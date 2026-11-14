// The wired 610 lifecycle, offline (, docs/driver/gsp-610-runtime plan §4, design "Memory, failure and
// rollback"). No GPU: a hand-built AD102 device (the test_mm pattern) with a register-level falcon model, the REAL
// hash-pinned 610 runtime bundle, and a mock GSP-RM queue peer running in a thread over the same shared ring memory
// the driver boots with. What this proves:
//
//   * the 610 profile's asset resolution loads every lifecycle asset through the C pins, and those pins equal the
//     prepared manifest's hashes (the two cannot drift silently);
//   * tinynv_gsp_init_sw builds the 610 startup args (152 B), queue geometry, WPR meta (boot610 layout), 952-byte
//     system info and registry table, and the records land in the command queue with the 610 MCTP/NVDM framing;
//   * the full boot (flcn FWSEC/scrubber/booter_load control flow + GSP start-up notice) and the object tree
//     (init_objects/channel/gr_context/open_client/queues) run against the mock peer with the 610 encoders at every
//     call site: 40-byte control headers, 376-byte channel allocs, 56-byte VA spaces, the 4032-byte GR static-info
//     request/reply carried across multiple queue elements, 3-byte schedules, 16-byte RUSD controls;
//   * failure injection at every stage (TINYNV_FAILPOINT) fails with the stage named, cleans up, and a retry with
//     the failure cleared boots the SAME frozen profile (no fallback, no mixed-version state);
//   * a boot whose start-up notice never comes is retried through unmake_boot/dev_reset_cold and comes up on the
//     second attempt with the profile intact;
//   * error-log / MMU-fault events are contained (err_state, first-error kept), the removed sequencer event and
//     unknown events are refused with a version-mismatch diagnostic, and a 570-framed record in a 610 queue is
//     refused by the codec;
//   * the unload notice + halt + FWSEC-SB + BTL1 booter_unload path tears the region down (WPR2 reads zero);
//   * after the 610 session, a fresh 570 software init in the same process succeeds (the offline rollback gate:
//     a separate clean process selecting 570; the hardware rollback is plan §6).
//
// This is control-flow / serialization evidence, NOT hardware evidence: the falcon model and the mock peer prove
// the driver's sequences and framing, never that the 610 firmware accepts them (plan §5).
#include "gpu.h"
#include "internal.h"
#include "nv_regs.h"
#include "rpc610.h"
#include "boot610.h"
#include "fw610.h"
#include "sha256.h"
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>

static int fails;
#define CHECK(cond, ...) do { if (!(cond)) { printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); fails++; } } while (0)

// ---- the fake card --------------------------------------------------------------------------------------------------

#define WINDOW (128u << 20)
#define VRAM_MB 24576u
#define REGS (16u << 20)

static uint32_t regs[REGS / 4];
#define ROM_SIZE 0x100000u
static uint8_t rom[ROM_SIZE];
static uint32_t fake_wpr2;          // the region register the falcon model places/tears down
static int gsp_starts, sec2_starts;
static uint64_t frts_expect;        // what FWSEC is asked to place (vram - 2 MiB), in register units

static uint32_t fk_rd32(tinynv_mmio_t *m, uint64_t off) {
  (void)m;
  // the falcons span more than one 4 KB page (RISCV regs sit at +0x1388/+0x1668), so match the engine RANGE and use
  // the 16-bit register offset within it, not a 4 KB page mask.
  int is_gsp = off >= 0x110000 && off < 0x120000, is_sec2 = off >= 0x840000 && off < 0x850000;
  uint64_t sub = off & 0xffff;
  if (is_gsp || is_sec2) {
    if (sub == NV_PFALCON_FALCON_DMATRFCMD) return 1u << NV_PFALCON_FALCON_DMATRFCMD_IDLE_LO; // drains at once, never full
    if (sub == NV_PFALCON_FALCON_CPUCTL) return 1u << NV_PFALCON_FALCON_CPUCTL_HALTED_LO;     // halts when started
    if (sub == NV_PFALCON_FALCON_HWCFG2) return 0;                                           // scrub done, falcon core
    if (sub == NV_PRISCV_RISCV_BCR_CTRL) return 1u << NV_PRISCV_RISCV_BCR_CTRL_VALID_LO;
    if (sub == NV_PRISCV_RISCV_CPUCTL) return 1u << NV_PRISCV_RISCV_CPUCTL_ACTIVE_STAT_LO;    // riscv active
  }
  if (off == NV_PFB_PRI_MMU_WPR2_ADDR_HI) return fake_wpr2;
  if (off == NV_PGC6_AON_SECURE_SCRATCH_GROUP_05(0)) return 0xff;   // gfw boot done
  if (off == NV_PGC6_AON_SECURE_SCRATCH_GROUP_05_PRIV_LEVEL_MASK) return 1;
  if (off == NV_PMC_BOOT_0) return 0x192000a1;                      // AD102
  if (off == NV_PGC6_AON_SECURE_SCRATCH_GROUP_42) return VRAM_MB;
  if (off == NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE) return regs[off / 4] & ~(1u << 31);
  return regs[off / 4];
}

static void fk_wr32(tinynv_mmio_t *m, uint64_t off, uint32_t v) {
  (void)m;
  regs[off / 4] = v;
  int is_gsp = off >= 0x110000 && off < 0x120000, is_sec2 = off >= 0x840000 && off < 0x850000;
  uint64_t sub = off & 0xffff;
  if (sub == NV_PFALCON_FALCON_CPUCTL && (v & (1u << NV_PFALCON_FALCON_CPUCTL_STARTCPU_LO))) {
    if (is_gsp) {
      gsp_starts++;
      // the first GSP start is FWSEC placing the region; later ones are the riscv handoff
      if (!fake_wpr2) fake_wpr2 = (uint32_t)(frts_expect >> 12);
    } else if (is_sec2) {
      sec2_starts++;
      if (regs[(0x840000 + NV_PFALCON_FALCON_MAILBOX0) / 4] == 0xff &&
          regs[(0x840000 + NV_PFALCON_FALCON_MAILBOX1) / 4] == 0xff)
        fake_wpr2 = 0;                       // booter_unload's "tear it down" mailboxes
      else
        regs[0x001180fc / 4] = 3u << 29;     // scrubber handoff (and booter_load leaves it set)
      // a successful SEC2 binary answers 0 in its mailboxes (the boot ROM's contract); the model ran it
      regs[(0x840000 + NV_PFALCON_FALCON_MAILBOX0) / 4] = 0;
      regs[(0x840000 + NV_PFALCON_FALCON_MAILBOX1) / 4] = 0;
    }
  }
}

static void fk_rd_block(tinynv_mmio_t *m, uint64_t off, void *dst, size_t n) {
  if (off >= 0x300000 && off + n <= 0x300000 + ROM_SIZE) { memcpy(dst, rom + (off - 0x300000), n); return; }
  for (size_t i = 0; i < n; i += 4) { uint32_t v = fk_rd32(m, off + i); memcpy((uint8_t *)dst + i, &v, n - i < 4 ? n - i : 4); }
}
static void fk_wr_block(tinynv_mmio_t *m, uint64_t off, const void *src, size_t n) {
  for (size_t i = 0; i < n; i += 4) { uint32_t v; memcpy(&v, (const uint8_t *)src + i, n - i < 4 ? n - i : 4); fk_wr32(m, off + i, v); }
}

// DMA: plain host memory. The buffers are kept until the process ends (dma_free only detaches): the mock peer thread
// reads the queue memory concurrently, and a real free between the driver's detach and the peer's next look would be
// a use-after-free in the TEST, not a driver bug. Bounded: a few hundred MB over the whole suite.
static struct { void *va; size_t size; } dma_keep[512];
static int ndma_keep;
static uint64_t next_iova = 0x100000000ull;

static int fk_dma_alloc(tinynv_pci_t *p, size_t size, tinynv_dma_t *out) {
  (void)p;
  void *va = calloc(1, (size + 4095) & ~4095ull);
  if (!va) return tinynv_fail("fake dma: out of memory");
  size_t np = ((size + 4095) & ~4095ull) / 4096;
  uint64_t *pages = calloc(np, 8);
  for (size_t i = 0; i < np; i++) pages[i] = next_iova + i * 4096;
  next_iova += np * 4096;
  *out = (tinynv_dma_t){.va = va, .size = (size + 4095) & ~4095ull, .pages = pages, .npages = np};
  tinynv_dma_bind_host(out);
  if (ndma_keep < 512) { dma_keep[ndma_keep].va = va; dma_keep[ndma_keep].size = size; ndma_keep++; }
  return 0;
}
static void fk_dma_free(tinynv_pci_t *p, tinynv_dma_t *d) { (void)p; free(d->pages); d->pages = NULL; d->va = NULL; }

static uint32_t fk_cfg_read(tinynv_pci_t *p, uint32_t off, uint32_t size) {
  (void)p;
  if (off == 0x00) return 0x268410deu; // the WIRE dword (vendor low, device high), as the real card answers
  if (off == 0x04) return 0x0007;
  if (off == 0x2c) return 0x14621234u;
  if (off == 0x08) return 0xa1;
  return 0;
}
static void fk_cfg_write(tinynv_pci_t *p, uint32_t off, uint32_t size, uint32_t val) { (void)p; (void)off; (void)size; (void)val; }
static int fk_bar_info(tinynv_pci_t *p, int bar, uint64_t *base, uint64_t *size) {
  (void)p; (void)bar;
  if (base) *base = 0;
  if (size) *size = WINDOW;
  return 0;
}
static int fk_reset(tinynv_pci_t *p) { (void)p; fake_wpr2 = 0; return 0; }
static void fk_quiesce(tinynv_pci_t *p) { (void)p; }

static tinynv_pci_t fk_pci = {
  .name = "0000:01:00.0", .live = 1,
  .cfg_read = fk_cfg_read, .cfg_write = fk_cfg_write, .bar_info = fk_bar_info,
  .dma_alloc = fk_dma_alloc, .dma_free = fk_dma_free, .reset = fk_reset, .quiesce = fk_quiesce,
};

static void *window_mem;

// ---- the synthetic board ROM the NORMAL vbios.c parser accepts -------------------------------------------------------
// The synthetic ROM targets vbios_610.c (BIT at 0x102, no expansion chain); the normal boot path's vbios.c
// wants the BIT at 0x1b0 and a base+extended expansion-rom chain. This builds that layout: a base image, an extended
// image (so exp_off measures back to 0), the BIT falcon-data token, the ucode table's FWSEC_PROD entry, and a v3
// descriptor + signature block + image with the interface/mapper/command/PKC slots fwsec_patch writes. Synthetic
// bytes throughout - the boot ROM's signature check is hardware's job (plan §5), not this suite's.
static void rom_w16(uint8_t *p, uint16_t v) { p[0] = v; p[1] = v >> 8; }
static void rom_w32(uint8_t *p, uint32_t v) { p[0] = v; p[1] = v >> 8; p[2] = v >> 16; p[3] = v >> 24; }

static void build_normal_rom(uint8_t *r) {
  memset(r, 0, ROM_SIZE);
  r[0] = 0x55; r[1] = 0xaa;
  // base image: 16 blocks of 0x200 = 0x2000 bytes, code type 0 (base)
  rom_w16(r + 0x18, 0x40);
  memcpy(r + 0x40, "PCIR", 4);
  rom_w16(r + 0x40 + 0x10, 16);
  r[0x40 + 0x14] = 0x00;
  // extended image at 0x2000, code type 0xe0 -> exp_off = 0x2000 - 0x2000 = 0. The PCIR pointer is RELATIVE to the
  // image start (the walk computes off + pci_blck), so it is 0x40 here, not the absolute 0x2040.
  rom_w16(r + 0x2000 + 0x18, 0x40);
  memcpy(r + 0x2040, "PCIR", 4);
  rom_w16(r + 0x2040 + 0x10, 16);
  r[0x2040 + 0x14] = 0xe0;
  r[0x2040 + 0x15] = 0x80;
  // BIT at 0x1b0: one falcon-data token (v2) pointing at the ucode table
  rom_w16(r + 0x1b0, 0xb8ff);
  rom_w32(r + 0x1b0 + 2, 0x00544942u);   // "BIT\0"
  r[0x1b0 + 8] = 12; r[0x1b0 + 9] = 8; r[0x1b0 + 10] = 1;
  r[0x1bc] = 0x70; r[0x1bd] = 2; rom_w16(r + 0x1be, 4); rom_w32(r + 0x1c0, 0x1c8);
  rom_w32(r + 0x1c8, 0x200);             // FalconUcodeTablePtr (exp_off = 0)
  // ucode table at 0x200: one FWSEC_PROD entry
  r[0x200] = 1; r[0x201] = 6; r[0x202] = 6; r[0x203] = 1; r[0x204] = 3; r[0x205] = 44;
  r[0x206] = 0x85; r[0x207] = 1; rom_w32(r + 0x208, 0x300);
  // descriptor at 0x300: v3 + a 768-byte signature block, then the image
  uint8_t *d = r + 0x300;
  rom_w32(d, (812u << 16) | (3u << 8) | 1u);
  rom_w32(d + 4, 0x1580);                // StoredSize: imem 0x800 + dmem 0xd80
  rom_w32(d + 8, 0x900);                 // PKCDataOffset (in dmem)
  rom_w32(d + 12, 0x100);                // InterfaceOffset (in dmem)
  rom_w32(d + 16, 0);                    // IMEMPhysBase
  rom_w32(d + 20, 0x800);                // IMEMLoadSize
  rom_w32(d + 24, 0);                    // IMEMVirtBase
  rom_w32(d + 28, 0);                    // DMEMPhysBase
  rom_w32(d + 32, 0xd80);                // DMEMLoadSize
  rom_w16(d + 36, 0x400);                // EngineIdMask
  d[38] = 1; d[39] = 2; rom_w16(d + 40, 3);
  memset(d + 44, 0x66, 768);             // signature block (last 0x180 = production sig)
  uint8_t *img = d + 812;
  memset(img, 0x99, 0x800);              // IMEM
  uint8_t *dmem = img + 0x800;
  // interface table at dmem 0x100: one DMEMMAPPER entry
  dmem[0x100] = 1; dmem[0x101] = 4; dmem[0x102] = 8; dmem[0x103] = 1;
  rom_w32(dmem + 0x104, 4);
  rom_w32(dmem + 0x108, 0x200);
  // mapper at dmem 0x200: command buffer at dmem 0x300
  uint8_t *mapper = dmem + 0x200;
  rom_w16(mapper + 4, 3); rom_w16(mapper + 6, 64);
  rom_w32(mapper + 8, 0x300); rom_w32(mapper + 12, 64);
}

// Build a device + mm the way test_mm does, frozen to the given profile. No gpu_open: the selection gate is
// test_runtime_profile's job; here the lifecycle above it is what is under test.
static int dev_setup(tinynv_gpu_t *g, const tinynv_runtime_profile_t *profile) {
  memset(g, 0, sizeof *g);
  g->profile = profile;
  tinynv_dev_t *d = &g->dev;
  d->pci = &fk_pci;
  d->runtime = profile;
  d->vram.ptr = window_mem;
  d->vram.size = WINDOW;
  d->mmio.ptr = NULL;
  d->mmio.rd32 = fk_rd32;
  d->mmio.wr32 = fk_wr32;
  d->mmio.rd_block = fk_rd_block;
  d->mmio.wr_block = fk_wr_block;
  d->mmio.size = REGS;
  d->vram_size = (uint64_t)VRAM_MB << 20;
  d->large_bar = 0;
  d->mmu_ver = 2;                       // AD102
  d->fmc_boot = 0;
  strcpy(d->chip_name, "AD102");
  strcpy(d->fw_name, "ad102");
  d->chip_id = 0x192;
  d->class_gpfifo = 0xc56f;             // AMPERE_CHANNEL_GPFIFO_A, the AD102 class (dev.c)
  d->class_compute = 0xc9c0;            // ADA_COMPUTE_A
  d->class_dma_copy = 0xc7b5;
  // the configuration snapshot a retry's reset_cold puts back (early_init normally takes it)
  d->cfg_saved = 1;
  d->cfg_cmd = 7;
  g->flcn.gpu = g;
  g->gsp.gpu = g;
  frts_expect = d->vram_size - 0x100000 - 0x100000;
  // the region FWSEC is asked to place; flcn_init_sw sets the same value, but the gsp-only arms need it too
  g->flcn.frts_offset = frts_expect;
  return tinynv_mm_init(&g->mm, d);
}

static void dev_teardown(tinynv_gpu_t *g) {
  tinynv_gsp_fini(g);
  tinynv_flcn_fini(g);
}

// ---- the mock GSP-RM peer --------------------------------------------------------------------------------------------
//
// A thread over the same ring memory the driver boots with. It decodes requests with the SAME 610 codec (itself
// tested by test_rpc610), echoes RM-style replies with the parameters filled, splits anything over the negotiated record ceiling into
// CONTINUATION_RECORDs exactly as the vendor host does, and can inject events / refusals / silence on command.

typedef struct {
  tinynv_gpu_t *g;
  atomic_int stop, mode;         // 0 = silent (no status header), 1 = header only (no init-done), 2 = full
  atomic_int generation;         // bumped by the test after a retry rebuilds the queues
  atomic_int inject_event;       // an event function to send once (0 = none)
  atomic_int refuse_next;        // answer the next rpc with rpc_result = 1
  atomic_int drop_replies;       // consume requests but never answer (timeout arm)
  atomic_int unload_parks_only;  // consume the unload notice and park, but send no reply record (silicon)
  atomic_int reply_empty_once;   // answer the next request with a ZERO-byte payload (the shape silicon sent)
  atomic_int saw_unload, saw_gr_multielement, saw_ctl40, saw_chan376, saw_va56, saw_sched3, saw_rusd16;
  int gen_seen;
  uint32_t rx_seq, tx_seq, rp, wp;
  uint8_t *pend;                 // request being reassembled
  uint32_t pend_len, pend_cap, pend_func;
} peer_t;

static peer_t P;

static uint8_t *qmem(peer_t *p, uint64_t *cmd_off, uint64_t *qsize) {
  tinynv_gsp_t *gsp = &p->g->gsp;
  uint8_t *ptr = (uint8_t *)gsp->queues.view.ptr;
  *cmd_off = gsp->cmd_q_off;
  *qsize = gsp->queue_size;
  return ptr;
}

static void peer_send_record(peer_t *p, uint8_t *mem, uint64_t stat_off, uint32_t func, const uint8_t *payload,
                             uint32_t len, uint32_t result) {
  uint8_t rec[TINYNV_RPC610_ELEM_SIZE_MAX];
  uint32_t count = 0;
  size_t total = 0;
  // encode with the driver codec, then set the result the way the firmware would (the codec sends PENDING)
  if (tinynv_rpc610_encode(rec, sizeof rec, func, payload, len, p->tx_seq, 4096, &count, &total)) {
    fprintf(stderr, "peer: encode failed: %s\n", tinynv_last_error());
    return;
  }
  // the codec set rpc_result=PENDING (a REQUEST value); the GSP side reports 0 for a delivered event/reply, nonzero
  // for a refusal. Set it unconditionally and recompute the checksum with the result in place.
  { uint32_t r = result; memcpy(rec + TINYNV_RPC610_ELEM_HDR_SIZE + 16, &r, 4);
    uint32_t rpc_len; memcpy(&rpc_len, rec + TINYNV_RPC610_ELEM_HDR_SIZE + 8, 4);
    memset(rec + 8, 0, 4);
    uint32_t ck = tinynv_rpc_checksum(rec, TINYNV_RPC610_ELEM_HDR_SIZE + rpc_len);
    memcpy(rec + 8, &ck, 4);
  }
  uint32_t wp;
  memcpy(&wp, mem + stat_off + 16, 4);
  uint64_t entries = stat_off + 4096, ring_len = 63ull * 4096;
  for (size_t i = 0; i < total; i++)
    mem[entries + ((uint64_t)wp * 4096 + i) % ring_len] = rec[i];
  wp = (wp + count) % 63;
  memcpy(mem + stat_off + 16, &wp, 4);
  p->tx_seq++;
}

static void peer_send_reply(peer_t *p, uint8_t *mem, uint64_t stat_off, uint32_t func, const uint8_t *payload,
                            uint32_t len, uint32_t result) {
  const uint32_t max = (uint32_t)tinynv_rpc610_first_payload_max();
  if (len <= max) {
    if (func == TINYNV_MSG_FUNCTION_GSP_RM_CONTROL && len + 48 > 4096)
      atomic_store(&p->saw_gr_multielement, 1);
    peer_send_record(p, mem, stat_off, func, payload, len, result); return;
  }
  peer_send_record(p, mem, stat_off, func, payload, max, result);
  for (uint32_t off = max; off < len; off += max)
    peer_send_record(p, mem, stat_off, TINYNV_MSG_FUNCTION_CONTINUATION_RECORD, payload + off,
                     len - off < max ? len - off : max, 0);
  if (func == TINYNV_MSG_FUNCTION_GSP_RM_CONTROL) atomic_store(&p->saw_gr_multielement, 1);
}

static void peer_fill_reply(peer_t *p, uint32_t func, uint8_t *buf, uint32_t len) {
  if (func == TINYNV_MSG_FUNCTION_GSP_RM_CONTROL && len >= 40) {
    uint32_t cmd;
    memcpy(&cmd, buf + 8, 4);
    uint8_t *params = buf + 40;
    uint32_t plen = len - 40;
    switch (cmd) {
      case TINYNV_CTRL_CMD_FIFO_GET_DEVICE_INFO_TABLE:
        if (plen >= sizeof(tinynv_fifo_device_info_t)) {
          tinynv_fifo_device_info_t *t = (void *)params;
          t->numEntries = 2;
          t->entries[0].engineData[2] = 1; t->entries[0].engineData[3] = 0;   // GR, runlist 0
          t->entries[0].numPbdmas = 2; strcpy(t->entries[0].engineName, "GR0");
          t->entries[1].engineData[2] = 5; t->entries[1].engineData[3] = 1;   // a copy engine, runlist 1
          t->entries[1].numPbdmas = 1; strcpy(t->entries[1].engineName, "CE0");
        }
        break;
      case TINYNV_CTRL_CMD_STATIC_KGR_GET_CONTEXT_BUFFERS_INFO:
        if (plen >= sizeof(tinynv_kgr_ctx_info_t)) {
          tinynv_kgr_ctx_info_t *t = (void *)params;
          for (int e = 0; e < 8; e++)
            for (int b = 0; b < TINYNV_GR_CTX_BUFFERS; b++) {
              t->engineContextBuffersInfo[e].engine[b].size = b == 0 ? 0x20000 : 0x1000;
              t->engineContextBuffersInfo[e].engine[b].alignment = 0x1000;
            }
        }
        break;
      case TINYNV_CTRL_CMD_STATIC_KGR_GET_INFO:
        if (plen >= TINYNV_RPC610_GR_STATIC_SIZE) {
          // the 610 stride: 63 entries of 8 bytes per engine. Fill the indices the driver reads.
          const struct { uint32_t idx, val; } V[] = {
            {TINYNV_GR_INFO_NUM_GPCS, 7}, {TINYNV_GR_INFO_NUM_TPC_PER_GPC, 6},
            {TINYNV_GR_INFO_NUM_SM_PER_TPC, 2}, {TINYNV_GR_INFO_MAX_WARPS_PER_SM, 48},
            {TINYNV_GR_INFO_SM_VERSION, 0x89},
          };
          for (size_t i = 0; i < sizeof(V) / sizeof(*V); i++) {
            uint8_t *at = params + (uint64_t)V[i].idx * 8;
            memcpy(at, &V[i].idx, 4);
            memcpy(at + 4, &V[i].val, 4);
          }
        }
        break;
      case TINYNV_CTRL_CMD_GET_WORK_SUBMIT_TOKEN:
        if (plen >= 4) { uint32_t tok = 0x123; memcpy(params, &tok, 4); }
        break;
      case TINYNV_CTRL_INIT_USER_SHARED_DATA:
        if (plen == 16) atomic_store(&p->saw_rusd16, 1);
        break;
      case TINYNV_CTRL_CMD_GPFIFO_SCHEDULE:
        if (plen == 3) atomic_store(&p->saw_sched3, 1);
        break;
      default: break;
    }
  } else if (func == TINYNV_MSG_FUNCTION_GSP_RM_ALLOC && len >= 32) {
    uint32_t cls;
    memcpy(&cls, buf + 12, 4);   // hClass sits at offset 12 in the 32-byte alloc header
    if (cls == 0xc56f && len - 32 == 376) {
      tinynv_rpc610_gpfifo_alloc_t channel;
      memcpy(&channel, buf + 32, sizeof channel);
      // A declared notifier needs backing memory. The privileged channel has
      // none; falsely advertising CTXDMA used to pass this mock but fail 610.
      CHECK(((channel.internalFlags >> 2) & 3) == 1 || channel.errorNotifierMem.size != 0,
            "channel declares an error notifier without backing memory");
      CHECK(((channel.internalFlags >> 4) & 3) == 1 || channel.eccErrorNotifierMem.size != 0,
            "channel declares an ECC notifier without backing memory");
      atomic_store(&p->saw_chan376, 1);
    }
    if (cls == 0x90f1 && len - 32 == 56) atomic_store(&p->saw_va56, 1);
  } else if (func == TINYNV_MSG_FUNCTION_UNLOADING_GUEST_DRIVER) {
    atomic_store(&p->saw_unload, 1);
    // the firmware parks the core: mailbox0 = LIBOS_INTERRUPT_PROCESSOR_SUSPENDED
    regs[NV_PGSP_FALCON_MAILBOX0 / 4] = 0x80000000u;
  }
}

static void *peer_main(void *arg) {
  peer_t *p = arg;
  while (!atomic_load(&p->stop)) {
    uint64_t cmd_off = 0, qsize = 0;
    uint8_t *mem = qmem(p, &cmd_off, &qsize);
    int mode = atomic_load(&p->mode);
    if (!mem || !cmd_off || !qsize || mode == 0) { usleep(200); continue; }
    uint64_t stat_off = cmd_off + qsize;
    int gen = atomic_load(&p->generation);
    if (gen != p->gen_seen) {
      // (re)attach: write the status queue header the driver's rpcq_attach waits for
      p->gen_seen = gen;
      p->rx_seq = p->tx_seq = 0;
      p->rp = 0;
      p->pend_len = p->pend_cap = p->pend_func = 0;
      tinynv_msgq_tx_header_t hdr;
      memset(&hdr, 0, sizeof hdr);
      hdr.size = (uint32_t)qsize; hdr.msgSize = 4096; hdr.msgCount = 63;
      hdr.flags = 1; hdr.rxHdrOff = 32; hdr.entryOff = 4096;
      memcpy(mem + stat_off, &hdr, sizeof hdr);
      if (mode == 2) {
        // a one-way libos print (UCODE_LIBOS_PRINT, 0x100C) arrives before init-done on a real 610 boot. The
        // driver must consume it as benign and still reach init-done; refusing it as unknown aborted the live boot.
        uint8_t libos_print[16];
        memset(libos_print, 0, sizeof libos_print);
        peer_send_record(p, mem, stat_off, TINYNV_MSG_EVENT_UCODE_LIBOS_PRINT, libos_print, sizeof libos_print, 0);
        peer_send_record(p, mem, stat_off, TINYNV_MSG_EVENT_GSP_INIT_DONE, NULL, 0, 0);
      }
    }
    if (mode < 2) { usleep(200); continue; }

    int ev = atomic_exchange(&p->inject_event, 0);
    if (ev) {
      uint8_t payload[64];
      memset(payload, 0, sizeof payload);
      if (ev == TINYNV_MSG_EVENT_OS_ERROR_LOG) { memcpy(payload, "0123456789ab", 12); memcpy(payload + 12, "mock fault line", 15); }
      peer_send_record(p, mem, stat_off, (uint32_t)ev, payload,
                       ev == TINYNV_MSG_EVENT_OS_ERROR_LOG ? 27 : 8, 0);
    }

    // consume one request record (with continuations) and answer it. A record can span several 4 KB elements (a
    // max-size record is 65536 bytes = 16 elements), so read the two headers, derive the count from the RPC
    // length, and assemble count elements with ring wrap - exactly as the driver's drain does.
    uint32_t wp;
    memcpy(&wp, mem + cmd_off + 16, 4);
    if (p->rp == wp) { usleep(100); continue; }
    uint64_t entries = cmd_off + 4096, ring_len = 63ull * 4096;
    uint8_t hdrs[TINYNV_RPC610_ELEM_HDR_SIZE + 32];
    for (int i = 0; i < (int)sizeof hdrs; i++)
      hdrs[i] = mem[entries + ((uint64_t)p->rp * 4096 + i) % ring_len];
    tinynv_msg_header_t mh;
    memcpy(&mh, hdrs + TINYNV_RPC610_ELEM_HDR_SIZE, sizeof mh);
    if (mh.length < 32 || mh.length > TINYNV_RPC610_ELEM_SIZE_MAX - TINYNV_RPC610_ELEM_HDR_SIZE) {
      fprintf(stderr, "peer: bad request rpc length %u\n", mh.length);
      p->rp = (p->rp + 1) % 63; p->rx_seq++;
      continue;
    }
    uint32_t count = (uint32_t)((TINYNV_RPC610_ELEM_HDR_SIZE + (uint64_t)mh.length + 4095) / 4096);
    static uint8_t rec[TINYNV_RPC610_ELEM_SIZE_MAX];
    size_t total = (size_t)count * 4096;
    for (size_t i = 0; i < total; i++)
      rec[i] = mem[entries + ((uint64_t)p->rp * 4096 + i) % ring_len];
    const uint8_t *payload;
    uint32_t payload_len;
    if (tinynv_rpc610_decode(rec, total, 4096, p->rx_seq, &mh, &payload, &payload_len, &count)) {
      fprintf(stderr, "peer: request decode failed: %s\n", tinynv_last_error());
      p->rp = (p->rp + 1) % 63;
      p->rx_seq++;
      continue;
    }
    p->rp = (p->rp + count) % 63;
    p->rx_seq++;
    // publish our read position where the driver's cmd queue expects it (stat header's rxHdrOff)
    memcpy(mem + stat_off + 32, &p->rp, 4);

    if (mh.function == TINYNV_MSG_FUNCTION_CONTINUATION_RECORD) {
      if (p->pend && p->pend_len + payload_len <= p->pend_cap) {
        memcpy(p->pend + p->pend_len, payload, payload_len);
        p->pend_len += payload_len;
      }
    } else {
      free(p->pend);
      p->pend_cap = payload_len + 8192;
      p->pend = malloc(p->pend_cap);
      memcpy(p->pend, payload, payload_len);
      p->pend_len = payload_len;
      p->pend_func = mh.function;
    }
    // the expected request total: RM alloc/control name it in the header; the rest are fixed-size payloads
    uint32_t want = p->pend_len;
    if (p->pend_func == TINYNV_MSG_FUNCTION_GSP_RM_ALLOC && p->pend_len >= 32) {
      uint32_t ps; memcpy(&ps, p->pend + 20, 4); want = 32 + ps;
    } else if (p->pend_func == TINYNV_MSG_FUNCTION_GSP_RM_CONTROL && p->pend_len >= 40) {
      uint32_t ps; memcpy(&ps, p->pend + 16, 4); want = 40 + ps;
      atomic_store(&p->saw_ctl40, 1);
    } else if (p->pend_func == TINYNV_MSG_FUNCTION_GSP_SET_SYSTEM_INFO) want = TINYNV_RPC610_SYSINFO_SIZE;
    else if (p->pend_func == TINYNV_MSG_FUNCTION_SET_REGISTRY && p->pend_len >= 4) memcpy(&want, p->pend, 4);
    else if (p->pend_func == TINYNV_MSG_FUNCTION_SET_PAGE_DIRECTORY) want = 48;
    else if (p->pend_func == TINYNV_MSG_FUNCTION_UNLOADING_GUEST_DRIVER) want = 8;
    if (p->pend_len < want) {
      CHECK(payload_len == 65536 - 16 - 32,
            "premature continuation: first RPC fragment is %u bytes", payload_len);
      continue;
    }

    if (!atomic_load(&p->drop_replies)) {
      peer_fill_reply(p, p->pend_func, p->pend, p->pend_len);
      // the two reply shapes the live 610 boot produced. For the unload notice gsp-rm parks its core (mailbox0
      // = LIBOS_INTERRUPT_PROCESSOR_SUSPENDED, set by peer_fill_reply above) and answers with the function echoed back
      // and NO payload bytes; the driver no longer waits for that reply, so the mailbox park is what completes the
      // unload. `reply_empty_once` hands the same zero-byte shape to whatever request comes next, which is how the
      // drain's accept-an-empty-reply rule is exercised on a call that DOES wait.
      int empty = (atomic_load(&p->unload_parks_only) &&
                   p->pend_func == TINYNV_MSG_FUNCTION_UNLOADING_GUEST_DRIVER) ||
                  atomic_exchange(&p->reply_empty_once, 0);
      if (empty) {
        peer_send_record(p, mem, stat_off, p->pend_func, NULL, 0, 0);
      } else {
        uint32_t result = atomic_exchange(&p->refuse_next, 0) ? 1 : 0;
        peer_send_reply(p, mem, stat_off, p->pend_func, p->pend, p->pend_len, result);
      }
    }
    free(p->pend);
    p->pend = NULL;
    p->pend_len = p->pend_cap = p->pend_func = 0;
  }
  return NULL;
}

static void peer_start(tinynv_gpu_t *g, int mode) {
  memset(&P, 0, sizeof P);
  P.g = g;
  atomic_store(&P.mode, mode);
  atomic_store(&P.generation, 1);
  pthread_t th;
  pthread_create(&th, NULL, peer_main, &P);
  pthread_detach(th);
}
static void peer_stop(void) { atomic_store(&P.stop, 1); usleep(2000); free(P.pend); P.pend = NULL; }

// ---- 1. the C pins equal the prepared manifest ----------------------------------------------------------------------

static void manifest_arm(const char *bundle) {
  char path[1024];
  snprintf(path, sizeof path, "%s/manifest.json", bundle);
  FILE *f = fopen(path, "rb");
  if (!f) {
    printf("  FAIL: no manifest at %s - prepare the bundle with tools/gsp610-runtime/run_prepare.sh\n", path);
    fails++;
    return;
  }
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  char *json = malloc(n + 1);
  if (fread(json, 1, n, f) != (size_t)n) { fclose(f); free(json); fails++; return; }
  json[n] = 0;
  fclose(f);
  const char *names[] = {"gsp", "bootloader", "bootloader_desc", "booter_load", "booter_unload", "scrubber",
                         "generic_bl", "generic_bl_desc"};
  for (size_t i = 0; i < sizeof(names) / sizeof(*names); i++) {
    // the manifest's sha256 for this asset
    char key[64];
    snprintf(key, sizeof key, "\"%s\"", names[i]);
    char *at = strstr(json, key);
    CHECK(at != NULL, "the manifest has no asset %s", names[i]);
    if (!at) continue;
    char *sha = strstr(at, "\"sha256\": \"");
    CHECK(sha != NULL, "the manifest's %s has no sha256", names[i]);
    if (!sha) continue;
    sha += 11;
    // the C pin: load through fw610 (which enforces it) and hash what came back
    tinynv_blob_t b;
    CHECK(tinynv_fw610_load(names[i], &b) == 0, "fw610 refused %s: %s", names[i], tinynv_last_error());
    if (b.data) {
      uint8_t digest[32];
      char hex[65];
      tinynv_sha256(b.data, b.size, digest);
      tinynv_sha256_hex(digest, hex);
      CHECK(!strncmp(hex, sha, 64), "the C pin for %s differs from the prepared manifest", names[i]);
      tinynv_fw_free(&b);
    }
  }
  free(json);
  printf("manifest: all 8 lifecycle assets load through the C pins, and the pins equal the prepared manifest\n");
}

// ---- 2. the software init, with the real bundle ---------------------------------------------------------------------

static void sw_init_arm(tinynv_gpu_t *g) {
  CHECK(tinynv_gsp_init_sw(g) == 0, "the 610 software init failed: %s", tinynv_last_error());
  tinynv_gsp_t *gsp = &g->gsp;
  // the 152-byte cached args with the 610 queue geometry
  CHECK(gsp->rm_args.size >= 152, "the 610 rm args block is %zu bytes, not the 152-byte layout", gsp->rm_args.size);
  tinynv_boot610_args_cached_t args;
  nv_rd_block(&gsp->rm_args.view, 0, &args, sizeof args);
  CHECK(args.bDmemStack == 1, "bDmemStack not set");
  CHECK(args.messageQueueInitArguments.queueElementHdrSize == 16 &&
        args.messageQueueInitArguments.queueElementSizeMin == 4096 &&
        args.messageQueueInitArguments.queueElementSizeMax == 65536 &&
        args.messageQueueInitArguments.queueHeaderAlign == 4 &&
        args.messageQueueInitArguments.queueElementAlign == 12, "the 610 queue geometry fields are wrong");
  CHECK(args.messageQueueInitArguments.sharedMemPhysAddr == gsp->queues.addrs[0], "the queue phys addr is wrong");
  CHECK(gsp->rm_state_monitor.size >= 4096 && gsp->rm_state_monitor.naddrs,
        "the required 610 state-monitor page is absent");
  CHECK(args.rmStateMonitorBufferArgs[0] == gsp->rm_state_monitor.addrs[0] &&
        args.rmStateMonitorBufferArgs[1] == 4096, "the state-monitor startup mapping is missing/wrong");
  uint8_t monitor[4096];
  nv_rd_block(&gsp->rm_state_monitor.view, 0, monitor, sizeof monitor);
  for (size_t i = 0; i < sizeof monitor; i++)
    if (monitor[i]) { CHECK(0, "the state-monitor page has stale data at %zu", i); break; }
  // the sysmem flush page 610's fatal KernelMemorySystem assert needs - present, in sysmem, and zeroed
  CHECK(gsp->sysmem_flush.size >= 4096 && gsp->sysmem_flush.naddrs && gsp->sysmem_flush.in_sysmem,
        "the required 610 sysmem flush page is absent or not in sysmem");
  uint8_t flush[4096];
  nv_rd_block(&gsp->sysmem_flush.view, 0, flush, sizeof flush);
  for (size_t i = 0; i < sizeof flush; i++)
    if (flush[i]) { CHECK(0, "the sysmem flush page has stale data at %zu", i); break; }
  // the WPR meta: the boot610 layout, every boundary ordered, the reservation covering it
  tinynv_wpr_meta_t m;
  nv_rd_block(&gsp->wpr_meta.view, 0, &m, sizeof m);
  CHECK(m.magic == TINYNV_GSP_FW_WPR_META_MAGIC && m.revision == TINYNV_GSP_FW_WPR_META_REVISION, "wpr meta magic");
  CHECK(m.sizeOfRadix3Elf == 84226048ull, "the 610 .fwimage size is %llu, not the pinned 84226048",
        (unsigned long long)m.sizeOfRadix3Elf);
  CHECK(m.nonWprHeapSize == 0x100000, "the AD102 non-WPR heap is %llu, not the matching 1 MiB",
        (unsigned long long)m.nonWprHeapSize);
  CHECK(m.gspFwWprEnd == g->dev.vram_size - 0x100000 && m.frtsOffset == frts_expect, "the carveout top is wrong");
  CHECK(m.nonWprHeapOffset < m.gspFwWprStart && m.gspFwWprStart < m.gspFwHeapOffset &&
        m.gspFwHeapOffset < m.gspFwOffset && m.gspFwOffset < m.bootBinOffset && m.bootBinOffset < m.frtsOffset,
        "the carveout boundaries are not ordered");
  CHECK(g->dev.vram_size - m.nonWprHeapOffset <= tinynv_runtime_610.reserve_top, "the carveout exceeds the reservation");
  CHECK(m.bootloaderCodeOffset == gsp->bl_desc610.code_offset &&
        m.bootloaderDataOffset == gsp->bl_desc610.data_offset &&
        m.bootloaderManifestOffset == gsp->bl_desc610.manifest_offset, "the boot descriptor offsets were not placed");
  // the generic BL staged for the host requests
  CHECK(gsp->gbl_fw.size == 768 && gsp->gbl_start_tag == 0xFD && gsp->gbl_code_size == 0x200,
        "the generic-BL ucode/descriptor were not staged (size %zu tag %#x code %#x)",
        gsp->gbl_fw.size, gsp->gbl_start_tag, gsp->gbl_code_size);
  // the two prefilled records in the command queue: 952-byte system info, then the registry table
  uint8_t *mem = (uint8_t *)gsp->queues.view.ptr;
  uint8_t rec[4096];
  tinynv_msg_header_t mh;
  const uint8_t *payload;
  uint32_t plen, count;
  memcpy(rec, mem + gsp->cmd_q_off + 4096, 4096);
  CHECK(tinynv_rpc610_decode(rec, 4096, 4096, 0, &mh, &payload, &plen, &count) == 0,
        "the first prefilled record is not valid 610 framing: %s", tinynv_last_error());
  CHECK(mh.function == TINYNV_MSG_FUNCTION_GSP_SET_SYSTEM_INFO && plen == TINYNV_RPC610_SYSINFO_SIZE,
        "the first record is func %u len %u, not the 952-byte system info", mh.function, plen);
  CHECK(payload[TINYNV_RPC610_SYSINFO_PASSTHRU_OFF] == 1, "bIsPassthru is not at its 610 offset 842");
  memcpy(rec, mem + gsp->cmd_q_off + 4096 + 4096, 4096);
  CHECK(tinynv_rpc610_decode(rec, 4096, 4096, 1, &mh, &payload, &plen, &count) == 0, "the registry record framing");
  CHECK(mh.function == TINYNV_MSG_FUNCTION_SET_REGISTRY, "the second record is func %u, not the registry", mh.function);
  CHECK(gsp->cmd_q.seq == 2, "the command sequence is %u after two records", gsp->cmd_q.seq);
  printf("sw init: 152-byte args + queue geometry, boot610 carveout (80.32 MiB image, 1 MiB non-WPR), generic BL staged, "
         "952-byte sysinfo + registry in 610 framing\n");
}

// ---- 3. per-stage failure injection ----------------------------------------------------------------------------------

static void failpoint_arm(void) {
  const char *stages[] = {"610-assets", "610-rm-args", "610-libos-args", "610-gsp-image", "610-bootloader",
                          "610-wpr-meta", "610-sysinfo", "610-registry", "610-rpc-send"};
  for (size_t i = 0; i < sizeof(stages) / sizeof(*stages); i++) {
    tinynv_gpu_t g;
    CHECK(dev_setup(&g, &tinynv_runtime_610) == 0, "dev setup: %s", tinynv_last_error());
    setenv("TINYNV_FAILPOINT", stages[i], 1);
    int rc = tinynv_gsp_init_sw(&g);
    CHECK(rc != 0, "failpoint %s did not fail the software init", stages[i]);
    CHECK(strstr(tinynv_last_error(), stages[i]) != NULL, "the %s failure does not name the stage: %s",
          stages[i], tinynv_last_error());
    CHECK(g.profile == &tinynv_runtime_610, "the failure disturbed the frozen profile");
    dev_teardown(&g);   // the cleanup must not crash on a half-built boot
    unsetenv("TINYNV_FAILPOINT");
    // and with the failure cleared, the SAME profile boots its software init again
    tinynv_gpu_t g2;
    CHECK(dev_setup(&g2, &tinynv_runtime_610) == 0, "dev setup (retry): %s", tinynv_last_error());
    CHECK(tinynv_gsp_init_sw(&g2) == 0, "the software init did not recover after %s was cleared: %s",
          stages[i], tinynv_last_error());
    CHECK(g2.profile == &tinynv_runtime_610, "the recovery changed the profile");
    dev_teardown(&g2);
  }
  printf("failpoints: %zu stages each fail with the stage named, clean up, and recover on the same frozen profile\n",
         sizeof(stages) / sizeof(*stages));
}

// ---- 4. the full boot + object tree against the mock peer -------------------------------------------------------------

static void boot_and_objects_arm(tinynv_gpu_t *g) {
  setenv("TINYNV_BOOT_ATTEMPTS", "1", 1);
  peer_start(g, 2);
  CHECK(tinynv_gpu_init_sw(g) == 0, "the full 610 software init (flcn+gsp) failed: %s", tinynv_last_error());
  // the BTL1 booters parsed at software init, not yet staged (the fuse is a device read)
  CHECK(g->flcn.booter.is_btl1 && g->flcn.booter.image.size == 0, "booter_load was not parsed as BTL1 / staged too early");
  CHECK(g->flcn.scrubber.is_btl1, "the scrubber was not parsed as BTL1");
  CHECK(tinynv_gpu_init_hw(g) == 0, "the 610 hardware boot failed: %s", tinynv_last_error());
  CHECK(g->gsp.up, "gsp-rm did not come up");
  // the PFB flush registers now name the sysmem flush page (HI then LO) - the write that lets 610's fatal
  // KernelMemorySystem validity assert pass. Offsets from ga100 dev_fb.h (absent from the generated nv_regs.h).
  {
    uint64_t fa = g->gsp.sysmem_flush.addrs[0];
    CHECK(regs[0x00100C40 / 4] == ((uint32_t)(fa >> 40) & 0x7F),
          "the PFB flush HI register does not name the sysmem flush page");
    CHECK(regs[0x00100C10 / 4] == (uint32_t)(fa >> 8),
          "the PFB flush LO register does not name the sysmem flush page");
  }
  CHECK(g->flcn.booter.image.size > 0 && g->flcn.scrubber.image.size > 0, "the BTL1 booters were not fuse-staged");
  CHECK(gsp_starts >= 1 && sec2_starts >= 2, "the falcon boot sequence did not run (gsp %d, sec2 %d starts)",
        gsp_starts, sec2_starts);
  CHECK(fake_wpr2 != 0, "FWSEC did not place the region");

  // the object tree: every 610 encoder at its real call site, the GR static-info request/reply spanning multiple queue elements
  CHECK(tinynv_gsp_init_objects(g) == 0, "init_objects: %s", tinynv_last_error());
  CHECK(tinynv_gsp_init_channel(g) == 0, "init_channel: %s", tinynv_last_error());
  CHECK(tinynv_gsp_init_gr_context(g) == 0, "init_gr_context: %s", tinynv_last_error());
  CHECK(tinynv_gsp_open_client(g) == 0, "open_client: %s", tinynv_last_error());
  CHECK(tinynv_gsp_init_queues(g) == 0, "init_queues: %s", tinynv_last_error());
  CHECK(atomic_load(&P.saw_ctl40), "no 40-byte control header reached the peer");
  CHECK(atomic_load(&P.saw_chan376), "no 376-byte channel alloc reached the peer");
  CHECK(atomic_load(&P.saw_va56), "no 56-byte VA space alloc reached the peer");
  CHECK(atomic_load(&P.saw_sched3), "no 3-byte schedule reached the peer");
  CHECK(atomic_load(&P.saw_gr_multielement), "the GR static-info reply never spanned multiple queue elements");
  CHECK(g->gsp.max_gpcs == 7 && g->gsp.max_tpc_per_gpc == 6 && g->gsp.max_sm_per_tpc == 2 &&
        g->gsp.max_warps_per_sm == 48 && g->gsp.sm_version == 0x89,
        "the GR geometry did not come back through the 610 stride (gpcs %u tpc %u sm %u warps %u ver %#x)",
        g->gsp.max_gpcs, g->gsp.max_tpc_per_gpc, g->gsp.max_sm_per_tpc, g->gsp.max_warps_per_sm, g->gsp.sm_version);
  CHECK(g->gsp.compute_q.token == (0x123 | (0 << 16) | (1u << 30)), "the work submit token is %#x", g->gsp.compute_q.token);

  // a one-control round trip through the public API (the 40-byte header + echo copy-back)
  uint64_t heap = 0;
  CHECK(tinynv_gsp_free_heap(g, &heap) == 0, "free_heap: %s", tinynv_last_error());

  // sensors: the 16-byte controls, then a field read out of the 3152-byte block
  CHECK(tinynv_gsp_sensors_init(g, 0xff, 100) == 0, "sensors_init: %s", tinynv_last_error());
  CHECK(atomic_load(&P.saw_rusd16), "the 16-byte RUSD init control did not reach the peer");
  CHECK(g->gsp.rusd_mem.size >= TINYNV_RPC610_RUSD_SIZE, "the RUSD buffer is %llu bytes, not 3152",
        (unsigned long long)g->gsp.rusd_mem.size);
  {
    // the firmware writes the block; here the test does, at the 610 offsets, with valid timestamps
    uint8_t *raw = g->gsp.rusd_mem.dma.va;
    memset(raw, 0, TINYNV_RPC610_RUSD_SIZE);
    uint64_t ts = 4;
    memcpy(raw + 2312, &ts, 8);              // temperatures[0] timestamp
    uint32_t temp = 55;
    memcpy(raw + 2320, &temp, 4);            // its current value
    tinynv_rusd_t out;
    CHECK(tinynv_gsp_sensors_read(g, &out) == 0, "sensors_read: %s", tinynv_last_error());
    CHECK(out.temperatures[0].ts == 4, "the 610 sensor block was not read at its new offsets");
  }
  printf("boot+objects: full 610 boot (FWSEC/scrubber/booter_load), object tree with 40B controls / 376B channels / "
         "56B VA spaces / 3B schedules, GR multi-element record, RUSD 16B controls + 3152B block\n");
}

// ---- 4b. a zero-byte reply is COMPLETE, not the head of a split one ---------------------------------------------------

static void empty_reply_arm(tinynv_gpu_t *g) {
  // the shape silicon answered the unload notice with - the function echoed back and no payload at all. The
  // drain used to read any reply shorter than the request's own size as the first fragment of a split one and wait for
  // continuations that cannot come, so a working unload became a 10 s timeout blaming the firmware. One control call
  // answered empty proves the reply is now taken as complete: it returns, and nothing is written through to the caller.
  uint64_t fh = 0xdeadbeefull;
  atomic_store(&P.reply_empty_once, 1);
  CHECK(tinynv_gsp_free_heap(g, &fh) == 0, "a zero-byte rpc reply was not accepted as complete: %s",
        tinynv_last_error());
  CHECK(fh == 0, "the empty reply wrote through the caller's parameter (%#llx)", (unsigned long long)fh);
  printf("empty reply: a zero-byte answer is taken as complete, not held as the head of a split one\n");
}

// ---- 5. failure containment on the status queue ----------------------------------------------------------------------

static void containment_arm(tinynv_gpu_t *g) {
  // an error log: contained, kept, and the poll reports it
  atomic_store(&P.inject_event, TINYNV_MSG_EVENT_OS_ERROR_LOG);
  usleep(5000);
  CHECK(tinynv_gsp_poll(g) == 1, "the error log was not contained");
  CHECK(g->gsp.err_state && g->gsp.err_fn == TINYNV_MSG_EVENT_OS_ERROR_LOG, "the error log was not kept");
  // an mmu fault: same containment, the first-error record kept
  atomic_store(&P.inject_event, TINYNV_MSG_EVENT_MMU_FAULT_QUEUED);
  usleep(5000);
  CHECK(tinynv_gsp_poll(g) == 1, "the mmu fault was not contained");
  CHECK(g->gsp.err_fn == TINYNV_MSG_EVENT_MMU_FAULT_QUEUED, "the mmu fault was not kept");
  // the removed sequencer event: refused with a version-mismatch diagnostic, never run through the 570 interpreter
  g->gsp.err_state = 0;
  atomic_store(&P.inject_event, TINYNV_MSG_EVENT_GSP_RUN_CPU_SEQUENCER);
  usleep(5000);
  CHECK(tinynv_gsp_poll(g) == 1, "the removed sequencer event was not refused");
  CHECK(strstr(tinynv_last_error(), "610 removed") != NULL, "the sequencer refusal is not the version-mismatch message: %s",
        tinynv_last_error());
  CHECK(g->gsp.cpu_seq_requests == 0, "the 570 sequencer interpreter ran under the 610 profile");
  // an unknown event: refused, not guessed
  atomic_store(&P.inject_event, 4999);
  usleep(5000);
  CHECK(tinynv_gsp_poll(g) == 1, "an unknown event was not refused");
  // a 570-framed record in the 610 queue: the codec refuses it (mixed-version state cannot be consumed)
  {
    tinynv_gsp_t *gsp = &g->gsp;
    uint8_t *mem = (uint8_t *)gsp->queues.view.ptr;
    uint64_t stat_off = gsp->cmd_q_off + gsp->queue_size;
    uint32_t wp;
    memcpy(&wp, mem + stat_off + 16, 4);
    uint8_t *slot = mem + stat_off + 4096 + (uint64_t)wp * 4096;
    memset(slot, 0, 4096);
    // a 570 element header: authTag/aad, then checkSum/seqNum/elemCount at 32/36/40 - no MCTP/NVDM prefix
    tinynv_msg_header_t mh;
    memset(&mh, 0, sizeof mh);
    mh.header_version = TINYNV_MSG_HEADER_VERSION;
    mh.signature = TINYNV_MSG_SIGNATURE_VALID;
    mh.function = TINYNV_MSG_EVENT_OS_ERROR_LOG;
    mh.length = 32;
    memcpy(slot + 48, &mh, sizeof mh);   // the 570 element header is 48 bytes
    wp = (wp + 1) % 63;
    memcpy(mem + stat_off + 16, &wp, 4);
    usleep(5000);
    CHECK(tinynv_gsp_poll(g) == 1, "a 570-framed record was consumed by the 610 queue");
    // the corrupt record is fatal to the queue (as on hardware); the test owns this memory, so skip past it to let
    // the unload arm run. The driver itself never silently skips a record it could not decode.
    uint32_t rp;
    memcpy(&rp, mem + gsp->cmd_q_off + 32, 4);
    rp = (rp + 1) % 63;
    memcpy(mem + gsp->cmd_q_off + 32, &rp, 4);
  }
  printf("containment: error log + mmu fault kept, removed sequencer / unknown event / 570 framing all refused\n");
}

// ---- 6. the unload ---------------------------------------------------------------------------------------------------

static void unload_arm(tinynv_gpu_t *g) {
  // the primary arm mirrors what the live 610 boot showed: the notice is taken, the core parks, the reply is empty.
  atomic_store(&P.unload_parks_only, 1);
  CHECK(tinynv_gpu_unload(g) == 0, "the 610 unload failed: %s", tinynv_last_error());
  CHECK(atomic_load(&P.saw_unload), "the unload notice never reached the peer");
  CHECK(!g->gsp.up, "gsp-rm is still marked up after the unload");
  CHECK(g->flcn.unloaded, "the region was not torn down");
  CHECK(fake_wpr2 == 0, "WPR2 is still up after the unload (%#x)", fake_wpr2);
  CHECK(g->flcn.unload.is_btl1, "booter_unload was not the BTL1 container");
  printf("unload: notice + halt + FWSEC-SB + BTL1 booter_unload, region down (WPR2 = 0)\n");
}

// ---- 7. the retry: a silent boot, then the same profile comes up ------------------------------------------------------

static void retry_arm(void) {
  tinynv_gpu_t g;
  CHECK(dev_setup(&g, &tinynv_runtime_610) == 0, "dev setup: %s", tinynv_last_error());
  setenv("TINYNV_INIT_TIMEOUT", "0.3", 1);
  setenv("TINYNV_BOOT_ATTEMPTS", "2", 1);
  peer_start(&g, 1);   // header only: the start-up notice never comes on the first attempt
  // the first attempt times out; unmake_boot resets the card cold and rebuilds; the second attempt is answered
  // (the test flips the peer to full after the first timeout would have fired)
  atomic_store(&P.mode, 1);
  // a watcher thread flips the peer to full once the first attempt has timed out and the queues are rebuilt
  pthread_t watch;
  struct { tinynv_gpu_t *g; } *wctx = malloc(sizeof *wctx);
  wctx->g = &g;
  extern void *retry_watcher(void *);
  pthread_create(&watch, NULL, retry_watcher, wctx);
  pthread_detach(watch);
  int rc = tinynv_gpu_boot_firmware(&g);
  CHECK(rc == 0, "the retried 610 boot failed: %s", tinynv_last_error());
  CHECK(g.profile == &tinynv_runtime_610, "the retry changed the frozen profile");
  CHECK(g.gsp.up, "the retried boot did not come up");
  peer_stop();
  dev_teardown(&g);
  unsetenv("TINYNV_INIT_TIMEOUT");
  unsetenv("TINYNV_BOOT_ATTEMPTS");
  printf("retry: a silent first boot is torn down (reset cold) and the SAME 610 profile comes up on attempt 2\n");
}

void *retry_watcher(void *ctx) {
  tinynv_gpu_t *g = ((struct { tinynv_gpu_t *g; } *)ctx)->g;
  free(ctx);
  // wait for the first attempt's timeout, then the wipe (cmd_q_off zeroed), then the rebuild placing the queues
  // again - only then is the new ring memory the peer's to attach to.
  for (int i = 0; i < 5000; i++) {
    if (g->gsp.init_timed_out) break;
    usleep(1000);
  }
  for (int i = 0; i < 5000; i++) {
    if (!g->gsp.cmd_q_off) break;
    usleep(1000);
  }
  for (int i = 0; i < 5000; i++) {
    if (g->gsp.cmd_q_off && g->gsp.queues.view.ptr) break;
    usleep(1000);
  }
  atomic_fetch_add(&P.generation, 1);
  atomic_store(&P.mode, 2);
  return NULL;
}

// ---- 8. the rollback gate: a SEPARATE clean process selects 570 after the 610 session --------------------------------
// The design defines rollback as a separate clean process selecting 570 after a successful shutdown (or physical
// recovery) - never swapping live firmware under active queues. Offline, that is a forked child with a fresh device,
// mm and frozen 570 profile, after the parent's 610 session was fully torn down. The child inherits no driver state
// it uses: it builds everything anew. (The on-silicon separate-process rollback is plan §6.)
static int rollback_child(void) {
  tinynv_gpu_t g;
  int bad = 0;
  if (dev_setup(&g, &tinynv_runtime_570)) { printf("  FAIL(rollback child): dev setup: %s\n", tinynv_last_error()); return 1; }
  if (tinynv_gpu_init_sw(&g)) { printf("  FAIL(rollback child): 570 software init: %s\n", tinynv_last_error()); bad = 1; }
  if (g.profile != &tinynv_runtime_570) { printf("  FAIL(rollback child): wrong profile\n"); bad = 1; }
  // the 570 cached args are the 72-byte layout: no 610 queue-geometry fields (offset 32 is srInitArguments, zero).
  // (alloc_boot_mem rounds the block to a page, so the allocation size is not the struct size.)
  uint32_t at32 = 0;
  nv_rd_block(&g.gsp.rm_args.view, 32, &at32, 4);
  if (at32 != 0) { printf("  FAIL(rollback child): 570 args carry 610 geometry at 32 (%#x)\n", at32); bad = 1; }
  if (g.flcn.booter.is_btl1) { printf("  FAIL(rollback child): 570 booter parsed as BTL1\n"); bad = 1; }
  dev_teardown(&g);
  return bad;
}

static void rollback_arm(void) {
  pid_t pid = fork();
  if (pid == 0) _exit(rollback_child());
  int status = 0;
  waitpid(pid, &status, 0);
  CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0, "the separate-process 570 rollback failed (status %#x)", status);
  printf("rollback: a separate clean process selects 570 and its software init succeeds after the 610 session\n");
}

int main(int argc, char **argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  const char *bundle = argc > 1 ? argv[1] : "../build/gsp610-runtime/bundle";
  // the firmware root is the bundle itself: it carries nvidia/ad102/gsp-610.43.03/ AND (via the shared root) 570.
  // For the 570 rollback arm the root must also carry nvidia/<chip>/gsp; point TINYNV_FW_DIR at the real root and
  // let the 610 bundle be found through it. The bundle arg is used for the manifest cross-check.
  setenv("TINYNV_FW_DIR", bundle, 1);
  unsetenv("TINYNV_FAILPOINT");
  unsetenv("TINYNV_GSP_RUNTIME");
  window_mem = calloc(1, WINDOW);
  build_normal_rom(rom);

  manifest_arm(bundle);

  tinynv_gpu_t g;
  if (dev_setup(&g, &tinynv_runtime_610)) { printf("  FAIL: dev setup: %s\n", tinynv_last_error()); return 1; }
  sw_init_arm(&g);
  dev_teardown(&g);

  failpoint_arm();

  // the boot/objects/containment/unload session on one device
  memset(regs, 0, sizeof regs);
  fake_wpr2 = 0; gsp_starts = sec2_starts = 0;
  if (dev_setup(&g, &tinynv_runtime_610)) { printf("  FAIL: dev setup: %s\n", tinynv_last_error()); return 1; }
  boot_and_objects_arm(&g);
  empty_reply_arm(&g);
  containment_arm(&g);
  unload_arm(&g);
  peer_stop();
  dev_teardown(&g);

  retry_arm();

  // the 570 rollback needs the 570 firmware tree; the bundle root only carries the 610 subtree. Switch the root for
  // that arm (fw_root is read per load); when no 570 tree is present, report the gap honestly.
  const char *root570 = argc > 2 ? argv[2] : "third_party/firmware";
  setenv("TINYNV_FW_DIR", root570, 1);
  tinynv_blob_t probe;
  if (tinynv_fw_load("ad102", "booter_load-" TINYNV_FW_VER ".bin", NULL, &probe) == 0) {
    tinynv_fw_free(&probe);
    rollback_arm();
  } else {
    printf("rollback: SKIPPED - no 570 firmware tree at %s (an environmental gap, not a pass)\n", root570);
  }

  printf(fails ? "lifecycle610: %d checks FAILED\n" : "lifecycle610: all checks passed (offline; no GPU, mock peer only)\n",
         fails);
  return fails ? 1 : 0;
}
