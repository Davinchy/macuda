// Placing GSP-RM's firmware in memory, and describing where it is.
//
// This mirrors tinygrad's NV_GSP.init_sw allocation by allocation, because the addresses end up inside the structures the
// boot firmware reads: the chain of trust message names the metadata block, the metadata block names the image, and the
// image is described by a page table the GPU walks. Allocate in a different order and every one of those addresses moves,
// which is why the replay checks each allocation's size against the recording and hands back the recorded address.
//
// Nothing here touches the GPU. It is all host memory and firmware files, which is what makes it testable offline.
#include "gpu.h"
#include "internal.h"
#include "nv_regs.h"
#include "rpc610.h"   // tinynv_rpc_checksum: the XOR-fold shared with the 610 element codec
#include "gsp610_host.h"
#include "fw610.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MB (1ull << 20)
#define PAGE 0x1000ull

// the PFB sysmem-flush address registers are absent from the generated nv_regs.h. Offsets and fields from the
// pinned 610 swref/published/ampere/ga100/dev_fb.h: LO 0x100C10 ADR_39_08 (31:0, address >> 8); HI 0x100C40 ADR_63_40
// (23:0, mask 0x7F, address >> 40).
#define NV_PFB_NISO_FLUSH_SYSMEM_ADDR         0x00100C10u
#define NV_PFB_NISO_FLUSH_SYSMEM_ADDR_HI      0x00100C40u
#define NV_PFB_NISO_FLUSH_SYSMEM_ADDR_HI_MASK 0x7Fu

// the frozen profile decides every version-dependent branch below. Pointer identity, not a string compare:
// the selector hands out exactly these two descriptors and nothing else can become one (runtime_profile.c).
static int is610(const tinynv_gpu_t *g) { return g->profile == &tinynv_runtime_610; }
static int gsp_schedule_group(tinynv_gpu_t *g, uint32_t client, uint32_t group);

// The staged generic-bootloader ucode for the 610 host-request executor (loaded at software init, fw610).
static tinynv_gsp610_bl_ucode_t gsp610_bl(const tinynv_gsp_t *gsp) {
  tinynv_gsp610_bl_ucode_t bl = {.image = gsp->gbl_fw.data, .image_len = gsp->gbl_fw.size,
                                 .start_tag = gsp->gbl_start_tag, .code_size = gsp->gbl_code_size};
  return bl;
}

// The firmware this driver is pinned to. The hashes are the version: a firmware image is code the GPU's secure boot
// executes, so the file's name is not evidence of anything and the content is checked before it is used.
static const char *GSP_SHA = "a8c3ebeed280323aedb51c061f321e73379cce7a9ae643a33dd03915df027f7f";

// GSP-RM itself ships once, under ga102, and runs on every architecture here. Its RISC-V bootloader does NOT: there is
// one per chip, under that chip's own directory, and they are different images with different hashes. This was a single
// constant pinned to Blackwell's, which meant an Ampere card loaded its own correct bootloader and then had it refused
// for hashing to something other than a Blackwell image - a refusal that reads like a corrupt download.
static const char *bl_sha(const char *fw_name) {
  if (!strcmp(fw_name, "gb202")) return "d40b48e431d1707dc77af3605db358ed7a32ebfc2830eb74de2eddb4d3025071";
  if (!strcmp(fw_name, "ga102")) return "82428f532240727e95bb3083fbaaba9b2cc7b937314323f2d546ce7245f27fad";
  if (!strcmp(fw_name, "ad102")) return "65ab2e6b6e0fca95365c4deac79a34582abcfeb15b6ae234138f22e7183118a8";
  return NULL;
}

static uint64_t round_up(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }
static uint64_t round_down(uint64_t v, uint64_t a) { return v / a * a; }

// chip names are ascii and the firmware section names are their lowercase form; not locale's business
static char lower_ascii(char c) { return c >= 'A' && c <= 'Z' ? (char)(c - 'A' + 'a') : c; }

// NVIDIA names the memory regions it hands its operating system with short ascii strings packed into a 64 bit field,
// most significant byte first: "LOGINIT", "RMARGS" and so on.
static uint64_t id8_of(const char *s) {
  uint64_t v = 0;
  for (; *s; s++) v = (v << 8) | (uint8_t)*s;
  return v;
}

// GSP-RM can ask the driver to touch registers on its behalf: a short program of writes, read-modify-writes, polls and
// delays, sent as a stream of words. This exists because a few sequences have to run from the host side of the bus.
//
// The recorded 5090 boot never sends one -- there is not a single register write between the chain of trust and the
// windows being retargeted -- so this path is written from the oracle and has not been exercised by the replay. It is
// here rather than left as a failure because a board that does send one would otherwise stop dead.
// Returns -1 with the reason in tinynv_last_error if the sequence could not be completed. It used to be void and
// `return` silently from every falcon step that failed, after which the caller sat in a 60 s wait for a start-up notice
// that could never come and reported only the wait - the real failure was overwritten before anyone read it. On the
// 3060 (2026-09-17) that was three boots in a row with nothing between the sequence line and the timeout.
static int seq_fail(uint32_t op, unsigned nops, size_t i, size_t have, const char *why) {
  char inner[256]; snprintf(inner, sizeof inner, "%s", tinynv_last_error());
  fprintf(stderr, "tinynv: register sequence ABORTED at operation %u (op %#x, word %zu of %zu): %s%s%s\n",
          nops, op, i, have, why, *inner ? " - " : "", inner);
  return tinynv_fail("register sequence aborted at operation %u (op %#x): %s%s%s", nops, op, why, *inner ? " - " : "", inner);
}
static int run_cpu_seq(tinynv_gpu_t *g, const uint8_t *msg, size_t len) {
  // say so, every time. The recorded 5090 boot never takes this path; the recorded 3060 boot takes it once, and replays
  // through this code with no divergence, so on an Ampere card one request is the expected shape of a boot and a second
  // is news.
  g->gsp.cpu_seq_requests++;
  fprintf(stderr, "tinynv: gsp-rm asked the driver to run a register sequence (%zu bytes, request %u)\n", len, g->gsp.cpu_seq_requests);
  // rpc_run_cpu_sequencer_v17_00 is FORTY bytes, not thirty-two: bufferSizeDWord at 0, cmdIndex at 4, then a register
  // save area of EIGHT words, and only then the commands. This read 32 and took the command count from the last word of
  // that, which is regSaveArea[5], and then started decoding at regSaveArea[6]. Nothing caught it because the recorded
  // 5090 boot never sends a sequence at all, so until an Ampere card asked for one this function had never run.
  const size_t hdr = 40;
  if (len < hdr) return tinynv_fail("register sequence request is %zu bytes, shorter than its %zu byte header", len, hdr);
  uint32_t cmd_index;
  memcpy(&cmd_index, msg + 4, 4);
  const uint32_t *w = (const uint32_t *)(const void *)(msg + hdr);
  size_t have = (len - hdr) / 4, i = 0;
  if (cmd_index < have) have = cmd_index;

#define NEXT(out) do { if (i >= have) return seq_fail(op, nops, i, have, "the sequence ended inside an operation"); memcpy(&(out), &w[i++], 4); } while (0)
#define STEP(call, why) do { if (call) return seq_fail(op, nops, i, have, why); } while (0)
  uint32_t op = 0; unsigned nops = 0;
  while (i < have) {
    uint32_t addr, val, mask, us;
    NEXT(op); nops++;
    switch (op) {
      case 0x0: NEXT(addr); NEXT(val); tinynv_wr32(&g->dev, addr, val); break;
      case 0x1:
        NEXT(addr); NEXT(val); NEXT(mask);
        tinynv_wr32(&g->dev, addr, (tinynv_rd32(&g->dev, addr) & ~mask) | (val & mask));
        break;
      case 0x2: {
        uint32_t ignored;
        NEXT(addr); NEXT(mask); NEXT(val); NEXT(ignored); NEXT(ignored);
        STEP(tinynv_wait_reg(&g->dev, addr, mask, val, 10000, "a register poll gsp-rm asked for"), "a register poll never matched");
        break;
      }
      case 0x3: {
        NEXT(us);
        struct timespec ts = {.tv_sec = us / 1000000, .tv_nsec = (long)(us % 1000000) * 1000L};
        nanosleep(&ts, NULL);
        break;
      }
      // Save a register into the request's own save area. The oracle decodes that area into a local copy and writes
      // there, so the stored value goes nowhere and is never read back - but the READ happens, and on a replay a read
      // that does not happen is a divergence. So the read is what matters and the value is kept beside it.
      case 0x4: {
        uint32_t index;
        NEXT(addr); NEXT(index);
        uint32_t v = tinynv_rd32(&g->dev, addr);
        if (index < sizeof(g->gsp.cpu_seq_saved) / sizeof(g->gsp.cpu_seq_saved[0])) g->gsp.cpu_seq_saved[index] = v;
        break;
      }
      // The remaining four ask the driver to drive the GSP falcon itself, which is why flcn.c exposes these.
      case 0x5:
        STEP(tinynv_flcn_reset(g, g->flcn.falcon, 0), "the gsp falcon did not come out of reset");
        tinynv_flcn_disable_ctx_req(&g->dev, g->flcn.falcon);
        break;
      case 0x6: tinynv_flcn_start_cpu(&g->dev, g->flcn.falcon); break;
      case 0x7: STEP(tinynv_flcn_wait_cpu_halted(&g->dev, g->flcn.falcon), "the gsp falcon never halted"); break;
      // Resume: the gsp falcon comes back on its riscv core with gsp-rm's arguments in the mailbox, then SEC2 is started
      // and has to say it handed off. SEC2's own mailbox reports a refusal, as it does for booter_load.
      case 0x8: {
        STEP(tinynv_flcn_reset(g, g->flcn.falcon, 1), "the gsp falcon did not come out of reset on its riscv core");
        tinynv_wr32(&g->dev, NV_PGSP_FALCON_MAILBOX0, (uint32_t)g->gsp.libos_args_sysmem);
        tinynv_wr32(&g->dev, NV_PGSP_FALCON_MAILBOX1, (uint32_t)(g->gsp.libos_args_sysmem >> 32));
        tinynv_flcn_start_cpu(&g->dev, g->flcn.sec2);
        STEP(tinynv_wait_reg(&g->dev, NV_PGC6_BSI_SECURE_SCRATCH_14,
                             1u << NV_PGC6_BSI_SECURE_SCRATCH_14_BOOT_STAGE_3_HANDOFF_LO,
                             1u << NV_PGC6_BSI_SECURE_SCRATCH_14_BOOT_STAGE_3_HANDOFF_LO, 10000,
                             "waiting for sec2 to hand off"), "sec2 never handed off");
        uint32_t mbx = tinynv_rd32(&g->dev, g->flcn.sec2 + NV_PFALCON_FALCON_MAILBOX0);
        if (mbx) { tinynv_fail("sec2 refused the sequencer's resume step: mailbox %#x", mbx); return seq_fail(op, nops, i, have, "sec2 refused"); }
        break;
      }
      default: tinynv_fail("gsp-rm asked for register operation %#x, which this driver does not know", op); return seq_fail(op, nops, i, have, "unknown operation");
    }
  }
#undef NEXT
#undef STEP
  fprintf(stderr, "tinynv: register sequence done: %u operations, %zu of %zu words\n", nops, i, have);
  return 0;
}

// --- the shared ring ------------------------------------------------------------------------------------------------
//
// Both directions live in one allocation the GPU reaches through its own page table. Each direction has a header saying
// how big its slots are and how many there are, followed by the slots themselves. The writer advances a write pointer in
// its own header; the reader advances a read pointer that, by this queue's convention, is stored in the *other*
// direction's header. So neither side ever writes to the other's header, and a single shared mapping is enough.

// the 610 libos has three tasks the 570 one does not - DEBUG, ROOT and RM_STATE_MONITOR - and the pinned
// 610 kgspSetupLibosLogs provides a log region for EACH of them (kernel_gsp.c logInitValues: LOGDBG 64K, LOGROOT 4K,
// LOGRMON 4K, on top of INIT/INTR/RM/MNOC/KRNL). The 570 baremetal list is only INIT/INTR/RM/MNOC (+KRNL), which is
// what the 570 path keeps. Offsets inside the 2 MiB log allocation; the region ORDER matches the vendor's.
static const struct { const char *name; uint64_t off, size; } LOGS610[] = {
  {"LOGINIT", 0x00000, 0x10000}, {"LOGINTR", 0x10000, 0x10000}, {"LOGRM", 0x20000, 0x10000},
  {"LOGMNOC", 0x30000, 0x10000}, {"LOGDBG", 0x40000, 0x10000}, {"LOGROOT", 0x50000, 0x1000},
  {"LOGRMON", 0x51000, 0x1000}, {"LOGKRNL", 0x60000, 0x10000},
};

// when GSP-RM never writes the status queue header, the only witnesses are its own log buffers (libos
// writes LOGINIT/LOGRM there from its first instructions) and the falcon's state. Read-only, failure path only -
// a successful boot never reaches this, so the recorded 570 operation stream is unchanged.
static void gsp_silent_boot_diag(tinynv_gpu_t *g) {
  tinynv_gsp_t *gsp = &g->gsp;
  tinynv_dev_t *d = &g->dev;
  if (!d->pci || !d->pci->live) return;
  uint32_t mb0 = tinynv_rd32(d, NV_PGSP_FALCON_MAILBOX0);
  uint32_t mb1 = tinynv_rd32(d, NV_PGSP_FALCON_MAILBOX0 + 4);
  uint32_t os = tinynv_rd32(d, g->flcn.falcon + NV_PFALCON_FALCON_OS);
  uint32_t cpuctl = tinynv_rd32(d, g->flcn.falcon + NV_PRISCV_RISCV_CPUCTL);
  fprintf(stderr, "tinynv: silent-boot diag: gsp mailbox0 %#x mailbox1 %#x falcon_os %#x cpuctl %#x (active %u, halted %u)\n",
          mb0, mb1, os, cpuctl,
          NV_GET(cpuctl, NV_PRISCV_RISCV_CPUCTL, ACTIVE_STAT), NV_GET(cpuctl, NV_PRISCV_RISCV_CPUCTL, HALTED));
  // did GSP-RM consume the pre-filled command-ring records (sysinfo/registry)? its read pointer lives in the
  // status header's rx area; the command header's own writePtr is ours. Dump both headers raw.
  if (gsp->queues.size && gsp->cmd_q_off) {
    uint32_t hdr[8];
    nv_rd_block(&gsp->queues.view, gsp->cmd_q_off, hdr, sizeof(hdr));
    fprintf(stderr, "tinynv: silent-boot diag: cmd hdr ver=%u size=%#x msgSize=%#x msgCount=%u writePtr=%u flags=%#x rxHdrOff=%#x entryOff=%#x\n",
            hdr[0], hdr[1], hdr[2], hdr[3], hdr[4], hdr[5], hdr[6], hdr[7]);
    nv_rd_block(&gsp->queues.view, gsp->cmd_q_off + gsp->queue_size, hdr, sizeof(hdr));
    fprintf(stderr, "tinynv: silent-boot diag: stat hdr ver=%u size=%#x msgSize=%#x msgCount=%u writePtr=%u flags=%#x rxHdrOff=%#x entryOff=%#x\n",
            hdr[0], hdr[1], hdr[2], hdr[3], hdr[4], hdr[5], hdr[6], hdr[7]);
  }
  // the exact blocks handed to the firmware: WPR meta, cached RM args, libos regions. Read back from where the
  // firmware reads them, so a placement/encoding mistake is visible rather than assumed.
  if (gsp->wpr_meta.size) {
    uint64_t m[32];
    nv_rd_block(&gsp->wpr_meta.view, 0, m, sizeof m);
    fprintf(stderr, "tinynv: silent-boot diag: wpr meta @%#llx:", (unsigned long long)gsp->wpr_meta.addrs[0]);
    for (int k = 0; k < 32; k++) fprintf(stderr, "%s%016llx", k % 4 ? " " : "\n  ", (unsigned long long)m[k]);
    fprintf(stderr, "\n");
  }
  if (gsp->rm_args.size) {
    uint64_t a[20];
    size_t n = gsp->rm_args.size < sizeof a ? gsp->rm_args.size : sizeof a;
    nv_rd_block(&gsp->rm_args.view, 0, a, n);
    fprintf(stderr, "tinynv: silent-boot diag: rm args @%#llx (%zu B):", (unsigned long long)gsp->rm_args.addrs[0], gsp->rm_args.size);
    for (size_t k = 0; k < n / 8; k++) fprintf(stderr, "%s%016llx", k % 4 ? " " : "\n  ", (unsigned long long)a[k]);
    fprintf(stderr, "\n");
  }
  if (gsp->libos_args.size) {
    uint64_t r[40];
    nv_rd_block(&gsp->libos_args.view, 0, r, sizeof r);
    fprintf(stderr, "tinynv: silent-boot diag: libos args @%#llx:", (unsigned long long)gsp->libos_args.addrs[0]);
    for (int k = 0; k < 40; k++) fprintf(stderr, "%s%016llx", k % 4 ? " " : "\n  ", (unsigned long long)r[k]);
    fprintf(stderr, "\n");
  }
  // read back what booter_load unpacked into the WPR image slot, through the retargeting BAR1 window
  // (read-only). The first words must be the libos boot code; the embedded task ELFs must start with \x7fELF.
  if (is610(g) && gsp->wpr_meta.size) {
    uint64_t meta[32];
    nv_rd_block(&gsp->wpr_meta.view, 0, meta, sizeof meta);
    uint64_t gsp_off = meta[17]; // gspFwOffset (tinynv_wpr_meta_t word order)
    static const uint64_t probes[] = {0, 0x6e000, 0x120c000, 0x5052000};
    // the probes read the BAR1 view at full-VRAM offsets, reachable only when the window spans the image
    // (a large BAR). nv_rd32 does not bounds-check a .ptr-backed view, and the offline mock's window is 128 MiB, so
    // reading gsp_off there runs past the mapping and faults. Skip the FB probes when the window cannot reach the
    // last probe; on a large-BAR card the window spans VRAM and every probe reads exactly as before.
    if (gsp_off + probes[sizeof probes / sizeof probes[0] - 1] + 16 <= d->vram.size) {
      for (size_t k = 0; k < sizeof probes / sizeof probes[0]; k++) {
        uint32_t w[4];
        for (int j = 0; j < 4; j++) w[j] = nv_rd32(&d->vram, gsp_off + probes[k] + 4 * j);
        fprintf(stderr, "tinynv: silent-boot diag: fb image @%#llx+%#llx: %08x %08x %08x %08x\n",
                (unsigned long long)gsp_off, (unsigned long long)probes[k], w[0], w[1], w[2], w[3]);
      }
      // control: a NON-WPR FB address. If this also reads ff, the window itself is not answering (the firmware owns
      // BAR1 while it runs); if it reads data, the ff above is the WPR read protection, not a missing image.
      uint32_t c[4];
      for (int j = 0; j < 4; j++) c[j] = nv_rd32(&d->vram, 0x100000 + 4 * j);
      fprintf(stderr, "tinynv: silent-boot diag: fb non-wpr @0x100000: %08x %08x %08x %08x\n", c[0], c[1], c[2], c[3]);
    }
  }
  if (!gsp->logbuf.size || gsp->logbuf.size < 5 * 0x10000) return;
  uint8_t *buf = malloc(0x10000);
  if (!buf) return;
  static const char *LOGS[] = {"INIT", "INTR", "RM", "MNOC", "KRNL"};
  const int nlogs = is610(g) ? 8 : 5;
  for (int i = 0; i < nlogs; i++) {
    const uint64_t loff = is610(g) ? LOGS610[i].off : 0x10000ull * i;
    const size_t lsize = is610(g) ? (size_t)LOGS610[i].size : 0x10000;
    const char *lname = is610(g) ? LOGS610[i].name : LOGS[i];
    nv_rd_block(&gsp->logbuf.view, loff, buf, lsize);
    // libos logs are BINARY nvlog entries (packed u64: metadata VA 47:0, totalArgs 55:48, taskId 63:56 -
    // libos_log.h), not text. Count non-zero words and dump them raw; a text scan stops at the first zero byte
    // inside the first entry and reads as "1 byte written". Word 0 is the ring's put pointer; records are packed
    // from word 1 up to it, so the NEWEST entries (where RM logs its start-up result) sit at the high end. Dump the
    // whole committed region, not a fixed prefix - a 96-word cap hid the RM tail that names the init failure.
    const uint64_t *w = (const uint64_t *)buf;
    size_t used = 0;
    for (size_t k = 0; k < lsize / 8; k++) if (w[k]) used = k + 1;
    if (!used) { fprintf(stderr, "tinynv: gsp log %s: empty\n", lname); continue; }
    fprintf(stderr, "tinynv: gsp log %s: %zu non-zero u64 words (last at %#zx, put=%llu):\n", lname, used,
            (used - 1) * 8, (unsigned long long)w[0]);
    for (size_t k = 0; k < used; k++)
      fprintf(stderr, "  [%04zu] %016llx  task=%llu args=%llu meta=%#llx\n", k, (unsigned long long)w[k],
              (unsigned long long)(w[k] >> 56), (unsigned long long)((w[k] >> 48) & 0xff),
              (unsigned long long)(w[k] & 0xffffffffffffull));
  }
  free(buf);
}

// a boot stage that fails with the firmware ALREADY UP leaves its reason inside RM's own log rings, and
// nothing on the 610 path read them: the silent-boot dump above only runs when the status queue never initialises,
// and its env-var arm only after a successful boot. A refusal several stages in - the first channel allocation, say -
// therefore reported the envelope's status code and nothing else, while RM's own NV_PRINTF(LEVEL_ERROR) explaining
// itself sat in LOGRM unread. Called from the boot-stage failure path. Read-only, and 610-gated HERE so the gate
// cannot be forgotten at a call site and the 570 profile issues no operation it did not issue before.
void tinynv_gsp_boot_stage_diag(tinynv_gpu_t *g) {
  if (!is610(g)) return;
  fprintf(stderr, "tinynv: a boot stage failed with gsp-rm already up; reading its own log rings for the reason\n");
  gsp_silent_boot_diag(g);
}

// Wait for a queue's header to say its slots have appeared, then take our copy of it from memory rather than assuming
// what we wrote. For the command queue that is immediate: we wrote it. For the status queue it is the whole of GSP-RM's
// start-up, which on this card is several seconds of spinning.
static int rpcq_attach(tinynv_gpu_t *g, tinynv_rpcq_t *q, uint64_t base, const char *what) {
  tinynv_gsp_t *gsp = &g->gsp;
  q->base = base;
  q->seq = 0;
  double deadline = tinynv_now_s() + 60.0;
  uint32_t entry_off;
  do {
    entry_off = nv_rd32(&gsp->queues.view, base + offsetof(tinynv_msgq_tx_header_t, entryOff));
  } while (entry_off != PAGE && tinynv_now_s() < deadline);
  if (entry_off != PAGE) {
    if (!strcmp(what, "status")) gsp_silent_boot_diag(g);
    return tinynv_fail("the %s queue never initialised: its entry offset read %#x, not %#llx",
                       what, entry_off, (unsigned long long)PAGE);
  }
  nv_rd_block(&gsp->queues.view, base, &q->tx, sizeof(q->tx));
  q->entries = base + q->tx.entryOff;
  if (!q->tx.msgSize || !q->tx.msgCount) return tinynv_fail("the %s queue describes %u slots of %u bytes", what,
                                                            q->tx.msgCount, q->tx.msgSize);
  return 0;
}

// The element header's own checksum: exclusive-or of the record as 64 bit words, folded in half. Zero length tails are
// padded, which matters because a record is only ever a whole number of slots on the wire but is checksummed at its
// natural length.
// Shared with the 610 element codec (rpc610.c): the algorithm is byte-identical in 570 and 610 (_checkSum32), so it
// lives here once. renamed it from static rpc_checksum; the 570 call sites are unchanged.
uint32_t tinynv_rpc_checksum(const uint8_t *data, size_t n) {
  uint64_t sum = 0;
  size_t whole = n / 8 * 8;
  for (size_t i = 0; i < whole; i += 8) {
    uint64_t w;
    memcpy(&w, data + i, 8);
    sum ^= w;
  }
  if (n > whole) { // the odd tail, zero extended, which is what padding to eight bytes amounts to
    uint64_t w = 0;
    memcpy(&w, data + whole, n - whole);
    sum ^= w;
  }
  return (uint32_t)(sum >> 32) ^ (uint32_t)sum;
}

// Put one record in the ring and ring the doorbell. A record that runs off the end of the ring wraps to the start, which
// is why the copy is in two parts.
static int rpc_send_record(tinynv_gpu_t *g, uint32_t func, const void *payload, size_t len) {
  tinynv_gsp_t *gsp = &g->gsp;
  tinynv_rpcq_t *q = &gsp->cmd_q;
  size_t hdrs = sizeof(tinynv_msg_element_t) + sizeof(tinynv_msg_header_t);
  uint32_t elem_count = (uint32_t)((len + hdrs + q->tx.msgSize - 1) / q->tx.msgSize);
  size_t total = (size_t)elem_count * q->tx.msgSize;

  uint8_t *rec = calloc(1, total);
  if (!rec) return tinynv_fail("out of memory for a %zu byte rpc record", total);

  tinynv_msg_header_t *mh = (tinynv_msg_header_t *)(rec + sizeof(tinynv_msg_element_t));
  mh->header_version = TINYNV_MSG_HEADER_VERSION;
  mh->signature = TINYNV_MSG_SIGNATURE_VALID;
  mh->rpc_result = mh->rpc_result_private = TINYNV_MSG_RESULT_PENDING;
  mh->function = func;
  mh->length = (uint32_t)(len + sizeof(tinynv_msg_header_t));
  memcpy(rec + hdrs, payload, len);

  tinynv_msg_element_t *eh = (tinynv_msg_element_t *)rec;
  eh->elemCount = elem_count;
  eh->seqNum = q->seq;
  // the checksum covers the record at its natural length, with the checksum field still zero
  eh->checkSum = tinynv_rpc_checksum(rec, sizeof(*eh) + mh->length);

  uint64_t ring = q->entries, ring_len = (uint64_t)q->tx.msgSize * q->tx.msgCount;
  uint32_t wp = nv_rd32(&gsp->queues.view, q->base + offsetof(tinynv_msgq_tx_header_t, writePtr));
  uint64_t off = (uint64_t)wp * q->tx.msgSize;
  size_t first = total < ring_len - off ? total : (size_t)(ring_len - off);
  nv_wr_block(&gsp->queues.view, ring + off, rec, first);
  if (first < total) nv_wr_block(&gsp->queues.view, ring, rec + first, total - first);
  nv_wr32(&gsp->queues.view, q->base + offsetof(tinynv_msgq_tx_header_t, writePtr), (wp + elem_count) % q->tx.msgCount);
  __sync_synchronize(); // the doorbell must not be seen before the record it announces

  // TINYNV_RPC_TRACE names every record and where it landed. A divergence in a replay is reported as an offset in the
  // shared allocation, which says nothing about which call went wrong; this is what turns that offset into a function.
  if (getenv("TINYNV_RPC_TRACE"))
  {
    // an allocation names the class it is creating, which is the only part that identifies it at a glance
    char what[64] = "";
    if (func == TINYNV_MSG_FUNCTION_GSP_RM_ALLOC && len >= sizeof(tinynv_rpc_rm_alloc_t)) {
      tinynv_rpc_rm_alloc_t a;
      memcpy(&a, payload, sizeof(a));
      snprintf(what, sizeof(what), " class %#x object %#x parent %#x", a.hClass, a.hObject, a.hParent);
    }
    fprintf(stderr, "tinynv: rpc function %u, %zu bytes payload, %u element(s) at mem:0 %#llx (seq %u)%s\n",
            func, len, elem_count, (unsigned long long)(ring + off), q->seq, what);
  }

  q->seq++;
  tinynv_wr32(&g->dev, NV_PGSP_QUEUE_HEAD(0), 0);
  free(rec);
  return 0;
}

// A payload too long for one record is continued in further records. The first carries the real function number and the
// rest a continuation marker, which is how the far side knows to join them back together.
// the 610 profile frames records with the MCTP/NVDM element codec (rpc610) and splits at the vendor's ONE
// PAGE, not the queue's 16-element ceiling. The 570 path below is unchanged operation for operation.
static int rpc_send_record610(tinynv_gpu_t *g, uint32_t func, const void *payload, size_t len) {
  tinynv_gsp_t *gsp = &g->gsp;
  tinynv_rpcq_t *q = &gsp->cmd_q;
  if (tinynv_failpoint("610-rpc-send")) return -1;
  // A record may span up to sixteen page-sized queue elements; the codec enforces
  // the negotiated ceiling, including its transport header.
  uint8_t *rec = calloc(1, TINYNV_RPC610_ELEM_SIZE_MAX);
  if (!rec) return tinynv_fail("out of memory for a 610 rpc record");
  uint32_t elem_count = 0;
  size_t total = 0;
  if (tinynv_rpc610_encode(rec, TINYNV_RPC610_ELEM_SIZE_MAX, func, payload, len, q->seq, q->tx.msgSize,
                           &elem_count, &total)) { free(rec); return -1; }

  uint64_t ring = q->entries, ring_len = (uint64_t)q->tx.msgSize * q->tx.msgCount;
  uint32_t wp = nv_rd32(&gsp->queues.view, q->base + offsetof(tinynv_msgq_tx_header_t, writePtr));
  uint64_t off = (uint64_t)wp * q->tx.msgSize;
  size_t first = total < ring_len - off ? total : (size_t)(ring_len - off);
  nv_wr_block(&gsp->queues.view, ring + off, rec, first);
  if (first < total) nv_wr_block(&gsp->queues.view, ring, rec + first, total - first);
  nv_wr32(&gsp->queues.view, q->base + offsetof(tinynv_msgq_tx_header_t, writePtr), (wp + elem_count) % q->tx.msgCount);
  __sync_synchronize(); // the doorbell must not be seen before the record it announces
  if (getenv("TINYNV_RPC_TRACE"))
    fprintf(stderr, "tinynv: 610 rpc function %u, %zu bytes payload, %u element(s) at mem:0 %#llx (seq %u)\n",
            func, len, elem_count, (unsigned long long)(ring + off), q->seq);
  q->seq++;
  tinynv_wr32(&g->dev, NV_PGSP_QUEUE_HEAD(0), 0);
  free(rec);
  return 0;
}

static int rpc_send(tinynv_gpu_t *g, uint32_t func, const void *payload, size_t len) {
  if (is610(g)) {
    const size_t max = tinynv_rpc610_first_payload_max();
    const uint8_t *p = payload;
    if (rpc_send_record610(g, func, p, len < max ? len : max)) return -1;
    for (size_t off = max; off < len; off += max)
      if (rpc_send_record610(g, TINYNV_MSG_FUNCTION_CONTINUATION_RECORD, p + off, len - off < max ? len - off : max))
        return -1;
    return 0;
  }
  tinynv_rpcq_t *q = &g->gsp.cmd_q;
  size_t max = (size_t)q->tx.msgSize * 16 - sizeof(tinynv_msg_element_t) - sizeof(tinynv_msg_header_t);
  const uint8_t *p = payload;
  if (rpc_send_record(g, func, p, len < max ? len : max)) return -1;
  for (size_t off = max; off < len; off += max)
    if (rpc_send_record(g, TINYNV_MSG_FUNCTION_CONTINUATION_RECORD, p + off, len - off < max ? len - off : max)) return -1;
  return 0;
}

// names for the wire, so a refusal reads as the thing that was refused rather than a number to look up in a
// header. Only what this driver's boot path actually puts on the wire; anything else says so instead of guessing.
static const char *rpc610_function_name(uint32_t func) {
  switch (func) {
    case TINYNV_MSG_FUNCTION_GSP_RM_ALLOC: return "an object allocation";
    case TINYNV_MSG_FUNCTION_GSP_RM_CONTROL: return "an object control call";
    case TINYNV_MSG_FUNCTION_UNLOADING_GUEST_DRIVER: return "the unload notice";
    default: return "a call";
  }
}

static const char *rpc610_class_name(uint32_t cls) {
  switch (cls) {
    case TINYNV_CLASS_ROOT: return "the root client";
    case TINYNV_CLASS_DEVICE: return "a device";
    case TINYNV_CLASS_SUBDEVICE: return "a subdevice";
    case TINYNV_CLASS_VASPACE: return "a VA space";
    case TINYNV_CLASS_MEMORY_VIRTUAL: return "virtual memory";
    case TINYNV_CLASS_CHANNEL_GROUP: return "a channel group";
    case TINYNV_CLASS_CONTEXT_SHARE: return "a context share";
    case TINYNV_CLASS_GPFIFO_AMPERE: return "an Ampere GPFIFO channel";
    case TINYNV_CLASS_GPFIFO_BLACKWELL: return "a Blackwell GPFIFO channel";
    case TINYNV_CLASS_COMPUTE_AMPERE: return "an Ampere compute object";
    case TINYNV_CLASS_COMPUTE_ADA: return "an Ada compute object";
    case TINYNV_CLASS_DMA_COPY_AMPERE: return "an Ampere copy object";
    case TINYNV_CLASS_DMA_COPY_BLACKWELL: return "a Blackwell copy object";
    case TINYNV_CLASS_USER_SHARED_DATA: return "a user shared data block";
    default: return "an object this driver has no name for";
  }
}

// what a refused allocation put on the wire, next to what gsp-rm handed back. The 512-byte error buffer that
// carries the reason to the caller cannot hold a 376-byte parameter blob, and the blob is the evidence: gsp-rm rewrites
// the fields it owns as its construct runs, in a known order - cid becomes its own counter first thing, then the
// client-supplied internal fields (internalFlags, both notifier descriptors, the process ids) are cleared, and
// hPhysChannelGroup is filled with the TSG it allocated internally to wrap a channel whose parent is a device. Which of
// those happened says how far the handler got before it refused, which is the question the error code alone cannot
// answer: a status of 0x1f arrives from the rpc decode layer and from a dozen places inside the construct alike. And a
// byte that differs where nothing should have touched it is a serializer bug, visible at a glance. stderr, so no length
// limit applies and the live log keeps all of it. Failure path only, and only for the 610 wire format.
static void dump_refused_blob(const char *what, const uint8_t *b, uint32_t len) {
  fprintf(stderr, "tinynv: refusal diag: %s (%u bytes)\n", what, len);
  for (uint32_t i = 0; i < len; i += 16) {
    fprintf(stderr, "tinynv:   %04x:", i);
    for (uint32_t j = i; j < i + 16 && j < len; j++) fprintf(stderr, " %02x", b[j]);
    fprintf(stderr, "\n");
  }
}

static void dump_refused_alloc610(tinynv_gpu_t *g, const tinynv_rpc_rm_alloc_t *ah, const uint8_t *payload,
                                  uint32_t payload_len) {
  tinynv_gsp_t *gsp = &g->gsp;
  fprintf(stderr,
          "tinynv: refusal diag: echoed header hClient %#x hParent %#x hObject %#x hClass %#x status %#x paramsSize %u "
          "flags %#x reserved %#x\n",
          ah->hClient, ah->hParent, ah->hObject, ah->hClass, ah->status, ah->paramsSize, ah->flags, ah->reserved);
  const uint8_t *echo = payload + sizeof *ah;
  uint32_t echo_len = payload_len - (uint32_t)sizeof *ah;
  dump_refused_blob("the parameters this driver sent", gsp->last_alloc_sent, gsp->last_alloc_sent_len);
  dump_refused_blob("the parameters gsp-rm echoed back", echo, echo_len);
  if (ah->hClass != TINYNV_CLASS_GPFIFO_AMPERE && ah->hClass != TINYNV_CLASS_GPFIFO_BLACKWELL) return;
  if (gsp->last_alloc_sent_len != sizeof(tinynv_rpc610_gpfifo_alloc_t) ||
      echo_len < sizeof(tinynv_rpc610_gpfifo_alloc_t))
    return;
  const tinynv_rpc610_gpfifo_alloc_t *s = (const tinynv_rpc610_gpfifo_alloc_t *)gsp->last_alloc_sent;
  tinynv_rpc610_gpfifo_alloc_t e;
  memcpy(&e, echo, sizeof e);
  fprintf(stderr, "tinynv: refusal diag: channel parameters, sent -> echoed. a value gsp-rm changed is one it "
                  "reached; the notes say when it changes each\n");
  fprintf(stderr, "tinynv:   cid               %#10x -> %#-10x replaced by rm's own counter, before any validation\n",
          s->cid, e.cid);
  fprintf(stderr, "tinynv:   internalFlags     %#10x -> %#-10x cleared once the client's own fields are dropped\n",
          s->internalFlags, e.internalFlags);
  fprintf(stderr, "tinynv:   hPhysChannelGroup %#10x -> %#-10x the tsg rm wrapped this channel in, if it got that far\n",
          s->hPhysChannelGroup, e.hPhysChannelGroup);
  fprintf(stderr, "tinynv:   ProcessID         %#10x -> %#-10x cleared with internalFlags\n", s->ProcessID, e.ProcessID);
  fprintf(stderr, "tinynv:   errorNotifierMem  %#10llx -> %#-10llx cleared with internalFlags\n",
          (unsigned long long)s->errorNotifierMem.base, (unsigned long long)e.errorNotifierMem.base);
  fprintf(stderr, "tinynv:   flags             %#10x -> %#-10x read, never rewritten: userd index/page, privilege\n",
          s->flags, e.flags);
  fprintf(stderr, "tinynv:   engineType        %#10x -> %#-10x 1 is gr0, which is what compute runs on\n",
          s->engineType, e.engineType);
  fprintf(stderr, "tinynv:   hVASpace          %#10x -> %#-10x pointer-based; must be the only one of the two set\n",
          s->hVASpace, e.hVASpace);
  fprintf(stderr, "tinynv:   hHandleVASpace    %#10x -> %#-10x new in 610, handle-based; sent as zero\n",
          s->hHandleVASpace, e.hHandleVASpace);
  fprintf(stderr, "tinynv:   hContextShare     %#10x -> %#-10x must be zero: it and hVASpace cannot both be set\n",
          s->hContextShare, e.hContextShare);
  fprintf(stderr, "tinynv:   hObjectError      %#10x -> %#-10x zero on the privileged channel\n", s->hObjectError,
          e.hObjectError);
  fprintf(stderr, "tinynv:   gpFifoEntries     %#10x -> %#-10x ring depth\n", s->gpFifoEntries, e.gpFifoEntries);
  fprintf(stderr, "tinynv:   gpFifoOffset      %#10llx -> %#-10llx the ring's address\n",
          (unsigned long long)s->gpFifoOffset, (unsigned long long)e.gpFifoOffset);
  fprintf(stderr, "tinynv:   subDeviceId       %#10x -> %#-10x one-hot subdevice mask\n", s->subDeviceId, e.subDeviceId);
  const struct { const char *n; const tinynv_memory_desc_t *a, *b; } D[] = {
    {"instanceMem", &s->instanceMem, &e.instanceMem}, {"userdMem", &s->userdMem, &e.userdMem},
    {"ramfcMem", &s->ramfcMem, &e.ramfcMem},         {"mthdbufMem", &s->mthdbufMem, &e.mthdbufMem}};
  for (unsigned i = 0; i < sizeof D / sizeof *D; i++)
    fprintf(stderr, "tinynv:   %-14s base %#llx size %#llx space %u attr %u -> base %#llx size %#llx space %u attr %u\n",
            D[i].n, (unsigned long long)D[i].a->base, (unsigned long long)D[i].a->size, D[i].a->addressSpace,
            D[i].a->cacheAttrib, (unsigned long long)D[i].b->base, (unsigned long long)D[i].b->size,
            D[i].b->addressSpace, D[i].b->cacheAttrib);
  if (memcmp(s, &e, sizeof e) == 0)
    fprintf(stderr, "tinynv: refusal diag: the echo is byte-identical to what was sent, so gsp-rm rewrote nothing: "
                    "its channel construct never ran, and the refusal came from the layer that decoded the message\n");
}

// the 610 status-queue drain. Same ring discipline as 570 (pointer bounds, one ring's worth per call), but
// the record framing is the MCTP/NVDM element codec, the sequence number is checked per record (the 610 host's
// rxSeqNum rule), the removed register sequencer is refused through the host-request dispatcher, and a split reply
// is reassembled: RM echoes the request, so the caller's head+params size is the exact total to wait for.
static int rpc_drain610(tinynv_gpu_t *g, uint32_t want, uint32_t want_total, int *seen, uint8_t **reply,
                        uint32_t *reply_len) {
  tinynv_gsp_t *gsp = &g->gsp;
  tinynv_rpcq_t *q = &gsp->stat_q;
  __sync_synchronize();
  for (uint32_t taken = 0;; taken++) {
    uint32_t rp = nv_rd32(&gsp->queues.view, q->rx_off);
    uint32_t wp = nv_rd32(&gsp->queues.view, q->base + offsetof(tinynv_msgq_tx_header_t, writePtr));
    if (rp == wp) return 0;
    if (wp >= q->tx.msgCount || rp >= q->tx.msgCount)
      return tinynv_fail("the 610 status queue's pointers are outside the ring (read %u, write %u, %u slots): the "
                         "memory gsp-rm writes has been corrupted", rp, wp, q->tx.msgCount);
    if (taken >= q->tx.msgCount)
      return tinynv_fail("the 610 status queue never empties (%u records taken, write pointer %u)", taken, wp);

    // The element count is DERIVED from the RPC length (610 stores no elemCount), so read the two headers first.
    uint64_t slot = q->entries + (uint64_t)rp * q->tx.msgSize;
    uint8_t hdrs[TINYNV_RPC610_ELEM_HDR_SIZE + 32];
    nv_rd_block(&gsp->queues.view, slot, hdrs, sizeof hdrs);
    tinynv_msg_header_t mh;
    memcpy(&mh, hdrs + TINYNV_RPC610_ELEM_HDR_SIZE, sizeof mh);
    if (mh.length < sizeof(tinynv_msg_header_t) || mh.length > TINYNV_RPC610_ELEM_SIZE_MAX - TINYNV_RPC610_ELEM_HDR_SIZE)
      return tinynv_fail("the 610 status record declares %u rpc bytes, outside [%zu, %u]", mh.length,
                         sizeof(tinynv_msg_header_t), TINYNV_RPC610_ELEM_SIZE_MAX - TINYNV_RPC610_ELEM_HDR_SIZE);
    size_t msg_len = TINYNV_RPC610_ELEM_HDR_SIZE + (size_t)mh.length;
    uint32_t count = (uint32_t)((msg_len + q->tx.msgSize - 1) / q->tx.msgSize);
    if (count > q->tx.msgCount)
      return tinynv_fail("the 610 status record spans %u elements, more than the ring's %u slots", count, q->tx.msgCount);
    size_t total = (size_t)count * q->tx.msgSize;
    uint8_t *rec = malloc(total);
    if (!rec) return tinynv_fail("out of memory for a %zu byte 610 status record", total);
    uint64_t ring_len = (uint64_t)q->tx.msgSize * q->tx.msgCount, off = (uint64_t)rp * q->tx.msgSize;
    size_t first = total <= ring_len - off ? total : (size_t)(ring_len - off);
    nv_rd_block(&gsp->queues.view, q->entries + off, rec, first);
    if (first < total) nv_rd_block(&gsp->queues.view, q->entries, rec + first, total - first);

    const uint8_t *payload = NULL;
    uint32_t payload_len = 0, elem_count = 0;
    if (tinynv_rpc610_decode(rec, total, q->tx.msgSize, q->seq, &mh, &payload, &payload_len, &elem_count)) {
      free(rec);
      return -1;
    }
    q->seq++;
    nv_wr32(&gsp->queues.view, q->rx_off, (rp + elem_count) % q->tx.msgCount);
    __sync_synchronize();
    if (tinynv_failpoint("610-rpc-reply")) { free(rec); return -1; }

    uint32_t func = mh.function;
    if (mh.rpc_result) {
      // this refusal used to be one clause and the record was freed unread. But GSP-RM ECHOES the request
      // back, so the record being thrown away is the only thing on the wire that says WHICH object was refused - and
      // it carries a second status, the alloc/control header's own, which the envelope's result does not duplicate.
      // The two together separate the cases that look identical from the outside: an envelope result with a ZERO
      // echoed status means the message was turned away while being decoded, before any class handler ran; two equal
      // values mean the handler ran and refused on its own terms. Six wrong guesses were spent on a refusal that
      // never named its class. Read before freeing, on the failure path only.
      //
      // The echoed status leads the detail, because the error buffer is 512 bytes and the first run that carried a
      // detail at all had it appended last - the truncation fell exactly there, and the one field that decides which
      // of the two cases this is never reached the log. Everything after it is ordered by how much it narrows the
      // search. The field-by-field comparison goes to stderr, which has no such limit.
      char detail[352];
      detail[0] = 0;
      if (func == TINYNV_MSG_FUNCTION_GSP_RM_ALLOC && payload_len >= sizeof(tinynv_rpc_rm_alloc_t)) {
        tinynv_rpc_rm_alloc_t ah;
        memcpy(&ah, payload, sizeof ah);
        // kept for the caller, which on the 610 channel sweep has to report a status per attempt and cannot
        // get one out of the failure text (gsp.h:last_alloc_status).
        gsp->last_alloc_status = ah.status;
        snprintf(detail, sizeof detail,
                 ": gsp-rm's own status for the call is %#x, %s; the echo names %s (class %#x), handle %#x under "
                 "parent %#x for client %#x, %u parameter bytes, flags %#x",
                 ah.status,
                 ah.status ? "so the class handler ran and refused on its own terms"
                           : "zero, so the envelope refused the message while decoding it, before the class handler "
                             "ran",
                 rpc610_class_name(ah.hClass), ah.hClass, ah.hObject, ah.hParent, ah.hClient, ah.paramsSize, ah.flags);
        dump_refused_alloc610(g, &ah, payload, payload_len);
      } else if (func == TINYNV_MSG_FUNCTION_GSP_RM_CONTROL && payload_len >= sizeof(tinynv_rpc610_rm_control_t)) {
        tinynv_rpc610_rm_control_t ch;
        memcpy(&ch, payload, sizeof ch);
        snprintf(detail, sizeof detail,
                 ": gsp-rm's own status for the call is %#x; the echo names control %#x on object %#x for client %#x, "
                 "%u parameter bytes",
                 ch.status, ch.cmd, ch.hObject, ch.hClient, ch.paramsSize);
        if (payload_len > sizeof ch)
          dump_refused_blob("the control parameters gsp-rm echoed back", payload + sizeof ch,
                            payload_len - (uint32_t)sizeof ch);
      }
      free(rec);
      return tinynv_fail("gsp-rm refused rpc %u (%s) with result %#x%s. The reply was %u bytes", func,
                         rpc610_function_name(func), mh.rpc_result, detail, payload_len);
    }

    // The sequencer's replacements: validated and executed before any other interpretation. A malformed request,
    // the removed sequencer event or an unknown event is a refusal here, never a guess (gsp610_host.c).
    tinynv_gsp610_bl_ucode_t bl = gsp610_bl(gsp);
    int hr = tinynv_gsp610_host_request(g, func, payload, payload_len, &bl);
    if (hr < 0) { free(rec); return -1; }
    if (hr == 0) { free(rec); continue; }   // a host request, executed

    if (func == TINYNV_MSG_EVENT_OS_ERROR_LOG && payload_len > 12)
      fprintf(stderr, "tinynv: gsp log: %.*s\n", (int)(payload_len - 12), (const char *)payload + 12);
    if (func == TINYNV_MSG_EVENT_OS_ERROR_LOG || func == TINYNV_MSG_EVENT_MMU_FAULT_QUEUED) {
      gsp->err_state = 1;
      gsp->err_fn = func;
      gsp->err_len = payload_len;
      uint32_t keep = payload_len < sizeof(gsp->err_head) ? payload_len : (uint32_t)sizeof(gsp->err_head);
      if (keep) memcpy(gsp->err_head, payload, keep);
      fprintf(stderr, "tinynv: gsp-rm reported %s (event %u, %u byte payload)\n",
              func == TINYNV_MSG_EVENT_MMU_FAULT_QUEUED ? "an mmu fault" : "an error log", func, payload_len);
      for (uint32_t i = 0; i < keep; i += 16) {
        fprintf(stderr, "tinynv:   %04x:", i);
        for (uint32_t j = i; j < i + 16 && j < keep; j++) fprintf(stderr, " %02x", gsp->err_head[j]);
        fprintf(stderr, "\n");
      }
    }

    if (func == TINYNV_MSG_FUNCTION_CONTINUATION_RECORD) {
      if (gsp->cont_buf) {
        if (gsp->cont_len + payload_len > gsp->cont_want) {
          free(rec);
          return tinynv_fail("a 610 continuation overruns the reply being assembled (%u + %u > %u)",
                             gsp->cont_len, payload_len, gsp->cont_want);
        }
        memcpy(gsp->cont_buf + gsp->cont_len, payload, payload_len);
        gsp->cont_len += payload_len;
        if (gsp->cont_len >= gsp->cont_want) {
          if (seen) *seen = 1;
          if (reply) { *reply = gsp->cont_buf; *reply_len = gsp->cont_len; }
          else free(gsp->cont_buf);
          gsp->cont_buf = NULL; gsp->cont_len = gsp->cont_want = 0;
          free(rec);
          return 0;
        }
      }
      // else: the tail of a reply nobody is waiting for (a poll drains everything); drop it with the record.
      free(rec);
      continue;
    }

    // A new record while an assembly is pending means the firmware abandoned the split reply. Say so and drop it;
    // delivering a half reply as if it were whole is the confidently-wrong answer this path exists to prevent.
    if (gsp->cont_buf) {
      fprintf(stderr, "tinynv: a 610 reply was left half-assembled (%u of %u bytes) when the next record arrived; "
                      "dropping it\n", gsp->cont_len, gsp->cont_want);
      free(gsp->cont_buf);
      gsp->cont_buf = NULL; gsp->cont_len = gsp->cont_want = 0;
    }

    if (func == want) {
      uint32_t need = want_total ? want_total : payload_len;
      // an EMPTY payload is a complete reply, never the head of a split one. rpc_send fills the first record to
      // first_payload_max before it emits any CONTINUATION_RECORD, so a fragment of a genuinely split reply is never
      // zero-length; treating one as fragment 1 waits for bytes that cannot arrive. That is exactly how a working 610
      // unload became a 10 s timeout blaming the firmware (see tinynv_gsp_unload). 570 matches by function alone.
      if (payload_len >= need || payload_len == 0) {
        if (seen) *seen = 1;
        if (reply) {
          *reply = malloc(payload_len ? payload_len : 1);
          if (!*reply) { free(rec); return tinynv_fail("out of memory for a %u byte rpc reply", payload_len); }
          memcpy(*reply, payload, payload_len);
          *reply_len = payload_len;
        }
        free(rec);
        return 0;
      }
      // The reply is split: hold what arrived and let the continuation records finish it. The total is the
      // request's own head+params size, which RM echoes back exactly.
      gsp->cont_buf = malloc(need);
      if (!gsp->cont_buf) { free(rec); return tinynv_fail("out of memory for a %u byte split reply", need); }
      memcpy(gsp->cont_buf, payload, payload_len);
      gsp->cont_len = payload_len;
      gsp->cont_want = need;
      free(rec);
      continue;
    }
    // A record this driver is not waiting for. Most are events it has already acted on above; naming the ones it drops
    // is what turns "gsp-rm never answered" into "gsp-rm answered with function N", which is a different bug. Opt-in,
    // so the recorded 570 boot and the ordinary path issue no extra operation.
    if (getenv("TINYNV_RPC_TRACE"))
      fprintf(stderr, "tinynv: 610 status record function %u, %u byte payload, dropped while waiting for %u\n", func,
              payload_len, want);
    free(rec);
  }
}

// Take whatever GSP-RM has put in the status queue, and say whether the message we are waiting for was among it.
//
// Two kinds of message are acted on rather than merely delivered: a request to run a register sequence on GSP-RM's
// behalf, and a log line, which is the only way to see what the firmware thinks went wrong.
static int rpc_drain(tinynv_gpu_t *g, uint32_t want, uint32_t want_total, int *seen, uint8_t **reply, uint32_t *reply_len) {
  if (is610(g)) return rpc_drain610(g, want, want_total, seen, reply, reply_len);
  (void)want_total; // 570 replies are matched by function alone, as they always were
  tinynv_gsp_t *gsp = &g->gsp;
  tinynv_rpcq_t *q = &gsp->stat_q;
  __sync_synchronize();
  for (uint32_t taken = 0;; taken++) {
    uint32_t rp = nv_rd32(&gsp->queues.view, q->rx_off);
    uint32_t wp = nv_rd32(&gsp->queues.view, q->base + offsetof(tinynv_msgq_tx_header_t, writePtr));
    if (rp == wp) return 0;
    // A ring that says otherwise is not a ring any more. On 2026-09-21 a boot whose card had lost and regained its
    // windows mid-way left a zero-length record here, and the drain went round it for ten minutes without a word:
    // zero bytes is zero slots, and zero slots does not move the read pointer. So: a record is at least one slot, the
    // write pointer is inside the ring, and one call takes at most a ring's worth before it calls the memory garbage.
    if (wp >= q->tx.msgCount || rp >= q->tx.msgCount)
      return tinynv_fail("the status queue's pointers are outside the ring (read %u, write %u, %u slots): the memory "
                         "gsp-rm writes has been corrupted", rp, wp, q->tx.msgCount);
    if (taken >= q->tx.msgCount)
      return tinynv_fail("the status queue never empties (%u records taken, write pointer %u): the memory gsp-rm writes "
                         "has been corrupted", taken, wp);

    // read the position again rather than reuse the one the comparison saw. it costs nothing, it is what the oracle
    // does, and it does not assume that this side is the only writer of a field that lives in shared memory.
    uint64_t slot = q->entries + (uint64_t)nv_rd32(&gsp->queues.view, q->rx_off) * q->tx.msgSize;
    tinynv_msg_header_t mh;
    nv_rd_block(&gsp->queues.view, slot + sizeof(tinynv_msg_element_t), &mh, sizeof(mh));

    uint8_t *msg = NULL;
    if (mh.length) {
      if (!(msg = malloc(mh.length))) return tinynv_fail("out of memory for a %u byte rpc reply", mh.length);
      nv_rd_block(&gsp->queues.view, slot + sizeof(tinynv_msg_element_t) + sizeof(tinynv_msg_header_t), msg, mh.length);
    }

    if (mh.function == TINYNV_MSG_EVENT_GSP_RUN_CPU_SEQUENCER && run_cpu_seq(g, msg, mh.length)) { free(msg); return -1; }
    else if (mh.function == TINYNV_MSG_EVENT_OS_ERROR_LOG && mh.length > 12)
      fprintf(stderr, "tinynv: gsp log: %.*s\n", (int)(mh.length - 12), (const char *)msg + 12);
    if (mh.function == TINYNV_MSG_EVENT_OS_ERROR_LOG || mh.function == TINYNV_MSG_EVENT_MMU_FAULT_QUEUED) {
      gsp->err_state = 1;
      // Keep what it said. An MMU fault arrives as a notification - the addresses live in a fault buffer this driver
      // does not register yet - so the payload is often short and sometimes empty, and saying so is still better than
      // the driver claiming something was logged when nothing was.
      gsp->err_fn = mh.function;
      gsp->err_len = mh.length;
      uint32_t keep = mh.length < sizeof(gsp->err_head) ? mh.length : (uint32_t)sizeof(gsp->err_head);
      if (msg && keep) memcpy(gsp->err_head, msg, keep);
      fprintf(stderr, "tinynv: gsp-rm reported %s (event %u, %u byte payload)\n",
              mh.function == TINYNV_MSG_EVENT_MMU_FAULT_QUEUED ? "an mmu fault - some engine touched an address that is "
                                                                 "not mapped, so look at what a descriptor points at"
                                                               : "an error log",
              mh.function, mh.length);
      for (uint32_t i = 0; i < keep; i += 16) {
        fprintf(stderr, "tinynv:   %04x:", i);
        for (uint32_t j = i; j < i + 16 && j < keep; j++) fprintf(stderr, " %02x", gsp->err_head[j]);
        fprintf(stderr, "\n");
      }
    }

    // advance past however many slots this message occupied
    uint32_t slots = (mh.length + q->tx.msgSize - 1) / q->tx.msgSize;
    if (!slots) slots = 1;
    nv_wr32(&gsp->queues.view, q->rx_off, (nv_rd32(&gsp->queues.view, q->rx_off) + slots) % q->tx.msgCount);
    __sync_synchronize();

    uint32_t result = mh.rpc_result, func = mh.function;
    if (result) { free(msg); return tinynv_fail("gsp-rm refused rpc %u with result %#x", func, result); }
    // stop at the message that was asked for, rather than draining whatever else has arrived behind it. the read pointer
    // is already past it, so the rest of the queue is still there to be read next time.
    if (func == want) {
      if (seen) *seen = 1;
      if (reply) { *reply = msg; *reply_len = mh.length; } else free(msg);
      return 0;
    }
    free(msg);
  }
}

static void fault_report(tinynv_gpu_t *g);

// Read whatever GSP-RM has said and has not been collected. Nothing drains the status queue once the boot is over, so
// an engine that faults reports it into a queue nobody is reading: the driver sees work that never completes and has no
// idea why, which is exactly the position this was written from. Returns 1 if the firmware has reported an error.
int tinynv_gsp_poll(tinynv_gpu_t *g) {
  int seen = 0;
  if (!g->gsp.stat_q.base) return 0;               // the queue does not exist before the firmware is up
  if (rpc_drain(g, 0xffffffffu, 0, &seen, NULL, NULL)) return 1; // a refusal is itself news, and it has been recorded
  // The event says a fault happened. Where it happened has to be asked for, and this is the only moment anyone is
  // looking - so ask now, while the answer still describes the fault that is being reported rather than a later one.
  if (g->gsp.err_state) fault_report(g);
  return g->gsp.err_state;
}

// Where GSP-RM is while the driver waits for its start-up notice, said every few seconds on a live card.
//
// Added 2026-09-19 after the fault record of a failed boot showed GSP-RM had NOT died after its register sequence: it
// delivered ~35 queued status messages the moment a later process turned bus mastering back on, to queue addresses that
// no longer existed. So during the 60 s wait it was either not allowed to write to host memory, or slower than 60 s.
// This separates the two: the queue pointers and the log put pointers are host memory it writes (moving = its DMA
// works), the command register says whether bus mastering is still on, and the falcon registers say whether the core
// is running. Live only, and only once a wait has lasted five seconds, so the recorded boot - where the notice arrives
// on the first look - issues exactly the operations it always did.
static void init_wait_report(tinynv_gpu_t *g, double elapsed) {
  tinynv_gsp_t *gsp = &g->gsp;
  tinynv_dev_t *d = &g->dev;
  tinynv_pci_t *pci = d->pci;
  uint32_t rp = nv_rd32(&gsp->queues.view, gsp->stat_q.rx_off);
  uint32_t wp = nv_rd32(&gsp->queues.view, gsp->stat_q.base + offsetof(tinynv_msgq_tx_header_t, writePtr));
  uint32_t took = nv_rd32(&gsp->queues.view, gsp->cmd_q.rx_off); // how far gsp-rm has read OUR queue
  uint32_t logput[3] = {0, 0, 0};
  for (int i = 0; i < 3; i++) logput[i] = nv_rd32(&gsp->logbuf.view, 0x10000u * (unsigned)i); // INIT, INTR, RM
  uint32_t vd = pci->cfg_read(pci, 0x00, 4), cmd = pci->cfg_read(pci, 0x04, 2);
  uint32_t mbx0 = tinynv_rd32(d, NV_PGSP_FALCON_MAILBOX0), mbx1 = tinynv_rd32(d, NV_PGSP_FALCON_MAILBOX1);
  uint32_t rv = tinynv_rd32(d, g->flcn.falcon + NV_PRISCV_RISCV_CPUCTL);
  uint32_t sec = tinynv_rd32(d, g->flcn.sec2 + NV_PFALCON_FALCON_MAILBOX0);
  uint32_t wpr = tinynv_rd32(d, NV_PFB_PRI_MMU_WPR2_ADDR_HI);
  unsigned lgen = 0, lw = 0;
  tinynv_dev_link(d, &lgen, &lw);
  // Every register reading all ones does not mean the same thing as config space reading all ones. The first is a window
  // that is not decoding; the second is a card that is not there. Saying which is which is the whole value of this line:
  // on 2026-09-21 it reported "bus master OFF" about a card whose CONFIG space was readable and whose memory window had
  // simply been switched off underneath a booting firmware, and the two readings look identical if only one is printed.
  fprintf(stderr, "tinynv: +%3.0fs no start-up notice yet | status rp %u wp %u, gsp read %u of our commands, log put INIT %u "
                  "INTR %u RM %u | %04x:%04x command %#06x (memory %s, bus master %s) | gsp mailbox %#x %#x, riscv %s%s, "
                  "sec2 mailbox %#x, wpr2 hi %#x, link x%u gen%u\n",
          elapsed, rp, wp, took, logput[0], logput[1], logput[2], vd & 0xffff, vd >> 16, cmd,
          (cmd & 0x2) ? "on" : "OFF", (cmd & 0x4) ? "on" : "OFF", mbx0, mbx1,
          NV_GET(rv, NV_PRISCV_RISCV_CPUCTL, ACTIVE_STAT) ? "active" : "NOT active",
          NV_GET(rv, NV_PRISCV_RISCV_CPUCTL, HALTED) ? " halted" : "", sec, wpr, lw, lgen);

  // AND PUT IT BACK. A command register that loses its memory window and bus mastering while a firmware is starting is
  // never a legitimate state: with the window off every register reads all ones, and with bus mastering off nothing
  // GSP-RM writes reaches this host, so the driver waits for a notice that physically cannot arrive and then reports the
  // firmware as silent. That is what every failed boot on this card was (2026-09-17 and 2026-09-19); the proof is that a
  // LATER process setting this register made the firmware deliver the whole backlog at once, into memory that by then
  // had been freed, which is where the machine's fault records came from. Nothing in this driver clears it, macOS logged
  // nothing when it happened, and the card never left the bus. So: restore it and say so. TINYNV_REASSERT_BME=0 leaves
  // it alone, for seeing the untreated failure again.
  // AND PUT THEM BACK. Restoring the command register alone is not enough and proves it: on 2026-09-21 doing exactly
  // that brought the start-up notice in immediately - gsp-rm had been running all along with nowhere to write - and the
  // boot then died in the page tables, because the base address registers were still zero and everything written
  // through the memory window went nowhere. Both halves or neither.
  const char *knob = getenv("TINYNV_REASSERT_BME");
  if (knob && !strcmp(knob, "0")) {
    if ((cmd & 0x6) != 0x6)
      fprintf(stderr, "tinynv: the card has lost its windows under a booting gsp-rm and TINYNV_REASSERT_BME=0 leaves it "
                      "that way, so nothing it writes can reach this host\n");
    return;
  }
  tinynv_dev_restore_decode(d);
}

static int rpc_wait(tinynv_gpu_t *g, uint32_t want, uint32_t want_total, double timeout_s, const char *what, uint8_t **reply, uint32_t *reply_len) {
  double start = tinynv_now_s(), deadline = start + timeout_s, next_report = start + 5.0;
  int seen = 0, gave_up = 0;
  const unsigned restored_before = g->dev.windows_restored;   // only a restore made during THIS wait counts below
  int report = want == TINYNV_MSG_EVENT_GSP_INIT_DONE && g->dev.pci->live;
  if (reply) { *reply = NULL; *reply_len = 0; }
  // The configuration is looked at every quarter second, not every five: when it goes, knowing WHEN it went - how long
  // after the register sequence, what the falcons were doing - is the evidence the open question needs, and putting it
  // back within a quarter second is less time for anything written through the window to be lost.
  double next_cfg = start + 0.25;
  do {
    if (rpc_drain(g, want, want_total, &seen, reply, reply_len)) return -1;
    if (seen) {
      if (report && tinynv_now_s() - start > 5.0)
        fprintf(stderr, "tinynv: the start-up notice arrived after %.1f s\n", tinynv_now_s() - start);
      return 0;
    }
    if (report && tinynv_now_s() >= next_cfg) {
      next_cfg += 0.25;
      uint32_t cmd = g->dev.pci->cfg_read(g->dev.pci, 0x04, 2);
      if (cmd != 0xffff && (cmd & 0x6) != 0x6) {
        fprintf(stderr, "tinynv: %.2f s after the start-up wait began, the command register reads %#06x\n",
                tinynv_now_s() - start, cmd);
        init_wait_report(g, tinynv_now_s() - start);
      }
    }
    if (report && tinynv_now_s() >= next_report) {
      init_wait_report(g, tinynv_now_s() - start);
      next_report += 5.0;
    }
    // A firmware whose windows went and came back either answers within a few seconds of the restore (it had finished
    // and was only waiting for somewhere to write) or it never will (it was mid-init and read zeroed configuration).
    // Ten seconds separates those; the rest of a sixty second wait only delays the retry that cures the second case.
    if (report && g->dev.windows_restored > restored_before && tinynv_now_s() - g->dev.windows_restored_at > 10.0) {
      fprintf(stderr, "tinynv: ten seconds after the windows were put back there is still no start-up notice: giving this boot up\n");
      gave_up = 1;
      break;
    }
  } while (tinynv_now_s() < deadline);
  // Whatever arrives now belongs to a request nobody is waiting for any more.
  g->gsp.rpc_desync = 1;
  // a half-assembled split reply dies with the wait; the desync drain drops any late continuation.
  if (g->gsp.cont_buf) { free(g->gsp.cont_buf); g->gsp.cont_buf = NULL; g->gsp.cont_len = g->gsp.cont_want = 0; }
  if (want == TINYNV_MSG_EVENT_GSP_INIT_DONE) g->gsp.init_timed_out = 1;
  // A timeout on the start-up notice is the one this driver keeps hitting on a live 3060, so say enough to tell a silent
  // death apart from a message we failed to recognise. If GSP-RM's write pointer sits exactly where our read pointer is,
  // it stopped talking after the register sequence (a hung or browned-out core); if it moved past us, it sent something
  // we did not act on and the bug is here.
  tinynv_gsp_t *gsp = &g->gsp;
  if (want == TINYNV_MSG_EVENT_GSP_INIT_DONE && gsp->stat_q.base) {
    uint32_t rp = nv_rd32(&gsp->queues.view, gsp->stat_q.rx_off);
    uint32_t wp = nv_rd32(&gsp->queues.view, gsp->stat_q.base + offsetof(tinynv_msgq_tx_header_t, writePtr));
    fprintf(stderr, "tinynv: at the timeout the status queue read pointer is %u and gsp-rm's write pointer is %u (%s); "
                    "%u register sequence(s) were handled and the firmware %s reported an error\n",
            rp, wp, rp == wp ? "gsp-rm went silent after the register sequence - a hung or under-powered core, not a "
                               "message this driver mishandled" : "gsp-rm sent more than this driver consumed",
            gsp->cpu_seq_requests, gsp->err_state ? "HAS" : "has not");
  }
  if (gave_up)
    return tinynv_fail("gsp-rm never sent %s (message %u): its configuration was cleared under it %.1f s into the boot and "
                       "ten seconds after it was put back there was still nothing", what, want, g->dev.windows_restored_at - start);
  return tinynv_fail("gsp-rm never sent %s (message %u) in %.0f s", what, want, timeout_s);
}

// --- the object tree ------------------------------------------------------------------------------------------------
//
// A call is a request record followed by its parameters, and a reply that carries the same parameters back with whatever
// GSP-RM filled in. Both directions use the ring built above; the only new idea is that the caller now waits for its own
// answer rather than for an event.
static uint32_t next_handle(tinynv_gsp_t *gsp) { return gsp->next_handle++; }

static int rpc_call(tinynv_gpu_t *g, uint32_t func, void *head, size_t head_len, void *params, size_t params_len,
                    const char *what) {
  // A previous call gave up on its reply. Clear the queue before asking anything else, because a reply carries the
  // function it answers and nothing that identifies the request: a late reply to one control is indistinguishable from
  // the answer to the next one, and rpc_wait would return it, copying the wrong parameters back to a caller with no
  // way to tell. The fault report makes two controls in a row on a path that is already failing, which is exactly
  // where this would land and exactly where a confidently wrong answer is worst.
  //
  // Only after a timeout, so the ordinary path issues no operation it did not issue before and the recorded boot is
  // reproduced exactly as it was.
  if (g->gsp.rpc_desync) {
    int seen = 0;
    fprintf(stderr, "tinynv: a previous rpc gave up waiting; emptying the reply queue so a late answer cannot be "
                    "mistaken for this one\n");
    if (rpc_drain(g, 0xffffffffu, 0, &seen, NULL, NULL)) return -1;
    g->gsp.rpc_desync = 0;
  }
  size_t total = head_len + params_len;
  uint8_t *buf = malloc(total ? total : 1);
  if (!buf) return tinynv_fail("out of memory for a %zu byte %s", total, what);
  memcpy(buf, head, head_len);
  if (params_len) memcpy(buf + head_len, params, params_len);
  int rc = rpc_send(g, func, buf, total);
  free(buf);
  if (rc) return -1;

  uint8_t *reply = NULL;
  uint32_t reply_len = 0;
  if (rpc_wait(g, func, (uint32_t)(head_len + params_len), 10.0, what, &reply, &reply_len)) return -1;
  // gsp-rm answers with the request echoed back and the parameters filled in, so take them from where they were sent
  if (params && params_len && reply_len >= head_len + params_len) memcpy(params, reply + head_len, params_len);
  // The echoed header carries RM's OWN status, and it used to be freed here with the rest of the reply. That status is
  // the only thing on the wire that says WHY a call was refused: the rpc envelope's result stays 0, because the rpc
  // itself worked perfectly - it delivered a refusal. Six wrong guesses at a refused copy object were bought with this
  // silence, and the answer had been arriving in this buffer the whole time.
  if (reply_len >= head_len) memcpy(head, reply, head_len);
  free(reply);
  return 0;
}

// Create one object under another and return its handle. The root client is the exception: it names itself.
// `handle` of zero means this driver picks one from its counter, which is what every caller here wants.
static int rm_alloc_as(tinynv_gpu_t *g, uint32_t client, uint32_t parent, uint32_t handle, uint32_t cls, void *params,
                       size_t params_len, uint32_t *out) {
  tinynv_gsp_t *gsp = &g->gsp;
  tinynv_rpc_rm_alloc_t head;
  memset(&head, 0, sizeof(head));
  head.hClient = client;
  head.hParent = parent;
  head.hObject = handle ? handle : next_handle(gsp);
  head.hClass = cls;
  head.paramsSize = (uint32_t)params_len;
  // two allocation payloads grew (rpc610): the VA space (48 -> 56, pasid appended) and the GPFIFO channel
  // (368 -> 376, hHandleVASpace inserted). Encode the caller's 570-shaped working struct into a scratch buffer at
  // the 610 layout; the echo copies back into the scratch, never into the caller's differently-shaped struct. No
  // caller reads channel/vaspace alloc params back (handles come from the header), which test_lifecycle610 pins.
  uint8_t enc610[400];
  void *send_params = params;
  size_t send_len = params_len;
  if (is610(g) && params && params_len == sizeof(tinynv_vaspace_alloc_t) && cls == TINYNV_CLASS_VASPACE) {
    tinynv_rpc610_vaspace_encode((tinynv_rpc610_vaspace_alloc_t *)enc610, params);
    send_params = enc610; send_len = sizeof(tinynv_rpc610_vaspace_alloc_t);
    head.paramsSize = (uint32_t)send_len;
  } else if (is610(g) && params && params_len == sizeof(tinynv_gpfifo_alloc_t) && cls == g->dev.class_gpfifo) {
    tinynv_rpc610_channel_encode((tinynv_rpc610_gpfifo_alloc_t *)enc610, params);
    send_params = enc610; send_len = sizeof(tinynv_rpc610_gpfifo_alloc_t);
    head.paramsSize = (uint32_t)send_len;
  }
  // keep a copy of the exact parameter bytes about to go on the wire, so the refusal path can set them
  // against what gsp-rm echoes back (rpc_drain610 -> dump_refused_alloc610). The request is gone by the time the
  // answer arrives, and the echo alone cannot say whether a differing byte came from gsp-rm rewriting it or from
  // this driver serializing it wrong. Cleared first, so a refusal never reports the previous allocation's bytes.
  // 610 only: the 570 profile gets no echo to compare against, and copying there would be work for nothing.
  gsp->last_alloc_sent_len = 0;
  gsp->last_alloc_status = 0;
  if (is610(g) && send_params && send_len <= sizeof gsp->last_alloc_sent) {
    memcpy(gsp->last_alloc_sent, send_params, send_len);
    gsp->last_alloc_sent_len = (uint32_t)send_len;
  }
  if (rpc_call(g, TINYNV_MSG_FUNCTION_GSP_RM_ALLOC, &head, sizeof(head), send_params, send_len, "object allocation")) return -1;
  // Said, not enforced. A nonzero status here has always been ignored, so some allocation on the boot path may well be
  // returning one and working anyway; making it fatal today would break a boot that currently comes up, and finding
  // that out costs a card window. So this reports and carries on, and the caller's own failure - if there is one - is
  // still the thing that stops the run.
  if (head.status)
    fprintf(stderr, "tinynv: gsp-rm returned status %#x allocating class %#x under parent %#x\n", head.status, cls, parent);
  uint32_t made = cls == TINYNV_CLASS_ROOT ? client : head.hObject;
  if (out) *out = made;
  return 0;
}

static int rm_alloc(tinynv_gpu_t *g, uint32_t parent, uint32_t cls, void *params, size_t params_len, uint32_t *out) {
  return rm_alloc_as(g, TINYNV_RM_PRIV_ROOT, parent, 0, cls, params, params_len, out);
}

static int rm_control_as(tinynv_gpu_t *g, uint32_t client, uint32_t object, uint32_t cmd, void *params, size_t params_len) {
  // the 610 control header grew 24 -> 40 B (rpc610). The alloc header (32 B) is unchanged and shared. The
  // status field sits at the same offset in both, so the read-back below is identical; only the header size differs.
  if (is610(g)) {
    tinynv_rpc610_rm_control_t head;
    tinynv_rpc610_rm_control_fill(&head, client, object, cmd, (uint32_t)params_len);
    if (rpc_call(g, TINYNV_MSG_FUNCTION_GSP_RM_CONTROL, &head, sizeof(head), params, params_len, "object control call"))
      return -1;
    g->gsp.last_ctrl_status = head.status;
    if (head.status)
      fprintf(stderr, "tinynv: gsp-rm returned status %#x for control %#x on object %#x\n", head.status, cmd, object);
    return 0;
  }
  tinynv_rpc_rm_control_t head;
  memset(&head, 0, sizeof(head));
  head.hClient = client;
  head.hObject = object;
  head.cmd = cmd;
  head.paramsSize = (uint32_t)params_len;
  if (rpc_call(g, TINYNV_MSG_FUNCTION_GSP_RM_CONTROL, &head, sizeof(head), params, params_len, "object control call"))
    return -1;
  g->gsp.last_ctrl_status = head.status;
  if (head.status)  // reported, not enforced, for the reason given in rm_alloc_as
    fprintf(stderr, "tinynv: gsp-rm returned status %#x for control %#x on object %#x\n", head.status, cmd, object);
  return 0;
}

static int rm_control(tinynv_gpu_t *g, uint32_t object, uint32_t cmd, void *params, size_t params_len) {
  return rm_control_as(g, TINYNV_RM_PRIV_ROOT, object, cmd, params, params_len);
}

// A field is valid when its timestamp is neither zero (not reported here) nor an odd sequence in the high range (a
// write in progress). Straight from RUSD_SEQ_DATA_VALID in the vendored header.
static int rusd_seq_ok(uint64_t seq) {
  return (seq < TINYNV_RUSD_SEQ_START && seq != TINYNV_RUSD_TS_INVALID) ||
         (seq >= TINYNV_RUSD_SEQ_START && (seq & 1) == 0);
}

int tinynv_gsp_free_heap(tinynv_gpu_t *g, uint64_t *out) {
  tinynv_gsp_free_heap_t p;
  memset(&p, 0, sizeof(p));
  if (rm_control(g, g->gsp.subdevice, TINYNV_CTRL_GET_GSP_RM_FREE_HEAP, &p, sizeof(p))) return -1;
  *out = p.freeHeapSize;
  return 0;
}

int tinynv_gsp_sensors_init(tinynv_gpu_t *g, uint64_t poll_mask, uint32_t freq_ms) {
  tinynv_gsp_t *gsp = &g->gsp;
  if (gsp->rusd_armed) return 0;
  // Host memory, because that is where the firmware puts it: ADDR_SYSMEM in the vendor's own allocation of this
  // buffer. It is 464 bytes on 570 (3152 on 610, whose block grew - rpc610) and it costs one of the 128 host mappings
  // the broker will ever hand out, which is a small finite resource that only goes up - worth stating rather than
  // waving through.
  const size_t rusd_sz = is610(g) ? TINYNV_RPC610_RUSD_SIZE : sizeof(tinynv_rusd_t);
  if (tinynv_mm_alloc_buffer(&g->mm, rusd_sz, 1, 1, 0, 0, 1, &gsp->rusd_mem)) return -1;

  // The class allocation is attempted but NOT required, and that distinction is the whole of why this still exists.
  //
  // On this card GSP-RM refuses RM_USER_SHARED_DATA with NV_ERR_NOT_SUPPORTED (0x56) - the same answer Session C gets
  // for it from the guest side. But that class is the object a CLIENT allocates to be handed a mapping of the block.
  // We are not a client of RM here; we are the CPU-side resource manager, and the two controls below are the plumbing
  // that side uses to tell the firmware where the buffer is and what to keep fresh. So the refusal may be about the
  // client-facing object rather than about the mechanism, and the controls are worth trying on their own.
  //
  // If that reasoning is wrong the controls will refuse too, and we will have learned it for the cost of two messages.
  tinynv_rusd_alloc_t ap = {.polledDataMask = poll_mask};
  if (rm_alloc(g, gsp->subdevice, TINYNV_CLASS_USER_SHARED_DATA, &ap, sizeof(ap), &gsp->rusd_obj))
    gsp->rusd_obj = 0;   // carry on without it

  // the two RUSD controls changed for 610 (init gained bInit before physAddr@8; poll renamed its interval
  // field at the same offset). The encoders place the fields; the 570 payloads are unchanged.
  if (is610(g)) {
    tinynv_rpc610_rusd_init_t ip;
    tinynv_rpc610_rusd_init_fill(&ip, gsp->rusd_mem.ranges[0].paddr);
    if (rm_control(g, gsp->subdevice, TINYNV_CTRL_INIT_USER_SHARED_DATA, &ip, sizeof(ip))) goto undo;
    tinynv_rpc610_rusd_poll_t pp;
    tinynv_rpc610_rusd_poll_fill(&pp, poll_mask, freq_ms);
    if (rm_control(g, gsp->subdevice, TINYNV_CTRL_USER_SHARED_DATA_POLL, &pp, sizeof(pp))) goto undo;
  } else {
    tinynv_rusd_init_t ip = {.physAddr = gsp->rusd_mem.ranges[0].paddr};
    if (rm_control(g, gsp->subdevice, TINYNV_CTRL_INIT_USER_SHARED_DATA, &ip, sizeof(ip))) goto undo;

    tinynv_rusd_poll_t pp = {.polledDataMask = poll_mask, .pollFrequencyMs = freq_ms};
    if (rm_control(g, gsp->subdevice, TINYNV_CTRL_USER_SHARED_DATA_POLL, &pp, sizeof(pp))) goto undo;
  }

  gsp->rusd_armed = 1;
  return 0;

undo:
  // Leave nothing half-armed: a buffer the firmware was told about but a poll it never got would read as a card that
  // reports nothing, which is indistinguishable from a card that cannot.
  gsp->rusd_obj = 0;
  gsp->rusd_armed = 0;
  if (gsp->rusd_mem.size) tinynv_vmap_free(&g->mm, &gsp->rusd_mem);
  return -1;
}

int tinynv_gsp_sensors_read(tinynv_gpu_t *g, tinynv_rusd_t *out) {
  tinynv_gsp_t *gsp = &g->gsp;
  memset(out, 0, sizeof(*out));
  if (!gsp->rusd_armed || !gsp->rusd_mem.dma.va) return tinynv_fail("sensors were not asked for on this device");
  // the 610 block is a different layout (3152 B, new offsets); its reader maps the used fields into the
  // same 570-shaped target with the same per-field timestamp retry protocol.
  if (is610(g)) return tinynv_rpc610_rusd_read(gsp->rusd_mem.dma.va, TINYNV_RPC610_RUSD_SIZE, out);
  const volatile tinynv_rusd_t *src = (const volatile tinynv_rusd_t *)gsp->rusd_mem.dma.va;

  // Field by field, because each carries its own timestamp and the firmware updates them independently. Read the
  // timestamp, copy, read it again: unchanged means nothing moved underneath. Ten attempts, then leave that field zero
  // and let the caller see a timestamp of zero, which reads the same as "not reported here" - both mean do not trust
  // this number, which is the only thing a caller needs to know.
#define FIELD(f) do {                                                                  \
    for (int a = 0; a < 10; a++) {                                                     \
      uint64_t seq = src->f.ts;                                                        \
      __atomic_thread_fence(__ATOMIC_ACQUIRE);                                         \
      if (!rusd_seq_ok(seq)) continue;                                                 \
      memcpy(&out->f, (const void *)&src->f, sizeof(out->f));                          \
      __atomic_thread_fence(__ATOMIC_ACQUIRE);                                         \
      if (seq == src->f.ts) break;                                                     \
      memset(&out->f, 0, sizeof(out->f));                                              \
    }                                                                                  \
  } while (0)
  FIELD(bar1MemoryInfo); FIELD(pmaMemoryInfo); FIELD(clkPublicDomainInfos); FIELD(clkThrottleReason);
  FIELD(perfDevUtil); FIELD(memEcc); FIELD(perfCurrentPstate); FIELD(powerLimitGpu);
  FIELD(temperatures[0]); FIELD(temperatures[1]); FIELD(avgPowerUsage); FIELD(instPowerUsage); FIELD(pciBusData);
#undef FIELD
  return 0;
}


// --- what faulted --------------------------------------------------------------------------------------------------
//
// An MMU fault arrives on the status queue as a bare notification: some engine touched an address that is not mapped,
// and nothing about which address or what it was doing. That is the least useful true sentence a driver can print, and
// it was printed through several hardware sessions before this existed.
//
// The addresses are not in that message and they are not in a buffer this driver forgot to register. They are held for
// the channel and asked for afterwards, of the debugger object that is already allocated on the compute channel
// because the recorded boot allocates one. The oracle does exactly this and registers no fault buffer either, which is
// what makes this change cost nothing at boot: nothing here runs until something has already gone wrong, so the boot
// the replay pins is untouched.
//
// Two questions, in this order. The first reads the per-SM error registers and, with them, a valid bit for a fault on
// the target channel - so it also answers "was this an MMU fault at all", which the notification does not. Only if
// that bit is set is the second worth asking, and it returns up to four addresses with the access that caused each.
static const char *fault_type_name(uint32_t t) {
  switch (t) {
    case TINYNV_PFAULT_TYPE_PDE:                  return "no page directory entry";
    case TINYNV_PFAULT_TYPE_PDE_SIZE:             return "page directory entry of the wrong size";
    case TINYNV_PFAULT_TYPE_PTE:                  return "no page table entry - nothing is mapped there";
    case TINYNV_PFAULT_TYPE_VA_LIMIT_VIOLATION:   return "address beyond the end of the address space";
    case TINYNV_PFAULT_TYPE_UNBOUND_INST_BLOCK:   return "unbound instance block - the channel was never set up";
    case TINYNV_PFAULT_TYPE_PRIV_VIOLATION:       return "privilege violation";
    case TINYNV_PFAULT_TYPE_RO_VIOLATION:         return "wrote to a read-only mapping";
    case TINYNV_PFAULT_TYPE_WO_VIOLATION:         return "read from a write-only mapping";
    case TINYNV_PFAULT_TYPE_PITCH_MASK_VIOLATION: return "pitch mask violation";
    case TINYNV_PFAULT_TYPE_WORK_CREATION:        return "work creation fault";
    case TINYNV_PFAULT_TYPE_UNSUPPORTED_APERTURE: return "unsupported aperture - the memory is not of a kind this "
                                                         "engine can reach";
    case TINYNV_PFAULT_TYPE_CC_VIOLATION:         return "confidential computing violation";
    case TINYNV_PFAULT_TYPE_UNSUPPORTED_KIND:     return "unsupported kind";
    case TINYNV_PFAULT_TYPE_REGION_VIOLATION:     return "region violation";
    case TINYNV_PFAULT_TYPE_POISONED:             return "poisoned memory";
    case TINYNV_PFAULT_TYPE_ATOMIC_VIOLATION:     return "atomic on memory that does not support it";
    default:                                      return "a fault kind this driver does not name";
  }
}

// The physical variants matter more than they look: a fault on a physical address is not in this driver's page tables
// at all, so the whole of the mapping code is the wrong place to start looking.
static const char *fault_access_name(uint32_t a) {
  switch (a) {
    case TINYNV_PFAULT_ACCESS_READ:          return "read";
    case TINYNV_PFAULT_ACCESS_WRITE:         return "write";
    case TINYNV_PFAULT_ACCESS_ATOMIC:        return "atomic";
    case TINYNV_PFAULT_ACCESS_PREFETCH:      return "prefetch, so the fault is past the end of something rather than "
                                                    "at a pointer that was wrong";
    case TINYNV_PFAULT_ACCESS_ATOMIC_WEAK:   return "weak atomic";
    case TINYNV_PFAULT_ACCESS_PHYS_READ:     return "read of a PHYSICAL address";
    case TINYNV_PFAULT_ACCESS_PHYS_WRITE:    return "write to a PHYSICAL address";
    case TINYNV_PFAULT_ACCESS_PHYS_ATOMIC:   return "atomic on a PHYSICAL address";
    case TINYNV_PFAULT_ACCESS_PHYS_PREFETCH: return "prefetch of a PHYSICAL address";
    default:                                 return "an access kind this driver does not name";
  }
}

static void fault_report(tinynv_gpu_t *g) {
  tinynv_gsp_t *gsp = &g->gsp;
  if (gsp->fault_reported) return;
  // Before the boot has built them there is nothing to ask and nothing to ask it of. This is not an error: the same
  // poll runs during start-up, where a report from the firmware is about the boot rather than about a launch.
  if (!gsp->user_debugger || !gsp->compute_q.channel) return;
  gsp->fault_reported = 1;

  // Four and a half kilobytes of registers, which is a lot for a stack in a path that is already unwinding.
  tinynv_sm_error_states_t *st = calloc(1, sizeof(*st));
  if (!st) return;
  st->hTargetChannel = gsp->compute_q.channel;
  st->numSMsToRead = TINYNV_DEBUG_MAX_SMS_PER_CALL;
  if (rm_control_as(g, TINYNV_RM_USER_ROOT, gsp->user_debugger, TINYNV_CTRL_CMD_DEBUG_READ_ALL_SM_ERROR_STATES, st,
                    sizeof(*st))) {
    // Worth saying rather than swallowing: "the debugger did not answer" is itself a fact about how far gone the
    // firmware is, and it is different from "there was nothing to report".
    fprintf(stderr, "tinynv: asked the debugger what faulted and it did not answer (%s)\n", tinynv_last_error());
    free(st);
    return;
  }

  if (st->mmuFault.valid) {
    tinynv_mmu_fault_info_t info;
    memset(&info, 0, sizeof(info));
    if (rm_control_as(g, TINYNV_RM_USER_ROOT, gsp->user_debugger, TINYNV_CTRL_CMD_DEBUG_READ_MMU_FAULT_INFO, &info,
                      sizeof(info))) {
      fprintf(stderr, "tinynv: an mmu fault is flagged on the compute channel but its addresses did not come back "
                      "(%s); the fault word is %#x\n", tinynv_last_error(), st->mmuFault.faultInfo);
      free(st);
      return;
    }
    uint32_t n = info.count < TINYNV_MMU_FAULT_ENTRIES ? info.count : TINYNV_MMU_FAULT_ENTRIES;
    if (n) {
      // The first entry is the one the caller is told about: the list is newest-first for this fault, and a message
      // that names four addresses names three that are probably older than the failure being reported.
      gsp->fault_have = 1;
      gsp->fault_va = info.mmuFaultInfoList[0].faultAddress;
      gsp->fault_type = info.mmuFaultInfoList[0].faultType;
      gsp->fault_access = info.mmuFaultInfoList[0].accessType;
    }
    fprintf(stderr, "tinynv: mmu fault on the compute channel, %u address%s recorded:\n", n, n == 1 ? "" : "es");
    for (uint32_t i = 0; i < n; i++)
      fprintf(stderr, "tinynv:   %#018llx  %s (%s)\n", (unsigned long long)info.mmuFaultInfoList[i].faultAddress,
              fault_type_name(info.mmuFaultInfoList[i].faultType),
              fault_access_name(info.mmuFaultInfoList[i].accessType));
    // The interface says so, and it is the difference between "it faulted four times" and "it faulted once and this
    // list has not been cleared since". Nothing here clears it, so the older entries can be from earlier runs.
    if (n > 1) fprintf(stderr, "tinynv:   (this list is kept until newer faults push entries out, so an older one may "
                               "be a repeat rather than a separate fault)\n");
  } else {
    // No MMU fault, so the addresses are not the story. What is left is the SMs' own error registers, which say a
    // kernel did something illegal rather than touched something unmapped.
    int shown = 0;
    for (uint32_t i = 0; i < st->numSMsToRead && i < TINYNV_DEBUG_MAX_SMS_PER_CALL; i++) {
      const tinynv_sm_error_state_t *e = &st->smErrorStateArray[i];
      if (!e->hwwGlobalEsr && !e->hwwWarpEsr && !e->hwwCgaEsr) continue;
      if (!gsp->fault_sm_have) { gsp->fault_sm_have = 1; gsp->fault_sm_esr = e->hwwWarpEsr ? e->hwwWarpEsr : e->hwwGlobalEsr; }
      if (shown == 8) { fprintf(stderr, "tinynv:   (more SMs report the same; the first eight are enough)\n"); break; }
      fprintf(stderr, "tinynv:   SM %u: global esr %#x, warp esr %#x, warp pc %#llx, address %#llx\n", i,
              e->hwwGlobalEsr, e->hwwWarpEsr, (unsigned long long)e->hwwWarpEsrPc64,
              (unsigned long long)e->hwwEsrAddr);
      shown++;
    }
    if (!shown)
      fprintf(stderr, "tinynv: the debugger reports no mmu fault and no SM error, so whatever gsp-rm is complaining "
                      "about did not happen inside a kernel on the compute channel\n");
  }
  free(st);
}

// The objects every later thing hangs off: a privileged client, the device, the physical chip under it, and an address
// space. This is the oracle's init_golden_image up to the point where it starts asking about engines.
int tinynv_gsp_init_objects(tinynv_gpu_t *g) {
  tinynv_gsp_t *gsp = &g->gsp;
  gsp->next_handle = TINYNV_RM_FIRST_HANDLE;

  tinynv_nv0000_alloc_t root;
  memset(&root, 0, sizeof(root));
  if (rm_alloc(g, 0, TINYNV_CLASS_ROOT, &root, sizeof(root), &gsp->priv_root)) return -1;

  tinynv_nv0080_alloc_t dev;
  memset(&dev, 0, sizeof(dev));
  dev.hClientShare = TINYNV_RM_PRIV_ROOT;
  if (rm_alloc(g, gsp->priv_root, TINYNV_CLASS_DEVICE, &dev, sizeof(dev), &gsp->device)) return -1;

  tinynv_nv2080_alloc_t sub;
  memset(&sub, 0, sizeof(sub));
  if (rm_alloc(g, gsp->device, TINYNV_CLASS_SUBDEVICE, &sub, sizeof(sub), &gsp->subdevice)) return -1;

  tinynv_vaspace_alloc_t va;
  memset(&va, 0, sizeof(va));
  if (rm_alloc(g, gsp->device, TINYNV_CLASS_VASPACE, &va, sizeof(va), &gsp->vaspace)) return -1;

  // which runlist each engine is on. entry word 2 names the engine, word 3 its runlist, which is what a channel needs.
  tinynv_fifo_device_info_t *info = calloc(1, sizeof(*info));
  if (!info) return tinynv_fail("out of memory for the engine table");
  if (rm_control(g, gsp->subdevice, TINYNV_CTRL_CMD_FIFO_GET_DEVICE_INFO_TABLE, info, sizeof(*info))) { free(info); return -1; }
  gsp->nengines = info->numEntries < TINYNV_FIFO_DEVICE_ENTRIES ? info->numEntries : TINYNV_FIFO_DEVICE_ENTRIES;
  for (uint32_t i = 0; i < gsp->nengines; i++) {
    gsp->engines[i] = info->entries[i].engineData[2];
    gsp->runlists[i] = info->entries[i].engineData[3];
  }
  // Read-only, fetched on every ordinary boot regardless of TINYNV_RUNQUEUE_SPLIT, printed on none of them until
  // now: numPbdmas/pbdmaIds per engine. NVIDIA's real kernel_fifo_*.c validates a channel's requested runqueue
  // against this count for its OWN engine before scheduling it (kfifoGetNumRunqueues_HAL, runqueue bounds-checked
  // against numSrcPbdmaIds); this driver's channel-alloc path never has, for any engine, on any boot. If the copy
  // (DMA_COPY) engine reports numPbdmas=1, runqueue ONE is not a real hardware lane for it at all, and setting
  // NVOS04_FLAGS_GROUP_CHANNEL_RUNQUEUE_ONE on that channel is undefined behaviour GSP-RM never rejects - which
  // would fully explain both hardware outcomes seen 2026-09-18 (a NULL-deref crash and, on a different build, a
  // hang ending in a real DART fault) as one root cause landing two different ways, not two different bugs.
  for (uint32_t i = 0; i < gsp->nengines; i++)
    fprintf(stderr, "tinynv: engine %-16s type 0x%x runlist %u numPbdmas %u pbdmaIds [%u,%u]\n",
            info->entries[i].engineName, gsp->engines[i], gsp->runlists[i],
            info->entries[i].numPbdmas, info->entries[i].pbdmaIds[0], info->entries[i].pbdmaIds[1]);
  free(info);

  // Half a gigabyte of address space whose page tables are built now and described to GSP-RM, so that the two sides
  // share a tree rather than each building its own. The levels go root first; the first is only as big as its entry
  // count, the rest are a page each.
  uint64_t res_size = 512ull << 20;
  uint64_t res_va = tinynv_mm_alloc_va(&g->mm, res_size, 0);
  if (res_va == TINYNV_BAD_ADDR) return -1;

  tinynv_pt_t levels[TINYNV_MAX_PT_LEVELS];
  int nlevels = 0;
  if (tinynv_mm_page_tables(&g->mm, g->mm.root_page_table, res_va, res_size, levels, &nlevels)) return -1;
  if (nlevels > TINYNV_MAX_PDE_LEVELS) return tinynv_fail("the page table is %d levels deep, more than the %d gsp-rm takes",
                                                          nlevels, TINYNV_MAX_PDE_LEVELS);

  tinynv_copy_pdes_t pdes;
  memset(&pdes, 0, sizeof(pdes));
  pdes.pageSize = res_size;
  pdes.numLevelsToCopy = 3; // the oracle's count, which is one fewer than it fills; gsp-rm reads what it needs
  pdes.virtAddrLo = res_va;
  pdes.virtAddrHi = res_va + res_size - 1;
  for (int i = 0; i < nlevels; i++) {
    pdes.levels[i].physAddress = levels[i].paddr;
    pdes.levels[i].size = i == 0 ? (uint64_t)g->mm.pte_cnt[0] * 8 : 0x1000;
    pdes.levels[i].aperture = 1;
    // the shift is the log of what one entry at this level covers
    uint64_t covers = g->mm.pte_covers[i];
    uint8_t shift = 0;
    while (covers > 1) { covers >>= 1; shift++; }
    pdes.levels[i].pageShift = shift;
  }
  if (rm_control(g, gsp->vaspace, TINYNV_CTRL_CMD_VASPACE_COPY_SERVER_RESERVED_PDES, &pdes, sizeof(pdes))) return -1;
  gsp->reserved_va = res_va;
  gsp->reserved_levels = nlevels;
  return 0;
}

// Create the channel work reaches the GPU through.
//
// The ring of command-buffer pointers lives in memory the GPU can address; the areas GSP-RM keeps the channel's state in
// are named by physical address, not virtual. One page holds both the ring and, above it, the small block the hardware
// writes its progress into, which is why the userd descriptor points into the same allocation at a fixed offset.
// The areas GSP-RM keeps a channel's own state in, which it wants by physical address and which the caller never sees.
static int fill_channel_state(tinynv_gpu_t *g, uint32_t client, tinynv_gpfifo_alloc_t *p, tinynv_vmap_t *ramfc_out,
                              tinynv_bootmem_t *mthd_out) {
  tinynv_vmap_t ramfc;
  if (tinynv_mm_valloc(&g->mm, 0x1000, 0x1000, 1, 0, &ramfc)) return -1;
  tinynv_bootmem_t mthd;
  if (tinynv_alloc_boot_mem(&g->mm, 0x5000, NULL, 0, &mthd)) return -1;
  p->ramfcMem = (tinynv_memory_desc_t){.base = ramfc.ranges[0].paddr, .size = 0x200, .addressSpace = 2};
  p->instanceMem = (tinynv_memory_desc_t){.base = ramfc.ranges[0].paddr, .size = 0x1000, .addressSpace = 2};
  p->mthdbufMem = (tinynv_memory_desc_t){.base = mthd.paddr, .size = 0x5000, .addressSpace = 2};

  // an ordinary client's channel also names where its error block and its progress block live
  if (client != TINYNV_RM_PRIV_ROOT && p->hObjectError) {
    // Truthful, and it has never been. addressSpace 0 is ADDR_UNKNOWN and the base was zero, while every other
    // descriptor on these lines correctly says 2 (ADDR_FBMEM). RM asserts the error notifier lives in video memory
    // (kchannelUpdateNotifierMem: NV_ASSERT_OR_RETURN(addressSpace == ADDR_FBMEM, NV_ERR_INVALID_STATE)) and so
    // every GET_WORK_SUBMIT_TOKEN this driver has ever issued came back 0x40 - AFTER writing the token, which is the
    // only reason the card works at all. hObjectError already holds the notifier's physical address.
    // The real size is 48 bytes: three NvNotification entries of 16 (NV_CHANNELGPFIFO_NOTIFICATION_TYPE__SIZE_1),
    // which is also what NVIDIA's own scrubber channel declares. 0xecc was invented.
    p->errorNotifierMem = (tinynv_memory_desc_t){.base = p->hObjectError, .size = 48, .addressSpace = 2};
    p->internalFlags |= (3u << 2);   // ERROR_NOTIFIER_TYPE = MEMORY (field 3:2), else RM sees UNKNOWN and asserts
    p->userdMem = (tinynv_memory_desc_t){.base = p->hUserdMemory[0] + p->userdOffset[0], .size = 0x400, .addressSpace = 2};
  }
  if (is610(g) && !p->hObjectEccError)
    p->internalFlags = (p->internalFlags & ~(3u << 4)) | (1u << 4);
  if (ramfc_out) *ramfc_out = ramfc;
  if (mthd_out) *mthd_out = mthd;
  return 0;
}

int tinynv_gsp_init_channel(tinynv_gpu_t *g) {
  tinynv_gsp_t *gsp = &g->gsp;
  const uint32_t entries = 32;

  if (tinynv_mm_valloc(&g->mm, 4 << 10, 0x1000, 1, 0, &gsp->gpfifo)) return -1;

  tinynv_gpfifo_alloc_t p;
  memset(&p, 0, sizeof(p));
  p.gpFifoOffset = gsp->gpfifo.va;
  p.gpFifoEntries = entries;
  p.engineType = 1; // graphics, which is what compute runs on
  p.cid = 3;
  p.hVASpace = gsp->vaspace;
  // No error/ECC notifier is allocated for the privileged channel. 610 consumes
  // these types directly: CTXDMA (2) with an empty descriptor fails memdescCreate.
  // Keep the historical 570 wire value; 610 uses KERNEL (2), NONE (1), NONE (1).
  p.internalFlags = is610(g) ? (2u | (1u << 2) | (1u << 4)) : 0x1a;
  p.flags = 0x200320;
  // the hardware's progress block sits just past the ring's entries, in the same page
  uint64_t userd_off = (uint64_t)entries * 8;
  p.userdOffset[0] = userd_off;
  if (fill_channel_state(g, TINYNV_RM_PRIV_ROOT, &p, &gsp->ramfc, &gsp->mthdbuf)) return -1;
  p.userdMem = (tinynv_memory_desc_t){.base = gsp->gpfifo.ranges[0].paddr + userd_off, .size = 0x20, .addressSpace = 2};

  if (rm_alloc(g, gsp->device, g->dev.class_gpfifo, &p, sizeof(p), &gsp->channel)) return -1;

  // which runlist this channel is on, which submitting work later has to name
  gsp->channel_runlist = 0;
  for (uint32_t i = 0; i < gsp->nengines; i++)
    if (gsp->engines[i] == p.engineType) gsp->channel_runlist = gsp->runlists[i];
  return 0;
}

static uint64_t round_up_to(uint64_t v, uint64_t a) { return a ? (v + a - 1) / a * a : v; }

// The buffers the graphics engine keeps its state in, and telling GSP-RM where they are.
//
// GSP-RM is asked how big each must be, the driver allocates them, and each is described by whichever of its addresses
// the engine wants: some by physical address, some by virtual, one by physical only. The identifiers and the peculiar
// index arithmetic below are the oracle's, and neither is documented anywhere else.
int tinynv_gsp_init_gr_context(tinynv_gpu_t *g) {
  tinynv_gsp_t *gsp = &g->gsp;

  tinynv_kgr_ctx_info_t *info = calloc(1, sizeof(*info));
  if (!info) return tinynv_fail("out of memory for the context buffer sizes");
  if (rm_control(g, gsp->subdevice, TINYNV_CTRL_CMD_STATIC_KGR_GET_CONTEXT_BUFFERS_INFO, info, sizeof(*info))) {
    free(info);
    return -1;
  }
  const tinynv_ctx_buffer_info_t *e = info->engineContextBuffersInfo[0].engine;

  // index 0 is the context itself, which gets a fixed extra allowance; 16 is the patch buffer; 17 upward are the
  // configuration areas, addressed here as 3..10 with an offset of 14, one of which wants 2 MB alignment
  uint64_t gr_size = gsp->gr_size = round_up_to((uint64_t)e[0].size + 0x40000, e[0].alignment);
  uint64_t patch_size = gsp->patch_size = round_up_to(e[16].size, e[16].alignment);
  uint64_t cfg[11];
  for (int x = 3; x <= 10; x++) cfg[x] = round_up_to(e[x + 14].size, x == 5 ? (2u << 20) : e[x + 14].alignment);

  // id, size, wanted by physical address, wanted by virtual address. id 1 is the engine's own local patch buffer, which
  // is allocated but never promoted, so it is not in this list at all.
  const struct { uint16_t id; uint64_t size; int phys, virt; } WANT[] = {
    {0, gr_size, 1, 1},    {2, patch_size, 1, 1},
    {3, cfg[3], 0, 1},     {4, cfg[4], 0, 1},   {5, cfg[5], 0, 1},   {6, cfg[6], 0, 1},
    {9, cfg[9], 1, 1},     {10, cfg[10], 1, 0}, {11, cfg[10], 1, 1},
  };
  const size_t n = sizeof(WANT) / sizeof(*WANT);

  tinynv_promote_ctx_t *prom = calloc(1, sizeof(*prom));
  if (!prom) { free(info); return tinynv_fail("out of memory for the context promotion"); }
  prom->engineType = 1;
  prom->hChanClient = TINYNV_RM_PRIV_ROOT;
  prom->hObject = gsp->channel;
  prom->entryCount = (uint32_t)n;
  for (size_t i = 0; i < n; i++) {
    tinynv_vmap_t *m = &gsp->grctx[i];
    if (tinynv_mm_valloc(&g->mm, WANT[i].size, 0x1000, 1, 0, m)) { free(prom); free(info); return -1; }
    prom->promoteEntry[i].bufferId = WANT[i].id;
    prom->promoteEntry[i].gpuVirtAddr = WANT[i].virt ? m->va : 0;
    prom->promoteEntry[i].bInitialize = (uint8_t)WANT[i].phys;
    prom->promoteEntry[i].gpuPhysAddr = WANT[i].phys ? m->ranges[0].paddr : 0;
    prom->promoteEntry[i].size = WANT[i].phys ? WANT[i].size : 0;
    prom->promoteEntry[i].physAttr = WANT[i].phys ? 4 : 0;
    prom->promoteEntry[i].bNonmapped = (uint8_t)(WANT[i].phys && !WANT[i].virt);
  }
  gsp->ngrctx = (int)n;

  int rc = rm_control(g, gsp->subdevice, TINYNV_CTRL_CMD_GPU_PROMOTE_CTX, prom, sizeof(*prom));
  free(prom);
  free(info);
  if (rc) return -1;

  // the classes the channel will actually run: compute, and the copy engine behind memcpy
  if (rm_alloc(g, gsp->channel, g->dev.class_compute, NULL, 0, &gsp->compute_obj)) return -1;
  return rm_alloc(g, gsp->channel, g->dev.class_dma_copy, NULL, 0, &gsp->dma_copy_obj);
}

// The client the driver works through, as against the privileged one that set the graphics context up.
//
// Its own device, address space and channel group, because the work submitted later is not privileged and GSP-RM treats
// the privileged client specially. The address space here is externally owned: the page tables are the driver's, and
// GSP-RM is told where their root is rather than building its own.
int tinynv_gsp_open_client(tinynv_gpu_t *g) {
  tinynv_gsp_t *gsp = &g->gsp;
  const uint32_t cl = TINYNV_RM_USER_ROOT;

  // before this client allocates anything, take the memory that has to stay where the processor can reach it
  if (tinynv_mm_reserve_bar_pool(&g->mm)) return -1;

  tinynv_nv0000_alloc_t root;
  memset(&root, 0, sizeof(root));
  if (rm_alloc_as(g, cl, 0, 0, TINYNV_CLASS_ROOT, &root, sizeof(root), &gsp->user_root)) return -1;

  tinynv_nv0080_alloc_t dev;
  memset(&dev, 0, sizeof(dev));
  dev.hClientShare = cl;
  dev.vaMode = 0; // several address spaces are allowed but not required
  if (rm_alloc_as(g, cl, cl, 0, TINYNV_CLASS_DEVICE, &dev, sizeof(dev), &gsp->user_device)) return -1;

  tinynv_nv2080_alloc_t sub;
  memset(&sub, 0, sizeof(sub));
  if (rm_alloc_as(g, cl, gsp->user_device, 0, TINYNV_CLASS_SUBDEVICE, &sub, sizeof(sub), &gsp->user_subdevice)) return -1;

  tinynv_memory_virtual_alloc_t vm;
  memset(&vm, 0, sizeof(vm));
  vm.limit = 0x1ffffffffffffull;
  if (rm_alloc_as(g, cl, gsp->user_device, 0, TINYNV_CLASS_MEMORY_VIRTUAL, &vm, sizeof(vm), &gsp->user_virtmem)) return -1;

  // ask for clocks: without this the chip runs at its idle point and a benchmark measures the governor
  tinynv_perf_boost_t boost;
  memset(&boost, 0, sizeof(boost));
  boost.duration = 0xffffffff;
  boost.flags = (TINYNV_PERF_BOOST_CUDA_YES << 4) | (TINYNV_PERF_BOOST_CUDA_PRIORITY_HIGH << 6) | TINYNV_PERF_BOOST_CMD_BOOST_TO_MAX;
  if (rm_control_as(g, cl, gsp->user_subdevice, TINYNV_CTRL_CMD_PERF_BOOST, &boost, sizeof(boost))) return -1;

  // An address space this client owns, but whose page tables the driver builds: "externally owned" means GSP-RM is told
  // where the root of the tree is rather than making one. Page faulting is on, which is what lets a kernel touch memory
  // that has not been paged in rather than dying.
  tinynv_vaspace_alloc_t va;
  memset(&va, 0, sizeof(va));
  va.vaBase = 0x1000;
  va.vaSize = 0x1fffffb000000ull;
  va.flags = TINYNV_VASPACE_FLAGS_ENABLE_PAGE_FAULTING | TINYNV_VASPACE_FLAGS_IS_EXTERNALLY_OWNED;
  if (rm_alloc_as(g, cl, gsp->user_device, 0, TINYNV_CLASS_VASPACE, &va, sizeof(va), &gsp->user_vaspace)) return -1;

  // and tell the firmware where that space's page tables begin, since it is not making them
  tinynv_set_page_directory_t pd;
  memset(&pd, 0, sizeof(pd));
  pd.hClient = cl;
  pd.hDevice = gsp->user_device;
  pd.pasid = 0xffffffff;
  pd.params.physAddress = g->mm.root_page_table;
  pd.params.numEntries = g->mm.pte_cnt[0];
  pd.params.flags = 0x8; // every channel in the space, rather than one named below
  pd.params.hVASpace = gsp->user_vaspace;
  pd.params.subDeviceId = 1;
  pd.params.pasid = 0xffffffff;
  if (rpc_call(g, TINYNV_MSG_FUNCTION_SET_PAGE_DIRECTORY, &pd, sizeof(pd), NULL, 0, "handing over the page tables")) return -1;

  tinynv_channel_group_alloc_t grp;
  memset(&grp, 0, sizeof(grp));
  grp.engineType = TINYNV_ENGINE_TYPE_GRAPHICS;
  if (rm_alloc_as(g, cl, gsp->user_device, 0, TINYNV_CLASS_CHANNEL_GROUP, &grp, sizeof(grp), &gsp->user_group)) return -1;

  // TINYNV_CE_GROUP=<engine type>: ask GSP-RM for a SECOND channel group, on a copy engine of its own, and report
  // what it says. Nothing is scheduled on it and nothing uses it - this allocates and stops, because the question
  // it answers is worth 20 lines before anyone writes 120.
  //
  // WHY: every copy this driver issues goes out on gsp->copy_q, which lives in the graphics group above. The engine
  // table says CE0 reports numPbdmas=1 with pbdmaIds [0,1] - the same PBDMA 0 that GR0 uses - and sits on runlist 0
  // with it, so a copy and a kernel cannot be in flight at once. CE1 reports runlist 1 and PBDMAs [8,9], which GR0
  // never touches. The expert stream measures 7.03 GiB/s against a 7.33 GiB/s PCIe 4.0 x4 ceiling, so it cannot be
  // made faster - only overlapped, and only if it can leave on a runlist of its own.
  //
  // This is NOT the 2026-09-18 experiment. That one set RUNQUEUE_ONE on a channel whose engine reports one PBDMA,
  // asking for a hardware lane that does not exist, and crashed two different ways (see the engine-table comment
  // above). This asks for a different ENGINE, which genuinely has its own PBDMA and its own runlist.
  {
    const char *ce = getenv("TINYNV_CE_GROUP");
    if (ce) {
      uint32_t et = (uint32_t)strtoul(ce, NULL, 0), rl = 0xffffffffu;
      for (uint32_t i = 0; i < gsp->nengines; i++) if (gsp->engines[i] == et) rl = gsp->runlists[i];
      if (rl == 0xffffffffu) {
        fprintf(stderr, "tinynv: CE group probe: no engine of type 0x%x in the table; not asking\n", et);
      } else {
        tinynv_channel_group_alloc_t g2;
        memset(&g2, 0, sizeof(g2));
        g2.engineType = et;
        uint32_t h = 0;
        int rc = rm_alloc_as(g, cl, gsp->user_device, 0, TINYNV_CLASS_CHANNEL_GROUP, &g2, sizeof(g2), &h);
        fprintf(stderr, "tinynv: CE group probe: engineType 0x%x (runlist %u) -> rc %d, handle 0x%x\n", et, rl, rc, h);
        if (!rc) {
          gsp->ce_group = h; gsp->ce_runlist = rl; gsp->ce_engine = et;
          tinynv_ctxshare_alloc_t cs;
          memset(&cs, 0, sizeof(cs));
          cs.hVASpace = gsp->user_vaspace;
          // SYNC, not ASYNC - and this one flag was the whole refusal. RM allows an ASYNC subcontext only on GR or on a
          // GRCE (kfifoValidateEngineAndSubctxType_GP102: "ASYNC Subcontext only supported on GR/GRCE"), and a CE counts
          // as a GRCE only when it shares GR0's runlist AND a PBDMA (_kfifoIsValidCETag_GP102). This CE is on a runlist
          // of its own - that separation is the entire point of allocating it - so it is not a GRCE and never can be.
          // The flag was copied from the graphics group's share below, where it is correct. Here it made GSP-RM refuse
          // the DMA_COPY object on an otherwise perfectly good channel, with a bare status and no reason.
          cs.flags = TINYNV_CTXSHARE_FLAGS_SUBCONTEXT_SYNC;
          if (rm_alloc_as(g, cl, h, 0, TINYNV_CLASS_CONTEXT_SHARE, &cs, sizeof(cs), &gsp->ce_ctxshare))
            fprintf(stderr, "tinynv: CE group probe: context share FAILED\n");
        }
      }
    }
  }

  // The scheduler's timeslice for this group, when asked for.
  //
  // A's raw columns put the token boundary's remaining ~700 us in the card scheduling two channels against each
  // other: the copy channel waiting to be picked up while compute finishes, or compute waiting after the copy ran.
  // On a bench with one channel active neither happens - 0 of 80 launches past 5 ms. A shorter timeslice should
  // shrink every such wait, not only this one, which is what makes it worth testing before any workaround that
  // avoids the second channel instead of explaining it.
  //
  // Off unless asked for. This is a scheduler setting on every channel we create, on a path with no recording behind
  // it, and the failure mode of a bad value is not a slow decode - it is a scheduling behaviour nothing here would
  // recognise.
  { const char *e = getenv("TINYNV_TIMESLICE_US");
    uint64_t us = e && *e ? strtoull(e, NULL, 10) : 0;
    if (us) {
      tinynv_timeslice_t ts = {.timesliceUs = us};
      if (rm_control_as(g, cl, gsp->user_group, TINYNV_CTRL_CMD_SET_TIMESLICE, &ts, sizeof(ts))) return -1;
      fprintf(stderr, "libtinynv: the channel group's timeslice was asked to be %llu us\n", (unsigned long long)us);
    } }

  tinynv_ctxshare_alloc_t share;
  memset(&share, 0, sizeof(share));
  share.hVASpace = gsp->user_vaspace;
  share.flags = TINYNV_CTXSHARE_FLAGS_SUBCONTEXT_ASYNC;
  if (rm_alloc_as(g, cl, gsp->user_group, 0, TINYNV_CLASS_CONTEXT_SHARE, &share, sizeof(share), &gsp->user_ctxshare)) return -1;
  { const char *e = getenv("TINYNV_SPLIT_CTXSHARE");
    if (e && *e && *e != '0') {
      tinynv_ctxshare_alloc_t cs;
      memset(&cs, 0, sizeof(cs));
      cs.hVASpace = gsp->user_vaspace;
      cs.flags = TINYNV_CTXSHARE_FLAGS_SUBCONTEXT_ASYNC;
      if (rm_alloc_as(g, cl, gsp->user_group, 0, TINYNV_CLASS_CONTEXT_SHARE, &cs, sizeof(cs), &gsp->copy_ctxshare))
        return -1;
      fprintf(stderr, "libtinynv: the copy channel was given a context share of its own (was asked for)\n");
    } }

  // the shape of the die: how many partitions it can have, not how many are turned on
  // the 610 static-info reply grew to 4032 B (63 entries per engine, rpc610); the indices read are the same.
  if (is610(g)) {
    uint8_t *raw = calloc(1, TINYNV_RPC610_GR_STATIC_SIZE);
    if (!raw) return tinynv_fail("out of memory for the chip's geometry");
    if (rm_control_as(g, cl, gsp->user_subdevice, TINYNV_CTRL_CMD_STATIC_KGR_GET_INFO, raw, TINYNV_RPC610_GR_STATIC_SIZE)) {
      free(raw);
      return -1;
    }
    int rc = tinynv_rpc610_gr_info_get(raw, TINYNV_RPC610_GR_STATIC_SIZE, 0, TINYNV_GR_INFO_NUM_GPCS, &gsp->max_gpcs) ||
             tinynv_rpc610_gr_info_get(raw, TINYNV_RPC610_GR_STATIC_SIZE, 0, TINYNV_GR_INFO_NUM_TPC_PER_GPC, &gsp->max_tpc_per_gpc) ||
             tinynv_rpc610_gr_info_get(raw, TINYNV_RPC610_GR_STATIC_SIZE, 0, TINYNV_GR_INFO_NUM_SM_PER_TPC, &gsp->max_sm_per_tpc) ||
             tinynv_rpc610_gr_info_get(raw, TINYNV_RPC610_GR_STATIC_SIZE, 0, TINYNV_GR_INFO_MAX_WARPS_PER_SM, &gsp->max_warps_per_sm) ||
             tinynv_rpc610_gr_info_get(raw, TINYNV_RPC610_GR_STATIC_SIZE, 0, TINYNV_GR_INFO_SM_VERSION, &gsp->sm_version);
    free(raw);
    if (rc) return -1;
  } else {
  tinynv_gr_static_info_t *info = calloc(1, sizeof(*info));
  if (!info) return tinynv_fail("out of memory for the chip's geometry");
  if (rm_control_as(g, cl, gsp->user_subdevice, TINYNV_CTRL_CMD_STATIC_KGR_GET_INFO, info, sizeof(*info))) { free(info); return -1; }
  const tinynv_gr_info_t *il = info->engineInfo[0].infoList;
  // these are maxima for the die, not the enabled counts; see the note in gsp.h before reporting them to anyone
  gsp->max_gpcs = il[TINYNV_GR_INFO_NUM_GPCS].data;
  gsp->max_tpc_per_gpc = il[TINYNV_GR_INFO_NUM_TPC_PER_GPC].data;
  gsp->max_sm_per_tpc = il[TINYNV_GR_INFO_NUM_SM_PER_TPC].data;
  gsp->max_warps_per_sm = il[TINYNV_GR_INFO_MAX_WARPS_PER_SM].data;
  gsp->sm_version = il[TINYNV_GR_INFO_SM_VERSION].data;
  free(info);
  }

  // A page of device memory with no processor mapping, allocated before anything else this client does. What it holds is
  // not visible from the boundary: it is unzeroed, unmapped from here, and one page table entry wide. It is reproduced
  // because leaving it out moves every address after it, which is how its absence was found.
  if (tinynv_mm_alloc_buffer(&g->mm, 0x1000, 0, 0, 0, 0, 0, &gsp->scratch)) return -1;

  // and the buffer copies are staged through. A remote cannot map the host's own pages, so every transfer in either
  // direction goes through this one allocation in chunks; it is taken once and kept for the life of the device.
  return tinynv_mm_alloc_buffer(&g->mm, 128ull << 20, 1, 0, 0, 0, 0, &gsp->staging);
}

// Hand the graphics engine a set of context buffers, saying for each whether the engine should take its physical
// address, its virtual address, or both. The same set is promoted twice: once by physical address so the engine can
// initialise them, then again by virtual address so it can reach them while running.
static int promote(tinynv_gpu_t *g, uint32_t client, uint32_t object, tinynv_vmap_t *bufs, const uint64_t *sizes, int n,
                   int by_phys) {
  tinynv_gsp_t *gsp = &g->gsp;
  tinynv_promote_ctx_t *prom = calloc(1, sizeof(*prom));
  if (!prom) return tinynv_fail("out of memory for a context promotion");
  prom->engineType = 1;
  prom->hChanClient = client;
  prom->hObject = object;
  prom->entryCount = (uint32_t)n;
  for (int i = 0; i < n; i++) {
    prom->promoteEntry[i].bufferId = (uint16_t)i;
    prom->promoteEntry[i].gpuVirtAddr = by_phys ? 0 : bufs[i].va;
    prom->promoteEntry[i].bInitialize = (uint8_t)by_phys;
    prom->promoteEntry[i].gpuPhysAddr = by_phys ? bufs[i].ranges[0].paddr : 0;
    prom->promoteEntry[i].size = by_phys ? sizes[i] : 0;
    prom->promoteEntry[i].physAttr = by_phys ? 4 : 0;
    prom->promoteEntry[i].bNonmapped = (uint8_t)by_phys; // physical only, so the engine is told not to expect a mapping
  }
  int rc = rm_control_as(g, client, gsp->user_subdevice, TINYNV_CTRL_CMD_GPU_PROMOTE_CTX, prom, sizeof(*prom));
  free(prom);
  return rc;
}

// One channel work is submitted through: its ring, the block the hardware writes its progress into just above the ring,
// and the object that says what kind of work it runs.
// group/runlist are parameters because a channel's ENGINE is decided by the group it is allocated in, and its
// doorbell must name that engine's runlist. Everything used to assume the graphics group and gsp->channel_runlist.
// The error notifier is FORTY-EIGHT BYTES of hardware: three NvNotification entries of 16
// (NV_CHANNELGPFIFO_NOTIFICATION_TYPE__SIZE_1 - ERROR, WORK_SUBMIT_TOKEN, KEY_ROTATION_STATUS), which is what
// NVIDIA's own scrubber channel declares and what this driver already tells GSP-RM in fill_channel_state.
//
// It is allocated at 48 MB anyway, a million times over, and that is not just waste: it comes from the same
// bottom-up allocator that serves the 256 MB processor-visible window, so two channels put the cursor near 232 MB
// and a third pushed it to 280 MB, past the window, which is what made the CE1 probe fail (2026-09-20).
//
// Shrinking it to a page was tried and REVERTED the same day, because it does not fail - it exposes something else:
//   fill_channel_state failed: mapping 0x1020011000+0x1000: 0x1020011000 is already mapped
// The boot-memory allocator and the general VA allocator hand out overlapping addresses, and the only reason that
// has never been seen is that a 48 MB hole sat between them. That is a real defect in the address space, it is
// worth fixing on its own terms, and it is not a side quest of a notifier resize. Until then the waste stays,
// documented, rather than a landmine relocated.
#define TINYNV_NOTIFIER_BYTES (48ull << 20)

static int new_queue(tinynv_gpu_t *g, tinynv_queue_t *q, uint64_t offset, uint32_t entries, int compute,
                     uint32_t group, uint32_t runlist, uint32_t ctxshare, uint32_t engine, uint64_t notifier_bytes) {
  tinynv_gsp_t *gsp = &g->gsp;
  const uint32_t cl = TINYNV_RM_USER_ROOT;

  // Where the engine reports faults. Nothing in this driver has ever read a byte of it, and its size is a parameter
  // only so the cost is visible at the call sites - see TINYNV_NOTIFIER_BYTES for what it should be and why it
  // is not that yet.
  if (tinynv_mm_alloc_buffer(&g->mm, notifier_bytes, 0, 0, 1, 0, 0, &q->notifier))
    return tinynv_fail("a queue's %llu MB error notifier could not be allocated - the reserve is exhausted, not RM "
                       "refusing", (unsigned long long)(notifier_bytes >> 20));

  tinynv_gpfifo_alloc_t p;
  memset(&p, 0, sizeof(p));
  p.gpFifoOffset = gsp->fifo_mem.va + offset;
  p.gpFifoEntries = entries;
  // A blocked semaphore acquire should end in a reported error, not a machine nobody can get back. Without this bit
  // RM leaves PBDMA_ACQUIRE_TIMEOUT_FALSE and a mis-ordered acquire waits forever: the host spins at 100% CPU, the
  // process has to be killed, and the card comes back only after a physical replug - which is exactly how the
  // 2026-09-18 runqueue-split attempt ended. With it, the same mistake costs about two seconds and an RC error.
  p.flags |= (1u << 27);   // NVOS04_FLAGS_CHANNEL_PBDMA_ACQUIRE_TIMEOUT 27:27 = TRUE
  // these handles are physical addresses: that is what the allocator hands back for memory the driver owns outright
  if (q->notifier.ranges[0].paddr > 0xffffffffull)
    return tinynv_fail("the error notifier is at %#llx, above what this field can carry in 32 bits",
                       (unsigned long long)q->notifier.ranges[0].paddr);
  p.hObjectError = (uint32_t)q->notifier.ranges[0].paddr;
  p.hObjectBuffer = (uint32_t)gsp->fifo_mem.ranges[0].paddr;
  p.hUserdMemory[0] = (uint32_t)gsp->fifo_mem.ranges[0].paddr;
  p.userdOffset[0] = (uint64_t)entries * 8 + offset;
  // Both channels have always shared ONE context share, which is one subcontext, which means the engine has to switch
  // context between them rather than holding both.
  //
  // Session A established the mechanism today: the compute and copy channels contend at every token boundary and the
  // runlist's switching decides who waits, costing ~700 us on about half the tokens. Their proposed cheap fix was to
  // put the two channels in one channel group - but they are ALREADY in one group, gsp->user_group, so that is not
  // available. The subcontext is the thing that is actually shared and not required to be.
  //
  // TINYNV_SPLIT_CTXSHARE=1 gives the copy channel its own, allocated with the same SUBCONTEXT_ASYNC flag. Off by
  // default and it is a guess, not a diagnosis: two channels in separate async subcontexts SHOULD be able to hold the
  // engine together where one subcontext cannot, but nothing here has measured that and no recording contains it.
  // The context share MUST belong to the same channel group as the channel. Passing one from another group is
  // rejected outright - which is how the first CE1 attempt failed on 2026-09-20: the group was granted, and then
  // the channel in it was handed the graphics group's share.
  // NVIDIA's kernel_channel.c (kchannelConstruct_IMPL, the per-runlist-CHRAM branch): if the channel names an
  // engine it MUST equal the TSG's, otherwise the channel inherits the TSG's. Left at 0 the graphics path has
  // always worked; a CE group needs it stated, because zero does not resolve to "unspecified" here.
  if (engine) p.engineType = engine;
  p.hContextShare = ctxshare ? ctxshare : gsp->user_ctxshare;
  if (!ctxshare && !compute && gsp->copy_ctxshare) p.hContextShare = gsp->copy_ctxshare;
  // A different axis from the ctxshare guess above, from NVIDIA's own header rather than a hypothesis about our
  // code: alloc_channel.h's NVOS04_FLAGS_GROUP_CHANNEL_RUNQUEUE (bit 4:4, GP10x+) "specifies which runqueue the
  // allocated channel will be executed on in a TSG. Channels on different runqueues within a TSG may be able to
  // feed methods into the engine simultaneously." Both our channels have always defaulted to runqueue 0 (`p.flags`
  // is memset to zero and nothing here has ever set this bit) - so the copy and compute channels contending for
  // one runlist slot at every token boundary (the mechanism this file's own comments above settled) has never
  // been tested against the one flag NVIDIA's own driver uses to let two channels in a TSG feed the engine at
  // once. TINYNV_RUNQUEUE_SPLIT=1 puts the copy channel on runqueue ONE; compute stays on the default. Off by
  // default: NVIDIA's own doc hedges ("may be able to"), and this could land like TINYNV_TIMESLICE_US - a real,
  // exercised mechanism that doesn't net out to a shorter token. One hardware run decides it.
  //
  // RUN, 2026-09-18: it does not just fail to help - it crashes. The GPFIFO alloc itself succeeds (GSP-RM accepts
  // the channel on runqueue ONE without complaint, boot proceeds, ggml_cuda_init reports the card normally), but
  // the FIRST real copy-engine job after it - the first H2D weight upload in llama_model_loader::load_all_data -
  // SIGSEGVs inside tinynv_exec_upload (NULL deref; symbolicated: tinynv_exec_upload+216 <- tinynv_memcpy_htod
  // <- cudaMemcpyAsync <- ggml_backend_cuda_buffer_set_tensor <- load_all_data). Card verified undamaged after
  // (preflight clean, no re-enumeration, op-verify 450/450 at chain depths 32/64/128 on the stock binary).
  //
  // RETRIED on a different build the same day: no crash this time, but a hang - the process spun at 100% CPU
  // (confirmed via ps: state R, not blocked) at the same point, consistent with tinynv_exec_idle's wait looping
  // on a completion value that never arrives. Had to be force-killed; the card then showed a genuine DART fault
  // (not the benign re-enumeration pattern this project usually sees), needing a physical replug. Two different
  // failure shapes, same program point, on two different builds - consistent with one root defect resolving as
  // undefined behaviour rather than two separate bugs.
  //
  // ROOT CAUSE FOUND, 2026-09-18, without touching hardware again: this driver has always fetched, and always
  // thrown away, exactly the fact that answers this. tinynv_gsp_init_objects's FIFO_GET_DEVICE_INFO_TABLE call
  // (above this function) returns numPbdmas/pbdmaIds per engine - printed for the first time on the ordinary
  // boot that produced this comment: GR0 (compute) reports numPbdmas=2, but EVERY copy engine instance (CE0
  // through CE7) reports numPbdmas=1. NVIDIA's real driver validates a channel's requested runqueue against its
  // OWN engine's PBDMA count before scheduling it (kfifoGetNumRunqueues_HAL, bounds-checked against
  // numSrcPbdmaIds in kernel_fifo_gm107.c); this driver's channel-alloc path never has, for any engine, on any
  // boot. So NVOS04_FLAGS_GROUP_CHANNEL_RUNQUEUE_ONE on the copy channel asks GSP-RM for a PBDMA index the copy
  // engine does not have - an out-of-range hardware selector that the allocation call happily accepts (nothing
  // in the accept path checks it), and which then produces undefined behaviour whose exact shape depends on
  // incidental memory layout - the crash and the hang are the same defect landing two different ways.
  //
  // This also settles the mechanism question, not just the crash: the flag's own wording ("channels on different
  // runqueues within a TSG may be able to feed methods into THE ENGINE simultaneously" - singular) describes two
  // channels of the SAME engine sharing that engine's PBDMA pair, not two channels of DIFFERENT engines (compute
  // vs copy) coordinating with each other. GR0 having 2 PBDMAs could in principle let two COMPUTE channels feed
  // concurrently; it says nothing about the compute<->copy handoff this project actually needs to shorten. Not a
  // missing-plumbing problem - the wrong mechanism for this job, now confirmed by hardware data rather than
  // inferred from a crash. Left OFF, in the tree, as the evidence - same status as TINYNV_SPLIT_CTXSHARE above.
  { const char *e = getenv("TINYNV_RUNQUEUE_SPLIT");
    if (e && *e && *e != '0' && !compute) {
      p.flags |= (1u << 4); // NVOS04_FLAGS_GROUP_CHANNEL_RUNQUEUE_ONE, field 4:4
      fprintf(stderr, "libtinynv: the copy channel was put on runqueue ONE (was asked for)\n");
    } }
  // The inner failure is printed rather than replaced: tinynv_fail writes into one buffer, so wrapping it here threw
  // away the message that said what actually went wrong and left only the name of the caller.
  if (fill_channel_state(g, cl, &p, NULL, NULL)) {
    fprintf(stderr, "tinynv: fill_channel_state failed for group %#x: %s\n", group, tinynv_last_error());
    return -1;
  }
  if (rm_alloc_as(g, cl, group, 0, g->dev.class_gpfifo, &p, sizeof(p), &q->channel))
    return tinynv_fail("GSP-RM refused a GPFIFO channel in group %#x (engine %#x, runlist %u)", group, engine, runlist);

  if (compute) {
    if (rm_alloc_as(g, cl, q->channel, 0, g->dev.class_compute, NULL, 0, &q->object)) return -1;

    // A compute object on an ordinary client gets its own copy of the first three context buffers, promoted twice: once
    // by physical address so the engine can initialise them, then by virtual address so it can reach them.
    const uint64_t sizes[3] = {gsp->gr_size, gsp->patch_size, gsp->patch_size};
    for (int i = 0; i < 3; i++)
      if (tinynv_mm_valloc(&g->mm, sizes[i], 0x1000, 1, 0, &q->grctx[i])) return -1;
    if (promote(g, cl, q->channel, q->grctx, sizes, 3, 1)) return -1;

    // The second promotion allocates a second set and then does not use it. That is the oracle's accident, not a design:
    // it reaches for the already-allocated buffer with a dictionary lookup whose default is the allocation call, and
    // python evaluates that default whether or not the key is there. A membership test, or a default that is not
    // evaluated until it is needed, was meant. It is in the recording, so it is reproduced -- and it costs about seven
    // megabytes of video memory per compute channel, which is worth knowing before anyone wonders where it went.
    //
    // It is an upstream bug and worth reporting there, but NOT before this driver stops depending on the recording
    // matching: fixing it in the oracle would change the trace we pinned, and the pin is what makes the first hardware
    // run a sequence the card has already accepted. After that, both the fix upstream and the trim here. (Session A.)
    tinynv_vmap_t discarded[3];
    for (int i = 0; i < 3; i++)
      if (tinynv_mm_valloc(&g->mm, sizes[i], 0x1000, 1, 0, &discarded[i])) return -1;
    if (promote(g, cl, q->channel, q->grctx, sizes, 3, 0)) return -1;

    tinynv_debugger_alloc_t dbg;
    memset(&dbg, 0, sizeof(dbg));
    dbg.hAppClient = cl;
    dbg.hClass3dObject = q->object;
    if (rm_alloc_as(g, cl, gsp->user_device, 0, TINYNV_CLASS_DEBUGGER, &dbg, sizeof(dbg), &gsp->user_debugger)) return -1;
  } else {
    // Only stated when the queue names an engine: the graphics-group copy channel has always allocated this object
    // with NULL params and works, and changing that path is a risk with nothing to gain.
    tinynv_dma_copy_alloc_t ce;
    memset(&ce, 0, sizeof(ce));
    ce.version = TINYNV_DMA_COPY_PARAMS_VERSION_1;
    ce.engineType = engine;
    // The copy class is the chip's own (dev.c chose it); TINYNV_CE_CLASS overrides it for an engine-named channel
    // only: BLACKWELL_DMA_COPY_B (0xcab5) is what the graphics-group channel uses; _A is 0xc9b5. Which classes a
    // given CE advertises is not documented anywhere we have, so it is a knob rather than an assumption.
    uint32_t cls = g->dev.class_dma_copy;
    if (engine) { const char *cs2 = getenv("TINYNV_CE_CLASS"); if (cs2) cls = (uint32_t)strtoul(cs2, NULL, 0); }
    if (rm_alloc_as(g, cl, q->channel, 0, cls, engine ? &ce : NULL, engine ? sizeof(ce) : 0, &q->object))
      return tinynv_fail("GSP-RM refused copy class %#x on the channel in group %#x (engine %#x)", cls, group, engine);
  }

  // the token names this channel at the doorbell. gsp-rm fills in only the channel part; the runlist and, on this
  // architecture, an enable bit are the driver's to add.
  tinynv_work_submit_token_t tok;
  memset(&tok, 0, sizeof(tok));
  tok.workSubmitToken = 0xffffffffu;
  if (rm_control_as(g, cl, q->channel, TINYNV_CTRL_CMD_GET_WORK_SUBMIT_TOKEN, &tok, sizeof(tok))) return -1;
  // The sentinel surviving means RM never wrote a token, and OR-ing it produced a doorbell naming channel 0xfff on
  // runlist 0x7f - which is not refused by anything, it is simply never fetched, so the ring stops and the host
  // spins forever. Refusing here turns the worst failure this path has into a sentence.
  if (tok.workSubmitToken == 0xffffffffu)
    return tinynv_fail("gsp-rm did not fill in a work submit token for channel %#x (rm status %#x); the doorbell "
                       "would be invented, and an invented doorbell is never rung by anything real", q->channel,
                       gsp->last_ctrl_status);
  if (gsp->last_ctrl_status)
    fprintf(stderr, "tinynv: gsp-rm answered the work submit token control with status %#x but did fill the token "
                    "in (%#x) - using it, and that is luck rather than contract\n",
            gsp->last_ctrl_status, tok.workSubmitToken);
  // GB202's doorbell is VECTOR 11:0 (the channel id), RUNLIST_ID 22:16 and RUNLIST_DOORBELL 30:30 (gb202/dev_vm.h).
  // Take ONLY the channel id from RM and place the other two fields here, rather than OR-ing the whole word: it is
  // not documented whether GSP hands back a bare chid or a complete doorbell, and an OR of two different runlist
  // ids yields a third that belongs to neither.
  { uint32_t rm_rl = (tok.workSubmitToken >> 16) & 0x7f;
    if (rm_rl && rm_rl != runlist)
      fprintf(stderr, "tinynv: gsp-rm put runlist %u in the token for a channel this driver placed on runlist %u - "
                      "one of the two is wrong, and the doorbell can only be built from one of them\n", rm_rl, runlist); }
  q->token = (tok.workSubmitToken & 0xfffu) | (runlist << 16) | (1u << 30);
  q->entries = entries;
  q->ring_va = gsp->fifo_mem.va + offset;
  // The control block sits immediately above the ring - that is what userdOffset told GSP-RM a moment ago - and GPPut is
  // a field inside it. Its offset is NVIDIA's, checked against the vendored header in test_headers; it was guessed at
  // 0x10 until the first real submission went nowhere, because a boot recording contains no submissions and nothing in
  // the suite could tell a wrong address from a right one.
  q->gpput_off = offset + (uint64_t)entries * 8 + TINYNV_GPFIFO_GPPUT_OFF;
  return 0;
}

// The compute and copy channels, then scheduling for the group they belong to.
int tinynv_gsp_init_queues(tinynv_gpu_t *g) {
  tinynv_gsp_t *gsp = &g->gsp;
  // one allocation holds both rings and both control blocks, and it must be writable from here, so it comes out of the
  // reserve rather than wherever the physical allocator happens to be
  if (tinynv_mm_alloc_buffer(&g->mm, 3ull << 20, 0, 1, 0, 1, 0, &gsp->fifo_mem)) return -1;

  if (new_queue(g, &gsp->compute_q, 0, 0x10000, 1, gsp->user_group, gsp->channel_runlist, 0, 0, TINYNV_NOTIFIER_BYTES)) return -1;
  if (new_queue(g, &gsp->copy_q, 0x100000, 0x10000, 0, gsp->user_group, gsp->channel_runlist, 0, 0, TINYNV_NOTIFIER_BYTES)) return -1;

  // TINYNV_CE_GROUP: carry the probe one step further - a real CHANNEL in the copy-engine group, with its own
  // DMA_COPY object, and the group scheduled. Still no traffic: this only answers whether the allocation path
  // works end to end before anyone writes the routing and the cross-channel ordering that would use it.
  if (gsp->ce_group) {
    if (new_queue(g, &gsp->ce_copy_q, 0x200000, 0x10000, 0, gsp->ce_group, gsp->ce_runlist, gsp->ce_ctxshare,
                  gsp->ce_engine, TINYNV_NOTIFIER_BYTES)) {
      fprintf(stderr, "tinynv: CE channel probe: FAILED in group 0x%x: %s\n", gsp->ce_group, tinynv_last_error());
    } else {
      int rc = gsp_schedule_group(g, TINYNV_RM_USER_ROOT, gsp->ce_group);
      fprintf(stderr, "tinynv: CE channel probe: channel 0x%x token 0x%x on runlist %u, schedule rc %d\n",
              gsp->ce_copy_q.channel, gsp->ce_copy_q.token, gsp->ce_runlist, rc);
    }
  }

  return gsp_schedule_group(g, TINYNV_RM_USER_ROOT, gsp->user_group);
}

// schedule one channel group. The 610 params grew 2 -> 3 B (bSkipEnable appended, zero - the driver wants
// the group enabled); the 570 payload is unchanged. One helper so both call sites cannot drift apart.
static int gsp_schedule_group(tinynv_gpu_t *g, uint32_t client, uint32_t group) {
  if (is610(g)) {
    tinynv_rpc610_gpfifo_schedule_t s;
    tinynv_rpc610_schedule_fill(&s, 1, 0);
    return rm_control_as(g, client, group, TINYNV_CTRL_CMD_GPFIFO_SCHEDULE, &s, sizeof(s));
  }
  tinynv_gpfifo_schedule_t sched;
  memset(&sched, 0, sizeof(sched));
  sched.bEnable = 1;
  return rm_control_as(g, client, group, TINYNV_CTRL_CMD_GPFIFO_SCHEDULE, &sched, sizeof(sched));
}

// The queue GSP-RM answers on, and the arguments that tell it where the queue is.
static int init_rm_args(tinynv_gpu_t *g) {
  tinynv_gsp_t *gsp = &g->gsp;
  const uint64_t queue_size = gsp->queue_size = 0x40000;

  // the gpu reaches the queues through a page table of its own, which lives in the same allocation just below them
  uint64_t queue_pte_cnt = (queue_size * 2) / PAGE;
  uint64_t pte_cnt = queue_pte_cnt + round_up(queue_pte_cnt * 8, PAGE) / PAGE;
  uint64_t pt_size = round_up(pte_cnt * 8, PAGE);
  if (tinynv_alloc_boot_mem(&g->mm, pt_size + queue_size * 2, NULL, 1, &gsp->queues)) return -1;
  gsp->cmd_q_off = pt_size;

  for (size_t i = 0; i < gsp->queues.naddrs; i++) nv_wr_block(&gsp->queues.view, i * 8, &gsp->queues.addrs[i], 8);

  // the 610 cached-args block grew 72 -> 152 B and its msgq init args 32 -> 64 B (boot610). The encoder
  // fills the same logical inputs plus the five new queue-geometry fields; the 570 block is unchanged.
  if (is610(g)) {
    tinynv_boot610_args_cached_t args;
    tinynv_boot610_fill_args_cached(&args, gsp->queues.addrs[0], (uint32_t)pte_cnt, pt_size, pt_size + queue_size);
    // kgspAllocRmStateMonitorBuffer + kgspPopulateGspRmInitArgs in pinned 610 kernel_gsp.c:
    // INIT maps this page unconditionally. A zero-sized pair returns LibosErrorArgument before RM starts.
    // LOGRMON is a separate diagnostic ring and cannot stand in for this shared state page.
    uint8_t zero[PAGE];
    memset(zero, 0, sizeof zero);
    if (tinynv_alloc_boot_mem(&g->mm, PAGE, zero, -1, &gsp->rm_state_monitor)) return -1;
    args.rmStateMonitorBufferArgs[0] = gsp->rm_state_monitor.addrs[0];
    args.rmStateMonitorBufferArgs[1] = PAGE;
    // KernelMemorySystem's statePreLoad asserts the PFB sysmem-flush address registers are non-zero
    // (kmemsysAssertSysmemFlushBufferValid_GA100); 610 made that fatal where 570 tolerated it. The CPU-RM answers it
    // with a resident sysmem page (kmemsysInitFlushSysmemBuffer_GA100). Allocate one here, explicitly in sysmem: the
    // GPU's flush is a sysmem read, so a VRAM page would be wrong, and -1 would resolve to VRAM on a large-BAR card.
    // The register itself is programmed in the hardware path (tinynv_gsp_program_sysmem_flush); init_sw stays GPU-free.
    if (tinynv_alloc_boot_mem(&g->mm, PAGE, zero, 1, &gsp->sysmem_flush)) return -1;
    // zero the whole args page before placing the 152-byte block - the DMA pages carry stale bytes (see
    // init_libos_args), and the RMARGS region hands libos a full page.
    if (tinynv_alloc_boot_mem(&g->mm, sizeof(args), NULL, -1, &gsp->rm_args)) return -1;
    nv_wr_block(&gsp->rm_args.view, 0, zero, PAGE);
    nv_wr_block(&gsp->rm_args.view, 0, &args, sizeof args);
  } else {
    tinynv_gsp_args_cached_t args;
    memset(&args, 0, sizeof(args));
    args.bDmemStack = 1;
    args.messageQueueInitArguments.sharedMemPhysAddr = gsp->queues.addrs[0];
    args.messageQueueInitArguments.pageTableEntryCount = (uint32_t)pte_cnt;
    args.messageQueueInitArguments.cmdQueueOffset = pt_size;
    args.messageQueueInitArguments.statQueueOffset = pt_size + queue_size;
    if (tinynv_alloc_boot_mem(&g->mm, sizeof(args), &args, -1, &gsp->rm_args)) return -1;
  }
  gsp->rm_args_sysmem = gsp->rm_args.addrs[0];

  tinynv_msgq_tx_header_t hdr;
  memset(&hdr, 0, sizeof(hdr));
  hdr.version = 0;
  hdr.size = (uint32_t)queue_size;
  hdr.msgSize = (uint32_t)PAGE;
  hdr.msgCount = (uint32_t)((queue_size - PAGE) / PAGE);
  hdr.writePtr = 0;
  hdr.flags = 1;
  hdr.rxHdrOff = (uint32_t)sizeof(hdr);
  hdr.entryOff = (uint32_t)PAGE;
  nv_wr_block(&gsp->queues.view, pt_size, &hdr, sizeof(hdr));

  // read the header back the way the oracle does, so what the driver believes about the queue comes from the queue
  return rpcq_attach(g, &gsp->cmd_q, pt_size, "command");
}

// The log buffers and the argument block GSP-RM's operating system starts with.
static int init_libos_args(tinynv_gpu_t *g) {
  tinynv_gsp_t *gsp = &g->gsp;
  static const char *LOGS[] = {"INIT", "INTR", "RM", "MNOC", "KRNL"};

  if (tinynv_alloc_boot_mem(&g->mm, 2 * MB, NULL, -1, &gsp->logbuf)) return -1;
  // VRAM keeps its old contents across boots, and the libos log buffers are rings whose first word is the
  // put pointer - stale words past it read as log entries from a previous boot. Zero the 610 logs so a silent boot's
  // dump is truthful. 610-gated: the 570 recorded operation stream stays byte-identical.
  if (is610(g) || getenv("TINYNV_DUMP_GSP_LOGS")) {
    uint8_t zero[0x10000];
    memset(zero, 0, sizeof zero);
    for (uint64_t off = 0; off < 2 * MB; off += sizeof zero) nv_wr_block(&gsp->logbuf.view, off, zero, sizeof zero);
  }
  if (tinynv_alloc_boot_mem(&g->mm, PAGE, NULL, -1, &gsp->libos_args)) return -1;
  // the vendor portMemSet's the WHOLE 4 KiB init-args block before filling the regions (kgspSetupLibosInitArgs),
  // and libos scans it until a zero id8. The TinyGPU DMA pages are NOT zeroed (the stale log buffers prove it), so the
  // bytes past the six regions are whatever the previous boot left there - garbage regions. Zero the block first.
  // 610-gated: the 570 recorded operation stream stays byte-identical.
  if (is610(g)) {
    uint8_t zero[PAGE];
    memset(zero, 0, sizeof zero);
    nv_wr_block(&gsp->libos_args.view, 0, zero, PAGE);
  }
  gsp->libos_args_sysmem = gsp->libos_args.addrs[0];

  tinynv_libos_region_t regions[9];
  memset(regions, 0, sizeof(regions));
  int nregions;
  if (is610(g)) {
    // the vendor's 610 region set and order (see LOGS610 above); RMARGS last, exactly as kgspSetupLibosInitArgs
    for (int i = 0; i < 8; i++) {
      regions[i].kind = TINYNV_LIBOS_MEMORY_REGION_CONTIGUOUS;
      regions[i].loc = TINYNV_LIBOS_MEMORY_REGION_LOC_SYSMEM;
      regions[i].size = LOGS610[i].size;
      regions[i].id8 = id8_of(LOGS610[i].name);
      regions[i].pa = gsp->logbuf.addrs[0] + LOGS610[i].off;
    }
    nregions = 8;
  } else {
    for (int i = 0; i < 5; i++) {
      char name[16];
      snprintf(name, sizeof(name), "LOG%s", LOGS[i]);
      regions[i].kind = TINYNV_LIBOS_MEMORY_REGION_CONTIGUOUS;
      regions[i].loc = TINYNV_LIBOS_MEMORY_REGION_LOC_SYSMEM;
      regions[i].size = 0x10000;
      regions[i].id8 = id8_of(name);
      regions[i].pa = gsp->logbuf.addrs[0] + 0x10000ull * i;
    }
    nregions = 5;
  }
  regions[nregions].kind = TINYNV_LIBOS_MEMORY_REGION_CONTIGUOUS;
  regions[nregions].loc = TINYNV_LIBOS_MEMORY_REGION_LOC_SYSMEM;
  regions[nregions].size = PAGE;
  regions[nregions].id8 = id8_of("RMARGS");
  regions[nregions].pa = gsp->rm_args_sysmem;
  nv_wr_block(&gsp->libos_args.view, 0, regions, sizeof(tinynv_libos_region_t) * (nregions + 1));
  return 0;
}

// GSP-RM's own firmware, under the three level page table the GPU walks to read it.
//
// The image is 60 MB and the GPU reads it by physical page, so it is described by a radix tree: a root page of pointers
// to a page of pointers to pages of pointers to the image's pages. Each level is sized from the one below it, which is
// why the loop runs backwards.
static int init_gsp_image(tinynv_gpu_t *g) {
  tinynv_gsp_t *gsp = &g->gsp;
  if (is610(g) && tinynv_failpoint("610-gsp-image")) return -1;
  // the oracle loads this one from the ga102 directory whatever the chip is: it is the same file for every architecture.
  // the version infix comes from the frozen runtime profile (570: TINYNV_FW_VER, byte-identical name).
  // the 610 GSP ELF is a bundle asset (fw610), hash-pinned to the 610 manifest - never the 570 ga102 file.
  if (is610(g)) {
    if (tinynv_fw610_load("gsp", &gsp->gsp_fw)) return -1;
  } else {
    char fw_name[64];
    snprintf(fw_name, sizeof fw_name, "gsp-%s.bin", g->profile->version);
    if (tinynv_fw_load("ga102", fw_name, GSP_SHA, &gsp->gsp_fw)) return -1;
  }

  const uint8_t *image, *sig;
  size_t image_len, sig_len;
  char sig_name[64];
  // ".fwsignature_gb20x" for a GB202: the family, lowercased, with the last digit replaced by an x
  snprintf(sig_name, sizeof(sig_name), ".fwsignature_%c%c%c%cx", lower_ascii(g->dev.chip_name[0]), lower_ascii(g->dev.chip_name[1]),
           lower_ascii(g->dev.chip_name[2]), lower_ascii(g->dev.chip_name[3]));
  if (tinynv_elf_section(&gsp->gsp_fw, ".fwimage", &image, &image_len)) return -1;
  if (tinynv_elf_section(&gsp->gsp_fw, sig_name, &sig, &sig_len)) return -1;
  gsp->gsp_image_size = image_len;
  // The reservation at the top of video memory was sized against this image (fw_layout.h): a bigger one is refused
  // here, where the message can say why, rather than placed under the manager's feet.
  if (image_len > g->profile->image_bound)
    return tinynv_fail("the GSP-RM image is %zu bytes, past the %llu MB the firmware reservation was sized for - raise "
                       "TINYNV_FW_IMAGE_BOUND and the reservation with it (fw_layout.h)",
                       image_len, (unsigned long long)(g->profile->image_bound >> 20));

  size_t npages[4], offsets[4] = {0, 0, 0, 0};
  npages[3] = round_up(image_len, PAGE) / PAGE;
  for (int i = 3; i > 0; i--) npages[i - 1] = ((npages[i] - 1) >> (TINYNV_LIBOS_MEMORY_REGION_RADIX_PAGE_LOG2 - 3)) + 1;
  for (int i = 1; i < 4; i++) offsets[i] = offsets[i - 1] + npages[i - 1] * PAGE;

  if (tinynv_alloc_boot_mem(&g->mm, offsets[3] + image_len, NULL, -1, &gsp->radix3)) return -1;
  nv_wr_block(&gsp->radix3.view, offsets[3], image, image_len);

  // each level holds the addresses of the pages of the level below it
  size_t cur = 0;
  for (int i = 0; i < 3; i++) {
    cur += npages[i];
    if (cur + npages[i + 1] > gsp->radix3.naddrs) return tinynv_fail("the gsp image page table does not fit its allocation");
    nv_wr_block(&gsp->radix3.view, offsets[i], &gsp->radix3.addrs[cur], npages[i + 1] * 8);
  }

  return tinynv_alloc_boot_mem(&g->mm, sig_len, sig, -1, &gsp->signature);
}

// The RISC-V bootloader that starts GSP-RM, in NVIDIA's own container rather than an ELF.
static int init_boot_binary_image(tinynv_gpu_t *g) {
  tinynv_gsp_t *gsp = &g->gsp;
  if (is610(g) && tinynv_failpoint("610-bootloader")) return -1;
  // the 610 bootloader is a RAW image plus a SEPARATE 84-byte v5 descriptor (boot610), not the 570 NVFW
  // container. Parse the descriptor against the raw image, keep the three offsets the shared WPR-meta fill reads
  // (and appVersion for the host-request core-resume), and place the raw bytes. No cast between the two layouts.
  if (is610(g)) {
    if (tinynv_fw610_load("bootloader", &gsp->bl_fw)) return -1;
    if (tinynv_fw610_load("bootloader_desc", &gsp->bl_desc_fw)) return -1;
    if (tinynv_boot610_desc_parse(gsp->bl_desc_fw.data, gsp->bl_desc_fw.size, (uint32_t)gsp->bl_fw.size,
                                 &gsp->bl_desc610)) return -1;
    gsp->bootloader_size = gsp->bl_fw.size;
    if (gsp->bootloader_size > g->profile->bootbin_bound)
      return tinynv_fail("the 610 riscv bootloader is %zu bytes, past the %llu MB the firmware reservation was sized "
                         "for (fw_layout_610.h)", gsp->bootloader_size,
                         (unsigned long long)(g->profile->bootbin_bound >> 20));
    // the shared descriptor fields the WPR-meta fill and the core-resume step read, at their unchanged wire offsets
    memset(&gsp->bl_desc, 0, sizeof(gsp->bl_desc));
    gsp->bl_desc.manifestOffset = gsp->bl_desc610.manifest_offset;
    gsp->bl_desc.monitorDataOffset = gsp->bl_desc610.data_offset;
    gsp->bl_desc.monitorCodeOffset = gsp->bl_desc610.code_offset;
    memcpy(&gsp->bl_desc.appVersion, gsp->bl_desc_fw.data + 28, 4);
    return tinynv_alloc_boot_mem(&g->mm, gsp->bl_fw.size, gsp->bl_fw.data, -1, &gsp->bootloader);
  }
  const char *sha = bl_sha(g->dev.fw_name);
  if (!sha) return tinynv_fail("no riscv bootloader hash is pinned for %s, so its image cannot be trusted", g->dev.fw_name);
  char bl_name[64];
  snprintf(bl_name, sizeof bl_name, "bootloader-%s.bin", g->profile->version);
  if (tinynv_fw_load(g->dev.fw_name, bl_name, sha, &gsp->bl_fw)) return -1;

  tinynv_nvfw_bin_hdr_t h;
  if (gsp->bl_fw.size < sizeof(h)) return tinynv_fail("the riscv bootloader is truncated");
  memcpy(&h, gsp->bl_fw.data, sizeof(h));
  if ((uint64_t)h.data_offset + h.data_size > gsp->bl_fw.size || (uint64_t)h.header_offset + sizeof(gsp->bl_desc) > gsp->bl_fw.size)
    return tinynv_fail("the riscv bootloader's header points outside the file");
  memcpy(&gsp->bl_desc, gsp->bl_fw.data + h.header_offset, sizeof(gsp->bl_desc));
  gsp->bootloader_size = h.data_size;
  if (h.data_size > g->profile->bootbin_bound)   // same reason as the image bound above
    return tinynv_fail("the riscv bootloader is %u bytes, past the %llu MB the firmware reservation was sized for "
                       "(fw_layout.h)", h.data_size, (unsigned long long)(g->profile->bootbin_bound >> 20));
  return tinynv_alloc_boot_mem(&g->mm, h.data_size, gsp->bl_fw.data + h.data_offset, -1, &gsp->bootloader);
}

// One structure naming everything above, which is what the chain of trust message actually points at.
static int init_wpr_meta(tinynv_gpu_t *g) {
  tinynv_gsp_t *gsp = &g->gsp;
  if (init_gsp_image(g)) return -1;
  if (init_boot_binary_image(g)) return -1;

  // the 610 carveout is re-derived from the matching pinned sizes (boot610/fw_layout_610), overflow-checked
  // and bounds-verified, not the 570 arithmetic with a bigger image dropped in. The WPR meta block itself is
  // byte-identical (asserted in test_headers_610), so the shared fill places every boundary.
  if (is610(g)) {
    if (tinynv_failpoint("610-wpr-meta")) return -1;
    if (g->dev.fmc_boot)
      return tinynv_fail("the 610.43.03 profile is the AD102 vbios boot path; a chain-of-trust device cannot run it");
    tinynv_boot610_layout_t l;
    if (tinynv_boot610_vbios_layout(g->dev.vram_size, gsp->gsp_image_size, gsp->bootloader_size, &l)) return -1;
    if (g->flcn.frts_offset != l.frts_offset)
      return tinynv_fail("fwsec was told to place the protected region at %#llx but the 610 metadata says %#llx",
                         (unsigned long long)g->flcn.frts_offset, (unsigned long long)l.frts_offset);
    tinynv_wpr_meta_t m;
    tinynv_boot610_fill_wpr_meta(&m, &l, gsp->radix3.addrs[0], gsp->gsp_image_size, gsp->bootloader.addrs[0],
                                 gsp->bootloader_size, &gsp->bl_desc610, gsp->signature.addrs[0]);
    if (tinynv_alloc_boot_mem(&g->mm, sizeof(m), &m, -1, &gsp->wpr_meta)) return -1;
    gsp->wpr_meta_sysmem = gsp->wpr_meta.addrs[0];
    return 0;
  }

  tinynv_wpr_meta_t m;
  memset(&m, 0, sizeof(m));
  m.magic = TINYNV_GSP_FW_WPR_META_MAGIC;
  m.revision = TINYNV_GSP_FW_WPR_META_REVISION;
  m.sysmemAddrOfRadix3Elf = gsp->radix3.addrs[0];
  m.sizeOfRadix3Elf = gsp->gsp_image_size;
  m.sysmemAddrOfBootloader = gsp->bootloader.addrs[0];
  m.sizeOfBootloader = gsp->bootloader_size;
  m.sysmemAddrOfSignature = gsp->signature.addrs[0];
  m.sizeOfSignature = 0x1000;
  m.bootloaderCodeOffset = gsp->bl_desc.monitorCodeOffset;
  m.bootloaderDataOffset = gsp->bl_desc.monitorDataOffset;
  m.bootloaderManifestOffset = gsp->bl_desc.manifestOffset;

  if (g->dev.fmc_boot) {
    // on the chain-of-trust path the firmware places its own carveout, so these are sizes and not offsets. They are
    // named in fw_layout.h, which is also what sizes the reservation the memory manager holds back for them: change one
    // there and the static assert beside them says whether the reservation still covers it.
    m.vgaWorkspaceSize = TINYNV_FW_VGA_WORKSPACE;
    m.pmuReservedSize = TINYNV_FW_PMU_RESERVED;
    m.nonWprHeapSize = TINYNV_FW_NONWPR_HEAP;
    m.gspFwHeapSize = TINYNV_FW_HEAP_SIZE;
    m.frtsSize = TINYNV_FW_FRTS_SIZE;
  } else {
    // On the vbios path nothing lays the carveout out for us, so the driver does it and hands GSP-RM every boundary.
    // Downwards from the top of video memory: the vga workspace, the write-protected region FWSEC has already placed,
    // the boot binary, gsp-rm's own image, its heap, and the non-wpr heap under that. Every boundary is rounded DOWN,
    // so a region that does not divide evenly eats into its own space rather than into the one below it.
    uint64_t vga_sz = 0x100000, vga_off = g->dev.vram_size - vga_sz;
    uint64_t frts_sz = 0x100000, frts_off = vga_off - frts_sz;
    uint64_t boot_off = frts_off - gsp->bootloader_size;
    uint64_t gsp_off = round_down(boot_off - gsp->gsp_image_size, 0x10000);
    uint64_t heap_sz = 0x8100000, heap_off = round_down(gsp_off - heap_sz, 0x100000);
    uint64_t wpr_start = round_down(heap_off - PAGE, 0x100000);
    uint64_t non_wpr_sz = 0x100000, non_wpr_off = round_down(wpr_start - non_wpr_sz, 0x100000);

    m.vgaWorkspaceSize = vga_sz;
    m.vgaWorkspaceOffset = vga_off;
    m.gspFwWprEnd = vga_off;
    m.frtsSize = frts_sz;
    m.frtsOffset = frts_off;
    m.bootBinOffset = boot_off;
    m.gspFwOffset = gsp_off;
    m.gspFwHeapSize = heap_sz;
    m.gspFwHeapOffset = heap_off;
    m.gspFwWprStart = wpr_start;
    m.nonWprHeapSize = non_wpr_sz;
    m.nonWprHeapOffset = non_wpr_off;
    m.gspFwRsvdStart = non_wpr_off;
    m.fbSize = g->dev.vram_size;

    // FWSEC was told where to place the region before this structure existed, from its own arithmetic in flcn.c. If the
    // two ever disagree, GSP-RM is handed a region that is not the one on the chip and nothing says so - so the oracle
    // compares them, and this does too.
    if (g->flcn.frts_offset != m.frtsOffset)
      return tinynv_fail("fwsec was told to place the protected region at %#llx but the metadata says %#llx",
                         (unsigned long long)g->flcn.frts_offset, (unsigned long long)m.frtsOffset);
  }

  if (tinynv_alloc_boot_mem(&g->mm, sizeof(m), &m, -1, &gsp->wpr_meta)) return -1;
  gsp->wpr_meta_sysmem = gsp->wpr_meta.addrs[0];
  return 0;
}

// The bus address GSP-RM is told, packed the way NVIDIA packs it. A name that is not a PCI address gives zero, which
// covers the socket backend over thunderbolt and the replay backend, and is what the oracle reports for them.
static uint64_t bdf_as_int(const char *name) {
  unsigned dom, bus, dev, fn;
  if (!name || sscanf(name, "%4x:%2x:%2x.%1x", &dom, &bus, &dev, &fn) != 4) return 0;
  return ((uint64_t)bus << 8) | ((uint64_t)dev << 3) | fn; // the domain is not part of what gsp-rm is told
}

// What GSP-RM is told about the machine around it: where its own windows are, what it is on the bus, and how much of the
// address space the driver will hand out.
static int rpc_set_gsp_system_info(tinynv_gpu_t *g) {
  tinynv_pci_t *pci = g->dev.pci;
  tinynv_gsp_system_info_t info;
  memset(&info, 0, sizeof(info));
  if (pci->bar_info(pci, 0, &info.gpuPhysAddr, NULL) || pci->bar_info(pci, 1, &info.gpuPhysFbAddr, NULL) ||
      pci->bar_info(pci, 3, &info.gpuPhysInstAddr, NULL))
    return -1;
  // where the config space is mirrored into the register window; the chain-of-trust chips moved it
  info.pciConfigMirrorBase = g->dev.fmc_boot ? 0x92000 : 0x88000;
  info.pciConfigMirrorSize = 0x1000;
  info.nvDomainBusDeviceFunc = bdf_as_int(pci->name);
  info.bIsPassthru = 1;
  info.PCIDeviceID = pci->cfg_read(pci, 0x00, 4);    // vendor and device together
  info.PCISubDeviceID = pci->cfg_read(pci, 0x2c, 4); // subsystem vendor and device together
  info.PCIRevisionID = pci->cfg_read(pci, 0x08, 1);
  info.maxUserVa = 0x7ffffffff000ull;
  // the 610 GspSystemInfo grew 928 -> 952 B (virtualConfigBits inserted, bIsPassthru moved to 842). The
  // encoder places every field the driver fills at its 610 offset; the 570 payload is unchanged.
  if (is610(g)) {
    uint8_t buf[TINYNV_RPC610_SYSINFO_SIZE];
    if (tinynv_rpc610_system_info_encode(buf, sizeof buf, &info)) return -1;
    return rpc_send(g, TINYNV_MSG_FUNCTION_GSP_SET_SYSTEM_INFO, buf, sizeof buf);
  }
  return rpc_send(g, TINYNV_MSG_FUNCTION_GSP_SET_SYSTEM_INFO, &info, sizeof(info));
}

// The handful of driver settings GSP-RM reads out of a packed table: a header, then one entry per key, then the key names
// laid end to end with each entry pointing at its own name.
static int rpc_set_registry_table(tinynv_gpu_t *g) {
  static const struct { const char *name; uint32_t value; } TABLE[] = {
    {"RMForcePcieConfigSave", 1},
    {"RMSecBusResetEnable", 1},
  };
  // Which of the two to send. The oracle sends both and so does the recorded boot, so the default is both. On hardware
  // they are suspects: "save the pcie config and put it back" and "you may reset the bus" are the two things a
  // firmware could be doing when a card's windows vanish under it. TINYNV_REGISTRY=none sends an empty table, =nosave
  // and =nosbr drop one each.
  size_t n = sizeof(TABLE) / sizeof(*TABLE), first = 0;
  const char *pick = getenv("TINYNV_REGISTRY");
  if (pick && !strcmp(pick, "none")) n = 0;
  else if (pick && !strcmp(pick, "nosave")) { first = 1; n = 1; }
  else if (pick && !strcmp(pick, "nosbr")) n = 1;
  if (pick) fprintf(stderr, "tinynv: registry table: TINYNV_REGISTRY=%s, %zu of 2 entries sent\n", pick, n);
  // TINYNV_PCIE_LINK_SPEED=lock|gen1|gen2|gen3|gen4 adds RMPcieLinkSpeed (nvrm_registry.h): generations above the
  // named one disabled, and bit 31, LOCK_AT_LOAD, so the firmware keeps the link where it found it instead of
  // retraining it during its init - which on this enclosure is the moment the configuration is cleared. Live cards
  // only, and only when asked: the recorded boot sends exactly two entries.
  struct { const char *name; uint32_t value; } extra[2];
  size_t n_extra = 0;
  // ON BY DEFAULT on a live vbios-path card since 2026-09-21: the first boot of the day on a 16 GT/s link that did not
  // lose its configuration was the one with the link locked, and the loss had always come 1.5 s into init - where the
  // firmware, left to itself, drops the link to Gen1 for power. TINYNV_PCIE_LINK_SPEED=off sends the two entries the
  // recording sends and nothing else.
  // "auto", the default on a live vbios-path card since 2026-09-21 evening: at the instant of every loss of the
  // card's configuration the link read Gen4, on boots that had opened at Gen1 and on boots that sent LOCK_AT_LOAD
  // alike - the firmware retrains the link to its maximum about 1.5 s into init, before any lock applies, and this
  // enclosure clears the endpoint's configuration on that transition. Boots that open at Gen4 never lose, because
  // there is nothing to retrain to. So, that evening: read the generation the link is on now and forbid every
  // generation above it. Superseded the same night by the two findings below; auto is now Gen3, always.
  const char *ls = getenv("TINYNV_PCIE_LINK_SPEED");
  if (!ls && g->dev.pci->live && !g->dev.fmc_boot) ls = "auto";
  if (ls && !strcmp(ls, "auto")) {
    unsigned lgen = 0, lw = 0;
    tinynv_dev_link(&g->dev, &lgen, &lw);
    // AND NEVER ABOVE GEN3. On a fresh 16 GT/s enumeration with only Gen5 forbidden, every boot lost its configuration
    // on every attempt (23 losses in 18 tries, 2026-09-21 15:00) and the loss lines showed the link flapping between
    // Gen4 and Gen1 the whole time; capped at Gen3 the same card came up first try, three of three, starting from that
    // same Gen4 link. Gen4 is not a speed this enclosure holds, and a Thunderbolt tunnel carries no more than a Gen3 x4
    // link's worth of PCIe in any case, so nothing is given up.
    // AND ALWAYS GEN3, whatever the link is on now (15:20): the link idles down to Gen1 after every unload, and a boot
    // that asked for Gen3 from that Gen1 link came up first try with the link at 8 GT/s ten seconds in and there for
    // the rest of the process (sampled from the host every 3 s) - the retrain up to Gen3 is one this enclosure holds,
    // and the same gate ran in 21 s instead of 37. Only Gen4 was ever the fault. lgen is read for the log line.
    (void)lgen; (void)lw;
    ls = "gen3";
  }
  if (ls && *ls && strcmp(ls, "off") && g->dev.pci->live) {
    uint32_t v = 1u << 31;                       // LOCK_AT_LOAD
    int gen = !strncmp(ls, "gen", 3) ? atoi(ls + 3) : 0;
    if (gen >= 1 && gen < 2) v |= 2u << 0;       // ALLOW_GEN2_DISABLE
    if (gen >= 1 && gen < 3) v |= 2u << 2;       // ALLOW_GEN3_DISABLE
    if (gen >= 1 && gen < 4) v |= 2u << 4;       // ALLOW_GEN4_DISABLE
    if (gen >= 1 && gen < 5) v |= 2u << 6;       // ALLOW_GEN5_DISABLE
    extra[n_extra].name = "RMPcieLinkSpeed"; extra[n_extra].value = v; n_extra++;
    fprintf(stderr, "tinynv: registry table: RMPcieLinkSpeed=%#x (%s, locked at load)\n", v, ls);
  }
  size_t hdr_size = sizeof(tinynv_registry_table_t), entries_size = sizeof(tinynv_registry_entry_t) * (n + n_extra), names_size = 0;
  for (size_t i = 0; i < n; i++) names_size += strlen(TABLE[first + i].name) + 1;
  for (size_t i = 0; i < n_extra; i++) names_size += strlen(extra[i].name) + 1;

  size_t total = hdr_size + entries_size + names_size;
  uint8_t *buf = calloc(1, total);
  if (!buf) return tinynv_fail("out of memory for the registry table");
  tinynv_registry_table_t *h = (tinynv_registry_table_t *)buf;
  h->size = (uint32_t)total;
  h->numEntries = (uint32_t)(n + n_extra);
  tinynv_registry_entry_t *e = (tinynv_registry_entry_t *)(buf + hdr_size);
  size_t at = 0;
  for (size_t i = 0; i < n; i++) {
    size_t len = strlen(TABLE[first + i].name) + 1;
    e[i].nameOffset = (uint32_t)(hdr_size + entries_size + at);
    e[i].type = TINYNV_REGISTRY_TYPE_DWORD;
    e[i].data = TABLE[first + i].value;
    e[i].length = 4;
    memcpy(buf + hdr_size + entries_size + at, TABLE[first + i].name, len);
    at += len;
  }
  for (size_t i = 0; i < n_extra; i++) {
    size_t len = strlen(extra[i].name) + 1;
    e[n + i].nameOffset = (uint32_t)(hdr_size + entries_size + at);
    e[n + i].type = TINYNV_REGISTRY_TYPE_DWORD;
    e[n + i].data = extra[i].value;
    e[n + i].length = 4;
    memcpy(buf + hdr_size + entries_size + at, extra[i].name, len);
    at += len;
  }
  int rc = rpc_send(g, TINYNV_MSG_FUNCTION_SET_REGISTRY, buf, total);
  free(buf);
  return rc;
}

// the hardware half of the sysmem flush buffer. 610's KernelMemorySystem statePreLoad returns
// NV_ERR_INVALID_STATE unless the PFB flush address registers name a real page; the CPU-RM programs them HI then LO
// (kmemsysProgramSysmemFlushBuffer_GA100). PFB is not cleared by falcon reset, and a full-card reset re-runs init_hw,
// so a boot retry re-programs it. 570 never wrote these registers and its assert is non-fatal, so this is a no-op there.
int tinynv_gsp_program_sysmem_flush(tinynv_gpu_t *g) {
  if (!is610(g)) return 0;
  tinynv_gsp_t *gsp = &g->gsp;
  if (!gsp->sysmem_flush.naddrs) return tinynv_fail("the 610 sysmem flush buffer was not allocated");
  uint64_t addr = gsp->sysmem_flush.addrs[0];
  if ((addr >> 40) > NV_PFB_NISO_FLUSH_SYSMEM_ADDR_HI_MASK)
    return tinynv_fail("the sysmem flush buffer address %#llx does not fit the PFB flush register",
                       (unsigned long long)addr);
  tinynv_wr32(&g->dev, NV_PFB_NISO_FLUSH_SYSMEM_ADDR_HI,
              (uint32_t)(addr >> 40) & NV_PFB_NISO_FLUSH_SYSMEM_ADDR_HI_MASK);
  tinynv_wr32(&g->dev, NV_PFB_NISO_FLUSH_SYSMEM_ADDR, (uint32_t)(addr >> 8));
  return 0;
}

int tinynv_gsp_init_hw(tinynv_gpu_t *g) {
  tinynv_gsp_t *gsp = &g->gsp;

  // the status queue's header does not exist until GSP-RM writes it, so attaching to it is the wait for the firmware to
  // start. on this card that is several seconds.
  if (rpcq_attach(g, &gsp->stat_q, gsp->cmd_q_off + gsp->queue_size, "status")) return -1;
  // each side keeps its read position in the other's header, so the queues have to be told about each other
  tinynv_msgq_tx_header_t cmd_tx;
  nv_rd_block(&gsp->queues.view, gsp->cmd_q_off, &cmd_tx, sizeof(cmd_tx));
  gsp->stat_q.rx_off = gsp->cmd_q_off + cmd_tx.rxHdrOff;
  gsp->cmd_q.rx_off = gsp->stat_q.base + gsp->stat_q.tx.rxHdrOff;

  // Sixty seconds unless told otherwise: every boot that has worked got its notice in a few, and a longer wait is for
  // finding out whether a boot that does not is slow or stuck (see init_wait_report).
  double init_timeout = 60.0;
  const char *t = getenv("TINYNV_INIT_TIMEOUT");
  if (t && atof(t) > 0) init_timeout = atof(t);
  if (g->dev.pci->live) {
    // where the buffers GSP-RM writes to are, so a fault record's addresses can be read against them
    const tinynv_bootmem_t *m[] = {&gsp->queues, &gsp->logbuf, &gsp->rm_args, &gsp->libos_args};
    const char *n[] = {"queues", "logs", "rm args", "libos args"};
    fprintf(stderr, "tinynv: host buffers gsp-rm writes:");
    for (int i = 0; i < 4; i++) {
      size_t split = 0;
      for (size_t k = 1; k < m[i]->naddrs; k++) split += m[i]->addrs[k] != m[i]->addrs[k - 1] + 0x1000;
      fprintf(stderr, " %s %#llx+%#zx%s", n[i], m[i]->naddrs ? (unsigned long long)m[i]->addrs[0] : 0ull,
              m[i]->naddrs * (size_t)0x1000, split ? " (NOT contiguous)" : "");
    }
    fprintf(stderr, "\n");
  }
  if (is610(g) && tinynv_failpoint("610-init-done")) return -1;
  if (rpc_wait(g, TINYNV_MSG_EVENT_GSP_INIT_DONE, 0, init_timeout, "its start-up notice", NULL, NULL)) {
    if (is610(g)) gsp_silent_boot_diag(g);
    return -1;
  }
  if (gsp->err_state) return tinynv_fail("gsp-rm started but reported an error on the way up");

  // one more look at the configuration before anything is written through the window again
  if (g->dev.pci->live) tinynv_dev_restore_decode(&g->dev);

  // with the firmware up, the two windows onto instance memory are retargeted at it
  tinynv_wr32(&g->dev, NV_PBUS_BAR1_BLOCK, 0);
  if (g->dev.fmc_boot) tinynv_wr32(&g->dev, NV_VIRTUAL_FUNCTION_PRIV_FUNC_BAR1_BLOCK_LOW_ADDR, 0);
  gsp->up = 1;
  // opt-in comparison dump (TINYNV_DUMP_GSP_LOGS=1) - the same state/log readout the silent-boot failure
  // path uses, taken after a SUCCESSFUL boot so a 570 control can be compared against a silent 610 one.
  if (getenv("TINYNV_DUMP_GSP_LOGS")) gsp_silent_boot_diag(g);
  return 0;
}

// why an unload notice went unanswered. The decisive reading is how far gsp-rm has read OUR command queue: if
// its read pointer reached the record this driver wrote, gsp-rm took the notice and chose not to answer it, which is a
// firmware-side teardown question; if it is still behind, the record never reached it, which is a host-side delivery
// question. The status-queue pointers say whether it wrote anything back at all, the log put pointers say whether its
// DMA into host memory still works, and the mailbox with the RISC-V control register say whether the core is alive or
// has already parked itself. This is init_wait_report's question set applied to the way down instead of the way up.
static void gsp_unload_diag(tinynv_gpu_t *g, uint32_t wp_before) {
  tinynv_gsp_t *gsp = &g->gsp;
  tinynv_dev_t *d = &g->dev;
  uint32_t wp = nv_rd32(&gsp->queues.view, gsp->cmd_q.base + offsetof(tinynv_msgq_tx_header_t, writePtr));
  uint32_t took = nv_rd32(&gsp->queues.view, gsp->cmd_q.rx_off);   // how far gsp-rm has read OUR queue
  uint32_t rp = nv_rd32(&gsp->queues.view, gsp->stat_q.rx_off);
  uint32_t swp = nv_rd32(&gsp->queues.view, gsp->stat_q.base + offsetof(tinynv_msgq_tx_header_t, writePtr));
  uint32_t logput[3] = {0, 0, 0};
  for (int i = 0; i < 3; i++) logput[i] = nv_rd32(&gsp->logbuf.view, 0x10000u * (unsigned)i); // INIT, INTR, RM
  uint32_t mbx0 = tinynv_rd32(d, NV_PGSP_FALCON_MAILBOX0), mbx1 = tinynv_rd32(d, NV_PGSP_FALCON_MAILBOX1);
  uint32_t rv = tinynv_rd32(d, g->flcn.falcon + NV_PRISCV_RISCV_CPUCTL);
  uint32_t cmd = d->pci->cfg_read(d->pci, 0x04, 2);
  fprintf(stderr, "tinynv: unload diagnostic | command queue: write pointer %u -> %u and gsp-rm has read %u (%s) | "
                  "status queue rp %u wp %u (%s) | log put INIT %u INTR %u RM %u | gsp mailbox %#x %#x, riscv %s%s | "
                  "command register %#06x (memory %s, bus master %s)\n",
          wp_before, wp, took, took == wp ? "it took the notice" : "it never read the notice",
          rp, swp, rp == swp ? "it wrote nothing back" : "it sent something this driver did not act on",
          logput[0], logput[1], logput[2], mbx0, mbx1,
          NV_GET(rv, NV_PRISCV_RISCV_CPUCTL, ACTIVE_STAT) ? "active" : "NOT active",
          NV_GET(rv, NV_PRISCV_RISCV_CPUCTL, HALTED) ? " halted" : "", cmd, (cmd & 0x2) ? "on" : "OFF",
          (cmd & 0x4) ? "on" : "OFF");
  // And the log rings, which are the only place gsp-rm can say why it will not go down.
  gsp_silent_boot_diag(g);
}

// Tell GSP-RM the driver is going, and wait for it to halt.
//
// The oracle sends the same notice (NV_GSP.fini_hw) and stops there, because on Linux the next open resets the card
// anyway. Here the halt is waited for, as NVIDIA's kgspUnloadRm does: once libos has parked the core the GSP falcon's
// first mailbox reads 0x80000000 (LIBOS_INTERRUPT_PROCESSOR_SUSPENDED), and only then may that falcon be reset and
// handed other code, which is what the unload's second half (tinynv_flcn_unload_hw) does.
int tinynv_gsp_unload(tinynv_gpu_t *g) {
  tinynv_gsp_t *gsp = &g->gsp;
  if (!gsp->up) return 0;
  double t0 = tinynv_now_s();
  tinynv_unloading_guest_driver_t u = {.bInPMTransition = 0, .bGc6Entering = 0, .newLevel = 1u << 6};
  if (is610(g) && tinynv_failpoint("610-unload-notice")) return -1;
  // where the write pointer stood before the notice, so the diagnostic can say whether the record this driver
  // wrote was the one gsp-rm stopped short of.
  uint32_t wp_before =
      is610(g) ? nv_rd32(&gsp->queues.view, gsp->cmd_q.base + offsetof(tinynv_msgq_tx_header_t, writePtr)) : 0;
  // on 610 the notice is sent WITHOUT waiting for a reply. gsp-rm does answer it - the drain trace on the live
  // boot (logs/g610-12-boot610-20261112-231521.log) records `function 47, 0 byte payload` - but that answer is EMPTY, and
  // rpc_call asks rpc_wait for want_total = the REQUEST's own 8 bytes. rpc_drain610 read a reply shorter than want_total
  // as the head of a split one, so it held a complete empty reply as fragment 1 of a continuation that cannot come, and
  // timed out 10 s later reporting "gsp-rm never sent the unload notice": a host-side matching defect wearing the
  // costume of a firmware teardown refusal. What the firmware actually does is take the notice, print once more (event
  // 4108), answer it empty and park - mailbox0 becomes 0x80000000 (LIBOS_INTERRUPT_PROCESSOR_SUSPENDED) and the RISC-V
  // halts. The pinned 610 kgspUnloadRm_IMPL treats the rpc the same way: it records rpcStatus, goes on to
  // kgspWaitForProcessorSuspend_HAL, dumps the logs and tears down, and only reports rpcStatus at the very end. So the
  // parked core is the evidence the firmware went down and the reply is incidental. 570 keeps waiting, unchanged.
  int sent = is610(g) ? rpc_send(g, TINYNV_MSG_FUNCTION_UNLOADING_GUEST_DRIVER, &u, sizeof u)
                      : rpc_call(g, TINYNV_MSG_FUNCTION_UNLOADING_GUEST_DRIVER, &u, sizeof u, NULL, 0, "the unload notice");
  if (sent) {
    if (is610(g)) gsp_unload_diag(g, wp_before);
    return -1;
  }
  // Poll the mailbox AND keep draining the queue: on its way down GSP-RM may ask for a register sequence (the sequencer's
  // core-reset and wait-for-halt operations exist for exactly this), and a request left unanswered is a firmware that
  // waits for us while we wait for it.
  double deadline = tinynv_now_s() + 10.0;
  uint32_t mbx;
  for (;;) {
    int seen = 0;
    if (is610(g) && tinynv_failpoint("610-unload-halt")) return -1;
    if (rpc_drain(g, 0xffffffffu, 0, &seen, NULL, NULL)) return -1;
    mbx = tinynv_rd32(&g->dev, NV_PGSP_FALCON_MAILBOX0);
    if (mbx == 0x80000000u) break;
    if (tinynv_now_s() > deadline) {
      if (is610(g)) {
        gsp_unload_diag(g, wp_before);
        return tinynv_fail("gsp-rm took the unload notice but its core never parked in 10 s (gsp mailbox %#x)", mbx);
      }
      return tinynv_fail("gsp-rm acknowledged the unload notice but did not halt in 10 s (gsp mailbox %#x)", mbx);
    }
  }
  gsp->up = 0;
  if (is610(g))
    fprintf(stderr, "tinynv: gsp-rm took the unload notice and its core parked, %.2f s\n", tinynv_now_s() - t0);
  else
    fprintf(stderr, "tinynv: gsp-rm acknowledged the unload notice and halted, %.2f s\n", tinynv_now_s() - t0);
  return 0;
}

int tinynv_gsp_init_sw(tinynv_gpu_t *g) {
  g->gsp.gpu = g;
  // the 610 profile preflights the WHOLE lifecycle bundle (boot AND unload AND the host-request ucode) and
  // stages the generic-bootloader ucode before any firmware is placed. A missing/wrong-hash asset is an error here,
  // never a substitution. The 570 path issues no extra operation.
  if (is610(g)) {
    if (g->dev.fmc_boot || strcmp(g->dev.fw_name, "ad102"))
      return tinynv_fail("the 610.43.03 profile is the AD102 vbios boot path only; this device is %s%s", g->dev.fw_name,
                         g->dev.fmc_boot ? " (chain-of-trust)" : "");
    if (tinynv_failpoint("610-assets")) return -1;
    if (tinynv_fw610_validate_all()) return -1;
    tinynv_gsp_t *gsp = &g->gsp;
    if (tinynv_fw610_load("generic_bl", &gsp->gbl_fw)) return -1;
    if (tinynv_fw610_load("generic_bl_desc", &gsp->gbl_desc_fw)) return -1;
    if (tinynv_fw610_bl_desc(&gsp->gbl_desc_fw, &gsp->gbl_start_tag, &gsp->gbl_code_size)) return -1;
    if (tinynv_failpoint("610-rm-args")) return -1;
  }
  if (init_rm_args(g)) return -1;
  if (is610(g) && tinynv_failpoint("610-libos-args")) return -1;
  if (init_libos_args(g)) return -1;
  if (init_wpr_meta(g)) return -1;
  // prefilled before the firmware exists: it reads both out of the ring as soon as it starts
  if (is610(g) && tinynv_failpoint("610-sysinfo")) return -1;
  if (rpc_set_gsp_system_info(g)) return -1;
  if (is610(g) && tinynv_failpoint("610-registry")) return -1;
  return rpc_set_registry_table(g);
}

void tinynv_gsp_fini(tinynv_gpu_t *g) {
  tinynv_gsp_t *gsp = &g->gsp;
  tinynv_bootmem_t *all[] = {&gsp->queues, &gsp->rm_args, &gsp->logbuf, &gsp->libos_args,
                             &gsp->radix3, &gsp->signature, &gsp->bootloader, &gsp->wpr_meta,
                             &gsp->rm_state_monitor, &gsp->sysmem_flush};
  for (size_t i = 0; i < sizeof(all) / sizeof(*all); i++) if (all[i]->size) tinynv_free_boot_mem(&g->mm, all[i]);
  tinynv_fw_free(&gsp->gsp_fw);
  tinynv_fw_free(&gsp->bl_fw);
  // the 610-only state (all zero on 570, so these are no-ops there).
  tinynv_fw_free(&gsp->bl_desc_fw);
  tinynv_fw_free(&gsp->gbl_fw);
  tinynv_fw_free(&gsp->gbl_desc_fw);
  free(gsp->cont_buf);
  gsp->cont_buf = NULL;
  gsp->cont_len = gsp->cont_want = 0;
}
