// The 610 host-request executor. See gsp610_host.h. Mirrors the pinned 610.43.03 kgspLoadAndExecuteHsBinary_GA102
// and kgspLoadAndExecuteGenericBootloader_TU102 (+ s_setupLoader / s_imemCopyTo / s_dmemCopyTo). Touches the device;
// exercised offline against a register-level fake. Every sub-call that fails already recorded its reason through
// tinynv_fail, so this file returns -1 without overwriting it (first-error diagnostics preserved); it only calls
// tinynv_fail for its own validation refusals.
#include "gsp610_host.h"
#include "internal.h"
#include "nv_regs.h"
#include <string.h>

// the mask of one [lo,hi] field, for a read-modify-write that leaves its neighbours alone (flcn.c's FLD)
#define FLD(reg, f) ((uint32_t)(((1ull << (reg##_##f##_HI - reg##_##f##_LO + 1)) - 1) << reg##_##f##_LO))

// LIBOS_INTERRUPT_PROCESSOR_SUSPENDED: the GSP mailbox0 bit that says the RISC-V core parked and the falcon is the
// host's to drive. FLCN_ERR_BINARY_NOT_STARTED: written to mailbox0 before launch so a binary that never starts is
// distinguishable from one that ran and returned 0 (flcnretval.h).
#define GSP610_SUSPEND_BIT 0x80000000u
#define GSP610_BINARY_NOT_STARTED 0xFEu
#define GSP610_BLK 256u
#define GSP610_DMEM_VA_INVALID 0xffffffffu
#define GSP610_FBIF_APERTURES 8u

static uint32_t rd32le(const uint8_t *p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

// Wait for GSP-RM to suspend its RISC-V core before the host takes the falcon. On timeout, name the mailbox value,
// which is what tells a core that never parked apart from one the host mis-read.
static int gsp610_wait_suspend(tinynv_gpu_t *g) {
  tinynv_dev_t *d = &g->dev;
  uint64_t mbox = g->flcn.falcon + NV_PFALCON_FALCON_MAILBOX0;
  if (tinynv_wait_reg(d, mbox, GSP610_SUSPEND_BIT, GSP610_SUSPEND_BIT, 10000,
                      "waiting for gsp-rm to suspend before a 610 host request")) {
    // keep the wait's own first error (the register value it read) and add the mailbox context, rather than overwrite
    char inner[256]; snprintf(inner, sizeof inner, "%s", tinynv_last_error());
    uint32_t v = tinynv_rd32(d, mbox);
    return tinynv_fail("gsp-rm did not suspend before the 610 host request: mailbox0 %#x (suspend bit %#x)%s%s",
                       v, GSP610_SUSPEND_BIT, *inner ? " - " : "", inner);
  }
  return 0;
}

// Resume the RISC-V core after the host ran a binary on the falcon: reset into RISC-V, hand back the libos boot args,
// start SEC2 to reload GSP-RM, wait for the handoff, and confirm the core came up. Mirrors kgspExecuteCoreResume_TU102
// and the 570 sequencer's CORE_RESUME step (gsp.c run_cpu_seq case 8), plus the 610 FALCON_OS / active checks.
static int gsp610_core_resume(tinynv_gpu_t *g) {
  tinynv_dev_t *d = &g->dev;
  if (tinynv_flcn_reset(g, g->flcn.falcon, 1)) return -1;   // "did not come out of reset on its riscv core"
  tinynv_wr32(d, NV_PGSP_FALCON_MAILBOX0, (uint32_t)g->gsp.libos_args_sysmem);
  tinynv_wr32(d, NV_PGSP_FALCON_MAILBOX1, (uint32_t)(g->gsp.libos_args_sysmem >> 32));
  tinynv_flcn_start_cpu(d, g->flcn.sec2);
  if (tinynv_wait_reg(d, NV_PGC6_BSI_SECURE_SCRATCH_14,
                      1u << NV_PGC6_BSI_SECURE_SCRATCH_14_BOOT_STAGE_3_HANDOFF_LO,
                      1u << NV_PGC6_BSI_SECURE_SCRATCH_14_BOOT_STAGE_3_HANDOFF_LO, 10000,
                      "waiting for sec2 to hand off after a 610 host request")) return -1;
  uint32_t mbx = tinynv_rd32(d, g->flcn.sec2 + NV_PFALCON_FALCON_MAILBOX0);
  if (mbx) return tinynv_fail("sec2 refused the 610 host request's resume step: mailbox %#x", mbx);
  // 610 programs FALCON_OS with the boot descriptor's appVersion and confirms the RISC-V core is active.
  tinynv_wr32(d, g->flcn.falcon + NV_PFALCON_FALCON_OS, g->gsp.bl_desc.appVersion);
  uint32_t rv = tinynv_rd32(d, g->flcn.falcon + NV_PRISCV_RISCV_CPUCTL);
  if (!NV_GET(rv, NV_PRISCV_RISCV_CPUCTL, ACTIVE_STAT))
    return tinynv_fail("the gsp riscv core is not active after the 610 host request's resume (cpuctl %#x)", rv);
  return 0;
}

// A binary that was never started leaves the sentinel in mailbox0. Check it before core resume overwrites the mailbox
// with the libos boot arguments, so a bad boot vector or PKC rejection cannot be mistaken for a completed host request.
static int gsp610_confirm_started(tinynv_gpu_t *g, uint64_t base) {
  uint32_t mailbox = tinynv_rd32(&g->dev, base + NV_PFALCON_FALCON_MAILBOX0);
  if (mailbox == GSP610_BINARY_NOT_STARTED)
    return tinynv_fail("the 610 binary left MAILBOX0 at BINARY_NOT_STARTED (%#x): it did not start", mailbox);
  return 0;
}

// Program the BROM registers for the boot ROM's PKC signature check (the 610 _kgspSetupPkcParams).
static void gsp610_pkc(tinynv_dev_t *d, uint64_t base, uint32_t sig_dmem, uint32_t eng_mask, uint32_t ucode_id) {
  tinynv_wr32(d, base + NV_PFALCON2_FALCON_BROM_PARAADDR(0), sig_dmem);
  tinynv_wr32(d, base + NV_PFALCON2_FALCON_BROM_ENGIDMASK, eng_mask);
  tinynv_wr32(d, base + NV_PFALCON2_FALCON_BROM_CURR_UCODE_ID,
              NV_SET(NV_PFALCON2_FALCON_BROM_CURR_UCODE_ID, VAL, ucode_id));
  tinynv_wr32(d, base + NV_PFALCON2_FALCON_MOD_SEL,
              NV_SET(NV_PFALCON2_FALCON_MOD_SEL, ALGO, NV_PFALCON2_FALCON_MOD_SEL_ALGO_RSA3K));
}

int tinynv_gsp610_exec_hs(tinynv_gpu_t *g, const tinynv_rpc610_load_exec_hs_t *p) {
  tinynv_dev_t *d = &g->dev;
  uint64_t base = g->flcn.falcon;
  fprintf(stderr, "tinynv: gsp-rm asked the host to run a 610 hs binary (imem %u B, dmem %u B, ucode id %u)\n",
          p->ucodeImemSize, p->ucodeDmemSize, p->ucodeId);

  if (gsp610_wait_suspend(g)) return -1;
  if (tinynv_flcn_reset(g, base, 0)) return -1;
  tinynv_flcn_disable_ctx_req(d, base);

  // TRANSCFG(0): fetch the DMA data from local FB, physically, engine id BAR2_FN0.
  tinynv_flcn_rmw(d, base + NV_PFALCON_FBIF_TRANSCFG(0),
                  FLD(NV_PFALCON_FBIF_TRANSCFG, TARGET) | FLD(NV_PFALCON_FBIF_TRANSCFG, MEM_TYPE) |
                      FLD(NV_PFALCON_FBIF_TRANSCFG, ENGINE_ID_FLAG),
                  NV_SET(NV_PFALCON_FBIF_TRANSCFG, TARGET, NV_PFALCON_FBIF_TRANSCFG_TARGET_LOCAL_FB) |
                      NV_SET(NV_PFALCON_FBIF_TRANSCFG, MEM_TYPE, NV_PFALCON_FBIF_TRANSCFG_MEM_TYPE_PHYSICAL) |
                      NV_SET(NV_PFALCON_FBIF_TRANSCFG, ENGINE_ID_FLAG, NV_PFALCON_FBIF_TRANSCFG_ENGINE_ID_FLAG_BAR2_FN0));

  // IMEM: secure. The source is the physical address the request names; the engine adds the VA back as it walks blocks.
  uint32_t cmd = NV_SET(NV_PFALCON_FALCON_DMATRFCMD, WRITE, 0) |
                 NV_SET(NV_PFALCON_FALCON_DMATRFCMD, SIZE, NV_PFALCON_FALCON_DMATRFCMD_SIZE_256B) |
                 NV_SET(NV_PFALCON_FALCON_DMATRFCMD, CTXDMA, 0) |
                 NV_SET(NV_PFALCON_FALCON_DMATRFCMD, IMEM, 1) | NV_SET(NV_PFALCON_FALCON_DMATRFCMD, SEC, 1);
  if (tinynv_flcn_dma(d, base, cmd, p->ucodeImemPA, p->ucodeImemVA, p->imemPhysAddr, p->ucodeImemSize)) return -1;

  // DMEM: not secure. A valid DMEM VA tags the transfer; otherwise the offset is zero.
  cmd = NV_SET(NV_PFALCON_FALCON_DMATRFCMD, WRITE, 0) |
        NV_SET(NV_PFALCON_FALCON_DMATRFCMD, SIZE, NV_PFALCON_FALCON_DMATRFCMD_SIZE_256B) |
        NV_SET(NV_PFALCON_FALCON_DMATRFCMD, CTXDMA, 0) |
        NV_SET(NV_PFALCON_FALCON_DMATRFCMD, IMEM, 0) | NV_SET(NV_PFALCON_FALCON_DMATRFCMD, SEC, 0);
  uint32_t dmem_off = 0;
  if (p->ucodeDmemVA != GSP610_DMEM_VA_INVALID) {
    cmd |= NV_SET(NV_PFALCON_FALCON_DMATRFCMD, SET_DMTAG, 1);
    dmem_off = p->ucodeDmemVA;
  }
  if (tinynv_flcn_dma(d, base, cmd, p->ucodeDmemPA, dmem_off, p->dmemPhysAddr, p->ucodeDmemSize)) return -1;

  gsp610_pkc(d, base, p->hsSigDmemAddr, p->engineIdMask, p->ucodeId);
  tinynv_wr32(d, base + NV_PFALCON_FALCON_MAILBOX0, GSP610_BINARY_NOT_STARTED);
  tinynv_wr32(d, base + NV_PFALCON_FALCON_BOOTVEC, p->ucodeImemVA);
  tinynv_flcn_start_cpu(d, base);
  if (tinynv_flcn_wait_cpu_halted(d, base)) return -1;
  if (gsp610_confirm_started(g, base)) return -1;
  return gsp610_core_resume(g);
}

// Copy `n` bytes into DMEM at `dest` through access port 0 (the 610 s_dmemCopyTo_TU102). dest and n are 4-aligned.
static int gsp610_dmem_copy(tinynv_dev_t *d, uint64_t base, uint32_t dest, const uint8_t *src, uint32_t n) {
  if (!n) return 0;
  if ((dest & 3) || (n & 3)) return tinynv_fail("the 610 dmem copy is not 4-byte aligned (dest %#x, %u B)", dest, n);
  uint32_t c = (dest & (FLD(NV_PFALCON_FALCON_DMEMC, OFFS) | FLD(NV_PFALCON_FALCON_DMEMC, BLK))) |
               NV_SET(NV_PFALCON_FALCON_DMEMC, AINCW, 1);
  tinynv_wr32(d, base + NV_PFALCON_FALCON_DMEMC(0), c);
  for (uint32_t i = 0; i < n; i += 4) tinynv_wr32(d, base + NV_PFALCON_FALCON_DMEMD(0), rd32le(src + i));
  return 0;
}

// Copy `n` bytes into IMEM at `dest` through access port 0, tagging each 256-byte block (the 610 s_imemCopyTo_TU102).
// dest is 256-aligned, n is 4-aligned; `tag` is the starting virtual-address tag (virtAddr >> 8).
static int gsp610_imem_copy(tinynv_dev_t *d, uint64_t base, uint32_t dest, const uint8_t *src, uint32_t n,
                            int secure, uint32_t tag) {
  if (!n) return 0;
  if (dest & (GSP610_BLK - 1)) return tinynv_fail("the 610 imem copy dest %#x is not 256-byte aligned", dest);
  if (n & 3) return tinynv_fail("the 610 imem copy size %u is not 4-byte aligned", n);
  uint32_t c = (dest & (FLD(NV_PFALCON_FALCON_IMEMC, OFFS) | FLD(NV_PFALCON_FALCON_IMEMC, BLK))) |
               NV_SET(NV_PFALCON_FALCON_IMEMC, AINCW, 1) | NV_SET(NV_PFALCON_FALCON_IMEMC, SECURE, secure ? 1 : 0);
  tinynv_wr32(d, base + NV_PFALCON_FALCON_IMEMC(0), c);
  uint32_t words = n / 4;
  for (uint32_t i = 0; i < words; i++) {
    if ((i & 63) == 0) {   // a tag per 256-byte block (64 words)
      tinynv_wr32(d, base + NV_PFALCON_FALCON_IMEMT(0), NV_SET(NV_PFALCON_FALCON_IMEMT, TAG, tag));
      tag++;
    }
    tinynv_wr32(d, base + NV_PFALCON_FALCON_IMEMD(0), rd32le(src + i * 4));
  }
  return 0;
}

// Program the loader aperture for the address space / cache attrib the request names (the 610 s_setupLoaderAperture).
static int gsp610_loader_aperture(tinynv_dev_t *d, uint64_t base, uint32_t ctx_dma, uint32_t addr_space,
                                  uint32_t cache_attrib) {
  uint32_t target;
  if (addr_space == 2) target = NV_PFALCON_FBIF_TRANSCFG_TARGET_LOCAL_FB;                 // ADDR_FBMEM
  else if (addr_space == 1 && cache_attrib == 0) target = NV_PFALCON_FBIF_TRANSCFG_TARGET_COHERENT_SYSMEM;
  else if (addr_space == 1 && cache_attrib == 1) target = NV_PFALCON_FBIF_TRANSCFG_TARGET_NONCOHERENT_SYSMEM;
  else return tinynv_fail("the 610 generic bootloader's address space %u / cache attrib %u is not a supported aperture",
                          addr_space, cache_attrib);
  tinynv_flcn_rmw(d, base + NV_PFALCON_FBIF_TRANSCFG(ctx_dma),
                  FLD(NV_PFALCON_FBIF_TRANSCFG, TARGET) | FLD(NV_PFALCON_FBIF_TRANSCFG, MEM_TYPE),
                  NV_SET(NV_PFALCON_FBIF_TRANSCFG, TARGET, target) |
                      NV_SET(NV_PFALCON_FBIF_TRANSCFG, MEM_TYPE, NV_PFALCON_FBIF_TRANSCFG_MEM_TYPE_PHYSICAL));
  return 0;
}

int tinynv_gsp610_exec_bl(tinynv_gpu_t *g, const tinynv_rpc610_load_exec_bl_t *p, const tinynv_gsp610_bl_ucode_t *bl) {
  tinynv_dev_t *d = &g->dev;
  uint64_t base = g->flcn.falcon;
  if (!bl || !bl->image) return tinynv_fail("the 610 generic-bootloader request arrived but no generic BL ucode is staged");
  if (p->dmemDesc.ctxDma >= GSP610_FBIF_APERTURES)
    return tinynv_fail("the 610 generic-bootloader ctxDma %u is past the %u FBIF apertures",
                       p->dmemDesc.ctxDma, GSP610_FBIF_APERTURES);
  uint32_t bl_size = (bl->code_size + GSP610_BLK - 1) / GSP610_BLK * GSP610_BLK;
  if (!bl_size || bl_size > bl->image_len)
    return tinynv_fail("the 610 generic BL code size %u (aligned %u) does not fit its %zu byte image",
                       bl->code_size, bl_size, bl->image_len);
  fprintf(stderr, "tinynv: gsp-rm asked the host to run the 610 generic bootloader (ctxDma %u, %u B code)\n",
          p->dmemDesc.ctxDma, bl_size);

  if (gsp610_wait_suspend(g)) return -1;
  if (tinynv_flcn_reset(g, base, 0)) return -1;
  tinynv_flcn_disable_ctx_req(d, base);

  uint64_t transcfg = base + NV_PFALCON_FBIF_TRANSCFG(p->dmemDesc.ctxDma);
  uint32_t saved = tinynv_rd32(d, transcfg);

  // s_setupLoader: aperture, the DMEM descriptor at offset 0, the BL at the top of IMEM, then BOOTVEC.
  int ret = 0;
  if (gsp610_loader_aperture(d, base, p->dmemDesc.ctxDma, p->addrSpace, p->cpuCacheAttrib)) {
    ret = -1;
    goto restore_aperture;
  }
  if (gsp610_dmem_copy(d, base, 0, (const uint8_t *)&p->dmemDesc, sizeof p->dmemDesc)) {
    ret = -1;
    goto restore_aperture;
  }
  uint32_t imem_blocks = NV_GET(tinynv_rd32(d, base + NV_PFALCON_FALCON_HWCFG), NV_PFALCON_FALCON_HWCFG, IMEM_SIZE);
  if (bl_size / GSP610_BLK > imem_blocks) {
    tinynv_fail("the 610 generic BL (%u blocks) does not fit the falcon's %u IMEM blocks", bl_size / GSP610_BLK,
                imem_blocks);
    ret = -1;
    goto restore_aperture;
  }
  uint32_t imem_dest = (imem_blocks - bl_size / GSP610_BLK) * GSP610_BLK;
  uint32_t virt_addr = bl->start_tag << 8;
  if (gsp610_imem_copy(d, base, imem_dest, bl->image, bl_size, 0, virt_addr >> 8)) {
    ret = -1;
    goto restore_aperture;
  }
  tinynv_wr32(d, base + NV_PFALCON_FALCON_BOOTVEC, virt_addr);

  tinynv_wr32(d, base + NV_PFALCON_FALCON_MAILBOX0, GSP610_BINARY_NOT_STARTED);
  tinynv_flcn_start_cpu(d, base);
  if (tinynv_flcn_wait_cpu_halted(d, base)) {
    ret = -1;
    goto restore_aperture;
  }
  if (gsp610_confirm_started(g, base)) {
    ret = -1;
    goto restore_aperture;
  }

restore_aperture:
  tinynv_wr32(d, transcfg, saved);   // restore the aperture the boot path had
  return ret ? ret : gsp610_core_resume(g);
}

int tinynv_gsp610_host_request(tinynv_gpu_t *g, uint32_t function, const uint8_t *msg, size_t len,
                               const tinynv_gsp610_bl_ucode_t *bl) {
  switch (tinynv_rpc610_event_kind(function)) {
    case TINYNV_RPC610_EVENT_LOAD_EXEC_HS: {
      tinynv_rpc610_load_exec_hs_t p;
      if (tinynv_rpc610_parse_load_exec_hs(msg, len, &p)) return -1;
      return tinynv_gsp610_exec_hs(g, &p);
    }
    case TINYNV_RPC610_EVENT_LOAD_EXEC_BL: {
      tinynv_rpc610_load_exec_bl_t p;
      if (tinynv_rpc610_parse_load_exec_bl(msg, len, &p)) return -1;
      return tinynv_gsp610_exec_bl(g, &p, bl);
    }
    case TINYNV_RPC610_EVENT_SEQUENCER_GONE:
      return tinynv_fail("gsp-rm sent the CPU register sequencer event (%u), which 610 removed: the 610 firmware uses "
                         "the load-exec host requests instead, so this is a version mismatch, not a sequence to run",
                         function);
    case TINYNV_RPC610_EVENT_LIBOS_PRINT:
      // UCODE_LIBOS_PRINT (0x100C) is a one-way libos print. CPU-RM logs the buffer and returns NV_OK with no
      // reply (_kgspRpcUcodeLibosPrint, kernel_gsp.c). It is neither a host request nor an error: consume it and keep
      // waiting for init-done. Refusing it as unknown aborted a boot whose RM had already initialised the status queue.
      return 1;
    case TINYNV_RPC610_EVENT_UNKNOWN:
      return tinynv_fail("gsp-rm sent event %u, which the 610 profile does not know how to handle", function);
    default:
      return 1;   // init-done / log / fault / rpc reply: not a host request
  }
}
