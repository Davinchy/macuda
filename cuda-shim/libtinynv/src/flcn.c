// The FSP chain of trust, which is how a Blackwell GPU comes to be running our firmware.
//
// The driver never starts the GSP processor directly. It places a small signed boot image (the FMC) in host memory, then
// sends the FSP — a separate security processor with the root of trust — a message saying where that image is, what it
// hashes to, who signed it, and where the arguments for it are. The FSP verifies the signature against the public key,
// checks the key against fuses, copies the image in, and releases the GSP falcon from its boot-time lockdown. Until that
// lockdown clears, every attempt to drive the chip is refused, so this is the gate the whole driver waits behind.
//
// The message goes through the FSP's mailbox: a window register that auto-increments, a data register written one word at
// a time, and a pair of queues whose head and tail pointers say when a message has been taken and when one has arrived.
#include "flcn.h"
#include "gpu.h"
#include "internal.h"
#include "nv_regs.h"
#include "nv_structs.h"
#include "vbios.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static const char *FMC_SHA = "cb59a35c1d4bd1274d7267fd10243c29f843ff41c851b9cbd59f5af2ddd7fece";
// the two booters SEC2 runs on the vbios path, from linux-firmware at the commit tools/fetch_firmware.sh pins
static const char *BOOTER_LOAD_SHA_GA102 = "4497e3eff7e95c774b8a569d17b27c08c9650158d10b229d2be81cdcad9a085b";
static const char *BOOTER_UNLOAD_SHA_GA102 = "8e63db5b78d7d3e349f20a2d11099c3d7109081393cb09ffc0a28133324ae009";

#define FSP_MAX_MSG 0x400

// ---- driving a falcon ---------------------------------------------------------------------------------------------
//
// Both boot paths end up here. A falcon is reset, has code and data moved into it by its own DMA engine, is told which
// signature to check and where, and is started; it halts when it is done and leaves its answer in a mailbox. None of
// this is Ampere-specific - it is how every falcon on the chip is driven - but only the vbios path uses it, because on
// Blackwell the FSP does all of it for us.

static void flcn_sleep_ms(int ms) {
  struct timespec ts = {.tv_sec = ms / 1000, .tv_nsec = (long)(ms % 1000) * 1000000L};
  nanosleep(&ts, NULL);
}

// the mask of one field, for a read-modify-write that must leave its neighbours alone
#define FLD(reg, f) ((uint32_t)(((1ull << (reg##_##f##_HI - reg##_##f##_LO + 1)) - 1) << reg##_##f##_LO))

// what the oracle's .update() does: read, replace these bits, write back. The bits not named belong to the engine and
// a blind write would clear them.
static void flcn_rmw(tinynv_dev_t *d, uint64_t off, uint32_t mask, uint32_t val) {
  tinynv_wr32(d, off, (tinynv_rd32(d, off) & ~mask) | val);
}

static int flcn_dma_free(tinynv_dev_t *d, uint64_t base) {
  return tinynv_wait_reg(d, base + NV_PFALCON_FALCON_DMATRFCMD, FLD(NV_PFALCON_FALCON_DMATRFCMD, FULL), 0, 5000,
                         "waiting for room in the falcon's dma queue");
}

// Move `size` bytes in 256 byte transfers. The engine takes the source once, as a base, and then one command per block;
// `mem_off` and `dest` advance together because the image's layout inside the falcon mirrors its layout in memory.
static int flcn_dma(tinynv_dev_t *d, uint64_t base, uint32_t cmd, uint32_t dest, uint32_t mem_off, uint64_t src, uint32_t size) {
  if (flcn_dma_free(d, base)) return -1;
  tinynv_wr32(d, base + NV_PFALCON_FALCON_DMATRFBASE, (uint32_t)(src >> 8));
  tinynv_wr32(d, base + NV_PFALCON_FALCON_DMATRFBASE1, (uint32_t)((src >> 8) >> 32) & 0x1ff);
  for (uint32_t done = 0; done < size; done += 256) {
    if (flcn_dma_free(d, base)) return -1;
    tinynv_wr32(d, base + NV_PFALCON_FALCON_DMATRFMOFFS, dest + done);
    tinynv_wr32(d, base + NV_PFALCON_FALCON_DMATRFFBOFFS, mem_off + done);
    tinynv_wr32(d, base + NV_PFALCON_FALCON_DMATRFCMD, cmd);
  }
  return tinynv_wait_reg(d, base + NV_PFALCON_FALCON_DMATRFCMD, FLD(NV_PFALCON_FALCON_DMATRFCMD, IDLE),
                         FLD(NV_PFALCON_FALCON_DMATRFCMD, IDLE), 10000, "waiting for the falcon's dma to drain");
}

// Let the engine address memory physically, without a context.
void tinynv_flcn_disable_ctx_req(tinynv_dev_t *d, uint64_t base) {
  flcn_rmw(d, base + NV_PFALCON_FBIF_CTL, FLD(NV_PFALCON_FBIF_CTL, ALLOW_PHYS_NO_CTX),
           NV_SET(NV_PFALCON_FBIF_CTL, ALLOW_PHYS_NO_CTX, 1));
  tinynv_wr32(d, base + NV_PFALCON_FALCON_DMACTL, 0);
}

int tinynv_flcn_wait_cpu_halted(tinynv_dev_t *d, uint64_t base) {
  return tinynv_wait_reg(d, base + NV_PFALCON_FALCON_CPUCTL, FLD(NV_PFALCON_FALCON_CPUCTL, HALTED),
                         FLD(NV_PFALCON_FALCON_CPUCTL, HALTED), 10000, "waiting for the falcon to halt");
}

// Two ways to start a core, and the chip says which: where the alias register is enabled it is the one that works.
void tinynv_flcn_start_cpu(tinynv_dev_t *d, uint64_t base) {
  if (NV_GET(tinynv_rd32(d, base + NV_PFALCON_FALCON_CPUCTL), NV_PFALCON_FALCON_CPUCTL, ALIAS_EN))
    tinynv_wr32(d, base + NV_PFALCON_FALCON_CPUCTL_ALIAS, NV_SET(NV_PFALCON_FALCON_CPUCTL_ALIAS, STARTCPU, 1));
  else
    tinynv_wr32(d, base + NV_PFALCON_FALCON_CPUCTL, NV_SET(NV_PFALCON_FALCON_CPUCTL, STARTCPU, 1));
}

// Reset a falcon and wait for it to finish scrubbing its own memory. `riscv` selects the RISC-V core for the next boot
// rather than the falcon one; without it, a core that HAS a RISC-V half is put back on the falcon half and told which
// chip it is on.
int tinynv_flcn_reset(tinynv_gpu_t *g, uint64_t base, int riscv) {
  tinynv_dev_t *d = &g->dev;
  uint64_t engine = (base == g->flcn.falcon) ? NV_PGSP_FALCON_ENGINE : NV_PSEC_FALCON_ENGINE;
  tinynv_wr32(d, engine, NV_SET(NV_PGSP_FALCON_ENGINE, RESET, 1));
  flcn_sleep_ms(100); // the engine is not answering until it is out of reset, and there is no bit that says so
  tinynv_wr32(d, engine, 0);

  if (tinynv_wait_reg(d, base + NV_PFALCON_FALCON_HWCFG2, FLD(NV_PFALCON_FALCON_HWCFG2, MEM_SCRUBBING), 0, 10000,
                      "waiting for the falcon to scrub its memory")) return -1;

  if (riscv) {
    tinynv_wr32(d, base + NV_PRISCV_RISCV_BCR_CTRL,
                NV_SET(NV_PRISCV_RISCV_BCR_CTRL, CORE_SELECT, 1) | NV_SET(NV_PRISCV_RISCV_BCR_CTRL, BRFETCH, 1));
  } else if (NV_GET(tinynv_rd32(d, base + NV_PFALCON_FALCON_HWCFG2), NV_PFALCON_FALCON_HWCFG2, RISCV)) {
    tinynv_wr32(d, base + NV_PRISCV_RISCV_BCR_CTRL, NV_SET(NV_PRISCV_RISCV_BCR_CTRL, CORE_SELECT, 0));
    if (tinynv_wait_reg(d, base + NV_PRISCV_RISCV_BCR_CTRL, FLD(NV_PRISCV_RISCV_BCR_CTRL, VALID),
                        FLD(NV_PRISCV_RISCV_BCR_CTRL, VALID), 10000, "waiting for the falcon core to be selected")) return -1;
    tinynv_wr32(d, base + NV_PFALCON_FALCON_RM, d->chip_id);
  }
  return 0;
}

// Run a signed heavy-secure image on a falcon and wait for it to halt.
//
// The boot ROM checks the image before running it, so the four registers after the transfers matter as much as the
// transfers: where in the image the signature is, which engines the image may run on, which ucode id it claims, and
// which algorithm signs it. Get one wrong and the core simply never comes out of reset, with nothing said about why.
static int flcn_execute_hs(tinynv_gpu_t *g, uint64_t base, uint64_t img_paddr, uint32_t code_off, uint32_t data_off,
                          uint32_t imem_pa, uint32_t imem_va, uint32_t imem_sz, uint32_t dmem_pa, uint32_t dmem_va,
                          uint32_t dmem_sz, uint32_t pkc_off, uint32_t engid, uint32_t ucodeid,
                          const uint64_t *mailbox, uint32_t out_mbox[2]) {
  tinynv_dev_t *d = &g->dev;

  tinynv_flcn_disable_ctx_req(d, base);

  // aperture 0 is video memory. It is not in the published headers under any name, which is why it is a bare 0 here
  // and in the oracle.
  const uint32_t ctx_dma = 0;
  flcn_rmw(d, base + NV_PFALCON_FBIF_TRANSCFG(ctx_dma),
           FLD(NV_PFALCON_FBIF_TRANSCFG, TARGET) | FLD(NV_PFALCON_FBIF_TRANSCFG, MEM_TYPE),
           NV_SET(NV_PFALCON_FBIF_TRANSCFG, TARGET, 0) |
           NV_SET(NV_PFALCON_FBIF_TRANSCFG, MEM_TYPE, NV_PFALCON_FBIF_TRANSCFG_MEM_TYPE_PHYSICAL));

  // code into instruction memory, secure; then data into data memory, not secure. The source is biased by the virtual
  // base because the engine adds it back when it walks the blocks.
  uint32_t cmd = NV_SET(NV_PFALCON_FALCON_DMATRFCMD, WRITE, 0) |
                 NV_SET(NV_PFALCON_FALCON_DMATRFCMD, SIZE, NV_PFALCON_FALCON_DMATRFCMD_SIZE_256B) |
                 NV_SET(NV_PFALCON_FALCON_DMATRFCMD, CTXDMA, ctx_dma) |
                 NV_SET(NV_PFALCON_FALCON_DMATRFCMD, IMEM, 1) | NV_SET(NV_PFALCON_FALCON_DMATRFCMD, SEC, 1);
  if (flcn_dma(d, base, cmd, imem_pa, imem_va, img_paddr + code_off - imem_va, imem_sz)) return -1;

  cmd = NV_SET(NV_PFALCON_FALCON_DMATRFCMD, WRITE, 0) |
        NV_SET(NV_PFALCON_FALCON_DMATRFCMD, SIZE, NV_PFALCON_FALCON_DMATRFCMD_SIZE_256B) |
        NV_SET(NV_PFALCON_FALCON_DMATRFCMD, CTXDMA, ctx_dma) |
        NV_SET(NV_PFALCON_FALCON_DMATRFCMD, IMEM, 0) | NV_SET(NV_PFALCON_FALCON_DMATRFCMD, SEC, 0);
  if (flcn_dma(d, base, cmd, dmem_pa, dmem_va, img_paddr + data_off - dmem_va, dmem_sz)) return -1;

  tinynv_wr32(d, base + NV_PFALCON2_FALCON_BROM_PARAADDR(0), pkc_off);
  tinynv_wr32(d, base + NV_PFALCON2_FALCON_BROM_ENGIDMASK, engid);
  tinynv_wr32(d, base + NV_PFALCON2_FALCON_BROM_CURR_UCODE_ID,
              NV_SET(NV_PFALCON2_FALCON_BROM_CURR_UCODE_ID, VAL, ucodeid));
  tinynv_wr32(d, base + NV_PFALCON2_FALCON_MOD_SEL,
              NV_SET(NV_PFALCON2_FALCON_MOD_SEL, ALGO, NV_PFALCON2_FALCON_MOD_SEL_ALGO_RSA3K));

  tinynv_wr32(d, base + NV_PFALCON_FALCON_BOOTVEC, imem_va);

  if (mailbox) {
    tinynv_wr32(d, base + NV_PFALCON_FALCON_MAILBOX0, (uint32_t)*mailbox);
    tinynv_wr32(d, base + NV_PFALCON_FALCON_MAILBOX1, (uint32_t)(*mailbox >> 32));
  }

  tinynv_flcn_start_cpu(d, base);
  if (tinynv_flcn_wait_cpu_halted(d, base)) return -1;

  if (out_mbox) {
    out_mbox[0] = tinynv_rd32(d, base + NV_PFALCON_FALCON_MAILBOX0);
    out_mbox[1] = tinynv_rd32(d, base + NV_PFALCON_FALCON_MAILBOX1);
  }
  return 0;
}

// ---- the vbios path's software init -------------------------------------------------------------------------------

// A booter image carries several signatures and the one to use is patched into the image at a place the image itself
// names. Nothing is verified here; the boot ROM does that, and a wrong byte shows up as a core that never starts.
// booter_load and booter_unload have the same shape (checked: same header layout, same engine id 1 and ucode id 3 in
// their metadata, only the code and data offsets differ), so one reader serves both.
// Parse and patch a booter image in host memory. `*image` is malloc'd and the caller places it.
static int flcn_parse_booter(tinynv_gpu_t *g, const char *which, const char *sha_ga102, tinynv_booter_t *out,
                             uint8_t **image_out, size_t *len_out) {
  memset(out, 0, sizeof *out);
  *image_out = NULL;
  // hash-pinned exactly as the fmc is, and for the same reason: this is code the gpu's secure boot executes
  if (strcmp(g->dev.fw_name, "ga102"))
    return tinynv_fail("no %s hash is pinned for %s, so its image cannot be trusted", which, g->dev.fw_name);
  char name[64];
  snprintf(name, sizeof name, "%s-" TINYNV_FW_VER ".bin", which);
  if (tinynv_fw_load(g->dev.fw_name, name, sha_ga102, &out->fw)) return -1;

  const uint8_t *b = out->fw.data;
  size_t n = out->fw.size;
  tinynv_nvfw_bin_hdr_t bh;
  tinynv_nvfw_hs_header_v2_t hs;
  tinynv_nvfw_hs_load_header_v2_t lh;
  tinynv_nvfw_hs_load_header_app_t app;
#define TAKE(dst, off, what) do { \
    if ((off) > n || sizeof(dst) > n - (off)) return tinynv_fail("%s's %s is outside the image", which, what); \
    memcpy(&(dst), b + (off), sizeof(dst)); } while (0)
  TAKE(bh, 0, "binary header");
  TAKE(hs, bh.header_offset, "heavy-secure header");
  TAKE(lh, hs.header_offset, "load header");
  TAKE(app, hs.header_offset + sizeof(lh), "first application entry");

  uint32_t patch_loc, patch_sig, num_sig;
  TAKE(patch_loc, hs.patch_loc, "signature patch location");
  TAKE(patch_sig, hs.patch_sig, "signature selector");
  TAKE(num_sig, hs.num_sig, "signature count");
  if (!num_sig) return tinynv_fail("%s declares no signatures", which);

  size_t sig_len = hs.sig_prod_size / num_sig;
  size_t sig_off = (size_t)hs.sig_prod_offset + patch_sig;
  if (sig_off > n || sig_len > n - sig_off) return tinynv_fail("%s's production signature is outside the image", which);
  if (bh.data_offset > n || bh.data_size > n - bh.data_offset) return tinynv_fail("%s's payload is outside the image", which);
  if (patch_loc > bh.data_size || sig_len > bh.data_size - patch_loc)
    return tinynv_fail("%s's signature slot is outside its own payload", which);
#undef TAKE

  uint8_t *image = malloc(bh.data_size);
  if (!image) return tinynv_fail("no memory for the %s image", which);
  memcpy(image, b + bh.data_offset, bh.data_size);
  memcpy(image + patch_loc, b + sig_off, sig_len);

  out->code_off = app.offset;
  out->code_sz = app.size;
  out->data_off = lh.os_data_offset;
  out->data_sz = lh.os_data_size;
  *image_out = image;
  *len_out = bh.data_size;
  return 0;
}

static int flcn_prep_booter(tinynv_gpu_t *g, const char *which, const char *sha_ga102, tinynv_booter_t *out) {
  uint8_t *image;
  size_t len;
  if (flcn_parse_booter(g, which, sha_ga102, out, &image, &len)) return -1;
  // video memory: SEC2's dma engine fetches this and is given a physical address
  int rc = tinynv_alloc_boot_mem(&g->mm, len, image, 0, &out->image);
  free(image);
  return rc;
}

static void flcn_free_booter(tinynv_gpu_t *g, tinynv_booter_t *b) {
  if (b->image.size) tinynv_free_boot_mem(&g->mm, &b->image);
  tinynv_fw_free(&b->fw);
  memset(b, 0, sizeof *b);
}

static int flcn_vbios_init_sw(tinynv_gpu_t *g) {
  tinynv_flcn_t *f = &g->flcn;
  // Two megabytes below the top of video memory, which is where the oracle puts it. The region's own size is declared
  // to the firmware in 4K units and the firmware fills it; the driver only says where.
  f->frts_offset = g->dev.vram_size - 0x100000 - 0x100000;
  if (tinynv_vbios_fwsec_frts(g, f->frts_offset, &f->fwsec)) return -1;
  return flcn_prep_booter(g, "booter_load", BOOTER_LOAD_SHA_GA102, &f->booter);
}

int tinynv_flcn_init_sw(tinynv_gpu_t *g) {
  tinynv_flcn_t *f = &g->flcn;
  f->gpu = g;
  f->falcon = 0x00110000;
  f->sec2 = 0x00840000;
  if (!g->dev.fmc_boot) return flcn_vbios_init_sw(g);

  // the arguments the boot firmware reads, allocated now and filled in once the gsp side knows its own addresses
  tinynv_gsp_fmc_boot_params_t empty;
  memset(&empty, 0, sizeof(empty));
  if (tinynv_alloc_boot_mem(&g->mm, sizeof(empty), &empty, -1, &f->boot_args)) return -1;
  f->boot_args_sysmem = f->boot_args.addrs[0];

  if (tinynv_fw_load(g->dev.fw_name, "fmc-" TINYNV_FW_VER ".bin", FMC_SHA, &f->fmc_fw)) return -1;
  const uint8_t *image;
  size_t image_len;
  if (tinynv_elf_section(&f->fmc_fw, "image", &image, &image_len)) return -1;
  if (tinynv_elf_section(&f->fmc_fw, "hash", &f->hash, &f->hash_len)) return -1;
  if (tinynv_elf_section(&f->fmc_fw, "signature", &f->sig, &f->sig_len)) return -1;
  if (tinynv_elf_section(&f->fmc_fw, "publickey", &f->pkey, &f->pkey_len)) return -1;

  tinynv_cot_payload_t probe;
  if (f->hash_len > sizeof(probe.hash384) || f->sig_len > sizeof(probe.signature) || f->pkey_len > sizeof(probe.publicKey))
    return tinynv_fail("the fmc's authentication material does not fit the chain of trust message");

  if (tinynv_alloc_boot_mem(&g->mm, image_len, image, -1, &f->fmc_image)) return -1;
  f->fmc_sysmem = f->fmc_image.addrs[0];
  return 0;
}

// Send one message to the FSP and wait for it to answer.
//
// The wire format is NVIDIA's data model over MCTP: a two word header saying this is both the first and the last packet
// of a message, addressed to the FSP's vendor id, carrying a message type. Note the padding rule, which is the oracle's
// and is reproduced rather than corrected: four zero bytes are appended unconditionally, so a payload that is already a
// multiple of four still grows. The recorded boot writes a queue tail of 0x364 for an 0x35c byte payload, which is only
// consistent with that reading, and the FSP is the thing that decides what is correct here.
static int kfsp_send_msg(tinynv_gpu_t *g, uint32_t nvdm_type, const void *payload, size_t len) {
  tinynv_dev_t *d = &g->dev;
  uint8_t buf[FSP_MAX_MSG];
  size_t total = 8 + len + (4 - (len % 4));
  if (total >= FSP_MAX_MSG) return tinynv_fail("the fsp message is %zu bytes, over the %d byte mailbox", total, FSP_MAX_MSG);

  memset(buf, 0, total);
  uint32_t h0 = (1u << 31) | (1u << 30); // single packet: both the first and the last
  uint32_t h1 = 0x7eu | (0x10deu << 8) | (nvdm_type << 24);
  memcpy(buf, &h0, 4);
  memcpy(buf + 4, &h1, 4);
  memcpy(buf + 8, payload, len);

  // point the window at the start of the mailbox and let it advance on every write
  tinynv_wr32(d, NV_PFSP_EMEMC(0), NV_SET(NV_PFSP_EMEMC, OFFS, 0) | NV_SET(NV_PFSP_EMEMC, BLK, 0) |
                                   NV_SET(NV_PFSP_EMEMC, AINCW, 1) | NV_SET(NV_PFSP_EMEMC, AINCR, 0));
  for (size_t i = 0; i < total; i += 4) {
    uint32_t w;
    memcpy(&w, buf + i, 4);
    tinynv_wr32(d, NV_PFSP_EMEMD(0), w);
  }

  // the tail names the last word written, then the head being written hands the message over
  tinynv_wr32(d, NV_PFSP_QUEUE_TAIL(0), (uint32_t)(total - 4));
  tinynv_wr32(d, NV_PFSP_QUEUE_HEAD(0), 0);

  // the FSP answers by advancing its own queue's head past our tail
  double deadline = tinynv_now_s() + 10.0;
  uint32_t head, tail;
  do {
    head = tinynv_rd32(d, NV_PFSP_MSGQ_HEAD(0));
    tail = tinynv_rd32(d, NV_PFSP_MSGQ_TAIL(0));
    if (head != tail) break;
  } while (tinynv_now_s() < deadline);
  if (head == tail) return tinynv_fail("the fsp did not answer the chain of trust message in 10 s (head and tail both %#x)", head);

  // turn the window around for reading and consume the reply, which the boot path does not otherwise look at
  tinynv_wr32(d, NV_PFSP_EMEMC(0), NV_SET(NV_PFSP_EMEMC, OFFS, 0) | NV_SET(NV_PFSP_EMEMC, BLK, 0) |
                                   NV_SET(NV_PFSP_EMEMC, AINCW, 0) | NV_SET(NV_PFSP_EMEMC, AINCR, 1));
  tinynv_wr32(d, NV_PFSP_MSGQ_TAIL(0), tinynv_rd32(d, NV_PFSP_MSGQ_HEAD(0)));
  return 0;
}

// ---- the vbios path's hardware boot -------------------------------------------------------------------------------
//
// Four steps, and each one is a gate on the next. FWSEC runs on the GSP falcon and places the write-protected region -
// there is a register that says whether it did, and it is checked, because FWSEC halts successfully either way. The
// falcon is then reset onto its RISC-V half and handed the address of GSP-RM's arguments. booter_load runs on SEC2 with
// the WPR metadata in its mailbox and unpacks GSP-RM into the region FWSEC placed; its mailbox comes back non-zero if
// it refused. Finally the GSP core is confirmed running, which is the same state the chain of trust leaves on Blackwell
// and where both paths rejoin.
static int flcn_vbios_init_hw(tinynv_gpu_t *g) {
  tinynv_flcn_t *f = &g->flcn;
  tinynv_gsp_t *gsp = &g->gsp;
  tinynv_dev_t *d = &g->dev;
  const tinynv_falcon_ucode_desc_v3_t *dsc = &f->fwsec.desc;

  if (!gsp->wpr_meta_sysmem || !gsp->libos_args_sysmem)
    return tinynv_fail("the vbios boot needs gsp's addresses: run the software init for both blocks first");

  // NVIDIA's driver fails here: "unexpected WPR2 already up, cannot proceed with booting GSP". A region that is up has a
  // firmware in it, or had one and nothing tore it down, and running FWSEC to place a new region over it is not a boot.
  // This costs no register read - early init read the register on arrival, and again after a reset - so the recorded
  // boot, on which the register reads zero here, is reproduced exactly.
  if (d->wpr2_hi_now)
    return tinynv_fail("the write-protected region is already up (hi %#x): gsp-rm cannot be booted over a live one. The "
                       "previous run should have unloaded it (TINYNV_UNLOAD=1, the default); a replug re-enumerates the card",
                       d->wpr2_hi_now);

  // FWSEC, out of the card's own rom, on the gsp falcon. Its data segment loads at virtual zero, which is why the
  // dmem virtual base is a literal here and the code's is not.
  if (tinynv_flcn_reset(g, f->falcon, 0)) return -1;
  if (flcn_execute_hs(g, f->falcon, f->fwsec.image.paddr, 0, dsc->IMEMLoadSize,
                      dsc->IMEMPhysBase, dsc->IMEMVirtBase, dsc->IMEMLoadSize,
                      dsc->DMEMPhysBase, 0, dsc->DMEMLoadSize,
                      dsc->PKCDataOffset, dsc->EngineIdMask, dsc->UcodeId, NULL, NULL)) return -1;

  // FWSEC halts whether or not it did the work, so the region itself is the test. Only when it failed is the error code
  // it leaves in the vbios scratch read (NVIDIA checks it first; the oracle never reads it, so on the recorded path the
  // read would be a divergence) - the region being placed is the whole of what matters, and the code says why it was not.
  if (!tinynv_rd32(d, NV_PFB_PRI_MMU_WPR2_ADDR_HI)) {
    uint32_t sc = tinynv_rd32(d, NV_PBUS_VBIOS_SCRATCH(0x0e));
    return tinynv_fail("fwsec ran and halted but the write-protected region is still unplaced (frts error code %#x)", sc >> 16);
  }

  // the same falcon again, now selecting its riscv core, and told where gsp-rm's arguments are
  if (tinynv_flcn_reset(g, f->falcon, 1)) return -1;
  tinynv_wr32(d, NV_PGSP_FALCON_MAILBOX0, (uint32_t)gsp->libos_args_sysmem);
  tinynv_wr32(d, NV_PGSP_FALCON_MAILBOX1, (uint32_t)(gsp->libos_args_sysmem >> 32));

  // booter_load on SEC2, carrying the wpr metadata. engine id 1 and ucode id 3 are what this image declares itself to
  // be; the boot rom refuses it under any other pair.
  if (tinynv_flcn_reset(g, f->sec2, 0)) return -1;
  uint32_t mbx[2] = {0, 0};
  uint64_t wpr = gsp->wpr_meta_sysmem;
  if (flcn_execute_hs(g, f->sec2, f->booter.image.paddr, f->booter.code_off, f->booter.data_off,
                      0, f->booter.code_off, f->booter.code_sz, 0, 0, f->booter.data_sz,
                      0x10, 1, 3, &wpr, mbx)) return -1;
  if (mbx[0]) return tinynv_fail("booter_load refused to unpack gsp-rm: mailbox %#x %#x", mbx[0], mbx[1]);

  tinynv_wr32(d, f->falcon + NV_PFALCON_FALCON_OS, 0);
  if (!NV_GET(tinynv_rd32(d, f->falcon + NV_PRISCV_RISCV_CPUCTL), NV_PRISCV_RISCV_CPUCTL, ACTIVE_STAT))
    return tinynv_fail("booter_load reported success but the gsp core is not running");
  return 0;
}

int tinynv_flcn_init_hw(tinynv_gpu_t *g) {
  tinynv_flcn_t *f = &g->flcn;
  tinynv_gsp_t *gsp = &g->gsp;
  if (!g->dev.fmc_boot) return flcn_vbios_init_hw(g);
  if (!gsp->wpr_meta_sysmem || !gsp->libos_args_sysmem)
    return tinynv_fail("the chain of trust needs gsp's addresses: run the software init for both blocks first");

  // now that the gsp side has placed everything, the boot firmware can be told where it all is
  tinynv_gsp_fmc_boot_params_t p;
  memset(&p, 0, sizeof(p));
  p.bootGspRmParams.target = TINYNV_GSP_DMA_TARGET_COHERENT_SYSTEM;
  p.bootGspRmParams.gspRmDescOffset = gsp->wpr_meta_sysmem;
  p.bootGspRmParams.gspRmDescSize = (uint32_t)sizeof(tinynv_wpr_meta_t);
  p.bootGspRmParams.bIsGspRmBoot = 1;
  p.gspRmParams.target = TINYNV_GSP_DMA_TARGET_COHERENT_SYSTEM;
  p.gspRmParams.bootArgsOffset = gsp->libos_args_sysmem;
  nv_wr_block(&f->boot_args.view, 0, &p, sizeof(p));

  tinynv_cot_payload_t cot;
  memset(&cot, 0, sizeof(cot));
  cot.version = 2;
  cot.size = (uint16_t)sizeof(cot);
  // the firmware places its own protected region; these say how big it should be and where, counted from the top of vram
  cot.frtsVidmemOffset = 0x1c00000;
  cot.frtsVidmemSize = 0x100000;
  cot.gspBootArgsSysmemOffset = f->boot_args_sysmem;
  cot.gspFmcSysmemOffset = f->fmc_sysmem;
  memcpy(cot.hash384, f->hash, f->hash_len);
  memcpy(cot.signature, f->sig, f->sig_len);
  memcpy(cot.publicKey, f->pkey, f->pkey_len);

  if (kfsp_send_msg(g, TINYNV_NVDM_TYPE_COT, &cot, sizeof(cot))) return -1;

  // the falcon is ours once the boot rom drops its lockdown; if it never does, the chain was refused
  return tinynv_wait_reg(&g->dev, f->falcon + NV_PFALCON_FALCON_HWCFG2,
                         1u << NV_PFALCON_FALCON_HWCFG2_RISCV_BR_PRIV_LOCKDOWN_LO, 0, 10000,
                         "waiting for the gsp falcon to leave boot lockdown");
}

void tinynv_flcn_fini(tinynv_gpu_t *g) {
  tinynv_flcn_t *f = &g->flcn;
  tinynv_vbios_fwsec_free(g, &f->fwsec);
  flcn_free_booter(g, &f->booter);
  tinynv_fw_free(&f->unload.fw);
  if (f->boot_args.size) tinynv_free_boot_mem(&g->mm, &f->boot_args);
  if (f->fmc_image.size) tinynv_free_boot_mem(&g->mm, &f->fmc_image);
  tinynv_fw_free(&f->fmc_fw);
  f->hash = f->sig = f->pkey = NULL;
}

// ---- the vbios path's unload -------------------------------------------------------------------------------------
//
// The mirror image of flcn_vbios_init_hw, in the order NVIDIA's kgspTeardown_TU102 does it and nouveau's tu102_gsp_fini
// copies: the GSP falcon reset onto its falcon core (GSP-RM has halted on the riscv one, see tinynv_gsp_unload), FWSEC
// run again with the SB command, then booter_unload on SEC2 with 0xff in both mailboxes - the value that means "not a
// suspend, tear it down" - and finally the region register, which must read zero. The oracle has none of this: on
// Linux it leaves the firmware resident and resets the card next time. On a 3060 over thunderbolt that reset path
// produced a firmware that came up and went silent, so the card is left the way NVIDIA leaves it instead.
//
// Two halves, because of where the images go. Both are written into the video memory the boot already used and proved -
// FWSEC-SB over the FRTS image (the same image, patched for the other command, so the same size) and booter_unload over
// booter_load (smaller) - which are addresses the processor wrote through the memory window and the falcons fetched
// from in this very boot. A fresh allocation at exit could land past the 256 MB the window shows. And the writes are
// made first, while GSP-RM is still up and everything about the window is exactly as it was all run.

int tinynv_flcn_unload_prepare(tinynv_gpu_t *g) {
  tinynv_flcn_t *f = &g->flcn;
  if (g->dev.fmc_boot) return 0;
  if (!f->fwsec.image.size || !f->booter.image.size)
    return tinynv_fail("the boot's fwsec and booter_load slots are gone, so there is nowhere proven to put the unload images");
  if (!f->fwsec.sb_image) return tinynv_fail("no fwsec-sb image was built at boot: %s", f->fwsec.sb_error);

  uint8_t *image;
  size_t len;
  if (flcn_parse_booter(g, "booter_unload", BOOTER_UNLOAD_SHA_GA102, &f->unload, &image, &len)) return -1;
  if (len > f->booter.image.size) {
    free(image);
    return tinynv_fail("booter_unload is %zu bytes and booter_load's slot only %zu", len, (size_t)f->booter.image.size);
  }
  nv_wr_block(&f->fwsec.image.view, 0, f->fwsec.sb_image, f->fwsec.image_len);
  nv_wr_block(&f->booter.image.view, 0, image, len);
  free(image);
  f->unload_staged = 1;
  return 0;
}

int tinynv_flcn_unload_hw(tinynv_gpu_t *g) {
  tinynv_flcn_t *f = &g->flcn;
  tinynv_dev_t *d = &g->dev;
  if (d->fmc_boot) return 0;
  if (!f->unload_staged) return tinynv_fail("the unload images were not staged");
  double t0 = tinynv_now_s();

  if (tinynv_flcn_reset(g, f->falcon, 0)) return -1;
  const tinynv_falcon_ucode_desc_v3_t *dsc = &f->fwsec.desc;
  if (flcn_execute_hs(g, f->falcon, f->fwsec.image.paddr, 0, dsc->IMEMLoadSize,
                      dsc->IMEMPhysBase, dsc->IMEMVirtBase, dsc->IMEMLoadSize,
                      dsc->DMEMPhysBase, 0, dsc->DMEMLoadSize,
                      dsc->PKCDataOffset, dsc->EngineIdMask, dsc->UcodeId, NULL, NULL)) return -1;
  // NVIDIA asserts on this and goes on to booter_unload regardless, because the region coming down is what matters
  uint32_t sb_err = tinynv_rd32(d, NV_PBUS_VBIOS_SCRATCH(0x15)) & 0xffff;
  if (sb_err) fprintf(stderr, "tinynv: fwsec-sb reported error %#x putting the pre-os applications back; tearing the region down anyway\n", sb_err);

  uint32_t mbx[2] = {0, 0};
  uint64_t arg = 0xffull | (0xffull << 32);
  if (tinynv_flcn_reset(g, f->sec2, 0)) return -1;
  if (flcn_execute_hs(g, f->sec2, f->booter.image.paddr, f->unload.code_off, f->unload.data_off,
                      0, f->unload.code_off, f->unload.code_sz, 0, 0, f->unload.data_sz, 0x10, 1, 3, &arg, mbx)) return -1;
  if (mbx[0]) return tinynv_fail("booter_unload refused to tear the region down: mailbox %#x %#x", mbx[0], mbx[1]);

  uint32_t wpr2 = tinynv_rd32(d, NV_PFB_PRI_MMU_WPR2_ADDR_HI);
  if (wpr2) return tinynv_fail("booter_unload reported success but the write-protected region is still up (hi %#x)", wpr2);
  f->unloaded = 1;
  fprintf(stderr, "tinynv: gsp-rm unloaded and its region torn down in %.2f s: the card is cold, and the next open needs no reset\n",
          tinynv_now_s() - t0);
  return 0;
}
