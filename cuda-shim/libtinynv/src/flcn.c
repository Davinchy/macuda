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
#include "fw_layout.h"
#include <string.h>
#include <time.h>

static const char *FMC_SHA = "cb59a35c1d4bd1274d7267fd10243c29f843ff41c851b9cbd59f5af2ddd7fece";

#define FSP_MAX_MSG 0x400

int tinynv_flcn_init_sw(tinynv_gpu_t *g) {
  tinynv_flcn_t *f = &g->flcn;
  f->gpu = g;
  f->falcon = 0x00110000;

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

int tinynv_flcn_init_hw(tinynv_gpu_t *g) {
  tinynv_flcn_t *f = &g->flcn;
  tinynv_gsp_t *gsp = &g->gsp;
  if (!g->dev.fmc_boot) return tinynv_fail("%s does not boot through the chain of trust", g->dev.chip_name);
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
  // the firmware places its own protected region; these say how big it should be and where, counted from the top of
  // vram - named in fw_layout.h beside the reservation they size
  cot.frtsVidmemOffset = TINYNV_FW_FRTS_FROM_END;
  cot.frtsVidmemSize = TINYNV_FW_FRTS_SIZE;
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
  if (f->boot_args.size) tinynv_free_boot_mem(&g->mm, &f->boot_args);
  if (f->fmc_image.size) tinynv_free_boot_mem(&g->mm, &f->fmc_image);
  tinynv_fw_free(&f->fmc_fw);
  f->hash = f->sig = f->pkey = NULL;
}
