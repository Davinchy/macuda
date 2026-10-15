// The chain of trust: how a Blackwell GPU is persuaded to run firmware we placed in memory.
//
// Nothing on this chip executes unsigned code. A separate security processor, the FSP, holds the root of trust; the
// driver hands it a message naming the boot image, its hash, its signature and the public key that signs it, and the FSP
// verifies the chain and releases the GSP falcon from lockdown.
//
// Ampere has no such processor. The image it runs to place the write-protected region is already on the card, inside the
// VBIOS, and the driver runs it on the GSP falcon itself; only then can booter_load, on SEC2, unpack GSP-RM. Both paths
// live here, because both end in the same place - a GSP falcon out of lockdown with our firmware in it - and the falcon
// itself is driven identically either way.
#ifndef TINYNV_FLCN_H
#define TINYNV_FLCN_H
#include "dev.h"
#include "fw.h"
#include "mmu.h"
#include "vbios.h"

// A booter image as SEC2 runs it: the signed bytes in video memory, and where its code and data are inside them.
// booter_load unpacks GSP-RM into the write-protected region at boot; booter_unload tears the region down at unload.
typedef struct {
  tinynv_blob_t fw;
  tinynv_bootmem_t image;
  uint32_t code_off, code_sz, data_off, data_sz;
} tinynv_booter_t;

typedef struct {
  tinynv_gpu_t *gpu;
  uint64_t falcon; // the register base of the GSP falcon
  uint64_t sec2;   // and of SEC2, the second falcon, which is where booter_load runs on the vbios path

  tinynv_bootmem_t boot_args; // what the boot firmware is told when it starts
  uint64_t boot_args_sysmem;

  tinynv_blob_t fmc_fw;
  tinynv_bootmem_t fmc_image;
  uint64_t fmc_sysmem;
  const uint8_t *hash, *sig, *pkey; // the authentication material, pointing into fmc_fw
  size_t hash_len, sig_len, pkey_len;

  // the vbios path: FWSEC out of the card's own rom, then booter_load over SEC2
  tinynv_fwsec_t fwsec;
  uint64_t frts_offset;         // where fwsec is asked to place the write-protected region
  tinynv_booter_t booter;       // booter_load, held for the life of the process as the oracle holds it
  tinynv_booter_t unload;       // booter_unload's offsets and firmware; its bytes go into booter_load's slot at unload
  int unload_staged;            // both unload images have been written into the boot's slots
  int unloaded;                 // the unload ran and the region is down: the next open finds a cold card
} tinynv_flcn_t;

// Driving a falcon, which GSP-RM's register sequencer asks the driver to do on its behalf (gsp.c run_cpu_seq).
int tinynv_flcn_reset(tinynv_gpu_t *g, uint64_t base, int riscv);
void tinynv_flcn_disable_ctx_req(tinynv_dev_t *d, uint64_t base);
void tinynv_flcn_start_cpu(tinynv_dev_t *d, uint64_t base);
int tinynv_flcn_wait_cpu_halted(tinynv_dev_t *d, uint64_t base);

int tinynv_flcn_init_sw(tinynv_gpu_t *g); // place the boot image and its arguments. no hardware.
int tinynv_flcn_init_hw(tinynv_gpu_t *g); // send the chain of trust message and wait for the falcon to be released.
void tinynv_flcn_fini(tinynv_gpu_t *g);
// Undo init_hw on the vbios path the way NVIDIA's driver does at unload: FWSEC-SB puts the pre-OS applications back and
// booter_unload tears the write-protected region down, so the next open finds a cold card and boots it the recorded
// way. _prepare writes both images into the slots the boot used, and runs while GSP-RM is still up; _hw drives the
// falcons, and runs after GSP-RM has been told and has halted (tinynv_gsp_unload). Both are no-ops on the chain of trust.
int tinynv_flcn_unload_prepare(tinynv_gpu_t *g);
int tinynv_flcn_unload_hw(tinynv_gpu_t *g);

#endif
