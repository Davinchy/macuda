// The chain of trust: how a Blackwell GPU is persuaded to run firmware we placed in memory.
//
// Nothing on this chip executes unsigned code. A separate security processor, the FSP, holds the root of trust; the
// driver hands it a message naming the boot image, its hash, its signature and the public key that signs it, and the FSP
// verifies the chain and releases the GSP falcon from lockdown. Ampere does this differently, out of the VBIOS, which is
// why this file is only the Blackwell path.
#ifndef TINYNV_FLCN_H
#define TINYNV_FLCN_H
#include "fw.h"
#include "mmu.h"

typedef struct tinynv_gpu tinynv_gpu_t;

typedef struct {
  tinynv_gpu_t *gpu;
  uint64_t falcon; // the register base of the GSP falcon

  tinynv_bootmem_t boot_args; // what the boot firmware is told when it starts
  uint64_t boot_args_sysmem;

  tinynv_blob_t fmc_fw;
  tinynv_bootmem_t fmc_image;
  uint64_t fmc_sysmem;
  const uint8_t *hash, *sig, *pkey; // the authentication material, pointing into fmc_fw
  size_t hash_len, sig_len, pkey_len;
} tinynv_flcn_t;

int tinynv_flcn_init_sw(tinynv_gpu_t *g); // place the boot image and its arguments. no hardware.
int tinynv_flcn_init_hw(tinynv_gpu_t *g); // send the chain of trust message and wait for the falcon to be released.
void tinynv_flcn_fini(tinynv_gpu_t *g);

#endif
