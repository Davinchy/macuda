// The card's own VBIOS, and the FWSEC image inside it.
#ifndef TINYNV_VBIOS_H
#define TINYNV_VBIOS_H
#include "mmu.h"
#include "nv_structs.h"
#include "vbios_structs.h"

typedef struct tinynv_gpu tinynv_gpu_t;

typedef struct {
  tinynv_falcon_ucode_desc_v3_t desc; // what the falcon launch needs: load sizes, bases, the signature's offset
  tinynv_bootmem_t image;             // the patched image, in video memory where the falcon's DMA can reach it
} tinynv_fwsec_t;

// Read the card's VBIOS, find the production FWSEC image, patch it to run the FRTS command for a region at
// `frts_offset`, and place it in video memory. On success `out` owns the image until tinynv_vbios_fwsec_free.
int tinynv_vbios_fwsec_frts(tinynv_gpu_t *g, uint64_t frts_offset, tinynv_fwsec_t *out);
void tinynv_vbios_fwsec_free(tinynv_gpu_t *g, tinynv_fwsec_t *f);
#endif
