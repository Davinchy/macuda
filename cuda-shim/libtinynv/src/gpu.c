// Bringing a GPU up, in the order the oracle does it.
//
// The order is not cosmetic. Both firmware blocks allocate before either touches hardware, because the chain of trust
// message the falcon sends carries the addresses of GSP-RM's structures inside it. Doing the falcon's hardware step
// before GSP-RM has placed its metadata would send the FSP a message pointing at nothing.
#include "gpu.h"
#include "internal.h"
#include <string.h>

int tinynv_gpu_open(tinynv_gpu_t *g, tinynv_pci_t *pci) {
  memset(g, 0, sizeof(*g));
  if (tinynv_dev_early_init(&g->dev, pci)) return -1;
  if (tinynv_dev_mmu_init(&g->dev)) return -1;
  if (tinynv_mm_init(&g->mm, &g->dev)) return -1;
  g->flcn.gpu = g;
  g->gsp.gpu = g;
  return 0;
}

int tinynv_gpu_init_sw(tinynv_gpu_t *g) {
  if (tinynv_flcn_init_sw(g)) return -1;
  return tinynv_gsp_init_sw(g);
}

int tinynv_gpu_init_hw(tinynv_gpu_t *g) {
  if (tinynv_flcn_init_hw(g)) return -1;
  return tinynv_gsp_init_hw(g);
}

void tinynv_gpu_close(tinynv_gpu_t *g) {
  tinynv_gsp_fini(g);
  tinynv_flcn_fini(g);
  tinynv_dev_quiesce(&g->dev);
}
