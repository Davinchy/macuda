// Bringing a GPU up, in the order the oracle does it.
//
// The order is not cosmetic. Both firmware blocks allocate before either touches hardware, because the chain of trust
// message the falcon sends carries the addresses of GSP-RM's structures inside it. Doing the falcon's hardware step
// before GSP-RM has placed its metadata would send the FSP a message pointing at nothing.
#include "gpu.h"
#include "internal.h"
#include <stdio.h>
#include <stdlib.h>
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

// Why the card is unloaded rather than left "idle warm" with the firmware resident, which is what this did until
// 2026-09-17: a resident firmware makes the next open reset the card first, and on a 3060 over thunderbolt that reset
// path produced a GSP-RM that came up, ran its register sequence and never sent its start-up notice - every second
// boot of the afternoon, and the only cure was a replug. The recorded boot, the one this driver reproduces operation
// for operation, is of a cold card. So leave the card cold: the way NVIDIA's driver and nouveau leave it, with the
// firmware told, halted and its region torn down. TINYNV_UNLOAD=0 restores the old behaviour, for comparison.
int tinynv_gpu_unload(tinynv_gpu_t *g) {
  if (!g->gsp.up) return 0;         // nothing came up, so there is nothing to tell
  if (!g->dev.pci->live) return 0;  // a recording ends where it ends
  if (g->dev.fmc_boot) return 0;    // the chain-of-trust path has no booter_unload; its reset-on-open has always worked
  const char *knob = getenv("TINYNV_UNLOAD");
  if (knob && !strcmp(knob, "0")) {
    fprintf(stderr, "libtinynv: TINYNV_UNLOAD=0: gsp-rm is left resident, and the next open will reset the card first\n");
    return 0;
  }
  if (tinynv_flcn_unload_prepare(g) || tinynv_gsp_unload(g) || tinynv_flcn_unload_hw(g)) {
    fprintf(stderr, "libtinynv: the unload did not complete: %s. The card is left as it is; the next open will reset it, "
                    "and if that fails too a replug re-enumerates it\n", tinynv_last_error());
    return -1;
  }
  return 0;
}

void tinynv_gpu_close(tinynv_gpu_t *g) {
  tinynv_gpu_unload(g);
  tinynv_gsp_fini(g);
  tinynv_flcn_fini(g);
  tinynv_dev_quiesce(&g->dev);
}
