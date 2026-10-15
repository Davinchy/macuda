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

// A boot that did not finish, undone far enough to be made again. GSP-RM is not asked to leave - it is the thing that
// never answered - so this is the falcon half of the unload alone: the images staged into the boot's own slots, the
// GSP falcon reset out from under whatever is on it, FWSEC-SB, booter_unload, the region register read back as zero.
// Then every host-side structure the firmware was handed is freed and rebuilt, because a firmware that half-started
// may have written into its queues and argument blocks, and a second boot has to begin from what the first began from.
static int unmake_boot(tinynv_gpu_t *g) {
  // Not the unload's falcon half: on a card that has just lost and regained its configuration SEC2 would not run
  // booter_unload (it sat "stopped", 2026-09-21), and a firmware that never answered cannot be asked to leave. The
  // reset through the backend is what the quiesce does after every failed boot, and every boot after one of those came
  // up cold today, so it is what a retry is made of.
  if (tinynv_dev_reset_cold(&g->dev)) return -1;
  tinynv_gsp_fini(g);
  tinynv_flcn_fini(g);
  memset(&g->gsp, 0, sizeof(g->gsp));
  memset(&g->flcn, 0, sizeof(g->flcn));
  g->flcn.gpu = g;
  g->gsp.gpu = g;
  return 0;
}

int tinynv_gpu_boot_firmware(tinynv_gpu_t *g) {
  // Three, because on the 3060 that loses its configuration under a booting firmware (2026-09-21) the loss is a
  // per-boot event: the boot after a lost one usually comes up. A retry is cheap - five seconds - against the
  // alternative, which is a process that reports a failed boot and a person who power-cycles the enclosure.
  // Six, not three: the window loss is roughly even odds per boot on this enclosure's supply, so three attempts
  // fail together about one run in eight (the 35B-A3B bench on 2026-09-21 lost all three, on a card at 38.8 C - not
  // thermal). Six brings all-failing to about one in fifty, at ~12 s per lost attempt. TINYNV_BOOT_ATTEMPTS
  // overrides. A workaround for a marginal supply, not a fix: a supply that held would not clear the configuration.
  int attempts = g->dev.pci->live && !g->dev.fmc_boot ? 6 : 1;
  { const char *e = getenv("TINYNV_BOOT_ATTEMPTS"); if (e && *e && atoi(e) > 0) attempts = atoi(e); }
  for (int attempt = 1;; attempt++) {
    if (tinynv_gpu_init_sw(g)) return -1;
    if (!tinynv_gpu_init_hw(g)) {
      if (attempt > 1) fprintf(stderr, "tinynv: the firmware came up on boot attempt %d\n", attempt);
      return 0;
    }
    if (attempt >= attempts || !g->gsp.init_timed_out) return -1;
    char why[512];
    snprintf(why, sizeof why, "%s", tinynv_last_error());
    fprintf(stderr, "tinynv: boot attempt %d did not finish (%s)%s; tearing the firmware down and booting again\n",
            attempt, why, g->dev.windows_restored ? " - the card's configuration had been cleared under it" : "");
    if (unmake_boot(g)) {
      fprintf(stderr, "tinynv: could not undo the failed boot (%s), so there will be no second attempt\n", tinynv_last_error());
      return tinynv_fail("%s", why);
    }
  }
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
  // THE ORDER IS THE POINT AND IT USED TO BE WRONG. It was gsp_fini, flcn_fini, quiesce - which took the mappings
  // away FIRST and dropped bus mastering afterwards. gsp_fini frees gsp->queues, which is host memory the firmware
  // DMAs into, so every exit opened a window where a card still mastering the bus pointed at translations that had
  // just been destroyed - the shape macOS latches as a fault that only a re-enumeration clears.
  //
  // Three things now happen in the only order that works:
  //
  //   1. TELL THE FIRMWARE TO LEAVE (the Ampere unload). An RPC, so it has to go out while the card can still read
  //      the queue - ahead of the quiesce, never after it. This is the half the quiesce-first commit promised would
  //      follow; on the chain-of-trust boot path (the 5090) it is a no-op by its own first checks.
  //   2. DROP BUS MASTERING, before anything is unmapped.
  //   3. TAKE THE MEMORY AWAY. gsp_fini frees the queues, by which point the card cannot reach them at all.
  //
  // Step 2 is the half that does not depend on the firmware cooperating: whatever the firmware believes it is still
  // doing, a card that is not a bus master cannot reach host memory.
  tinynv_gpu_unload(g);
  tinynv_dev_quiesce(&g->dev);
  tinynv_gsp_fini(g);
  tinynv_flcn_fini(g);
}
