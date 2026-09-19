// Does this driver's memory manager overlap the firmware's write-protected region? Asked, not computed.
//
// Since 2026-09-19 the manager holds back TINYNV_FW_RESERVE_TOP (fw_layout.h), derived from the sizes handed to the
// firmware, and the boot itself reads the two registers below and refuses the open on an overlap (tinynv.c,
// tinynv_mm_check_fw_carveout). This probe is the read-only measurement that came first and stays the record: run it
// after any change to those sizes and write its numbers into libtinynv-design.md SS4f-ter. What follows is the
// arithmetic as it stood when the 64 MB was flat, kept because it is why the probe exists.
//
// mmu.c held back a flat 64 MB: `managed = dev->vram_size - 64 * MB`, with the comment "the top of video memory is
// reserved for the firmware's own structures, so the manager never sees it". But the sizes this driver HANDS the
// firmware in the WPR metadata (gsp.c) add up to considerably more than that:
//
//     vgaWorkspace     0.1 MB
//     pmuReserved     24.1 MB
//     nonWprHeap       2.1 MB
//     gspFwHeap      135.0 MB
//     frts             1.0 MB
//     TOTAL          162.4 MB     against 64 MB held back
//
// Two numbers that should be related and are not. If the firmware places all of that at the top of video memory, the
// manager's range runs ~98 MB into it, and the only reason nothing has broken is that the manager allocates from a
// 32 GB pool that has never been more than half full - a latent overlap, not a safe one.
//
// BUT THAT IS ARITHMETIC AND ARITHMETIC IS WHAT HAS BEEN WRONG ALL DAY. nonWprHeap is named "non-WPR" and may not be
// in video memory at all; the firmware may place its carveout somewhere other than the top; the sizes we ask for may
// not be the sizes it takes. So this reads WPR2's actual base and limit off the chip - NV_PFB_PRI_MMU_WPR2_ADDR_LO
// and _HI, the same pair dev.c already reads to notice a live firmware - and compares them against where the manager
// actually stops. One register pair, read-only, nothing allocated and nothing changed.
//
// WHY IT MATTERS BEYOND TIDINESS: a guest is capped near 80 MiB of device memory because its allocations come out of
// GSP-RM's heap, and the obvious fix is to raise gspFwHeapSize. If the reservation does not track that constant,
// raising it walks the firmware's region further into memory the manager is already handing out - so this has to be
// answered BEFORE the heap is touched, not after.
#include "tinynv.h"
#include "fw_layout.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  if (!getenv("TINYNV_HW") || !getenv("TINYNV_SOCKET")) {
    printf("usage: TINYNV_HW=1 TINYNV_SOCKET=<socket> %s\n", argv[0]);
    return 2;
  }
  tinynv_device_t d = NULL;
  if (tinynv_device_get(&d, 0) != TINYNV_OK) { printf("  no device: %s\n", tinynv_last_error()); return 1; }
  tinynv_device_props_t props;
  if (tinynv_device_props(d, &props) != TINYNV_OK) {
    printf("  cannot open the card: %s\n", tinynv_last_error());
    return 1;
  }

  uint64_t base = 0, limit = 0, vram = 0, managed_end = 0;
  if (tinynv_device_wpr2_range(d, &base, &limit, &vram, &managed_end)) {
    printf("  could not read the wpr2 registers: %s\n", tinynv_last_error());
    return 1;
  }

  const double MB = 1048576.0;
  printf("  video memory   %.1f MB (%#llx)\n", vram / MB, (unsigned long long)vram);
  printf("  manager ends   %.1f MB (%#llx)\n", managed_end / MB, (unsigned long long)managed_end);
  printf("  wpr2           %#llx .. %#llx", (unsigned long long)base, (unsigned long long)limit);
  if (limit > base) printf("  (%.1f MB, starting %.1f MB below the top)", (limit - base) / MB, (vram - base) / MB);
  printf("\n");

  if (!base && !limit) {
    printf("  WPR2 IS NOT UP. The firmware has not placed its region, or placed it somewhere these registers do not\n"
           "  report, so this run says nothing either way and the arithmetic in the header stands unchecked.\n");
    return 1;
  }
  if (base >= managed_end) {
    printf("  NO OVERLAP: the firmware's region starts at or above where the manager stops, with %.1f MB of gap\n"
           "  (%.1f MB of it is the firmware's unprotected non-wpr heap, which sits just below wpr2). The %llu MB held\n"
           "  back is derived from the sizes handed to the firmware (fw_layout.h) and the boot checks it against these\n"
           "  registers, so raising gspFwHeapSize without raising the reservation now fails the build, not the card.\n",
           (base - managed_end) / MB, (double)TINYNV_FW_NONWPR_HEAP / MB,
           (unsigned long long)(TINYNV_FW_RESERVE_TOP >> 20));
    return base - managed_end >= TINYNV_FW_NONWPR_HEAP ? 0 : 1;
  }
  printf("  OVERLAP: the manager hands out addresses up to %#llx and the firmware's region starts at %#llx, so\n"
         "  %.1f MB of what this driver believes it owns is inside the write-protected region. Nothing has broken\n"
         "  because the manager allocates from %.1f GB and has never filled it - this is latent, not safe, and any\n"
         "  workload large enough to reach the top would corrupt the firmware or be refused by it.\n",
         (unsigned long long)managed_end, (unsigned long long)base, (managed_end - base) / MB, vram / MB / 1024.0);
  return 1;
}
