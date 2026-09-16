// What GSP-RM has to allocate from, asked rather than inferred.
//
// Session C bisected a guest's virtual address reservation to a hard ceiling of 2 MiB - granted at 2, refused above,
// in both of its address spaces. Two megabytes is exactly one 4 KB page table's worth of coverage at the small page
// size: 512 entries of eight bytes. That is too exact to be a capacity shortfall and reads as a floor: whatever backs
// a reservation gets one page and cannot get a second.
//
// The explanation that fits is that GSP-RM has no video memory to allocate page tables from. Nothing in this driver's
// bring-up gives it a heap, and nothing has ever needed one, because libtinynv allocates every byte itself and hands
// over descriptors - its own address space is a tree we build here and describe with COPY_SERVER_RESERVED_PDES. A
// guest client asking RM for its own space is the first thing to require RM to find memory on its own account.
//
// That is an inference, and the last several hours have been a lesson in what inferences cost when they are built on
// rather than checked. So this asks. One read-only control, no allocation, nothing changed.
//
// A zero, or a number far too small to build page tables from, confirms it and the fix is to give GSP-RM a heap. A
// large number refutes it, and then the ceiling is something else entirely and I would have spent days on the wrong
// thing.
#include "tinynv.h"
#include "gpu.h"
#include "internal.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  if (!getenv("TINYNV_HW") || !getenv("TINYNV_SOCKET")) {
    printf("usage: TINYNV_HW=1 TINYNV_SOCKET=<socket> %s\n", argv[0]);
    return 2;
  }
  tinynv_device_t d = NULL;
  if (tinynv_device_get(&d, 0) != TINYNV_OK) { printf("  no device: %s\n", tinynv_last_error()); return 1; }
  // Any real call boots the card; properties is the cheapest one that does.
  tinynv_device_props_t props;
  if (tinynv_device_props(d, &props) != TINYNV_OK) { printf("  cannot open the card: %s\n", tinynv_last_error()); return 1; }

  uint64_t freeheap = 0;
  if (tinynv_device_gsp_free_heap(d, &freeheap)) {
    // A refusal is also an answer, and a different one: it means the firmware will not even discuss its heap, which is
    // not the same as having none.
    printf("  gsp-rm refused the free-heap query: %s\n", tinynv_last_error());
    printf("  that is not the same as having no heap; it means this question cannot be asked this way.\n");
    return 1;
  }
  printf("  gsp-rm says it has %llu bytes of video memory free to allocate (%.2f MB)\n",
         (unsigned long long)freeheap, (double)freeheap / (1024.0 * 1024.0));
  // 2 MiB of virtual space needs one 4 KB page table. 4016 MiB, which is what libcuda asks for, needs 2008 of them at
  // the small page size - about 8 MB - plus the levels above. So the threshold worth naming is a few tens of megabytes.
  if (freeheap < (16ull << 20))
    printf("  too little to build page tables from: a 4016 MB reservation needs ~8 MB of leaf tables alone, so this\n"
           "  is consistent with the 2 MiB ceiling and the fix is to give gsp-rm a heap.\n");
  else
    printf("  that is enough to build page tables from, so the 2 MiB ceiling is NOT an empty heap and the diagnosis\n"
           "  was wrong. Look elsewhere before building anything.\n");
  return 0;
}
