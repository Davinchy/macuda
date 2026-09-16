// Does GSP-RM answer NV0041_CTRL_CMD_GET_SURFACE_PHYS_ATTR? 5a-ranges, and the one question the offline suite cannot
// reach.
//
// WHY IT MATTERS. To program page-table entries for a guest's allocation this driver needs the PHYSICAL pages behind
// it, and it does not have them: the guest's class-0x0040 allocation goes through our forwarding path, so GSP-RM
// holds the memory descriptor and we have never seen where it landed. NVIDIA's own UVM path reads pMemory->pMemDesc
// directly - in-kernel RM state that does not exist for us, because for us RM is firmware. The only route left is to
// ASK, with 0x410103: memOffset in, physical address out, contigSegmentSize saying how far it runs.
//
// WHY IT IS NOT SETTLED ON PAPER. ctrl0041.h says "This call is only currently supported in the MODS environment".
// In the open tree that is stale - flags 0x0, accessRight 0x0, compiled in, and memCtrlCmdGetSurfacePhysAttrLvm_IMPL
// is a straight call into the HAL with no gate. But the open tree is the CPU-side RM and what answers our forwarded
// control is CLOSED GSP firmware. "The control is ungated" is a fact about source we can read; "GSP-RM will answer
// it" is not. No recording contains the call, because nothing has ever made it.
//
// WHAT A NO MEANS. Everything in tinynv_c4b_map and tinynv_c4b_ranges still stands - they take ranges as a parameter
// precisely so that only the transport depends on this. But 5b's guest page-table tree would have nothing to put in
// it, so the shape of the whole C4b path changes and it changes BEFORE any of 5b is written. That is why this runs
// first in the window.
//
// THIS ALLOCATES ITS OWN OBJECT, so no guest, no daemon and no torch are involved. If the firmware answers for an
// object this driver allocated, it is the same control on the same class and the answer transfers. If it refuses,
// the next question is whether it refuses for everyone or only for us, and that one needs Session C's path.
//
// ONE CONFOUND NAMED BEFORE IT IS MEASURED: test_hw_heap.c's standing hypothesis is that GSP-RM has no video memory
// heap of its own. If that is right the ALLOCATION may fail for reasons having nothing to do with 0x410103, and a
// reader could take a heap problem for a control problem. The two failures are reported separately below and the
// allocation's is never described as an answer about the control.
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
  tinynv_device_props_t props;
  if (tinynv_device_props(d, &props) != TINYNV_OK) {
    printf("  cannot open the card: %s\n", tinynv_last_error());
    return 1;
  }

  // The shape Session C read out of the recording verbatim, not guessed: type 0 (NVOS32_TYPE_IMAGE), flags 0x1c101,
  // attr 0x18000000, hVASpace 0, and alignment EQUAL to size. Two of those - type and attr - I did not have and
  // would most likely have got wrong, and a wrong attr fails the allocation for a reason unrelated to this question.
  uint64_t paddr = 0, contig = 0;
  unsigned aperture = 0;
  int rc = tinynv_device_probe_phys_attr(d, 0x200000ull, 0, &paddr, &contig, &aperture);
  // In the form Session C's vm/trace/phys_walk.py reads, so the reply can be walked by their checker rather than by
  // my eye. Printed before any interpretation of it, and on failure too - the reply is the datum and the sentences
  // below are a reading of it.
  printf("C 00410103 probe %d in=%#llx out=%#llx contig=%#llx aperture=%u\n", rc, 0ull,
         (unsigned long long)paddr, (unsigned long long)contig, aperture);
  if (rc) {
    printf("  NO ANSWER: %s\n", tinynv_last_error());
    printf("  Read this carefully before concluding anything: if the sentence above is about the ALLOCATION, this\n"
           "  run says nothing at all about 0x410103 - see test_hw_heap.c, GSP-RM may have no heap to allocate from.\n"
           "  Only a failure of the CONTROL is an answer to the question this test asks.\n");
    return 1;
  }

  printf("  GSP-RM ANSWERED 0x410103: offset 0 is at physical %#llx, %llu bytes contiguous, aperture %u (%s)\n",
         (unsigned long long)paddr, (unsigned long long)contig, aperture,
         aperture == 0 ? "vidmem" : aperture == 1 ? "sysmem" : "neither vidmem nor sysmem, which is unexpected");

  // The answer has to be usable, not merely present. An unaligned base or a zero length would be an answer this
  // driver cannot build entries from, and that is a different result from a refusal - worse, because it looks like
  // success.
  int bad = 0;
  if (paddr & 0xfffull) {
    printf("  BUT the physical address is not page aligned, and entries store address >> 12 - the low bits would be\n"
           "  dropped silently and every page would point somewhere the guest was never granted.\n");
    bad++;
  }
  if (!contig) {
    printf("  BUT contigSegmentSize is ZERO, which consumes nothing: a walk that asks again at the same offset does\n"
           "  not terminate. tinynv_c4b_ranges refuses this rather than looping, so the walk is safe - but the\n"
           "  firmware cannot describe its own allocation and 5a-ranges has no usable answer.\n");
    bad++;
  }
  if (contig && contig < 0x200000ull)
    printf("  (the allocation is not contiguous: %llu bytes from offset 0, so a 2 MiB mapping needs %llu ranges and\n"
           "   the walk's multi-segment path is the one that matters rather than the single-range case)\n",
           (unsigned long long)contig, (unsigned long long)((0x200000ull + contig - 1) / contig));

  // Session C's question, one extra call: the 2 MiB this allocates and the 2 MiB ceiling test_hw_heap hypothesises
  // are the same number, and their recording has this allocation succeeding five times across four clients - always
  // at 2 MiB. If 4 MiB fails where 2 MiB worked, the ceiling is being met exactly rather than coincidentally, and
  // that is a fact about the heap worth having while the card is open.
  uint64_t p4 = 0, c4 = 0;
  unsigned a4 = 0;
  int rc4 = tinynv_device_probe_phys_attr(d, 0x400000ull, 0, &p4, &c4, &a4);
  printf("C 00410103 probe4m %d in=%#llx out=%#llx contig=%#llx aperture=%u\n", rc4, 0ull,
         (unsigned long long)p4, (unsigned long long)c4, a4);
  if (rc4)
    printf("  4 MiB did NOT work where 2 MiB did: %s\n"
           "  So the 2 MiB that succeeds is a CEILING being met exactly, not a size that happens to fit, and\n"
           "  test_hw_heap's hypothesis has a second piece of evidence. This says nothing about 0x410103.\n",
           tinynv_last_error());
  else
    printf("  4 MiB works too, so 2 MiB is not a ceiling for this kind of allocation and the 2 MiB in the recording\n"
           "  is what torch asked for rather than the most it could get.\n");

  // Is the multi-segment path in tinynv_c4b_ranges live code or dead code? Both sizes above came back FULLY
  // contiguous, so the walk took one range each and its segment-stitching has never met real firmware. A large
  // allocation is where a heap would have to fragment if it ever does. TINYNV_PROBE_BIG overrides the size.
  const char *big = getenv("TINYNV_PROBE_BIG");
  if (big) {
    uint64_t bs = strtoull(big, NULL, 0);
    uint64_t pb = 0, cb = 0;
    unsigned ab = 0;
    int rcb = tinynv_device_probe_phys_attr(d, bs, 0, &pb, &cb, &ab);
    printf("C 00410103 probebig %d in=%#llx out=%#llx contig=%#llx aperture=%u size=%#llx\n", rcb, 0ull,
           (unsigned long long)pb, (unsigned long long)cb, ab, (unsigned long long)bs);
    if (rcb)
      printf("  %#llx bytes could not be allocated: %s\n", (unsigned long long)bs, tinynv_last_error());
    else if (cb < bs)
      printf("  NOT CONTIGUOUS: %llu bytes from offset 0 of a %llu byte allocation, so a real mapping needs several\n"
             "  ranges and tinynv_c4b_ranges' segment stitching is live code that has now met real firmware.\n",
             (unsigned long long)cb, (unsigned long long)bs);
    else
      printf("  %#llx bytes came back contiguous in one piece as well. On this card, at these sizes, the walk always\n"
             "  takes one range - the multi-segment path is correct and unexercised, which is worth knowing before\n"
             "  trusting it.\n", (unsigned long long)bs);
  }

  if (!bad)
    printf("  5a-ranges is answered YES: the firmware will say where a class-0x0040 allocation lives, and\n"
           "  tinynv_c4b_ranges can be given a real transport. 5b's guest tree is worth writing.\n");
  return bad ? 1 : 0;
}
