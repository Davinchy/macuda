// What the firmware owns at the top of video memory, written down once.
//
// On the chain-of-trust path this driver hands the firmware SIZES - the WPR metadata in gsp.c, the FRTS placement in
// flcn.c - and the GSP-FMC/ACR places everything at the top of video memory itself, patching the offsets back into the
// meta. So the bytes the memory manager must never hand out are a consequence of these numbers, and until 2026-09-19
// the manager held back a flat 64 MB that was related to none of them (README "Known limits", second bullet;
// test/test_hw_wpr.c is the probe that first asked the chip). The layout, top down, as the vendor's own
// kernel_gsp_gh100.c lays it out and gsp_fw_wpr_meta.h draws it:
//
//     VGA workspace, PMU reservation, alignment          TINYNV_FW_FRTS_FROM_END covers all of it
//     ---- wpr2 end ----
//     FRTS, boot bin, GSP-RM ELF, the firmware heap,     write-protected: the FB MMU drops a non-secure write and
//     the LSB header and the meta                        blocks the read, so a buffer here "works" and reads back zeros
//     ---- wpr2 start ----
//     the firmware's non-WPR heap                        NOT protected: a buffer here corrupts GSP-RM's own heap and the
//                                                        failure is a garbled RPC or a firmware crash some time later
//     ---- what the manager may hand out ends here ----
//
// Every size is the vendor's own value for this card; test/test_headers.c pins the two that come from formulas to the
// headers they come from, and test/test_mm.c pins the manager's top to the reservation. tinynv.c reads the wpr2
// registers once the firmware has booted and refuses the open if the manager's top is above the region - the chip's
// word, not this arithmetic, is what a run actually trusts; this file only sizes the hold-back so that the check passes
// with room to spare.
#ifndef TINYNV_FW_LAYOUT_H
#define TINYNV_FW_LAYOUT_H

#define TINYNV_FW_VGA_WORKSPACE  0x20000ull     // 128 KB, kernel_gsp_gh100.c
#define TINYNV_FW_PMU_RESERVED   0x1820000ull   // 24.1 MB
#define TINYNV_FW_NONWPR_HEAP    0x220000ull    // 2.125 MB, the GB20x value, BELOW wpr2 and not write-protected
#define TINYNV_FW_HEAP_SIZE      0x8700000ull   // 135 MB: OS 22 + base RM 14 + 96 KB per GB x 32 + 48 KB x 2048 channels
#define TINYNV_FW_FRTS_SIZE      0x100000ull    // 1 MB, kernel_gsp_frts_tu102.c
#define TINYNV_FW_FRTS_FROM_END  0x1c00000ull   // 28 MB: ALIGN_UP(non-wpr heap + 4 KB + PMU reservation, 2 MB), kern_fsp_gh100.c
// Bounds the loaded images are checked against at load time, so a bigger firmware cannot outgrow the estimate below
// without the build or the load saying so. gsp-570.144.bin's .fwimage is 63,541,248 bytes; its bootloader's data is
// 200,704.
#define TINYNV_FW_IMAGE_BOUND    (64ull << 20)
#define TINYNV_FW_BOOTBIN_BOUND  (1ull << 20)
// What the firmware could own from the top, from the sizes above: FRTS and everything over it, the boot bin and the
// image at their bounds, the heap, a megabyte for the LSB header and the meta, the non-WPR heap below, and eight
// megabytes for the alignment the ACR applies (the heap absorbs the padding, kernel_gsp_gh100.c). About 240 MB.
#define TINYNV_FW_CARVEOUT_ESTIMATE                                                                                \
  (TINYNV_FW_FRTS_FROM_END + TINYNV_FW_FRTS_SIZE + TINYNV_FW_BOOTBIN_BOUND + TINYNV_FW_IMAGE_BOUND +               \
   TINYNV_FW_HEAP_SIZE + (1ull << 20) + TINYNV_FW_NONWPR_HEAP + (8ull << 20))
// Held back from the top of video memory, with room to spare. 256 MB is also the vendor's own "the firmware scrubs the
// last 256 MB of FB; no memory outside of this region may be used until it has scrubbed the rest" figure
// (gsp_fw_wpr_meta.h). On a 32,607 MB card any hold-back under 351 MB leaves the page-table reservation at the same
// 64 MB, so every address the recorded boot depends on is unchanged; test_mm and test_dev assert that.
#define TINYNV_FW_RESERVE_TOP    (256ull << 20)
_Static_assert(TINYNV_FW_CARVEOUT_ESTIMATE <= TINYNV_FW_RESERVE_TOP,
               "the sizes handed to the firmware outgrew the reservation: raise TINYNV_FW_RESERVE_TOP with them");

#endif
