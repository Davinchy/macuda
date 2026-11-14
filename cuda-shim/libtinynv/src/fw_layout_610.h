// What the 610.43.03 firmware owns at the top of video memory on the owner's RTX 4090 (AD102, vbios path).
//
// The 610 sibling of fw_layout.h. Every number here is derived from the MATCHING pinned 610.43.03 definition or the
// hash-pinned asset, not copied from 570 and not taken from the audit summary where the two disagree
// (see the non-WPR heap note below). The 570 file is untouched; this one is only reachable through the 610 profile.
//
// The layout is the same top-down vbios carveout gsp.c init_wpr_meta builds for 570: VGA workspace, the FRTS region
// FWSEC places, the boot binary, the GSP-RM image, the firmware heap, the LSB/meta page, and the non-WPR heap below
// wpr2. What changed for 610 is the IMAGE: .fwimage grew from 63,541,248 to 84,226,048 bytes, past the 570 64 MiB
// load bound, so the image bound and the reservation arithmetic are re-derived here. The heap formula and the AD102
// non-WPR heap are unchanged (verified against the pinned source, below).
#ifndef TINYNV_FW_LAYOUT_610_H
#define TINYNV_FW_LAYOUT_610_H

// The pinned GSP-RM .fwimage, from the bundle manifest (gsp-610.43.03.bin, sha256 572373620a…). This is the
// exact byte count the ELF section reader returns; the load bound below must cover it.
#define TINYNV_FW610_IMAGE_SIZE    84226048ull
// Load-time image bound. 570 uses 64 MiB for a 60.6 MiB image; 610's 80.32 MiB image needs a larger bound. 96 MiB
// covers the pinned image with ~16 MiB of headroom, the same "round up to a clean boundary" philosophy as 570. A
// bigger firmware cannot outgrow the reservation without the load (gsp.c) or the static assert below saying so.
#define TINYNV_FW610_IMAGE_BOUND   (96ull << 20)
// The pinned RISC-V bootloader image (bootloader-610.43.03.bin, 36,864 bytes) and its bound. Well under 1 MiB, so the
// 570 bootbin bound is kept; it is re-asserted against the pinned size rather than assumed.
#define TINYNV_FW610_BOOTBIN_SIZE  36864ull
#define TINYNV_FW610_BOOTBIN_BOUND (1ull << 20)
// The firmware heap the carveout check reasons about. The 610 _kgspCalculateFwHeapSize formula is unchanged for a
// 32 GB card (kernel_gsp.c: OS 22 MiB + base RM 14 MiB + 96 KiB/GB x 32 + 48 KiB x 2048 channels = 135 MiB), so this
// equals the 570 value. It is re-derived from the 610 gsp_fw_heap.h constants in test_headers_610, not copied.
#define TINYNV_FW610_HEAP_SIZE     0x8700000ull
// The heap the vbios path actually places (gsp.c init_wpr_meta). 570 uses 0x8100000 here; the formula is unchanged so
// 610 keeps it. Distinct from the carveout-check term above, exactly as in 570.
#define TINYNV_FW610_VBIOS_HEAP    0x8100000ull
// The AD102 non-WPR heap, from the MATCHING 610 HAL. g_kernel_gsp_nvoc.c dispatches AD102 (with TU10x/GA10x/AD10x) to
// kgspGetNonWprHeapSize_ed6b8b, which returns 1048576 (1 MiB) - the SAME value 570 dispatches AD102 to.
// The 2 MiB HAL (d505ea) serves GH100/GB100/GB10x, not Ada. The vbios path places 0x100000 for AD102;
// 610 keeps the matching 1 MiB. test_headers_610 pins the HAL value.
#define TINYNV_FW610_NONWPR_HEAP   0x100000ull
// VGA workspace and FRTS are board/FWSEC quantities, runtime-independent (vbios.c), unchanged for 610.
#define TINYNV_FW610_VGA_SIZE      0x100000ull
#define TINYNV_FW610_FRTS_SIZE     0x100000ull
// Held back from the top of video memory. The vbios carveout estimate below must fit inside it. 256 MiB still covers
// the larger 610 image (the estimate is ~238 MiB), so the hold-back - and therefore the page-table reservation and
// every low address - is unchanged from 570. This is CHECKED, not assumed: the static assertion is the proof.
#define TINYNV_FW610_RESERVE_TOP   (256ull << 20)

// The worst-case depth of the vbios carveout from the top of video memory, at the bounds above: VGA + FRTS + bootbin
// + image + heap + one meta page (rounded to 1 MiB) + non-WPR heap, plus 8 MiB for the round-downs the layout applies
// (gsp to 64 KiB, heap/wpr/non-WPR to 1 MiB). Mirrors the 570 estimate's structure but with the 610 image bound and
// the AD102 non-WPR heap. If the firmware outgrows this, the assert fires and the reservation must be raised with it.
#define TINYNV_FW610_CARVEOUT_ESTIMATE                                                                                   \
  (TINYNV_FW610_VGA_SIZE + TINYNV_FW610_FRTS_SIZE + TINYNV_FW610_BOOTBIN_BOUND + TINYNV_FW610_IMAGE_BOUND +            \
   TINYNV_FW610_VBIOS_HEAP + (1ull << 20) + TINYNV_FW610_NONWPR_HEAP + (8ull << 20))

_Static_assert(TINYNV_FW610_IMAGE_SIZE <= TINYNV_FW610_IMAGE_BOUND,
               "the pinned 610 .fwimage outgrew its load bound: raise TINYNV_FW610_IMAGE_BOUND and the reservation");
_Static_assert(TINYNV_FW610_BOOTBIN_SIZE <= TINYNV_FW610_BOOTBIN_BOUND,
               "the pinned 610 bootloader outgrew its load bound");
_Static_assert(TINYNV_FW610_CARVEOUT_ESTIMATE <= TINYNV_FW610_RESERVE_TOP,
               "the 610 sizes handed to the firmware outgrew the reservation: raise TINYNV_FW610_RESERVE_TOP with them");

#endif
