// The 610.43.03 boot-metadata adapters (, docs/driver/gsp-610-runtime plan §4, design "ABI boundary").
//
// Everything the 610 boot path builds or parses that DIFFERS from 570, as pure functions: no device, no GPU, no
// allocation. Each takes explicit inputs and returns 0 / -1 through tinynv_fail, so the offline suite drives it with
// synthetic fixtures and the real hash-pinned assets alike. The 570 path is untouched; these are only reached through
// the 610 profile, which this build still refuses at selection (the RPC/lifecycle integration is ).
//
// What changed for 610 and is handled here (matrix, re-verified against the pinned source):
//   * the booter/scrubber container is BTL1 v1, NOT the 570 NVFW image, and the signature is chosen by fuse at run
//     time (index = num_sigs - 1 - fuse_ver), not by a fixed patch_sig selector;
//   * the RISC-V bootloader is a raw image plus a SEPARATE 84-byte v5 descriptor, not an NVFW container;
//   * MESSAGE_QUEUE_INIT_ARGUMENTS grew 32 -> 64 B, GSP_ARGUMENTS_CACHED 72 -> 152 B, GSP_FMC_BOOT_PARAMS 80 -> 88 B;
//   * the .fwimage grew to 80.32 MiB, so the reservation is re-sized (fw_layout_610.h) and the vbios carveout is
//     re-derived and bounds-checked here.
// The WPR metadata block itself, the libos regions and the queue HEADER are byte-identical to 570 and are shared
// (nv_structs.h); test_headers_610 asserts that against the pinned 610 tree rather than assuming it.
#ifndef TINYNV_BOOT610_H
#define TINYNV_BOOT610_H
#include "nv_structs.h"
#include <stddef.h>
#include <stdint.h>

// --- the BTL1 v1 container (booter_load / booter_unload / scrubber) ----------------------------------------------
#define TINYNV_BTL1_MAGIC    0x314c5442u // 'BTL1' little-endian; NOT the 570 NVFW magic 0x000010de
#define TINYNV_BTL1_VERSION  1u
#define TINYNV_BTL1_SIG_SIZE 384u        // one RSA3K production signature
#define TINYNV_BTL1_PREFIX   72u         // magic+version (8) + 14 header words (56) + image_size+sig_bytes (8)

// The 14 header words, in wire order, shared by the BTL1 extractor and packer.
typedef struct {
  uint32_t os_code_off, os_code_size, os_data_off, os_data_size;
  uint32_t num_apps, app_code_off, app_code_size, app_data_off, app_data_size;
  uint32_t patch_loc, ucode_id, engine_id, num_sigs, sig_bytes;
} tinynv_btl1_hdr_t;

// A parsed, bound-checked BTL1 container. `image_off`/`sig_off` point into the caller's container bytes.
typedef struct {
  tinynv_btl1_hdr_t hdr;
  uint32_t image_size, sig_bytes;
  uint32_t sig_dmem_off; // patch_loc - os_data_off: where the boot ROM finds the signature in DMEM
  size_t image_off, sig_off;
} tinynv_btl1_t;

// The pinned AD102 geometry for one logical asset ("booter_load", "booter_unload", "scrubber"), or NULL if the name is
// not one of them. These are the BTL1_GEOMETRY values, from the hash-verified bindata archives.
const tinynv_btl1_hdr_t *tinynv_boot610_geom(const char *which, uint32_t *image_size_out);

// Parse and fully bound-check a BTL1 v1 container. When `expect` is non-NULL the geometry must match it exactly
// (wrong-chip / wrong-image defence). Refuses the 570 NVFW magic explicitly, so a renamed 570 file cannot pass.
int tinynv_boot610_btl1_parse(const uint8_t *data, size_t len, const tinynv_btl1_hdr_t *expect,
                              uint32_t expect_image_size, tinynv_btl1_t *out);

// Copy the image out of a parsed container and patch the production signature selected by the SEC2 fuse version.
// `image_out` receives a malloc'd buffer of image_size bytes the caller frees. fuse_ver must be < num_sigs.
int tinynv_boot610_btl1_image(const tinynv_btl1_t *b, const uint8_t *data, uint32_t fuse_ver,
                              uint8_t **image_out, size_t *len_out);

// The SEC2 ucode fuse register value -> the fuse version used for the signature index: highest
// set bit of the low 16 bits, plus one; zero when no bit is set.
uint32_t tinynv_boot610_fuse_version(uint32_t raw);

// --- the RISC-V boot descriptor (84-byte v5 wire form) -----------------------------------------------------------
// The 610 header struct RM_RISCV_UCODE_DESC grew to 92 B (bIsSmp/bIsPlicEnabled appended) but the pinned AD102 archive
// ships the 84-byte v5 wire form; the offsets the boot path reads are unchanged. This parses the WIRE form, not the
// 92-byte struct, and bounds every range against the raw bootloader image.
typedef struct {
  uint32_t manifest_offset, manifest_size;
  uint32_t data_offset, data_size;
  uint32_t code_offset, code_size;
  uint32_t image_size;
} tinynv_boot610_desc_t;
#define TINYNV_BOOT610_DESC_SIZE 84u
int tinynv_boot610_desc_parse(const uint8_t *desc, size_t len, uint32_t image_size, tinynv_boot610_desc_t *out);

// --- the 610 startup argument layouts ----------------------------------------------------------------------------
// MESSAGE_QUEUE_INIT_ARGUMENTS, 64 B (570: 32 B). The five new queue-geometry fields are filled from the 610
// message_queue_cpu.c defaults; the four fields 570 already fills keep their offsets.
typedef struct {
  uint64_t sharedMemPhysAddr;
  uint32_t pageTableEntryCount, _pad;
  uint64_t cmdQueueOffset, statQueueOffset;
  uint64_t queueElementHdrSize, queueElementSizeMin, queueElementSizeMax;
  uint32_t queueHeaderAlign, queueElementAlign;
} tinynv_boot610_msgq_args_t;
TINYNV_SIZE(tinynv_boot610_msgq_args_t, 64);
TINYNV_AT(tinynv_boot610_msgq_args_t, cmdQueueOffset, 16);
TINYNV_AT(tinynv_boot610_msgq_args_t, queueElementHdrSize, 32);
TINYNV_AT(tinynv_boot610_msgq_args_t, queueHeaderAlign, 56);

// The queue-geometry defaults the 610 host driver writes (message_queue_cpu.c, no Confidential Compute): the element
// header is the 16-byte MCTP/NVDM/checkSum/seqNum prefix, elements are page-sized up to 16 pages, header align 4,
// element align the 4 KiB page shift. Pinned here and asserted against the 610 source in test_headers_610.
#define TINYNV_BOOT610_ELEM_HDR_SIZE 16u
#define TINYNV_BOOT610_ELEM_SIZE_MIN 4096u
#define TINYNV_BOOT610_ELEM_SIZE_MAX (4096u * 16u)
#define TINYNV_BOOT610_HDR_ALIGN     4u
#define TINYNV_BOOT610_ELEM_ALIGN    12u

// GSP_ARGUMENTS_CACHED, 152 B (570: 72 B). Grows with the msgq args and appends three arg blocks; the fields the boot
// path fills keep their relative placement. gpuInstance moves 44 -> 76, bDmemStack 48 -> 80, profilerArgs 56 -> 88.
typedef struct {
  tinynv_boot610_msgq_args_t messageQueueInitArguments;
  uint8_t srInitArguments[12];
  uint32_t gpuInstance;
  uint8_t bDmemStack, _pad[7];
  uint64_t profilerArgs[2];
  uint64_t sysmemHeapArgs[2];
  uint64_t rmStateMonitorBufferArgs[2];
  uint64_t bindataArgs[2];
} tinynv_boot610_args_cached_t;
TINYNV_SIZE(tinynv_boot610_args_cached_t, 152);
TINYNV_AT(tinynv_boot610_args_cached_t, gpuInstance, 76);
TINYNV_AT(tinynv_boot610_args_cached_t, bDmemStack, 80);
TINYNV_AT(tinynv_boot610_args_cached_t, profilerArgs, 88);
TINYNV_AT(tinynv_boot610_args_cached_t, bindataArgs, 136);

// GSP_FMC_BOOT_PARAMS, 88 B (570: 80 B). The tail gains GSP_RM_MEM_PARAMS (8 B); the three blocks the boot path fills
// keep their offsets. The ACR params gain bInstInSysMode in what was padding (size unchanged, stays 0 here).
typedef struct {
  uint32_t flushSysmemAddrValLo, flushSysmemAddrValHi; // GSP_RM_MEM_PARAMS
} tinynv_boot610_rm_mem_params_t;
typedef struct {
  tinynv_gsp_fmc_init_params_t initParams;
  uint32_t _pad;
  tinynv_gsp_acr_boot_params_t bootGspRmParams;
  tinynv_gsp_rm_params_t gspRmParams;
  tinynv_gsp_spdm_params_t gspSpdmParams;
  tinynv_boot610_rm_mem_params_t gspRmMemParams;
} tinynv_boot610_fmc_boot_params_t;
TINYNV_SIZE(tinynv_boot610_fmc_boot_params_t, 88);
TINYNV_AT(tinynv_boot610_fmc_boot_params_t, bootGspRmParams, 8);
TINYNV_AT(tinynv_boot610_fmc_boot_params_t, gspRmParams, 40);
TINYNV_AT(tinynv_boot610_fmc_boot_params_t, gspSpdmParams, 56);
TINYNV_AT(tinynv_boot610_fmc_boot_params_t, gspRmMemParams, 80);

// Fill the 610 cached args from the same logical inputs the 570 init_rm_args uses, plus the queue geometry. The buffer
// is zeroed first so every field the boot path does not set (the new arg blocks, srInitArguments) is explicitly zero.
void tinynv_boot610_fill_args_cached(tinynv_boot610_args_cached_t *a, uint64_t shared_mem_pa, uint32_t pte_count,
                                     uint64_t cmd_off, uint64_t stat_off);

// --- the 610 vbios carveout --------------------------------------------------------------------------------------
// The top-down layout gsp.c init_wpr_meta builds, with the 610 sizes (fw_layout_610.h) and overflow-checked arithmetic.
// Pure: given the usable FB and the placed image/boot sizes, it computes every boundary and refuses a layout that does
// not fit, rather than wrapping. The WPR meta block is then filled from this by tinynv_boot610_fill_wpr_meta.
typedef struct {
  uint64_t fb_size;
  uint64_t vga_offset, vga_size;
  uint64_t frts_offset, frts_size;
  uint64_t boot_offset;
  uint64_t gsp_offset;
  uint64_t heap_offset, heap_size;
  uint64_t wpr_start, wpr_end;
  uint64_t nonwpr_offset, nonwpr_size;
} tinynv_boot610_layout_t;
int tinynv_boot610_vbios_layout(uint64_t vram, uint64_t image_size, uint64_t boot_size, tinynv_boot610_layout_t *out);

// Fill the shared WPR metadata block (byte-identical 570/610) from a computed layout and the placed addresses. The
// radix3/bootloader/signature addresses are the GPU-reachable physical addresses the boot path allocated.
void tinynv_boot610_fill_wpr_meta(tinynv_wpr_meta_t *m, const tinynv_boot610_layout_t *l, uint64_t radix3_pa,
                                  uint64_t image_size, uint64_t bootloader_pa, uint64_t bootloader_size,
                                  const tinynv_boot610_desc_t *desc, uint64_t signature_pa);

#endif
