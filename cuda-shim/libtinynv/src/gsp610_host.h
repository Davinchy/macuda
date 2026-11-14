// The 610 host-request executor (, docs/driver/gsp-610-runtime plan §4). The 610 firmware REMOVED the CPU
// register sequencer (event 4098, rpc_run_cpu_sequencer) and replaced it with two structured host requests:
// GSP_LOAD_EXEC_GENERIC_BOOTLOADER (4134) and GSP_LOAD_EXEC_HS_BINARY (4135). This module executes them on the GSP
// falcon, mirroring kgspLoadAndExecuteGenericBootloader_TU102 and kgspLoadAndExecuteHsBinary_GA102 from the pinned
// 610.43.03 source, using the same falcon primitives the 570 boot path uses (flcn.c).
//
// Unlike rpc610.c (pure codecs), this module TOUCHES THE DEVICE: it is the hardware-facing half of the sequencer
// replacement. It is exercised offline against a register-level fake card, which proves the control flow and the
// register sequence, never hardware completion (that is plan §5+). The 570 run_cpu_seq path is untouched.
#ifndef TINYNV_GSP610_HOST_H
#define TINYNV_GSP610_HOST_H
#include "gpu.h"
#include "rpc610.h"

// The SEC2 generic bootloader ucode the generic-bootloader host request loads into the GSP falcon's IMEM. It is a
// separate asset (g_bindata_ksec2GetBinArchiveBlUcode_TU102) from the boot chain; the caller loads it from the bundle
// and hands it here. start_tag / code_size are the RM_FLCN_BL_DESC fields (blStartTag, blImgHeader.blCodeSize).
typedef struct {
  const uint8_t *image;
  size_t image_len;
  uint32_t start_tag, code_size;
} tinynv_gsp610_bl_ucode_t;

// Execute one heavy-secure binary on the GSP falcon (the HS-binary host request). `p` is the validated payload.
// Returns 0, or -1 through tinynv_fail with the first error preserved.
int tinynv_gsp610_exec_hs(tinynv_gpu_t *g, const tinynv_rpc610_load_exec_hs_t *p);

// Execute the generic bootloader on the GSP falcon (the generic-bootloader host request). `p` is the validated
// payload; `bl` is the SEC2 generic bootloader ucode to load. Returns 0, or -1 through tinynv_fail.
int tinynv_gsp610_exec_bl(tinynv_gpu_t *g, const tinynv_rpc610_load_exec_bl_t *p, const tinynv_gsp610_bl_ucode_t *bl);

// Dispatch a status-queue message that may be a 610 host request. Returns:
//   0  - a load-exec host request was parsed and executed successfully;
//   1  - not a host request (init-done / log / fault / rpc reply): the caller handles it as usual;
//  -1  - a host request was malformed, or the event is the removed sequencer / unknown: tinynv_fail has the reason.
// `bl` may be NULL only if no generic-bootloader request arrives; a BL request with no ucode staged is a clear error.
int tinynv_gsp610_host_request(tinynv_gpu_t *g, uint32_t function, const uint8_t *msg, size_t len,
                               const tinynv_gsp610_bl_ucode_t *bl);

#endif