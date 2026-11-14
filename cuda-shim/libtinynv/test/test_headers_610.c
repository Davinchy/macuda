// Holds the driver's 610 boot-metadata declarations against NVIDIA's pinned 610.43.03 tree.
//
// The 610 sibling of test_headers.c, compiled as its OWN translation unit against the 610 header tree so the two
// releases' same-named headers never collide (design "ABI boundary": a separate assertion TU per pinned release).
// test_headers.c stays the 570 check and is untouched. This file asserts two things:
//   1. the structures the audit classifies SAME are byte-identical in 610, so the shared nv_structs.h code is safe;
//   2. the structures that CHANGED match the new boot610.h layouts exactly (size and every offset the boot path uses).
// It also re-derives the 610 reservation terms from the matching 610 definitions (heap formula, AD102 non-WPR HAL).
//
// This is a compile-time check: if a field moved, the build breaks here rather than the firmware reading garbage.
#include <stddef.h>
#include <stdio.h>
#include "nvtypes.h"
#include "nvgputypes.h"
#include "nvmisc.h"   // REF_NUM / NV_OFFSETOF, for re-deriving the MCTP/NVDM header constants
#include "gsp/gsp_fw_wpr_meta.h"
#include "gsp/gspifpub.h"
#include "kernel/gpu/fsp/kern_fsp_cot_payload.h"
#if defined(__has_include)
#  if __has_include("fsp/fsp_nvdm_format.h")
#    include "fsp/fsp_nvdm_format.h"
#  elif __has_include("nvdm_format.h")
#    include "nvdm_format.h"   // 610 folded fsp/fsp_nvdm_format.h into nvdm_format.h
#  endif
#else
#  include "fsp/fsp_nvdm_format.h"
#endif
#include "msgq/msgq_priv.h"
#include "kernel/gpu/gsp/gsp_init_args.h"
#include "libos_init_args.h"
#include "rmRiscvUcode.h"
#include "kernel/gpu/gsp/gsp_fw_heap.h"
#include "nv_structs.h"
#include "boot610.h"
#include "fw_layout_610.h"
// the RPC / resource-management boundary. The RPC structures are gated behind RPC_STRUCTURES; the message
// header lives in its own generated file. The GspSystemInfo / load-exec payloads are behind the NVOC port layer, so
// they are asserted against the VERBATIM-extracted pinned definitions (tools/gsp610-runtime/extract_structs.py).
#define RPC_STRUCTURES
#define RPC_GENERIC_UNION   // rpc_generic_union, which the message header's flexible payload array names
#include "g_rpc-structures.h"
#define RPC_MESSAGE_STRUCTURES
#include "g_rpc-message-header.h"
#include "nvos.h"   // NV_VASPACE_ALLOCATION_PARAMETERS
#include "kernel/vgpu/rpc_headers.h"
#include "kernel/vgpu/rpc_global_enums.h"
#include "class/cl00de.h"
#include "alloc/alloc_channel.h"
#include "class/cl90f1.h"
#include "ctrl/ctrla06c.h"
#include "ctrl/ctrl2080/ctrl2080internal.h"
#include "ctrl/ctrl0080/ctrl0080gr.h"
#include "mctp_format.h"
#if defined(__has_include)
#  if __has_include("nvdm_format.h")
#    include "nvdm_format.h"
#  endif
#endif
#include "swref/published/ampere/ga102/dev_falcon_v4.h"
// tu102's fbif header names all three aperture targets (ga102's omits NONCOHERENT); the values are chip-independent.
#include "swref/published/turing/tu102/dev_fbif_v4.h"
#include "extracted_610.h"
#include "rpc610.h"

#define SAME_SIZE(mine, theirs) _Static_assert(sizeof(mine) == sizeof(theirs), #mine " is not the size of " #theirs)
#define SAME_AT(mine, mf, theirs, tf) \
  _Static_assert(offsetof(mine, mf) == offsetof(theirs, tf), #mine "." #mf " is not where " #theirs "." #tf " is")

// --- 1. the SAME structures: shared code is safe only if 610 really kept them ------------------------------------
// The WPR metadata block is the heart of the boot path; the audit probed all 30 offsets and found them identical.
SAME_SIZE(tinynv_wpr_meta_t, GspFwWprMeta);
SAME_AT(tinynv_wpr_meta_t, magic, GspFwWprMeta, magic);
SAME_AT(tinynv_wpr_meta_t, revision, GspFwWprMeta, revision);
SAME_AT(tinynv_wpr_meta_t, sysmemAddrOfRadix3Elf, GspFwWprMeta, sysmemAddrOfRadix3Elf);
SAME_AT(tinynv_wpr_meta_t, sizeOfRadix3Elf, GspFwWprMeta, sizeOfRadix3Elf);
SAME_AT(tinynv_wpr_meta_t, sysmemAddrOfBootloader, GspFwWprMeta, sysmemAddrOfBootloader);
SAME_AT(tinynv_wpr_meta_t, sizeOfBootloader, GspFwWprMeta, sizeOfBootloader);
SAME_AT(tinynv_wpr_meta_t, bootloaderCodeOffset, GspFwWprMeta, bootloaderCodeOffset);
SAME_AT(tinynv_wpr_meta_t, bootloaderDataOffset, GspFwWprMeta, bootloaderDataOffset);
SAME_AT(tinynv_wpr_meta_t, bootloaderManifestOffset, GspFwWprMeta, bootloaderManifestOffset);
SAME_AT(tinynv_wpr_meta_t, sysmemAddrOfSignature, GspFwWprMeta, sysmemAddrOfSignature);
SAME_AT(tinynv_wpr_meta_t, sizeOfSignature, GspFwWprMeta, sizeOfSignature);
SAME_AT(tinynv_wpr_meta_t, gspFwRsvdStart, GspFwWprMeta, gspFwRsvdStart);
SAME_AT(tinynv_wpr_meta_t, nonWprHeapOffset, GspFwWprMeta, nonWprHeapOffset);
SAME_AT(tinynv_wpr_meta_t, nonWprHeapSize, GspFwWprMeta, nonWprHeapSize);
SAME_AT(tinynv_wpr_meta_t, gspFwWprStart, GspFwWprMeta, gspFwWprStart);
SAME_AT(tinynv_wpr_meta_t, gspFwHeapOffset, GspFwWprMeta, gspFwHeapOffset);
SAME_AT(tinynv_wpr_meta_t, gspFwHeapSize, GspFwWprMeta, gspFwHeapSize);
SAME_AT(tinynv_wpr_meta_t, gspFwOffset, GspFwWprMeta, gspFwOffset);
SAME_AT(tinynv_wpr_meta_t, bootBinOffset, GspFwWprMeta, bootBinOffset);
SAME_AT(tinynv_wpr_meta_t, frtsOffset, GspFwWprMeta, frtsOffset);
SAME_AT(tinynv_wpr_meta_t, frtsSize, GspFwWprMeta, frtsSize);
SAME_AT(tinynv_wpr_meta_t, gspFwWprEnd, GspFwWprMeta, gspFwWprEnd);
SAME_AT(tinynv_wpr_meta_t, fbSize, GspFwWprMeta, fbSize);
SAME_AT(tinynv_wpr_meta_t, vgaWorkspaceOffset, GspFwWprMeta, vgaWorkspaceOffset);
SAME_AT(tinynv_wpr_meta_t, vgaWorkspaceSize, GspFwWprMeta, vgaWorkspaceSize);
SAME_AT(tinynv_wpr_meta_t, pmuReservedSize, GspFwWprMeta, pmuReservedSize);
SAME_AT(tinynv_wpr_meta_t, verified, GspFwWprMeta, verified);
_Static_assert(GSP_FW_WPR_META_MAGIC == TINYNV_GSP_FW_WPR_META_MAGIC, "the 610 wpr metadata magic changed");
_Static_assert(GSP_FW_WPR_META_REVISION == TINYNV_GSP_FW_WPR_META_REVISION, "the 610 wpr metadata revision changed");

// The chain-of-trust payload and the libos regions are unchanged (the AD102 vbios path does not use COT, but the
// shared declarations must still hold so nothing drifts).
SAME_SIZE(tinynv_cot_payload_t, NVDM_PAYLOAD_COT);
SAME_AT(tinynv_cot_payload_t, gspFmcSysmemOffset, NVDM_PAYLOAD_COT, gspFmcSysmemOffset);
SAME_AT(tinynv_cot_payload_t, frtsVidmemOffset, NVDM_PAYLOAD_COT, frtsVidmemOffset);
SAME_AT(tinynv_cot_payload_t, gspBootArgsSysmemOffset, NVDM_PAYLOAD_COT, gspBootArgsSysmemOffset);
_Static_assert(NVDM_TYPE_COT == TINYNV_NVDM_TYPE_COT, "the 610 chain of trust message type changed");
SAME_SIZE(tinynv_libos_region_t, LibosMemoryRegionInitArgument);
SAME_AT(tinynv_libos_region_t, id8, LibosMemoryRegionInitArgument, id8);
SAME_AT(tinynv_libos_region_t, pa, LibosMemoryRegionInitArgument, pa);
SAME_AT(tinynv_libos_region_t, size, LibosMemoryRegionInitArgument, size);
SAME_AT(tinynv_libos_region_t, kind, LibosMemoryRegionInitArgument, kind);
SAME_AT(tinynv_libos_region_t, loc, LibosMemoryRegionInitArgument, loc);
_Static_assert(LIBOS_MEMORY_REGION_CONTIGUOUS == TINYNV_LIBOS_MEMORY_REGION_CONTIGUOUS, "610 libos region kinds changed");
_Static_assert(LIBOS_MEMORY_REGION_LOC_SYSMEM == TINYNV_LIBOS_MEMORY_REGION_LOC_SYSMEM, "610 libos region locations changed");
_Static_assert(LIBOS_MEMORY_REGION_RADIX_PAGE_LOG2 == TINYNV_LIBOS_MEMORY_REGION_RADIX_PAGE_LOG2, "610 radix page size changed");

// The queue HEADER is stable even though the element framing changed (that is ).
SAME_SIZE(tinynv_msgq_tx_header_t, msgqTxHeader);
SAME_AT(tinynv_msgq_tx_header_t, size, msgqTxHeader, size);
SAME_AT(tinynv_msgq_tx_header_t, msgSize, msgqTxHeader, msgSize);
SAME_AT(tinynv_msgq_tx_header_t, msgCount, msgqTxHeader, msgCount);
SAME_AT(tinynv_msgq_tx_header_t, writePtr, msgqTxHeader, writePtr);
SAME_AT(tinynv_msgq_tx_header_t, flags, msgqTxHeader, flags);
SAME_AT(tinynv_msgq_tx_header_t, rxHdrOff, msgqTxHeader, rxHdrOff);
SAME_AT(tinynv_msgq_tx_header_t, entryOff, msgqTxHeader, entryOff);

// The ACR/RM/SPDM boot blocks keep their size and used offsets; bInstInSysMode lands in what was padding.
SAME_SIZE(tinynv_gsp_acr_boot_params_t, GSP_ACR_BOOT_GSP_RM_PARAMS);
SAME_AT(tinynv_gsp_acr_boot_params_t, target, GSP_ACR_BOOT_GSP_RM_PARAMS, target);
SAME_AT(tinynv_gsp_acr_boot_params_t, gspRmDescSize, GSP_ACR_BOOT_GSP_RM_PARAMS, gspRmDescSize);
SAME_AT(tinynv_gsp_acr_boot_params_t, gspRmDescOffset, GSP_ACR_BOOT_GSP_RM_PARAMS, gspRmDescOffset);
SAME_AT(tinynv_gsp_acr_boot_params_t, bIsGspRmBoot, GSP_ACR_BOOT_GSP_RM_PARAMS, bIsGspRmBoot);
SAME_SIZE(tinynv_gsp_rm_params_t, GSP_RM_PARAMS);
SAME_AT(tinynv_gsp_rm_params_t, bootArgsOffset, GSP_RM_PARAMS, bootArgsOffset);
SAME_SIZE(tinynv_gsp_spdm_params_t, GSP_SPDM_PARAMS);
_Static_assert(GSP_DMA_TARGET_COHERENT_SYSTEM == TINYNV_GSP_DMA_TARGET_COHERENT_SYSTEM, "610 dma target numbering changed");

// --- 2. the CHANGED structures: the new boot610 layouts must match 610 exactly -----------------------------------
// MESSAGE_QUEUE_INIT_ARGUMENTS grew 32 -> 64 B; the four fields 570 fills keep their offsets, five are appended.
SAME_SIZE(tinynv_boot610_msgq_args_t, MESSAGE_QUEUE_INIT_ARGUMENTS);
SAME_AT(tinynv_boot610_msgq_args_t, sharedMemPhysAddr, MESSAGE_QUEUE_INIT_ARGUMENTS, sharedMemPhysAddr);
SAME_AT(tinynv_boot610_msgq_args_t, pageTableEntryCount, MESSAGE_QUEUE_INIT_ARGUMENTS, pageTableEntryCount);
SAME_AT(tinynv_boot610_msgq_args_t, cmdQueueOffset, MESSAGE_QUEUE_INIT_ARGUMENTS, cmdQueueOffset);
SAME_AT(tinynv_boot610_msgq_args_t, statQueueOffset, MESSAGE_QUEUE_INIT_ARGUMENTS, statQueueOffset);
SAME_AT(tinynv_boot610_msgq_args_t, queueElementHdrSize, MESSAGE_QUEUE_INIT_ARGUMENTS, queueElementHdrSize);
SAME_AT(tinynv_boot610_msgq_args_t, queueElementSizeMin, MESSAGE_QUEUE_INIT_ARGUMENTS, queueElementSizeMin);
SAME_AT(tinynv_boot610_msgq_args_t, queueElementSizeMax, MESSAGE_QUEUE_INIT_ARGUMENTS, queueElementSizeMax);
SAME_AT(tinynv_boot610_msgq_args_t, queueHeaderAlign, MESSAGE_QUEUE_INIT_ARGUMENTS, queueHeaderAlign);
SAME_AT(tinynv_boot610_msgq_args_t, queueElementAlign, MESSAGE_QUEUE_INIT_ARGUMENTS, queueElementAlign);

// GSP_ARGUMENTS_CACHED grew 72 -> 152 B.
SAME_SIZE(tinynv_boot610_args_cached_t, GSP_ARGUMENTS_CACHED);
SAME_AT(tinynv_boot610_args_cached_t, messageQueueInitArguments, GSP_ARGUMENTS_CACHED, messageQueueInitArguments);
SAME_AT(tinynv_boot610_args_cached_t, gpuInstance, GSP_ARGUMENTS_CACHED, gpuInstance);
SAME_AT(tinynv_boot610_args_cached_t, bDmemStack, GSP_ARGUMENTS_CACHED, bDmemStack);
SAME_AT(tinynv_boot610_args_cached_t, profilerArgs, GSP_ARGUMENTS_CACHED, profilerArgs);
SAME_AT(tinynv_boot610_args_cached_t, sysmemHeapArgs, GSP_ARGUMENTS_CACHED, sysmemHeapArgs);
SAME_AT(tinynv_boot610_args_cached_t, rmStateMonitorBufferArgs, GSP_ARGUMENTS_CACHED, rmStateMonitorBufferArgs);
SAME_AT(tinynv_boot610_args_cached_t, bindataArgs, GSP_ARGUMENTS_CACHED, bindataArgs);

// GSP_FMC_BOOT_PARAMS grew 80 -> 88 B (the tail gains GSP_RM_MEM_PARAMS).
SAME_SIZE(tinynv_boot610_fmc_boot_params_t, GSP_FMC_BOOT_PARAMS);
SAME_AT(tinynv_boot610_fmc_boot_params_t, bootGspRmParams, GSP_FMC_BOOT_PARAMS, bootGspRmParams);
SAME_AT(tinynv_boot610_fmc_boot_params_t, gspRmParams, GSP_FMC_BOOT_PARAMS, gspRmParams);
SAME_AT(tinynv_boot610_fmc_boot_params_t, gspSpdmParams, GSP_FMC_BOOT_PARAMS, gspSpdmParams);
SAME_AT(tinynv_boot610_fmc_boot_params_t, gspRmMemParams, GSP_FMC_BOOT_PARAMS, gspRmMemParams);
SAME_SIZE(tinynv_boot610_rm_mem_params_t, GSP_RM_MEM_PARAMS);

// RM_RISCV_UCODE_DESC: the 610 STRUCT grew to 92 B (bIsSmp/bIsPlicEnabled appended), but the offsets the boot path
// reads are unchanged, and the pinned AD102 archive ships the 84-byte v5 WIRE form (asserted against the real asset in
// test_boot610). So the shared 84-byte tinynv_riscv_ucode_desc_t still reads the used fields correctly; only a sizeof
// against the 610 struct would differ, and that is exactly why the wire form is parsed by offset, not by sizeof.
_Static_assert(sizeof(RM_RISCV_UCODE_DESC) == 92, "the 610 RM_RISCV_UCODE_DESC struct is 92 bytes (bIsSmp/bIsPlicEnabled)");
_Static_assert(sizeof(tinynv_riscv_ucode_desc_t) == 84, "the driver's descriptor stays the 84-byte v5 wire form");
SAME_AT(tinynv_riscv_ucode_desc_t, manifestOffset, RM_RISCV_UCODE_DESC, manifestOffset);
SAME_AT(tinynv_riscv_ucode_desc_t, monitorDataOffset, RM_RISCV_UCODE_DESC, monitorDataOffset);
SAME_AT(tinynv_riscv_ucode_desc_t, monitorCodeOffset, RM_RISCV_UCODE_DESC, monitorCodeOffset);

// --- 3. the reservation terms, re-derived from the matching 610 definitions --------------------------------------
// The heap formula is unchanged for a 32 GB card; 610 renamed SIZE_PER_GB_FB -> SIZE_PER_GB. Re-derive, don't copy.
_Static_assert(TINYNV_FW610_HEAP_SIZE == GSP_FW_HEAP_PARAM_OS_SIZE_LIBOS3_BAREMETAL + GSP_FW_HEAP_PARAM_BASE_RM_SIZE_GH100 +
                                          32 * GSP_FW_HEAP_PARAM_SIZE_PER_GB + GSP_FW_HEAP_PARAM_CLIENT_ALLOC_SIZE,
               "the 610 firmware heap is no longer the vendor's formula for a 32 GB card");
// The AD102 non-WPR heap: g_kernel_gsp_nvoc.c dispatches AD102 to kgspGetNonWprHeapSize_ed6b8b = 1048576 (1 MiB), the
// SAME HAL 570 uses. The generated HAL body is not includable standalone (it needs the whole NVOC port layer), so the
// value is pinned here as a source-verified constant; the dispatch evidence is recorded in the notes. This
// CORRECTS the audit's "AD102 doubled to 2 MiB" (that 2 MiB HAL, d505ea, serves GH100/GB100, not Ada).
_Static_assert(TINYNV_FW610_NONWPR_HEAP == 1048576, "the AD102 non-WPR heap is the matching ed6b8b HAL value (1 MiB)");

// --- 4. : the RPC envelope / function+event numbers that are SHARED with 570 --------------------------------
// The RPC message header is byte-identical; only its POSITION inside the queue element moved (the element framing is
// rpc610's codec). The signature and header version are unchanged.
SAME_SIZE(tinynv_msg_header_t, rpc_message_header_v03_00);
SAME_AT(tinynv_msg_header_t, header_version, rpc_message_header_v03_00, header_version);
SAME_AT(tinynv_msg_header_t, signature, rpc_message_header_v03_00, signature);
SAME_AT(tinynv_msg_header_t, length, rpc_message_header_v03_00, length);
SAME_AT(tinynv_msg_header_t, function, rpc_message_header_v03_00, function);
SAME_AT(tinynv_msg_header_t, rpc_result, rpc_message_header_v03_00, rpc_result);
SAME_AT(tinynv_msg_header_t, rpc_result_private, rpc_message_header_v03_00, rpc_result_private);
SAME_AT(tinynv_msg_header_t, sequence, rpc_message_header_v03_00, sequence);
_Static_assert(NV_VGPU_MSG_SIGNATURE_VALID == TINYNV_MSG_SIGNATURE_VALID, "the 610 rpc signature changed");
_Static_assert((NV_VGPU_MSG_HEADER_VERSION_MAJOR_TOT << 24) == TINYNV_MSG_HEADER_VERSION, "the 610 rpc header version changed");

// The RM ALLOC header is unchanged (32 B); the CONTROL header grew (below). set-page-directory / unloading are unchanged.
SAME_SIZE(tinynv_rpc_rm_alloc_t, rpc_gsp_rm_alloc_v03_00);
SAME_AT(tinynv_rpc_rm_alloc_t, hClient, rpc_gsp_rm_alloc_v03_00, hClient);
SAME_AT(tinynv_rpc_rm_alloc_t, hParent, rpc_gsp_rm_alloc_v03_00, hParent);
SAME_AT(tinynv_rpc_rm_alloc_t, hObject, rpc_gsp_rm_alloc_v03_00, hObject);
SAME_AT(tinynv_rpc_rm_alloc_t, hClass, rpc_gsp_rm_alloc_v03_00, hClass);
SAME_AT(tinynv_rpc_rm_alloc_t, status, rpc_gsp_rm_alloc_v03_00, status);
SAME_AT(tinynv_rpc_rm_alloc_t, paramsSize, rpc_gsp_rm_alloc_v03_00, paramsSize);
SAME_SIZE(tinynv_set_page_directory_t, rpc_set_page_directory_v1E_05);
SAME_SIZE(tinynv_unloading_guest_driver_t, rpc_unloading_guest_driver_v1F_07);

// Every function / event number the driver uses is unchanged; the two load-exec events are the 610 additions.
_Static_assert(NV_VGPU_MSG_FUNCTION_CONTINUATION_RECORD == TINYNV_MSG_FUNCTION_CONTINUATION_RECORD, "610 continuation fn changed");
_Static_assert(NV_VGPU_MSG_FUNCTION_GSP_SET_SYSTEM_INFO == TINYNV_MSG_FUNCTION_GSP_SET_SYSTEM_INFO, "610 set-system-info fn changed");
_Static_assert(NV_VGPU_MSG_FUNCTION_SET_REGISTRY == TINYNV_MSG_FUNCTION_SET_REGISTRY, "610 set-registry fn changed");
_Static_assert(NV_VGPU_MSG_FUNCTION_UNLOADING_GUEST_DRIVER == TINYNV_MSG_FUNCTION_UNLOADING_GUEST_DRIVER, "610 unload fn changed");
_Static_assert(NV_VGPU_MSG_FUNCTION_GSP_RM_CONTROL == TINYNV_MSG_FUNCTION_GSP_RM_CONTROL, "610 rm-control fn changed");
_Static_assert(NV_VGPU_MSG_FUNCTION_GSP_RM_ALLOC == TINYNV_MSG_FUNCTION_GSP_RM_ALLOC, "610 rm-alloc fn changed");
_Static_assert(NV_VGPU_MSG_FUNCTION_SET_PAGE_DIRECTORY == TINYNV_MSG_FUNCTION_SET_PAGE_DIRECTORY, "610 set-page-dir fn changed");
_Static_assert(NV_VGPU_MSG_EVENT_GSP_INIT_DONE == TINYNV_MSG_EVENT_GSP_INIT_DONE, "610 init-done event changed");
_Static_assert(NV_VGPU_MSG_EVENT_GSP_RUN_CPU_SEQUENCER == TINYNV_MSG_EVENT_GSP_RUN_CPU_SEQUENCER, "610 sequencer event number changed");
_Static_assert(NV_VGPU_MSG_EVENT_MMU_FAULT_QUEUED == TINYNV_MSG_EVENT_MMU_FAULT_QUEUED, "610 mmu-fault event changed");
_Static_assert(NV_VGPU_MSG_EVENT_OS_ERROR_LOG == TINYNV_MSG_EVENT_OS_ERROR_LOG, "610 error-log event changed");
_Static_assert(NV_VGPU_MSG_EVENT_GSP_LOAD_EXEC_GENERIC_BOOTLOADER == TINYNV_MSG_EVENT_GSP_LOAD_EXEC_GENERIC_BOOTLOADER, "610 load-exec-bl event");
_Static_assert(NV_VGPU_MSG_EVENT_GSP_LOAD_EXEC_HS_BINARY == TINYNV_MSG_EVENT_GSP_LOAD_EXEC_HS_BINARY, "610 load-exec-hs event");

// The 610 element framing constants, re-derived from the pinned MCTP/NVDM helpers (mctpCreateTransportHeader(1,1,0,0,0)
// and mctpCreateNvdmHeader(NVDM_TYPE_RM_RPC)).
_Static_assert(TINYNV_RPC610_MCTP_HEADER ==
                 (REF_NUM(MCTP_HEADER_VERSION, 0x1) | REF_NUM(MCTP_HEADER_SOM, 1) | REF_NUM(MCTP_HEADER_EOM, 1)),
               "the 610 MCTP transport header constant drifted from mctpCreateTransportHeader");
_Static_assert(TINYNV_RPC610_NVDM_HEADER ==
                 (REF_NUM(MCTP_MSG_HEADER_TYPE, MCTP_MSG_HEADER_TYPE_VENDOR_PCI) |
                  REF_NUM(MCTP_MSG_HEADER_VENDOR_ID, MCTP_MSG_HEADER_VENDOR_ID_NV) |
                  REF_NUM(MCTP_MSG_HEADER_NVDM_TYPE, NVDM_TYPE_RM_RPC)),
               "the 610 NVDM header constant drifted from mctpCreateNvdmHeader");
_Static_assert(NVDM_TYPE_RM_RPC == TINYNV_RPC610_NVDM_TYPE_RM_RPC, "the 610 RM_RPC NVDM type changed");
SAME_SIZE(tinynv_rpc610_elem_t, GSP_MSG_QUEUE_ELEMENT);
SAME_AT(tinynv_rpc610_elem_t, mctpHeader, GSP_MSG_QUEUE_ELEMENT, mctpHeader);
SAME_AT(tinynv_rpc610_elem_t, nvdmHeader, GSP_MSG_QUEUE_ELEMENT, nvdmHeader);
SAME_AT(tinynv_rpc610_elem_t, checkSum, GSP_MSG_QUEUE_ELEMENT, checkSum);
SAME_AT(tinynv_rpc610_elem_t, seqNum, GSP_MSG_QUEUE_ELEMENT, seqNum);
_Static_assert(TINYNV_RPC610_ELEM_HDR_SIZE == NV_OFFSETOF(GSP_MSG_QUEUE_ELEMENT, payload), "the 610 element header size changed");

// --- 5. : the CHANGED RM payloads ----------------------------------------------------------------------------
// The control header grew 24 -> 40 B; params moved to 40.
SAME_SIZE(tinynv_rpc610_rm_control_t, rpc_gsp_rm_control_v03_00);
SAME_AT(tinynv_rpc610_rm_control_t, hClient, rpc_gsp_rm_control_v03_00, hClient);
SAME_AT(tinynv_rpc610_rm_control_t, hObject, rpc_gsp_rm_control_v03_00, hObject);
SAME_AT(tinynv_rpc610_rm_control_t, cmd, rpc_gsp_rm_control_v03_00, cmd);
SAME_AT(tinynv_rpc610_rm_control_t, status, rpc_gsp_rm_control_v03_00, status);
SAME_AT(tinynv_rpc610_rm_control_t, paramsSize, rpc_gsp_rm_control_v03_00, paramsSize);
SAME_AT(tinynv_rpc610_rm_control_t, rmapiRpcFlags, rpc_gsp_rm_control_v03_00, rmapiRpcFlags);
SAME_AT(tinynv_rpc610_rm_control_t, rmctrlFlags, rpc_gsp_rm_control_v03_00, rmctrlFlags);
SAME_AT(tinynv_rpc610_rm_control_t, rmctrlAccessRight, rpc_gsp_rm_control_v03_00, rmctrlAccessRight);
SAME_AT(tinynv_rpc610_rm_control_t, reserved0, rpc_gsp_rm_control_v03_00, reserved0);
_Static_assert(offsetof(rpc_gsp_rm_control_v03_00, params) == 40, "the 610 control params are not at 40");

// GspSystemInfo: the exact 610 size / bIsPassthru offset the audit left unresolved, now computed from the
// verbatim-extracted pinned definition. The fields the driver fills at 0..96 keep their offsets.
_Static_assert(sizeof(GspSystemInfo) == TINYNV_RPC610_SYSINFO_SIZE, "the 610 GspSystemInfo size is not 952");
_Static_assert(offsetof(GspSystemInfo, bIsPassthru) == TINYNV_RPC610_SYSINFO_PASSTHRU_OFF, "the 610 bIsPassthru is not at 842");
_Static_assert(offsetof(GspSystemInfo, virtualConfigBits) == TINYNV_RPC610_SYSINFO_VIRTCONF_OFF, "the 610 virtualConfigBits is not at 840");
SAME_AT(tinynv_gsp_system_info_t, gpuPhysAddr, GspSystemInfo, gpuPhysAddr);
SAME_AT(tinynv_gsp_system_info_t, gpuPhysFbAddr, GspSystemInfo, gpuPhysFbAddr);
SAME_AT(tinynv_gsp_system_info_t, gpuPhysInstAddr, GspSystemInfo, gpuPhysInstAddr);
SAME_AT(tinynv_gsp_system_info_t, nvDomainBusDeviceFunc, GspSystemInfo, nvDomainBusDeviceFunc);
SAME_AT(tinynv_gsp_system_info_t, maxUserVa, GspSystemInfo, maxUserVa);
SAME_AT(tinynv_gsp_system_info_t, pciConfigMirrorBase, GspSystemInfo, pciConfigMirrorBase);
SAME_AT(tinynv_gsp_system_info_t, pciConfigMirrorSize, GspSystemInfo, pciConfigMirrorSize);
SAME_AT(tinynv_gsp_system_info_t, PCIDeviceID, GspSystemInfo, PCIDeviceID);
SAME_AT(tinynv_gsp_system_info_t, PCISubDeviceID, GspSystemInfo, PCISubDeviceID);
SAME_AT(tinynv_gsp_system_info_t, PCIRevisionID, GspSystemInfo, PCIRevisionID);

// The channel allocation grew 368 -> 376 B (hHandleVASpace inserted); every later field shifts.
SAME_SIZE(tinynv_rpc610_gpfifo_alloc_t, NV_CHANNEL_ALLOC_PARAMS);
SAME_AT(tinynv_rpc610_gpfifo_alloc_t, hObjectError, NV_CHANNEL_ALLOC_PARAMS, hObjectError);
SAME_AT(tinynv_rpc610_gpfifo_alloc_t, gpFifoOffset, NV_CHANNEL_ALLOC_PARAMS, gpFifoOffset);
SAME_AT(tinynv_rpc610_gpfifo_alloc_t, hVASpace, NV_CHANNEL_ALLOC_PARAMS, hVASpace);
SAME_AT(tinynv_rpc610_gpfifo_alloc_t, hHandleVASpace, NV_CHANNEL_ALLOC_PARAMS, hHandleVASpace);
SAME_AT(tinynv_rpc610_gpfifo_alloc_t, hUserdMemory, NV_CHANNEL_ALLOC_PARAMS, hUserdMemory);
SAME_AT(tinynv_rpc610_gpfifo_alloc_t, userdOffset, NV_CHANNEL_ALLOC_PARAMS, userdOffset);
SAME_AT(tinynv_rpc610_gpfifo_alloc_t, engineType, NV_CHANNEL_ALLOC_PARAMS, engineType);
SAME_AT(tinynv_rpc610_gpfifo_alloc_t, instanceMem, NV_CHANNEL_ALLOC_PARAMS, instanceMem);
SAME_AT(tinynv_rpc610_gpfifo_alloc_t, userdMem, NV_CHANNEL_ALLOC_PARAMS, userdMem);
SAME_AT(tinynv_rpc610_gpfifo_alloc_t, ramfcMem, NV_CHANNEL_ALLOC_PARAMS, ramfcMem);
SAME_AT(tinynv_rpc610_gpfifo_alloc_t, mthdbufMem, NV_CHANNEL_ALLOC_PARAMS, mthdbufMem);
SAME_AT(tinynv_rpc610_gpfifo_alloc_t, internalFlags, NV_CHANNEL_ALLOC_PARAMS, internalFlags);
SAME_AT(tinynv_rpc610_gpfifo_alloc_t, errorNotifierMem, NV_CHANNEL_ALLOC_PARAMS, errorNotifierMem);
SAME_AT(tinynv_rpc610_gpfifo_alloc_t, ProcessID, NV_CHANNEL_ALLOC_PARAMS, ProcessID);
SAME_AT(tinynv_rpc610_gpfifo_alloc_t, tpcConfigID, NV_CHANNEL_ALLOC_PARAMS, tpcConfigID);

// The VA space allocation grew 48 -> 56 B (pasid appended).
SAME_SIZE(tinynv_rpc610_vaspace_alloc_t, NV_VASPACE_ALLOCATION_PARAMETERS);
SAME_AT(tinynv_rpc610_vaspace_alloc_t, vaSize, NV_VASPACE_ALLOCATION_PARAMETERS, vaSize);
SAME_AT(tinynv_rpc610_vaspace_alloc_t, vaBase, NV_VASPACE_ALLOCATION_PARAMETERS, vaBase);
SAME_AT(tinynv_rpc610_vaspace_alloc_t, pasid, NV_VASPACE_ALLOCATION_PARAMETERS, pasid);

// The GR static-info stride grew 58 -> 63 entries; the indices the driver reads are unchanged.
_Static_assert(NV0080_CTRL_GR_INFO_MAX_SIZE == TINYNV_RPC610_GR_INFO_COUNT, "the 610 GR info count is not 63");
SAME_SIZE(tinynv_rpc610_gr_static_info_t, NV2080_CTRL_INTERNAL_STATIC_GR_GET_INFO_PARAMS);
SAME_SIZE(tinynv_rpc610_gr_engine_info_t, NV2080_CTRL_INTERNAL_STATIC_GR_INFO);
SAME_SIZE(tinynv_rpc610_gr_info_t, NV2080_CTRL_INTERNAL_GR_INFO);
_Static_assert(NV0080_CTRL_GR_INFO_INDEX_LITTER_NUM_GPCS == TINYNV_GR_INFO_NUM_GPCS, "610 NUM_GPCS index changed");
_Static_assert(NV0080_CTRL_GR_INFO_INDEX_LITTER_NUM_TPC_PER_GPC == TINYNV_GR_INFO_NUM_TPC_PER_GPC, "610 NUM_TPC index changed");
_Static_assert(NV0080_CTRL_GR_INFO_INDEX_LITTER_NUM_SM_PER_TPC == TINYNV_GR_INFO_NUM_SM_PER_TPC, "610 NUM_SM index changed");
_Static_assert(NV0080_CTRL_GR_INFO_INDEX_MAX_WARPS_PER_SM == TINYNV_GR_INFO_MAX_WARPS_PER_SM, "610 MAX_WARPS index changed");
_Static_assert(NV0080_CTRL_GR_INFO_INDEX_SM_VERSION == TINYNV_GR_INFO_SM_VERSION, "610 SM_VERSION index changed");

// The GPFIFO schedule grew 2 -> 3 B (bSkipEnable appended).
SAME_SIZE(tinynv_rpc610_gpfifo_schedule_t, NVA06C_CTRL_GPFIFO_SCHEDULE_PARAMS);
SAME_AT(tinynv_rpc610_gpfifo_schedule_t, bEnable, NVA06C_CTRL_GPFIFO_SCHEDULE_PARAMS, bEnable);
SAME_AT(tinynv_rpc610_gpfifo_schedule_t, bSkipSubmit, NVA06C_CTRL_GPFIFO_SCHEDULE_PARAMS, bSkipSubmit);
SAME_AT(tinynv_rpc610_gpfifo_schedule_t, bSkipEnable, NVA06C_CTRL_GPFIFO_SCHEDULE_PARAMS, bSkipEnable);

// RUSD: the init control grew 8 -> 16 B (bInit before physAddr@8); the poll control renamed its interval field.
SAME_SIZE(tinynv_rpc610_rusd_init_t, NV2080_CTRL_INTERNAL_INIT_USER_SHARED_DATA_PARAMS);
SAME_AT(tinynv_rpc610_rusd_init_t, bInit, NV2080_CTRL_INTERNAL_INIT_USER_SHARED_DATA_PARAMS, bInit);
SAME_AT(tinynv_rpc610_rusd_init_t, physAddr, NV2080_CTRL_INTERNAL_INIT_USER_SHARED_DATA_PARAMS, physAddr);
SAME_SIZE(tinynv_rpc610_rusd_poll_t, NV2080_CTRL_INTERNAL_USER_SHARED_DATA_SET_DATA_POLL_PARAMS);
SAME_AT(tinynv_rpc610_rusd_poll_t, polledDataMask, NV2080_CTRL_INTERNAL_USER_SHARED_DATA_SET_DATA_POLL_PARAMS, polledDataMask);
SAME_AT(tinynv_rpc610_rusd_poll_t, pollIntervalMs, NV2080_CTRL_INTERNAL_USER_SHARED_DATA_SET_DATA_POLL_PARAMS, pollIntervalMs);

// The 610 RUSD shared block: the offsets the reader maps, asserted against cl00de.h.
_Static_assert(sizeof(NV00DE_SHARED_DATA) == TINYNV_RPC610_RUSD_SIZE, "the 610 RUSD block is not 3152 bytes");
_Static_assert(offsetof(NV00DE_SHARED_DATA, bar1MemoryInfo) == 0, "610 RUSD bar1 offset");
_Static_assert(offsetof(NV00DE_SHARED_DATA, pmaMemoryInfo) == 16, "610 RUSD pma offset");
_Static_assert(offsetof(NV00DE_SHARED_DATA, clkPublicDomainInfos) == 72, "610 RUSD clk offset");
_Static_assert(offsetof(NV00DE_SHARED_DATA, clkThrottleInfo) == 112, "610 RUSD throttle offset");
_Static_assert(offsetof(NV00DE_SHARED_DATA, perfDevUtil) == 1992, "610 RUSD devutil offset");
_Static_assert(offsetof(NV00DE_SHARED_DATA, memEcc) == 2040, "610 RUSD ecc offset");
_Static_assert(offsetof(NV00DE_SHARED_DATA, perfCurrentPstate) == 2280, "610 RUSD pstate offset");
_Static_assert(offsetof(NV00DE_SHARED_DATA, powerLimitGpu) == 2296, "610 RUSD powerlimit offset");
_Static_assert(offsetof(NV00DE_SHARED_DATA, temperatures) == 2312, "610 RUSD temperatures offset");
_Static_assert(offsetof(NV00DE_SHARED_DATA, avgPowerUsage) == 2440, "610 RUSD avgpower offset");
_Static_assert(offsetof(NV00DE_SHARED_DATA, instPowerUsage) == 2464, "610 RUSD instpower offset");
_Static_assert(offsetof(NV00DE_SHARED_DATA, pciBusData) == 2504, "610 RUSD pci offset");
_Static_assert(offsetof(RUSD_CLK_THROTTLE_INFO, reasonMask) == 8, "610 throttle reasonMask is not at +8");

// --- 6. : the host-request payloads (the sequencer's replacements) ------------------------------------------
SAME_SIZE(tinynv_rpc610_bl_dmem_desc_t, RM_FLCN_BL_DMEM_DESC);
SAME_AT(tinynv_rpc610_bl_dmem_desc_t, ctxDma, RM_FLCN_BL_DMEM_DESC, ctxDma);
SAME_AT(tinynv_rpc610_bl_dmem_desc_t, codeDmaBase, RM_FLCN_BL_DMEM_DESC, codeDmaBase);
SAME_AT(tinynv_rpc610_bl_dmem_desc_t, dataDmaBase, RM_FLCN_BL_DMEM_DESC, dataDmaBase);
SAME_SIZE(tinynv_rpc610_load_exec_bl_t, GspLoadExecGenericBootloaderParams);
SAME_AT(tinynv_rpc610_load_exec_bl_t, dmemDesc, GspLoadExecGenericBootloaderParams, dmemDesc);
SAME_AT(tinynv_rpc610_load_exec_bl_t, dmemDescSize, GspLoadExecGenericBootloaderParams, dmemDescSize);
SAME_AT(tinynv_rpc610_load_exec_bl_t, addrSpace, GspLoadExecGenericBootloaderParams, addrSpace);
SAME_AT(tinynv_rpc610_load_exec_bl_t, cpuCacheAttrib, GspLoadExecGenericBootloaderParams, cpuCacheAttrib);
SAME_SIZE(tinynv_rpc610_load_exec_hs_t, GspLoadExecHsBinaryParams);
SAME_AT(tinynv_rpc610_load_exec_hs_t, imemPhysAddr, GspLoadExecHsBinaryParams, imemPhysAddr);
SAME_AT(tinynv_rpc610_load_exec_hs_t, dmemPhysAddr, GspLoadExecHsBinaryParams, dmemPhysAddr);
SAME_AT(tinynv_rpc610_load_exec_hs_t, ucodeImemVA, GspLoadExecHsBinaryParams, ucodeImemVA);
SAME_AT(tinynv_rpc610_load_exec_hs_t, ucodeImemPA, GspLoadExecHsBinaryParams, ucodeImemPA);
SAME_AT(tinynv_rpc610_load_exec_hs_t, ucodeImemSize, GspLoadExecHsBinaryParams, ucodeImemSize);
SAME_AT(tinynv_rpc610_load_exec_hs_t, ucodeDmemVA, GspLoadExecHsBinaryParams, ucodeDmemVA);
SAME_AT(tinynv_rpc610_load_exec_hs_t, ucodeDmemPA, GspLoadExecHsBinaryParams, ucodeDmemPA);
SAME_AT(tinynv_rpc610_load_exec_hs_t, ucodeDmemSize, GspLoadExecHsBinaryParams, ucodeDmemSize);
SAME_AT(tinynv_rpc610_load_exec_hs_t, hsSigDmemAddr, GspLoadExecHsBinaryParams, hsSigDmemAddr);
SAME_AT(tinynv_rpc610_load_exec_hs_t, engineIdMask, GspLoadExecHsBinaryParams, engineIdMask);
SAME_AT(tinynv_rpc610_load_exec_hs_t, ucodeId, GspLoadExecHsBinaryParams, ucodeId);

// --- 7. : the falcon registers the host-request executor programs, asserted against the pinned hwref --------
_Static_assert(NV_PFALCON_FALCON_HWCFG == 0x108, "610 HWCFG offset");
_Static_assert(NV_PFALCON_FALCON_IMEMC(0) == 0x180 && NV_PFALCON_FALCON_IMEMC(1) == 0x190, "610 IMEMC stride");
_Static_assert(NV_PFALCON_FALCON_IMEMD(0) == 0x184, "610 IMEMD offset");
_Static_assert(NV_PFALCON_FALCON_IMEMT(0) == 0x188, "610 IMEMT offset");
_Static_assert(NV_PFALCON_FALCON_DMEMC(0) == 0x1c0 && NV_PFALCON_FALCON_DMEMC(1) == 0x1c8, "610 DMEMC stride");
_Static_assert(NV_PFALCON_FALCON_DMEMD(0) == 0x1c4, "610 DMEMD offset");
_Static_assert(NV_PFALCON_FBIF_TRANSCFG_TARGET_LOCAL_FB == 0, "610 TRANSCFG LOCAL_FB");
_Static_assert(NV_PFALCON_FBIF_TRANSCFG_TARGET_COHERENT_SYSMEM == 1, "610 TRANSCFG COHERENT");
_Static_assert(NV_PFALCON_FBIF_TRANSCFG_TARGET_NONCOHERENT_SYSMEM == 2, "610 TRANSCFG NONCOHERENT");

int main(void) {
  printf("headers-610: shared layouts asserted byte-identical, changed boot layouts matched, reservation re-derived - "
         "wpr meta %zu, msgq init args %zu, args cached %zu, fmc boot params %zu, riscv desc struct %zu (wire 84)\n",
         sizeof(GspFwWprMeta), sizeof(MESSAGE_QUEUE_INIT_ARGUMENTS), sizeof(GSP_ARGUMENTS_CACHED),
         sizeof(GSP_FMC_BOOT_PARAMS), sizeof(RM_RISCV_UCODE_DESC));
  printf("headers-610: rpc/rm boundary asserted - rpc hdr %zu, control %zu (params@%zu), alloc %zu, sysinfo %zu "
         "(bIsPassthru@%zu), channel %zu, vaspace %zu, gr static %zu, schedule %zu, rusd %zu, load-exec bl %zu / hs %zu\n",
         sizeof(rpc_message_header_v03_00), sizeof(rpc_gsp_rm_control_v03_00), offsetof(rpc_gsp_rm_control_v03_00, params),
         sizeof(rpc_gsp_rm_alloc_v03_00), sizeof(GspSystemInfo), offsetof(GspSystemInfo, bIsPassthru),
         sizeof(NV_CHANNEL_ALLOC_PARAMS), sizeof(NV_VASPACE_ALLOCATION_PARAMETERS), sizeof(NV2080_CTRL_INTERNAL_STATIC_GR_GET_INFO_PARAMS),
         sizeof(NVA06C_CTRL_GPFIFO_SCHEDULE_PARAMS), sizeof(NV00DE_SHARED_DATA),
         sizeof(GspLoadExecGenericBootloaderParams), sizeof(GspLoadExecHsBinaryParams));
  return 0;
}
