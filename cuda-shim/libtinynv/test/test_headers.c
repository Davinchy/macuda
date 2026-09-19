// Holds the driver's own structure declarations against NVIDIA's.
//
// src/nv_structs.h re-declares the structures the boot path builds, so the library still compiles on a machine that has
// not fetched the 108 MB kernel-module tree. This file includes the real headers from the pinned tree and asserts that
// every size and every offset the driver relies on is the same in both. If upstream moves a field, the build breaks
// here, which is a much better place to find out than a GPU refusing a signature for no stated reason.
#include <stddef.h>
#include <stdio.h>
#include "nvtypes.h"
#include "nvgputypes.h"
#include "gsp/gsp_fw_wpr_meta.h"
#include "gsp/gspifpub.h"
#include "kernel/gpu/fsp/kern_fsp_cot_payload.h"
#include "fsp/fsp_nvdm_format.h"
#include "msgq/msgq_priv.h"
#include "kernel/gpu/gsp/gsp_init_args.h"
#include "libos_init_args.h"
#include "rmRiscvUcode.h"
#include "class/clc56f.h"
#include "ctrl/ctrl83de.h"
#include "swref/published/blackwell/gb202/dev_fault.h"
#include "nvos.h"
#include "ctrl/ctrl0041.h"
#include "swref/published/hopper/gh100/dev_mmu.h"
#include "kernel/vgpu/rpc_headers.h"
#include "class/cl00de.h"
#include "ctrl/ctrl2080/ctrl2080internal.h"
#include "ctrl/ctrla06c.h"
#include "kernel/gpu/gsp/gsp_fw_heap.h"
#include "nv_structs.h"
#include "fw_layout.h"

// The firmware heap this driver hands GSP-RM is the vendor's own formula for a 32 GB card, term by term
// (kernel_gsp.c, _kgspCalculateFwHeapSize): the LIBOS3 bare-metal OS carveout, the Hopper+ base RM size, 96 KB per
// GB of FB over 32 GB, and 48 KB per channel over 2048 channels. The non-WPR heap is the GB20x HAL's value
// (g_kernel_gsp_nvoc.h). If either side moves, the build breaks here rather than the firmware placing a bigger region
// under the manager's feet: these two sizes are what fw_layout.h sizes the reservation from.
_Static_assert(TINYNV_FW_HEAP_SIZE == GSP_FW_HEAP_PARAM_OS_SIZE_LIBOS3_BAREMETAL + GSP_FW_HEAP_PARAM_BASE_RM_SIZE_GH100 +
                                          32 * GSP_FW_HEAP_PARAM_SIZE_PER_GB_FB + GSP_FW_HEAP_PARAM_CLIENT_ALLOC_SIZE,
               "the firmware heap size handed to GSP-RM is no longer the vendor's formula for a 32 GB card");
_Static_assert(TINYNV_FW_NONWPR_HEAP == 2228224, "the GB20x non-WPR heap is 2,228,224 bytes (g_kernel_gsp_nvoc.h)");

// ours against theirs: the same number of bytes, and every field we touch in the same place
#define SAME_SIZE(mine, theirs) _Static_assert(sizeof(mine) == sizeof(theirs), #mine " is not the size of " #theirs)
#define SAME_AT(mine, mf, theirs, tf) \
  _Static_assert(offsetof(mine, mf) == offsetof(theirs, tf), #mine "." #mf " is not where " #theirs "." #tf " is")

// Where a released timestamp lands. Both halves of the refill instrument read the gpu's clock at +8 of a 16-byte
// semaphore report - one written by a host semaphore method, the other by a descriptor's release slot - and on
// 2026-09-15 that offset was a belief with a run behind it rather than a checked fact. NVIDIA's own NvGpuSemaphore
// settles it: payload at 0..7, timestamp at 8..15. If a future header moves it, the build breaks rather than the
// driver subtracting a payload from a clock and reporting the difference as idle time, which is a number that looks
// entirely plausible and is what this instrument produced twice in one day.
_Static_assert(offsetof(NvGpuSemaphore, timeStamp) == 8,
               "a released timestamp is not 8 bytes into the semaphore report");
_Static_assert(sizeof(NvGpuSemaphore) == 16, "a semaphore report with a timestamp is not 16 bytes");

// The one field in a channel's control block this driver writes. Nothing else checks it: the recorded boot makes no
// submissions, so a wrong offset here is a write into the control block that the engine never reads.
_Static_assert(TINYNV_GPFIFO_GPPUT_OFF == offsetof(AmpereAControlGPFifo, GPPut),
               "GPPut is not where the vendored headers put it");

// Freeing an rm object. The wire form and the ioctl form happen to be identical here - the generated
// NVOS00_PARAMETERS_v03_00 has the same four fields in the same places as nvos.h's NVOS00_PARAMETERS - but that is a
// fact about this structure rather than a rule about RPC payloads, so it is asserted against the one the RPC actually
// carries. The driver has never sent this message to a card, which is exactly why it is worth holding against the
// header rather than against my reading of it.
// The sensor block. Transcribing a 464-byte structure by hand is only defensible because every byte of it is held
// against the original here: if NVIDIA moves a field, the build breaks rather than the driver reading a temperature out
// of the middle of a power reading.
// The scheduler timeslice control. One field, and asserting our own size against 8 would only be checking arithmetic
// I did myself - it has to be held against NVIDIA's struct or it is not held against anything.
// The memory-allocation params, whose `size` is the only way to bound a guest's mapping offset (guarantee 8). Held
// against NVIDIA's struct field by field rather than transcribed: the one number we act on is at an offset nobody
// should be computing by hand.
SAME_SIZE(tinynv_nv0040_alloc_t, NV_MEMORY_ALLOCATION_PARAMS);
SAME_AT(tinynv_nv0040_alloc_t, size, NV_MEMORY_ALLOCATION_PARAMS, size);
SAME_AT(tinynv_nv0040_alloc_t, offset, NV_MEMORY_ALLOCATION_PARAMS, offset);
SAME_AT(tinynv_nv0040_alloc_t, flags, NV_MEMORY_ALLOCATION_PARAMS, flags);
SAME_AT(tinynv_nv0040_alloc_t, hVASpace, NV_MEMORY_ALLOCATION_PARAMS, hVASpace);

SAME_SIZE(tinynv_timeslice_t, NVA06C_CTRL_TIMESLICE_PARAMS);
SAME_AT(tinynv_timeslice_t, timesliceUs, NVA06C_CTRL_TIMESLICE_PARAMS, timesliceUs);

SAME_SIZE(tinynv_rusd_t, NV00DE_SHARED_DATA);
SAME_AT(tinynv_rusd_t, bar1MemoryInfo, NV00DE_SHARED_DATA, bar1MemoryInfo);
SAME_AT(tinynv_rusd_t, pmaMemoryInfo, NV00DE_SHARED_DATA, pmaMemoryInfo);
SAME_AT(tinynv_rusd_t, shadowErrCont, NV00DE_SHARED_DATA, shadowErrCont);
SAME_AT(tinynv_rusd_t, grInfo, NV00DE_SHARED_DATA, grInfo);
SAME_AT(tinynv_rusd_t, clkPublicDomainInfos, NV00DE_SHARED_DATA, clkPublicDomainInfos);
SAME_AT(tinynv_rusd_t, clkThrottleReason, NV00DE_SHARED_DATA, clkThrottleReason);
SAME_AT(tinynv_rusd_t, perfDevUtil, NV00DE_SHARED_DATA, perfDevUtil);
SAME_AT(tinynv_rusd_t, memEcc, NV00DE_SHARED_DATA, memEcc);
SAME_AT(tinynv_rusd_t, perfCurrentPstate, NV00DE_SHARED_DATA, perfCurrentPstate);
SAME_AT(tinynv_rusd_t, powerLimitGpu, NV00DE_SHARED_DATA, powerLimitGpu);
SAME_AT(tinynv_rusd_t, temperatures, NV00DE_SHARED_DATA, temperatures);
SAME_AT(tinynv_rusd_t, memRowRemap, NV00DE_SHARED_DATA, memRowRemap);
SAME_AT(tinynv_rusd_t, avgPowerUsage, NV00DE_SHARED_DATA, avgPowerUsage);
SAME_AT(tinynv_rusd_t, instPowerUsage, NV00DE_SHARED_DATA, instPowerUsage);
SAME_AT(tinynv_rusd_t, pciBusData, NV00DE_SHARED_DATA, pciBusData);
// And the fields actually read, inside their own sub-structures.
SAME_SIZE(tinynv_rusd_temp_t, RUSD_TEMPERATURE);
SAME_AT(tinynv_rusd_temp_t, temperature, RUSD_TEMPERATURE, temperature);
SAME_SIZE(tinynv_rusd_avgpwr_t, RUSD_AVG_POWER_USAGE);
SAME_SIZE(tinynv_rusd_clk_t, RUSD_CLK_PUBLIC_DOMAIN_INFOS);
SAME_SIZE(tinynv_rusd_util_t, RUSD_PERF_DEVICE_UTILIZATION);
SAME_SIZE(tinynv_rusd_powerlimit_t, RUSD_POWER_LIMITS);
SAME_SIZE(tinynv_rusd_throttle_t, RUSD_CLK_THROTTLE_REASON);
SAME_SIZE(tinynv_rusd_init_t, NV2080_CTRL_INTERNAL_INIT_USER_SHARED_DATA_PARAMS);
SAME_SIZE(tinynv_rusd_poll_t, NV2080_CTRL_INTERNAL_USER_SHARED_DATA_SET_DATA_POLL_PARAMS);
SAME_SIZE(tinynv_rusd_alloc_t, NV00DE_ALLOC_PARAMETERS);
_Static_assert(TINYNV_CLASS_USER_SHARED_DATA == RM_USER_SHARED_DATA, "the shared-data class number moved");
_Static_assert(TINYNV_CTRL_INIT_USER_SHARED_DATA == NV2080_CTRL_CMD_INTERNAL_INIT_USER_SHARED_DATA, "init ctrl moved");
_Static_assert(TINYNV_CTRL_USER_SHARED_DATA_POLL == NV2080_CTRL_CMD_INTERNAL_USER_SHARED_DATA_SET_DATA_POLL,
               "poll ctrl moved");
_Static_assert(TINYNV_RUSD_POLL_THERMAL == NV00DE_RUSD_POLL_THERMAL, "the thermal poll bit moved");
_Static_assert(TINYNV_RUSD_POLL_POWER == NV00DE_RUSD_POLL_POWER, "the power poll bit moved");

SAME_SIZE(tinynv_rpc_rm_free_t, NVOS00_PARAMETERS);
SAME_AT(tinynv_rpc_rm_free_t, hRoot, NVOS00_PARAMETERS, hRoot);
SAME_AT(tinynv_rpc_rm_free_t, hObjectParent, NVOS00_PARAMETERS, hObjectParent);
SAME_AT(tinynv_rpc_rm_free_t, hObjectOld, NVOS00_PARAMETERS, hObjectOld);
SAME_AT(tinynv_rpc_rm_free_t, status, NVOS00_PARAMETERS, status);
// And the function number, which is the other half of the message and the half that is a bare integer in our source.
_Static_assert(TINYNV_MSG_FUNCTION_FREE == NV_VGPU_MSG_FUNCTION_FREE,
               "the rm free function number is not what the pinned tree calls FREE");
_Static_assert(TINYNV_MSG_FUNCTION_GSP_RM_ALLOC == NV_VGPU_MSG_FUNCTION_GSP_RM_ALLOC,
               "the rm alloc function number is not what the pinned tree calls GSP_RM_ALLOC");

SAME_SIZE(tinynv_cot_payload_t, NVDM_PAYLOAD_COT);
SAME_AT(tinynv_cot_payload_t, version, NVDM_PAYLOAD_COT, version);
SAME_AT(tinynv_cot_payload_t, size, NVDM_PAYLOAD_COT, size);
SAME_AT(tinynv_cot_payload_t, gspFmcSysmemOffset, NVDM_PAYLOAD_COT, gspFmcSysmemOffset);
SAME_AT(tinynv_cot_payload_t, frtsVidmemOffset, NVDM_PAYLOAD_COT, frtsVidmemOffset);
SAME_AT(tinynv_cot_payload_t, frtsVidmemSize, NVDM_PAYLOAD_COT, frtsVidmemSize);
SAME_AT(tinynv_cot_payload_t, hash384, NVDM_PAYLOAD_COT, hash384);
SAME_AT(tinynv_cot_payload_t, publicKey, NVDM_PAYLOAD_COT, publicKey);
SAME_AT(tinynv_cot_payload_t, signature, NVDM_PAYLOAD_COT, signature);
SAME_AT(tinynv_cot_payload_t, gspBootArgsSysmemOffset, NVDM_PAYLOAD_COT, gspBootArgsSysmemOffset);

SAME_SIZE(tinynv_gsp_fmc_boot_params_t, GSP_FMC_BOOT_PARAMS);
SAME_AT(tinynv_gsp_fmc_boot_params_t, bootGspRmParams, GSP_FMC_BOOT_PARAMS, bootGspRmParams);
SAME_AT(tinynv_gsp_fmc_boot_params_t, gspRmParams, GSP_FMC_BOOT_PARAMS, gspRmParams);
SAME_AT(tinynv_gsp_fmc_boot_params_t, gspSpdmParams, GSP_FMC_BOOT_PARAMS, gspSpdmParams);
SAME_SIZE(tinynv_gsp_acr_boot_params_t, GSP_ACR_BOOT_GSP_RM_PARAMS);
SAME_AT(tinynv_gsp_acr_boot_params_t, target, GSP_ACR_BOOT_GSP_RM_PARAMS, target);
SAME_AT(tinynv_gsp_acr_boot_params_t, gspRmDescSize, GSP_ACR_BOOT_GSP_RM_PARAMS, gspRmDescSize);
SAME_AT(tinynv_gsp_acr_boot_params_t, gspRmDescOffset, GSP_ACR_BOOT_GSP_RM_PARAMS, gspRmDescOffset);
SAME_AT(tinynv_gsp_acr_boot_params_t, bIsGspRmBoot, GSP_ACR_BOOT_GSP_RM_PARAMS, bIsGspRmBoot);
SAME_SIZE(tinynv_gsp_rm_params_t, GSP_RM_PARAMS);
SAME_AT(tinynv_gsp_rm_params_t, bootArgsOffset, GSP_RM_PARAMS, bootArgsOffset);
SAME_SIZE(tinynv_gsp_spdm_params_t, GSP_SPDM_PARAMS);

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
SAME_AT(tinynv_wpr_meta_t, bootCount, GspFwWprMeta, bootCount);
SAME_AT(tinynv_wpr_meta_t, pmuReservedSize, GspFwWprMeta, pmuReservedSize);
SAME_AT(tinynv_wpr_meta_t, verified, GspFwWprMeta, verified);

SAME_SIZE(tinynv_msgq_tx_header_t, msgqTxHeader);
SAME_AT(tinynv_msgq_tx_header_t, size, msgqTxHeader, size);
SAME_AT(tinynv_msgq_tx_header_t, msgSize, msgqTxHeader, msgSize);
SAME_AT(tinynv_msgq_tx_header_t, msgCount, msgqTxHeader, msgCount);
SAME_AT(tinynv_msgq_tx_header_t, writePtr, msgqTxHeader, writePtr);
SAME_AT(tinynv_msgq_tx_header_t, flags, msgqTxHeader, flags);
SAME_AT(tinynv_msgq_tx_header_t, rxHdrOff, msgqTxHeader, rxHdrOff);
SAME_AT(tinynv_msgq_tx_header_t, entryOff, msgqTxHeader, entryOff);

SAME_SIZE(tinynv_msgq_init_args_t, MESSAGE_QUEUE_INIT_ARGUMENTS);
SAME_AT(tinynv_msgq_init_args_t, sharedMemPhysAddr, MESSAGE_QUEUE_INIT_ARGUMENTS, sharedMemPhysAddr);
SAME_AT(tinynv_msgq_init_args_t, pageTableEntryCount, MESSAGE_QUEUE_INIT_ARGUMENTS, pageTableEntryCount);
SAME_AT(tinynv_msgq_init_args_t, cmdQueueOffset, MESSAGE_QUEUE_INIT_ARGUMENTS, cmdQueueOffset);
SAME_AT(tinynv_msgq_init_args_t, statQueueOffset, MESSAGE_QUEUE_INIT_ARGUMENTS, statQueueOffset);
SAME_SIZE(tinynv_gsp_args_cached_t, GSP_ARGUMENTS_CACHED);
SAME_AT(tinynv_gsp_args_cached_t, messageQueueInitArguments, GSP_ARGUMENTS_CACHED, messageQueueInitArguments);
SAME_AT(tinynv_gsp_args_cached_t, gpuInstance, GSP_ARGUMENTS_CACHED, gpuInstance);
SAME_AT(tinynv_gsp_args_cached_t, bDmemStack, GSP_ARGUMENTS_CACHED, bDmemStack);

SAME_SIZE(tinynv_libos_region_t, LibosMemoryRegionInitArgument);
SAME_AT(tinynv_libos_region_t, id8, LibosMemoryRegionInitArgument, id8);
SAME_AT(tinynv_libos_region_t, pa, LibosMemoryRegionInitArgument, pa);
SAME_AT(tinynv_libos_region_t, size, LibosMemoryRegionInitArgument, size);
SAME_AT(tinynv_libos_region_t, kind, LibosMemoryRegionInitArgument, kind);
SAME_AT(tinynv_libos_region_t, loc, LibosMemoryRegionInitArgument, loc);

// What the debugger is asked after a fault. These two structures are read back from GSP-RM, so a field in the wrong
// place is not a compile error anywhere - it is a fault address taken from the middle of a register dump, which reads
// like a plausible pointer and sends the next hour somewhere wrong.
SAME_SIZE(tinynv_sm_error_state_t, NV83DE_SM_ERROR_STATE_REGISTERS);
SAME_AT(tinynv_sm_error_state_t, hwwGlobalEsr, NV83DE_SM_ERROR_STATE_REGISTERS, hwwGlobalEsr);
SAME_AT(tinynv_sm_error_state_t, hwwWarpEsr, NV83DE_SM_ERROR_STATE_REGISTERS, hwwWarpEsr);
SAME_AT(tinynv_sm_error_state_t, hwwEsrAddr, NV83DE_SM_ERROR_STATE_REGISTERS, hwwEsrAddr);
SAME_AT(tinynv_sm_error_state_t, hwwWarpEsrPc64, NV83DE_SM_ERROR_STATE_REGISTERS, hwwWarpEsrPc64);
SAME_AT(tinynv_sm_error_state_t, hwwCgaEsr, NV83DE_SM_ERROR_STATE_REGISTERS, hwwCgaEsr);

SAME_SIZE(tinynv_sm_error_states_t, NV83DE_CTRL_DEBUG_READ_ALL_SM_ERROR_STATES_PARAMS);
SAME_AT(tinynv_sm_error_states_t, hTargetChannel, NV83DE_CTRL_DEBUG_READ_ALL_SM_ERROR_STATES_PARAMS, hTargetChannel);
SAME_AT(tinynv_sm_error_states_t, numSMsToRead, NV83DE_CTRL_DEBUG_READ_ALL_SM_ERROR_STATES_PARAMS, numSMsToRead);
SAME_AT(tinynv_sm_error_states_t, smErrorStateArray, NV83DE_CTRL_DEBUG_READ_ALL_SM_ERROR_STATES_PARAMS, smErrorStateArray);
SAME_AT(tinynv_sm_error_states_t, mmuFaultInfo, NV83DE_CTRL_DEBUG_READ_ALL_SM_ERROR_STATES_PARAMS, mmuFaultInfo);
SAME_AT(tinynv_sm_error_states_t, mmuFault, NV83DE_CTRL_DEBUG_READ_ALL_SM_ERROR_STATES_PARAMS, mmuFault);
SAME_AT(tinynv_sm_error_states_t, startingSM, NV83DE_CTRL_DEBUG_READ_ALL_SM_ERROR_STATES_PARAMS, startingSM);
// the valid bit is one byte inside that structure, and reading it as a word would make every fault look valid
_Static_assert(sizeof(((tinynv_sm_error_states_t *)0)->mmuFault.valid) == sizeof(NvBool),
               "the mmu fault valid flag is not the width NvBool is");

SAME_SIZE(tinynv_mmu_fault_entry_t, NV83DE_CTRL_DEBUG_READ_MMU_FAULT_INFO_ENTRY);
SAME_AT(tinynv_mmu_fault_entry_t, faultAddress, NV83DE_CTRL_DEBUG_READ_MMU_FAULT_INFO_ENTRY, faultAddress);
SAME_AT(tinynv_mmu_fault_entry_t, faultType, NV83DE_CTRL_DEBUG_READ_MMU_FAULT_INFO_ENTRY, faultType);
SAME_AT(tinynv_mmu_fault_entry_t, accessType, NV83DE_CTRL_DEBUG_READ_MMU_FAULT_INFO_ENTRY, accessType);
SAME_SIZE(tinynv_mmu_fault_info_t, NV83DE_CTRL_DEBUG_READ_MMU_FAULT_INFO_PARAMS);
SAME_AT(tinynv_mmu_fault_info_t, mmuFaultInfoList, NV83DE_CTRL_DEBUG_READ_MMU_FAULT_INFO_PARAMS, mmuFaultInfoList);
SAME_AT(tinynv_mmu_fault_info_t, count, NV83DE_CTRL_DEBUG_READ_MMU_FAULT_INFO_PARAMS, count);
_Static_assert(TINYNV_MMU_FAULT_ENTRIES == NV83DE_CTRL_DEBUG_READ_MMU_FAULT_INFO_MAX_ENTRIES,
               "the driver reads a different number of fault entries than the interface returns");
_Static_assert(TINYNV_DEBUG_MAX_SMS_PER_CALL == NV83DE_CTRL_DEBUG_MAX_SMS_PER_CALL,
               "the driver asks for a different number of SMs than one call carries");

// And the fault kinds this driver spells out in order to name them, against THIS architecture's header rather than a
// general one. A wrong number here is a fault report that says the confident opposite of what happened, which is worse
// than no report at all; writing them from an older architecture's header puts a confidential-computing violation
// under the name COMPRESSION_FAILURE, and this assertion is what caught that.
_Static_assert(TINYNV_PFAULT_TYPE_PDE == NV_PFAULT_FAULT_TYPE_PDE, "fault type PDE");
_Static_assert(TINYNV_PFAULT_TYPE_PDE_SIZE == NV_PFAULT_FAULT_TYPE_PDE_SIZE, "fault type PDE_SIZE");
_Static_assert(TINYNV_PFAULT_TYPE_PTE == NV_PFAULT_FAULT_TYPE_PTE, "fault type PTE");
_Static_assert(TINYNV_PFAULT_TYPE_VA_LIMIT_VIOLATION == NV_PFAULT_FAULT_TYPE_VA_LIMIT_VIOLATION, "fault type VA_LIMIT_VIOLATION");
_Static_assert(TINYNV_PFAULT_TYPE_UNBOUND_INST_BLOCK == NV_PFAULT_FAULT_TYPE_UNBOUND_INST_BLOCK, "fault type UNBOUND_INST_BLOCK");
_Static_assert(TINYNV_PFAULT_TYPE_PRIV_VIOLATION == NV_PFAULT_FAULT_TYPE_PRIV_VIOLATION, "fault type PRIV_VIOLATION");
_Static_assert(TINYNV_PFAULT_TYPE_RO_VIOLATION == NV_PFAULT_FAULT_TYPE_RO_VIOLATION, "fault type RO_VIOLATION");
_Static_assert(TINYNV_PFAULT_TYPE_WO_VIOLATION == NV_PFAULT_FAULT_TYPE_WO_VIOLATION, "fault type WO_VIOLATION");
_Static_assert(TINYNV_PFAULT_TYPE_PITCH_MASK_VIOLATION == NV_PFAULT_FAULT_TYPE_PITCH_MASK_VIOLATION, "fault type PITCH_MASK_VIOLATION");
_Static_assert(TINYNV_PFAULT_TYPE_WORK_CREATION == NV_PFAULT_FAULT_TYPE_WORK_CREATION, "fault type WORK_CREATION");
_Static_assert(TINYNV_PFAULT_TYPE_UNSUPPORTED_APERTURE == NV_PFAULT_FAULT_TYPE_UNSUPPORTED_APERTURE, "fault type UNSUPPORTED_APERTURE");
_Static_assert(TINYNV_PFAULT_TYPE_CC_VIOLATION == NV_PFAULT_FAULT_TYPE_CC_VIOLATION, "fault type CC_VIOLATION");
_Static_assert(TINYNV_PFAULT_TYPE_UNSUPPORTED_KIND == NV_PFAULT_FAULT_TYPE_UNSUPPORTED_KIND, "fault type UNSUPPORTED_KIND");
_Static_assert(TINYNV_PFAULT_TYPE_REGION_VIOLATION == NV_PFAULT_FAULT_TYPE_REGION_VIOLATION, "fault type REGION_VIOLATION");
_Static_assert(TINYNV_PFAULT_TYPE_POISONED == NV_PFAULT_FAULT_TYPE_POISONED, "fault type POISONED");
_Static_assert(TINYNV_PFAULT_TYPE_ATOMIC_VIOLATION == NV_PFAULT_FAULT_TYPE_ATOMIC_VIOLATION, "fault type ATOMIC_VIOLATION");
_Static_assert(TINYNV_PFAULT_ACCESS_READ == NV_PFAULT_ACCESS_TYPE_READ, "access type READ");
_Static_assert(TINYNV_PFAULT_ACCESS_WRITE == NV_PFAULT_ACCESS_TYPE_WRITE, "access type WRITE");
_Static_assert(TINYNV_PFAULT_ACCESS_ATOMIC == NV_PFAULT_ACCESS_TYPE_ATOMIC, "access type ATOMIC");
_Static_assert(TINYNV_PFAULT_ACCESS_PREFETCH == NV_PFAULT_ACCESS_TYPE_PREFETCH, "access type PREFETCH");
_Static_assert(TINYNV_PFAULT_ACCESS_ATOMIC_WEAK == NV_PFAULT_ACCESS_TYPE_VIRT_ATOMIC_WEAK, "access type ATOMIC_WEAK");
_Static_assert(TINYNV_PFAULT_ACCESS_PHYS_READ == NV_PFAULT_ACCESS_TYPE_PHYS_READ, "access type PHYS_READ");
_Static_assert(TINYNV_PFAULT_ACCESS_PHYS_WRITE == NV_PFAULT_ACCESS_TYPE_PHYS_WRITE, "access type PHYS_WRITE");
_Static_assert(TINYNV_PFAULT_ACCESS_PHYS_ATOMIC == NV_PFAULT_ACCESS_TYPE_PHYS_ATOMIC, "access type PHYS_ATOMIC");
_Static_assert(TINYNV_PFAULT_ACCESS_PHYS_PREFETCH == NV_PFAULT_ACCESS_TYPE_PHYS_PREFETCH, "access type PHYS_PREFETCH");

SAME_SIZE(tinynv_riscv_ucode_desc_t, RM_RISCV_UCODE_DESC);
SAME_AT(tinynv_riscv_ucode_desc_t, manifestOffset, RM_RISCV_UCODE_DESC, manifestOffset);
SAME_AT(tinynv_riscv_ucode_desc_t, monitorDataOffset, RM_RISCV_UCODE_DESC, monitorDataOffset);
SAME_AT(tinynv_riscv_ucode_desc_t, monitorCodeOffset, RM_RISCV_UCODE_DESC, monitorCodeOffset);

// Four more structures the driver builds are deliberately NOT asserted here: the queue element header, the message
// header, the system information block and the registry table. Their upstream headers cannot be included without
// NVIDIA's whole port layer (PORT_BREAKPOINT, PORT_IS_CHECKED_BUILD and the rest), and writing a shim to satisfy that
// would mean asserting my declarations against my own shim, which proves nothing.
//
// They are checked a stronger way instead. Both records the driver builds out of them are written into the shared ring
// and compared against the recorded boot byte for byte, by hash, in test_dev: the system information record at trace
// line 174 and the registry record at 178. A field at the wrong offset changes those bytes and the replay refuses them.
// An offsetof assertion says a field is where a header says; the recording says the GPU accepted it.

// the constants that move between driver branches, which is why tinygrad keeps 570, 580 and 610 autogens side by side
_Static_assert(GSP_FW_WPR_META_MAGIC == TINYNV_GSP_FW_WPR_META_MAGIC, "the wpr metadata magic changed");
_Static_assert(GSP_FW_WPR_META_REVISION == TINYNV_GSP_FW_WPR_META_REVISION, "the wpr metadata revision changed");
_Static_assert(NVDM_TYPE_COT == TINYNV_NVDM_TYPE_COT, "the chain of trust message type changed");
_Static_assert(GSP_DMA_TARGET_COHERENT_SYSTEM == TINYNV_GSP_DMA_TARGET_COHERENT_SYSTEM, "the dma target numbering changed");
_Static_assert(LIBOS_MEMORY_REGION_CONTIGUOUS == TINYNV_LIBOS_MEMORY_REGION_CONTIGUOUS, "the libos region kinds changed");
_Static_assert(LIBOS_MEMORY_REGION_LOC_SYSMEM == TINYNV_LIBOS_MEMORY_REGION_LOC_SYSMEM, "the libos region locations changed");
_Static_assert(LIBOS_MEMORY_REGION_RADIX_PAGE_LOG2 == TINYNV_LIBOS_MEMORY_REGION_RADIX_PAGE_LOG2, "the radix page size changed");

// The read-only allocation flags guarantee 7 reads, and the aliasing that makes them mean less than their names say.
// NVIDIA defines five pairs of NVOS32 alloc flags on THE SAME BIT, not on adjacent ones (nvos.h:1496-1507, three
// separate TODO BUGs), so the flags word is not a clean bitfield and a bit test cannot recover which name the
// caller meant. The two pairs guarantee 7 reads are at nvos.h:1498-1501:
//
//   0x04000000  SPARSE                      / USER_READ_ONLY              TODO BUG 2488682
//   0x08000000  DEVICE_READ_ONLY            / ALLOCATE_KERNEL_PRIVILEGED  TODO BUG 2488682
//
// These assert the aliasing ITSELF, which is the unusual direction: they fail on the day NVIDIA fixes BUG 2488682 and
// splits the pairs. That is a good day - it means the bits carry one meaning each and the caveat in nv_structs.h and
// in tinynv_rm_obj_writable can be deleted - but it also means these constants have moved, so the build should stop
// and make someone re-read rather than keep testing a bit that now means something else.
_Static_assert(TINYNV_ALLOC_FLAGS_USER_READ_ONLY == NVOS32_ALLOC_FLAGS_USER_READ_ONLY, "USER_READ_ONLY moved");
_Static_assert(TINYNV_ALLOC_FLAGS_DEVICE_READ_ONLY == NVOS32_ALLOC_FLAGS_DEVICE_READ_ONLY, "DEVICE_READ_ONLY moved");
_Static_assert(NVOS32_ALLOC_FLAGS_SPARSE == NVOS32_ALLOC_FLAGS_USER_READ_ONLY,
               "SPARSE and USER_READ_ONLY are no longer the same bit - BUG 2488682 may be fixed, so re-read nvos.h "
               "and drop the ambiguity caveat instead of assuming this constant still means read-only");
_Static_assert(NVOS32_ALLOC_FLAGS_ALLOCATE_KERNEL_PRIVILEGED == NVOS32_ALLOC_FLAGS_DEVICE_READ_ONLY,
               "DEVICE_READ_ONLY and ALLOCATE_KERNEL_PRIVILEGED are no longer the same bit - see above");

// The load-bearing fact behind the one disambiguation we DO get. SPARSE lives in the virtual-only mask, and all three
// physical allocation paths refuse anything in that mask (video_mem.c:1188, system_mem.c:562, egm_mem.c:316). Class
// 0x0040 is VideoMemory (resource_list.h:460), so a class-0x0040 object carrying 0x04000000 cannot be a sparse
// allocation: RM would have failed it. If SPARSE ever leaves this mask that argument is gone.
_Static_assert((NVOS32_ALLOC_FLAGS_VIRTUAL_ONLY & NVOS32_ALLOC_FLAGS_SPARSE) == NVOS32_ALLOC_FLAGS_SPARSE,
               "SPARSE left the virtual-only mask, so a physical allocation can carry it and bit 0x04000000 is "
               "ambiguous on class 0x0040 too");

// The PCF encodings this driver writes into a leaf entry. PCF is NOT a caching field - it encodes privilege,
// read-only, atomic and caching in one number - and until 2026-09-15 pt.c wrote `uncached ? 1 : 0`, which silently
// also chose REGULAR, READ-WRITE and ATOMIC for every entry ever programmed. That made guarantee 7's GRANT half
// impossible to express: a guest allowed a read-only mapping got a writable one.
//
// Asserted against the pinned tree rather than trusted, because a renumbering here does not fail anything else. The
// values come from gh100, which is what tools/gen_nv_regs.py already pins VER3 against; Blackwell does not redefine
// them, and gb202's own dev_mmu.h carries no PCF definitions at all.
_Static_assert(TINYNV_PCF_RW_CACHED == NV_MMU_VER3_PTE_PCF_REGULAR_RW_ATOMIC_CACHED_ACE, "PCF read-write cached moved");
_Static_assert(TINYNV_PCF_RW_UNCACHED == NV_MMU_VER3_PTE_PCF_REGULAR_RW_ATOMIC_UNCACHED_ACE,
               "PCF read-write uncached moved");
_Static_assert(TINYNV_PCF_RO_CACHED == NV_MMU_VER3_PTE_PCF_REGULAR_RO_ATOMIC_CACHED_ACE,
               "PCF READ-ONLY cached moved - if this fires, every read-only mapping this driver programs is writing "
               "some other encoding, and guarantee 7 grants more than it was asked for again");
_Static_assert(TINYNV_PCF_RO_UNCACHED == NV_MMU_VER3_PTE_PCF_REGULAR_RO_ATOMIC_UNCACHED_ACE,
               "PCF read-only uncached moved");
// And that read-only is not accidentally equal to read-write, which is the one way the four above could all pass
// while the distinction does nothing.
_Static_assert(NV_MMU_VER3_PTE_PCF_REGULAR_RO_ATOMIC_CACHED_ACE != NV_MMU_VER3_PTE_PCF_REGULAR_RW_ATOMIC_CACHED_ACE,
               "NVIDIA's read-only and read-write PCF encodings are the same value, so nothing this driver writes "
               "can distinguish them");

// Where the physical pages of a guest's allocation come from. The struct we send for 0x410103 against NVIDIA's.
SAME_SIZE(tinynv_nv0041_phys_attr_t, NV0041_CTRL_GET_SURFACE_PHYS_ATTR_PARAMS);
SAME_AT(tinynv_nv0041_phys_attr_t, memOffset, NV0041_CTRL_GET_SURFACE_PHYS_ATTR_PARAMS, memOffset);
SAME_AT(tinynv_nv0041_phys_attr_t, memAperture, NV0041_CTRL_GET_SURFACE_PHYS_ATTR_PARAMS, memAperture);
SAME_AT(tinynv_nv0041_phys_attr_t, mmuContext, NV0041_CTRL_GET_SURFACE_PHYS_ATTR_PARAMS, mmuContext);
SAME_AT(tinynv_nv0041_phys_attr_t, contigSegmentSize, NV0041_CTRL_GET_SURFACE_PHYS_ATTR_PARAMS, contigSegmentSize);
_Static_assert(TINYNV_CTRL_CMD_GET_SURFACE_PHYS_ATTR == NV0041_CTRL_CMD_GET_SURFACE_PHYS_ATTR,
               "the surface-physical-attributes control id moved");

int main(void) {
  // Counted from this file at build time, not typed here. They used to be the literals 13 and 32, and they had stopped
  // being true: the line reported the same two numbers whatever the file contained, which is the one thing a summary
  // must not do. A number in a message that does not track what it describes is the same failure as a comment that is
  // true for a reason it does not give - it reads as evidence and is decoration.
  printf("headers: %zu structures, %zu fields and %zu constants checked against the pinned tree - wpr metadata %zu"
         " bytes, chain of trust message %zu, fmc boot parameters %zu, queue header %zu, libos region %zu, riscv"
         " descriptor %zu, sm error states %zu, mmu fault info %zu\n",
         (size_t)TINYNV_HDR_STRUCTS, (size_t)TINYNV_HDR_FIELDS, (size_t)TINYNV_HDR_CONSTS,
         sizeof(GspFwWprMeta), sizeof(NVDM_PAYLOAD_COT), sizeof(GSP_FMC_BOOT_PARAMS),
         sizeof(msgqTxHeader), sizeof(LibosMemoryRegionInitArgument), sizeof(RM_RISCV_UCODE_DESC),
         sizeof(NV83DE_CTRL_DEBUG_READ_ALL_SM_ERROR_STATES_PARAMS), sizeof(NV83DE_CTRL_DEBUG_READ_MMU_FAULT_INFO_PARAMS));
  return 0;
}
