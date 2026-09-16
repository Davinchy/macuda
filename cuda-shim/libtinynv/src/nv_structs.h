// The structures the GPU's firmware reads out of host memory during boot, declared so the driver can build them.
//
// These are NVIDIA's, from open-gpu-kernel-modules at the pinned commit. They are re-declared here rather than included
// so the library still builds on a machine that has not fetched the 108 MB tree; test/test_headers.c includes the real
// headers and asserts every size and offset used here against them, so a re-declaration that drifts breaks the build
// rather than the boot. Offsets are pinned below with static assertions for the same reason: a compiler that lays one of
// these out differently must fail here, not hand the firmware a structure it will read as garbage and refuse to sign.
#ifndef TINYNV_NV_STRUCTS_H
#define TINYNV_NV_STRUCTS_H
#include <stdint.h>
#include <stddef.h>

#define TINYNV_AT(t, f, n) _Static_assert(offsetof(t, f) == (n), #t "." #f " is not where the oracle puts it")
#define TINYNV_SIZE(t, n) _Static_assert(sizeof(t) == (n), #t " is not the size the oracle gives it")

// how the firmware should reach an address we hand it. everything the boot path passes lives in coherent host memory.
#define TINYNV_GSP_DMA_TARGET_COHERENT_SYSTEM 1

#define TINYNV_NVDM_TYPE_COT 0x14

// --- the chain of trust message -----------------------------------------------------------------------------------
// Sent to the FSP to authorise the GSP firmware: where the boot image is, what it hashes to, and who signed it. Packed,
// because the FSP reads it off the wire with no padding.
#pragma pack(push, 1)
typedef struct {
  uint16_t version, size;
  uint64_t gspFmcSysmemOffset, frtsSysmemOffset;
  uint32_t frtsSysmemSize;
  uint64_t frtsVidmemOffset; // counted from the end of video memory, not the start
  uint32_t frtsVidmemSize;
  uint32_t hash384[12], publicKey[96], signature[96];
  uint64_t gspBootArgsSysmemOffset;
} tinynv_cot_payload_t;
#pragma pack(pop)
TINYNV_SIZE(tinynv_cot_payload_t, 860);
TINYNV_AT(tinynv_cot_payload_t, gspFmcSysmemOffset, 4);
TINYNV_AT(tinynv_cot_payload_t, frtsVidmemOffset, 24);
TINYNV_AT(tinynv_cot_payload_t, hash384, 36);
TINYNV_AT(tinynv_cot_payload_t, publicKey, 84);
TINYNV_AT(tinynv_cot_payload_t, signature, 468);
TINYNV_AT(tinynv_cot_payload_t, gspBootArgsSysmemOffset, 852);

// --- what the boot firmware is told when it starts ------------------------------------------------------------------
typedef struct { uint32_t regkeys; } tinynv_gsp_fmc_init_params_t;

typedef struct {
  uint32_t target;        // which aperture gspRmDescOffset is in
  uint32_t gspRmDescSize;
  uint64_t gspRmDescOffset;
  uint64_t wprCarveoutOffset;
  uint32_t wprCarveoutSize;
  uint8_t bIsGspRmBoot;
  uint8_t _pad[3];
} tinynv_gsp_acr_boot_params_t;
TINYNV_SIZE(tinynv_gsp_acr_boot_params_t, 32);
TINYNV_AT(tinynv_gsp_acr_boot_params_t, gspRmDescOffset, 8);
TINYNV_AT(tinynv_gsp_acr_boot_params_t, bIsGspRmBoot, 28);

typedef struct { uint32_t target; uint32_t _pad; uint64_t bootArgsOffset; } tinynv_gsp_rm_params_t;
TINYNV_SIZE(tinynv_gsp_rm_params_t, 16);
TINYNV_AT(tinynv_gsp_rm_params_t, bootArgsOffset, 8);

typedef struct { uint32_t target; uint32_t _pad; uint64_t payloadBufferOffset; uint32_t payloadBufferSize; uint32_t _pad2; }
  tinynv_gsp_spdm_params_t;
TINYNV_SIZE(tinynv_gsp_spdm_params_t, 24);

typedef struct {
  tinynv_gsp_fmc_init_params_t initParams;
  uint32_t _pad;
  tinynv_gsp_acr_boot_params_t bootGspRmParams;
  tinynv_gsp_rm_params_t gspRmParams;
  tinynv_gsp_spdm_params_t gspSpdmParams;
} tinynv_gsp_fmc_boot_params_t;
TINYNV_SIZE(tinynv_gsp_fmc_boot_params_t, 80);
TINYNV_AT(tinynv_gsp_fmc_boot_params_t, bootGspRmParams, 8);
TINYNV_AT(tinynv_gsp_fmc_boot_params_t, gspRmParams, 40);
TINYNV_AT(tinynv_gsp_fmc_boot_params_t, gspSpdmParams, 56);

// --- where GSP-RM's own firmware is -------------------------------------------------------------------------------
// The descriptor the boot firmware reads to find the rest of GSP-RM. NVIDIA declares several of these fields in a union;
// this is the arm the boot path fills, and the rest is zero, which is what the oracle writes.
#define TINYNV_GSP_FW_WPR_META_MAGIC 0xdc3aae21371a60b3ull
#define TINYNV_GSP_FW_WPR_META_REVISION 1
typedef struct {
  uint64_t magic, revision;
  uint64_t sysmemAddrOfRadix3Elf, sizeOfRadix3Elf;
  uint64_t sysmemAddrOfBootloader, sizeOfBootloader;
  uint64_t bootloaderCodeOffset, bootloaderDataOffset, bootloaderManifestOffset;
  uint64_t sysmemAddrOfSignature, sizeOfSignature;
  uint64_t gspFwRsvdStart;
  uint64_t nonWprHeapOffset, nonWprHeapSize;
  uint64_t gspFwWprStart;
  uint64_t gspFwHeapOffset, gspFwHeapSize;
  uint64_t gspFwOffset;
  uint64_t bootBinOffset;
  uint64_t frtsOffset, frtsSize;
  uint64_t gspFwWprEnd, fbSize;
  uint64_t vgaWorkspaceOffset, vgaWorkspaceSize;
  uint64_t bootCount;
  uint64_t partitionRpcAddr;
  uint16_t partitionRpcRequestOffset, partitionRpcReplyOffset;
  uint32_t elfCodeOffset, elfDataOffset, elfCodeSize, elfDataSize, lsUcodeVersion;
  uint8_t gspFwHeapVfPartitionCount, flags;
  uint16_t padding;
  uint32_t pmuReservedSize;
  uint64_t verified;
} tinynv_wpr_meta_t;
TINYNV_SIZE(tinynv_wpr_meta_t, 256);
TINYNV_AT(tinynv_wpr_meta_t, sysmemAddrOfRadix3Elf, 16);
TINYNV_AT(tinynv_wpr_meta_t, sysmemAddrOfBootloader, 32);
TINYNV_AT(tinynv_wpr_meta_t, bootloaderCodeOffset, 48);
TINYNV_AT(tinynv_wpr_meta_t, sysmemAddrOfSignature, 72);
TINYNV_AT(tinynv_wpr_meta_t, nonWprHeapSize, 104);
TINYNV_AT(tinynv_wpr_meta_t, gspFwHeapSize, 128);
TINYNV_AT(tinynv_wpr_meta_t, frtsSize, 160);
TINYNV_AT(tinynv_wpr_meta_t, vgaWorkspaceSize, 192);
TINYNV_AT(tinynv_wpr_meta_t, pmuReservedSize, 244);
TINYNV_AT(tinynv_wpr_meta_t, verified, 248);

// --- the message queue GSP-RM answers on --------------------------------------------------------------------------
typedef struct {
  uint32_t version, size, msgSize, msgCount, writePtr, flags, rxHdrOff, entryOff;
} tinynv_msgq_tx_header_t;
TINYNV_SIZE(tinynv_msgq_tx_header_t, 32);
TINYNV_AT(tinynv_msgq_tx_header_t, writePtr, 16);
TINYNV_AT(tinynv_msgq_tx_header_t, entryOff, 28);

typedef struct {
  uint64_t sharedMemPhysAddr;
  uint32_t pageTableEntryCount, _pad;
  uint64_t cmdQueueOffset, statQueueOffset;
} tinynv_msgq_init_args_t;
TINYNV_SIZE(tinynv_msgq_init_args_t, 32);
TINYNV_AT(tinynv_msgq_init_args_t, cmdQueueOffset, 16);

typedef struct {
  tinynv_msgq_init_args_t messageQueueInitArguments;
  uint8_t srInitArguments[12];
  uint32_t gpuInstance;
  uint8_t bDmemStack, _pad[7];
  uint64_t profilerArgs[2];
} tinynv_gsp_args_cached_t;
TINYNV_SIZE(tinynv_gsp_args_cached_t, 72);
TINYNV_AT(tinynv_gsp_args_cached_t, gpuInstance, 44);
TINYNV_AT(tinynv_gsp_args_cached_t, bDmemStack, 48);
TINYNV_AT(tinynv_gsp_args_cached_t, profilerArgs, 56);

// --- the memory regions the GSP operating system is handed ---------------------------------------------------------
#define TINYNV_LIBOS_MEMORY_REGION_CONTIGUOUS 1
#define TINYNV_LIBOS_MEMORY_REGION_LOC_SYSMEM 1
#define TINYNV_LIBOS_MEMORY_REGION_RADIX_PAGE_LOG2 12
typedef struct {
  uint64_t id8, pa, size;
  uint8_t kind, loc, _pad[6];
} tinynv_libos_region_t;
TINYNV_SIZE(tinynv_libos_region_t, 32);
TINYNV_AT(tinynv_libos_region_t, pa, 8);
TINYNV_AT(tinynv_libos_region_t, kind, 24);
TINYNV_AT(tinynv_libos_region_t, loc, 25);

// --- the messages the driver and GSP-RM exchange -------------------------------------------------------------------
// Each message sits in one or more page-sized slots of a ring. An element header carries the sequence number, the slot
// count and a checksum over the whole record; inside it is the message header, and after that the payload.
#define TINYNV_MSG_SIGNATURE_VALID 0x43505256u // "VRPC" read the other way round
#define TINYNV_MSG_RESULT_PENDING 0xffffffffu
#define TINYNV_MSG_HEADER_VERSION (3u << 24)

#define TINYNV_MSG_FUNCTION_CONTINUATION_RECORD 71
#define TINYNV_MSG_FUNCTION_GSP_SET_SYSTEM_INFO 72
#define TINYNV_MSG_FUNCTION_SET_REGISTRY 73
#define TINYNV_MSG_EVENT_GSP_INIT_DONE 4097
#define TINYNV_MSG_EVENT_GSP_RUN_CPU_SEQUENCER 4098
#define TINYNV_MSG_EVENT_MMU_FAULT_QUEUED 4101
#define TINYNV_MSG_EVENT_OS_ERROR_LOG 4102

typedef struct {
  uint8_t authTagBuffer[16], aadBuffer[16]; // confidential computing only; zero here
  uint32_t checkSum, seqNum, elemCount, padding;
} tinynv_msg_element_t;
TINYNV_SIZE(tinynv_msg_element_t, 48);
TINYNV_AT(tinynv_msg_element_t, checkSum, 32);
TINYNV_AT(tinynv_msg_element_t, seqNum, 36);
TINYNV_AT(tinynv_msg_element_t, elemCount, 40);

typedef struct {
  uint32_t header_version, signature, length, function, rpc_result, rpc_result_private, sequence, u;
} tinynv_msg_header_t;
TINYNV_SIZE(tinynv_msg_header_t, 32);
TINYNV_AT(tinynv_msg_header_t, length, 8);
TINYNV_AT(tinynv_msg_header_t, function, 12);
TINYNV_AT(tinynv_msg_header_t, rpc_result, 16);

// What GSP-RM is told about the machine it is running in. Only the fields the driver fills are named; the rest is a hole
// that stays zero, and test_headers.c checks each named one against NVIDIA's own definition.
typedef struct {
  uint64_t gpuPhysAddr, gpuPhysFbAddr, gpuPhysInstAddr, gpuPhysIoAddr;
  uint64_t nvDomainBusDeviceFunc;
  uint64_t _hole40[4];
  uint64_t maxUserVa;
  uint32_t pciConfigMirrorBase, pciConfigMirrorSize;
  uint32_t PCIDeviceID, PCISubDeviceID, PCIRevisionID;
  uint8_t _hole100[840 - 100];
  uint8_t bIsPassthru;
  uint8_t _hole841[928 - 841];
} tinynv_gsp_system_info_t;
TINYNV_SIZE(tinynv_gsp_system_info_t, 928);
TINYNV_AT(tinynv_gsp_system_info_t, nvDomainBusDeviceFunc, 32);
TINYNV_AT(tinynv_gsp_system_info_t, maxUserVa, 72);
TINYNV_AT(tinynv_gsp_system_info_t, pciConfigMirrorBase, 80);
TINYNV_AT(tinynv_gsp_system_info_t, PCIDeviceID, 88);
TINYNV_AT(tinynv_gsp_system_info_t, PCIRevisionID, 96);
TINYNV_AT(tinynv_gsp_system_info_t, bIsPassthru, 840);

// The driver's registry, which is how a handful of behaviours are turned on inside GSP-RM.
#define TINYNV_REGISTRY_TYPE_DWORD 1
typedef struct { uint32_t size, numEntries; } tinynv_registry_table_t;
TINYNV_SIZE(tinynv_registry_table_t, 8);
typedef struct { uint32_t nameOffset; uint8_t type, _pad[3]; uint32_t data, length; } tinynv_registry_entry_t;
TINYNV_SIZE(tinynv_registry_entry_t, 16);
TINYNV_AT(tinynv_registry_entry_t, type, 4);
TINYNV_AT(tinynv_registry_entry_t, data, 8);
TINYNV_AT(tinynv_registry_entry_t, length, 12);

// --- the resource manager's objects ---------------------------------------------------------------------------------
// Everything the GPU does is an object in a tree inside GSP-RM: a client at the root, a device under it, a subdevice for
// the physical chip, an address space, channels. The driver creates them by remote call and refers to them by handle.
#define TINYNV_RM_PRIV_ROOT 0xc1e00004u // the handle GSP-RM expects the privileged client to use
#define TINYNV_RM_FIRST_HANDLE 0xcf000000u

#define TINYNV_MSG_FUNCTION_GSP_RM_CONTROL 76
// Read from NVIDIA's own rpc_global_enums.h in the pinned tree rather than guessed: X(RM, FREE, 10).
#define TINYNV_MSG_FUNCTION_FREE 10
#define TINYNV_MSG_FUNCTION_GSP_RM_ALLOC 103

#define TINYNV_CLASS_ROOT 0x0
#define TINYNV_CLASS_DEVICE 0x80
#define TINYNV_CLASS_SUBDEVICE 0x2080
#define TINYNV_CLASS_VASPACE 0x90f1

// Freeing one object. NVOS00_PARAMETERS in NVIDIA's nvos.h, and the whole of it: an object is named by the client it
// belongs to, its parent, and itself. Nothing else crosses, which is why a guest can ask for a free without either side
// keeping a translation table - the handles mean the same thing on both.
// Sensors, as the card reports them.
//
// GSP-RM writes this structure into host memory we allocate and hand it, then refreshes it on a timer of our choosing.
// So reading a temperature costs a load rather than a message, which matters here more than it would elsewhere: on this
// backend every register read is a round trip to another process.
//
// Transcribed from class/cl00de.h in the pinned tree. The whole structure is here rather than the four fields anyone
// cares about, because the offsets of those four depend on everything in front of them - and every size and offset the
// driver uses is asserted against the real header in test_headers.c, which is the only reason transcribing it is safe.
//
// NOT here, because NVIDIA does not publish it: fan speed. ctrl2080fan.h and ctrl2080thermal.h ship as empty stubs, and
// there is no fan field anywhere in this structure. Temperature, power, clocks and utilisation are the whole of what a
// card will tell us.
#define TINYNV_CLASS_USER_SHARED_DATA 0x00de

// Which groups the firmware should poll. Passed at allocation and again in the poll control.
#define TINYNV_RUSD_POLL_CLOCK   0x1
#define TINYNV_RUSD_POLL_PERF    0x2
#define TINYNV_RUSD_POLL_MEMORY  0x4
#define TINYNV_RUSD_POLL_POWER   0x8
#define TINYNV_RUSD_POLL_THERMAL 0x10
#define TINYNV_RUSD_POLL_PCI     0x20

// Hand the firmware the buffer, and tell it what to keep fresh. Both are subdevice controls, which is a thing this
// driver already does ten of.
// What GSP-RM has to allocate from, which this driver has never asked and has been reasoning about.
//
// Session C bisected a guest's virtual reservation to a ceiling of exactly 2 MiB - which is one 4 KB page table's
// worth of coverage at the small page size, 512 entries of eight bytes. That arithmetic says the thing backing a
// reservation gets one page and cannot get a second, and the obvious explanation is that GSP-RM has nothing to
// allocate page tables from, because nothing in our bring-up ever gives it a heap and nothing has ever needed one:
// libtinynv allocates every byte itself and hands over descriptors.
//
// That explanation is an inference. This is the control that measures it instead.
#define TINYNV_CTRL_GET_GSP_RM_FREE_HEAP 0x20800aeb
typedef struct { uint64_t freeHeapSize; } tinynv_gsp_free_heap_t;
TINYNV_SIZE(tinynv_gsp_free_heap_t, 8);

#define TINYNV_CTRL_INIT_USER_SHARED_DATA 0x20800afe
#define TINYNV_CTRL_USER_SHARED_DATA_POLL 0x20800aff
typedef struct { uint64_t physAddr; } tinynv_rusd_init_t;
typedef struct { uint64_t polledDataMask; uint32_t pollFrequencyMs; } tinynv_rusd_poll_t;
typedef struct { uint64_t polledDataMask; } tinynv_rusd_alloc_t;

// Each field is a timestamp followed by its contents. Read the timestamp, copy, read it again: if it changed, the
// firmware was writing while we read and it has to be retried. A timestamp of zero means the reading is not available
// on this platform, which is how we will find out whether a card in an enclosure reports temperature at all rather
// than assuming it does.
#define TINYNV_RUSD_TS_INVALID  0ull
#define TINYNV_RUSD_TS_WRITING  0xffffffffffffffffull
#define TINYNV_RUSD_SEQ_START   0xff00000000000000ull

typedef struct { uint64_t ts; uint32_t bar1Size, bar1AvailSize; } tinynv_rusd_bar1_t;
typedef struct { uint64_t ts; uint64_t totalPmaMemory, freePmaMemory; } tinynv_rusd_pma_t;
typedef struct { uint64_t ts; uint32_t shadowErrContVal; } tinynv_rusd_errcont_t;
typedef struct { uint64_t ts; uint8_t bCtxswLoggingEnabled; } tinynv_rusd_gr_t;
typedef struct { uint64_t ts; uint32_t targetClkMHz[4]; } tinynv_rusd_clk_t;   // graphics, memory, video, sm
typedef struct { uint64_t ts; uint32_t reasonMask; } tinynv_rusd_throttle_t;
typedef struct { uint32_t clkPercentBusy, samplingPeriodUs; } tinynv_rusd_engutil_t;
typedef struct { uint64_t ts; uint32_t gpuPercentBusy, memoryPercentBusy; tinynv_rusd_engutil_t engUtil[4]; }
  tinynv_rusd_util_t;
typedef struct { uint64_t ts; uint32_t currentPstate; } tinynv_rusd_pstate_t;
typedef struct { uint64_t correctedVolatile, correctedAggregate, uncorrectedVolatile, uncorrectedAggregate; }
  tinynv_rusd_memerr_t;
typedef struct { uint64_t ts; tinynv_rusd_memerr_t count[3]; } tinynv_rusd_ecc_t;
typedef struct { uint64_t ts; uint32_t requestedmW, enforcedmW; } tinynv_rusd_powerlimit_t;
// Signed fixed point with eight fractional bits, so 256ths of a degree celsius. NvTemp in nvfixedtypes.h.
typedef struct { uint64_t ts; int32_t temperature; } tinynv_rusd_temp_t;
typedef struct { uint32_t histogramMax, histogramHigh, histogramPartial, histogramLow, histogramNone,
                 correctableRows, uncorrectableRows; uint8_t isPending, hasFailureOccurred; } tinynv_rusd_remapinfo_t;
typedef struct { uint64_t ts; tinynv_rusd_remapinfo_t info; } tinynv_rusd_remap_t;
typedef struct { uint64_t ts; uint32_t averageGpuPower, averageModulePower, averageMemoryPower; } tinynv_rusd_avgpwr_t;
typedef struct { uint64_t ts; uint32_t instGpuPower, instModulePower, instCpuPower; } tinynv_rusd_instpwr_t;
typedef struct { uint64_t ts; uint32_t data[9]; } tinynv_rusd_pcie_t;

typedef struct {
  tinynv_rusd_bar1_t bar1MemoryInfo;
  tinynv_rusd_pma_t pmaMemoryInfo;
  tinynv_rusd_errcont_t shadowErrCont;
  tinynv_rusd_gr_t grInfo;
  tinynv_rusd_clk_t clkPublicDomainInfos;
  tinynv_rusd_throttle_t clkThrottleReason;
  tinynv_rusd_util_t perfDevUtil;
  tinynv_rusd_ecc_t memEcc;
  tinynv_rusd_pstate_t perfCurrentPstate;
  tinynv_rusd_powerlimit_t powerLimitGpu;
  tinynv_rusd_temp_t temperatures[2];   // gpu, then memory
  tinynv_rusd_remap_t memRowRemap;
  tinynv_rusd_avgpwr_t avgPowerUsage;
  tinynv_rusd_instpwr_t instPowerUsage;
  tinynv_rusd_pcie_t pciBusData;
} tinynv_rusd_t;

typedef struct { uint32_t hRoot, hObjectParent, hObjectOld, status; } tinynv_rpc_rm_free_t;
TINYNV_SIZE(tinynv_rpc_rm_free_t, 16);
TINYNV_AT(tinynv_rpc_rm_free_t, hObjectOld, 8);

typedef struct { uint32_t hClient, hParent, hObject, hClass, status, paramsSize, flags, reserved; } tinynv_rpc_rm_alloc_t;
TINYNV_SIZE(tinynv_rpc_rm_alloc_t, 32);
TINYNV_AT(tinynv_rpc_rm_alloc_t, hObject, 8);
TINYNV_AT(tinynv_rpc_rm_alloc_t, paramsSize, 20);

typedef struct { uint32_t hClient, hObject, cmd, status, paramsSize, flags; } tinynv_rpc_rm_control_t;
TINYNV_SIZE(tinynv_rpc_rm_control_t, 24);
TINYNV_AT(tinynv_rpc_rm_control_t, cmd, 8);
TINYNV_AT(tinynv_rpc_rm_control_t, paramsSize, 16);

typedef struct { uint32_t hClient, processID; char processName[100]; uint32_t _pad; uint64_t pOsPidInfo; } tinynv_nv0000_alloc_t;
TINYNV_SIZE(tinynv_nv0000_alloc_t, 120);
TINYNV_AT(tinynv_nv0000_alloc_t, processName, 8);
TINYNV_AT(tinynv_nv0000_alloc_t, pOsPidInfo, 112);

typedef struct {
  uint32_t deviceId, hClientShare, hTargetClient, hTargetDevice, flags, _pad;
  uint64_t vaSpaceSize, vaStartInternal, vaLimitInternal;
  uint32_t vaMode, _pad2;
} tinynv_nv0080_alloc_t;
TINYNV_SIZE(tinynv_nv0080_alloc_t, 56);
TINYNV_AT(tinynv_nv0080_alloc_t, hClientShare, 4);
TINYNV_AT(tinynv_nv0080_alloc_t, vaSpaceSize, 24);
TINYNV_AT(tinynv_nv0080_alloc_t, vaMode, 48);

typedef struct { uint32_t subDeviceId; } tinynv_nv2080_alloc_t;
TINYNV_SIZE(tinynv_nv2080_alloc_t, 4);

typedef struct {
  uint32_t index, flags;
  uint64_t vaSize, vaStartInternal, vaLimitInternal;
  uint32_t bigPageSize, _pad;
  uint64_t vaBase;
} tinynv_vaspace_alloc_t;
TINYNV_SIZE(tinynv_vaspace_alloc_t, 48);
TINYNV_AT(tinynv_vaspace_alloc_t, vaSize, 8);
TINYNV_AT(tinynv_vaspace_alloc_t, bigPageSize, 32);
TINYNV_AT(tinynv_vaspace_alloc_t, vaBase, 40);

// The engines the chip has, and which runlist each is on. A channel has to be created against the right runlist, so this
// table is the first thing the driver asks GSP-RM about once the object tree exists.
#define TINYNV_CTRL_CMD_FIFO_GET_DEVICE_INFO_TABLE 0x20801112u
#define TINYNV_FIFO_DEVICE_ENTRIES 32
#define TINYNV_FIFO_ENGINE_DATA 16
typedef struct {
  uint32_t engineData[TINYNV_FIFO_ENGINE_DATA];
  uint32_t pbdmaIds[2], pbdmaFaultIds[2], numPbdmas;
  char engineName[16];
} tinynv_fifo_device_entry_t;
TINYNV_SIZE(tinynv_fifo_device_entry_t, 100);
TINYNV_AT(tinynv_fifo_device_entry_t, pbdmaIds, 64);
TINYNV_AT(tinynv_fifo_device_entry_t, numPbdmas, 80);
TINYNV_AT(tinynv_fifo_device_entry_t, engineName, 84);

typedef struct {
  uint32_t baseIndex, numEntries;
  uint8_t bMore, _pad[3];
  tinynv_fifo_device_entry_t entries[TINYNV_FIFO_DEVICE_ENTRIES];
} tinynv_fifo_device_info_t;
TINYNV_SIZE(tinynv_fifo_device_info_t, 3212);
TINYNV_AT(tinynv_fifo_device_info_t, entries, 12);

// The tables the driver has already built, handed to GSP-RM so that both sides walk the same tree from the start rather
// than discovering each other's later.
#define TINYNV_CTRL_CMD_VASPACE_COPY_SERVER_RESERVED_PDES 0x90f10106u
#define TINYNV_MAX_PDE_LEVELS 6
typedef struct { uint64_t physAddress, size; uint32_t aperture; uint8_t pageShift, _pad[3]; } tinynv_pde_level_t;
TINYNV_SIZE(tinynv_pde_level_t, 24);
TINYNV_AT(tinynv_pde_level_t, aperture, 16);
TINYNV_AT(tinynv_pde_level_t, pageShift, 20);

typedef struct {
  uint32_t hSubDevice, subDeviceId;
  uint64_t pageSize, virtAddrLo, virtAddrHi;
  uint32_t numLevelsToCopy, _pad;
  tinynv_pde_level_t levels[TINYNV_MAX_PDE_LEVELS];
} tinynv_copy_pdes_t;
TINYNV_SIZE(tinynv_copy_pdes_t, 184);
TINYNV_AT(tinynv_copy_pdes_t, pageSize, 8);
TINYNV_AT(tinynv_copy_pdes_t, numLevelsToCopy, 32);
TINYNV_AT(tinynv_copy_pdes_t, levels, 40);

// --- a channel ------------------------------------------------------------------------------------------------------
// Work reaches the GPU through a channel: a ring of pointers to command buffers, plus several small areas GSP-RM keeps
// the channel's state in. The classes are per architecture; these are Blackwell's.
#define TINYNV_CLASS_GPFIFO 0xc96f
#define TINYNV_CLASS_COMPUTE 0xcec0
#define TINYNV_CLASS_DMA_COPY 0xcab5

typedef struct { uint64_t base, size; uint32_t addressSpace, cacheAttrib; } tinynv_memory_desc_t;
TINYNV_SIZE(tinynv_memory_desc_t, 24);
TINYNV_AT(tinynv_memory_desc_t, addressSpace, 16);

typedef struct {
  uint32_t hObjectError, hObjectBuffer;
  uint64_t gpFifoOffset;
  uint32_t gpFifoEntries, flags, hContextShare, hVASpace;
  uint32_t hUserdMemory[8];
  uint64_t userdOffset[8];
  uint32_t engineType, cid, subDeviceId, hObjectEccError;
  tinynv_memory_desc_t instanceMem, userdMem, ramfcMem, mthdbufMem;
  uint32_t hPhysChannelGroup, internalFlags;
  tinynv_memory_desc_t errorNotifierMem, eccErrorNotifierMem;
  uint32_t ProcessID, SubProcessID;
  uint8_t encryptIv[12], decryptIv[12], hmacNonce[32];
  uint32_t tpcConfigID;
  uint32_t _pad;
} tinynv_gpfifo_alloc_t;
TINYNV_SIZE(tinynv_gpfifo_alloc_t, 368);
TINYNV_AT(tinynv_gpfifo_alloc_t, gpFifoOffset, 8);
TINYNV_AT(tinynv_gpfifo_alloc_t, hVASpace, 28);
TINYNV_AT(tinynv_gpfifo_alloc_t, hUserdMemory, 32);
TINYNV_AT(tinynv_gpfifo_alloc_t, userdOffset, 64);
TINYNV_AT(tinynv_gpfifo_alloc_t, engineType, 128);
TINYNV_AT(tinynv_gpfifo_alloc_t, instanceMem, 144);
TINYNV_AT(tinynv_gpfifo_alloc_t, userdMem, 168);
TINYNV_AT(tinynv_gpfifo_alloc_t, ramfcMem, 192);
TINYNV_AT(tinynv_gpfifo_alloc_t, mthdbufMem, 216);
TINYNV_AT(tinynv_gpfifo_alloc_t, internalFlags, 244);
TINYNV_AT(tinynv_gpfifo_alloc_t, errorNotifierMem, 248);
TINYNV_AT(tinynv_gpfifo_alloc_t, ProcessID, 296);
TINYNV_AT(tinynv_gpfifo_alloc_t, tpcConfigID, 360);

// --- the graphics context ---------------------------------------------------------------------------------------------
// A channel that runs compute needs a set of buffers the graphics engine keeps its own state in: the context itself, a
// patch buffer, and a handful of configuration areas. GSP-RM says how big each must be and how it must be aligned, and
// the driver allocates them and hands back where they are. Some are wanted by physical address, some by virtual, some by
// both, which is what the flags on each entry say.
#define TINYNV_CTRL_CMD_STATIC_KGR_GET_CONTEXT_BUFFERS_INFO 0x20800a32u
#define TINYNV_CTRL_CMD_GPU_PROMOTE_CTX 0x2080012bu
#define TINYNV_GR_CTX_BUFFERS 26

typedef struct { uint32_t size, alignment; } tinynv_ctx_buffer_info_t;
TINYNV_SIZE(tinynv_ctx_buffer_info_t, 8);
typedef struct { tinynv_ctx_buffer_info_t engine[TINYNV_GR_CTX_BUFFERS]; } tinynv_gr_ctx_sizes_t;
TINYNV_SIZE(tinynv_gr_ctx_sizes_t, 208);
typedef struct { tinynv_gr_ctx_sizes_t engineContextBuffersInfo[8]; } tinynv_kgr_ctx_info_t;
TINYNV_SIZE(tinynv_kgr_ctx_info_t, 1664);

typedef struct {
  uint64_t gpuPhysAddr, gpuVirtAddr, size;
  uint32_t physAttr;
  uint16_t bufferId;
  uint8_t bInitialize, bNonmapped;
} tinynv_promote_entry_t;
TINYNV_SIZE(tinynv_promote_entry_t, 32);
TINYNV_AT(tinynv_promote_entry_t, physAttr, 24);
TINYNV_AT(tinynv_promote_entry_t, bufferId, 28);
TINYNV_AT(tinynv_promote_entry_t, bInitialize, 30);
TINYNV_AT(tinynv_promote_entry_t, bNonmapped, 31);

#define TINYNV_MAX_PROMOTE_ENTRIES 16
typedef struct {
  uint32_t engineType, hClient, ChID, hChanClient, hObject, hVirtMemory;
  uint64_t virtAddress, size;
  uint32_t entryCount, _pad;
  tinynv_promote_entry_t promoteEntry[TINYNV_MAX_PROMOTE_ENTRIES];
} tinynv_promote_ctx_t;
TINYNV_SIZE(tinynv_promote_ctx_t, 560);
TINYNV_AT(tinynv_promote_ctx_t, virtAddress, 24);
TINYNV_AT(tinynv_promote_ctx_t, entryCount, 40);
TINYNV_AT(tinynv_promote_ctx_t, promoteEntry, 48);

// --- the client the driver actually works through -----------------------------------------------------------------
// The objects above belong to a privileged client used to initialise the graphics context. Everything the driver does
// afterwards belongs to a second, ordinary client with its own device, address space and channel group. Two clients,
// because GSP-RM treats the privileged one specially and the work we submit is not privileged.
#define TINYNV_RM_USER_ROOT 0xc1000000u
#define TINYNV_CLASS_MEMORY_VIRTUAL 0x70
#define TINYNV_CLASS_CHANNEL_GROUP 0xa06c
#define TINYNV_CLASS_CONTEXT_SHARE 0x9067

#define TINYNV_CTRL_CMD_PERF_BOOST 0x2080200au
#define TINYNV_PERF_BOOST_CUDA_YES 1
#define TINYNV_PERF_BOOST_CUDA_PRIORITY_HIGH 1
#define TINYNV_PERF_BOOST_CMD_BOOST_TO_MAX 2

#define TINYNV_VASPACE_FLAGS_ENABLE_PAGE_FAULTING 64
#define TINYNV_VASPACE_FLAGS_IS_EXTERNALLY_OWNED 8
#define TINYNV_CTXSHARE_FLAGS_SUBCONTEXT_ASYNC 1
#define TINYNV_ENGINE_TYPE_GRAPHICS 1

typedef struct { uint64_t offset, limit; uint32_t hVASpace, _pad; } tinynv_memory_virtual_alloc_t;
TINYNV_SIZE(tinynv_memory_virtual_alloc_t, 24);
TINYNV_AT(tinynv_memory_virtual_alloc_t, limit, 8);
TINYNV_AT(tinynv_memory_virtual_alloc_t, hVASpace, 16);

// The four PCF encodings this driver writes into a leaf page-table entry, from NVIDIA's dev_mmu.h (gh100, which is
// what tools/gen_nv_regs.py already pins VER3 against; Blackwell does not redefine them and gb202's own dev_mmu.h
// carries no PCF definitions at all).
//
// PCF IS NOT A CACHING FIELD. It encodes privilege, read-only, atomic and caching in one number, and until
// 2026-09-15 pt.c wrote `uncached ? 1 : 0` - which silently also chose REGULAR, READ-WRITE and ATOMIC for every
// entry this driver has ever programmed. That made guarantee 7's GRANT half impossible to express: a guest allowed
// a read-only mapping, which tinynv_rm_map_check correctly permits even against read-only memory, was handed a
// WRITABLE one. Same defect as the mapping-type constant and pointed the same way - the refusal path was tested and
// the grant path gave away more than was asked for.
#define TINYNV_PCF_RW_CACHED   0x0u  // NV_MMU_VER3_PTE_PCF_REGULAR_RW_ATOMIC_CACHED_ACE
#define TINYNV_PCF_RW_UNCACHED 0x1u  // NV_MMU_VER3_PTE_PCF_REGULAR_RW_ATOMIC_UNCACHED_ACE
#define TINYNV_PCF_RO_CACHED   0x4u  // NV_MMU_VER3_PTE_PCF_REGULAR_RO_ATOMIC_CACHED_ACE
#define TINYNV_PCF_RO_UNCACHED 0x5u  // NV_MMU_VER3_PTE_PCF_REGULAR_RO_ATOMIC_UNCACHED_ACE

// How this driver learns where a guest's allocation actually IS. NV0041_CTRL_CMD_GET_SURFACE_PHYS_ATTR on the memory
// object: memOffset goes in and comes back as the physical address at that offset, with contigSegmentSize saying how
// many bytes are contiguous from there - so an allocation is walked segment by segment.
//
// WHY THIS AND NOT WHAT NVIDIA DOES. Their UVM path (nvGpuOpsGetExternalAllocPtes) reads pMemory->pMemDesc directly,
// which is in-kernel RM state. For us RM is GSP firmware and that route does not exist: the guest's class 0x0040
// allocation went through our forwarding path, so GSP-RM holds the descriptor and we have never seen the pages.
//
// TWO LIMITS, STATED RATHER THAN GLOSSED. ctrl0041.h says "This call is only currently supported in the MODS
// environment"; in the open tree that is stale - flags 0x0, accessRight 0x0, compiled in, and the implementation is a
// straight call into the HAL with no gate. But the open tree is the CPU-side RM, and what answers our forwarded
// control is CLOSED GSP firmware. "The control is ungated" is a fact about source we can read; "GSP-RM will answer
// it" is not, and no recording contains the call because nothing has ever made it.
#define TINYNV_CTRL_CMD_GET_SURFACE_PHYS_ATTR 0x410103u
#define TINYNV_PHYS_ATTR_APERTURE_VIDMEM 0u
typedef struct {
  uint64_t memOffset;  // [in] offset into the allocation; [out] the physical address there
  uint32_t memFormat, comprOffset, comprFormat, memAperture, gpuCacheAttr, gpuP2PCacheAttr, mmuContext;
  uint64_t contigSegmentSize;  // [out] bytes contiguous from memOffset to the end of the surface
} tinynv_nv0041_phys_attr_t;
TINYNV_SIZE(tinynv_nv0041_phys_attr_t, 48);
TINYNV_AT(tinynv_nv0041_phys_attr_t, memFormat, 8);
TINYNV_AT(tinynv_nv0041_phys_attr_t, mmuContext, 32);
TINYNV_AT(tinynv_nv0041_phys_attr_t, contigSegmentSize, 40);

#define TINYNV_CLASS_MEMORY_LOCAL_USER 0x0040u
// Whether the allocation itself is read-only, which is the input guarantee 7 compares a requested mapping against.
// Session C found these after I had looked and concluded they did not exist - I grepped, found NVOS02's READ_ONLY
// flags, saw they belonged to a different allocation interface, and stopped. These are NVOS32's, in the struct we
// already capture, four bytes in.
//
// NVIDIA marks both "TODO BUG 2488682: remove this after KMD transition", so guarantee 7's only source is a field its
// author intends to delete. Worth knowing before anything depends on it: if a future pin drops them, 7 loses its
// input again and the honest response is to refuse write mappings rather than to assume writable.
//
// AND THEY DO NOT CARRY THE MEANING CLEANLY EVEN NOW, which is the sharper problem and the one a reader hits first.
// Session C found this building the constant audit. Both bits are ALIASED - NVIDIA spells two names on the same bit,
// not on adjacent ones (nvos.h:1498-1501):
//
//   0x04000000  NVOS32_ALLOC_FLAGS_SPARSE            and  NVOS32_ALLOC_FLAGS_USER_READ_ONLY
//   0x08000000  NVOS32_ALLOC_FLAGS_DEVICE_READ_ONLY  and  NVOS32_ALLOC_FLAGS_ALLOCATE_KERNEL_PRIVILEGED
//
// so a bit test cannot recover which name the caller meant. What that costs guarantee 7, bit by bit:
//
//   0x04000000 is recoverable for the objects 7 actually sees, and only for those. SPARSE is in
//   NVOS32_ALLOC_FLAGS_VIRTUAL_ONLY (nvos.h:1517), and all three physical allocation paths refuse anything in that
//   mask with NV_ERR_INVALID_ARGUMENT - video_mem.c:1188, system_mem.c:562, egm_mem.c:316. Class 0x0040 is
//   VideoMemory (resource_list.h:460), a physical class. So a class-0x0040 object carrying this bit CANNOT be sparse:
//   RM would have failed the allocation before we ever recorded it. On any other class the bit is ambiguous, and the
//   refusal below is then a strict over-refusal rather than a read of the guest's intent.
//
//   0x08000000 is NOT recoverable and does not need to be. Both readings describe memory a guest must not get a write
//   mapping on: DEVICE_READ_ONLY says so directly, and ALLOCATE_KERNEL_PRIVILEGED is kernel-only memory that a
//   non-kernel client cannot even allocate (nvos.h:1447-1451). The ambiguity is benign here by construction, not by
//   luck - which is worth writing down, because "benign by luck" is a thing that stops being true quietly.
//
// So the honest form of guarantee 7 is NOT "refuses write mappings on read-only allocations". It is: refuses write
// mappings on allocations carrying a bit that means read-only, or - on classes other than 0x0040 - possibly sparse.
// The day someone sees a legitimate allocation refused with WOULD_WRITE, this paragraph is what tells them whether it
// is a bug. Neither bit is reachable from the recording either: the recorded allocation is flags 0x1c101, none of the
// four names set, so the first of these to appear will come from a workload and not from a test.
//
// test_headers.c asserts the aliasing itself, so the day NVIDIA fixes BUG 2488682 the build stops and this paragraph
// gets re-read rather than quietly outliving the thing it describes.
// UvmRmGpuMappingType, from NVIDIA's nv_uvm_types.h:461 - named so the next reader does not re-derive it:
//
//   Default = 0, ReadWriteAtomic = 1, ReadWrite = 2, ReadOnly = 3
//
// and uvm_mem.c:238 reads the same enum independently, mapping ReadOnly to UVM_PROT_READ_ONLY. Anything that is not
// ReadOnly is treated as wanting write, because the safe direction for a value we do not recognise is the stricter
// one.
//
// THIS WAS 2 FOR AN HOUR AND THE ERROR INVERTED THE GUARANTEE IT SERVES. With READ_ONLY spelled 2, a guest sending
// mappingType 2 - which the real driver maps as UVM_PROT_READ_WRITE - was classified here as read-only and GRANTED
// against a DEVICE_READ_ONLY allocation: a write mapping on read-only memory, allowed by the check written to stop
// it. Type 3, a genuinely read-only request, was refused instead - a harmless over-refusal that MASKED the real bug,
// because the visible behaviour still looked conservative. Session C found it by going to the header rather than
// trusting either of our files.
#define TINYNV_RM_MAP_TYPE_READ_ONLY 3u
#define TINYNV_ALLOC_FLAGS_USER_READ_ONLY   0x04000000u
#define TINYNV_ALLOC_FLAGS_DEVICE_READ_ONLY 0x08000000u
// Allocating memory, which is the one class a guest allocates for anything it then asks us to map: class 0x0040,
// NV01_MEMORY_LOCAL_USER. Session C read it out of the recording rather than inferring it - torch stops at the map of
// an object allocated as 0x0040 four calls earlier, and there is no second class behind it.
//
// TWO TRAPS IN THIS ONE. The call declares paramsSize 0 ON THE WIRE and the real size is 128, taken from the CLASS
// (rmapiGetClassAllocParamSize) because NVOS64 ignores the caller's paramsSize here - so a reader who trusts the
// declared size reads nothing. And `address` is an NvP64: a POINTER in a params struct a guest supplies. It is [OUT]
// and RM is expected to fill it rather than follow it, but check_ctrl_pointers.py covers CONTROLS and has never
// looked at allocation params, so "no pointer reaches the card from a guest" is verified for the map call and NOT for
// this one. Named here so the next reader does not inherit the stronger claim.
typedef struct {
  uint32_t owner, type, flags, width, height;
  int32_t pitch;
  uint32_t attr, attr2, format, comprCovg, zcullCovg, _pad0;
  uint64_t rangeLo, rangeHi, size, alignment, offset, limit, address;
  uint32_t ctagOffset, hVASpace, internalflags, tag;
  int32_t numaNode;
  uint32_t _pad1;
} tinynv_nv0040_alloc_t;
TINYNV_SIZE(tinynv_nv0040_alloc_t, 128);

typedef struct { uint32_t flags, duration; } tinynv_perf_boost_t;
TINYNV_SIZE(tinynv_perf_boost_t, 8);

// How long the scheduler lets this channel group hold an engine before it may switch to another. Issued on the
// channel GROUP, not a channel. See TINYNV_TIMESLICE_US.
#define TINYNV_CTRL_CMD_SET_TIMESLICE 0xa06c0103u
typedef struct { uint64_t timesliceUs; } tinynv_timeslice_t;
TINYNV_SIZE(tinynv_timeslice_t, 8);

typedef struct {
  uint32_t hObjectError, hObjectEccError, hVASpace, engineType;
  uint8_t bIsCallingContextVgpuPlugin, _pad[3];
} tinynv_channel_group_alloc_t;
TINYNV_SIZE(tinynv_channel_group_alloc_t, 20);
TINYNV_AT(tinynv_channel_group_alloc_t, engineType, 12);

typedef struct { uint32_t hVASpace, flags, subctxId; } tinynv_ctxshare_alloc_t;
TINYNV_SIZE(tinynv_ctxshare_alloc_t, 12);

// An externally owned address space is one whose page tables the driver builds, so GSP-RM has to be told where the root
// of the tree is. Without this the firmware walks nothing and every address in that space faults.
#define TINYNV_MSG_FUNCTION_SET_PAGE_DIRECTORY 54
typedef struct {
  uint64_t physAddress;
  uint32_t numEntries, flags, hVASpace, chId, subDeviceId, pasid;
} tinynv_set_page_directory_params_t;
TINYNV_SIZE(tinynv_set_page_directory_params_t, 32);
TINYNV_AT(tinynv_set_page_directory_params_t, hVASpace, 16);
TINYNV_AT(tinynv_set_page_directory_params_t, pasid, 28);

typedef struct {
  uint32_t hClient, hDevice, pasid, _pad;
  tinynv_set_page_directory_params_t params;
} tinynv_set_page_directory_t;
TINYNV_SIZE(tinynv_set_page_directory_t, 48);
TINYNV_AT(tinynv_set_page_directory_t, params, 16);

// What the chip is made of: how many partitions, how many cores in each, and which instruction set they speak. Asked
// before any channel is created, because the answer decides how a kernel is launched.
#define TINYNV_CTRL_CMD_STATIC_KGR_GET_INFO 0x20800a2au
#define TINYNV_GR_INFO_COUNT 58
#define TINYNV_GR_INFO_NUM_GPCS 20
#define TINYNV_GR_INFO_NUM_TPC_PER_GPC 23
#define TINYNV_GR_INFO_NUM_SM_PER_TPC 32
#define TINYNV_GR_INFO_MAX_WARPS_PER_SM 13
#define TINYNV_GR_INFO_SM_VERSION 12

typedef struct { uint32_t index, data; } tinynv_gr_info_t;
TINYNV_SIZE(tinynv_gr_info_t, 8);
typedef struct { tinynv_gr_info_t infoList[TINYNV_GR_INFO_COUNT]; } tinynv_gr_engine_info_t;
TINYNV_SIZE(tinynv_gr_engine_info_t, 464);
typedef struct { tinynv_gr_engine_info_t engineInfo[8]; } tinynv_gr_static_info_t;
TINYNV_SIZE(tinynv_gr_static_info_t, 3712);

// The two channels work is submitted through, and the objects that come with them.
#define TINYNV_CLASS_DEBUGGER 0x83de
#define TINYNV_CTRL_CMD_GET_WORK_SUBMIT_TOKEN 0xc36f0108u
#define TINYNV_CTRL_CMD_GPFIFO_SCHEDULE 0xa06c0101u

typedef struct { uint32_t hDebuggerClient_Obsolete, hAppClient, hClass3dObject; } tinynv_debugger_alloc_t;
TINYNV_SIZE(tinynv_debugger_alloc_t, 12);
typedef struct { uint32_t workSubmitToken; } tinynv_work_submit_token_t;
TINYNV_SIZE(tinynv_work_submit_token_t, 4);
typedef struct { uint8_t bEnable, bSkipSubmit; } tinynv_gpfifo_schedule_t;
TINYNV_SIZE(tinynv_gpfifo_schedule_t, 2);

// What the debugger object can be asked once something has gone wrong, and the only way this driver can learn an
// address from a fault. An MMU fault arrives as a bare notification on the status queue: the event says a fault
// happened and nothing whatever about where, which is the least useful true sentence a driver can print. The oracle
// does not register a fault buffer to fix that - it asks these two questions after the fact, of the debugger object
// that already exists on the compute channel. That is why reading fault addresses costs nothing at boot and cannot
// perturb the recorded one.
//
// The first call is the one that answers "was it an MMU fault at all": it reports the per-SM error registers and a
// valid bit for a fault on the target channel. Only if that bit is set is the second worth asking, and it returns up
// to four addresses with the kind of access that caused each.
#define TINYNV_CTRL_CMD_DEBUG_READ_ALL_SM_ERROR_STATES 0x83de030cu
#define TINYNV_CTRL_CMD_DEBUG_READ_MMU_FAULT_INFO      0x83de0328u
#define TINYNV_DEBUG_MAX_SMS_PER_CALL 100
#define TINYNV_MMU_FAULT_ENTRIES 4

typedef struct {
  uint32_t hwwGlobalEsr, hwwWarpEsr, hwwWarpEsrPc, hwwGlobalEsrReportMask, hwwWarpEsrReportMask;
  uint64_t hwwEsrAddr, hwwWarpEsrPc64;
  uint32_t hwwCgaEsr, hwwCgaEsrReportMask;
} tinynv_sm_error_state_t;
TINYNV_SIZE(tinynv_sm_error_state_t, 48);
TINYNV_AT(tinynv_sm_error_state_t, hwwEsrAddr, 24);
TINYNV_AT(tinynv_sm_error_state_t, hwwWarpEsrPc64, 32);
TINYNV_AT(tinynv_sm_error_state_t, hwwCgaEsr, 40);

// A hundred SMs of registers in one call, which is the ceiling the interface names and more than this chip enables.
// The deprecated mmuFaultInfo word before the structure is still part of the layout, so it is declared rather than
// skipped: leaving it out would move everything after it by four bytes and the valid bit would be read from padding.
typedef struct {
  uint32_t hTargetChannel, numSMsToRead;
  tinynv_sm_error_state_t smErrorStateArray[TINYNV_DEBUG_MAX_SMS_PER_CALL];
  uint32_t mmuFaultInfo;
  struct { uint8_t valid, _pad[3]; uint32_t faultInfo; } mmuFault;
  uint32_t startingSM;
} tinynv_sm_error_states_t;
TINYNV_SIZE(tinynv_sm_error_states_t, 4824);
TINYNV_AT(tinynv_sm_error_states_t, smErrorStateArray, 8);
TINYNV_AT(tinynv_sm_error_states_t, mmuFaultInfo, 4808);
TINYNV_AT(tinynv_sm_error_states_t, mmuFault, 4812);
TINYNV_AT(tinynv_sm_error_states_t, startingSM, 4820);

typedef struct { uint64_t faultAddress; uint32_t faultType, accessType; } tinynv_mmu_fault_entry_t;
TINYNV_SIZE(tinynv_mmu_fault_entry_t, 16);
typedef struct {
  tinynv_mmu_fault_entry_t mmuFaultInfoList[TINYNV_MMU_FAULT_ENTRIES];
  uint32_t count;
} tinynv_mmu_fault_info_t;
TINYNV_SIZE(tinynv_mmu_fault_info_t, 72);
TINYNV_AT(tinynv_mmu_fault_info_t, count, 64);

// The fault kinds worth naming, as this architecture numbers them. The words are what turn an address into a
// diagnosis: a PTE fault on a plausible-looking address means a mapping this driver did not make, an unbound instance
// block means a channel that was never set up, and a prefetch access means the fault is past the end of something
// rather than at a pointer that was wrong.
//
// These are Blackwell's numbers and they are NOT the ones a general table would give. Writing them from the older
// architectures - which is what a search across the vendored tree hands you first - puts COMPRESSION_FAILURE at 0xb,
// where GB202 has a confidential-computing violation, and omits the write-only violation at 0x7 entirely. The static
// assertions in test_headers are against gb202's own header for exactly that reason, and they caught both.
#define TINYNV_PFAULT_TYPE_PDE                  0x0u
#define TINYNV_PFAULT_TYPE_PDE_SIZE             0x1u
#define TINYNV_PFAULT_TYPE_PTE                  0x2u
#define TINYNV_PFAULT_TYPE_VA_LIMIT_VIOLATION   0x3u
#define TINYNV_PFAULT_TYPE_UNBOUND_INST_BLOCK   0x4u
#define TINYNV_PFAULT_TYPE_PRIV_VIOLATION       0x5u
#define TINYNV_PFAULT_TYPE_RO_VIOLATION         0x6u
#define TINYNV_PFAULT_TYPE_WO_VIOLATION         0x7u
#define TINYNV_PFAULT_TYPE_PITCH_MASK_VIOLATION 0x8u
#define TINYNV_PFAULT_TYPE_WORK_CREATION        0x9u
#define TINYNV_PFAULT_TYPE_UNSUPPORTED_APERTURE 0xau
#define TINYNV_PFAULT_TYPE_CC_VIOLATION         0xbu
#define TINYNV_PFAULT_TYPE_UNSUPPORTED_KIND     0xcu
#define TINYNV_PFAULT_TYPE_REGION_VIOLATION     0xdu
#define TINYNV_PFAULT_TYPE_POISONED             0xeu
#define TINYNV_PFAULT_TYPE_ATOMIC_VIOLATION     0xfu

// Access kinds. The low four are virtual addresses and the ones from eight up are the same accesses against physical
// addresses - a distinction that matters here more than it looks, because a physical-address fault means the fault is
// not in this driver's page tables at all.
#define TINYNV_PFAULT_ACCESS_READ          0x0u
#define TINYNV_PFAULT_ACCESS_WRITE         0x1u
#define TINYNV_PFAULT_ACCESS_ATOMIC        0x2u
#define TINYNV_PFAULT_ACCESS_PREFETCH      0x3u
#define TINYNV_PFAULT_ACCESS_ATOMIC_WEAK   0x4u
#define TINYNV_PFAULT_ACCESS_PHYS_READ     0x8u
#define TINYNV_PFAULT_ACCESS_PHYS_WRITE    0x9u
#define TINYNV_PFAULT_ACCESS_PHYS_ATOMIC   0xau
#define TINYNV_PFAULT_ACCESS_PHYS_PREFETCH 0xbu

// --- the container NVIDIA ships the riscv bootloader in ------------------------------------------------------------
typedef struct { uint32_t bin_magic, bin_ver, bin_size, header_offset, data_offset, data_size; } tinynv_nvfw_bin_hdr_t;
TINYNV_SIZE(tinynv_nvfw_bin_hdr_t, 24);

typedef struct {
  uint32_t version, bootloaderOffset, bootloaderSize, bootloaderParamOffset, bootloaderParamSize;
  uint32_t riscvElfOffset, riscvElfSize, appVersion;
  uint32_t manifestOffset, manifestSize, monitorDataOffset, monitorDataSize, monitorCodeOffset, monitorCodeSize;
  uint32_t bIsMonitorEnabled, swbromCodeOffset, swbromCodeSize, swbromDataOffset, swbromDataSize;
  uint32_t fbReservedSize, bSignedAsCode;
} tinynv_riscv_ucode_desc_t;
TINYNV_SIZE(tinynv_riscv_ucode_desc_t, 84);
TINYNV_AT(tinynv_riscv_ucode_desc_t, manifestOffset, 32);
TINYNV_AT(tinynv_riscv_ucode_desc_t, monitorDataOffset, 40);
TINYNV_AT(tinynv_riscv_ucode_desc_t, monitorCodeOffset, 48);

// Where GPPut sits inside a channel's control block, which is what the doorbell publishes. Checked against
// AmpereAControlGPFifo in the vendored headers by test_headers, because a wrong value here is a submission that is
// simply never noticed by the engine.
#define TINYNV_GPFIFO_GPGET_OFF 0x88   // what the engine has consumed; read only, and the other half of the question
#define TINYNV_GPFIFO_GPPUT_OFF 0x8c

#endif
