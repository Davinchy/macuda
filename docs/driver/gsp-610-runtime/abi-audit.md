# GSP 570.144 → 610.43.03 runtime ABI audit

This audit compares the firmware interfaces used by the 570.144 runtime with the 610.43.03
interfaces used by the experimental RTX 4090 profile. It is derived from the two pinned NVIDIA
source trees and the verified release bytes.

Companions: [spec](spec.md) (wins), [design](design.md), [plan](plan.md). Immutable inputs and
their hashes: [inputs.json](inputs.json). Reproduce with `sh tools/gsp610-runtime/run_audit.sh`.

> 1. **§1g AD102 non-WPR heap: SAME (1 MiB), not changed.** The pinned 610 HAL dispatch
>    (`g_kernel_gsp_nvoc.c`, `kgspGetNonWprHeapSize`) places AD102 in the `ed6b8b` group = 1048576, the same
>    value 570 dispatches; the 2 MiB HAL (`d505ea`) serves GH100/GB100/GB10x, not Ada.
>    `fw_layout_610.h` and `test_headers_610` pin the AD102 value.
> 2. **§1c/§4 GspSystemInfo resolved: 610 = 952 B, `virtualConfigBits` (u16) @840, `bIsPassthru` @842**
>    (570: 928 B, @840); fields 0..96 unchanged. Computed from the verbatim-extracted pinned definition and
>    asserted in `test_headers_610`.
> 3. **§1g image-bound blocker resolved:** the 610 reservation (`fw_layout_610.h`) raises the image bound to
>    96 MiB for the 80.32 MiB `.fwimage`; the worst-case carveout (~238 MiB) still fits the existing 256 MiB
>    hold-back — checked by static assertions, so `reserve_top` and every low address are unchanged.
> 4. **Load-exec protocol:** RUSD `bInit=1` is encoded; the AD102 load-exec host sequences are implemented
>    from the pinned HALs in `gsp610_host.c`; the booter containers are BTL1 v1 with a raw image and an
>    84-byte v5 descriptor with fuse-selected signatures. The
>    AD102 archive descriptor is 84 B on the wire even though the 610 struct is 92 B).
> 5. **Sequencer replacement payload sizes:** `GspLoadExecGenericBootloaderParams` = 112 B,
>    `GspLoadExecHsBinaryParams` = 80 B, `RM_FLCN_BL_DMEM_DESC` = 84 B (`RM_FLCN_U64` is `NvU64_ALIGN32`);
>    the bundle includes the corresponding `generic_bl` and `generic_bl_desc` assets.
> 6. **Pre-existing coverage note:** the ADA-vs-gb202 fault-numbering note in §4 —
>    `test_headers.c` still asserts gb202 numbering while AD102 runs mmu_ver 2; unchanged, flagged for a
>    future driver card.
> Hardware validation shows that the 610 boot chain reaches RM, creates its objects, supports compute,
> and unloads cleanly. Source classification establishes ABI compatibility; it does not broaden the
> validated device or workload coverage. See
> [spec.md](spec.md) and [usage.md](usage.md).

## 0. Method and pins

Two header/source trees are compared, each fetched and SHA-256-verified by
`tools/gsp610-runtime/fetch_inputs.sh` into this worktree's gitignored `third_party/`:

| Side | Source | Revision | SHA-256 (archive) |
|---|---|---|---|
| 570 | `NVIDIA/open-gpu-kernel-modules` | commit `81fe4fb417c8ac3b9bdcc1d56827d116743892a5` (release 570.86.16) | `c2f13d2132043f186cd831b28fb43b83e71cbe637761ffdd6110f2eeb9abd2f5` |
| 610 | `NVIDIA-kernel-module-source-610.43.03.tar.xz` (download.nvidia.com) | `NVIDIA_VERSION = 610.43.03` (`version.mk`) | `7e118923c7a23edc36114d63273a46e3e04e9af98695a42203e7ac2dfe9fc1dc` |

The 570 pin is the one `tools/fetch_nv_headers.sh` and the Makefile's `NVSRC` already use
(the deliberate headers-570.86.16 / firmware-570.144 skew documented in the README). The 610 pin is
the immutable NVIDIA 610.43.03 kernel-module source archive recorded in `inputs.json`.

Evidence is produced two ways, both reproducible:

1. **Compiled probe** (`tools/gsp610-runtime/abi_probe.c`): one translation unit including only
   NVIDIA headers, compiled once per tree (`build_probes.sh`), emitting `sizeof`/`offsetof`/constant
   records. 447 records; **381 same, 48 changed, 3 removed, 15 added**. Raw:
   `cuda-shim/build/gsp610-abi/abi_{570,610}.txt`; matrix: `abi_matrix.json`.
2. **Source-level compare** (`abi_text.py`): for definitions behind NVIDIA's port layer or inside
   `.c` files that cannot be compiled standalone, the typedef/function body is extracted from both
   trees, normalized, and compared. Raw: `text_diff.json`.

The driver-side inventory is built from actual field reads/writes in `src/gsp.c`, `src/flcn.c`,
`src/fw.c`, `src/vbios.c`, `src/dev.c`, `src/nv_structs.h`, `src/fw_layout.h`, not from typedef names.
The existing `test/test_headers.c` already pins the 570 declarations; this audit asks, for each, what
610 does.

**Classification key:** `same` = identical size/offsets/value (shared code is safe with an
assertion); `changed` = layout/size/constant moved (needs a version-specific serializer or field
placement); `removed` = gone in 610 (needs a replacement path); `added` = new in 610 (may be a new
mandatory host request).

## 1. Summary matrix (every used interface)

### 1a. Boot / startup metadata

| Interface | Driver use | 570 | 610 | Class | Action |
|---|---|---|---|---|---|
| `NVDM_PAYLOAD_COT` (FSP chain-of-trust) | `flcn.c` FMC path (Blackwell only) | 860 B, all offsets | identical | **same** | share; assert. Not on the AD102 vbios path |
| `NVDM_TYPE_COT` | `flcn.c` | 0x14 | 0x14 | **same** | share |
| `GSP_FMC_BOOT_PARAMS` | `flcn.c` FMC path | 80 B | **88 B** | **changed** | tail grew: `GSP_RM_MEM_PARAMS gspRmMemParams` (8 B) appended; `bootGspRmParams`/`gspRmParams`/`gspSpdmParams` offsets unchanged. Zero-fill tail for 610 |
| `GSP_ACR_BOOT_GSP_RM_PARAMS` | `flcn.c` | 32 B | 32 B (adds `NvBool bInstInSysMode` into existing pad) | **same** size/used-offsets | share; assert used fields; new bool stays 0 |
| `GSP_RM_PARAMS` / `GSP_SPDM_PARAMS` | `flcn.c` | 16 / 24 B | identical | **same** | share |
| `GSP_DMA_TARGET_COHERENT_SYSTEM` | `flcn.c`/`gsp.c` | 1 | 1 | **same** | share |
| `GspFwWprMeta` | `gsp.c init_wpr_meta` | 256 B, every offset | **identical** (all 30 probed offsets) | **same** | share; assert. Magic `0xdc3aae21371a60b3`, revision 1 unchanged |
| `GSP_FW_WPR_META_MAGIC` / `_REVISION` | `gsp.c` | 0xdc3aae21371a60b3 / 1 | identical | **same** | share |
| `RM_RISCV_UCODE_DESC` (bootloader container descriptor) | `gsp.c init_boot_binary_image` | 84 B | **92 B** | **changed** | appends `NvU32 bIsSmp; NvU32 bIsPlicEnabled;`. All used offsets (`manifestOffset` 32, `monitorDataOffset` 40, `monitorCodeOffset` 48) unchanged. Read by fixed offset is safe; size assert must be per-profile |
| `NVFW_BIN_HDR` (booter/bootloader outer header) | `flcn.c flcn_parse_booter`, `gsp.c` | magic `0x000010de`, 24 B | **container differs** (see §2) | **changed** | 610 uses bindata-derived `BTL1` containers and the profile has a dedicated parser |
| `MESSAGE_QUEUE_INIT_ARGUMENTS` | `gsp.c init_rm_args` | 32 B | **64 B** | **changed** | appends `queueElementHdrSize`(32), `queueElementSizeMin`(40), `queueElementSizeMax`(48), `queueHeaderAlign`(56), `queueElementAlign`(60). `sharedMemPhysAddr`/`pageTableEntryCount`/`cmdQueueOffset`/`statQueueOffset` offsets unchanged. 610 encoder must fill the new fields |
| `GSP_ARGUMENTS_CACHED` | `gsp.c init_rm_args` | 72 B | **152 B** | **changed** | grows with msgq args; `gpuInstance` 44→76, `bDmemStack` 48→80, `profilerArgs` 56→88; new `sysmemHeapArgs`(104), `rmStateMonitorBufferArgs`(120), `bindataArgs`(136). Dedicated 610 encoder |
| `GSP_SR_INIT_ARGUMENTS` | (zeroed `srInitArguments[12]`) | 12 B | 12 B | **same** | share |
| `LibosMemoryRegionInitArgument` | `gsp.c init_libos_args` | 32 B, all offsets | identical | **same** | share; assert |
| `LIBOS_MEMORY_REGION_CONTIGUOUS`/`LOC_SYSMEM`/`RADIX_PAGE_LOG2` | `gsp.c` | 1 / 1 / 12 | identical | **same** | share |
| `msgqTxHeader` (queue header) | `gsp.c` ring | 32 B, all offsets | **identical** | **same** | share; assert. The queue *header* is stable; the *element* framing is not (§1b) |

### 1b. Queue framing / RPC envelope / events

| Interface | Driver use | 570 | 610 | Class | Action |
|---|---|---|---|---|---|
| `GSP_MSG_QUEUE_ELEMENT` (element header) | `gsp.c rpc_send_record`/`rpc_drain` (`tinynv_msg_element_t`, 48 B) | `authTag[16]+aad[16]+checkSum+seqNum+elemCount` then aligned `rpc_message_header_v` (48 B before RPC) | **`mctpHeader+nvdmHeader+checkSum+seqNum+payload[]`** (16 B before RPC; no `elemCount`) | **changed** | **Versioned codec required.** 610 wraps every element in MCTP/NVDM (`NVDM_TYPE_RM_RPC = 0x25`); element count derived from length + `queueElementSizeMin`, not stored. See `message_queue_cpu.c:505-512` |
| `rpc_message_header_v03_00` | `gsp.c` (`tinynv_msg_header_t`, 32 B) | 32 B | **identical** (source-compare) | **same** | share; the RPC header itself is stable, its position inside the element moves |
| `rpc_message_rpc_union_field_v03_00` | `gsp.c` | — | identical | **same** | share |
| RPC checksum (XOR-fold of 64-bit words) | `gsp.c rpc_checksum` | zero-padded to 8 | same algorithm (`message_queue_cpu.c` zero-pads `msgLen & 7`) | **same** | share; verify against 610 fixture |
| `NV_VGPU_MSG_SIGNATURE_VALID` 0x43505256, header version 3<<24 | `gsp.c` | — | unchanged in `rpc_message_header_v` | **same** | share |
| Function `CONTINUATION_RECORD` 71 | `gsp.c rpc_send` | 71 | 71 | **same** | share |
| Function `GSP_SET_SYSTEM_INFO` 72 | `gsp.c` | 72 | 72 | **same** | share (payload changed, §1c) |
| Function `SET_REGISTRY` 73 | `gsp.c` | 73 | 73 | **same** | share |
| Function `UNLOADING_GUEST_DRIVER` 47 | `gsp.c tinynv_gsp_unload` | 47 | 47 | **same** | share |
| Function `GSP_RM_CONTROL` 76 | `gsp.c` | 76 | 76 | **same** | share (header changed, §1c) |
| Function `GSP_RM_ALLOC` 103 | `gsp.c` | 103 | 103 | **same** | share |
| Function `SET_PAGE_DIRECTORY` 54 | `gsp.c` | 54 | 54 | **same** | share |
| Event `GSP_INIT_DONE` 4097 | `gsp.c` | 4097 | 4097 | **same** | share |
| Event `GSP_RUN_CPU_SEQUENCER` 4098 | `gsp.c run_cpu_seq` | 4098 | enum value still 4098, **but struct + handler removed** | **removed** | see §1d |
| Event `MMU_FAULT_QUEUED` 4101 | `gsp.c` | 4101 | 4101 | **same** | share |
| Event `OS_ERROR_LOG` 4102 | `gsp.c` | 4102 | 4102 | **same** | share |
| Event `GSP_LOAD_EXEC_GENERIC_BOOTLOADER` 0x1026 (4134) | — | absent | **new** | **added** | 610 replacement for sequencer core-reset/start/halt (§1d) |
| Event `GSP_LOAD_EXEC_HS_BINARY` 0x1027 (4135) | — | absent | **new** | **added** | 610 replacement for sequencer HS launch (§1d) |

### 1c. RM allocation / control payloads

| Interface | Driver use | 570 | 610 | Class | Action |
|---|---|---|---|---|---|
| `rpc_gsp_rm_alloc_v03_00` (`tinynv_rpc_rm_alloc_t`) | `gsp.c rm_alloc_as` | 32 B | **identical** (source-compare) | **same** | share; assert |
| `rpc_gsp_rm_control_v03_00` (`tinynv_rpc_rm_control_t`) | `gsp.c rm_control_as` | 24 B (`…paramsSize,flags`) | **40 B** (`flags`→`rmapiRpcFlags`, +`rmctrlFlags`, +`rmctrlAccessRight`, +`reserved0` u64) | **changed** | 610 control header encoder; `params[]` now at 40 not 24 |
| `rpc_set_page_directory_v1E_05` (`tinynv_set_page_directory_t`) | `gsp.c` | 48 B | **identical** | **same** | share; assert |
| `rpc_unloading_guest_driver_v1F_07` (`tinynv_unloading_guest_driver_t`) | `gsp.c` | 8 B | **identical** | **same** | share; assert |
| `GspSystemInfo` (`tinynv_gsp_system_info_t`, 928 B) | `gsp.c rpc_set_gsp_system_info` | `bIsPassthru` @840 | **952 B; `NvU16 virtualConfigBits` @840, `bIsPassthru` @842**, asserted in `test_headers_610` | **changed** | All fields the driver fills at offsets 0–96 unchanged; 610 encoder in `rpc610.c` |
| `PACKED_REGISTRY_TABLE` / `PACKED_REGISTRY_ENTRY` (`tinynv_registry_*`) | `gsp.c rpc_set_registry_table` | 8 / 16 B | **identical** | **same** | share; assert |
| `NV0000_ALLOC_PARAMETERS` (`tinynv_nv0000_alloc_t`) | root client | 120 B | identical | **same** | share |
| `NV0080_ALLOC_PARAMETERS` (`tinynv_nv0080_alloc_t`) | device | 56 B | identical | **same** | share |
| `NV2080_ALLOC_PARAMETERS` (`tinynv_nv2080_alloc_t`) | subdevice | 4 B | identical | **same** | share |
| `NV_VASPACE_ALLOCATION_PARAMETERS` (`tinynv_vaspace_alloc_t`) | vaspace | 48 B | **56 B** (appends `NvU32 pasid`) | **changed** | used offsets unchanged; zero-fill `pasid`. Verify 610 accepts pasid=0 for externally-owned VA |
| `NV_MEMORY_VIRTUAL_ALLOCATION_PARAMS` (`tinynv_memory_virtual_alloc_t`) | class 0x70 | 24 B | identical | **same** | share |
| `NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS` (`tinynv_channel_group_alloc_t`) | TSG | 20 B | identical | **same** | share |
| `NV_CTXSHARE_ALLOCATION_PARAMETERS` (`tinynv_ctxshare_alloc_t`) | ctxshare | 12 B | identical | **same** | share |
| `NV83DE_ALLOC_PARAMETERS` (`tinynv_debugger_alloc_t`) | debugger | 12 B | identical | **same** | share |
| `NV00DE_ALLOC_PARAMETERS` (`tinynv_rusd_alloc_t`) | RUSD | 8 B | identical | **same** | share |
| `NVB0B5_ALLOCATION_PARAMETERS` (`tinynv_dma_copy_alloc_t`) | CE | 8 B, VERSION_1=1 | identical | **same** | share |
| `NV_CHANNEL_ALLOC_PARAMS` (`tinynv_gpfifo_alloc_t`) | GPFIFO channel | 368 B | **376 B** (`NvHandle hHandleVASpace` inserted after `hVASpace`) | **changed** | `hUserdMemory` 32→36, `userdOffset` 64→72, `engineType` 128→136, `instanceMem` 144→152, … `tpcConfigID` 360→368. Every field after `hVASpace` shifts; 610 channel encoder |
| `NV_MEMORY_DESC_PARAMS` (`tinynv_memory_desc_t`) | channel state | 24 B | identical | **same** | share |
| `NVOS00_PARAMETERS` (free) | object free | 16 B | identical | **same** | share |
| `NVOS04_FLAGS_CHANNEL_PBDMA_ACQUIRE_TIMEOUT` / `_GROUP_CHANNEL_RUNQUEUE` | `gsp.c new_queue` | 27:27 / 4:4 | identical | **same** | share |

### 1d. CPU register sequencer

| Interface | Driver use | 570 | 610 | Class | Action |
|---|---|---|---|---|---|
| `rpc_run_cpu_sequencer_v17_00` (40 B hdr: `bufferSizeDWord`,`cmdIndex`,`regSaveArea[8]`) | `gsp.c run_cpu_seq` | present | **removed from `g_rpc-structures.h`** | **removed** | 610 has no sequencer RPC |
| `GSP_SEQ_BUF_OPCODE` enum (REG_WRITE 0 … CORE_RESUME 8) | `gsp.c run_cpu_seq` switch | `rmgspseq.h` | **removed from `rmgspseq.h`** | **removed** | opcode stream gone |
| `GSP_SEQ_BUF_REG_SAVE_SIZE` (8) | `gsp.c` save area | defined | **removed** | **removed** | — |
| `GspLoadExecGenericBootloaderParams` | — | absent | **new** (`rmgspseq.h`: `RM_FLCN_BL_DMEM_DESC dmemDesc`, …) | **added** | 610 host-request for bootloader load/exec |
| `GspLoadExecHsBinaryParams` | — | absent | **new** (`imemPhysAddr`,`dmemPhysAddr`,`ucodeImemVA/PA/Size`,`ucodeDmemVA/PA/Size`,`hsSigDmemAddr`,`engineIdMask`,`ucodeId`) | **added** | 610 host-request for HS binary launch |
| handlers `_kgspRpcLoadAndExecuteGenericBootloader` / `…HsBinary` | — | absent | `kernel_gsp.c:430-457`, dispatched at 1430/1579 | **added** | the new mechanism CPU-RM serves |

**Impact.** The recorded Ampere/Ada boot runs a register sequence (`run_cpu_seq`, exercised once on
the 3060). On 610 that event is replaced by `GSP_LOAD_EXEC_GENERIC_BOOTLOADER` / `GSP_LOAD_EXEC_HS_BINARY`
with structured payloads instead of an opcode stream. macuda must implement these two host requests
for the 610 profile; the existing `run_cpu_seq` cannot be reused. This is a **required** adaptation,
not an optional feature. The 570 path stays as-is.

### 1e. Controls / GR / channel / memory

| Interface | Driver use | 570 | 610 | Class | Action |
|---|---|---|---|---|---|
| `NV2080_CTRL_CMD_FIFO_GET_DEVICE_INFO_TABLE` 0x20801112 + params | engine table | 3212 B, entry 100 B | identical | **same** | share |
| `NV90F1_CTRL_CMD_VASPACE_COPY_SERVER_RESERVED_PDES` 0x90f10106 + params | reserved PDEs | 184 B, level 24 B, `GMMU_FMT_MAX_LEVELS` | identical | **same** | share |
| `NV2080_CTRL_CMD_INTERNAL_STATIC_KGR_GET_CONTEXT_BUFFERS_INFO` 0x20800a32 + params | GR ctx sizes | 1664 B, 26 buffers × 8 engines | identical | **same** | share |
| `NV2080_CTRL_CMD_GPU_PROMOTE_CTX` 0x2080012b + params | ctx promote | 560 B, entry 32 B, max 16 | identical | **same** | share |
| `NV2080_CTRL_CMD_INTERNAL_STATIC_KGR_GET_INFO` 0x20800a2a + params | chip geometry | 3712 B (`infoList[58]`×8) | **4032 B** (`NV0080_CTRL_GR_INFO_MAX_SIZE` 58→**63**, per-engine 464→504) | **changed** | indices the driver reads (NUM_GPCS 20, NUM_TPC_PER_GPC 23, NUM_SM_PER_TPC 32, MAX_WARPS_PER_SM 13, SM_VERSION 12) unchanged; array stride grows. 610 reads at new stride |
| GR info index constants | `gsp.c` | 20/23/32/13/12 | identical | **same** | share |
| `NV2080_CTRL_CMD_PERF_BOOST` 0x2080200a + params | clocks | 8 B, flags | identical | **same** | share |
| `NVA06C_CTRL_CMD_SET_TIMESLICE` 0xa06c0103 + params | timeslice | 8 B | identical | **same** | share |
| `NVA06C_CTRL_CMD_GPFIFO_SCHEDULE` 0xa06c0101 + params | schedule | 2 B (`bEnable,bSkipSubmit`) | **3 B** (adds `bSkipEnable`) | **changed** | used offsets unchanged; zero-fill `bSkipEnable` |
| `NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN` 0xc36f0108 + params | doorbell token | 4 B | identical | **same** | share |
| `NV83DE_CTRL_CMD_DEBUG_READ_ALL_SM_ERROR_STATES` 0x83de030c + params | fault diag | 4824 B, SM state 48 B | identical | **same** | share |
| `NV83DE_CTRL_CMD_DEBUG_READ_MMU_FAULT_INFO` 0x83de0328 + params | fault addrs | 72 B, entry 16 B, max 4 | identical | **same** | share |
| `NV2080_CTRL_CMD_INTERNAL_GPU_GET_GSP_RM_FREE_HEAP` 0x20800aeb + params | heap probe | 8 B | identical | **same** | share |
| `NV2080_CTRL_CMD_INTERNAL_INIT_USER_SHARED_DATA` 0x20800afe + params | RUSD init | 8 B (`physAddr`@0) | **16 B** (adds `NvBool bInit` before `physAddr`@8) | **changed** | 610 must set `bInit=1` and place `physAddr`@8 |
| `NV2080_CTRL_CMD_INTERNAL_USER_SHARED_DATA_SET_DATA_POLL` 0x20800aff + params | RUSD poll | `pollFrequencyMs`@8 | **`pollIntervalMs`@8** (rename) | **changed** | same offset/width; rename only |
| Classes: ROOT 0x0, DEVICE 0x80, SUBDEVICE 0x2080, VASPACE 0x90f1, MEMORY_VIRTUAL 0x70, CHANNEL_GROUP 0xa06c, CONTEXT_SHARE 0x9067, DEBUGGER 0x83de, USER_SHARED_DATA 0xde | object tree | — | identical | **same** | share |
| Classes: GPFIFO AMPERE 0xc56f / BLACKWELL 0xc96f, COMPUTE AMPERE 0xc7c0 / ADA 0xc9c0 / BLACKWELL 0xcec0, DMA_COPY AMPERE 0xc7b5 / BLACKWELL 0xcab5 | `dev.c` per-chip | — | identical | **same** | share. AD102 uses ADA_COMPUTE_A 0xc9c0 |
| `AmpereAControlGPFifo.GPPut` @0x8c / `GPGet` @0x88 | doorbell | 512 B block | identical | **same** | share |
| `NvGpuSemaphore.timeStamp` @8 | release | 16 B | identical | **same** | share |
| `NV_MMU_VER3_PTE_PCF_REGULAR_RW_ATOMIC_CACHED_ACE` / `_UNCACHED_ACE` | `pt.c` | 0 / 1 | identical (gh100 `dev_mmu.h`) | **same** | share |
| ADA `ad102/dev_fault.h` fault/access types | fault naming | — | **byte-identical** | **same** | share. (Driver currently asserts gb202 numbering in `test_headers.c`; AD102 runs mmu_ver 2 — see §4 note) |

### 1f. RUSD shared sensor block (`NV00DE_SHARED_DATA`)

| Field | 570 off | 610 off | Class |
|---|---|---|---|
| struct size | 464 | **3152** | **changed** |
| `bar1MemoryInfo`,`pmaMemoryInfo`,`shadowErrCont`,`grInfo`,`clkPublicDomainInfos` | 0… | unchanged | same |
| `clkThrottleReason` (16 B) | 96 | **removed** → `clkThrottleInfo` (1880 B) @112 | **changed** |
| `perfDevUtil` | 112 | 1992 | changed |
| `memEcc` | 160 | 2040 | changed |
| `perfCurrentPstate` | 264 | 2280 | changed |
| `powerLimitGpu` | 280 | 2296 | changed |
| `temperatures[2]` | 296 | 2312 | changed |
| `memRowRemap` | 328 | 2392 | changed |
| `avgPowerUsage` | 368 | 2440 | changed |
| `instPowerUsage` | 392 | 2464 | changed |
| `pciBusData` | 416 | 2504 | changed |
| `powerPolicyStatus` | — | 2488 | added |
| `fanCoolerStatus` | — | 2552 | added |
| `procUtil` | — | 2624 | added |
| `RUSD_CLK_PUBLIC_DOMAIN_INFOS` size | 24 | 40 | changed |

**Action.** Telemetry is optional (spec: "telemetry where supported"). The 610 RUSD block is a
different layout; the driver's `tinynv_rusd_t` transcription and every `FIELD()` offset must be
per-profile. The timestamp/sequence validity protocol (`RUSD_SEQ_*`) is unchanged. A 610 RUSD
adapter reads the fields the driver actually uses (`temperatures`, `powerLimitGpu`, `clkPublicDomainInfos`,
`avgPowerUsage`, `instPowerUsage`, `pciBusData`) at their 610 offsets. Not a boot blocker.

### 1g. Firmware reservation policy

| Item | 570 | 610 | Class | Action |
|---|---|---|---|---|
| `_kgspCalculateFwHeapSize` formula | OS carveout + base RM + per-GB + client alloc | same terms; `SIZE_PER_GB_FB`→`SIZE_PER_GB`; adds zero-FB branch | **changed (cosmetic)** | formula value unchanged for a 32 GB card |
| `GSP_FW_HEAP_PARAM_OS_SIZE_LIBOS3_BAREMETAL` | 22 MiB | 22 MiB | same | share |
| `GSP_FW_HEAP_PARAM_BASE_RM_SIZE_GH100` | 14 MiB | 14 MiB | same | share |
| `GSP_FW_HEAP_PARAM_SIZE_PER_GB(_FB)` | 96 KiB | 96 KiB (renamed) | same value | share |
| `GSP_FW_HEAP_PARAM_CLIENT_ALLOC_SIZE` | 48 KiB×2048 | identical | same | share |
| `TINYNV_FW_HEAP_SIZE` (0x8700000 = 135 MiB) | valid | still valid (same formula) | same | keep |
| non-WPR heap HAL, **AD102** | `ed6b8b` = **1 MiB** | `ed6b8b` = **1 MiB** | **same** (the `d505ea` 2 MiB HAL serves GH100/GB100/GB10x, not Ada) | AD102 vbios-path `non_wpr_sz` stays 0x100000; `fw_layout_610.h` uses the matching 1 MiB |
| non-WPR heap HAL, GB20x | `1bb8e3` = 2228224 (2.125 MiB) | `b5d989` = **3145728 (3 MiB)** | **changed** | `test_headers.c` asserts 2228224; 610 profile needs 3 MiB |
| FRTS region size | 1 MiB (`FWSECLIC_FRTS_REGION_SIZE_1MB_IN_4K` 0x100) | identical | same | share |
| **GSP `.fwimage` size** | 63,541,248 B (60.6 MiB) | **84,226,048 B (80.3 MiB)** | **changed** | **exceeds `TINYNV_FW_IMAGE_BOUND` (64 MiB).** 610 profile must raise the image bound and `TINYNV_FW_RESERVE_TOP` and re-check placed WPR/non-WPR bounds |
| GSP ELF section names | `.fwimage`,`.fwsignature_ad10x`,`.fwsignature_gb20x` | identical | same | share parser |
| `.fwsignature_ad10x` offset | 63,561,836 | 84,267,118 | changed (image grew) | read from ELF, not hardcoded |

## 2. Assets / container formats (plan §2 prerequisite findings)

Recorded here because the ABI boundary includes the boot image containers; full asset preparation
is plan §2.

- **570 AD102 booter/bootloader** (linux-firmware, `third_party/firmware/nvidia/ad102/gsp/`): outer
  header magic `0x000010de` (`NVFW_BIN_HDR`), which `flcn_parse_booter` parses (HS header v2 + load
  header v2 + app entry). Hashes verified in `inputs.json`.
- **610 runtime booters**: magic **`BTL1` (`0x314c5442`)**, a different container from the 570 NVFW
  image. They must not be renamed into a 570 firmware filename.
- **610 NVIDIA driver booter packaging**: `kernel_gsp_booter.c` uses BINDATA archives with separate
  labels (`IMAGE_PROD`, `SIG_PROD`, `PATCH_SIG`, `NUM_SIGS`) and geometry fields
  (`osCodeOffset`,`osDataOffset`,`appCodeOffset`,`appDataOffset`), not an in-image NVFW header.
  `BTL1` does not appear anywhere in the 610 NVIDIA tree; the preparation tooling builds the
  container deterministically from the release bindata.
- **610 AD102 boot assets exist in the pinned source** as generated bindata:
  `g_bindata_kgspGetBinArchiveBooterLoadUcode_AD102.c`,
  `g_bindata_kgspGetBinArchiveBooterUnloadUcode_AD102.c`,
  `g_bindata_kgspGetBinArchiveGspRmBoot_AD102.c`,
  `g_bindata_ksec2GetBinArchiveSecurescrubUcode_AD10X.c`. So load/unload booters, GSP boot and the
  scrubber are all obtainable from the pinned 610 release (extraction is plan §2).
- **GSP ELF**: the bundle's `gsp_ga10x.bin` (sha256 `572373620a…`, verified) has the same
  `.fwimage`/`.fwsignature_ad10x` sections macuda's parser reads; only the size grew (§1g).

The 610 profile therefore has its own BTL1 parser. The 570 `flcn_parse_booter` path remains
unchanged.

**Board-provided VBIOS structures are runtime-independent.** `vbios.c` parses the RTX 4090's own
ROM (BIT header, falcon ucode table, FWSEC descriptor v3, falcon app interface, DMEM mapper, FRTS
command — the packed structs in `vbios_structs.h`). These come from the board, not from the 570/610
firmware files, so selecting the 610 runtime does not change them; the FWSEC-FRTS/SB path is
unchanged by this migration. Only the *firmware-file* containers (booter/bootloader/scrubber:
`NVFW_BIN_HDR` + HS header v2 + load header v2 + app entry, parsed by `flcn_parse_booter`) are
version-dependent and are handled by the 610 BTL1 path. The HS-v2/load-v2 layouts themselves
are tinygrad-autogen-derived (not present as includable NVIDIA headers in either tree); their 610
applicability is established by the validated BTL1 extraction and boot path.

## 3. What is safe to share (assertion-backed)

The following are byte-identical across the pins and can be shared with a per-release static
assertion (design: "If layouts are identical, share code with assertions demonstrating why"):
WPR metadata, COT payload, libos region, msgq **header**, RPC message header, registry table,
RM alloc RPC header, set-page-directory, unloading-guest-driver, all RPC function/event **numbers**,
all RM class numbers, all control **command numbers**, FIFO device-info table, copy-PDEs, GR ctx
buffers info, promote-ctx, perf-boost, timeslice, work-submit-token, SM error states, MMU fault info,
free-heap, memory descriptors, NVOS00/04, GPFifo control block, semaphore, MMU PCF, ADA fault types,
and the firmware heap formula values.

## 4. Notes and limits

- **Evidence boundary.** The matrix is a source/byte audit. The separate offline lifecycle and live
  checks cover the implemented request and boot paths on one RTX 4090; they do not establish support
  for other devices or broader workloads.
- **`GspSystemInfo`:** the 610 definition is 952 bytes with `virtualConfigBits` at offset 840
  and `bIsPassthru` at 842. The 610 encoder and `test_headers_610` pin the layout.
- **ADA vs gb202 fault numbering.** `test_headers.c` asserts `TINYNV_PFAULT_*` against gb202's
  `dev_fault.h`, but AD102 runs mmu_ver 2 and its faults are named by `ada/ad102/dev_fault.h`
  (byte-identical 570↔610). This is a pre-existing driver concern, not a 610 delta; flagged for the
  reconciliation task, not changed here.
- **RUSD initialization:** the 610 control sets `bInit=1` and places `physAddr` at offset 8;
  `rpc610.c` encodes that layout.
- **Sequencer replacement:** `gsp610_host.c` mirrors the pinned `kgspLoadAndExecuteHsBinary_HAL`
  sequence and its payload sizes.
- **Firmware image reservation:** the 610 GSP image is 80 MiB, so its profile raises the image
  bound while static assertions retain the existing low-address and carveout bounds.

## 5. Reproduction

```
sh tools/gsp610-runtime/run_audit.sh
# -> cuda-shim/build/gsp610-abi/{abi_570.txt,abi_610.txt,text_diff.json,abi_matrix.json}
# -> docs/driver/gsp-610-runtime/inputs.json   (17 hash checks, all_match=true)
```

Sources: `tools/gsp610-runtime/{fetch_inputs.sh,build_probes.sh,abi_probe.c,abi_text.py,abi_diff.py,make_inputs.py,run_audit.sh}`.
