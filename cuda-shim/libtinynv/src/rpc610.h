// The 610.43.03 RPC / resource-management adapters (docs/driver/gsp-610-runtime plan §4, design "ABI boundary"
// and "Integration map").
//
// Everything the 610 queue/RPC/event and RM allocation/control path builds or parses that DIFFERS from 570, as pure
// functions: no device, no GPU, no allocation. Each takes explicit inputs and returns 0 / -1 through tinynv_fail, so
// the offline suite drives them with synthetic fixtures. The 570 path is untouched; these are only reached through the
// 610 profile after it is selected and frozen for the open.
//
// What changed for 610 and is handled here (matrix, re-verified against the pinned source):
//   * the queue ELEMENT framing: 570's 48-byte authTag/aad/checkSum/seqNum/elemCount header becomes a 16-byte
//     MCTP/NVDM/checkSum/seqNum prefix (message_queue_cpu.c). The element count is derived from the RPC length and
//     queueElementSizeMin, not stored. The queue HEADER and the 32-byte RPC header inside are byte-identical (shared,
//     asserted in test_headers_610).
//   * rpc_gsp_rm_control grew 24 -> 40 B (flags -> rmapiRpcFlags, +rmctrlFlags, +rmctrlAccessRight, +reserved0 u64);
//     params[] now sits at 40. The ALLOC header (32 B) is unchanged and shared.
//   * GspSystemInfo grew 928 -> 952 B; virtualConfigBits (u16) is inserted at 840 and bIsPassthru moves 840 -> 842.
//     Every field the driver fills at 0..96 keeps its offset.
//   * NV_CHANNEL_ALLOC_PARAMS grew 368 -> 376 B (hHandleVASpace inserted after hVASpace); every later field shifts.
//   * NV_VASPACE_ALLOCATION_PARAMETERS grew 48 -> 56 B (pasid appended; zero for externally-owned VA).
//   * the GR static-info array stride grew: NV0080_CTRL_GR_INFO_MAX_SIZE 58 -> 63, per-engine 464 -> 504, total
//     3712 -> 4032. The indices the driver reads are unchanged.
//   * GPFIFO_SCHEDULE params grew 2 -> 3 B (bSkipEnable appended, zero).
//   * RUSD: INIT_USER_SHARED_DATA grew 8 -> 16 B (bInit before physAddr@8); SET_DATA_POLL renamed pollFrequencyMs ->
//     pollIntervalMs (same offset/width); the shared sensor block grew 464 -> 3152 B with new offsets.
//   * the CPU register sequencer (event 4098, rpc_run_cpu_sequencer) is REMOVED. Its replacements are the two new
//     host requests GSP_LOAD_EXEC_GENERIC_BOOTLOADER (4134) and GSP_LOAD_EXEC_HS_BINARY (4135), whose structured
//     payloads are parsed/validated here and executed by gsp610_host.c.
//
// Unchanged encoders are SHARED with assertions, not copied: the RPC message header, the RM alloc header, the
// set-page-directory and unloading-guest-driver payloads, the registry table, every function/event number and every
// class/control command number are asserted byte-identical against the pinned 610 tree in test_headers_610.c.
#ifndef TINYNV_RPC610_H
#define TINYNV_RPC610_H
#include "nv_structs.h"
#include <stddef.h>
#include <stdint.h>

// --- the 610 queue element (MCTP/NVDM framing) ---------------------------------------------------------------------
// GSP_MSG_QUEUE_ELEMENT without Confidential Compute: a 16-byte prefix, then the RPC header, then the payload.
// The 570 element's authTag/aad/elemCount are gone; the count is derived from the length.
typedef struct {
  uint32_t mctpHeader, nvdmHeader, checkSum, seqNum;
} tinynv_rpc610_elem_t;
TINYNV_SIZE(tinynv_rpc610_elem_t, 16);
TINYNV_AT(tinynv_rpc610_elem_t, checkSum, 8);
TINYNV_AT(tinynv_rpc610_elem_t, seqNum, 12);

#define TINYNV_RPC610_ELEM_HDR_SIZE 16u
// MCTP transport header: version 1, SOM=1, EOM=1, SEID/DEID/SEQ/TAG/TO = 0 (mctpCreateTransportHeader(1,1,0,0,0)).
#define TINYNV_RPC610_MCTP_HEADER 0xC0000001u
// NVDM over MCTP: type Vendor-Defined PCI (0x7e), IC=0, vendor 0x10de, NVDM type RM_RPC (0x25).
#define TINYNV_RPC610_NVDM_HEADER 0x2510DE7Eu
#define TINYNV_RPC610_MCTP_VERSION 0x1u
#define TINYNV_RPC610_VENDOR_ID_NV 0x10deu
#define TINYNV_RPC610_NVDM_TYPE_RM_RPC 0x25u

// The queue geometry the 610 host uses (message_queue_cpu.c, no CC): page-sized elements up to 16 pages. These are the
// same values boot610 puts in the startup args; pinned here so the codec and the args cannot drift apart.
#define TINYNV_RPC610_ELEM_SIZE_MIN 4096u
#define TINYNV_RPC610_ELEM_SIZE_MAX (4096u * 16u)
// GSP overrides rpcConstruct's generic one-page default in _kgspInitRpcInfrastructure:
// gspMsgQueueGetMaxRpcSize = queueElementSizeMax - queueElementHdrSize (no CC).
#define TINYNV_RPC610_MAX_RPC_SIZE (TINYNV_RPC610_ELEM_SIZE_MAX - TINYNV_RPC610_ELEM_HDR_SIZE)

// The shared XOR-fold checksum (gsp.c tinynv_rpc_checksum): exclusive-or of the record as 64-bit words, folded in
// half, zero-padded to 8. Identical algorithm in 570 and 610 (message_queue_cpu.c _checkSum32); shared, not copied.
// Declared here so the codec and the tests reach the one implementation.
uint32_t tinynv_rpc_checksum(const uint8_t *data, size_t n);

// Encode one RPC record into `out` (which must hold elem_count * elem_size bytes). Builds the 16-byte MCTP/NVDM
// element header, the shared 32-byte RPC header and the payload, zero-pads to the element boundary and sets the
// checksum so the whole-record fold is zero. `elem_size` is the queue's slot size (a power of two >= the minimum).
// Returns 0 and writes *elem_count / *rec_len, or -1 through tinynv_fail. `len` must fit one record's RPC ceiling.
int tinynv_rpc610_encode(uint8_t *out, size_t out_cap, uint32_t func, const void *payload, size_t len, uint32_t seq,
                         uint32_t elem_size, uint32_t *elem_count, size_t *rec_len);

// Validate and decode one assembled record (elem_count * elem_size bytes). Checks, in the 610 host's order: the RPC
// length bounds, the whole-record checksum fold == 0, the MCTP version and vendor id, and the sequence number. On
// success *hdr is the RPC header and *payload / *payload_len point into `rec`. The NVDM type is checked too (this queue
// carries only RM_RPC). Returns 0 or -1 through tinynv_fail with the specific reason.
int tinynv_rpc610_decode(const uint8_t *rec, size_t rec_len, uint32_t elem_size, uint32_t expect_seq,
                         tinynv_msg_header_t *hdr, const uint8_t **payload, uint32_t *payload_len,
                         uint32_t *elem_count);

// How many payload bytes fit the FIRST record and each CONTINUATION record at the negotiated GSP record ceiling.
static inline size_t tinynv_rpc610_first_payload_max(void) { return TINYNV_RPC610_MAX_RPC_SIZE - sizeof(tinynv_msg_header_t); }
static inline size_t tinynv_rpc610_cont_payload_max(void) { return TINYNV_RPC610_MAX_RPC_SIZE - sizeof(tinynv_msg_header_t); }

// --- the 610 RM control header (40 B; 570 was 24) ------------------------------------------------------------------
// rpc_gsp_rm_control_v03_00: flags became rmapiRpcFlags and rmctrlFlags/rmctrlAccessRight/reserved0 were appended, so
// params[] moved 24 -> 40. The driver sends all flags zero (it sends no rmapi/rmctrl flags), so the encoder places the
// five unchanged fields and zeroes the rest.
typedef struct {
  uint32_t hClient, hObject, cmd, status, paramsSize;
  uint32_t rmapiRpcFlags, rmctrlFlags, rmctrlAccessRight;
  uint64_t reserved0;
} tinynv_rpc610_rm_control_t;
TINYNV_SIZE(tinynv_rpc610_rm_control_t, 40);
TINYNV_AT(tinynv_rpc610_rm_control_t, cmd, 8);
TINYNV_AT(tinynv_rpc610_rm_control_t, paramsSize, 16);
TINYNV_AT(tinynv_rpc610_rm_control_t, rmapiRpcFlags, 20);
TINYNV_AT(tinynv_rpc610_rm_control_t, reserved0, 32);

void tinynv_rpc610_rm_control_fill(tinynv_rpc610_rm_control_t *c, uint32_t client, uint32_t object, uint32_t cmd,
                                   uint32_t params_size);

// --- the 610 GspSystemInfo (952 B; 570 was 928) --------------------------------------------------------------------
// The fields the driver fills at 0..96 keep their offsets; virtualConfigBits (u16) lands at 840 and bIsPassthru moves
// to 842. The encoder takes the driver's 570-shaped system info and places every field at its 610 offset, zeroing the
// rest (the new tail fields are all features this driver does not use). The exact size/offset are asserted against the
// verbatim-extracted pinned definition in test_headers_610 (extract_structs.py).
#define TINYNV_RPC610_SYSINFO_SIZE 952u
#define TINYNV_RPC610_SYSINFO_PASSTHRU_OFF 842u
#define TINYNV_RPC610_SYSINFO_VIRTCONF_OFF 840u
int tinynv_rpc610_system_info_encode(uint8_t *out, size_t out_cap, const tinynv_gsp_system_info_t *in);

// --- the 610 channel allocation (376 B; 570 was 368) ---------------------------------------------------------------
// NV_CHANNEL_ALLOC_PARAMS with hHandleVASpace inserted after hVASpace. Every field from hUserdMemory on shifts by 4 (and
// the u64-aligned ones by 8). The encoder copies field-by-field from the driver's 570-shaped working struct; the new
// hHandleVASpace is zero (handle-based VA space is not used - the driver hands physical descriptors). No cast between
// the incompatible layouts.
typedef struct {
  uint32_t hObjectError, hObjectBuffer;
  uint64_t gpFifoOffset;
  uint32_t gpFifoEntries, flags, hContextShare, hVASpace;
  uint32_t hHandleVASpace;              // NEW in 610
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
} tinynv_rpc610_gpfifo_alloc_t;
TINYNV_SIZE(tinynv_rpc610_gpfifo_alloc_t, 376);
TINYNV_AT(tinynv_rpc610_gpfifo_alloc_t, hVASpace, 28);
TINYNV_AT(tinynv_rpc610_gpfifo_alloc_t, hHandleVASpace, 32);
TINYNV_AT(tinynv_rpc610_gpfifo_alloc_t, hUserdMemory, 36);
TINYNV_AT(tinynv_rpc610_gpfifo_alloc_t, userdOffset, 72);
TINYNV_AT(tinynv_rpc610_gpfifo_alloc_t, engineType, 136);
TINYNV_AT(tinynv_rpc610_gpfifo_alloc_t, instanceMem, 152);
TINYNV_AT(tinynv_rpc610_gpfifo_alloc_t, userdMem, 176);
TINYNV_AT(tinynv_rpc610_gpfifo_alloc_t, ramfcMem, 200);
TINYNV_AT(tinynv_rpc610_gpfifo_alloc_t, mthdbufMem, 224);
TINYNV_AT(tinynv_rpc610_gpfifo_alloc_t, internalFlags, 252);
TINYNV_AT(tinynv_rpc610_gpfifo_alloc_t, errorNotifierMem, 256);
TINYNV_AT(tinynv_rpc610_gpfifo_alloc_t, ProcessID, 304);
TINYNV_AT(tinynv_rpc610_gpfifo_alloc_t, tpcConfigID, 368);

void tinynv_rpc610_channel_encode(tinynv_rpc610_gpfifo_alloc_t *out, const tinynv_gpfifo_alloc_t *in);

// --- the 610 VA space allocation (56 B; 570 was 48) ----------------------------------------------------------------
// NV_VASPACE_ALLOCATION_PARAMETERS with pasid appended. Used offsets unchanged; pasid is zero for an externally-owned
// VA space whose page tables the driver builds (the 610 driver only sets a real pasid for SVM/ATS, which this is not).
typedef struct {
  uint32_t index, flags;
  uint64_t vaSize, vaStartInternal, vaLimitInternal;
  uint32_t bigPageSize, _pad;
  uint64_t vaBase;
  uint32_t pasid;                       // NEW in 610
  uint32_t _pad2;
} tinynv_rpc610_vaspace_alloc_t;
TINYNV_SIZE(tinynv_rpc610_vaspace_alloc_t, 56);
TINYNV_AT(tinynv_rpc610_vaspace_alloc_t, vaBase, 40);
TINYNV_AT(tinynv_rpc610_vaspace_alloc_t, pasid, 48);

void tinynv_rpc610_vaspace_encode(tinynv_rpc610_vaspace_alloc_t *out, const tinynv_vaspace_alloc_t *in);

// --- the 610 GR static info (4032 B; 570 was 3712) -----------------------------------------------------------------
// NV0080_CTRL_GR_INFO_MAX_SIZE grew 58 -> 63, so each engine's infoList is 63 entries (504 B) and the 8-engine block is
// 4032 B. The indices the driver reads (NUM_GPCS 20, NUM_TPC_PER_GPC 23, NUM_SM_PER_TPC 32, MAX_WARPS_PER_SM 13,
// SM_VERSION 12) are unchanged; only the stride grows. The reader bounds-checks the buffer and the index.
#define TINYNV_RPC610_GR_INFO_COUNT 63u
#define TINYNV_RPC610_GR_ENGINE_SIZE 504u
#define TINYNV_RPC610_GR_STATIC_SIZE 4032u
typedef struct { uint32_t index, data; } tinynv_rpc610_gr_info_t;
TINYNV_SIZE(tinynv_rpc610_gr_info_t, 8);
typedef struct { tinynv_rpc610_gr_info_t infoList[TINYNV_RPC610_GR_INFO_COUNT]; } tinynv_rpc610_gr_engine_info_t;
TINYNV_SIZE(tinynv_rpc610_gr_engine_info_t, 504);
typedef struct { tinynv_rpc610_gr_engine_info_t engineInfo[8]; } tinynv_rpc610_gr_static_info_t;
TINYNV_SIZE(tinynv_rpc610_gr_static_info_t, 4032);

// Read one GR info value from a raw 610 static-info reply. `engine` < 8, `index` < 63. Returns 0 or -1 (bad length /
// out-of-range), never reads past the buffer.
int tinynv_rpc610_gr_info_get(const void *params, size_t len, unsigned engine, unsigned index, uint32_t *data);

// --- the 610 GPFIFO schedule (3 B; 570 was 2) ----------------------------------------------------------------------
// NVA06C_CTRL_GPFIFO_SCHEDULE_PARAMS gained bSkipEnable. Used offsets unchanged; the new byte is zero (do not skip
// enable - the driver wants the group scheduled).
typedef struct { uint8_t bEnable, bSkipSubmit, bSkipEnable; } tinynv_rpc610_gpfifo_schedule_t;
TINYNV_SIZE(tinynv_rpc610_gpfifo_schedule_t, 3);
void tinynv_rpc610_schedule_fill(tinynv_rpc610_gpfifo_schedule_t *s, uint8_t enable, uint8_t skip_submit);

// --- the 610 RUSD controls and shared block ------------------------------------------------------------------------
// INIT_USER_SHARED_DATA grew 8 -> 16 B: bInit (NvBool) before physAddr@8. bInit=1 arms the block (the 610 header:
// physAddr "0 to de-initialize", bInit the explicit init flag). SET_DATA_POLL is 16 B in both, but 610 renamed
// pollFrequencyMs -> pollIntervalMs at the same offset 8.
typedef struct { uint8_t bInit, _pad[7]; uint64_t physAddr; } tinynv_rpc610_rusd_init_t;
TINYNV_SIZE(tinynv_rpc610_rusd_init_t, 16);
TINYNV_AT(tinynv_rpc610_rusd_init_t, physAddr, 8);
typedef struct { uint64_t polledDataMask; uint32_t pollIntervalMs; uint32_t _pad; } tinynv_rpc610_rusd_poll_t;
TINYNV_SIZE(tinynv_rpc610_rusd_poll_t, 16);
TINYNV_AT(tinynv_rpc610_rusd_poll_t, pollIntervalMs, 8);

void tinynv_rpc610_rusd_init_fill(tinynv_rpc610_rusd_init_t *p, uint64_t phys_addr);
void tinynv_rpc610_rusd_poll_fill(tinynv_rpc610_rusd_poll_t *p, uint64_t poll_mask, uint32_t interval_ms);

// The 610 shared sensor block is 3152 B (570: 464) with new offsets and a larger clock/throttle section. The reader
// maps the fields the driver actually uses into the driver's 570-shaped tinynv_rusd_t, each with its own timestamp
// retry protocol (read ts, copy, re-read ts). Fields 610 removed or grew are handled explicitly: clkThrottleReason is
// read from the prefix of the new clkThrottleInfo, the clock domains keep only targetClkMHz, memEcc keeps the legacy
// counts prefix. A field whose timestamp never settles is left zero ("not reported"), exactly as the 570 reader.
#define TINYNV_RPC610_RUSD_SIZE 3152u
int tinynv_rpc610_rusd_read(const volatile void *raw, size_t len, tinynv_rusd_t *out);

// --- the 610 host requests (the CPU sequencer's replacements) ------------------------------------------------------
// Event numbers (rpc_global_enums.h): the sequencer event 4098 is REMOVED in 610; these two are added.
#define TINYNV_MSG_EVENT_GSP_LOAD_EXEC_GENERIC_BOOTLOADER 0x1026u  // 4134
#define TINYNV_MSG_EVENT_GSP_LOAD_EXEC_HS_BINARY 0x1027u           // 4135

// RM_FLCN_BL_DMEM_DESC (rmflcnbl.h): the descriptor the generic bootloader reads from DMEM to load the rest of the
// code. RM_FLCN_U64 is NvU64_ALIGN32 (two u32, 4-byte aligned), so the whole descriptor is 84 B.
typedef struct { uint32_t lo, hi; } tinynv_rpc610_flcn_u64_t;
typedef struct {
  uint32_t reserved[4];
  uint32_t signature[4];
  uint32_t ctxDma;
  tinynv_rpc610_flcn_u64_t codeDmaBase;
  uint32_t nonSecureCodeOff, nonSecureCodeSize;
  uint32_t secureCodeOff, secureCodeSize;
  uint32_t codeEntryPoint;
  tinynv_rpc610_flcn_u64_t dataDmaBase;
  uint32_t dataSize, argc, argv;
} tinynv_rpc610_bl_dmem_desc_t;
TINYNV_SIZE(tinynv_rpc610_bl_dmem_desc_t, 84);
TINYNV_AT(tinynv_rpc610_bl_dmem_desc_t, ctxDma, 32);
TINYNV_AT(tinynv_rpc610_bl_dmem_desc_t, codeDmaBase, 36);
TINYNV_AT(tinynv_rpc610_bl_dmem_desc_t, dataDmaBase, 64);

// GspLoadExecGenericBootloaderParams (rmgspseq.h): the descriptor plus its size, the address space and cache attrib.
typedef struct {
  tinynv_rpc610_bl_dmem_desc_t dmemDesc;
  uint32_t dmemDescSize, addrSpace, cpuCacheAttrib;
  uint32_t reserved[4];
} tinynv_rpc610_load_exec_bl_t;
TINYNV_SIZE(tinynv_rpc610_load_exec_bl_t, 112);
TINYNV_AT(tinynv_rpc610_load_exec_bl_t, dmemDescSize, 84);
TINYNV_AT(tinynv_rpc610_load_exec_bl_t, addrSpace, 88);
TINYNV_AT(tinynv_rpc610_load_exec_bl_t, cpuCacheAttrib, 92);

// GspLoadExecHsBinaryParams (rmgspseq.h): where the signed heavy-secure binary lives and how to launch it.
typedef struct {
  uint64_t imemPhysAddr, dmemPhysAddr;
  uint64_t reserved64[2];
  uint32_t ucodeImemVA, ucodeImemPA, ucodeImemSize;
  uint32_t ucodeDmemVA, ucodeDmemPA, ucodeDmemSize;
  uint32_t hsSigDmemAddr, engineIdMask, ucodeId;
  uint32_t reserved32[3];
} tinynv_rpc610_load_exec_hs_t;
TINYNV_SIZE(tinynv_rpc610_load_exec_hs_t, 80);
TINYNV_AT(tinynv_rpc610_load_exec_hs_t, ucodeImemVA, 32);
TINYNV_AT(tinynv_rpc610_load_exec_hs_t, hsSigDmemAddr, 56);
TINYNV_AT(tinynv_rpc610_load_exec_hs_t, ucodeId, 64);

// Parse + validate the two host-request payloads. The generic bootloader checks dmemDescSize matches the struct, the
// ctxDma index is a real aperture (< 8), the address space is one the loader supports, and the code/data ranges are
// 256-aligned and do not wrap. The HS binary checks the IMEM/DMEM sizes are 256-aligned and non-zero and the engine
// mask / ucode id are present. These are the bounds the 610 host asserts before it touches a falcon; refusing here
// means a malformed request is a clear error, not a hung core.
int tinynv_rpc610_parse_load_exec_bl(const uint8_t *msg, size_t len, tinynv_rpc610_load_exec_bl_t *out);
int tinynv_rpc610_parse_load_exec_hs(const uint8_t *msg, size_t len, tinynv_rpc610_load_exec_hs_t *out);

// --- event classification ------------------------------------------------------------------------------------------
// What the 610 status queue may carry and what this driver does with it. The sequencer event is classified separately
// so a 610 firmware that (wrongly) sends it is refused with a specific diagnostic rather than run through the 570
// opcode interpreter, which no longer matches any 610 request.
typedef enum {
  TINYNV_RPC610_EVENT_INIT_DONE = 0,   // 4097: the start-up notice
  TINYNV_RPC610_EVENT_OS_ERROR_LOG,    // 4102: a log line / error report
  TINYNV_RPC610_EVENT_MMU_FAULT,       // 4101: a fault notification
  TINYNV_RPC610_EVENT_LIBOS_PRINT,     // 4108: a one-way libos print; consume it, never a host request or an error
  TINYNV_RPC610_EVENT_LOAD_EXEC_BL,    // 4134: run the generic bootloader on the host's behalf
  TINYNV_RPC610_EVENT_LOAD_EXEC_HS,    // 4135: run a heavy-secure binary on the host's behalf
  TINYNV_RPC610_EVENT_REPLY,           // a function number: an RPC reply, matched by the caller
  TINYNV_RPC610_EVENT_SEQUENCER_GONE,  // 4098: removed in 610 - refuse
  TINYNV_RPC610_EVENT_UNKNOWN          // anything else - refuse with a diagnostic
} tinynv_rpc610_event_kind_t;
tinynv_rpc610_event_kind_t tinynv_rpc610_event_kind(uint32_t function);

#endif
