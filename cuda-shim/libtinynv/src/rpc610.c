// The 610.43.03 RPC / resource-management adapters. See rpc610.h for the contract and the audit citations. Pure
// functions: no device, no GPU, no allocation. Every refusal goes through tinynv_fail so the offline suite can assert
// on the reason, and every length/offset computation is bounds-checked rather than trusted.
#include "rpc610.h"
#include "internal.h"
#include <string.h>

// --- little-endian readers/writers that stay inside the buffer -----------------------------------------------------
static uint32_t rd32(const uint8_t *p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static void wr32(uint8_t *p, uint32_t v) {
  p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24);
}
static uint64_t rd64(const uint8_t *p) { return (uint64_t)rd32(p) | ((uint64_t)rd32(p + 4) << 32); }
static void wr64(uint8_t *p, uint64_t v) { wr32(p, (uint32_t)v); wr32(p + 4, (uint32_t)(v >> 32)); }

static uint64_t round_up64(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }

// --- the shared checksum -------------------------------------------------------------------------------------------
// The XOR-fold is byte-identical in 570 (gsp.c) and 610 (message_queue_cpu.c _checkSum32). It lives in gsp.c as
// tinynv_rpc_checksum and is shared, not duplicated; test_headers_610 asserts the algorithm's inputs (the element
// layout) and the codec tests assert the fold against fixtures. This translation unit only calls it.

// --- the 610 queue element codec -----------------------------------------------------------------------------------
int tinynv_rpc610_encode(uint8_t *out, size_t out_cap, uint32_t func, const void *payload, size_t len, uint32_t seq,
                         uint32_t elem_size, uint32_t *elem_count, size_t *rec_len) {
  if (elem_count) *elem_count = 0;
  if (rec_len) *rec_len = 0;
  if (!out) return tinynv_fail("the 610 rpc encoder was given no output buffer");
  if (elem_size < TINYNV_RPC610_ELEM_SIZE_MIN || (elem_size & (elem_size - 1)))
    return tinynv_fail("the 610 queue element size %u is not a power of two >= %u", elem_size, TINYNV_RPC610_ELEM_SIZE_MIN);
  // GSP negotiates maxRpcSize from the queue ceiling, excluding the transport header.
  size_t rpc_max = TINYNV_RPC610_MAX_RPC_SIZE;
  if (rpc_max > TINYNV_RPC610_ELEM_SIZE_MAX - TINYNV_RPC610_ELEM_HDR_SIZE)
    rpc_max = TINYNV_RPC610_ELEM_SIZE_MAX - TINYNV_RPC610_ELEM_HDR_SIZE;
  if (len + sizeof(tinynv_msg_header_t) > rpc_max)
    return tinynv_fail("a %zu byte 610 rpc payload does not fit one record (ceiling %zu bytes of header+payload)",
                       len, rpc_max);

  size_t msg_len = TINYNV_RPC610_ELEM_HDR_SIZE + sizeof(tinynv_msg_header_t) + len;
  uint32_t count = (uint32_t)round_up64(msg_len, elem_size) / elem_size;
  size_t total = (size_t)count * elem_size;
  if (out_cap < total)
    return tinynv_fail("the 610 rpc record needs %zu bytes, the buffer holds %zu", total, out_cap);

  memset(out, 0, total);
  tinynv_msg_header_t mh;
  memset(&mh, 0, sizeof mh);
  mh.header_version = TINYNV_MSG_HEADER_VERSION;
  mh.signature = TINYNV_MSG_SIGNATURE_VALID;
  mh.rpc_result = mh.rpc_result_private = TINYNV_MSG_RESULT_PENDING;
  mh.function = func;
  mh.length = (uint32_t)(len + sizeof(tinynv_msg_header_t));
  memcpy(out + TINYNV_RPC610_ELEM_HDR_SIZE, &mh, sizeof mh);
  if (len) memcpy(out + TINYNV_RPC610_ELEM_HDR_SIZE + sizeof mh, payload, len);

  wr32(out + 0, TINYNV_RPC610_MCTP_HEADER);
  wr32(out + 4, TINYNV_RPC610_NVDM_HEADER);
  wr32(out + 12, seq);
  // checkSum at 8 stays zero for the fold, then is set so the whole-record fold is zero (the 610 host's rule).
  wr32(out + 8, tinynv_rpc_checksum(out, msg_len));

  if (elem_count) *elem_count = count;
  if (rec_len) *rec_len = total;
  return 0;
}

int tinynv_rpc610_decode(const uint8_t *rec, size_t rec_len, uint32_t elem_size, uint32_t expect_seq,
                         tinynv_msg_header_t *hdr, const uint8_t **payload, uint32_t *payload_len,
                         uint32_t *elem_count) {
  if (elem_count) *elem_count = 0;
  if (payload) *payload = NULL;
  if (payload_len) *payload_len = 0;
  if (!rec) return tinynv_fail("the 610 rpc decoder was given no record");
  if (elem_size < TINYNV_RPC610_ELEM_SIZE_MIN || (elem_size & (elem_size - 1)))
    return tinynv_fail("the 610 queue element size %u is not a power of two >= %u", elem_size, TINYNV_RPC610_ELEM_SIZE_MIN);
  if (rec_len < TINYNV_RPC610_ELEM_HDR_SIZE + sizeof(tinynv_msg_header_t))
    return tinynv_fail("the 610 rpc record is %zu bytes, shorter than its element+rpc headers", rec_len);

  // The RPC length drives everything, exactly as the 610 host derives the element count from it.
  uint32_t rpc_len = rd32(rec + TINYNV_RPC610_ELEM_HDR_SIZE + 8);
  size_t msg_len = TINYNV_RPC610_ELEM_HDR_SIZE + (size_t)rpc_len;
  if (rpc_len < sizeof(tinynv_msg_header_t))
    return tinynv_fail("the 610 rpc length %u is shorter than the %zu byte rpc header", rpc_len, sizeof(tinynv_msg_header_t));
  if (msg_len > TINYNV_RPC610_ELEM_SIZE_MAX)
    return tinynv_fail("the 610 rpc length %u exceeds the %u byte queue element ceiling", rpc_len, TINYNV_RPC610_ELEM_SIZE_MAX);
  uint32_t count = (uint32_t)round_up64(msg_len, elem_size) / elem_size;
  if ((size_t)count * elem_size > rec_len)
    return tinynv_fail("the 610 rpc record declares %u elements (%zu bytes) but only %zu were assembled",
                       count, (size_t)count * elem_size, rec_len);

  // The whole-record fold must be zero (the checksum was set to make it so). Computed at the natural length, the tail
  // zero-extended, which is what the 610 host does; the assembled padding is zero so this reads only real bytes.
  if (tinynv_rpc_checksum(rec, msg_len) != 0)
    return tinynv_fail("the 610 rpc record's checksum does not fold to zero: the memory gsp-rm wrote is corrupt");

  uint32_t mctp = rd32(rec + 0), nvdm = rd32(rec + 4), seq = rd32(rec + 12);
  if ((mctp & 0xfu) != TINYNV_RPC610_MCTP_VERSION)
    return tinynv_fail("the 610 rpc MCTP header version is %u, not %u", mctp & 0xfu, TINYNV_RPC610_MCTP_VERSION);
  if (((nvdm >> 8) & 0xffffu) != TINYNV_RPC610_VENDOR_ID_NV)
    return tinynv_fail("the 610 rpc NVDM vendor id is %#x, not NVIDIA %#x", (nvdm >> 8) & 0xffffu, TINYNV_RPC610_VENDOR_ID_NV);
  if (((nvdm >> 24) & 0xffu) != TINYNV_RPC610_NVDM_TYPE_RM_RPC)
    return tinynv_fail("the 610 rpc NVDM type is %#x, not RM_RPC %#x", (nvdm >> 24) & 0xffu, TINYNV_RPC610_NVDM_TYPE_RM_RPC);
  if (seq != expect_seq)
    return tinynv_fail("the 610 rpc sequence number is %u, expected %u: a record was lost or the memory is corrupt",
                       seq, expect_seq);

  if (hdr) memcpy(hdr, rec + TINYNV_RPC610_ELEM_HDR_SIZE, sizeof *hdr);
  if (payload) *payload = rec + TINYNV_RPC610_ELEM_HDR_SIZE + sizeof(tinynv_msg_header_t);
  if (payload_len) *payload_len = rpc_len - (uint32_t)sizeof(tinynv_msg_header_t);
  if (elem_count) *elem_count = count;
  return 0;
}

// --- the 610 RM control header -------------------------------------------------------------------------------------
void tinynv_rpc610_rm_control_fill(tinynv_rpc610_rm_control_t *c, uint32_t client, uint32_t object, uint32_t cmd,
                                   uint32_t params_size) {
  memset(c, 0, sizeof *c);
  c->hClient = client;
  c->hObject = object;
  c->cmd = cmd;
  c->paramsSize = params_size;
  // rmapiRpcFlags / rmctrlFlags / rmctrlAccessRight / reserved0 stay zero: this driver sends no rmapi/rmctrl flags.
}

// --- the 610 GspSystemInfo -----------------------------------------------------------------------------------------
int tinynv_rpc610_system_info_encode(uint8_t *out, size_t out_cap, const tinynv_gsp_system_info_t *in) {
  if (!out || !in) return tinynv_fail("the 610 system-info encoder was given a null buffer");
  if (out_cap < TINYNV_RPC610_SYSINFO_SIZE)
    return tinynv_fail("the 610 system-info buffer holds %zu bytes, the 610 struct is %u", out_cap, TINYNV_RPC610_SYSINFO_SIZE);
  memset(out, 0, TINYNV_RPC610_SYSINFO_SIZE);
  // Every field the driver fills keeps its 570 offset through 96 (asserted in test_headers_610). Copy them by name so
  // a future drift in either layout is a compile error, not a silent misplacement.
  wr64(out + 0, in->gpuPhysAddr);
  wr64(out + 8, in->gpuPhysFbAddr);
  wr64(out + 16, in->gpuPhysInstAddr);
  wr64(out + 24, in->gpuPhysIoAddr);
  wr64(out + 32, in->nvDomainBusDeviceFunc);
  wr64(out + 72, in->maxUserVa);
  wr32(out + 80, in->pciConfigMirrorBase);
  wr32(out + 84, in->pciConfigMirrorSize);
  wr32(out + 88, in->PCIDeviceID);
  wr32(out + 92, in->PCISubDeviceID);
  wr32(out + 96, in->PCIRevisionID);
  // bIsPassthru moved 840 -> 842 (virtualConfigBits, which this driver leaves zero, was inserted at 840).
  out[TINYNV_RPC610_SYSINFO_PASSTHRU_OFF] = in->bIsPassthru;
  return 0;
}

// --- the 610 channel allocation ------------------------------------------------------------------------------------
void tinynv_rpc610_channel_encode(tinynv_rpc610_gpfifo_alloc_t *out, const tinynv_gpfifo_alloc_t *in) {
  memset(out, 0, sizeof *out);
  out->hObjectError = in->hObjectError;
  out->hObjectBuffer = in->hObjectBuffer;
  out->gpFifoOffset = in->gpFifoOffset;
  out->gpFifoEntries = in->gpFifoEntries;
  out->flags = in->flags;
  out->hContextShare = in->hContextShare;
  out->hVASpace = in->hVASpace;
  out->hHandleVASpace = 0;   // handle-based VA space is not used; the driver hands physical descriptors
  for (int i = 0; i < 8; i++) { out->hUserdMemory[i] = in->hUserdMemory[i]; out->userdOffset[i] = in->userdOffset[i]; }
  out->engineType = in->engineType;
  out->cid = in->cid;
  out->subDeviceId = in->subDeviceId;
  out->hObjectEccError = in->hObjectEccError;
  out->instanceMem = in->instanceMem;
  out->userdMem = in->userdMem;
  out->ramfcMem = in->ramfcMem;
  out->mthdbufMem = in->mthdbufMem;
  out->hPhysChannelGroup = in->hPhysChannelGroup;
  out->internalFlags = in->internalFlags;
  out->errorNotifierMem = in->errorNotifierMem;
  out->eccErrorNotifierMem = in->eccErrorNotifierMem;
  out->ProcessID = in->ProcessID;
  out->SubProcessID = in->SubProcessID;
  memcpy(out->encryptIv, in->encryptIv, sizeof out->encryptIv);
  memcpy(out->decryptIv, in->decryptIv, sizeof out->decryptIv);
  memcpy(out->hmacNonce, in->hmacNonce, sizeof out->hmacNonce);
  out->tpcConfigID = in->tpcConfigID;
}

// --- the 610 VA space allocation -----------------------------------------------------------------------------------
void tinynv_rpc610_vaspace_encode(tinynv_rpc610_vaspace_alloc_t *out, const tinynv_vaspace_alloc_t *in) {
  memset(out, 0, sizeof *out);
  out->index = in->index;
  out->flags = in->flags;
  out->vaSize = in->vaSize;
  out->vaStartInternal = in->vaStartInternal;
  out->vaLimitInternal = in->vaLimitInternal;
  out->bigPageSize = in->bigPageSize;
  out->vaBase = in->vaBase;
  out->pasid = 0;   // externally-owned VA space: no process address-space id
}

// --- the 610 GR static info ----------------------------------------------------------------------------------------
int tinynv_rpc610_gr_info_get(const void *params, size_t len, unsigned engine, unsigned index, uint32_t *data) {
  if (data) *data = 0;
  if (!params) return tinynv_fail("the 610 GR info reader was given no reply");
  if (len < TINYNV_RPC610_GR_STATIC_SIZE)
    return tinynv_fail("the 610 GR static-info reply is %zu bytes, the 610 layout is %u", len, TINYNV_RPC610_GR_STATIC_SIZE);
  if (engine >= 8) return tinynv_fail("the 610 GR info engine %u is past the 8 the reply carries", engine);
  if (index >= TINYNV_RPC610_GR_INFO_COUNT)
    return tinynv_fail("the 610 GR info index %u is past the %u the 610 array carries", index, TINYNV_RPC610_GR_INFO_COUNT);
  const uint8_t *p = (const uint8_t *)params + (size_t)engine * TINYNV_RPC610_GR_ENGINE_SIZE + (size_t)index * 8;
  if (data) *data = rd32(p + 4);   // the data word of {index, data}
  return 0;
}

// --- the 610 GPFIFO schedule ---------------------------------------------------------------------------------------
void tinynv_rpc610_schedule_fill(tinynv_rpc610_gpfifo_schedule_t *s, uint8_t enable, uint8_t skip_submit) {
  s->bEnable = enable;
  s->bSkipSubmit = skip_submit;
  s->bSkipEnable = 0;   // do not skip enable: the driver wants the group scheduled
}

// --- the 610 RUSD controls -----------------------------------------------------------------------------------------
void tinynv_rpc610_rusd_init_fill(tinynv_rpc610_rusd_init_t *p, uint64_t phys_addr) {
  memset(p, 0, sizeof *p);
  p->bInit = 1;   // arm the block (610: bInit is the explicit init flag, physAddr 0 would de-initialize)
  p->physAddr = phys_addr;
}
void tinynv_rpc610_rusd_poll_fill(tinynv_rpc610_rusd_poll_t *p, uint64_t poll_mask, uint32_t interval_ms) {
  memset(p, 0, sizeof *p);
  p->polledDataMask = poll_mask;
  p->pollIntervalMs = interval_ms;   // 610 renamed pollFrequencyMs -> pollIntervalMs, same offset/width
}

// --- the 610 RUSD shared block reader ------------------------------------------------------------------------------
// The timestamp validity rule is unchanged (RUSD_SEQ_*): a field is valid when its timestamp is neither zero nor an
// odd sequence in the high range. Same helper the 570 reader uses.
static int rusd610_seq_ok(uint64_t seq) {
  return (seq < TINYNV_RUSD_SEQ_START && seq != TINYNV_RUSD_TS_INVALID) ||
         (seq >= TINYNV_RUSD_SEQ_START && (seq & 1) == 0);
}

// Copy one field with the read-ts / copy / re-read-ts retry. `off` is the field's 610 offset, `n` the bytes to copy
// into the 570-shaped target. Ten attempts, then leave the target zero ("not reported"), exactly as the 570 reader.
static void rusd610_field(const volatile uint8_t *raw, size_t off, void *dst, size_t n) {
  memset(dst, 0, n);
  for (int a = 0; a < 10; a++) {
    uint64_t seq = rd64((const uint8_t *)raw + off);
    __atomic_thread_fence(__ATOMIC_ACQUIRE);
    if (!rusd610_seq_ok(seq)) continue;
    memcpy(dst, (const void *)(raw + off), n);
    __atomic_thread_fence(__ATOMIC_ACQUIRE);
    if (seq == rd64((const uint8_t *)raw + off)) return;
    memset(dst, 0, n);
  }
}

int tinynv_rpc610_rusd_read(const volatile void *raw, size_t len, tinynv_rusd_t *out) {
  memset(out, 0, sizeof *out);
  if (!raw) return tinynv_fail("the 610 sensor reader was given no shared block");
  if (len < TINYNV_RPC610_RUSD_SIZE)
    return tinynv_fail("the 610 sensor block is %zu bytes, the 610 layout is %u", len, TINYNV_RPC610_RUSD_SIZE);
  const volatile uint8_t *r = (const volatile uint8_t *)raw;

  // Fields whose 610 layout is byte-identical to 570, at their new offsets (asserted in test_headers_610).
  rusd610_field(r, 0, &out->bar1MemoryInfo, sizeof out->bar1MemoryInfo);          // RUSD_BAR1_MEMORY_INFO
  rusd610_field(r, 16, &out->pmaMemoryInfo, sizeof out->pmaMemoryInfo);           // RUSD_PMA_MEMORY_INFO
  rusd610_field(r, 1992, &out->perfDevUtil, sizeof out->perfDevUtil);             // RUSD_PERF_DEVICE_UTILIZATION
  rusd610_field(r, 2280, &out->perfCurrentPstate, sizeof out->perfCurrentPstate); // RUSD_PERF_CURRENT_PSTATE
  rusd610_field(r, 2296, &out->powerLimitGpu, sizeof out->powerLimitGpu);         // RUSD_POWER_LIMITS
  rusd610_field(r, 2312, &out->temperatures[0], sizeof out->temperatures[0]);     // GPU
  rusd610_field(r, 2328, &out->temperatures[1], sizeof out->temperatures[1]);     // MEMORY
  rusd610_field(r, 2440, &out->avgPowerUsage, sizeof out->avgPowerUsage);         // RUSD_AVG_POWER_USAGE
  rusd610_field(r, 2464, &out->instPowerUsage, sizeof out->instPowerUsage);       // RUSD_INST_POWER_USAGE
  rusd610_field(r, 2504, &out->pciBusData, sizeof out->pciBusData);               // RUSD_PCIE_DATA

  // memEcc grew (a new RUSD_ECC_INFO tail); the legacy counts prefix the driver reads is unchanged. Copy just it.
  rusd610_field(r, 2040, &out->memEcc, sizeof out->memEcc);

  // clkThrottleReason became the prefix of the much larger clkThrottleInfo: ts @112, reasonMask @120. The 570 target
  // is {ts, reasonMask}; read those two words with the same retry, leaving the rest of the 610 block unread.
  for (int a = 0; a < 10; a++) {
    uint64_t seq = rd64((const uint8_t *)r + 112);
    __atomic_thread_fence(__ATOMIC_ACQUIRE);
    if (!rusd610_seq_ok(seq)) continue;
    out->clkThrottleReason.ts = seq;
    out->clkThrottleReason.reasonMask = rd32((const uint8_t *)r + 120);
    __atomic_thread_fence(__ATOMIC_ACQUIRE);
    if (seq == rd64((const uint8_t *)r + 112)) break;
    out->clkThrottleReason.ts = 0;
    out->clkThrottleReason.reasonMask = 0;
  }

  // clkPublicDomainInfos: each 610 domain is {targetClkMHz, actualClkKHz}; the 570 target keeps only targetClkMHz[4].
  // Read the block timestamp once, then the four target words; re-read the timestamp to confirm nothing moved.
  for (int a = 0; a < 10; a++) {
    uint64_t seq = rd64((const uint8_t *)r + 72);
    __atomic_thread_fence(__ATOMIC_ACQUIRE);
    if (!rusd610_seq_ok(seq)) continue;
    out->clkPublicDomainInfos.ts = seq;
    for (int d = 0; d < 4; d++) out->clkPublicDomainInfos.targetClkMHz[d] = rd32((const uint8_t *)r + 80 + (size_t)d * 8);
    __atomic_thread_fence(__ATOMIC_ACQUIRE);
    if (seq == rd64((const uint8_t *)r + 72)) break;
    memset(&out->clkPublicDomainInfos, 0, sizeof out->clkPublicDomainInfos);
  }
  return 0;
}

// --- the 610 host-request payloads ---------------------------------------------------------------------------------
int tinynv_rpc610_parse_load_exec_bl(const uint8_t *msg, size_t len, tinynv_rpc610_load_exec_bl_t *out) {
  memset(out, 0, sizeof *out);
  if (!msg) return tinynv_fail("the 610 generic-bootloader request is empty");
  if (len < sizeof *out)
    return tinynv_fail("the 610 generic-bootloader request is %zu bytes, shorter than its %zu byte payload", len, sizeof *out);
  memcpy(out, msg, sizeof *out);
  // The 610 host asserts the descriptor size matches its own struct before it copies it into DMEM.
  if (out->dmemDescSize != sizeof(tinynv_rpc610_bl_dmem_desc_t))
    return tinynv_fail("the 610 generic-bootloader descriptor size is %u, not the %zu byte RM_FLCN_BL_DMEM_DESC",
                       out->dmemDescSize, sizeof(tinynv_rpc610_bl_dmem_desc_t));
  // ctxDma must name a real FBIF aperture (NV_PFALCON_FBIF_TRANSCFG__SIZE_1 = 8).
  if (out->dmemDesc.ctxDma >= 8)
    return tinynv_fail("the 610 generic-bootloader ctxDma %u is past the 8 FBIF apertures", out->dmemDesc.ctxDma);
  // The loader supports FB memory and coherent/uncached system memory; anything else is a refusal, not a guess.
  if (out->addrSpace != 1 && out->addrSpace != 2)
    return tinynv_fail("the 610 generic-bootloader address space %u is not SYSMEM(1) or FBMEM(2)", out->addrSpace);
  // The code/data ranges the bootloader will DMA must be 256-aligned and must not wrap the 64-bit address.
  const tinynv_rpc610_bl_dmem_desc_t *d = &out->dmemDesc;
  if ((d->nonSecureCodeOff & 255) || (d->nonSecureCodeSize & 255) ||
      (d->secureCodeOff & 255) || (d->secureCodeSize & 255) || (d->dataSize & 255))
    return tinynv_fail("the 610 generic-bootloader code/data range is not 256-byte aligned");
  uint64_t code_base = (uint64_t)d->codeDmaBase.lo | ((uint64_t)d->codeDmaBase.hi << 32);
  uint64_t data_base = (uint64_t)d->dataDmaBase.lo | ((uint64_t)d->dataDmaBase.hi << 32);
  if (d->nonSecureCodeSize && code_base + d->nonSecureCodeOff + d->nonSecureCodeSize < code_base)
    return tinynv_fail("the 610 generic-bootloader non-secure code range wraps the address space");
  if (d->secureCodeSize && code_base + d->secureCodeOff + d->secureCodeSize < code_base)
    return tinynv_fail("the 610 generic-bootloader secure code range wraps the address space");
  if (d->dataSize && data_base + d->dataSize < data_base)
    return tinynv_fail("the 610 generic-bootloader data range wraps the address space");
  return 0;
}

int tinynv_rpc610_parse_load_exec_hs(const uint8_t *msg, size_t len, tinynv_rpc610_load_exec_hs_t *out) {
  memset(out, 0, sizeof *out);
  if (!msg) return tinynv_fail("the 610 hs-binary request is empty");
  if (len < sizeof *out)
    return tinynv_fail("the 610 hs-binary request is %zu bytes, shorter than its %zu byte payload", len, sizeof *out);
  memcpy(out, msg, sizeof *out);
  // The IMEM/DMEM transfers are 256-byte blocks; a zero or misaligned size is a malformed request.
  if (!out->ucodeImemSize || (out->ucodeImemSize & 255))
    return tinynv_fail("the 610 hs-binary IMEM size %u is zero or not 256-byte aligned", out->ucodeImemSize);
  if (!out->ucodeDmemSize || (out->ucodeDmemSize & 255))
    return tinynv_fail("the 610 hs-binary DMEM size %u is zero or not 256-byte aligned", out->ucodeDmemSize);
  if (!out->engineIdMask)
    return tinynv_fail("the 610 hs-binary engine id mask is zero: the boot ROM would refuse the signature");
  // ucodeId is an 8-bit BROM field (NV_PFALCON2_FALCON_BROM_CURR_UCODE_ID_VAL 7:0).
  if (out->ucodeId > 0xff)
    return tinynv_fail("the 610 hs-binary ucode id %u is past the 8-bit BROM field", out->ucodeId);
  return 0;
}

// --- event classification ------------------------------------------------------------------------------------------
tinynv_rpc610_event_kind_t tinynv_rpc610_event_kind(uint32_t function) {
  switch (function) {
    case TINYNV_MSG_EVENT_GSP_INIT_DONE: return TINYNV_RPC610_EVENT_INIT_DONE;
    case TINYNV_MSG_EVENT_OS_ERROR_LOG: return TINYNV_RPC610_EVENT_OS_ERROR_LOG;
    case TINYNV_MSG_EVENT_MMU_FAULT_QUEUED: return TINYNV_RPC610_EVENT_MMU_FAULT;
    case TINYNV_MSG_EVENT_UCODE_LIBOS_PRINT: return TINYNV_RPC610_EVENT_LIBOS_PRINT;
    case TINYNV_MSG_EVENT_GSP_LOAD_EXEC_GENERIC_BOOTLOADER: return TINYNV_RPC610_EVENT_LOAD_EXEC_BL;
    case TINYNV_MSG_EVENT_GSP_LOAD_EXEC_HS_BINARY: return TINYNV_RPC610_EVENT_LOAD_EXEC_HS;
    case TINYNV_MSG_EVENT_GSP_RUN_CPU_SEQUENCER: return TINYNV_RPC610_EVENT_SEQUENCER_GONE;
    default:
      // Function numbers below the event range are RPC replies, matched by the caller; anything else is unknown.
      if (function < TINYNV_MSG_EVENT_GSP_INIT_DONE) return TINYNV_RPC610_EVENT_REPLY;
      return TINYNV_RPC610_EVENT_UNKNOWN;
  }
}
