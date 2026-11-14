// The 610 RPC / resource-management adapters under test, with no GPU.
//
// Drives the pure rpc610 codecs/encoders with synthetic fixtures, and the host-request executor (gsp610_host) against a
// register-level fake falcon. Covers what the card requires: changed request/response SIZES (every encoder places its
// fields at the 610 offsets), CONTINUATION (a payload too long for one record is split and re-joined), STATUS/EVENT
// handling (decode + classification), and UNSUPPORTED messages (the removed sequencer and unknown events are refused
// with a diagnostic). Nothing here runs on a GPU or proves the 610 firmware accepts these requests; the fake proves
// control flow and the register sequence only (hardware validation is plan §5+).
#include "rpc610.h"
#include "gsp610_host.h"
#include "internal.h"
#include "nv_regs.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;
#define CHECK(cond, ...) do { if (!(cond)) { printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); fails++; } } while (0)

static void wr32(uint8_t *p, uint32_t v) { p[0]=v; p[1]=v>>8; p[2]=v>>16; p[3]=v>>24; }
static uint32_t rd32(const uint8_t *p) { return p[0] | ((uint32_t)p[1]<<8) | ((uint32_t)p[2]<<16) | ((uint32_t)p[3]<<24); }
static void wr64(uint8_t *p, uint64_t v) { wr32(p, (uint32_t)v); wr32(p+4, (uint32_t)(v>>32)); }
static uint64_t rd64(const uint8_t *p) { return rd32(p) | ((uint64_t)rd32(p+4)<<32); }

// ---- the element codec -------------------------------------------------------------------------------------------
static void codec_roundtrip(void) {
  uint8_t rec[TINYNV_RPC610_ELEM_SIZE_MAX];
  uint8_t payload[100];
  for (int i = 0; i < 100; i++) payload[i] = (uint8_t)(i * 3 + 1);
  uint32_t count = 0; size_t rec_len = 0;
  CHECK(tinynv_rpc610_encode(rec, sizeof rec, TINYNV_MSG_FUNCTION_GSP_RM_ALLOC, payload, sizeof payload, 7,
                             TINYNV_RPC610_ELEM_SIZE_MIN, &count, &rec_len) == 0, "encode refused a 100 byte payload");
  CHECK(count == 1 && rec_len == 4096, "a 100 byte payload should be one 4096 byte element (got %u, %zu)", count, rec_len);
  // the element header is the MCTP/NVDM prefix, then the RPC header
  CHECK(rd32(rec) == TINYNV_RPC610_MCTP_HEADER, "mctp header not at 0");
  CHECK(rd32(rec + 4) == TINYNV_RPC610_NVDM_HEADER, "nvdm header not at 4");
  CHECK(rd32(rec + 12) == 7, "seq not at 12");
  CHECK(rd32(rec + 16 + 12) == TINYNV_MSG_FUNCTION_GSP_RM_ALLOC, "rpc function not at element+12");
  CHECK(rd32(rec + 16 + 8) == 100 + 32, "rpc length not payload+header");
  CHECK(memcmp(rec + 48, payload, 100) == 0, "payload not at element+rpc header");
  // the whole-record fold is zero (the checksum was set to make it so)
  CHECK(tinynv_rpc_checksum(rec, 16 + 132) == 0, "the encoded record does not fold to zero");

  tinynv_msg_header_t hdr; const uint8_t *pl = NULL; uint32_t pl_len = 0, dcount = 0;
  CHECK(tinynv_rpc610_decode(rec, rec_len, TINYNV_RPC610_ELEM_SIZE_MIN, 7, &hdr, &pl, &pl_len, &dcount) == 0,
        "decode refused a valid record: %s", tinynv_last_error());
  CHECK(hdr.function == TINYNV_MSG_FUNCTION_GSP_RM_ALLOC && pl_len == 100 && dcount == 1, "decode returned wrong meta");
  CHECK(pl && memcmp(pl, payload, 100) == 0, "decode payload mismatch");

  // The real GR-info control is 40 + 4032 bytes. It must be ONE RPC
  // occupying two queue elements, not a 4064-byte RPC plus a continuation.
  uint8_t gr[40 + 4032] = {0};
  CHECK(tinynv_rpc610_encode(rec, sizeof rec, 76, gr, sizeof gr, 8,
                             4096, &count, &rec_len) == 0,
        "GR-info control was rejected by the obsolete one-page RPC ceiling");
  CHECK(count == 2 && rec_len == 8192, "GR-info must occupy two elements in one record");
  CHECK(tinynv_rpc610_decode(rec, rec_len, 4096, 8, &hdr, &pl, &pl_len, &dcount) == 0 &&
        pl_len == sizeof gr && dcount == 2, "GR-info multi-element record did not round-trip");
}

static void codec_refusals(void) {
  uint8_t rec[TINYNV_RPC610_ELEM_SIZE_MAX];
  uint32_t count; size_t rec_len;
  CHECK(tinynv_rpc610_encode(rec, sizeof rec, 76, NULL, 0, 0, TINYNV_RPC610_ELEM_SIZE_MIN, &count, &rec_len) == 0,
        "encode refused an empty payload");
  // a payload past the one-record ceiling is refused (the caller must split it)
  CHECK(tinynv_rpc610_encode(rec, sizeof rec, 76, rec, TINYNV_RPC610_MAX_RPC_SIZE, 0, TINYNV_RPC610_ELEM_SIZE_MIN,
                             &count, &rec_len) != 0, "encode accepted a payload past the record ceiling");
  // a too-small / non-power-of-two element size is refused
  CHECK(tinynv_rpc610_encode(rec, sizeof rec, 76, NULL, 0, 0, 1000, &count, &rec_len) != 0, "encode accepted elem size 1000");
  // a buffer too small for the record is refused
  CHECK(tinynv_rpc610_encode(rec, 100, 76, NULL, 0, 0, TINYNV_RPC610_ELEM_SIZE_MIN, &count, &rec_len) != 0,
        "encode accepted a 100 byte buffer for a 4096 byte record");

  // decode: corrupt the checksum, the MCTP version, the vendor, the seq, the length
  uint8_t good[4096];
  CHECK(tinynv_rpc610_encode(good, sizeof good, 76, "abcd", 4, 3, TINYNV_RPC610_ELEM_SIZE_MIN, &count, &rec_len) == 0,
        "encode for decode-refusal fixtures failed");
  tinynv_msg_header_t hdr; const uint8_t *pl; uint32_t pl_len, dc;
  uint8_t bad[4096];
  memcpy(bad, good, sizeof good); bad[20] ^= 0xff;   // a payload byte -> checksum no longer folds to zero
  CHECK(tinynv_rpc610_decode(bad, rec_len, TINYNV_RPC610_ELEM_SIZE_MIN, 3, &hdr, &pl, &pl_len, &dc) != 0,
        "decode accepted a corrupted record");
  memcpy(bad, good, sizeof good); wr32(bad, 0x2);      // MCTP version 2
  CHECK(tinynv_rpc610_decode(bad, rec_len, TINYNV_RPC610_ELEM_SIZE_MIN, 3, &hdr, &pl, &pl_len, &dc) != 0,
        "decode accepted MCTP version 2");
  memcpy(bad, good, sizeof good); wr32(bad + 4, rd32(bad + 4) & ~(0xffffu << 8));   // vendor 0
  CHECK(tinynv_rpc610_decode(bad, rec_len, TINYNV_RPC610_ELEM_SIZE_MIN, 3, &hdr, &pl, &pl_len, &dc) != 0,
        "decode accepted a non-NVIDIA vendor");
  memcpy(bad, good, sizeof good); wr32(bad + 4, (rd32(bad + 4) & ~(0xffu << 24)) | (0x14u << 24));   // NVDM COT not RPC
  CHECK(tinynv_rpc610_decode(bad, rec_len, TINYNV_RPC610_ELEM_SIZE_MIN, 3, &hdr, &pl, &pl_len, &dc) != 0,
        "decode accepted a non-RM_RPC NVDM type");
  CHECK(tinynv_rpc610_decode(good, rec_len, TINYNV_RPC610_ELEM_SIZE_MIN, 4, &hdr, &pl, &pl_len, &dc) != 0,
        "decode accepted the wrong sequence number");
  memcpy(bad, good, sizeof good); wr32(bad + 16 + 8, 0);   // rpc length 0 (< the rpc header)
  CHECK(tinynv_rpc610_decode(bad, rec_len, TINYNV_RPC610_ELEM_SIZE_MIN, 3, &hdr, &pl, &pl_len, &dc) != 0,
        "decode accepted a zero rpc length");
  memcpy(bad, good, sizeof good); wr32(bad + 16 + 8, TINYNV_RPC610_ELEM_SIZE_MAX + 1);   // past the ceiling
  CHECK(tinynv_rpc610_decode(bad, rec_len, TINYNV_RPC610_ELEM_SIZE_MIN, 3, &hdr, &pl, &pl_len, &dc) != 0,
        "decode accepted a length past the element ceiling");
  CHECK(tinynv_rpc610_decode(good, 100, TINYNV_RPC610_ELEM_SIZE_MIN, 3, &hdr, &pl, &pl_len, &dc) != 0,
        "decode accepted a record shorter than the assembled elements");
}

// A payload larger than one record is split into a first record + continuation records, and re-joined. This mirrors
// rpc_send's continuation loop with the negotiated 16-page queue ceiling.
static void codec_continuation(void) {
  size_t total = TINYNV_RPC610_MAX_RPC_SIZE * 2 + 500;   // spans three records
  uint8_t *payload = malloc(total);
  for (size_t i = 0; i < total; i++) payload[i] = (uint8_t)(i * 11 + 5);
  uint8_t rec[TINYNV_RPC610_ELEM_SIZE_MAX];
  uint32_t count; size_t rec_len;
  size_t first_max = tinynv_rpc610_first_payload_max(), cont_max = tinynv_rpc610_cont_payload_max();
  CHECK(first_max == 65536 - 16 - 32 && cont_max == first_max, "the 610 split is not the GSP queue ceiling");

  uint8_t *joined = malloc(total);
  size_t done = 0; uint32_t seq = 100; int records = 0;
  size_t off = 0;
  while (off < total) {
    size_t chunk = (off == 0 ? first_max : cont_max);
    if (chunk > total - off) chunk = total - off;
    uint32_t func = (off == 0 ? TINYNV_MSG_FUNCTION_GSP_SET_SYSTEM_INFO : TINYNV_MSG_FUNCTION_CONTINUATION_RECORD);
    CHECK(tinynv_rpc610_encode(rec, sizeof rec, func, payload + off, chunk, seq, TINYNV_RPC610_ELEM_SIZE_MIN,
                               &count, &rec_len) == 0, "continuation encode failed at off %zu", off);
    tinynv_msg_header_t hdr; const uint8_t *pl; uint32_t pl_len, dc;
    CHECK(tinynv_rpc610_decode(rec, rec_len, TINYNV_RPC610_ELEM_SIZE_MIN, seq, &hdr, &pl, &pl_len, &dc) == 0,
          "continuation decode failed at off %zu: %s", off, tinynv_last_error());
    CHECK(hdr.function == func, "continuation record %d has the wrong function", records);
    CHECK(pl_len == chunk, "continuation record %d length %u != %zu", records, pl_len, chunk);
    memcpy(joined + done, pl, pl_len);
    done += pl_len; off += chunk; seq++; records++;
  }
  CHECK(records == 3, "a %zu byte payload should split into 3 records (got %d)", total, records);
  CHECK(done == total && memcmp(joined, payload, total) == 0, "the re-joined continuation payload differs");
  free(payload); free(joined);
}

// ---- the RM encoders: fields at the 610 offsets ------------------------------------------------------------------
static void rm_control_encoder(void) {
  tinynv_rpc610_rm_control_t c;
  tinynv_rpc610_rm_control_fill(&c, 0xc1000000, 0xcf000001, TINYNV_CTRL_CMD_PERF_BOOST, 8);
  CHECK(sizeof c == 40, "the 610 control header is not 40 bytes");
  uint8_t *p = (uint8_t *)&c;
  CHECK(rd32(p) == 0xc1000000 && rd32(p + 4) == 0xcf000001, "control client/object misplaced");
  CHECK(rd32(p + 8) == TINYNV_CTRL_CMD_PERF_BOOST && rd32(p + 16) == 8, "control cmd/paramsSize misplaced");
  CHECK(rd32(p + 20) == 0 && rd32(p + 24) == 0 && rd32(p + 28) == 0 && rd64(p + 32) == 0,
        "the new control flag fields are not zero");
}

static void sysinfo_encoder(void) {
  tinynv_gsp_system_info_t in;
  memset(&in, 0, sizeof in);
  in.gpuPhysAddr = 0x100000000ull; in.gpuPhysFbAddr = 0x200000000ull; in.gpuPhysInstAddr = 0x300000000ull;
  in.nvDomainBusDeviceFunc = 0x0100; in.maxUserVa = 0x7ffffffff000ull;
  in.pciConfigMirrorBase = 0x88000; in.pciConfigMirrorSize = 0x1000;
  in.PCIDeviceID = 0x268410de; in.PCISubDeviceID = 0x12345678; in.PCIRevisionID = 0xa1;
  in.bIsPassthru = 1;
  uint8_t out[TINYNV_RPC610_SYSINFO_SIZE];
  CHECK(tinynv_rpc610_system_info_encode(out, sizeof out, &in) == 0, "sysinfo encode refused");
  CHECK(rd64(out) == 0x100000000ull && rd64(out + 8) == 0x200000000ull && rd64(out + 16) == 0x300000000ull,
        "sysinfo phys addrs misplaced");
  CHECK(rd64(out + 32) == 0x0100 && rd64(out + 72) == 0x7ffffffff000ull, "sysinfo bdf/maxUserVa misplaced");
  CHECK(rd32(out + 80) == 0x88000 && rd32(out + 88) == 0x268410de && rd32(out + 96) == 0xa1,
        "sysinfo pci fields misplaced");
  CHECK(out[842] == 1, "bIsPassthru is not at the 610 offset 842");
  CHECK(out[840] == 0 && out[841] == 0, "virtualConfigBits is not zero");
  // the 570 offset 840 must NOT carry bIsPassthru any more
  CHECK(tinynv_rpc610_system_info_encode(out, 100, &in) != 0, "sysinfo encode accepted a too-small buffer");
}

static void channel_encoder(void) {
  tinynv_gpfifo_alloc_t in;
  memset(&in, 0, sizeof in);
  in.gpFifoOffset = 0xabcdef000ull; in.gpFifoEntries = 0x10000; in.flags = 0x200320;
  in.hVASpace = 0xcf000002; in.engineType = 1; in.cid = 3; in.internalFlags = 0x1a;
  in.hUserdMemory[0] = 0x1000; in.userdOffset[0] = 0x20000;
  in.instanceMem = (tinynv_memory_desc_t){.base = 0x4000, .size = 0x1000, .addressSpace = 2};
  in.tpcConfigID = 0x55;
  tinynv_rpc610_gpfifo_alloc_t out;
  tinynv_rpc610_channel_encode(&out, &in);
  CHECK(sizeof out == 376, "the 610 channel alloc is not 376 bytes");
  uint8_t *p = (uint8_t *)&out;
  CHECK(rd64(p + 8) == 0xabcdef000ull, "channel gpFifoOffset misplaced");
  CHECK(rd32(p + 28) == 0xcf000002, "channel hVASpace misplaced");
  CHECK(rd32(p + 32) == 0, "the new hHandleVASpace is not zero");
  CHECK(rd32(p + 36) == 0x1000, "channel hUserdMemory not at 36");
  CHECK(rd64(p + 72) == 0x20000, "channel userdOffset not at 72");
  CHECK(rd32(p + 136) == 1 && rd32(p + 140) == 3, "channel engineType/cid not at 136/140");
  CHECK(rd64(p + 152) == 0x4000, "channel instanceMem not at 152");
  CHECK(rd32(p + 252) == 0x1a, "channel internalFlags not at 252");
  CHECK(rd32(p + 368) == 0x55, "channel tpcConfigID not at 368");
}

static void vaspace_encoder(void) {
  tinynv_vaspace_alloc_t in;
  memset(&in, 0, sizeof in);
  in.vaBase = 0x1000; in.vaSize = 0x1fffffb000000ull;
  in.flags = TINYNV_VASPACE_FLAGS_ENABLE_PAGE_FAULTING | TINYNV_VASPACE_FLAGS_IS_EXTERNALLY_OWNED;
  tinynv_rpc610_vaspace_alloc_t out;
  tinynv_rpc610_vaspace_encode(&out, &in);
  CHECK(sizeof out == 56, "the 610 vaspace alloc is not 56 bytes");
  uint8_t *p = (uint8_t *)&out;
  CHECK(rd64(p + 8) == 0x1fffffb000000ull && rd64(p + 40) == 0x1000, "vaspace vaSize/vaBase misplaced");
  CHECK(rd32(p + 4) == in.flags, "vaspace flags misplaced");
  CHECK(rd32(p + 48) == 0, "the new pasid is not zero");
}

static void gr_reader(void) {
  tinynv_rpc610_gr_static_info_t info;
  memset(&info, 0, sizeof info);
  // place recognizable values at the indices the driver reads, in engine 0
  info.engineInfo[0].infoList[TINYNV_GR_INFO_NUM_GPCS].data = 7;
  info.engineInfo[0].infoList[TINYNV_GR_INFO_NUM_TPC_PER_GPC].data = 6;
  info.engineInfo[0].infoList[TINYNV_GR_INFO_NUM_SM_PER_TPC].data = 2;
  info.engineInfo[0].infoList[TINYNV_GR_INFO_MAX_WARPS_PER_SM].data = 48;
  info.engineInfo[0].infoList[TINYNV_GR_INFO_SM_VERSION].data = 0x89;
  info.engineInfo[1].infoList[TINYNV_GR_INFO_NUM_GPCS].data = 99;   // a different engine
  uint32_t v = 0;
  CHECK(sizeof info == 4032, "the 610 GR static info is not 4032 bytes");
  CHECK(tinynv_rpc610_gr_info_get(&info, sizeof info, 0, TINYNV_GR_INFO_NUM_GPCS, &v) == 0 && v == 7, "GR NUM_GPCS read");
  CHECK(tinynv_rpc610_gr_info_get(&info, sizeof info, 0, TINYNV_GR_INFO_SM_VERSION, &v) == 0 && v == 0x89, "GR SM_VERSION read");
  CHECK(tinynv_rpc610_gr_info_get(&info, sizeof info, 1, TINYNV_GR_INFO_NUM_GPCS, &v) == 0 && v == 99, "GR engine stride");
  CHECK(tinynv_rpc610_gr_info_get(&info, sizeof info, 0, 63, &v) != 0, "GR accepted index 63 (past the 63-entry array)");
  CHECK(tinynv_rpc610_gr_info_get(&info, sizeof info, 8, 0, &v) != 0, "GR accepted engine 8");
  CHECK(tinynv_rpc610_gr_info_get(&info, 100, 0, 0, &v) != 0, "GR accepted a short reply");
  // the 570 stride (464) would read the wrong engine; confirm the 610 stride is what separates engines
  CHECK(TINYNV_RPC610_GR_ENGINE_SIZE == 504, "the 610 per-engine GR stride is not 504");
}

static void schedule_rusd_encoders(void) {
  tinynv_rpc610_gpfifo_schedule_t s;
  tinynv_rpc610_schedule_fill(&s, 1, 0);
  CHECK(sizeof s == 3 && s.bEnable == 1 && s.bSkipSubmit == 0 && s.bSkipEnable == 0, "the 610 schedule is not 3 zero-tail bytes");

  tinynv_rpc610_rusd_init_t ri;
  tinynv_rpc610_rusd_init_fill(&ri, 0x12345000ull);
  CHECK(sizeof ri == 16 && ri.bInit == 1, "the 610 RUSD init is not 16 bytes with bInit=1");
  CHECK(rd64((uint8_t *)&ri + 8) == 0x12345000ull, "RUSD physAddr not at 8");
  tinynv_rpc610_rusd_poll_t rp;
  tinynv_rpc610_rusd_poll_fill(&rp, 0x3f, 100);
  CHECK(sizeof rp == 16 && rd64((uint8_t *)&rp) == 0x3f && rd32((uint8_t *)&rp + 8) == 100,
        "the 610 RUSD poll mask/interval misplaced");
}

// ---- the 610 RUSD shared-block reader ----------------------------------------------------------------------------
static void rusd_reader(void) {
  uint8_t *raw = calloc(1, TINYNV_RPC610_RUSD_SIZE);
  // a valid timestamp (even, below the seq-start range) for each field
  uint64_t ts = 42;
  wr64(raw + 0, ts); wr32(raw + 8, 256); wr32(raw + 12, 128);            // bar1
  wr64(raw + 16, ts); wr64(raw + 24, 24ull << 30); wr64(raw + 32, 20ull << 30);   // pma
  wr64(raw + 72, ts);                                                    // clk domains
  for (int d = 0; d < 4; d++) { wr32(raw + 80 + d * 8, 1000 + d); wr32(raw + 84 + d * 8, 900000 + d); }
  wr64(raw + 112, ts); wr32(raw + 120, 0x5);                             // throttle reasonMask
  wr64(raw + 1992, ts); wr32(raw + 2000, 55); wr32(raw + 2004, 33);      // perfDevUtil
  wr64(raw + 2040, ts); wr64(raw + 2048, 1);                             // memEcc legacy prefix
  wr64(raw + 2280, ts); wr32(raw + 2288, 0);                             // pstate
  wr64(raw + 2296, ts); wr32(raw + 2304, 450000); wr32(raw + 2308, 450000);   // powerLimit
  wr64(raw + 2312, ts); wr32(raw + 2320, 65 * 256);                      // temperature GPU (65 C)
  wr64(raw + 2328, ts); wr32(raw + 2336, 70 * 256);                      // temperature MEMORY
  wr64(raw + 2440, ts); wr32(raw + 2448, 300000);                        // avgPower
  wr64(raw + 2464, ts); wr32(raw + 2472, 310000);                        // instPower
  wr64(raw + 2504, ts); wr32(raw + 2512, 4);                             // pciBusData

  tinynv_rusd_t out;
  CHECK(tinynv_rpc610_rusd_read(raw, TINYNV_RPC610_RUSD_SIZE, &out) == 0, "rusd read refused");
  CHECK(out.bar1MemoryInfo.bar1Size == 256 && out.bar1MemoryInfo.ts == ts, "rusd bar1 not mapped");
  CHECK(out.pmaMemoryInfo.totalPmaMemory == (24ull << 30), "rusd pma not mapped");
  CHECK(out.clkPublicDomainInfos.targetClkMHz[0] == 1000 && out.clkPublicDomainInfos.targetClkMHz[3] == 1003,
        "rusd clk targetClkMHz not extracted at the 610 stride");
  CHECK(out.clkThrottleReason.reasonMask == 0x5, "rusd throttle reasonMask not read from clkThrottleInfo prefix");
  CHECK(out.perfDevUtil.gpuPercentBusy == 55, "rusd perfDevUtil not mapped");
  CHECK(out.memEcc.count[0].correctedVolatile == 1, "rusd memEcc legacy prefix not mapped");
  CHECK(out.powerLimitGpu.enforcedmW == 450000, "rusd powerLimit not mapped");
  CHECK(out.temperatures[0].temperature == 65 * 256 && out.temperatures[1].temperature == 70 * 256,
        "rusd temperatures not mapped");
  CHECK(out.avgPowerUsage.averageGpuPower == 300000 && out.instPowerUsage.instGpuPower == 310000, "rusd power not mapped");
  CHECK(out.pciBusData.data[0] == 4, "rusd pciBusData not mapped");

  // a field whose timestamp is invalid (zero) is left zero, not trusted
  wr64(raw + 2312, 0);   // GPU temperature now "not reported"
  CHECK(tinynv_rpc610_rusd_read(raw, TINYNV_RPC610_RUSD_SIZE, &out) == 0, "rusd read refused (invalid ts)");
  CHECK(out.temperatures[0].temperature == 0 && out.temperatures[0].ts == 0, "rusd trusted an invalid timestamp");
  CHECK(out.temperatures[1].temperature == 70 * 256, "rusd lost the valid MEMORY temperature");
  CHECK(tinynv_rpc610_rusd_read(raw, 100, &out) != 0, "rusd read accepted a short block");
  free(raw);
}

// ---- the host-request payload parsers ----------------------------------------------------------------------------
static void host_parsers(void) {
  tinynv_rpc610_load_exec_hs_t hs;
  memset(&hs, 0, sizeof hs);
  hs.imemPhysAddr = 0x100000; hs.dmemPhysAddr = 0x200000;
  hs.ucodeImemVA = 0; hs.ucodeImemPA = 0; hs.ucodeImemSize = 4096;
  hs.ucodeDmemVA = 0xffffffff; hs.ucodeDmemPA = 0x1000; hs.ucodeDmemSize = 256;
  hs.hsSigDmemAddr = 0x100; hs.engineIdMask = 1; hs.ucodeId = 3;
  tinynv_rpc610_load_exec_hs_t out;
  CHECK(tinynv_rpc610_parse_load_exec_hs((uint8_t *)&hs, sizeof hs, &out) == 0, "hs parse refused a valid payload");
  CHECK(out.ucodeImemSize == 4096 && out.ucodeId == 3, "hs parse lost fields");
  hs.ucodeImemSize = 100;   // not 256-aligned
  CHECK(tinynv_rpc610_parse_load_exec_hs((uint8_t *)&hs, sizeof hs, &out) != 0, "hs parse accepted a misaligned imem size");
  hs.ucodeImemSize = 4096; hs.engineIdMask = 0;
  CHECK(tinynv_rpc610_parse_load_exec_hs((uint8_t *)&hs, sizeof hs, &out) != 0, "hs parse accepted a zero engine mask");
  CHECK(tinynv_rpc610_parse_load_exec_hs((uint8_t *)&hs, 10, &out) != 0, "hs parse accepted a short payload");

  tinynv_rpc610_load_exec_bl_t bl;
  memset(&bl, 0, sizeof bl);
  bl.dmemDescSize = sizeof(tinynv_rpc610_bl_dmem_desc_t);
  bl.dmemDesc.ctxDma = 0; bl.addrSpace = 1; bl.cpuCacheAttrib = 0;
  bl.dmemDesc.nonSecureCodeSize = 256; bl.dmemDesc.dataSize = 256;
  tinynv_rpc610_load_exec_bl_t bout;
  CHECK(tinynv_rpc610_parse_load_exec_bl((uint8_t *)&bl, sizeof bl, &bout) == 0, "bl parse refused a valid payload");
  bl.dmemDescSize = 999;
  CHECK(tinynv_rpc610_parse_load_exec_bl((uint8_t *)&bl, sizeof bl, &bout) != 0, "bl parse accepted a wrong desc size");
  bl.dmemDescSize = sizeof(tinynv_rpc610_bl_dmem_desc_t); bl.dmemDesc.ctxDma = 8;
  CHECK(tinynv_rpc610_parse_load_exec_bl((uint8_t *)&bl, sizeof bl, &bout) != 0, "bl parse accepted ctxDma 8");
  bl.dmemDesc.ctxDma = 0; bl.addrSpace = 5;
  CHECK(tinynv_rpc610_parse_load_exec_bl((uint8_t *)&bl, sizeof bl, &bout) != 0, "bl parse accepted addrSpace 5");
  bl.addrSpace = 1; bl.dmemDesc.nonSecureCodeSize = 100;
  CHECK(tinynv_rpc610_parse_load_exec_bl((uint8_t *)&bl, sizeof bl, &bout) != 0, "bl parse accepted a misaligned code size");
}

static void event_classification(void) {
  CHECK(tinynv_rpc610_event_kind(TINYNV_MSG_EVENT_GSP_INIT_DONE) == TINYNV_RPC610_EVENT_INIT_DONE, "init-done kind");
  CHECK(tinynv_rpc610_event_kind(TINYNV_MSG_EVENT_OS_ERROR_LOG) == TINYNV_RPC610_EVENT_OS_ERROR_LOG, "error-log kind");
  CHECK(tinynv_rpc610_event_kind(TINYNV_MSG_EVENT_MMU_FAULT_QUEUED) == TINYNV_RPC610_EVENT_MMU_FAULT, "mmu-fault kind");
  CHECK(tinynv_rpc610_event_kind(TINYNV_MSG_EVENT_UCODE_LIBOS_PRINT) == TINYNV_RPC610_EVENT_LIBOS_PRINT, "libos-print kind");
  CHECK(tinynv_rpc610_event_kind(TINYNV_MSG_EVENT_GSP_LOAD_EXEC_GENERIC_BOOTLOADER) == TINYNV_RPC610_EVENT_LOAD_EXEC_BL, "load-exec-bl kind");
  CHECK(tinynv_rpc610_event_kind(TINYNV_MSG_EVENT_GSP_LOAD_EXEC_HS_BINARY) == TINYNV_RPC610_EVENT_LOAD_EXEC_HS, "load-exec-hs kind");
  CHECK(tinynv_rpc610_event_kind(TINYNV_MSG_EVENT_GSP_RUN_CPU_SEQUENCER) == TINYNV_RPC610_EVENT_SEQUENCER_GONE, "sequencer-gone kind");
  CHECK(tinynv_rpc610_event_kind(TINYNV_MSG_FUNCTION_GSP_RM_ALLOC) == TINYNV_RPC610_EVENT_REPLY, "rpc-reply kind");
  CHECK(tinynv_rpc610_event_kind(0x1020) == TINYNV_RPC610_EVENT_UNKNOWN, "unknown event kind");
}

// ---- the host-request executor against a register-level fake falcon ----------------------------------------------
// The fake models just enough of the GSP/SEC2 falcons to prove the executor's control flow and register sequence: it
// answers the status polls (suspend, scrub-done, DMA idle, halt, handoff, riscv-active) and records every write. It does
// NOT model signature verification or real DMA data movement - that is hardware, plan §5+.
#define FAKE_GSP 0x110000u
#define FAKE_SEC2 0x840000u
#define FAKE_BAR0 0x1000000u
typedef struct {
  uint32_t regs[FAKE_BAR0 / 4];
  struct { uint32_t off, val; } log[8192];
  int nlog;
  int suspend_allowed;   // when 0, the GSP mailbox never reports suspend (the wait must time out)
  int sentinel_stays;    // when 1, the launched binary leaves BINARY_NOT_STARTED in mailbox0
  int launch_phase;      // CPU start happened; mailbox0 now reports the binary result instead of suspend
} fake_falcon_t;
static fake_falcon_t FF;

static uint32_t ff_rd32(tinynv_mmio_t *m, uint64_t off) {
  (void)m;
  if (off >= FAKE_BAR0) return ~0u;
  uint32_t o = (uint32_t)off;
  // Before the request, GSP mailbox0 carries the suspend bit. After the falcon starts it carries the binary result, or
  // the launch sentinel when the test models a binary that never ran.
  if (o == FAKE_GSP + 0x40) {
    if (FF.launch_phase) return FF.sentinel_stays ? 0xFEu : 0;
    return FF.suspend_allowed ? 0x80000000u : 0;
  }
  // DMA command: idle, not full
  if (o == FAKE_GSP + 0x118 || o == FAKE_SEC2 + 0x118) return 2;
  return FF.regs[o / 4];
}
static void ff_wr32(tinynv_mmio_t *m, uint64_t off, uint32_t v) {
  (void)m;
  if (off >= FAKE_BAR0) return;
  uint32_t o = (uint32_t)off;
  if (FF.nlog < 8192) { FF.log[FF.nlog].off = o; FF.log[FF.nlog].val = v; FF.nlog++; }
  FF.regs[o / 4] = v;
  uint32_t base = (o & 0xfff000) == 0x840000 ? FAKE_SEC2 : FAKE_GSP;
  uint32_t rel = o - base;
  // starting a falcon halts it (the fake's ucode "runs" instantly) and, for SEC2, signals the handoff
  if ((rel == 0x100 || rel == 0x130) && (v & 2)) {
    if (base == FAKE_GSP) FF.launch_phase = 1;
    FF.regs[(base + 0x100) / 4] |= 0x10;   // HALTED
    if (base == FAKE_SEC2) {
      FF.regs[0x1180f8 / 4] |= (1u << 26);   // SCRATCH_14 handoff
      FF.regs[(FAKE_SEC2 + 0x40) / 4] = 0;   // SEC2 mailbox0 = success
    }
  }
}
static int ff_logged(uint32_t off, uint32_t val) {
  for (int i = 0; i < FF.nlog; i++) if (FF.log[i].off == off && FF.log[i].val == val) return 1;
  return 0;
}
static void fake_falcon_init(tinynv_gpu_t *g) {
  memset(&FF, 0, sizeof FF);
  FF.suspend_allowed = 1;
  FF.regs[(FAKE_GSP + 0x108) / 4] = 0x100;   // HWCFG IMEM_SIZE = 256 blocks
  FF.regs[(FAKE_GSP + 0x1388) / 4] = 0x80;   // RISCV_CPUCTL ACTIVE_STAT
  memset(g, 0, sizeof *g);
  g->dev.mmio = (tinynv_mmio_t){.ptr = NULL, .rd32 = ff_rd32, .wr32 = ff_wr32};
  g->dev.chip_id = 0x192;
  g->flcn.falcon = FAKE_GSP;
  g->flcn.sec2 = FAKE_SEC2;
  g->gsp.libos_args_sysmem = 0xfeed000ull;
  g->gsp.bl_desc.appVersion = 0x610;
}

static void exec_hs_fake(void) {
  tinynv_gpu_t g;
  fake_falcon_init(&g);
  tinynv_rpc610_load_exec_hs_t hs;
  memset(&hs, 0, sizeof hs);
  hs.imemPhysAddr = 0x100000; hs.dmemPhysAddr = 0x200000;
  hs.ucodeImemVA = 0x100; hs.ucodeImemPA = 0; hs.ucodeImemSize = 4096;
  hs.ucodeDmemVA = 0xffffffff; hs.ucodeDmemPA = 0x1000; hs.ucodeDmemSize = 256;
  hs.hsSigDmemAddr = 0x80; hs.engineIdMask = 1; hs.ucodeId = 3;
  CHECK(tinynv_gsp610_exec_hs(&g, &hs) == 0, "exec_hs failed against the fake: %s", tinynv_last_error());
  // the register sequence: TRANSCFG local-FB, PKC, mailbox NOT_STARTED, bootvec, then the resume's FALCON_OS
  CHECK(ff_logged(FAKE_GSP + 0x600, 0x4), "exec_hs did not program TRANSCFG(0) physical/local-FB");
  CHECK(ff_logged(FAKE_GSP + 0x1210, 0x80), "exec_hs did not set BROM_PARAADDR to hsSigDmemAddr");
  CHECK(ff_logged(FAKE_GSP + 0x119c, 1), "exec_hs did not set BROM_ENGIDMASK");
  CHECK(ff_logged(FAKE_GSP + 0x40, 0xFE), "exec_hs did not write MAILBOX0 = BINARY_NOT_STARTED");
  CHECK(ff_logged(FAKE_GSP + 0x104, 0x100), "exec_hs did not set BOOTVEC to ucodeImemVA");
  CHECK(ff_logged(FAKE_GSP + 0x80, 0x610), "exec_hs resume did not program FALCON_OS");
  CHECK(ff_logged(0x110040, 0xfeed000u), "exec_hs resume did not hand back the libos args low word");
  // the IMEM DMA was commanded secure (SEC=1, IMEM=1): cmd bits 4 and 2..3
  int saw_imem_dma = 0;
  for (int i = 0; i < FF.nlog; i++)
    if (FF.log[i].off == FAKE_GSP + 0x118 && (FF.log[i].val & 0x10) && (FF.log[i].val & 0x4)) saw_imem_dma = 1;
  CHECK(saw_imem_dma, "exec_hs did not command a secure IMEM DMA");

  // with suspend withheld, the executor must time out and say so (not hang or run the falcon)
  fake_falcon_init(&g);
  FF.suspend_allowed = 0;
  CHECK(tinynv_gsp610_exec_hs(&g, &hs) != 0, "exec_hs ran without gsp-rm suspending");
  CHECK(strstr(tinynv_last_error(), "suspend") != NULL, "exec_hs timeout did not name the suspend wait");

  // A halted binary that left the 0xFE sentinel must fail before core resume overwrites mailbox0.
  fake_falcon_init(&g);
  FF.sentinel_stays = 1;
  CHECK(tinynv_gsp610_exec_hs(&g, &hs) != 0, "exec_hs accepted a binary that never started");
  CHECK(strstr(tinynv_last_error(), "BINARY_NOT_STARTED") != NULL,
        "exec_hs sentinel failure did not preserve a useful diagnostic: %s", tinynv_last_error());
  CHECK(!ff_logged(FAKE_GSP + 0x40, 0xfeed000u), "exec_hs resumed and overwrote mailbox0 after a sentinel halt");
  CHECK(!ff_logged(FAKE_GSP + 0x80, 0x610), "exec_hs completed core resume after a sentinel halt");
}

static void exec_bl_fake(void) {
  tinynv_gpu_t g;
  fake_falcon_init(&g);
  uint8_t bl_image[1024];
  for (int i = 0; i < 1024; i++) bl_image[i] = (uint8_t)i;
  tinynv_gsp610_bl_ucode_t bl = {.image = bl_image, .image_len = sizeof bl_image, .start_tag = 0xfd, .code_size = 0x200};
  tinynv_rpc610_load_exec_bl_t p;
  memset(&p, 0, sizeof p);
  p.dmemDescSize = sizeof(tinynv_rpc610_bl_dmem_desc_t);
  p.dmemDesc.ctxDma = 0; p.addrSpace = 1; p.cpuCacheAttrib = 0;   // coherent sysmem
  p.dmemDesc.nonSecureCodeSize = 256; p.dmemDesc.dataSize = 256;
  CHECK(tinynv_gsp610_exec_bl(&g, &p, &bl) == 0, "exec_bl failed against the fake: %s", tinynv_last_error());
  // the aperture is coherent sysmem (target 1, physical): TRANSCFG = (1<<0)|(1<<2) = 5
  CHECK(ff_logged(FAKE_GSP + 0x600, 0x5), "exec_bl did not program the coherent-sysmem aperture");
  // BOOTVEC = start_tag << 8
  CHECK(ff_logged(FAKE_GSP + 0x104, 0xfd00), "exec_bl did not set BOOTVEC to start_tag<<8");
  // the DMEM descriptor was written through DMEMC/DMEMD (port 0), and the BL through IMEMC/IMEMD
  CHECK(ff_logged(FAKE_GSP + 0x1c0, 0x1000000), "exec_bl did not set DMEMC(0) auto-increment at offset 0");
  int saw_imemc = 0, saw_imemt = 0;
  for (int i = 0; i < FF.nlog; i++) {
    if (FF.log[i].off == FAKE_GSP + 0x180) saw_imemc = 1;
    if (FF.log[i].off == FAKE_GSP + 0x188) saw_imemt = 1;
  }
  CHECK(saw_imemc && saw_imemt, "exec_bl did not copy the BL through IMEMC/IMEMT");
  CHECK(ff_logged(FAKE_GSP + 0x40, 0xFE), "exec_bl did not write MAILBOX0 = BINARY_NOT_STARTED");

  // no ucode staged is a clear refusal, not a crash
  fake_falcon_init(&g);
  CHECK(tinynv_gsp610_exec_bl(&g, &p, NULL) != 0, "exec_bl ran with no generic BL ucode staged");
  // a BL larger than IMEM is refused
  fake_falcon_init(&g);
  tinynv_gsp610_bl_ucode_t too_big = {.image = bl_image, .image_len = sizeof bl_image, .start_tag = 0xfd, .code_size = 0x200};
  const uint32_t pre_error_transcfg = 0x123u;
  FF.regs[(FAKE_GSP + 0x600) / 4] = pre_error_transcfg;
  FF.regs[(FAKE_GSP + 0x108) / 4] = 1;   // only 1 IMEM block
  CHECK(tinynv_gsp610_exec_bl(&g, &p, &too_big) != 0, "exec_bl accepted a BL larger than IMEM");
  CHECK(FF.regs[(FAKE_GSP + 0x600) / 4] == pre_error_transcfg,
        "exec_bl did not restore TRANSCFG after an IMEM-capacity error (got %#x, want %#x)",
        FF.regs[(FAKE_GSP + 0x600) / 4], pre_error_transcfg);

  // The parser rejects this first; the executor also refuses an unvalidated direct call before forming MMIO.
  fake_falcon_init(&g);
  p.dmemDesc.ctxDma = 8;
  CHECK(tinynv_gsp610_exec_bl(&g, &p, &bl) != 0, "exec_bl accepted an out-of-range ctxDma");
  CHECK(strstr(tinynv_last_error(), "ctxDma 8") != NULL,
        "exec_bl ctxDma refusal did not name the bad index: %s", tinynv_last_error());
  CHECK(FF.nlog == 0, "exec_bl touched the device before refusing ctxDma 8");
  p.dmemDesc.ctxDma = 0;

  // A generic BL that never starts must not reach core resume, and its temporary aperture must be restored first.
  fake_falcon_init(&g);
  const uint32_t saved_transcfg = 0x123u;
  FF.regs[(FAKE_GSP + 0x600) / 4] = saved_transcfg;
  FF.sentinel_stays = 1;
  CHECK(tinynv_gsp610_exec_bl(&g, &p, &bl) != 0, "exec_bl accepted a binary that never started");
  CHECK(strstr(tinynv_last_error(), "BINARY_NOT_STARTED") != NULL,
        "exec_bl sentinel failure did not preserve a useful diagnostic: %s", tinynv_last_error());
  CHECK(!ff_logged(FAKE_GSP + 0x40, 0xfeed000u), "exec_bl resumed and overwrote mailbox0 after a sentinel halt");
  CHECK(!ff_logged(FAKE_GSP + 0x80, 0x610), "exec_bl completed core resume after a sentinel halt");
  CHECK(FF.regs[(FAKE_GSP + 0x600) / 4] == saved_transcfg,
        "exec_bl did not restore TRANSCFG after a sentinel halt (got %#x, want %#x)",
        FF.regs[(FAKE_GSP + 0x600) / 4], saved_transcfg);
}

static void dispatcher_fake(void) {
  tinynv_gpu_t g;
  fake_falcon_init(&g);
  // a non-host-request event is passed back (return 1), not executed
  CHECK(tinynv_gsp610_host_request(&g, TINYNV_MSG_EVENT_GSP_INIT_DONE, NULL, 0, NULL) == 1,
        "the dispatcher did not pass init-done back");
  CHECK(tinynv_gsp610_host_request(&g, TINYNV_MSG_FUNCTION_GSP_RM_ALLOC, NULL, 0, NULL) == 1,
        "the dispatcher did not pass an rpc reply back");
  // a one-way libos print is consumed (return 1), not refused as unknown - it aborted a live boot before.
  CHECK(tinynv_gsp610_host_request(&g, TINYNV_MSG_EVENT_UCODE_LIBOS_PRINT, NULL, 0, NULL) == 1,
        "the dispatcher did not consume a benign libos print");
  // the removed sequencer is refused with a diagnostic naming the version mismatch
  CHECK(tinynv_gsp610_host_request(&g, TINYNV_MSG_EVENT_GSP_RUN_CPU_SEQUENCER, NULL, 0, NULL) == -1,
        "the dispatcher did not refuse the removed sequencer");
  CHECK(strstr(tinynv_last_error(), "610 removed") != NULL, "the sequencer refusal did not name the removal");
  // an unknown event is refused
  CHECK(tinynv_gsp610_host_request(&g, 0x1020, NULL, 0, NULL) == -1, "the dispatcher did not refuse an unknown event");
  // a malformed HS payload is refused before any falcon access
  fake_falcon_init(&g);
  uint8_t short_msg[4] = {0};
  CHECK(tinynv_gsp610_host_request(&g, TINYNV_MSG_EVENT_GSP_LOAD_EXEC_HS_BINARY, short_msg, sizeof short_msg, NULL) == -1,
        "the dispatcher accepted a short HS payload");
  CHECK(FF.nlog == 0, "the dispatcher touched the falcon before validating the payload");
  // a valid HS payload is executed end to end through the dispatcher
  fake_falcon_init(&g);
  tinynv_rpc610_load_exec_hs_t hs;
  memset(&hs, 0, sizeof hs);
  hs.imemPhysAddr = 0x100000; hs.dmemPhysAddr = 0x200000;
  hs.ucodeImemSize = 4096; hs.ucodeDmemVA = 0xffffffff; hs.ucodeDmemPA = 0x1000; hs.ucodeDmemSize = 256;
  hs.hsSigDmemAddr = 0x80; hs.engineIdMask = 1; hs.ucodeId = 3;
  CHECK(tinynv_gsp610_host_request(&g, TINYNV_MSG_EVENT_GSP_LOAD_EXEC_HS_BINARY, (uint8_t *)&hs, sizeof hs, NULL) == 0,
        "the dispatcher did not execute a valid HS request: %s", tinynv_last_error());
}

int main(void) {
  codec_roundtrip();
  codec_refusals();
  codec_continuation();
  rm_control_encoder();
  sysinfo_encoder();
  channel_encoder();
  vaspace_encoder();
  gr_reader();
  schedule_rusd_encoders();
  rusd_reader();
  host_parsers();
  event_classification();
  exec_hs_fake();
  exec_bl_fake();
  dispatcher_fake();
  printf(fails ? "rpc610: %d checks FAILED\n"
               : "rpc610: all checks passed (codec/continuation/encoders/rusd/host-requests; fake falcon, no GPU)\n",
         fails);
  return fails ? 1 : 0;
}
