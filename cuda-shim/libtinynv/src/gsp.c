// Placing GSP-RM's firmware in memory, and describing where it is.
//
// This mirrors tinygrad's NV_GSP.init_sw allocation by allocation, because the addresses end up inside the structures the
// boot firmware reads: the chain of trust message names the metadata block, the metadata block names the image, and the
// image is described by a page table the GPU walks. Allocate in a different order and every one of those addresses moves,
// which is why the replay checks each allocation's size against the recording and hands back the recorded address.
//
// Nothing here touches the GPU. It is all host memory and firmware files, which is what makes it testable offline.
#include "gpu.h"
#include "internal.h"
#include "nv_regs.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MB (1ull << 20)
#define PAGE 0x1000ull

// The firmware this driver is pinned to. The hashes are the version: a firmware image is code the GPU's secure boot
// executes, so the file's name is not evidence of anything and the content is checked before it is used.
static const char *GSP_SHA = "a8c3ebeed280323aedb51c061f321e73379cce7a9ae643a33dd03915df027f7f";
static const char *BL_SHA = "d40b48e431d1707dc77af3605db358ed7a32ebfc2830eb74de2eddb4d3025071";

static uint64_t round_up(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }

// chip names are ascii and the firmware section names are their lowercase form; not locale's business
static char lower_ascii(char c) { return c >= 'A' && c <= 'Z' ? (char)(c - 'A' + 'a') : c; }

// NVIDIA names the memory regions it hands its operating system with short ascii strings packed into a 64 bit field,
// most significant byte first: "LOGINIT", "RMARGS" and so on.
static uint64_t id8_of(const char *s) {
  uint64_t v = 0;
  for (; *s; s++) v = (v << 8) | (uint8_t)*s;
  return v;
}

// GSP-RM can ask the driver to touch registers on its behalf: a short program of writes, read-modify-writes, polls and
// delays, sent as a stream of words. This exists because a few sequences have to run from the host side of the bus.
//
// The recorded 5090 boot never sends one -- there is not a single register write between the chain of trust and the
// windows being retargeted -- so this path is written from the oracle and has not been exercised by the replay. It is
// here rather than left as a failure because a board that does send one would otherwise stop dead.
static void run_cpu_seq(tinynv_gpu_t *g, const uint8_t *msg, size_t len) {
  // say so, every time. this path was never taken in the recording, so on real hardware a request is the driver leaving
  // the ground the oracle covers, and that should be visible in the log rather than inferred afterwards.
  g->gsp.cpu_seq_requests++;
  fprintf(stderr, "tinynv: gsp-rm asked the driver to run a register sequence (%zu bytes, request %u). "
                  "the recorded boot never did this, so this path is unproven.\n", len, g->gsp.cpu_seq_requests);
  const size_t hdr = 32; // rpc_run_cpu_sequencer_v17_00: the command count is the last word of it
  if (len < hdr) return;
  uint32_t cmd_index;
  memcpy(&cmd_index, msg + hdr - 4, 4);
  const uint32_t *w = (const uint32_t *)(const void *)(msg + hdr);
  size_t have = (len - hdr) / 4, i = 0;
  if (cmd_index < have) have = cmd_index;

#define NEXT(out) do { if (i >= have) return; memcpy(&(out), &w[i++], 4); } while (0)
  while (i < have) {
    uint32_t op, addr, val, mask, us;
    NEXT(op);
    switch (op) {
      case 0x0: NEXT(addr); NEXT(val); tinynv_wr32(&g->dev, addr, val); break;
      case 0x1:
        NEXT(addr); NEXT(val); NEXT(mask);
        tinynv_wr32(&g->dev, addr, (tinynv_rd32(&g->dev, addr) & ~mask) | (val & mask));
        break;
      case 0x2: {
        uint32_t ignored;
        NEXT(addr); NEXT(mask); NEXT(val); NEXT(ignored); NEXT(ignored);
        tinynv_wait_reg(&g->dev, addr, mask, val, 10000, "a register poll gsp-rm asked for");
        break;
      }
      case 0x3: {
        NEXT(us);
        struct timespec ts = {.tv_sec = us / 1000000, .tv_nsec = (long)(us % 1000000) * 1000L};
        nanosleep(&ts, NULL);
        break;
      }
      default: tinynv_fail("gsp-rm asked for register operation %#x, which this driver does not know", op); return;
    }
  }
#undef NEXT
}

// --- the shared ring ------------------------------------------------------------------------------------------------
//
// Both directions live in one allocation the GPU reaches through its own page table. Each direction has a header saying
// how big its slots are and how many there are, followed by the slots themselves. The writer advances a write pointer in
// its own header; the reader advances a read pointer that, by this queue's convention, is stored in the *other*
// direction's header. So neither side ever writes to the other's header, and a single shared mapping is enough.

// Wait for a queue's header to say its slots have appeared, then take our copy of it from memory rather than assuming
// what we wrote. For the command queue that is immediate: we wrote it. For the status queue it is the whole of GSP-RM's
// start-up, which on this card is several seconds of spinning.
static int rpcq_attach(tinynv_gpu_t *g, tinynv_rpcq_t *q, uint64_t base, const char *what) {
  tinynv_gsp_t *gsp = &g->gsp;
  q->base = base;
  q->seq = 0;
  double deadline = tinynv_now_s() + 60.0;
  uint32_t entry_off;
  do {
    entry_off = nv_rd32(&gsp->queues.view, base + offsetof(tinynv_msgq_tx_header_t, entryOff));
  } while (entry_off != PAGE && tinynv_now_s() < deadline);
  if (entry_off != PAGE) return tinynv_fail("the %s queue never initialised: its entry offset read %#x, not %#llx",
                                            what, entry_off, (unsigned long long)PAGE);
  nv_rd_block(&gsp->queues.view, base, &q->tx, sizeof(q->tx));
  q->entries = base + q->tx.entryOff;
  if (!q->tx.msgSize || !q->tx.msgCount) return tinynv_fail("the %s queue describes %u slots of %u bytes", what,
                                                            q->tx.msgCount, q->tx.msgSize);
  return 0;
}

// The element header's own checksum: exclusive-or of the record as 64 bit words, folded in half. Zero length tails are
// padded, which matters because a record is only ever a whole number of slots on the wire but is checksummed at its
// natural length.
static uint32_t rpc_checksum(const uint8_t *data, size_t n) {
  uint64_t sum = 0;
  size_t whole = n / 8 * 8;
  for (size_t i = 0; i < whole; i += 8) {
    uint64_t w;
    memcpy(&w, data + i, 8);
    sum ^= w;
  }
  if (n > whole) { // the odd tail, zero extended, which is what padding to eight bytes amounts to
    uint64_t w = 0;
    memcpy(&w, data + whole, n - whole);
    sum ^= w;
  }
  return (uint32_t)(sum >> 32) ^ (uint32_t)sum;
}

// Put one record in the ring and ring the doorbell. A record that runs off the end of the ring wraps to the start, which
// is why the copy is in two parts.
static int rpc_send_record(tinynv_gpu_t *g, uint32_t func, const void *payload, size_t len) {
  tinynv_gsp_t *gsp = &g->gsp;
  tinynv_rpcq_t *q = &gsp->cmd_q;
  size_t hdrs = sizeof(tinynv_msg_element_t) + sizeof(tinynv_msg_header_t);
  uint32_t elem_count = (uint32_t)((len + hdrs + q->tx.msgSize - 1) / q->tx.msgSize);
  size_t total = (size_t)elem_count * q->tx.msgSize;

  uint8_t *rec = calloc(1, total);
  if (!rec) return tinynv_fail("out of memory for a %zu byte rpc record", total);

  tinynv_msg_header_t *mh = (tinynv_msg_header_t *)(rec + sizeof(tinynv_msg_element_t));
  mh->header_version = TINYNV_MSG_HEADER_VERSION;
  mh->signature = TINYNV_MSG_SIGNATURE_VALID;
  mh->rpc_result = mh->rpc_result_private = TINYNV_MSG_RESULT_PENDING;
  mh->function = func;
  mh->length = (uint32_t)(len + sizeof(tinynv_msg_header_t));
  memcpy(rec + hdrs, payload, len);

  tinynv_msg_element_t *eh = (tinynv_msg_element_t *)rec;
  eh->elemCount = elem_count;
  eh->seqNum = q->seq;
  // the checksum covers the record at its natural length, with the checksum field still zero
  eh->checkSum = rpc_checksum(rec, sizeof(*eh) + mh->length);

  uint64_t ring = q->entries, ring_len = (uint64_t)q->tx.msgSize * q->tx.msgCount;
  uint32_t wp = nv_rd32(&gsp->queues.view, q->base + offsetof(tinynv_msgq_tx_header_t, writePtr));
  uint64_t off = (uint64_t)wp * q->tx.msgSize;
  size_t first = total < ring_len - off ? total : (size_t)(ring_len - off);
  nv_wr_block(&gsp->queues.view, ring + off, rec, first);
  if (first < total) nv_wr_block(&gsp->queues.view, ring, rec + first, total - first);
  nv_wr32(&gsp->queues.view, q->base + offsetof(tinynv_msgq_tx_header_t, writePtr), (wp + elem_count) % q->tx.msgCount);
  __sync_synchronize(); // the doorbell must not be seen before the record it announces

  q->seq++;
  tinynv_wr32(&g->dev, NV_PGSP_QUEUE_HEAD(0), 0);
  free(rec);
  return 0;
}

// A payload too long for one record is continued in further records. The first carries the real function number and the
// rest a continuation marker, which is how the far side knows to join them back together.
static int rpc_send(tinynv_gpu_t *g, uint32_t func, const void *payload, size_t len) {
  tinynv_rpcq_t *q = &g->gsp.cmd_q;
  size_t max = (size_t)q->tx.msgSize * 16 - sizeof(tinynv_msg_element_t) - sizeof(tinynv_msg_header_t);
  const uint8_t *p = payload;
  if (rpc_send_record(g, func, p, len < max ? len : max)) return -1;
  for (size_t off = max; off < len; off += max)
    if (rpc_send_record(g, TINYNV_MSG_FUNCTION_CONTINUATION_RECORD, p + off, len - off < max ? len - off : max)) return -1;
  return 0;
}

// Take whatever GSP-RM has put in the status queue, and say whether the message we are waiting for was among it.
//
// Two kinds of message are acted on rather than merely delivered: a request to run a register sequence on GSP-RM's
// behalf, and a log line, which is the only way to see what the firmware thinks went wrong.
static int rpc_drain(tinynv_gpu_t *g, uint32_t want, int *seen, uint8_t **reply, uint32_t *reply_len) {
  tinynv_gsp_t *gsp = &g->gsp;
  tinynv_rpcq_t *q = &gsp->stat_q;
  __sync_synchronize();
  for (;;) {
    uint32_t rp = nv_rd32(&gsp->queues.view, q->rx_off);
    uint32_t wp = nv_rd32(&gsp->queues.view, q->base + offsetof(tinynv_msgq_tx_header_t, writePtr));
    if (rp == wp) return 0;

    // read the position again rather than reuse the one the comparison saw. it costs nothing, it is what the oracle
    // does, and it does not assume that this side is the only writer of a field that lives in shared memory.
    uint64_t slot = q->entries + (uint64_t)nv_rd32(&gsp->queues.view, q->rx_off) * q->tx.msgSize;
    tinynv_msg_header_t mh;
    nv_rd_block(&gsp->queues.view, slot + sizeof(tinynv_msg_element_t), &mh, sizeof(mh));

    uint8_t *msg = NULL;
    if (mh.length) {
      if (!(msg = malloc(mh.length))) return tinynv_fail("out of memory for a %u byte rpc reply", mh.length);
      nv_rd_block(&gsp->queues.view, slot + sizeof(tinynv_msg_element_t) + sizeof(tinynv_msg_header_t), msg, mh.length);
    }

    if (mh.function == TINYNV_MSG_EVENT_GSP_RUN_CPU_SEQUENCER) run_cpu_seq(g, msg, mh.length);
    else if (mh.function == TINYNV_MSG_EVENT_OS_ERROR_LOG && mh.length > 12)
      fprintf(stderr, "tinynv: gsp log: %.*s\n", (int)(mh.length - 12), (const char *)msg + 12);
    if (mh.function == TINYNV_MSG_EVENT_OS_ERROR_LOG || mh.function == TINYNV_MSG_EVENT_MMU_FAULT_QUEUED) {
      gsp->err_state = 1;
      // Keep what it said. An MMU fault arrives as a notification - the addresses live in a fault buffer this driver
      // does not register yet - so the payload is often short and sometimes empty, and saying so is still better than
      // the driver claiming something was logged when nothing was.
      gsp->err_fn = mh.function;
      gsp->err_len = mh.length;
      uint32_t keep = mh.length < sizeof(gsp->err_head) ? mh.length : (uint32_t)sizeof(gsp->err_head);
      if (msg && keep) memcpy(gsp->err_head, msg, keep);
      fprintf(stderr, "tinynv: gsp-rm reported %s (event %u, %u byte payload)\n",
              mh.function == TINYNV_MSG_EVENT_MMU_FAULT_QUEUED ? "an mmu fault - some engine touched an address that is "
                                                                 "not mapped, so look at what a descriptor points at"
                                                               : "an error log",
              mh.function, mh.length);
      for (uint32_t i = 0; i < keep; i += 16) {
        fprintf(stderr, "tinynv:   %04x:", i);
        for (uint32_t j = i; j < i + 16 && j < keep; j++) fprintf(stderr, " %02x", gsp->err_head[j]);
        fprintf(stderr, "\n");
      }
    }

    // advance past however many slots this message occupied
    uint32_t slots = (mh.length + q->tx.msgSize - 1) / q->tx.msgSize;
    nv_wr32(&gsp->queues.view, q->rx_off, (nv_rd32(&gsp->queues.view, q->rx_off) + slots) % q->tx.msgCount);
    __sync_synchronize();

    uint32_t result = mh.rpc_result, func = mh.function;
    if (result) { free(msg); return tinynv_fail("gsp-rm refused rpc %u with result %#x", func, result); }
    // stop at the message that was asked for, rather than draining whatever else has arrived behind it. the read pointer
    // is already past it, so the rest of the queue is still there to be read next time.
    if (func == want) {
      if (seen) *seen = 1;
      if (reply) { *reply = msg; *reply_len = mh.length; } else free(msg);
      return 0;
    }
    free(msg);
  }
}

static void fault_report(tinynv_gpu_t *g);

// Read whatever GSP-RM has said and has not been collected. Nothing drains the status queue once the boot is over, so
// an engine that faults reports it into a queue nobody is reading: the driver sees work that never completes and has no
// idea why, which is exactly the position this was written from. Returns 1 if the firmware has reported an error.
int tinynv_gsp_poll(tinynv_gpu_t *g) {
  int seen = 0;
  if (!g->gsp.stat_q.base) return 0;               // the queue does not exist before the firmware is up
  if (rpc_drain(g, 0xffffffffu, &seen, NULL, NULL)) return 1; // a refusal is itself news, and it has been recorded
  // The event says a fault happened. Where it happened has to be asked for, and this is the only moment anyone is
  // looking - so ask now, while the answer still describes the fault that is being reported rather than a later one.
  if (g->gsp.err_state) fault_report(g);
  return g->gsp.err_state;
}

static int rpc_wait(tinynv_gpu_t *g, uint32_t want, double timeout_s, const char *what, uint8_t **reply, uint32_t *reply_len) {
  double deadline = tinynv_now_s() + timeout_s;
  int seen = 0;
  if (reply) { *reply = NULL; *reply_len = 0; }
  do {
    if (rpc_drain(g, want, &seen, reply, reply_len)) return -1;
    if (seen) return 0;
  } while (tinynv_now_s() < deadline);
  // Whatever arrives now belongs to a request nobody is waiting for any more.
  g->gsp.rpc_desync = 1;
  return tinynv_fail("gsp-rm never sent %s (message %u) in %.0f s", what, want, timeout_s);
}

// --- the object tree ------------------------------------------------------------------------------------------------
//
// A call is a request record followed by its parameters, and a reply that carries the same parameters back with whatever
// GSP-RM filled in. Both directions use the ring built above; the only new idea is that the caller now waits for its own
// answer rather than for an event.
static uint32_t next_handle(tinynv_gsp_t *gsp) { return gsp->next_handle++; }

static int rpc_call(tinynv_gpu_t *g, uint32_t func, const void *head, size_t head_len, void *params, size_t params_len,
                    const char *what) {
  // A previous call gave up on its reply. Clear the queue before asking anything else, because a reply carries the
  // function it answers and nothing that identifies the request: a late reply to one control is indistinguishable from
  // the answer to the next one, and rpc_wait would return it, copying the wrong parameters back to a caller with no
  // way to tell. The fault report makes two controls in a row on a path that is already failing, which is exactly
  // where this would land and exactly where a confidently wrong answer is worst.
  //
  // Only after a timeout, so the ordinary path issues no operation it did not issue before and the recorded boot is
  // reproduced exactly as it was.
  if (g->gsp.rpc_desync) {
    int seen = 0;
    fprintf(stderr, "tinynv: a previous rpc gave up waiting; emptying the reply queue so a late answer cannot be "
                    "mistaken for this one\n");
    if (rpc_drain(g, 0xffffffffu, &seen, NULL, NULL)) return -1;
    g->gsp.rpc_desync = 0;
  }
  size_t total = head_len + params_len;
  uint8_t *buf = malloc(total ? total : 1);
  if (!buf) return tinynv_fail("out of memory for a %zu byte %s", total, what);
  memcpy(buf, head, head_len);
  if (params_len) memcpy(buf + head_len, params, params_len);
  int rc = rpc_send(g, func, buf, total);
  free(buf);
  if (rc) return -1;

  uint8_t *reply = NULL;
  uint32_t reply_len = 0;
  if (rpc_wait(g, func, 10.0, what, &reply, &reply_len)) return -1;
  // gsp-rm answers with the request echoed back and the parameters filled in, so take them from where they were sent
  if (params && params_len && reply_len >= head_len + params_len) memcpy(params, reply + head_len, params_len);
  free(reply);
  return 0;
}

// Create one object under another and return its handle. The root client is the exception: it names itself.
// `handle` of zero means this driver picks one from its counter, which is what every internal caller wants. A caller
// forwarding for a guest passes the handle the guest chose: GSP-RM accepts that, proven on hardware, and it is the
// property the whole forwarding design rests on - (client, object) then means the same thing on both sides of the
// boundary and neither needs a translation table.
static int rm_alloc_as(tinynv_gpu_t *g, uint32_t client, uint32_t parent, uint32_t handle, uint32_t cls, void *params,
                       size_t params_len, uint32_t *out) {
  tinynv_gsp_t *gsp = &g->gsp;
  tinynv_rpc_rm_alloc_t head;
  memset(&head, 0, sizeof(head));
  head.hClient = client;
  head.hParent = parent;
  head.hObject = handle ? handle : next_handle(gsp);
  head.hClass = cls;
  head.paramsSize = (uint32_t)params_len;
  // Asked before the message goes out, because the card's refusal for this is a bare status with no reason - and that
  // reads as GSP-RM rejecting a guest-chosen handle, which is a different and much more alarming conclusion than the
  // true one. It cost Session C a card window to establish that distinction once already. A handle still recorded as
  // live is the caller's fault and this says so in those words.
  {
    uint32_t want = head.hObject;
    for (int i = 0; i < gsp->nobjs; i++)
      if (gsp->objs[i].client == client && gsp->objs[i].handle == want)
        return tinynv_fail("object %#x under client %#x is already allocated and has not been freed; the card would "
                           "refuse this with a bare status and no reason", want, client);
  }
  if (rpc_call(g, TINYNV_MSG_FUNCTION_GSP_RM_ALLOC, &head, sizeof(head), params, params_len, "object allocation")) return -1;
  uint32_t made = cls == TINYNV_CLASS_ROOT ? client : head.hObject;
  // Recorded here because this is the only place an object comes into existence, and the record has to be complete
  // before any free is ever sent - a guest that dies mid-kernel sends nothing at all, so teardown needs it whether or
  // not the frees were well behaved. Full is refused rather than silently forgotten: a record that quietly stops being
  // complete is worse than one that stops.
  if (gsp->nobjs == gsp->objs_cap) {
    int want = gsp->objs_cap ? gsp->objs_cap * 2 : TINYNV_RM_OBJECTS_FIRST;
    tinynv_rm_obj_t *grown = realloc(gsp->objs, (size_t)want * sizeof(*grown));
    if (!grown) return tinynv_fail("out of memory for the record of %d rm objects", want);
    gsp->objs = grown;
    gsp->objs_cap = want;
  }
  if (gsp->nobjs + 1 >= TINYNV_RM_OBJECTS_LOUD && !gsp->objs_warned) {
    gsp->objs_warned = 1;
    fprintf(stderr, "tinynv: %d rm objects are allocated and none have been freed lately, which is what a leak looks "
                    "like from here. Not refusing; saying so once.\n", gsp->nobjs + 1);
  }
  // The allocation's size, for the one class a guest allocates and then asks us to map.
  //
  // Only when the caller actually handed over the whole struct. The wire form declares paramsSize 0 for this class and
  // the real size comes from the class rather than the call (rmapiGetClassAllocParamSize) - so a forwarder may or may
  // not have substituted it by the time it reaches here, and reading 128 bytes out of a buffer somebody described as
  // empty is a read overrun in this process. Unknown stays 0, and an object of unknown size cannot be mapped.
  uint64_t size = 0;
  uint32_t alloc_flags = 0;
  if (cls == TINYNV_CLASS_MEMORY_LOCAL_USER && params && params_len >= sizeof(tinynv_nv0040_alloc_t)) {
    const tinynv_nv0040_alloc_t *p = params;
    size = p->size;
    alloc_flags = p->flags;
  }
  gsp->objs[gsp->nobjs++] = (tinynv_rm_obj_t){
      .client = client, .parent = parent, .handle = made, .cls = cls, .alloc_flags = alloc_flags, .size = size};
  if (out) *out = made;
  return 0;
}

// Where an object sits in the record, or -1. Named by client and handle together: handles are only unique within a
// client, which is the property that lets a guest choose its own and is also why a free has to carry both.
// Walking up rather than testing one step, because the tree is deeper than two: a compute object sits under a channel
// under a group under a device under a client. Bounded by n, since a cycle in the record would otherwise spin here and
// the record is ours to trust only as far as it has been checked.
int tinynv_rm_is_descendant(const tinynv_rm_obj_t *objs, int n, uint32_t client, uint32_t handle, uint32_t ancestor) {
  if (handle == ancestor) return 0;
  uint32_t at = handle;
  for (int steps = 0; steps <= n; steps++) {
    int found = -1;
    for (int i = 0; i < n; i++)
      if (objs[i].client == client && objs[i].handle == at) { found = i; break; }
    if (found < 0) return 0;
    uint32_t parent = objs[found].parent;
    if (parent == ancestor) return 1;
    if (!parent || parent == at) return 0;   // reached a root, or a self-parent, which a client is
    at = parent;
  }
  return 0;
}

tinynv_rm_free_check_t tinynv_rm_free_check(const tinynv_rm_obj_t *objs, int n, uint32_t client, uint32_t handle,
                                            int *at) {
  int found = -1;
  for (int i = 0; i < n; i++)
    if (objs[i].client == client && objs[i].handle == handle) { found = i; break; }
  if (found < 0) return TINYNV_RM_FREE_UNKNOWN;
  // A handle is only unique within a client, so the parent test is scoped to the client too. Without that, two clients
  // that both chose the same handle - which they are entitled to do, and a guest choosing its own makes it likely -
  // would each appear to be the other's parent.
  for (int i = 0; i < n; i++)
    if (objs[i].client == client && objs[i].parent == handle) return TINYNV_RM_FREE_HAS_CHILDREN;
  if (at) *at = found;
  return TINYNV_RM_FREE_OK;
}

// Whether there is anything on the other end. The public entry points in tinynv.c boot the card on first use, so
// nothing reaching RM through THAT surface can arrive early. These three are not on that surface - they take a gpu
// rather than a device and never pass through it - so a caller that has not booted gets here with no message queue,
// and writes a request into memory the firmware has never been told about. That is a silent nothing rather than an
// error, which is the worst available outcome, so it is named here.
// Is GSP-RM able to take an RPC? This tested `queues.view.size != 0`, which is true from the moment the queue memory is
// ALLOCATED - before the header is written, and before rpcq_attach spins (up to sixty seconds on this card) waiting for
// the firmware to publish its entry offset. So it answered "up" for the whole of the firmware's own start-up, and an
// RPC issued in that window would have gone into a queue with no slots described.
//
// Nothing in this driver could reach the window, because our own calls all come after boot. Session C's forwarder can:
// it takes calls from a guest, which arrive whenever the guest sends them. C asked whether their `cmd_q.entries` test
// was the same check as this one and declined to collapse the two without an answer, which was right - they are not
// the same, and theirs is the one that is correct. cmd_q.entries is set only after the firmware has initialised its
// side and we have read the description back from it, which is exactly the condition "can take an RPC" means.
static int rm_up(tinynv_gpu_t *g) { return g->gsp.cmd_q.entries != 0; }

int tinynv_gsp_rm_free(tinynv_gpu_t *g, uint32_t client, uint32_t handle) {
  if (!rm_up(g)) return tinynv_fail("rm free asked for before gsp-rm is up");
  tinynv_gsp_t *gsp = &g->gsp;
  // Freeing an object with live children CASCADES rather than being refused, because that is what RM itself does and
  // refusing it broke the ordinary teardown path.
  //
  // I had this wrong, and Session C caught it on hardware: libcuda frees the objects it cares about, leaves the rest,
  // and frees the client to sweep up. My validator refused the subdevice and then the device, and the card was only
  // left clean because C's eviction sweep ran afterwards - a safety net written for a guest dying, doing the work of
  // the normal path.
  //
  // Read rather than assumed, in the vendored resource server: clientUpdatePendingFreeList recursively adds a target's
  // children AND its dependants to the pending free list, children first. That is true of ANY resource and not only a
  // client, which is broader than C's hypothesis and broader than what I would have implemented from it.
  for (int pass = 0; pass < gsp->nobjs + 1; pass++) {
    int child = -1;
    for (int j = gsp->nobjs - 1; j >= 0; j--)
      if (gsp->objs[j].client == client && tinynv_rm_is_descendant(gsp->objs, gsp->nobjs, client, gsp->objs[j].handle,
                                                                  handle)) { child = j; break; }
    if (child < 0) break;
    // Deepest-last in the record, so the newest descendant goes first and a parent is never freed before its own
    // children. One at a time, re-scanning, because the array moves under us as things are removed.
    if (tinynv_gsp_rm_free(g, client, gsp->objs[child].handle)) return -1;
  }

  int i = 0;
  switch (tinynv_rm_free_check(gsp->objs, gsp->nobjs, client, handle, &i)) {
    case TINYNV_RM_FREE_UNKNOWN:
      // RM answers OK for a resource that does not exist; this refuses instead, and deliberately. We cannot forward a
      // free for an object we never recorded because we do not know its parent, so this is a limitation stated as a
      // refusal rather than a policy. It has already earned its place: it named a bug in C's forwarder in a minute.
      return tinynv_fail("free of object %#x under client %#x, which this driver never allocated", handle, client);
    case TINYNV_RM_FREE_HAS_CHILDREN:
      return tinynv_fail("object %#x under client %#x still has children after the cascade, which is a bug in the "
                         "record rather than in the caller", handle, client);
    case TINYNV_RM_FREE_OK: break;
  }

  tinynv_rpc_rm_free_t p;
  memset(&p, 0, sizeof(p));
  p.hRoot = client;
  p.hObjectParent = gsp->objs[i].parent;
  p.hObjectOld = handle;
  if (rpc_call(g, TINYNV_MSG_FUNCTION_FREE, &p, sizeof(p), NULL, 0, "object free")) return -1;

  // Forgotten only after the card has agreed, and the order of the rest is preserved because teardown reads it
  // backwards and that order is what makes children come first.
  memmove(&gsp->objs[i], &gsp->objs[i + 1], (size_t)(gsp->nobjs - i - 1) * sizeof(gsp->objs[0]));
  gsp->nobjs--;
  return 0;
}

int tinynv_rm_range_safe(uint32_t lo, uint32_t hi) {
  if (hi <= lo) return 1;   // empty sweeps nothing
  if (lo <= TINYNV_RM_PRIV_ROOT && TINYNV_RM_PRIV_ROOT < hi) return 0;
  if (lo <= TINYNV_RM_USER_ROOT && TINYNV_RM_USER_ROOT < hi) return 0;
  return 1;
}

int tinynv_gsp_rm_free_clients(tinynv_gpu_t *g, uint32_t lo, uint32_t hi) {
  tinynv_gsp_t *gsp = &g->gsp;
  // The driver's own clients are never in range, whatever was asked for. Session C's range stops one below the
  // privileged client on purpose, and an off-by-one there would tear down the client that owns the channels this
  // process is submitting through - a mistake that would present as the card dying rather than as a bad argument. So it
  // is refused here rather than left to every caller to get right.
  if (!tinynv_rm_range_safe(lo, hi))
    return tinynv_fail("the range %#x..%#x contains one of this driver's own clients (%#x, %#x)", lo, hi,
                       TINYNV_RM_PRIV_ROOT, TINYNV_RM_USER_ROOT);

  int refused = 0;
  // Backwards, which is creation order reversed, which is children before parents. Skipping an out-of-range object
  // does not break that: a child is created after its parent and carries the same client, so the two are either both
  // in range or both out.
  for (int i = gsp->nobjs - 1; i >= 0; i--) {
    if (gsp->objs[i].client < lo || gsp->objs[i].client >= hi) continue;
    if (i >= gsp->nobjs) continue;   // the array shrank under us as things were freed
    if (tinynv_gsp_rm_free(g, gsp->objs[i].client, gsp->objs[i].handle)) refused++;
  }
  return refused;
}

int tinynv_gsp_rm_free_all(tinynv_gpu_t *g) {
  tinynv_gsp_t *gsp = &g->gsp;
  int failed = 0;
  // Backwards, which is reverse creation order, which is children before parents. Keep going after a failure: one
  // object refusing to go is not a reason to leave the rest allocated, and the count at the end is the honest report.
  while (gsp->nobjs) {
    int n = gsp->nobjs;
    if (tinynv_gsp_rm_free(g, gsp->objs[n - 1].client, gsp->objs[n - 1].handle)) {
      failed++;
      gsp->nobjs = n - 1;   // step over it rather than spinning on it
    }
  }
  return failed ? tinynv_fail("%d of the objects torn down were refused by the card", failed) : 0;
}

static int rm_alloc(tinynv_gpu_t *g, uint32_t parent, uint32_t cls, void *params, size_t params_len, uint32_t *out) {
  return rm_alloc_as(g, TINYNV_RM_PRIV_ROOT, parent, 0, cls, params, params_len, out);
}

static int rm_control_as(tinynv_gpu_t *g, uint32_t client, uint32_t object, uint32_t cmd, void *params, size_t params_len) {
  tinynv_rpc_rm_control_t head;
  memset(&head, 0, sizeof(head));
  head.hClient = client;
  head.hObject = object;
  head.cmd = cmd;
  head.paramsSize = (uint32_t)params_len;
  return rpc_call(g, TINYNV_MSG_FUNCTION_GSP_RM_CONTROL, &head, sizeof(head), params, params_len, "object control call");
}

static int rm_control(tinynv_gpu_t *g, uint32_t object, uint32_t cmd, void *params, size_t params_len) {
  return rm_control_as(g, TINYNV_RM_PRIV_ROOT, object, cmd, params, params_len);
}

// THE RM SURFACE, and the reason it is exactly these three.
//
// Everything that creates an object goes through one function, because the record that makes an object freeable is
// written there and nowhere else. A caller that allocates around this - a forwarder building objects for a guest, say -
// creates something no free can ever find, and the failure is silent until teardown or until the guest's second run.
// So this is not a convenience wrapper over rm_alloc_as; it is the only door.
int tinynv_gsp_rm_alloc(tinynv_gpu_t *g, uint32_t client, uint32_t parent, uint32_t handle, uint32_t cls, void *params,
                        size_t params_len, uint32_t *out) {
  if (!rm_up(g)) return tinynv_fail("rm alloc asked for before gsp-rm is up");
  return rm_alloc_as(g, client, parent, handle, cls, params, params_len, out);
}

// Controls create nothing, so there is nothing to record and this is a straight pass-through.
int tinynv_gsp_rm_control(tinynv_gpu_t *g, uint32_t client, uint32_t object, uint32_t cmd, void *params,
                          size_t params_len) {
  if (!rm_up(g)) return tinynv_fail("rm control asked for before gsp-rm is up");
  return rm_control_as(g, client, object, cmd, params, params_len);
}


// A field is valid when its timestamp is neither zero (not reported here) nor an odd sequence in the high range (a
// write in progress). Straight from RUSD_SEQ_DATA_VALID in the vendored header.
static int rusd_seq_ok(uint64_t seq) {
  return (seq < TINYNV_RUSD_SEQ_START && seq != TINYNV_RUSD_TS_INVALID) ||
         (seq >= TINYNV_RUSD_SEQ_START && (seq & 1) == 0);
}

int tinynv_gsp_free_heap(tinynv_gpu_t *g, uint64_t *out) {
  tinynv_gsp_free_heap_t p;
  memset(&p, 0, sizeof(p));
  if (rm_control(g, g->gsp.subdevice, TINYNV_CTRL_GET_GSP_RM_FREE_HEAP, &p, sizeof(p))) return -1;
  *out = p.freeHeapSize;
  return 0;
}

int tinynv_gsp_sensors_init(tinynv_gpu_t *g, uint64_t poll_mask, uint32_t freq_ms) {
  tinynv_gsp_t *gsp = &g->gsp;
  if (gsp->rusd_armed) return 0;
  // Host memory, because that is where the firmware puts it: ADDR_SYSMEM in the vendor's own allocation of this
  // buffer. It is 464 bytes and it costs one of the 128 host mappings the broker will ever hand out, which is a small
  // finite resource that only goes up - worth stating rather than waving through.
  if (tinynv_mm_alloc_buffer(&g->mm, sizeof(tinynv_rusd_t), 1, 1, 0, 0, 1, &gsp->rusd_mem)) return -1;

  // The class allocation is attempted but NOT required, and that distinction is the whole of why this still exists.
  //
  // On this card GSP-RM refuses RM_USER_SHARED_DATA with NV_ERR_NOT_SUPPORTED (0x56) - the same answer Session C gets
  // for it from the guest side. But that class is the object a CLIENT allocates to be handed a mapping of the block.
  // We are not a client of RM here; we are the CPU-side resource manager, and the two controls below are the plumbing
  // that side uses to tell the firmware where the buffer is and what to keep fresh. So the refusal may be about the
  // client-facing object rather than about the mechanism, and the controls are worth trying on their own.
  //
  // If that reasoning is wrong the controls will refuse too, and we will have learned it for the cost of two messages.
  tinynv_rusd_alloc_t ap = {.polledDataMask = poll_mask};
  if (rm_alloc(g, gsp->subdevice, TINYNV_CLASS_USER_SHARED_DATA, &ap, sizeof(ap), &gsp->rusd_obj))
    gsp->rusd_obj = 0;   // carry on without it

  tinynv_rusd_init_t ip = {.physAddr = gsp->rusd_mem.ranges[0].paddr};
  if (rm_control(g, gsp->subdevice, TINYNV_CTRL_INIT_USER_SHARED_DATA, &ip, sizeof(ip))) goto undo;

  tinynv_rusd_poll_t pp = {.polledDataMask = poll_mask, .pollFrequencyMs = freq_ms};
  if (rm_control(g, gsp->subdevice, TINYNV_CTRL_USER_SHARED_DATA_POLL, &pp, sizeof(pp))) goto undo;

  gsp->rusd_armed = 1;
  return 0;

undo:
  // Leave nothing half-armed: a buffer the firmware was told about but a poll it never got would read as a card that
  // reports nothing, which is indistinguishable from a card that cannot.
  gsp->rusd_obj = 0;
  gsp->rusd_armed = 0;
  if (gsp->rusd_mem.size) tinynv_vmap_free(&g->mm, &gsp->rusd_mem);
  return -1;
}

int tinynv_gsp_sensors_read(tinynv_gpu_t *g, tinynv_rusd_t *out) {
  tinynv_gsp_t *gsp = &g->gsp;
  memset(out, 0, sizeof(*out));
  if (!gsp->rusd_armed || !gsp->rusd_mem.dma.va) return tinynv_fail("sensors were not asked for on this device");
  const volatile tinynv_rusd_t *src = (const volatile tinynv_rusd_t *)gsp->rusd_mem.dma.va;

  // Field by field, because each carries its own timestamp and the firmware updates them independently. Read the
  // timestamp, copy, read it again: unchanged means nothing moved underneath. Ten attempts, then leave that field zero
  // and let the caller see a timestamp of zero, which reads the same as "not reported here" - both mean do not trust
  // this number, which is the only thing a caller needs to know.
#define FIELD(f) do {                                                                  \
    for (int a = 0; a < 10; a++) {                                                     \
      uint64_t seq = src->f.ts;                                                        \
      __atomic_thread_fence(__ATOMIC_ACQUIRE);                                         \
      if (!rusd_seq_ok(seq)) continue;                                                 \
      memcpy(&out->f, (const void *)&src->f, sizeof(out->f));                          \
      __atomic_thread_fence(__ATOMIC_ACQUIRE);                                         \
      if (seq == src->f.ts) break;                                                     \
      memset(&out->f, 0, sizeof(out->f));                                              \
    }                                                                                  \
  } while (0)
  FIELD(bar1MemoryInfo); FIELD(pmaMemoryInfo); FIELD(clkPublicDomainInfos); FIELD(clkThrottleReason);
  FIELD(perfDevUtil); FIELD(memEcc); FIELD(perfCurrentPstate); FIELD(powerLimitGpu);
  FIELD(temperatures[0]); FIELD(temperatures[1]); FIELD(avgPowerUsage); FIELD(instPowerUsage); FIELD(pciBusData);
#undef FIELD
  return 0;
}


// --- what faulted --------------------------------------------------------------------------------------------------
//
// An MMU fault arrives on the status queue as a bare notification: some engine touched an address that is not mapped,
// and nothing about which address or what it was doing. That is the least useful true sentence a driver can print, and
// it was printed through several hardware sessions before this existed.
//
// The addresses are not in that message and they are not in a buffer this driver forgot to register. They are held for
// the channel and asked for afterwards, of the debugger object that is already allocated on the compute channel
// because the recorded boot allocates one. The oracle does exactly this and registers no fault buffer either, which is
// what makes this change cost nothing at boot: nothing here runs until something has already gone wrong, so the boot
// the replay pins is untouched.
//
// Two questions, in this order. The first reads the per-SM error registers and, with them, a valid bit for a fault on
// the target channel - so it also answers "was this an MMU fault at all", which the notification does not. Only if
// that bit is set is the second worth asking, and it returns up to four addresses with the access that caused each.
static const char *fault_type_name(uint32_t t) {
  switch (t) {
    case TINYNV_PFAULT_TYPE_PDE:                  return "no page directory entry";
    case TINYNV_PFAULT_TYPE_PDE_SIZE:             return "page directory entry of the wrong size";
    case TINYNV_PFAULT_TYPE_PTE:                  return "no page table entry - nothing is mapped there";
    case TINYNV_PFAULT_TYPE_VA_LIMIT_VIOLATION:   return "address beyond the end of the address space";
    case TINYNV_PFAULT_TYPE_UNBOUND_INST_BLOCK:   return "unbound instance block - the channel was never set up";
    case TINYNV_PFAULT_TYPE_PRIV_VIOLATION:       return "privilege violation";
    case TINYNV_PFAULT_TYPE_RO_VIOLATION:         return "wrote to a read-only mapping";
    case TINYNV_PFAULT_TYPE_WO_VIOLATION:         return "read from a write-only mapping";
    case TINYNV_PFAULT_TYPE_PITCH_MASK_VIOLATION: return "pitch mask violation";
    case TINYNV_PFAULT_TYPE_WORK_CREATION:        return "work creation fault";
    case TINYNV_PFAULT_TYPE_UNSUPPORTED_APERTURE: return "unsupported aperture - the memory is not of a kind this "
                                                         "engine can reach";
    case TINYNV_PFAULT_TYPE_CC_VIOLATION:         return "confidential computing violation";
    case TINYNV_PFAULT_TYPE_UNSUPPORTED_KIND:     return "unsupported kind";
    case TINYNV_PFAULT_TYPE_REGION_VIOLATION:     return "region violation";
    case TINYNV_PFAULT_TYPE_POISONED:             return "poisoned memory";
    case TINYNV_PFAULT_TYPE_ATOMIC_VIOLATION:     return "atomic on memory that does not support it";
    default:                                      return "a fault kind this driver does not name";
  }
}

// The physical variants matter more than they look: a fault on a physical address is not in this driver's page tables
// at all, so the whole of the mapping code is the wrong place to start looking.
static const char *fault_access_name(uint32_t a) {
  switch (a) {
    case TINYNV_PFAULT_ACCESS_READ:          return "read";
    case TINYNV_PFAULT_ACCESS_WRITE:         return "write";
    case TINYNV_PFAULT_ACCESS_ATOMIC:        return "atomic";
    case TINYNV_PFAULT_ACCESS_PREFETCH:      return "prefetch, so the fault is past the end of something rather than "
                                                    "at a pointer that was wrong";
    case TINYNV_PFAULT_ACCESS_ATOMIC_WEAK:   return "weak atomic";
    case TINYNV_PFAULT_ACCESS_PHYS_READ:     return "read of a PHYSICAL address";
    case TINYNV_PFAULT_ACCESS_PHYS_WRITE:    return "write to a PHYSICAL address";
    case TINYNV_PFAULT_ACCESS_PHYS_ATOMIC:   return "atomic on a PHYSICAL address";
    case TINYNV_PFAULT_ACCESS_PHYS_PREFETCH: return "prefetch of a PHYSICAL address";
    default:                                 return "an access kind this driver does not name";
  }
}

static void fault_report(tinynv_gpu_t *g) {
  tinynv_gsp_t *gsp = &g->gsp;
  if (gsp->fault_reported) return;
  // Before the boot has built them there is nothing to ask and nothing to ask it of. This is not an error: the same
  // poll runs during start-up, where a report from the firmware is about the boot rather than about a launch.
  if (!gsp->user_debugger || !gsp->compute_q.channel) return;
  gsp->fault_reported = 1;

  // Four and a half kilobytes of registers, which is a lot for a stack in a path that is already unwinding.
  tinynv_sm_error_states_t *st = calloc(1, sizeof(*st));
  if (!st) return;
  st->hTargetChannel = gsp->compute_q.channel;
  st->numSMsToRead = TINYNV_DEBUG_MAX_SMS_PER_CALL;
  if (rm_control_as(g, TINYNV_RM_USER_ROOT, gsp->user_debugger, TINYNV_CTRL_CMD_DEBUG_READ_ALL_SM_ERROR_STATES, st,
                    sizeof(*st))) {
    // Worth saying rather than swallowing: "the debugger did not answer" is itself a fact about how far gone the
    // firmware is, and it is different from "there was nothing to report".
    fprintf(stderr, "tinynv: asked the debugger what faulted and it did not answer (%s)\n", tinynv_last_error());
    free(st);
    return;
  }

  if (st->mmuFault.valid) {
    tinynv_mmu_fault_info_t info;
    memset(&info, 0, sizeof(info));
    if (rm_control_as(g, TINYNV_RM_USER_ROOT, gsp->user_debugger, TINYNV_CTRL_CMD_DEBUG_READ_MMU_FAULT_INFO, &info,
                      sizeof(info))) {
      fprintf(stderr, "tinynv: an mmu fault is flagged on the compute channel but its addresses did not come back "
                      "(%s); the fault word is %#x\n", tinynv_last_error(), st->mmuFault.faultInfo);
      free(st);
      return;
    }
    uint32_t n = info.count < TINYNV_MMU_FAULT_ENTRIES ? info.count : TINYNV_MMU_FAULT_ENTRIES;
    if (n) {
      // The first entry is the one the caller is told about: the list is newest-first for this fault, and a message
      // that names four addresses names three that are probably older than the failure being reported.
      gsp->fault_have = 1;
      gsp->fault_va = info.mmuFaultInfoList[0].faultAddress;
      gsp->fault_type = info.mmuFaultInfoList[0].faultType;
      gsp->fault_access = info.mmuFaultInfoList[0].accessType;
    }
    fprintf(stderr, "tinynv: mmu fault on the compute channel, %u address%s recorded:\n", n, n == 1 ? "" : "es");
    for (uint32_t i = 0; i < n; i++)
      fprintf(stderr, "tinynv:   %#018llx  %s (%s)\n", (unsigned long long)info.mmuFaultInfoList[i].faultAddress,
              fault_type_name(info.mmuFaultInfoList[i].faultType),
              fault_access_name(info.mmuFaultInfoList[i].accessType));
    // The interface says so, and it is the difference between "it faulted four times" and "it faulted once and this
    // list has not been cleared since". Nothing here clears it, so the older entries can be from earlier runs.
    if (n > 1) fprintf(stderr, "tinynv:   (this list is kept until newer faults push entries out, so an older one may "
                               "be a repeat rather than a separate fault)\n");
  } else {
    // No MMU fault, so the addresses are not the story. What is left is the SMs' own error registers, which say a
    // kernel did something illegal rather than touched something unmapped.
    int shown = 0;
    for (uint32_t i = 0; i < st->numSMsToRead && i < TINYNV_DEBUG_MAX_SMS_PER_CALL; i++) {
      const tinynv_sm_error_state_t *e = &st->smErrorStateArray[i];
      if (!e->hwwGlobalEsr && !e->hwwWarpEsr && !e->hwwCgaEsr) continue;
      if (!gsp->fault_sm_have) { gsp->fault_sm_have = 1; gsp->fault_sm_esr = e->hwwWarpEsr ? e->hwwWarpEsr : e->hwwGlobalEsr; }
      if (shown == 8) { fprintf(stderr, "tinynv:   (more SMs report the same; the first eight are enough)\n"); break; }
      fprintf(stderr, "tinynv:   SM %u: global esr %#x, warp esr %#x, warp pc %#llx, address %#llx\n", i,
              e->hwwGlobalEsr, e->hwwWarpEsr, (unsigned long long)e->hwwWarpEsrPc64,
              (unsigned long long)e->hwwEsrAddr);
      shown++;
    }
    if (!shown)
      fprintf(stderr, "tinynv: the debugger reports no mmu fault and no SM error, so whatever gsp-rm is complaining "
                      "about did not happen inside a kernel on the compute channel\n");
  }
  free(st);
}

// The objects every later thing hangs off: a privileged client, the device, the physical chip under it, and an address
// space. This is the oracle's init_golden_image up to the point where it starts asking about engines.
int tinynv_gsp_init_objects(tinynv_gpu_t *g) {
  tinynv_gsp_t *gsp = &g->gsp;
  gsp->next_handle = TINYNV_RM_FIRST_HANDLE;

  tinynv_nv0000_alloc_t root;
  memset(&root, 0, sizeof(root));
  if (rm_alloc(g, 0, TINYNV_CLASS_ROOT, &root, sizeof(root), &gsp->priv_root)) return -1;

  tinynv_nv0080_alloc_t dev;
  memset(&dev, 0, sizeof(dev));
  dev.hClientShare = TINYNV_RM_PRIV_ROOT;
  if (rm_alloc(g, gsp->priv_root, TINYNV_CLASS_DEVICE, &dev, sizeof(dev), &gsp->device)) return -1;

  tinynv_nv2080_alloc_t sub;
  memset(&sub, 0, sizeof(sub));
  if (rm_alloc(g, gsp->device, TINYNV_CLASS_SUBDEVICE, &sub, sizeof(sub), &gsp->subdevice)) return -1;

  tinynv_vaspace_alloc_t va;
  memset(&va, 0, sizeof(va));
  if (rm_alloc(g, gsp->device, TINYNV_CLASS_VASPACE, &va, sizeof(va), &gsp->vaspace)) return -1;

  // which runlist each engine is on. entry word 2 names the engine, word 3 its runlist, which is what a channel needs.
  tinynv_fifo_device_info_t *info = calloc(1, sizeof(*info));
  if (!info) return tinynv_fail("out of memory for the engine table");
  if (rm_control(g, gsp->subdevice, TINYNV_CTRL_CMD_FIFO_GET_DEVICE_INFO_TABLE, info, sizeof(*info))) { free(info); return -1; }
  gsp->nengines = info->numEntries < TINYNV_FIFO_DEVICE_ENTRIES ? info->numEntries : TINYNV_FIFO_DEVICE_ENTRIES;
  for (uint32_t i = 0; i < gsp->nengines; i++) {
    gsp->engines[i] = info->entries[i].engineData[2];
    gsp->runlists[i] = info->entries[i].engineData[3];
  }
  // Read-only, fetched on every ordinary boot regardless of TINYNV_RUNQUEUE_SPLIT, printed on none of them until
  // now: numPbdmas/pbdmaIds per engine. NVIDIA's real kernel_fifo_*.c validates a channel's requested runqueue
  // against this count for its OWN engine before scheduling it (kfifoGetNumRunqueues_HAL, runqueue bounds-checked
  // against numSrcPbdmaIds); this driver's channel-alloc path never has, for any engine, on any boot. If the copy
  // (DMA_COPY) engine reports numPbdmas=1, runqueue ONE is not a real hardware lane for it at all, and setting
  // NVOS04_FLAGS_GROUP_CHANNEL_RUNQUEUE_ONE on that channel is undefined behaviour GSP-RM never rejects - which
  // would fully explain both hardware outcomes seen 2026-09-18 (a NULL-deref crash and, on a different build, a
  // hang ending in a real DART fault) as one root cause landing two different ways, not two different bugs.
  for (uint32_t i = 0; i < gsp->nengines; i++)
    fprintf(stderr, "tinynv: engine %-16s type 0x%x runlist %u numPbdmas %u pbdmaIds [%u,%u]\n",
            info->entries[i].engineName, gsp->engines[i], gsp->runlists[i],
            info->entries[i].numPbdmas, info->entries[i].pbdmaIds[0], info->entries[i].pbdmaIds[1]);
  free(info);

  // Half a gigabyte of address space whose page tables are built now and described to GSP-RM, so that the two sides
  // share a tree rather than each building its own. The levels go root first; the first is only as big as its entry
  // count, the rest are a page each.
  uint64_t res_size = 512ull << 20;
  uint64_t res_va = tinynv_mm_alloc_va(&g->mm, res_size, 0);
  if (res_va == TINYNV_BAD_ADDR) return -1;

  tinynv_pt_t levels[TINYNV_MAX_PT_LEVELS];
  int nlevels = 0;
  if (tinynv_mm_page_tables(&g->mm, g->mm.root_page_table, res_va, res_size, levels, &nlevels)) return -1;
  if (nlevels > TINYNV_MAX_PDE_LEVELS) return tinynv_fail("the page table is %d levels deep, more than the %d gsp-rm takes",
                                                          nlevels, TINYNV_MAX_PDE_LEVELS);

  tinynv_copy_pdes_t pdes;
  memset(&pdes, 0, sizeof(pdes));
  pdes.pageSize = res_size;
  pdes.numLevelsToCopy = 3; // the oracle's count, which is one fewer than it fills; gsp-rm reads what it needs
  pdes.virtAddrLo = res_va;
  pdes.virtAddrHi = res_va + res_size - 1;
  for (int i = 0; i < nlevels; i++) {
    pdes.levels[i].physAddress = levels[i].paddr;
    pdes.levels[i].size = i == 0 ? (uint64_t)g->mm.pte_cnt[0] * 8 : 0x1000;
    pdes.levels[i].aperture = 1;
    // the shift is the log of what one entry at this level covers
    uint64_t covers = g->mm.pte_covers[i];
    uint8_t shift = 0;
    while (covers > 1) { covers >>= 1; shift++; }
    pdes.levels[i].pageShift = shift;
  }
  if (rm_control(g, gsp->vaspace, TINYNV_CTRL_CMD_VASPACE_COPY_SERVER_RESERVED_PDES, &pdes, sizeof(pdes))) return -1;
  gsp->reserved_va = res_va;
  gsp->reserved_levels = nlevels;
  return 0;
}

// Create the channel work reaches the GPU through.
//
// The ring of command-buffer pointers lives in memory the GPU can address; the areas GSP-RM keeps the channel's state in
// are named by physical address, not virtual. One page holds both the ring and, above it, the small block the hardware
// writes its progress into, which is why the userd descriptor points into the same allocation at a fixed offset.
// The areas GSP-RM keeps a channel's own state in, which it wants by physical address and which the caller never sees.
static int fill_channel_state(tinynv_gpu_t *g, uint32_t client, tinynv_gpfifo_alloc_t *p, tinynv_vmap_t *ramfc_out,
                              tinynv_bootmem_t *mthd_out) {
  tinynv_vmap_t ramfc;
  if (tinynv_mm_valloc(&g->mm, 0x1000, 0x1000, 1, 0, &ramfc)) return -1;
  tinynv_bootmem_t mthd;
  if (tinynv_alloc_boot_mem(&g->mm, 0x5000, NULL, 0, &mthd)) return -1;
  p->ramfcMem = (tinynv_memory_desc_t){.base = ramfc.ranges[0].paddr, .size = 0x200, .addressSpace = 2};
  p->instanceMem = (tinynv_memory_desc_t){.base = ramfc.ranges[0].paddr, .size = 0x1000, .addressSpace = 2};
  p->mthdbufMem = (tinynv_memory_desc_t){.base = mthd.paddr, .size = 0x5000, .addressSpace = 2};

  // an ordinary client's channel also names where its error block and its progress block live
  if (client != TINYNV_RM_PRIV_ROOT && p->hObjectError) {
    p->errorNotifierMem = (tinynv_memory_desc_t){.base = 0, .size = 0xecc, .addressSpace = 0};
    p->userdMem = (tinynv_memory_desc_t){.base = p->hUserdMemory[0] + p->userdOffset[0], .size = 0x400, .addressSpace = 2};
  }
  if (ramfc_out) *ramfc_out = ramfc;
  if (mthd_out) *mthd_out = mthd;
  return 0;
}

int tinynv_gsp_init_channel(tinynv_gpu_t *g) {
  tinynv_gsp_t *gsp = &g->gsp;
  const uint32_t entries = 32;

  if (tinynv_mm_valloc(&g->mm, 4 << 10, 0x1000, 1, 0, &gsp->gpfifo)) return -1;

  tinynv_gpfifo_alloc_t p;
  memset(&p, 0, sizeof(p));
  p.gpFifoOffset = gsp->gpfifo.va;
  p.gpFifoEntries = entries;
  p.engineType = 1; // graphics, which is what compute runs on
  p.cid = 3;
  p.hVASpace = gsp->vaspace;
  p.internalFlags = 0x1a;
  p.flags = 0x200320;
  // the hardware's progress block sits just past the ring's entries, in the same page
  uint64_t userd_off = (uint64_t)entries * 8;
  p.userdOffset[0] = userd_off;
  if (fill_channel_state(g, TINYNV_RM_PRIV_ROOT, &p, &gsp->ramfc, &gsp->mthdbuf)) return -1;
  p.userdMem = (tinynv_memory_desc_t){.base = gsp->gpfifo.ranges[0].paddr + userd_off, .size = 0x20, .addressSpace = 2};

  if (rm_alloc(g, gsp->device, TINYNV_CLASS_GPFIFO, &p, sizeof(p), &gsp->channel)) return -1;

  // which runlist this channel is on, which submitting work later has to name
  gsp->channel_runlist = 0;
  for (uint32_t i = 0; i < gsp->nengines; i++)
    if (gsp->engines[i] == p.engineType) gsp->channel_runlist = gsp->runlists[i];
  return 0;
}

static uint64_t round_up_to(uint64_t v, uint64_t a) { return a ? (v + a - 1) / a * a : v; }

// The buffers the graphics engine keeps its state in, and telling GSP-RM where they are.
//
// GSP-RM is asked how big each must be, the driver allocates them, and each is described by whichever of its addresses
// the engine wants: some by physical address, some by virtual, one by physical only. The identifiers and the peculiar
// index arithmetic below are the oracle's, and neither is documented anywhere else.
int tinynv_gsp_init_gr_context(tinynv_gpu_t *g) {
  tinynv_gsp_t *gsp = &g->gsp;

  tinynv_kgr_ctx_info_t *info = calloc(1, sizeof(*info));
  if (!info) return tinynv_fail("out of memory for the context buffer sizes");
  if (rm_control(g, gsp->subdevice, TINYNV_CTRL_CMD_STATIC_KGR_GET_CONTEXT_BUFFERS_INFO, info, sizeof(*info))) {
    free(info);
    return -1;
  }
  const tinynv_ctx_buffer_info_t *e = info->engineContextBuffersInfo[0].engine;

  // index 0 is the context itself, which gets a fixed extra allowance; 16 is the patch buffer; 17 upward are the
  // configuration areas, addressed here as 3..10 with an offset of 14, one of which wants 2 MB alignment
  uint64_t gr_size = gsp->gr_size = round_up_to((uint64_t)e[0].size + 0x40000, e[0].alignment);
  uint64_t patch_size = gsp->patch_size = round_up_to(e[16].size, e[16].alignment);
  uint64_t cfg[11];
  for (int x = 3; x <= 10; x++) cfg[x] = round_up_to(e[x + 14].size, x == 5 ? (2u << 20) : e[x + 14].alignment);

  // id, size, wanted by physical address, wanted by virtual address. id 1 is the engine's own local patch buffer, which
  // is allocated but never promoted, so it is not in this list at all.
  const struct { uint16_t id; uint64_t size; int phys, virt; } WANT[] = {
    {0, gr_size, 1, 1},    {2, patch_size, 1, 1},
    {3, cfg[3], 0, 1},     {4, cfg[4], 0, 1},   {5, cfg[5], 0, 1},   {6, cfg[6], 0, 1},
    {9, cfg[9], 1, 1},     {10, cfg[10], 1, 0}, {11, cfg[10], 1, 1},
  };
  const size_t n = sizeof(WANT) / sizeof(*WANT);

  tinynv_promote_ctx_t *prom = calloc(1, sizeof(*prom));
  if (!prom) { free(info); return tinynv_fail("out of memory for the context promotion"); }
  prom->engineType = 1;
  prom->hChanClient = TINYNV_RM_PRIV_ROOT;
  prom->hObject = gsp->channel;
  prom->entryCount = (uint32_t)n;
  for (size_t i = 0; i < n; i++) {
    tinynv_vmap_t *m = &gsp->grctx[i];
    if (tinynv_mm_valloc(&g->mm, WANT[i].size, 0x1000, 1, 0, m)) { free(prom); free(info); return -1; }
    prom->promoteEntry[i].bufferId = WANT[i].id;
    prom->promoteEntry[i].gpuVirtAddr = WANT[i].virt ? m->va : 0;
    prom->promoteEntry[i].bInitialize = (uint8_t)WANT[i].phys;
    prom->promoteEntry[i].gpuPhysAddr = WANT[i].phys ? m->ranges[0].paddr : 0;
    prom->promoteEntry[i].size = WANT[i].phys ? WANT[i].size : 0;
    prom->promoteEntry[i].physAttr = WANT[i].phys ? 4 : 0;
    prom->promoteEntry[i].bNonmapped = (uint8_t)(WANT[i].phys && !WANT[i].virt);
  }
  gsp->ngrctx = (int)n;

  int rc = rm_control(g, gsp->subdevice, TINYNV_CTRL_CMD_GPU_PROMOTE_CTX, prom, sizeof(*prom));
  free(prom);
  free(info);
  if (rc) return -1;

  // the classes the channel will actually run: compute, and the copy engine behind memcpy
  if (rm_alloc(g, gsp->channel, TINYNV_CLASS_COMPUTE, NULL, 0, &gsp->compute_obj)) return -1;
  return rm_alloc(g, gsp->channel, TINYNV_CLASS_DMA_COPY, NULL, 0, &gsp->dma_copy_obj);
}

// The client the driver works through, as against the privileged one that set the graphics context up.
//
// Its own device, address space and channel group, because the work submitted later is not privileged and GSP-RM treats
// the privileged client specially. The address space here is externally owned: the page tables are the driver's, and
// GSP-RM is told where their root is rather than building its own.
int tinynv_gsp_open_client(tinynv_gpu_t *g) {
  tinynv_gsp_t *gsp = &g->gsp;
  const uint32_t cl = TINYNV_RM_USER_ROOT;

  // before this client allocates anything, take the memory that has to stay where the processor can reach it
  if (tinynv_mm_reserve_bar_pool(&g->mm)) return -1;

  tinynv_nv0000_alloc_t root;
  memset(&root, 0, sizeof(root));
  if (rm_alloc_as(g, cl, 0, 0, TINYNV_CLASS_ROOT, &root, sizeof(root), &gsp->user_root)) return -1;

  tinynv_nv0080_alloc_t dev;
  memset(&dev, 0, sizeof(dev));
  dev.hClientShare = cl;
  dev.vaMode = 0; // several address spaces are allowed but not required
  if (rm_alloc_as(g, cl, cl, 0, TINYNV_CLASS_DEVICE, &dev, sizeof(dev), &gsp->user_device)) return -1;

  tinynv_nv2080_alloc_t sub;
  memset(&sub, 0, sizeof(sub));
  if (rm_alloc_as(g, cl, gsp->user_device, 0, TINYNV_CLASS_SUBDEVICE, &sub, sizeof(sub), &gsp->user_subdevice)) return -1;

  tinynv_memory_virtual_alloc_t vm;
  memset(&vm, 0, sizeof(vm));
  vm.limit = 0x1ffffffffffffull;
  if (rm_alloc_as(g, cl, gsp->user_device, 0, TINYNV_CLASS_MEMORY_VIRTUAL, &vm, sizeof(vm), &gsp->user_virtmem)) return -1;

  // ask for clocks: without this the chip runs at its idle point and a benchmark measures the governor
  tinynv_perf_boost_t boost;
  memset(&boost, 0, sizeof(boost));
  boost.duration = 0xffffffff;
  boost.flags = (TINYNV_PERF_BOOST_CUDA_YES << 4) | (TINYNV_PERF_BOOST_CUDA_PRIORITY_HIGH << 6) | TINYNV_PERF_BOOST_CMD_BOOST_TO_MAX;
  if (rm_control_as(g, cl, gsp->user_subdevice, TINYNV_CTRL_CMD_PERF_BOOST, &boost, sizeof(boost))) return -1;

  // An address space this client owns, but whose page tables the driver builds: "externally owned" means GSP-RM is told
  // where the root of the tree is rather than making one. Page faulting is on, which is what lets a kernel touch memory
  // that has not been paged in rather than dying.
  tinynv_vaspace_alloc_t va;
  memset(&va, 0, sizeof(va));
  va.vaBase = 0x1000;
  va.vaSize = 0x1fffffb000000ull;
  va.flags = TINYNV_VASPACE_FLAGS_ENABLE_PAGE_FAULTING | TINYNV_VASPACE_FLAGS_IS_EXTERNALLY_OWNED;
  if (rm_alloc_as(g, cl, gsp->user_device, 0, TINYNV_CLASS_VASPACE, &va, sizeof(va), &gsp->user_vaspace)) return -1;

  // and tell the firmware where that space's page tables begin, since it is not making them
  tinynv_set_page_directory_t pd;
  memset(&pd, 0, sizeof(pd));
  pd.hClient = cl;
  pd.hDevice = gsp->user_device;
  pd.pasid = 0xffffffff;
  pd.params.physAddress = g->mm.root_page_table;
  pd.params.numEntries = g->mm.pte_cnt[0];
  pd.params.flags = 0x8; // every channel in the space, rather than one named below
  pd.params.hVASpace = gsp->user_vaspace;
  pd.params.subDeviceId = 1;
  pd.params.pasid = 0xffffffff;
  if (rpc_call(g, TINYNV_MSG_FUNCTION_SET_PAGE_DIRECTORY, &pd, sizeof(pd), NULL, 0, "handing over the page tables")) return -1;

  tinynv_channel_group_alloc_t grp;
  memset(&grp, 0, sizeof(grp));
  grp.engineType = TINYNV_ENGINE_TYPE_GRAPHICS;
  if (rm_alloc_as(g, cl, gsp->user_device, 0, TINYNV_CLASS_CHANNEL_GROUP, &grp, sizeof(grp), &gsp->user_group)) return -1;

  // The scheduler's timeslice for this group, when asked for.
  //
  // A's raw columns put the token boundary's remaining ~700 us in the card scheduling two channels against each
  // other: the copy channel waiting to be picked up while compute finishes, or compute waiting after the copy ran.
  // On a bench with one channel active neither happens - 0 of 80 launches past 5 ms. A shorter timeslice should
  // shrink every such wait, not only this one, which is what makes it worth testing before any workaround that
  // avoids the second channel instead of explaining it.
  //
  // Off unless asked for. This is a scheduler setting on every channel we create, on a path with no recording behind
  // it, and the failure mode of a bad value is not a slow decode - it is a scheduling behaviour nothing here would
  // recognise.
  { const char *e = getenv("TINYNV_TIMESLICE_US");
    uint64_t us = e && *e ? strtoull(e, NULL, 10) : 0;
    if (us) {
      tinynv_timeslice_t ts = {.timesliceUs = us};
      if (rm_control_as(g, cl, gsp->user_group, TINYNV_CTRL_CMD_SET_TIMESLICE, &ts, sizeof(ts))) return -1;
      fprintf(stderr, "libtinynv: the channel group's timeslice was asked to be %llu us\n", (unsigned long long)us);
    } }

  tinynv_ctxshare_alloc_t share;
  memset(&share, 0, sizeof(share));
  share.hVASpace = gsp->user_vaspace;
  share.flags = TINYNV_CTXSHARE_FLAGS_SUBCONTEXT_ASYNC;
  if (rm_alloc_as(g, cl, gsp->user_group, 0, TINYNV_CLASS_CONTEXT_SHARE, &share, sizeof(share), &gsp->user_ctxshare)) return -1;
  { const char *e = getenv("TINYNV_SPLIT_CTXSHARE");
    if (e && *e && *e != '0') {
      tinynv_ctxshare_alloc_t cs;
      memset(&cs, 0, sizeof(cs));
      cs.hVASpace = gsp->user_vaspace;
      cs.flags = TINYNV_CTXSHARE_FLAGS_SUBCONTEXT_ASYNC;
      if (rm_alloc_as(g, cl, gsp->user_group, 0, TINYNV_CLASS_CONTEXT_SHARE, &cs, sizeof(cs), &gsp->copy_ctxshare))
        return -1;
      fprintf(stderr, "libtinynv: the copy channel was given a context share of its own (was asked for)\n");
    } }

  // the shape of the die: how many partitions it can have, not how many are turned on
  tinynv_gr_static_info_t *info = calloc(1, sizeof(*info));
  if (!info) return tinynv_fail("out of memory for the chip's geometry");
  if (rm_control_as(g, cl, gsp->user_subdevice, TINYNV_CTRL_CMD_STATIC_KGR_GET_INFO, info, sizeof(*info))) { free(info); return -1; }
  const tinynv_gr_info_t *il = info->engineInfo[0].infoList;
  // these are maxima for the die, not the enabled counts; see the note in gsp.h before reporting them to anyone
  gsp->max_gpcs = il[TINYNV_GR_INFO_NUM_GPCS].data;
  gsp->max_tpc_per_gpc = il[TINYNV_GR_INFO_NUM_TPC_PER_GPC].data;
  gsp->max_sm_per_tpc = il[TINYNV_GR_INFO_NUM_SM_PER_TPC].data;
  gsp->max_warps_per_sm = il[TINYNV_GR_INFO_MAX_WARPS_PER_SM].data;
  gsp->sm_version = il[TINYNV_GR_INFO_SM_VERSION].data;
  free(info);

  // A page of device memory with no processor mapping, allocated before anything else this client does. What it holds is
  // not visible from the boundary: it is unzeroed, unmapped from here, and one page table entry wide. It is reproduced
  // because leaving it out moves every address after it, which is how its absence was found.
  if (tinynv_mm_alloc_buffer(&g->mm, 0x1000, 0, 0, 0, 0, 0, &gsp->scratch)) return -1;

  // and the buffer copies are staged through. A remote cannot map the host's own pages, so every transfer in either
  // direction goes through this one allocation in chunks; it is taken once and kept for the life of the device.
  return tinynv_mm_alloc_buffer(&g->mm, 128ull << 20, 1, 0, 0, 0, 0, &gsp->staging);
}

// Hand the graphics engine a set of context buffers, saying for each whether the engine should take its physical
// address, its virtual address, or both. The same set is promoted twice: once by physical address so the engine can
// initialise them, then again by virtual address so it can reach them while running.
static int promote(tinynv_gpu_t *g, uint32_t client, uint32_t object, tinynv_vmap_t *bufs, const uint64_t *sizes, int n,
                   int by_phys) {
  tinynv_gsp_t *gsp = &g->gsp;
  tinynv_promote_ctx_t *prom = calloc(1, sizeof(*prom));
  if (!prom) return tinynv_fail("out of memory for a context promotion");
  prom->engineType = 1;
  prom->hChanClient = client;
  prom->hObject = object;
  prom->entryCount = (uint32_t)n;
  for (int i = 0; i < n; i++) {
    prom->promoteEntry[i].bufferId = (uint16_t)i;
    prom->promoteEntry[i].gpuVirtAddr = by_phys ? 0 : bufs[i].va;
    prom->promoteEntry[i].bInitialize = (uint8_t)by_phys;
    prom->promoteEntry[i].gpuPhysAddr = by_phys ? bufs[i].ranges[0].paddr : 0;
    prom->promoteEntry[i].size = by_phys ? sizes[i] : 0;
    prom->promoteEntry[i].physAttr = by_phys ? 4 : 0;
    prom->promoteEntry[i].bNonmapped = (uint8_t)by_phys; // physical only, so the engine is told not to expect a mapping
  }
  int rc = rm_control_as(g, client, gsp->user_subdevice, TINYNV_CTRL_CMD_GPU_PROMOTE_CTX, prom, sizeof(*prom));
  free(prom);
  return rc;
}

// One channel work is submitted through: its ring, the block the hardware writes its progress into just above the ring,
// and the object that says what kind of work it runs.
static int new_queue(tinynv_gpu_t *g, tinynv_queue_t *q, uint64_t offset, uint32_t entries, int compute) {
  tinynv_gsp_t *gsp = &g->gsp;
  const uint32_t cl = TINYNV_RM_USER_ROOT;

  // where the engine reports faults, and in enough detail that it is large
  if (tinynv_mm_alloc_buffer(&g->mm, 48ull << 20, 0, 0, 1, 0, 0, &q->notifier)) return -1;

  tinynv_gpfifo_alloc_t p;
  memset(&p, 0, sizeof(p));
  p.gpFifoOffset = gsp->fifo_mem.va + offset;
  p.gpFifoEntries = entries;
  // these handles are physical addresses: that is what the allocator hands back for memory the driver owns outright
  p.hObjectError = (uint32_t)q->notifier.ranges[0].paddr;
  p.hObjectBuffer = (uint32_t)gsp->fifo_mem.ranges[0].paddr;
  p.hUserdMemory[0] = (uint32_t)gsp->fifo_mem.ranges[0].paddr;
  p.userdOffset[0] = (uint64_t)entries * 8 + offset;
  // Both channels have always shared ONE context share, which is one subcontext, which means the engine has to switch
  // context between them rather than holding both.
  //
  // Session A established the mechanism today: the compute and copy channels contend at every token boundary and the
  // runlist's switching decides who waits, costing ~700 us on about half the tokens. Their proposed cheap fix was to
  // put the two channels in one channel group - but they are ALREADY in one group, gsp->user_group, so that is not
  // available. The subcontext is the thing that is actually shared and not required to be.
  //
  // TINYNV_SPLIT_CTXSHARE=1 gives the copy channel its own, allocated with the same SUBCONTEXT_ASYNC flag. Off by
  // default and it is a guess, not a diagnosis: two channels in separate async subcontexts SHOULD be able to hold the
  // engine together where one subcontext cannot, but nothing here has measured that and no recording contains it.
  p.hContextShare = gsp->user_ctxshare;
  if (!compute && gsp->copy_ctxshare) p.hContextShare = gsp->copy_ctxshare;
  // A different axis from the ctxshare guess above, from NVIDIA's own header rather than a hypothesis about our
  // code: alloc_channel.h's NVOS04_FLAGS_GROUP_CHANNEL_RUNQUEUE (bit 4:4, GP10x+) "specifies which runqueue the
  // allocated channel will be executed on in a TSG. Channels on different runqueues within a TSG may be able to
  // feed methods into the engine simultaneously." Both our channels have always defaulted to runqueue 0 (`p.flags`
  // is memset to zero and nothing here has ever set this bit) - so the copy and compute channels contending for
  // one runlist slot at every token boundary (the mechanism this file's own comments above settled) has never
  // been tested against the one flag NVIDIA's own driver uses to let two channels in a TSG feed the engine at
  // once. TINYNV_RUNQUEUE_SPLIT=1 puts the copy channel on runqueue ONE; compute stays on the default. Off by
  // default: NVIDIA's own doc hedges ("may be able to"), and this could land like TINYNV_TIMESLICE_US - a real,
  // exercised mechanism that doesn't net out to a shorter token. One hardware run decides it.
  //
  // RUN, 2026-09-18: it does not just fail to help - it crashes. The GPFIFO alloc itself succeeds (GSP-RM accepts
  // the channel on runqueue ONE without complaint, boot proceeds, ggml_cuda_init reports the card normally), but
  // the FIRST real copy-engine job after it - the first H2D weight upload in llama_model_loader::load_all_data -
  // SIGSEGVs inside tinynv_exec_upload (NULL deref; symbolicated: tinynv_exec_upload+216 <- tinynv_memcpy_htod
  // <- cudaMemcpyAsync <- ggml_backend_cuda_buffer_set_tensor <- load_all_data). Card verified undamaged after
  // (preflight clean, no re-enumeration, op-verify 450/450 at chain depths 32/64/128 on the stock binary).
  //
  // RETRIED on a different build the same day: no crash this time, but a hang - the process spun at 100% CPU
  // (confirmed via ps: state R, not blocked) at the same point, consistent with tinynv_exec_idle's wait looping
  // on a completion value that never arrives. Had to be force-killed; the card then showed a genuine DART fault
  // (not the benign re-enumeration pattern this project usually sees), needing a physical replug. Two different
  // failure shapes, same program point, on two different builds - consistent with one root defect resolving as
  // undefined behaviour rather than two separate bugs.
  //
  // ROOT CAUSE FOUND, 2026-09-18, without touching hardware again: this driver has always fetched, and always
  // thrown away, exactly the fact that answers this. tinynv_gsp_init_objects's FIFO_GET_DEVICE_INFO_TABLE call
  // (above this function) returns numPbdmas/pbdmaIds per engine - printed for the first time on the ordinary
  // boot that produced this comment: GR0 (compute) reports numPbdmas=2, but EVERY copy engine instance (CE0
  // through CE7) reports numPbdmas=1. NVIDIA's real driver validates a channel's requested runqueue against its
  // OWN engine's PBDMA count before scheduling it (kfifoGetNumRunqueues_HAL, bounds-checked against
  // numSrcPbdmaIds in kernel_fifo_gm107.c); this driver's channel-alloc path never has, for any engine, on any
  // boot. So NVOS04_FLAGS_GROUP_CHANNEL_RUNQUEUE_ONE on the copy channel asks GSP-RM for a PBDMA index the copy
  // engine does not have - an out-of-range hardware selector that the allocation call happily accepts (nothing
  // in the accept path checks it), and which then produces undefined behaviour whose exact shape depends on
  // incidental memory layout - the crash and the hang are the same defect landing two different ways.
  //
  // This also settles the mechanism question, not just the crash: the flag's own wording ("channels on different
  // runqueues within a TSG may be able to feed methods into THE ENGINE simultaneously" - singular) describes two
  // channels of the SAME engine sharing that engine's PBDMA pair, not two channels of DIFFERENT engines (compute
  // vs copy) coordinating with each other. GR0 having 2 PBDMAs could in principle let two COMPUTE channels feed
  // concurrently; it says nothing about the compute<->copy handoff this project actually needs to shorten. Not a
  // missing-plumbing problem - the wrong mechanism for this job, now confirmed by hardware data rather than
  // inferred from a crash. Left OFF, in the tree, as the evidence - same status as TINYNV_SPLIT_CTXSHARE above.
  { const char *e = getenv("TINYNV_RUNQUEUE_SPLIT");
    if (e && *e && *e != '0' && !compute) {
      p.flags |= (1u << 4); // NVOS04_FLAGS_GROUP_CHANNEL_RUNQUEUE_ONE, field 4:4
      fprintf(stderr, "libtinynv: the copy channel was put on runqueue ONE (was asked for)\n");
    } }
  if (fill_channel_state(g, cl, &p, NULL, NULL)) return -1;
  if (rm_alloc_as(g, cl, gsp->user_group, 0, TINYNV_CLASS_GPFIFO, &p, sizeof(p), &q->channel)) return -1;

  if (compute) {
    if (rm_alloc_as(g, cl, q->channel, 0, TINYNV_CLASS_COMPUTE, NULL, 0, &q->object)) return -1;

    // A compute object on an ordinary client gets its own copy of the first three context buffers, promoted twice: once
    // by physical address so the engine can initialise them, then by virtual address so it can reach them.
    const uint64_t sizes[3] = {gsp->gr_size, gsp->patch_size, gsp->patch_size};
    for (int i = 0; i < 3; i++)
      if (tinynv_mm_valloc(&g->mm, sizes[i], 0x1000, 1, 0, &q->grctx[i])) return -1;
    if (promote(g, cl, q->channel, q->grctx, sizes, 3, 1)) return -1;

    // The second promotion allocates a second set and then does not use it. That is the oracle's accident, not a design:
    // it reaches for the already-allocated buffer with a dictionary lookup whose default is the allocation call, and
    // python evaluates that default whether or not the key is there. A membership test, or a default that is not
    // evaluated until it is needed, was meant. It is in the recording, so it is reproduced -- and it costs about seven
    // megabytes of video memory per compute channel, which is worth knowing before anyone wonders where it went.
    //
    // It is an upstream bug and worth reporting there, but NOT before this driver stops depending on the recording
    // matching: fixing it in the oracle would change the trace we pinned, and the pin is what makes the first hardware
    // run a sequence the card has already accepted. After that, both the fix upstream and the trim here. (Session A.)
    tinynv_vmap_t discarded[3];
    for (int i = 0; i < 3; i++)
      if (tinynv_mm_valloc(&g->mm, sizes[i], 0x1000, 1, 0, &discarded[i])) return -1;
    if (promote(g, cl, q->channel, q->grctx, sizes, 3, 0)) return -1;

    tinynv_debugger_alloc_t dbg;
    memset(&dbg, 0, sizeof(dbg));
    dbg.hAppClient = cl;
    dbg.hClass3dObject = q->object;
    if (rm_alloc_as(g, cl, gsp->user_device, 0, TINYNV_CLASS_DEBUGGER, &dbg, sizeof(dbg), &gsp->user_debugger)) return -1;
  } else if (rm_alloc_as(g, cl, q->channel, 0, TINYNV_CLASS_DMA_COPY, NULL, 0, &q->object)) return -1;

  // the token names this channel at the doorbell. gsp-rm fills in only the channel part; the runlist and, on this
  // architecture, an enable bit are the driver's to add.
  tinynv_work_submit_token_t tok;
  memset(&tok, 0, sizeof(tok));
  tok.workSubmitToken = 0xffffffffu;
  if (rm_control_as(g, cl, q->channel, TINYNV_CTRL_CMD_GET_WORK_SUBMIT_TOKEN, &tok, sizeof(tok))) return -1;
  q->token = tok.workSubmitToken | (gsp->channel_runlist << 16) | (1u << 30);
  q->entries = entries;
  q->ring_va = gsp->fifo_mem.va + offset;
  // The control block sits immediately above the ring - that is what userdOffset told GSP-RM a moment ago - and GPPut is
  // a field inside it. Its offset is NVIDIA's, checked against the vendored header in test_headers; it was guessed at
  // 0x10 until the first real submission went nowhere, because a boot recording contains no submissions and nothing in
  // the suite could tell a wrong address from a right one.
  q->gpput_off = offset + (uint64_t)entries * 8 + TINYNV_GPFIFO_GPPUT_OFF;
  return 0;
}

// The compute and copy channels, then scheduling for the group they belong to.
int tinynv_gsp_init_queues(tinynv_gpu_t *g) {
  tinynv_gsp_t *gsp = &g->gsp;
  // one allocation holds both rings and both control blocks, and it must be writable from here, so it comes out of the
  // reserve rather than wherever the physical allocator happens to be
  if (tinynv_mm_alloc_buffer(&g->mm, 3ull << 20, 0, 1, 0, 1, 0, &gsp->fifo_mem)) return -1;

  if (new_queue(g, &gsp->compute_q, 0, 0x10000, 1)) return -1;
  if (new_queue(g, &gsp->copy_q, 0x100000, 0x10000, 0)) return -1;

  tinynv_gpfifo_schedule_t sched;
  memset(&sched, 0, sizeof(sched));
  sched.bEnable = 1;
  return rm_control_as(g, TINYNV_RM_USER_ROOT, gsp->user_group, TINYNV_CTRL_CMD_GPFIFO_SCHEDULE, &sched, sizeof(sched));
}

// The queue GSP-RM answers on, and the arguments that tell it where the queue is.
static int init_rm_args(tinynv_gpu_t *g) {
  tinynv_gsp_t *gsp = &g->gsp;
  const uint64_t queue_size = gsp->queue_size = 0x40000;

  // the gpu reaches the queues through a page table of its own, which lives in the same allocation just below them
  uint64_t queue_pte_cnt = (queue_size * 2) / PAGE;
  uint64_t pte_cnt = queue_pte_cnt + round_up(queue_pte_cnt * 8, PAGE) / PAGE;
  uint64_t pt_size = round_up(pte_cnt * 8, PAGE);
  if (tinynv_alloc_boot_mem(&g->mm, pt_size + queue_size * 2, NULL, 1, &gsp->queues)) return -1;
  gsp->cmd_q_off = pt_size;

  for (size_t i = 0; i < gsp->queues.naddrs; i++) nv_wr_block(&gsp->queues.view, i * 8, &gsp->queues.addrs[i], 8);

  tinynv_gsp_args_cached_t args;
  memset(&args, 0, sizeof(args));
  args.bDmemStack = 1;
  args.messageQueueInitArguments.sharedMemPhysAddr = gsp->queues.addrs[0];
  args.messageQueueInitArguments.pageTableEntryCount = (uint32_t)pte_cnt;
  args.messageQueueInitArguments.cmdQueueOffset = pt_size;
  args.messageQueueInitArguments.statQueueOffset = pt_size + queue_size;
  if (tinynv_alloc_boot_mem(&g->mm, sizeof(args), &args, -1, &gsp->rm_args)) return -1;
  gsp->rm_args_sysmem = gsp->rm_args.addrs[0];

  tinynv_msgq_tx_header_t hdr;
  memset(&hdr, 0, sizeof(hdr));
  hdr.version = 0;
  hdr.size = (uint32_t)queue_size;
  hdr.msgSize = (uint32_t)PAGE;
  hdr.msgCount = (uint32_t)((queue_size - PAGE) / PAGE);
  hdr.writePtr = 0;
  hdr.flags = 1;
  hdr.rxHdrOff = (uint32_t)sizeof(hdr);
  hdr.entryOff = (uint32_t)PAGE;
  nv_wr_block(&gsp->queues.view, pt_size, &hdr, sizeof(hdr));

  // read the header back the way the oracle does, so what the driver believes about the queue comes from the queue
  return rpcq_attach(g, &gsp->cmd_q, pt_size, "command");
}

// The log buffers and the argument block GSP-RM's operating system starts with.
static int init_libos_args(tinynv_gpu_t *g) {
  tinynv_gsp_t *gsp = &g->gsp;
  static const char *LOGS[] = {"INIT", "INTR", "RM", "MNOC", "KRNL"};

  if (tinynv_alloc_boot_mem(&g->mm, 2 * MB, NULL, -1, &gsp->logbuf)) return -1;
  if (tinynv_alloc_boot_mem(&g->mm, PAGE, NULL, -1, &gsp->libos_args)) return -1;
  gsp->libos_args_sysmem = gsp->libos_args.addrs[0];

  tinynv_libos_region_t regions[6];
  memset(regions, 0, sizeof(regions));
  for (int i = 0; i < 5; i++) {
    char name[16];
    snprintf(name, sizeof(name), "LOG%s", LOGS[i]);
    regions[i].kind = TINYNV_LIBOS_MEMORY_REGION_CONTIGUOUS;
    regions[i].loc = TINYNV_LIBOS_MEMORY_REGION_LOC_SYSMEM;
    regions[i].size = 0x10000;
    regions[i].id8 = id8_of(name);
    regions[i].pa = gsp->logbuf.addrs[0] + 0x10000ull * i;
  }
  regions[5].kind = TINYNV_LIBOS_MEMORY_REGION_CONTIGUOUS;
  regions[5].loc = TINYNV_LIBOS_MEMORY_REGION_LOC_SYSMEM;
  regions[5].size = PAGE;
  regions[5].id8 = id8_of("RMARGS");
  regions[5].pa = gsp->rm_args_sysmem;
  nv_wr_block(&gsp->libos_args.view, 0, regions, sizeof(regions));
  return 0;
}

// GSP-RM's own firmware, under the three level page table the GPU walks to read it.
//
// The image is 60 MB and the GPU reads it by physical page, so it is described by a radix tree: a root page of pointers
// to a page of pointers to pages of pointers to the image's pages. Each level is sized from the one below it, which is
// why the loop runs backwards.
static int init_gsp_image(tinynv_gpu_t *g) {
  tinynv_gsp_t *gsp = &g->gsp;
  // the oracle loads this one from the ga102 directory whatever the chip is: it is the same file for every architecture
  if (tinynv_fw_load("ga102", "gsp-" TINYNV_FW_VER ".bin", GSP_SHA, &gsp->gsp_fw)) return -1;

  const uint8_t *image, *sig;
  size_t image_len, sig_len;
  char sig_name[64];
  // ".fwsignature_gb20x" for a GB202: the family, lowercased, with the last digit replaced by an x
  snprintf(sig_name, sizeof(sig_name), ".fwsignature_%c%c%c%cx", lower_ascii(g->dev.chip_name[0]), lower_ascii(g->dev.chip_name[1]),
           lower_ascii(g->dev.chip_name[2]), lower_ascii(g->dev.chip_name[3]));
  if (tinynv_elf_section(&gsp->gsp_fw, ".fwimage", &image, &image_len)) return -1;
  if (tinynv_elf_section(&gsp->gsp_fw, sig_name, &sig, &sig_len)) return -1;
  gsp->gsp_image_size = image_len;
  // The reservation at the top of video memory was sized against this image (fw_layout.h): a bigger one is refused
  // here, where the message can say why, rather than placed under the manager's feet.
  if (image_len > TINYNV_FW_IMAGE_BOUND)
    return tinynv_fail("the GSP-RM image is %zu bytes, past the %llu MB the firmware reservation was sized for - raise "
                       "TINYNV_FW_IMAGE_BOUND and the reservation with it (fw_layout.h)",
                       image_len, (unsigned long long)(TINYNV_FW_IMAGE_BOUND >> 20));

  size_t npages[4], offsets[4] = {0, 0, 0, 0};
  npages[3] = round_up(image_len, PAGE) / PAGE;
  for (int i = 3; i > 0; i--) npages[i - 1] = ((npages[i] - 1) >> (TINYNV_LIBOS_MEMORY_REGION_RADIX_PAGE_LOG2 - 3)) + 1;
  for (int i = 1; i < 4; i++) offsets[i] = offsets[i - 1] + npages[i - 1] * PAGE;

  if (tinynv_alloc_boot_mem(&g->mm, offsets[3] + image_len, NULL, -1, &gsp->radix3)) return -1;
  nv_wr_block(&gsp->radix3.view, offsets[3], image, image_len);

  // each level holds the addresses of the pages of the level below it
  size_t cur = 0;
  for (int i = 0; i < 3; i++) {
    cur += npages[i];
    if (cur + npages[i + 1] > gsp->radix3.naddrs) return tinynv_fail("the gsp image page table does not fit its allocation");
    nv_wr_block(&gsp->radix3.view, offsets[i], &gsp->radix3.addrs[cur], npages[i + 1] * 8);
  }

  return tinynv_alloc_boot_mem(&g->mm, sig_len, sig, -1, &gsp->signature);
}

// The RISC-V bootloader that starts GSP-RM, in NVIDIA's own container rather than an ELF.
static int init_boot_binary_image(tinynv_gpu_t *g) {
  tinynv_gsp_t *gsp = &g->gsp;
  if (tinynv_fw_load(g->dev.fw_name, "bootloader-" TINYNV_FW_VER ".bin", BL_SHA, &gsp->bl_fw)) return -1;

  tinynv_nvfw_bin_hdr_t h;
  if (gsp->bl_fw.size < sizeof(h)) return tinynv_fail("the riscv bootloader is truncated");
  memcpy(&h, gsp->bl_fw.data, sizeof(h));
  if ((uint64_t)h.data_offset + h.data_size > gsp->bl_fw.size || (uint64_t)h.header_offset + sizeof(gsp->bl_desc) > gsp->bl_fw.size)
    return tinynv_fail("the riscv bootloader's header points outside the file");
  memcpy(&gsp->bl_desc, gsp->bl_fw.data + h.header_offset, sizeof(gsp->bl_desc));
  gsp->bootloader_size = h.data_size;
  if (h.data_size > TINYNV_FW_BOOTBIN_BOUND)   // same reason as the image bound above
    return tinynv_fail("the riscv bootloader is %u bytes, past the %llu MB the firmware reservation was sized for "
                       "(fw_layout.h)", h.data_size, (unsigned long long)(TINYNV_FW_BOOTBIN_BOUND >> 20));
  return tinynv_alloc_boot_mem(&g->mm, h.data_size, gsp->bl_fw.data + h.data_offset, -1, &gsp->bootloader);
}

// One structure naming everything above, which is what the chain of trust message actually points at.
static int init_wpr_meta(tinynv_gpu_t *g) {
  tinynv_gsp_t *gsp = &g->gsp;
  if (init_gsp_image(g)) return -1;
  if (init_boot_binary_image(g)) return -1;

  tinynv_wpr_meta_t m;
  memset(&m, 0, sizeof(m));
  m.magic = TINYNV_GSP_FW_WPR_META_MAGIC;
  m.revision = TINYNV_GSP_FW_WPR_META_REVISION;
  m.sysmemAddrOfRadix3Elf = gsp->radix3.addrs[0];
  m.sizeOfRadix3Elf = gsp->gsp_image_size;
  m.sysmemAddrOfBootloader = gsp->bootloader.addrs[0];
  m.sizeOfBootloader = gsp->bootloader_size;
  m.sysmemAddrOfSignature = gsp->signature.addrs[0];
  m.sizeOfSignature = 0x1000;
  m.bootloaderCodeOffset = gsp->bl_desc.monitorCodeOffset;
  m.bootloaderDataOffset = gsp->bl_desc.monitorDataOffset;
  m.bootloaderManifestOffset = gsp->bl_desc.manifestOffset;

  if (!g->dev.fmc_boot) return tinynv_fail("the vbios boot path does not build this structure the same way");
  // on the chain-of-trust path the firmware places its own carveout, so these are sizes and not offsets. They are
  // named in fw_layout.h, which is also what sizes the reservation the memory manager holds back for them: change one
  // there and the static assert beside them says whether the reservation still covers it.
  m.vgaWorkspaceSize = TINYNV_FW_VGA_WORKSPACE;
  m.pmuReservedSize = TINYNV_FW_PMU_RESERVED;
  m.nonWprHeapSize = TINYNV_FW_NONWPR_HEAP;
  m.gspFwHeapSize = TINYNV_FW_HEAP_SIZE;
  m.frtsSize = TINYNV_FW_FRTS_SIZE;

  if (tinynv_alloc_boot_mem(&g->mm, sizeof(m), &m, -1, &gsp->wpr_meta)) return -1;
  gsp->wpr_meta_sysmem = gsp->wpr_meta.addrs[0];
  return 0;
}

// The bus address GSP-RM is told, packed the way NVIDIA packs it. A name that is not a PCI address gives zero, which
// covers the socket backend over thunderbolt and the replay backend, and is what the oracle reports for them.
static uint64_t bdf_as_int(const char *name) {
  unsigned dom, bus, dev, fn;
  if (!name || sscanf(name, "%4x:%2x:%2x.%1x", &dom, &bus, &dev, &fn) != 4) return 0;
  return ((uint64_t)bus << 8) | ((uint64_t)dev << 3) | fn; // the domain is not part of what gsp-rm is told
}

// What GSP-RM is told about the machine around it: where its own windows are, what it is on the bus, and how much of the
// address space the driver will hand out.
static int rpc_set_gsp_system_info(tinynv_gpu_t *g) {
  tinynv_pci_t *pci = g->dev.pci;
  tinynv_gsp_system_info_t info;
  memset(&info, 0, sizeof(info));
  if (pci->bar_info(pci, 0, &info.gpuPhysAddr, NULL) || pci->bar_info(pci, 1, &info.gpuPhysFbAddr, NULL) ||
      pci->bar_info(pci, 3, &info.gpuPhysInstAddr, NULL))
    return -1;
  // where the config space is mirrored into the register window; the chain-of-trust chips moved it
  info.pciConfigMirrorBase = g->dev.fmc_boot ? 0x92000 : 0x88000;
  info.pciConfigMirrorSize = 0x1000;
  info.nvDomainBusDeviceFunc = bdf_as_int(pci->name);
  info.bIsPassthru = 1;
  info.PCIDeviceID = pci->cfg_read(pci, 0x00, 4);    // vendor and device together
  info.PCISubDeviceID = pci->cfg_read(pci, 0x2c, 4); // subsystem vendor and device together
  info.PCIRevisionID = pci->cfg_read(pci, 0x08, 1);
  info.maxUserVa = 0x7ffffffff000ull;
  return rpc_send(g, TINYNV_MSG_FUNCTION_GSP_SET_SYSTEM_INFO, &info, sizeof(info));
}

// The handful of driver settings GSP-RM reads out of a packed table: a header, then one entry per key, then the key names
// laid end to end with each entry pointing at its own name.
static int rpc_set_registry_table(tinynv_gpu_t *g) {
  static const struct { const char *name; uint32_t value; } TABLE[] = {
    {"RMForcePcieConfigSave", 1},
    {"RMSecBusResetEnable", 1},
  };
  const size_t n = sizeof(TABLE) / sizeof(*TABLE);
  size_t hdr_size = sizeof(tinynv_registry_table_t), entries_size = sizeof(tinynv_registry_entry_t) * n, names_size = 0;
  for (size_t i = 0; i < n; i++) names_size += strlen(TABLE[i].name) + 1;

  size_t total = hdr_size + entries_size + names_size;
  uint8_t *buf = calloc(1, total);
  if (!buf) return tinynv_fail("out of memory for the registry table");
  tinynv_registry_table_t *h = (tinynv_registry_table_t *)buf;
  h->size = (uint32_t)total;
  h->numEntries = (uint32_t)n;
  tinynv_registry_entry_t *e = (tinynv_registry_entry_t *)(buf + hdr_size);
  size_t at = 0;
  for (size_t i = 0; i < n; i++) {
    size_t len = strlen(TABLE[i].name) + 1;
    e[i].nameOffset = (uint32_t)(hdr_size + entries_size + at);
    e[i].type = TINYNV_REGISTRY_TYPE_DWORD;
    e[i].data = TABLE[i].value;
    e[i].length = 4;
    memcpy(buf + hdr_size + entries_size + at, TABLE[i].name, len);
    at += len;
  }
  int rc = rpc_send(g, TINYNV_MSG_FUNCTION_SET_REGISTRY, buf, total);
  free(buf);
  return rc;
}

int tinynv_gsp_init_hw(tinynv_gpu_t *g) {
  tinynv_gsp_t *gsp = &g->gsp;

  // the status queue's header does not exist until GSP-RM writes it, so attaching to it is the wait for the firmware to
  // start. on this card that is several seconds.
  if (rpcq_attach(g, &gsp->stat_q, gsp->cmd_q_off + gsp->queue_size, "status")) return -1;
  // each side keeps its read position in the other's header, so the queues have to be told about each other
  tinynv_msgq_tx_header_t cmd_tx;
  nv_rd_block(&gsp->queues.view, gsp->cmd_q_off, &cmd_tx, sizeof(cmd_tx));
  gsp->stat_q.rx_off = gsp->cmd_q_off + cmd_tx.rxHdrOff;
  gsp->cmd_q.rx_off = gsp->stat_q.base + gsp->stat_q.tx.rxHdrOff;

  if (rpc_wait(g, TINYNV_MSG_EVENT_GSP_INIT_DONE, 60.0, "its start-up notice", NULL, NULL)) return -1;
  if (gsp->err_state) return tinynv_fail("gsp-rm started but reported an error on the way up");

  // with the firmware up, the two windows onto instance memory are retargeted at it
  tinynv_wr32(&g->dev, NV_PBUS_BAR1_BLOCK, 0);
  if (g->dev.fmc_boot) tinynv_wr32(&g->dev, NV_VIRTUAL_FUNCTION_PRIV_FUNC_BAR1_BLOCK_LOW_ADDR, 0);
  return 0;
}

int tinynv_gsp_init_sw(tinynv_gpu_t *g) {
  g->gsp.gpu = g;
  if (init_rm_args(g)) return -1;
  if (init_libos_args(g)) return -1;
  if (init_wpr_meta(g)) return -1;
  // prefilled before the firmware exists: it reads both out of the ring as soon as it starts
  if (rpc_set_gsp_system_info(g)) return -1;
  return rpc_set_registry_table(g);
}

void tinynv_gsp_fini(tinynv_gpu_t *g) {
  // The record itself, which is host memory of ours and nothing the card knows about. Freeing the objects it describes
  // is a separate decision and deliberately not taken here yet.
  free(g->gsp.objs);
  g->gsp.objs = NULL;
  g->gsp.nobjs = g->gsp.objs_cap = 0;
  tinynv_gsp_t *gsp = &g->gsp;
  tinynv_bootmem_t *all[] = {&gsp->queues, &gsp->rm_args, &gsp->logbuf, &gsp->libos_args,
                             &gsp->radix3, &gsp->signature, &gsp->bootloader, &gsp->wpr_meta};
  for (size_t i = 0; i < sizeof(all) / sizeof(*all); i++) if (all[i]->size) tinynv_free_boot_mem(&g->mm, all[i]);
  tinynv_fw_free(&gsp->gsp_fw);
  tinynv_fw_free(&gsp->bl_fw);
}

// --- the mapping record ---------------------------------------------------------------------------------------
//
// Separated from the code that programs page tables for the same reason tinynv_rm_free_check is separated from the
// code that sends a free: the refusals are the point, and they can be driven with no card. See gsp.h.

tinynv_rm_unmap_check_t tinynv_rm_unmap_check(const tinynv_rm_map_t *maps, int n, uint32_t client, uint64_t root,
                                              uint64_t va, uint64_t length, int *at) {
  for (int i = 0; i < n; i++) {
    if (maps[i].client != client || maps[i].root != root) continue;
    if (maps[i].va != va) continue;
    // Base matches. Length must too - see TINYNV_RM_UNMAP_PARTIAL.
    if (maps[i].length != length) return TINYNV_RM_UNMAP_PARTIAL;
    if (at) *at = i;
    return TINYNV_RM_UNMAP_OK;
  }
  // A base that falls INSIDE a live mapping is the partial case rather than the unknown one, and the distinction is
  // worth making: "you did not map this" and "you mapped it and are unmapping half of it" are different mistakes and
  // only the second is a guest getting the arithmetic right and the policy wrong.
  for (int i = 0; i < n; i++)
    if (maps[i].client == client && maps[i].root == root && va > maps[i].va && va < maps[i].va + maps[i].length)
      return TINYNV_RM_UNMAP_PARTIAL;
  return TINYNV_RM_UNMAP_UNKNOWN;
}

int tinynv_rm_object_is_mapped(const tinynv_rm_map_t *maps, int n, uint32_t client, uint32_t handle) {
  for (int i = 0; i < n; i++)
    if (maps[i].client == client && maps[i].handle == handle) return 1;
  return 0;
}

// Guarantee 7's input. Unknown size means the params never reached us, so the flags did not either - and an unknown
// permission is refused exactly like an unknown bound rather than assumed permissive.
//
// This answers a NARROWER question than its name: both bits are aliased in nvos.h, so what it really reports is
// "carries a bit that means read-only, or on a class other than 0x0040 possibly sparse". The full reasoning, and why
// 0x08000000's ambiguity is benign while 0x04000000's is only bounded, is at TINYNV_ALLOC_FLAGS_USER_READ_ONLY in
// nv_structs.h. The outcome is the same under every reading - refuse - so the ambiguity costs a caller nothing except
// the ability to trust the REASON, and the reason is the part anyone debugging a refusal will act on.
int tinynv_rm_obj_writable(const tinynv_rm_obj_t *o) {
  if (!o || !o->size) return 0;
  return !(o->alloc_flags & (TINYNV_ALLOC_FLAGS_USER_READ_ONLY | TINYNV_ALLOC_FLAGS_DEVICE_READ_ONLY));
}

// --- the C4b mapping decision -------------------------------------------------------------------------------------
//
// Every guarantee in docs/driver/libtinynv-design.md §4g that can be decided from the records is decided here, and
// nothing that programs a page table is. The split is the same one the free path uses and for the same reason: the
// REFUSALS are the feature, and a feature that needs a card to exercise is a feature nobody exercises.
//
// Guarantees not decided here, and where they live instead. 3's alignment IS decided here as of 2026-09-15 - this
// comment previously said it was "arithmetic in tinynv_c4b_map", a function that does not exist and never did, and
// nothing anywhere checked it. The tree's own coverage bound is still not decided here; 5's teardown ordering is
// the free path consulting tinynv_rm_object_is_mapped; 6 is pt.c's invalidate-and-wait; 9 follows from 1 and 8 rather
// than being checked; 10's page-table budget is not implemented and is recorded as open; 12 is
// tinynv_rm_unmap_check.
tinynv_rm_map_check_t tinynv_rm_map_check(const tinynv_rm_obj_t *objs, int nobjs, const tinynv_rm_map_t *maps,
                                          int nmaps, uint64_t driver_root, const tinynv_rm_map_req_t *req) {
  if (!req) return TINYNV_RM_MAP_NO_OBJECT;

  // 2 first, because it is the one whose failure is silent and catastrophic: a guest mapping programmed into the
  // driver's tree WORKS, and gives a guest an address in the same space as the command ring. Checked before anything
  // about the request itself, so a malformed request into our tree is still refused for the right reason.
  if (req->root == driver_root || !req->root) return TINYNV_RM_MAP_NOT_GUEST_TREE;

  // 3, the part that is about the request rather than the tree. The wrap tests come before the bound tests they
  // protect: an overflowed end address passes a naive containment check, which is the whole reason the order matters.
  if (!req->length) return TINYNV_RM_MAP_BAD_GEOMETRY;
  if (req->va + req->length < req->va) return TINYNV_RM_MAP_BAD_GEOMETRY;
  if (req->offset + req->length < req->offset) return TINYNV_RM_MAP_BAD_GEOMETRY;

  // 3's other half, the alignment, WHICH WAS ENFORCED NOWHERE until 2026-09-15. The comment below this function used
  // to say it was "arithmetic in tinynv_c4b_map" - a function that does not exist and never did. Nothing else covered
  // it either: tinynv_mm_map_range takes vaddr and size and walks, with no alignment test anywhere on the way, and
  // the only rounding in pt.c is in tinynv_mm_valloc, which is OUR allocation path and not a guest's.
  //
  // THE PAGE SIZE IS OURS, NEVER THE GUEST'S - a guest that could choose the granularity could choose one that spans
  // outside its window. So this compares against the constant this driver programs and does not read anything from
  // the request that could change it. The offset is included because it sets the physical base of the mapping: an
  // unaligned offset misaligns every entry even when va and length are clean.
  if ((req->va | req->length | req->offset) & (TINYNV_MAP_PAGE - 1)) return TINYNV_RM_MAP_MISALIGNED;

  // 1. A handle under another client is NOT FOUND rather than refused - "refused" tells a guest the handle exists.
  const tinynv_rm_obj_t *o = NULL;
  for (int i = 0; i < nobjs; i++)
    if (objs[i].client == req->client && objs[i].handle == req->handle) { o = &objs[i]; break; }
  if (!o) return TINYNV_RM_MAP_NO_OBJECT;

  // 8. An allocation whose params never reached us has no bound, and a missing bound is refused rather than assumed
  // generous - the alternative is a guest naming bytes past the end of its own allocation with nothing able to say so.
  if (!o->size) return TINYNV_RM_MAP_SIZE_UNKNOWN;
  if (req->offset + req->length > o->size) return TINYNV_RM_MAP_OUT_OF_BOUNDS;

  // 7. The request's side is gpuMappingType; the allocation's is its NVOS32 read-only flags. Mapping type 1 is
  // read-write in the traffic we have; anything that is not explicitly the read-only type is treated as wanting
  // write, because the safe direction for an unrecognised value is the stricter one.
  //
  // Both sides of this comparison are weaker than they look, and in opposite ways. The request's side was WRONG for
  // an hour - TINYNV_RM_MAP_TYPE_READ_ONLY spelled 2 instead of 3, which inverted this test on the one value that
  // matters. The allocation's side is not wrong but is imprecise: the bits it reads are aliased onto SPARSE and
  // ALLOCATE_KERNEL_PRIVILEGED, so a refusal here means "carries a read-only bit", not "is read-only". See
  // nv_structs.h. Both were found by reading NVIDIA's headers rather than by testing, which is the standing lesson:
  // an offline suite can only disagree with itself.
  if (req->mapping_type != TINYNV_RM_MAP_TYPE_READ_ONLY && !tinynv_rm_obj_writable(o))
    return TINYNV_RM_MAP_WOULD_WRITE;

  // 4. Overlap in the same tree, which is not the same question as tinynv_mm_map_range's - that one asks what the
  // page tables say, this one asks what we recorded, and a disagreement between them is itself worth finding.
  for (int i = 0; i < nmaps; i++) {
    if (maps[i].root != req->root) continue;
    uint64_t lo = maps[i].va, hi = maps[i].va + maps[i].length;
    if (req->va < hi && lo < req->va + req->length) return TINYNV_RM_MAP_ALREADY_MAPPED;
  }
  return TINYNV_RM_MAP_OK;
}

// Does GSP-RM answer NV0041_CTRL_CMD_GET_SURFACE_PHYS_ATTR, and what does it say? Diagnostic; see
// test/test_hw_physattr.c for the whole reasoning. Allocates a class-0x0040 object, asks where one offset of it
// lives, frees it again. Nothing else in the driver depends on this working.
//
// It allocates its OWN object rather than using a guest's, which makes this a card question that needs no daemon, no
// guest and no torch. If the firmware answers for an object this driver allocated, it is the same control on the same
// class and the answer transfers. If it refuses, that is when the guest's path has to be brought in, to find out
// whether it refuses for everyone or only for us.
int tinynv_gsp_probe_phys_attr(tinynv_gpu_t *g, uint64_t size, uint64_t offset, uint64_t *paddr, uint64_t *contig,
                               uint32_t *aperture) {
  if (!g || !paddr || !contig || !aperture) return -1;
  tinynv_gsp_t *gsp = &g->gsp;
  // The recorded shape, read out of the trace by Session C rather than invented: type 0 (NVOS32_TYPE_IMAGE), flags
  // 0x1c101, attr 0x18000000, hVASpace 0 - a pure physical allocation with no address space - and ALIGNMENT EQUAL TO
  // SIZE. Every other field is zero in the recording and is left zero here. `owner` is the recording's client and
  // will differ for us, which is the one field that cannot be reproduced and does not need to be.
  tinynv_nv0040_alloc_t a;
  memset(&a, 0, sizeof(a));
  a.owner = TINYNV_RM_PRIV_ROOT;
  a.type = 0;
  a.flags = 0x1c101u;
  a.attr = 0x18000000u;
  a.size = size;
  a.alignment = size;
  uint32_t mem = 0;
  if (rm_alloc_as(g, TINYNV_RM_PRIV_ROOT, gsp->device, 0, TINYNV_CLASS_MEMORY_LOCAL_USER, &a, sizeof(a), &mem))
    return tinynv_fail("the class 0x0040 allocation this probe needs was refused, so nothing was learned about "
                       "0x410103: %s", tinynv_last_error());

  tinynv_nv0041_phys_attr_t q;
  memset(&q, 0, sizeof(q));
  q.memOffset = offset;
  int rc = rm_control_as(g, TINYNV_RM_PRIV_ROOT, mem, TINYNV_CTRL_CMD_GET_SURFACE_PHYS_ATTR, &q, sizeof(q));
  if (!rc) {
    *paddr = q.memOffset;
    *contig = q.contigSegmentSize;
    *aperture = q.memAperture;
  }
  // Freed either way. A probe that leaves an allocation behind on a refusal turns "the control does not work" into
  // "the control does not work and the heap is smaller now", and the second run would measure the first.
  tinynv_gsp_rm_free(g, TINYNV_RM_PRIV_ROOT, mem);
  return rc;
}

// Walk an allocation into physical ranges. See gsp.h for why the query is a function pointer.
int tinynv_c4b_ranges(tinynv_phys_query_fn ask, void *ctx, uint64_t offset, uint64_t length,
                      tinynv_paddr_range_t *out, int maxout, int *nout) {
  if (!ask || !out || !nout || !length || maxout <= 0)
    return tinynv_fail("a physical-range walk needs a query, somewhere to put the answer and a non-zero length");
  if (offset + length < offset) return tinynv_fail("the range to walk wraps: offset %#llx plus length %#llx",
                                                   (unsigned long long)offset, (unsigned long long)length);
  *nout = 0;
  uint64_t done = 0;
  while (done < length) {
    uint64_t paddr = 0, contig = 0;
    if (ask(ctx, offset + done, &paddr, &contig))
      return tinynv_fail("the card would not say where offset %#llx of this allocation is",
                         (unsigned long long)(offset + done));
    // Zero is the one answer that must be refused rather than retried: it consumes nothing, and asking again at the
    // same offset is a loop with no exit. A firmware that cannot answer has to say so by failing, not by saying nil.
    if (!contig) return tinynv_fail("the card reports 0 contiguous bytes at offset %#llx, which is not an answer and "
                                    "cannot be walked past", (unsigned long long)(offset + done));
    if (paddr & (TINYNV_MAP_PAGE - 1))
      return tinynv_fail("physical address %#llx for offset %#llx is not page aligned - entries store address >> 12, "
                         "so the low bits would be dropped silently and every page would be wrong",
                         (unsigned long long)paddr, (unsigned long long)(offset + done));
    uint64_t take = contig < length - done ? contig : length - done;
    // Only the tail may be short of a page, and it cannot be: length is page-aligned by guarantee 3 and every take
    // before it is either a whole page multiple or clipped to what remains. A non-multiple here means the card
    // answered with a granularity we do not program.
    if (take & (TINYNV_MAP_PAGE - 1))
      return tinynv_fail("a contiguous segment of %#llx bytes at offset %#llx is not a multiple of the page size "
                         "this driver programs", (unsigned long long)take, (unsigned long long)(offset + done));
    if (*nout >= maxout)
      return tinynv_fail("this allocation needs more than %d physical ranges to describe - refusing rather than "
                         "mapping the first %d, which would look like a firmware disagreement later", maxout, maxout);
    out[(*nout)++] = (tinynv_paddr_range_t){.paddr = paddr, .size = take};
    done += take;
  }
  return 0;
}

// Decide, program, record. See gsp.h for why `ranges` is a parameter rather than something this function obtains.
tinynv_rm_map_check_t tinynv_c4b_map(tinynv_mm_t *mm, const tinynv_rm_obj_t *objs, int nobjs, tinynv_rm_map_t *maps,
                                     int *nmaps, int maxmaps, uint64_t driver_root, const tinynv_rm_map_req_t *req,
                                     const tinynv_paddr_range_t *ranges, int nranges) {
  if (!mm || !maps || !nmaps || !req || (!ranges && nranges)) return TINYNV_RM_MAP_NO_OBJECT;

  // THE DECISION FIRST, AND NOTHING BEFORE IT. Every guarantee that can refuse gets to refuse before a single entry
  // is written, so a refused request cannot leave anything behind to tidy up.
  tinynv_rm_map_check_t d = tinynv_rm_map_check(objs, nobjs, maps, *nmaps, driver_root, req);
  if (d != TINYNV_RM_MAP_OK) return d;

  // The ranges are checked like anything else that arrives from outside, because they DO arrive from outside: GSP-RM
  // supplies them and this driver cannot verify them against anything it already holds. An unaligned physical base
  // is the dangerous one - tinynv_pt_set_entry writes paddr >> 12, so the low bits are dropped SILENTLY and every
  // entry would point twelve bits away from the memory the guest was granted.
  uint64_t total = 0;
  for (int i = 0; i < nranges; i++) {
    if ((ranges[i].paddr | ranges[i].size) & (TINYNV_MAP_PAGE - 1)) return TINYNV_RM_MAP_BAD_RANGES;
    if (!ranges[i].size) return TINYNV_RM_MAP_BAD_RANGES;
    if (total + ranges[i].size < total) return TINYNV_RM_MAP_BAD_RANGES;
    total += ranges[i].size;
  }
  if (!nranges || total != req->length) return TINYNV_RM_MAP_BAD_RANGES;

  // Room to RECORD it, checked before programming rather than after. A mapping we cannot record is one the free path
  // cannot find, and guarantee 5 exists precisely because an allocation going away under a live mapping leaves page
  // tables pointing at memory the allocator can hand to somebody else. Refusing is the only safe order.
  if (*nmaps >= maxmaps) return TINYNV_RM_MAP_NO_ROOM;

  // GUARANTEE 7'S GRANT HALF, and the line that makes the decision mean something. tinynv_rm_map_check has already
  // refused a write mapping on read-only memory; this is the other case - a read-only mapping it ALLOWED, which must
  // be programmed read-only rather than merely permitted. Until 2026-09-15 there was no encoding that could say it,
  // so this line would have been a comment describing something the hardware never heard.
  uint32_t flags = (req->mapping_type == TINYNV_RM_MAP_TYPE_READ_ONLY) ? TINYNV_PTE_READONLY : 0;

  if (tinynv_mm_map_range(mm, req->root, req->va, req->length, ranges, nranges, flags)) {
    // Programming can fail partway and leave entries behind. Tear the range back down so the caller's "refused"
    // means what it says: a failed map leaves NOTHING, not a partial mapping nothing has recorded.
    tinynv_mm_unmap_range(mm, req->root, req->va, req->length);
    return TINYNV_RM_MAP_BAD_RANGES;
  }

  maps[(*nmaps)++] = (tinynv_rm_map_t){
      .client = req->client, .handle = req->handle, .root = req->root, .va = req->va, .length = req->length};
  return TINYNV_RM_MAP_OK;
}

const char *tinynv_rm_map_why(tinynv_rm_map_check_t c) {
  switch (c) {
    case TINYNV_RM_MAP_OK: return "the mapping is allowed";
    case TINYNV_RM_MAP_NO_OBJECT: return "no such allocation under this client";
    case TINYNV_RM_MAP_NOT_GUEST_TREE:
      return "that page-table tree is the driver's own, and a guest mapping in it would share an address space with "
             "the command ring";
    case TINYNV_RM_MAP_BAD_GEOMETRY: return "zero length, or an address or offset that wraps";
    case TINYNV_RM_MAP_ALREADY_MAPPED: return "something is already mapped over that range in that tree";
    case TINYNV_RM_MAP_WOULD_WRITE:
      // Deliberately not "an allocation that is read-only". NVIDIA shares these bits with SPARSE and
      // ALLOCATE_KERNEL_PRIVILEGED, so this sentence is what stops someone reading a correct strict refusal as a bug,
      // or a puzzling one as correct. nv_structs.h has which is which.
      return "a writable mapping was asked for an allocation carrying a read-only flag bit - a bit NVIDIA also spells "
             "SPARSE and ALLOCATE_KERNEL_PRIVILEGED, so see nv_structs.h before treating this refusal as settled";
    case TINYNV_RM_MAP_OUT_OF_BOUNDS: return "offset plus length runs past the end of the allocation";
    case TINYNV_RM_MAP_SIZE_UNKNOWN:
      return "this allocation's size never reached the driver, so there is no bound to check the offset against";
    case TINYNV_RM_MAP_BAD_RANGES:
      return "the physical ranges behind this allocation do not describe the request - wrong total, or not page "
             "aligned. This is NOT the guest's request being wrong: those come from GSP-RM, so it is a disagreement "
             "between this driver and the firmware and should be looked for there";
    case TINYNV_RM_MAP_NO_ROOM:
      return "the mapping record is full, and a mapping that cannot be recorded cannot be torn down when its "
             "allocation goes away - so it is refused rather than programmed and forgotten";
    case TINYNV_RM_MAP_MISALIGNED:
      return "the address, length or offset is not a multiple of the page size this driver programs - the granularity "
             "is ours and never the guest's, so this is refused rather than rounded to fit";
  }
  return "refused";
}
