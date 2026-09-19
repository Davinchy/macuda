// Building command buffers, and handing them to the GPU.
//
// Work reaches the engine as a stream of methods: a header naming a subchannel, a method number and how many values
// follow, then the values. A batch is wrapped in a wait on a semaphore and a release of it, so the engine runs the batch
// only after whatever it depends on, and says so when it is done.
//
// Handing one over is three writes in this order, and the order is the protocol rather than an implementation detail: a
// ring entry naming the buffer and its length, the write pointer that publishes it, then the doorbell. Over the socket
// backend each is its own message delivered in order; one combined write would be a single copy on the far side with no
// ordering promise inside it, and the engine could see a published pointer to an entry that had not landed.
//
// The recorded boot confirms the shape: eight ring entries carrying lengths of 22, 16, 21, 32 and so on, each followed
// by a write pointer and a doorbell carrying this driver's own computed token. The contents of those buffers were not
// recorded - they live in host memory the client maps directly - so they are checked instead against what the oracle
// builds for the same inputs, byte for byte, in test_submit.
#include "submit.h"
#include "gpu.h"
#include "internal.h"
#include "nv_regs.h"
#include <stdio.h>
#include <string.h>

#define NV_METHOD(subc, mthd, n) ((2u << 28) | ((uint32_t)(n) << 16) | ((uint32_t)(subc) << 13) | ((uint32_t)(mthd) >> 2))
// The same header with SEC_OP = NON_INC_METHOD (3, at 31:29) instead of INC_METHOD (1). An incrementing header walks
// the method address forward one dword at a time, which is right for a run of distinct registers and wrong for a FIFO
// at a single address: inline data sent incrementing would scatter its payload across whatever methods follow
// LOAD_INLINE_DATA rather than being written to memory.
#define NV_METHOD_NI(subc, mthd, n) ((6u << 28) | ((uint32_t)(n) << 16) | ((uint32_t)(subc) << 13) | ((uint32_t)(mthd) >> 2))
#define NV_FLAG(field, value) ((uint32_t)(value) << field##_LO)

static int push(tinynv_cmdbuf_t *c, uint32_t w) {
  if (c->n >= c->cap) return tinynv_fail("the command buffer holds %u dwords and the batch needs more", c->cap);
  c->words[c->n++] = w;
  return 0;
}

int tinynv_cmd_method(tinynv_cmdbuf_t *c, uint32_t subc, uint32_t mthd, const uint32_t *vals, uint32_t n) {
  if (push(c, NV_METHOD(subc, mthd, n))) return -1;
  for (uint32_t i = 0; i < n; i++) if (push(c, vals[i])) return -1;
  return 0;
}

// Every value to the SAME method address, for a register that is a queue rather than a location.
static int method_ni(tinynv_cmdbuf_t *c, uint32_t subc, uint32_t mthd, const uint32_t *vals, uint32_t n) {
  if (push(c, NV_METHOD_NI(subc, mthd, n))) return -1;
  for (uint32_t i = 0; i < n; i++) if (push(c, vals[i])) return -1;
  return 0;
}

// A small host-to-device copy carried in the pushbuffer itself.
//
// The seven tiny copies at the start of every decode token - 20480, 4, 16, 8, 8, 512 and 4 bytes - each cost a
// copy-engine batch, a doorbell and a wait, which Session A timed at ~50 us apiece including their two
// synchronisations: ~370 us a token, 2.3% of a dense token and 4.8% of a mixture's, to move 21 KB. This writes the
// bytes as methods in a batch that is going out anyway, so the copy engine is not involved at all and there is no
// second queue to wait on.
//
// It reaches memory the processor cannot: the BAR window is 256 MB against 32 GB of video memory, so writing small
// copies through it would work for our own regions and fail for a caller's tensors. The compute engine has no such
// limit - it writes wherever the page tables say.
//
// The four registers are consecutive (0x180, 0x184, 0x188, 0x18c), so one incrementing header sets all of them.
// LOAD_INLINE_DATA is not consecutive with anything: it is a FIFO, and it needs the non-incrementing header above.
int tinynv_cmd_inline_upload(tinynv_cmdbuf_t *c, uint64_t dst, const void *src, uint32_t n) {
  if (!n || (n & 3u)) return tinynv_fail("an inline upload is a whole number of dwords, not %u bytes", n);
  if (dst & 3u) return tinynv_fail("an inline upload writes dwords, so %#llx must be 4-byte aligned",
                                   (unsigned long long)dst);
  uint32_t set[4] = {n, 1, (uint32_t)(dst >> 32), (uint32_t)dst};   // LINE_LENGTH_IN, LINE_COUNT, OFFSET_OUT_UPPER/LOWER
  if (tinynv_cmd_method(c, 1, NVC6C0_LINE_LENGTH_IN, set, 4)) return -1;
  // FLUSH_ONLY, not FLUSH_DISABLE. This moved from an instrument to the data path the moment anything routed a real
  // copy through it, and the failure mode of an unflushed write is a kernel reading stale memory - which shows up as
  // a wrong answer somewhere else entirely, with nothing pointing back here. The flush costs a little on a path that
  // exists to be cheap, and a cheap wrong answer is not the trade.
  uint32_t go = NV_FLAG(NVC6C0_LAUNCH_DMA_DST_MEMORY_LAYOUT, NVC6C0_LAUNCH_DMA_DST_MEMORY_LAYOUT_PITCH) |
                NV_FLAG(NVC6C0_LAUNCH_DMA_COMPLETION_TYPE, NVC6C0_LAUNCH_DMA_COMPLETION_TYPE_FLUSH_ONLY) |
                NV_FLAG(NVC6C0_LAUNCH_DMA_REDUCTION_ENABLE, NVC6C0_LAUNCH_DMA_REDUCTION_ENABLE_FALSE) |
                NV_FLAG(NVC6C0_LAUNCH_DMA_SYSMEMBAR_DISABLE, NVC6C0_LAUNCH_DMA_SYSMEMBAR_DISABLE_FALSE);
  if (tinynv_cmd_method(c, 1, NVC6C0_LAUNCH_DMA, &go, 1)) return -1;
  uint32_t words[TINYNV_INLINE_MAX / 4];
  if (n > TINYNV_INLINE_MAX) return tinynv_fail("%u bytes is past what this driver will put in a pushbuffer", n);
  memcpy(words, src, n);          // the caller's buffer need not be aligned for a dword read; ours is
  return method_ni(c, 1, NVC6C0_LOAD_INLINE_DATA, words, n / 4);
}

// A semaphore operation: the address low half first, then the value low half first, then what to do with it.
static int semaphore(tinynv_cmdbuf_t *c, uint64_t addr, uint64_t value, uint32_t flags) {
  uint32_t v[5] = {(uint32_t)addr, (uint32_t)(addr >> 32), (uint32_t)value, (uint32_t)(value >> 32), flags};
  return tinynv_cmd_method(c, 0, NVC56F_SEM_ADDR_LO, v, 5);
}

int tinynv_cmd_wait(tinynv_cmdbuf_t *c, uint64_t addr, uint64_t value) {
  // "acquire when circularly greater or equal": the comparison wraps, so a counter that has gone round still satisfies
  return semaphore(c, addr, value, NV_FLAG(NVC56F_SEM_EXECUTE_PAYLOAD_SIZE, NVC56F_SEM_EXECUTE_PAYLOAD_SIZE_64BIT) |
                                   NV_FLAG(NVC56F_SEM_EXECUTE_OPERATION, NVC56F_SEM_EXECUTE_OPERATION_ACQ_CIRC_GEQ));
}

// The gpu's clock, written when the front end REACHES this method rather than when the engine drains. That difference
// is the whole reason this exists separately from the release below: a release-with-timestamp always carries
// RELEASE_WFI, so it times the END of a batch however it is placed, and using one at the head of a batch to ask "when
// did this start" silently answers "when did it finish" instead. Which is what the refill instrument did until
// 2026-09-15, making its idle figure the finish-to-finish period - the gap plus a whole batch of work.
int tinynv_cmd_timestamp(tinynv_cmdbuf_t *c, uint64_t addr, uint64_t value) {
  return semaphore(c, addr, value, NV_FLAG(NVC56F_SEM_EXECUTE_PAYLOAD_SIZE, NVC56F_SEM_EXECUTE_PAYLOAD_SIZE_64BIT) |
                                   NV_FLAG(NVC56F_SEM_EXECUTE_OPERATION, NVC56F_SEM_EXECUTE_OPERATION_RELEASE) |
                                   NV_FLAG(NVC56F_SEM_EXECUTE_RELEASE_TIMESTAMP,
                                           NVC56F_SEM_EXECUTE_RELEASE_TIMESTAMP_EN));
}

int tinynv_cmd_release_ts(tinynv_cmdbuf_t *c, uint64_t addr, uint64_t value, int timestamp) {
  // released only once the engine is idle, so the value means the work is done rather than merely issued. asking for a
  // timestamp writes the gpu's own clock alongside the payload, which is how a queue times the work it ran.
  if (semaphore(c, addr, value, NV_FLAG(NVC56F_SEM_EXECUTE_PAYLOAD_SIZE, NVC56F_SEM_EXECUTE_PAYLOAD_SIZE_64BIT) |
                                NV_FLAG(NVC56F_SEM_EXECUTE_OPERATION, NVC56F_SEM_EXECUTE_OPERATION_RELEASE) |
                                NV_FLAG(NVC56F_SEM_EXECUTE_RELEASE_WFI, NVC56F_SEM_EXECUTE_RELEASE_WFI_EN) |
                                NV_FLAG(NVC56F_SEM_EXECUTE_RELEASE_TIMESTAMP,
                                        timestamp ? NVC56F_SEM_EXECUTE_RELEASE_TIMESTAMP_EN
                                                  : NVC56F_SEM_EXECUTE_RELEASE_TIMESTAMP_DIS)))
    return -1;
  // Restored after replay refused it. The recorded stream carries release_ts with no interrupt after it - that is the
  // oracle's convention for a stamp taken inside a batch, and rebuilding those buffers byte for byte is a check worth
  // more than the tidiness of un-overloading this argument. A batch-ending release that also wants a clock is
  // tinynv_cmd_release_clocked below, which is profile-only and appears in no recording.
  if (timestamp) return 0;
  uint32_t zero = 0;
  return tinynv_cmd_method(c, 0, NVC56F_NON_STALL_INTERRUPT, &zero, 1);
}

int tinynv_cmd_release(tinynv_cmdbuf_t *c, uint64_t addr, uint64_t value) {
  return tinynv_cmd_release_ts(c, addr, value, 0);
}

// Ends a batch the way tinynv_cmd_release does - waits for idle, announces itself with the interrupt - and leaves the
// gpu's clock at +8 of the report. Used only under the profile, so it is in no recording and replay never sees it.
int tinynv_cmd_release_clocked(tinynv_cmdbuf_t *c, uint64_t addr, uint64_t value) {
  if (tinynv_cmd_release_ts(c, addr, value, 1)) return -1;
  uint32_t zero = 0;
  return tinynv_cmd_method(c, 0, NVC56F_NON_STALL_INTERRUPT, &zero, 1);
}

// The copy engine's own release. The host semaphore methods above work on any channel, and on the compute channel their
// RELEASE_WFI genuinely means "the engine has finished". On a copy channel it does not mean the same thing: the transfer
// is handed to the copy engine, and a host-method release can retire without the engine having drained - so the payload
// lands, whatever is waiting on it starts, and the copy is still writing. Synchronously that is invisible, because the
// host's round trip to the next batch is far longer than the copy; asynchronously it is a kernel reading a buffer that
// is still being filled.
//
// So a copy is completed the way the oracle completes one: a semaphore address set on the engine, then a transfer with
// nothing to transfer whose only effect is the release. FLUSH_ENABLE is the load-bearing half - it is what makes the
// engine's writes visible before the payload lands, and it is the part a host-method release cannot express.
//
// The payload is one word, so 32 bits, where the compute path releases 64. Both write the same timeline, which is sound
// only while the counter stays inside 32 bits: a one word release leaves the high half alone, and the high half is zero
// because every 64 bit release below 2**32 writes it zero. exec.c refuses to go past that rather than wrap quietly.
// The copy engine's clock, into an address of its own, leaving the timeline release above untouched.
//
// Separate rather than promoting that release to four words, deliberately: the timeline release is what every wait in
// this driver depends on, its payload is read back as 64 bits, and a four word release writes its payload differently.
// Changing it to carry a clock would put the whole submission path's correctness behind a detail of a report format,
// under a profile flag, to answer a question about timing. This costs one more semaphore launch on a copy batch and
// risks nothing.
int tinynv_cmd_copy_timestamp(tinynv_cmdbuf_t *c, uint64_t addr, uint64_t value) {
  uint32_t v[3] = {(uint32_t)(addr >> 32), (uint32_t)addr, (uint32_t)value};
  if (tinynv_cmd_method(c, 4, NVC6B5_SET_SEMAPHORE_A, v, 3)) return -1;
  uint32_t go = NV_FLAG(NVC6B5_LAUNCH_DMA_FLUSH_ENABLE, NVC6B5_LAUNCH_DMA_FLUSH_ENABLE_TRUE) |
                NV_FLAG(NVC6B5_LAUNCH_DMA_SEMAPHORE_TYPE, NVC6B5_LAUNCH_DMA_SEMAPHORE_TYPE_RELEASE_FOUR_WORD_SEMAPHORE);
  return tinynv_cmd_method(c, 4, NVC6B5_LAUNCH_DMA, &go, 1);
}

int tinynv_cmd_copy_release(tinynv_cmdbuf_t *c, uint64_t addr, uint64_t value) {
  uint32_t v[3] = {(uint32_t)(addr >> 32), (uint32_t)addr, (uint32_t)value};   // high half first, like the transfer methods
  if (tinynv_cmd_method(c, 4, NVC6B5_SET_SEMAPHORE_A, v, 3)) return -1;
  uint32_t go = NV_FLAG(NVC6B5_LAUNCH_DMA_FLUSH_ENABLE, NVC6B5_LAUNCH_DMA_FLUSH_ENABLE_TRUE) |
                NV_FLAG(NVC6B5_LAUNCH_DMA_SEMAPHORE_TYPE, NVC6B5_LAUNCH_DMA_SEMAPHORE_TYPE_RELEASE_ONE_WORD_SEMAPHORE);
  return tinynv_cmd_method(c, 4, NVC6B5_LAUNCH_DMA, &go, 1);
}

// How much local memory the die needs, which is not a property of any kernel: every thread that could be resident at
// once needs its own, so the size is the per-thread figure multiplied out by the whole die and rounded at each step.
// Getting this wrong is not a wrong answer, it is threads reading each other's stack.
uint64_t tinynv_local_memory_size(const tinynv_die_t *die, uint32_t slm_per_thread, uint64_t *bytes_per_tpc) {
  uint64_t per_warp = ((uint64_t)slm_per_thread * 32 + 0x1ff) & ~0x1ffull;      // a warp is 32 threads
  uint64_t tpc = (per_warp * die->max_warps_per_sm * die->num_sm_per_tpc + 0x7fff) & ~0x7fffull;
  if (bytes_per_tpc) *bytes_per_tpc = tpc;
  return (tpc * die->num_tpc_per_gpc * die->num_gpcs + 0x1ffff) & ~0x1ffffull;
}

// Tells the compute engine where that memory is and how much of it a cluster may use.
int tinynv_cmd_local_memory(tinynv_cmdbuf_t *c, uint64_t addr, uint64_t bytes_per_tpc) {
  uint32_t at[2] = {(uint32_t)(addr >> 32), (uint32_t)addr};   // high half first, like the shader windows
  if (tinynv_cmd_method(c, 1, NVC6C0_SET_SHADER_LOCAL_MEMORY_A, at, 2)) return -1;
  // the third value is the throttle: 0xff is every cluster, which is what the oracle asks for and what the card took
  uint32_t sz[3] = {(uint32_t)(bytes_per_tpc >> 32), (uint32_t)bytes_per_tpc, 0xff};
  return tinynv_cmd_method(c, 1, NVC6C0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_A, sz, 3);
}

// Makes the shader caches forget what they hold, so a kernel reads what the last one wrote rather than what it cached.
int tinynv_cmd_memory_barrier(tinynv_cmdbuf_t *c) {
  uint32_t f = NV_FLAG(NVC6C0_INVALIDATE_SHADER_CACHES_NO_WFI_INSTRUCTION, NVC6C0_INVALIDATE_SHADER_CACHES_NO_WFI_INSTRUCTION_TRUE) |
               NV_FLAG(NVC6C0_INVALIDATE_SHADER_CACHES_NO_WFI_GLOBAL_DATA, NVC6C0_INVALIDATE_SHADER_CACHES_NO_WFI_GLOBAL_DATA_TRUE) |
               NV_FLAG(NVC6C0_INVALIDATE_SHADER_CACHES_NO_WFI_CONSTANT, NVC6C0_INVALIDATE_SHADER_CACHES_NO_WFI_CONSTANT_TRUE);
  return tinynv_cmd_method(c, 1, NVC6C0_INVALIDATE_SHADER_CACHES_NO_WFI, &f, 1);
}

// Which class a subchannel runs: the compute engine on one, the copy engine on another, named once per channel.
int tinynv_cmd_set_object(tinynv_cmdbuf_t *c, uint32_t subc, uint32_t cls) {
  return tinynv_cmd_method(c, subc, NVC6C0_SET_OBJECT, &cls, 1);
}

// Where a kernel's local or shared memory lives. These take the high half first, unlike the semaphore methods in the
// same command buffer, which is the sort of thing only a byte-for-byte comparison catches.
int tinynv_cmd_shader_window(tinynv_cmdbuf_t *c, int shared, uint64_t window) {
  uint32_t v[2] = {(uint32_t)(window >> 32), (uint32_t)window};
  return tinynv_cmd_method(c, 1, shared ? NVC6C0_SET_SHADER_SHARED_MEMORY_WINDOW_A : NVC6C0_SET_SHADER_LOCAL_MEMORY_WINDOW_A,
                           v, 2);
}

// The batch that binds a channel to what it runs, and the windows a kernel's local and shared memory live in.
int tinynv_cmd_compute_setup(tinynv_cmdbuf_t *c, uint64_t sem_addr, uint64_t sem_value, uint32_t compute_class,
                             uint64_t local_window, uint64_t shared_window) {
  if (tinynv_cmd_wait(c, sem_addr, sem_value)) return -1;
  if (tinynv_cmd_set_object(c, 1, compute_class)) return -1;
  if (tinynv_cmd_shader_window(c, 0, local_window)) return -1;
  if (tinynv_cmd_shader_window(c, 1, shared_window)) return -1;
  return tinynv_cmd_release(c, sem_addr, sem_value + 1);
}

int tinynv_cmd_copy_setup(tinynv_cmdbuf_t *c, uint64_t sem_addr, uint64_t sem_value, uint32_t copy_class) {
  if (tinynv_cmd_wait(c, sem_addr, sem_value)) return -1;
  if (tinynv_cmd_set_object(c, 4, copy_class)) return -1;
  return tinynv_cmd_release(c, sem_addr, sem_value + 1);
}

// The copy engine. It is a different engine with its own methods, and - the part that catches people - its own way of
// releasing a semaphore: not the host's semaphore methods the compute path uses, but a semaphore address set on the
// engine and then a transfer launched with nothing to transfer, whose only effect is the release.
//
// A transfer is three methods: where from and where to, how many bytes, and go. The addresses go high half first, which
// the method names say (OFFSET_IN_UPPER before OFFSET_IN_LOWER) and which is the opposite of the host semaphore methods
// in the same command buffer.
int tinynv_cmd_copy(tinynv_cmdbuf_t *c, uint64_t dst, uint64_t src, uint64_t bytes) {
  // A transfer's length field is 32 bits, so anything larger is issued as several. The oracle steps by 1<<31 rather than
  // the full 1<<32, and this matches it: the recorded traces are only evidence for what it does.
  const uint64_t step = 1ull << 31;
  for (uint64_t off = 0; off < bytes; off += step) {
    uint64_t s = src + off, d = dst + off;
    uint32_t ends[4] = {(uint32_t)(s >> 32), (uint32_t)s, (uint32_t)(d >> 32), (uint32_t)d};
    if (tinynv_cmd_method(c, 4, NVC6B5_OFFSET_IN_UPPER, ends, 4)) return -1;
    uint32_t len = (uint32_t)(bytes - off < step ? bytes - off : step);
    if (tinynv_cmd_method(c, 4, NVC6B5_LINE_LENGTH_IN, &len, 1)) return -1;
    uint32_t go = NV_FLAG(NVC6B5_LAUNCH_DMA_DATA_TRANSFER_TYPE, NVC6B5_LAUNCH_DMA_DATA_TRANSFER_TYPE_NON_PIPELINED) |
                  NV_FLAG(NVC6B5_LAUNCH_DMA_SRC_MEMORY_LAYOUT, NVC6B5_LAUNCH_DMA_SRC_MEMORY_LAYOUT_PITCH) |
                  NV_FLAG(NVC6B5_LAUNCH_DMA_DST_MEMORY_LAYOUT, NVC6B5_LAUNCH_DMA_DST_MEMORY_LAYOUT_PITCH);
    if (tinynv_cmd_method(c, 4, NVC6B5_LAUNCH_DMA, &go, 1)) return -1;
  }
  return 0;
}

// A release from the copy engine: the address and payload, then a launch that carries only the release. One word writes
// the payload alone; four words write a timestamp after it, which is how the queue times itself.
static int dma_semaphore(tinynv_cmdbuf_t *c, uint64_t addr, uint32_t value, uint32_t words) {
  uint32_t v[3] = {(uint32_t)(addr >> 32), (uint32_t)addr, value};
  if (tinynv_cmd_method(c, 4, NVC6B5_SET_SEMAPHORE_A, v, 3)) return -1;
  uint32_t go = NV_FLAG(NVC6B5_LAUNCH_DMA_FLUSH_ENABLE, NVC6B5_LAUNCH_DMA_FLUSH_ENABLE_TRUE) |
                NV_FLAG(NVC6B5_LAUNCH_DMA_SEMAPHORE_TYPE, words);
  return tinynv_cmd_method(c, 4, NVC6B5_LAUNCH_DMA, &go, 1);
}

int tinynv_cmd_dma_signal(tinynv_cmdbuf_t *c, uint64_t addr, uint32_t value) {
  return dma_semaphore(c, addr, value, NVC6B5_LAUNCH_DMA_SEMAPHORE_TYPE_RELEASE_ONE_WORD_SEMAPHORE);
}
int tinynv_cmd_dma_timestamp(tinynv_cmdbuf_t *c, uint64_t addr) {
  return dma_semaphore(c, addr, 0, NVC6B5_LAUNCH_DMA_SEMAPHORE_TYPE_RELEASE_FOUR_WORD_SEMAPHORE);
}

// A launch, which is two methods and nothing else: here is the descriptor, and go. Everything about the kernel is in
// the descriptor - see qmd.c - so this is the same four dwords whatever is being run.
int tinynv_cmd_launch(tinynv_cmdbuf_t *c, uint64_t qmd_addr) {
  uint32_t addr = (uint32_t)(qmd_addr >> 8);
  if (tinynv_cmd_method(c, 1, NVC6C0_SEND_PCAS_A, &addr, 1)) return -1;
  uint32_t action = NVC6C0_SEND_SIGNALING_PCAS2_B_PCAS_ACTION_PREFETCH_SCHEDULE;
  return tinynv_cmd_method(c, 1, NVC6C0_SEND_SIGNALING_PCAS2_B, &action, 1);
}

// What the ring is handed: where the buffer is, how long it is, and a bit the engine requires set.
uint64_t tinynv_ring_entry(uint64_t cmdbuf_va, uint32_t dwords) {
  return cmdbuf_va | ((uint64_t)dwords << 42) | (1ull << 41);
}

// Where the ring actually stands: what the driver has published, and what the engine has consumed. Worth reading only
// when something has gone wrong, and then worth a great deal - "the work was never picked up" has three quite different
// causes and these two numbers separate them. If put has not moved, the submission never happened. If put has moved and
// get is behind it, the engine has been handed the work and is not taking it, which means its front end is stuck rather
// than its engine busy. If get has caught put, everything was consumed and the work ran without saying so, which is a
// different bug entirely and not one in this file.
void tinynv_queue_state(tinynv_gpu_t *g, tinynv_queue_t *q, uint32_t *get, uint32_t *put, uint64_t *published) {
  uint64_t base = g->gsp.fifo_mem.ranges[0].paddr;
  *get = q->entries ? nv_rd32(&g->dev.vram, base + q->gpput_off - (TINYNV_GPFIFO_GPPUT_OFF - TINYNV_GPFIFO_GPGET_OFF)) : 0;
  *put = q->entries ? nv_rd32(&g->dev.vram, base + q->gpput_off) : 0;
  *published = q->put;
}

// Read back a ring entry the driver wrote. The engine not taking a batch has exactly three causes and this separates
// one of them: an entry that does not say what the driver thinks it says was never a valid piece of work, however
// correct the command buffer behind it.
uint64_t tinynv_ring_entry_read(tinynv_gpu_t *g, tinynv_queue_t *q, uint64_t slot) {
  if (!q->entries) return 0;
  uint64_t entry = 0;
  nv_rd_block(&g->dev.vram, g->gsp.fifo_mem.ranges[0].paddr + (q->ring_va - g->gsp.fifo_mem.va) + slot * 8, &entry, 8);
  return entry;
}

int tinynv_submit_stage(tinynv_gpu_t *g, tinynv_queue_t *q, uint64_t cmdbuf_va, uint32_t dwords) {
  if (!q->entries) return tinynv_fail("this channel has no ring");
  if (g->nstaged == (int)(sizeof g->staged / sizeof g->staged[0]))
    return tinynv_fail("more batches staged than there is room to announce (%d)", g->nstaged);
  uint64_t entry = tinynv_ring_entry(cmdbuf_va, dwords);

  // The rings live in the processor-visible reserve, which is video memory, so they are written through the window onto
  // it at their physical address. The allocation is contiguous by construction, which is why one base plus an offset is
  // enough to find any of it.
  uint64_t base = g->gsp.fifo_mem.ranges[0].paddr;

  uint32_t slot = (uint32_t)(q->put % q->entries);
  uint64_t ring_off = q->ring_va - g->gsp.fifo_mem.va;
  nv_wr_block(&g->dev.vram, base + ring_off + (uint64_t)slot * 8, &entry, 8);
  q->put++;
  nv_wr32(&g->dev.vram, base + q->gpput_off, (uint32_t)(q->put % q->entries));
  g->staged[g->nstaged++] = q;
  return 0;
}

int tinynv_submit_ring_send(tinynv_gpu_t *g) {
  if (!g->nstaged) return 0;
  // One in flight at a time: the socket carries one conversation, and an announcement owed must be paid first.
  if (g->ring_pending && tinynv_submit_ring_complete(g)) return -1;
  uint64_t base = g->gsp.fifo_mem.ranges[0].paddr;

  // Read a write pointer back before ringing anything, and note which bar each of these is on: the rings and the
  // pointers are written through the window onto video memory, and the doorbell is a register in the other one.
  // Ordering holds along a path, not between two of them - so a doorbell can arrive ahead of the pointer it is
  // announcing. The engine wakes, looks at a write pointer that has not moved yet, finds nothing to do and goes back to
  // sleep; the pointer lands a moment later with no doorbell behind it, and the batch sits in the ring, ready and never
  // dispatched.
  //
  // That is what a whole token of decode looked like. It was not visible while the only thing written before the
  // pointer was eight bytes of ring entry; putting the command arena in video memory put tens of kilobytes ahead of it
  // on the same path, and the race opened. Prefill never saw it because its batches are few and large and the engine is
  // rarely idle to be woken. This read cannot pass the writes ahead of it, so it costs one round trip per announcement
  // and makes the announcement mean what it says. nouveau reads its channel control page back for this reason.
  //
  // ONE read covers every batch staged, which is the whole point of the split. A non-posted read cannot pass any write
  // issued before it on this path, and that is true of all of their pointers and not only the one it names.
  //
  // And check the first few, because on this backend the read is a message to a server process rather than a load from
  // a mapped bar, and a message that never reaches the device orders nothing the device can see. If what comes back is
  // not what was just written, this read is not the fence it is being trusted as, and everything reasoned on top of it
  // is wrong. Said once, loudly, rather than assumed for the rest of the session.
  tinynv_queue_t *last = g->staged[g->nstaged - 1];
  g->ring_want = (uint32_t)(last->put % last->entries);
  nv_rd32_send(&g->dev.vram, base + last->gpput_off);
  for (int i = 0; i < g->nstaged; i++) g->ring_q[i] = g->staged[i];
  g->ring_n = g->nstaged;
  g->nstaged = 0;
  g->ring_pending = 1;
  return 0;
}

int tinynv_submit_ring_complete(tinynv_gpu_t *g) {
  if (!g->ring_pending) return 0;
  uint32_t want = g->ring_want, got = nv_rd32_recv(&g->dev.vram);
  if (got != want && g->submit_checked < 4) {
    g->submit_checked++;
    fprintf(stderr, "tinynv: the write pointer read back as %u where %u was just written - the read before the doorbell "
            "is not reaching the device, so it is not ordering anything the engine sees\n", got, want);
  } else if (got == want && !g->submit_checked) {
    g->submit_checked = 1;
    fprintf(stderr, "tinynv: the write pointer reads back as written, so the read before the doorbell reaches the "
            "device\n");
  }
  for (int i = 0; i < g->ring_n; i++) tinynv_wr32(&g->dev, TINYNV_DOORBELL, g->ring_q[i]->token);
  g->ring_n = 0;
  g->ring_pending = 0;
  return 0;
}

int tinynv_submit_ring(tinynv_gpu_t *g) {
  if (tinynv_submit_ring_send(g)) return -1;
  return tinynv_submit_ring_complete(g);
}

int tinynv_submit(tinynv_gpu_t *g, tinynv_queue_t *q, uint64_t cmdbuf_va, uint32_t dwords) {
  if (tinynv_submit_stage(g, q, cmdbuf_va, dwords)) return -1;
  return tinynv_submit_ring(g);
}
