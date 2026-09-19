// Command buffers, and the three writes that hand one to the GPU. See submit.c for why the order of those three is the
// protocol rather than a detail.
#ifndef TINYNV_SUBMIT_H
#define TINYNV_SUBMIT_H
#include <stdint.h>

// where the doorbell sits in the register window: the usermode region, plus the offset within it
#define TINYNV_DOORBELL (0xbb0000ull + 0x90)


typedef struct tinynv_gpu tinynv_gpu_t;
typedef struct tinynv_queue tinynv_queue_t;
// Handing a batch to the GPU, in two halves so that several can share one fence.
//
// The three writes are ring entry, write pointer, doorbell, and between the second and the third there has to be a
// read: the pointer goes to video memory and the doorbell to a register window, ordering holds along a path and not
// between two of them, so without it the doorbell can arrive ahead of the pointer it announces. On this backend that
// read is a message to another process, so it costs a full round trip AND drains whatever writes are queued ahead of
// it - which is why halving the number of reads is worth more than the nominal microseconds.
//
// One read fences every write issued before it on that path, whatever address it names. So staging N batches and then
// ringing once costs one read rather than N. tinynv_submit is the two halves back to back, for callers with one batch.
int tinynv_submit_stage(tinynv_gpu_t *g, tinynv_queue_t *q, uint64_t cmdbuf_va, uint32_t dwords);
int tinynv_submit_ring(tinynv_gpu_t *g);

// A batch under construction. The caller owns the storage, because a batch is built into memory the GPU will read.
typedef struct { uint32_t *words, n, cap; } tinynv_cmdbuf_t;

int tinynv_cmd_method(tinynv_cmdbuf_t *c, uint32_t subc, uint32_t mthd, const uint32_t *vals, uint32_t n);
int tinynv_cmd_wait(tinynv_cmdbuf_t *c, uint64_t addr, uint64_t value);
int tinynv_cmd_release(tinynv_cmdbuf_t *c, uint64_t addr, uint64_t value);
// The same thing on a copy channel, where the host method's RELEASE_WFI does not mean the copy engine has drained.
int tinynv_cmd_copy_release(tinynv_cmdbuf_t *c, uint64_t addr, uint64_t value);
// The same release, optionally writing the gpu's clock alongside the payload. A timestamped release is taken inside a
// batch and carries no interrupt; the one that ends a batch does.
int tinynv_cmd_release_ts(tinynv_cmdbuf_t *c, uint64_t addr, uint64_t value, int timestamp);
// The gpu clock at the point the front end reaches this method - no wait for idle, so it can time a batch's start.
int tinynv_cmd_timestamp(tinynv_cmdbuf_t *c, uint64_t addr, uint64_t value);
// A batch-ending release that also leaves the clock at +8 of its report. Profile-only; appears in no recording.
int tinynv_cmd_release_clocked(tinynv_cmdbuf_t *c, uint64_t addr, uint64_t value);
// The copy engine's clock into an address of its own, leaving its timeline release alone. Profile-only.
int tinynv_cmd_copy_timestamp(tinynv_cmdbuf_t *c, uint64_t addr, uint64_t value);
// The largest host-to-device copy ONE inline-upload call can carry: LOAD_INLINE_DATA's non-incrementing header holds
// a 13-bit dword count, so 8,191 dwords. Which copies actually take this path is a runtime choice, ex->inline_max
// (TINYNV_INLINE_MAX, 4,096 bytes by default - past a few KB the pushbuffer itself has to reach the card, so inlining
// stops paying for a lone copy); this constant only sizes the call and the buffer it copies through. Raised from
// 4,096 on 2026-09-19 so that the 8,192-byte upload every decode token still hands to the copy engine can be measured
// riding the pushbuffer instead (docs/driver/moe-next-steps.md, option 1).
#define TINYNV_INLINE_MAX 32764
// A small host-to-device copy written as methods in a command batch - no copy engine, no second queue, no doorbell of
// its own. `n` is a whole number of dwords, at most TINYNV_INLINE_MAX, and `dst` is 4-byte aligned.
int tinynv_cmd_inline_upload(tinynv_cmdbuf_t *c, uint64_t dst, const void *src, uint32_t n);
// How many dwords such an upload will add to a batch, so a caller can reserve the room.
#define TINYNV_INLINE_DWORDS(n) (8u + (uint32_t)(n) / 4u)   // 5 to set the four registers, 2 to launch, 1 header for the data
int tinynv_cmd_memory_barrier(tinynv_cmdbuf_t *c);

// The die's shape, which is what decides how much local memory has to exist for a kernel that uses any.
typedef struct { uint32_t num_gpcs, num_tpc_per_gpc, num_sm_per_tpc, max_warps_per_sm; } tinynv_die_t;
uint64_t tinynv_local_memory_size(const tinynv_die_t *die, uint32_t slm_per_thread, uint64_t *bytes_per_tpc);
int tinynv_cmd_local_memory(tinynv_cmdbuf_t *c, uint64_t addr, uint64_t bytes_per_tpc);
int tinynv_cmd_set_object(tinynv_cmdbuf_t *c, uint32_t subc, uint32_t cls);
int tinynv_cmd_shader_window(tinynv_cmdbuf_t *c, int shared, uint64_t window);
int tinynv_cmd_compute_setup(tinynv_cmdbuf_t *c, uint64_t sem_addr, uint64_t sem_value, uint32_t compute_class,
                             uint64_t local_window, uint64_t shared_window);
int tinynv_cmd_copy_setup(tinynv_cmdbuf_t *c, uint64_t sem_addr, uint64_t sem_value, uint32_t copy_class);
// The copy engine: a transfer, and its own kind of semaphore release. See submit.c - it is not the host semaphore the
// compute path uses, and its addresses go high half first.
int tinynv_cmd_copy(tinynv_cmdbuf_t *c, uint64_t dst, uint64_t src, uint64_t bytes);
int tinynv_cmd_dma_signal(tinynv_cmdbuf_t *c, uint64_t addr, uint32_t value);
int tinynv_cmd_dma_timestamp(tinynv_cmdbuf_t *c, uint64_t addr);

// The two methods that run a kernel. The kernel itself is described by the descriptor at qmd_addr; see qmd.h.
int tinynv_cmd_launch(tinynv_cmdbuf_t *c, uint64_t qmd_addr);

// The ring entry naming a command buffer: its address, its length in dwords, and the bit the engine requires.
uint64_t tinynv_ring_entry(uint64_t cmdbuf_va, uint32_t dwords);
int tinynv_submit(tinynv_gpu_t *g, tinynv_queue_t *q, uint64_t cmdbuf_va, uint32_t dwords);
// The ring's two ends, for when work is not being picked up: what the engine has consumed, what the driver published
// to the hardware, and how many entries the driver believes it has written.
void tinynv_queue_state(tinynv_gpu_t *g, tinynv_queue_t *q, uint32_t *get, uint32_t *put, uint64_t *published);
// What is actually sitting in a ring slot now, to be compared against what the driver meant to put there.
uint64_t tinynv_ring_entry_read(tinynv_gpu_t *g, tinynv_queue_t *q, uint64_t slot);

#endif
