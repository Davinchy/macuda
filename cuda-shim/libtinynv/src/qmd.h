// The kernel descriptor: what a launch actually is.
//
// A launch is not a command stream. The command stream carries two methods - here is a descriptor, go run it - and
// everything about the kernel lives in a 384 byte block in memory that those two methods point at. Grid and block
// dimensions, where the code is, where its constant buffers are, how many registers each thread wants, which caches to
// invalidate first, and which semaphore to release when it finishes: all of it is bitfields in that block, and several
// of them do not start on a byte boundary.
//
// The field positions come from the generator, never from reading a header, for the same reason the registers do. What
// is transcribed from the python oracle is the far smaller question of which fields a launch sets and to what, and that
// is what test_qmd compares byte for byte.
#ifndef TINYNV_QMD_H
#define TINYNV_QMD_H
#include <stdint.h>
#include "nv_regs.h"

#define TINYNV_QMD_CONSTBUFS 8
// The descriptor occupies 384 bytes but sits in a 256-aligned slot, and constant buffer 0 follows it in the same one.
#define TINYNV_QMD_SLOT_BYTES ((TINYNV_QMD_BYTES + 0xff) & ~0xff)
#define TINYNV_QMD_RELEASES 2
// Constant buffer 0 holds the driver's own parameters and then the kernel's. Blackwell's driver params reach index 223,
// so the buffer is at least this long whatever the kernel asks for.
// The most shared memory one block can have, measured on a GB202 rather than quoted: 100,352 bytes was written and read
// back word for word, and 131,200 - what ggml's flash attention asks for - was accepted into the descriptor and then
// never scheduled. A carveout the hardware cannot provide is not refused; the launch simply never completes.
//
// This is the single place that number lives. The descriptor's configuration ladder tops out here, the refusal above it
// quotes it, and tinynv_device_props reports it, so the driver and anything built on it cannot disagree about it. It is
// a per-block figure and not the memory a core has: consumer Blackwell has 128 KB of L1 and shared per core and makes
// this much of it available to a block. 228 KB is Hopper and does not belong here.
#define TINYNV_SMEM_MAX_PER_BLOCK (100 * 1024)

#define TINYNV_QMD_CBUF0_MIN_DWORDS 224
// Where the launch geometry lives inside those driver parameters, measured on a 5090 rather than read from a header:
// blockDim at word 216 and gridDim at word 220, each three words. See tinynv_qmd_cbuf0.
#define TINYNV_CBUF0_NTID 216
#define TINYNV_CBUF0_NCTAID 220

typedef struct {
  uint8_t b[TINYNV_QMD_BYTES];
  // Which bits each field has claimed. Nothing on the GPU reads this; it is here because a byte-for-byte comparison
  // against a reference built the same way cannot catch two fields being written over each other - the reference makes
  // the identical mistake and the bytes still agree. A claim on a bit that was already claimed is counted here instead.
  uint8_t claimed[TINYNV_QMD_BYTES];
  uint32_t overlaps;
} tinynv_qmd_t;

// a field, named by the two bit positions the generator emitted for it
#define TINYNV_QMD_F(name) TINYNV_QMD_##name##_LO, TINYNV_QMD_##name##_HI

int tinynv_qmd_set(tinynv_qmd_t *q, uint32_t lo, uint32_t hi, uint64_t v);
uint64_t tinynv_qmd_get(const tinynv_qmd_t *q, uint32_t lo, uint32_t hi);

// What the program itself fixes, the same for every launch of it. Sizes are in bytes; this shifts them where the field
// wants them shifted, so a caller never has to know which fields are named for a shift.
typedef struct {
  uint32_t regs;             // registers per thread, from the cubin
  uint32_t shmem;            // shared memory the kernel asks for, already rounded the way the oracle rounds it
  uint32_t slm_per_thread;   // local memory per thread, a property of the device rather than the kernel
  uint32_t prog_size;        // the kernel's code size, which bounds how much of it is worth prefetching
  uint32_t sass_version;     // tinynv_sass_version() of the device's sm_version
  uint32_t constbuf_size[TINYNV_QMD_CONSTBUFS]; // 0 for a bank the program does not use
  int constbuf_used[TINYNV_QMD_CONSTBUFS];
} tinynv_qmd_program_t;

// What changes per launch: the shape, and the addresses that are only known once memory is allocated.
typedef struct {
  uint32_t grid[3], block[3];
  uint64_t program_addr;
  uint64_t constbuf_addr[TINYNV_QMD_CONSTBUFS];
  int constbuf_set[TINYNV_QMD_CONSTBUFS];
} tinynv_qmd_launch_t;

uint32_t tinynv_sass_version(uint32_t sm_version);

int tinynv_qmd_program(tinynv_qmd_t *q, const tinynv_qmd_program_t *p);
int tinynv_qmd_launch(tinynv_qmd_t *q, const tinynv_qmd_launch_t *l);
// Releases a semaphore when the kernel finishes. There are two slots; this takes the first free one and returns which,
// or -1 when both are taken and the caller has to fall back on a release in the command stream.
int tinynv_qmd_release(tinynv_qmd_t *q, uint64_t addr, uint64_t payload, int timestamp);
// Check one link of a chain: that it releases `want_payload`, and that it schedules the descriptor at `next_va` - or
// nothing at all, when next_va is zero and this is the last link. Returns 0, or -1 with *why set to a reason.
// `releases` says which shape this chain is in: 1 for the shipping one, where every link reports its own completion,
// and 0 for a link in a tail-only chain, which must NOT release - a check that accepted either would prove nothing
// about the shape actually built, and the two have different failure modes.
int tinynv_qmd_link_check(const tinynv_qmd_t *q, int releases, uint64_t want_payload, uint64_t next_va,
                          const char **why);
// Chains one launch onto another: the previous descriptor starts this one itself, so the second launch costs no methods.
// Chain `prev` to the descriptor at `next_qmd_addr`, so finishing schedules it. `prefetch` fetches that descriptor
// early: what the oracle does, and the first thing to turn off when a chain is not being scheduled at all.
int tinynv_qmd_chain(tinynv_qmd_t *prev, uint64_t next_qmd_addr, int prefetch);

// Constant buffer 0's driver parameters, which are where a kernel finds the windows its shared and local memory live in.
// Returns how many dwords were written, or 0 if the buffer is too short for them.
// `grid` and `block` may be NULL, which leaves the geometry words zero - which is what the python oracle produces, and
// what the byte-for-byte comparisons against it therefore expect.
uint32_t tinynv_qmd_cbuf0(uint32_t *out, uint32_t cap, uint64_t shared_window, uint64_t local_window,
                          const uint32_t grid[3], const uint32_t block[3]);

#endif
