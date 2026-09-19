// Writing the kernel descriptor. See qmd.h for what it is.
//
// Two things about the encoding are worth stating before the code, because both are easy to get subtly wrong and
// neither is visible in a field's name.
//
// A field is a bit range, not a byte range. Writing one means reading the bytes it touches, replacing its bits and
// writing them back - the neighbours in those bytes belong to other fields and must survive. The generator checks that
// no field reaches more than 64 bits past its first byte, so a 64 bit window at that byte is always enough.
//
// Several fields are named for a shift they do not perform. `program_address_lower_shifted4` holds the address already
// divided by sixteen, and it is the writer's job to divide it. The functions here take plain addresses and sizes and do
// the shifting, so no caller has to remember which ones are shifted and by how much.
#include "qmd.h"
#include "internal.h"
#include <stdlib.h>
#include <string.h>

// Indexed fields - one per constant buffer, one per release slot - come out of the generator as eight or two separately
// named macros. These gather them back into something addressable by index.
#define F2(name, i) {TINYNV_QMD_##name##_##i##_LO, TINYNV_QMD_##name##_##i##_HI}
#define TABLE8(name) {F2(name,0), F2(name,1), F2(name,2), F2(name,3), F2(name,4), F2(name,5), F2(name,6), F2(name,7)}
#define TABLE2(name) {F2(name,0), F2(name,1)}
// the release slots are spelled with the index inside the name rather than after it, so they need their own gathering
#define FR(name, i, sfx) {TINYNV_QMD_RELEASE##i##name##sfx##_LO, TINYNV_QMD_RELEASE##i##name##sfx##_HI}

typedef uint16_t field_t[2];

static const field_t cb_lo[TINYNV_QMD_CONSTBUFS]   = TABLE8(CONSTANT_BUFFER_ADDR_LOWER_SHIFTED6);
static const field_t cb_hi[TINYNV_QMD_CONSTBUFS]   = TABLE8(CONSTANT_BUFFER_ADDR_UPPER_SHIFTED6);
static const field_t cb_size[TINYNV_QMD_CONSTBUFS] = TABLE8(CONSTANT_BUFFER_SIZE_SHIFTED4);
static const field_t cb_valid[TINYNV_QMD_CONSTBUFS] = TABLE8(CONSTANT_BUFFER_VALID);
static const field_t rel_size[TINYNV_QMD_RELEASES] = TABLE2(RELEASE_STRUCTURE_SIZE);

static const field_t rel_enable[TINYNV_QMD_RELEASES] = {
  {TINYNV_QMD_RELEASE0_ENABLE_LO, TINYNV_QMD_RELEASE0_ENABLE_HI},
  {TINYNV_QMD_RELEASE1_ENABLE_LO, TINYNV_QMD_RELEASE1_ENABLE_HI}};
static const field_t rel_addr_lo[TINYNV_QMD_RELEASES] = {
  {TINYNV_QMD_RELEASE_SEMAPHORE0_ADDR_LOWER_LO, TINYNV_QMD_RELEASE_SEMAPHORE0_ADDR_LOWER_HI},
  {TINYNV_QMD_RELEASE_SEMAPHORE1_ADDR_LOWER_LO, TINYNV_QMD_RELEASE_SEMAPHORE1_ADDR_LOWER_HI}};
static const field_t rel_addr_hi[TINYNV_QMD_RELEASES] = {
  {TINYNV_QMD_RELEASE_SEMAPHORE0_ADDR_UPPER_LO, TINYNV_QMD_RELEASE_SEMAPHORE0_ADDR_UPPER_HI},
  {TINYNV_QMD_RELEASE_SEMAPHORE1_ADDR_UPPER_LO, TINYNV_QMD_RELEASE_SEMAPHORE1_ADDR_UPPER_HI}};
static const field_t rel_pay_lo[TINYNV_QMD_RELEASES] = {
  {TINYNV_QMD_RELEASE_SEMAPHORE0_PAYLOAD_LOWER_LO, TINYNV_QMD_RELEASE_SEMAPHORE0_PAYLOAD_LOWER_HI},
  {TINYNV_QMD_RELEASE_SEMAPHORE1_PAYLOAD_LOWER_LO, TINYNV_QMD_RELEASE_SEMAPHORE1_PAYLOAD_LOWER_HI}};
static const field_t rel_pay_hi[TINYNV_QMD_RELEASES] = {
  {TINYNV_QMD_RELEASE_SEMAPHORE0_PAYLOAD_UPPER_LO, TINYNV_QMD_RELEASE_SEMAPHORE0_PAYLOAD_UPPER_HI},
  {TINYNV_QMD_RELEASE_SEMAPHORE1_PAYLOAD_UPPER_LO, TINYNV_QMD_RELEASE_SEMAPHORE1_PAYLOAD_UPPER_HI}};

static const field_t prog_lo = {TINYNV_QMD_F(PROGRAM_ADDRESS_LOWER_SHIFTED4)};
static const field_t prog_hi = {TINYNV_QMD_F(PROGRAM_ADDRESS_UPPER_SHIFTED4)};
static const field_t prefetch_lo = {TINYNV_QMD_F(PROGRAM_PREFETCH_ADDR_LOWER_SHIFTED)};
static const field_t prefetch_hi = {TINYNV_QMD_F(PROGRAM_PREFETCH_ADDR_UPPER_SHIFTED)};

static const field_t grid_f[3] = {{TINYNV_QMD_F(GRID_WIDTH)}, {TINYNV_QMD_F(GRID_HEIGHT)}, {TINYNV_QMD_F(GRID_DEPTH)}};
static const field_t block_f[3] = {{TINYNV_QMD_F(CTA_THREAD_DIMENSION0)}, {TINYNV_QMD_F(CTA_THREAD_DIMENSION1)},
                                   {TINYNV_QMD_F(CTA_THREAD_DIMENSION2)}};

static uint64_t field_mask(uint32_t lo, uint32_t hi) {
  uint32_t w = hi - lo + 1;
  return w >= 64 ? ~0ull : ((1ull << w) - 1);
}

uint64_t tinynv_qmd_get(const tinynv_qmd_t *q, uint32_t lo, uint32_t hi) {
  uint64_t num = 0;
  for (uint32_t i = lo / 8; i <= hi / 8; i++) num |= (uint64_t)q->b[i] << (8 * (i - lo / 8));
  return (num >> (lo % 8)) & field_mask(lo, hi);
}

int tinynv_qmd_set(tinynv_qmd_t *q, uint32_t lo, uint32_t hi, uint64_t v) {
  if (hi >= TINYNV_QMD_BYTES * 8 || lo > hi) return tinynv_fail("descriptor field [%u,%u] is not inside the block", lo, hi);
  uint64_t mask = field_mask(lo, hi);
  if (v & ~mask) return tinynv_fail("%#llx does not fit the %u bit descriptor field at %u", (unsigned long long)v, hi - lo + 1, lo);

  uint64_t num = 0;
  for (uint32_t i = lo / 8; i <= hi / 8; i++) num |= (uint64_t)q->b[i] << (8 * (i - lo / 8));
  num = (num & ~(mask << (lo % 8))) | (v << (lo % 8));
  for (uint32_t i = lo / 8; i <= hi / 8; i++) q->b[i] = (uint8_t)(num >> (8 * (i - lo / 8)));

  // and note which bits this claimed, so a second field claiming any of them is counted rather than silently winning
  for (uint32_t bit = lo; bit <= hi; bit++) {
    uint8_t m = (uint8_t)(1u << (bit % 8));
    if (q->claimed[bit / 8] & m) q->overlaps++;
    q->claimed[bit / 8] |= m;
  }
  return 0;
}

#define SET(q, name, v) do { if (tinynv_qmd_set((q), TINYNV_QMD_F(name), (v))) return -1; } while (0)
#define SETF(q, f, v) do { if (tinynv_qmd_set((q), (f)[0], (f)[1], (v))) return -1; } while (0)

// An address split across a lower and an upper field, which is how every address in the descriptor is held.
static int set_addr(tinynv_qmd_t *q, const field_t lo, const field_t hi, uint64_t addr) {
  SETF(q, lo, (uint32_t)addr);
  SETF(q, hi, addr >> 32);
  return 0;
}

uint32_t tinynv_sass_version(uint32_t sm_version) { return ((sm_version & 0xf00) >> 4) | (sm_version & 0xf); }

int tinynv_qmd_program(tinynv_qmd_t *q, const tinynv_qmd_program_t *p) {
  // How much shared memory the hardware is configured to make available: the smallest of the three sizes it offers that
  // still covers what the kernel asked for. A kernel wanting more than the largest cannot run at all.
  static const uint32_t offered[] = {32 * 1024, 64 * 1024, TINYNV_SMEM_MAX_PER_BLOCK};
  uint32_t smem_cfg = 0, max_cfg = 0x1a;   // 0x1a is what the card took, and stays the default
  for (size_t i = 0; i < sizeof(offered) / sizeof(*offered); i++)
    if (offered[i] >= p->shmem) { smem_cfg = offered[i] / 4096 + 1; break; }

  // Above the largest configuration a card has been seen to accept, a carveout has to be asked for explicitly, and this
  // driver will not do it on a guess: ggml's flash attention wants 131,200 bytes and the shim's opt-in goes to 227 KB,
  // neither of which appears in any recording. TINYNV_SMEM_CEILING_KB exists to measure that, not to enable it - the
  // probe in test_hw_smem sets it, makes a kernel write a pattern through the whole span it was promised and read it
  // back, and reports what the card really gave. When a number is measured, it belongs in `offered` above with the
  // measurement written next to it, and this knob goes back to being unused.
  // The sizes above are not arbitrary and neither are these: shared memory is carved out in a fixed set of sizes, and a
  // request for one that is not in the set is not rejected - the descriptor is accepted and the block is simply never
  // scheduled, which arrives as a launch that never completes and a firmware with nothing to report. Measured: asking
  // for 162 KB, a number between two rungs, stalled for the full timeout on an otherwise healthy card.
  //
  // MEASURED ON A GB202, 2026-09-14, and the answer is that there is nothing above the three rungs above.
  //
  //   100,352 bytes: every one of 25,088 words written and read back. The card really gave it.
  //   131,200 bytes (the 132 KB rung, which is what ggml's flash attention asks for): never scheduled.
  //   163,840 bytes (162 KB, a size between two rungs):                               never scheduled.
  //
  // A size the hardware cannot carve out is not refused. The descriptor is accepted, the block is never placed, and the
  // launch simply never completes with the firmware reporting nothing - so the failure looks like a hang rather than a
  // limit, which is why it had to be measured rather than assumed.
  //
  // 228 KB is a Hopper number. Consumer Blackwell has 128 KB of L1 and shared memory per core with 100 KB of it
  // available to a block, and this part behaves exactly that way. The rungs are kept because they are the right ladder
  // for a card that does have them, and TINYNV_SMEM_CEILING_KB is what would reach them; on this card nothing does.
  static const uint32_t beyond[] = {132 * 1024, 164 * 1024, 196 * 1024, 228 * 1024};
  if (!smem_cfg) {
    const char *ask = getenv("TINYNV_SMEM_CEILING_KB");
    uint64_t permitted = ask ? strtoull(ask, NULL, 0) * 1024 : 0;
    uint64_t ceiling = 0;
    for (size_t i = 0; i < sizeof(beyond) / sizeof(*beyond); i++)
      if (beyond[i] >= p->shmem && beyond[i] <= permitted) { ceiling = beyond[i]; break; }
    if (ceiling) {
      uint64_t cfg = ceiling / 4096 + 1;
      uint64_t most = (1ull << (TINYNV_QMD_MAX_SM_CONFIG_SHARED_MEM_SIZE_HI -
                                TINYNV_QMD_MAX_SM_CONFIG_SHARED_MEM_SIZE_LO + 1)) - 1;
      if (cfg > most)
        return tinynv_fail("a %llu byte carveout needs configuration %llu and the descriptor's field holds %llu",
                           (unsigned long long)ceiling, (unsigned long long)cfg, (unsigned long long)most);
      (void)0;
      smem_cfg = (uint32_t)cfg;
      max_cfg = (uint32_t)cfg;  // the ceiling is being asked for, so it is also the most this kernel may have
    }
  }
  if (!smem_cfg)
    // Above this the hardware needs a carveout asked for explicitly, which this driver has never seen a card accept and
    // will not guess at: the recorded session used none of it. Saying which number was wanted and which is the ceiling
    // is more use than a generic failure, because the next step is measuring one launch above it, not reading more code.
    return tinynv_fail("the kernel wants %u bytes of shared memory and this card's largest carveout is %u; measured on "
                       "a GB202, anything above that is accepted into the descriptor and then never scheduled",
                       p->shmem, (unsigned)TINYNV_SMEM_MAX_PER_BLOCK);

  SET(q, QMD_MAJOR_VERSION, TINYNV_QMD_VERSION);
  SET(q, QMD_TYPE, TINYNV_QMDV_QMD_TYPE_GRID_CTA);
  SET(q, REGISTER_COUNT, p->regs);
  SET(q, SHARED_MEMORY_SIZE_SHIFTED7, p->shmem >> 7);
  SET(q, SHADER_LOCAL_MEMORY_HIGH_SIZE_SHIFTED4, p->slm_per_thread >> 4);
  SET(q, QMD_GROUP_ID, 0x3f);
  // The invalidates and the barrier are the oracle's on every launch (ops_nv.py:289-291: the four cache invalidates,
  // the constant-bank invalidate, L1_SYSMEMBAR). Both are measurement knobs since 2026-09-19 (exec.c): with the delta
  // delivery defaults the MoE decode is engine-busy 66% of a token and still a third slower than native, and what a
  // descriptor makes the engine do at its tail - a system-scope barrier over Thunderbolt per kernel, five cache
  // invalidates per launch - is the remaining per-launch cost this driver chooses. The per-chain command-stream
  // invalidate (submit.c, INVALIDATE_SHADER_CACHES_NO_WFI) stays whatever these say.
  int inv_all = p->invalidate == TINYNV_QMD_INVALIDATE_ALL, inv_cb0 = p->invalidate != TINYNV_QMD_INVALIDATE_NONE;
  SET(q, INVALIDATE_TEXTURE_HEADER_CACHE, inv_all);
  SET(q, INVALIDATE_TEXTURE_SAMPLER_CACHE, inv_all);
  SET(q, INVALIDATE_TEXTURE_DATA_CACHE, inv_all);
  SET(q, INVALIDATE_SHADER_DATA_CACHE, inv_all);
  SET(q, API_VISIBLE_CALL_LIMIT, 1);
  SET(q, SAMPLER_INDEX, 1);
  SET(q, BARRIER_COUNT, 1);
  if (tinynv_qmd_membar(q, p->membar)) return -1;
  SET(q, CONSTANT_BUFFER_INVALIDATE_0, inv_cb0);
  SET(q, MIN_SM_CONFIG_SHARED_MEM_SIZE, smem_cfg);
  SET(q, TARGET_SM_CONFIG_SHARED_MEM_SIZE, smem_cfg);
  SET(q, MAX_SM_CONFIG_SHARED_MEM_SIZE, max_cfg);
  SET(q, PROGRAM_PREFETCH_SIZE, p->prog_size >> 8 < 0x1ff ? p->prog_size >> 8 : 0x1ff);
  SET(q, SASS_VERSION, p->sass_version);

  for (int i = 0; i < TINYNV_QMD_CONSTBUFS; i++) {
    if (!p->constbuf_used[i]) continue;
    // The field is named for a shift it is not given: the oracle writes the size in bytes. Reproduced rather than
    // corrected, because a descriptor that disagrees with the oracle is a descriptor whose only witness disagrees with
    // it. Worth raising upstream once the first kernel of this driver's own has run.
    SETF(q, cb_size[i], p->constbuf_size[i]);
    SETF(q, cb_valid[i], 1);
  }
  return 0;
}

int tinynv_qmd_membar(tinynv_qmd_t *q, int membar) {
  uint64_t v = membar == TINYNV_QMD_MEMBAR_GPU ? TINYNV_QMDV_CWD_MEMBAR_TYPE_L1_MEMBAR
             : membar == TINYNV_QMD_MEMBAR_NONE ? TINYNV_QMDV_CWD_MEMBAR_TYPE_L1_NONE
                                                : TINYNV_QMDV_CWD_MEMBAR_TYPE_L1_SYSMEMBAR;
  SET(q, CWD_MEMBAR_TYPE, v);
  return 0;
}

int tinynv_qmd_launch(tinynv_qmd_t *q, const tinynv_qmd_launch_t *l) {
  for (int i = 0; i < 3; i++) {
    SETF(q, grid_f[i], l->grid[i]);
    SETF(q, block_f[i], l->block[i]);
  }
  // the code's address twice: once for the launch, divided by sixteen, and once for the prefetcher, divided by 256
  if (set_addr(q, prog_lo, prog_hi, l->program_addr >> 4)) return -1;
  if (set_addr(q, prefetch_lo, prefetch_hi, l->program_addr >> 8)) return -1;
  for (int i = 0; i < TINYNV_QMD_CONSTBUFS; i++)
    if (l->constbuf_set[i] && set_addr(q, cb_lo[i], cb_hi[i], l->constbuf_addr[i] >> 6)) return -1;
  return 0;
}

int tinynv_qmd_release(tinynv_qmd_t *q, uint64_t addr, uint64_t payload, int timestamp) {
  int slot = -1;
  for (int i = 0; i < TINYNV_QMD_RELEASES; i++)
    if (!tinynv_qmd_get(q, rel_enable[i][0], rel_enable[i][1])) { slot = i; break; }
  if (slot < 0) return -1; // both taken: the caller releases in the command stream instead, which costs methods
  if (set_addr(q, rel_addr_lo[slot], rel_addr_hi[slot], addr)) return -1;
  if (set_addr(q, rel_pay_lo[slot], rel_pay_hi[slot], payload)) return -1;
  SETF(q, rel_enable[slot], 1);
  // a four word structure when it carries a timestamp, two words when it is only the payload
  SETF(q, rel_size[slot], timestamp ? TINYNV_QMDV_RELEASE_STRUCTURE_SIZE_SEMAPHORE_FOUR_WORDS
                                    : TINYNV_QMDV_RELEASE_STRUCTURE_SIZE_SEMAPHORE_TWO_WORDS);
  return slot;
}

// Is this link of a chain the one it was meant to be? Separated from the code that builds chains so that it can be
// tested against descriptors built wrong on purpose - a check that cannot fail is worse than no check, because it reads
// like one. `next_va` is where the following launch lives, or zero for the last link, which must schedule nothing: a
// stale pointer there sends the engine to a descriptor the chain does not own.
int tinynv_qmd_link_check(const tinynv_qmd_t *q, int releases, uint64_t want_payload, uint64_t next_va,
                          const char **why) {
#define NO(msg) do { if (why) *why = (msg); return -1; } while (0)
  int has = tinynv_qmd_get(q, TINYNV_QMD_F(RELEASE0_ENABLE)) != 0;
  if (!releases) {
    // This link was asked to stay silent - either it is not the tail of a tail-only chain, or nothing in this chain
    // releases at all because the command stream ends the batch itself. One that releases anyway puts a value on the
    // timeline nothing waits for and, worse, makes the shape look like the shipping one in a report. The reason is
    // deliberately not spelled as "only at its tail": two modes share this branch, and a refusal that names the wrong
    // one sends the reader to the wrong knob.
    if (has) NO("it releases a value where the chain it is in was built to release nothing");
  } else {
    if (!has) NO("it releases nothing, so the timeline never reaches it");
    uint64_t payload = tinynv_qmd_get(q, TINYNV_QMD_F(RELEASE_SEMAPHORE0_PAYLOAD_LOWER)) |
                       tinynv_qmd_get(q, TINYNV_QMD_F(RELEASE_SEMAPHORE0_PAYLOAD_UPPER)) << 32;
    if (payload != want_payload) NO("it releases the wrong value, so the timeline is not a sequence");
  }
  uint64_t enabled = tinynv_qmd_get(q, TINYNV_QMD_F(DEPENDENT_QMD0_ENABLE));
  uint64_t next = tinynv_qmd_get(q, TINYNV_QMD_F(DEPENDENT_QMD0_POINTER));
  if (next_va) {
    if (!enabled) NO("it does not schedule the launch after it, so the two may run together");
    if (next != next_va >> 8) NO("it schedules the wrong descriptor");
  } else if (enabled) {
    NO("the last link still schedules something, which the chain does not own");
  }
  return 0;
#undef NO
}

int tinynv_qmd_chain(tinynv_qmd_t *prev, uint64_t next_qmd_addr, int prefetch) {
  // The pointer is 32 bits holding the address shifted by eight, so it reaches 1 TB and no further. Every address this
  // driver hands a descriptor is far below that, but the arithmetic is silent about it and the field write would only
  // say "does not fit", which names the field rather than the reason.
  if (next_qmd_addr >> 40)
    return tinynv_fail("a descriptor at %#llx cannot be chained to: the pointer reaches 1 TB and no further",
                       (unsigned long long)next_qmd_addr);
  SET(prev, DEPENDENT_QMD0_POINTER, next_qmd_addr >> 8);
  SET(prev, DEPENDENT_QMD0_ACTION, TINYNV_QMDV_DEPENDENT_QMD0_ACTION_QMD_SCHEDULE);
  // Prefetch fetches the dependent descriptor early. It is an optimisation, not a requirement - and a suspect, because
  // a descriptor being made ready early is exactly where a kernel's resources could be reserved before its predecessor
  // has given them back. This driver has already measured once that a shared memory carveout which cannot be met is not
  // refused: the descriptor is accepted and the block is simply never scheduled, which looks like a hang and says
  // nothing. Two kernels of 57 KB each cannot both be carved out of the 100 KB an SM has.
  SET(prev, DEPENDENT_QMD0_PREFETCH, prefetch ? 1 : 0);
  SET(prev, DEPENDENT_QMD0_ENABLE, 1);
  return 0;
}

uint32_t tinynv_qmd_cbuf0(uint32_t *out, uint32_t cap, uint64_t shared_window, uint64_t local_window,
                          const uint32_t grid[3], const uint32_t block[3]) {
  if (cap < TINYNV_QMD_CBUF0_MIN_DWORDS) return 0;
  memset(out, 0, (size_t)cap * 4);
  out[188] = (uint32_t)shared_window;
  out[189] = (uint32_t)(shared_window >> 32);
  out[190] = (uint32_t)local_window;
  out[191] = (uint32_t)(local_window >> 32);
  out[223] = 0xfffdc0; // the address the hardware traps to, which the kernel reads from here rather than being told

  // blockDim and gridDim. A CUDA kernel does not get these from a special register on this architecture - it loads them
  // out of the parameter region of constant bank 0, and if the driver does not put them there they read as zero. The
  // symptom is not a failure: `i = blockIdx.x * blockDim.x + threadIdx.x` becomes `i = threadIdx.x`, so every block
  // writes over the first block's output and a launch of one block is right while anything larger is quietly wrong.
  //
  // Nothing in the recorded sessions writes them, because tinygrad compiles its launch sizes in as literals and never
  // asks - so these two offsets are the first thing in this driver that no recording could supply. They were measured:
  // the region was filled with one marker per word and a kernel asked what it saw, which came back as the markers at
  // word 216 and word 220. Nothing is inferred from a header here, and the dims kernel in spike/ re-checks all six.
  if (grid) for (int i = 0; i < 3; i++) out[TINYNV_CBUF0_NCTAID + i] = grid[i];
  if (block) for (int i = 0; i < 3; i++) out[TINYNV_CBUF0_NTID + i] = block[i];
  return cap;
}
