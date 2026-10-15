// Whether recycling the scratch region a piece at a time ever hands out bytes an engine is still reading.
//
// This is the one failure in the scheme that hardware would not report. A descriptor whose bytes are given away while
// the engine reads them does not fault: the engine reads whatever is there now and launches it, so the symptom is a
// wrong answer thousands of launches later, or nothing at all on a run that happened to be lucky. A bump arena that
// drained everything at the wrap could not have the bug; one that reuses a piece while the rest of the region stays
// live can, and the difference is entirely in the bookkeeping.
//
// So the bookkeeping is driven here against a record of who wrote each byte, with no card and no driver. The region's
// three decisions - where would this land, what has to finish first, who owns it now - are the same functions exec.c
// calls per allocation; everything around them (the engine, the timeline, the flush) is modelled, because what is
// being checked is the arithmetic and not the plumbing.
//
// The invariant: at the moment a byte is handed out, the batch that last used it has finished. Recorded per byte, so
// an off-by-one at a piece boundary, a tail belonging to no piece, or a span cleared before it was recorded all fail
// here rather than on the card. Over-waiting is allowed and counted, never failed: the region tracks pieces, not
// bytes, so it rounds outwards on purpose.
#include "exec.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t rng_state = 0x243f6a8885a308d3ull;
static uint64_t rnd(uint64_t n) {
  rng_state ^= rng_state << 13; rng_state ^= rng_state >> 7; rng_state ^= rng_state << 17;
  return n ? rng_state % n : 0;
}

#define MAX_PENDING 256

// Two policies over the same sequence of allocations, so the comparison means something. DRAIN is what the region did
// before: one wrap point, and coming round waits for everything outstanding. PIECES is what it does now.
enum policy { POLICY_DRAIN, POLICY_PIECES };

struct result {
  uint64_t handouts, wraps, waits, wasted, stall;
};

struct model {
  tinynv_exec_region_t rg;
  uint64_t *owner;          // per byte: the value of the last submitted batch to use it, 0 for never used
  uint64_t engine;          // what the engine has finished; only ever rises
  uint64_t counter;         // the timeline, handed out one per batch
  struct { uint64_t off, bytes; } pending[MAX_PENDING];
  int npending;
  struct result r;
};

// What has been built reaches video memory. In the driver this is push_shadow, and it is a separate act from
// submitting on purpose: a push can happen with nothing submitted, which is exactly why ownership is recorded from the
// used span and not this one.
static void push(struct model *m) {
  m->rg.dirty_lo = m->rg.dirty_hi = 0;
}

// A batch reaches the engine. Everything allocated since the last one belongs to it.
static void submit(struct model *m) {
  if (!m->npending) return;
  uint64_t v = ++m->counter;
  push(m);                     // submit_batch pushes before it hands over, and the driver's spans are clear after it
  tinynv_arena_submitted(&m->rg, v);
  for (int i = 0; i < m->npending; i++)
    for (uint64_t b = m->pending[i].off; b < m->pending[i].off + m->pending[i].bytes; b++) m->owner[b] = v;
  m->npending = 0;
}

// The engine gets on with it, always a fixed number of batches behind. A lag of zero would make every wait free and
// prove nothing. It is the same model under both policies, which is what makes the two stall figures comparable: the
// engine's progress is a property of the engine, not of how the host chooses to wait for it.
static void engine_runs(struct model *m, uint64_t lag) {
  uint64_t at = m->counter > lag ? m->counter - lag : 0;
  if (at > m->engine) m->engine = at;
}

// Waiting costs the host however far ahead of the engine it had to reach.
static void stall_until(struct model *m, uint64_t value) {
  if (value <= m->engine) return;
  m->r.stall += value - m->engine;
  m->engine = value;
}

static int one_run(uint64_t size, uint64_t maxbytes, uint64_t align, uint64_t lag, int chain, unsigned iters,
                   enum policy pol, const char *what, struct result *out) {
  struct model m;
  memset(&m, 0, sizeof(m));
  m.rg.size = size;
  m.owner = calloc((size_t)size, sizeof(uint64_t));
  if (!m.owner) return printf("  FAIL: out of memory for the %llu byte record\n", (unsigned long long)size), 1;
  rng_state = 0x243f6a8885a308d3ull;    // the same sequence of allocations under both policies

  for (unsigned it = 0; it < iters; it++) {
    uint64_t bytes = (rnd(maxbytes) + align) & ~(align - 1);
    if (bytes > size) bytes = align;

    int wrap = 0;
    uint64_t off = tinynv_arena_place(&m.rg, bytes, align, &wrap);
    if (wrap) {
      // What exec.c does when the region comes round: send what is built, then rewind. The chain guard there means
      // there is never anything unsubmitted at this point, which is what makes the rewind safe to model this way.
      push(&m);                  // arena() pushes before it comes round, whether or not there is anything to submit
      submit(&m);
      m.r.wraps++;
      if (pol == POLICY_DRAIN) stall_until(&m, m.counter);   // the old cost: everything outstanding, every lap
      if (tinynv_arena_rewind(&m.rg)) {
        printf("  FAIL (%s): came round at iteration %u with a span neither submitted nor pushed\n", what, it);
        free(m.owner); return 1;
      }
      off = tinynv_arena_place(&m.rg, bytes, align, &wrap);
      if (wrap) { printf("  FAIL: %llu bytes will not fit a %llu byte region even from the bottom\n",
                         (unsigned long long)bytes, (unsigned long long)size); free(m.owner); return 1; }
    }

    // What the bytes about to be handed out are really waiting for, which the region does not know: it tracks pieces.
    uint64_t highest = 0;
    for (uint64_t b = off; b < off + bytes; b++) if (m.owner[b] > highest) highest = m.owner[b];
    int was_free = highest <= m.engine;

    if (pol == POLICY_PIECES) {
      uint64_t want = tinynv_arena_enter(&m.rg, off, bytes);
      if (want > m.counter) {
        printf("  FAIL: asked to wait for %llu with only %llu ever submitted\n",
               (unsigned long long)want, (unsigned long long)m.counter);
        free(m.owner); return 1;
      }
      if (want > m.engine) { m.r.waits++; if (was_free) m.r.wasted++; stall_until(&m, want); }
    }

    // The invariant, byte by byte, under both policies - which also says the harness is measuring what it claims.
    if (highest > m.engine) {
      printf("  FAIL (%s): handed out %llu..%llu at iteration %u; the engine has finished %llu and batch %llu is "
             "still reading it\n", what, (unsigned long long)off, (unsigned long long)(off + bytes),
             it, (unsigned long long)m.engine, (unsigned long long)highest);
      free(m.owner); return 1;
    }

    tinynv_arena_take(&m.rg, off, bytes);
    m.pending[m.npending].off = off;
    m.pending[m.npending].bytes = bytes;
    if (++m.npending >= chain || m.npending >= MAX_PENDING) submit(&m);
    engine_runs(&m, lag);
    m.r.handouts++;
  }
  *out = m.r;
  free(m.owner);
  return 0;
}

// The tail of a region whose size is not a multiple of the piece count belongs to a piece like everything else. It is
// checked on its own because the arithmetic that gets it wrong - a fixed piece size, with the last piece short - looks
// right and leaves those bytes tracked by nothing at all.
static int tail_is_tracked(void) {
  tinynv_exec_region_t rg;
  int bad = 0;
  for (uint64_t size = 1000; size < 1000 + 64; size++) {
    memset(&rg, 0, sizeof(rg));
    rg.size = size;
    tinynv_arena_mark(&rg, size - 1, size, 77);            // the very last byte
    if (tinynv_arena_wait_for(&rg, size - 1, 1) != 77) {
      printf("  FAIL: the last byte of a %llu byte region is recorded against no piece\n", (unsigned long long)size);
      bad = 1;
    }
    // and the first, which is the other end the arithmetic can fall off
    memset(&rg, 0, sizeof(rg));
    rg.size = size;
    tinynv_arena_mark(&rg, 0, 1, 9);
    if (tinynv_arena_wait_for(&rg, 0, 1) != 9) {
      printf("  FAIL: the first byte of a %llu byte region is recorded against no piece\n", (unsigned long long)size);
      bad = 1;
    }
  }
  if (!bad) printf("  every byte of a region belongs to a piece, at 64 sizes that divide unevenly\n");
  return bad;
}

// One piece being reused must not wait for work that only touched another. This is the whole point of the change, so
// it is asserted rather than left to the rates below. Written in terms of the piece count so that it still says
// something when the count is varied.
static int pieces_are_independent(void) {
  tinynv_exec_region_t rg;
  const uint64_t n = TINYNV_EXEC_SEGMENTS;
  memset(&rg, 0, sizeof(rg));
  rg.size = 1 << 20;
  // Where a piece starts, exactly. Dividing the size by the count and multiplying back lands inside the previous
  // piece whenever the division truncates, which is a bug in a test rather than in the thing tested, but it reads as
  // the thing tested failing.
  #define PIECE_AT(i) (((i) * rg.size + n - 1) / n)
  uint64_t piece = PIECE_AT(1);
  tinynv_arena_mark(&rg, 0, piece, 100);                        // the first piece was used by batch 100
  tinynv_arena_mark(&rg, PIECE_AT(n - 1), rg.size, 900);        // the last by batch 900
  if (tinynv_arena_wait_for(&rg, 0, 64) != 100)
    return printf("  FAIL: reusing the first piece waits for something that never touched it\n"), 1;
  if (n >= 3 && tinynv_arena_wait_for(&rg, PIECE_AT(n / 2), 64) != 0)
    return printf("  FAIL: reusing an untouched piece waits at all\n"), 1;
  if (tinynv_arena_wait_for(&rg, PIECE_AT(n - 1) - 32, 64) != 900)
    return printf("  FAIL: an allocation straddling two pieces does not wait for the later of them\n"), 1;

  // And entering a piece clears it, because from there to the end of the lap it holds only this lap's own work. If it
  // did not, every allocation inside a piece would wait for the batch just submitted - a stall per launch instead of
  // per lap, which is what the first version of this did.
  memset(&rg, 0, sizeof(rg));
  rg.size = 1 << 20;
  tinynv_arena_mark(&rg, 0, piece, 100);
  if (tinynv_arena_enter(&rg, 0, 64) != 100)
    return printf("  FAIL: entering a piece does not report what it was waiting for\n"), 1;
  if (tinynv_arena_enter(&rg, 64, 64) != 0)
    return printf("  FAIL: a second allocation in a piece already entered asks about it again\n"), 1;
  printf("  a piece is waited for once, when it is entered, and for nothing that never touched it (%llu pieces)\n",
         (unsigned long long)n);
  return 0;
}

int main(void) {
  if (tail_is_tracked()) return 1;
  if (pieces_are_independent()) return 1;

  // Shapes worth driving: decode, where descriptors are small and chains are long and the region is sized for it;
  // a region far too small for its work, where every lap is a wait and the scheme must still be correct; and awkward
  // sizes and alignments that no real allocation would ask for.
  struct { uint64_t size, maxbytes, align, lag; int chain; unsigned iters; const char *what; } runs[] = {
    { 1u << 20, 256,   64,  8,  32, 200000, "decode-shaped" },
    { 1u << 20, 256,   64,  64, 32, 200000, "decode, engine far behind" },
    { 8192,     256,   64,  8,  32, 200000, "region far too small" },
    { 4096,     1024,  256, 4,  4,   50000, "big allocations, short chains" },
    { 1000003,  333,   1,   16, 17, 200000, "awkward size and alignment" },
    { 1u << 16, 64,    64,  0,  1,   50000, "engine keeping up" },
    { 1u << 16, 4096,  64,  32, 8,   50000, "allocations a fraction of the region" },
    // The regime where the number of pieces is the question rather than a detail: a lap of this region is about
    // sixteen batches, so the engine is most of a lap behind. Split in two, the half being reused is half a lap old
    // and the wait is real; split into eight, it is seven eighths of a lap old and there is nothing to wait for.
    { 1u << 16, 256,   64,  8,  32,  50000, "engine most of a lap behind" },
  };
  printf("  %-34s %8s %7s %8s %12s %12s\n", "", "handouts", "laps", "waits", "stall drain", "stall pieces");
  for (unsigned i = 0; i < sizeof(runs) / sizeof(runs[0]); i++) {
    struct result drain, pieces;
    if (one_run(runs[i].size, runs[i].maxbytes, runs[i].align, runs[i].lag, runs[i].chain, runs[i].iters,
                POLICY_DRAIN, runs[i].what, &drain)) return 1;
    if (one_run(runs[i].size, runs[i].maxbytes, runs[i].align, runs[i].lag, runs[i].chain, runs[i].iters,
                POLICY_PIECES, runs[i].what, &pieces)) return 1;
    if (pieces.stall > drain.stall) {
      printf("  FAIL (%s): recycling in pieces waited longer than draining every lap, %llu against %llu\n",
             runs[i].what, (unsigned long long)pieces.stall, (unsigned long long)drain.stall);
      return 1;
    }
    printf("  %-34s %8llu %7llu %8llu %12llu %12llu  (%llu of the waits were the piece rounding up)\n",
           runs[i].what, (unsigned long long)pieces.handouts, (unsigned long long)pieces.wraps,
           (unsigned long long)pieces.waits, (unsigned long long)drain.stall, (unsigned long long)pieces.stall,
           (unsigned long long)pieces.wasted);
  }
  printf("  no byte was ever handed out while the batch that last used it was still running, under either policy\n");
  printf("  stall is batches of engine time the host spent waiting; lower is better and neither column is a rate\n");
  return 0;
}
