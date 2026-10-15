// The sub-launch diff behind TINYNV_DELTA_DELIVERY, checked without a GPU.
//
// tinynv_delta_runs decides which bytes of a launch's descriptor span cross the link as inline patches and which are
// trusted to be there already. A run a dword short delivers a stale dword the kernel reads as its parameter, and
// nothing on the card reports that - the answer is simply wrong somewhere else. So every property the emit loop in
// tinynv_exec_flush relies on is pinned here: runs are dword multiples, they cover every differing dword and no other,
// they never leave [lo,hi), they merge exactly at the gap and not past it, an overflow still reports the envelope,
// and at the default gap a launch's runs never cost more pushbuffer than its one envelope would - the inequality the
// check pass's "cheaper of the two" decision leans on. Then the two knob parsers, both spellings of each ask, so a
// sweep reads off the startup line what it asked for.
//
// Seen to fail before it was trusted: with the merge test in tinynv_delta_runs changed from <= to <, the gap tests
// below fail; with the envelope tracking stopped at overflow, the overflow test fails (2026-09-19).
#include "exec.h"
#include <stdio.h>
#include <string.h>

static int fails;
// Every CHECK counts itself, and the run asserts it executed at least as many as there are CHECK sites in this file -
// a number the Makefile greps out at build time, so it cannot go stale the way a literal would. See test_submit.c for
// where that rule came from.
static int checks;
#define CHECK(cond, ...) do { checks++; if (!(cond)) { printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); fails++; } } while (0)

#define SPAN 1536u   // a real launch: a 512-byte descriptor slot plus a 1 KB constant buffer, both 256-aligned

// Every run is a dword multiple inside [lo,hi), in order, non-overlapping, separated by more than `gap` bytes, and
// together they cover exactly the dwords that differ. The one property that makes a patch safe, stated once.
static void check_cover(const char *what, const uint8_t *a, const uint8_t *b, uint32_t lo, uint32_t hi, uint32_t gap,
                        const tinynv_delta_span_t *runs, int k) {
  for (int r = 0; r < k; r++) {
    CHECK(!(runs[r].lo & 3u) && !(runs[r].hi & 3u), "%s: run %d [%u,%u) is not a whole number of dwords", what, r,
          runs[r].lo, runs[r].hi);
    CHECK(runs[r].lo >= lo && runs[r].hi <= hi && runs[r].hi > runs[r].lo, "%s: run %d [%u,%u) leaves [%u,%u)", what,
          r, runs[r].lo, runs[r].hi, lo, hi);
    if (r) CHECK(runs[r].lo > runs[r - 1].hi + gap, "%s: runs %d and %d are %u apart with gap %u - they should be one",
                 what, r - 1, r, runs[r].lo - runs[r - 1].hi, gap);
    // and a run never starts or ends on an unchanged dword: that would be a byte the envelope's shape leaked in
    CHECK(memcmp(a + runs[r].lo, b + runs[r].lo, 4) && memcmp(a + runs[r].hi - 4, b + runs[r].hi - 4, 4),
          "%s: run %d [%u,%u) starts or ends on a dword that did not change", what, r, runs[r].lo, runs[r].hi);
  }
  for (uint32_t j = lo; j < hi; j += 4) {
    int differs = memcmp(a + j, b + j, 4) != 0, covered = 0;
    for (int r = 0; r < k; r++) if (j >= runs[r].lo && j < runs[r].hi) covered = 1;
    if (differs && !covered) { CHECK(0, "%s: dword at %u differs and no run covers it - a stale parameter", what, j); return; }
  }
}

static uint32_t cost(const tinynv_delta_span_t *s, int k) {   // the pushbuffer price the check pass uses
  uint32_t dw = 0;
  for (int r = 0; r < k; r++) dw += TINYNV_INLINE_DWORDS(s[r].hi - s[r].lo);
  return dw;
}

int main(void) {
  static uint8_t a[SPAN * 3], b[SPAN * 3];
  tinynv_delta_span_t runs[64], env;
  const uint32_t lo = SPAN, hi = 2 * SPAN;   // the launch under test sits between two neighbours

  // identical: nothing to send, and the envelope says so the way the caller reads it (hi <= lo)
  memset(a, 0x5a, sizeof a); memcpy(b, a, sizeof a);
  int k = tinynv_delta_runs(a, b, lo, hi, 32, runs, 64, &env);
  CHECK(k == 0, "identical spans gave %d runs", k);
  CHECK(env.hi <= env.lo, "identical spans gave a non-empty envelope [%u,%u)", env.lo, env.hi);

  // one byte, mid-dword: the run is that dword, no wider, and the envelope is the same dword
  b[lo + 5] ^= 1;
  k = tinynv_delta_runs(a, b, lo, hi, 32, runs, 64, &env);
  CHECK(k == 1 && runs[0].lo == lo + 4 && runs[0].hi == lo + 8, "one changed byte at +5 gave %d runs, first [%u,%u)",
        k, k ? runs[0].lo : 0, k ? runs[0].hi : 0);
  CHECK(env.lo == lo + 4 && env.hi == lo + 8, "envelope [%u,%u) is not the one dword", env.lo, env.hi);
  check_cover("one byte", a, b, lo, hi, 32, runs, k);

  // the first and last dwords of the span: runs reach the edges and stop there, the envelope is the whole span
  memcpy(b, a, sizeof a); b[lo] ^= 1; b[hi - 1] ^= 1;
  k = tinynv_delta_runs(a, b, lo, hi, 32, runs, 64, &env);
  CHECK(k == 2 && runs[0].lo == lo && runs[0].hi == lo + 4 && runs[1].lo == hi - 4 && runs[1].hi == hi,
        "edges gave %d runs, [%u,%u) and [%u,%u)", k, runs[0].lo, runs[0].hi, k > 1 ? runs[1].lo : 0,
        k > 1 ? runs[1].hi : 0);
  CHECK(env.lo == lo && env.hi == hi, "edge envelope is [%u,%u), not the span", env.lo, env.hi);
  check_cover("edges", a, b, lo, hi, 32, runs, k);

  // changes in the neighbours are not this launch's: the scan stays inside [lo,hi)
  memcpy(b, a, sizeof a); b[lo - 1] ^= 1; b[hi] ^= 1;
  k = tinynv_delta_runs(a, b, lo, hi, 32, runs, 64, &env);
  CHECK(k == 0 && env.hi <= env.lo, "changes just outside the span were reported: %d runs, envelope [%u,%u)", k,
        env.lo, env.hi);

  // the gap: two dwords with exactly 32 unchanged bytes between them are one patch at gap 32 (the tie merges), two at
  // gap 28, and at gap 0 only touching dwords merge
  memcpy(b, a, sizeof a); b[lo + 64] ^= 1; b[lo + 64 + 4 + 32] ^= 1;
  k = tinynv_delta_runs(a, b, lo, hi, 32, runs, 64, &env);
  CHECK(k == 1 && runs[0].lo == lo + 64 && runs[0].hi == lo + 64 + 4 + 32 + 4,
        "a 32-byte gap at gap=32 gave %d runs, first [%u,%u) - the tie should merge", k, runs[0].lo, runs[0].hi);
  CHECK(cost(runs, k) == TINYNV_INLINE_DWORDS(40), "the merged run costs %u dwords, not %u", cost(runs, k),
        TINYNV_INLINE_DWORDS(40));
  k = tinynv_delta_runs(a, b, lo, hi, 28, runs, 64, &env);
  CHECK(k == 2, "a 32-byte gap at gap=28 gave %d runs, not 2", k);
  check_cover("gap 28", a, b, lo, hi, 28, runs, k);
  // at gap 28 the two calls cost 16 header dwords for 2 data dwords; merged at 32 it was 8 for 10 - the same 18, which
  // is the tie the default sits on
  CHECK(cost(runs, k) == 18 && TINYNV_INLINE_DWORDS(40) == 18, "the split costs %u dwords and the merge %u - the "
        "default gap is meant to be the exact tie", cost(runs, k), TINYNV_INLINE_DWORDS(40));
  memcpy(b, a, sizeof a); b[lo + 100] ^= 1; b[lo + 104] ^= 1; b[lo + 112] ^= 1;   // dwords 25, 26 and 28
  k = tinynv_delta_runs(a, b, lo, hi, 0, runs, 64, &env);
  CHECK(k == 2 && runs[0].lo == lo + 100 && runs[0].hi == lo + 108 && runs[1].lo == lo + 112 && runs[1].hi == lo + 116,
        "gap 0 gave %d runs: touching dwords should merge and one unchanged dword should split", k);
  check_cover("gap 0", a, b, lo, hi, 0, runs, k);
  k = tinynv_delta_runs(a, b, lo, hi, 4, runs, 64, &env);
  CHECK(k == 1 && runs[0].lo == lo + 100 && runs[0].hi == lo + 116, "gap 4 gave %d runs; one unchanged dword should "
        "now ride along", k);

  // overflow: more runs than the caller has room for reports -1, and the envelope is still exact - the check pass
  // falls back to it for that launch, so it has to be right when nothing else is
  memcpy(b, a, sizeof a);
  for (uint32_t j = lo + 8; j + 4 <= hi - 8; j += 64) b[j] ^= 1;   // 24 runs, 64 bytes apart
  k = tinynv_delta_runs(a, b, lo, hi, 32, runs, 64, &env);
  CHECK(k == 24, "24 isolated dwords gave %d runs", k);
  check_cover("24 runs", a, b, lo, hi, 32, runs, k);
  tinynv_delta_span_t full_env = env;
  k = tinynv_delta_runs(a, b, lo, hi, 32, runs, 8, &env);
  CHECK(k == -1, "24 runs in room for 8 gave %d, not -1", k);
  CHECK(env.lo == full_env.lo && env.hi == full_env.hi, "the envelope after overflow is [%u,%u), was [%u,%u) with "
        "room - the fallback would patch short", env.lo, env.hi, full_env.lo, full_env.hi);
  CHECK(env.lo == lo + 8 && env.hi == lo + 8 + 23 * 64 + 4, "the overflow envelope [%u,%u) is not first-to-last",
        env.lo, env.hi);
  k = tinynv_delta_runs(a, b, lo, hi, 32, runs, 0, &env);
  CHECK(k == -1 && env.lo == full_env.lo && env.hi == full_env.hi, "no room at all: %d runs, envelope [%u,%u)", k,
        env.lo, env.hi);
  k = tinynv_delta_runs(a, b, lo, hi, 32, runs, 24, &env);
  CHECK(k == 24, "exactly enough room gave %d", k);

  // the inequality the check pass leans on: at the default gap the runs never cost more than the envelope, for any
  // pattern - each gap kept is longer than the 8 header dwords it saves. Random patterns, and a few adversarial ones:
  // gaps of exactly 36 bytes (the smallest kept) and dense alternation.
  uint32_t seed = 0x2545f491u;
  int worst_ok = 1, seen_multi = 0;
  for (int t = 0; t < 400 && worst_ok; t++) {
    memcpy(b, a, sizeof a);
    seed = seed * 1103515245u + 12345u;
    int density = 1 + (int)((seed >> 16) % 40);   // one dword in `density` differs
    for (uint32_t j = lo; j < hi; j += 4) {
      seed = seed * 1103515245u + 12345u;
      if ((int)((seed >> 16) % (unsigned)density) == 0) b[j + ((seed >> 8) & 3u)] ^= (uint8_t)(1 + (seed & 0x7f));
    }
    k = tinynv_delta_runs(a, b, lo, hi, 32, runs, 64, &env);
    if (k < 0) { k = 1; runs[0] = env; }   // what the check pass does when a launch overflows its room
    else check_cover("random", a, b, lo, hi, 32, runs, k);
    if (k > 1) seen_multi++;
    if (env.hi > env.lo && cost(runs, k) > TINYNV_INLINE_DWORDS(env.hi - env.lo)) {
      worst_ok = 0;
      printf("  pattern %d (density 1/%d): %d runs cost %u dwords, the envelope %u\n", t, density, k, cost(runs, k),
             TINYNV_INLINE_DWORDS(env.hi - env.lo));
    }
  }
  CHECK(worst_ok, "a pattern's runs cost more than its envelope at the default gap - the tie arithmetic is off");
  CHECK(seen_multi > 100, "only %d of 400 random patterns produced more than one run - the property was not exercised",
        seen_multi);
  memcpy(b, a, sizeof a);
  for (uint32_t j = lo; j + 4 <= hi; j += 40) b[j] ^= 1;   // 36-byte gaps: kept, and each costs 9 data dwords vs 8 of headers
  k = tinynv_delta_runs(a, b, lo, hi, 32, runs, 64, &env);
  CHECK(k == (int)((hi - lo + 39) / 40), "36-byte gaps gave %d runs", k);
  CHECK(cost(runs, k) < TINYNV_INLINE_DWORDS(env.hi - env.lo), "36-byte gaps: runs %u dwords, envelope %u - splitting "
        "should just win", cost(runs, k), TINYNV_INLINE_DWORDS(env.hi - env.lo));
  memcpy(b, a, sizeof a);
  for (uint32_t j = lo; j + 4 <= hi; j += 8) b[j] ^= 1;   // every other dword: one run, the whole span
  k = tinynv_delta_runs(a, b, lo, hi, 32, runs, 64, &env);
  CHECK(k == 1 && runs[0].lo == lo && runs[0].hi == hi - 4, "every other dword gave %d runs, first [%u,%u)", k,
        runs[0].lo, runs[0].hi);

  // the knobs. NULL is unset, "" is set-but-empty; both are the default, like every other knob in exec.c.
  static const struct { const char *e; uint32_t want; } gap[] = {
    {NULL, 32}, {"", 32}, {"32", 32}, {"0", 0}, {"33", 32}, {"35", 32}, {"36", 36}, {"-8", 0}, {"abc", 0},
    {"100000", 65536}, {"64", 64}};
  for (unsigned i = 0; i < sizeof gap / sizeof *gap; i++)
    CHECK(tinynv_exec_delta_gap(gap[i].e) == gap[i].want, "TINYNV_DELTA_GAP=%s gave %u, expected %u",
          gap[i].e ? gap[i].e : "(unset)", tinynv_exec_delta_gap(gap[i].e), gap[i].want);
  static const struct { const char *e; uint32_t want; } agg[] = {
    {NULL, 128u * 1024u}, {"", 128u * 1024u}, {"64", 64u * 1024u}, {"0", 128u * 1024u}, {"-1", 128u * 1024u},
    {"abc", 128u * 1024u}, {"256", 256u * 1024u}, {"1000", 256u * 1024u}, {"1", 1024u}};
  for (unsigned i = 0; i < sizeof agg / sizeof *agg; i++)
    CHECK(tinynv_exec_delta_aggregate(agg[i].e) == agg[i].want, "TINYNV_DELTA_AGGREGATE_KB=%s gave %u, expected %u",
          agg[i].e ? agg[i].e : "(unset)", tinynv_exec_delta_aggregate(agg[i].e), agg[i].want);
  // and the default gap IS the header cost, in bytes - the arithmetic the tie test above depends on
  CHECK(tinynv_exec_delta_gap(NULL) == 4u * (TINYNV_INLINE_DWORDS(0)), "the default gap %u is not the %u bytes a "
        "call's headers cost", tinynv_exec_delta_gap(NULL), 4u * TINYNV_INLINE_DWORDS(0));

  if (checks < TINYNV_DELTA_CHECKS) {
    printf("  FAIL: %d checks ran but this file has %d CHECK sites - a block was skipped and the run would have "
           "reported success\n", checks, TINYNV_DELTA_CHECKS);
    fails++;
  }
  printf(fails ? "%d checks failed\n" : "all %d checks passed\n", fails ? fails : checks);
  return fails ? 1 : 0;
}
