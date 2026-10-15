// Recycling the scratch region a piece at a time: where an allocation lands, what has to finish before its bytes may
// be handed out, and who owns them afterwards.
//
// These are here rather than in exec.c because they are the whole of the decision and none of the machinery. They read
// and write one struct, call nothing, and touch no device, which is what lets test/test_arena.c drive them against a
// byte-by-byte record of who wrote what - the failure they exist to prevent is bytes given back while an engine still
// reads them, and that produces a wrong answer with no fault, so there is nothing on the card that would report it.
#include "exec.h"

// Which piece an offset falls in. Multiplying before dividing rather than cutting the region into pieces of a fixed
// size, because a region whose size is not a multiple of the count would leave a tail belonging to no piece at all -
// bytes handed out with nothing recorded against them, which is the one failure this whole scheme exists to prevent.
static uint64_t seg_of(const tinynv_exec_region_t *rg, uint64_t off) {
  return off * TINYNV_EXEC_SEGMENTS / rg->size;
}

uint64_t tinynv_arena_place(const tinynv_exec_region_t *rg, uint64_t bytes, uint64_t align, int *wrap) {
  uint64_t off = (rg->next + align - 1) & ~(align - 1);
  *wrap = off + bytes > rg->size;
  return off;
}

// The highest value any piece under this allocation is waiting on, which is all a caller needs: a channel retires in
// order, so reaching the highest means reaching every lower one too.
uint64_t tinynv_arena_wait_for(const tinynv_exec_region_t *rg, uint64_t off, uint64_t bytes) {
  if (!bytes || !rg->size || off + bytes > rg->size) return 0;
  uint64_t want = 0;
  for (uint64_t i = seg_of(rg, off); i <= seg_of(rg, off + bytes - 1); i++)
    if (rg->seg_value[i] > want) want = rg->seg_value[i];
  return want;
}

// Widening a span rather than tracking each allocation keeps it to one write per region: within a region, within a
// batch, these are consecutive.
static void widen(uint64_t *lo, uint64_t *hi, uint64_t a, uint64_t b) {
  if (*hi == *lo) { *lo = a; *hi = b; return; }
  if (a < *lo) *lo = a;
  if (b > *hi) *hi = b;
}

// Entering the pieces an allocation falls in, which is the only moment their record matters. Returns the highest value
// they were waiting on and clears them: from here to the end of the lap those pieces belong to the work being built,
// and asking again would be asking about ourselves.
//
// Pieces between the last one entered and this allocation are stepped over without being read or cleared. Nothing is
// being handed out in them - they are alignment, not bytes - so their record is still last lap's and still wanted.
uint64_t tinynv_arena_enter(tinynv_exec_region_t *rg, uint64_t off, uint64_t bytes) {
  if (!bytes || !rg->size || off + bytes > rg->size) return 0;
  uint64_t s0 = seg_of(rg, off), s1 = seg_of(rg, off + bytes - 1), want = 0;
  if (s0 < rg->seg_entered) s0 = rg->seg_entered;
  for (uint64_t i = s0; i <= s1; i++) {
    if (rg->seg_value[i] > want) want = rg->seg_value[i];
    rg->seg_value[i] = 0;
  }
  if (s1 + 1 > rg->seg_entered) rg->seg_entered = s1 + 1;
  return want;
}

void tinynv_arena_take(tinynv_exec_region_t *rg, uint64_t off, uint64_t bytes) {
  rg->next = off + bytes;
  widen(&rg->dirty_lo, &rg->dirty_hi, off, off + bytes);
  widen(&rg->used_lo, &rg->used_hi, off, off + bytes);
}

// Remember which pieces a batch used, so that coming round to one of them later waits for that batch and not for
// everything. Values only ever rise, so what is kept is the newest batch to have touched the piece.
void tinynv_arena_mark(tinynv_exec_region_t *rg, uint64_t lo, uint64_t hi, uint64_t value) {
  if (hi <= lo || !rg->size || hi > rg->size) return;
  for (uint64_t i = seg_of(rg, lo); i <= seg_of(rg, hi - 1); i++)
    if (value > rg->seg_value[i]) rg->seg_value[i] = value;
}

void tinynv_arena_submitted(tinynv_exec_region_t *rg, uint64_t value) {
  tinynv_arena_mark(rg, rg->used_lo, rg->used_hi, value);
  rg->used_lo = rg->used_hi = 0;
}

// Start the region again from the bottom.
//
// THE CALLER MUST HAVE SUBMITTED AND PUSHED FIRST, and nothing in this file enforces or can enforce that. arena() in
// exec.c does it - push_shadow then flush, both immediately above its call to this - and the two spans below are empty
// by the time control arrives here BECAUSE OF THAT and for no reason visible from here. Said the other way round: this
// function cannot tell a correct call from one that is about to hand live bytes away, so the guarantee lives at the
// call site and this comment is the only thing that says so.
//
// So it says so and then checks it. A span left over is the observable half of the precondition - bytes handed out
// that no submission has given a value, or built and never pushed - and returning it makes the claim something a
// caller can be caught breaking rather than something a reader has to agree with. It costs two comparisons on a path
// taken once a lap.
//
// The spans are cleared regardless. An offset from before the rewind left in one would widen the next span across the
// whole region, which is a slow correct push rather than a silent wrong one, and that is the right way to fail here.
int tinynv_arena_rewind(tinynv_exec_region_t *rg) {
  int held = rg->dirty_hi == rg->dirty_lo && rg->used_hi == rg->used_lo;
  rg->next = 0;
  rg->seg_entered = 0;    // a new lap; every piece is somebody else's again until it is entered
  rg->dirty_lo = rg->dirty_hi = 0;
  rg->used_lo = rg->used_hi = 0;
  return held ? 0 : -1;
}
