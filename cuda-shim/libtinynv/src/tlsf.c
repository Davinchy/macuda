// The allocator the addresses come from.
//
// This is a two level segregated fit allocator, and it is here rather than a bump pointer because the addresses it
// returns are not private: they go into page tables, into the structures GSP-RM reads, and into the commands the GPU
// executes. A different allocator gives different addresses, and every one of those is then wrong in a way that only
// shows up as a fault a long way from here.
//
// The behaviour that matters, and the reason a bump pointer could not stand in: an aligned allocation that leaves a gap
// below it keeps that gap free, and a later small allocation is placed in the gap rather than after the big one. Session
// A predicted this would be the first address to diverge once 2 MB aligned allocations appeared, and it was, exactly
// there. Free blocks are kept in buckets by size, and a request takes the first block in the smallest bucket that fits,
// which is what makes the choice reproducible rather than merely reasonable.
#include "internal.h"
#include "tlsf.h"
#include <stdlib.h>
#include <string.h>

#define LV2_COUNT 16
#define LV2_BITS 5 // (16).bit_length(), the oracle's lv2_cnt.bit_length()

static int bit_length(uint64_t v) {
  int n = 0;
  while (v) { n++; v >>= 1; }
  return n;
}

static uint64_t round_up(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }

static int lv1_of(uint64_t size) { return bit_length(size); }

static int lv2_of(uint64_t size) {
  int bl = bit_length(size);
  int shift = bl - LV2_BITS;
  return (int)((size - (1ull << (bl - 1))) >> (shift > 0 ? shift : 0));
}

// blocks are a doubly linked list in address order; free ones are also in a bucket
typedef struct blk {
  uint64_t start, size;
  struct blk *next, *prev;   // address order
  struct blk *bnext, *bprev; // within a bucket
  int free;
} blk_t;

struct tinynv_tlsf {
  uint64_t base, size;
  blk_t *head;
  blk_t **buckets; // [lv1][lv2], flattened
  int lv1_max;
};

static blk_t **bucket_of(tinynv_tlsf_t *a, uint64_t size) {
  int l1 = lv1_of(size);
  if (l1 > a->lv1_max) l1 = a->lv1_max;
  return &a->buckets[(size_t)l1 * LV2_COUNT + lv2_of(size)];
}

static void bucket_remove(tinynv_tlsf_t *a, blk_t *b) {
  blk_t **head = bucket_of(a, b->size);
  if (b->bprev) b->bprev->bnext = b->bnext;
  else if (*head == b) *head = b->bnext;
  if (b->bnext) b->bnext->bprev = b->bprev;
  b->bnext = b->bprev = NULL;
  b->free = 0;
}

// The oracle appends to a bucket and takes from the front, so within one bucket the oldest block is chosen. Keeping a
// singly linked stack would take the newest and pick different addresses, so insertion goes to the tail.
static void bucket_append(tinynv_tlsf_t *a, blk_t *b) {
  blk_t **head = bucket_of(a, b->size);
  b->bnext = b->bprev = NULL;
  b->free = 1;
  if (!*head) { *head = b; return; }
  blk_t *t = *head;
  while (t->bnext) t = t->bnext;
  t->bnext = b;
  b->bprev = t;
}

tinynv_tlsf_t *tinynv_tlsf_new(uint64_t size, uint64_t base) {
  tinynv_tlsf_t *a = calloc(1, sizeof(*a));
  if (!a) return NULL;
  a->base = base;
  a->size = size;
  a->lv1_max = bit_length(size);
  a->buckets = calloc((size_t)(a->lv1_max + 1) * LV2_COUNT, sizeof(blk_t *));
  if (!a->buckets) { free(a); return NULL; }
  if (size) {
    blk_t *b = calloc(1, sizeof(*b));
    b->start = 0;
    b->size = size;
    a->head = b;
    bucket_append(a, b);
  }
  return a;
}

void tinynv_tlsf_free_all(tinynv_tlsf_t *a) {
  if (!a) return;
  for (blk_t *b = a->head; b;) { blk_t *n = b->next; free(b); b = n; }
  free(a->buckets);
  free(a);
}

// split b at `at` bytes in, leaving b with `at` and a new free block with the rest
static blk_t *split(tinynv_tlsf_t *a, blk_t *b, uint64_t at) {
  blk_t *rest = calloc(1, sizeof(*rest));
  if (!rest) return NULL;
  bucket_remove(a, b);
  rest->start = b->start + at;
  rest->size = b->size - at;
  rest->next = b->next;
  rest->prev = b;
  if (b->next) b->next->prev = rest;
  b->next = rest;
  b->size = at;
  bucket_append(a, b);
  bucket_append(a, rest);
  return rest;
}

uint64_t tinynv_tlsf_alloc(tinynv_tlsf_t *a, uint64_t req_size, uint64_t align) {
  if (!align) align = 1;
  if (req_size < 16) req_size = 16; // the oracle's minimum block
  uint64_t want = req_size + align - 1;
  if (want < 16) want = 16;
  // round up to the next bucket, so any block in it is certainly big enough
  int bl = bit_length(want);
  int shift = bl - LV2_BITS;
  want = round_up(want, 1ull << (shift > 0 ? shift : 0));

  for (int l1 = lv1_of(want); l1 <= a->lv1_max; l1++) {
    for (int l2 = (l1 == bit_length(want) ? lv2_of(want) : 0); l2 < LV2_COUNT; l2++) {
      blk_t *b = a->buckets[(size_t)l1 * LV2_COUNT + l2];
      if (!b) continue;
      if (b->size < want) continue;

      // give back the misaligned head, then anything past the request
      uint64_t aligned = round_up(b->start, align);
      if (aligned != b->start) {
        blk_t *rest = split(a, b, aligned - b->start);
        if (!rest) return TINYNV_TLSF_FAIL;
        b = rest;
      }
      if (b->size > req_size && !split(a, b, req_size)) return TINYNV_TLSF_FAIL;
      bucket_remove(a, b);
      return b->start + a->base;
    }
  }
  return TINYNV_TLSF_FAIL;
}

void tinynv_tlsf_release(tinynv_tlsf_t *a, uint64_t addr) {
  uint64_t at = addr - a->base;
  blk_t *b = a->head;
  while (b && b->start != at) b = b->next;
  if (!b || b->free) return;
  bucket_append(a, b);
  // merge with free neighbours so the space can be used by a larger request later
  while (b->prev && b->prev->free) {
    blk_t *p = b->prev;
    bucket_remove(a, p);
    bucket_remove(a, b);
    p->size += b->size;
    p->next = b->next;
    if (b->next) b->next->prev = p;
    free(b);
    b = p;
    bucket_append(a, b);
  }
  while (b->next && b->next->free) {
    blk_t *n = b->next;
    bucket_remove(a, b);
    bucket_remove(a, n);
    b->size += n->size;
    b->next = n->next;
    if (n->next) n->next->prev = b;
    free(n);
    bucket_append(a, b);
  }
}
