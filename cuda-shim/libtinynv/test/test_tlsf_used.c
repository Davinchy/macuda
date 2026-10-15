// The allocator's own count of what it has handed out - what tinynv_mem_info turns into "free".
//
// Checked against a second route rather than against itself: after every step, tinynv_tlsf_used must equal the sum of
// tinynv_tlsf_block_size over every address still live, which the allocator computes from its block list and not from
// the counter. A release it refuses (an address it never handed out, or one already freed) must leave the count alone.
#include "tlsf.h"
#include <stdio.h>
#include <stdint.h>

static int fails, checks;
#define CHECK(cond, ...) do { checks++; if (!(cond)) { printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); fails++; } } while (0)

#define N 64
static uint64_t live[N];
static int nlive;
static uint64_t sum_live(const tinynv_tlsf_t *a) {
  uint64_t s = 0;
  for (int i = 0; i < nlive; i++) s += tinynv_tlsf_block_size(a, live[i]);
  return s;
}

int main(void) {
  tinynv_tlsf_t *a = tinynv_tlsf_new(1ull << 30, 0x4200000);   // a 1 GiB region at a base only 2 MiB-aligned, like VRAM's
  CHECK(a && tinynv_tlsf_size(a) == 1ull << 30, "size");
  CHECK(tinynv_tlsf_used(a) == 0, "a new allocator has handed out nothing");
  static const uint64_t sizes[] = {4096, 65536, 2u << 20, 12345, 1u << 20, 16, 3u << 20, 4096};
  static const uint64_t aligns[] = {4096, 65536, 2u << 20, 1, 4096, 1, 1u << 20, 4096};
  for (int round = 0; round < 3; round++) {
    for (size_t i = 0; i < sizeof sizes / sizeof sizes[0]; i++) {
      uint64_t p = tinynv_tlsf_alloc(a, sizes[i] << round, aligns[i]);
      CHECK(p != TINYNV_TLSF_FAIL, "alloc %llu", (unsigned long long)(sizes[i] << round));
      live[nlive++] = p;
      CHECK(tinynv_tlsf_used(a) == sum_live(a), "after alloc %d: counter %llu, blocks sum %llu", nlive,
            (unsigned long long)tinynv_tlsf_used(a), (unsigned long long)sum_live(a));
    }
    // release every other one, which leaves holes the next round's merges and aligned splits have to work around
    for (int i = nlive - 1; i >= 0; i -= 2) {
      uint64_t before = tinynv_tlsf_used(a), sz = tinynv_tlsf_block_size(a, live[i]);
      CHECK(tinynv_tlsf_release(a, live[i]) == 0, "release");
      CHECK(tinynv_tlsf_used(a) == before - sz, "a release of %llu moved the count by %lld", (unsigned long long)sz,
            (long long)(before - tinynv_tlsf_used(a)));
      live[i] = live[--nlive];
      CHECK(tinynv_tlsf_used(a) == sum_live(a), "after release: counter %llu, blocks sum %llu",
            (unsigned long long)tinynv_tlsf_used(a), (unsigned long long)sum_live(a));
    }
  }
  // refusals must not move the count
  uint64_t before = tinynv_tlsf_used(a);
  CHECK(tinynv_tlsf_release(a, 0x4200000 + 7) != 0, "an address never handed out must be refused");
  uint64_t p = tinynv_tlsf_alloc(a, 8192, 4096);
  CHECK(tinynv_tlsf_release(a, p) == 0, "release");
  uint64_t mid = tinynv_tlsf_used(a);
  CHECK(tinynv_tlsf_release(a, p) != 0, "a second release of the same block must be refused");
  CHECK(tinynv_tlsf_used(a) == mid && mid == before, "refused releases moved the count: %llu -> %llu",
        (unsigned long long)before, (unsigned long long)tinynv_tlsf_used(a));
  while (nlive) CHECK(tinynv_tlsf_release(a, live[--nlive]) == 0, "final release");
  CHECK(tinynv_tlsf_used(a) == 0, "everything back, yet %llu counted as handed out", (unsigned long long)tinynv_tlsf_used(a));
  tinynv_tlsf_free_all(a);
  printf(fails ? "%d of %d checks failed\n" : "allocator count: all %d checks passed\n", fails ? fails : checks, checks);
  return fails ? 1 : 0;
}
