// The allocator that decides addresses. See tlsf.c for why it is not a bump pointer.
#ifndef TINYNV_TLSF_H
#define TINYNV_TLSF_H
#include <stdint.h>

#define TINYNV_TLSF_FAIL ((uint64_t)-1)

typedef struct tinynv_tlsf tinynv_tlsf_t;

tinynv_tlsf_t *tinynv_tlsf_new(uint64_t size, uint64_t base);
void tinynv_tlsf_free_all(tinynv_tlsf_t *a);
uint64_t tinynv_tlsf_alloc(tinynv_tlsf_t *a, uint64_t req_size, uint64_t align);
// 0 on success, -1 for an address that names no live block - a double release or a caller's bad pointer, refused
// rather than silently merged into the free list.
int tinynv_tlsf_release(tinynv_tlsf_t *a, uint64_t addr);
// The size of the LIVE block starting exactly at `addr`, or 0 if there is no such block. A physical free hands this
// its record's size: a disagreement is a corrupted record rather than a caller's mistake, and is refused as one.
uint64_t tinynv_tlsf_block_size(const tinynv_tlsf_t *a, uint64_t addr);
// Bytes in live blocks, and the size the allocator manages - so "how much is free" is a question the ALLOCATOR answers
// (tinynv_mem_info), never the physical memory size, which says nothing about what an allocation would get.
uint64_t tinynv_tlsf_used(const tinynv_tlsf_t *a);
uint64_t tinynv_tlsf_size(const tinynv_tlsf_t *a);

#endif
