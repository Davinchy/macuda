// Pinned host memory pool: a bounded number of driver mappings, carved for cudaMallocHost/cudaHostAlloc/cudaFreeHost.
#ifndef TINYCUDART_HOSTPOOL_H
#define TINYCUDART_HOSTPOOL_H
#include <stddef.h>
#include "tinynv.h"
typedef struct { size_t slabs, bytes_mapped, bytes_in_use, lifetime_allocs; } tinycudart_hostpool_stats_t;
int  tinycudart_hostpool_init(tinynv_device_t dev);
int  tinycudart_host_alloc(size_t n, void **out);   // 0 on success; -1 = pinned memory unavailable (caller falls back)
int  tinycudart_host_free(void *p);                 // 0 on success; -1 = not a pool pointer
int  tinycudart_host_owns(const void *p);           // 1 if p lies in pinned memory this pool handed out
void tinycudart_hostpool_stats(tinycudart_hostpool_stats_t *s);
#endif
