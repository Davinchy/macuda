// Pinned host memory pool for libtinycudart.
//
// Why a pool: over TinyGPU the dext keeps every DMA mapping until the socket closes and allows at most 128 per session,
// and each mapping is described in at most 32 physical segments. libtinynv therefore cannot release a host mapping
// mid-session (tinynv_host_free only drops the page list). A runtime that mapped and unmapped per cudaMallocHost /
// cudaFreeHost would exhaust the session after 128 *lifetime* allocations even with a handful live. So: take a few
// large slabs from the driver (64 MB each, the size tinygrad found fits the segment cap), carve them with a first-fit
// free list, coalesce on free, and never hand a slab back. Driver mappings are bounded by the number of slabs.
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tinynv.h"
#include "hostpool.h"

#define SLAB_BYTES (64u << 20)
#define MIN_SLAB   (4u << 20)   // if the driver refuses 64 MB (segment cap), halve down to this before giving up
#define GRAN       4096u        // page granularity: DMA pages are 4 KB, and cudaMallocHost promises >= 256 B alignment
#define MAX_SLABS  96           // hard ceiling on driver mappings this pool will ever hold (the dext allows 128/session)

typedef struct block { size_t off, size; int used; struct block *next; } block_t;
typedef struct { unsigned char *raw, *base; size_t size; block_t *blocks; } slab_t;

static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static slab_t g_slabs[MAX_SLABS];
static int g_nslabs;
static tinynv_device_t g_dev;
static size_t g_in_use, g_lifetime_allocs, g_mapped;

static size_t round_up(size_t v, size_t a) { return (v + a - 1) / a * a; }

// ask the driver for a mapping of at least `want` bytes; a refusal for size (the segment cap) is retried smaller
static slab_t *new_slab(size_t want) {
  if (g_nslabs >= MAX_SLABS) return NULL;
  size_t size = want > SLAB_BYTES ? round_up(want, GRAN) : SLAB_BYTES;
  void *raw = NULL;
  for (;;) {
    tinynv_status_t s = tinynv_host_alloc(g_dev, size + GRAN, &raw);
    if (s == TINYNV_OK) break;
    if (s != TINYNV_ERR_INVALID || size <= MIN_SLAB || want > size / 2) return NULL;
    size /= 2;
  }
  slab_t *sl = &g_slabs[g_nslabs++];
  sl->raw = raw;
  sl->base = (unsigned char *)round_up((uintptr_t)raw, GRAN);
  sl->size = size;
  sl->blocks = calloc(1, sizeof(block_t));
  sl->blocks->size = size;
  g_mapped += size;
  return sl;
}

static void *carve(slab_t *sl, size_t size) {
  for (block_t *b = sl->blocks; b; b = b->next) {
    if (b->used || b->size < size) continue;
    if (b->size > size) { // split: the remainder stays free after us
      block_t *rest = calloc(1, sizeof(block_t));
      rest->off = b->off + size; rest->size = b->size - size; rest->next = b->next;
      b->size = size; b->next = rest;
    }
    b->used = 1;
    return sl->base + b->off;
  }
  return NULL;
}

int tinycudart_hostpool_init(tinynv_device_t dev) { g_dev = dev; return 0; }

int tinycudart_host_alloc(size_t n, void **out) {
  if (!n) n = 1;
  size_t size = round_up(n, GRAN);
  pthread_mutex_lock(&g_lock);
  void *p = NULL;
  for (int i = 0; i < g_nslabs && !p; i++) p = carve(&g_slabs[i], size);
  if (!p) { slab_t *sl = new_slab(size); if (sl) p = carve(sl, size); }
  if (p) { g_in_use += size; g_lifetime_allocs++; }
  pthread_mutex_unlock(&g_lock);
  *out = p;
  return p ? 0 : -1;
}

int tinycudart_host_free(void *p) {
  if (!p) return 0;
  int rc = -1;
  pthread_mutex_lock(&g_lock);
  for (int i = 0; i < g_nslabs && rc; i++) {
    slab_t *sl = &g_slabs[i];
    if ((unsigned char *)p < sl->base || (unsigned char *)p >= sl->base + sl->size) continue;
    size_t off = (size_t)((unsigned char *)p - sl->base);
    for (block_t *prev = NULL, *b = sl->blocks; b; prev = b, b = b->next) {
      if (b->off != off || !b->used) continue;
      b->used = 0; g_in_use -= b->size; rc = 0;
      if (b->next && !b->next->used) { block_t *n = b->next; b->size += n->size; b->next = n->next; free(n); }
      if (prev && !prev->used) { prev->size += b->size; prev->next = b->next; free(b); }
      break;
    }
  }
  pthread_mutex_unlock(&g_lock);
  return rc;
}

int tinycudart_host_owns(const void *p) {
  pthread_mutex_lock(&g_lock);
  int owns = 0;
  for (int i = 0; i < g_nslabs && !owns; i++)
    owns = (const unsigned char *)p >= g_slabs[i].base && (const unsigned char *)p < g_slabs[i].base + g_slabs[i].size;
  pthread_mutex_unlock(&g_lock);
  return owns;
}

void tinycudart_hostpool_stats(tinycudart_hostpool_stats_t *s) {
  pthread_mutex_lock(&g_lock);
  s->slabs = (size_t)g_nslabs; s->bytes_mapped = g_mapped; s->bytes_in_use = g_in_use; s->lifetime_allocs = g_lifetime_allocs;
  pthread_mutex_unlock(&g_lock);
}
