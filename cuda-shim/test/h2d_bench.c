// h2d_bench: what the copy engine charges to bring N bytes of host memory into video memory, through the public API
// (tinynv_memcpy_htod stages into pinned host memory and the copy engine pulls it across; the sync waits for the
// timeline). Prices the DMA delivery of the descriptor arena: one flush at depth 32 is 48 KB, at 128 it is 192 KB.
#include "tinynv.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (double)t.tv_sec + (double)t.tv_nsec * 1e-9; }
#define T(call) do { tinynv_status_t st_ = (call); if (st_ != TINYNV_OK) { printf("  %s: %s\n    %s\n", #call, tinynv_status_str(st_), tinynv_last_error()); return 1; } } while (0)
int main(void) {
  setvbuf(stdout, NULL, _IONBF, 0);
  tinynv_device_t dev; tinynv_stream_t s; tinynv_devptr_t d;
  T(tinynv_init()); T(tinynv_device_get(&dev, 0)); T(tinynv_stream_create(dev, &s)); T(tinynv_malloc(dev, 4u << 20, &d));
  unsigned char *src = malloc(4u << 20); for (size_t i = 0; i < (4u << 20); i++) src[i] = (unsigned char)(i * 3 + 1);
  T(tinynv_memcpy_htod(s, d, src, 4u << 20)); T(tinynv_stream_sync(s));
  double t0 = now(); for (int i = 0; i < 100; i++) T(tinynv_stream_sync(s)); double sy = (now() - t0) / 100 * 1e6;
  printf("  a sync alone, nothing pending:            %8.1f us\n", sy);
  printf("  one host-to-device copy of N bytes, then a sync (the pair, and less the sync alone)\n");
  printf("  %10s %12s %12s %10s\n", "bytes", "pair us", "copy us", "MB/s");
  const size_t sizes[] = {4096, 16384, 49152, 98304, 196608, 1048576, 4194304};
  for (size_t si = 0; si < sizeof(sizes) / sizeof(*sizes); si++) {
    size_t n = sizes[si]; int reps = n >= 1048576 ? 10 : 30;
    t0 = now(); for (int i = 0; i < reps; i++) { T(tinynv_memcpy_htod(s, d, src, n)); T(tinynv_stream_sync(s)); }
    double pair = (now() - t0) / reps * 1e6, cost = pair - sy;
    printf("  %10zu %12.1f %12.1f %10.1f\n", n, pair, cost, (double)n / cost);
  }
  printf("  eight 48 KB copies back to back, then one sync (the pipelined price per copy)\n");
  { int reps = 20; t0 = now(); for (int i = 0; i < reps; i++) { for (int j = 0; j < 8; j++) T(tinynv_memcpy_htod(s, d + (tinynv_devptr_t)(j * 49152), src, 49152)); T(tinynv_stream_sync(s)); }
    double pair = (now() - t0) / reps * 1e6; printf("  %10s %12.1f %12.1f\n", "8 x 49152", pair, (pair - sy) / 8); }
  unsigned char *back = malloc(49152); T(tinynv_memcpy_dtoh(s, back, d, 49152)); T(tinynv_stream_sync(s));
  printf("  the last 48 KB reads back %s\n", memcmp(back, src, 49152) ? "WRONG" : "as written");
  T(tinynv_free(dev, d)); free(src); free(back);
  return 0;
}
