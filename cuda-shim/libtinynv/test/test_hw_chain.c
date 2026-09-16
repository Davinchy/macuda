// Does work still happen in the right order when nothing waits for it?
//
// Every other hardware test here syncs between operations, so each would pass against a driver that reordered or
// clobbered everything in between. That was fine while the driver waited for every batch. It does not wait any more:
// launches are submitted and the caller returns, scratch is reused once the engine has passed it, and work on the copy
// engine is ordered against work on the compute engine by a semaphore acquire in the command stream rather than by the
// host. All three of those are wrong-answer bugs if they are wrong, not crashes.
//
// So this builds a chain where every step needs the one before it, submits the whole thing without a single sync, and
// checks the arithmetic at the end. c = a + b, then c = c + b, over and over: the result is a + Nb, and a step that ran
// early, late or twice gives a different number.
#include "tinynv.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (double)t.tv_sec + t.tv_nsec * 1e-9; }

#define N 256
// Long enough to wrap the scratch arena several times over. That matters more than the depth: a wrap hands back
// scratch an engine may still be reading, and it used to be safe only because every batch was waited on. At roughly
// 1.7 KB of descriptor, constant buffer and command buffer per launch against a 4 MB arena, this is a few wraps.
#define LINKS 8000
// Each crossing adds two to the total and hands the result between the engines twice.
#define CROSSINGS 500

int main(int argc, char **argv) {
  const char *path = argc > 1 ? argv[1] : "../spike/vecadd.sm120.cubin";
  if (!getenv("TINYNV_HW") || !getenv("TINYNV_SOCKET")) {
    printf("usage: TINYNV_HW=1 TINYNV_SOCKET=<socket> %s [cubin]\n", argv[0]);
    return 2;
  }
  printf("libtinynv build %s\n", tinynv_build_id());

  FILE *f = fopen(path, "rb");
  if (!f) { printf("  cannot open %s\n", path); return 1; }
  fseek(f, 0, SEEK_END);
  long len = ftell(f);
  rewind(f);
  void *blob = malloc((size_t)len);
  if (!blob || fread(blob, 1, (size_t)len, f) != (size_t)len) { printf("  cannot read %s\n", path); return 1; }
  fclose(f);

  tinynv_device_t dev; tinynv_module_t mod; tinynv_kernel_t k; tinynv_kernel_info_t info; tinynv_stream_t s;
  tinynv_devptr_t a, b, c, d; tinynv_status_t st;
#define T(call) do { if ((st = (call)) != TINYNV_OK) { printf("  %s: %s\n    %s\n", #call, tinynv_status_str(st), tinynv_last_error()); return 1; } } while (0)
  T(tinynv_init());
  T(tinynv_device_get(&dev, 0));
  T(tinynv_module_load(dev, blob, (size_t)len, &mod));
  T(tinynv_get_kernel(mod, "vecadd", &k));
  T(tinynv_kernel_info(k, &info));
  T(tinynv_stream_create(dev, &s));
  T(tinynv_malloc(dev, N * sizeof(float), &a));
  T(tinynv_malloc(dev, N * sizeof(float), &b));
  T(tinynv_malloc(dev, N * sizeof(float), &c));
  T(tinynv_malloc(dev, N * sizeof(float), &d));

  static float ha[N], hb[N], hc[N];
  for (int i = 0; i < N; i++) { ha[i] = (float)i; hb[i] = 1.0f; hc[i] = -1.0f; }
  T(tinynv_memcpy_htod(s, a, ha, sizeof(ha)));
  T(tinynv_memcpy_htod(s, b, hb, sizeof(hb)));
  T(tinynv_memcpy_htod(s, c, hc, sizeof(hc)));

  unsigned char params[64];
  unsigned n = N;
  size_t plen = (size_t)info.params[3].offset + sizeof(int);
#define SET(x, y, z) do { memset(params, 0, sizeof(params)); \
  uint64_t p0 = (uint64_t)(x), p1 = (uint64_t)(y), p2 = (uint64_t)(z); \
  memcpy(params + info.params[0].offset, &p0, 8); memcpy(params + info.params[1].offset, &p1, 8); \
  memcpy(params + info.params[2].offset, &p2, 8); memcpy(params + info.params[3].offset, &n, 4); } while (0)

  // c = a + b, and then c = c + b another LINKS-1 times. Not one sync in the whole chain.
  double t0 = now();
  SET(a, b, c);
  T(tinynv_launch(s, k, N / 64, 1, 1, 64, 1, 1, 0, params, plen));
  for (int i = 1; i < LINKS; i++) {
    SET(c, b, c);
    T(tinynv_launch(s, k, N / 64, 1, 1, 64, 1, 1, 0, params, plen));
  }
  double issued = now() - t0;
  T(tinynv_stream_sync(s));
  double done = now() - t0;
  printf("submitted %d dependent launches with no synchronisation between them\n", LINKS);
  printf("  issuing them took %.3f s, %.1f us each; everything finished %.3f s in, %.1f us each\n",
         issued, issued * 1e6 / LINKS, done, done * 1e6 / LINKS);

  // and then the part with one copy at the end could not test: the engines interleaved.
  //
  // Compute and copy run independently, and nothing but a semaphore acquire in the command stream orders them. A single
  // copy after a long chain barely exercises that - by the time it is submitted the chain has usually finished anyway.
  // So this alternates them, each step depending on the last: the kernel adds to c, the copy moves c to d, the kernel
  // adds to d, the copy moves it back. A copy that starts before the kernel it follows, or a kernel that starts before
  // the copy that feeds it, loses or repeats an addition and the count at the end is wrong.
  for (int i = 0; i < CROSSINGS; i++) {
    SET(c, b, c);
    T(tinynv_launch(s, k, N / 64, 1, 1, 64, 1, 1, 0, params, plen));
    T(tinynv_memcpy_dtod(s, d, c, N * sizeof(float)));
    SET(d, b, d);
    T(tinynv_launch(s, k, N / 64, 1, 1, 64, 1, 1, 0, params, plen));
    T(tinynv_memcpy_dtod(s, c, d, N * sizeof(float)));
  }
  printf("and %d crossings between the engines, each depending on the one before\n", CROSSINGS);

  T(tinynv_memcpy_dtod(s, d, c, N * sizeof(float)));
  T(tinynv_stream_sync(s));
  T(tinynv_memcpy_dtoh(s, hc, d, sizeof(hc)));

  int wrong = 0, first = -1;
  for (int i = 0; i < N; i++) {
    float want = (float)i + (float)(LINKS + 2 * CROSSINGS);   // a + b added once per launch, b being all ones
    if (hc[i] != want) { if (first < 0) first = i; wrong++; }
  }
  if (wrong) {
    printf("  FAIL: %d of %d wrong; the first is [%d] = %g where %g belongs\n", wrong, N, first, (double)hc[first],
           (double)((float)first + (float)(LINKS + 2 * CROSSINGS)));
    printf("        a chain that computes too few is work that ran out of order or scratch reused too early;\n"
           "        too many is a batch submitted twice.\n");
    return 1;
  }
  printf("all %d correct after a %d-deep chain and %d engine crossings: ordering holds without the host waiting\n",
         N, LINKS, CROSSINGS);
  T(tinynv_free(dev, a)); T(tinynv_free(dev, b)); T(tinynv_free(dev, c)); T(tinynv_free(dev, d));
  free(blob);
  return 0;
}
