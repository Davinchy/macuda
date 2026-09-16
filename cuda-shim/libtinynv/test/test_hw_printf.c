// A cubin that references the device runtime's printf, run on the card.
//
// test_reloc proves such a cubin loads and that the unresolved word holds zero. This proves the kernel then runs and
// computes correctly, which is the part that matters: resolving a symbol to zero is only safe if a kernel that never
// reaches the call is unaffected by it, and every quantised ggml kernel is exactly that - a printf on an assert path
// that no thread takes.
#include "tinynv.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define N 64
int main(int argc, char **argv) {
  const char *path = argc > 1 ? argv[1] : "../spike/printf_kernel.sm120.cubin";
  if (!getenv("TINYNV_HW") || !getenv("TINYNV_SOCKET")) {
    printf("usage: TINYNV_HW=1 TINYNV_SOCKET=<socket> %s [cubin]\n", argv[0]);
    return 2;
  }
  FILE *f = fopen(path, "rb");
  if (!f) { printf("  cannot open %s\n", path); return 1; }
  fseek(f, 0, SEEK_END);
  long len = ftell(f);
  rewind(f);
  void *blob = malloc((size_t)len);
  if (!blob || fread(blob, 1, (size_t)len, f) != (size_t)len) { printf("  cannot read %s\n", path); return 1; }
  fclose(f);

  printf("libtinynv build %s\n", tinynv_build_id());
  tinynv_device_t dev; tinynv_module_t mod; tinynv_kernel_t k; tinynv_kernel_info_t info;
  tinynv_stream_t s; tinynv_devptr_t out; tinynv_status_t st;
#define T(c) do { if ((st = (c)) != TINYNV_OK) { printf("  %s: %s\n    %s\n", #c, tinynv_status_str(st), tinynv_last_error()); return 1; } } while (0)
  T(tinynv_init());
  T(tinynv_device_get(&dev, 0));
  T(tinynv_module_load(dev, blob, (size_t)len, &mod));
  T(tinynv_get_kernel(mod, "printf_kernel", &k));
  T(tinynv_kernel_info(k, &info));
  T(tinynv_stream_create(dev, &s));
  T(tinynv_malloc(dev, N * sizeof(unsigned), &out));

  static unsigned got[N];
  for (int i = 0; i < N; i++) got[i] = 0xdeadbeefu;
  T(tinynv_memcpy_htod(s, out, got, sizeof(got)));

  unsigned char params[64];
  memset(params, 0, sizeof(params));
  uint64_t p = (uint64_t)out;
  unsigned n = N;
  memcpy(params + info.params[0].offset, &p, sizeof(p));
  memcpy(params + info.params[1].offset, &n, sizeof(n));
  T(tinynv_launch(s, k, 1, 1, 1, N, 1, 1, 0, params, (size_t)info.params[1].offset + sizeof(n)));
  T(tinynv_stream_sync(s));
  T(tinynv_memcpy_dtoh(s, got, out, sizeof(got)));

  int wrong = 0, first = -1;
  for (int i = 0; i < N; i++) if (got[i] != (unsigned)i * 3u + 1u) { if (first < 0) first = i; wrong++; }
  if (wrong) {
    printf("  FAIL: %d of %d wrong; the first is [%d] = %#x where %u belongs\n", wrong, N, first, got[first],
           (unsigned)first * 3u + 1u);
    return 1;
  }
  printf("all %d results correct: a kernel whose cubin references the device printf loads, launches and computes\n", N);
  free(blob);
  return 0;
}
