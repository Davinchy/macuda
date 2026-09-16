// A kernel on the card, through the same interface the cudart shim calls and nothing else.
//
// Everything below goes through cuda-shim/include/tinynv.h: no driver internals, no private state. That is the point -
// if this passes, the path libtinycudart takes has been walked end to end, and what is left between here and llama.cpp
// is the shim's own marshalling rather than anything underneath it.
//
// The arithmetic is chosen so a wrong answer cannot look right. c[i] = a[i] + b[i] with a[i] = i and b[i] = 2i gives
// 3i, which differs in every element: a kernel that ran on stale memory, ran on the wrong buffer, or did not run at all
// fails on the first element rather than on some unlucky one.
//
// Needs the socket and TINYNV_HW=1, the same two locks test_hw asks for, so nothing here runs by accident.
#include "tinynv.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;
#define CHECK(cond, ...) do { if (!(cond)) { printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); fails++; } } while (0)
#define OK(call) do { tinynv_status_t st_ = (call); \
  if (st_ != TINYNV_OK) { printf("  FAIL: %s: %s\n         %s\n", #call, tinynv_status_str(st_), tinynv_last_error()); \
                          return ++fails; } } while (0)

#define N 256

int main(int argc, char **argv) {
  const char *cubin_path = argc > 1 ? argv[1] : "../spike/vecadd.sm120.cubin";
  const char *kernel = argc > 2 ? argv[2] : "vecadd";
  // The block size, and therefore how many blocks: a run that is right for one block and wrong past it cannot say
  // whether the grid failed to advance or the kernel's length argument arrived wrong, and one launch of a different
  // shape separates them.
  const unsigned block = argc > 3 ? (unsigned)strtoul(argv[3], NULL, 0) : 64;
  if (!getenv("TINYNV_HW") || !getenv("TINYNV_SOCKET")) {
    printf("usage: TINYNV_HW=1 TINYNV_SOCKET=<socket> %s [cubin] [kernel]\n", argv[0]);
    printf("  runs a kernel on a real gpu, so it needs both. take the gpu lock first.\n");
    return 2;
  }

  FILE *f = fopen(cubin_path, "rb");
  if (!f) { printf("  cannot open %s\n", cubin_path); return 1; }
  fseek(f, 0, SEEK_END);
  long len = ftell(f);
  rewind(f);
  void *blob = malloc((size_t)len);
  if (!blob || fread(blob, 1, (size_t)len, f) != (size_t)len) { printf("  cannot read %s\n", cubin_path); return 1; }
  fclose(f);

  printf("libtinynv build %s\n", tinynv_build_id());
  tinynv_device_t dev;
  OK(tinynv_init());
  OK(tinynv_device_get(&dev, 0));

  tinynv_device_props_t props;
  OK(tinynv_device_props(dev, &props));
  printf("device: %s, sm %d.%d, %d cores at most, %.1f GB\n", props.name, props.cc_major, props.cc_minor, props.sm_count,
         (double)props.total_mem / (1024.0 * 1024.0 * 1024.0));
  CHECK(props.cc_major > 0, "the device reports compute capability 0, which means it cannot run anything");

  tinynv_module_t mod;
  tinynv_kernel_t k;
  tinynv_kernel_info_t info;
  OK(tinynv_module_load(dev, blob, (size_t)len, &mod));
  OK(tinynv_get_kernel(mod, kernel, &k));
  OK(tinynv_kernel_info(k, &info));
  printf("kernel %s: %d parameters at c[0][%#x], %d registers, %d bytes of shared memory\n", kernel, info.num_params,
         info.param_base, info.regs, info.static_smem);
  CHECK(info.num_params == 4, "%s takes %d parameters, this test passes 4", kernel, info.num_params);
  if (fails) return fails;

  tinynv_devptr_t a, b, c;
  OK(tinynv_malloc(dev, N * sizeof(float), &a));
  OK(tinynv_malloc(dev, N * sizeof(float), &b));
  OK(tinynv_malloc(dev, N * sizeof(float), &c));
  printf("allocated three buffers at %#llx %#llx %#llx\n", (unsigned long long)a, (unsigned long long)b,
         (unsigned long long)c);
  CHECK(!(a % TINYNV_DEVPTR_ALIGN || b % TINYNV_DEVPTR_ALIGN || c % TINYNV_DEVPTR_ALIGN),
        "a device pointer came back unaligned, which the interface promises it never does");

  static float ha[N], hb[N], hc[N];
  for (int i = 0; i < N; i++) { ha[i] = (float)i; hb[i] = (float)(2 * i); hc[i] = -1.0f; }

  tinynv_stream_t s;
  OK(tinynv_stream_create(dev, &s));

  // Before any kernel: does memory survive a round trip at all? A launch that comes back with the first quarter right
  // has two possible causes - one block of four ran, or the copy moved a quarter - and they are indistinguishable from
  // the results. This separates them, and it is worth having permanently: a driver whose copies are short produces
  // wrong answers everywhere and blames the kernels.
  static float probe[N], back[N];
  for (int i = 0; i < N; i++) { probe[i] = (float)(1000 + i); back[i] = -2.0f; }
  OK(tinynv_memcpy_htod(s, a, probe, sizeof(probe)));
  OK(tinynv_memcpy_dtoh(s, back, a, sizeof(back)));
  int bad = -1;
  for (int i = 0; i < N && bad < 0; i++) if (back[i] != probe[i]) bad = i;
  CHECK(bad < 0, "a %zu byte round trip through device memory differs from element %d (%g, not %g): %d of %zu bytes "
        "survived, so the copy is short, not the kernel", sizeof(probe), bad, bad >= 0 ? (double)back[bad] : 0.0,
        bad >= 0 ? (double)probe[bad] : 0.0, bad * 4, sizeof(probe));
  if (bad < 0) printf("round trip: %zu bytes through device memory and back, unchanged\n", sizeof(probe));
  if (fails) return fails;

  OK(tinynv_memcpy_htod(s, a, ha, sizeof(ha)));
  OK(tinynv_memcpy_htod(s, b, hb, sizeof(hb)));
  // and the output buffer is filled with something that is not the answer, so "the kernel never wrote" is a failure
  // rather than a zero that happens to be right
  OK(tinynv_memcpy_htod(s, c, hc, sizeof(hc)));
  printf("copied %zu bytes up through the copy engine\n", sizeof(ha) * 3);

  // the parameter block, laid out from the kernel's own table the way the shim marshals it
  unsigned char params[256];
  memset(params, 0, sizeof(params));
  uint64_t ptrs[3] = {(uint64_t)a, (uint64_t)b, (uint64_t)c};
  int n = N;
  size_t params_len = 0;
  for (int i = 0; i < info.num_params; i++) {
    const void *src = i < 3 ? (const void *)&ptrs[i] : (const void *)&n;
    size_t want = i < 3 ? sizeof(uint64_t) : sizeof(int);
    CHECK((size_t)info.params[i].size == want, "parameter %d is %d bytes, this test passes %zu", i, info.params[i].size, want);
    memcpy(params + info.params[i].offset, src, want);
    if ((size_t)info.params[i].offset + want > params_len) params_len = (size_t)info.params[i].offset + want;
  }
  if (fails) return fails;

  printf("launching %s grid=(%u,1,1) block=(%u,1,1) with %zu bytes of parameters\n", kernel, N / block, block, params_len);
  OK(tinynv_launch(s, k, N / block, 1, 1, block, 1, 1, 0, params, params_len));
  OK(tinynv_stream_sync(s));

  OK(tinynv_memcpy_dtoh(s, hc, c, sizeof(hc)));
  int wrong = 0, first = -1;
  for (int i = 0; i < N; i++)
    if (hc[i] != (float)(3 * i)) { if (first < 0) first = i; wrong++; }
  CHECK(!wrong, "%d of %d results are wrong; the first is c[%d] = %g where %g belongs", wrong, N, first,
        first >= 0 ? (double)hc[first] : 0.0, first >= 0 ? (double)(3 * first) : 0.0);
  if (!wrong) printf("all %d results correct: c[1] = %g, c[255] = %g\n", N, (double)hc[1], (double)hc[255]);

  OK(tinynv_free(dev, a));
  OK(tinynv_free(dev, b));
  OK(tinynv_free(dev, c));
  OK(tinynv_module_unload(mod));
  free(blob);
  printf(fails ? "%d checks failed\n" : "all checks passed: a kernel ran on the card and computed the right numbers\n", fails);
  return fails ? 1 : 0;
}
