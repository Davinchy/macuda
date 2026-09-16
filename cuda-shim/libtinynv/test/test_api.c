// Walks tinynv.h exactly the way libtinycudart does: init, take the device, load a cubin, look the kernel up, ask for its
// parameter layout, then allocate, copy and launch. Checks the layout against the cubin the shim will really be handed.
#include "tinynv.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;
#define CHECK(cond, ...) do { if (!(cond)) { fails++; printf("FAIL %s:%d ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

int main(int argc, char **argv) {
  if (argc < 3) { fprintf(stderr, "usage: %s <cubin> <expected param_base>\n", argv[0]); return 2; }
  long want_base = strtol(argv[2], NULL, 0);

  FILE *f = fopen(argv[1], "rb");
  if (!f) { perror("open"); return 1; }
  fseek(f, 0, SEEK_END);
  long len = ftell(f);
  fseek(f, 0, SEEK_SET);
  void *cubin = malloc((size_t)len);
  if (fread(cubin, 1, (size_t)len, f) != (size_t)len) { perror("read"); return 1; }
  fclose(f);

  CHECK(tinynv_init() == TINYNV_OK, "init");
  CHECK(tinynv_device_count() >= 1, "no devices");

  tinynv_device_t dev;
  CHECK(tinynv_device_get(&dev, 0) == TINYNV_OK, "device_get");
  tinynv_device_props_t props;
  CHECK(tinynv_device_props(dev, &props) == TINYNV_OK, "props");
  printf("device: %s\n", props.name);

  tinynv_module_t mod;
  CHECK(tinynv_module_load(dev, cubin, (size_t)len, &mod) == TINYNV_OK, "module_load");
  tinynv_kernel_t k;
  CHECK(tinynv_get_kernel(mod, "vecadd", &k) == TINYNV_OK, "get_kernel");
  CHECK(tinynv_get_kernel(mod, "nosuchkernel", &k) != TINYNV_OK || 0, "a missing kernel must not be found");
  CHECK(tinynv_get_kernel(mod, "vecadd", &k) == TINYNV_OK, "get_kernel again");

  tinynv_kernel_info_t info;
  CHECK(tinynv_kernel_info(k, &info) == TINYNV_OK, "kernel_info");
  CHECK(info.param_base == (int)want_base, "param_base %#x, expected %#lx", info.param_base, want_base);
  CHECK(info.num_params == 4, "%d params", info.num_params);
  const int off[] = {0, 8, 16, 24}, sz[] = {8, 8, 8, 4};
  for (int i = 0; i < 4 && i < info.num_params; i++) {
    CHECK(info.params[i].offset == off[i], "param %d offset %d", i, info.params[i].offset);
    CHECK(info.params[i].size == sz[i], "param %d size %d", i, info.params[i].size);
  }
  CHECK(info.regs > 0, "regs %d", info.regs);

  // the rest of the flow: three buffers, a copy in, the launch the shim would issue for vecadd<<<4,64>>>
  tinynv_stream_t stream;
  CHECK(tinynv_stream_create(dev, &stream) == TINYNV_OK, "stream_create");
  tinynv_devptr_t a, b, c;
  CHECK(tinynv_malloc(dev, 256 * 4, &a) == TINYNV_OK, "malloc a");
  CHECK(tinynv_malloc(dev, 256 * 4, &b) == TINYNV_OK, "malloc b");
  CHECK(tinynv_malloc(dev, 256 * 4, &c) == TINYNV_OK, "malloc c");

  // Alignment is part of the contract, so it is checked rather than hoped for. Awkward sizes on purpose: a run of
  // allocations that all happen to land aligned proves nothing, and this is exactly how the bug got through once -
  // it passed on the luck of the heap and failed on the next run at a different call.
  {
    static const size_t awkward[] = {1, 17, 33, 129, 255, 257, 1000, 4095};
    int misaligned = 0;
    tinynv_devptr_t p[sizeof(awkward) / sizeof(*awkward)];
    for (size_t i = 0; i < sizeof(awkward) / sizeof(*awkward); i++) {
      CHECK(tinynv_malloc(dev, awkward[i], &p[i]) == TINYNV_OK, "malloc %zu bytes", awkward[i]);
      if ((uintptr_t)p[i] % TINYNV_DEVPTR_ALIGN) {
        printf("  FAIL: a %zu byte allocation landed at %#llx, which is not %d aligned\n", awkward[i],
               (unsigned long long)(uintptr_t)p[i], TINYNV_DEVPTR_ALIGN);
        misaligned++;
      }
    }
    for (size_t i = 0; i < sizeof(awkward) / sizeof(*awkward); i++) tinynv_free(dev, p[i]);
    fails += misaligned;
    if (!misaligned) printf("device pointers: %zu awkward sizes, all %d aligned\n",
                            sizeof(awkward) / sizeof(*awkward), TINYNV_DEVPTR_ALIGN);
  }

  float host[256];
  for (int i = 0; i < 256; i++) host[i] = (float)i;
  CHECK(tinynv_memcpy_htod(stream, a, host, sizeof(host)) == TINYNV_OK, "htod");
  memset(host, 0, sizeof(host));
  CHECK(tinynv_memcpy_dtoh(stream, host, a, sizeof(host)) == TINYNV_OK, "dtoh");
  CHECK(host[255] == 255.0f, "copy round trip gave %f", (double)host[255]);

  // the blob the classic launch ABI accumulates: three pointers then the count, which is what the cubin declares
  struct { uint64_t a, b, c; int n; } args = {a, b, c, 256};
  CHECK(tinynv_launch(stream, k, 4, 1, 1, 64, 1, 1, 0, &args, 28) == TINYNV_OK, "launch");
  CHECK(tinynv_launch(stream, k, 4, 1, 1, 2048, 1, 1, 0, &args, 28) != TINYNV_OK, "a block of 2048 threads must be refused");
  CHECK(tinynv_stream_wait_event(stream, NULL) == TINYNV_OK, "wait_event on the null device");
  CHECK(tinynv_launch(stream, k, 1, 1, 1, 64, 1, 1, 0, &args, 1024) != TINYNV_OK, "too many parameter bytes must be refused");

  CHECK(tinynv_stream_sync(stream) == TINYNV_OK, "sync");
  CHECK(tinynv_stream_destroy(stream) == TINYNV_OK, "stream_destroy");
  CHECK(tinynv_free(dev, a) == TINYNV_OK, "free");
  CHECK(tinynv_module_unload(mod) == TINYNV_OK, "module_unload");

  free(cubin);
  printf(fails ? "%d checks failed\n" : "all checks passed\n", fails);
  return fails ? 1 : 0;
}
