// Does a kernel get told the truth about its own launch?
//
// This exists because of a bug that every other test passed. On this architecture a CUDA kernel reads blockDim and
// gridDim out of the parameter region of constant bank 0, not from a register, and if the driver does not put them
// there they read as zero. Nothing fails: `i = blockIdx.x * blockDim.x + threadIdx.x` becomes `i = threadIdx.x`, so a
// launch of one block is exactly right and every larger launch quietly writes over the first block's output.
//
// A vector add of the right size passes. A matrix multiply with one block passes. The only thing that catches it is
// asking a kernel what it was told, which is what this does - every block of a three dimensional grid, all six values,
// compared against what was asked for.
#include "tinynv.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// deliberately all different, and none of them a power of two in every position: a grid that is right by symmetry is
// not right
#define GX 4
#define GY 3
#define GZ 2
#define BX 32
#define BY 5
#define BZ 3
#define NBLOCK (GX * GY * GZ)
#define PER 8

int main(int argc, char **argv) {
  const char *path = argc > 1 ? argv[1] : "../spike/dims.sm120.cubin";
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
  tinynv_device_t dev;
  tinynv_module_t mod;
  tinynv_kernel_t k;
  tinynv_kernel_info_t info;
  tinynv_stream_t s;
  tinynv_devptr_t out;
  tinynv_status_t st;
#define T(c) do { if ((st = (c)) != TINYNV_OK) { printf("  %s: %s\n    %s\n", #c, tinynv_status_str(st), tinynv_last_error()); return 1; } } while (0)
  T(tinynv_init());
  T(tinynv_device_get(&dev, 0));
  T(tinynv_module_load(dev, blob, (size_t)len, &mod));
  T(tinynv_get_kernel(mod, "dims", &k));
  T(tinynv_kernel_info(k, &info));
  T(tinynv_stream_create(dev, &s));
  T(tinynv_malloc(dev, NBLOCK * PER * sizeof(unsigned), &out));

  static unsigned r[NBLOCK * PER];
  for (int i = 0; i < NBLOCK * PER; i++) r[i] = 0xdeadbeefu;
  T(tinynv_memcpy_htod(s, out, r, sizeof(r)));

  unsigned char params[64];
  memset(params, 0, sizeof(params));
  uint64_t p = (uint64_t)out;
  memcpy(params + info.params[0].offset, &p, sizeof(p));
  T(tinynv_launch(s, k, GX, GY, GZ, BX, BY, BZ, 0, params, (size_t)info.params[0].offset + sizeof(p)));
  T(tinynv_stream_sync(s));
  T(tinynv_memcpy_dtoh(s, r, out, sizeof(r)));

  const unsigned want[6] = {GX, GY, GZ, BX, BY, BZ};
  static const char *name[6] = {"gridDim.x", "gridDim.y", "gridDim.z", "blockDim.x", "blockDim.y", "blockDim.z"};
  int missing = 0, wrong = 0;
  for (int b = 0; b < NBLOCK; b++) {
    const unsigned *v = r + b * PER;
    if (v[7] != 0xa5a5a5a5u) { missing++; continue; }
    for (int j = 0; j < 6; j++)
      if (v[j] != want[j]) {
        if (!wrong) printf("  FAIL: block %d was told %s = %u, and it is %u\n", b, name[j], v[j], want[j]);
        wrong++;
      }
  }
  if (missing) printf("  FAIL: %d of %d blocks never ran\n", missing, NBLOCK);
  if (!missing && !wrong)
    printf("all %d blocks of a (%d,%d,%d) grid of (%d,%d,%d) blocks ran and were told the truth about all six\n",
           NBLOCK, GX, GY, GZ, BX, BY, BZ);
  free(blob);
  return missing || wrong ? 1 : 0;
}
