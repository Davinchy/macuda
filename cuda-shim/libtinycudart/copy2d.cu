// copy2d.cu — the strided device-to-device copy behind cudaMemcpy2DAsync, as ONE kernel launch.
//
// ggml's same-type tensor copies (cont/permute paths) reduce to "height rows of width contiguous bytes" and call
// cudaMemcpy2DAsync. The first version of that entry point issued one copy-engine batch per row, and every batch is a
// submit with a socket round trip in it, so a copy of a few thousand rows cost a few thousand round trips: SD 1.5 spent
// 99.6% of its host time there (docs/SHARED-STATUS.md 2026-09-15 ~02:00). A launch costs about a microsecond and the
// kernel touches the bytes at memory speed, which is what a copy should cost.
//
// vec is the widest unit that src, dst, both pitches and width are all multiples of: 16, 8, 4 or 1. Rows beyond 65535
// are chunked by the host (grid.y limit); a thread strides over its row in vec-sized units.
// Build: nvcc -arch=sm_120 -O3 -cubin -o copy2d.cubin copy2d.cu   (remote nvcc on the Pop!_OS box)
#include <cuda_runtime.h>
#include <stdint.h>
extern "C" __global__ void __launch_bounds__(256) tinycudart_copy2d(const unsigned char* __restrict__ src, unsigned char* __restrict__ dst,
                                                                  long long spitch, long long dpitch, long long width, long long height, int vec) {
  long long row = blockIdx.y;
  if (row >= height) return;
  const unsigned char* s = src + row * spitch;
  unsigned char* d = dst + row * dpitch;
  long long units = width / vec;
  long long stride = (long long)gridDim.x * blockDim.x;
  for (long long u = (long long)blockIdx.x * blockDim.x + threadIdx.x; u < units; u += stride) {
    switch (vec) {
      case 16: ((uint4*)d)[u] = ((const uint4*)s)[u]; break;
      case 8:  ((uint2*)d)[u] = ((const uint2*)s)[u]; break;
      case 4:  ((unsigned int*)d)[u] = ((const unsigned int*)s)[u]; break;
      default: d[u] = s[u]; break;
    }
  }
}
