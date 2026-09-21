// One property per kernel, so each attribute the driver reports can be tied to one change in the image.
__constant__ float cst[512];
extern "C" __global__ void k_plain(float *o) { o[threadIdx.x] = threadIdx.x; }
extern "C" __global__ void k_local(float *o, int n) {
  float buf[64];                                   // indexed by a runtime value: lives in local memory
  for (int i = 0; i < 64; i++) buf[i] = o[i] * i;
  o[threadIdx.x] = buf[(n + threadIdx.x) & 63];
}
extern "C" __global__ void k_const(float *o) { o[threadIdx.x] = cst[threadIdx.x & 511]; }
extern "C" __global__ void __launch_bounds__(128, 4) k_bounds(float *o) { o[threadIdx.x] = 2.0f * threadIdx.x; }
extern "C" __global__ void k_bigstack(float *o, int n) {
  float buf[1024];                                 // too large to keep in registers: a real per-thread stack
  for (int i = 0; i < 1024; i++) buf[(i * n + threadIdx.x) & 1023] = o[i & 255] + i;
  o[threadIdx.x] = buf[(n * threadIdx.x) & 1023];
}
#include <cstdio>
__device__ __noinline__ float callee(const float *o, int n) {
  float t[256];                                    // the CALLEE's frame, not the kernel's
  for (int i = 0; i < 256; i++) t[(i * n) & 255] = o[i & 63] * i;
  return t[n & 255];
}
extern "C" __global__ void k_calls(float *o, int n) { o[threadIdx.x] = callee(o, n + threadIdx.x); }
extern "C" __global__ void k_printf(float *o, int n) { if (n == 12345) printf("%d %f\n", n, o[0]); o[threadIdx.x] = 1; }
