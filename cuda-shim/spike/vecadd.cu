#include <cuda_runtime.h>
#include "tinycudart_compat.h"
// Minimal CUDA TU to prove the clang host/device split (device compiled on Linux, host compiled on Mac).
extern "C" __global__ void vecadd(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}
#include <cstdio>
extern "C" int run_vecadd_host();   // defined in host driver code below via clang's <<<>>> lowering
int main() {
  // host code that clang lowers to __cudaRegisterFatBinary + cudaLaunchKernel against our stub cudart
  const int n = 256; float *a=nullptr,*b=nullptr,*c=nullptr;
  cudaMalloc((void**)&a, n*sizeof(float));
  cudaMalloc((void**)&b, n*sizeof(float));
  cudaMalloc((void**)&c, n*sizeof(float));
  dim3 grid((n+63)/64), block(64);
  vecadd<<<grid, block>>>(a, b, c, n);
  cudaDeviceSynchronize();
  printf("host reached: launch issued via cudaLaunchKernel\n");
  return 0;
}
