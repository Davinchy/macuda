// Self-contained: pull in cuda_runtime.h then declare the classic launch ABI clang emits (removed from CUDA 13 headers).
#pragma once
#include <cuda_runtime.h>
extern "C" {
  cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem = 0, cudaStream_t stream = 0);
  cudaError_t cudaSetupArgument(const void* arg, size_t size, size_t offset);
  cudaError_t cudaLaunch(const void* func);
}
