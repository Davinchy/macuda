// bw_bench — streaming read and write bandwidth of device memory through the shim's page tables, one direction at a time.
//   bw_bench [MiB=256] [launches=50]
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>
#include "tinynv.h"
extern unsigned char gemm_cubin[]; extern unsigned int gemm_cubin_len;
extern tinynv_device_t tinycudart_device(void); extern tinynv_stream_t tinycudart_default_stream(void);
static double now(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+ts.tv_nsec*1e-9; }
int main(int argc, char** argv){
  size_t mib = argc>1 ? (size_t)atoi(argv[1]) : 256; int reps = argc>2 ? atoi(argv[2]) : 50;
  size_t bytes = mib<<20; long long n4 = (long long)(bytes/16);
  float *buf, *out; cudaMalloc((void**)&buf, bytes); cudaMalloc((void**)&out, 4096); cudaMemset(buf, 0, bytes);
  tinynv_module_t mod; if (tinynv_module_load(tinycudart_device(), gemm_cubin, gemm_cubin_len, &mod)!=TINYNV_OK){ fprintf(stderr,"module_load failed\n"); return 1; }
  tinynv_kernel_t kw, kr; if (tinynv_get_kernel(mod,"tinyblas_bw_write",&kw)!=TINYNV_OK || tinynv_get_kernel(mod,"tinyblas_bw_read",&kr)!=TINYNV_OK){ fprintf(stderr,"kernels missing\n"); return 1; }
  tinynv_stream_t s = tinycudart_default_stream();
  struct __attribute__((packed)) { uint64_t p; int64_t n4; float v; } pw = { (uint64_t)buf, n4, 1.0f };
  struct __attribute__((packed)) { uint64_t p; int64_t n4; uint64_t out; } pr = { (uint64_t)buf, n4, (uint64_t)out };
  unsigned grid = 170*8, block = 256;
  // warm-up one of each, then time
  tinynv_launch(s,kw,grid,1,1,block,1,1,0,&pw,sizeof pw); tinynv_launch(s,kr,grid,1,1,block,1,1,0,&pr,sizeof pr); cudaDeviceSynchronize();
  double t0=now(); for (int i=0;i<reps;i++) tinynv_launch(s,kw,grid,1,1,block,1,1,0,&pw,sizeof pw); cudaDeviceSynchronize(); double tw=now()-t0;
  t0=now(); for (int i=0;i<reps;i++) tinynv_launch(s,kr,grid,1,1,block,1,1,0,&pr,sizeof pr); cudaDeviceSynchronize(); double tr=now()-t0;
  printf("write-only: %zu MiB x %d = %.1f GB/s (%.1f us per launch)\n", mib, reps, (double)bytes*reps/tw/1e9, tw/reps*1e6);
  printf("read-only:  %zu MiB x %d = %.1f GB/s (%.1f us per launch)\n", mib, reps, (double)bytes*reps/tr/1e9, tr/reps*1e6);
  return 0;
}
