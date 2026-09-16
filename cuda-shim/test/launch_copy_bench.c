// launch_copy_bench — what does a launch cost when the stream also carries copies? A diffusion step through the shim is
// ~5000 launches, ~128 device-to-device cudaMemcpyAsync and ~11 synchronizations, and runs at ~95 us per launch where LLM
// decode (launches only) runs at ~11. This isolates the difference: N small GEMM launches alone, then the same with a
// d2d copy every K launches, then with a memset every K. Wall time per launch tells which shape of traffic is expensive.
//   launch_copy_bench <N> <K> <mode: launches|copies|memsets|copies-small>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
static double now(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+ts.tv_nsec*1e-9; }
int main(int argc, char** argv){
  int N = argc>1 ? atoi(argv[1]) : 4000, K = argc>2 ? atoi(argv[2]) : 40; const char* mode = argc>3 ? argv[3] : "launches";
  size_t big = strcmp(mode,"copies-small")==0 ? 4096 : (size_t)1<<20;
  float *A,*B,*C,*X,*Y; cudaMalloc((void**)&A,64*64*4); cudaMalloc((void**)&B,64*64*4); cudaMalloc((void**)&C,64*64*4); cudaMalloc((void**)&X,big); cudaMalloc((void**)&Y,big);
  cudaMemset(A,0,64*64*4); cudaMemset(B,0,64*64*4); cudaMemset(C,0,64*64*4); cudaMemset(X,0,big);
  cublasHandle_t h; cublasCreate(&h); float one=1.f, zero=0.f; cudaStream_t s=0;
  // warm up: the first launch loads the module
  cublasSgemm(h,CUBLAS_OP_N,CUBLAS_OP_N,64,64,64,&one,A,64,B,64,&zero,C,64); cudaStreamSynchronize(s);
  double t0=now(); int copies=0;
  for (int i=0;i<N;i++){
    cublasSgemm(h,CUBLAS_OP_N,CUBLAS_OP_N,64,64,64,&one,A,64,B,64,&zero,C,64);
    if (i%K==K-1){
      if (strncmp(mode,"copies",6)==0){ cudaMemcpyAsync(Y,X,big,cudaMemcpyDeviceToDevice,s); copies++; }
      else if (strcmp(mode,"memsets")==0){ cudaMemsetAsync(Y,0,big,s); copies++; }
    }
  }
  cudaStreamSynchronize(s); double t1=now();
  printf("%s: %d launches, %d extra ops (%s, %zu bytes each): %.1f ms total = %.1f us per launch\n", mode, N, copies, mode, big, (t1-t0)*1e3, (t1-t0)*1e6/N);
  return 0;
}
