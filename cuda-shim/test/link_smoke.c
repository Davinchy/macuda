// Links libtinycudart + libtinycublas + libtinynv.a through the REAL CUDA headers' prototypes (so our C ABI matches what
// clang-compiled ggml will call), and exercises the pinned host pool on the null device: allocation churn must not
// grow the number of driver mappings, which is the whole point of the pool (the dext caps a session at 128).
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include "hostpool.h"
static int fails;
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("  FAIL: " __VA_ARGS__); printf("\n"); } } while (0)
int main(void) {
  int n = 0; CHECK(cudaGetDeviceCount(&n) == cudaSuccess && n >= 1, "device count %d", n);
  tinycudart_hostpool_stats_t st;
  // 1. churn: 300 lifetime allocations must cost ONE driver mapping
  for (int i = 0; i < 300; i++) { void *p = NULL; CHECK(cudaMallocHost(&p, 1 << 20) == cudaSuccess && p, "mallochost %d", i); memset(p, i, 1 << 20); CHECK(cudaFreeHost(p) == cudaSuccess, "freehost %d", i); }
  tinycudart_hostpool_stats(&st);
  CHECK(st.slabs == 1 && st.lifetime_allocs == 300 && st.bytes_in_use == 0, "after churn: slabs=%zu lifetime=%zu in_use=%zu", st.slabs, st.lifetime_allocs, st.bytes_in_use);
  // 2. fragmentation: fill a slab, free every other block, refill the holes with smaller blocks, still one slab
  void *big[8]; for (int i = 0; i < 8; i++) CHECK(cudaMallocHost(&big[i], 8 << 20) == cudaSuccess, "8 MB #%d", i);
  for (int i = 0; i < 8; i += 2) CHECK(cudaFreeHost(big[i]) == cudaSuccess, "free #%d", i);
  void *small[4]; for (int i = 0; i < 4; i++) CHECK(cudaMallocHost(&small[i], 4 << 20) == cudaSuccess, "4 MB #%d", i);
  tinycudart_hostpool_stats(&st); CHECK(st.slabs == 1, "holes reused: slabs=%zu", st.slabs);
  for (int i = 1; i < 8; i += 2) cudaFreeHost(big[i]); for (int i = 0; i < 4; i++) cudaFreeHost(small[i]);
  tinycudart_hostpool_stats(&st); CHECK(st.bytes_in_use == 0, "all freed: in_use=%zu", st.bytes_in_use);
  // 3. a request larger than a slab gets a dedicated mapping, which is reused after free
  void *huge = NULL; CHECK(cudaHostAlloc(&huge, 100 << 20, cudaHostAllocPortable) == cudaSuccess, "100 MB");
  tinycudart_hostpool_stats(&st); CHECK(st.slabs == 2, "dedicated: slabs=%zu", st.slabs);
  cudaFreeHost(huge); CHECK(cudaMallocHost(&huge, 100 << 20) == cudaSuccess, "100 MB again");
  tinycudart_hostpool_stats(&st); CHECK(st.slabs == 2, "dedicated reused: slabs=%zu", st.slabs); cudaFreeHost(huge);
  // 4. the unsupported host APIs fail the way ggml expects (error, then cudaGetLastError clears it)
  char buf[4096]; CHECK(cudaHostRegister(buf, sizeof buf, cudaHostRegisterPortable) == cudaErrorNotSupported, "HostRegister");
  CHECK(cudaGetLastError() == cudaErrorNotSupported && cudaGetLastError() == cudaSuccess, "last error clears");
  CHECK(cudaHostUnregister(buf) == cudaSuccess, "HostUnregister"); CHECK(cudaFreeHost(NULL) == cudaSuccess, "free NULL");
  // 5. cuBLAS links, finds its kernel in the embedded cubin, and launches on the null device
  cublasHandle_t h = NULL; CHECK(cublasCreate(&h) == CUBLAS_STATUS_SUCCESS && h, "cublasCreate");
  // A 48x48x48 GEMM: the kernel's 16x16 blocks give a 3x3 grid, so this launch is MULTI-block on purpose. On Blackwell a kernel
  // reads blockDim/gridDim out of constant bank 0, written by the driver; a driver that gets that wrong makes every block
  // compute block 0's tile, and a single-block test would pass anyway (Session B's finding, 03:15). All 2304 results checked.
  enum { N = 48 }; float alpha = 1, beta = 0; void *A, *B, *C; cudaMalloc(&A, N * N * 4); cudaMalloc(&B, N * N * 4); cudaMalloc(&C, N * N * 4);
  static float hA[N * N], hB[N * N], hC[N * N], want[N * N];
  for (int i = 0; i < N * N; i++) { hA[i] = (float)(i % 7) - 3; hB[i] = (float)((i * 5) % 4) - 1; }
  for (int col = 0; col < N; col++) for (int row = 0; row < N; row++) { float acc = 0; for (int k = 0; k < N; k++) acc += hA[row + k * N] * hB[k + col * N]; want[row + col * N] = acc; } // column-major
  CHECK(cudaMemcpy(A, hA, sizeof hA, cudaMemcpyHostToDevice) == cudaSuccess && cudaMemcpy(B, hB, sizeof hB, cudaMemcpyHostToDevice) == cudaSuccess, "gemm inputs");
  CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, A, N, B, N, &beta, C, N) == CUBLAS_STATUS_SUCCESS, "Sgemm");
  CHECK(cudaDeviceSynchronize() == cudaSuccess && cudaMemcpy(hC, C, sizeof hC, cudaMemcpyDeviceToHost) == cudaSuccess, "gemm readback");
  { struct cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    if (strstr(p.name, "null")) printf("gemm result check skipped on the null device (nothing executes)\n");
    else { int bad = 0, first = -1; for (int i = 0; i < N * N; i++) if (hC[i] != want[i]) { if (first < 0) first = i; bad++; }
           CHECK(bad == 0, "GEMM on hardware: %d of %d results wrong, first at [%d,%d] (got %g want %g)", bad, N * N, first % N, first / N, first >= 0 ? hC[first] : 0, first >= 0 ? want[first] : 0);
           if (!bad) printf("GEMM on hardware: %d of %d results correct across a 3x3 grid of blocks -- the stack computed something real\n", N * N, N * N); } }
  { // batched triangular solve: the pointer arrays live in device memory, so they go through cudaMalloc too
    void *Aarr, *Barr; cudaMalloc(&Aarr, 2 * sizeof(void*)); cudaMalloc(&Barr, 2 * sizeof(void*));
    void *ptrs[2] = { A, B }; cudaMemcpy(Aarr, ptrs, sizeof ptrs, cudaMemcpyHostToDevice); cudaMemcpy(Barr, ptrs, sizeof ptrs, cudaMemcpyHostToDevice);
    CHECK(cublasStrsmBatched(h, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, 4, 4, &alpha, (const float * const *)Aarr, 4, (float * const *)Barr, 4, 2) == CUBLAS_STATUS_SUCCESS, "StrsmBatched");
    CHECK(cublasStrsmBatched(h, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, 4, 4, &alpha, (const float * const *)Aarr, 4, (float * const *)Barr, 4, 2) == CUBLAS_STATUS_NOT_SUPPORTED, "StrsmBatched refuses the unimplemented side");
  }
  { // f16-output GemmEx: the path that was wrong (Ctype=CUDA_R_16F, ggml-cuda.cu:1397). Value-checked on a real device
    // against a reference built from the SAME f16-rounded inputs (the GPU reads f16, accumulates f32), two-sided tolerance.
    enum { M4 = 4 };
    __fp16 *Ah16=(__fp16*)malloc(M4*M4*2), *Bh16=(__fp16*)malloc(M4*M4*2);
    for (int i=0;i<M4*M4;i++){ Ah16[i]=(__fp16)((float)(i%7)-3); Bh16[i]=(__fp16)((float)((i*5)%4)-1); }
    float want16[M4*M4];
    for (int col=0; col<M4; col++) for (int row=0; row<M4; row++) { float acc=0; for (int kk=0;kk<M4;kk++) acc += (float)Ah16[row+kk*M4]*(float)Bh16[kk+col*M4]; want16[row+col*M4]=acc; }
    void *Ad,*Bd,*Cd; cudaMalloc(&Ad,M4*M4*2); cudaMalloc(&Bd,M4*M4*2); cudaMalloc(&Cd,M4*M4*2);
    cudaMemcpy(Ad,Ah16,M4*M4*2,cudaMemcpyHostToDevice); cudaMemcpy(Bd,Bh16,M4*M4*2,cudaMemcpyHostToDevice);
    __fp16 alpha16=(__fp16)1.0f, beta16=(__fp16)0.0f;   // ggml's f16 matmul passes COMPUTE_16F, so alpha/beta are HALF, not float
    CHECK(cublasGemmEx(h,CUBLAS_OP_N,CUBLAS_OP_N,M4,M4,M4,&alpha16,Ad,CUDA_R_16F,M4,Bd,CUDA_R_16F,M4,&beta16,Cd,CUDA_R_16F,M4,CUBLAS_COMPUTE_16F,CUBLAS_GEMM_DEFAULT)==CUBLAS_STATUS_SUCCESS,"GemmEx f16 out");
    struct cudaDeviceProp pp; cudaGetDeviceProperties(&pp,0);
    if (!strstr(pp.name,"null")) { __fp16 hCd[M4*M4]; cudaDeviceSynchronize(); cudaMemcpy(hCd,Cd,M4*M4*2,cudaMemcpyDeviceToHost);
      int bad=0, first=-1; for (int i=0;i<M4*M4;i++){ float g=(float)hCd[i]; if (g!=g || fabsf(g-want16[i]) > 0.01f*fabsf(want16[i])+0.01f) { if(first<0)first=i; bad++; } }
      CHECK(bad==0,"f16-out GEMM on hardware: %d of %d wrong, first [%d,%d] (got %g want %g)",bad,M4*M4,first%M4,first/M4,first>=0?(float)hCd[first]:0,first>=0?want16[first]:0);
      if(!bad) printf("f16-out GEMM on hardware: %d of %d correct -- Ctype=CUDA_R_16F honoured\n",M4*M4,M4*M4); }
    free(Ah16); free(Bh16); }
  cublasDestroy(h);
  // 6. streams, events, async copies, fills, queries, attributes, occupancy, and an explicit launch with marshalled args
  cudaStream_t s1 = NULL; cudaEvent_t ev = NULL; float hbuf[64]; memset(hbuf, 0, sizeof hbuf);
  CHECK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking) == cudaSuccess && s1, "stream create");
  CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming) == cudaSuccess && ev, "event create");
  CHECK(cudaMemcpyAsync(A, hbuf, 64, cudaMemcpyHostToDevice, s1) == cudaSuccess, "memcpyAsync h2d");
  CHECK(cudaMemcpy2DAsync(B, 32, hbuf, 16, 16, 2, cudaMemcpyHostToDevice, s1) == cudaSuccess, "memcpy2DAsync");
  CHECK(cudaMemsetAsync(C, 0, 64, s1) == cudaSuccess && cudaMemset(C, 1, 64) == cudaSuccess, "memset");
  CHECK(cudaEventRecord(ev, s1) == cudaSuccess && cudaStreamWaitEvent(NULL, ev, 0) == cudaSuccess && cudaEventSynchronize(ev) == cudaSuccess, "event record/wait/sync");
  CHECK(cudaStreamSynchronize(s1) == cudaSuccess && cudaStreamDestroy(s1) == cudaSuccess && cudaEventDestroy(ev) == cudaSuccess, "stream sync/destroy");
  struct cudaDeviceProp prop; CHECK(cudaGetDeviceProperties(&prop, 0) == cudaSuccess && prop.warpSize == 32 && prop.maxThreadsPerBlock >= 1024, "device props (%s, warp %d)", prop.name, prop.warpSize);
  int coop = 1; CHECK(cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, 0) == cudaSuccess && coop == 0, "cooperative launch attr must be 0");
  size_t fr = 0, tot = 0; CHECK(cudaMemGetInfo(&fr, &tot) == cudaSuccess, "memgetinfo");
  CHECK(cudaStreamBeginCapture(NULL, cudaStreamCaptureModeGlobal) == cudaErrorNotSupported && cudaGetLastError() == cudaErrorNotSupported, "graph capture refused");
  CHECK(cudaGetLastError() == cudaSuccess && cudaPeekAtLastError() == cudaSuccess, "error state clean");
  printf("runtime surface: streams, events, async copies, fills, props, attrs exercised on the null device\n");
  tinycudart_hostpool_stats(&st);
  printf("hostpool: %zu driver mappings for %zu lifetime allocations, %zu MB mapped\n", st.slabs, st.lifetime_allocs, st.bytes_mapped >> 20);
  printf("link smoke: %s\n", fails ? "FAILED" : "all checks passed");
  return fails ? 1 : 0;
}
