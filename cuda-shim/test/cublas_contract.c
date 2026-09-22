// The type contract of libtinycublas's Ex entry points (G's review of the bf16 fix, 2026-09-21), on the null device: A and
// B of one type, a compute type cuBLAS defines for that input type, and C as that compute type allows. Anything else is
// NOT_SUPPORTED. Before this, A=f16 with B=bf16 ran on the f16 kernel (and the reverse on the bf16 one), and every compute
// type except exactly COMPUTE_16F had its alpha/beta read as float, 64F and 32I included.
// Each row is checked through all three entry points: GemmEx, GemmStridedBatchedEx (batch 2) and GemmBatchedEx (2 pointers).
#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
static int fails, rows;
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("  FAIL: " __VA_ARGS__); printf("\n"); } } while (0)
static cublasHandle_t h;
static void *A, *B, *C, **Ap, **Bp, **Cp;
static const float one_f = 1.f, zero_f = 0.f;
static const uint16_t one_h = 0x3c00, zero_h = 0;
static void row(const char *what, cudaDataType at, cudaDataType bt, cudaDataType ct, cublasComputeType_t cm, cublasStatus_t want) {
  const void *al = (cm == CUBLAS_COMPUTE_16F || cm == CUBLAS_COMPUTE_16F_PEDANTIC) ? (const void *)&one_h : (const void *)&one_f;
  const void *be = (cm == CUBLAS_COMPUTE_16F || cm == CUBLAS_COMPUTE_16F_PEDANTIC) ? (const void *)&zero_h : (const void *)&zero_f;
  cublasStatus_t s1 = cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, 32, 32, 32, al, A, at, 32, B, bt, 32, be, C, ct, 32, cm, CUBLAS_GEMM_DEFAULT);
  cublasStatus_t s2 = cublasGemmStridedBatchedEx(h, CUBLAS_OP_T, CUBLAS_OP_N, 32, 32, 32, al, A, at, 32, 1024, B, bt, 32, 1024, be, C, ct, 32,
                                                 1024, 2, cm, CUBLAS_GEMM_DEFAULT);
  cublasStatus_t s3 = cublasGemmBatchedEx(h, CUBLAS_OP_T, CUBLAS_OP_N, 32, 32, 32, al, (const void *const *)Ap, at, 32, (const void *const *)Bp, bt, 32,
                                          be, (void *const *)Cp, ct, 32, 2, cm, CUBLAS_GEMM_DEFAULT);
  rows++;
  int ok = s1 == want && s2 == want && s3 == want;
  printf("  %s %-48s GemmEx %d, StridedBatchedEx %d, BatchedEx %d (want %d)\n", ok ? "ok  " : "FAIL", what, s1, s2, s3, want);
  if (!ok) fails++;
}
int main(void) {
  CHECK(cublasCreate(&h) == CUBLAS_STATUS_SUCCESS, "cublasCreate");
  CHECK(cudaMalloc(&A, 1 << 16) == cudaSuccess && cudaMalloc(&B, 1 << 16) == cudaSuccess && cudaMalloc(&C, 1 << 16) == cudaSuccess, "malloc");
  void *hp[2]; CHECK(cudaMalloc((void **)&Ap, 16) == cudaSuccess && cudaMalloc((void **)&Bp, 16) == cudaSuccess && cudaMalloc((void **)&Cp, 16) == cudaSuccess, "ptr arrays");
  hp[0] = A; hp[1] = (char *)A + 4096; cudaMemcpy(Ap, hp, 16, cudaMemcpyHostToDevice);
  hp[0] = B; hp[1] = (char *)B + 4096; cudaMemcpy(Bp, hp, 16, cudaMemcpyHostToDevice);
  hp[0] = C; hp[1] = (char *)C + 8192; cudaMemcpy(Cp, hp, 16, cudaMemcpyHostToDevice);
  printf("served (SUCCESS on the null device, where launches are reported, not executed):\n");
  row("bf16 x bf16 -> f32, COMPUTE_32F (ggml's bf16)", CUDA_R_16BF, CUDA_R_16BF, CUDA_R_32F, CUBLAS_COMPUTE_32F, CUBLAS_STATUS_SUCCESS);
  row("bf16 x bf16 -> bf16, COMPUTE_32F", CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, CUBLAS_COMPUTE_32F, CUBLAS_STATUS_SUCCESS);
  row("f16 x f16 -> f16, COMPUTE_16F (ggml's f16)", CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUBLAS_COMPUTE_16F, CUBLAS_STATUS_SUCCESS);
  row("f16 x f16 -> f32, COMPUTE_32F", CUDA_R_16F, CUDA_R_16F, CUDA_R_32F, CUBLAS_COMPUTE_32F, CUBLAS_STATUS_SUCCESS);
  row("f32 x f32 -> f32, COMPUTE_32F_FAST_TF32", CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_STATUS_SUCCESS);
  printf("refused (NOT_SUPPORTED = %d):\n", CUBLAS_STATUS_NOT_SUPPORTED);
  row("f16 x bf16 (mismatched A/B)", CUDA_R_16F, CUDA_R_16BF, CUDA_R_32F, CUBLAS_COMPUTE_32F, CUBLAS_STATUS_NOT_SUPPORTED);
  row("bf16 x f16 (mismatched A/B)", CUDA_R_16BF, CUDA_R_16F, CUDA_R_32F, CUBLAS_COMPUTE_32F, CUBLAS_STATUS_NOT_SUPPORTED);
  row("f32 x f16 (mismatched A/B)", CUDA_R_32F, CUDA_R_16F, CUDA_R_32F, CUBLAS_COMPUTE_32F, CUBLAS_STATUS_NOT_SUPPORTED);
  row("bf16 with COMPUTE_16F", CUDA_R_16BF, CUDA_R_16BF, CUDA_R_32F, CUBLAS_COMPUTE_16F, CUBLAS_STATUS_NOT_SUPPORTED);
  row("f16 COMPUTE_16F with an f32 C", CUDA_R_16F, CUDA_R_16F, CUDA_R_32F, CUBLAS_COMPUTE_16F, CUBLAS_STATUS_NOT_SUPPORTED);
  row("f32 with COMPUTE_64F", CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUBLAS_COMPUTE_64F, CUBLAS_STATUS_NOT_SUPPORTED);
  row("f32 with COMPUTE_32I", CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUBLAS_COMPUTE_32I, CUBLAS_STATUS_NOT_SUPPORTED);
  row("bf16 with COMPUTE_32F_FAST_TF32", CUDA_R_16BF, CUDA_R_16BF, CUDA_R_32F, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_STATUS_NOT_SUPPORTED);
  printf("\n%d row(s), %d failure(s)\n", rows, fails);
  return fails ? 1 : 0;
}
