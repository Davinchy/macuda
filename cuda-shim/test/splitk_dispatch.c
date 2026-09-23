// Split-K dispatch rows (B, 2026-09-23; root docs/review/20260923-smalln-design-B.md §3 rows 1, 2 and 6). Card-free: the
// null device reports launches and executes nothing. Part 1 asserts the plan, a pure function of the shape, at the
// design's table and at every boundary it introduces. Part 2 issues the design's five decode shapes through cublasGemmEx
// so that test/splitk_check.sh can read the launch trace in both arms (TINYCUBLAS_SPLITK=1 and unset).
#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
int tinycublas_splitk_plan(int m, int n, int k, int batch);
unsigned tinycublas_splitk_ws_allocs(void);
static int fails, rows;
static void plan(const char *what, int m, int n, int k, int batch, int want) {
  int got = tinycublas_splitk_plan(m, n, k, batch); rows++;
  printf("  %s %-44s m=%-5d n=%-3d k=%-5d batch=%d -> S=%d (want %d)\n", got == want ? "ok  " : "FAIL", what, m, n, k, batch, got, want);
  if (got != want) fails++;
}
int main(void) {
  printf("plan rows (pure function):\n");
  plan("k/v (3B decode)", 256, 32, 2048, 1, 16);
  plan("q/o (3B decode)", 2048, 32, 2048, 1, 8);
  plan("down (3B decode)", 2048, 32, 11008, 1, 8);
  plan("gate/up (3B decode): gx 172, never split", 11008, 32, 2048, 1, 1);
  plan("diffusion mix: gy 4, never split", 1280, 256, 1280, 1, 1);
  plan("boundary gx 169 splits", 169 * 64, 32, 2048, 1, 2);
  plan("boundary gx 170 does not", 170 * 64, 32, 2048, 1, 1);
  plan("boundary n 64 (gy 1) splits", 2048, 64, 2048, 1, 8);
  plan("boundary n 65 (gy 2) does not", 2048, 65, 2048, 1, 1);
  plan("k 2047 cannot keep 4 tiles in 16 slices", 256, 32, 2047, 1, 8);
  plan("k 255: 4 tiles need k >= 256 for S 2", 2048, 32, 255, 1, 1);
  plan("workspace bound: batch 3 of q/o is 12 MiB", 2048, 32, 2048, 3, 1);
  plan("workspace bound: batch 2 of q/o is 8 MiB", 2048, 32, 2048, 2, 8);
  printf("\n%d plan row(s), %d failure(s)\n", rows, fails);
  // D's unit row (04:5x): the workspace is made at a handle's FIRST splitting GEMM. hN never splits (gate/up, diffusion) and
  // must make 0 allocations; hS splits three times (k/v, q/o, down) and must make exactly 1. splitk_check.sh reads the lines.
  cublasHandle_t hN, hS;
  if (cublasCreate(&hN) != CUBLAS_STATUS_SUCCESS || cublasCreate(&hS) != CUBLAS_STATUS_SUCCESS) { printf("FAIL: cublasCreate\n"); return 1; }
  printf("ws allocs after cublasCreate x2: %u\n", tinycublas_splitk_ws_allocs());
  void *A, *B, *C;
  if (cudaMalloc(&A, 64 << 20) || cudaMalloc(&B, 4 << 20) || cudaMalloc(&C, 8 << 20)) { printf("FAIL: malloc\n"); return 1; }
  const float one = 1.f, zero = 0.f;
  const int shapes[5][3] = { {11008, 32, 2048}, {1280, 256, 1280}, {256, 32, 2048}, {2048, 32, 2048}, {2048, 32, 11008} };
  for (int i = 0; i < 5; i++) {
    int m = shapes[i][0], n = shapes[i][1], k = shapes[i][2];
    cublasHandle_t h = i < 2 ? hN : hS;
    if (i == 2) printf("ws allocs after the non-splitting handle's GEMMs: %u\n", tinycublas_splitk_ws_allocs());
    cublasStatus_t s = cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &one, A, CUDA_R_16BF, k, B, CUDA_R_16BF, k, &zero, C, CUDA_R_32F, m,
                                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    printf("gemm m=%d n=%d k=%d status %d\n", m, n, k, (int)s);
    if (s != CUBLAS_STATUS_SUCCESS) fails++;
  }
  printf("ws allocs after the splitting handle's 3 GEMMs: %u\n", tinycublas_splitk_ws_allocs());
  cublasDestroy(hN); cublasDestroy(hS);
  return fails;
}
