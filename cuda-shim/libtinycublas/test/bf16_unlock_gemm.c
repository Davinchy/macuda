// bf16_unlock_gemm.c — BF16 correctness and timing workload for the tensor-unlock handoff.
//
// What this is: the plan Phase 4 "small known-correct BF16 GEMM" after the standalone unlock transaction, plus the
// stock baseline the same binary captures before it. It boots the UNCHANGED macuda runtime firmware through the
// normal shim path (cudaMalloc + cublasCreate), reads the unlock registers back through the live device
// (tensor_unlock_postboot, read-only), runs BF16 GEMMs through the shim's own cublasGemmEx / GemmStridedBatchedEx
// against a CPU double reference, reads the registers back again after the workload, and times one large GEMM for
// the stock/unlocked comparison. It changes no kernel, no cubin, no arithmetic and no tolerance: the gate is the
// bf16-20260921 protocol (max |x-ref| / max |ref|, any non-finite value refused first; 2e-3 for f32 output,
// 1e-2 for bf16 output — the output rounding alone is up to 2^-8 relative for bf16).
//
// The reference is computed on the CPU in double from the DECODED bf16 inputs, so it is independent of every GPU
// kernel: on this shim the cuBLAS entry points dispatch to the same tc2 kernels under test, which is why a
// GPU-side "reference" (bench_gemm_contract's real-cuBLAS comparison) cannot be the correctness authority here.
//
// Usage (the runner tools/rtx4090-live.sh gemm sets the environment and holds the GPU lock):
//   TINYNV_SOCKET=<sock> bf16_unlock_gemm [--expect stock|unlocked|any] [--no-bench]
// --expect gates the post-boot AND post-workload register state; default 'any' only reports it.
// Exit: 0 all cases within tolerance and the expected state verified; 1 a case failed or the state disagreed;
// 2 usage/setup; 3 a CUDA/cuBLAS call failed.
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "tinynv.h"
#include "tensor_unlock.h"

#define CR(x) do { cudaError_t r_ = (x); if (r_ != cudaSuccess) { printf("ERROR %s: %s (line %d)\n", #x, cudaGetErrorString(r_), __LINE__); exit(3); } } while (0)
#define CB(x) do { cublasStatus_t r_ = (x); if (r_ != CUBLAS_STATUS_SUCCESS) { printf("ERROR %s: cublas %d (line %d)\n", #x, (int)r_, __LINE__); exit(3); } } while (0)

static int fails, cases_run;

// ---- bf16 host codec (round-to-nearest-even, the same encoding the inputs carry) ------------------------------------
static uint16_t f2bf(float f) {
  uint32_t u; memcpy(&u, &f, 4);
  if (((u >> 23) & 0xff) == 0xff) return (uint16_t)(u >> 16); // Inf/NaN keep their shape
  uint32_t lsb = (u >> 16) & 1, r = (u & 0x7fff) + 0x7fff + lsb;
  return (uint16_t)((u + r) >> 16);
}
static float frand(void) { return (float)rand() / (float)RAND_MAX * 2.f - 1.f; }
static float bf2f(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }

// ---- the bf16-20260921 gate, unchanged: non-finite refused first, then max-abs relative to max-abs ref --------------
static double gate_err(const float *got, const double *ref, size_t n) {
  for (size_t i = 0; i < n; i++) if (!isfinite(got[i]) || !isfinite(ref[i])) return INFINITY;
  double mx = 0, mr = 0;
  for (size_t i = 0; i < n; i++) { double d = fabs((double)got[i] - ref[i]); if (d > mx) mx = d; if (fabs(ref[i]) > mr) mr = fabs(ref[i]); }
  return mr == 0 ? INFINITY : mx / mr;
}
// The gate must actually refuse non-finite output (the v1 fmax reduction dropped NaN and read error 0). Host-only
// selftest: no GPU fault is injected to prove this.
static void gate_selftest(void) {
  float got[2] = { 1.f, 1.f }; double ref[2] = { 1.f, 1.f };
  int ok = 1;
  got[1] = NAN;  ok &= isinf(gate_err(got, ref, 2));
  got[1] = INFINITY; ok &= isinf(gate_err(got, ref, 2));
  got[1] = 1.f; ref[1] = NAN; ok &= isinf(gate_err(got, ref, 2));
  got[1] = 1.001f; ok &= gate_err(got, ref, 2) > 0; // a real difference still measures
  printf("gate selftest (NaN/Inf refused, finite difference measured): %s\n", ok ? "ok" : "FAIL");
  if (!ok) fails++;
}

// ---- one GEMM case: column-major, exactly the cuBLAS contract --------------------------------------------------------
// opA/opB: 0 = N, 1 = T. A is (opA ? k x m : m x k) with lda, B likewise; C is m x n with ldc.
// bf16 inputs, f32 or bf16 output, alpha/beta float, COMPUTE_32F. batch uses GemmStridedBatchedEx.
static void case_gemm(cublasHandle_t h, const char *what, int opA, int opB, int m, int n, int k,
                      int lda, int ldb, int ldc, int batch, int bf16_out, float alpha, float beta, size_t c_off_bytes) {
  // column-major extents: opA=N -> A is m x k (lda*k); opA=T -> k x m (lda*m); opB=N -> B is k x n (ldb*n);
  // opB=T -> n x k (ldb*k). The leading dimension times the COLUMN count, not the contracted dimension.
  size_t asz = (size_t)lda * (opA ? m : k) * batch, bsz = (size_t)ldb * (opB ? k : n) * batch;
  size_t csz = (size_t)ldc * n * batch;
  uint16_t *hA = malloc(asz * 2), *hB = malloc(bsz * 2);
  float *hC0 = malloc(csz * 4);
  if (!hA || !hB || !hC0) { printf("ERROR oom\n"); exit(3); }
  for (size_t i = 0; i < asz; i++) hA[i] = f2bf(frand());
  for (size_t i = 0; i < bsz; i++) hB[i] = f2bf(frand());
  for (size_t i = 0; i < csz; i++) hC0[i] = frand();
  void *A, *B; char *C;
  CR(cudaMalloc(&A, asz * 2)); CR(cudaMalloc(&B, bsz * 2)); CR(cudaMalloc((void **)&C, csz * (bf16_out ? 2 : 4) + 64));
  CR(cudaMemcpy(A, hA, asz * 2, cudaMemcpyHostToDevice));
  CR(cudaMemcpy(B, hB, bsz * 2, cudaMemcpyHostToDevice));
  if (bf16_out) {
    uint16_t *hCbf = malloc(csz * 2);
    for (size_t i = 0; i < csz; i++) hCbf[i] = f2bf(hC0[i]);
    CR(cudaMemcpy(C + c_off_bytes, hCbf, csz * 2, cudaMemcpyHostToDevice));
    free(hCbf);
  } else {
    CR(cudaMemcpy(C + c_off_bytes, hC0, csz * 4, cudaMemcpyHostToDevice));
  }
  long long sA = (long long)lda * (opA ? m : k), sB = (long long)ldb * (opB ? k : n), sC = (long long)ldc * n;
  cudaDataType out_t = bf16_out ? CUDA_R_16BF : CUDA_R_32F;
  if (batch == 1)
    CB(cublasGemmEx(h, opA ? CUBLAS_OP_T : CUBLAS_OP_N, opB ? CUBLAS_OP_T : CUBLAS_OP_N, m, n, k, &alpha,
                    A, CUDA_R_16BF, lda, B, CUDA_R_16BF, ldb, &beta, C + c_off_bytes, out_t, ldc,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
  else
    CB(cublasGemmStridedBatchedEx(h, opA ? CUBLAS_OP_T : CUBLAS_OP_N, opB ? CUBLAS_OP_T : CUBLAS_OP_N, m, n, k, &alpha,
                                  A, CUDA_R_16BF, lda, sA, B, CUDA_R_16BF, ldb, sB, &beta, C + c_off_bytes, out_t, ldc, sC,
                                  batch, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
  CR(cudaDeviceSynchronize());
  // read back and compare only the m x n submatrix of each batch element (ldc may exceed m)
  float *got = malloc((size_t)m * n * batch * 4);
  double *ref = malloc((size_t)m * n * batch * 8);
  if (bf16_out) {
    uint16_t *rb = malloc(csz * 2);
    CR(cudaMemcpy(rb, C + c_off_bytes, csz * 2, cudaMemcpyDeviceToHost));
    for (int b = 0; b < batch; b++) for (int j = 0; j < n; j++) for (int i = 0; i < m; i++)
      got[((size_t)b * n + j) * m + i] = bf2f(rb[(size_t)b * sC + (size_t)j * ldc + i]);
    free(rb);
  } else {
    float *rb = malloc(csz * 4);
    CR(cudaMemcpy(rb, C + c_off_bytes, csz * 4, cudaMemcpyDeviceToHost));
    for (int b = 0; b < batch; b++) for (int j = 0; j < n; j++) for (int i = 0; i < m; i++)
      got[((size_t)b * n + j) * m + i] = rb[(size_t)b * sC + (size_t)j * ldc + i];
    free(rb);
  }
  // CPU double reference from the decoded bf16 inputs: independent of every GPU kernel
  for (int b = 0; b < batch; b++) for (int j = 0; j < n; j++) for (int i = 0; i < m; i++) {
    double acc = 0;
    for (int q = 0; q < k; q++) {
      double a = bf2f(hA[(size_t)b * sA + (opA ? (size_t)i * lda + q : (size_t)q * lda + i)]);
      double bb = bf2f(hB[(size_t)b * sB + (opB ? (size_t)q * ldb + j : (size_t)j * ldb + q)]);
      acc += a * bb;
    }
    double c0 = hC0[(size_t)b * sC + (size_t)j * ldc + i];
    if (bf16_out) c0 = bf2f(f2bf((float)c0)); // the GPU's initial C carries the bf16 rounding too
    ref[((size_t)b * n + j) * m + i] = (double)alpha * acc + (double)beta * c0;
  }
  double e = gate_err(got, ref, (size_t)m * n * batch);
  if (getenv("BF16_GEMM_DEBUG") && !(e <= (bf16_out ? 1e-2 : 2e-3))) {
    // diagnose WHICH contract broke: re-reduce the same outputs against alternative references
    size_t nn = (size_t)m * n * batch;
    double *r_a1 = malloc(nn * 8), *r_b0 = malloc(nn * 8), *r_b1 = malloc(nn * 8);
    // recompute the three alternatives from the raw pieces
    for (int b = 0; b < batch; b++) for (int j = 0; j < n; j++) for (int i = 0; i < m; i++) {
      double acc = 0;
      for (int q = 0; q < k; q++) {
        double a = bf2f(hA[(size_t)b * sA + (opA ? (size_t)i * lda + q : (size_t)q * lda + i)]);
        double bb = bf2f(hB[(size_t)b * sB + (opB ? (size_t)q * ldb + j : (size_t)j * ldb + q)]);
        acc += a * bb;
      }
      double c0 = hC0[(size_t)b * sC + (size_t)j * ldc + i];
      if (bf16_out) c0 = bf2f(f2bf((float)c0));
      size_t x = ((size_t)b * n + j) * m + i;
      r_a1[x] = acc + (double)beta * c0;          // alpha ignored (=1)
      r_b0[x] = (double)alpha * acc;              // beta ignored (=0)
      r_b1[x] = (double)alpha * acc + c0;         // beta = 1
    }
    printf("  debug: got[0..3] = %.6g %.6g %.6g %.6g\n", got[0], got[1], got[2], got[3]);
    printf("  debug: ref[0..3] = %.6g %.6g %.6g %.6g\n", ref[0], ref[1], ref[2], ref[3]);
    printf("  debug: err vs alpha=1: %.3e, vs beta=0: %.3e, vs beta=1: %.3e\n",
           gate_err(got, r_a1, nn), gate_err(got, r_b0, nn), gate_err(got, r_b1, nn));
    double worst = 0; size_t wi = 0;
    for (size_t x = 0; x < nn; x++) { double d = fabs(got[x] - ref[x]); if (d > worst) { worst = d; wi = x; } }
    printf("  debug: worst at flat %zu (i=%zu j=%zu): got %.6g ref %.6g\n", wi, wi % m, wi / m, got[wi], ref[wi]);
    // where does the wrong value come from? scan every reference element for the closest match
    double best = 1e300; size_t bi = 0;
    for (size_t x = 0; x < nn; x++) { double d = fabs(got[wi] - ref[x]); if (d < best) { best = d; bi = x; } }
    printf("  debug: closest ref to worst got: flat %zu (i=%zu j=%zu) ref %.6g residual %.3e\n", bi, bi % m, bi / m, ref[bi], best);
    if (!opA && !opB && batch == 1) {
      // which k-subset does the GPU's accumulation for the worst element correspond to?
      size_t i = wi % m, j = wi / m;
      double full = 0, first64 = 0, last16 = 0, first16 = 0;
      for (int q = 0; q < k; q++) {
        double t = (double)bf2f(hA[(size_t)q * lda + i]) * bf2f(hB[j * ldb + q]);
        full += t;
        if (q < 64) first64 += t;
        if (q >= 64) last16 += t;
        if (q < 16) first16 += t;
      }
      double c0 = hC0[j * ldc + i];
      printf("  debug: partial sums: full=%.6g first64=%.6g last16=%.6g first16=%.6g (k=%d lda=%d ldb=%d)\n", full, first64, last16, first16, k, lda, ldb);
      printf("  debug: worst got %.6g ; alpha*full+beta*c0=%.6g ; alpha*first64+beta*c0=%.6g ; alpha*(full-last16)+beta*c0=%.6g ; alpha*(full-first16)+beta*c0=%.6g ; c0=%.6g\n",
             got[wi], alpha * full + beta * c0, alpha * first64 + beta * c0, alpha * (full - last16) + beta * c0,
             alpha * (full - first16) + beta * c0, c0);
    }
    // region errors: first 64 columns vs the rest, first 64 rows vs the rest
    for (int reg = 0; reg < 4; reg++) {
      double mx = 0, mr = 0;
      for (int b = 0; b < batch; b++) for (int j = 0; j < n; j++) for (int i = 0; i < m; i++) {
        if ((j >= 64) != (reg & 1) || (i >= 64) != ((reg >> 1) & 1)) continue;
        size_t x = ((size_t)b * n + j) * m + i;
        double d = fabs(got[x] - ref[x]); if (d > mx) mx = d; if (fabs(ref[x]) > mr) mr = fabs(ref[x]);
      }
      printf("  debug: region i%s64 j%s64: relerr %.3e\n", (reg >> 1) & 1 ? ">=" : "<", reg & 1 ? ">=" : "<", mr ? mx / mr : 0);
    }
    // hypothesis refs for the j>=64 region: prev taken from j-64, or acc taken from j-64
    if (n > 64 && !opA && !opB && batch == 1) {
      double mx1 = 0, mr1 = 0, mx2 = 0, mr2 = 0;
      for (int j = 64; j < n; j++) for (int i = 0; i < m; i++) {
        double accj = 0, accp = 0;
        for (int q = 0; q < k; q++) {
          double a = bf2f(hA[(size_t)q * lda + i]);
          accj += a * bf2f(hB[(size_t)j * ldb + q]);
          accp += a * bf2f(hB[(size_t)(j - 64) * ldb + q]);
        }
        double c0j = hC0[(size_t)j * ldc + i], c0p = hC0[(size_t)(j - 64) * ldc + i];
        size_t x = (size_t)j * m + i;
        double h1 = (double)alpha * accj + (double)beta * c0p;   // right acc, prev from j-64
        double h2 = (double)alpha * accp + (double)beta * c0j;   // acc from j-64, right prev
        double d1 = fabs(got[x] - h1), d2 = fabs(got[x] - h2);
        if (d1 > mx1) mx1 = d1; if (fabs(h1) > mr1) mr1 = fabs(h1);
        if (d2 > mx2) mx2 = d2; if (fabs(h2) > mr2) mr2 = fabs(h2);
      }
      printf("  debug: j>=64 vs prev-from-j-64: %.3e ; vs acc-from-j-64: %.3e\n", mr1 ? mx1 / mr1 : 0, mr2 ? mx2 / mr2 : 0);
    }
    // per-column max abs difference: which columns are wrong
    printf("  debug: per-col max|d|:");
    for (int j = 0; j < n; j++) {
      double mx = 0;
      for (int b = 0; b < batch; b++) for (int i = 0; i < m; i++) {
        double d = fabs(got[((size_t)b * n + j) * m + i] - ref[((size_t)b * n + j) * m + i]);
        if (d > mx) mx = d;
      }
      printf(" %d:%.2g", j, mx);
    }
    printf("\n");
    free(r_a1); free(r_b0); free(r_b1);
  }
  double tol = bf16_out ? 1e-2 : 2e-3;
  int ok = e <= tol;
  cases_run++; if (!ok) fails++;
  printf("  %-4s %-34s %c%c m %4d n %4d k %4d batch %d out %-4s alpha %.1f beta %.1f: err %.2e (tol %.0e)%s\n",
         ok ? "ok" : "FAIL", what, opA ? 'T' : 'N', opB ? 'T' : 'N', m, n, k, batch, bf16_out ? "bf16" : "f32",
         alpha, beta, e, tol, ok ? "" : "  <-- OVER TOLERANCE / NON-FINITE");
  fflush(stdout);
  free(got); free(ref); free(hA); free(hB); free(hC0);
  cudaFree(A); cudaFree(B); cudaFree(C);
}

// ---- the register readout, printed raw and gated ----------------------------------------------------------------------
static int read_state(tinynv_device_t d, const char *phase, int expect) {
  tensor_unlock_state_t s;
  if (tensor_unlock_postboot(d, &s)) {
    printf("tensor-state[%s]: UNAVAILABLE (no live booted device)\n", phase);
    fails++;
    return -1;
  }
  printf("tensor-state[%s]: chip=%#x feat=%#x readout=%#x (masked %#x) ss0=%#x ss1=%#x valid=%d unlocked=%d stock=%d\n",
         phase, s.chip, s.feat, s.readout, s.readout & TU_UNLOCK_READOUT_MASK, s.ss0, s.ss1, s.valid, s.unlocked, s.stock);
  fflush(stdout);
  if (expect == 1 && !s.unlocked) { printf("  FAIL: expected the verified UNLOCKED state after %s\n", phase); fails++; }
  if (expect == 0 && !s.stock) { printf("  FAIL: expected the STOCK state after %s\n", phase); fails++; }
  if (!s.valid) { printf("  FAIL: inaccessible register reads are never evidence of success\n"); fails++; }
  return 0;
}

static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static int cmp_d(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return x < y ? -1 : x > y; }

// ---- multi-shape measurement suite ----------------------------------------------------------------------------------
// Same-cubin GEMM timing on the shapes that matter for this experiment, plus the card's own sensors around every
// shape (temperature, power, clocks, pstate, throttle reasons) so a stock/unlocked comparison can say whether the
// thermal/power state matched. Nothing here changes a kernel, a cubin, the arithmetic or a tolerance.

static void print_sensors(tinynv_device_t d, const char *phase) {
  tinynv_sensors_t s;
  if (tinynv_sensors(d, &s) != TINYNV_OK) { printf("  sensors[%s]: unavailable (%s)\n", phase, tinynv_last_error()); return; }
  printf("  sensors[%s]:", phase);
  if (s.gpu_temp_ok) printf(" gpu %.0fC", s.gpu_temp_c); else printf(" gpu-temp n/a");
  if (s.mem_temp_ok) printf(" mem %.0fC", s.mem_temp_c);
  if (s.power_ok) printf(" power %.1fW", s.gpu_power_mw / 1000.0);
  if (s.limit_ok) printf(" limit %.0fW", s.power_limit_mw / 1000.0);
  if (s.clock_ok) printf(" sm %uMHz mem %uMHz", s.sm_mhz, s.memory_mhz); else printf(" clocks n/a");
  if (s.pstate_ok) printf(" pstate P%u", s.pstate);
  if (s.util_ok) printf(" busy %u%%", s.gpu_busy_pct);
  if (s.throttle_ok && s.throttle) {
    const char *words[8];
    int n = tinynv_throttle_str(s.throttle, words, 8);
    printf(" throttle:");
    for (int i = 0; i < n; i++) printf(" %s", words[i]);
  }
  printf("\n");
  fflush(stdout);
}

// One TN GEMM shape (ggml's prefill call: opA=T, opB=N, bf16 in, f32 out, COMPUTE_32F), timed through the same
// cublasGemmEx the correctness cases use. Wall clock around call+sync: this driver waits on every submitted batch
// before the call returns. 3 warmup + 10 measured repeats, median/min/max and every repeat printed.
// Correctness of the timed output: the FULL C is scanned for non-finite values (refused first, the protocol's gate),
// then 64 deterministic elements are checked against a CPU double reference from the decoded bf16 inputs with the
// SAME bf16-20260921 tolerance (2e-3, f32 output). The 10-case matrix above remains the full correctness authority.
// Returns the median ms (for the modeled pp256 sum), or -1 on a correctness failure.
static double bench_shape(cublasHandle_t h, tinynv_device_t d, int m, int n, int k, const char *label) {
  size_t asz = (size_t)k * m, bsz = (size_t)k * n, csz = (size_t)m * n;
  uint16_t *hA = malloc(asz * 2), *hB = malloc(bsz * 2);
  for (size_t i = 0; i < asz; i++) hA[i] = f2bf(frand());
  for (size_t i = 0; i < bsz; i++) hB[i] = f2bf(frand());
  void *A, *B, *C;
  CR(cudaMalloc(&A, asz * 2)); CR(cudaMalloc(&B, bsz * 2)); CR(cudaMalloc(&C, csz * 4));
  CR(cudaMemcpy(A, hA, asz * 2, cudaMemcpyHostToDevice)); CR(cudaMemcpy(B, hB, bsz * 2, cudaMemcpyHostToDevice));
  CR(cudaMemset(C, 0, csz * 4));
  const float one = 1.f, zero = 0.f;
  printf("shape %s (TN %dx%dx%d bf16->f32):\n", label, m, n, k);
  print_sensors(d, "pre");
  double t[10];
  for (int i = 0; i < 3 + 10; i++) {
    double t0 = now_s();
    CB(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &one, A, CUDA_R_16BF, k, B, CUDA_R_16BF, k, &zero,
                    C, CUDA_R_32F, m, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    CR(cudaDeviceSynchronize());
    double dt = (now_s() - t0) * 1e3;
    if (i >= 3) t[i - 3] = dt;
    printf("  bench warmup/measured %2d: %.3f ms\n", i, dt);
  }
  print_sensors(d, "post");
  // correctness of the timed output: non-finite scan first, then the 64-element double-reference spot check
  float *got = malloc(csz * 4);
  CR(cudaMemcpy(got, C, csz * 4, cudaMemcpyDeviceToHost));
  int nonfinite = 0;
  for (size_t i = 0; i < csz; i++) if (!isfinite(got[i])) { nonfinite = 1; break; }
  double spot_err = 0;
  int spot_ok = !nonfinite;
  if (!nonfinite) {
    uint64_t rng = 0x9e3779b97f4a7c15ull ^ (uint64_t)(m * 131 + n * 7 + k);
    double mx = 0, mr = 0;
    for (int s = 0; s < 64; s++) {
      rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; // xorshift64: deterministic per shape
      size_t idx = (size_t)(rng % csz);
      size_t i = idx % m, j = idx / m;
      double acc = 0;
      for (int q = 0; q < k; q++)
        acc += (double)bf2f(hA[(size_t)i * k + q]) * (double)bf2f(hB[(size_t)j * k + q]); // opA=T lda=k: A[i*k+q]; opB=N ldb=k: B[j*k+q]
      double derr = fabs((double)got[idx] - acc);
      if (derr > mx) mx = derr;
      if (fabs(acc) > mr) mr = fabs(acc);
    }
    spot_err = mr == 0 ? INFINITY : mx / mr;
    spot_ok = spot_err <= 2e-3;
  }
  cases_run++;
  if (!spot_ok) fails++;
  printf("  output check: non-finite %s, 64-element double-reference spot err %.2e (tol 2e-3): %s\n",
         nonfinite ? "PRESENT" : "absent", spot_err, spot_ok ? "ok" : "FAIL");
  double s[10]; memcpy(s, t, sizeof t); qsort(s, 10, sizeof s[0], cmp_d);
  double med = (s[4] + s[5]) / 2, flops = 2.0 * m * n * k;
  printf("bench bf16 TN %dx%dx%d f32-out: median %.3f ms (%.2f TFLOP/s), min %.3f, max %.3f, all:",
         m, n, k, med, flops / (med * 1e-3) / 1e12, s[0], s[9]);
  for (int i = 0; i < 10; i++) printf(" %.3f", t[i]);
  printf(" ms\n");
  fflush(stdout);
  free(got); free(hA); free(hB); cudaFree(A); cudaFree(B); cudaFree(C);
  return spot_ok ? med : -1;
}

static void bench(cublasHandle_t h, tinynv_device_t d) {
  // The pp256 family is the EXACT call shape ggml makes for Qwen2.5-3B BF16 prefill (bf16-20260921/README.md:
  // cublasGemmEx(T, N, m=out features, n=256, k=in features), per-pp256 call counts 72/72/70/35 on the 36-layer
  // model). 2048^3 is the primary comparison; n=1 decode and n=300 partial tiles are secondary.
  struct { int m, n, k; const char *label; int pp_calls; } shapes[] = {
    { 2048, 2048, 2048, "large square (primary comparison)", 0 },
    { 2048,  256, 2048, "pp256 Q/O", 72 },
    {  256,  256, 2048, "pp256 K/V", 72 },
    {11008,  256, 2048, "pp256 gate/up", 70 },
    { 2048,  256,11008, "pp256 down", 35 },
    { 2048,    1, 2048, "decode GEMV (secondary)", 0 },
    { 2048,  300, 2048, "partial tiles n=300", 0 },
  };
  double pp_sum = 0; int pp_ok = 1;
  for (size_t i = 0; i < sizeof shapes / sizeof shapes[0]; i++) {
    double med = bench_shape(h, d, shapes[i].m, shapes[i].n, shapes[i].k, shapes[i].label);
    if (shapes[i].pp_calls) {
      if (med < 0) pp_ok = 0;
      else pp_sum += med * shapes[i].pp_calls;
    }
  }
  if (pp_ok)
    printf("modeled pp256 GEMM time (Qwen2.5-3B dims, 36 layers, call counts 72/72/70/35 x measured medians): "
           "%.1f ms of GEMM compute per 256-token prefill — a compute-side model, NOT an end-to-end prefill "
           "measurement (attention, norms, dispatch and host overhead are not included)\n", pp_sum);
  else
    printf("modeled pp256 GEMM time: NOT reported (a prefill shape failed its output check)\n");
  fflush(stdout);
}

int main(int argc, char **argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  int expect = -1, do_bench = 1; // -1 any, 0 stock, 1 unlocked
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--expect") && i + 1 < argc) {
      i++;
      if (!strcmp(argv[i], "stock")) expect = 0;
      else if (!strcmp(argv[i], "unlocked")) expect = 1;
      else if (!strcmp(argv[i], "any")) expect = -1;
      else { printf("unknown --expect %s\n", argv[i]); return 2; }
    } else if (!strcmp(argv[i], "--no-bench")) do_bench = 0;
    else { printf("usage: %s [--expect stock|unlocked|any] [--no-bench]\n", argv[0]); return 2; }
  }
  const char *sock = getenv("TINYNV_SOCKET");
  if (!sock || !*sock) { printf("refusing the null device: set TINYNV_SOCKET (this test validates real hardware)\n"); return 2; }
  printf("bf16_unlock_gemm: driver build %s, expect %s\n", tinynv_build_id(),
         expect == 1 ? "unlocked" : expect == 0 ? "stock" : "any (report only)");
  gate_selftest();
  // the normal handoff: unchanged runtime firmware boot through the shim's own path. The init idiom is a real
  // alloc/free: this shim's cudaFree does not lazy-init (cudaFree(0) would pass a NULL device to tinynv_free).
  void *init_probe = NULL;
  CR(cudaMalloc(&init_probe, 256));
  CR(cudaFree(init_probe));
  struct cudaDeviceProp prop; CR(cudaGetDeviceProperties(&prop, 0));
  printf("device: %s, cc %d.%d, %d SMs, %.1f GiB\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount,
         prop.totalGlobalMem / 1073741824.0);
  if (prop.major == 0) { printf("null device reported; refusing\n"); return 2; }
  tinynv_device_t d = NULL;
  if (tinynv_device_get(&d, 0) != TINYNV_OK || !d) { printf("no tinynv device: %s\n", tinynv_last_error()); return 2; }
  read_state(d, "post-boot", expect);
  cublasHandle_t h; CB(cublasCreate(&h));
  srand(7); // the same seed in stock and unlocked sessions: identical inputs, identical cubins
  printf("correctness (CPU double reference from decoded bf16 inputs; bf16-20260921 gate and tolerances):\n");
  case_gemm(h, "aligned NN",            0, 0, 256, 256, 256, 256, 256, 256, 1, 0, 1.f, 0.f, 0);
  case_gemm(h, "ggml TN",               1, 0, 512, 64, 256, 256, 256, 512, 1, 0, 1.f, 0.f, 0);
  case_gemm(h, "NT",                    0, 1, 96, 80, 64, 96, 80, 96, 1, 0, 1.f, 0.f, 0);
  case_gemm(h, "TT",                    1, 1, 64, 64, 64, 64, 64, 64, 1, 0, 1.f, 0.f, 0);
  case_gemm(h, "partial tiles batched", 1, 0, 48, 40, 72, 72, 72, 48, 3, 0, 1.f, 0.f, 0);
  {
    // env-tunable for diagnosis (defaults are the contract case): isolate alpha scaling from the beta*C read
    float ab_alpha = getenv("BF16_AB_ALPHA") ? strtof(getenv("BF16_AB_ALPHA"), NULL) : 0.5f;
    float ab_beta = getenv("BF16_AB_BETA") ? strtof(getenv("BF16_AB_BETA"), NULL) : 2.f;
    case_gemm(h, "alpha/beta + C",        0, 0, 128, 96, 80, 128, 80, 128, 1, 0, ab_alpha, ab_beta, 0);
  }
  case_gemm(h, "bf16 output",           0, 0, 128, 128, 128, 128, 128, 128, 1, 1, 1.f, 0.f, 0);
  case_gemm(h, "ldc33 (staging path)",  1, 0, 32, 32, 32, 32, 32, 33, 1, 0, 1.f, 0.f, 0);
  case_gemm(h, "C base +16 (staging)",  1, 0, 32, 32, 32, 32, 32, 32, 1, 0, 1.f, 0.f, 16);
  if (!getenv("BF16_SKIP_LARGE"))
    case_gemm(h, "large TN",              1, 0, 1024, 1024, 1024, 1024, 1024, 1024, 1, 0, 1.f, 0.f, 0);
  read_state(d, "post-workload", expect);
  if (do_bench && !getenv("BF16_NO_BENCH")) { printf("timing (identical shapes/seed/cubin in stock and unlocked sessions):\n"); bench(h, d); }
  cublasDestroy(h);
  printf("\n%d case(s), %d failure(s)\n", cases_run, fails);
  return fails ? 1 : 0;
}
