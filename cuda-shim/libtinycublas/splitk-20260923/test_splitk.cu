// test_splitk.cu - the split-K GEMM (tinyblas_gemm_*_tc2k_*) against the kernel it splits (B, 2026-09-23).
// Design: docs/review/20260923-smalln-design-B.md (root repo). Runs on any sm_80+ GPU; banked on the 3090 box (sm_86).
//   test_splitk <gemm cubin>
// Rows, each printed on entry and each ending PASS or FAIL; the exit code is the number of FAILs:
//   EXACT   split(S) == sum over slices, IN SLICE ORDER, of the unsplit tc2s kernel run on that slice's k-range, compared
//           with float == (so +0 and -0 agree and nothing else may differ). alpha 1, beta 0, f32 out, where tc2s's
//           direct store writes the accumulator bits unchanged. The two kernels share one main loop, so the partials
//           must be identical and the only arithmetic the split adds is that ordered sum.
//   REF     split(S) against a double-precision host reference, with the contract test's gate (2e-3 f32 out, 1e-2 bf16 out).
//   R       (D, 02:2x) plan-free: max|split - fp64| over the FULL k, divided by the same for the UNSPLIT tc2s kernel on the
//           same inputs, must be <= 2. EXACT cuts slices with the kernel's own rule, so a rule that dropped a K remainder
//           or overlapped slices would pass it; this row never uses the rule.
//   AGAIN   the same GEMM a second time on the same ws/cnt with C poisoned in between: must be EXACT again (the counters
//           reset themselves), and every counter must read 0 afterwards.
// Mutants are separate cubins built from a sed of gemm.cu (run-3090.sh); each must make this test exit nonzero.
#include <cuda.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>

#define CK(x) do { CUresult r_ = (x); if (r_) { const char *s_ = 0; cuGetErrorString(r_, &s_); printf("ERROR %s: %d %s (line %d)\n", #x, r_, s_ ? s_ : "", __LINE__); exit(99); } } while (0)

static CUmodule mod;
static int fails;

static uint32_t rng = 12345u;
static float frand(void){ rng = rng * 1664525u + 1013904223u; return ((rng >> 8) * (1.f / 16777216.f)) * 2.f - 1.f; }

struct Case { const char* name; int m, n, k, S, batch; };

// ggml's call: op(A) = A^T with A stored k-contiguous (lda = k), B k-contiguous (ldb = k), C column-major ldc = m.
static void launch(const char* kname, CUdeviceptr A, CUdeviceptr B, CUdeviceptr C, long long sA, long long sB, long long sC,
                   int m, int n, int k, int lda, int ldb, int ldc, float alpha, float beta, int out,
                   CUdeviceptr ws, CUdeviceptr cnt, int S, int batch){
  CUfunction f; CK(cuModuleGetFunction(&f, mod, kname));
  int split = S > 0;
  void* args[] = { &A, &B, &C, &sA, &sB, &sC, &m, &n, &k, &lda, &ldb, &ldc, &alpha, &beta, &out, &ws, &cnt, &S };
  unsigned gx = (m + 63) / 64, gy = (n + 63) / 64, gz = split ? batch * S : batch;
  CK(cuLaunchKernel(f, gx, gy, gz, 128, 1, 1, 0, 0, args, 0));
  CK(cuCtxSynchronize());
}

static void run(const Case& c){
  printf("CASE %s m=%d n=%d k=%d S=%d batch=%d\n", c.name, c.m, c.n, c.k, c.S, c.batch);
  const int m = c.m, n = c.n, k = c.k, S = c.S, nb = c.batch;
  const long long sA = (long long)m * k, sB = (long long)n * k, sC = (long long)m * n;
  std::vector<__nv_bfloat16> hA(sA * nb), hB(sB * nb);
  for (auto& x : hA) x = __float2bfloat16(frand());
  for (auto& x : hB) x = __float2bfloat16(frand());
  CUdeviceptr dA, dB, dC, dP, ws, cnt;
  const unsigned gx = (m + 63) / 64, gy = (n + 63) / 64, tiles = gx * gy;
  const size_t wsz = (size_t)nb * S * tiles * 64 * 64 * 4, csz = (size_t)nb * tiles * 4;
  CK(cuMemAlloc(&dA, hA.size() * 2)); CK(cuMemAlloc(&dB, hB.size() * 2));
  CK(cuMemAlloc(&dC, sC * nb * 4)); CK(cuMemAlloc(&dP, sC * 4)); CK(cuMemAlloc(&ws, wsz)); CK(cuMemAlloc(&cnt, csz));
  CK(cuMemcpyHtoD(dA, hA.data(), hA.size() * 2)); CK(cuMemcpyHtoD(dB, hB.data(), hB.size() * 2));
  CK(cuMemsetD8(cnt, 0, csz));

  // The expected result: per slice, tc2s on that k-range (same slicing rule as the kernel), summed on the host in order.
  const int BK = 32, nk_all = (k + BK - 1) / BK, per = (nk_all + S - 1) / S;
  std::vector<float> expect(sC * nb, 0.f), part(sC);
  for (int b = 0; b < nb; b++)
    for (int q = 0; q < S; q++){
      const int t0 = q * per < nk_all ? q * per : nk_all, t1 = t0 + per < nk_all ? t0 + per : nk_all;
      const int k0 = t0 * BK, kl = (t1 * BK < k ? t1 * BK : k) - k0;
      if (kl <= 0) { for (long long e = 0; e < sC; e++) expect[b * sC + e] += 0.f; continue; }
      launch("tinyblas_gemm_bf16_tc2s_tn", dA + (b * sA + k0) * 2, dB + (b * sB + k0) * 2, dP, 0, 0, 0,
             m, n, kl, k, k, m, 1.f, 0.f, 0, 0, 0, -1, 1);
      CK(cuMemcpyDtoH(part.data(), dP, sC * 4));
      for (long long e = 0; e < sC; e++) expect[b * sC + e] += part[e];
    }

  std::vector<float> got(sC * nb);
  for (int pass = 0; pass < 2; pass++){
    const char* row = pass == 0 ? "EXACT" : "AGAIN";
    printf("  %s: entered\n", row);
    CK(cuMemsetD32(dC, 0x7fc00000u, sC * nb));   // poison C with NaN: a tile nobody wrote cannot pass
    launch("tinyblas_gemm_bf16_tc2k_tn", dA, dB, dC, sA, sB, sC, m, n, k, k, k, m, 1.f, 0.f, 0, ws, cnt, S, nb);
    CK(cuMemcpyDtoH(got.data(), dC, sC * nb * 4));
    long long bad = 0, first = -1;
    for (long long e = 0; e < sC * nb; e++) if (!(got[e] == expect[e])) { if (first < 0) first = e; bad++; }
    if (bad) { printf("  %s FAIL: %lld of %lld differ; first at %lld: got %.9g want %.9g\n", row, bad, sC * nb, first, got[first], expect[first]); fails++; }
    else printf("  %s PASS: %lld of %lld equal\n", row, sC * nb, sC * nb);
  }
  std::vector<unsigned> hc(nb * tiles);
  CK(cuMemcpyDtoH(hc.data(), cnt, csz));
  long long nz = 0; for (unsigned v : hc) nz += v != 0;
  if (nz) { printf("  COUNTERS FAIL: %lld of %u nonzero after the GEMM\n", nz, nb * tiles); fails++; }
  else printf("  COUNTERS PASS: all %u zero\n", nb * tiles);

  // REF, with alpha/beta and bf16 output too: the split path's own epilogue, against doubles.
  for (int out = 0; out <= 2; out += 2){
    const float alpha = out ? 0.75f : 1.f, beta = out ? 0.5f : 0.f;
    printf("  REF out=%s alpha=%g beta=%g: entered\n", out ? "bf16" : "f32", alpha, beta);
    const int osz = out ? 2 : 4;
    std::vector<unsigned char> c0(sC * nb * osz);
    for (long long e = 0; e < sC * nb; e++){ float v = frand(); if (out) { __nv_bfloat16 h = __float2bfloat16(v); memcpy(&c0[e * 2], &h, 2); } else memcpy(&c0[e * 4], &v, 4); }
    CK(cuMemcpyHtoD(dC, c0.data(), c0.size()));
    launch("tinyblas_gemm_bf16_tc2k_tn", dA, dB, dC, sA, sB, sC, m, n, k, k, k, m, alpha, beta, out, ws, cnt, S, nb);
    std::vector<unsigned char> c1(c0.size()); CK(cuMemcpyDtoH(c1.data(), dC, c1.size()));
    double maxerr = 0, maxref = 0; int nonfinite = 0;
    for (int b = 0; b < nb; b++) for (int j = 0; j < n; j++) for (int i = 0; i < m; i++){
      double r = 0;
      for (int kk = 0; kk < k; kk++) r += (double)__bfloat162float(hA[b * sA + (long long)i * k + kk]) * (double)__bfloat162float(hB[b * sB + (long long)j * k + kk]);
      const long long e = b * sC + i + (long long)j * m;
      double prev, g;
      if (out) { __nv_bfloat16 h; memcpy(&h, &c0[e * 2], 2); prev = __bfloat162float(h); memcpy(&h, &c1[e * 2], 2); g = __bfloat162float(h); }
      else { float f; memcpy(&f, &c0[e * 4], 4); prev = f; memcpy(&f, &c1[e * 4], 4); g = f; }
      r = alpha * r + beta * prev;
      if (!isfinite(g)) nonfinite++;
      maxerr = fmax(maxerr, fabs(g - r)); maxref = fmax(maxref, fabs(r));
    }
    const double rel = maxerr / maxref, tol = out ? 1e-2 : 2e-3;
    if (nonfinite || !(rel <= tol)) { printf("  REF FAIL: rel %.3g (tol %.0e), %d non-finite\n", rel, tol, nonfinite); fails++; }
    else printf("  REF PASS: rel %.3g (tol %.0e)\n", rel, tol);
    if (out == 0) {   // R: the unsplit kernel over the full k on the same inputs, against the same doubles
      printf("  R: entered\n");
      std::vector<float> s1(sC * nb);
      for (int b = 0; b < nb; b++) {
        launch("tinyblas_gemm_bf16_tc2s_tn", dA + b * sA * 2, dB + b * sB * 2, dP, 0, 0, 0, m, n, k, k, k, m, 1.f, 0.f, 0, 0, 0, -1, 1);
        CK(cuMemcpyDtoH(s1.data() + b * sC, dP, sC * 4));
      }
      double e1 = 0, eon = 0;
      for (int b = 0; b < nb; b++) for (int j = 0; j < n; j++) for (int i = 0; i < m; i++){
        double r = 0;
        for (int kk = 0; kk < k; kk++) r += (double)__bfloat162float(hA[b * sA + (long long)i * k + kk]) * (double)__bfloat162float(hB[b * sB + (long long)j * k + kk]);
        const long long e = b * sC + i + (long long)j * m; float g; memcpy(&g, &c1[e * 4], 4);
        e1 = fmax(e1, fabs((double)s1[e] - r)); eon = fmax(eon, fabs((double)g - r));
      }
      const double R = e1 > 0 ? eon / e1 : (eon > 0 ? INFINITY : 1.0);
      if (!(R <= 2.0)) { printf("  R FAIL: split err %.3g vs unsplit %.3g, R %.3g (> 2)\n", eon, e1, R); fails++; }
      else printf("  R PASS: split err %.3g vs unsplit %.3g, R %.3g (<= 2)\n", eon, e1, R);
    }
  }
  cuMemFree(dA); cuMemFree(dB); cuMemFree(dC); cuMemFree(dP); cuMemFree(ws); cuMemFree(cnt);
}

int main(int argc, char** argv){
  if (argc < 2) { fprintf(stderr, "usage: %s <gemm cubin>\n", argv[0]); return 99; }
  CK(cuInit(0)); CUdevice d; CK(cuDeviceGet(&d, 0)); CUcontext ctx; CK(cuDevicePrimaryCtxRetain(&ctx, d)); CK(cuCtxSetCurrent(ctx));
  char nm[128]; CK(cuDeviceGetName(nm, sizeof nm, d)); printf("device: %s\ncubin: %s\n", nm, argv[1]);
  CK(cuModuleLoad(&mod, argv[1]));
  const Case cases[] = {
    { "kv (3B decode)",        256, 32,  2048, 16, 1 },
    { "q/o (3B decode)",      2048, 32,  2048,  8, 1 },
    { "down (3B decode)",     2048, 32, 11008,  8, 1 },
    { "tails",                 200, 20,  1000,  5, 1 },
    { "empty last slice",      128, 64,    96,  4, 1 },   // nk 3, per 1: slice 3 has no k-tiles and must still count
    { "batch 3",               256, 48,   512,  4, 3 },
  };
  for (const Case& c : cases) run(c);
  printf("RESULT: %d FAIL(s)\n", fails);
  return fails;
}
