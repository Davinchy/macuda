// splitk_card.c - the split-K GEMM on the card, through libtinynv (B, 2026-09-23; D's gate on prereg b79fb72f, item S1).
// The same rows as libtinycublas/splitk-20260923/test_splitk.cu (which went 30/30 on the 3090, sm_86), for the sm_120
// cubin the shim embeds, on the three Qwen2.5-3B decode shapes. Run under the protocol as a probe:
//   sh tools/nv_shim_step.sh B probe <tree>/build/shim/splitk_card <tree>/libtinycublas/gemm.cubin
// Rows, each printed on entry and ending PASS or FAIL; the exit code is the number of FAILs:
//   EXACT     split(S) == the sum IN SLICE ORDER of tc2s run on each slice's k-range (float ==, alpha 1, beta 0, f32 out)
//   AGAIN     the same split GEMM again on the same ws/cnt, C poisoned with NaN in between: EXACT again
//   COUNTERS  every counter reads 0 afterwards
//   REF       split(S) against a double-precision host reference, max|x-ref|/max|ref| <= 2e-3 (f32 out)
// On the NULL device nothing executes: every row reads memory the kernels never wrote, so a null run is a PLUMBING check
// only and prints that it is VOID as evidence. Only a run whose first line names a real card counts.
// The parameter blobs are built from tinynv_kernel_info's offsets, not from a struct, and checked against the layout
// the shim's _Static_assert pins (tc2k: ws 0x58, cnt 0x60, S 0x68, 108 bytes; tc2s: 84 bytes).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "tinynv.h"

static int fails;
static tinynv_stream_t st;
static uint32_t rng = 12345u;
static float frand(void){ rng = rng * 1664525u + 1013904223u; return ((rng >> 8) * (1.f / 16777216.f)) * 2.f - 1.f; }
static uint16_t f2bf(float f){ uint32_t u; memcpy(&u, &f, 4); u += 0x7fffu + ((u >> 16) & 1u); return (uint16_t)(u >> 16); }   // RNE; inputs finite
static float bf2f(uint16_t h){ uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }
#define CK(x) do { tinynv_status_t s_ = (x); if (s_ != TINYNV_OK) { printf("ERROR %s: %s (%s) line %d\n", #x, tinynv_status_str(s_), tinynv_last_error(), __LINE__); exit(99); } } while (0)

// Lay the values out at the offsets the cubin declares. vals[i] points at param i's bytes, sizes[i] its size.
static size_t blob(tinynv_kernel_t k, unsigned char *out, const void **vals, const int *sizes, int n, int want_len){
  tinynv_kernel_info_t ki; CK(tinynv_kernel_info(k, &ki));
  if (ki.num_params != n) { printf("ERROR: the kernel declares %d params, the call has %d\n", ki.num_params, n); exit(99); }
  size_t len = 0; memset(out, 0, 256);
  for (int i = 0; i < n; i++){
    if (ki.params[i].size != sizes[i]) { printf("ERROR: param %d is %d bytes in the cubin, %d here\n", i, ki.params[i].size, sizes[i]); exit(99); }
    memcpy(out + ki.params[i].offset, vals[i], sizes[i]);
    if ((size_t)(ki.params[i].offset + sizes[i]) > len) len = ki.params[i].offset + sizes[i];
  }
  if ((int)len != want_len) { printf("ERROR: blob is %zu bytes, the shim sends %d\n", len, want_len); exit(99); }
  return len;
}

static void gemm(tinynv_kernel_t k, int split, uint64_t A, uint64_t B, uint64_t C, int m, int n, int kk, int lda, int ldb, int ldc,
                 uint64_t ws, uint64_t cnt, int S){
  int64_t z = 0; float alpha = 1.f, beta = 0.f; int out = 0;
  const void *v[18] = { &A, &B, &C, &z, &z, &z, &m, &n, &kk, &lda, &ldb, &ldc, &alpha, &beta, &out, &ws, &cnt, &S };
  const int sz[18] = { 8, 8, 8, 8, 8, 8, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 8, 4 };
  unsigned char p[256]; size_t len = blob(k, p, v, sz, split ? 18 : 15, split ? 108 : 84);
  CK(tinynv_launch(st, k, (m + 63) / 64, (n + 63) / 64, split ? S : 1, 128, 1, 1, 0, p, len));
  CK(tinynv_stream_sync(st));
}

int main(int argc, char **argv){
  if (argc < 2) { fprintf(stderr, "usage: %s <gemm.cubin (sm_120)>\n", argv[0]); return 99; }
  FILE *f = fopen(argv[1], "rb"); if (!f) { printf("ERROR: no cubin at %s\n", argv[1]); return 99; }
  fseek(f, 0, SEEK_END); long cl = ftell(f); fseek(f, 0, SEEK_SET); void *cub = malloc(cl);
  if (fread(cub, 1, cl, f) != (size_t)cl) { printf("ERROR: short read\n"); return 99; } fclose(f);
  CK(tinynv_init()); tinynv_device_t d; CK(tinynv_device_get(&d, 0));
  tinynv_device_props_t pr; CK(tinynv_device_props(d, &pr));
  int null_dev = pr.sm_count == 0 || strstr(pr.name, "null") != NULL;
  printf("device: %s (sm %d.%d, %d SMs)%s\nlibtinynv build %s\ncubin: %s (%ld bytes)\n", pr.name, pr.cc_major, pr.cc_minor, pr.sm_count,
         null_dev ? "  <- NULL DEVICE: nothing executes, every row below is VOID as evidence" : "", tinynv_build_id(), argv[1], cl);
  tinynv_module_t mod; CK(tinynv_module_load(d, cub, cl, &mod));
  tinynv_kernel_t ks, kk; CK(tinynv_get_kernel(mod, "tinyblas_gemm_bf16_tc2s_tn", &ks)); CK(tinynv_get_kernel(mod, "tinyblas_gemm_bf16_tc2k_tn", &kk));
  CK(tinynv_stream_create(d, &st));
  const struct { const char *name; int m, n, k, S; } cs[] = {
    { "k/v (3B decode)", 256, 32, 2048, 16 }, { "q/o (3B decode)", 2048, 32, 2048, 8 }, { "down (3B decode)", 2048, 32, 11008, 8 } };
  for (int c = 0; c < 3; c++){
    const int m = cs[c].m, n = cs[c].n, k = cs[c].k, S = cs[c].S;
    printf("CASE %s m=%d n=%d k=%d S=%d\n", cs[c].name, m, n, k, S);
    const size_t na = (size_t)m * k, nbv = (size_t)n * k, nc = (size_t)m * n, tiles = (size_t)((m + 63) / 64) * ((n + 63) / 64);
    uint16_t *hA = malloc(na * 2), *hB = malloc(nbv * 2); float *exp_ = calloc(nc, 4), *part = malloc(nc * 4), *got = malloc(nc * 4);
    for (size_t i = 0; i < na; i++) hA[i] = f2bf(frand());
    for (size_t i = 0; i < nbv; i++) hB[i] = f2bf(frand());
    tinynv_devptr_t dA, dB, dC, dP, ws, cnt;
    CK(tinynv_malloc(d, na * 2, &dA)); CK(tinynv_malloc(d, nbv * 2, &dB)); CK(tinynv_malloc(d, nc * 4, &dC)); CK(tinynv_malloc(d, nc * 4, &dP));
    CK(tinynv_malloc(d, (size_t)S * tiles * 64 * 64 * 4, &ws)); CK(tinynv_malloc(d, tiles * 4, &cnt));
    CK(tinynv_memcpy_htod(st, dA, hA, na * 2)); CK(tinynv_memcpy_htod(st, dB, hB, nbv * 2)); CK(tinynv_memset(st, cnt, 0, tiles * 4)); CK(tinynv_stream_sync(st));
    const int BK = 32, nk_all = (k + BK - 1) / BK, per = (nk_all + S - 1) / S;
    for (int q = 0; q < S; q++){
      const int t0 = q * per < nk_all ? q * per : nk_all, t1 = t0 + per < nk_all ? t0 + per : nk_all, k0 = t0 * BK, kl = (t1 * BK < k ? t1 * BK : k) - k0;
      if (kl <= 0) continue;
      CK(tinynv_memset(st, dP, 0xff, nc * 4));
      gemm(ks, 0, dA + (uint64_t)k0 * 2, dB + (uint64_t)k0 * 2, dP, m, n, kl, k, k, m, 0, 0, 1);
      CK(tinynv_memcpy_dtoh(st, part, dP, nc * 4)); CK(tinynv_stream_sync(st));
      for (size_t e = 0; e < nc; e++) exp_[e] += part[e];
    }
    for (int pass = 0; pass < 2; pass++){
      const char *row = pass ? "AGAIN" : "EXACT"; printf("  %s: entered\n", row);
      CK(tinynv_memset(st, dC, 0xff, nc * 4));   // all-ones = NaN: a tile nobody wrote cannot pass
      gemm(kk, 1, dA, dB, dC, m, n, k, k, k, m, ws, cnt, S);
      CK(tinynv_memcpy_dtoh(st, got, dC, nc * 4)); CK(tinynv_stream_sync(st));
      size_t bad = 0, first = 0; for (size_t e = 0; e < nc; e++) if (!(got[e] == exp_[e])) { if (!bad) first = e; bad++; }
      if (bad) { printf("  %s FAIL: %zu of %zu differ; first at %zu: got %.9g want %.9g\n", row, bad, nc, first, got[first], exp_[first]); fails++; }
      else printf("  %s PASS: %zu of %zu equal\n", row, nc, nc);
    }
    unsigned *hc = malloc(tiles * 4); CK(tinynv_memcpy_dtoh(st, hc, cnt, tiles * 4)); CK(tinynv_stream_sync(st));
    size_t nz = 0; for (size_t t = 0; t < tiles; t++) nz += hc[t] != 0;
    if (nz) { printf("  COUNTERS FAIL: %zu of %zu nonzero\n", nz, tiles); fails++; } else printf("  COUNTERS PASS: all %zu zero\n", tiles);
    printf("  REF: entered\n");
    double maxerr = 0, maxref = 0; int nonfinite = 0;
    for (int j = 0; j < n; j++) for (int i = 0; i < m; i++){
      double r = 0; for (int t = 0; t < k; t++) r += (double)bf2f(hA[(size_t)i * k + t]) * (double)bf2f(hB[(size_t)j * k + t]);
      double g = got[i + (size_t)j * m]; if (!isfinite(g)) nonfinite++;
      maxerr = fmax(maxerr, fabs(g - r)); maxref = fmax(maxref, fabs(r));
    }
    double rel = maxerr / maxref;
    if (nonfinite || !(rel <= 2e-3)) { printf("  REF FAIL: rel %.3g, %d non-finite\n", rel, nonfinite); fails++; } else printf("  REF PASS: rel %.3g (tol 2e-3)\n", rel);
    tinynv_free(d, dA); tinynv_free(d, dB); tinynv_free(d, dC); tinynv_free(d, dP); tinynv_free(d, ws); tinynv_free(d, cnt);
    free(hA); free(hB); free(exp_); free(part); free(got); free(hc);
  }
  printf("RESULT: %d FAIL(s)%s\n", fails, null_dev ? "  (NULL DEVICE: VOID as evidence)" : "");
  return fails;
}
