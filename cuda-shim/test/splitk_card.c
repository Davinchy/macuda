// splitk_card.c - the split-K GEMM on the card, through libtinynv (B, 2026-09-23; D's gate on prereg b79fb72f, item S1).
// The same rows as libtinycublas/splitk-20260923/test_splitk.cu (which went 30/30 on the 3090, sm_86), for the sm_120
// cubin the shim embeds, on the three Qwen2.5-3B decode shapes. Run under the protocol as a probe:
//   sh tools/nv_shim_step.sh B probe <tree>/build/shim/splitk_card <tree>/libtinycublas/gemm.cubin
// Rows, each printed on entry and ending PASS or FAIL; the exit code is the number of FAILs:
//   EXACT     split(S) == the sum IN SLICE ORDER of tc2s run on each slice's k-range (float ==, alpha 1, beta 0, f32 out)
//   AGAIN     the same split GEMM again on the same ws/cnt, C poisoned with NaN in between: EXACT again
//   COUNTERS  every counter reads 0 afterwards
//   REF       split(S) against a double-precision host reference, max|x-ref|/max|ref| <= 2e-3 (f32 out)
//   R         (D, 02:2x) plan-free: max|split - fp64| / max|unsplit tc2s over the FULL k - fp64| <= 2, same inputs. EXACT
//             cuts slices with the kernel's own rule; this row never uses it, so a dropped K remainder fails here.
//   BURST     (D's F1, 03:30) 8 split-K GEMMs launched back to back on ONE stream with NO sync between them, sharing one
//             workspace and one counter array, as one cublas handle does; each output then checked with R against its own
//             fp64 reference (and for non-finite values). This is the row that reads the NORMAL-mode hazard: chained
//             launches with state carried across calls. Run it with the knobs unset.
//   argv[2] = "twostreams": ONLY the burst, alternating across TWO streams that share the workspace - the case the design
//             forbids (ggml keeps one handle per stream). It is the arm that shows BURST can fail; predicted FAIL, unless
//             this driver serialises streams, which would then be the finding.
// The last line is "RESULT: PASS ..." or "RESULT: FAIL ..." - the slot driver keys on it, so no other outcome word may
// appear on it.
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

static void gemm_on(tinynv_stream_t s, int sync, tinynv_kernel_t k, int split, uint64_t A, uint64_t B, uint64_t C, int m, int n, int kk,
                    int lda, int ldb, int ldc, uint64_t ws, uint64_t cnt, int S){
  int64_t z = 0; float alpha = 1.f, beta = 0.f; int out = 0;
  const void *v[18] = { &A, &B, &C, &z, &z, &z, &m, &n, &kk, &lda, &ldb, &ldc, &alpha, &beta, &out, &ws, &cnt, &S };
  const int sz[18] = { 8, 8, 8, 8, 8, 8, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 8, 4 };
  unsigned char p[256]; size_t len = blob(k, p, v, sz, split ? 18 : 15, split ? 108 : 84);
  CK(tinynv_launch(s, k, (m + 63) / 64, (n + 63) / 64, split ? S : 1, 128, 1, 1, 0, p, len));
  if (sync) CK(tinynv_stream_sync(s));
}
static void gemm(tinynv_kernel_t k, int split, uint64_t A, uint64_t B, uint64_t C, int m, int n, int kk, int lda, int ldb, int ldc,
                 uint64_t ws, uint64_t cnt, int S){ gemm_on(st, 1, k, split, A, B, C, m, n, kk, lda, ldb, ldc, ws, cnt, S); }

// BURST: NB split GEMMs of the q/o shape, each on its own inputs and output, one workspace, no sync until all are issued.
static int burst(tinynv_device_t d, tinynv_kernel_t ks, tinynv_kernel_t kk, int nstreams){
  enum { NB = 8 }; const int m = 2048, n = 32, k = 2048, S = 8; const size_t na = (size_t)m * k, nbv = (size_t)n * k, nc = (size_t)m * n, tiles = 32;
  printf("BURST: entered - %d split GEMMs m=%d n=%d k=%d S=%d, one workspace, %d stream(s), no sync between launches\n", NB, m, n, k, S, nstreams);
  tinynv_stream_t s2 = st; if (nstreams == 2) CK(tinynv_stream_create(d, &s2));
  uint16_t *hA = malloc(na * 2 * NB), *hB = malloc(nbv * 2); for (size_t i = 0; i < na * NB; i++) hA[i] = f2bf(frand()); for (size_t i = 0; i < nbv; i++) hB[i] = f2bf(frand());
  tinynv_devptr_t dA, dB, dC, dP, ws, cnt;
  CK(tinynv_malloc(d, na * 2 * NB, &dA)); CK(tinynv_malloc(d, nbv * 2, &dB)); CK(tinynv_malloc(d, nc * 4 * NB, &dC)); CK(tinynv_malloc(d, nc * 4, &dP));
  CK(tinynv_malloc(d, (size_t)S * tiles * 64 * 64 * 4, &ws)); CK(tinynv_malloc(d, tiles * 4, &cnt));
  CK(tinynv_memcpy_htod(st, dA, hA, na * 2 * NB)); CK(tinynv_memcpy_htod(st, dB, hB, nbv * 2)); CK(tinynv_memset(st, cnt, 0, tiles * 4));
  CK(tinynv_memset(st, dC, 0xff, nc * 4 * NB)); CK(tinynv_stream_sync(st));
  for (int i = 0; i < NB; i++)
    gemm_on((i & 1) ? s2 : st, 0, kk, 1, dA + (uint64_t)i * na * 2, dB, dC + (uint64_t)i * nc * 4, m, n, k, k, k, m, ws, cnt, S);
  CK(tinynv_stream_sync(st)); if (s2 != st) CK(tinynv_stream_sync(s2));
  float *got = malloc(nc * 4 * NB), *part = malloc(nc * 4); CK(tinynv_memcpy_dtoh(st, got, dC, nc * 4 * NB)); CK(tinynv_stream_sync(st));
  unsigned hc[32]; CK(tinynv_memcpy_dtoh(st, hc, cnt, sizeof hc)); CK(tinynv_stream_sync(st));
  int nz = 0; for (int t = 0; t < 32; t++) nz += hc[t] != 0;
  int bad = 0; double worst = 0;
  for (int i = 0; i < NB; i++){
    CK(tinynv_memset(st, dP, 0xff, nc * 4)); gemm(ks, 0, dA + (uint64_t)i * na * 2, dB, dP, m, n, k, k, k, m, 0, 0, 1);
    CK(tinynv_memcpy_dtoh(st, part, dP, nc * 4)); CK(tinynv_stream_sync(st));
    const uint16_t *a = hA + (size_t)i * na; const float *g = got + (size_t)i * nc; double e1 = 0, eon = 0; int nf = 0;
    for (int j = 0; j < n; j++) for (int r_ = 0; r_ < m; r_++){
      double r = 0; for (int t = 0; t < k; t++) r += (double)bf2f(a[(size_t)r_ * k + t]) * (double)bf2f(hB[(size_t)j * k + t]);
      size_t e = r_ + (size_t)j * m; nf += !isfinite(part[e]) || !isfinite(g[e]);
      e1 = fmax(e1, fabs((double)part[e] - r)); eon = fmax(eon, fabs((double)g[e] - r));
    }
    double R = nf ? INFINITY : e1 > 0 ? eon / e1 : (eon > 0 ? INFINITY : 1.0);
    printf("  burst[%d] on stream %d: R %.3g%s\n", i, (i & 1) && nstreams == 2 ? 2 : 1, R, nf ? " (non-finite)" : "");
    if (!(R <= 2.0)) bad++; if (R > worst) worst = R;
  }
  int f = bad || nz;
  if (f) printf("  BURST FAIL: %d of %d outputs over R 2 (worst %.3g), %d counter(s) nonzero\n", bad, NB, worst, nz);
  else printf("  BURST PASS: all %d outputs R <= 2 (worst %.3g), counters all zero\n", NB, worst);
  tinynv_free(d, dA); tinynv_free(d, dB); tinynv_free(d, dC); tinynv_free(d, dP); tinynv_free(d, ws); tinynv_free(d, cnt);
  if (s2 != st) tinynv_stream_destroy(s2);
  free(hA); free(hB); free(got); free(part);
  return f;
}

int main(int argc, char **argv){
  // Unbuffered, as test_hw_namedbar already is: a verdict that only exists in the stdio buffer is a verdict the exit
  // path can eat. Both wedges of 2026-09-25 trapped this binary's entire run that way - the drain hung in atexit,
  // which runs BEFORE stdio flush, and the signal that ended the process ended the buffer with it.
  setvbuf(stdout, NULL, _IONBF, 0);
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
  if (argc > 2 && !strcmp(argv[2], "twostreams")) {   // the arm that must be able to FAIL: two streams, one workspace
    int f = burst(d, ks, kk, 2);
    printf("RESULT: %s - twostreams burst%s\n", f ? "FAIL" : "PASS", null_dev ? " (NULL DEVICE: VOID as evidence)" : "");
    return f;
  }
  const struct { const char *name; int m, n, k, S; } cs[] = {
    { "k/v (3B decode)", 256, 32, 2048, 16 }, { "q/o (3B decode)", 2048, 32, 2048, 8 }, { "down (3B decode)", 2048, 32, 11008, 8 },
    { "tails (k remainder)", 200, 20, 1000, 5 } };
  const int ncases = sizeof cs / sizeof cs[0];
  for (int c = 0; c < ncases; c++){
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
    printf("  R: entered\n");
    CK(tinynv_memset(st, dP, 0xff, nc * 4));
    gemm(ks, 0, dA, dB, dP, m, n, k, k, k, m, 0, 0, 1);                     // the unsplit kernel over the full k: no plan
    CK(tinynv_memcpy_dtoh(st, part, dP, nc * 4)); CK(tinynv_stream_sync(st));
    double e1 = 0, eon = 0;
    for (int j = 0; j < n; j++) for (int i = 0; i < m; i++){
      double r = 0; for (int t = 0; t < k; t++) r += (double)bf2f(hA[(size_t)i * k + t]) * (double)bf2f(hB[(size_t)j * k + t]);
      size_t e = i + (size_t)j * m; e1 = fmax(e1, fabs((double)part[e] - r)); eon = fmax(eon, fabs((double)got[e] - r));
    }
    // fmax drops NaN, so a NaN on either side must fail here explicitly (the null run showed R PASS on unwritten output)
    int nf = 0; for (size_t e = 0; e < nc; e++) nf += !isfinite(part[e]) || !isfinite(got[e]);
    double R = nf ? INFINITY : e1 > 0 ? eon / e1 : (eon > 0 ? INFINITY : 1.0);
    if (!(R <= 2.0)) { printf("  R FAIL: split err %.3g vs unsplit %.3g, R %.3g (> 2)\n", eon, e1, R); fails++; }
    else printf("  R PASS: split err %.3g vs unsplit %.3g, R %.3g (<= 2)\n", eon, e1, R);
    tinynv_free(d, dA); tinynv_free(d, dB); tinynv_free(d, dC); tinynv_free(d, dP); tinynv_free(d, ws); tinynv_free(d, cnt);
    free(hA); free(hB); free(exp_); free(part); free(got); free(hc);
  }
  fails += burst(d, ks, kk, 1);
  const int rows = ncases * 5 + 1;
  if (fails) printf("RESULT: FAIL - %d row(s) of %d%s\n", fails, rows, null_dev ? " (NULL DEVICE: VOID as evidence)" : "");
  else printf("RESULT: PASS - all %d rows%s\n", rows, null_dev ? " (NULL DEVICE: VOID as evidence)" : "");
  return fails;
}
