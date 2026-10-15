// bench_gemm_contract.cu - the vectorised tc2 GEMM kernels (f16 and bf16) against real cuBLAS, beyond llama.cpp's one
// call shape (G's review of the bf16 fix, 2026-09-21: other transposes, outputs, coefficients, batches and tails were
// unvalidated), and the direct store's alignment contract.
//   bench_gemm_contract <gemm cubin> matrix            every transpose x output x (alpha, beta), strided batch 3, tails
//   bench_gemm_contract <gemm cubin> matrix-control    the same with the kernel's alpha x1.05: every case must be refused
//   bench_gemm_contract <gemm cubin> align <f16|bf16> <ldc33|base16>   ONE case per process: a misaligned store can
//                                                      poison the context, so no case may run after another in the same one
// The reference is cublasGemmStridedBatchedEx with COMPUTE_32F. The gate: max |x - ref| / max |ref|, with any non-finite
// value refused first. The tolerance is 2e-3 for f32 output and 1e-2 for bf16/f16 output (the output rounding alone is up to
// 2^-8 relative for bf16).
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <vector>

#define CK(x) do { CUresult r_ = (x); if (r_) { const char *s_ = 0; cuGetErrorString(r_, &s_); printf("ERROR %s: %d %s (line %d)\n", #x, r_, s_ ? s_ : "", __LINE__); exit(3); } } while (0)
#define CR(x) do { cudaError_t r_ = (x); if (r_) { printf("ERROR %s: %s (line %d)\n", #x, cudaGetErrorString(r_), __LINE__); exit(3); } } while (0)
#define CB(x) do { cublasStatus_t r_ = (x); if (r_) { printf("ERROR %s: cublas %d (line %d)\n", #x, (int)r_, __LINE__); exit(3); } } while (0)

static CUmodule mod;
static float g_alpha_mutant = 1.f;   /* matrix-control: the KERNEL's alpha only, x1.05, so every case must go over tolerance */
static cublasHandle_t h;
enum { F32 = 0, F16 = 1, BF16 = 2 };
static cudaDataType dt(int t) { return t == F32 ? CUDA_R_32F : t == F16 ? CUDA_R_16F : CUDA_R_16BF; }
static int esz(int t) { return t == F32 ? 4 : 2; }
static uint16_t enc(float v, int t) {
	uint16_t r;
	if (t == F16) { __half x = __float2half(v); memcpy(&r, &x, 2); } else { __nv_bfloat16 x = __float2bfloat16(v); memcpy(&r, &x, 2); }
	return r;
}
static float dec(const void *p, size_t i, int t) {
	if (t == F32) return ((const float *)p)[i];
	uint16_t u = ((const uint16_t *)p)[i];
	if (t == F16) { __half x; memcpy(&x, &u, 2); return __half2float(x); }
	__nv_bfloat16 x; memcpy(&x, &u, 2); return __bfloat162float(x);
}
static double relerr(const std::vector<float> &x, const std::vector<float> &ref) {
	double mx = 0, mr = 0;
	for (size_t i = 0; i < x.size(); i++) if (!std::isfinite(x[i]) || !std::isfinite(ref[i])) return INFINITY;
	for (size_t i = 0; i < x.size(); i++) { mx = fmax(mx, fabs((double)x[i] - ref[i])); mr = fmax(mr, fabs((double)ref[i])); }
	return mr == 0 ? INFINITY : mx / mr;
}

/* one case: returns the error of the tc2 kernel against cuBLAS; cbase_off shifts C's base by that many bytes */
static double run_case(int in, int oa, int ob, int out, float alpha, float beta, int m, int n, int k, int batch, int ldc,
                       size_t cbase_off, const char **kname_out) {
	const int lda = oa == 0 ? m : k, ldb = ob == 0 ? k : n;
	const long long sA = (long long)lda * (oa == 0 ? k : m), sB = (long long)ldb * (ob == 0 ? n : k), sC = (long long)ldc * n;
	std::vector<uint16_t> hA((size_t)sA * batch), hB((size_t)sB * batch);
	for (auto &v : hA) v = enc((float)rand() / RAND_MAX * 2.f - 1.f, in);
	for (auto &v : hB) v = enc((float)rand() / RAND_MAX * 2.f - 1.f, in);
	const size_t cbytes = (size_t)sC * batch * esz(out);
	std::vector<unsigned char> hC0(cbytes);
	for (size_t i = 0; i < (size_t)sC * batch; i++) {
		float v = (float)rand() / RAND_MAX * 2.f - 1.f;
		if (out == F32) memcpy(&hC0[i * 4], &v, 4); else { uint16_t u = enc(v, out); memcpy(&hC0[i * 2], &u, 2); }
	}
	void *A, *B; char *Cref, *Ctc;
	CR(cudaMalloc(&A, hA.size() * 2)); CR(cudaMalloc(&B, hB.size() * 2));
	CR(cudaMalloc((void **)&Cref, cbytes + 64)); CR(cudaMalloc((void **)&Ctc, cbytes + 64));
	CR(cudaMemcpy(A, hA.data(), hA.size() * 2, cudaMemcpyHostToDevice)); CR(cudaMemcpy(B, hB.data(), hB.size() * 2, cudaMemcpyHostToDevice));
	CR(cudaMemcpy(Cref, hC0.data(), cbytes, cudaMemcpyHostToDevice));
	CR(cudaMemcpy(Ctc + cbase_off, hC0.data(), cbytes, cudaMemcpyHostToDevice));
	CB(cublasGemmStridedBatchedEx(h, oa ? CUBLAS_OP_T : CUBLAS_OP_N, ob ? CUBLAS_OP_T : CUBLAS_OP_N, m, n, k, &alpha, A, dt(in), lda, sA,
	                              B, dt(in), ldb, sB, &beta, Cref, dt(out), ldc, sC, batch, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
	// the kernel and its tile choice exactly as cublas.c's launch_gemm_tc makes them (for the 5090's 170 SMs)
	long long blocks128 = (long long)((m + 127) / 128) * ((n + 127) / 128) * batch;
	int small = blocks128 < 2LL * 170, tile = small ? 64 : 128, nt = small ? 128 : 256;
	static char kname[64];
	snprintf(kname, sizeof kname, "tinyblas_gemm_%s_tc2%s_%c%c", in == BF16 ? "bf16" : "f16", small ? "s" : "", oa ? 't' : 'n', ob ? 't' : 'n');
	*kname_out = kname;
	CUfunction f; CK(cuModuleGetFunction(&f, mod, kname));
	struct __attribute__((packed)) { uint64_t A, B, C; int64_t sA, sB, sC; int32_t m, n, k, lda, ldb, ldc; float alpha, beta; int32_t out; } q =
		{ (uint64_t)A, (uint64_t)B, (uint64_t)(Ctc + cbase_off), sA, sB, sC, m, n, k, lda, ldb, ldc, alpha * g_alpha_mutant, beta, out };
	size_t len = sizeof q;
	void *extra[] = { CU_LAUNCH_PARAM_BUFFER_POINTER, &q, CU_LAUNCH_PARAM_BUFFER_SIZE, &len, CU_LAUNCH_PARAM_END };
	CK(cuLaunchKernel(f, (m + tile - 1) / tile, (n + tile - 1) / tile, batch, nt, 1, 1, 0, 0, NULL, extra));
	CR(cudaDeviceSynchronize());
	std::vector<unsigned char> r1(cbytes), r2(cbytes);
	CR(cudaMemcpy(r1.data(), Cref, cbytes, cudaMemcpyDeviceToHost)); CR(cudaMemcpy(r2.data(), Ctc + cbase_off, cbytes, cudaMemcpyDeviceToHost));
	// compare only the m x n submatrix of each batch element (ldc may exceed m)
	std::vector<float> ref, got;
	for (int b = 0; b < batch; b++)
		for (int j = 0; j < n; j++)
			for (int i = 0; i < m; i++) {
				size_t idx = (size_t)b * sC + (size_t)j * ldc + i;
				ref.push_back(dec(r1.data(), idx, out)); got.push_back(dec(r2.data(), idx, out));
			}
	cudaFree(A); cudaFree(B); cudaFree(Cref); cudaFree(Ctc);
	return relerr(got, ref);
}

int main(int argc, char **argv) {
	if (argc < 3) { printf("usage: bench_gemm_contract <cubin> matrix | align <f16|bf16> <ldc33|base16>\n"); return 2; }
	CR(cudaFree(0));
	CK(cuModuleLoad(&mod, argv[1]));
	CB(cublasCreate(&h));
	srand(7);
	if (!strcmp(argv[2], "matrix-control")) g_alpha_mutant = 1.05f;
	if (!strcmp(argv[2], "matrix") || !strcmp(argv[2], "matrix-control")) {
		int bad = 0, n = 0;
		for (int in = F16; in <= BF16; in++)
			for (int op = 0; op < 4; op++)
				for (int o = 0; o < 2; o++)
					for (int ab = 0; ab < 2; ab++) {
						int out = o == 0 ? F32 : in;
						float alpha = ab ? 0.5f : 1.f, beta = ab ? 2.f : 0.f;
						const char *kn;
						// m 48 and k 72 are multiples of 8 but not of 16/32: partial fragments and a partial K slab
						double e = run_case(in, op >> 1, op & 1, out, alpha, beta, 48, 40, 72, 3, 48, 0, &kn);
						double tol = out == F32 ? 2e-3 : 1e-2;
						int ok = e <= tol;
						n++; bad += !ok;
						printf("  %-4s %-26s %c%c out %-4s alpha %.1f beta %.1f  m 48 n 40 k 72 batch 3: err %.2e %s\n",
						       ok ? "ok" : "FAIL", kn, (op >> 1) ? 'T' : 'N', (op & 1) ? 'T' : 'N', out == F32 ? "f32" : out == F16 ? "f16" : "bf16",
						       alpha, beta, e, ok ? "" : "<-- over tolerance");
					}
		if (g_alpha_mutant != 1.f) {
			printf("bench_gemm_contract matrix-control (kernel alpha x1.05): %d case(s), %d over tolerance - %s\n", n, bad,
			       bad == n ? "EVERY case refused, as it must" : "SOME CASE PASSED: the matrix cannot see a 5%% error");
			return bad == n ? 0 : 1;
		}
		printf("bench_gemm_contract matrix: %d case(s), %d over tolerance\n", n, bad);
		return bad ? 1 : 0;
	}
	if (!strcmp(argv[2], "align") && argc >= 5) {
		int in = !strcmp(argv[3], "bf16") ? BF16 : F16, ldc33 = !strcmp(argv[4], "ldc33");
		const char *kn;
		// T/N m = n = k = 32, f32 out, alpha 1, beta 0: every fragment fully inside, so the old kernel stores directly
		double e = run_case(in, 1, 0, F32, 1.f, 0.f, 32, 32, 32, 1, ldc33 ? 33 : 32, ldc33 ? 0 : 16, &kn);
		printf("bench_gemm_contract align %s %s (%s): err %.2e %s\n", argv[3], argv[4], kn, e, e <= 2e-3 ? "PASS" : "FAIL");
		return e <= 2e-3 ? 0 : 1;
	}
	printf("unknown mode\n");
	return 2;
}
