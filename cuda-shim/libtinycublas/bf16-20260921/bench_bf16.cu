// bench_bf16.cu - libtinycublas' bf16 GEMM path against real cuBLAS, on the 3090, with the shapes llama.cpp (ad6c668) issues
// for Qwen2.5-3B at pp256: cublasGemmEx(CUBLAS_OP_T, CUBLAS_OP_N, m, n=256, k, bf16 A lda=k, bf16 B ldb=k, f32 C ldc=m,
// COMPUTE_32F), as logged by the shim on the null device (TINYCUDART_TRACE=1, 2-layer model with the real dimensions).
//   bench_bf16 <gemm_sm86.cubin>
// Per shape: the scalar kernel the shim uses for bf16 today (tinyblas_gemm_f32, in=2), the bf16 tc2 kernel (the fix), and
// cuBLAS. Every result is checked against cuBLAS: max |x - ref| / max |ref|. CONTROL, which must FAIL the check: the f16 tc2
// kernel run on the same bf16 bits (the dtype confusion a careless fix would make); and the gate's own mutants, an all-NaN
// result and one planted NaN, which must be refused (G's review: the first cut's reduction accepted an all-NaN result).
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <vector>

#define CK(x) do { CUresult r_ = (x); if (r_) { const char *s_ = 0; cuGetErrorString(r_, &s_); printf("%s: %d %s (line %d)\n", #x, r_, s_ ? s_ : "", __LINE__); exit(2); } } while (0)
#define CR(x) do { cudaError_t r_ = (x); if (r_) { printf("%s: %s (line %d)\n", #x, cudaGetErrorString(r_), __LINE__); exit(2); } } while (0)
#define CB(x) do { cublasStatus_t r_ = (x); if (r_) { printf("%s: cublas %d (line %d)\n", #x, (int)r_, __LINE__); exit(2); } } while (0)

struct shape { int m, n, k, count; const char *what; };
static const shape S[] = {
	{ 2048, 256, 2048, 72, "Q, O (2 x 36)" },
	{ 256, 256, 2048, 72, "K, V (2 x 36)" },
	{ 11008, 256, 2048, 70, "gate, up (2 x 35; the last layer's FFN runs on 1 token)" },
	{ 2048, 256, 11008, 35, "down (35)" },
};
static const double TOL = 2e-3;   // f32 accumulation of bf16 products in different orders: well under this; a dtype error is O(1)

static CUfunction f_scalar, f_tc2[2], f_f16tc2[2];
static float time_launch(CUfunction f, unsigned gx, unsigned gy, unsigned bx, unsigned by, void *blob, size_t len, int iters)
{
	void *extra[] = { CU_LAUNCH_PARAM_BUFFER_POINTER, blob, CU_LAUNCH_PARAM_BUFFER_SIZE, &len, CU_LAUNCH_PARAM_END };
	cudaEvent_t a, b; CR(cudaEventCreate(&a)); CR(cudaEventCreate(&b));
	CK(cuLaunchKernel(f, gx, gy, 1, bx, by, 1, 0, 0, NULL, extra)); CR(cudaDeviceSynchronize());   // warm-up
	CR(cudaEventRecord(a));
	for (int i = 0; i < iters; i++) CK(cuLaunchKernel(f, gx, gy, 1, bx, by, 1, 0, 0, NULL, extra));
	CR(cudaEventRecord(b)); CR(cudaEventSynchronize(b));
	float ms; CR(cudaEventElapsedTime(&ms, a, b)); return ms / iters;
}
/* THE GATE REFUSES ANY NON-FINITE VALUE FIRST (G's review, 2026-09-21): fmax(mx, NaN) keeps mx, so the reduction below
 * alone read an all-NaN result - an unwritten C, prefilled 0xff - as zero error against a finite reference. */
static double relerr(const std::vector<float> &x, const std::vector<float> &ref)
{
	double mx = 0, mr = 0;
	for (size_t i = 0; i < x.size(); i++) if (!std::isfinite(x[i]) || !std::isfinite(ref[i])) return INFINITY;
	for (size_t i = 0; i < x.size(); i++) { mx = fmax(mx, fabs((double)x[i] - ref[i])); mr = fmax(mr, fabs((double)ref[i])); }
	return mr == 0 ? INFINITY : mx / mr;
}

int main(int argc, char **argv)
{
	if (argc < 2) { printf("usage: bench_bf16 <gemm_sm86.cubin>\n"); return 2; }
	CR(cudaFree(0));
	CUmodule mod; CK(cuModuleLoad(&mod, argv[1]));
	CK(cuModuleGetFunction(&f_scalar, mod, "tinyblas_gemm_f32"));
	CK(cuModuleGetFunction(&f_tc2[0], mod, "tinyblas_gemm_bf16_tc2_tn")); CK(cuModuleGetFunction(&f_tc2[1], mod, "tinyblas_gemm_bf16_tc2s_tn"));
	CK(cuModuleGetFunction(&f_f16tc2[0], mod, "tinyblas_gemm_f16_tc2_tn")); CK(cuModuleGetFunction(&f_f16tc2[1], mod, "tinyblas_gemm_f16_tc2s_tn"));
	cudaDeviceProp p; CR(cudaGetDeviceProperties(&p, 0));
	printf("bench_bf16: %s, sm_%d%d, %d SMs; cubin %s. Kernel choice as the shim's host code makes it for the 5090 (170 SMs).\n",
	       p.name, p.major, p.minor, p.multiProcessorCount, argv[1]);
	cublasHandle_t h; CB(cublasCreate(&h));
	srand(1);
	double t_scalar = 0, t_tc2 = 0, t_cublas = 0;
	int bad = 0;
	for (const shape &s : S) {
		const int m = s.m, n = s.n, k = s.k, lda = k, ldb = k, ldc = m;
		std::vector<__nv_bfloat16> hA((size_t)k * m), hB((size_t)k * n);
		for (auto &v : hA) v = __float2bfloat16((float)rand() / RAND_MAX * 2.f - 1.f);
		for (auto &v : hB) v = __float2bfloat16((float)rand() / RAND_MAX * 2.f - 1.f);
		__nv_bfloat16 *A, *B; float *C;
		CR(cudaMalloc(&A, hA.size() * 2)); CR(cudaMalloc(&B, hB.size() * 2)); CR(cudaMalloc(&C, (size_t)m * n * 4));
		CR(cudaMemcpy(A, hA.data(), hA.size() * 2, cudaMemcpyHostToDevice)); CR(cudaMemcpy(B, hB.data(), hB.size() * 2, cudaMemcpyHostToDevice));
		std::vector<float> ref((size_t)m * n), got((size_t)m * n);
		const float one = 1.f, zero = 0.f;
		// cuBLAS, exactly as ggml calls it
		CB(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &one, A, CUDA_R_16BF, lda, B, CUDA_R_16BF, ldb, &zero, C, CUDA_R_32F, ldc,
		                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
		CR(cudaDeviceSynchronize()); CR(cudaMemcpy(ref.data(), C, ref.size() * 4, cudaMemcpyDeviceToHost));
		cudaEvent_t a, b; CR(cudaEventCreate(&a)); CR(cudaEventCreate(&b));
		const int iters = 20;
		CR(cudaEventRecord(a));
		for (int i = 0; i < iters; i++)
			CB(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &one, A, CUDA_R_16BF, lda, B, CUDA_R_16BF, ldb, &zero, C, CUDA_R_32F, ldc,
			                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
		CR(cudaEventRecord(b)); CR(cudaEventSynchronize(b));
		float ms_cublas; CR(cudaEventElapsedTime(&ms_cublas, a, b)); ms_cublas /= iters;
		// the scalar kernel: tinyblas_gemm_f32(A,B,C,m,n,k,lda,ldb,ldc,alpha,beta,opA,opB,in,out), block 16x16
		struct __attribute__((packed)) { uint64_t A, B, C; int32_t m, n, k, lda, ldb, ldc; float alpha, beta; int32_t opA, opB, in, out; } ps =
			{ (uint64_t)A, (uint64_t)B, (uint64_t)C, m, n, k, lda, ldb, ldc, 1.f, 0.f, 1, 0, 2, 0 };
		CR(cudaMemset(C, 0xff, (size_t)m * n * 4));
		float ms_scalar = time_launch(f_scalar, (m + 15) / 16, (n + 15) / 16, 16, 16, &ps, sizeof ps, s.m * (long long)s.k > 10000000 ? 2 : 5);
		CR(cudaMemcpy(got.data(), C, got.size() * 4, cudaMemcpyDeviceToHost));
		double e_scalar = relerr(got, ref);
		// the bf16 tc2 kernel, chosen as the host code chooses it (cublas.c launch_gemm_tc: small when < 2 x 170 128-tiles)
		long long blocks128 = (long long)((m + 127) / 128) * ((n + 127) / 128);
		int small = blocks128 < 2LL * 170, tile = small ? 64 : 128, nt = small ? 128 : 256;
		struct __attribute__((packed)) { uint64_t A, B, C; int64_t sA, sB, sC; int32_t m, n, k, lda, ldb, ldc; float alpha, beta; int32_t out; } q =
			{ (uint64_t)A, (uint64_t)B, (uint64_t)C, 0, 0, 0, m, n, k, lda, ldb, ldc, 1.f, 0.f, 0 };
		CR(cudaMemset(C, 0xff, (size_t)m * n * 4));
		float ms_tc2 = time_launch(f_tc2[small], (m + tile - 1) / tile, (n + tile - 1) / tile, nt, 1, &q, sizeof q, iters);
		CR(cudaMemcpy(got.data(), C, got.size() * 4, cudaMemcpyDeviceToHost));
		double e_tc2 = relerr(got, ref);
		// THE GATE'S OWN MUTANTS (G): an all-NaN result (C as prefilled, the kernel never run) and the correct result with one
		// NaN planted; both must be refused, and the pre-fix reduction accepted the first
		std::vector<float> nanrow(got.size(), NAN), onenan = got;
		onenan[onenan.size() / 2] = NAN;
		double e_allnan = relerr(nanrow, ref), e_onenan = relerr(onenan, ref);
		if (!(e_allnan > TOL) || !(e_onenan > TOL)) { printf("  THE GATE ACCEPTED A NaN: all-NaN %.1e, one NaN %.1e\n", e_allnan, e_onenan); bad++; }
		// CONTROL: the f16 kernel on the same bf16 bits - the check must refuse it
		CR(cudaMemset(C, 0xff, (size_t)m * n * 4));
		time_launch(f_f16tc2[small], (m + tile - 1) / tile, (n + tile - 1) / tile, nt, 1, &q, sizeof q, 1);
		CR(cudaMemcpy(got.data(), C, got.size() * 4, cudaMemcpyDeviceToHost));
		double e_ctl = relerr(got, ref);
		double gf = 2.0 * m * n * k / 1e9;
		printf("  m %5d n %d k %5d x%-2d %-45s scalar %8.3f ms (%5.2f TF/s, err %.1e)  bf16 tc2%s %7.3f ms (%5.1f TF/s, err %.1e)  "
		       "cuBLAS %7.3f ms (%5.1f TF/s)  CONTROL f16-on-bf16 err %.1e %s\n",
		       m, n, k, s.count, s.what, ms_scalar, gf / ms_scalar, e_scalar, small ? "s" : " ", ms_tc2, gf / ms_tc2, e_tc2,
		       ms_cublas, gf / ms_cublas, e_ctl, e_ctl > TOL ? "REFUSED (as it must)" : "PASSED - THE CHECK IS BLIND");
		printf("      gate mutants: all-NaN err %.1e, one NaN err %.1e - %s\n", e_allnan, e_onenan,
		       e_allnan > TOL && e_onenan > TOL ? "both REFUSED (as they must)" : "ACCEPTED - THE GATE IS BLIND");
		if (e_scalar > TOL || e_tc2 > TOL || !(e_ctl > TOL)) bad++;
		t_scalar += ms_scalar * s.count; t_tc2 += ms_tc2 * s.count; t_cublas += ms_cublas * s.count;
		cudaFree(A); cudaFree(B); cudaFree(C);
	}
	printf("bench_bf16: the bf16 GEMMs of one pp256 on the 36-layer model, on this card: scalar %.1f ms, bf16 tc2 %.1f ms, cuBLAS %.1f ms "
	       "(scalar / tc2 = %.1fx, tc2 / cuBLAS = %.2fx)\n", t_scalar, t_tc2, t_cublas, t_scalar / t_tc2, t_tc2 / t_cublas);
	printf("bench_bf16: %s\n", bad ? "CHECK FAILED" : "every result within tolerance of cuBLAS, and the control refused");
	return bad ? 1 : 0;
}
