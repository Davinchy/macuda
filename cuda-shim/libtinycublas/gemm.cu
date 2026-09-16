// Correctness-first general GEMM for libtinycublas' residual F16/F32 path (ggml only reaches cuBLAS for unquantized GEMMs,
// mostly prompt-time). One kernel, all dtype/transpose cases; a tensor-core version comes later (perf is prompt-only).
// C[m,n] = alpha * op(A)[m,k] * op(B)[k,n] + beta * C, column-major (cuBLAS convention). in AND out dtype each f32/f16/bf16;
// compute is always f32. ggml calls this with Ctype=CUDA_R_16F for f16 matmuls, so writing f32 there overruns a half-sized dst.
#include <cuda_fp16.h>
#include <cuda_bf16.h>
extern "C" __global__ void tinyblas_gemm_f32(
    const void* A, const void* B, void* C, int m, int n, int k,
    int lda, int ldb, int ldc, float alpha, float beta,
    int opA, int opB, int in_dtype, int out_dtype) {   // opA/opB: 0=N,1=T ; dtypes: 0=f32,1=f16,2=bf16
  int col = blockIdx.y * blockDim.y + threadIdx.y;   // n
  int row = blockIdx.x * blockDim.x + threadIdx.x;   // m
  if (row >= m || col >= n) return;
  float acc = 0.f;
  for (int e = 0; e < k; e++) {
    int ai = (opA==0) ? (row + (long)e*lda) : (e + (long)row*lda);   // col-major op(A)
    int bi = (opB==0) ? (e + (long)col*ldb) : (col + (long)e*ldb);
    float a, b;
    if (in_dtype==0) { a=((const float*)A)[ai]; b=((const float*)B)[bi]; }
    else if (in_dtype==1) { a=__half2float(((const __half*)A)[ai]); b=__half2float(((const __half*)B)[bi]); }
    else { a=__bfloat162float(((const __nv_bfloat16*)A)[ai]); b=__bfloat162float(((const __nv_bfloat16*)B)[bi]); }
    acc += a*b;
  }
  long ci = row + (long)col*ldc;
  float prev = 0.f;
  if (beta != 0.f) {
    if (out_dtype==0) prev=((const float*)C)[ci];
    else if (out_dtype==1) prev=__half2float(((const __half*)C)[ci]);
    else prev=__bfloat162float(((const __nv_bfloat16*)C)[ci]);
  }
  float outv = alpha*acc + beta*prev;
  if (out_dtype==0) ((float*)C)[ci]=outv;
  else if (out_dtype==1) ((__half*)C)[ci]=__float2half(outv);
  else ((__nv_bfloat16*)C)[ci]=__float2bfloat16(outv);
}

// Batched triangular solve, the one case ggml's SOLVE_TRI issues when its own fast kernel's limits (n<=64, k<=32) are
// exceeded: cublasStrsmBatched(side=RIGHT, uplo=UPPER, trans=N, diag=NON_UNIT): X * A = alpha * B, solved in place in B,
// A upper-triangular n x n (lda), X/B k x n (ldb), column-major. Each thread owns one row of X and forward-substitutes
// across the columns, reading the already-solved entries back from X. Correctness first; one launch per batch.
extern "C" __global__ void tinyblas_trsm_rupn_f32(const float* A, float* X, int k, int n, int lda, int ldb, float alpha) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;   // row of X
  if (i >= k) return;
  for (int j = 0; j < n; j++) {
    float c = alpha * X[i + (size_t)j * ldb];
    for (int l = 0; l < j; l++) c -= X[i + (size_t)l * ldb] * A[l + (size_t)j * lda];
    X[i + (size_t)j * ldb] = c / A[j + (size_t)j * lda];
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// Tensor-core GEMM for the f16 path (2026-09-14). One SDXL-Turbo UNet step at 512² launched the scalar kernel above
// 9,117 times, once per batch element, with 2,171 f16<->f32 converts around it; that was 1.4 s of a 1.4 s step. This
// one: f16 inputs, f32 accumulate on the tensor cores (wmma m16n16k16), 64x64x32 tiles staged through shared memory,
// four warps each owning a 32x32 quarter, and the whole strided batch in one launch (blockIdx.z). Column-major cuBLAS
// semantics, both transposes, arbitrary m/n/k with zero-padded tiles, alpha/beta, f32/f16/bf16 output. Shared rows are
// padded so every wmma fragment pointer is 32-byte aligned (a requirement, not a preference). Global loads pick the thread
// mapping that walks the contiguous dimension for the given transpose so they coalesce.
#include <mma.h>
using namespace nvcuda;
extern "C" __global__ void __launch_bounds__(128) tinyblas_gemm_f16_tc(
    const __half* A, const __half* B, void* C,
    long long sA, long long sB, long long sC,
    int m, int n, int k, int lda, int ldb, int ldc,
    float alpha, float beta, int opA, int opB, int out_dtype) {
  enum { BM = 64, BN = 64, BK = 32, APAD = 16, BPAD = 16, CPAD = 8, NT = 128 };
  __shared__ __align__(32) __half As[BM][BK + APAD];   // As[mi][ki] = op(A)[m0+mi][k0+ki]   (row = m, col = k)
  __shared__ __align__(32) __half Bs[BK][BN + BPAD];   // Bs[ki][ni] = op(B)[k0+ki][n0+ni]   (row = k, col = n)
  __shared__ __align__(32) float  Cs[BM][BN + CPAD];
  A += (long long)blockIdx.z * sA;
  B += (long long)blockIdx.z * sB;
  char* Cb = (char*)C + (long long)blockIdx.z * sC * (out_dtype == 0 ? 4 : 2);
  const int tid = threadIdx.x, warp = tid >> 5, wm = warp >> 1, wn = warp & 1;
  const int m0 = blockIdx.x * BM, n0 = blockIdx.y * BN;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
  #pragma unroll
  for (int i = 0; i < 2; i++)
    #pragma unroll
    for (int j = 0; j < 2; j++) wmma::fill_fragment(acc[i][j], 0.f);
  const __half zero = __float2half(0.f);
  for (int k0 = 0; k0 < k; k0 += BK) {
    if (opA == 0) {            // A is m x k col-major: consecutive m are contiguous -> threads walk m
      #pragma unroll
      for (int i = 0; i < (BM * BK) / NT; i++) { int idx = tid + i * NT; int ki = idx / BM, mi = idx % BM; int gm = m0 + mi, gk = k0 + ki;
        As[mi][ki] = (gm < m && gk < k) ? A[gm + (long long)gk * lda] : zero; }
    } else {                   // op(A) = A^T, A stored k x m: consecutive k are contiguous -> threads walk k
      #pragma unroll
      for (int i = 0; i < (BM * BK) / NT; i++) { int idx = tid + i * NT; int mi = idx / BK, ki = idx % BK; int gm = m0 + mi, gk = k0 + ki;
        As[mi][ki] = (gm < m && gk < k) ? A[gk + (long long)gm * lda] : zero; }
    }
    if (opB == 0) {            // B is k x n col-major: consecutive k contiguous -> threads walk k
      #pragma unroll
      for (int i = 0; i < (BK * BN) / NT; i++) { int idx = tid + i * NT; int ni = idx / BK, ki = idx % BK; int gn = n0 + ni, gk = k0 + ki;
        Bs[ki][ni] = (gk < k && gn < n) ? B[gk + (long long)gn * ldb] : zero; }
    } else {                   // op(B) = B^T, B stored n x k: consecutive n contiguous -> threads walk n
      #pragma unroll
      for (int i = 0; i < (BK * BN) / NT; i++) { int idx = tid + i * NT; int ki = idx / BN, ni = idx % BN; int gn = n0 + ni, gk = k0 + ki;
        Bs[ki][ni] = (gk < k && gn < n) ? B[gn + (long long)gk * ldb] : zero; }
    }
    __syncthreads();
    #pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a[2];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b[2];
      #pragma unroll
      for (int i = 0; i < 2; i++) wmma::load_matrix_sync(a[i], &As[wm * 32 + i * 16][kk], BK + APAD);
      #pragma unroll
      for (int j = 0; j < 2; j++) wmma::load_matrix_sync(b[j], &Bs[kk][wn * 32 + j * 16], BN + BPAD);
      #pragma unroll
      for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 2; j++) wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
    }
    __syncthreads();
  }
  #pragma unroll
  for (int i = 0; i < 2; i++)
    #pragma unroll
    for (int j = 0; j < 2; j++) wmma::store_matrix_sync(&Cs[wm * 32 + i * 16][wn * 32 + j * 16], acc[i][j], BN + CPAD, wmma::mem_row_major);
  __syncthreads();
  for (int idx = tid; idx < BM * BN; idx += NT) {   // consecutive threads walk m: coalesced column-major stores
    int mi = idx % BM, ni = idx / BM; int gm = m0 + mi, gn = n0 + ni;
    if (gm >= m || gn >= n) continue;
    long long ci = gm + (long long)gn * ldc;
    float v = alpha * Cs[mi][ni];
    if (beta != 0.f) {
      float prev = out_dtype == 0 ? ((const float*)Cb)[ci]
                 : out_dtype == 1 ? __half2float(((const __half*)Cb)[ci])
                                  : __bfloat162float(((const __nv_bfloat16*)Cb)[ci]);
      v += beta * prev;
    }
    if (out_dtype == 0) ((float*)Cb)[ci] = v;
    else if (out_dtype == 1) ((__half*)Cb)[ci] = __float2half(v);
    else ((__nv_bfloat16*)Cb)[ci] = __float2bfloat16(v);
  }
}

// f32-input GEMM on the tensor cores in TF32 (wmma m16n16k8). ggml asks cuBLAS for f32 GEMMs whenever a matmul carries
// GGML_PREC_F32 (stable-diffusion.cpp sets it on attention), and on a real card those run on TF32 tensor cores because
// ggml selects CUBLAS_TF32_TENSOR_OP_MATH - so this is the faithful equivalent, not a shortcut: f32 range, 10-bit
// mantissa on the inputs, f32 accumulation. Same tiling and batching as the f16 kernel; inputs are staged as f32 and
// rounded to TF32 inside the fragments, as the wmma contract requires.
extern "C" __global__ void __launch_bounds__(128) tinyblas_gemm_f32_tc(
    const float* A, const float* B, void* C,
    long long sA, long long sB, long long sC,
    int m, int n, int k, int lda, int ldb, int ldc,
    float alpha, float beta, int opA, int opB, int out_dtype) {
  enum { BM = 64, BN = 64, BK = 32, APAD = 8, BPAD = 8, CPAD = 8, NT = 128 };
  __shared__ __align__(32) float As[BM][BK + APAD];
  __shared__ __align__(32) float Bs[BK][BN + BPAD];
  __shared__ __align__(32) float Cs[BM][BN + CPAD];
  A += (long long)blockIdx.z * sA;
  B += (long long)blockIdx.z * sB;
  char* Cb = (char*)C + (long long)blockIdx.z * sC * (out_dtype == 0 ? 4 : 2);
  const int tid = threadIdx.x, warp = tid >> 5, wm = warp >> 1, wn = warp & 1;
  const int m0 = blockIdx.x * BM, n0 = blockIdx.y * BN;
  wmma::fragment<wmma::accumulator, 16, 16, 8, float> acc[2][2];
  #pragma unroll
  for (int i = 0; i < 2; i++)
    #pragma unroll
    for (int j = 0; j < 2; j++) wmma::fill_fragment(acc[i][j], 0.f);
  for (int k0 = 0; k0 < k; k0 += BK) {
    if (opA == 0) {
      #pragma unroll
      for (int i = 0; i < (BM * BK) / NT; i++) { int idx = tid + i * NT; int ki = idx / BM, mi = idx % BM; int gm = m0 + mi, gk = k0 + ki;
        As[mi][ki] = (gm < m && gk < k) ? A[gm + (long long)gk * lda] : 0.f; }
    } else {
      #pragma unroll
      for (int i = 0; i < (BM * BK) / NT; i++) { int idx = tid + i * NT; int mi = idx / BK, ki = idx % BK; int gm = m0 + mi, gk = k0 + ki;
        As[mi][ki] = (gm < m && gk < k) ? A[gk + (long long)gm * lda] : 0.f; }
    }
    if (opB == 0) {
      #pragma unroll
      for (int i = 0; i < (BK * BN) / NT; i++) { int idx = tid + i * NT; int ni = idx / BK, ki = idx % BK; int gn = n0 + ni, gk = k0 + ki;
        Bs[ki][ni] = (gk < k && gn < n) ? B[gk + (long long)gn * ldb] : 0.f; }
    } else {
      #pragma unroll
      for (int i = 0; i < (BK * BN) / NT; i++) { int idx = tid + i * NT; int ki = idx / BN, ni = idx % BN; int gn = n0 + ni, gk = k0 + ki;
        Bs[ki][ni] = (gk < k && gn < n) ? B[gn + (long long)gk * ldb] : 0.f; }
    }
    __syncthreads();
    #pragma unroll
    for (int kk = 0; kk < BK; kk += 8) {
      wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> a[2];
      wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major> b[2];
      #pragma unroll
      for (int i = 0; i < 2; i++) { wmma::load_matrix_sync(a[i], &As[wm * 32 + i * 16][kk], BK + APAD);
        #pragma unroll
        for (int t = 0; t < a[i].num_elements; t++) a[i].x[t] = wmma::__float_to_tf32(a[i].x[t]); }
      #pragma unroll
      for (int j = 0; j < 2; j++) { wmma::load_matrix_sync(b[j], &Bs[kk][wn * 32 + j * 16], BN + BPAD);
        #pragma unroll
        for (int t = 0; t < b[j].num_elements; t++) b[j].x[t] = wmma::__float_to_tf32(b[j].x[t]); }
      #pragma unroll
      for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 2; j++) wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
    }
    __syncthreads();
  }
  #pragma unroll
  for (int i = 0; i < 2; i++)
    #pragma unroll
    for (int j = 0; j < 2; j++) wmma::store_matrix_sync(&Cs[wm * 32 + i * 16][wn * 32 + j * 16], acc[i][j], BN + CPAD, wmma::mem_row_major);
  __syncthreads();
  for (int idx = tid; idx < BM * BN; idx += NT) {
    int mi = idx % BM, ni = idx / BM; int gm = m0 + mi, gn = n0 + ni;
    if (gm >= m || gn >= n) continue;
    long long ci = gm + (long long)gn * ldc;
    float v = alpha * Cs[mi][ni];
    if (beta != 0.f) {
      float prev = out_dtype == 0 ? ((const float*)Cb)[ci]
                 : out_dtype == 1 ? __half2float(((const __half*)Cb)[ci])
                                  : __bfloat162float(((const __nv_bfloat16*)Cb)[ci]);
      v += beta * prev;
    }
    if (out_dtype == 0) ((float*)Cb)[ci] = v;
    else if (out_dtype == 1) ((__half*)Cb)[ci] = __float2half(v);
    else ((__nv_bfloat16*)Cb)[ci] = __float2bfloat16(v);
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// f16 GEMM, second cut (2026-09-14, after the first profile of a diffusion step put the first cut at 173 us a launch):
// 128x128x32 tiles, 8 warps each owning 64x32, 16-byte vector loads along whichever dimension the transpose leaves
// contiguous, staged with cp.async into a double-buffered shared tile so the next K-slab loads while this one multiplies.
// The wmma layout follows the storage: op(A) with m contiguous is a col_major matrix_a fragment over As[k][m], with k
// contiguous a row_major one over As[m][k]; likewise for B. Four instantiations, one per transpose pair, selected on the
// host. Requires the contiguous extent and its leading dimension to be multiples of 8 elements and the base pointers
// 16-byte aligned; the host falls back to the first-cut kernel otherwise. Same epilogue: f32 tile through shared memory,
// alpha/beta, f32/f16/bf16 out, coalesced column-major stores.
#include <cuda_pipeline.h>
template <int OPA, int OPB, int BM, int BN, int WN_FRAGS>
__device__ __forceinline__ void tinyblas_gemm_f16_tc2(
    const __half* A, const __half* B, void* C,
    long long sA, long long sB, long long sC,
    int m, int n, int k, int lda, int ldb, int ldc,
    float alpha, float beta, int out_dtype) {
  enum { BK = 32, PAD = 8, NBUF = 2, NWARPS = (BM / 32) * (BN / (16 * WN_FRAGS)), NT = NWARPS * 32, WCOLS = BN / (16 * WN_FRAGS) };
  // A tile: OPA==0 -> As[k][m] (m contiguous, col_major fragment, ldm = BM+PAD); OPA==1 -> As[m][k] (row_major, ldm = BK+PAD)
  // B tile: OPB==0 -> Bs[n][k] (k contiguous, col_major fragment, ldm = BK+PAD); OPB==1 -> Bs[k][n] (row_major, ldm = BN+PAD)
  constexpr int A_ROWS = OPA == 0 ? BK : BM, A_COLS = OPA == 0 ? BM : BK, A_LD = A_COLS + PAD;
  constexpr int B_ROWS = OPB == 0 ? BN : BK, B_COLS = OPB == 0 ? BK : BN, B_LD = B_COLS + PAD;
  __shared__ __align__(128) __half As[NBUF][A_ROWS * A_LD];
  __shared__ __align__(128) __half Bs[NBUF][B_ROWS * B_LD];
  __shared__ __align__(128) float  stage[NWARPS][16 * 16];   // one 16x16 f32 fragment per warp for the general epilogue
  A += (long long)blockIdx.z * sA;
  B += (long long)blockIdx.z * sB;
  char* Cb = (char*)C + (long long)blockIdx.z * sC * (out_dtype == 0 ? 4 : 2);
  const int tid = threadIdx.x, warp = tid >> 5, wm = warp / WCOLS, wn = warp % WCOLS;   // each warp: 32 rows x (16*WN_FRAGS) cols
  const int m0 = blockIdx.x * BM, n0 = blockIdx.y * BN;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][WN_FRAGS];
  #pragma unroll
  for (int i = 0; i < 2; i++)
    #pragma unroll
    for (int j = 0; j < WN_FRAGS; j++) wmma::fill_fragment(acc[i][j], 0.f);
  // one 16-byte chunk = 8 halves; (A_ROWS*A_COLS/8)/NT chunks per thread for A, likewise for B
  auto load_tiles = [&](int buf, int k0) {
    #pragma unroll
    for (int c = 0; c < (A_ROWS * A_COLS / 8) / NT; c++) {
      int chunk = tid + c * NT; int row = chunk / (A_COLS / 8), col8 = (chunk % (A_COLS / 8)) * 8;
      const __half* src; bool ok;
      if (OPA == 0) { int gk = k0 + row, gm = m0 + col8; ok = gk < k && gm < m; src = A + gm + (long long)gk * lda; }   // As[k][m]: 8 consecutive m
      else          { int gm = m0 + row, gk = k0 + col8; ok = gm < m && gk < k; src = A + gk + (long long)gm * lda; }   // As[m][k]: 8 consecutive k
      __half* dst = &As[buf][row * A_LD + col8];
      if (ok) __pipeline_memcpy_async(dst, src, 16); else *(uint4*)dst = make_uint4(0, 0, 0, 0);
    }
    #pragma unroll
    for (int c = 0; c < (B_ROWS * B_COLS / 8) / NT; c++) {
      int chunk = tid + c * NT; int row = chunk / (B_COLS / 8), col8 = (chunk % (B_COLS / 8)) * 8;
      const __half* src; bool ok;
      if (OPB == 0) { int gn = n0 + row, gk = k0 + col8; ok = gn < n && gk < k; src = B + gk + (long long)gn * ldb; }   // Bs[n][k]: 8 consecutive k
      else          { int gk = k0 + row, gn = n0 + col8; ok = gk < k && gn < n; src = B + gn + (long long)gk * ldb; }   // Bs[k][n]: 8 consecutive n
      __half* dst = &Bs[buf][row * B_LD + col8];
      if (ok) __pipeline_memcpy_async(dst, src, 16); else *(uint4*)dst = make_uint4(0, 0, 0, 0);
    }
    __pipeline_commit();
  };
  const int nk = (k + BK - 1) / BK;
  load_tiles(0, 0);
  for (int t = 0; t < nk; t++) {
    const int buf = t & 1;
    if (t + 1 < nk) load_tiles(buf ^ 1, (t + 1) * BK);
    if (t + 1 < nk) __pipeline_wait_prior(1); else __pipeline_wait_prior(0);
    __syncthreads();
    #pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      #pragma unroll
      for (int i = 0; i < 2; i++) {
        const int mi = wm * 32 + i * 16;
        if (OPA == 0) { wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::col_major> a; wmma::load_matrix_sync(a, &As[buf][kk * A_LD + mi], A_LD);
          #pragma unroll
          for (int j = 0; j < WN_FRAGS; j++) { const int ni = wn * (16 * WN_FRAGS) + j * 16;
            if (OPB == 0) { wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b; wmma::load_matrix_sync(b, &Bs[buf][ni * B_LD + kk], B_LD); wmma::mma_sync(acc[i][j], a, b, acc[i][j]); }
            else          { wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b; wmma::load_matrix_sync(b, &Bs[buf][kk * B_LD + ni], B_LD); wmma::mma_sync(acc[i][j], a, b, acc[i][j]); } }
        } else { wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a; wmma::load_matrix_sync(a, &As[buf][mi * A_LD + kk], A_LD);
          #pragma unroll
          for (int j = 0; j < WN_FRAGS; j++) { const int ni = wn * (16 * WN_FRAGS) + j * 16;
            if (OPB == 0) { wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b; wmma::load_matrix_sync(b, &Bs[buf][ni * B_LD + kk], B_LD); wmma::mma_sync(acc[i][j], a, b, acc[i][j]); }
            else          { wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b; wmma::load_matrix_sync(b, &Bs[buf][kk * B_LD + ni], B_LD); wmma::mma_sync(acc[i][j], a, b, acc[i][j]); } }
        }
      }
    }
    __syncthreads();
  }
  // Epilogue, fragment by fragment. A fragment that is fully inside the matrix with alpha 1, beta 0 and f32 output goes
  // straight to memory as a column-major 16x16 block; everything else is staged through the warp's own 16x16 buffer so
  // alpha, beta, bounds and the output type are handled per element.
  const int lane = tid & 31;
  #pragma unroll
  for (int i = 0; i < 2; i++) {
    #pragma unroll
    for (int j = 0; j < WN_FRAGS; j++) {
      const int gm0 = m0 + wm * 32 + i * 16, gn0 = n0 + wn * (16 * WN_FRAGS) + j * 16;
      if (gm0 >= m || gn0 >= n) continue;
      if (out_dtype == 0 && alpha == 1.f && beta == 0.f && gm0 + 16 <= m && gn0 + 16 <= n) {
        wmma::store_matrix_sync((float*)Cb + gm0 + (long long)gn0 * ldc, acc[i][j], ldc, wmma::mem_col_major);
        continue;
      }
      wmma::store_matrix_sync(stage[warp], acc[i][j], 16, wmma::mem_col_major);   // stage[mi + ni*16]
      __syncwarp();
      #pragma unroll
      for (int e = 0; e < 8; e++) {
        const int idx = lane + e * 32, mi = idx & 15, ni = idx >> 4;
        const int gm = gm0 + mi, gn = gn0 + ni;
        if (gm < m && gn < n) {
          const long long ci = gm + (long long)gn * ldc;
          float v = alpha * stage[warp][idx];
          if (beta != 0.f) {
            float prev = out_dtype == 0 ? ((const float*)Cb)[ci] : out_dtype == 1 ? __half2float(((const __half*)Cb)[ci]) : __bfloat162float(((const __nv_bfloat16*)Cb)[ci]);
            v += beta * prev;
          }
          if (out_dtype == 0) ((float*)Cb)[ci] = v; else if (out_dtype == 1) ((__half*)Cb)[ci] = __float2half(v); else ((__nv_bfloat16*)Cb)[ci] = __float2bfloat16(v);
        }
      }
      __syncwarp();
    }
  }
}
extern "C" __global__ void __launch_bounds__(256) tinyblas_gemm_f16_tc2_nn(const __half* A, const __half* B, void* C, long long sA, long long sB, long long sC, int m, int n, int k, int lda, int ldb, int ldc, float alpha, float beta, int out_dtype) { tinyblas_gemm_f16_tc2<0,0,128,128,4>(A,B,C,sA,sB,sC,m,n,k,lda,ldb,ldc,alpha,beta,out_dtype); }
extern "C" __global__ void __launch_bounds__(128) tinyblas_gemm_f16_tc2s_nn(const __half* A, const __half* B, void* C, long long sA, long long sB, long long sC, int m, int n, int k, int lda, int ldb, int ldc, float alpha, float beta, int out_dtype) { tinyblas_gemm_f16_tc2<0,0,64,64,2>(A,B,C,sA,sB,sC,m,n,k,lda,ldb,ldc,alpha,beta,out_dtype); }
extern "C" __global__ void __launch_bounds__(256) tinyblas_gemm_f16_tc2_nt(const __half* A, const __half* B, void* C, long long sA, long long sB, long long sC, int m, int n, int k, int lda, int ldb, int ldc, float alpha, float beta, int out_dtype) { tinyblas_gemm_f16_tc2<0,1,128,128,4>(A,B,C,sA,sB,sC,m,n,k,lda,ldb,ldc,alpha,beta,out_dtype); }
extern "C" __global__ void __launch_bounds__(128) tinyblas_gemm_f16_tc2s_nt(const __half* A, const __half* B, void* C, long long sA, long long sB, long long sC, int m, int n, int k, int lda, int ldb, int ldc, float alpha, float beta, int out_dtype) { tinyblas_gemm_f16_tc2<0,1,64,64,2>(A,B,C,sA,sB,sC,m,n,k,lda,ldb,ldc,alpha,beta,out_dtype); }
extern "C" __global__ void __launch_bounds__(256) tinyblas_gemm_f16_tc2_tn(const __half* A, const __half* B, void* C, long long sA, long long sB, long long sC, int m, int n, int k, int lda, int ldb, int ldc, float alpha, float beta, int out_dtype) { tinyblas_gemm_f16_tc2<1,0,128,128,4>(A,B,C,sA,sB,sC,m,n,k,lda,ldb,ldc,alpha,beta,out_dtype); }
extern "C" __global__ void __launch_bounds__(128) tinyblas_gemm_f16_tc2s_tn(const __half* A, const __half* B, void* C, long long sA, long long sB, long long sC, int m, int n, int k, int lda, int ldb, int ldc, float alpha, float beta, int out_dtype) { tinyblas_gemm_f16_tc2<1,0,64,64,2>(A,B,C,sA,sB,sC,m,n,k,lda,ldb,ldc,alpha,beta,out_dtype); }
extern "C" __global__ void __launch_bounds__(256) tinyblas_gemm_f16_tc2_tt(const __half* A, const __half* B, void* C, long long sA, long long sB, long long sC, int m, int n, int k, int lda, int ldb, int ldc, float alpha, float beta, int out_dtype) { tinyblas_gemm_f16_tc2<1,1,128,128,4>(A,B,C,sA,sB,sC,m,n,k,lda,ldb,ldc,alpha,beta,out_dtype); }
extern "C" __global__ void __launch_bounds__(128) tinyblas_gemm_f16_tc2s_tt(const __half* A, const __half* B, void* C, long long sA, long long sB, long long sC, int m, int n, int k, int lda, int ldb, int ldc, float alpha, float beta, int out_dtype) { tinyblas_gemm_f16_tc2<1,1,64,64,2>(A,B,C,sA,sB,sC,m,n,k,lda,ldb,ldc,alpha,beta,out_dtype); }

// ---------------------------------------------------------------------------------------------------------------------
// Bandwidth probes (2026-09-14): read-only kernels reach ~1570 GB/s on this card, every read+write copy caps at ~508.
// These two split the direction: one streams 16-byte writes, the other streams 16-byte reads and keeps one word so the
// loads cannot be dropped. Launched by test/launch_copy_bench.c through libtinynv directly.
extern "C" __global__ void tinyblas_bw_write(float4* p, long long n4, float v) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x, stride = (long long)gridDim.x * blockDim.x;
  float4 val = make_float4(v, v, v, v);
  for (; i < n4; i += stride) p[i] = val;
}
extern "C" __global__ void tinyblas_bw_read(const float4* p, long long n4, float* out) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x, stride = (long long)gridDim.x * blockDim.x;
  float acc = 0.f;
  for (; i < n4; i += stride) { float4 x = p[i]; acc += x.x + x.y + x.z + x.w; }
  if (acc == 123456.789f) out[0] = acc;   // practically never true; keeps the loads alive
}
