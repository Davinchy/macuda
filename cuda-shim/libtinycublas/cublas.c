// libtinycublas — the ~8 cuBLAS entry points ggml-cuda calls, mapped to one GEMM cubin launched via tinynv.h.
// Only the residual unquantized GEMMs reach here (FORCE_MMQ keeps quantized weights on the MMQ path). Correctness first.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "tinynv.h"
#include <cublas_v2.h>   // cuBLAS types (handle/status/operation/computeType) from the CUDA-13 tree; C-compatible

// A GEMM launch goes through the capture when one is running on this stream, exactly as a ggml kernel does.
// Without this the GEMMs are invisible to a graph: they run once, while the caller is capturing, and never on a
// replay - which on 2026-09-19 rendered every stable-diffusion image blank, because sd's text encoder and
// diffusion model are mostly unquantized matmuls and all of them arrive here rather than through libtinycudart.
extern int tinycudart_capturing_nv(tinynv_stream_t s);
extern int tinycudart_capture_launch(tinynv_kernel_t k, const char *name, unsigned gx, unsigned gy, unsigned gz,
                                     unsigned bx, unsigned by, unsigned bz, unsigned smem, const void *params, size_t plen);
static tinynv_status_t blas_launch(tinynv_stream_t s, tinynv_kernel_t k, const char *name, unsigned gx, unsigned gy,
                                   unsigned gz, unsigned bx, unsigned by, unsigned bz, const void *p, size_t plen) {
  if (tinycudart_capturing_nv(s))
    return tinycudart_capture_launch(k, name, gx, gy, gz, bx, by, bz, 0, p, plen) == 0 ? TINYNV_OK : TINYNV_ERR_LAUNCH;
  return tinynv_launch(s, k, gx, gy, gz, bx, by, bz, 0, p, plen);
}
extern tinynv_device_t tinycudart_device(void);       // provided by libtinycudart
extern tinynv_stream_t tinycudart_default_stream(void);
extern int tinycudart_trace(void); extern void tinycudart_count_launch(const char*); extern void tinycudart_time_launch(const char*, double); extern double tinycudart_now_ns(void);   // the runtime's trace/stats, so cuBLAS launches are seen too
extern unsigned char gemm_cubin[]; extern unsigned int gemm_cubin_len;   // gemm.cubin embedded by the Makefile (xxd -i)
static tinynv_module_t g_mod; static int g_mod_tried;
static tinynv_kernel_t tinycublas_kernel(const char* name){   // any kernel in the embedded cubin, resolved once
  static struct { const char* name; tinynv_kernel_t k; int tried; } cache[24]; 
  if (!g_mod && !g_mod_tried){ g_mod_tried=1; if (tinynv_module_load(tinycudart_device(), gemm_cubin, gemm_cubin_len, &g_mod)!=TINYNV_OK){ fprintf(stderr,"[tinycublas] cubin failed to load\n"); g_mod=NULL; } }
  if (!g_mod) return NULL;
  for (int i=0;i<24;i++){
    if (cache[i].name && !strcmp(cache[i].name,name)) return cache[i].k;
    if (!cache[i].name){ cache[i].name=name; if (tinynv_get_kernel(g_mod,name,&cache[i].k)!=TINYNV_OK){ fprintf(stderr,"[tinycublas] %s not in the cubin\n",name); cache[i].k=NULL; } return cache[i].k; }
  }
  return NULL;
}
tinynv_kernel_t tinycublas_gemm_kernel(void){ return tinycublas_kernel("tinyblas_gemm_f32"); }
static tinynv_kernel_t tinycublas_gemm_tc_kernel(void){ return tinycublas_kernel("tinyblas_gemm_f16_tc"); }
static tinynv_kernel_t tinycublas_gemm_tf32_kernel(void){ return tinycublas_kernel("tinyblas_gemm_f32_tc"); }
// The vectorised kernels, f16 (in_dtype 1) or bf16 (in_dtype 2): the same template, instantiated per input type.
static const char* const tc2_names[2][8]={
  {"tinyblas_gemm_f16_tc2_nn","tinyblas_gemm_f16_tc2_nt","tinyblas_gemm_f16_tc2_tn","tinyblas_gemm_f16_tc2_tt",
   "tinyblas_gemm_f16_tc2s_nn","tinyblas_gemm_f16_tc2s_nt","tinyblas_gemm_f16_tc2s_tn","tinyblas_gemm_f16_tc2s_tt"},
  {"tinyblas_gemm_bf16_tc2_nn","tinyblas_gemm_bf16_tc2_nt","tinyblas_gemm_bf16_tc2_tn","tinyblas_gemm_bf16_tc2_tt",
   "tinyblas_gemm_bf16_tc2s_nn","tinyblas_gemm_bf16_tc2s_nt","tinyblas_gemm_bf16_tc2s_tn","tinyblas_gemm_bf16_tc2s_tt"}};
static tinynv_kernel_t tinycublas_gemm_tc2_kernel(int bf16, int opA, int opB, int small){
  return tinycublas_kernel(tc2_names[bf16?1:0][(small?4:0)+(opA?2:0)+(opB?1:0)]); }
static int tc2_enabled(void){ static int v=-1; if(v<0){ const char* e=getenv("TINYCUBLAS_GEMM"); v=!(e&&strcmp(e,"v1")==0); } return v; }
// The tensor-core path takes every f16-input GEMM unless TINYCUBLAS_TC=0 asks for the scalar kernel (a bisect knob).
static int tc_enabled(void){ static int v=-1; if(v<0){ const char* e=getenv("TINYCUBLAS_TC"); v=(e&&*e=='0')?0:1; } return v; }

struct tinyblas_handle { tinynv_stream_t stream; int math_mode; };

static int dtype_code(cudaDataType t){ return t==CUDA_R_32F?0 : t==CUDA_R_16F?1 : t==CUDA_R_16BF?2 : -1; }
// cublas alpha/beta are in the COMPUTE type: CUBLAS_COMPUTE_16F means they are __half, not float. ggml's f16 matmul uses
// COMPUTE_16F, so reading them as float (4 bytes off a 2-byte value) gave a garbage scale that overflowed f16 to inf.
static float half_bits_to_float(uint16_t h){
  uint32_t sign=(uint32_t)(h>>15)&1u, exp=(h>>10)&0x1fu, man=h&0x3ffu, f;
  if(exp==0){ if(man==0) f=sign<<31; else { int e=-1; do { e++; man<<=1; } while(!(man&0x400u)); man&=0x3ffu; f=(sign<<31)|((uint32_t)(127-15+1-e)<<23)|(man<<13); } }
  else if(exp==0x1f) f=(sign<<31)|(0xffu<<23)|(man<<13);
  else f=(sign<<31)|((exp-15+127)<<23)|(man<<13);
  float out; memcpy(&out,&f,4); return out;
}
static float cublas_scalar(const void* p, cublasComputeType_t ct){ return (ct==CUBLAS_COMPUTE_16F || ct==CUBLAS_COMPUTE_16F_PEDANTIC) ? half_bits_to_float(*(const uint16_t*)p) : *(const float*)p; }
// THE TYPE CONTRACT these entry points serve (G's review of the bf16 fix, 2026-09-21). A and B must be ONE type, the compute
// type must be one cuBLAS defines for it, and alpha/beta are read in the compute type. Everything else is refused
// NOT_SUPPORTED, rather than run on a kernel for another input type (A=f16 with B=bf16 used to reach the f16 kernel and the
// reverse the bf16 one) or with its scalars read at the wrong width (every compute type but exactly COMPUTE_16F was read as
// float, 64F and 32I included). Every kernel here accumulates in f32; COMPUTE_16F's half accumulation is not reproduced.
//   A/B f32:  COMPUTE_32F, 32F_PEDANTIC, 32F_FAST_TF32, 32F_FAST_16F, 32F_FAST_16BF (scalars float)
//   A/B f16:  COMPUTE_16F, 16F_PEDANTIC (scalars half; C must be f16, as cuBLAS requires), COMPUTE_32F, 32F_PEDANTIC (float)
//   A/B bf16: COMPUTE_32F, 32F_PEDANTIC (scalars float)
static int gemm_contract(cudaDataType At, cudaDataType Bt, cudaDataType Ct, cublasComputeType_t ct, int *in, int *out){
  *in = dtype_code(At); *out = dtype_code(Ct);
  if (*in < 0 || *out < 0 || dtype_code(Bt) != *in) return -1;
  int c32 = ct==CUBLAS_COMPUTE_32F || ct==CUBLAS_COMPUTE_32F_PEDANTIC;
  int c16 = ct==CUBLAS_COMPUTE_16F || ct==CUBLAS_COMPUTE_16F_PEDANTIC;
  if (*in == 0) return (c32 || ct==CUBLAS_COMPUTE_32F_FAST_TF32 || ct==CUBLAS_COMPUTE_32F_FAST_16F || ct==CUBLAS_COMPUTE_32F_FAST_16BF) ? 0 : -1;
  if (*in == 1) return c32 || (c16 && *out == 1) ? 0 : -1;
  return c32 ? 0 : -1;
}

// One launch for a whole strided batch on the tensor cores: grid (m/64, n/64, batch), 128 threads. Strides are in elements
// of the respective dtype, as cuBLAS defines them; gridDim.z is capped at 65535 so a larger batch goes in chunks.
static cublasStatus_t launch_gemm_tc(struct tinyblas_handle* h, cublasOperation_t opA, cublasOperation_t opB,
    int m,int n,int k, float alpha, const void*A,int lda,long long sA, const void*B,int ldb,long long sB,
    float beta, void*C,int ldc,long long sC, int batch, int in_dtype, int out_dtype){
  // The vectorised f16 kernel needs the contiguous extent and its leading dimension to be multiples of 8 elements and
  // 16-byte aligned bases (per batch element too); everything else takes the first-cut kernel.
  int oa=(opA==CUBLAS_OP_N?0:1), ob=(opB==CUBLAS_OP_N?0:1);
  int v2 = (in_dtype==1 || in_dtype==2) && tc2_enabled()
        && ((oa==0 ? m : k) % 8 == 0) && (lda % 8 == 0) && ((ob==0 ? k : n) % 8 == 0) && (ldb % 8 == 0)
        && (((uintptr_t)A & 15) == 0) && (((uintptr_t)B & 15) == 0) && (sA % 8 == 0) && (sB % 8 == 0);
  if (v2) {
    // 128x128 tiles only when they fill the card (about two blocks per SM); the diffusion mix is mostly m=1280, n=256,
    // which would put 20 blocks on 170 SMs, so those take 64x64 tiles (four times the blocks, half the threads each)
    long long blocks128 = (long long)((m+127)/128) * ((n+127)/128) * batch;
    int small = blocks128 < 2LL * 170;
    tinynv_kernel_t kern = tinycublas_gemm_tc2_kernel(in_dtype==2, oa, ob, small);
    if (!kern) return CUBLAS_STATUS_NOT_INITIALIZED;
    const int tile = small ? 64 : 128, nthreads = small ? 128 : 256;
    // param blob matches tinyblas_gemm_f16_tc2_xx(A,B,C,sA,sB,sC,m,n,k,lda,ldb,ldc,alpha,beta,out): 84 bytes, no padding
    struct __attribute__((packed)) { uint64_t A,B,C; int64_t sA,sB,sC; int32_t m,n,k,lda,ldb,ldc; float alpha,beta; int32_t out; } q;
    q.sA=sA; q.sB=sB; q.sC=sC; q.m=m; q.n=n; q.k=k; q.lda=lda; q.ldb=ldb; q.ldc=ldc; q.alpha=alpha; q.beta=beta; q.out=out_dtype;
    unsigned gx=(unsigned)((m+tile-1)/tile), gy=(unsigned)((n+tile-1)/tile); int osz=out_dtype==0?4:2;
    const char* kname=tc2_names[in_dtype==2?1:0][(small?4:0)+oa*2+ob];
    for (int b0=0; b0<batch; b0+=65535){
      int nb = batch-b0 < 65535 ? batch-b0 : 65535;
      q.A=(uint64_t)((const char*)A+(long long)b0*sA*2); q.B=(uint64_t)((const char*)B+(long long)b0*sB*2); q.C=(uint64_t)((char*)C+(long long)b0*sC*osz);
      if (tinycudart_trace()) fprintf(stderr,"[trace] launch %s grid=(%u,%u,%d) block=(%d,1,1) m=%d n=%d k=%d out=%d\n",kname,gx,gy,nb,nthreads,m,n,k,out_dtype);
      tinycudart_count_launch(kname); double t0=tinycudart_now_ns();
      tinynv_status_t st = blas_launch(h?h->stream:tinycudart_default_stream(), kern, kname, gx,gy,(unsigned)nb, (unsigned)nthreads,1,1, &q, sizeof(q));
      tinycudart_time_launch(kname, tinycudart_now_ns()-t0);
      if (st!=TINYNV_OK) return CUBLAS_STATUS_EXECUTION_FAILED;
    }
    return CUBLAS_STATUS_SUCCESS;
  }
  // bf16 has only the vectorised kernels: a bf16 GEMM they cannot take goes back to the scalar kernel, never to the f16 one
  if (in_dtype==2) return CUBLAS_STATUS_NOT_SUPPORTED;
  tinynv_kernel_t kern = in_dtype==0 ? tinycublas_gemm_tf32_kernel() : tinycublas_gemm_tc_kernel();
  if (!kern) return CUBLAS_STATUS_NOT_INITIALIZED;
  const int esz = in_dtype==0 ? 4 : 2; const char* kname = in_dtype==0 ? "tinyblas_gemm_f32_tc" : "tinyblas_gemm_f16_tc";
  // param blob matches tinyblas_gemm_f16_tc(A,B,C,sA,sB,sC,m,n,k,lda,ldb,ldc,alpha,beta,opA,opB,out): 92 bytes, no padding
  struct __attribute__((packed)) { uint64_t A,B,C; int64_t sA,sB,sC; int32_t m,n,k,lda,ldb,ldc; float alpha,beta; int32_t opA,opB,out; } p;
  p.sA=sA; p.sB=sB; p.sC=sC; p.m=m; p.n=n; p.k=k; p.lda=lda; p.ldb=ldb; p.ldc=ldc; p.alpha=alpha; p.beta=beta;
  p.opA=(opA==CUBLAS_OP_N?0:1); p.opB=(opB==CUBLAS_OP_N?0:1); p.out=out_dtype;
  unsigned gx=(unsigned)((m+63)/64), gy=(unsigned)((n+63)/64);
  int osz = out_dtype==0?4:2;
  for (int b0=0; b0<batch; b0+=65535){
    int nb = batch-b0 < 65535 ? batch-b0 : 65535;
    p.A=(uint64_t)((const char*)A+(long long)b0*sA*esz); p.B=(uint64_t)((const char*)B+(long long)b0*sB*esz); p.C=(uint64_t)((char*)C+(long long)b0*sC*osz);
    if (tinycudart_trace()) fprintf(stderr,"[trace] launch %s grid=(%u,%u,%d) block=(128,1,1) m=%d n=%d k=%d out=%d\n",kname,gx,gy,nb,m,n,k,out_dtype);
    tinycudart_count_launch(kname); double t0=tinycudart_now_ns();
    tinynv_status_t s = blas_launch(h?h->stream:tinycudart_default_stream(), kern, kname, gx,gy,(unsigned)nb, 128,1,1, &p, sizeof(p));
    tinycudart_time_launch(kname, tinycudart_now_ns()-t0);
    if (s!=TINYNV_OK) return CUBLAS_STATUS_EXECUTION_FAILED;
  }
  return CUBLAS_STATUS_SUCCESS;
}

static cublasStatus_t launch_gemm(struct tinyblas_handle* h, cublasOperation_t opA, cublasOperation_t opB,
    int m,int n,int k, float alpha, const void*A,int lda, const void*B,int ldb, float beta, void*C,int ldc, int in_dtype, int out_dtype){
  if ((in_dtype==1 || in_dtype==0) && tc_enabled() && (in_dtype==0 ? tinycublas_gemm_tf32_kernel() : tinycublas_gemm_tc_kernel()))
    return launch_gemm_tc(h, opA,opB, m,n,k, alpha, A,lda,0, B,ldb,0, beta, C,ldc,0, 1, in_dtype, out_dtype);
  // bf16 (2026-09-21): the vectorised tensor-core kernels when the shape allows them, as f16 does; the scalar kernel
  // below only for a shape they refuse. This used to take the scalar kernel always, which held a bf16 Qwen2.5-3B prompt
  // to 307 tok/s on the 5090 (logs/shim-bench-20260921-141436.log): 62.6x slower than the vectorised kernel on the
  // 3090, measured over that model's pp256 shapes.
  if (in_dtype==2 && tc_enabled()) {
    cublasStatus_t st = launch_gemm_tc(h, opA,opB, m,n,k, alpha, A,lda,0, B,ldb,0, beta, C,ldc,0, 1, in_dtype, out_dtype);
    if (st != CUBLAS_STATUS_NOT_SUPPORTED) return st;
  }
  tinynv_kernel_t kern = tinycublas_gemm_kernel();
  if (!kern) return CUBLAS_STATUS_NOT_INITIALIZED;
  // param blob matches tinyblas_gemm_f32(A,B,C,m,n,k,lda,ldb,ldc,alpha,beta,opA,opB,in_dtype)
  struct __attribute__((packed)) { uint64_t A,B,C; int32_t m,n,k,lda,ldb,ldc; float alpha,beta; int32_t opA,opB,in,out; } p;
  p.A=(uint64_t)A; p.B=(uint64_t)B; p.C=(uint64_t)C; p.m=m; p.n=n; p.k=k; p.lda=lda; p.ldb=ldb; p.ldc=ldc;
  p.alpha=alpha; p.beta=beta; p.opA=(opA==CUBLAS_OP_N?0:1); p.opB=(opB==CUBLAS_OP_N?0:1); p.in=in_dtype; p.out=out_dtype;
  unsigned bx=16, by=16, gx=(m+bx-1)/bx, gy=(n+by-1)/by;
  if (tinycudart_trace()) fprintf(stderr,"[trace] launch tinyblas_gemm_f32 grid=(%u,%u,1) block=(%u,%u,1) m=%d n=%d k=%d in=%d out=%d\n",gx,gy,bx,by,m,n,k,in_dtype,out_dtype);
  tinycudart_count_launch("tinyblas_gemm_f32"); double t0=tinycudart_now_ns();
  tinynv_status_t s = blas_launch(h?h->stream:tinycudart_default_stream(), kern, "tinyblas_gemm_f32", gx,gy,1, bx,by,1, &p, sizeof(p));
  tinycudart_time_launch("tinyblas_gemm_f32", tinycudart_now_ns()-t0);
  return s==TINYNV_OK ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_EXECUTION_FAILED;
}

cublasStatus_t cublasCreate_v2(cublasHandle_t* h){ struct tinyblas_handle* t=calloc(1,sizeof(*t)); t->stream=tinycudart_default_stream(); *h=(cublasHandle_t)t;
  return tinycublas_gemm_kernel() ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_NOT_INITIALIZED; }
cublasStatus_t cublasDestroy_v2(cublasHandle_t h){ free(h); return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasSetStream_v2(cublasHandle_t h, cudaStream_t s){ ((struct tinyblas_handle*)h)->stream=(tinynv_stream_t)s; return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasSetMathMode(cublasHandle_t h, cublasMath_t m){ ((struct tinyblas_handle*)h)->math_mode=(int)m; return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasSgemm_v2(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m,int n,int k,
    const float* alpha, const float* A,int lda, const float* B,int ldb, const float* beta, float* C,int ldc){
  return launch_gemm((struct tinyblas_handle*)h, ta,tb, m,n,k, *alpha,A,lda,B,ldb,*beta,C,ldc, 0, 0);
}
cublasStatus_t cublasGemmEx(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m,int n,int k,
    const void* alpha, const void* A, cudaDataType Atype,int lda, const void* B, cudaDataType Btype,int ldb,
    const void* beta, void* C, cudaDataType Ctype,int ldc, cublasComputeType_t ct, cublasGemmAlgo_t algo){
  (void)algo; int in, out; if (gemm_contract(Atype, Btype, Ctype, ct, &in, &out)) return CUBLAS_STATUS_NOT_SUPPORTED;
  return launch_gemm((struct tinyblas_handle*)h, ta,tb, m,n,k, cublas_scalar(alpha,ct),A,lda,B,ldb,cublas_scalar(beta,ct),C,ldc, in, out);
}
// strided/batched: loop the single-GEMM launch (correctness-first; a batched kernel comes with the tensor-core version)
cublasStatus_t cublasGemmStridedBatchedEx(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m,int n,int k,
    const void* alpha, const void* A, cudaDataType At,int lda, long long sA, const void* B, cudaDataType Bt,int ldb, long long sB,
    const void* beta, void* C, cudaDataType Ct,int ldc, long long sC, int batch, cublasComputeType_t ct, cublasGemmAlgo_t algo){
  (void)algo; int in, out; if (gemm_contract(At, Bt, Ct, ct, &in, &out)) return CUBLAS_STATUS_NOT_SUPPORTED;
  float al=cublas_scalar(alpha,ct), be=cublas_scalar(beta,ct);
  int esz = in==0?4:2, osz = out==0?4:2;   // A/B stride in input elements, C stride in output elements
  if ((in==1 || in==0) && tc_enabled() && (in==0 ? tinycublas_gemm_tf32_kernel() : tinycublas_gemm_tc_kernel()))
    return launch_gemm_tc((struct tinyblas_handle*)h, ta,tb, m,n,k, al, A,lda,sA, B,ldb,sB, be, C,ldc,sC, batch, in, out);
  if (in==2 && tc_enabled()) {   // bf16: one batched launch on the vectorised kernels when they take it, as f16
    cublasStatus_t st = launch_gemm_tc((struct tinyblas_handle*)h, ta,tb, m,n,k, al, A,lda,sA, B,ldb,sB, be, C,ldc,sC, batch, in, out);
    if (st != CUBLAS_STATUS_NOT_SUPPORTED) return st;
  }
  for (int i=0;i<batch;i++){
    cublasStatus_t s=launch_gemm((struct tinyblas_handle*)h, ta,tb, m,n,k, al,
      (const char*)A+(long)i*sA*esz,lda, (const char*)B+(long)i*sB*esz,ldb, be,
      (char*)C+(long)i*sC*osz,ldc, in, out);
    if (s!=CUBLAS_STATUS_SUCCESS) return s;
  }
  return CUBLAS_STATUS_SUCCESS;
}
const char* tinycublas_note(void){ return "libtinycublas: residual GEMM subset over one tinyblas_gemm cubin"; }

// --- the rest of the linked surface ---
const char* cublasGetStatusString(cublasStatus_t s){ switch(s){ case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS"; case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
  case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED"; case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED"; default: return "CUBLAS_STATUS_ERROR"; } }
cublasStatus_t cublasSetWorkspace_v2(cublasHandle_t h, void* ws, size_t n){ (void)h;(void)ws;(void)n; return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasSgemmStridedBatched(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m,int n,int k, const float* alpha,
    const float* A,int lda,long long sA, const float* B,int ldb,long long sB, const float* beta, float* C,int ldc,long long sC, int batch){
  return cublasGemmStridedBatchedEx(h,ta,tb,m,n,k,alpha,A,CUDA_R_32F,lda,sA,B,CUDA_R_32F,ldb,sB,beta,C,CUDA_R_32F,ldc,sC,batch,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT);
}
// pointer-array batches: the arrays of matrix pointers live in DEVICE memory, so read them back before looping
static cublasStatus_t batched(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m,int n,int k, float alpha,
    const void* const* Ad,int lda, const void* const* Bd,int ldb, float beta, void* const* Cd,int ldc, int batch, int in, int out){
  if (batch<=0) return CUBLAS_STATUS_SUCCESS;
  void **A=malloc(sizeof(void*)*batch), **B=malloc(sizeof(void*)*batch), **C=malloc(sizeof(void*)*batch);
  tinynv_stream_t s=((struct tinyblas_handle*)h)->stream; cublasStatus_t rc=CUBLAS_STATUS_EXECUTION_FAILED;
  if (tinynv_memcpy_dtoh(s,A,(tinynv_devptr_t)Ad,sizeof(void*)*batch)==TINYNV_OK && tinynv_memcpy_dtoh(s,B,(tinynv_devptr_t)Bd,sizeof(void*)*batch)==TINYNV_OK
      && tinynv_memcpy_dtoh(s,C,(tinynv_devptr_t)Cd,sizeof(void*)*batch)==TINYNV_OK && tinynv_stream_sync(s)==TINYNV_OK){
    rc=CUBLAS_STATUS_SUCCESS;
    for (int i=0;i<batch && rc==CUBLAS_STATUS_SUCCESS;i++) rc=launch_gemm((struct tinyblas_handle*)h,ta,tb,m,n,k,alpha,A[i],lda,B[i],ldb,beta,C[i],ldc,in,out);
  }
  free(A); free(B); free(C); return rc;
}
cublasStatus_t cublasSgemmBatched(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m,int n,int k, const float* alpha,
    const float* const A[],int lda, const float* const B[],int ldb, const float* beta, float* const C[],int ldc, int batch){
  return batched(h,ta,tb,m,n,k,*alpha,(const void* const*)A,lda,(const void* const*)B,ldb,*beta,(void* const*)C,ldc,batch,0,0);
}
cublasStatus_t cublasGemmBatchedEx(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m,int n,int k, const void* alpha,
    const void* const A[], cudaDataType At,int lda, const void* const B[], cudaDataType Bt,int ldb, const void* beta, void* const C[], cudaDataType Ct,int ldc,
    int batch, cublasComputeType_t ct, cublasGemmAlgo_t algo){
  (void)algo; int in, out; if (gemm_contract(At, Bt, Ct, ct, &in, &out)) return CUBLAS_STATUS_NOT_SUPPORTED;
  return batched(h,ta,tb,m,n,k,cublas_scalar(alpha,ct),A,lda,B,ldb,cublas_scalar(beta,ct),C,ldc,batch,in,out);
}
cublasStatus_t cublasStrsmBatched(cublasHandle_t h, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag,
    int m,int n, const float* alpha, const float* const A[],int lda, float* const B[],int ldb, int batch){
  // the case ggml issues (solve_tri.cu): X * A = alpha * B with A upper-triangular, in place in B. Others are not needed yet.
  if (side!=CUBLAS_SIDE_RIGHT || uplo!=CUBLAS_FILL_MODE_UPPER || trans!=CUBLAS_OP_N || diag!=CUBLAS_DIAG_NON_UNIT) return CUBLAS_STATUS_NOT_SUPPORTED;
  tinynv_kernel_t kern=tinycublas_kernel("tinyblas_trsm_rupn_f32"); if (!kern) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (batch<=0) return CUBLAS_STATUS_SUCCESS;
  tinynv_stream_t s=((struct tinyblas_handle*)h)->stream;
  void **Ah=malloc(sizeof(void*)*batch), **Bh=malloc(sizeof(void*)*batch); cublasStatus_t rc=CUBLAS_STATUS_EXECUTION_FAILED;
  if (tinynv_memcpy_dtoh(s,Ah,(tinynv_devptr_t)A,sizeof(void*)*batch)==TINYNV_OK && tinynv_memcpy_dtoh(s,Bh,(tinynv_devptr_t)B,sizeof(void*)*batch)==TINYNV_OK && tinynv_stream_sync(s)==TINYNV_OK){
    rc=CUBLAS_STATUS_SUCCESS;
    for (int i=0;i<batch && rc==CUBLAS_STATUS_SUCCESS;i++){
      struct __attribute__((packed)) { uint64_t A,X; int32_t k,n,lda,ldb; float alpha; } p={ (uint64_t)Ah[i],(uint64_t)Bh[i], m,n,lda,ldb,*alpha };
      unsigned bx=256, gx=(unsigned)((m+255)/256);
      if (tinycudart_trace()) fprintf(stderr,"[trace] launch tinyblas_trsm_rupn_f32 grid=(%u,1,1) block=(%u,1,1) k=%d n=%d batch %d/%d\n",gx,bx,m,n,i+1,batch);
      tinycudart_count_launch("tinyblas_trsm_rupn_f32");
      if (blas_launch(s,kern, "tinyblas_trsm_rupn_f32", gx,1,1, bx,1,1, &p, sizeof(p))!=TINYNV_OK) rc=CUBLAS_STATUS_EXECUTION_FAILED;
    }
  }
  free(Ah); free(Bh); return rc;
}
