// Logging stand-in for libtinycudart (old launch ABI). Proves the host runtime flow; cudaLaunch is where libtinynv submits a QMD.
#include <cstdio>
#include <cstdlib>
#include <cstddef>
extern "C" {
typedef int cudaError_t;
struct dim3 { unsigned x, y, z; };
static int nreg=0, nfn=0, nsetup=0;
void** __cudaRegisterFatBinary(void* p){ printf("[shim] __cudaRegisterFatBinary(%p)  <- embedded sm_120 fatbin\n", p); nreg++; return (void**)p; }
void __cudaRegisterFatBinaryEnd(void** h){ (void)h; }
void __cudaUnregisterFatBinary(void** h){ (void)h; printf("[shim] __cudaUnregisterFatBinary\n"); }
void __cudaRegisterFunction(void** h, const char* hostFun, char* devFun, const char* name, int tl, void* a, void* b, void* c, void* d, int* e){
  (void)h;(void)devFun;(void)tl;(void)a;(void)b;(void)c;(void)d;(void)e; printf("[shim] __cudaRegisterFunction name='%s' hoststub=%p\n", name, hostFun); nfn++; }
cudaError_t cudaConfigureCall(dim3 g, dim3 b, size_t sh, void* st){ (void)st; printf("[shim] cudaConfigureCall grid=(%u,%u,%u) block=(%u,%u,%u) shmem=%zu\n", g.x,g.y,g.z,b.x,b.y,b.z,sh); return 0; }
cudaError_t cudaSetupArgument(const void* arg, size_t size, size_t off){ (void)arg; printf("[shim]   cudaSetupArgument size=%zu offset=%zu\n", size, off); nsetup++; return 0; }
cudaError_t cudaLaunch(const void* func){ printf("[shim] cudaLaunch func=%p (%d args)  <== libtinynv builds the QMD + rings the doorbell here\n", func, nsetup); nsetup=0; return 0; }
cudaError_t cudaMalloc(void** p, size_t sz){ *p=malloc(sz); printf("[shim] cudaMalloc(%zu) -> %p\n", sz, *p); return 0; }
cudaError_t cudaDeviceSynchronize(){ printf("[shim] cudaDeviceSynchronize  (fatbins=%d funcs=%d)\n", nreg, nfn); return 0; }
}
