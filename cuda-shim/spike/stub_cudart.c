// Logging stub of the cudart ABI the compiler-generated host code calls. Proves the registration+launch flow reaches us.
#include <stdio.h>
#include <stdint.h>
typedef int cudaError_t;
typedef unsigned long size_t_;
static int reg_count=0, fn_count=0, launch_count=0;
// fatbin registration
void** __cudaRegisterFatBinary(void* fatCubin){ printf("[tinycudart] __cudaRegisterFatBinary(%p)\n", fatCubin); reg_count++; return (void**)fatCubin; }
void   __cudaRegisterFatBinaryEnd(void** h){ printf("[tinycudart] __cudaRegisterFatBinaryEnd(%p)\n",(void*)h); }
void   __cudaUnregisterFatBinary(void** h){ printf("[tinycudart] __cudaUnregisterFatBinary(%p)\n",(void*)h); }
void   __cudaRegisterFunction(void**h,const char*hostfun,char*devfun,const char*name,int tl,void*tid,void*bd,void*dg,void*db,int*ws){
  printf("[tinycudart] __cudaRegisterFunction name=%s hostfun=%p\n", name, hostfun); fn_count++; }
void   __cudaRegisterVar(void**h,char*hv,char*dv,const char*name,int ext,size_t s,int c,int g){ printf("[tinycudart] __cudaRegisterVar %s\n",name); }
unsigned __cudaPopCallConfiguration(void*g,void*b,void*sh,void*st){ return 0; }
unsigned __cudaPushCallConfiguration(uint64_t gx,uint64_t gy,uint64_t bx,uint64_t by,unsigned sh,void*st){ return 0; }
cudaError_t cudaMalloc(void** p, unsigned long sz){ static char pool[1<<20]; static unsigned long off=0; *p=pool+off; off+=sz; printf("[tinycudart] cudaMalloc(%lu) -> %p\n", sz, *p); return 0; }
cudaError_t cudaLaunchKernel(const void*fn,uint64_t gx,uint64_t gy,uint64_t gz,uint64_t bx,uint64_t by,uint64_t bz,void**args,unsigned long sh,void*st){
  printf("[tinycudart] cudaLaunchKernel fn=%p grid=(%llu) block=(%llu) args=%p  <-- THIS is where libtinynv would submit a QMD\n",fn,(unsigned long long)gx,(unsigned long long)bx,(void*)args); launch_count++; return 0; }
cudaError_t cudaDeviceSynchronize(void){ printf("[tinycudart] cudaDeviceSynchronize (reg=%d fn=%d launch=%d)\n",reg_count,fn_count,launch_count); return 0; }
