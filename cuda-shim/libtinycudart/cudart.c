// libtinycudart — the CUDA runtime ABI subset that clang-compiled ggml-cuda host code calls, backed by libtinynv (tinynv.h).
// Implements the classic launch ABI (cudaConfigureCall/SetupArgument/Launch) clang emits when it can't detect the CUDA version.
#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <stdint.h>
#include "tinynv.h"
#include "hostpool.h"
typedef int cudaError_t;
enum { cudaSuccess=0, cudaErrorInvalidValue=1, cudaErrorMemoryAllocation=2, cudaErrorInvalidDeviceFunction=98, cudaErrorNotSupported=801, cudaErrorUnknown=999 };
enum { cudaMemcpyHostToHost=0, cudaMemcpyHostToDevice=1, cudaMemcpyDeviceToHost=2, cudaMemcpyDeviceToDevice=3, cudaMemcpyDefault=4 };
typedef struct { unsigned x,y,z; } dim3;
typedef void* cudaStream_t;
extern int tinycudart_extract_cubin(const void*, const void**, size_t*);
extern int tinycudart_cubin_kernels(const void*, size_t, char***);
extern char* tinycudart_demangle(const char*); // demangle.cpp, LLVM's demangler
// Host stubs are mangled by clang on macOS, device code by nvcc on Linux, and they disagree in two places: clang marks a
// `static` kernel with the internal-linkage L (_ZL...), nvcc does not; and int64_t is `long long` (x) on macOS but `long`
// (l) on Linux, likewise uint64_t (y vs m). The parameters are the same size on both sides, so the launch layout is
// identical; only the names differ. Names are therefore matched by their DEMANGLED text with "long long" collapsed to
// "long", which covers both differences at once without guessing at positions in the mangled string.
static char* canon(const char* mangled){
  char* d=tinycudart_demangle(mangled); if (!d) return NULL;
  // nvcc wraps a TU's static device functions in a namespace named _INTERNAL_<hash>_<file>_<hash>_<n>; clang does not.
  // A function-pointer template argument therefore reads `&_INTERNAL_...::f(...)` on one side and `&f(...)` on the other.
  char* out=malloc(strlen(d)+1); char* o=out; const char* p=d;
  while (*p){
    if (!strncmp(p,"_INTERNAL_",10)){ const char* q=p; while (*q && (isalnum((unsigned char)*q)||*q=='_')) q++; if (q[0]==':'&&q[1]==':'){ p=q+2; continue; } }
    if (!strncmp(p,"long long",9)){ memcpy(o,"long",4); o+=4; p+=9; continue; } // int64_t is long long on macOS, long on Linux
    if (isdigit((unsigned char)*p)){ // an integer literal (a template value argument): 128ll on macOS is 128l on Linux, 3ull is 3ul
      while (isdigit((unsigned char)*p)) *o++=*p++;
      if (p[0]=='u') *o++=*p++;
      if (p[0]=='l'&&p[1]=='l'&&!isalnum((unsigned char)p[2])&&p[2]!='_'){ *o++='l'; p+=2; }
      continue;
    }
    *o++=*p++;
  }
  *o=0; free(d); return out;
}

static tinynv_device_t g_dev; static tinynv_stream_t g_default_stream; static int g_init;
static int stats_on(void); static void stats_dump(void);
// host-stub address -> kernel. ggml registers ~8000 kernels and launches thousands per token, so this is an open-addressing
// hash on the stub address rather than a linear table; it grows by doubling.
typedef struct { const void* hostfun; tinynv_kernel_t k; const char* name; } func_t;
static func_t* g_funcs; static size_t g_fcap, g_nfuncs;
static size_t fslot(const void* p, size_t cap){ uintptr_t x=(uintptr_t)p; x^=x>>17; x*=0x9E3779B97F4A7C15ull; x^=x>>29; return (size_t)x & (cap-1); }
static func_t* flookup(const void* p){ if (!g_fcap) return NULL; for (size_t i=fslot(p,g_fcap);;i=(i+1)&(g_fcap-1)){ if (!g_funcs[i].hostfun) return NULL; if (g_funcs[i].hostfun==p) return &g_funcs[i]; } }
static void finsert(const void* p, tinynv_kernel_t k, const char* name){
  if ((g_nfuncs+1)*2 > g_fcap){ size_t ncap=g_fcap?g_fcap*2:16384; func_t* nf=calloc(ncap,sizeof(func_t));
    for (size_t i=0;i<g_fcap;i++) if (g_funcs[i].hostfun){ size_t j=fslot(g_funcs[i].hostfun,ncap); while (nf[j].hostfun) j=(j+1)&(ncap-1); nf[j]=g_funcs[i]; }
    free(g_funcs); g_funcs=nf; g_fcap=ncap; }
  func_t* f=flookup(p); if (f){ f->k=k; return; }
  size_t i=fslot(p,g_fcap); while (g_funcs[i].hostfun) i=(i+1)&(g_fcap-1); g_funcs[i].hostfun=p; g_funcs[i].k=k; g_funcs[i].name=name; g_nfuncs++;
}
// classic-ABI launch state (globals for the scaffold; real impl = thread-local)
static dim3 g_grid, g_block; static size_t g_shmem; static tinynv_stream_t g_stream;
static unsigned char g_params[4096]; static size_t g_params_len;
static cudaError_t g_lasterr = cudaSuccess;

// A failed driver init (a named socket that cannot be opened, a dead server) means NO device, loudly: ggml then reports
// zero CUDA devices and llama.cpp falls back to the CPU. Carrying on with no device would compute on garbage silently.
static int g_nodev;
static void ensure_init(void){
  if (g_init) return;
  g_init=1;
  if (tinynv_init()!=TINYNV_OK || tinynv_device_get(&g_dev,0)!=TINYNV_OK || tinynv_stream_create(g_dev,&g_default_stream)!=TINYNV_OK){
    fprintf(stderr,"[tinycudart] driver init failed: %s -- reporting no CUDA devices\n", tinynv_last_error()); g_nodev=1; g_dev=NULL; g_default_stream=NULL; return; }
  tinycudart_hostpool_init(g_dev);
  // Lend the driver a copy kernel for downloads (libtinycudart/copy1d.cu): a download served by a kernel on the compute
  // engine has no copy-engine handoff at the token boundary. Registration alone changes nothing; the driver takes the
  // path only under TINYNV_DOWNLOAD_VIA_COMPUTE=1 and names it at start-up. A refusal is loud and leaves the copy engine.
  { extern unsigned char copy1d_cubin[]; extern unsigned int copy1d_cubin_len; tinynv_module_t mod=NULL; tinynv_kernel_t k=NULL; tinynv_status_t r;
    if ((r=tinynv_module_load(g_dev,copy1d_cubin,copy1d_cubin_len,&mod))!=TINYNV_OK || (r=tinynv_get_kernel(mod,"tinycudart_copy1d",&k))!=TINYNV_OK ||
        (r=tinynv_set_download_kernel(g_dev,k,256,16))!=TINYNV_OK)
      fprintf(stderr,"[tinycudart] the download kernel was not registered (%s): downloads stay on the copy engine\n", tinynv_last_error()); }
  // And the keep-alive kernel (libtinycudart/keepalive.cu): one warp polling a host flag, launched by the driver after a
  // synchronisation under TINYNV_KEEPALIVE=1 so the compute engine stays scheduled across a token boundary.
  { extern unsigned char keepalive_cubin[]; extern unsigned int keepalive_cubin_len; tinynv_module_t mod=NULL; tinynv_kernel_t k=NULL; tinynv_status_t r;
    if ((r=tinynv_module_load(g_dev,keepalive_cubin,keepalive_cubin_len,&mod))!=TINYNV_OK || (r=tinynv_get_kernel(mod,"tinynv_keepalive",&k))!=TINYNV_OK ||
        (r=tinynv_set_keepalive_kernel(g_dev,k))!=TINYNV_OK)
      fprintf(stderr,"[tinycudart] the keep-alive kernel was not registered (%s): TINYNV_KEEPALIVE would do nothing\n", tinynv_last_error()); }
  fprintf(stderr,"[tinycudart] libtinynv build %s\n", tinynv_build_id());  // a stale binary announces itself (B's 4892427)
  atexit(stats_dump);   // always: the flushed-sync count below prints whenever it is non-zero; the rest only with TINYCUDART_STATS=1
}
// accessors for the sibling libraries (libtinycublas)
tinynv_device_t tinycudart_device(void){ ensure_init(); return g_dev; }
tinynv_stream_t tinycudart_default_stream(void){ ensure_init(); return g_default_stream; }
tinynv_kernel_t tinycudart_kernel_for(const void* hostfun){ func_t* f=flookup(hostfun); return f?f->k:NULL; }
const char* tinycudart_kernel_name(const void* hostfun){ func_t* f=flookup(hostfun); return f?f->name:"(never registered)"; }
int  tinycudart_peek_error(void){ return g_lasterr; }
void tinycudart_set_error(int e){ g_lasterr=e; }

// Each fatbin gets its own handle (the ABI treats the returned void** as opaque), so functions register against the
// module they came from and the module is unloaded exactly once.
typedef struct { tinynv_module_t mod; int kernels, resolved, renamed, matched; int nnames; char **names, **canon; } fatbin_handle_t;
static int g_modules, g_kernels_total, g_resolved_total, g_renamed_total, g_matched_total;
static fatbin_handle_t** g_handles; static int g_nhandles, g_hcap;
// every driver failure is reported with the driver's own words, since ggml only ever sees "CUDA error"
static cudaError_t drv(const char* what, tinynv_status_t st){ if (st!=TINYNV_OK) fprintf(stderr,"[tinycudart] %s failed: %s (%s)\n", what, tinynv_status_str(st), tinynv_last_error()); return st==TINYNV_OK?cudaSuccess:cudaErrorUnknown; }
// TINYCUDART_TRACE=1: one line per launch / copy / fill / sync, with the kernel name and sizes -- the CUDA-API-level view of a run,
// which is where a wrong result is first localised to a kernel. TINYCUDART_STATS=1: at exit, launches per kernel and bytes moved.
static int g_trace=-1, g_stats=-1;
int tinycudart_trace(void){ if (g_trace<0){ const char* v=getenv("TINYCUDART_TRACE"); g_trace=(v&&*v=='1'); } return g_trace; }
static int stats_on(void){ if (g_stats<0){ const char* v=getenv("TINYCUDART_STATS"); g_stats=(v&&*v=='1'); } return g_stats; }
typedef struct { const char* name; unsigned long launches; double ns; } kstat_t;
static unsigned long g_syncs;   // stream/device/event waits and synchronous copies: each one breaks the launch chain
static kstat_t* g_kstats; static size_t g_nkstats, g_kcap; static unsigned long g_launches, g_copies; static unsigned long long g_bytes_h2d, g_bytes_d2h, g_bytes_d2d, g_bytes_set;
// The caller's own time between one launch returning and the next arriving (ggml-cuda's dispatch and llama.cpp above it),
// against the time inside tinynv_launch. Paired only when the previous CUDA call was also a launch, so a wait or a copy
// between them is not counted as the caller thinking. The stats' own name loops run outside both brackets.
double tinycudart_now_ns(void);
static double g_last_ret, g_between_ns; static int g_last_was_launch; static unsigned long g_between_n;
// The host's turnaround after a wait: from a synchronisation returning (the logits read back, on a decode) to the next
// launch, whatever copies happen between - sampling, the next graph's build and scheduling, the input uploads. This is
// the host's share of the token boundary the kernel profile measures on the engine's clock.
static double g_sync_ret, g_turn_ns, g_turn_max; static int g_after_sync; static unsigned long g_turn_n;
void tinycudart_count_launch(const char* name){ g_launches++; if (!stats_on()) return;
  { double now=tinycudart_now_ns(); if (g_last_was_launch && g_last_ret>0) { g_between_ns+=now-g_last_ret; g_between_n++; }
    // the first four are the load, the prompt and the warm-up, whose turnarounds are not a token's
    if (g_after_sync) { static unsigned long seen; double d=now-g_sync_ret; g_after_sync=0;
      if (seen++ >= 4) { g_turn_ns+=d; g_turn_n++; if (d>g_turn_max) g_turn_max=d; } } }
  for (size_t i=0;i<g_nkstats;i++) if (g_kstats[i].name==name){ g_kstats[i].launches++; return; }
  if (g_nkstats==g_kcap){ g_kcap=g_kcap?g_kcap*2:256; g_kstats=realloc(g_kstats,g_kcap*sizeof(*g_kstats)); }
  g_kstats[g_nkstats].name=name; g_kstats[g_nkstats].launches=1; g_nkstats++; }
// Wall time around tinynv_launch, per kernel. Under TINYNV_SYNC=1 TINYNV_CHAIN_DEPTH=1 every launch is waited on before
// the call returns, so this is the kernel's execution time plus a fixed round trip; in the default async mode it is only the
// submission cost. The dump says which mode it was.
void tinycudart_time_launch(const char* name, double ns){ if (!stats_on()) return;
  for (size_t i=0;i<g_nkstats;i++) if (g_kstats[i].name==name){ g_kstats[i].ns+=ns; break; }
  g_last_ret=tinycudart_now_ns(); g_last_was_launch=1; }
void tinycudart_count_sync(void){ g_syncs++; g_last_was_launch=0; }
static unsigned long g_syncs_flushed;   // answered without waiting: the driver said nothing outstanding needed it (inline uploads only)
void tinycudart_count_sync_flushed(void){ g_syncs_flushed++; }
static double g_api_ns[8]; static unsigned long g_api_n[8];
void tinycudart_time_api(int kind, double ns){ g_last_was_launch=0; if (!stats_on()||kind<0||kind>7) return; g_api_ns[kind]+=ns; g_api_n[kind]++;
  // the FIRST synchronisation after the last launch (the logits read back, on a decode) starts the turnaround; the input
  // copies that follow before the next launch are part of it, not a new start
  if ((kind==1||kind==3||kind==6) && !g_after_sync) { g_sync_ret=tinycudart_now_ns(); g_after_sync=1; } }
double tinycudart_now_ns(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec*1e9+ts.tv_nsec; }
void tinycudart_count_copy(int kind, unsigned long long n){ g_copies++; g_last_was_launch=0; if (kind==1) g_bytes_h2d+=n; else if (kind==2) g_bytes_d2h+=n; else if (kind==3) g_bytes_d2d+=n; else g_bytes_set+=n; }
static int cmp_kstat(const void* a, const void* b){ double x=((const kstat_t*)a)->ns, y=((const kstat_t*)b)->ns; if (x!=y) return x<y?1:-1;
  unsigned long lx=((const kstat_t*)a)->launches, ly=((const kstat_t*)b)->launches; return lx<ly?1:lx>ly?-1:0; }
static void stats_dump(void){
  { extern void tinycudart_graph_stats(void); tinycudart_graph_stats(); }
  if (g_syncs_flushed) fprintf(stderr,"[tinycudart] %lu synchronizations were answered without a wait - the driver said nothing outstanding needed one (inline uploads only), so they flushed and returned\n", g_syncs_flushed);
  if (!stats_on()) return;
  fprintf(stderr,"[tinycudart] stats: %lu launches, %lu copies/fills: h2d %.1f MB, d2h %.1f MB, d2d %.1f MB, memset %.1f MB\n", g_launches, g_copies,
          g_bytes_h2d/1048576.0, g_bytes_d2h/1048576.0, g_bytes_d2d/1048576.0, g_bytes_set/1048576.0);
  const char* sync=getenv("TINYNV_SYNC"); const char* depth=getenv("TINYNV_CHAIN_DEPTH");
  double total=0; for (size_t i=0;i<g_nkstats;i++) total+=g_kstats[i].ns;
  fprintf(stderr,"[tinycudart] stats: %lu synchronizations (waits + synchronous copies); launch wall time %.1f ms total, %s\n", g_syncs, total/1e6,
          (sync&&*sync=='1'&&depth&&*depth=='1') ? "sync + depth 1: this is execution time per kernel plus one round trip each"
                                                  : "async: this is submission cost only, NOT execution time");
  { static const char* kinds[8]={"memcpyAsync","memcpy","memsetAsync","streamSynchronize","eventRecord","streamWaitEvent","eventSynchronize","?"};
    for (int i=0;i<8;i++) if (g_api_n[i]) fprintf(stderr,"[tinycudart]   api %-18s %8lu calls  %9.1f ms  %7.1f us/call\n", kinds[i], g_api_n[i], g_api_ns[i]/1e6, g_api_ns[i]/1e3/g_api_n[i]); }
  if (g_between_n) fprintf(stderr,"[tinycudart] stats: between consecutive launches the caller spent %.2f us a launch (%lu pairs: ggml-cuda's dispatch and "
                           "llama.cpp above it); inside tinynv_launch %.2f us a launch - the host's cost per launch is their sum\n",
                           g_between_ns/1e3/g_between_n, g_between_n, g_launches?total/1e3/g_launches:0.0);
  if (g_turn_n) fprintf(stderr,"[tinycudart] stats: after a synchronisation returned, the host took %.1f us to issue the next launch (%lu times after the first four, worst %.1f us): "
                       "sampling, the next graph's build and scheduling, input uploads - the host's share of the token boundary\n",
                       g_turn_ns/1e3/g_turn_n, g_turn_n, g_turn_max/1e3);
  qsort(g_kstats,g_nkstats,sizeof(*g_kstats),cmp_kstat);
  for (size_t i=0;i<g_nkstats && i<40;i++) fprintf(stderr,"[tinycudart]   %8lu  %9.1f ms  %7.1f us/launch  %s\n", g_kstats[i].launches, g_kstats[i].ns/1e6,
          g_kstats[i].launches ? g_kstats[i].ns/1e3/g_kstats[i].launches : 0.0, g_kstats[i].name);
  if (g_nkstats>40) fprintf(stderr,"[tinycudart]   ... and %zu more kernels\n", g_nkstats-40);
}
static int quiet(void){ static int q=-1; if (q<0){ const char* v=getenv("TINYCUDART_VERBOSE"); q=!(v&&*v=='1'); } return q; }

void** __cudaRegisterFatBinary(void* fatCubin){
  ensure_init();
  fatbin_handle_t* h=calloc(1,sizeof(*h));
  const void* cubin; size_t len;
  if (g_nodev) return (void**)h;
  if (tinycudart_extract_cubin(fatCubin,&cubin,&len)!=0){ fprintf(stderr,"[tinycudart] no cubin in fatbin\n"); return (void**)h; }
  if (tinynv_module_load(g_dev,cubin,len,&h->mod)!=TINYNV_OK){ fprintf(stderr,"[tinycudart] module_load failed (%zu byte cubin)\n",len); h->mod=NULL; }
  else { g_modules++; if (!quiet()) fprintf(stderr,"[tinycudart] loaded module: cubin %zu bytes\n", len); }
  h->nnames=tinycudart_cubin_kernels(cubin,len,&h->names); if (h->nnames<0) h->nnames=0;
  h->canon=calloc((size_t)h->nnames,sizeof(char*)); for (int i=0;i<h->nnames;i++) h->canon[i]=canon(h->names[i]);
  if (g_nhandles==g_hcap){ g_hcap=g_hcap?g_hcap*2:256; g_handles=realloc(g_handles,(size_t)g_hcap*sizeof(*g_handles)); }
  g_handles[g_nhandles++]=h;
  return (void**)h;
}
void __cudaRegisterFatBinaryEnd(void** hh){
  fatbin_handle_t* h=(fatbin_handle_t*)hh; if (!h) return;
  // (totals are accumulated per registration: clang only calls this when it detected a CUDA >= 10.1 SDK, which it did not)
  if (!quiet() || h->resolved<h->kernels)
    fprintf(stderr,"[tinycudart] module %d: %d kernels registered, %d resolved (%d via internal-linkage rename)%s\n",
            g_modules, h->kernels, h->resolved, h->renamed, h->resolved<h->kernels ? " -- unresolved ones fail at launch, not here" : "");
}
void __cudaUnregisterFatBinary(void** hh){ fatbin_handle_t* h=(fatbin_handle_t*)hh; if (!h) return; if (h->mod) tinynv_module_unload(h->mod); h->mod=NULL; free(h); }
void __cudaRegisterFunction(void** hh,const char* hostFun,char* devFun,const char* name,int tl,void* a,void* b,void* c,void* d,int* e){
  (void)devFun;(void)tl;(void)a;(void)b;(void)c;(void)d;(void)e;
  fatbin_handle_t* h=(fatbin_handle_t*)hh; tinynv_kernel_t k=NULL; int renamed=0;
  if (h && h->mod){
    if (tinynv_get_kernel(h->mod,name,&k)!=TINYNV_OK){
      // clang mangles a `static __global__` kernel with the Itanium internal-linkage marker (_ZL...); nvcc's device
      // compiler gives the same kernel external linkage in the cubin and mangles it without the marker (_Z...).
      // ggml declares nearly every kernel static, so this is the common case, not the exception.
      if (name[0]=='_'&&name[1]=='Z'&&name[2]=='L'){ char alt[1024]; snprintf(alt,sizeof alt,"_Z%s",name+3); if (tinynv_get_kernel(h->mod,alt,&k)==TINYNV_OK) renamed=1; else k=NULL; }
      else k=NULL;
      if (!k){ // the platform seam: compare demangled text with 64-bit spellings collapsed, then look up nvcc's exact name
        char* c=canon(name);
        if (c){ for (int i=0;i<h->nnames;i++) if (h->canon[i] && !strcmp(h->canon[i],c)){ if (tinynv_get_kernel(h->mod,h->names[i],&k)==TINYNV_OK) renamed=2; else k=NULL; break; } free(c); }
      }
    }
  }
  if (h){ h->kernels++; if (k){ h->resolved++; h->renamed+=(renamed==1); h->matched+=(renamed==2); } }
  g_kernels_total++; if (k){ g_resolved_total++; g_renamed_total+=(renamed==1); g_matched_total+=(renamed==2); }
  if (!k && !quiet()) fprintf(stderr,"[tinycudart] kernel '%s' not in its module (fails only if launched)\n",name);
  finsert(hostFun,k,strdup(name));
}
void __cudaRegisterVar(void**h,char*hv,char*dv,const char*n,int ext,size_t s,int c,int g){(void)h;(void)hv;(void)dv;(void)n;(void)ext;(void)s;(void)c;(void)g;}

cudaError_t cudaConfigureCall(dim3 grid, dim3 block, size_t sh, cudaStream_t stream){
  g_grid=grid; g_block=block; g_shmem=sh; g_stream=((uintptr_t)stream>2)?(tinynv_stream_t)stream:g_default_stream; /* NULL, cudaStreamLegacy (1), cudaStreamPerThread (2) -> default */ g_params_len=0;
  memset(g_params,0,sizeof(g_params)); return cudaSuccess;
}
cudaError_t cudaSetupArgument(const void* arg, size_t size, size_t offset){
  if (offset+size<=sizeof(g_params)){ memcpy(g_params+offset,arg,size); if (offset+size>g_params_len) g_params_len=offset+size; }
  return cudaSuccess;
}
cudaError_t cudaLaunch(const void* func){
  func_t* f=flookup(func); tinynv_kernel_t k=f?f->k:NULL;
  if (!k){ const char* nm=f?f->name:"(never registered)";
    fprintf(stderr,"[tinycudart] cudaLaunch: kernel '%s' (%p) was registered but not found in its module\n",nm,func); return g_lasterr=cudaErrorUnknown; }
  if (tinycudart_trace()) fprintf(stderr,"[trace] launch %s grid=(%u,%u,%u) block=(%u,%u,%u) smem=%zu params=%zu\n", f->name, g_grid.x,g_grid.y,g_grid.z, g_block.x,g_block.y,g_block.z, g_shmem, g_params_len);
  { extern int tinycudart_capturing_nv(tinynv_stream_t); extern cudaError_t tinycudart_capture_launch(tinynv_kernel_t, const char*, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, const void*, size_t);
    if (tinycudart_capturing_nv(g_stream)) return g_lasterr=tinycudart_capture_launch(k, f->name, g_grid.x,g_grid.y,g_grid.z, g_block.x,g_block.y,g_block.z, (unsigned)g_shmem, g_params, g_params_len);
    extern void tinycudart_capture_offstream(const char *, tinynv_stream_t); tinycudart_capture_offstream("a classic-ABI launch on another stream", g_stream); }
  tinycudart_count_launch(f->name);
  double t0=stats_on()?tinycudart_now_ns():0;
  tinynv_status_t s=tinynv_launch(g_stream,k, g_grid.x,g_grid.y,g_grid.z, g_block.x,g_block.y,g_block.z,
                                  (unsigned)g_shmem, g_params, g_params_len);
  if (stats_on()) tinycudart_time_launch(f->name, tinycudart_now_ns()-t0);
  if (s!=TINYNV_OK){ char what[1200]; snprintf(what,sizeof what,"cudaLaunch(%s grid=%u,%u,%u block=%u,%u,%u smem=%zu params=%zu)",f->name,g_grid.x,g_grid.y,g_grid.z,g_block.x,g_block.y,g_block.z,g_shmem,g_params_len); drv(what,s); }
  return g_lasterr = (s==TINYNV_OK? cudaSuccess : cudaErrorUnknown);
}
cudaError_t cudaMalloc(void** p, size_t sz){ { extern void tinycudart_capture_note(const char*); tinycudart_capture_note("cudaMalloc"); } ensure_init(); tinynv_devptr_t d=0; tinynv_status_t s=tinynv_malloc(g_dev,sz,&d); *p=(void*)d;
  if (tinycudart_trace()) fprintf(stderr,"[trace] malloc %zu -> %#llx\n", sz, (unsigned long long)d);
  return g_lasterr=drv("cudaMalloc",s); }
cudaError_t cudaFree(void* p){ { extern void tinycudart_capture_note(const char*); tinycudart_capture_note("cudaFree"); } if (tinycudart_trace()) fprintf(stderr,"[trace] free %#llx\n",(unsigned long long)(tinynv_devptr_t)p);
  return g_lasterr=(tinynv_free(g_dev,(tinynv_devptr_t)p)==TINYNV_OK?cudaSuccess:cudaErrorUnknown); }
cudaError_t cudaMemcpy(void* dst,const void* src,size_t n,int kind){ { extern void tinycudart_capture_note(const char*); tinycudart_capture_note("cudaMemcpy (synchronous)"); } tinycudart_count_sync();
  if (tinycudart_trace()) fprintf(stderr,"[trace] memcpy kind=%d %zu bytes (sync)\n", kind, n);
  tinycudart_count_copy(kind, n);
  tinynv_status_t s=TINYNV_ERR_INVALID;
  if (kind==cudaMemcpyHostToDevice) s=tinynv_memcpy_htod(g_default_stream,(tinynv_devptr_t)dst,src,n);
  else if (kind==cudaMemcpyDeviceToHost) s=tinynv_memcpy_dtoh(g_default_stream,dst,(tinynv_devptr_t)src,n);
  else if (kind==cudaMemcpyDeviceToDevice) s=tinynv_memcpy_dtod(g_default_stream,(tinynv_devptr_t)dst,(tinynv_devptr_t)src,n);
  else { memcpy(dst,src,n); s=TINYNV_OK; }
  if (s==TINYNV_OK) tinynv_stream_sync(g_default_stream);
  return g_lasterr=(s==TINYNV_OK?cudaSuccess:cudaErrorUnknown);
}
cudaError_t cudaDeviceSynchronize(void){ { extern void tinycudart_capture_note(const char*); tinycudart_capture_note("cudaDeviceSynchronize"); } tinycudart_count_sync(); return g_lasterr=(tinynv_stream_sync(g_default_stream)==TINYNV_OK?cudaSuccess:cudaErrorUnknown); }
cudaError_t cudaGetLastError(void){ cudaError_t e=g_lasterr; g_lasterr=cudaSuccess; return e; }
const char* cudaGetErrorString(cudaError_t e){
  switch(e){ case cudaSuccess: return "no error"; case cudaErrorInvalidValue: return "invalid argument";
    case cudaErrorMemoryAllocation: return "out of memory"; case cudaErrorNotSupported: return "operation not supported";
    case cudaErrorInvalidDeviceFunction: return "invalid device function (kernel not registered or not found in its module)"; default: return "tinycudart error"; }
}
cudaError_t cudaSetDevice(int d){(void)d;return cudaSuccess;} cudaError_t cudaGetDevice(int* d){*d=0;return cudaSuccess;}
cudaError_t cudaGetDeviceCount(int* n){
  ensure_init(); *n=g_nodev?0:tinynv_device_count();
  static int summarized; // registration runs before main; this is the first runtime call ggml makes afterwards
  if (!summarized++) fprintf(stderr,"[tinycudart] %d modules loaded, %d kernels registered, %d resolved (%d exact, %d by internal-linkage rename, %d by demangled match), %d absent from their module\n",
                             g_modules, g_kernels_total, g_resolved_total, g_resolved_total-g_renamed_total-g_matched_total, g_renamed_total, g_matched_total, g_kernels_total-g_resolved_total);
  return cudaSuccess;
}

// --- pinned host memory: served from the pool (hostpool.c) so driver mappings stay bounded; see hostpool.c for why ---
cudaError_t cudaMallocHost(void** p, size_t n){ ensure_init(); if(!p) return g_lasterr=cudaErrorInvalidValue;
  return g_lasterr=(tinycudart_host_alloc(n,p)==0?cudaSuccess:cudaErrorMemoryAllocation); }   // ggml falls back to a CPU buffer on failure
cudaError_t cudaHostAlloc(void** p, size_t n, unsigned flags){ (void)flags; return cudaMallocHost(p,n); } // Mapped is accepted; GetDevicePointer says no
cudaError_t cudaFreeHost(void* p){ ensure_init(); return g_lasterr=(tinycudart_host_free(p)==0?cudaSuccess:cudaErrorInvalidValue); }
cudaError_t cudaHostRegister(void* p, size_t n, unsigned flags){ (void)p;(void)n;(void)flags; return g_lasterr=cudaErrorNotSupported; } // ggml falls back to unpinned
cudaError_t cudaHostUnregister(void* p){ (void)p; return cudaSuccess; }
cudaError_t cudaHostGetDevicePointer(void** d, void* h, unsigned flags){ (void)d;(void)h;(void)flags; return g_lasterr=cudaErrorNotSupported; }
