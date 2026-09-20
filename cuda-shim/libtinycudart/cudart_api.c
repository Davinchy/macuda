// libtinycudart, part 2: the runtime entry points ggml-cuda links against beyond the classic launch ABI. This file is
// compiled against the REAL cuda_runtime_api.h so every signature and struct layout (cudaDeviceProp, cudaLaunchConfig_t,
// cudaFuncAttributes) is checked by the compiler rather than transcribed. Backed by tinynv.h; the null device makes all
// of it runnable without a GPU. What is deliberately unsupported returns cudaErrorNotSupported, which ggml treats as
// advisory (graphs, peer access, managed memory, cooperative launch): each is either off in our build config or
// guarded by a capability query we answer "no" to.
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime_api.h>
#include "tinynv.h"
#include "hostpool.h"
// the header may rename these to versioned/per-thread variants; ggml's objects reference the plain names
#undef cudaGetDeviceProperties
#undef cudaLaunchKernelExC
#undef cudaMemcpyAsync
#undef cudaMemcpy2DAsync
#undef cudaMemsetAsync
#undef cudaStreamSynchronize
#undef cudaStreamWaitEvent
#undef cudaEventRecord
#undef cudaMemcpyPeerAsync
#undef cudaStreamBeginCapture

extern tinynv_device_t tinycudart_device(void);
extern tinynv_stream_t tinycudart_default_stream(void);
extern tinynv_kernel_t tinycudart_kernel_for(const void *hostfun);
extern const char *tinycudart_kernel_name(const void *hostfun);
extern int tinycudart_peek_error(void);
extern void tinycudart_set_error(int e);
extern int tinycudart_trace(void);
extern void tinycudart_count_launch(const char *name);
extern void tinycudart_count_sync(void);
extern void tinycudart_time_launch(const char *name, double ns);
extern void tinycudart_time_api(int kind, double ns);
extern double tinycudart_now_ns(void);
extern void tinycudart_count_copy(int kind, unsigned long long n);
extern void tinycudart_count_sync_flushed(void);   // a synchronise the driver said needed no wait: flushed and answered
// the graph subset (graph.c): is this driver stream being captured, and the recorders the paths below hand off to
extern int tinycudart_capturing_nv(tinynv_stream_t s);
extern cudaError_t tinycudart_capture_launch(tinynv_kernel_t k, const char *name, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz, unsigned smem, const void *params, size_t plen);
extern cudaError_t tinycudart_capture_memcpy(void *dst, const void *src, size_t n, int kind);
extern cudaError_t tinycudart_capture_memset(void *p, int v, size_t n);
extern cudaError_t tinycudart_capture_refuse(void);
extern void tinycudart_capture_note(const char *what);   // something a capture could not record happened anyway
extern cudaError_t tinycudart_capture_memcpy2d(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, int kind);
extern void tinycudart_capture_offstream(const char *what, tinynv_stream_t s);
#define TRACE(...) do { if (tinycudart_trace()) { fprintf(stderr, "[trace] %.1f ", tinycudart_now_ns() / 1e3); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } while (0)

// functions, not macros: the argument carries the driver call, and it must run exactly once
static cudaError_t S(tinynv_status_t r) {
  cudaError_t e = r == TINYNV_OK ? cudaSuccess : r == TINYNV_ERR_OOM ? cudaErrorMemoryAllocation : r == TINYNV_ERR_INVALID ? cudaErrorInvalidValue : cudaErrorUnknown;
  if (r != TINYNV_OK) fprintf(stderr, "[tinycudart] driver call failed: %s (%s)\n", tinynv_status_str(r), tinynv_last_error()); // ggml only sees "CUDA error"
  tinycudart_set_error(e); return e;
}
static cudaError_t E(cudaError_t e) { tinycudart_set_error(e); return e; }
// NULL, cudaStreamLegacy (1) and cudaStreamPerThread (2) are magic values, not driver handles: all of them are the default stream here
static tinynv_stream_t st(cudaStream_t s) { return (uintptr_t)s > 2 ? (tinynv_stream_t)s : tinycudart_default_stream(); }
// The driver marks a device that cannot run anything with cc_major 0 (its null device, until the GSP boots and reports
// the real numbers). ggml decides its whole dispatch from these — tensor-core paths, MMQ, flash attention, how much to
// offload — so for host-side testing the null device is presented with the target's profile, the RTX 5090 (GB202,
// sm_120, 170 SMs, 32 GB, 228 KB shared per SM), and says so once. TINYCUDART_NULL_PROFILE=0 turns that off.
// The dynamic shared memory a block may opt in to, per architecture, as real CUDA reports it. Consumer Blackwell (sm_120,
// this card) has 128 KB of L1/shared per SM with 100 KB configurable and 99 KB available to one block, like Ada; the
// 227 KB figure is Hopper/datacenter. Session B measured it on the 5090: 100,352 bytes schedules and runs, 131,200 is
// accepted by the descriptor and NEVER SCHEDULED, which looks like a hang, not a limit. So the shim must not advertise it.
static int optin_smem_for(int cc_major, int cc_minor) {
  if (cc_major == 9) return 232448;                       // Hopper: 227 KB
  if (cc_major == 8 && cc_minor == 0) return 166912;      // A100: 163 KB
  if (cc_major == 12 || cc_major == 8 || cc_major == 7) return 101376;  // consumer Ampere/Ada/Blackwell, Turing: 99 KB
  return 48 * 1024;
}
static tinynv_device_props_t props(void) {
  tinynv_device_props_t p; memset(&p, 0, sizeof p); tinynv_device_props(tinycudart_device(), &p);
  static int announced;
  const char *env = getenv("TINYCUDART_NULL_PROFILE");
  if (p.cc_major == 0 && !(env && *env == '0')) {
    if (!announced++) fprintf(stderr, "[tinycudart] %s: presenting the RTX 5090 profile (sm_120, 170 SMs, 32 GB) so ggml dispatches as it will on the target\n", p.name);
    p.cc_major = 12; p.cc_minor = 0; p.sm_count = 170; p.total_mem = 32607ull << 20; p.warp_size = 32;
    p.max_threads_per_block = 1024; p.max_shared_per_block = 48 * 1024; p.max_shared_per_sm = 100 * 1024;  // consumer Blackwell, measured
  }
  return p;
}

// --- streams and events (in-order queues and timeline values in the driver) ---
cudaError_t cudaStreamCreateWithFlags(cudaStream_t *s, unsigned flags) { (void)flags; tinynv_stream_t t = NULL; tinynv_status_t r = tinynv_stream_create(tinycudart_device(), &t); *s = (cudaStream_t)t; return S(r); }
cudaError_t cudaStreamDestroy(cudaStream_t s) { return (uintptr_t)s > 2 ? S(tinynv_stream_destroy((tinynv_stream_t)s)) : E(cudaSuccess); }
cudaError_t cudaStreamSynchronize(cudaStream_t s) { if (tinycudart_capturing_nv(st(s))) return tinycudart_capture_refuse(); double _t0=tinycudart_now_ns(); cudaError_t _r; { tinycudart_count_sync(); TRACE("streamSynchronize %p", (void *)s); if (!tinynv_stream_needs_wait(st(s))) { tinycudart_count_sync_flushed(); _r=S(tinynv_stream_flush(st(s))); } else _r=S(tinynv_stream_sync(st(s))); } tinycudart_time_api(3, tinycudart_now_ns()-_t0); return _r; }
cudaError_t cudaStreamWaitEvent(cudaStream_t s, cudaEvent_t e, unsigned flags) { if (tinycudart_capturing_nv(st(s))) return E(cudaSuccess); double _t0=tinycudart_now_ns(); cudaError_t _r; { (void)flags; _r=S(tinynv_stream_wait_event(st(s), (tinynv_event_t)e)); } tinycudart_time_api(5, tinycudart_now_ns()-_t0); return _r; }
cudaError_t cudaEventCreateWithFlags(cudaEvent_t *e, unsigned flags) { (void)flags; tinynv_event_t t = NULL; tinynv_status_t r = tinynv_event_create(tinycudart_device(), &t); *e = (cudaEvent_t)t; return S(r); }
cudaError_t cudaEventDestroy(cudaEvent_t e) { (void)e; return E(cudaSuccess); } // the contract has no event destroy; a timeline value is a few bytes, kept for the process lifetime
cudaError_t cudaEventRecord(cudaEvent_t e, cudaStream_t s) { if (tinycudart_capturing_nv(st(s))) return E(cudaSuccess); double _t0=tinycudart_now_ns(); cudaError_t _r; { _r=S(tinynv_event_record((tinynv_event_t)e, st(s))); } tinycudart_time_api(4, tinycudart_now_ns()-_t0); return _r; }
cudaError_t cudaEventSynchronize(cudaEvent_t e) { tinycudart_capture_note("cudaEventSynchronize"); double _t0=tinycudart_now_ns(); cudaError_t _r; { tinycudart_count_sync(); TRACE("eventSynchronize %p", (void *)e); _r=S(tinynv_event_sync((tinynv_event_t)e)); } tinycudart_time_api(6, tinycudart_now_ns()-_t0); return _r; }

// --- copies and fills ---
// cudaMemcpyDefault needs to know which side a pointer is on; the only host memory we can vouch for is the pinned pool
static tinynv_status_t copy(tinynv_stream_t s, void *dst, const void *src, size_t n, enum cudaMemcpyKind kind) {
  if (kind == cudaMemcpyDefault) kind = tinycudart_host_owns(dst) ? cudaMemcpyDeviceToHost : tinycudart_host_owns(src) ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice;
  switch (kind) {
    case cudaMemcpyHostToDevice:   return tinynv_memcpy_htod(s, (tinynv_devptr_t)dst, src, n);
    case cudaMemcpyDeviceToHost:   return tinynv_memcpy_dtoh(s, dst, (tinynv_devptr_t)src, n);
    case cudaMemcpyDeviceToDevice: return tinynv_memcpy_dtod(s, (tinynv_devptr_t)dst, (tinynv_devptr_t)src, n);
    case cudaMemcpyHostToHost:     memcpy(dst, src, n); return TINYNV_OK;
    default: return TINYNV_ERR_INVALID;
  }
}
cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t n, enum cudaMemcpyKind kind, cudaStream_t s) { if (tinycudart_capturing_nv(st(s))) return tinycudart_capture_memcpy(dst, src, n, (int)kind); tinycudart_capture_offstream("cudaMemcpyAsync on another stream", st(s)); double _t0=tinycudart_now_ns(); cudaError_t _r; { TRACE("memcpyAsync kind=%d %zu bytes stream=%p", (int)kind, n, (void *)s); tinycudart_count_copy((int)kind, n); _r=S(copy(st(s), dst, src, n, kind)); } tinycudart_time_api(0, tinycudart_now_ns()-_t0); return _r; }
cudaError_t cudaMemcpyPeerAsync(void *dst, int ddev, const void *src, int sdev, size_t n, cudaStream_t s) { (void)ddev; (void)sdev; tinycudart_capture_note("cudaMemcpyPeerAsync"); return S(copy(st(s), dst, src, n, cudaMemcpyDeviceToDevice)); } // one device
// A strided device-to-device copy is ONE kernel launch (libtinycudart/copy2d.cu); anything else - a host side, or the
// null device, which runs no kernels - is `height` row copies. The row loop was the whole of SD 1.5's host time: one
// copy-engine batch and one socket round trip per row (docs/SHARED-STATUS.md 2026-09-15). The kernel is loaded once, on
// first use, from the cubin embedded in this archive; if it cannot be loaded the row loop is still correct, only slow.
extern unsigned char copy2d_cubin[]; extern unsigned int copy2d_cubin_len;
static tinynv_status_t copy2d_kernel(tinynv_stream_t s, void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height) {
  static tinynv_module_t mod; static tinynv_kernel_t k; static int tried, ok;
  if (!tried) {
    tried = 1;
    tinynv_device_props_t p; memset(&p, 0, sizeof p);
    if (tinynv_device_props(tinycudart_device(), &p) == TINYNV_OK && p.cc_major > 0 &&
        tinynv_module_load(tinycudart_device(), copy2d_cubin, copy2d_cubin_len, &mod) == TINYNV_OK &&
        tinynv_get_kernel(mod, "tinycudart_copy2d", &k) == TINYNV_OK) ok = 1;
    else fprintf(stderr, "[tinycudart] strided copies go row by row: the copy2d kernel is not available on this device\n");
  }
  if (!ok) return TINYNV_ERR_INVALID;
  unsigned long long a = (unsigned long long)(uintptr_t)dst | (unsigned long long)(uintptr_t)src | dpitch | spitch | width;
  int vec = !(a & 15) ? 16 : !(a & 7) ? 8 : !(a & 3) ? 4 : 1;
  struct { unsigned long long src, dst; long long spitch, dpitch, width, height; int vec; } __attribute__((packed)) prm;
  unsigned long long units = width / (unsigned)vec;
  unsigned gx = (unsigned)((units + 255) / 256); if (gx > 4096) gx = 4096; if (!gx) gx = 1;
  for (size_t r0 = 0; r0 < height; r0 += 65535) {
    size_t rows = height - r0 < 65535 ? height - r0 : 65535;
    prm.src = (unsigned long long)(uintptr_t)src + (unsigned long long)r0 * spitch; prm.dst = (unsigned long long)(uintptr_t)dst + (unsigned long long)r0 * dpitch;
    prm.spitch = (long long)spitch; prm.dpitch = (long long)dpitch; prm.width = (long long)width; prm.height = (long long)rows; prm.vec = vec;
    tinynv_status_t x = tinynv_launch(s, k, gx, (unsigned)rows, 1, 256, 1, 1, 0, &prm, sizeof prm);
    if (x != TINYNV_OK) return x;
  }
  return TINYNV_OK;
}
static cudaError_t cudaMemcpy2DAsync_noted(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width,
                                           size_t height, enum cudaMemcpyKind kind, cudaStream_t s);
// The strided copy: a capture cannot record it (there is no 2-D node here), so it says so and runs, which is a
// wrong replay rather than a wrong result now - the counter above is what makes that visible.
cudaError_t cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t s) {
  if (tinycudart_capturing_nv(st(s))) return tinycudart_capture_memcpy2d(dst, dpitch, src, spitch, width, height, (int)kind);
  return cudaMemcpy2DAsync_noted(dst, dpitch, src, spitch, width, height, kind, s);
}
static cudaError_t cudaMemcpy2DAsync_noted(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t s) {
  TRACE("memcpy2DAsync kind=%d %zux%zu (dpitch %zu spitch %zu)", (int)kind, width, height, dpitch, spitch); tinycudart_count_copy((int)kind, (unsigned long long)width * height);
  if (width > dpitch || width > spitch) return E(cudaErrorInvalidValue);
  if (dpitch == width && spitch == width) return S(copy(st(s), dst, src, width * height, kind));
  if (kind == cudaMemcpyDefault) kind = tinycudart_host_owns(dst) ? cudaMemcpyDeviceToHost : tinycudart_host_owns(src) ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice;
  if (kind == cudaMemcpyDeviceToDevice && height > 1 && copy2d_kernel(st(s), dst, dpitch, src, spitch, width, height) == TINYNV_OK) return E(cudaSuccess);
  for (size_t r = 0; r < height; r++) { tinynv_status_t x = copy(st(s), (char *)dst + r * dpitch, (const char *)src + r * spitch, width, kind); if (x != TINYNV_OK) return S(x); }
  return E(cudaSuccess);
}
cudaError_t cudaMemsetAsync(void *p, int v, size_t n, cudaStream_t s) { if (tinycudart_capturing_nv(st(s))) return tinycudart_capture_memset(p, v, n); double _t0=tinycudart_now_ns(); cudaError_t _r; { TRACE("memsetAsync %zu bytes = %d", n, v); tinycudart_count_copy(0, n); _r=S(tinynv_memset(st(s), (tinynv_devptr_t)p, v, n)); } tinycudart_time_api(2, tinycudart_now_ns()-_t0); return _r; }
cudaError_t cudaMemset(void *p, int v, size_t n) { tinycudart_capture_note("cudaMemset (synchronous)"); tinynv_status_t r = tinynv_memset(tinycudart_default_stream(), (tinynv_devptr_t)p, v, n); if (r == TINYNV_OK) r = tinynv_stream_sync(tinycudart_default_stream()); return S(r); }
cudaError_t cudaMallocManaged(void **p, size_t n, unsigned flags) { (void)p; (void)n; (void)flags; return E(cudaErrorNotSupported); } // only under GGML_CUDA_ENABLE_UNIFIED_MEMORY
cudaError_t cudaMemGetInfo(size_t *free_, size_t *total) { tinynv_device_props_t p = props(); if (total) *total = p.total_mem; if (free_) *free_ = p.total_mem; return E(cudaSuccess); } // TODO: subtract live cudaMalloc bytes

// --- device queries ---
cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp *d, int dev) {
  (void)dev; tinynv_device_props_t p = props(); memset(d, 0, sizeof *d);
  snprintf(d->name, sizeof d->name, "%s", p.name);
  d->totalGlobalMem = p.total_mem; d->major = p.cc_major; d->minor = p.cc_minor; d->multiProcessorCount = p.sm_count; d->warpSize = p.warp_size ? p.warp_size : 32;
  d->maxThreadsPerBlock = p.max_threads_per_block ? p.max_threads_per_block : 1024;
  d->sharedMemPerBlock = p.max_shared_per_block ? (size_t)p.max_shared_per_block : 48 * 1024;
  { int lim = optin_smem_for(p.cc_major, p.cc_minor); int drv = p.max_shared_per_sm ? p.max_shared_per_sm - 1024 : lim; d->sharedMemPerBlockOptin = (size_t)(drv < lim ? drv : lim); }
  d->sharedMemPerMultiprocessor = p.max_shared_per_sm; d->regsPerBlock = 65536; d->regsPerMultiprocessor = 65536;
  d->maxThreadsPerMultiProcessor = 2048; d->maxThreadsDim[0] = 1024; d->maxThreadsDim[1] = 1024; d->maxThreadsDim[2] = 64;
  d->maxGridSize[0] = 2147483647; d->maxGridSize[1] = 65535; d->maxGridSize[2] = 65535; d->integrated = 0; d->pciDomainID = 0; d->pciBusID = 1; d->pciDeviceID = 0;
  return E(cudaSuccess);
}
cudaError_t cudaGetDeviceProperties_v2(struct cudaDeviceProp *d, int dev) { return cudaGetDeviceProperties(d, dev); }
cudaError_t cudaDeviceGetAttribute(int *v, enum cudaDeviceAttr a, int dev) {
  (void)dev; tinynv_device_props_t p = props();
  switch (a) {
    case cudaDevAttrCooperativeLaunch: *v = 0; break;                     // so ggml never issues a cooperative launch
    case cudaDevAttrComputeCapabilityMajor: *v = p.cc_major; break;
    case cudaDevAttrComputeCapabilityMinor: *v = p.cc_minor; break;
    case cudaDevAttrMultiProcessorCount: *v = p.sm_count; break;
    case cudaDevAttrWarpSize: *v = p.warp_size ? p.warp_size : 32; break;
    case cudaDevAttrMaxThreadsPerBlock: *v = p.max_threads_per_block ? p.max_threads_per_block : 1024; break;
    case cudaDevAttrMaxSharedMemoryPerBlock: *v = p.max_shared_per_block ? p.max_shared_per_block : 48 * 1024; break;
    case cudaDevAttrMaxSharedMemoryPerBlockOptin: { int lim = optin_smem_for(p.cc_major, p.cc_minor); int drv = p.max_shared_per_sm ? p.max_shared_per_sm - 1024 : lim; *v = drv < lim ? drv : lim; } break;
    case cudaDevAttrMaxSharedMemoryPerMultiprocessor: *v = p.max_shared_per_sm; break;
    case cudaDevAttrIntegrated: case cudaDevAttrConcurrentManagedAccess: case cudaDevAttrManagedMemory: *v = 0; break;
    // The launch geometry and resource limits of the part, as libraries ask for them. CUB's DeviceScan asks for the
    // maximum grid width between its two kernels and gives up silently on an error, so a table that answered a dozen
    // attributes and refused the rest made every cumulative sum over a long row come back as memory contents
    // (test-backend-ops CUMSUM ne=[242004,1,1,1], found 2026-09-14). These are the GB202's values; anything below marked
    // "nominal" cannot change a result, only a heuristic.
    case cudaDevAttrMaxGridDimX: *v = 2147483647; break;
    case cudaDevAttrMaxGridDimY: case cudaDevAttrMaxGridDimZ: *v = 65535; break;
    case cudaDevAttrMaxBlockDimX: case cudaDevAttrMaxBlockDimY: *v = 1024; break;
    case cudaDevAttrMaxBlockDimZ: *v = 64; break;
    case cudaDevAttrMaxThreadsPerMultiProcessor: *v = 1536; break;
    case cudaDevAttrMaxBlocksPerMultiprocessor: *v = 32; break;                                    // nominal
    case cudaDevAttrMaxRegistersPerBlock: case cudaDevAttrMaxRegistersPerMultiprocessor: *v = 65536; break;
    case cudaDevAttrTotalConstantMemory: *v = 65536; break;
    case cudaDevAttrReservedSharedMemoryPerBlock: *v = 1024; break;
    case cudaDevAttrTextureAlignment: *v = 512; break;
    case cudaDevAttrMaxPitch: *v = 2147483647; break;
    case cudaDevAttrClockRate: *v = 2407000; break;                                                // kHz, nominal
    case cudaDevAttrMemoryClockRate: *v = 14001000; break;                                         // kHz, nominal
    case cudaDevAttrGlobalMemoryBusWidth: *v = 512; break;
    case cudaDevAttrL2CacheSize: *v = 100663296; break;                                             // 96 MB
    case cudaDevAttrUnifiedAddressing: case cudaDevAttrCanMapHostMemory: case cudaDevAttrConcurrentKernels:
    case cudaDevAttrGpuOverlap: case cudaDevAttrStreamPrioritiesSupported: case cudaDevAttrGlobalL1CacheSupported:
    case cudaDevAttrLocalL1CacheSupported: case cudaDevAttrComputePreemptionSupported:
    case cudaDevAttrCanUseHostPointerForRegisteredMem: *v = 1; break;
    case cudaDevAttrAsyncEngineCount: *v = 1; break;                                                // one copy engine exposed
    case cudaDevAttrSingleToDoublePrecisionPerfRatio: *v = 64; break;
    case cudaDevAttrPciBusId: *v = 1; break;
    case cudaDevAttrPciDeviceId: case cudaDevAttrPciDomainId: case cudaDevAttrComputeMode: case cudaDevAttrEccEnabled:
    case cudaDevAttrTccDriver: case cudaDevAttrKernelExecTimeout: case cudaDevAttrHostNativeAtomicSupported:
    case cudaDevAttrPageableMemoryAccess: case cudaDevAttrIsMultiGpuBoard: case cudaDevAttrMultiGpuBoardGroupID:
    case cudaDevAttrMemoryPoolsSupported: case cudaDevAttrGPUDirectRDMASupported: *v = 0; break;
    default: {
      // Refuse as before, but say so once per attribute: a library that swallows this error (CUB does) leaves no other trace.
      static int said[512]; if ((int)a >= 0 && (int)a < 512 && !said[(int)a]) { said[(int)a] = 1;
        fprintf(stderr, "[tinycudart] cudaDeviceGetAttribute: attribute %d is not modelled; returning cudaErrorInvalidValue\n", (int)a); }
      *v = 0; return E(cudaErrorInvalidValue); }
  }
  return E(cudaSuccess);
}
cudaError_t cudaDeviceGetPCIBusId(char *buf, int len, int dev) { (void)dev; snprintf(buf, (size_t)len, "0000:01:00.0"); return E(cudaSuccess); }
cudaError_t cudaDeviceCanAccessPeer(int *can, int a, int b) { (void)a; (void)b; *can = 0; return E(cudaSuccess); }
cudaError_t cudaDeviceEnablePeerAccess(int dev, unsigned flags) { (void)dev; (void)flags; return E(cudaErrorNotSupported); }
cudaError_t cudaSetDeviceFlags(unsigned flags) { (void)flags; return E(cudaSuccess); }
cudaError_t cudaPeekAtLastError(void) { return tinycudart_peek_error(); }

// --- per-kernel attributes and occupancy ---
#define MAXATTR 4096
static struct { const void *fn; int max_dyn_smem; } g_attr[MAXATTR]; static int g_nattr;
static int dyn_smem_limit(const void *fn) { for (int i = 0; i < g_nattr; i++) if (g_attr[i].fn == fn) return g_attr[i].max_dyn_smem; return 48 * 1024; }
cudaError_t cudaFuncSetAttribute(const void *fn, enum cudaFuncAttribute a, int value) {
  if (a != cudaFuncAttributeMaxDynamicSharedMemorySize) return E(cudaSuccess); // the others are hints
  { tinynv_device_props_t p = props(); int lim = optin_smem_for(p.cc_major, p.cc_minor); int drv = p.max_shared_per_sm ? p.max_shared_per_sm - 1024 : lim; if (drv < lim) lim = drv;
    if (value > lim) { fprintf(stderr, "[tinycudart] cudaFuncSetAttribute: %s asks for %d bytes of dynamic shared memory, this device allows %d\n", tinycudart_kernel_name(fn), value, lim); return E(cudaErrorInvalidValue); } }
  for (int i = 0; i < g_nattr; i++) if (g_attr[i].fn == fn) { g_attr[i].max_dyn_smem = value; return E(cudaSuccess); }
  if (g_nattr < MAXATTR) { g_attr[g_nattr].fn = fn; g_attr[g_nattr].max_dyn_smem = value; g_nattr++; }
  return E(cudaSuccess);
}
cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes *a, const void *fn) {
  tinynv_kernel_t k = tinycudart_kernel_for(fn); tinynv_kernel_info_t ki; memset(a, 0, sizeof *a);
  if (!k || tinynv_kernel_info(k, &ki) != TINYNV_OK) { fprintf(stderr, "[tinycudart] cudaFuncGetAttributes: %s (%p) has no device code\n", tinycudart_kernel_name(fn), fn); return E(cudaErrorInvalidDeviceFunction); }
  tinynv_device_props_t p = props();
  a->numRegs = ki.regs; a->sharedSizeBytes = (size_t)ki.static_smem; a->maxThreadsPerBlock = p.max_threads_per_block ? p.max_threads_per_block : 1024;
  a->constSizeBytes = (size_t)(ki.param_base + (ki.num_params ? ki.params[ki.num_params - 1].offset + ki.params[ki.num_params - 1].size : 0));
  a->maxDynamicSharedSizeBytes = dyn_smem_limit(fn); a->binaryVersion = a->ptxVersion = p.cc_major * 10 + p.cc_minor;
  return E(cudaSuccess);
}
// a conservative occupancy model: registers, shared memory and the thread ceiling, each rounded the way the SM does
cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *n, const void *fn, int block, size_t dyn, unsigned flags) {
  (void)flags; tinynv_kernel_t k = tinycudart_kernel_for(fn); tinynv_kernel_info_t ki;
  if (!k || tinynv_kernel_info(k, &ki) != TINYNV_OK) { *n = 0; fprintf(stderr, "[tinycudart] cudaOccupancy: %s (%p) has no device code\n", tinycudart_kernel_name(fn), fn); return E(cudaErrorInvalidDeviceFunction); }
  tinynv_device_props_t p = props(); int warps = (block + 31) / 32, regs = ki.regs ? ki.regs : 8;
  int by_regs = 65536 / (regs * warps * 32), by_threads = 2048 / (warps * 32);
  size_t smem = (size_t)ki.static_smem + dyn + 1024, per_sm = p.max_shared_per_sm ? (size_t)p.max_shared_per_sm : 100 * 1024;
  int by_smem = (int)(per_sm / smem); int r = by_regs < by_threads ? by_regs : by_threads; if (by_smem < r) r = by_smem; if (r > 32) r = 32; if (r < 0) r = 0;
  *n = r; return E(cudaSuccess);
}

// --- explicit launches: the argument array is marshalled into the constant-bank layout the cubin declares ---
static cudaError_t launch_args(tinynv_stream_t s, const void *fn, dim3 g, dim3 b, size_t dyn, void **args) {
  tinynv_kernel_t k = tinycudart_kernel_for(fn); tinynv_kernel_info_t ki;
  if (!k || tinynv_kernel_info(k, &ki) != TINYNV_OK) { fprintf(stderr, "[tinycudart] launch of unregistered function %p\n", fn); return E(cudaErrorInvalidDeviceFunction); }
  unsigned char blob[4096]; size_t len = 0; memset(blob, 0, sizeof blob);
  for (int i = 0; i < ki.num_params; i++) {
    size_t off = (size_t)ki.params[i].offset, sz = (size_t)ki.params[i].size;
    if (off + sz > sizeof blob) return E(cudaErrorInvalidValue);
    memcpy(blob + off, args[i], sz); if (off + sz > len) len = off + sz;
  }
  TRACE("launchEx %s grid=(%u,%u,%u) block=(%u,%u,%u) smem=%zu params=%zu", tinycudart_kernel_name(fn), g.x, g.y, g.z, b.x, b.y, b.z, dyn, len);
  if (tinycudart_capturing_nv(s))
    return tinycudart_capture_launch(k, tinycudart_kernel_name(fn), g.x, g.y, g.z, b.x, b.y, b.z, (unsigned)dyn, blob, len);
  tinycudart_capture_offstream("a kernel launch on another stream", s);
  tinycudart_count_launch(tinycudart_kernel_name(fn));
  if ((int)dyn > dyn_smem_limit(fn)) fprintf(stderr, "[tinycudart] %s: %zu bytes of dynamic shared memory exceeds the %d the kernel opted in to\n", tinycudart_kernel_name(fn), dyn, dyn_smem_limit(fn));
  double t0 = tinycudart_now_ns();
  tinynv_status_t st = tinynv_launch(s, k, g.x, g.y, g.z, b.x, b.y, b.z, (unsigned)dyn, blob, len);
  tinycudart_time_launch(tinycudart_kernel_name(fn), tinycudart_now_ns() - t0);
  return S(st);
}
cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_t *c, const void *fn, void **args) {
  // attributes (programmatic stream serialisation etc.) are scheduling hints; ignoring them keeps the launch fully ordered, which is correct
  return launch_args(st(c->stream), fn, c->gridDim, c->blockDim, c->dynamicSmemBytes, args);
}
cudaError_t cudaLaunchCooperativeKernel(const void *fn, dim3 g, dim3 b, void **args, size_t smem, cudaStream_t s) {
  (void)fn; (void)g; (void)b; (void)args; (void)smem; (void)s; return E(cudaErrorNotSupported); // grid-wide sync needs hardware support we do not expose; cudaDevAttrCooperativeLaunch says 0
}
