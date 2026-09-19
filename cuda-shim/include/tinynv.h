// tinynv.h — the driver interface libtinynv (Session B) implements and libtinycudart (Session A) calls.
// A↔B contract. Mirrors tinygrad's ops_nv: module_load≈NVProgram, malloc/memcpy≈NVAllocator, event≈timeline signal.
// Design notes: opaque handles; kernels are launched with CUDA grid/block semantics (blocks × threads-per-block), so the
// driver sets the QMD cta_raster + cta_thread_dimension directly (this is where tinygrad's launch wrapper fell short in M2).
#ifndef TINYNV_H
#define TINYNV_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct tinynv_device* tinynv_device_t;
typedef struct tinynv_module* tinynv_module_t;   // a loaded cubin
typedef struct tinynv_kernel* tinynv_kernel_t;   // one __global__ entry within a module
typedef struct tinynv_stream* tinynv_stream_t;   // in-order queue (GPFIFO sub-queue)
typedef struct tinynv_event*  tinynv_event_t;    // timeline value
typedef uint64_t tinynv_devptr_t;                // device VA

typedef enum { TINYNV_OK=0, TINYNV_ERR_NODEV, TINYNV_ERR_OOM, TINYNV_ERR_LAUNCH, TINYNV_ERR_INVALID, TINYNV_ERR_DRIVER } tinynv_status_t;

// Why the last call failed, in a sentence, for the thread that made it. A status says which kind of thing went wrong;
// this says which thing. Without it a launch refused for asking 131,200 bytes of shared memory when the driver has
// evidence for 102,400 arrives as "bad launch", and the difference between those two is an hour.
//
// Per thread, and valid until the next failing call on that thread: a success does not clear it, so the reason for a
// failure survives the calls a caller makes while reacting to it. Reads "no error" when nothing has failed yet.
const char *tinynv_last_error(void);

// Which build this is: the commit it came from, plus "-dirty" when the tree had uncommitted changes. Worth logging
// before a run, because a binary linked before a fix landed reports the same symptoms the fix was for, and three runs
// in this project have been diagnosed against code they were not running.
const char *tinynv_build_id(void);

// device lifecycle
tinynv_status_t tinynv_init(void);
int             tinynv_device_count(void);
tinynv_status_t tinynv_device_get(tinynv_device_t* out, int ordinal);
// props the cudart shim needs for cudaGetDeviceProperties/DeviceGetAttribute
typedef struct {
  char name[256]; int cc_major, cc_minor; int sm_count; int warp_size;
  size_t total_mem; int max_threads_per_block; int max_shared_per_block; int max_shared_per_sm;
} tinynv_device_props_t;
tinynv_status_t tinynv_device_props(tinynv_device_t, tinynv_device_props_t* out);

// modules & kernels (cubin is a raw ELF, already extracted from the fatbin by the shim)
tinynv_status_t tinynv_module_load(tinynv_device_t, const void* cubin, size_t len, tinynv_module_t* out);
tinynv_status_t tinynv_module_unload(tinynv_module_t);
tinynv_status_t tinynv_get_kernel(tinynv_module_t, const char* name, tinynv_kernel_t* out);
// param layout the shim needs to marshal args correctly (from the cubin's EIATTR_KPARAM_INFO 0x17 / PARAM_CBANK 0x0a)
typedef struct { int num_params; int param_base; struct { int offset, size; } params[64]; int regs, static_smem; } tinynv_kernel_info_t;
tinynv_status_t tinynv_kernel_info(tinynv_kernel_t, tinynv_kernel_info_t* out);

// memory
// Every device pointer is aligned to at least TINYNV_DEVPTR_ALIGN bytes. This is a guarantee, not an accident of the
// allocator: callers lay tensors out from a buffer's base assuming it, and silently lose the difference to padding when
// it does not hold. ggml assumes at least 128 and CUDA promises 256, so this promises 256.
#define TINYNV_DEVPTR_ALIGN 256
tinynv_status_t tinynv_malloc(tinynv_device_t, size_t, tinynv_devptr_t* out);
tinynv_status_t tinynv_free(tinynv_device_t, tinynv_devptr_t);
tinynv_status_t tinynv_memcpy_htod(tinynv_stream_t, tinynv_devptr_t dst, const void* src, size_t);
tinynv_status_t tinynv_memcpy_dtoh(tinynv_stream_t, void* dst, tinynv_devptr_t src, size_t);
tinynv_status_t tinynv_memcpy_dtod(tinynv_stream_t, tinynv_devptr_t dst, tinynv_devptr_t src, size_t);
tinynv_status_t tinynv_memset(tinynv_stream_t, tinynv_devptr_t dst, int value, size_t);
// pinned host memory (honours the dext 32-segment DMA cap; may return TINYNV_ERR_INVALID → shim falls back to pageable)
tinynv_status_t tinynv_host_alloc(tinynv_device_t, size_t, void** out);
tinynv_status_t tinynv_host_free(tinynv_device_t, void*);

// streams / events / launch
tinynv_status_t tinynv_stream_create(tinynv_device_t, tinynv_stream_t* out);
tinynv_status_t tinynv_stream_destroy(tinynv_stream_t);
tinynv_status_t tinynv_stream_sync(tinynv_stream_t);

// Does a synchronise on this stream have to WAIT, or only hand over what has been built?
//
// Non-zero means something outstanding requires the engine to finish: a kernel, a transfer, a download. Zero means the
// only work outstanding is host-to-device data that rides in the command stream itself, and a caller whose host buffer
// is the thing it wants back is already safe - the bytes are in the batch that carries them, and anything that later
// reads them is behind that batch on the same queue. Such a caller must still call tinynv_stream_flush, so the batch
// actually goes out; it need not poll for completion.
//
// This lives here rather than being a second success code from tinynv_memcpy_htod because `!= TINYNV_OK` is the
// prevailing idiom at every call site in the shim, so a second SUCCESS value in that enum would be read as a failure
// by code that is correct today. And it is answered by the driver rather than mirrored by the caller: the driver
// knows what it submitted, a caller counting alongside it can drift, and the failure mode of drift here is a
// synchronise that returns while a kernel is still running.
int tinynv_stream_needs_wait(tinynv_stream_t);

// Hand over everything built so far without waiting for it. See tinynv_stream_needs_wait.
tinynv_status_t tinynv_stream_flush(tinynv_stream_t);

// Lend the driver a kernel that copies device memory to a staging buffer, so a download can be served by the COMPUTE
// engine instead of the copy engine.
//
// Why a caller has to provide it: a decode's token boundary has the copy engine and the compute engine contending,
// and the card's scheduling of the two decides who waits - measured at ~500-1,100 us on about half of all tokens.
// Serving the download from the compute channel removes the second engine from the boundary entirely. The driver
// carries no device code of its own and should not start: a driver that embeds a cubin has to keep embedding one for
// every architecture it ever meets. A caller already has kernels resident.
//
// The kernel must take exactly three 8-byte parameters in this order - destination, source, byte count - and copy
// `nbytes` bytes, bounds-checking its own threads. It is validated against the cubin's OWN declaration at
// registration rather than trusted: wrong parameter counts or sizes are refused here, loudly, instead of producing a
// wrong copy at the first download. `block_threads` and `bytes_per_thread` say how the driver should size the grid;
// they are the kernel's properties and only it knows them.
//
// Registering a kernel does not turn this on. TINYNV_DOWNLOAD_VIA_COMPUTE=1 does, and it names itself at start-up.
tinynv_status_t tinynv_set_download_kernel(tinynv_device_t, tinynv_kernel_t,
                                           unsigned block_threads, unsigned bytes_per_thread);
// Lend the driver a keep-alive kernel: one block that polls a flag in host memory and exits when the flag is set or
// after a cycle budget (the watchdog). Under TINYNV_KEEPALIVE=1 the driver launches it after every synchronisation
// - the moment the compute engine has gone idle - so the engine stays scheduled while the host samples and builds
// the next token, and sets the flag just before the next chain is handed over (or before any wait), so nothing
// ever waits on the spin. Measured reason: after the engine idles at a token boundary it takes ~0.3-0.5 ms to reach
// the next batch, where a mid-token seam costs ~40 us. Two parameters, both 8 bytes: the flag's address and the
// cycle budget. Registration alone changes nothing; the knob does, and names itself at start-up.
tinynv_status_t tinynv_set_keepalive_kernel(tinynv_device_t, tinynv_kernel_t);
tinynv_status_t tinynv_event_create(tinynv_device_t, tinynv_event_t* out);
tinynv_status_t tinynv_event_record(tinynv_event_t, tinynv_stream_t);
tinynv_status_t tinynv_event_sync(tinynv_event_t);
tinynv_status_t tinynv_stream_wait_event(tinynv_stream_t, tinynv_event_t);
// the hot path. params is one contiguous blob laid out per tinynv_kernel_info (param_base..). dyn_smem in bytes.
tinynv_status_t tinynv_launch(tinynv_stream_t, tinynv_kernel_t,
                              unsigned gx, unsigned gy, unsigned gz,       // grid (blocks)
                              unsigned bx, unsigned by, unsigned bz,       // block (threads)
                              unsigned dyn_smem, const void* params, size_t params_len);
// What the card says about itself. Temperature, power, clocks and how busy it is, read from a block GSP-RM keeps
// fresh in host memory - so this costs a load rather than a message, which matters on a backend where a register read
// is a round trip to another process. Fan speed is not here because NVIDIA does not publish it.
//
// A field whose `_ok` is zero was not reported: either the card does not measure it on this platform, or the firmware
// was mid-write every time we looked. Do not print those as zero; say they are not available.
typedef struct {
  int gpu_temp_ok, mem_temp_ok, power_ok, limit_ok, clock_ok, util_ok, pstate_ok, throttle_ok;
  double gpu_temp_c, mem_temp_c;       // celsius
  unsigned gpu_power_mw, mem_power_mw; // milliwatts, averaged by the firmware
  unsigned power_limit_mw;             // what it is currently allowed to draw
  unsigned graphics_mhz, memory_mhz, video_mhz, sm_mhz;
  unsigned gpu_busy_pct, mem_busy_pct;
  unsigned pstate;
  unsigned throttle;                   // bitmask; see tinynv_throttle_str
} tinynv_sensors_t;

tinynv_status_t tinynv_sensors(tinynv_device_t, tinynv_sensors_t* out);
// How much video memory the firmware believes it can still allocate. Diagnostic; see test/test_hw_heap.c.
int tinynv_device_gsp_free_heap(tinynv_device_t, uint64_t* out);
// Where the firmware's write-protected region actually is, beside the top of what this driver's manager hands out.
// Diagnostic; see test/test_hw_wpr.c. All four are bytes.
int tinynv_device_wpr2_range(tinynv_device_t, uint64_t* base, uint64_t* limit, uint64_t* vram, uint64_t* managed_end);
// Does the firmware answer NV0041_CTRL_CMD_GET_SURFACE_PHYS_ATTR? Allocates its own class-0x0040 object of `size`,
// asks where `offset` lives, frees it. Diagnostic; see test/test_hw_physattr.c for why it is a question at all.
int tinynv_device_probe_phys_attr(tinynv_device_t, uint64_t size, uint64_t offset, uint64_t* paddr, uint64_t* contig,
                                  unsigned* aperture);
// The reasons the card is holding its clocks back, as words. Returns how many were written.
int tinynv_throttle_str(unsigned mask, const char** out, int max);

const char* tinynv_status_str(tinynv_status_t);

#ifdef __cplusplus
}
#endif
#endif
