// copy1d.cu — a plain device-to-device byte copy as ONE kernel launch, for the driver's download path.
//
// The logits read-back at every token boundary is a copy-engine transfer, and the copy engine and the compute engine
// hand the card to each other at that boundary through the runlist scheduler: ~0.5 ms for the copy engine to pick the
// copy up after the token's last kernel, and on about half the tokens another ~0.7 ms for the compute engine to pick the
// next token's first batch up after the copy (docs/SHARED-STATUS.md 2026-09-15 09:50-10:10; the timeslice curve
// established the mechanism and two cheaper levers on it failed). A copy done by a kernel on the compute queue is a
// queue after a queue, not a switch: libtinynv launches this kernel with dst = its host-visible staging buffer's GPU
// address and src = the caller's device address, waits, and memcpys to the caller as it does today.
//
// vec is the widest unit that src, dst and nbytes are all multiples of: 16, 8, 4 or 1; a thread strides over the copy
// in vec-sized units. Exactly three 8-byte parameters - destination, source, byte count - which is what
// tinynv_set_download_kernel validates against the cubin's own declaration; the driver writes them at the offsets the
// cubin declares and sizes the grid from the block_threads / bytes_per_thread the registration states (256 / 16).
// Build: nvcc -arch=sm_120 -O3 -cubin -o copy1d.cubin copy1d.cu   (remote nvcc)
#include <cuda_runtime.h>
#include <stdint.h>
extern "C" __global__ void __launch_bounds__(256) tinycudart_copy1d(unsigned char* __restrict__ dst, const unsigned char* __restrict__ src,
                                                                  unsigned long long nbytes) {
  // exactly three 8-byte parameters, as the driver's registration checks against this cubin's own declaration; the
  // unit width is derived here from what all three are multiples of, so no fourth parameter has to be agreed on
  unsigned long long a = (unsigned long long)(uintptr_t)dst | (unsigned long long)(uintptr_t)src | nbytes;
  int vec = !(a & 15) ? 16 : !(a & 7) ? 8 : !(a & 3) ? 4 : 1;
  unsigned long long units = nbytes / vec;
  unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
  for (unsigned long long u = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; u < units; u += stride) {
    switch (vec) {
      case 16: ((uint4*)dst)[u] = ((const uint4*)src)[u]; break;
      case 8:  ((uint2*)dst)[u] = ((const uint2*)src)[u]; break;
      case 4:  ((unsigned int*)dst)[u] = ((const unsigned int*)src)[u]; break;
      default: dst[u] = src[u]; break;
    }
  }
}
