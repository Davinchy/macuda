// occ_xcheck.cpp - tinynv_kernel_thread_limit against NVIDIA's own occupancy calculator, every case.
//
// Not in make test: it needs cuda-13/include/cuda_occupancy.h, which lives untracked in the main checkout only.
//   clang++ -std=c++17 -O1 -I<main checkout>/cuda-13/include -Isrc -I../include -o occ_xcheck tools/occ_xcheck.cpp build/libtinynv.a
// Measured 2026-09-21 (B): 8224 of 8224 agree (regs 0..256 x block 32..1024). Controls: the old always-1024 rule
// disagrees with NVIDIA in 3712 cases, and this rule without the round-up to 4 sub-partitions in 240, so a clean
// result here is one this comparison could have failed.
// NVIDIA's occupancy calculator (cuda-13/include/cuda_occupancy.h) vs tinynv_kernel_thread_limit, every case.
// A block of B threads for a kernel with R registers: NVIDIA says it fits if cudaOccMaxActiveBlocksPerMultiprocessor
// returns at least one block; ours says it fits if B <= the limit. Only registers vary: no shared memory, no bound.
#include "cuda_occupancy.h"
#include <cstdio>
#include <cstring>
extern "C" {
#include "cubin.h"
}
int main() {
  cudaOccDeviceProp prop;
  prop.computeMajor = 12; prop.computeMinor = 0;          // GB202, sm_120
  prop.maxThreadsPerBlock = 1024; prop.maxThreadsPerMultiprocessor = 1536;   // 48 warps per SM on sm_120
  prop.regsPerBlock = 65536; prop.regsPerMultiprocessor = 65536; prop.warpSize = 32;
  prop.sharedMemPerBlock = 49152; prop.sharedMemPerMultiprocessor = 102400; prop.numSms = 170;
  prop.sharedMemPerBlockOptin = 101376; prop.reservedSharedMemPerBlock = 1024;
  cudaOccDeviceState state;
  long cases = 0, agree = 0, disagree = 0, ctl_old = 0, ctl_nosub = 0;
  for (int r = 0; r <= 256; r++) {
    tinynv_kernel_desc_t d; memset(&d, 0, sizeof d); d.regs = (uint32_t)r;
    uint32_t mine = tinynv_kernel_thread_limit(&d);
    for (int b = 32; b <= 1024; b += 32) {
      cudaOccFuncAttributes attr;
      attr.maxThreadsPerBlock = 1024; attr.numRegs = r; attr.sharedSizeBytes = 0;
      attr.partitionedGCConfig = PARTITIONED_GC_OFF; attr.shmemLimitConfig = FUNC_SHMEM_LIMIT_DEFAULT;
      attr.maxDynamicSharedSizeBytes = 0; attr.numBlockBarriers = 1;
      cudaOccResult res;
      cudaOccError e = cudaOccMaxActiveBlocksPerMultiprocessor(&res, &prop, &attr, &state, b, 0);
      int nv_fits = e == CUDA_OCC_SUCCESS && res.activeBlocksPerMultiprocessor > 0;
      int my_fits = (uint32_t)b <= mine;
      /* CONTROLS: the rule this replaces (always 1024), and ours without the round-up to 4 sub-partitions */
      int old_fits = 1;
      uint32_t pw = ((uint32_t)r * 32 + 255) / 256 * 256, w = (uint32_t)(b + 31) / 32;
      int nosub_fits = r <= 256 && (!pw || pw * w <= 65536);
      if (e == CUDA_OCC_SUCCESS) { ctl_old += old_fits != nv_fits; ctl_nosub += nosub_fits != nv_fits; }
      cases++;
      if (nv_fits == my_fits) agree++;
      else if (disagree++ < 10) printf("  DISAGREE regs %d block %d: NVIDIA %s (err %d, blocks %d, limiter %#x), ours %s (limit %u)\n",
                                       r, b, nv_fits ? "fits" : "does not fit", (int)e, res.activeBlocksPerMultiprocessor,
                                       res.limitingFactors, my_fits ? "fits" : "does not fit", mine);
    }
  }
  printf("%ld cases (regs 0..256 x block 32..1024): agree %ld, disagree %ld\n", cases, agree, disagree);
  printf("controls: the old always-1024 rule disagrees with NVIDIA in %ld cases; ours without sub-partition rounding in %ld\n",
         ctl_old, ctl_nosub);
  return disagree || !ctl_old || !ctl_nosub ? 1 : 0;
}
