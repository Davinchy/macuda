// test_hw_namedbar's kernels (B, 2026-09-24; the C8 FP8 fault: BAR.SYNC 7 in a launch granted BARRIER_COUNT 1).
// Built with CUDA 13.0: nvcc -arch=sm_120a -cubin -o namedbar.sm120a.cubin namedbar.cu (3090 box).
// nb_named: 64 threads sync on NAMED barrier 1 (bar.sync 1, 64) between a shared write and a read of the other half, so
//           the cubin declares EIATTR_NUM_BARRIERS 2. out[t] = the value thread (t ^ 32) wrote, + 1: correct only if the
//           barrier worked. nb_plain: the same data flow on barrier 0 (__syncthreads), NUM_BARRIERS 1 - the control.
extern "C" __global__ void nb_named(unsigned *out) {
  __shared__ unsigned s[64];
  unsigned t = threadIdx.x;
  s[t] = 0xB0000000u + t;
  asm volatile("bar.sync 1, 64;" ::: "memory");
  out[t] = s[t ^ 32u] + 1u;
}
extern "C" __global__ void nb_plain(unsigned *out) {
  __shared__ unsigned s[64];
  unsigned t = threadIdx.x;
  s[t] = 0xB0000000u + t;
  __syncthreads();
  out[t] = s[t ^ 32u] + 1u;
}
