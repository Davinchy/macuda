// A kernel that fails without touching memory it should not.
//
// It exists for one check: that the driver's fault report can tell an illegal kernel from an unmapped address. Those
// arrive at the driver as the same bare notification from GSP-RM, they lead opposite ways, and until the report asked
// the debugger there was no way to know which had happened.
//
// `trap` is the instruction the hardware raises on, so this reports through the SMs' own error registers and sets no
// MMU fault valid bit - the other branch of the report, and the one nothing else exercises. The write beforehand is
// there so the kernel has done something real first: a launch that faults on its first instruction would also pass a
// report that only ever says "something went wrong", and this way the buffer says whether the kernel was running.
extern "C" __global__ void trap_after_write(float *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = 1.0f;
  __syncthreads();
  if (i == 0) asm volatile("trap;");
}
