// A kernel that can print, which is the only reason a cubin references an undefined symbol.
//
// nvcc turns device-side printf into a call through `vprintf`, a function the real driver supplies out of its device
// runtime, so the cubin carries an R_CUDA_64 relocation against a symbol it does not contain. Across all 183 cubins of
// a full ggml build it is the only undefined symbol there is - 115 of them, every one from an assert path - and
// refusing it cost every quantised model's decode path until 2026-09-14.
//
// The print is behind a condition that is never true, which is exactly how ggml's asserts carry it: the relocation is
// in the cubin whether or not any thread ever reaches the call.
extern "C" __global__ void printf_kernel(unsigned *out, unsigned n) {
  unsigned i = threadIdx.x;
  if (i < n) out[i] = i * 3u + 1u;
  if (n == 0xffffffffu) printf("unreachable, and the reason this cubin needs vprintf: %u\n", i);
}
