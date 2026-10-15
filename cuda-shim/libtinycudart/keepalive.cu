// keepalive.cu - built on the Pop!_OS box: nvcc -cubin -O3 -gencode arch=compute_120a,code=sm_120a -o keepalive.cubin keepalive.cu
// tinynv_keepalive.cu - keeps the compute engine scheduled across a token boundary. One block of one warp polls a flag
// in host memory and exits the moment the host sets it (just before it hands over the next chain), or after max_cycles
// as the watchdog, so a host that never comes back cannot leave the engine spinning. Nothing waits on this kernel: it
// is launched after the token's last download and the next chain's acquire is satisfied as soon as it exits.
extern "C" __global__ void tinynv_keepalive(const volatile unsigned int *flag, unsigned long long max_cycles) {
  if (threadIdx.x != 0) return;
  unsigned long long t0 = clock64();
  while (*flag == 0u) {
    if (clock64() - t0 > max_cycles) break;
    __nanosleep(2000);
  }
}
