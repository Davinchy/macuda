// Proves how much dynamic shared memory a launch actually got, rather than that the descriptor was accepted.
//
// The descriptor carries one shared-memory size and a configuration chosen to cover it, and above 100 KB that
// configuration is territory no recorded session has been through. A launch that comes back without an error proves
// only that the front end liked the numbers; it says nothing about whether the memory is there. So the kernel writes a
// pattern through the whole span it was promised, reads it back, and reports the first word that disagrees.
//
// The pattern is index-dependent, so memory that aliases onto a smaller real allocation shows up as a mismatch rather
// than as a plausible value, and the fold means a byte-swapped or shifted read does not pass either.
extern "C" __global__ void smem_probe(unsigned *first_bad, unsigned bytes) {
  extern __shared__ unsigned char raw[];
  unsigned *s = reinterpret_cast<unsigned *>(raw);
  const unsigned words = bytes / 4;
  for (unsigned i = threadIdx.x; i < words; i += blockDim.x) s[i] = i ^ 0x5a5a5a5au;
  __syncthreads();
  for (unsigned i = threadIdx.x; i < words; i += blockDim.x)
    if (s[i] != (i ^ 0x5a5a5a5au)) atomicMin(first_bad, i);
}
