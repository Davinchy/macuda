// A kernel that reads a __device__ table, which is how every IQ-quantised llama.cpp kernel finds its dequantisation
// grid. The table lives in .nv.global.init; its address is not known until the image is placed, so the compiler emits a
// relocation into a constant bank - bank 4 in the ggml cubins - and the driver has to apply it and bind that bank.
//
// Skipping either step does not fail to load. The kernel launches, reads a pointer that is zero or stale, and returns
// whatever that address happened to hold. So this reads every entry and writes what it found, and the caller checks the
// values rather than the absence of an error.
__device__ const unsigned table[64] = {
  0x00000000u, 0x11111111u, 0x22222222u, 0x33333333u, 0x44444444u, 0x55555555u, 0x66666666u, 0x77777777u,
  0x88888888u, 0x99999999u, 0xaaaaaaaau, 0xbbbbbbbbu, 0xccccccccu, 0xddddddddu, 0xeeeeeeeeu, 0xffffffffu,
  0x0f0f0f0fu, 0x1e1e1e1eu, 0x2d2d2d2du, 0x3c3c3c3cu, 0x4b4b4b4bu, 0x5a5a5a5au, 0x69696969u, 0x78787878u,
  0x87878787u, 0x96969696u, 0xa5a5a5a5u, 0xb4b4b4b4u, 0xc3c3c3c3u, 0xd2d2d2d2u, 0xe1e1e1e1u, 0xf0f0f0f0u,
  0x01020304u, 0x05060708u, 0x090a0b0cu, 0x0d0e0f10u, 0x11121314u, 0x15161718u, 0x191a1b1cu, 0x1d1e1f20u,
  0x21222324u, 0x25262728u, 0x292a2b2cu, 0x2d2e2f30u, 0x31323334u, 0x35363738u, 0x393a3b3cu, 0x3d3e3f40u,
  0xdeadbeefu, 0xcafebabeu, 0xfeedfaceu, 0x8badf00du, 0x0defaced2u & 0xffffffffu, 0xfacefeedu, 0xbaddcafeu, 0xdeadc0deu,
  0x13579bdfu, 0x2468ace0u, 0xfdb97531u, 0x0eca8642u, 0xa1b2c3d4u, 0xe5f60718u, 0x293a4b5cu, 0x6d7e8f90u};

extern "C" __global__ void cbank(unsigned *out, unsigned n) {
  for (unsigned i = threadIdx.x; i < n; i += blockDim.x) out[i] = table[i];
}
