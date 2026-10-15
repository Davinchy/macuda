// What a kernel is told about its own launch geometry, written down so it can be read rather than inferred.
//
// On this architecture blockDim and gridDim do not come from special registers: the driver has to place them in the
// parameter region of constant bank 0, and nothing in the recorded sessions does, because tinygrad compiles its sizes in
// as literals and never asks. Their positions were found by filling that region with one marker per word and reading
// back which markers a kernel returned; this kernel is that instrument, and it stays as the check that they are right.
extern "C" __global__ void dims(unsigned *out) {
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    unsigned *p = out + (blockIdx.z * gridDim.y * gridDim.x + blockIdx.y * gridDim.x + blockIdx.x) * 8;
    p[0] = gridDim.x;  p[1] = gridDim.y;  p[2] = gridDim.z;
    p[3] = blockDim.x; p[4] = blockDim.y; p[5] = blockDim.z;
    p[6] = blockIdx.x * 100 + blockIdx.y * 10 + blockIdx.z;
    p[7] = 0xa5a5a5a5u;
  }
}
