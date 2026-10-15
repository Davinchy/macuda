// The fatbin and wrong-architecture fixtures for test_module_load_data.c: one trivial kernel.
extern "C" __global__ void kfat(float *p) { p[threadIdx.x] = 1.0f; }
