// Compile-time cluster settings, one per kernel, so the records that carry them can be told apart (sm_90+ only).
extern "C" __global__ void c_plain(float *o) { o[threadIdx.x] = 1; }
extern "C" __global__ void __cluster_dims__(2, 1, 1) c_dims2(float *o) { o[threadIdx.x] = 2; }
extern "C" __global__ void __cluster_dims__(2, 2, 1) c_dims22(float *o) { o[threadIdx.x] = 3; }
extern "C" __global__ void __launch_bounds__(256, 1, 4) c_maxrank(float *o) { o[threadIdx.x] = 4; }
