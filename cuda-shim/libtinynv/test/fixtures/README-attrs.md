# attrs-* fixtures: per-function attributes, with the real driver's answers

Built by CUDA 13.0 (nvcc V13.0.88) on the 3090 box, 2026-09-21, from the two sources here:

    nvcc -cubin -arch=sm_80 attrs-props.cu -o attrs-p80.cubin                                  sha256 299d5cdbe665db28add486f3a7b6c3d05185474121146028b605a6995bae2e6d
    nvcc -cubin -gencode arch=compute_75,code=sm_86 attrs-props.cu -o attrs-p75v86.cubin        sha256 9ff15c154a0878ba26ed3c8c0e81a47153174d271fd0e3dca6d34ad4ce3c4740
    nvcc -cubin -arch=sm_120 attrs-clusters.cu -o attrs-c120.cubin                              sha256 d38e6f6283be8781f2637c6ba7efa3a612f90725eac150ef66f53317573f68c7

Each kernel changes one property: plain, a local array the compiler kept in registers, __constant__ data, a launch
bound, a 4 KB per-thread array (real stack), a call into a function with a 1 KB frame, printf, and, for sm_120,
__cluster_dims__ (2,1,1) and (2,2,1) and a maximum cluster rank.

What NVIDIA's driver 580.126.18 on the RTX 3090 reported through cuFuncGetAttribute is the expected table in
test_kernel_attrs.c. The sm_120 file cannot load on that card, so its cluster expectations come from the source.
