# libtinynv test fixtures — captured bytes, tracked

`torch-cu128-assertfail-vprintf.sm120.cubin` — one sm_120 image out of `libtorch_cuda.so` from torch **2.10.0+cu128**
(the bench's x86_64 copy, `/mnt/4tb/l-shim/cu128-site`; the guest runs the same torch release, aarch64): fatbin 295,
entry 6, LZ4-decoded by D's `fatinv.c` with liblz4 and a hand decoder agreeing (`DUMP=295:6`). 24776 bytes, sha256
`b8a29f4faff847fba8b41379494b3e8d81f7ffcb39fd5a5acf9e593e604fc166`. Two kernels, 21 relocations, all R_CUDA_64.

It is the smallest image in that library that references BOTH undefined symbols the loader allows, `__assertfail`
and `vprintf`, so `test_reloc` exercises both names and both rename-refusals on real bytes. Unlike
`../spike/printf_kernel.sm120.cubin` (build output, untracked, and two different builds of it exist in two worktrees),
this one is in the repository, so that arm of `make test` cannot skip.
