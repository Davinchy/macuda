# libtinynv test fixtures

Everything tracked here is built from this repository's own sources (`README-abi.md`, `README-attrs.md`).

## NVIDIA-built images: not shipped

Three tests run on images cut out of NVIDIA's own libraries. Those bytes are NVIDIA's, so they are not in this
repository. `make test` looks for them in `test/fixtures-nvidia/` (untracked; override with `NV_FIXTURES=<dir>`) and
prints `SKIPPED:` for each one that is absent - never a silent pass. To run them, extract the images below and check
the sha256 before trusting a result.

| file | bytes | sha256 | used by | where it comes from |
|---|---|---|---|---|
| `torch-cu128-assertfail-vprintf.sm120.cubin` | 24776 | `b8a29f4faff847fba8b41379494b3e8d81f7ffcb39fd5a5acf9e593e604fc166` | `test_reloc` | `libtorch_cuda.so`, torch 2.10.0+cu128 (x86_64): fatbin 295, entry 6, LZ4-decoded. Two kernels, 21 relocations, all R_CUDA_64; the smallest image there referencing BOTH undefined symbols the loader allows, `__assertfail` and `vprintf`. |
| `torch-cu128-unbounded-regs.sm120.cubin` | 396640 | `a162a3aa09e8d76e37843920d9b5157f16944382a774c6e98e9d68c1354141c2` | `test_thread_limit` | `libtorch_cuda.so`, torch 2.10.0+cu128: an sm_120 image whose kernels declare no `__launch_bounds__`, so their register count caps threads per block. |
| `cublaslt-cu128-deviceconstants.sm120.cubin` | 13616 | `2e8c744ccc4a3456fce7ecc5a7a060761c33ab4e07156fefe5730a873c52403f` | `test_global` | cuBLASLt from CUDA 12.8: the sm_120 image holding the device-constants data object `cublasCreate` looks up by name. |
