# bf16 GEMMs on the tensor cores (B, 2026-09-21): the evidence for the fix in this branch

**The debt.** llama.cpp's bf16 prompt processing through the shim read pp256 = 306.56 tok/s on the 5090 (root
`logs/shim-bench-20260921-141436.log`; Qwen2.5-3B-Instruct BF16). Q8_0 reads 15,254 through ggml's own MMQ kernels, and Metal
bf16 reads 2,209.

**The cause** is `libtinycublas/cublas.c`. `launch_gemm` and `cublasGemmStridedBatchedEx` sent only f32 (in_dtype 0) and f16
(1) to the tensor-core kernels. bf16 (2) always took `tinyblas_gemm_f32`, the correctness-first scalar kernel: one thread
per output element, a serial dot product over k, no tiling and no shared memory.

**The calls** (`trace-before.txt`, `trace-after.txt`) come from the shim's own trace (TINYCUDART_TRACE=1) on the null
device: the validated `build/bin/llama-bench-null`, llama.cpp ad6c668. The model is a shapes-only GGUF made by
`mk_shapes_gguf.py`: the real dimensions, 2 layers, zero weights. ggml ad6c668 calls `cublasGemmEx(CUBLAS_OP_T,
CUBLAS_OP_N, m = out features, n = 256, k = in features, bf16 A lda=k, bf16 B ldb=k, f32 C ldc=m, CUBLAS_COMPUTE_32F)`,
ggml-cuda.cu:1552, with `prefer_f32_output` on NVIDIA at :1514. Per pp256 on the 36-layer model:
- m 2048, k 2048 (Q, O): 72 calls;
- m 256, k 2048 (K, V): 72;
- m 11008, k 2048 (gate, up): 70;
- m 2048, k 11008 (down): 35.

The last layer's FFN and the output head run on the one output token, as matrix-vector products, not through cuBLAS.

**The fix, as this branch holds it (v2, commit 81b9826).**
- `gemm.cu` templates the vectorised `tinyblas_gemm_tc2` on its input element and adds eight `tinyblas_gemm_bf16_tc2*`
  entry points.
- The direct fragment store (`wmma::store_matrix_sync` straight into C) runs only when C's base is 32-byte aligned and
  `ldc % 4 == 0`, wmma's documented contract. Every other tile goes through the staging path. This changes the f16 tc2
  kernels too, since they share the template.
- `cublas.c` routes CUDA_R_16BF to the bf16 tc2 kernels under the f16 path's alignment rule. A bf16 GEMM they refuse goes
  to the scalar kernel, never to the f16 one.
- `cublasGemmEx`, `GemmStridedBatchedEx` and `GemmBatchedEx` refuse (CUBLAS_STATUS_NOT_SUPPORTED) any combination the
  library does not serve:
  - A and B of different types, or a type it has no code for;
  - bf16 with any compute type but COMPUTE_32F or 32F_PEDANTIC;
  - f16 with any compute type but those two, except 16F (or 16F_PEDANTIC) with an f16 C;
  - f32 with any compute type outside the 32F family (for example 64F, 32I or 16F).
- Under both 16F compute types, alpha and beta are read as half.
- `gemm.cubin` is rebuilt for sm_120 with nvcc 13.0 on the 3090 box, and `build-gemm-cubin.sh` rebuilds it from source.

**What changed from v1 (commits 39d8bb7 and 2dbc56f), per G's review** (root `docs/review/20260921-194653-bf16-gemm-fix.md`):

1. **The direct store's alignment.**
   - v1 stored straight into C under the tile-bounds test alone.
   - v2 adds the alignment test.
   - `align-v1.out` and `align-v2.out` hold four cases, each in its own process: f16 and bf16, each with ldc 33 and with C
     at base + 16. **All eight pass on both.**
   - The reason is in `cubin-compare-v2.txt`: every global store these kernels compile to is a 32-bit `STG.E` or a 16-bit
     `STG.E.U16`, on sm_86 and on sm_120, v1 and v2. With no vector store, alignment cannot fail with this toolkit.
   - **So v2 enforces the specification, but no failure of v1 is observable on this toolchain.** The alignment rows are not
     a control that has been seen to fail.
2. **The bench gate's NaN blindness.**
   - v1's reduction, `fmax` over |x − ref|, drops NaN, so an all-NaN or one-NaN result read error 0.
   - v2 refuses any non-finite value before the reduction.
   - `gate_check.c` and `gate-check.out` show both reductions on those two inputs: v1 accepts both, v2 refuses both.
   - `bench_bf16.cu` (v2) runs both mutants at every shape, and all are refused.
3. **The type contract.**
   - `test/cublas_contract.c` has 13 rows through all three Ex entry points, runs on the null device, and is in `make test`.
   - `contract-test.out`: this branch passes 13 of 13. The same test linked against main 715b281's libtinycublas fails 8,
     each one a combination main served silently.
4. **The matrix beyond llama.cpp's one call shape.**
   - `bench_gemm_contract.cu matrix` runs every tc2 kernel against cuBLAS, 32 cases:
     - f16 and bf16;
     - all four transposes;
     - f32 output and same-type output;
     - (alpha, beta) of (1, 0) and (0.5, 2);
     - m 48, n 40, k 72 (partial fragments and a partial K slab);
     - strided batch 3.
   - `matrix-v2.out`: 32 of 32 within tolerance.
   - **Control, `matrix-control.out`:** the kernel's alpha scaled by 1.05. All 32 are refused.
5. **The cubin's provenance.**
   - `build-gemm-cubin.sh` builds sm_120 twice from one source, requires the two builds to match, then compares the result
     with the committed cubin.
   - `repro-v2.out`: 81b9826's `gemm.cu` (sha256 8aacfa93…) builds as `7fce087a…` twice, and that is the committed cubin.
   - **Control, `repro-control.out`:** the same source against v1's cubin (`ddedbafa…`) reads DIFFERS, rc 1.

**The SASS** (`cubin-compare-v2.txt`, addresses stripped). Against main 715b281's cubin:
- the eight f16 tc2 kernels changed (the store predicate);
- the other six main held are identical: the scalar, f16 tc, f32 tc, trsm and the bandwidth kernels.

Against v1, all 16 tc2 kernels changed and the same 6 are identical. v1's own comparison is `cubin-compare.txt`: there, all
14 of main's kernels were identical.

**Measured on the 3090** (`bench-3090-v2.out`; `bench_bf16.cu` v2 with the sm_86 build of 81b9826's gemm.cu, against the
box's cuBLAS, with the shapes above and the kernel choice the host makes for the 5090's 170 SMs). Over one pp256 of the
36-layer model:

| | total | throughput | error vs cuBLAS |
|---|---|---|---|
| scalar | 2,712.5 ms | 0.4-0.5 TF/s | ≤ 5.5e-6 |
| bf16 tc2 (v2) | 43.2 ms | 31-39 TF/s on the large shapes, 6.6 on K/V | ≤ 1.8e-5 |
| cuBLAS | 34.4 ms | | reference |

- That is **62.9x** over the scalar kernel, and 1.25x cuBLAS's time.
- **CONTROL, refused at every shape:** the f16 kernel run on the same bf16 bits (error ~8).
- `bench-3090.out` is v1's run (2dbc56f's bench_bf16.cu, v1's cubin): 62.1x.

**The dispatch control** is the null-device trace (`trace-before.txt`, `trace-after.txt`, v1). With the fix linked in, all
bf16 GEMMs are `tinyblas_gemm_bf16_tc2s_tn`; before it, every one is `tinyblas_gemm_f32 ... in=2`.

**The slot's own wrapper lines, dry** (`dry-runs.txt`: DRY=1, null device, 81b9826 relinked into llama.cpp ad6c668, against
the validated `build/bin`):
- **bench:** all 44 GEMM launches are `tinyblas_gemm_bf16_tc2s_tn`, against 44 `tinyblas_gemm_f32` for the validated set.
- **simple:**
  - The default 13-token prompt never reaches cuBLAS: 0 tinyblas launches. ggml runs so few tokens as matrix-vector
    products.
  - The pinned 106-token `correctness-prompt.txt` gives 11 `tinyblas_gemm_bf16_tc2s_tn` launches, against 11
    `tinyblas_gemm_f32`. The shim's `TINYCUDART_STATS=1` table names the kernel with its count: that row is the card run's
    witness.
- **The comparator:** `cmp_texts.sh` reads the two dry texts IDENTICAL (562 bytes; sha b60bb8fbfc6b; `dry-simple-*.txt`).
  **Control:** one byte changed past the prompt (`dry-simple-perturbed.txt`) reads DIFFER, rc 1.
- **ops:**
  - `MUL_MAT type_a=bf16`: 151 cases, with 330 tc2 launches and 1201 scalar launches. Which rule sends each scalar launch
    there is not traced.
  - `MUL_MAT type_a=f16`: 281 cases. This is how the changed f16 tc2 kernels get exercised. The bench and simple traces
    above launch none of them (flash attention is on).
  - **The f16 split between tc2s and the legacy f16_tc kernel is not stable across null runs.** 27 cases flip between two
    modes in either binary, the base 715b281 included. The f16 steps are therefore read case by case, never by kernel
    count.
- **K0, the control arm** (D's review): base 715b281 relinked exactly as V (`build/bin-k0`, sha256s in `dry-runs.txt`). It
  reads `libtinynv build 715b281`, with 44 and 11 `tinyblas_gemm_f32` launches, and its text is identical to V's.
