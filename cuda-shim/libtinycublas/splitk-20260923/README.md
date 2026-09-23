# Split-K for the small-N decode GEMMs (B, 2026-09-23): the evidence on this branch

Design and registered bands: root `docs/review/20260923-smalln-design-B.md`. **Default OFF.** `TINYCUBLAS_SPLITK=1`
turns it on. With it unset, every GEMM takes exactly the kernel it took before, and the SASS below shows that kernel is
unchanged, so the unset run is the reference arm. No card timing has been taken. The card step is requested separately,
as a prereg.

**What changed**
- `gemm.cu`: the `tinyblas_gemm_tc2` template gains `bool SPLIT = false` and three arguments (`ws`, `cnt`, `S`).
  - Eight new entry points `tinyblas_gemm_{f16,bf16}_tc2k_{nn,nt,tn,tt}` carry tc2s's 64x64 tile with SPLIT on.
  - The last block of each tile sums the S partials in slice order, applies alpha/beta once, and resets its counter.
- `cublas.c`:
  - `tinycublas_splitk_plan(m,n,k,batch)` is a pure function. It splits only when gy == 1 and gx < 170 (the S loop enforces the second), into the
    smallest power of two ≥ 170/gx, capped at 16, with ≥ 4 k-tiles per slice and within the workspace bound.
  - An 8 MiB workspace and 16384 counters per handle, made at `cublasCreate` (never inside a capture), and only when the
    knob is on.
  - The kernel cache grows from 24 to 32 entries, because there are now 27 names and a full cache returns NULL.
  - The param blob's offsets are pinned by `_Static_assert` to the cubin's `EIATTR_KPARAM_INFO`: ws at 0x58, cnt at
    0x60, S at 0x68, 108 bytes.
- `gemm.cubin`: sm_120, nvcc 13.0 on the 3090 box, built twice from `gemm.cu` sha256 `bbb21c8d…`, both builds
  `4ee3b384…` (`build-gemm-cubin.sh`).

**Evidence, all card-free. The 3090 box's clock is CST = Mac + 1 h.**
- `run-3090-2.out`: `test_splitk.cu` on the RTX 3090 (sm_86), from the `gemm.cu` that is committed here.
  - **Reference: 30 PASS, 0 FAIL.** Six cases: the 3B decode k/v, q/o and down shapes, tails (m 200, n 20, k 1000),
    an empty last slice, and batch 3. Five rows each:
    - EXACT: `==` against the ordered host sum of tc2s run on each slice;
    - AGAIN: the same GEMM on the same counters, with C poisoned in between;
    - COUNTERS: all zero afterwards;
    - REF f32 and REF bf16: against doubles, with the contract test's tolerances.
  - **Each of three mutants makes it fail, and each sed changed exactly one line.**
    - drop_slice: EXACT and REF fail.
    - reverse_order: EXACT fails on 5941/8192 and 40149/65536. REF is not expected to catch an order change.
    - no_reset: AGAIN comes out NaN and COUNTERS reads nonzero.
  - A run-1 from the pre-fix source (see the next item) also passed. It is not kept, because its source is not the
    committed one.
- `sass-compare.out`: per-function sm_120 SASS, addresses stripped, main `0438864`'s `gemm.cu` against this one.
  **0 of 22 functions differ**, and the only new functions are the 8 tc2k.
  - **The comparison was seen to fail first:** the first version of this change read 18 DIFF (every tc2 and tc2s). The
    causes were a signed `bz` and an added branch before the first tile load. Both were fixed to that reading. The
    18-DIFF output was seen in session and not banked.
- `shim-tests.out`: link_smoke, cublas_contract (13/13), stats_grid_check (7/7) and splitk_check (11/11), after a
  clean rebuild. **libtinynv's own suite was not run.** This worktree has no `third_party` (an untracked symlink), and
  this change touches no libtinynv file.
- **Plan mutants, rebuilt from clean and hashed, because an in-place rebuild changed nothing the first time** (the
  same-second make trap: three "readings" were one binary):
  - moving the S loop's target 170→171 fails the "gx 170 does not" row;
  - letting gy 2 split fails the "n 65" row;
  - the restored source reads 13/13 again.
  - A first mutant on the `gx >= 170` guard did NOT fail, and it was right not to. That guard was unreachable: the S
    loop already leaves S at 1 for gx ≥ 170. It is removed.
- splitk_check's own first run failed one row on a **checker** defect: its count pattern also matched the null device's
  "would launch" line (6 ≠ 3). Anchored to `[trace] launch`.
- `test/splitk_dispatch.c` and `test/splitk_check.sh` (in `make test`), on the null device:
  - 13 plan rows at the design's table and at every boundary: gx 169/170, n 64/65, k 2047, k 255, and the
    workspace at batch 2/3.
  - The trace, read in both arms: ON puts k/v at (4,1,16) and q/o and down at (32,1,8), with gate/up (172,1,1) and the
    diffusion mix (20,4,1) unchanged. OFF has no tc2k anywhere.

**Not shown here**
- Any timing.
- Behaviour under a graph capture (the workspace is made before any capture, but that has not been exercised).
- Two streams sharing one handle while split GEMMs are in flight on both. They would share the workspace. ggml keeps a
  handle per stream.
