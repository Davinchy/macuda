# RTX 4090 tensor unlock validation

## Offline

The implementation is covered by:

- malformed-input tests for the bundle manifest and every generated asset;
- C asset-pin checks against `tools/rtx4090-unlock/manifest.json`;
- fake-AD102 transaction tests for every register and completion gate;
- socket timeout and interrupted-transfer tests;
- recovery-hold tests for termination signals, power-off confirmation, and
  bounded cleanup;
- runtime-gate tests for inert default behavior, strict option parsing, and
  postboot state loss; and
- BF16 GEMM correctness tests using a CPU double-precision reference.

These tests prove control flow, parsing, and cleanup ordering. They do not prove
that firmware completed a hardware operation.

## Hardware

The first supervised transaction ran on one desktop RTX 4090 (`10de:2684`,
AD102 chip architecture `0x192`) through the existing TinyGPU transport.

| Item | Result |
|---|---|
| Transaction | `applied-and-verified` in 1.89 s |
| Stage gates | FWSEC-FRTS, SEC2 completion, privileged permissions, selectors, effective readout before and after reset |
| DMA | Seven checked buffers, all IOVAs below 4 GiB, contiguous and aligned; seven freed |
| Runtime handoff | Override retained across FLR and the unchanged 570.144 runtime boot |
| BF16 correctness | 10/10 cases in the stock session and both initial unlocked sessions |
| Later correctness | Three multi-shape unlocked sessions passed 17/17 cases each |
| Lifecycle | Seven open/workload/close cycles, each ending with a verified cold unload |
| Faults | No new DART or MMU fault and no device re-enumeration |

The stock 2048-square TN BF16/FP32 GEMM medians were 0.335 ms and 0.357 ms.
Unlocked medians were 0.330 ms and 0.327 ms. These samples are from separate
sessions and do not establish an application speedup.

An unlocked-only shape suite recorded:

| Shape | Median samples |
|---|---|
| 2048×2048×2048 | 0.311–0.328 ms |
| 2048×256×2048 | 0.120–0.121 ms |
| 256×256×2048 | 0.078–0.080 ms |
| 11008×256×2048 | 0.248–0.273 ms |
| 2048×256×11008 | 0.241–0.265 ms |
| 2048×1×2048 | 0.099 ms |
| 2048×300×2048 | 0.110–0.111 ms in the final two sessions |

Every numerical check remained within the existing BF16 tolerances (`2e-3` for
FP32 output and `1e-2` for BF16 output), with non-finite values rejected.

## Gaps

- The new-shape suite still needs a matched stock session after a physical power
  cycle.
- The 2048-square stock/unlocked comparison is not interleaved across power
  cycles.
- Seven of ten lifecycle cycles have been recorded.
- End-to-end BF16 prefill and a register-resident instruction probe have not run.
- Recovery from a deliberately injected live firmware failure has not been
  exercised.
- Unexpected process death still bypasses the recovery hold and the installed
  transport cannot retain DMA mappings.

These gaps keep the feature experimental and off by default.

## Reproduce

```sh
make -C tools/rtx4090-probe
make -C tools/rtx4090-probe check

make -C cuda-shim/libtinynv BUILD=build-codex-rtx \
  build/test_tensor_unlock \
  build/test_socket_timeout \
  build/mock_ad102_server \
  build/runtime_unlock_harness

python3 cuda-shim/libtinynv/test/test_probe_recovery.py \
  tools/rtx4090-probe/build/rtx4090-probe \
  cuda-shim/libtinynv/build-codex-rtx/mock_ad102_server \
  cuda-shim/libtinynv/third_party/unlock-610.43.03

python3 cuda-shim/libtinynv/test/test_runtime_unlock.py \
  cuda-shim/libtinynv/build-codex-rtx/runtime_unlock_harness \
  cuda-shim/libtinynv/build-codex-rtx/mock_ad102_server \
  cuda-shim/libtinynv/third_party/unlock-610.43.03
```

Live commands and recovery requirements are in [usage.md](usage.md).
