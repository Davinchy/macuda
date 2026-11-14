# rtx4090-probe

Standalone, supervised AD102 tensor-unlock transaction for one desktop NVIDIA
RTX 4090 (`10de:2684`). The feature is experimental and opt-in. It changes only
macuda userspace code and uses the existing TinyGPU transport unchanged.

Read the [specification](../../docs/driver/rtx4090-tensor-unlock/spec.md),
[usage and recovery guide](../../docs/driver/rtx4090-tensor-unlock/usage.md),
and [validation record](../../docs/driver/rtx4090-tensor-unlock/validation.md)
before attempting the live transaction.

## Operation

- `preflight` reads PCI identity, AD102 chip architecture, reset capability,
  GFW state, FEAT/readout/SS0/SS1, and WPR2. It performs no writes, DMA, or reset.
- `unlock` runs the pinned transaction: cold-state gates, validated DMA,
  FWSEC-FRTS from the board ROM, the V67 privilege payload, SEC2 completion and
  permission checks, selector writes, FLR, and retention readback.

Every asset is checked against its manifest size and SHA-256 before any device
address is published. The transaction restores the original signature span and
preserves the first failure through cleanup.

## Build

```sh
make -C tools/rtx4090-probe
make -C tools/rtx4090-probe check
```

Prepare and validate the gitignored firmware bundle once:

```sh
python3 tools/rtx4090-unlock/prepare_bundle.py
python3 tools/rtx4090-unlock/validate_bundle.py \
  cuda-shim/libtinynv/third_party/unlock-610.43.03
```

The probe searches `--bundle`, `$TINYNV_UNLOCK_BUNDLE`, then the prepared bundle
under `cuda-shim/libtinynv/third_party/unlock-610.43.03`.

## Live protocol

`tools/rtx4090-live.sh` builds and validates the probe, runs the repository
preflight, ensures the TinyGPU server is available, checks the GPU lock, and
records a transcript. It never wraps the probe in a killing timeout.

```sh
# Read-only.
sh tools/rtx4090-live.sh preflight

# Power-cycle the enclosure before this command.
sh tools/rtx4090-live.sh unlock

# Stock baseline before the transaction, then post-unlock handoff.
sh tools/rtx4090-live.sh gemm stock
sh tools/rtx4090-live.sh gemm unlocked

# Measure the current state without requiring stock or unlocked.
sh tools/rtx4090-live.sh gemm any
```

Only a physical enclosure power cycle is verified to clear the override in this
implementation. FLR and a cold unload retain it. Keep the enclosure on an
independent power supply and do not launch a second GPU client.

The PTY helper is available for supervised automation:

```sh
python3 tools/rtx4090-live-drive.py
```

It answers the initial supervision prompt after an operator has verified a
stock-cold preflight. It does not confirm power-off recovery. During a recovery
hold, power off the enclosure and then either create the file named by
`$TINYNV_TENSOR_UNLOCK_CONFIRM_OFF_FILE` (default
`/tmp/macuda-tensor-unlock-confirm-off`) or send `SIGUSR1` to the probe.

## Recovery

If a failure occurs after firmware may have started and containment cannot be
proven, the probe keeps the process, socket, DMA mappings, and lock alive while
it waits for power-off confirmation. It performs no further GPU work.

Do not kill the process. `SIGKILL`, a client crash, server loss, or host failure
bypasses the recovery hold, and the installed transport cannot retain mappings.

## Tests

The hardware-free suites cover transaction gates, asset parsing and drift,
socket timeouts and `EINTR`, recovery signaling, lock ownership, runtime option
parsing, postboot selector loss, and continuation of DMA cleanup. These tests
prove control flow and ordering, not firmware completion on hardware.

```sh
make -C tools/rtx4090-probe check
python3 -m unittest discover -s tools/rtx4090-unlock -p 'test_*.py' -v

make -C cuda-shim/libtinynv BUILD=build-codex-rtx \
  build/test_tensor_unlock \
  build/test_socket_timeout \
  build/mock_ad102_server \
  build/runtime_unlock_harness
```

## Exit codes

| Code | Meaning |
|---|---|
| `0` | `applied-and-verified` or `verified-existing` |
| `1` | contained failure; buffers released and PCI state restored |
| `2` | usage, lock, or socket setup failure |
| `3` | recovery completed only after physical power-off confirmation |
