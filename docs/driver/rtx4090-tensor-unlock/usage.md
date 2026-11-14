# RTX 4090 tensor unlock usage

> This is an experimental hardware operation for one RTX 4090 on the existing
> TinyGPU transport. Use exclusive ownership, keep the enclosure independently
> powered, and never place the live process under a timeout that kills it.

## Build and prepare assets

From the repository root:

```sh
make -C tools/rtx4090-probe
make -C tools/rtx4090-probe check

python3 tools/rtx4090-unlock/prepare_bundle.py
python3 tools/rtx4090-unlock/validate_bundle.py \
  cuda-shim/libtinynv/third_party/unlock-610.43.03
```

The preparation step downloads the pinned public NVIDIA inputs once and stores a
gitignored bundle. No network access occurs during device open.

## Inspect the card

```sh
sh tools/rtx4090-live.sh preflight
```

Preflight is read-only. It checks the device ID, AD102 chip identity, reset
capability, GFW state, current selectors, WPR2, and transport accessibility. The
possible classifications are:

- `stock-cold`;
- `retained-unlock`;
- `warm`;
- `wrong-device`;
- `wrong-chip`; or
- `inaccessible`.

Run `sh tools/tinygpu-server.sh status` and `sh tools/gpu-lock.sh status` before a
live session. Do not continue if another client owns the card.

## Run the supervised unlock

The only verified stock-clearing operation is a physical enclosure power cycle.
Leave the enclosure off for at least 30 seconds, power it on, and confirm the
preflight result.

```sh
sh tools/rtx4090-live.sh unlock
```

The runner acquires the GPU lock, verifies the complete asset bundle, runs the
transaction, logs every stage, and keeps the card cold after a contained exit. A
successful result is `applied-and-verified`; an eligible card that already has the
override reports `verified-existing`.

## Verify the runtime handoff

Before the transaction, a stock baseline may be collected only when preflight
reports `stock-cold`:

```sh
sh tools/rtx4090-live.sh gemm stock
```

After a successful unlock:

```sh
sh tools/rtx4090-live.sh gemm unlocked
```

The handoff boots the unchanged 570.144 runtime, re-reads the unlock registers,
runs BF16 correctness cases against a CPU double reference, records the selected
GEMM shapes, and performs a cold unload. `gemm any` reports the current state
without requiring stock or unlocked.

Applications may use the integrated gate:

```sh
TINYNV_TENSOR_UNLOCK=1 <application>
```

Unset or `TINYNV_TENSOR_UNLOCK=0` leaves the path inert. Other values fail during
initialization.

## Restore stock state

`TINYNV_TENSOR_UNLOCK=0` prevents a new transaction; it does not clear an override
that is already present. FLR and unload retain selectors on the tested card.

To return to stock:

1. stop all GPU clients and leave the GPU lock free;
2. power off the enclosure for at least 30 seconds;
3. power it on; and
4. run preflight and require `stock-cold`.

Do not infer stock state from an environment variable.

## Recovery

If a failure occurs after firmware may have started and containment cannot be
proven, the process enters a recovery hold. It retains the socket, DMA mappings,
and GPU lock, performs no further work, and prints the last verified stage.

Power off the enclosure physically. Only then send the configured confirmation
input or signal. The process must not be killed; `SIGKILL`, a crash, server death,
or host failure bypasses the hold and the installed transport cannot retain the
mappings.

A cleanup operation that proves engine halt, WPR2 down, no pending transaction,
and bus mastering disabled does not need the recovery hold. It preserves the first
error, restores PCI state, releases buffers, and exits normally.

## Diagnostics

Relevant environment options:

| Variable | Purpose |
|---|---|
| `TINYNV_UNLOCK_BUNDLE` | Override the prepared bundle directory |
| `TINYNV_SOCKET_TIMEOUT_S` | Bound socket receive waits; default 30 seconds |
| `TINYNV_FAILPOINT` | Offline failure injection; remove it from live runs |

The live runner stores timestamps, stage output, sensor data, and workload logs
under its build directory. Keep failures and neutral results; do not benchmark a
stock card as unlocked or omit regressing shapes.

## Current evidence

The reviewed results are summarized in [validation.md](validation.md). At the time
of this branch, the transaction and handoff are proven on one RTX 4090, but
cross-power-cycle stock comparison, end-to-end BF16 prefill, and ten-cycle
lifecycle coverage remain incomplete.
