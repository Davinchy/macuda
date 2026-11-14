# GSP 610 runtime operating guide

This guide prepares and exercises the experimental 610.43.03 runtime for the
RTX 4090. The 570.144 runtime remains the default.

## Prerequisites

- macOS host with the current TinyGPU application, server, and DriverKit
  extension installed unchanged;
- an RTX 4090 with PCI identity `10de:2684`;
- the repository build prerequisites;
- network access for the first preparation, or the pinned inputs available
  locally; and
- an exclusive GPU lock before any hardware command.

Read [spec.md](spec.md) and [design.md](design.md) before running a live
session.

## Prepare firmware and headers

The scripts write only to gitignored paths under
`cuda-shim/libtinynv/third_party` and `cuda-shim/build`.

```sh
# Preserve the default 570 input tree.
sh cuda-shim/libtinynv/tools/fetch_firmware.sh

# Fetch and verify the 570 and 610 header/source trees.
sh tools/gsp610-runtime/fetch_inputs.sh

# Fetch the official 610.43.03 release artifacts, build the runtime bundle,
# validate it, and run the GSP preparation tests.
sh tools/gsp610-runtime/run_prepare.sh
```

`run_prepare.sh` produces:

```text
cuda-shim/build/gsp610-runtime/bundle/
```

The NVIDIA release package is large. For an offline build, pass the extracted
release members:

```sh
GSP610_RELEASE_GSP=/path/to/gsp_ga10x.bin \
GSP610_RELEASE_LICENSE=/path/to/LICENSE \
sh tools/gsp610-runtime/run_prepare.sh
```

Revalidate the resulting bundle at any time:

```sh
python3 tools/gsp610-runtime/validate_runtime.py
```

The validator exits nonzero on the first missing, malformed, truncated,
wrong-chip, or hash-mismatched asset.

## Run the offline tests

From the repository root:

```sh
python3 -m py_compile tools/gsp610-runtime/*.py
python3 -m unittest discover -s tools/gsp610-runtime -p 'test_*.py' -v

make -C cuda-shim/libtinynv \
  BUILD=build-gsp610 \
  BUNDLE610="$PWD/cuda-shim/build/gsp610-runtime/bundle" \
  test-runtime-profile
```

`test-runtime-profile` runs:

- selector parsing and dispatch;
- 570 and 610 boot metadata tests;
- 610 RPC/RM encoding and host-request tests;
- the complete mock GSP-RM lifecycle and failpoint matrix;
- separately compiled 570 and 610 header assertions; and
- real-bundle checks when the prepared assets are present.

An existing recorded boot can also exercise the 570 replay equivalence path:

```sh
make -C cuda-shim/libtinynv \
  BUILD=build-gsp610 \
  TRACE=/path/to/trace-without-extension \
  BUNDLE610="$PWD/cuda-shim/build/gsp610-runtime/bundle" \
  test-runtime-profile
```

## Build for a live session

The driver test must come from this same tree.

```sh
make -C cuda-shim/libtinynv BUILD=../build/gsp610-nv

cd cuda-shim
make BUILD=build/gsp610-shim CUBIN_ARCH=sm_86
```

Override the default locations with:

```sh
TINYNV_NV_BUILD=/absolute/path/to/libtinynv/build
TINYNV_SHIM_BUILD=/absolute/path/to/cuda-shim/build
```

## Select the runtime

The selector accepts two exact values:

```sh
# Default 570.144 behavior. An unset value and the explicit value are
# equivalent.
TINYNV_FW_DIR=/path/to/firmware ./your-app
TINYNV_GSP_RUNTIME=570.144 TINYNV_FW_DIR=/path/to/firmware ./your-app

# Experimental 610.43.03 on the RTX 4090.
TINYNV_GSP_RUNTIME=610.43.03 \
TINYNV_FW_DIR="$PWD/cuda-shim/build/gsp610-runtime/bundle" \
./your-app
```

Other values, partial versions, whitespace, unsupported devices, and
recorded/replay backends fail before firmware placement. There is no automatic
fallback after a failed 610 transaction.

## Supervised live session

The runner supports:

```sh
sh tools/gsp610-live.sh preflight
sh tools/gsp610-live.sh boot610
sh tools/gsp610-live.sh lifecycle610
sh tools/gsp610-live.sh lifecycle570
```

`preflight` is read-only: it validates the bundle and checks the system,
TinyGPU server, and lock status. The other modes take the local GPU lock,
ensure the server is healthy, and run one GPU process at a time.

`boot610` performs one controlled 610 open and cold unload. `lifecycle610` and
`lifecycle570` perform one init probe followed by five boot/cold-unload cycles
of the upstream `test_hw` program by default. Set `REPS` to change the cycle
count, or set `TINYNV_WORKLOAD` to another compatible test program.

The runner deliberately has no `timeout(1)` wrapper, failed-boot retry loop,
or automatic fallback. If a process may retain DMA ownership, leave the
process, socket, and mappings alive. Physical recovery requires an actual
owner power-off.

## Failure diagnostics

Every failure names the stage and preserves the first concrete error.

- `TINYNV_DUMP_GSP_LOGS=1` requests the detailed GSP log dump after a
  successful open.
- A silent 610 boot always dumps mailbox, queue, carveout, log-ring, and
  firmware-image state.
- `TINYNV_RPC_TRACE=1` reports dropped or mismatched status records.
- `TINYNV_FAILPOINT=<stage>` exists for offline tests. Do not set it for a
  live session.

The test harness can exit nonzero solely because its bus-mastering evidence
check expected mastering to start enabled. The runner accepts that exit code
only when the log proves both `GSP-RM is up` and a cold unload.

## Restore 570

After a 610 process has exited or been cleanly contained, start a separate
default-profile process with the 570 firmware root:

```sh
env -u TINYNV_GSP_RUNTIME \
  TINYNV_FW_DIR="$PWD/cuda-shim/libtinynv/third_party/firmware" \
  ./your-app
```

Never swap firmware under active queues or mappings. The branch does not
provide crash-safe retention across `SIGKILL`, server death, or host failure.

## Current support boundary

Live testing on one RTX 4090 established that the 610 runtime reaches GSP-RM,
creates its resource-management objects, runs BF16 computation, and performs
clean cold unloads. The public lifecycle test reproduces the boot and unload
path. Other GPUs, broader CUDA applications, crash-safe recovery, default
promotion, and a performance advantage remain outside this branch's validated
scope.
