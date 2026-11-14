# GSP 610 runtime implementation plan

## Objective

Add a selectable 610.43.03 runtime for RTX 4090 userspace bring-up while
preserving 570.144 as the default. The work must be independently buildable
from the public NVIDIA 610.43.03 source and release packages.

## Work items

### 1. Audit the ABI and pin inputs

Inventory the structures, constants, events, controls, and host requests used
by libtinynv. Compare the existing 570 source pin with NVIDIA 610.43.03 and
record every same, changed, removed, and added boundary.

Status: complete. See [abi-audit.md](abi-audit.md) and
[inputs.json](inputs.json).

### 2. Prepare the runtime bundle

Extract the AD102 boot, unload, scrubber, bootloader, and generic-bootloader
assets from the pinned source. Obtain the GSP ELF from the official NVIDIA
release. Validate geometry and hashes and emit a deterministic manifest.

Status: complete. See `tools/gsp610-runtime/run_prepare.sh`.

### 3. Add an immutable runtime selector

Implement exact parsing, the RTX 4090 allowlist, live-device enforcement, and
profile retention across retry and unload. Keep unset and explicit 570
selection equivalent.

Status: complete. See `src/runtime_profile.{h,c}`.

### 4. Add 610 boot and RPC adapters

Implement the 610 metadata, startup arguments, carveout, queue envelope, RPC
continuations, RM controls, channel and VA-space allocation, RUSD mapping,
GR-info stride, and load-exec host requests.

Status: complete. See `src/boot610.{h,c}`, `src/rpc610.{h,c}`, and
`src/gsp610_host.{h,c}`.

### 5. Integrate the runtime lifecycle

Wire the selected asset set through open, reset, retry, containment, and
unload. Add a complete mock-peer lifecycle, failpoint coverage, profile
retention checks, and separate-process 570 rollback coverage.

Status: complete. See `test/test_lifecycle610.c`.

### 6. Validate on the target GPU

Build the driver and test workload from the same tree. Run known-answer BF16
workloads with one process at a time, verify tensor state and carveout before
and after work, and require a cold unload after each cycle.

Status: complete for the maintainer's RTX 4090. Five cycles passed under both
570.144 and 610.43.03 with the same binary. See
[validation.md](validation.md).

### 7. Deliver operating and recovery instructions

Document immutable inputs, offline preparation, selector behavior, build
commands, the live-session protocol, diagnostics, and rollback.

Status: complete. See [usage.md](usage.md).

## Acceptance criteria

Before merge:

```sh
python3 -m py_compile tools/gsp610-runtime/*.py
python3 -m unittest discover -s tools/gsp610-runtime -p 'test_*.py' -v
make -C cuda-shim/libtinynv BUILD=build-gsp610 \
  BUNDLE610="$PWD/cuda-shim/build/gsp610-runtime/bundle" \
  test-runtime-profile
git diff --check upstream/main...HEAD
```

must pass with no private worktree paths or machine-specific state.
on another feature branch.

## Deliberate exclusions

This branch does not add:

- default 610 selection;
- other GPU profiles;
- an automatic fallback after a failed 610 boot;
- crash-safe retention beyond the existing driver behavior; or
- changes to CUDA arithmetic or the compute-library implementation.
