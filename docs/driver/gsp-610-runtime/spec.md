# GSP 610 runtime specification

## Status

This branch adds an experimental, opt-in GSP 610 runtime for the RTX 4090
(`10de:2684`, AD102). The existing 570.144 runtime remains the default and keeps
its current firmware paths, hashes, lifecycle, and recorded behavior.

The port uses NVIDIA 610.43.03 userspace ABI definitions and vendor firmware
assets. It does not replace the installed TinyGPU application, server, or
DriverKit extension.

## Scope

The runtime must:

- select 610 only when `TINYNV_GSP_RUNTIME=610.43.03` is set explicitly;
- accept only the exact versions `570.144` and `610.43.03`;
- restrict the 610 profile to a live RTX 4090;
- keep selection frozen across boot, retry, unload, and close;
- validate every required boot and unload asset before placing firmware;
- reject mixed-version or unsupported state before device work;
- preserve the complete 570.144 path when 610 is not selected;
- report the failing stage and preserve the first concrete error;
- perform a cold unload after a successful session; and
- avoid automatic fallback to 570 after a failed 610 transaction.

Runtime selection and tensor configuration are separate settings. Selecting
610 does not alter CUDA arithmetic or the compute-library implementation, and this branch does not
change CUDA, cuBLAS, or kernel arithmetic.

The initial support boundary is one RTX 4090 behind the current TinyGPU
transport. Other Ada, Hopper, and Blackwell devices keep the existing 570
policy.

## Firmware inputs

The 610 runtime is derived from immutable inputs:

| Input | Pin |
|---|---|
| NVIDIA kernel module source | `610.43.03`, SHA-256 `7e118923c7a23edc36114d63273a46e3e04e9af98695a42203e7ac2dfe9fc1dc` |
| NVIDIA Linux driver release | `NVIDIA-Linux-x86_64-610.43.03.run`, SHA-256 `45e2d4c134a23c35e50f253a4aa63e7e5e8d17e3d185d4a07c8a58e9612ed392` |
| GSP ELF | `gsp_ga10x.bin`, SHA-256 `572373620a37418f24dc16b5031c39338778c3257e48e8408de9a57291b24f3a` |
| Runtime manifest | produced deterministically by `tools/gsp610-runtime/run_prepare.sh` |

Booter load, booter unload, scrubber, GSP boot image, descriptor, and
generic-bootloader assets are extracted from the pinned NVIDIA source. The
board VBIOS remains responsible for FWSEC-FRTS/SB data.

## Validation requirements

Offline validation must cover:

- strict selector parsing and frozen profile lifetime;
- firmware path, container, hash, size, alignment, and wrong-chip checks;
- 570 and 610 ABI assertions from separate translation units;
- RPC framing, continuation, reply, event, and allocation payload tests;
- boot metadata, startup arguments, carveout, and load-exec host requests;
- per-stage failure cleanup and retry with the selected profile retained;
- a complete boot/object-tree lifecycle against a mock GSP-RM peer; and
- the existing 570 header, firmware, and memory tests.

Hardware validation must use an exclusive GPU lock and one process at a time.
The current implementation has completed five known-answer BF16
open/workload/cold-unload cycles under 610.43.03 and five equivalent cycles
under 570.144 with the same test binary.

### Remaining limits

This does not claim:

- support for GPUs other than the allowlisted RTX 4090;
- general CUDA compatibility beyond the tested shim workload;
- independent FP16 and FP32 hardware validation;
- recovery guarantees after `SIGKILL`, client death, server death, or host
  failure;
- a performance advantage over 570.144; or
- default promotion of 610.

## Related documents

- [Design](design.md)
- [Implementation plan](plan.md)
- [Operating guide](usage.md)
- [ABI audit](abi-audit.md)
- [Validation record](validation.md)
- [Immutable inputs](inputs.json)
