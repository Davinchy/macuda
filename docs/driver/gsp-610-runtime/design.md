# GSP 610 runtime design

## Runtime profiles

`src/runtime_profile.{h,c}` defines two immutable profiles.

`tinynv_runtime_570` is the default. It derives its firmware directory and
layout from the existing 570 definitions and keeps the existing per-chip hash
tables.

`tinynv_runtime_610` pins `610.43.03`, the AD102 directory
`nvidia/ad102/gsp-610.43.03`, the RTX 4090 allowlist, and the larger firmware
reservation. Selection happens before early device setup. An invalid profile
fails before a BAR is mapped, and 610 requires a live device so a recorded 570
stream cannot be consumed by the new path.

The selected profile is stored outside the boot state cleared by retry. Boot,
reset, retry, and unload use the same profile; close never reselects.

## Asset preparation

`TINYNV_FW_DIR` remains the common root. The 610 subtree contains eight
validated assets:

| Asset | Source |
|---|---|
| `gsp-610.43.03.bin` | official NVIDIA 610.43.03 Linux release |
| `bootloader-610.43.03.bin` | AD102 GSP boot BINDATA |
| `bootloader_desc-610.43.03.bin` | AD102 BINDATA descriptor |
| `booter_load-610.43.03.bin` | AD102 load booter BINDATA |
| `booter_unload-610.43.03.bin` | AD102 unload booter BINDATA |
| `scrubber-610.43.03.bin` | AD10X secure scrubber BINDATA |
| `generic_bl-610.43.03.bin` | generic bootloader ucode |
| `generic_bl_desc-610.43.03.bin` | generic bootloader descriptor |

`tools/gsp610-runtime/runtime_assets.py` extracts and validates the vendor
BINDATA archives, assembles BTL1 v1 booter containers, checks fused-signature
geometry, and writes a manifest without timestamps. The C pin table in
`src/fw610.c` is cross-checked against that manifest by
`test_lifecycle610`.

The whole lifecycle bundle is validated before firmware placement. Missing,
truncated, renamed, wrong-chip, or hash-mismatched assets are hard failures.

## Firmware layout

The 610 GSP `.fwimage` is 84,226,048 bytes. The 610 profile raises the image
bound to 96 MiB while retaining the existing 256 MiB reservation ceiling.
Static assertions check the image bound, heap, non-WPR heap, WPR placement,
and the worst-case carveout.

The 610 GSP ELF is parsed with the same section mechanism as 570. Signature
and image offsets are read from the ELF instead of copied from the smaller
570 image.

## ABI boundary

The detailed size and offset evidence is in [abi-audit.md](abi-audit.md).
Changed interfaces use dedicated encoders and parsers:

- `boot610.{h,c}` handles boot metadata, startup arguments, BTL1 assets, and
  carveout placement;
- `rpc610.{h,c}` handles the 610 MCTP/NVDM queue envelope, 40-byte control
  header, variable element count, continuation records, 952-byte system-info
  payload, 376-byte channel allocation, 56-byte VA-space allocation, RUSD
  offsets, and GR-info stride; and
- `gsp610_host.{h,c}` handles the load-exec generic-bootloader and
  high-secure-binary requests that replace the removed CPU register
  sequencer.

Unchanged structures are shared only where `test_headers.c`,
`test_headers_610.c`, or the RPC/lifecycle tests assert their layout.

## Lifecycle

The 610 open path:

1. parses and freezes the runtime profile;
2. reads the device identity without recording a 570 operation;
3. validates the complete 610 asset bundle;
4. maps and resets the device;
5. runs the scrubber and signed booter chain;
6. starts libos and GSP-RM with the 610 argument layouts;
7. creates the RM object tree, channels, queues, and memory mappings; and
8. checks firmware carveout and tensor state before workload submission.

Unload sends the 570-compatible guest-driver notice, waits for the processor
to park, and tears down the firmware region. A zero-length reply is treated as
complete rather than as the first fragment of a continuation record.

On failure, cleanup and retry retain the selected profile. A separate process
must perform 570 rollback after the 610 process exits and has left the card
cold.

## Diagnostics

The implementation keeps first-failure reporting and adds:

- `TINYNV_DUMP_GSP_LOGS=1` for an explicit GSP log dump;
- an automatic silent-boot diagnostic;
- `TINYNV_RPC_TRACE=1` for dropped or mismatched status records; and
- `TINYNV_FAILPOINT=<stage>` for offline failure injection.

Failure injection is disabled in normal operation and the live runner removes
it from the environment.

## Compatibility

The 570.144 path remains the default and its recorded operations are
unchanged. The branch does not alter CUDA kernels, cuBLAS dispatch, TinyGPU, or
the device extension. Runtime 610 and tensor configuration are independent
settings.
