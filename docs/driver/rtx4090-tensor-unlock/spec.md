# RTX 4090 tensor unlock specification

## Status

Experimental and opt-in. The implementation targets one desktop NVIDIA RTX 4090
(`10de:2684`, AD102) reached through the existing TinyGPU transport. The installed
TinyGPU application, server, and signed DriverKit extension remain unchanged.

The unlock runs in macuda userspace before the normal runtime firmware boot. It is
default-off, depends on device-specific firmware assets, and is not a supported
CUDA precision mode.

## Objective

Enable the AD102 tensor speed override, then verify that macuda's existing BF16
matrix-multiply path still produces correct results after the normal runtime boot.

The feature changes no CUDA API, kernel arithmetic, tolerance, clock, power limit,
VBIOS, or normal runtime firmware. BF16 input and FP32 accumulation remain the
primary validation path.

## Interface

`TINYNV_TENSOR_UNLOCK` accepts only:

| Value | Behavior |
|---|---|
| unset or `0` | Skip all unlock reads, allocations, and device writes |
| `1` | Require an eligible card and a verified unlock before application work starts |

Any other non-empty value is rejected. Turning the option off does not restore
stock hardware state because the powered card may retain the override.

The observable result is one of:

- `verified-existing`: a retained unlock passed the effective-state gates;
- `applied-and-verified`: this process applied and verified the transaction;
- `unsupported`: the device or transport is outside the supported scope;
- `failed-preflight`: no firmware launch was attempted;
- `failed-after-start`: firmware may have started and cleanup had to be contained.

The runtime rechecks the override after any initialization reset and after the
normal runtime firmware boot, before creating application queues.

## Safety and lifecycle requirements

- Accept only the desktop RTX 4090 device ID and a matching AD102 chip identity.
- Require exclusive GPU ownership and a cold, inspectable starting state.
- Validate every asset by size and SHA-256 before publishing any device address.
- Check that transaction buffers have aligned, contiguous DART IOVAs below 4 GiB.
- Verify Falcon completion, privileged-write permissions, selectors, and effective
  register state. A mailbox marker alone is not success.
- Treat reset RPC success as insufficient; verify identity, BAR state, firmware
  state, WPR2 state, and selector retention after every reset.
- Preserve the first failure and restore the stock signature span on every exit.
- Release device-visible memory only after firmware DMA is proven stopped.
- If containment cannot be proven, keep the process, socket, mappings, and GPU lock
  alive and request physical enclosure power-off before disconnecting.

Catchable termination enters the recovery hold. Signal handlers do not perform
reentrant socket operations. The hold is best effort: `SIGKILL`, process crash,
server death, or host failure can still force an unsafe disconnect. Physical power
cycling is the only verified way to return the card to stock.

## Validation criteria

Offline validation must cover:

- strict option parsing and inert default behavior;
- wrong-device, wrong-chip, inaccessible-register, and malformed-asset refusal;
- asset pin drift, transaction stage failures, and reset retention loss;
- socket timeout and interrupted-transfer behavior;
- recovery-hold signaling and buffer-retention ordering;
- BF16 correctness and the normal runtime handoff.

Live validation uses one GPU process at a time and must record:

- the standalone transaction result and all stage gates;
- override state after the runtime firmware boot and after representative work;
- BF16 output checks against a CPU reference under unchanged tolerances;
- a clean cold unload with no new DART or MMU fault; and
- stock and unlocked timing on the same kernel and input shapes.

The current validation record and remaining gaps are in [validation.md](validation.md).

## Provenance

The asset pipeline pins the independently exercised
`JustAResearcher/gpu-unlock-ada` revision
`8855e7bca8ff142204e02301331fd56b26699819`, NVIDIA 610.43.03 inputs, vendored
license texts, and every generated asset hash. The 610.43.03 unlock bundle is
separate from macuda's 570.144 runtime bundle. See [assets.md](assets.md) and
[inputs.md](inputs.md).

The normal runtime is reached through the transport constraints in
[transport.md](transport.md).
