# RTX 4090 tensor unlock design

## Components

| Component | Responsibility |
|---|---|
| `src/tensor_unlock.{c,h}` | Eligibility, transaction state, register gates, recovery hold, post-boot verification |
| `src/unlock610/` | Portable parsers and asset preparation derived from the pinned upstream revision |
| `tools/rtx4090-unlock/` | Deterministic asset acquisition, generation, validation, and manifest |
| `tools/rtx4090-probe/` | Standalone supervised transaction executable |
| `tools/rtx4090-live.sh` | Locked preflight, unlock, handoff, and benchmark workflow |
| `libtinynv` runtime gate | Preboot opt-in transaction and postboot state verification |

The runtime gate is the integration point used by normal applications. The
standalone probe drives the same transaction and recovery code so the hardware
operation can be exercised before enabling the library-wide gate.

## Startup ordering

The transaction must complete before macuda creates persistent MMU, allocator, and
queue state:

1. parse and freeze `TINYNV_TENSOR_UNLOCK`;
2. open the TinyGPU transport and claim the GPU lock;
3. inspect PCI identity, chip identity, reset capability, WPR2, and current
   override state;
4. validate the asset bundle and all parsed ranges;
5. allocate and validate transaction DMA buffers;
6. run FWSEC-FRTS, the signed payload, and the privileged writes;
7. set and verify the tensor selectors;
8. perform and verify the reset;
9. start the normal 570.144 runtime firmware from clean state; and
10. verify the override again before application work.

No MMU or allocator object may survive step 8.

## State machine

```text
OFF
  -> REQUESTED
  -> ASSETS_VALIDATED
  -> DEVICE_CLAIMED
  -> COLD_STATE_VERIFIED
  -> TRANSACTION_APPLIED
  -> SELECTORS_VERIFIED
  -> RESET_CONTAINED
  -> RUNTIME_BOOTED
  -> POSTBOOT_VERIFIED
  -> READY
```

Every failure records the first concrete stage. A partially launched firmware
stage never retries in the same open. Normal runtime boot retries are allowed only
after the configured reset path and selector retention have been verified.

## Register and asset gates

The AD102 register manifest includes the FEAT permission register, effective
readout, SS0, SS1, WPR/window registers, SEC2 completion state, reset state, ACR
scratch, and chip identity. All-ones, `0xbadf____`, and other invalid read patterns
fail closed.

The asset bundle contains:

- the signed load booter;
- the GSP boot image and version-5 descriptor;
- the four-PLM V67 privilege payload;
- the 610.43.03 GSP ELF used for its AD102 signature section; and
- the vendor license and complete hash manifest.

The loader re-verifies sizes and hashes at runtime. It parses BTL1 geometry,
descriptor fields, ELF sections, and radix layout after verification.

## Memory model

The TinyGPU client distinguishes:

- CPU-mapped shared memory;
- DART host IOVAs used by device DMA; and
- GPU physical VRAM addresses.

Every published host IOVA must be below 4 GiB, page-aligned, and contiguous for
the length consumed by Falcon DMA. The client parses and validates the transport's
segment table before overwriting the first mapped page. A release fence separates
CPU stores from a subsequent device doorbell or BAR write.

The transaction uses direct transport allocations, separate from macuda's normal
boot allocator. Buffers become device-owned when their addresses are published
and stay mapped until the firmware engine is halted or the enclosure is powered
off.

## Reset and recovery

The transport reset maps to a PCI function reset and restores configuration space.
The reset RPC result is not trusted. The implementation verifies the same state
used by the existing cold-reset path, plus the unlock selectors.

For a contained failure, cleanup order is:

1. stop new GPU work;
2. clear bus mastering and read it back;
3. reset and prove SEC2/GSP halt, WPR2 down, and no pending transaction;
4. restore PCI command state; and
5. release buffers and the socket.

If any halt gate is unavailable, the probe enters a recovery hold instead. The hold
keeps the transport alive, reports the last verified stage, and waits for physical
power-off confirmation.

## Runtime handoff

When the unlock is already retained, the preboot gate performs read-only
verification and skips firmware loading. After the normal runtime boot, macuda
reads FEAT, effective readout, SS0, and SS1 through the live BAR0 window. A loss of
state fails the open before application work begins.

## Known limits

- The installed transport does not retain mappings after unexpected process death.
- Sleep/resume, hot unplug, concurrent clients, and automatic recovery are outside
  this implementation.
- The override survives FLR and cold unload; only physical power removal reliably
  clears it.
- The target remains one RTX 4090 on the current TinyGPU transport.
