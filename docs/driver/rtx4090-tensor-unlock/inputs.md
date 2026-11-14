# RTX 4090 tensor unlock input provenance

The unlock bundle is derived from immutable NVIDIA 610.43.03 inputs and the
payload layout at `JustAResearcher/gpu-unlock-ada@8855e7bca8ff142204e02301331fd56b26699819`.

## Runtime baseline

The normal macuda runtime remains the existing 570.144 firmware set. The unlock
does not replace or modify those files. The AD102 runtime images remain pinned by
`cuda-shim/libtinynv/tools/fetch_firmware.sh`:

| Runtime file | SHA-256 |
|---|---|
| `ad102/booter_load-570.144.bin` | `8b293e19b637c5e22c87a2428d1c71bb13e0904e8a88ac6b3c6c1f2679c6e37a` |
| `ad102/booter_unload-570.144.bin` | `975b85a14ded8e430d30f000c3c1afdd55c15dee04f35ff9dfd876acd7e67186` |
| `ad102/bootloader-570.144.bin` | `65ab2e6b6e0fca95365c4deac79a34582abcfeb15b6ae234138f22e7183118a8` |
| `ad102/scrubber-570.144.bin` | `c397358e5c4258dab070966589c7b172eb00771fd15d0bb74b8e22f327c1909b` |
| `ga102/gsp/gsp-570.144.bin` | `a8c3ebeed280323aedb51c061f321e73379cce7a9ae643a33dd03915df027f7f` |

The firmware source commit and all chip-specific runtime pins remain defined in
the existing fetch script and `SHA256SUMS`.

## Unlock source

| Repository | Revision | Role |
|---|---|---|
| `JustAResearcher/gpu-unlock-ada` | `8855e7bca8ff142204e02301331fd56b26699819` | Independently exercised AD102 transaction, payload layout, and boot assets |
| `NVIDIA/open-gpu-kernel-modules` | `81fe4fb417c8ac3b9bdcc1d56827d116743892a5` | Existing 570 header baseline |
| NVIDIA 610.43.03 source archive | release `610.43.03` | 610 boot, unload, scrubber, bootloader, and header inputs |

The selected payload produces the measured four-PLM V67 image:

| Generated value | Pin |
|---|---|
| Size | 64,000 bytes |
| SHA-256 | `44c36b2d6b3424e53e18f3e55db85c833655de071ce66ce42749588a4a7dc8cb` |

The pinned source's AD102 booter geometry is:

- image size `56832`;
- OS code offset `0`, size `256`;
- application code offset `256`, size `32000`;
- OS data offset `32256`, size `24576`;
- patch offset `32272`;
- ucode ID `3`;
- engine ID `1`;
- two signatures totaling `768` bytes.

The GSP boot descriptor is version 5 with image size `36864`, data offset `2048`,
code offset `18432`, and manifest offset `0`. Its signature comes from the
`.fwsignature_ad10x` section of the pinned 610.43.03 GSP ELF.

## Transaction invariants

- The starting state must be cold, with WPR2 down and the firmware boot sequence
  complete.
- The device must be the desktop RTX 4090 (`10de:2684`) with AD102 architecture
  identifier `0x192`.
- The four privileged writes are followed by readback.
- FEAT permission, effective readout, SS0, and SS1 are checked before and after
  reset and again after the normal runtime firmware boot.
- The stock signature span is restored before reset and on failure paths.
- The override survives FLR; physical power removal is required to clear it.

The complete asset hashes and reproduction commands are in [assets.md](assets.md).
The transaction and transport gates are in [design.md](design.md) and
[transport.md](transport.md).
