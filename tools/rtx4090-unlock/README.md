# RTX 4090 unlock bundle

Deterministic preparation and offline validation for the AD102 tensor-unlock
assets. Nothing in this directory contacts the GPU, TinyGPU server, or enclosure.
Firmware bytes are generated into a gitignored directory and are not
redistributed.

The bundle derives from `JustAResearcher/gpu-unlock-ada` at
`8855e7bca8ff142204e02301331fd56b26699819`, with two pinned public NVIDIA
610.43.03 inputs. `manifest.json` is the source of truth for input hashes,
generated-source hashes, output sizes and hashes, boot geometry, payload writes,
device eligibility, and notices.

## Prepare and validate

```sh
python3 tools/rtx4090-unlock/prepare_bundle.py

# Use only the existing artifact cache, with no network access.
python3 tools/rtx4090-unlock/prepare_bundle.py --offline

python3 tools/rtx4090-unlock/validate_bundle.py \
  cuda-shim/libtinynv/third_party/unlock-610.43.03

python3 -m unittest discover -s tools/rtx4090-unlock -p 'test_*.py' -v
```

The output contains:

| File | Role |
|---|---|
| `booter_load_ad102.bin` | Signed BTL1 load booter |
| `gsp_boot_ad102.bin` | GSP boot image for the FWSEC-FRTS stage |
| `gsp_boot_desc_ad102.bin` | Version 5 GSP boot descriptor |
| `v67-ad102-610.43.03-four-plm.bin` | Four-PLM V67 privilege payload |
| `gsp_ga10x.bin` | 610.43.03 ELF used for its AD102 signature section |
| `manifest.json`, `SHA256SUMS`, notices | Provenance and verification data |

The generated bundle is separate from macuda's unchanged 570.144 runtime
firmware. It does not replace or modify normal runtime files.

## Integrity limits

Preparation and validation prove asset integrity and structure. They do not
prove hardware behavior, reset retention, the 610-to-570 handoff, or DMA
containment on a particular transport. The recorded live evidence and remaining
gaps are in
[validation.md](../../docs/driver/rtx4090-tensor-unlock/validation.md).
