# RTX 4090 tensor unlock assets

## Source revision

The selected implementation is `JustAResearcher/gpu-unlock-ada` revision
`8855e7bca8ff142204e02301331fd56b26699819`. That revision records an independently
exercised RTX 4090 stock/unlocked comparison and is the source of the pinned
payload layout used here.

The upstream repository is GPL-2.0-only. Its license, notice, and the licenses for
the payload-derived components are preserved under
`tools/rtx4090-unlock/upstream/`.

NVIDIA firmware bytes are not covered by the open-source module license. The
extracted NVIDIA license is stored with the prepared bundle and is excluded from
version control.

## Inputs

| Input | SHA-256 |
|---|---|
| NVIDIA kernel module source 610.43.03 | `7e118923c7a23edc36114d63273a46e3e04e9af98695a42203e7ac2dfe9fc1dc` |
| NVIDIA Linux driver release 610.43.03 | `45e2d4c134a23c35e50f253a4aa63e7e5e8d17e3d185d4a07c8a58e9612ed392` |
| GSP ELF `gsp_ga10x.bin` | `572373620a37418f24dc16b5031c39338778c3257e48e8408de9a57291b24f3a` |
| NVIDIA license extracted from the release | `c692fb14ad4f499c6aee1b9e144e6905148c420e6225b17cb2937366c1e507ad` |

The manifest also records the pinned source files and the input package
substitution. A replacement package was accepted only after the extracted ELF and
license matched the independently recorded hashes above.

## Generated bundle

| Asset | Size | SHA-256 |
|---|---:|---|
| `booter_load_ad102.bin` | 57,672 | `6ee05b20e1dc244c08c052f36ef28469b8d09f4955d1fe33e48a179e11a7fd49` |
| `gsp_boot_ad102.bin` | 36,864 | `45d6335a8185d6ff13e8d1484cea73f68d3d13aa1c19dc080ed1c3482dd5bac0` |
| `gsp_boot_desc_ad102.bin` | 84 | `83f8f3f1f5f9496e0ea8fc7103c2beb64d0b6de2f5ffe725f13514de5a95fa42` |
| `v67-ad102-610.43.03-four-plm.bin` | 64,000 | `44c36b2d6b3424e53e18f3e55db85c833655de071ce66ce42749588a4a7dc8cb` |
| `gsp_ga10x.bin` | 84,277,400 | `572373620a37418f24dc16b5031c39338778c3257e48e8408de9a57291b24f3a` |

The C loader embeds the same pins and rejects drift at runtime. The offline
validator additionally checks BTL1 container geometry, descriptor version and
offsets, ELF sections, signature bounds, radix sizing, and device eligibility.

## Reproduction

```sh
python3 tools/rtx4090-unlock/prepare_bundle.py
python3 tools/rtx4090-unlock/validate_bundle.py \
  cuda-shim/libtinynv/third_party/unlock-610.43.03
python3 -m unittest discover -s tools/rtx4090-unlock -p 'test_*.py' -v
```

The same inputs produce a byte-identical bundle. No timestamp or local worktree
path is written to the manifest.
