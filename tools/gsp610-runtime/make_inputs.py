#!/usr/bin/env python3
"""Record immutable NVIDIA inputs for the 570-to-610 runtime ABI audit.

Writes docs/driver/gsp-610-runtime/inputs.json. Every hash is computed from bytes on disk and
cross-checked against the source/release pins. A mismatch is reported, never silently accepted.
No GPU contact.

Usage: python3 make_inputs.py [--repo ROOT]
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path

def sha256(p: Path) -> str | None:
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def elf_sections(p: Path) -> dict:
    """Return {name: (offset, size)} for an ELF firmware image (bounded parse)."""
    import struct
    d = p.read_bytes()
    if d[:4] != b"\x7fELF" or d[4] != 2:
        return {}
    shoff = struct.unpack_from("<Q", d, 0x28)[0]
    shentsize = struct.unpack_from("<H", d, 0x3A)[0]
    shnum = struct.unpack_from("<H", d, 0x3C)[0]
    shstrndx = struct.unpack_from("<H", d, 0x3E)[0]

    def sh(i):
        return struct.unpack_from("<IIQQQQIIQQ", d, shoff + i * shentsize)

    st = sh(shstrndx)
    stroff = st[4]
    out = {}
    for i in range(shnum):
        s = sh(i)
        nm = d[stroff + s[0]:stroff + s[0] + 64].split(b"\x00")[0].decode()
        out[nm] = (s[4], s[5])
    return out


def firmware_pins(repo: Path) -> dict:
    """Parse the chip/name/sha table out of fetch_firmware.sh."""
    txt = (repo / "cuda-shim/libtinynv/tools/fetch_firmware.sh").read_text()
    pins = {}
    for m in re.finditer(r'"(\w+)\s+([\w.\-]+\.bin)\s+([0-9a-f]{64})"', txt):
        chip, name, sha = m.groups()
        pins[f"nvidia/{chip}/gsp/{name}"] = sha
    return pins


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
    args = ap.parse_args()
    repo = Path(args.repo)
    tp = repo / "cuda-shim/libtinynv/third_party"

    out: dict = {
        "schema": "gsp610-inputs/1",
        "purpose": "Immutable NVIDIA inputs for the 570-to-610 runtime ABI audit",
    }

    # --- header/source trees (the two pins the audit compares) ---
    ogkm_commit = "81fe4fb417c8ac3b9bdcc1d56827d116743892a5"
    sums = (repo / "cuda-shim/libtinynv/tools/SHA256SUMS").read_text()
    ogkm_want = next((l.split()[0] for l in sums.splitlines() if ogkm_commit in l), None)
    ogkm_tar = tp / f"ogkm-{ogkm_commit}.tar.gz"
    kms_tar = tp / "NVIDIA-kernel-module-source-610.43.03.tar.xz"
    out["header_trees"] = {
        "570": {
            "repo": "NVIDIA/open-gpu-kernel-modules",
            "commit": ogkm_commit,
            "release": "570.86.16",
            "tarball": ogkm_tar.name,
            "sha256": sha256(ogkm_tar),
            "sha256_expected": ogkm_want,
            "extracted": (tp / f"open-gpu-kernel-modules-{ogkm_commit}/src").is_dir(),
        },
        "610": {
            "repo": "NVIDIA-kernel-module-source (download.nvidia.com)",
            "release": "610.43.03",
            "tarball": kms_tar.name,
            "sha256": sha256(kms_tar),
            "sha256_expected": "7e118923c7a23edc36114d63273a46e3e04e9af98695a42203e7ac2dfe9fc1dc",
            "extracted": (tp / "NVIDIA-kernel-module-source-610.43.03/src").is_dir(),
            "version_mk": (tp / "NVIDIA-kernel-module-source-610.43.03/version.mk").read_text().splitlines()[0]
            if (tp / "NVIDIA-kernel-module-source-610.43.03/version.mk").is_file() else None,
        },
    }

    # --- 570 runtime firmware ---
    fw_root = tp / "firmware"
    pins = firmware_pins(repo)
    fw = {}
    for rel, want in pins.items():
        got = sha256(fw_root / rel)
        fw[rel] = {"sha256": got, "sha256_expected": want, "match": got == want}
    out["firmware_570"] = {
        "root": "cuda-shim/libtinynv/third_party/firmware",
        "files": fw,
    }

    # --- official 610 release files used by the runtime bundle ---
    release_dir = tp / "gsp610-runtime/release"
    gsp610 = release_dir / "gsp_ga10x.bin"
    nvidia_license = release_dir / "NVIDIA-LICENSE"
    gsp610_sha = sha256(gsp610)
    license_sha = sha256(nvidia_license)
    out["release_610"] = {
        "package": "NVIDIA-Linux-x86_64-610.43.03.run",
        "package_sha256_expected": "45e2d4c134a23c35e50f253a4aa63e7e5e8d17e3d185d4a07c8a58e9612ed392",
        "gsp_elf": {
            "path": "cuda-shim/libtinynv/third_party/gsp610-runtime/release/gsp_ga10x.bin",
            "sha256": gsp610_sha,
            "sha256_expected": "572373620a37418f24dc16b5031c39338778c3257e48e8408de9a57291b24f3a",
            "match": gsp610_sha == "572373620a37418f24dc16b5031c39338778c3257e48e8408de9a57291b24f3a",
        },
        "license": {
            "path": "cuda-shim/libtinynv/third_party/gsp610-runtime/release/NVIDIA-LICENSE",
            "sha256": license_sha,
            "sha256_expected": "c692fb14ad4f499c6aee1b9e144e6905148c420e6225b17cb2937366c1e507ad",
            "match": license_sha == "c692fb14ad4f499c6aee1b9e144e6905148c420e6225b17cb2937366c1e507ad",
        },
    }

    # --- GSP ELF section geometry (computed from the actual images) ---
    gsp570 = fw_root / "nvidia/ga102/gsp/gsp-570.144.bin"
    s570, s610 = elf_sections(gsp570), elf_sections(gsp610)
    out["gsp_elf"] = {
        "570": {"path": "cuda-shim/libtinynv/third_party/firmware/nvidia/ga102/gsp/gsp-570.144.bin",
                "fwimage_size": s570.get(".fwimage", (None, None))[1],
                "fwsignature_ad10x_offset": s570.get(".fwsignature_ad10x", (None,))[0]},
        "610": {"path": out["release_610"]["gsp_elf"]["path"],
                "fwimage_size": s610.get(".fwimage", (None, None))[1],
                "fwsignature_ad10x_offset": s610.get(".fwsignature_ad10x", (None,))[0]},
        "section_names_570": sorted(s570),
        "section_names_610": sorted(s610),
        "note": "610 .fwimage is 80 MiB, exceeding the 570-profile TINYNV_FW_IMAGE_BOUND of 64 MiB",
    }

    # --- verification roll-up ---
    checks = []
    for rel, r in fw.items():
        checks.append((f"firmware_570:{rel}", r["match"]))
    checks.append(("release_610:gsp", out["release_610"]["gsp_elf"]["match"]))
    checks.append(("release_610:license", out["release_610"]["license"]["match"]))
    for tag in ("570", "610"):
        t = out["header_trees"][tag]
        checks.append((f"header_tree:{tag}", t["sha256"] == t["sha256_expected"]))
    out["verification"] = {
        "all_match": all(ok for _, ok in checks),
        "failed": [name for name, ok in checks if not ok],
        "checked": len(checks),
    }

    dest = repo / "docs/driver/gsp-610-runtime/inputs.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {dest}")
    print(f"verification: {out['verification']['checked']} checks, "
          f"all_match={out['verification']['all_match']}, failed={out['verification']['failed']}")


if __name__ == "__main__":
    main()
