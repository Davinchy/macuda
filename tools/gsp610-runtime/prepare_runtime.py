#!/usr/bin/env python3
""": deterministic preparation of the AD102 610.43.03 normal-runtime bundle.

Offline relative to GPU access: fetch/extract/validate only; nothing here opens a device or
contacts a GPU machine. The runtime bundle is written to a versioned subtree
(nvidia/ad102/gsp-610.43.03/) separate from the 570 firmware root. Same inputs =>
byte-identical bundle; no timestamps are written.

Inputs (all hash-pinned, see runtime_assets.py):
  * NVIDIA-kernel-module-source-610.43.03.tar.xz, extracted by fetch_inputs.sh into this
    worktree's third_party.
  * The official NVIDIA-Linux-x86_64-610.43.03.run package, extracted by
    fetch_release_fw.sh into the same worktree.

Usage:
  python3 prepare_runtime.py [--out DIR] [--force]
Defaults:
  out = <repo>/cuda-shim/build/gsp610-runtime/bundle   (gitignored build output)
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))

import runtime_assets as ra  # noqa: E402

TP = REPO / "cuda-shim/libtinynv/third_party"
SOURCE_ROOT = TP / "NVIDIA-kernel-module-source-610.43.03"
DEFAULT_OUT = REPO / "cuda-shim/build/gsp610-runtime/bundle"
RELEASE_ROOT = TP / "gsp610-runtime/release"
GSP_ELF = RELEASE_ROOT / "gsp_ga10x.bin"
NVIDIA_LICENSE = RELEASE_ROOT / "NVIDIA-LICENSE"


def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_source_tree() -> Path:
    if not (SOURCE_ROOT / "src/nvidia/generated").is_dir():
        log("pinned 610 source tree absent: running fetch_inputs.sh")
        subprocess.run(["sh", str(HERE / "fetch_inputs.sh")], check=True)
    if not (SOURCE_ROOT / "src/nvidia/generated").is_dir():
        raise ra.AssetError(f"610 source tree still missing at {SOURCE_ROOT}")
    return SOURCE_ROOT


def verify_release_assets() -> bool:
    """Return true only for the pinned GSP ELF and NVIDIA license."""
    try:
        return ((GSP_ELF.is_file() and not GSP_ELF.is_symlink()
                 and ra.sha256_file(GSP_ELF) == ra.ASSET_PINS["gsp"])
                and (NVIDIA_LICENSE.is_file() and not NVIDIA_LICENSE.is_symlink()
                     and ra.sha256_file(NVIDIA_LICENSE) == ra.NVIDIA_LICENSE_SHA))
    except OSError:
        return False


def ensure_release_assets() -> None:
    if verify_release_assets():
        log("ok: official 610.43.03 GSP ELF and license verified")
        return
    log("pinned 610.43.03 release assets absent or invalid: running fetch_release_fw.sh")
    subprocess.run(["sh", str(HERE / "fetch_release_fw.sh")], check=True)
    if not verify_release_assets():
        raise ra.AssetError(f"release assets under {RELEASE_ROOT} failed hash verification")


def prepare(out: Path, force: bool) -> None:
    if out.is_dir() and not force:
        try:
            ra.validate_bundle(out, quiet=True)
            log(f"PASS: existing bundle already valid: {out} (use --force to rebuild)")
            return
        except (ra.AssetError, OSError, ValueError):
            log("existing bundle invalid or incomplete; rebuilding")

    source = ensure_source_tree()
    assets = ra.extract_runtime_assets(source)
    log(f"ok: {len(assets)} assets extracted from the pinned 610.43.03 bindata and pin-verified")

    # GSP ELF: use the official 610.43.03 release member after hash + section verification.
    ensure_release_assets()
    gsp = GSP_ELF.read_bytes()
    if ra.sha256(gsp) != ra.ASSET_PINS["gsp"]:
        raise ra.AssetError("official 610.43.03 GSP ELF failed hash verification")
    sections = ra.elf_sections(gsp, "gsp")
    for required in (".fwimage", ".fwsignature_ad10x", ".fwversion"):
        if required not in sections:
            raise ra.AssetError(f"official GSP ELF lacks {required}")
    off, size = sections[".fwversion"]
    if gsp[off:off + size] != ra.GSP_FWVERSION:
        raise ra.AssetError("official GSP ELF .fwversion is not 610.43.03")
    assets["gsp"] = gsp
    log("ok: GSP ELF extracted from the official 610.43.03 release (sections checked)")

    manifest = ra.build_manifest(assets)
    out.mkdir(parents=True, exist_ok=True)
    asset_dir = out / ra.ASSET_DIR
    asset_dir.mkdir(parents=True, exist_ok=True)
    for logical, data in assets.items():
        (asset_dir / ra.ASSET_FILES[logical]).write_bytes(data)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (out / "NOTICE.md").write_text(ra.NOTICE_TEXT)
    shutil.copyfile(NVIDIA_LICENSE, out / "NVIDIA-LICENSE")
    shutil.copyfile(source / "COPYING", out / "SOURCE-COPYING")

    names = [f"{ra.ASSET_DIR}/{ra.ASSET_FILES[k]}" for k in sorted(ra.ASSET_FILES)]
    names += ["manifest.json", "NOTICE.md", "NVIDIA-LICENSE", "SOURCE-COPYING"]
    lines = [f"{ra.sha256_file(out / n)}  {n}" for n in sorted(names)]
    (out / "SHA256SUMS").write_text("\n".join(lines) + "\n")

    ra.validate_bundle(out)
    log(f"PASS: runtime bundle prepared: {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--force", action="store_true", help="rebuild even if the output already validates")
    args = parser.parse_args()
    try:
        prepare(args.out, args.force)
    except (ra.AssetError, OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
