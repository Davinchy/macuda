#!/usr/bin/env python3
"""Deterministic preparation of the pinned AD102 610.43.03 unlock bundle.

Pipeline (all inputs and outputs SHA-256 pinned in manifest.json):
  1. verify the vendored upstream files against the pin manifest;
  2. fetch the two pinned public NVIDIA artifacts (cached, hash-verified,
     never executed/installed); --offline uses the cache only;
  3. extract exactly the two pinned generated AD102 bindata sources from the
     NVIDIA source archive (member-name allowlist, regular files, size cap);
  4. run the vendored upstream extractor (chip=ad102) and payload generator;
  5. assemble the bundle with notices and the pin manifest, then validate it
     with validate_bundle.py before reporting success.

Same inputs => byte-identical bundle. No timestamps are written. Nothing here
contacts a GPU; the bundle is an offline artifact for the later transaction
card. NVIDIA firmware bytes are NOT committed to git (third_party is ignored).

Usage:
  python3 prepare_bundle.py [--output DIR] [--cache DIR] [--offline] [--force]
Defaults:
  output = <repo>/cuda-shim/libtinynv/third_party/unlock-610.43.03
  cache  = <repo>/cuda-shim/libtinynv/third_party/unlock-artifacts
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "upstream"))

import validate_bundle as vb  # noqa: E402
import fetch_artifacts  # noqa: E402  (vendored, pinned)
import extract_boot_assets  # noqa: E402  (vendored, pinned)
import generate_v67  # noqa: E402  (vendored, pinned)

DEFAULT_OUTPUT = REPO / "cuda-shim/libtinynv/third_party/unlock-610.43.03"
DEFAULT_CACHE = REPO / "cuda-shim/libtinynv/third_party/unlock-artifacts"
MAX_SOURCE_MEMBER = 64 * 1024 * 1024


def sha256_file(path: Path) -> str:
    return vb.sha256_file(path)


def verify_vendored(manifest: dict) -> None:
    for rel, want in manifest["upstream"]["vendored_files_sha256"].items():
        path = HERE / "upstream" / rel
        if not path.is_file() or path.is_symlink():
            raise vb.ValidationError("vendored upstream file missing: " + rel)
        got = sha256_file(path)
        if got != want:
            raise vb.ValidationError(
                "vendored %s sha256 %s != pin %s (upstream %s @ %s)"
                % (rel, got, want, manifest["upstream"]["repository"], manifest["upstream"]["commit"]))
    print("ok: %d vendored upstream files match the pinned revision"
          % len(manifest["upstream"]["vendored_files_sha256"]))


def extract_generated_sources(archive: Path, pins: dict, dest: Path) -> Path:
    """Extract only the pinned bindata members; refuse anything else."""
    root = dest / "nvidia-source"
    gen_dir = root / "src/nvidia/generated"
    gen_dir.mkdir(parents=True, exist_ok=True)
    wanted = {"src/nvidia/generated/" + name for name in pins}
    seen = set()
    with tarfile.open(archive, mode="r:*") as tar:
        for member in tar:
            # archive members carry a top-level directory prefix
            parts = Path(member.name).parts
            rel = "/".join(parts[1:]) if len(parts) > 1 else ""
            if rel not in wanted:
                continue
            if not member.isfile() or member.size > MAX_SOURCE_MEMBER:
                raise vb.ValidationError("bad source archive member: " + member.name)
            if rel in seen:
                raise vb.ValidationError("duplicate source archive member: " + member.name)
            seen.add(rel)
            stream = tar.extractfile(member)
            if stream is None:
                raise vb.ValidationError("unreadable source archive member: " + member.name)
            data = stream.read()
            name = Path(rel).name
            got = hashlib.sha256(data).hexdigest()
            if got != pins[name]:
                raise vb.ValidationError("generated source %s sha256 %s != pin %s" % (name, got, pins[name]))
            (gen_dir / name).write_bytes(data)
    if seen != wanted:
        raise vb.ValidationError("source archive missing pinned members: %s" % sorted(wanted - seen))
    print("ok: %d generated bindata sources extracted and hash-verified" % len(seen))
    return root


def build_from_package(package: Path, staging: Path, manifest: dict) -> None:
    """AD102 subset of the vendored generate_v67 main(): hash-verify everything."""
    pkg_spec = manifest["inputs"][1]
    if sha256_file(package) != pkg_spec["sha256"]:
        raise vb.ValidationError("package sha256 does not match the pinned nvidia-utils artifact")
    pkginfo = generate_v67.extract_member(package, ".PKGINFO").decode("utf-8", errors="replace")
    if ("pkgver = " + pkg_spec["version"]) not in pkginfo.splitlines():
        raise vb.ValidationError("package metadata version does not match " + pkg_spec["version"])
    gsp = generate_v67.extract_member(package, generate_v67.FIRMWARE_PREFIX + "gsp_ga10x.bin")
    if generate_v67.sha256(gsp) != manifest["assets"]["gsp_ga10x.bin"]["sha256"]:
        raise vb.ValidationError("gsp_ga10x.bin sha256 does not match the pinned artifact")
    payload = generate_v67.build_ad102_payload()
    payload_name = manifest["recipe"]["payload_filename"]
    if generate_v67.sha256(payload) != manifest["assets"][payload_name]["sha256"]:
        raise vb.ValidationError("generated AD102 payload differs from its pinned sha256")
    license_text = generate_v67.extract_member(package, "usr/share/licenses/nvidia-utils/LICENSE")
    if generate_v67.sha256(license_text) != manifest["notices"]["NVIDIA-LICENSE"]["sha256"]:
        raise vb.ValidationError("NVIDIA license notice differs from the pinned package")
    (staging / "gsp_ga10x.bin").write_bytes(gsp)
    (staging / payload_name).write_bytes(payload)
    (staging / "NVIDIA-LICENSE").write_bytes(license_text)
    print("ok: package extracted; AD102 payload generated (sha256 %s)"
          % manifest["assets"][payload_name]["sha256"][:16])


def assemble(staging: Path, output: Path, manifest: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    names = list(manifest["assets"]) + list(manifest["notices"])
    for name in names:
        src = staging / name
        if not src.is_file():
            # notices are vendored, not generated
            src = HERE / "upstream" / name
        if not src.is_file() or src.is_symlink():
            raise vb.ValidationError("missing bundle input: " + name)
        dst = output / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    shutil.copyfile(HERE / "manifest.json", output / "manifest.json")
    lines = []
    for name in sorted(names + ["manifest.json"]):
        lines.append("%s  %s" % (sha256_file(output / name), name))
    (output / "SHA256SUMS").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--offline", action="store_true", help="verify cached inputs without network")
    parser.add_argument("--force", action="store_true", help="rebuild even if the output already validates")
    args = parser.parse_args()

    manifest = json.loads((HERE / "manifest.json").read_text())
    try:
        verify_vendored(manifest)
        if args.output.is_dir() and not args.force:
            try:
                vb.validate_bundle(args.output, manifest, quiet=True)
                print("PASS: existing bundle already valid: %s (use --force to rebuild)" % args.output)
                return 0
            except (vb.ValidationError, OSError):
                print("existing bundle invalid or incomplete; rebuilding")
        source_spec, package_spec = manifest["inputs"]
        source = fetch_artifacts.fetch(source_spec, args.cache, args.offline)
        print("ok: input %s sha256-verified" % source.name)
        package = fetch_artifacts.fetch(package_spec, args.cache, args.offline)
        print("ok: input %s sha256-verified" % package.name)
        with tempfile.TemporaryDirectory(prefix="rtx4090-unlock-") as tmp:
            staging = Path(tmp) / "staging"
            staging.mkdir()
            root = extract_generated_sources(source, manifest["generated_source_pins"], Path(tmp))
            extract_boot_assets.extract_assets(root, staging, chip="ad102")
            print("ok: booter/GSP boot/descriptor extracted (upstream pinned extractor, chip=ad102)")
            build_from_package(package, staging, manifest)
            assemble(staging, args.output, manifest)
        vb.validate_bundle(args.output, manifest)
    except (vb.ValidationError, OSError, ValueError) as error:
        print("FAIL: %s" % error, file=sys.stderr)
        return 1
    print("PASS: bundle prepared and validated: %s" % args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
