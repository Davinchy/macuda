#!/usr/bin/env python3
"""Offline validator for the pinned AD102 610.43.03 unlock bundle.

Checks, fail-closed, before any device use:
  * every required asset and notice exists as a regular file (no symlinks),
    with the exact pinned size and SHA-256;
  * BTL1 booter bundle structure against the pinned Falcon geometry;
  * GSP boot descriptor fields and ranges against the boot image;
  * V67 payload word-exact layout (fill allowlist: any unpinned deviation,
    including a different recipe's image, is rejected);
  * GSP ELF section table and the pinned .fwsignature_ad10x section bounds.

Structural checks are independent of the hash checks: a corrupted asset that
was re-hashed into a forged manifest still fails structure, and a structurally
valid asset with wrong bytes fails the pinned hash. This validator proves
asset integrity and layout only — never hardware behavior.

Usage:
  python3 validate_bundle.py BUNDLE_DIR [--manifest manifest.json] [--quiet]
Exit status 0 = all checks pass; 1 = at least one failure (first error printed).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE / "manifest.json"

# Hard caps so a malformed header can never drive a huge allocation/read.
MAX_BOOTER_IMAGE = 16 * 1024 * 1024
MAX_BOOT_IMAGE = 1 * 1024 * 1024
MAX_SIGNATURES = 64 * 384


class ValidationError(ValueError):
    pass


def _need(cond, msg):
    if not cond:
        raise ValidationError(msg)


def sha256_file(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def check_payload(data: bytes, layout: dict) -> None:
    """Word-exact V67 AD102 payload check (fill allowlist)."""
    size = layout["size"]
    _need(len(data) == size, "payload size %d != pinned %d" % (len(data), size))
    _need(size % 4 == 0, "payload layout size not a multiple of 4")
    fill = int(layout["fill_dword"], 16)
    patched = {int(o, 16): int(v, 16) for o, v in layout["patched_words"].items()}
    for off, val in patched.items():
        _need(0 <= off <= size - 4 and off % 4 == 0,
              "manifest payload patch offset %#x out of range/unaligned" % off)
        _need(0 <= val <= 0xFFFFFFFF, "manifest payload patch value overflow")
    words = struct.unpack("<%dI" % (size // 4), data)
    for i, word in enumerate(words):
        off = i * 4
        expect = patched.get(off, fill)
        _need(word == expect,
              "payload word %#06x = %#010x, expected %#010x" % (off, word, expect))


def check_booter(data: bytes, geo: dict) -> None:
    """BTL1 booter bundle: header, geometry, ranges, signature table."""
    image_size = geo["image_size"]
    sig_bytes = geo["signature_total_bytes"]
    _need(0 < image_size <= MAX_BOOTER_IMAGE, "geometry image size out of bounds")
    _need(0 < sig_bytes <= MAX_SIGNATURES, "geometry signature size out of bounds")
    total = 72 + image_size + sig_bytes
    _need(len(data) == total, "booter size %d != 72+%d+%d" % (len(data), image_size, sig_bytes))
    magic, version = struct.unpack_from("<II", data, 0)
    _need(magic == int(geo["magic"], 16), "booter magic %#x != BTL1" % magic)
    _need(version == geo["bundle_version"], "booter bundle version %d != %d" % (version, geo["bundle_version"]))
    fields = struct.unpack_from("<14I", data, 8)
    (os_code_off, os_code_size, os_data_off, os_data_size, num_apps,
     app_code_off, app_code_size, app_data_off, app_data_size,
     patch_loc, ucode_id, engine_id, num_sigs, sig_len) = fields
    expected = (
        (os_code_off, geo["os_code_offset"], "os_code offset"),
        (os_code_size, geo["os_code_size"], "os_code size"),
        (os_data_off, geo["os_data_offset"], "os_data offset"),
        (os_data_size, geo["os_data_size"], "os_data size"),
        (num_apps, geo["application_count"], "application count"),
        (app_code_off, geo["app_code_offset"], "app_code offset"),
        (app_code_size, geo["app_code_size"], "app_code size"),
        (app_data_off, geo["app_data_offset"], "app_data offset"),
        (app_data_size, geo["app_data_size"], "app_data size"),
        (patch_loc, geo["signature_patch_offset"], "signature patch offset"),
        (ucode_id, geo["ucode_id"], "ucode id"),
        (engine_id, geo["engine_id"], "engine id"),
        (num_sigs, geo["signature_count"], "signature count"),
        (sig_len, sig_bytes, "signature table bytes"),
    )
    for got, want, name in expected:
        _need(got == want, "booter %s %#x != pinned %#x" % (name, got, want))
    _need(num_sigs > 0 and sig_len == num_sigs * geo["signature_bytes_each"],
          "booter signature table inconsistent")
    tail_image, tail_sigs = struct.unpack_from("<II", data, 64)
    _need(tail_image == image_size and tail_sigs == sig_len,
          "booter trailing size fields inconsistent")
    for off, size, name in ((os_code_off, os_code_size, "os_code"),
                            (os_data_off, os_data_size, "os_data"),
                            (app_code_off, app_code_size, "app_code")):
        _need(off + size <= image_size, "booter %s range outside image" % name)
        _need(off % 256 == 0 and size % 256 == 0, "booter %s range not 256-byte aligned" % name)
    _need(os_data_off <= patch_loc and patch_loc + geo["signature_bytes_each"] <= os_data_off + os_data_size,
          "booter signature patch outside DMEM")


def check_boot_desc(desc: bytes, image_len: int, expected: dict) -> None:
    """84-byte GSP boot descriptor: version, self-size, bounded ranges."""
    _need(len(desc) == 84, "boot descriptor size %d != 84" % len(desc))
    _need(0 < image_len <= MAX_BOOT_IMAGE, "boot image size out of bounds")
    fields = struct.unpack("<21I", desc)
    _need(fields[0] == expected["version"], "boot descriptor version %d != %d" % (fields[0], expected["version"]))
    _need(fields[14] == 1, "boot descriptor word 14 != 1")
    _need(fields[19] == image_len, "boot descriptor image size %d != %d" % (fields[19], image_len))
    _need(fields[19] == expected["image_size"], "boot descriptor image size != pinned")
    _need(fields[20] == 0, "boot descriptor word 20 != 0")
    for index in (1, 3, 8, 10, 12):
        off, size = fields[index], fields[index + 1]
        _need(size > 0, "boot descriptor range at word %d has zero size" % index)
        _need(off + size <= image_len, "boot descriptor range at word %d outside image" % index)
    _need(fields[8] == expected["manifest_offset"], "boot descriptor manifest offset mismatch")
    _need(fields[10] == expected["data_offset"], "boot descriptor data offset mismatch")
    _need(fields[12] == expected["code_offset"], "boot descriptor code offset mismatch")


def check_gsp_elf(data: bytes, spec: dict) -> None:
    """ELF64 header + section table bounds + pinned signature section."""
    _need(len(data) >= 64, "GSP ELF truncated header")
    _need(data[:4] == b"\x7fELF", "GSP ELF bad magic")
    _need(data[4] == 2 and data[5] == 1, "GSP ELF not ELF64 little-endian")
    e_shoff, = struct.unpack_from("<Q", data, 40)
    e_shentsize, e_shnum, e_shstrndx = struct.unpack_from("<HHH", data, 58)
    _need(e_shentsize == 64, "GSP ELF unexpected section header size %d" % e_shentsize)
    _need(0 < e_shnum <= 4096, "GSP ELF section count out of bounds")
    _need(e_shstrndx < e_shnum, "GSP ELF shstrtab index out of range")
    _need(e_shoff + e_shnum * e_shentsize <= len(data), "GSP ELF section table outside file")

    def section(i):
        base = e_shoff + i * e_shentsize
        name, _typ, _flags, _addr, off, size = struct.unpack_from("<IIQQQQ", data, base)
        return name, off, size

    str_name, str_off, str_size = section(e_shstrndx)
    _need(str_off + str_size <= len(data), "GSP ELF shstrtab outside file")
    strtab = data[str_off:str_off + str_size]
    want = spec["signature_section"].encode("ascii")
    found = None
    for i in range(e_shnum):
        name, off, size = section(i)
        _need(name < str_size, "GSP ELF section %d name outside shstrtab" % i)
        end = strtab.find(b"\0", name)
        _need(end != -1, "GSP ELF unterminated section name")
        if strtab[name:end] == want:
            _need(found is None, "GSP ELF duplicate signature section")
            found = (off, size)
    _need(found is not None, "GSP ELF missing section " + spec["signature_section"])
    off, size = found
    _need(off == spec["signature_offset"] and size == spec["signature_size"],
          "GSP ELF signature section at %#x+%d != pinned %#x+%d"
          % (off, size, spec["signature_offset"], spec["signature_size"]))
    _need(off + size <= len(data), "GSP ELF signature section outside file")


def validate_bundle(bundle: Path, manifest: dict, quiet: bool = False) -> list:
    """Validate a prepared bundle directory. Returns list of pass lines;
    raises ValidationError on the first failure."""
    passed = []

    def note(msg):
        passed.append(msg)
        if not quiet:
            print("ok: " + msg)

    def load(name, spec, structural=None):
        path = bundle / name
        _need(not path.is_symlink(), "%s is a symlink" % name)
        _need(path.is_file(), "%s missing or not a regular file" % name)
        size = path.stat().st_size
        _need(size == spec["size"], "%s size %d != pinned %d" % (name, size, spec["size"]))
        digest = sha256_file(path)
        _need(digest == spec["sha256"], "%s sha256 %s != pinned %s" % (name, digest, spec["sha256"]))
        data = path.read_bytes()
        if structural is not None:
            structural(data)
        note("%s size+sha256%s" % (name, "+structure" if structural else ""))
        return data

    assets = manifest["assets"]
    payload_name = manifest["recipe"]["payload_filename"]
    booter = load("booter_load_ad102.bin", assets["booter_load_ad102.bin"],
                  lambda d: check_booter(d, manifest["booter_geometry"]))
    boot = load("gsp_boot_ad102.bin", assets["gsp_boot_ad102.bin"])
    load("gsp_boot_desc_ad102.bin", assets["gsp_boot_desc_ad102.bin"],
         lambda d: check_boot_desc(d, len(boot), manifest["gsp_boot_descriptor"]))
    load(payload_name, assets[payload_name],
         lambda d: check_payload(d, manifest["payload_layout"]))
    load("gsp_ga10x.bin", assets["gsp_ga10x.bin"],
         lambda d: check_gsp_elf(d, manifest["gsp_elf"]))
    for name, spec in manifest["notices"].items():
        path = bundle / name
        _need(not path.is_symlink(), "notice %s is a symlink" % name)
        _need(path.is_file(), "notice %s missing" % name)
        if "sha256" in spec:
            _need(sha256_file(path) == spec["sha256"], "notice %s sha256 mismatch" % name)
        note("notice " + name)
    # The bundle must carry its own pin manifest, byte-identical to the tool's.
    bundled_manifest = bundle / "manifest.json"
    _need(bundled_manifest.is_file() and not bundled_manifest.is_symlink(), "bundle manifest.json missing")
    _need(json.loads(bundled_manifest.read_text()) == manifest,
          "bundle manifest.json differs from the pinned tool manifest")
    note("bundle manifest.json matches pinned manifest")
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    try:
        passed = validate_bundle(args.bundle, manifest, args.quiet)
    except (ValidationError, OSError, struct.error) as error:
        print("FAIL: %s" % error, file=sys.stderr)
        return 1
    print("PASS: %d checks, bundle %s" % (len(passed), args.bundle))
    return 0


if __name__ == "__main__":
    sys.exit(main())
