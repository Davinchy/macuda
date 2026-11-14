#!/usr/bin/env python3
""": offline unit tests for 610 runtime asset preparation/validation.

Run: python3 -m unittest discover -s tools/gsp610-runtime -p 'test_*.py' -v

Covers valid / corrupt / truncated / wrong-chip cases for every asset format, the BINDATA
extractor, manifest immutability and cross-asset lifecycle constraints, using small synthetic
fixtures. Integration tests against the real prepared bundle and the pinned source tree run
when those artifacts exist (they are gitignored build outputs) and skip otherwise; the real
bundle is additionally validated by validate_runtime.py in run_prepare.sh.

No GPU is contacted. Synthetic fixtures are parser tests, NOT evidence of hardware behaviour.
"""
from __future__ import annotations

import json
import os
import struct
import tempfile
import unittest
import zlib
from pathlib import Path

import runtime_assets as ra

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
REAL_BUNDLE = Path(os.environ.get("GSP610_RUNTIME_BUNDLE",
                                  str(REPO / "cuda-shim/build/gsp610-runtime/bundle")))
SOURCE_ROOT = REPO / "cuda-shim/libtinynv/third_party/NVIDIA-kernel-module-source-610.43.03"


# --- synthetic fixture builders -------------------------------------------------------------

def synth_btl1(image_size=2048, os_data_off=1024, os_data_size=512, app_code_size=512,
               patch_loc=1024, ucode_id=3, engine_id=1, num_sigs=2, num_apps=1):
    image = bytes((i * 7 + 3) & 0xFF for i in range(image_size))
    sigs = bytes((i * 13 + 1) & 0xFF for i in range(num_sigs * ra.SIG_SIZE))
    header9 = (0, 256, os_data_off, os_data_size, num_apps, 256, app_code_size, 256, 0)
    return ra.pack_btl1(image, header9, patch_loc, ucode_id, engine_id, num_sigs, sigs)


SYNTH_GEOM = dict(image_size=2048, os_code_off=0, os_code_size=256,
                  os_data_off=1024, os_data_size=512, num_apps=1,
                  app_code_off=256, app_code_size=512, app_data_off=256, app_data_size=0,
                  patch_loc=1024, ucode_id=3, engine_id=1, num_sigs=2, sig_bytes=768)


def synth_desc(image_size=1024):
    f = [0] * 21
    f[0] = 5
    f[1], f[2] = 0, 256          # bootloader
    f[3], f[4] = 256, 16         # params
    f[8], f[9] = 512, 64         # manifest
    f[10], f[11] = 576, 64       # monitor data
    f[12], f[13] = 640, 128      # monitor code
    f[14] = 1
    f[19] = image_size
    return struct.pack("<21I", *f)


def synth_elf(fwimage_size=4096, sig_size=64, version=b"610.43.03\x00",
              drop_ad10x=False, elf32=False):
    """Minimal little-endian ELF64 with the GSP sections macuda reads."""
    names = [b"", b".fwimage", b".fwsignature_ad10x", b".fwversion", b".shstrtab"]
    if drop_ad10x:
        names[2] = b".fwsignature_ga10x"  # wrong chip: Ampere signature, no Ada one
    strtab = b"\0".join(names) + b"\0"
    offsets, pos = [], 64
    payloads = [b"\xA5" * fwimage_size, b"\x5A" * sig_size, version]
    for p in payloads:
        offsets.append(pos)
        pos += len(p)
    str_off = pos
    pos += len(strtab)
    shoff = pos
    shnum = 5
    name_off, acc = [], 0
    for n in names:
        name_off.append(acc)
        acc += len(n) + 1
    header = struct.pack("<4sBBBBB7xHHIQQQIHHHHHH",
                         b"\x7fELF", 1 if elf32 else 2, 1, 1, 0, 0,
                         2, 0xF3, 1, 0, 0, shoff, 0, 64, 0, 0, 64, shnum, 4)
    table = b""
    for i in range(shnum):
        off = offsets[i - 1] if 1 <= i <= 3 else str_off
        size = len(payloads[i - 1]) if 1 <= i <= 3 else len(strtab)
        table += struct.pack("<IIQQQQIIQQ", name_off[i], 1, 0, 0, off, size, 0, 0, 1, 0)
    return header + b"".join(payloads) + strtab + table


SYNTH_ELF_SPEC = {"fwimage_size": 4096, "fwversion": b"610.43.03\x00", "ad10x_sig_size": 64}


def synth_assets():
    return {
        "gsp": synth_elf(),
        "bootloader": b"\x11" * 1024,
        "bootloader_desc": synth_desc(1024),
        "booter_load": synth_btl1(),
        "booter_unload": synth_btl1(image_size=1536, os_data_off=768, os_data_size=512,
                                    app_code_size=256, patch_loc=768),
        "scrubber": synth_btl1(image_size=1280, os_data_off=768, os_data_size=512,
                               app_code_size=256, patch_loc=768, ucode_id=13),
        # : the generic-bootloader host-request ucode. The geometry constants are pinned, so the synthetic
        # image/desc must match them (the sha pins are recomputed from these synthetic bytes by synth_pins).
        "generic_bl": bytes((i * 5 + 2) & 0xFF for i in range(ra.GENERIC_BL_IMAGE_SIZE)),
        "generic_bl_desc": struct.pack("<6I", *ra.GENERIC_BL_DESC_VALUES),
    }


def synth_geometry():
    g = dict(SYNTH_GEOM)
    return {
        "booter_load": g,
        "booter_unload": dict(g, image_size=1536, os_data_off=768, app_code_size=256, patch_loc=768),
        "scrubber": dict(g, image_size=1280, os_data_off=768, os_data_size=512,
                         app_code_size=256, patch_loc=768, ucode_id=13),
    }


def write_bundle(root: Path, assets: dict) -> dict:
    manifest = ra.build_manifest(assets)
    (root / ra.ASSET_DIR).mkdir(parents=True, exist_ok=True)
    for logical, data in assets.items():
        (root / ra.ASSET_DIR / ra.ASSET_FILES[logical]).write_bytes(data)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    for notice in manifest["notices"]:
        (root / notice).write_text("notice\n")
    return manifest


def synth_pins(assets):
    return {k: ra.sha256(v) for k, v in assets.items()}


class BundleCase(unittest.TestCase):
    """Base: a fresh synthetic bundle + its pins per test."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="gsp610-test-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.assets = synth_assets()
        self.manifest = write_bundle(self.root, self.assets)
        self.pins = synth_pins(self.assets)
        self.geom = synth_geometry()

    def validate(self, **kw):
        return ra.validate_bundle(self.root, pins=self.pins, geometry=self.geom,
                                  elf=SYNTH_ELF_SPEC, quiet=True, **kw)

    def rewrite(self, logical, data):
        (self.root / ra.ASSET_DIR / ra.ASSET_FILES[logical]).write_bytes(data)

    def assertFails(self, needle, **kw):
        with self.assertRaises(ra.AssetError) as ctx:
            self.validate(**kw)
        self.assertIn(needle, str(ctx.exception))


# --- happy path ------------------------------------------------------------------------------

class TestValidBundle(BundleCase):
    def test_valid_synthetic_bundle(self):
        manifest = self.validate()
        self.assertEqual(manifest["profile"], "610.43.03")
        self.assertEqual(set(manifest["assets"]), set(ra.ASSET_FILES))

    def test_manifest_deterministic(self):
        a = json.dumps(ra.build_manifest(self.assets), sort_keys=True)
        b = json.dumps(ra.build_manifest(synth_assets()), sort_keys=True)
        self.assertEqual(a, b)

    def test_btl1_pack_deterministic(self):
        self.assertEqual(synth_btl1(), synth_btl1())


# --- corrupt / truncated / missing ------------------------------------------------------------

class TestCorruptAndTruncated(BundleCase):
    def test_corrupt_byte_fails_hash(self):
        data = bytearray(self.assets["booter_load"])
        data[ra.BTL1_PREFIX + 100] ^= 0xFF
        self.rewrite("booter_load", bytes(data))
        self.assertFails("corrupt asset")

    def test_truncated_asset_fails_size(self):
        self.rewrite("booter_unload", self.assets["booter_unload"][:-16])
        self.assertFails("truncated or replaced")

    def test_truncated_to_header_fails(self):
        self.rewrite("scrubber", self.assets["scrubber"][:10])
        self.assertFails("truncated or replaced")

    def test_missing_asset_file(self):
        (self.root / ra.ASSET_DIR / ra.ASSET_FILES["gsp"]).unlink()
        self.assertFails("asset file missing")

    def test_missing_manifest(self):
        (self.root / "manifest.json").unlink()
        self.assertFails("manifest missing")

    def test_missing_notice(self):
        (self.root / "NVIDIA-LICENSE").unlink()
        self.assertFails("notice missing")

    def test_empty_asset(self):
        self.rewrite("bootloader", b"")
        self.assertFails("truncated or replaced")

    # : the generic-bootloader host-request ucode
    def test_generic_bl_truncated(self):
        self.rewrite("generic_bl", self.assets["generic_bl"][:-8])
        self.assertFails("truncated or replaced")

    def test_generic_bl_desc_corrupt(self):
        data = bytearray(self.assets["generic_bl_desc"])
        data[0] ^= 0xFF
        self.rewrite("generic_bl_desc", bytes(data))
        self.assertFails("corrupt asset")

    def test_generic_bl_desc_wrong_geometry(self):
        # a same-size desc with a different blCodeSize, re-pinned and re-manifested so the hash passes and the
        # geometry check is what refuses it
        bad = struct.pack("<6I", 0xFD, 0, 0, 0x400, 0x200, 0x100)   # blCodeSize 0x400 != pinned 0x200
        self.assets["generic_bl_desc"] = bad
        self.pins["generic_bl_desc"] = ra.sha256(bad)
        self.manifest = write_bundle(self.root, self.assets)
        self.assertFails("pinned RM_FLCN_BL_DESC geometry")


# --- wrong chip / tampered manifest ------------------------------------------------------------

class TestWrongChipAndTamper(BundleCase):
    def test_wrong_chip_blob_with_rewritten_manifest(self):
        # A GA102-style booter (different geometry) with a fully rewritten manifest must still
        # be refused: the frozen pins are the anchor, not the manifest.
        foreign = synth_btl1(image_size=4096, os_data_off=2048, os_data_size=1024,
                             app_code_size=1024, patch_loc=2048)
        self.assets["booter_load"] = foreign
        write_bundle(self.root, self.assets)  # manifest now matches the foreign bytes
        self.assertFails("wrong chip")

    def test_manifest_hash_tamper(self):
        m = json.loads((self.root / "manifest.json").read_text())
        m["assets"]["scrubber"]["sha256"] = "0" * 64
        (self.root / "manifest.json").write_text(json.dumps(m))
        self.assertFails("!= manifest")

    def test_manifest_chip_rejected(self):
        m = json.loads((self.root / "manifest.json").read_text())
        m["chip"] = "ga102"
        (self.root / "manifest.json").write_text(json.dumps(m))
        self.assertFails("chip")

    def test_manifest_profile_rejected(self):
        m = json.loads((self.root / "manifest.json").read_text())
        m["profile"] = "570.144"
        (self.root / "manifest.json").write_text(json.dumps(m))
        self.assertFails("profile")

    def test_incomplete_asset_set_rejected(self):
        m = json.loads((self.root / "manifest.json").read_text())
        del m["assets"]["booter_unload"]  # unload chain missing
        (self.root / "manifest.json").write_text(json.dumps(m))
        self.assertFails("boot AND unload")

    def test_elf_without_ada_signature_is_wrong_chip(self):
        elf = synth_elf(drop_ad10x=True)
        self.rewrite("gsp", elf)
        self.pins["gsp"] = ra.sha256(elf)
        m = json.loads((self.root / "manifest.json").read_text())
        m["assets"]["gsp"]["size"] = len(elf)
        m["assets"]["gsp"]["sha256"] = self.pins["gsp"]
        (self.root / "manifest.json").write_text(json.dumps(m))
        self.assertFails(".fwsignature_ad10x absent")


# --- cross-asset lifecycle constraints ----------------------------------------------------------

class TestLifecycleConstraints(BundleCase):
    def test_unload_larger_than_load_rejected(self):
        big = synth_btl1(image_size=4096, os_data_off=2048, os_data_size=1024,
                         app_code_size=1024, patch_loc=2048)
        self.assets["booter_unload"] = big
        write_bundle(self.root, self.assets)
        self.pins = synth_pins(self.assets)
        self.geom["booter_unload"] = ra.parse_btl1(big)
        self.assertFails("would not fit")

    def test_desc_image_size_mismatch_rejected(self):
        desc = synth_desc(2048)  # claims a 2048-byte image; bootloader file is 1024
        self.assets["bootloader_desc"] = desc
        write_bundle(self.root, self.assets)
        self.pins = synth_pins(self.assets)
        self.assertFails("disagree")


# --- BTL1 parser unit cases ----------------------------------------------------------------------

class TestBtl1Parser(unittest.TestCase):
    def test_valid(self):
        geom = ra.parse_btl1(synth_btl1(), expect=SYNTH_GEOM)
        self.assertEqual(geom["ucode_id"], 3)
        self.assertEqual(geom["patch_loc"], 1024)

    def test_nvfw_magic_rejected(self):
        data = bytearray(synth_btl1())
        data[0:4] = struct.pack("<I", 0x000010DE)  # a 570 NVFW image is not a 610 booter
        with self.assertRaisesRegex(ra.AssetError, "not a BTL1"):
            ra.parse_btl1(bytes(data))

    def test_bad_version(self):
        data = bytearray(synth_btl1())
        data[4:8] = struct.pack("<I", 2)
        with self.assertRaisesRegex(ra.AssetError, "version"):
            ra.parse_btl1(bytes(data))

    def test_truncated_header(self):
        with self.assertRaisesRegex(ra.AssetError, "truncated"):
            ra.parse_btl1(synth_btl1()[:40])

    def test_truncated_image(self):
        with self.assertRaisesRegex(ra.AssetError, "truncated or trailing"):
            ra.parse_btl1(synth_btl1()[:-100])

    def test_trailing_data(self):
        with self.assertRaisesRegex(ra.AssetError, "truncated or trailing"):
            ra.parse_btl1(synth_btl1() + b"\0" * 8)

    def test_sig_size_fields_disagree(self):
        data = bytearray(synth_btl1())
        struct.pack_into("<I", data, 8 + 56 + 4, 999)
        with self.assertRaisesRegex(ra.AssetError, "disagree"):
            ra.parse_btl1(bytes(data))

    def test_num_apps_not_one(self):
        with self.assertRaisesRegex(ra.AssetError, "num_apps"):
            ra.parse_btl1(synth_btl1(num_apps=2))

    def test_zero_signatures(self):
        with self.assertRaisesRegex(ra.AssetError, "signature count"):
            ra.parse_btl1(synth_btl1(num_sigs=0))

    def test_signature_size_not_rsa3k(self):
        data = bytearray(synth_btl1())
        # claim 3 signatures while keeping 768 bytes -> 256-byte signatures, not RSA3K
        struct.pack_into("<I", data, 8 + 12 * 4, 3)
        with self.assertRaisesRegex(ra.AssetError, "signature bytes"):
            ra.parse_btl1(bytes(data))

    def test_misaligned_transfer(self):
        with self.assertRaisesRegex(ra.AssetError, "256-aligned"):
            ra.parse_btl1(synth_btl1(os_data_off=1000))

    def test_transfer_outside_image(self):
        with self.assertRaisesRegex(ra.AssetError, "256-aligned"):
            ra.parse_btl1(synth_btl1(image_size=1024, os_data_off=1024, os_data_size=512,
                                     patch_loc=1024))

    def test_patch_outside_dmem(self):
        with self.assertRaisesRegex(ra.AssetError, "patch location"):
            ra.parse_btl1(synth_btl1(patch_loc=512))  # before os_data_off

    def test_patch_overruns_dmem(self):
        with self.assertRaisesRegex(ra.AssetError, "patch location"):
            ra.parse_btl1(synth_btl1(patch_loc=1024 + 512 - 16))

    def test_zero_ucode_id(self):
        with self.assertRaisesRegex(ra.AssetError, "zero ucode/engine"):
            ra.parse_btl1(synth_btl1(ucode_id=0))

    def test_wrong_geometry_expectation(self):
        with self.assertRaisesRegex(ra.AssetError, "wrong chip"):
            ra.parse_btl1(synth_btl1(), expect=dict(SYNTH_GEOM, image_size=56832))

    def test_scrubber_ucode_id_pinned(self):
        scrub = synth_btl1(ucode_id=13)
        ra.parse_btl1(scrub, expect=dict(SYNTH_GEOM, ucode_id=13))
        with self.assertRaisesRegex(ra.AssetError, "wrong chip"):
            ra.parse_btl1(scrub, expect=dict(SYNTH_GEOM, ucode_id=3))


# --- boot descriptor unit cases -------------------------------------------------------------------

class TestBootDesc(unittest.TestCase):
    def test_valid(self):
        d = ra.parse_boot_desc(synth_desc(1024), 1024)
        self.assertEqual(d["manifest_offset"], 512)
        self.assertEqual(d["monitor_data_offset"], 576)
        self.assertEqual(d["monitor_code_offset"], 640)

    def test_wrong_length(self):
        with self.assertRaisesRegex(ra.AssetError, "84"):
            ra.parse_boot_desc(synth_desc(1024)[:-4], 1024)

    def test_wrong_version(self):
        desc = bytearray(synth_desc(1024))
        struct.pack_into("<I", desc, 0, 4)
        with self.assertRaisesRegex(ra.AssetError, "version"):
            ra.parse_boot_desc(bytes(desc), 1024)

    def test_image_size_mismatch(self):
        with self.assertRaisesRegex(ra.AssetError, "disagree"):
            ra.parse_boot_desc(synth_desc(1024), 2048)

    def test_range_outside_image(self):
        desc = bytearray(synth_desc(1024))
        struct.pack_into("<I", desc, 12 * 4, 4096)  # monitorCodeOffset past the image
        with self.assertRaisesRegex(ra.AssetError, "outside the image"):
            ra.parse_boot_desc(bytes(desc), 1024)

    def test_unsupported_feature(self):
        desc = bytearray(synth_desc(1024))
        struct.pack_into("<I", desc, 15 * 4, 1)  # swbromCodeOffset nonzero
        with self.assertRaisesRegex(ra.AssetError, "unsupported"):
            ra.parse_boot_desc(bytes(desc), 1024)


# --- ELF unit cases ---------------------------------------------------------------------------------

class TestElf(unittest.TestCase):
    def test_valid_sections(self):
        s = ra.elf_sections(synth_elf())
        self.assertIn(".fwimage", s)
        self.assertIn(".fwsignature_ad10x", s)
        self.assertEqual(s[".fwversion"][1], len(b"610.43.03\x00"))

    def test_not_elf(self):
        with self.assertRaisesRegex(ra.AssetError, "not an ELF"):
            ra.elf_sections(b"MZ" + b"\0" * 100)

    def test_elf32_rejected(self):
        with self.assertRaisesRegex(ra.AssetError, "ELF64"):
            ra.elf_sections(synth_elf(elf32=True))

    def test_truncated(self):
        with self.assertRaisesRegex(ra.AssetError, "not an ELF"):
            ra.elf_sections(synth_elf()[:32])

    def test_section_past_eof(self):
        data = bytearray(synth_elf())
        shoff, = struct.unpack_from("<Q", data, 0x28)
        struct.pack_into("<Q", data, shoff + 64 + 24, 1 << 40)  # .fwimage offset
        with self.assertRaisesRegex(ra.AssetError, "past the end"):
            ra.elf_sections(bytes(data))

    def test_missing_ada_signature(self):
        s = ra.elf_sections(synth_elf(drop_ad10x=True))
        self.assertNotIn(".fwsignature_ad10x", s)


# --- BINDATA extractor unit cases --------------------------------------------------------------------

def gen_array(label, payload, compressed=False, declared=None, stored=None, crc=None, isize=None):
    size = len(payload) if declared is None else declared
    if compressed:
        comp = zlib.compressobj(9, zlib.DEFLATED, -15)
        raw = comp.compress(payload) + comp.flush()
        trailer = struct.pack("<II", zlib.crc32(payload) if crc is None else crc,
                              len(payload) if isize is None else isize)
        body = raw + trailer
        lines = [f"// COMPRESSION: YES", f"// DATA SIZE (bytes): {size}",
                 f"// COMPRESSED SIZE (bytes): {len(body) if stored is None else stored}"]
    else:
        body = payload
        lines = [f"// COMPRESSION: NO", f"// DATA SIZE (bytes): {size}"]
    tokens = ", ".join(f"0x{b:02x}" for b in body)
    return "\n".join(lines) + f"\nstatic unsigned char BINDATA_LABEL_{label}_data[] = {{ {tokens} }};\n"


class TestBindataExtractor(unittest.TestCase):
    def test_uncompressed(self):
        self.assertEqual(ra.array_bytes(gen_array("X", b"hello"), "X"), b"hello")

    def test_compressed_with_gzip_trailer(self):
        payload = bytes(range(256)) * 4
        self.assertEqual(ra.array_bytes(gen_array("X", payload, compressed=True), "X"), payload)

    def test_bad_crc(self):
        with self.assertRaisesRegex(ra.AssetError, "checksum"):
            ra.array_bytes(gen_array("X", b"hello", compressed=True, crc=123), "X")

    def test_bad_isize(self):
        with self.assertRaisesRegex(ra.AssetError, "checksum or size"):
            ra.array_bytes(gen_array("X", b"hello", compressed=True, isize=99), "X")

    def test_truncated_stream(self):
        text = gen_array("X", b"hello world", compressed=True)
        # drop the last two body bytes (inside the trailer) by rewriting the array shorter
        payload = b"hello world"
        comp = zlib.compressobj(9, zlib.DEFLATED, -15)
        raw = comp.compress(payload) + comp.flush()
        body = (raw + struct.pack("<II", zlib.crc32(payload), len(payload)))[:-2]
        tokens = ", ".join(f"0x{b:02x}" for b in body)
        text = (f"// COMPRESSION: YES\n// DATA SIZE (bytes): {len(payload)}\n"
                f"// COMPRESSED SIZE (bytes): {len(body)}\n"
                f"static unsigned char BINDATA_LABEL_X_data[] = {{ {tokens} }};\n")
        with self.assertRaises(ra.AssetError):
            ra.array_bytes(text, "X")

    def test_size_mismatch_uncompressed(self):
        with self.assertRaisesRegex(ra.AssetError, "header says"):
            ra.array_bytes(gen_array("X", b"hello", declared=99), "X")

    def test_missing_label(self):
        with self.assertRaisesRegex(ra.AssetError, "missing"):
            ra.array_bytes(gen_array("Y", b"x"), "X")

    def test_non_byte_token(self):
        text = "// COMPRESSION: NO\n// DATA SIZE (bytes): 1\n" \
               "static unsigned char BINDATA_LABEL_X_data[] = { 0x123 };\n"
        with self.assertRaisesRegex(ra.AssetError, "non-byte"):
            ra.array_bytes(text, "X")

    # : the COMPLEX_STRUCT extractor (the generic-BL RM_FLCN_BL_DESC literal)
    @staticmethod
    def _complex(values):
        inner = ", ".join(str(v) for v in values[2:])
        return (f"static const RM_FLCN_BL_DESC BINDATA_LABEL_X_data = {{\n"
                f"    {values[0]}, {values[1]},\n    {{ {inner } }}\n}};\n"
                f"// a pointer reference that must not be mistaken for the initializer\n"
                f"const void *p = &BINDATA_LABEL_X_data;\n")

    def test_complex_struct_packs_le(self):
        vals = (0xFD, 0, 0, 0x200, 0x200, 0x100)
        self.assertEqual(ra.complex_struct_bytes(self._complex(vals), "X", 6), struct.pack("<6I", *vals))

    def test_complex_struct_wrong_word_count(self):
        with self.assertRaisesRegex(ra.AssetError, "words"):
            ra.complex_struct_bytes(self._complex((1, 2, 3)), "X", 6)

    def test_complex_struct_missing(self):
        with self.assertRaisesRegex(ra.AssetError, "missing or duplicate"):
            ra.complex_struct_bytes("// nothing here\n", "X", 6)


# --- integration against real artifacts (skip when absent) ---------------------------------------------

@unittest.skipUnless(REAL_BUNDLE.is_dir(), "real runtime bundle not prepared yet")
class TestRealBundle(unittest.TestCase):
    def test_real_bundle_validates_against_frozen_pins(self):
        ra.validate_bundle(REAL_BUNDLE, quiet=True)

    def test_real_bundle_rejects_corruption(self):
        with tempfile.TemporaryDirectory(prefix="gsp610-corrupt-") as tmp:
            copy = Path(tmp) / "bundle"
            import shutil
            shutil.copytree(REAL_BUNDLE, copy)
            target = copy / ra.ASSET_DIR / ra.ASSET_FILES["booter_unload"]
            data = bytearray(target.read_bytes())
            data[-1] ^= 0xFF
            target.write_bytes(bytes(data))
            with self.assertRaises(ra.AssetError):
                ra.validate_bundle(copy, quiet=True)


@unittest.skipUnless((SOURCE_ROOT / "src/nvidia/generated").is_dir(),
                     "pinned 610 source tree not fetched yet")
class TestRealExtraction(unittest.TestCase):
    def test_extraction_reproduces_frozen_pins(self):
        assets = ra.extract_runtime_assets(SOURCE_ROOT)
        for logical, data in assets.items():
            self.assertEqual(ra.sha256(data), ra.ASSET_PINS[logical], logical)

    def test_extraction_deterministic(self):
        a = ra.extract_runtime_assets(SOURCE_ROOT)
        b = ra.extract_runtime_assets(SOURCE_ROOT)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
