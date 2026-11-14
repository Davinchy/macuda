#!/usr/bin/env python3
"""Offline malformed-input tests for the AD102 unlock bundle validator.

Runs without any downloaded NVIDIA artifact: structural fixtures are synthetic
and built from the pinned manifest geometry. The real generated payload is
reproduced locally by the vendored pinned generator (pure Python).

  python3 -m unittest discover -s tools/rtx4090-unlock -p 'test_*.py' -v
"""
from __future__ import annotations

import copy
import hashlib
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "upstream"))

import validate_bundle as vb  # noqa: E402
import generate_v67  # noqa: E402  (vendored, pinned)

MANIFEST = json.loads((HERE / "manifest.json").read_text())
GEO = MANIFEST["booter_geometry"]
DESC_SPEC = MANIFEST["gsp_boot_descriptor"]
LAYOUT = MANIFEST["payload_layout"]
PAYLOAD_NAME = MANIFEST["recipe"]["payload_filename"]


def build_booter(geo=None, image=None, sigs=None):
    """Synthesize a BTL1 bundle consistent with the given geometry."""
    geo = geo or GEO
    image = image if image is not None else bytes((i * 7 + 3) & 0xFF for i in range(geo["image_size"]))
    sigs = sigs if sigs is not None else bytes((i * 13 + 1) & 0xFF for i in range(geo["signature_total_bytes"]))
    hdr = struct.pack(
        "<14I",
        geo["os_code_offset"], geo["os_code_size"], geo["os_data_offset"], geo["os_data_size"],
        geo["application_count"], geo["app_code_offset"], geo["app_code_size"],
        geo["app_data_offset"], geo["app_data_size"], geo["signature_patch_offset"],
        geo["ucode_id"], geo["engine_id"], geo["signature_count"], geo["signature_total_bytes"])
    prefix = struct.pack("<II", int(geo["magic"], 16), geo["bundle_version"]) + hdr
    prefix += struct.pack("<II", len(image), len(sigs))
    return prefix + image + sigs


def build_desc(image_len=None):
    image_len = image_len if image_len is not None else DESC_SPEC["image_size"]
    words = [0] * 21
    words[0] = DESC_SPEC["version"]
    words[1], words[2] = 0, 2048
    words[3], words[4] = 2048, 16384
    words[8], words[9] = DESC_SPEC["manifest_offset"], 2048
    words[10], words[11] = DESC_SPEC["data_offset"], 16384
    words[12], words[13] = DESC_SPEC["code_offset"], image_len - DESC_SPEC["code_offset"]
    words[14] = 1
    words[19] = image_len
    words[20] = 0
    return struct.pack("<21I", *words)


def build_elf(sig_offset=512, sig_size=4096, section_name=b".fwsignature_ad10x"):
    """Minimal ELF64 LSB with a null section, shstrtab and one signature section."""
    strtab = b"\0" + section_name + b"\0.shstrtab\0"
    name_off = 1
    shstr_name_off = 1 + len(section_name) + 1
    body = b"\0" * (sig_offset - 64)
    sig = bytes((i * 11 + 5) & 0xFF for i in range(sig_size))
    str_off = sig_offset + sig_size
    shoff = str_off + len(strtab)
    eh = struct.pack("<4sBBBBB7xHHIQQQIHHHHHH",
                     b"\x7fELF", 2, 1, 1, 0, 0,
                     2, 62, 1, 0, 0, shoff, 0, 64, 0, 0, 64, 3, 2)
    sh_null = struct.pack("<IIQQQQIIQQ", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    sh_sig = struct.pack("<IIQQQQIIQQ", name_off, 1, 0, 0, sig_offset, sig_size, 0, 0, 1, 0)
    sh_str = struct.pack("<IIQQQQIIQQ", shstr_name_off, 3, 0, 0, str_off, len(strtab), 0, 0, 1, 0)
    return eh + body + sig + strtab + sh_null + sh_sig + sh_str


def elf_spec(sig_offset=512, sig_size=4096):
    return {"signature_section": ".fwsignature_ad10x",
            "signature_offset": sig_offset, "signature_size": sig_size}


def flip(data: bytes, offset: int) -> bytes:
    out = bytearray(data)
    out[offset] ^= 0x01
    return bytes(out)


class PayloadTests(unittest.TestCase):
    def setUp(self):
        self.payload = generate_v67.build_ad102_payload()

    def test_valid_payload_matches_pin(self):
        self.assertEqual(hashlib.sha256(self.payload).hexdigest(),
                         MANIFEST["assets"][PAYLOAD_NAME]["sha256"])
        vb.check_payload(self.payload, LAYOUT)

    def test_deterministic_generation(self):
        self.assertEqual(self.payload, generate_v67.build_ad102_payload())

    def test_manifest_layout_matches_vendored_generator(self):
        pinned = {int(o, 16): int(v, 16) for o, v in LAYOUT["patched_words"].items()}
        self.assertEqual(pinned, generate_v67.AD102_PATCHED_WORDS)
        self.assertEqual(LAYOUT["size"], generate_v67.PAYLOAD_SIZE)
        self.assertEqual(int(LAYOUT["fill_dword"], 16), generate_v67.FILL_DWORD)

    def test_wrong_recipe_ga102_rejected(self):
        with self.assertRaises(vb.ValidationError):
            vb.check_payload(generate_v67.build_payload(), LAYOUT)

    def test_truncated_rejected(self):
        with self.assertRaises(vb.ValidationError):
            vb.check_payload(self.payload[:-4], LAYOUT)

    def test_extended_rejected(self):
        with self.assertRaises(vb.ValidationError):
            vb.check_payload(self.payload + b"\x9c\x01\x00\x00", LAYOUT)

    def test_fill_corruption_rejected(self):
        with self.assertRaises(vb.ValidationError):
            vb.check_payload(flip(self.payload, 0x100), LAYOUT)

    def test_every_pinned_word_mutation_rejected(self):
        for off in generate_v67.AD102_PATCHED_WORDS:
            with self.subTest(offset=hex(off)):
                with self.assertRaises(vb.ValidationError):
                    vb.check_payload(flip(self.payload, off), LAYOUT)

    def test_extra_gadget_word_rejected(self):
        extra = bytearray(self.payload)
        extra[0x2000:0x2004] = (0x0000257E).to_bytes(4, "little")  # Zagitta-refactor write entry
        with self.assertRaises(vb.ValidationError):
            vb.check_payload(bytes(extra), LAYOUT)

    def test_all_ones_rejected(self):
        with self.assertRaises(vb.ValidationError):
            vb.check_payload(b"\xff" * LAYOUT["size"], LAYOUT)

    def test_all_zero_rejected(self):
        with self.assertRaises(vb.ValidationError):
            vb.check_payload(b"\x00" * LAYOUT["size"], LAYOUT)


class BooterTests(unittest.TestCase):
    def setUp(self):
        self.booter = build_booter()

    def test_valid_synthetic_booter(self):
        vb.check_booter(self.booter, GEO)

    def test_bad_magic(self):
        with self.assertRaises(vb.ValidationError):
            vb.check_booter(flip(self.booter, 0), GEO)

    def test_bad_version(self):
        with self.assertRaises(vb.ValidationError):
            vb.check_booter(flip(self.booter, 4), GEO)

    def test_truncated(self):
        with self.assertRaises(vb.ValidationError):
            vb.check_booter(self.booter[:-1], GEO)

    def test_extended(self):
        with self.assertRaises(vb.ValidationError):
            vb.check_booter(self.booter + b"\0", GEO)

    def test_header_geometry_mismatch(self):
        for word in range(14):
            with self.subTest(word=word):
                with self.assertRaises(vb.ValidationError):
                    vb.check_booter(flip(self.booter, 8 + 4 * word), GEO)

    def test_trailing_size_mismatch(self):
        with self.assertRaises(vb.ValidationError):
            vb.check_booter(flip(self.booter, 64), GEO)

    def test_patch_outside_dmem(self):
        geo = copy.deepcopy(GEO)
        geo["signature_patch_offset"] = geo["os_data_offset"] + geo["os_data_size"]
        with self.assertRaises(vb.ValidationError):
            vb.check_booter(build_booter(geo), geo)

    def test_misaligned_range(self):
        geo = copy.deepcopy(GEO)
        geo["os_data_offset"] += 4
        geo["signature_patch_offset"] += 4
        with self.assertRaises(vb.ValidationError):
            vb.check_booter(build_booter(geo), geo)

    def test_range_outside_image(self):
        geo = copy.deepcopy(GEO)
        geo["os_data_size"] += 256  # still 256-aligned, now overflows the image
        with self.assertRaises(vb.ValidationError):
            vb.check_booter(build_booter(geo), geo)

    def test_zero_signature_count(self):
        geo = copy.deepcopy(GEO)
        geo["signature_count"] = 0
        with self.assertRaises(vb.ValidationError):
            vb.check_booter(build_booter(geo), geo)

    def test_signature_table_inconsistent(self):
        geo = copy.deepcopy(GEO)
        geo["signature_total_bytes"] = 384  # count says 2
        with self.assertRaises(vb.ValidationError):
            vb.check_booter(build_booter(geo), geo)

    def test_oversized_geometry_rejected(self):
        geo = copy.deepcopy(GEO)
        geo["image_size"] = vb.MAX_BOOTER_IMAGE + 256
        with self.assertRaises(vb.ValidationError):
            vb.check_booter(b"", geo)


class BootDescTests(unittest.TestCase):
    def setUp(self):
        self.desc = build_desc()
        self.image_len = DESC_SPEC["image_size"]

    def check(self, desc, image_len=None):
        vb.check_boot_desc(desc, image_len if image_len is not None else self.image_len, DESC_SPEC)

    def test_valid_synthetic_desc(self):
        self.check(self.desc)

    def test_wrong_length(self):
        for bad in (self.desc[:-1], self.desc + b"\0"):
            with self.assertRaises(vb.ValidationError):
                self.check(bad)

    def test_wrong_version(self):
        with self.assertRaises(vb.ValidationError):
            self.check(flip(self.desc, 0))

    def test_image_size_mismatch(self):
        with self.assertRaises(vb.ValidationError):
            self.check(self.desc, self.image_len - 4)

    def test_word14_mismatch(self):
        with self.assertRaises(vb.ValidationError):
            self.check(flip(self.desc, 14 * 4))

    def test_word20_nonzero(self):
        with self.assertRaises(vb.ValidationError):
            self.check(flip(self.desc, 20 * 4))

    def test_zero_size_range(self):
        words = list(struct.unpack("<21I", self.desc))
        words[11] = 0
        with self.assertRaises(vb.ValidationError):
            self.check(struct.pack("<21I", *words))

    def test_range_overflow(self):
        words = list(struct.unpack("<21I", self.desc))
        words[13] += 4
        with self.assertRaises(vb.ValidationError):
            self.check(struct.pack("<21I", *words))

    def test_offset_field_mismatch(self):
        for word in (8, 10, 12):
            with self.subTest(word=word):
                with self.assertRaises(vb.ValidationError):
                    self.check(flip(self.desc, word * 4))


class GspElfTests(unittest.TestCase):
    def setUp(self):
        self.elf = build_elf()
        self.spec = elf_spec()

    def test_valid_synthetic_elf(self):
        vb.check_gsp_elf(self.elf, self.spec)

    def test_bad_magic(self):
        with self.assertRaises(vb.ValidationError):
            vb.check_gsp_elf(flip(self.elf, 0), self.spec)

    def test_not_elf64(self):
        bad = bytearray(self.elf)
        bad[4] = 1
        with self.assertRaises(vb.ValidationError):
            vb.check_gsp_elf(bytes(bad), self.spec)

    def test_big_endian_rejected(self):
        bad = bytearray(self.elf)
        bad[5] = 2
        with self.assertRaises(vb.ValidationError):
            vb.check_gsp_elf(bytes(bad), self.spec)

    def test_truncated_header(self):
        with self.assertRaises(vb.ValidationError):
            vb.check_gsp_elf(self.elf[:32], self.spec)

    def test_section_table_outside_file(self):
        with self.assertRaises(vb.ValidationError):
            vb.check_gsp_elf(self.elf[:-8], self.spec)

    def test_shstrtab_index_out_of_range(self):
        bad = bytearray(self.elf)
        struct.pack_into("<H", bad, 62, 99)
        with self.assertRaises(vb.ValidationError):
            vb.check_gsp_elf(bytes(bad), self.spec)

    def test_missing_signature_section(self):
        with self.assertRaises(vb.ValidationError):
            vb.check_gsp_elf(build_elf(section_name=b".fwsignature_ga10x"), self.spec)

    def test_wrong_signature_offset(self):
        with self.assertRaises(vb.ValidationError):
            vb.check_gsp_elf(self.elf, elf_spec(sig_offset=513))

    def test_wrong_signature_size(self):
        with self.assertRaises(vb.ValidationError):
            vb.check_gsp_elf(self.elf, elf_spec(sig_size=4095))

    def test_signature_outside_file(self):
        with self.assertRaises(vb.ValidationError):
            vb.check_gsp_elf(self.elf, elf_spec(sig_size=len(self.elf)))

    def test_duplicate_signature_section(self):
        # append a second identical section header and bump e_shnum
        shoff, = struct.unpack_from("<Q", self.elf, 40)
        sh_sig = self.elf[shoff + 64:shoff + 128]
        bad = bytearray(self.elf + sh_sig)
        struct.pack_into("<H", bad, 60, 4)
        with self.assertRaises(vb.ValidationError):
            vb.check_gsp_elf(bytes(bad), self.spec)


class BundleTests(unittest.TestCase):
    """Full validate_bundle() flow on a synthetic-but-pinned mini bundle.

    The 84 MB GSP ELF is replaced by a small synthetic ELF with its own pins in
    a test manifest, so these tests run offline; the real bundle is validated
    by prepare_bundle.py / validate_bundle.py against the real manifest.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="rtx4090-test-")
        self.bundle = Path(self.tmp.name) / "bundle"
        self.bundle.mkdir()
        self.manifest = copy.deepcopy(MANIFEST)
        elf = build_elf()
        self.manifest["gsp_elf"].update(elf_spec())
        self.manifest["assets"]["gsp_ga10x.bin"] = {
            "size": len(elf), "sha256": hashlib.sha256(elf).hexdigest()}
        boot = bytes((i * 3 + 1) & 0xFF for i in range(DESC_SPEC["image_size"]))
        files = {
            "booter_load_ad102.bin": build_booter(),
            "gsp_boot_ad102.bin": boot,
            "gsp_boot_desc_ad102.bin": build_desc(),
            PAYLOAD_NAME: generate_v67.build_ad102_payload(),
            "gsp_ga10x.bin": elf,
        }
        for name, data in files.items():
            (self.bundle / name).write_bytes(data)
            self.manifest["assets"][name] = {
                "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        self.manifest["notices"] = {}
        for name in ("NVIDIA-LICENSE", "UPSTREAM-NOTICE.md"):
            data = ("test notice " + name).encode()
            (self.bundle / name).write_bytes(data)
            self.manifest["notices"][name] = {"sha256": hashlib.sha256(data).hexdigest()}
        (self.bundle / "manifest.json").write_text(json.dumps(self.manifest))

    def tearDown(self):
        self.tmp.cleanup()

    def validate(self):
        return vb.validate_bundle(self.bundle, self.manifest, quiet=True)

    def test_happy_path(self):
        self.assertTrue(self.validate())

    def test_missing_asset(self):
        (self.bundle / "gsp_boot_ad102.bin").unlink()
        with self.assertRaises(vb.ValidationError):
            self.validate()

    def test_missing_notice(self):
        (self.bundle / "NVIDIA-LICENSE").unlink()
        with self.assertRaises(vb.ValidationError):
            self.validate()

    def test_symlink_asset_rejected(self):
        target = self.bundle / PAYLOAD_NAME
        data = target.read_bytes()
        target.unlink()
        outside = Path(self.tmp.name) / "outside.bin"
        outside.write_bytes(data)
        target.symlink_to(outside)
        with self.assertRaises(vb.ValidationError):
            self.validate()

    def test_wrong_size(self):
        path = self.bundle / PAYLOAD_NAME
        path.write_bytes(path.read_bytes()[:-1])
        with self.assertRaises(vb.ValidationError):
            self.validate()

    def test_wrong_hash(self):
        path = self.bundle / PAYLOAD_NAME
        path.write_bytes(flip(path.read_bytes(), 0x100))
        with self.assertRaises(vb.ValidationError):
            self.validate()

    def test_rehashed_corruption_still_fails_structure(self):
        # Forge the manifest hash for a corrupted payload: structure must reject.
        path = self.bundle / PAYLOAD_NAME
        bad = flip(path.read_bytes(), 0xF9E0)  # mailbox marker word
        path.write_bytes(bad)
        self.manifest["assets"][PAYLOAD_NAME]["sha256"] = hashlib.sha256(bad).hexdigest()
        (self.bundle / "manifest.json").write_text(json.dumps(self.manifest))
        with self.assertRaises(vb.ValidationError):
            self.validate()

    def test_rehashed_wrong_recipe_still_fails_structure(self):
        path = self.bundle / PAYLOAD_NAME
        bad = generate_v67.build_payload()  # GA102 image, same size
        path.write_bytes(bad)
        self.manifest["assets"][PAYLOAD_NAME]["sha256"] = hashlib.sha256(bad).hexdigest()
        (self.bundle / "manifest.json").write_text(json.dumps(self.manifest))
        with self.assertRaises(vb.ValidationError):
            self.validate()

    def test_bundled_manifest_mismatch(self):
        (self.bundle / "manifest.json").write_text("{}")
        with self.assertRaises(vb.ValidationError):
            self.validate()

    def test_directory_instead_of_file(self):
        path = self.bundle / "gsp_boot_desc_ad102.bin"
        path.unlink()
        path.mkdir()
        with self.assertRaises(vb.ValidationError):
            self.validate()


class ProvenanceTests(unittest.TestCase):
    def test_vendored_files_match_pin(self):
        for rel, want in MANIFEST["upstream"]["vendored_files_sha256"].items():
            with self.subTest(file=rel):
                path = HERE / "upstream" / rel
                self.assertTrue(path.is_file())
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), want)

    def test_upstream_pin_identity(self):
        self.assertEqual(MANIFEST["upstream"]["commit"],
                         "8855e7bca8ff142204e02301331fd56b26699819")
        self.assertEqual(MANIFEST["upstream"]["repository"],
                         "https://github.com/JustAResearcher/gpu-unlock-ada")

    def test_frozen_input_pins(self):
        # Frozen input expectations documented in inputs.md.
        assets = MANIFEST["assets"]
        self.assertEqual(assets[PAYLOAD_NAME]["sha256"],
                         "44c36b2d6b3424e53e18f3e55db85c833655de071ce66ce42749588a4a7dc8cb")
        self.assertEqual(assets["booter_load_ad102.bin"]["sha256"],
                         "6ee05b20e1dc244c08c052f36ef28469b8d09f4955d1fe33e48a179e11a7fd49")
        self.assertEqual(assets["gsp_boot_ad102.bin"]["sha256"],
                         "45d6335a8185d6ff13e8d1484cea73f68d3d13aa1c19dc080ed1c3482dd5bac0")
        self.assertEqual(assets["gsp_boot_desc_ad102.bin"]["sha256"],
                         "83f8f3f1f5f9496e0ea8fc7103c2beb64d0b6de2f5ffe725f13514de5a95fa42")
        self.assertEqual(assets["gsp_ga10x.bin"]["sha256"],
                         "572373620a37418f24dc16b5031c39338778c3257e48e8408de9a57291b24f3a")
        inputs = {i["filename"]: i["sha256"] for i in MANIFEST["inputs"]}
        self.assertEqual(inputs["NVIDIA-kernel-module-source-610.43.03.tar.xz"],
                         "7e118923c7a23edc36114d63273a46e3e04e9af98695a42203e7ac2dfe9fc1dc")
        # container substitution (2026-11-12): CachyOS archive purged the pinned
        # x86_64_v3 container; the Arch Linux Archive x86_64 container is pinned
        # instead and its members must equal the frozen member pins.
        pkg = MANIFEST["inputs"][1]
        self.assertEqual(inputs["nvidia-utils-610.43.03-1-x86_64.pkg.tar.zst"],
                         "c3367ff68632ac7daf6ca50ef7cbdd1cb5d09f7fe0b26125b312c73e52a34cf4")
        self.assertEqual(pkg["member_pins"]["usr/lib/firmware/nvidia/610.43.03/gsp_ga10x.bin"],
                         assets["gsp_ga10x.bin"]["sha256"])
        self.assertEqual(pkg["member_pins"]["usr/share/licenses/nvidia-utils/LICENSE"],
                         MANIFEST["notices"]["NVIDIA-LICENSE"]["sha256"])
        sub = MANIFEST["input_substitutions"][0]
        self.assertEqual(sub["original_sha256"],
                         "e17e7c92eeffa178b0fe1f40eb17a47095f8fc3ce4a77bf115cf5bd5a4df86e9")
        self.assertEqual(sub["replacement_sha256"], pkg["sha256"])

    def test_allowlist_is_desktop_4090_only(self):
        allow = MANIFEST["device_allowlist"]
        self.assertEqual(len(allow), 1)
        self.assertEqual(allow[0]["vendor"], "0x10de")
        self.assertEqual(allow[0]["device"], "0x2684")
        self.assertEqual(allow[0]["pmc_boot_0_arch"], "0x192")


if __name__ == "__main__":
    unittest.main()
