#!/usr/bin/env python3
""": deterministic preparation and validation of the AD102 610.43.03 *runtime* bundle.

Scope (plan §2, design "Assets and filesystem"): the complete normal boot AND unload asset
chain for the RTX 4090 (10de:2684, AD102). This module never contacts a GPU and never writes
outside its output directory.

Asset formats (explicit; the 610 profile in plan §4 adds the matching libtinynv parsers):

  booter_load / booter_unload / scrubber
      BTL1 v1 containers assembled from NVIDIA's generated BINDATA archives. Layout, all
      little endian:
        u32 magic = 0x314C5442 ('BTL1'), u32 version = 1
        14 x u32 header: os_code_off, os_code_size, os_data_off, os_data_size, num_apps,
                         app_code_off, app_code_size, app_data_off, app_data_size,
                         patch_loc, ucode_id, engine_id, num_sigs, sig_bytes
        u32 image_size, u32 sig_bytes (again, bound-checked against the header)
        image bytes, then num_sigs x 384-byte RSA3K production signatures
      These are NOT 570 NVFW images (magic 0x000010de); flcn_parse_booter must not be pointed
      at them. Semantics follow 610 kernel_gsp_booter.c: num_apps == 1, the PATCH_SIG selector
      is 0 (NVIDIA asserts this; BTL1 therefore does not store it), and the signature to patch
      at patch_loc is chosen at runtime by fuse version: index = num_sigs - 1 - fuse_ver
      (s_patchBooterUcodeSignature). patch_loc is an absolute image offset inside the DMEM
      (os_data) range; hs dmem address = patch_loc - os_data_off.

  bootloader / bootloader_desc
      The raw GSP RISC-V boot image and its separate 84-byte version-5 descriptor
      (RM_RISCV_UCODE_DESC wire form), from g_bindata_kgspGetBinArchiveGspRmBoot_AD102.c.
      The 610 header struct grows to 92 bytes (bIsSmp/bIsPlicEnabled appended) but the AD102
      archive descriptor is 84 bytes and every offset macuda reads (manifestOffset 32,
      monitorDataOffset 40, monitorCodeOffset 48) is unchanged — see abi-audit.md.

  gsp
      The ga10x GSP-RM ELF from the official NVIDIA 610.43.03 Linux release, verified by hash
      and ELF-section geometry. .fwimage is 80 MiB: the 610 profile must raise
      TINYNV_FW_IMAGE_BOUND (plan §4); this tool only records the size.

  FWSEC-FRTS / FWSEC-SB are board-provided (the RTX 4090's own VBIOS, parsed by vbios.c) and
  are deliberately NOT file assets; the manifest records that authoritative reason.

Every derived byte is pinned below from the hash-verified 610.43.03 source and release
packages. Repeated preparation reproduces every hash; the manifest carries no timestamps.
"""
from __future__ import annotations

import hashlib
import json
import re
import struct
import zlib
from pathlib import Path

SCHEMA = "gsp610-runtime-manifest/1"
PROFILE = "610.43.03"
CHIP = "ad102"
DEVICE = "10de:2684"
ASSET_DIR = "nvidia/ad102/gsp-610.43.03"

BTL1_MAGIC = 0x314C5442
BTL1_VERSION = 1
BTL1_HDR_WORDS = 14
BTL1_PREFIX = 8 + BTL1_HDR_WORDS * 4 + 8  # magic+ver, header, image_size+sig_bytes
SIG_SIZE = 384  # RSA3K production signature
MAX_ASSET = 128 * 1024 * 1024

# --- immutable input pins -----------------------------------------------------------------

SOURCE_TARBALL = {
    "name": "NVIDIA-kernel-module-source-610.43.03.tar.xz",
    "url": "https://download.nvidia.com/XFree86/NVIDIA-kernel-module-source/NVIDIA-kernel-module-source-610.43.03.tar.xz",
    "sha256": "7e118923c7a23edc36114d63273a46e3e04e9af98695a42203e7ac2dfe9fc1dc",
    "license": "GPL-2.0 OR MIT (see SOURCE-COPYING in the bundle)",
}

# generated bindata sources inside the tarball (src/nvidia/generated/)
BINDATA_PINS = {
    "g_bindata_kgspGetBinArchiveBooterLoadUcode_AD102.c":
        "0504f640bc36d03c3b617b1d80cf6eee1c8022cdfffb52ed22405aaa316a1460",
    "g_bindata_kgspGetBinArchiveBooterUnloadUcode_AD102.c":
        "11310bcd02216e0242eb5ccfda6f9f92f6b8c926829793d454a62580a83d11e0",
    "g_bindata_kgspGetBinArchiveGspRmBoot_AD102.c":
        "da110719ed6b680f26f3f7158f6731d1ff20fa859857238e23ac4935bd4cd91c",
    "g_bindata_ksec2GetBinArchiveSecurescrubUcode_AD10X.c":
        "94dbc82226634bf7d1cf6d00691a5a5389d0517c687e8c17aa533f527ad6f312",
    # : the SEC2 generic bootloader the 610 GSP_LOAD_EXEC_GENERIC_BOOTLOADER host request loads (TU102, shared).
    "g_bindata_ksec2GetBinArchiveBlUcode_TU102.c":
        "fc0bc277eeee754ea3071ae0fe8360a52193565bba7c07dcea60b00e8d56d9c8",
}

# logical asset -> filename inside ASSET_DIR
ASSET_FILES = {
    "gsp": "gsp-610.43.03.bin",
    "bootloader": "bootloader-610.43.03.bin",
    "bootloader_desc": "bootloader_desc-610.43.03.bin",
    "booter_load": "booter_load-610.43.03.bin",
    "booter_unload": "booter_unload-610.43.03.bin",
    "scrubber": "scrubber-610.43.03.bin",
    # : the generic-bootloader host-request ucode (image + RM_FLCN_BL_DESC), not part of the boot/unload chain.
    "generic_bl": "generic_bl-610.43.03.bin",
    "generic_bl_desc": "generic_bl_desc-610.43.03.bin",
}

# frozen derived-asset pins (sha256), computed from the pinned source and release packages
ASSET_PINS = {
    "gsp": "572373620a37418f24dc16b5031c39338778c3257e48e8408de9a57291b24f3a",
    "bootloader": "45d6335a8185d6ff13e8d1484cea73f68d3d13aa1c19dc080ed1c3482dd5bac0",
    "bootloader_desc": "83f8f3f1f5f9496e0ea8fc7103c2beb64d0b6de2f5ffe725f13514de5a95fa42",
    "booter_load": "6ee05b20e1dc244c08c052f36ef28469b8d09f4955d1fe33e48a179e11a7fd49",
    "booter_unload": "bf1495f15d00c3a41a6371747510be3d4803b98addf9097cf9e55bb3f7c2f6b3",
    "scrubber": "1cfc78136a4dff4993b7e5a1a09345f6a67d0762555e5bd9660e406831379348",
    # : derived from the pinned TU102 generic-BL bindata (image raw-DEFLATE, desc the packed RM_FLCN_BL_DESC).
    "generic_bl": "f21f1cfbb8fffa8067685314367fc4c50f3f84dd99eda7c19db4292d9269b9d9",
    "generic_bl_desc": "3196d0c5e8817736651f0d888951d4f7aca10dfa4dedd908bac760e8c3335faf",
}

# : the pinned generic-BL geometry (RM_FLCN_BL_DESC): blStartTag, blDmemDescLoadOff, then the image header
# {blCodeOffset, blCodeSize, blDataOffset, blDataSize}. The executor loads blCodeSize bytes at start_tag<<8.
GENERIC_BL_DESC_VALUES = (0xFD, 0, 0, 0x200, 0x200, 0x100)
GENERIC_BL_IMAGE_SIZE = 768
GENERIC_BL_DESC_SIZE = 24

# BTL1 geometry the pinned AD102 images must declare (from HEADER_PROD/PATCH_*/NUM_SIGS).
BTL1_GEOMETRY = {
    "booter_load": dict(image_size=56832, os_code_off=0, os_code_size=256,
                        os_data_off=32256, os_data_size=24576, num_apps=1,
                        app_code_off=256, app_code_size=32000, app_data_off=256, app_data_size=0,
                        patch_loc=32272, ucode_id=3, engine_id=1, num_sigs=2, sig_bytes=768),
    "booter_unload": dict(image_size=40704, os_code_off=0, os_code_size=256,
                          os_data_off=20480, os_data_size=19968, num_apps=1,
                          app_code_off=256, app_code_size=20224, app_data_off=256, app_data_size=0,
                          patch_loc=20496, ucode_id=3, engine_id=1, num_sigs=2, sig_bytes=768),
    "scrubber": dict(image_size=7424, os_code_off=0, os_code_size=256,
                     os_data_off=6656, os_data_size=768, num_apps=1,
                     app_code_off=256, app_code_size=6400, app_data_off=256, app_data_size=0,
                     patch_loc=6656, ucode_id=13, engine_id=1, num_sigs=2, sig_bytes=768),
}

BOOT_DESC_SIZE = 84
BOOTLOADER_IMAGE_SIZE = 36864
GSP_FWIMAGE_SIZE = 84226048
GSP_FWVERSION = b"610.43.03\x00"
GSP_AD10X_SIG_SIZE = 4096

# NVIDIA license shipped with the official 610.43.03 driver release
NVIDIA_LICENSE_SHA = "c692fb14ad4f499c6aee1b9e144e6905148c420e6225b17cb2937366c1e507ad"


class AssetError(ValueError):
    """A preparation or validation failure with a concrete reason."""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --- generated BINDATA extraction ----------------------------------------------------------

def array_bytes(text: str, label: str) -> bytes:
    """Extract one BINDATA_LABEL_<label>_data[] array from a generated NVIDIA .c file.

    Handles NVIDIA's raw-DEFLATE + gzip-trailer storage (no gzip header) and verifies the
    trailer CRC32/ISIZE, the declared size and the byte-token grammar. Derived outputs are
    checked against their frozen pins.
    """
    marker = f"BINDATA_LABEL_{label}_data[]"
    if text.count(marker) != 1:
        raise AssetError(f"missing or duplicate {label}")
    start = text.index(marker)
    window = text[max(0, start - 400):start]
    compression = re.findall(r"COMPRESSION: (YES|NO)", window)
    sizes = re.findall(r"DATA SIZE \(bytes\): (\d+)", window)
    if not compression or not sizes:
        raise AssetError(f"missing storage declaration for {label}")
    expected = int(sizes[-1])
    if not 0 < expected <= MAX_ASSET:
        raise AssetError(f"{label}: declared size {expected} outside extraction limit")
    declaration = re.match(re.escape(marker) + r"\s*=\s*\{([^}]*)\};", text[start:], re.S)
    if not declaration:
        raise AssetError(f"{label}: malformed byte-array declaration")
    tokens = declaration[1].strip().rstrip(",").split(",")
    if any(not re.fullmatch(r"\s*0x[0-9a-fA-F]{2}\s*", x) for x in tokens):
        raise AssetError(f"{label}: non-byte token in generated array")
    raw = bytes(int(item, 16) for item in tokens)
    if compression[-1] == "YES":
        stored = re.findall(r"COMPRESSED SIZE \(bytes\): (\d+)", window)
        if not stored or len(raw) != int(stored[-1]):
            raise AssetError(f"{label}: compressed storage size mismatch")
        inflater = zlib.decompressobj(-15)
        data = inflater.decompress(raw, expected + 1)
        if not inflater.eof or len(inflater.unused_data) != 8 or inflater.unconsumed_tail:
            raise AssetError(f"{label}: truncated, trailing or oversized compressed stream")
        crc, size = struct.unpack("<II", inflater.unused_data)
        if size != expected or crc != zlib.crc32(data):
            raise AssetError(f"{label}: compressed checksum or size mismatch")
    else:
        data = raw
    if len(data) != expected:
        raise AssetError(f"{label}: extracted {len(data)} bytes, header says {expected}")
    return data


def complex_struct_bytes(text: str, label: str, nwords: int) -> bytes:
    """Extract a COMPLEX_STRUCT bindata literal (a C struct initializer, not a byte array) as nwords LE u32.

    : the generic-BL UCODE_DESC is an RM_FLCN_BL_DESC literal. Only integer tokens (hex or decimal) are
    accepted, and the count must match exactly, so a changed struct layout is a refusal rather than a silent pack.
    """
    # the initializer is 'BINDATA_LABEL_<label>_data = {...}'; the bare name also appears as a pointer reference, so
    # anchor on the assignment to pick out exactly the declaration.
    decl = re.compile(r"BINDATA_LABEL_" + re.escape(label) + r"_data\s*=\s*\{(.*?)\};", re.S)
    matches = decl.findall(text)
    if len(matches) != 1:
        raise AssetError(f"missing or duplicate {label} initializer ({len(matches)})")
    m = decl.search(text)
    tokens = re.findall(r"0x[0-9a-fA-F]+|\b\d+\b", m.group(1))
    if len(tokens) != nwords:
        raise AssetError(f"{label}: complex struct has {len(tokens)} words, expected {nwords}")
    return struct.pack(f"<{nwords}I", *(int(t, 0) for t in tokens))


# --- BTL1 container ------------------------------------------------------------------------

def pack_btl1(image: bytes, header9: tuple, patch_loc: int, ucode_id: int, engine_id: int,
              num_sigs: int, signatures: bytes) -> bytes:
    """Pack the BTL1 v1 container used by the 610 booter chain."""
    if len(header9) != 9:
        raise AssetError("header9 must have 9 fields")
    hdr = struct.pack("<14I", *header9, patch_loc, ucode_id, engine_id, num_sigs, len(signatures))
    prefix = struct.pack("<II", BTL1_MAGIC, BTL1_VERSION) + hdr + struct.pack("<II", len(image), len(signatures))
    return prefix + image + signatures


def parse_btl1(data: bytes, name: str = "btl1", expect: dict | None = None) -> dict:
    """Parse and fully bound-check a BTL1 v1 container. Returns the geometry dict.

    `expect` (optional) pins the exact geometry (wrong-chip / wrong-image defence).
    """
    if len(data) < BTL1_PREFIX:
        raise AssetError(f"{name}: truncated BTL1 header ({len(data)} < {BTL1_PREFIX} bytes)")
    magic, version = struct.unpack_from("<II", data, 0)
    if magic != BTL1_MAGIC:
        raise AssetError(f"{name}: not a BTL1 container (magic {magic:#x}; a 570 NVFW image or a "
                         "renamed file cannot be used as a 610 booter)")
    if version != BTL1_VERSION:
        raise AssetError(f"{name}: unsupported BTL1 version {version}")
    words = struct.unpack_from("<14I", data, 8)
    (os_code_off, os_code_size, os_data_off, os_data_size, num_apps,
     app_code_off, app_code_size, app_data_off, app_data_size,
     patch_loc, ucode_id, engine_id, num_sigs, sig_bytes) = words
    image_size, sig_bytes2 = struct.unpack_from("<II", data, 8 + 56)
    if sig_bytes != sig_bytes2:
        raise AssetError(f"{name}: signature size fields disagree ({sig_bytes} vs {sig_bytes2})")
    if len(data) != BTL1_PREFIX + image_size + sig_bytes:
        raise AssetError(f"{name}: file is {len(data)} bytes, header declares "
                         f"{BTL1_PREFIX + image_size + sig_bytes} (truncated or trailing data)")
    if num_apps != 1:
        raise AssetError(f"{name}: num_apps {num_apps} != 1 (610 kernel_gsp_booter.c requires 1)")
    if not 1 <= num_sigs <= 8:
        raise AssetError(f"{name}: implausible signature count {num_sigs}")
    if sig_bytes != num_sigs * SIG_SIZE:
        raise AssetError(f"{name}: {sig_bytes} signature bytes != {num_sigs} x {SIG_SIZE}")
    if not image_size or image_size > MAX_ASSET:
        raise AssetError(f"{name}: implausible image size {image_size}")
    for label, off, size in (("os_code", os_code_off, os_code_size),
                             ("os_data", os_data_off, os_data_size),
                             ("app_code", app_code_off, app_code_size)):
        if not size or off % 256 or size % 256 or off + size > image_size:
            raise AssetError(f"{name}: {label} range {off:#x}+{size:#x} is not a valid 256-aligned "
                             "falcon transfer inside the image")
    if app_data_size and (app_data_off % 256 or app_data_size % 256
                          or app_data_off + app_data_size > image_size):
        raise AssetError(f"{name}: app_data range outside the image or misaligned")
    if not ucode_id or not engine_id:
        raise AssetError(f"{name}: zero ucode/engine id")
    # the signature is patched into DMEM: patch_loc must sit inside the os_data range with
    # room for one full signature (NVIDIA: hsSigDmemAddr = patchLoc - dataOffset)
    if patch_loc < os_data_off or patch_loc + SIG_SIZE > os_data_off + os_data_size:
        raise AssetError(f"{name}: signature patch location {patch_loc:#x} outside the DMEM range "
                         f"{os_data_off:#x}+{os_data_size:#x}")
    geom = dict(image_size=image_size, os_code_off=os_code_off, os_code_size=os_code_size,
                os_data_off=os_data_off, os_data_size=os_data_size, num_apps=num_apps,
                app_code_off=app_code_off, app_code_size=app_code_size,
                app_data_off=app_data_off, app_data_size=app_data_size,
                patch_loc=patch_loc, ucode_id=ucode_id, engine_id=engine_id,
                num_sigs=num_sigs, sig_bytes=sig_bytes)
    if expect is not None:
        for key, want in expect.items():
            if geom.get(key) != want:
                raise AssetError(f"{name}: {key} is {geom.get(key)!r}, pinned {want!r} "
                                 "(wrong chip or wrong image for this profile)")
    return geom


def btl1_image(data: bytes) -> bytes:
    geom = parse_btl1(data)
    return data[BTL1_PREFIX:BTL1_PREFIX + geom["image_size"]]


# --- RISC-V boot descriptor (version-5, 84-byte wire form) ----------------------------------

def parse_boot_desc(desc: bytes, image_size: int, name: str = "bootloader_desc") -> dict:
    """Validate the 84-byte v5 descriptor against its image."""
    if len(desc) != BOOT_DESC_SIZE:
        raise AssetError(f"{name}: descriptor is {len(desc)} bytes, expected {BOOT_DESC_SIZE}")
    f = struct.unpack("<21I", desc)
    if f[0] != 5:
        raise AssetError(f"{name}: descriptor version {f[0]} != 5")
    if f[14] != 1 or f[19] != image_size or f[20] != 0:
        raise AssetError(f"{name}: descriptor flags/image-size disagree with the bootloader image")
    for i in (1, 3, 8, 10, 12):
        off, size = f[i], f[i + 1]
        if not size or off > image_size or size > image_size - off:
            raise AssetError(f"{name}: descriptor range [{i}] {off:#x}+{size:#x} outside the image")
    for i in list(range(5, 8)) + list(range(15, 19)):
        if f[i]:
            raise AssetError(f"{name}: unsupported descriptor feature field [{i}] = {f[i]}")
    return dict(version=f[0], bootloader_offset=f[1], bootloader_size=f[2],
                param_offset=f[3], param_size=f[4], riscv_elf_offset=f[5], riscv_elf_size=f[6],
                manifest_offset=f[8], manifest_size=f[9],
                monitor_data_offset=f[10], monitor_data_size=f[11],
                monitor_code_offset=f[12], monitor_code_size=f[13],
                image_size=f[19])


# --- minimal bounded ELF64 section reader ---------------------------------------------------

def elf_sections(data: bytes, name: str = "elf") -> dict:
    """Return {section_name: (offset, size)} for a little-endian ELF64, fully bounds-checked."""
    if len(data) < 64 or data[:4] != b"\x7fELF":
        raise AssetError(f"{name}: not an ELF image")
    if data[4] != 2 or data[5] != 1:
        raise AssetError(f"{name}: expected little-endian ELF64")
    shoff, = struct.unpack_from("<Q", data, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", data, 0x3A)
    if not shnum or shstrndx >= shnum or shentsize < 64:
        raise AssetError(f"{name}: bad section table header")
    if shoff + shnum * shentsize > len(data):
        raise AssetError(f"{name}: section table outside the file")

    def entry(i):
        base = shoff + i * shentsize
        nm, = struct.unpack_from("<I", data, base)
        off, size = struct.unpack_from("<QQ", data, base + 24)
        return nm, off, size

    _, str_off, str_size = entry(shstrndx)
    if str_off + str_size > len(data):
        raise AssetError(f"{name}: section name table outside the file")
    strtab = data[str_off:str_off + str_size]
    out = {}
    for i in range(shnum):
        nm, off, size = entry(i)
        if nm >= str_size:
            continue
        end = strtab.find(b"\0", nm)
        if end < 0:
            continue
        sname = strtab[nm:end].decode("ascii", errors="replace")
        if off + size > len(data):
            raise AssetError(f"{name}: section {sname} runs past the end of the file")
        out[sname] = (off, size)
    return out


# --- extraction from the pinned source tree --------------------------------------------------

def extract_runtime_assets(source_root: Path) -> dict:
    """Extract every AD102 610 runtime asset (except the GSP ELF) from the pinned source tree.

    Returns {logical_name: bytes}. Every generated source is hash-verified first; every
    derived asset is checked against its frozen pin before being returned.
    """
    gen = source_root / "src/nvidia/generated"
    texts = {}
    for fname, want in BINDATA_PINS.items():
        path = gen / fname
        if not path.is_file():
            raise AssetError(f"pinned generated source missing: {path} (run fetch_inputs.sh)")
        data = path.read_bytes()
        if sha256(data) != want:
            raise AssetError(f"generated source {fname} sha256 mismatch: not the pinned 610.43.03 input")
        texts[fname] = data.decode("ascii")

    out = {}
    for logical, fname, chip_label in (
        ("booter_load", "g_bindata_kgspGetBinArchiveBooterLoadUcode_AD102.c", "BooterLoadUcode_AD102"),
        ("booter_unload", "g_bindata_kgspGetBinArchiveBooterUnloadUcode_AD102.c", "BooterUnloadUcode_AD102"),
        ("scrubber", "g_bindata_ksec2GetBinArchiveSecurescrubUcode_AD10X.c", "SecurescrubUcode_AD10X"),
    ):
        t = texts[fname]
        image = array_bytes(t, "IMAGE_PROD")
        header = array_bytes(t, "HEADER_PROD")
        sigs = array_bytes(t, "SIG_PROD")
        patch_loc, = struct.unpack("<I", array_bytes(t, "PATCH_LOC"))
        patch_sig, = struct.unpack("<I", array_bytes(t, "PATCH_SIG"))
        fuse_ver, engine_id, ucode_id = struct.unpack("<3I", array_bytes(t, "PATCH_META"))
        num_sigs, = struct.unpack("<I", array_bytes(t, "NUM_SIGS"))
        if len(header) != 36:
            raise AssetError(f"{logical}: HEADER_PROD is {len(header)} bytes, expected 36")
        if patch_sig != 0:
            raise AssetError(f"{logical}: PATCH_SIG {patch_sig} != 0 (610 requires selector 0; "
                             "signature choice is fuse-based at runtime)")
        header9 = struct.unpack("<9I", header)
        blob = pack_btl1(image, header9, patch_loc, ucode_id, engine_id, num_sigs, sigs)
        parse_btl1(blob, logical, expect=BTL1_GEOMETRY[logical])
        out[logical] = blob

    boot = texts["g_bindata_kgspGetBinArchiveGspRmBoot_AD102.c"]
    image = array_bytes(boot, "UCODE_IMAGE_PROD")
    desc = array_bytes(boot, "UCODE_DESC_PROD")
    parse_boot_desc(desc, len(image))
    if len(image) != BOOTLOADER_IMAGE_SIZE:
        raise AssetError(f"bootloader image is {len(image)} bytes, pinned {BOOTLOADER_IMAGE_SIZE}")
    out["bootloader"] = image
    out["bootloader_desc"] = desc

    # : the SEC2 generic bootloader for the GSP_LOAD_EXEC_GENERIC_BOOTLOADER host request. The image is a
    # raw-DEFLATE byte array; the descriptor is a COMPLEX_STRUCT (an RM_FLCN_BL_DESC literal), so it is parsed as six
    # integers and packed little-endian, then checked against the pinned geometry.
    gbl = texts["g_bindata_ksec2GetBinArchiveBlUcode_TU102.c"]
    gbl_image = array_bytes(gbl, "UCODE_IMAGE")
    gbl_desc = complex_struct_bytes(gbl, "UCODE_DESC", 6)
    if len(gbl_image) != GENERIC_BL_IMAGE_SIZE:
        raise AssetError(f"generic_bl image is {len(gbl_image)} bytes, pinned {GENERIC_BL_IMAGE_SIZE}")
    if struct.unpack("<6I", gbl_desc) != GENERIC_BL_DESC_VALUES:
        raise AssetError("generic_bl descriptor does not match the pinned RM_FLCN_BL_DESC geometry")
    out["generic_bl"] = gbl_image
    out["generic_bl_desc"] = gbl_desc

    for logical, data in out.items():
        if sha256(data) != ASSET_PINS[logical]:
            raise AssetError(f"derived {logical} differs from its frozen pin — refusing to ship it")
    return out


# --- manifest --------------------------------------------------------------------------------

def build_manifest(assets: dict) -> dict:
    """Deterministic manifest: logical assets, formats, sizes, hashes, geometry, provenance."""
    entries = {}
    for logical in sorted(ASSET_FILES):
        data = assets[logical]
        fmt = {"gsp": "elf64-gsp", "bootloader": "raw-riscv-boot-image",
               "bootloader_desc": "riscv-ucode-desc-v5",
               "generic_bl": "raw-falcon-bl-image", "generic_bl_desc": "rm-flcn-bl-desc"}.get(logical, "btl1-v1")
        entry = {
            "path": f"{ASSET_DIR}/{ASSET_FILES[logical]}",
            "format": fmt,
            "size": len(data),
            "sha256": sha256(data),
        }
        if fmt == "btl1-v1":
            entry["geometry"] = parse_btl1(data, logical)
            entry["signature_selection"] = "index = num_sigs - 1 - fuse_ver (runtime fuse read)"
        entries[logical] = entry
    return {
        "schema": SCHEMA,
        "profile": PROFILE,
        "chip": CHIP,
        "device": DEVICE,
        "purpose": "AD102 610.43.03 normal GSP-RM runtime: complete boot AND unload chain.",
        "inputs": {
            "source_tarball": SOURCE_TARBALL,
            "generated_bindata_sha256": BINDATA_PINS,
            "gsp_elf_provenance": "official NVIDIA-Linux-x86_64-610.43.03.run firmware/gsp_ga10x.bin; "
                                  "verified by hash and ELF-section geometry",
        },
        "assets": entries,
        "lifecycle": {
            "boot": ["fwsec-frts (VBIOS-provided)", "scrubber", "booter_load", "bootloader", "gsp"],
            "unload": ["fwsec-sb (VBIOS-provided)", "booter_unload"],
            "host_requests": ["generic_bl + generic_bl_desc (GSP_LOAD_EXEC_GENERIC_BOOTLOADER, loaded on demand)"],
            "board_provided": {
                "fwsec_frts_sb": "Parsed from the RTX 4090's own VBIOS by src/vbios.c "
                                 "(FWSEC descriptor v3, FRTS/SB commands). Runtime-independent: "
                                 "selecting 610 does not change board ROM contents, so these are "
                                 "not file assets of this bundle.",
            },
        },
        "constraints": {
            "booter_unload_fits_booter_load_slot": True,
            "gsp_fwimage_size": GSP_FWIMAGE_SIZE,
            "note": "610 .fwimage (80 MiB) exceeds the 570 TINYNV_FW_IMAGE_BOUND (64 MiB); "
                    "reservation re-sizing is plan §4, not an asset change.",
        },
        "notices": ["NOTICE.md", "NVIDIA-LICENSE", "SOURCE-COPYING"],
    }


# --- bundle validation ------------------------------------------------------------------------

def validate_bundle(root: Path, pins: dict | None = None,
                    geometry: dict | None = None, elf: dict | None = None,
                    quiet: bool = False) -> dict:
    """Validate a prepared runtime bundle end to end. Raises AssetError on the first problem.

    `pins`/`geometry`/`elf` default to the frozen AD102 610.43.03 constants; tests may pass
    synthetic ones. The manifest is checked against the pins, so a rewritten manifest cannot
    legitimise wrong-chip or corrupted bytes.
    """
    pins = ASSET_PINS if pins is None else pins
    geometry = BTL1_GEOMETRY if geometry is None else geometry
    elf = {"fwimage_size": GSP_FWIMAGE_SIZE, "fwversion": GSP_FWVERSION,
           "ad10x_sig_size": GSP_AD10X_SIG_SIZE} if elf is None else elf
    root = Path(root)
    mpath = root / "manifest.json"
    if not mpath.is_file():
        raise AssetError(f"manifest missing: {mpath}")
    manifest = json.loads(mpath.read_text())
    if manifest.get("schema") != SCHEMA:
        raise AssetError(f"manifest schema {manifest.get('schema')!r} != {SCHEMA!r}")
    for key, want in (("profile", PROFILE), ("chip", CHIP), ("device", DEVICE)):
        if manifest.get(key) != want:
            raise AssetError(f"manifest {key} {manifest.get(key)!r} != {want!r} "
                             "(this validator only accepts the AD102 RTX 4090 610.43.03 profile)")
    entries = manifest.get("assets", {})
    if set(entries) != set(ASSET_FILES):
        raise AssetError(f"manifest asset set {sorted(entries)} != required {sorted(ASSET_FILES)} "
                         "(boot AND unload chain must be complete)")

    blobs = {}
    for logical in sorted(ASSET_FILES):
        entry = entries[logical]
        rel = entry.get("path")
        if rel != f"{ASSET_DIR}/{ASSET_FILES[logical]}":
            raise AssetError(f"{logical}: unexpected manifest path {rel!r}")
        path = root / rel
        if not path.is_file():
            raise AssetError(f"{logical}: asset file missing: {path}")
        data = path.read_bytes()
        if len(data) != entry.get("size"):
            raise AssetError(f"{logical}: file is {len(data)} bytes, manifest says {entry.get('size')} "
                             "(truncated or replaced)")
        digest = sha256(data)
        if digest != entry.get("sha256"):
            raise AssetError(f"{logical}: sha256 {digest} != manifest {entry.get('sha256')} (corrupt asset)")
        if digest != pins[logical]:
            raise AssetError(f"{logical}: sha256 {digest} != frozen pin {pins[logical]} "
                             "(wrong chip, wrong version or tampered manifest)")
        blobs[logical] = data

    # format-level parsing with the pinned geometry
    for logical in ("booter_load", "booter_unload", "scrubber"):
        geom = parse_btl1(blobs[logical], logical, expect=geometry[logical])
        if entries[logical].get("geometry") != geom:
            raise AssetError(f"{logical}: manifest geometry disagrees with the container")
    desc = parse_boot_desc(blobs["bootloader_desc"], len(blobs["bootloader"]))
    if entries["bootloader_desc"].get("size") != BOOT_DESC_SIZE:
        raise AssetError("bootloader_desc: manifest size is not the 84-byte v5 descriptor")

    # : the generic-bootloader host-request ucode. The image is the pinned size; the descriptor is the packed
    # RM_FLCN_BL_DESC and its code size must fit the image (the executor loads blCodeSize bytes).
    if len(blobs["generic_bl"]) != GENERIC_BL_IMAGE_SIZE:
        raise AssetError(f"generic_bl: image is {len(blobs['generic_bl'])} bytes, pinned {GENERIC_BL_IMAGE_SIZE}")
    if len(blobs["generic_bl_desc"]) != GENERIC_BL_DESC_SIZE:
        raise AssetError(f"generic_bl_desc: is {len(blobs['generic_bl_desc'])} bytes, expected {GENERIC_BL_DESC_SIZE}")
    gbl_vals = struct.unpack("<6I", blobs["generic_bl_desc"])
    if gbl_vals != GENERIC_BL_DESC_VALUES:
        raise AssetError("generic_bl_desc: does not match the pinned RM_FLCN_BL_DESC geometry")
    if gbl_vals[3] > len(blobs["generic_bl"]):
        raise AssetError("generic_bl_desc: blCodeSize is larger than the generic_bl image")

    sections = elf_sections(blobs["gsp"], "gsp")
    for required in (".fwimage", ".fwsignature_ad10x", ".fwversion"):
        if required not in sections:
            raise AssetError(f"gsp: required section {required} absent (wrong chip or wrong image)")
    off, size = sections[".fwimage"]
    if size != elf["fwimage_size"]:
        raise AssetError(f"gsp: .fwimage is {size} bytes, pinned {elf['fwimage_size']}")
    off, size = sections[".fwsignature_ad10x"]
    if size != elf["ad10x_sig_size"]:
        raise AssetError(f"gsp: .fwsignature_ad10x is {size} bytes, expected {elf['ad10x_sig_size']}")
    off, size = sections[".fwversion"]
    if blobs["gsp"][off:off + size] != elf["fwversion"]:
        raise AssetError("gsp: .fwversion does not match the pinned runtime version")

    # cross-asset lifecycle constraints
    load_geom = parse_btl1(blobs["booter_load"])
    unload_geom = parse_btl1(blobs["booter_unload"])
    if unload_geom["image_size"] > load_geom["image_size"]:
        raise AssetError("booter_unload image is larger than booter_load's: the unload path reuses "
                         "the boot slot (flcn_unload_prepare) and would not fit")
    if desc["image_size"] != len(blobs["bootloader"]):
        raise AssetError("bootloader descriptor image size disagrees with the bootloader file")

    for notice in manifest.get("notices", []):
        if not (root / notice).is_file():
            raise AssetError(f"required notice missing: {notice}")

    if not quiet:
        print(f"PASS: {root}: {len(blobs)} assets validated (boot + unload), "
              f"profile {PROFILE}, chip {CHIP}, device {DEVICE}")
    return manifest


NOTICE_TEXT = """# AD102 610.43.03 GSP-RM runtime bundle — provenance and notices

Experimental runtime artifact for the RTX 4090 (10de:2684, AD102).
No NVIDIA endorsement is implied.

- GSP-RM ELF (`gsp-610.43.03.bin`): NVIDIA driver firmware, extracted byte-identical from the
  official `NVIDIA-Linux-x86_64-610.43.03.run` member `firmware/gsp_ga10x.bin`
  (sha256 572373620a37418f24dc16b5031c39338778c3257e48e8408de9a57291b24f3a). See NVIDIA-LICENSE.
- Booter load/unload, scrubber, RISC-V bootloader + descriptor: extracted deterministically from
  the generated BINDATA archives of `NVIDIA-kernel-module-source-610.43.03.tar.xz`
  (sha256 7e118923c7a23edc36114d63273a46e3e04e9af98695a42203e7ac2dfe9fc1dc),
  licensed GPL-2.0 OR MIT (SOURCE-COPYING).
- FWSEC-FRTS/SB come from the board's own VBIOS at runtime and are not part of this bundle.

Container formats and validation: tools/gsp610-runtime/runtime_assets.py.
Firmware bytes are gitignored build outputs.
"""
