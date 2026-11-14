#!/usr/bin/env python3
"""Offline probe for the 610 libos log metadata.

Reads ONLY the hash-pinned 610 GSP ELF from the prepared runtime bundle; no GPU, no network. It answers three
questions about decoding the silent-boot libos logs (LOGINIT stops at meta VA 0x4bb0, LOGKRNL at 0x1ac0):

  1. Where are the embedded task ELFs and their LOAD segments? (section headers are zeroed, program headers survive)
  2. What VA does the boot-region string table map at? (anchor: "FB offset %llx fwWprStart %llx")
  3. Are the libosLogMetadata structs present, so a meta VA can be tied to a format string?

Finding (reproducible): the .logging metadata structs are STRIPPED from the shipped image (libos_log.h: "emitted into
the .logging section which is stripped from the image as the final build step"). A whole-image scan for
{filename_ptr, format_ptr, line<20000, argc<20} against the boot-region string table returns ZERO candidates. The
format strings survive, but the meta-VA -> string map does not, so the exact source of a malformed log record cannot
be named from the shipped bundle alone. It would need an unstripped log ELF, which is not part of the pinned runtime
asset set.

Usage: python3 tools/gsp610-runtime/decode_nvlog.py [path/to/gsp-610.43.03.bin]
"""
import struct
import sys
import os

DEFAULT = os.path.join(os.path.dirname(__file__), "..", "..",
                       "cuda-shim/build/gsp610-runtime/bundle/nvidia/ad102/gsp-610.43.03/gsp-610.43.03.bin")
ANCHOR = b"FB offset %llx fwWprStart %llx"


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    d = open(path, "rb").read()
    assert d[:4] == b"\x7fELF", "not the GSP ELF"
    # .fwimage is the first PROGBITS section; locate it by name.
    e_shoff = struct.unpack_from("<Q", d, 0x28)[0]
    e_shentsize, e_shnum, e_shstrndx = struct.unpack_from("<HHH", d, 0x3A)
    secs = [struct.unpack_from("<IIQQQQ", d, e_shoff + i * e_shentsize) for i in range(e_shnum)]
    stro = secs[e_shstrndx][4]
    def nm(n):
        return d[stro + n:d.index(b"\0", stro + n)].decode()
    fw = next(s for s in secs if nm(s[0]) == ".fwimage")
    img = d[fw[4]:fw[4] + fw[5]]
    print(f"fwimage: off={fw[4]:#x} size={fw[5]:#x}")

    # 1. embedded task ELFs (program headers survive; section headers are zeroed)
    print("\nembedded task ELFs (LOAD segments):")
    bases = []
    i = 0
    while True:
        i = img.find(b"\x7fELF", i)
        if i < 0:
            break
        bases.append(i)
        i += 4
    for base in bases:
        h = img[base:base + 64]
        if len(h) < 64:
            continue
        typ, mach, ver, entry, phoff, shoff, flags, ehsz, phentsz, phnum, shentsz, shnum, shstr = \
            struct.unpack_from("<HHIQQQIHHHHHH", h, 16)
        if typ != 2 or phnum == 0 or phnum > 16:
            continue
        loads = []
        for k in range(phnum):
            t, fl, off, va, pa, fsz, msz, al = struct.unpack_from("<IIQQQQQQ", img, base + phoff + k * phentsz)
            if t == 1:
                loads.append((va, off, fsz))
        if loads:
            print(f"  base={base:#09x} entry={entry:#x} " +
                  " ".join(f"LOAD va={va:#x} off={off:#x} sz={fsz:#x}" for va, off, fsz in loads))

    # 2. boot-region string table VA base, anchored on the known format string
    ai = img.find(ANCHOR)
    print(f"\nanchor string {ANCHOR!r} at fwimage offset {ai:#x}")
    # The pointer table referencing it reveals the VA base. Several qwords may coincidentally look like a pointer
    # to the anchor; the GENUINE base is the one referenced by the most qwords (the real string-pointer table).
    refs = {}
    for off in range(0, len(img) - 8, 8):
        v = struct.unpack_from("<Q", img, off)[0]
        b = v - ai
        if b > 0 and (b & 0xFFF) == 0 and 0x1000000 <= v <= 0x6000000:
            refs[b] = refs.get(b, 0) + 1
    if not refs:
        print("  could not anchor the string VA base")
        return
    base = max(refs, key=refs.get)
    print(f"  candidate bases by reference count: {[(hex(b), n) for b, n in sorted(refs.items(), key=lambda kv: -kv[1])[:4]]}")
    print(f"  boot-region string table maps at VA base {base:#x} (string VA {base + ai:#x})")

    # collect boot-region string VAs (the region before the first task ELF)
    first_task = bases[0] if bases else 0x6E000
    strvas = set()
    i = 0
    while i < first_task:
        j = img.find(b"\0", i)
        if j < 0 or j > first_task:
            break
        s = img[i:j]
        if len(s) >= 4 and all(32 <= c < 127 for c in s):
            strvas.add(base + i)
        i = j + 1
    print(f"  {len(strvas)} boot-region strings collected")

    # 3. whole-image scan for libosLogMetadata {filename_ptr, format_ptr, line(u32), argc(u8), lvl(u8), spec(u8)}
    hits = 0
    for off in range(0, len(img) - 24, 8):
        fn, fmt = struct.unpack_from("<QQ", img, off)
        if fmt in strvas:
            line = struct.unpack_from("<I", img, off + 16)[0]
            ac, pl = img[off + 20], img[off + 21]
            if line < 20000 and ac < 20 and pl < 8:
                hits += 1
    print(f"\nlibosLogMetadata candidates referencing the string table: {hits}")
    if hits == 0:
        print("CONCLUSION: the .logging metadata is stripped from the shipped image. The meta-VA -> format-string")
        print("map cannot be reconstructed from the bundle alone; naming the exact stalled libos operation needs the")
        print("unstripped log ELF (.fwlogging / gsp_log.bin), which is not a pinned runtime asset.")
    else:
        print("CONCLUSION: metadata structs ARE present; decode them against the string table to name the stall.")


if __name__ == "__main__":
    main()
