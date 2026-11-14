#!/usr/bin/env python3
"""Extract the embedded libos task ELFs from the pinned 610 GSP image and disassemble their
RISC-V LOAD segments. This is an offline diagnostic for the hash-pinned bundle ELF and needs
llvm-objdump with the riscv64 target.

The packed log metadata address is not a task-local constant, so a constant search cannot locate
its call site. Resolving an individual record would require an unstripped log ELF, which is not
present in the official 610.43.03 release.

usage: python3 tools/gsp610-runtime/disasm_tasks.py [gsp-elf] [out-dir]
"""
import os
import struct
import subprocess
import sys

DEFAULT_ELF = os.path.join(os.path.dirname(__file__), "..", "..",
                           "cuda-shim/build/gsp610-runtime/bundle/nvidia/ad102/gsp-610.43.03/gsp-610.43.03.bin")
OBJDUMP = os.environ.get("OBJDUMP", "/opt/homebrew/opt/llvm/bin/llvm-objdump")


def wrap(blob, va, path):
    """minimal ELF64 with one SHF_ALLOC|EXEC section so llvm-objdump disassembles a raw LOAD segment"""
    shstr = b"\0.text\0.shstrtab\0"
    textoff = 64
    shoff = (textoff + len(blob) + 7) & ~7
    stro = shoff + 3 * 64
    eh = struct.pack("<4sBBBBB7xHHIQQQIHHHHHH", b"\x7fELF", 2, 1, 1, 0, 0, 2, 0xF3, 1, va, 0, shoff, 0, 64, 0, 0, 64, 3, 2)
    z = struct.pack("<IIQQQQIIQQ", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    s1 = struct.pack("<IIQQQQIIQQ", 1, 1, 6, va, textoff, len(blob), 0, 0, 4, 0)
    s2 = struct.pack("<IIQQQQIIQQ", 7, 3, 0, 0, stro, len(shstr), 0, 0, 1, 0)
    open(path, "wb").write(eh + blob + b"\0" * (shoff - (textoff + len(blob))) + z + s1 + s2 + shstr)


def main():
    elf = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ELF
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), "..", "..",
                                                            "cuda-shim/build/gsp610-09-diag")
    os.makedirs(out, exist_ok=True)
    d = open(elf, "rb").read()
    assert d[:4] == b"\x7fELF"
    e_shoff = struct.unpack_from("<Q", d, 0x28)[0]
    entsz, num, strndx = struct.unpack_from("<HHH", d, 0x3A)
    secs = [struct.unpack_from("<IIQQQQIIQQ", d, e_shoff + i * entsz) for i in range(num)]
    st = secs[strndx]
    names = d[st[4]:st[4] + st[5]]
    fw = next(s for s in secs if names[s[0]:].split(b"\0")[0] == b".fwimage")
    img = d[fw[4]:fw[4] + fw[5]]
    i, n = 0, 0
    while True:
        i = img.find(b"\x7fELF", i)
        if i < 0:
            break
        base = i
        i += 4
        if img[base + 4] != 2 or img[base + 5] != 1:  # ELFCLASS64, LSB
            continue
        phoff = struct.unpack_from("<Q", img, base + 0x20)[0]
        phentsz, phnum = struct.unpack_from("<HH", img, base + 0x36)
        if phnum == 0 or phnum > 16:
            continue
        loads = []
        for k in range(phnum):
            t, fl, off, va, pa, fsz, msz, al = struct.unpack_from("<IIQQQQQQ", img, base + phoff + k * phentsz)
            if t == 1:
                loads.append((va, off, fsz))
        if not loads:
            continue
        for si, (va, off, fsz) in enumerate(loads):
            blob = img[base + off:base + off + fsz]
            fn = os.path.join(out, f"task{n}_seg{si}")
            open(fn + ".bin", "wb").write(blob)
            if fsz > 0x100000:
                print(f"task{n} seg{si}: fwimage base={base:#x} va={va:#x} size={fsz:#x} -> raw only (disassembly capped at 1 MiB)")
                continue
            wrap(blob, va, fn + ".elf")
            with open(fn + ".dis", "w") as f:
                subprocess.run([OBJDUMP, "-d", "--mattr=+m,+a,+c", fn + ".elf"], stdout=f, stderr=subprocess.DEVNULL)
            print(f"task{n} seg{si}: fwimage base={base:#x} va={va:#x} size={fsz:#x} -> {fn}.dis")
        n += 1
    print(f"{n} embedded task ELFs disassembled into {out}")


if __name__ == "__main__":
    main()
