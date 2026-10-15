#!/usr/bin/env python3
"""Make a cubin whose kernel has a very long name, to test the reader against one.

Session A's llama.cpp survey found two kernels out of 8012 whose mangled names are 474 characters: CUB's
DeviceThreeWayPartitionKernel instantiations. The reader stored names in a 256 byte buffer and built the names of the
sections describing each kernel from the stored name, so for those two the parameter base, shared memory size and
register count came from sections that were never found. Nothing in the local corpus is longer than 117 characters, so
nothing here would ever have caught it.

This takes a real cubin and renames its kernel to whatever length is asked for, everywhere the name appears: the section
names that describe the kernel, and the symbol the register count is attributed through. The Mercury sections are left
under the original name on purpose, so the result also proves the reader does not match those by accident.

    python tools/make_long_name_cubin.py in.cubin out.cubin [length]
"""
from __future__ import annotations
import struct, sys

def rd(b, off, fmt): return struct.unpack_from(fmt, b, off)[0]

def cstr(b, off):
  return b[off:b.index(b"\0", off)].decode()

def retarget(b: bytearray, sec_off: int, field: int, value: int):
  struct.pack_into("<Q", b, sec_off + field, value)

def main(src: str, dst: str, length: int = 474):
  b = bytearray(open(src, "rb").read())
  shoff = rd(b, 0x28, "<Q")
  shent, shnum, shstrndx = struct.unpack_from("<HHH", b, 0x3a)

  def sh(i): return shoff + i * shent
  def sname(i):
    strtab = rd(b, sh(shstrndx) + 0x18, "<Q")
    return cstr(b, strtab + rd(b, sh(i), "<I"))

  # the kernel is the one section holding machine code whose name starts with .text.
  kernels = [sname(i)[6:] for i in range(shnum) if rd(b, sh(i) + 4, "<I") == 1 and sname(i).startswith(".text.")]
  if len(kernels) != 1: raise SystemExit(f"{src} has {len(kernels)} kernels; this tool expects exactly one")
  old = kernels[0]
  if length <= len(old): raise SystemExit(f"asked for {length} characters, which is no longer than {old}")
  # keep the original as a prefix so the result is still recognisable, and pad with a legal identifier tail
  new = old + "_" + "L" * (length - len(old) - 1)

  # the sections the reader builds from the kernel's name, and must therefore still find under the new one
  described = {f".text.{old}": f".text.{new}", f".nv.info.{old}": f".nv.info.{new}",
               f".nv.constant0.{old}": f".nv.constant0.{new}", f".nv.shared.{old}": f".nv.shared.{new}"}

  # rebuild the section name table with the new names appended, rather than moving anything already in it
  shstr_off, shstr_size = rd(b, sh(shstrndx) + 0x18, "<Q"), rd(b, sh(shstrndx) + 0x20, "<Q")
  table = bytearray(b[shstr_off:shstr_off + shstr_size])
  added: dict[str, int] = {}
  for want in described.values():
    added[want] = len(table)
    table += want.encode() + b"\0"

  # and the symbol name table, because the register count is attributed through the symbol, not the section
  symtabs = [i for i in range(shnum) if rd(b, sh(i) + 4, "<I") == 2]
  strtab_idx = rd(b, sh(symtabs[0]) + 0x28, "<I") if symtabs else None
  sym_table = None
  if strtab_idx is not None:
    so, ss = rd(b, sh(strtab_idx) + 0x18, "<Q"), rd(b, sh(strtab_idx) + 0x20, "<Q")
    sym_table = bytearray(b[so:so + ss])
    sym_new_off = len(sym_table)
    sym_table += new.encode() + b"\0"

  # append both tables at the end of the file and point their sections at them; nothing already placed has to move
  def append(blob: bytearray) -> int:
    while len(b) % 8: b.append(0)
    at = len(b)
    b.extend(blob)
    return at

  new_shstr = append(table)
  retarget(b, sh(shstrndx), 0x18, new_shstr)
  retarget(b, sh(shstrndx), 0x20, len(table))

  if sym_table is not None:
    new_strtab = append(sym_table)
    retarget(b, sh(strtab_idx), 0x18, new_strtab)
    retarget(b, sh(strtab_idx), 0x20, len(sym_table))
    for s in symtabs:
      off, size = rd(b, sh(s) + 0x18, "<Q"), rd(b, sh(s) + 0x20, "<Q")
      for e in range(size // 24):
        at = off + e * 24
        if cstr(b, new_strtab + rd(b, at, "<I")) == old: struct.pack_into("<I", b, at, sym_new_off)

  renamed = 0
  for i in range(shnum):
    cur = sname(i)
    if cur in described:
      struct.pack_into("<I", b, sh(i), added[described[cur]])
      renamed += 1

  open(dst, "wb").write(b)
  print(f"{dst}: kernel renamed to {len(new)} characters, {renamed} sections retargeted")

if __name__ == "__main__":
  if len(sys.argv) < 3: raise SystemExit(__doc__)
  main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 474)
