#!/usr/bin/env python3
"""Check a whole ggml build against the limits libtinynv has, before a GPU run finds one instead.

This is weak evidence and worth being clear about that: it reads what is in the cubins and compares it against what the
driver handles, so it catches a build that has grown past a limit and catches nothing else. Both of the walls this
driver has actually hit on hardware - a kernel reading its launch geometry from a constant bank nothing wrote, and a
relocation against the device runtime's printf - were invisible to a static read of exactly this kind. Run it to rule
out the boring failures, not to gain confidence about the interesting ones.

    python tools/check_cubins.py ../build/obj
"""
from __future__ import annotations
import collections, pathlib, re, struct, sys

MAX_CONSTBUFS = 8      # tinynv_image_t holds this many
MAX_PARAMS = 128       # TINYNV_MAX_PARAMS
KNOWN_UNDEFINED = {"vprintf"}   # the device runtime's printf; see image.c

def cubins(path: pathlib.Path):
  """Every sm_* ELF inside a fatbin embedded in a host object, or the file itself if it is already one."""
  b = path.read_bytes()
  if b[:4] == b"\x7fELF" and struct.unpack_from("<H", b, 0x12)[0] == 190:
    yield b
    return
  at = b.find(struct.pack("<I", 0xBA55ED50))
  if at < 0: return
  _, _, hsize, fatsize = struct.unpack_from("<IHHQ", b, at)
  off, end = at + hsize, at + hsize + fatsize
  while off < end:
    kind, _, ehsize = struct.unpack_from("<HHH", b, off)
    padded, = struct.unpack_from("<Q", b, off + 8)
    payload = b[off + ehsize:off + ehsize + padded]
    if kind == 2 and payload[:4] == b"\x7fELF": yield payload
    off += ehsize + padded

def sections(c: bytes):
  shoff, = struct.unpack_from("<Q", c, 0x28)
  shent, shnum, shstr = struct.unpack_from("<HHH", c, 0x3a)
  so, _ = struct.unpack_from("<QQ", c, shoff + shstr * shent + 0x18)
  out = []
  for i in range(shnum):
    name, typ, _flags, _addr, off, size, link, _info = struct.unpack_from("<IIQQQQII", c, shoff + i * shent)
    out.append(dict(n=c[so + name:c.find(b"\0", so + name)].decode(), typ=typ, off=off, size=size, link=link))
  return out

def undefined_symbols(c: bytes, S):
  for s in S:
    if s["typ"] not in (4, 9): continue
    ent, symtab = (24 if s["typ"] == 4 else 16), S[s["link"]]
    strtab = S[symtab["link"]]
    for r in range(0, s["size"], ent):
      _off, info = struct.unpack_from("<QQ", c, s["off"] + r)
      at = symtab["off"] + (info >> 32) * 24
      st_name, _, _, st_shndx = struct.unpack_from("<IBBH", c, at)
      if st_shndx == 0:
        base = strtab["off"] + st_name
        yield c[base:c.find(b"\0", base)].decode() or "<unnamed>"

def main(roots: list[str]) -> int:
  files = [p for r in roots for p in (sorted(pathlib.Path(r).rglob("*")) if pathlib.Path(r).is_dir()
                                      else [pathlib.Path(r)]) if p.is_file()]
  banks, undef = collections.Counter(), collections.Counter()
  kernels = longest = biggest = most_sections = 0
  seen = 0
  for f in files:
    for c in cubins(f):
      seen += 1
      biggest, S = max(biggest, len(c)), sections(c)
      most_sections = max(most_sections, len(S))
      for s in S:
        if s["typ"] == 1 and s["n"].startswith(".text."):
          kernels += 1
          longest = max(longest, len(s["n"]) - 6)
        if (m := re.fullmatch(r"\.nv\.constant(\d+)(?:\..*)?", s["n"])): banks[int(m.group(1))] += 1
      for u in undefined_symbols(c, S): undef[u] += 1

  print(f"{len(files)} files, {seen} cubins, {kernels} kernels")
  print(f"constant banks in use: {sorted(banks)}  (this driver binds 0-{MAX_CONSTBUFS - 1})")
  print(f"longest kernel name {longest}, largest cubin {biggest / 1e6:.1f} MB, most sections {most_sections}")
  print(f"undefined symbols relocated against: {dict(undef) or 'none'}")

  bad = 0
  for b in banks:
    if b >= MAX_CONSTBUFS: print(f"  OVER: constant bank {b} is past the {MAX_CONSTBUFS} this driver binds"); bad += 1
  for u in undef:
    if u not in KNOWN_UNDEFINED: print(f"  OVER: '{u}' is an undefined symbol the driver will refuse"); bad += 1
  print("nothing here exceeds a limit" if not bad else f"{bad} things would be refused")
  return bad

if __name__ == "__main__":
  raise SystemExit(1 if main(sys.argv[1:] or ["../build/obj"]) else 0)
