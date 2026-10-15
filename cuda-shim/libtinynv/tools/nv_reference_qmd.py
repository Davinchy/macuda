#!/usr/bin/env python3
"""Emit the exact kernel descriptor the oracle builds for a real cubin, so the C writer can be compared against it.

This is the oracle's own code, not a re-derivation of it. `NVProgramData` and `QMD` come straight out of tinygrad and
run here against a stub device: nothing in either class touches hardware while it builds a descriptor, so the bytes it
produces for a given cubin and a given set of device properties are the bytes it would produce on the card.

That matters more here than it did for the command buffers. A command buffer is a handful of method numbers; a
descriptor is a hundred and thirty fields at bit positions that mostly do not fall on byte boundaries, several of them
named for a shift they are not given. Comparing against the oracle's own output is the only way to be sure the C writer
sets the same fields to the same values and puts them in the same bits.

The comparison is deliberately not handed the answers. The C side reads the same cubin with its own reader and derives
the register count, the shared memory size and the program size itself, so a divergence in the reader shows up here too.
What this file does carry is the device's own properties - local memory per thread, sass version, the two windows -
because those come from the card rather than from the kernel.

    python tools/nv_reference_qmd.py ../spike/vecadd.sm120.cubin test/reference-qmd.txt
"""
from __future__ import annotations
import os
import pathlib, sys

sys.path.insert(0, os.environ.get("TINYGRAD_SRC", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "tinygrad")))
from tinygrad.dtype import dtypes                          # noqa: E402
from tinygrad.device import TinyELF                        # noqa: E402
from tinygrad.helpers import round_up                      # noqa: E402
from tinygrad.runtime.autogen import nv_570 as g           # noqa: E402
from tinygrad.runtime import ops_nv                        # noqa: E402

# The card's own numbers. Fixed here rather than queried so this runs with no GPU; the combined recording session is
# what checks them against the card.
#
# TWO ARCHITECTURES, because tinygrad's QMD picks its version - and therefore its field table, its grid field names and
# whether a release carries payload64b - from the device's compute class alone. Handing the stub an Ampere class is
# enough to make the oracle emit a v3 descriptor, which is how a reference for a 256 byte descriptor can be generated
# with no Ampere card present.
#
# The Ampere sm_version is a PLACEHOLDER, and marked as one. A real card reports it from the RM control our own boot
# issues in tinynv_gsp_open_client, which is a stage no recording here reaches yet, so there is no measured value to
# use. It feeds exactly one descriptor field (SASS_VERSION) and the reference header states it, so the C side is told
# the same number and the comparison stays a comparison of ENCODING. Replace it when a client-stage capture exists.
SLM_PER_THREAD = 0x40
ARCHS = {
  "blackwell": (lambda: g.BLACKWELL_COMPUTE_B, 0xa04),  # measured on the 5090
  "ampere":    (lambda: g.AMPERE_COMPUTE_B,    0x806),  # sm_version is a placeholder, see above
}
ARCH = os.environ.get("REFQMD_ARCH", "blackwell")
if ARCH not in ARCHS: raise SystemExit(f"REFQMD_ARCH={ARCH}: expected one of {', '.join(ARCHS)}")
COMPUTE_CLASS, SM_VERSION = ARCHS[ARCH][0](), ARCHS[ARCH][1]
SHARED_WINDOW, LOCAL_WINDOW = 0x729400000000, 0x729300000000
# placeholders with distinct halves, so a swapped high and low word shows up as a mismatch rather than a coincidence
# Every one is in range for the field that holds it, which is checked below rather than assumed: a descriptor holds an
# address in two halves, and the upper half of several of them is narrower than 32 bits.
PROGRAM_ADDR, CBUF0_ADDR = 0x7f1122330000, 0x7f4455660000
SEM_PAYLOAD = 0x99aabbccddeeff00
# THE RELEASE SEMAPHORE ADDRESS IS NARROWER ON AMPERE, and by a lot: v3 splits it 8 bits upper + 32 lower, so a
# descriptor can only release a semaphore in the low 1 TB, where v5 splits it 25 + 32 and reaches 57 bits. The
# placeholder below is 47 bits and fits Blackwell comfortably; on Ampere it does not fit at all, and the check further
# down caught it rather than letting this file compare the C writer against a silently truncated address.
#
# This is not just a detail of the reference. A driver that places a release semaphore above 2^40 on an Ampere card
# gets no error: the address is truncated into the field and the release lands somewhere else, which is a hang with no
# message. tinygrad's own VA allocator is based at 64 GB with a 16 TB span, so that is reachable.
SEM_ADDR = {"blackwell": 0x7f8899aabbcc, "ampere": 0x99aabbccdd}[ARCH]
NEXT_QMD = 0x8899aabb00   # the chain pointer is 32 bits holding the address shifted by eight, so it reaches 1 TB and no further
GRID, BLOCK = (17, 3, 2), (256, 2, 1)

class StubIface: compute_class = COMPUTE_CLASS
class StubDev:
  """Everything NVProgramData reads off a device while building a descriptor, and nothing else."""
  iface = StubIface()
  renderer = object()   # not a NAKRenderer: this is the cubin path, which is the one libtinynv implements
  slm_per_thread, sass_version = SLM_PER_THREAD, ((SM_VERSION & 0xf00) >> 4) | (SM_VERSION & 0xf)
  shared_mem_window, local_mem_window = SHARED_WINDOW, LOCAL_WINDOW
  def __init__(self): self.local_memory_asked = 0
  def _ensure_has_local_memory(self, required): self.local_memory_asked = required

def capture_writes():
  """Record every (field, value) the oracle sets, and give an address the same truncation the real path gives it.

  An address never reaches `QMD.write` as an integer in the driver: it is a symbolic value, patched into the block at
  link time after a cast to the widest type the field can hold, which is what cuts a 64 bit address down to the 32 bit
  half the field wants. Handing the same call a plain integer skips the cast and it is rejected for not fitting. So the
  cast is done here, by the same rule, and each truncation is recorded rather than performed quietly.
  """
  seen: list[dict] = []
  cuts: list[tuple[str, int, int]] = []
  orig = ops_nv.QMD.write
  def write(self, **kw):
    out = {}
    for k, v in kw.items():
      hi, lo = ops_nv.QMD.fields[self.pref][k.upper()]
      width = hi - lo + 1
      if isinstance(v, int) and v >> width:
        bits = max((n for n in (64, 32, 16, 8) if n <= width), default=width)
        cuts.append((k, v, bits))
        v &= (1 << bits) - 1
      out[k] = v
    seen.append(dict(out))
    return orig(self, **out)
  ops_nv.QMD.write = write
  return seen, cuts

def kernel_name(lib: bytes) -> str:
  """The one kernel in the cubin, by its .text section. A cubin with several would need one named on the command line."""
  from tinygrad.runtime.support.elf import elf_loader
  _img, sections, _relocs = elf_loader(lib, force_section_align=128)
  names = [s.name[len(".text."):] for s in sections if s.name.startswith(".text.")]
  if len(names) != 1: raise SystemExit(f"expected one kernel, found {len(names)}: {names[:4]}")
  return names[0]

def program_size(lib: bytes, name: str) -> int:
  from tinygrad.runtime.support.elf import elf_loader
  img, sections, _ = elf_loader(lib, force_section_align=128)
  for s in sections:
    if s.name == f".text.{name}": return s.header.sh_size
  return len(img)

def main(cubin_path: str, out_path: str) -> int:
  lib = pathlib.Path(cubin_path).read_bytes()
  name = kernel_name(lib)
  prog_size = program_size(lib, name)
  writes, cuts = capture_writes()

  # A signature only sets how many bytes of arguments follow the driver parameters, which is what kernargs_size is for.
  # Three pointers and a count is the shape of every kernel this driver has to launch first.
  sig = ((None, 0, dtypes.uint64, ()), (None, 1, dtypes.uint64, ()), (None, 2, dtypes.uint64, ()), ("n", 3, dtypes.int, ()))
  dev = StubDev()
  data = ops_nv.NVProgramData(dev, TinyELF(lib=lib, name=name, target=None, signature=sig))
  template = bytes(data.qmd.mv)

  # what the oracle derived, recovered from the fields it set. shared memory is rounded to 128 and local memory to 32,
  # so shifting them back is lossless; the check is that the C side derives the same numbers from the same cubin.
  fields = {k: v for w in writes for k, v in w.items()}

  # Several fields are named for a shift on one architecture and not on the other - shared_memory_size_shifted7 against
  # shared_memory_size - so the value is read by stem and the shift taken from whichever name the oracle used, rather
  # than by a name that is only right for one of them.
  def by_stem(stem: str):
    for k, v in fields.items():
      if k == stem: return v, 0
      if k.startswith(stem + "_shifted"): return v, int(k[len(stem) + len("_shifted"):])
    raise KeyError(f"the oracle set no field with stem {stem!r}")

  shmem_raw, shmem_sh = by_stem("shared_memory_size")
  shmem = shmem_raw << shmem_sh
  regs = fields["register_count"] if "register_count" in fields else fields["register_count_v"]
  slm_raw, slm_sh = by_stem("shader_local_memory_high_size")
  if slm_raw << slm_sh != SLM_PER_THREAD:
    return err(f"local memory per thread came back as {slm_raw << slm_sh:#x}")
  if fields["program_prefetch_size"] != min(prog_size >> 8, 0x1ff):
    return err(f"prefetch size {fields['program_prefetch_size']:#x} does not match a {prog_size:#x} byte program")

  # then the three things that happen to a descriptor once memory exists: the launch's own shape and addresses, a
  # semaphore release, and being chained onto by the launch that follows. Each is cumulative, so a divergence names the
  # step that introduced it rather than only the end state.
  qmd = ops_nv.QMD(dev, bytearray(template))
  qmd.write(**dict(zip(qmd.grid, GRID)), **{f"cta_thread_dimension{j}": b for j, b in enumerate(BLOCK)})
  qmd.set_program_addr(PROGRAM_ADDR)
  qmd.set_constant_buf_addr(0, CBUF0_ADDR)
  launched = bytes(qmd.mv)
  if not qmd.set_release(SEM_ADDR, SEM_PAYLOAD): return err("the descriptor had no free release slot")
  slot = 0 if qmd.read("release0_enable") else 1
  released = bytes(qmd.mv)
  qmd.write(dependent_qmd0_pointer=NEXT_QMD >> 8, dependent_qmd0_action=1, dependent_qmd0_prefetch=1, dependent_qmd0_enable=1)
  chained = bytes(qmd.mv)

  # Every cast the address path performed, checked to be the intended split and nothing more. A lower half keeps the low
  # 32 bits because the upper half carries the rest; an upper half that lost anything means the address does not fit the
  # descriptor at all, and this reference would be comparing against a silently wrong number.
  for k, v, bits in cuts:
    if "_upper" in k: return err(f"{k} lost bits: {v:#x} does not fit its field even after the cast")
    if "_lower" in k and (bits != 32 or v & 0xffffffff != v & ((1 << bits) - 1)):
      return err(f"{k} was cut to {bits} bits, not the low 32 the upper half complements")
    if "_lower" not in k and "_upper" not in k: return err(f"{k}={v:#x} does not fit its {bits} bit field")

  qmd_sz = round_up(ops_nv.QMD(dev).sz * 4, 256)
  lines = [
    f"# generated by tools/nv_reference_qmd.py from tinygrad's own NVProgramData and QMD",
    f"# cubin {cubin_path} kernel {name}",
    f"device arch {ARCH} slm_per_thread {SLM_PER_THREAD:#x} sm_version {SM_VERSION:#x} shared_window {SHARED_WINDOW:#x} "
    f"local_window {LOCAL_WINDOW:#x}",
    f"derived regs {regs} shmem {shmem:#x} prog_size {prog_size:#x} const0_size {data.constbufs[0][1]:#x} "
    f"kernargs_size {data.kernargs_size:#x} qmd_slot {qmd_sz:#x} cbuf0_dwords {len(data.cbuf_0)}",
    f"launch grid {GRID[0]} {GRID[1]} {GRID[2]} block {BLOCK[0]} {BLOCK[1]} {BLOCK[2]} program_addr {PROGRAM_ADDR:#x} "
    f"cbuf0_addr {CBUF0_ADDR:#x}",
    f"release addr {SEM_ADDR:#x} payload {SEM_PAYLOAD:#x} slot {slot} next_qmd {NEXT_QMD:#x}",
    f"template {template.hex()}",
    f"launched {launched.hex()}",
    f"released {released.hex()}",
    f"chained {chained.hex()}",
    f"cbuf0 {b''.join(int(x).to_bytes(4, 'little') for x in data.cbuf_0).hex()}",
  ]
  pathlib.Path(out_path).write_text("\n".join(lines) + "\n")
  print(f"wrote {out_path}: {name}, {regs} registers, {shmem:#x} shared, {prog_size:#x} of code, "
        f"{len(data.cbuf_0)} dwords of constant buffer 0, release in slot {slot}; "
        f"{len(cuts)} addresses split across their two fields")
  return 0

def err(msg: str) -> int:
  print(f"  {msg}")
  return 1

if __name__ == "__main__":
  if len(sys.argv) < 3: raise SystemExit("usage: [REFQMD_ARCH=blackwell|ampere] nv_reference_qmd.py <cubin> <out>")
  raise SystemExit(main(sys.argv[1], sys.argv[2]))
