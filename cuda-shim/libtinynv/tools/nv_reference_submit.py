#!/usr/bin/env python3
"""Emit the exact command-buffer words the oracle builds, so the C encoder can be compared against them byte for byte.

The recording gives the launch's protocol but not its payload: the command buffer lives in host memory the client maps
directly, so its contents cross neither recorder. The bytes are still deterministic, though - the oracle builds them from
the method numbers and a handful of addresses - so they can be produced here and compared, which is an oracle for the
contents even though nothing recorded them.

The two halves check each other. The recorded ring entries carry each command buffer's length in dwords, so a reference
build is not free to be merely self-consistent: the channel setup batches must come out at exactly 22 and 16 dwords, the
lengths the card was actually handed. Both do.

Addresses are placeholders, and deliberately so: a real batch's semaphore address comes from the driver's own
allocations, and fixing them here is what makes the comparison byte-exact rather than approximate.

    python tools/nv_reference_submit.py test/reference-submit.txt
"""
from __future__ import annotations
import os
import pathlib, sys

sys.path.insert(0, os.environ.get("TINYGRAD_SRC", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "tinygrad")))
from tinygrad.runtime.autogen import nv_570 as g  # noqa: E402

# the placeholders the C side uses too. any values would do; these are picked to have distinct halves so a swapped
# high and low word shows up as a mismatch rather than as a coincidence.
SEM_ADDR, SEM_VALUE = 0x1122334455667788, 0x99aabbccddeeff00
LOCAL_WINDOW, SHARED_WINDOW = 0x729300000000, 0x729400000000

def nvm(subc: int, mthd: int, *vals: int, typ: int = 2) -> list[int]:
  """One method header and its values. The header's length field counts dwords, so a 64 bit value counts as two."""
  return [(typ << 28) | (len(vals) << 16) | (subc << 13) | (mthd >> 2), *vals]

def lo_hi(v: int) -> list[int]: return [v & 0xffffffff, v >> 32]   # the semaphore methods take the low half first
def hi_lo(v: int) -> list[int]: return [v >> 32, v & 0xffffffff]   # the shader window methods take the high half first

def nv_flags(reg: str, **kw: str) -> int:
  """The oracle's own encoder, not a re-derivation: a field is (hi, lo) and its named value shifts into lo."""
  out = 0
  for name, val in kw.items():
    out |= getattr(g, f"{reg}_{name}_{val}".upper()) << getattr(g, f"{reg}_{name}".upper())[1]
  return out

def sem(addr: int, value: int, flags: int) -> list[int]:
  return nvm(0, g.NVC56F_SEM_ADDR_LO, *lo_hi(addr), *lo_hi(value), flags)

def wait(addr: int, value: int) -> list[int]:
  return sem(addr, value, nv_flags("NVC56F_SEM_EXECUTE", payload_size="64bit", operation="acq_circ_geq"))

def release(addr: int, value: int) -> list[int]:
  return (sem(addr, value, nv_flags("NVC56F_SEM_EXECUTE", payload_size="64bit", operation="release",
                                    release_wfi="en", release_timestamp="dis"))
          + nvm(0, g.NVC56F_NON_STALL_INTERRUPT, 0x0))

def compute_setup() -> list[int]:
  return (wait(SEM_ADDR, SEM_VALUE)
          + nvm(1, g.NVC6C0_SET_OBJECT, g.BLACKWELL_COMPUTE_B)
          + nvm(1, g.NVC6C0_SET_SHADER_LOCAL_MEMORY_WINDOW_A, *hi_lo(LOCAL_WINDOW))
          + nvm(1, g.NVC6C0_SET_SHADER_SHARED_MEMORY_WINDOW_A, *hi_lo(SHARED_WINDOW))
          + release(SEM_ADDR, SEM_VALUE + 1))

def launch(qmd_addr: int = 0x7f1122330000) -> list[int]:
  """The two methods that run a kernel. Everything else about it is in the descriptor at qmd_addr; see nv_reference_qmd.py."""
  return (nvm(1, g.NVC6C0_SEND_PCAS_A, (qmd_addr >> 8) & 0xffffffff)
          + nvm(1, g.NVC6C0_SEND_SIGNALING_PCAS2_B, g.NVC6C0_SEND_SIGNALING_PCAS2_B_PCAS_ACTION_PREFETCH_SCHEDULE))

def copy_release(addr: int, value: int) -> list[int]:
  """The copy engine's own release, which is not the host semaphore methods the compute path uses: a semaphore address
  set on the engine, then a transfer with nothing to transfer whose only effect is the release. NVCopyQueue.semaphore in
  ops_nv.py. flush_enable is the load-bearing half - it is what makes the engine's writes visible before the payload
  lands, and it is why a host-method release on this channel is not the same thing.

  The payload is one word, so 32 bits, where the compute path releases 64. Both write the same timeline here, which is
  sound only while the counter stays inside 32 bits: a one word release leaves the high half alone, and the high half is
  zero because every 64 bit release below 2**32 writes it zero. run() refuses to go past that rather than wrap quietly."""
  return (nvm(4, g.NVC6B5_SET_SEMAPHORE_A, *hi_lo(addr), value & 0xffffffff)
          + nvm(4, g.NVC6B5_LAUNCH_DMA, nv_flags("NVC6B5_LAUNCH_DMA", flush_enable="true",
                                                 semaphore_type="release_one_word_semaphore")))

def copy_setup() -> list[int]:
  return wait(SEM_ADDR, SEM_VALUE) + nvm(4, g.NVC6C0_SET_OBJECT, g.BLACKWELL_DMA_COPY_B) + release(SEM_ADDR, SEM_VALUE + 1)

# What the card was actually handed, from the ring entries in traces/5090-wire. A batch named here is pinned to a length
# the hardware accepted; a batch not named here is checked against the oracle and against nothing else, and that is worth
# knowing when reading a green test. The recorded session ran no kernel, so the launch is in the second group.
RECORDED_DWORDS = {"compute setup": 22, "copy setup": 16}

def main(out_path: str) -> int:
  batches = {"compute setup": compute_setup(), "copy setup": copy_setup(), "launch": launch(),
             "copy release": copy_release(SEM_ADDR, SEM_VALUE)}
  bad = 0
  lines = [f"# generated by tools/nv_reference_submit.py from the oracle's own method numbers",
           f"# sem_addr {SEM_ADDR:#x} sem_value {SEM_VALUE:#x} local {LOCAL_WINDOW:#x} shared {SHARED_WINDOW:#x}"]
  for name, words in batches.items():
    if (want := RECORDED_DWORDS.get(name)) is not None and len(words) != want:
      print(f"  {name}: {len(words)} dwords, but the card was handed {want}")
      bad += 1
    lines.append(f"{name.replace(' ', '_')} {len(words)} " + " ".join(f"{w:#010x}" for w in words))
  pathlib.Path(out_path).write_text("\n".join(lines) + "\n")
  if not bad:
    print(f"wrote {out_path}: " + ", ".join(f"{n} {len(w)} dwords" + ("" if n in RECORDED_DWORDS else " (no recorded length)")
                                              for n, w in batches.items())
          + " - the pinned ones matching the lengths in the recorded ring entries")
  return bad

if __name__ == "__main__":
  raise SystemExit(1 if main(sys.argv[1] if len(sys.argv) > 1 else "test/reference-submit.txt") else 0)
