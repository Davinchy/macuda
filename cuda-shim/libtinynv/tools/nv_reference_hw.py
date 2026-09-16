#!/usr/bin/env python3
"""Turn a recorded session's launch snapshot into a reference the C descriptor writer is compared against.

This is the one reference in the suite whose bytes came off a card. Everything else compares libtinynv against the
python oracle, which is two derivations agreeing; these are the descriptors and command buffers a 5090 was actually
handed, captured inside the driver at the moment they existed as bytes, with the command buffer lengths independently
confirmed by the ring entries in the wire trace of the same session.

Addresses are the one thing not compared, and by construction rather than by concession: in the driver they are symbolic
until link time, so the bytes recorded here have zeros where they go, and the C driver is handed them by its caller
rather than deriving them. What is compared is everything the driver computes.

The bits each field claimed are emitted too. Byte equality cannot see a field written to the value the block already
held - writing a zero over a zero - so a transcription that sets the wrong field, or misses one whose value happens to
be zero, would still pass. The claimed mask catches both.

    python tools/nv_reference_hw.py traces/5090-launch2-launch.json test/reference-hw.txt
"""
from __future__ import annotations
import os
import json, pathlib, struct, sys

sys.path.insert(0, os.environ.get("TINYGRAD_SRC", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "tinygrad")))
from tinygrad.runtime.autogen import nv_570 as g  # noqa: E402

PREFIX, QMD_BYTES = "NVCEC0_QMDV05_00_", 384
FIELDS = {**{n[len(PREFIX):]: v for n, v in vars(g).items() if n.startswith(PREFIX) and isinstance(v, tuple)},
          **{f"{n[len(PREFIX):]}_{i}": fn(i) for n, fn in vars(g).items() if n.startswith(PREFIX) and callable(fn)
             for i in range(8)}}

def claimed_mask(field_writes: list[dict]) -> bytes:
  """Which bits of the block the oracle's field writes touched, one bit per bit, in the same order as the bytes."""
  mask = bytearray(QMD_BYTES)
  for w in field_writes:
    for name in w:
      hi, lo = FIELDS[name.upper()]
      for b in range(lo, hi + 1): mask[b // 8] |= 1 << (b % 8)
  return bytes(mask)

# The method numbers and flag words a batch is built from, so a recorded buffer can be named as a sequence of operations
# and rebuilt rather than only stared at. Anything not recognised is emitted as its method number: a batch carrying one
# is reported as not rebuilt, which is the honest thing for the test to say about it.
M = {(0, g.NVC56F_SEM_ADDR_LO): "sem", (0, g.NVC56F_NON_STALL_INTERRUPT): "nonstall",
     (1, g.NVC6C0_SET_OBJECT): "set_object", (1, g.NVC6C0_INVALIDATE_SHADER_CACHES_NO_WFI): "barrier", (1, g.NVC6C0_SET_SHADER_LOCAL_MEMORY_WINDOW_A): "local_window",
     (1, g.NVC6C0_SET_SHADER_SHARED_MEMORY_WINDOW_A): "shared_window", (1, g.NVC6C0_SEND_PCAS_A): "pcas",
     (1, g.NVC6C0_SET_SHADER_LOCAL_MEMORY_A): "slm_at", (1, g.NVC6C0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_A): "slm_size",
     (1, g.NVC6C0_SEND_SIGNALING_PCAS2_B): "pcas2", (4, g.NVC6C0_SET_OBJECT): "set_object4",
     (4, g.NVC6B5_SET_SEMAPHORE_A): "dma_sem", (4, g.NVC6B5_LAUNCH_DMA): "dma_go",
     (4, g.NVC6B5_OFFSET_IN_UPPER): "dma_ends", (4, g.NVC6B5_LINE_LENGTH_IN): "dma_len"}

def nv_flags(reg: str, **kw: str) -> int:
  out = 0
  for name, val in kw.items(): out |= getattr(g, f"{reg}_{name}_{val}".upper()) << getattr(g, f"{reg}_{name}".upper())[1]
  return out

ACQUIRE = nv_flags("NVC56F_SEM_EXECUTE", payload_size="64bit", operation="acq_circ_geq")
RELEASE = nv_flags("NVC56F_SEM_EXECUTE", payload_size="64bit", operation="release", release_wfi="en", release_timestamp="dis")
RELEASE_TS = nv_flags("NVC56F_SEM_EXECUTE", payload_size="64bit", operation="release", release_wfi="en", release_timestamp="en")
BARRIER = nv_flags("NVC6C0_INVALIDATE_SHADER_CACHES_NO_WFI", instruction="true", global_data="true", constant="true")
DMA_COPY = nv_flags("NVC6B5_LAUNCH_DMA", data_transfer_type="non_pipelined", src_memory_layout="pitch", dst_memory_layout="pitch")
DMA_TS = nv_flags("NVC6B5_LAUNCH_DMA", flush_enable="true", semaphore_type="release_four_word_semaphore")
DMA_SIGNAL = nv_flags("NVC6B5_LAUNCH_DMA", flush_enable="true", semaphore_type="release_one_word_semaphore")

def decode(blob: bytes) -> list[str]:
  w = list(struct.unpack(f"<{len(blob) // 4}I", blob))
  ops, i = [], 0
  while i < len(w):
    h = w[i]
    n, subc, mthd = (h >> 16) & 0x1fff, (h >> 13) & 7, (h & 0x1fff) << 2
    vals, name = w[i + 1:i + 1 + n], M.get((subc, mthd))
    i += 1 + n
    if name == "sem":
      # the payload's low word is carried too: most are patched in at link time and read back as zero, but not all
      kind = {ACQUIRE: "wait", RELEASE: "release", RELEASE_TS: "release_ts"}.get(vals[4], f"sem?{vals[4]:#x}")
      ops.append(f"{kind}:{vals[2]}" if not kind.startswith("sem?") else kind)
    elif name == "barrier": ops.append("barrier" if vals[0] == BARRIER else f"barrier?{vals[0]:#x}")
    elif name == "slm_at": ops.append(f"local_memory:{vals[0] << 32 | vals[1]:#x}")   # high half first
    elif name == "slm_size":
      # The size is a derivation from the die's shape, carried here only so the test can check its own against it. It is
      # folded onto the address named a moment ago, because the two methods are one operation.
      if ops and ops[-1].startswith("local_memory:"): ops[-1] += f":{vals[0] << 32 | vals[1]:#x}:{vals[2]:#x}"
      else: ops.append("local_memory_size?")
    elif name == "nonstall": ops.append("nonstall")
    elif name == "set_object": ops.append(f"set_object:{vals[0]:#x}")
    elif name == "set_object4": ops.append(f"set_object4:{vals[0]:#x}")
    elif name in ("local_window", "shared_window"): ops.append(name)
    elif name == "pcas": ops.append("launch")
    elif name == "pcas2": pass                      # the second half of a launch, already named
    elif name == "dma_sem": ops.append(f"dma_sem:{vals[2]}")
    elif name == "dma_ends": ops.append("dma_ends")
    elif name == "dma_len": ops.append(f"dma_len:{vals[0]}")
    elif name == "dma_go":
      f = vals[0]
      last = ops[-1] if ops else ""
      if f == DMA_COPY: ops[-2:] = [f"copy:{last.split(':')[1]}"] if last.startswith("dma_len") else ops[-2:] + ["copy?"]
      elif f == DMA_TS: ops[-1:] = ["dma_timestamp"] if last.startswith("dma_sem") else ops[-1:] + ["dma_ts?"]
      elif f == DMA_SIGNAL: ops[-1:] = [f"dma_signal:{last.split(':')[1]}"] if last.startswith("dma_sem") else ["dma_sig?"]
      else: ops.append(f"dma_go?{f:#x}")
    else: ops.append(f"unknown:{subc}:{mthd:#x}:{n}")
  # a release is always followed by the interrupt that goes with it; name the pair once
  out: list[str] = []
  for o in ops:
    if o == "nonstall" and out and out[-1].startswith("release:"): continue
    out.append(o)
  return out

def main(snap_path: str, out_path: str) -> int:
  snap = json.load(open(snap_path))
  dev = snap["device"]
  # The descriptor bytes have to come from the submit, not from the launch that built them. A launch's release and the
  # chaining onto it are written afterwards, by the release and by the next launch, so the copy taken while encoding the
  # launch is the descriptor before those - and the field list, read at the end, already has them. Taking one from each
  # gives a reference that contradicts itself, which is how this was found.
  final = [q for s in snap["submits"] for q in s["qmds"]]
  if len(final) != len(snap["execs"]):
    print(f"  {len(final)} descriptors were submitted but {len(snap['execs'])} launches were encoded")
    return 1
  lines = [f"# from {snap_path}: descriptors and command buffers a 5090 was handed, captured in the driver",
           f"# tinygrad {snap.get('tinygrad_commit','?')}, {dev['arch']} sm_version {dev['sm_version']}, "
           f"{len(snap['execs'])} launches, {len(snap['submits'])} command buffers",
           f"device slm_per_thread={int(dev['slm_per_thread'])} sass_version={int(dev['sass_version'], 16)} "
           f"shared_window={dev['shared_mem_window']} local_window={dev['local_mem_window']} "
           f"num_gpcs={dev['num_gpcs']} num_tpc_per_gpc={dev['num_tpc_per_gpc']} "
           f"num_sm_per_tpc={dev['num_sm_per_tpc']} max_warps_per_sm={dev['max_warps_per_sm']}"]

  for i, e in enumerate(snap["execs"]):
    tmpl = {k: v for w in e["template_fields"] for k, v in w.items()}
    built = [w for w in e["built_fields"]]
    qmd = bytes.fromhex(final[i])
    # the release slots, in the order they were taken, and whether each carried a timestamp. the field's two named
    # values are four words and two words, and the oracle picks four when a timestamp is wanted.
    releases = []
    for w in built:
      for k, v in w.items():
        if k.startswith("release_structure_size_"): releases.append("ts" if v == 0 else "no")
    chained = any("dependent_qmd0_enable" in w for w in built)
    grid = [next(v for w in built for k, v in w.items() if k == f"grid_{d}") for d in ("width", "height", "depth")]
    block = [next(v for w in built for k, v in w.items() if k == f"cta_thread_dimension{j}") for j in range(3)]

    # A program size that gives back the prefetch size the card was handed. The snapshot does not carry the size itself -
    # the oracle keeps it in a local - and the field is clamped, so this is the one input reconstructed rather than read.
    # It reaches exactly one field, which is compared like every other.
    lines.append(f"launch={i} regs={tmpl['register_count']} shmem={tmpl['shared_memory_size_shifted7'] << 7} "
                 f"slm_per_thread={tmpl['shader_local_memory_high_size_shifted4'] << 4} "
                 f"prog_size={tmpl['program_prefetch_size'] << 8} const0_size={tmpl['constant_buffer_size_shifted4_0']} "
                 f"grid={','.join(map(str, grid))} block={','.join(map(str, block))} "
                 f"releases={','.join(releases) or '-'} chain={int(chained)} "
                 f"cbuf0_dwords={e['cbuf_0_len']} qmd_slot={e['qmd_sz']}")
    lines.append(f"qmd={qmd[:QMD_BYTES].hex()}")
    lines.append(f"claimed={claimed_mask(e['template_fields'] + built).hex()}")
    lines.append(f"cbuf0={qmd[e['qmd_sz']:e['qmd_sz'] + e['cbuf_0_len'] * 4].hex()}")

  for j, s in enumerate(snap["submits"]):
    ops = decode(bytes.fromhex(s["blob"]))
    lines.append(f"cmdbuf={j} cls={s['cls']} dwords={s['dwords']} patches={','.join(map(str, s['patch_offsets'])) or '-'} "
                 f"ops={','.join(ops)}")
    lines.append(f"words={s['blob']}")

  pathlib.Path(out_path).write_text("\n".join(lines) + "\n")
  print(f"wrote {out_path}: {len(snap['execs'])} descriptors and {len(snap['submits'])} command buffers "
        f"({', '.join(str(s['dwords']) for s in snap['submits'])} dwords) from the card")
  return 0

if __name__ == "__main__":
  if len(sys.argv) < 3: raise SystemExit("usage: nv_reference_hw.py <snapshot.json> <out>")
  raise SystemExit(main(sys.argv[1], sys.argv[2]))
