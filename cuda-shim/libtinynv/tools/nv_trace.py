"""Record everything the python driver does at the PCI boundary, so the C driver can be built against it with no GPU.

A GSP bring-up is thousands of register reads whose answers come from hardware and, once GSP is alive, from memory the GPU
itself wrote. Replaying those answers in order lets libtinynv boot offline and be diffed write for write against the driver
that is known to work. This records the answers; `pci_replay.c` gives them back.

Usage:  python tools/nv_trace.py <out-prefix> <script.py> [args...]
Writes <prefix>.trace (one line per operation) and <prefix>.blob (the bytes every read returned).

Nothing in tinygrad is modified: the driver classes are patched in this process only.
"""
from __future__ import annotations
import os, struct, sys, time
from tinygrad.helpers import OSX

FNV_OFF, FNV_PRIME, MASK = 0xcbf29ce484222325, 0x100000001b3, (1 << 64) - 1

def fnv1a(b: bytes) -> int:
  h = FNV_OFF
  for x in b: h = ((h ^ x) * FNV_PRIME) & MASK
  return h

class Recorder:
  # a read whose bytes repeat (a poll waiting for hardware) is written once with a count, or the trace of a boot is gigabytes
  INLINE = 32 # writes at most this long are kept whole, so a divergence can be read off the trace itself
  COALESCE_VALUE = True # never turn this off outside selftest(); see the comment in read()

  def __init__(self, prefix: str):
    self.txt, self.blob = open(f"{prefix}.trace", "w"), open(f"{prefix}.blob", "wb")
    self.blob_off, self.pending, self.t0, self.counts = 0, None, time.perf_counter(), {}
    self.txt.write("# tinynv-trace v1\n")

  def _flush(self):
    if self.pending is None: return
    _, line, n = self.pending
    self.txt.write(line + (f" x{n}\n" if n > 1 else "\n"))
    self.pending = None

  def op(self, line: str):
    self.counts[line.split()[0]] = self.counts.get(line.split()[0], 0) + 1
    self._flush()
    self.txt.write(line + "\n")

  def read(self, kind: str, res: str, off: int, data: bytes):
    # A read that repeats identically is a poll waiting on hardware: keep one copy of the bytes and a count, or a boot
    # with a ten second timeout writes millions of lines. Replay hands the same bytes back that many times.
    #
    # THE VALUE IS PART OF THE KEY, AND THAT IS LOAD BEARING. It means a coalesced run holds exactly one value, so the
    # read that first saw a change always begins a new line. The replay checker relies on this: it forgives a driver that
    # served fewer repeats than were recorded, on the grounds that every repeat returned the same bytes. Coalesce by
    # place alone and that stops being true - a run whose last read observed the change could be under-served, and a
    # driver that stopped one read early would decide on stale state with nothing to catch it. selftest() guards this.
    h = fnv1a(data)
    key = (kind, res, off, len(data), h) if self.COALESCE_VALUE else (kind, res, off, len(data))
    self.counts[kind] = self.counts.get(kind, 0) + 1
    if self.pending is not None and self.pending[0] == key:
      self.pending = (key, self.pending[1], self.pending[2] + 1)
      return
    self._flush()
    at = self.blob_off
    self.blob.write(data)
    self.blob_off += len(data)
    self.pending = (key, f"{kind} {res} {off:#x} {len(data)} {at:#x} {h:#x}", 1)

  def write(self, kind: str, res: str, off: int, data: bytes):
    tail = " " + data.hex() if len(data) <= self.INLINE else ""
    self.op(f"{kind} {res} {off:#x} {len(data)} {fnv1a(data):#x}{tail}")

  def note(self, text: str): self.op(f"# {text}")

  def close(self):
    self._flush()
    self.txt.write(f"# {sum(self.counts.values())} ops in {time.perf_counter() - self.t0:.2f}s: "
                   + ", ".join(f"{k}={v}" for k, v in sorted(self.counts.items())) + "\n")
    self.txt.close()
    self.blob.close()

def _pack(fmt: str, val) -> bytes:
  # a value crossing this boundary is an int, a sequence of them, or anything with a buffer (memoryview, array.array)
  if isinstance(val, (bytes, bytearray, memoryview)): return bytes(val)
  if hasattr(val, "tobytes"): return val.tobytes()
  if isinstance(val, (list, tuple)): return struct.pack(f"<{len(val)}{fmt}", *val)
  if isinstance(val, int): return struct.pack(f"<{fmt}", val)
  return bytes(memoryview(val))

def _guard(fn):
  # the recorder must never be why a gpu run dies: a boot that fails halfway leaves hardware to clean up
  def wrapper(*a, **kw):
    try: return fn(*a, **kw)
    except Exception as e: _guard.errors.append(repr(e))  # noqa: BLE001
  wrapper.__name__ = fn.__name__
  return wrapper
_guard.errors = []

# Whether work submission is visible to this recorder, which depends on how the driver submits.
#
# On a remote device the driver lowers the ring entry, the write pointer and the doorbell into a compiled write(2) on the
# socket, so they never pass through MMIOInterface or the PCI device and nothing here sees them. Those three writes are
# the whole of the submission protocol, and their order is the protocol, so a trace taken that way cannot be evidence
# about submission however complete it looks. It says so in its own header rather than leaving that to be discovered.
def submission_is_visible() -> bool:
  from tinygrad.helpers import OSX
  return not OSX  # the remote path is macOS-only today; a local device writes the ring through the recorded interface

# Why this recorder and the wire proxy are complementary, and why both are needed rather than either.
#
# Host memory never crosses the socket. The server hands the client a file descriptor and the client maps it, so every
# access to the GSP message queues, the firmware images and the boot structures is an ordinary memory access that only an
# in-process recorder like this one can see. The wire proxy records everything that does cross the socket - all of the
# register and window access, including the three writes that submit work - and none of the host memory.
#
# So a wire trace cannot replay a boot and a class trace cannot replay a launch. Run both at once: they are recording
# different halves of the same session, not the same thing twice.
#
# The obvious-looking shortcut does not work, and it is written down here so nobody tries it twice. Turning off the
# compiled write(2) would send submission through this recorder, but the doorbell is a "CPU" buffer whose pointer is a
# byte offset into a remote window rather than an address in this process: it is only ever valid because the compiled
# path intercepts it. Without that path the first doorbell write goes to address 0xbb0090 of the recording process.
def install(prefix: str) -> Recorder:
  from tinygrad.runtime.support import system as sysmod
  from tinygrad.runtime.support.hcq import MMIOInterface
  rec = Recorder(prefix)
  rec.note("submission: recorded" if submission_is_visible() else
           "submission: NOT RECORDED - on a remote device the ring entry, write pointer and doorbell are lowered into a "
           "write(2) on the socket and never reach this recorder. run tools/nv_wire_proxy.py alongside for those.")
  # every window the driver can touch, by address: host memory the device reads, and on linux the mapped bars as well,
  # because there both arrive as a plain MMIOInterface over an mmap and only the address tells them apart
  chunks: list[tuple[int, int, int]] = [] # (base, size, id) of host mappings
  bars: list[tuple[int, int, int, int]] = [] # (base, size, bar, offset within the bar)

  def resource_of(mm) -> tuple[str, int] | None:
    for base, size, i in chunks:
      if base <= mm.addr < base + size: return (f"mem:{i}", mm.addr - base)
    for base, size, bar, off in bars:
      if base <= mm.addr < base + size: return (f"bar:{bar}", off + (mm.addr - base))
    return None

  def chunk_base(res: str) -> int: return next(b for b, _, i in chunks if res == f"mem:{i}")

  @_guard
  def _rec_read(kind, res, off, fmt, val): rec.read(kind, res, off, _pack(fmt, val))

  @_guard
  def _rec_write(kind, res, off, fmt, val): rec.write(kind, res, off, _pack(fmt, val))

  # the class in use: the socket backed device on macOS, the sysfs one on linux
  D = sysmod.APLRemotePCIDevice if OSX else sysmod.PCIDevice
  for name, kind in (("read_config", "cfgr"), ("write_config", "cfgw")):
    orig = getattr(D, name)
    def make(orig=orig, kind=kind, name=name):
      def fn(self, offset, *a):
        r = orig(self, offset, *a)
        if name == "read_config": rec.op(f"cfgr {offset:#x} {a[0]} {r:#x}")
        else: rec.op(f"cfgw {offset:#x} {a[1]} {a[0]:#x}")
        return r
      return fn
    setattr(D, name, make())

  orig_reset, orig_bar, orig_sys = D.reset, D.bar_info, D.alloc_sysmem
  def reset(self):
    rec.op("reset")
    return orig_reset(self)
  def bar_info(self, bar):
    base, size = orig_bar(self, bar)
    rec.op(f"bari {bar} {base:#x} {size:#x}")
    return base, size
  def alloc_sysmem(self, size, vaddr=0, contiguous=False):
    view, paddrs = orig_sys(self, size, vaddr, contiguous)
    chunks.append((view.addr, size, len(chunks)))
    rec.op(f"dma mem:{len(chunks) - 1} {size:#x} {len(paddrs)} " + " ".join(f"{p:#x}" for p in paddrs[:4]))
    return view, paddrs
  D.reset, D.bar_info, D.alloc_sysmem = reset, bar_info, alloc_sysmem

  # on linux a mapped bar is a plain MMIOInterface over an mmap, so remember where each one landed to classify accesses
  orig_map_bar = D.map_bar
  def map_bar(self, bar, off=0, addr=0, size=None, fmt='B'):
    view = orig_map_bar(self, bar, off, addr, size, fmt)
    if getattr(view, "ptr", None) is None and not isinstance(view, sysmod.RemoteMMIOInterface):
      bars.append((view.addr, view.nbytes, bar, off))
    return view
  D.map_bar = map_bar

  # device registers: every access is a message to the server, which is exactly what the C driver will issue
  R = sysmod.RemoteMMIOInterface
  orig_get, orig_set = R.__getitem__, R.__setitem__
  def r_get(self, index):
    val = orig_get(self, index)
    sl = index if isinstance(index, slice) else slice(index, index + 1)
    _rec_read("mmior", f"bar:{self.residx}", self.off + (sl.start or 0) * self.el_sz, self.fmt, val)
    return val
  def r_set(self, index, val):
    start = (index.start or 0) * self.el_sz if isinstance(index, slice) else index * self.el_sz
    _rec_write("mmiow", f"bar:{self.residx}", self.off + start, self.fmt, val)
    return orig_set(self, index, val)
  R.__getitem__, R.__setitem__ = r_get, r_set

  # host memory the gpu also writes: the rpc queues and signals. without these the replay cannot answer a poll on GSP.
  orig_mget, orig_mset = MMIOInterface.__getitem__, MMIOInterface.__setitem__
  def m_get(self, index):
    val = orig_mget(self, index)
    if (r := resource_of(self)) is not None:
      sl = index if isinstance(index, slice) else slice(index, index + 1)
      _rec_read("mmior" if r[0].startswith("bar") else "memr", r[0], r[1] + (sl.start or 0) * struct.calcsize(self.fmt), self.fmt, val)
    return val
  def m_set(self, index, val):
    if (r := resource_of(self)) is not None:
      el = struct.calcsize(self.fmt)
      start = (index.start or 0) * el if isinstance(index, slice) else index * el
      _rec_write("mmiow" if r[0].startswith("bar") else "memw", r[0], r[1] + start, self.fmt, val)
    return orig_mset(self, index, val)
  MMIOInterface.__getitem__, MMIOInterface.__setitem__ = m_get, m_set

  return rec

if __name__ == "__main__" and "--selftest" not in sys.argv:
  prefix, script = sys.argv[1], sys.argv[2]
  rec = install(prefix)
  sys.argv = sys.argv[2:]
  try:
    with open(script) as f: code = compile(f.read(), script, "exec")
    exec(code, {"__name__": "__main__", "__file__": script})  # noqa: S102 # pylint: disable=exec-used
  finally:
    if _guard.errors: rec.note(f"recorder errors: {len(_guard.errors)}: {_guard.errors[:3]}")
    rec.close()
    if _guard.errors: print(f"WARNING: recorder hit {len(_guard.errors)} errors, e.g. {_guard.errors[0]}", file=sys.stderr)
    print(f"trace: {prefix}.trace ({os.path.getsize(prefix + '.trace')} B), blob {os.path.getsize(prefix + '.blob')} B", file=sys.stderr)


def _coalescing_check(with_value: bool) -> list[str]:
  """Record a run of identical reads, then a changed one, then a different place. Returns what is wrong, if anything."""
  import tempfile
  bad = []
  with tempfile.TemporaryDirectory() as d:
    Recorder.COALESCE_VALUE = with_value
    r = Recorder(f"{d}/t")
    r.read("mmior", "bar:0", 0x100, b"\x01\x00\x00\x00")  # same place, same value, three times
    r.read("mmior", "bar:0", 0x100, b"\x01\x00\x00\x00")
    r.read("mmior", "bar:0", 0x100, b"\x01\x00\x00\x00")
    r.read("mmior", "bar:0", 0x100, b"\x02\x00\x00\x00")  # the read that sees the change must start a new line
    r.read("mmior", "bar:0", 0x104, b"\x02\x00\x00\x00")  # a different place must too
    r.close()
    Recorder.COALESCE_VALUE = True
    lines = [l.split() for l in open(f"{d}/t.trace") if l.strip() and not l.startswith("#")]

    if len(lines) != 3: bad.append(f"produced {len(lines)} lines, expected 3")
    else:
      if lines[0][-1] != "x3": bad.append(f"three identical reads did not coalesce: {lines[0]}")
      if lines[1][-1] == "x2" or lines[1][2] != "0x100": bad.append(f"a changed value was folded into the run: {lines[1]}")
      if lines[2][2] != "0x104": bad.append(f"a different place was folded in: {lines[2]}")
  return bad

def selftest() -> int:
  """Guard the invariant the replay checker leans on: a coalesced run holds one value, and only one.

  The negative half matters as much as the positive one. Coalescing by place alone still produces a plausible trace, and
  the replay would still pass; it would just have stopped being able to tell a driver that read stale state from one
  that did not. So the check is shown refusing that recorder, not only accepting this one."""
  fails = 0
  if (bad := _coalescing_check(with_value=True)):
    for b in bad: print(f"  {b}")
    fails += len(bad)
  if not _coalescing_check(with_value=False):
    print("  coalescing by place alone passed the same checks: this test cannot tell the two recorders apart")
    fails += 1
  print("recorder: a coalesced run holds one value, a changed value starts a new run, and coalescing by place is refused"
        if not fails else f"recorder: {fails} checks failed")
  return fails

if __name__ == "__main__" and "--selftest" in sys.argv:
  sys.exit(1 if selftest() else 0)
