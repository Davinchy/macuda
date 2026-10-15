"""Capture the ground truth for a kernel launch: the bytes the oracle builds, and the inputs that produced them.

Neither recorder can supply this on its own. The kernel descriptor and the command buffer are assembled in host memory
the client maps directly, so the wire proxy never sees them; and they are written through paths the class recorder does
not patch, so it does not see them either. What both recorders do capture is the launch's protocol - the ring entry, the
write pointer, the doorbell - which pins the command buffer's length and nothing about its contents.

So this records the contents at the only place they exist as bytes: inside the driver, as it builds them. Three hooks,
none of which changes what the driver does.

  QMD.write      every (field, value) the oracle sets, in order. This is the one that matters most. libtinynv's
                 descriptor writer is a transcription of which fields a launch sets and to what; the bit positions come
                 from the generator and cannot be wrong, but the choice of fields is a transcription and could be. This
                 records what the oracle actually chose, on this card, for this kernel.
  HWQueue.submit the command buffer as bytes, with the offsets whose values are addresses filled in only at link time.
                 Comparing against this means comparing everything the driver computes, with the addresses - which the C
                 side is handed rather than deriving - left out of the comparison by construction.
  exec           the launch's own inputs: grid, block, and the program's properties read out of its elf.

Run it under both recorders, which is the whole point of running it at all:

    python tools/nv_wire_proxy.py --trace traces/x-wire --connect $TINYGPU_SOCK --listen /tmp/proxy.sock &
    APL_REMOTE_SOCK=/tmp/proxy.sock python tools/nv_trace.py traces/x-class tools/nv_snapshot_launch.py traces/x-launch.json

The three files then describe one session: what crossed the socket, what happened in host memory, and what the launch
was built out of.
"""
from __future__ import annotations
import json, os, sys, time

OUT = sys.argv[1] if len(sys.argv) > 1 else "launch-snapshot.json"
snap: dict = {"qmds": [], "submits": [], "execs": [], "programs": [], "errors": []}

def guard(fn):
  # a snapshot must never be why a gpu run dies: this session is expensive and the recording is worth more than the json
  def wrap(*a, **kw):
    try: return fn(*a, **kw)
    except Exception as e: snap["errors"].append(repr(e))  # noqa: BLE001
  return wrap

def install():
  from tinygrad.runtime import ops_nv
  from tinygrad.runtime.support.hcq2 import HWQueue

  # every field a descriptor is given, in the order it is given, per descriptor instance
  fields: dict[int, list] = {}
  orig_write = ops_nv.QMD.write
  def qmd_write(self, **kw):
    r = orig_write(self, **kw)
    @guard
    def note():
      # only the plain integers: an address arrives as a UOp and is patched in later, and is recorded with the patches
      fields.setdefault(id(self), []).append({k: (v if isinstance(v, int) else f"<uop {type(v).__name__}>")
                                              for k, v in kw.items()})
    note()
    return r
  ops_nv.QMD.write = qmd_write

  orig_exec = ops_nv.NVComputeQueue.exec
  def nvexec(self, call, prg):
    r = orig_exec(self, call, prg)
    @guard
    def note():
      data, _lib = ops_nv.nv_build_program(self.dev, prg, self.devs)
      snap["execs"].append({
        "global_size": list(prg.arg.global_size), "local_size": list(prg.arg.local_size),
        "name": getattr(prg.arg, "name", None),
        "qmd_sz": self.qmd_sz, "stride": self.stride,
        "prog_off": data.prog_off, "kernargs_size": data.kernargs_size,
        "constbufs": {str(k): list(v) for k, v in data.constbufs.items()},
        "cbuf_0_len": len(data.cbuf_0),
        "cbuf_0": [int(x) for x in data.cbuf_0],
        "max_threads": data.max_threads,
        "template_qmd": bytes(data.qmd.mv).hex(),
        "template_fields": fields.get(id(data.qmd), []),
        "built_qmd": bytes(self.qmds[-1].mv).hex() if self.qmds else None,
        "built_fields": fields.get(id(self.qmds[-1]), []) if self.qmds else [],
        "built_patches": sorted(int(o) for o in self.qmds[-1].patches) if self.qmds else [],
      })
    note()
    return r
  ops_nv.NVComputeQueue.exec = nvexec

  # NVQueue defines its own submit and never calls up, so patching HWQueue's catches nothing: the base method only
  # raises. Both are patched so this keeps working whichever class ends up owning it.
  targets = [c for c in (ops_nv.NVQueue, HWQueue) if "submit" in vars(c)]
  orig_submits = {c: c.submit for c in targets}
  def make_submit(orig_submit):
   def submit(self, cmdbuf):
    @guard
    def note():
      snap["submits"].append({
        "queue": str(getattr(self, "queue", "?")), "cls": type(self).__name__,
        "dwords": len(self.blob) // 4, "blob": bytes(self.blob).hex(),
        # the offsets an address is written into. zero in the blob above, and supplied by the caller in the C driver too,
        # so leaving them out of the comparison is not a weakening of it.
        "patch_offsets": sorted(int(o) for o, _ in self.patches),
        "qmds": [bytes(q.mv).hex() for q in getattr(self, "qmds", [])],
      })
    note()
    return orig_submit(self, cmdbuf)
   return submit
  for c, orig in orig_submits.items(): c.submit = make_submit(orig)

def device_facts(dev) -> dict:
  keys = ["arch", "sm_version", "sass_version", "slm_per_thread", "shared_mem_window", "local_mem_window",
          "num_gpcs", "num_tpc_per_gpc", "num_sm_per_tpc", "max_warps_per_sm", "pma_enabled"]
  out = {k: getattr(dev, k, None) for k in keys}
  out["compute_class"] = hex(getattr(dev.iface, "compute_class", 0))
  return {k: (hex(v) if isinstance(v, int) and k.endswith(("window", "version", "class")) else v) for k, v in out.items()}

def workload(dev) -> None:
  from tinygrad import Tensor, Device
  # one small kernel with real parameters, then a second so a chained launch is captured too: the second descriptor is
  # started by the first rather than by methods of its own, and that is a different code path worth having bytes for
  out = (Tensor([1., 2., 3.], device="NV") * 2 + 1).tolist()
  print(f"kernel ok: {out}", flush=True)
  a = Tensor.rand(256, 256, device="NV")
  b = Tensor.rand(256, 256, device="NV")
  (a @ b).realize()
  Device["NV"].synchronize()
  print("matmul ok", flush=True)

def main() -> int:
  install()
  t0 = time.perf_counter()
  from tinygrad import Device
  dev = Device["NV"]
  print(f"NV up in {time.perf_counter()-t0:.2f}s: renderer={type(dev.renderer).__name__} arch={getattr(dev,'arch','?')}", flush=True)

  # A workload that raises leaves the card with bus master still on and the driver's dma mappings live, which shows up
  # afterwards as pci-dart-error-data latched on the IOKit nub and a preflight that refuses to run anything. It cost a
  # session once. Whatever happens in the workload, the device gets shut down properly and the snapshot still gets
  # written, because a recording that captured a boot is worth keeping even when the kernel it was after never ran.
  failure = None
  try: workload(dev)
  except Exception as e:  # noqa: BLE001
    failure = repr(e)
    snap["errors"].append(f"workload: {failure}")
    print(f"workload failed: {failure}", flush=True)
  finally:
    try: dev.finalize()
    except Exception as e: snap["errors"].append(f"finalize: {e!r}")  # noqa: BLE001
    else: print("device finalized: bus master cleared, dma mappings dropped", flush=True)

  snap["failure"] = failure
  snap["device"] = device_facts(dev)
  snap["tinygrad_commit"] = os.popen("git -C " + os.environ.get("TINYGRAD_SRC", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "tinygrad")) + " rev-parse --short HEAD").read().strip()
  with open(OUT, "w") as f: json.dump(snap, f, indent=1)
  print(f"snapshot: {OUT} - {len(snap['execs'])} launches, {len(snap['submits'])} submits, "
        f"{len(snap['errors'])} hook errors", flush=True)
  if snap["errors"]: print(f"  first error: {snap['errors'][0]}", flush=True)
  return 1 if failure else 0

if __name__ == "__main__": raise SystemExit(main())
