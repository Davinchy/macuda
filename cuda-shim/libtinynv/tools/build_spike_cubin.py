#!/usr/bin/env python3
"""Compile one of the spike kernels to a cubin for this architecture.

The cubins are not in the repository - `spike/*.cubin` is ignored, and rightly, since they are build output - but some
of the tests need one, so there has to be a command that produces it rather than a paragraph explaining how. It goes
through the same compiler tinygrad uses, which on macOS is a server in a container or on another machine; env.sh says
which.

    python tools/build_spike_cubin.py ../spike/printf_kernel.cu
"""
from __future__ import annotations
import os
import pathlib, sys

sys.path.insert(0, os.environ.get("TINYGRAD_SRC", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "tinygrad")))
from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler  # noqa: E402

def main(src: str, arch: str = "sm_120") -> int:
  p = pathlib.Path(src)
  if not p.exists(): return err(f"{src} does not exist")
  out = p.with_suffix(f".{arch.replace('sm_', 'sm')}.cubin")
  lib = NVRTCCompiler(arch, ptx=False, cache_key=f"spike_{p.stem}").compile(p.read_text())
  if lib[:4] != b"\x7fELF": return err(f"the compiler returned {len(lib)} bytes that are not an elf")
  out.write_bytes(lib)
  print(f"wrote {out}: {len(lib)} bytes for {arch}")
  return 0

def err(msg: str) -> int:
  print(f"  {msg}")
  return 1

if __name__ == "__main__":
  if len(sys.argv) < 2: raise SystemExit("usage: build_spike_cubin.py <kernel.cu> [arch]")
  raise SystemExit(main(sys.argv[1], *sys.argv[2:3]))
