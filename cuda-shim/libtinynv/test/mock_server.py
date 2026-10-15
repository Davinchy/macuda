#!/usr/bin/env python3
"""Serve the python stand-in for TinyGPU.app on a unix socket so the C backend can be tested without a GPU.

The mock is the one the python driver's own tests use (tinygrad/test/unit/test_remote_pci.py), so both clients are
checked against the same server. Usage: mock_server.py <socket path> [seconds]
"""
import sys, os, time, pathlib

sys.path.insert(0, os.environ.get("TINYGRAD_SRC", str(pathlib.Path(__file__).resolve().parents[3] / "tinygrad")))
from test.unit.test_remote_pci import MockTinyGPUServer  # noqa: E402

if __name__ == "__main__":
  path, timeout = sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
  srv = MockTinyGPUServer(path, bars={0: 0x20000, 1: 4 << 20, 3: 0x10000})
  srv.start()
  print(f"mock tinygpu server on {path}", flush=True)
  deadline = time.time() + timeout
  while time.time() < deadline and os.path.exists(path): time.sleep(0.05)
  srv.close()
