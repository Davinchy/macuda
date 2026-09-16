#!/usr/bin/env python3
"""Prove the wire proxy records what the class-level recorder cannot, without a GPU.

Three things have to hold, and the third is the whole point:

  - a session driven through the proxy behaves exactly as one driven straight at the server, including the reply that
    carries a file descriptor, which has to be relayed as ancillary data or the client fails in a way that looks like the
    server's fault;
  - the trace it writes parses as a trace, in the same vocabulary the replay backend already reads;
  - writes appear in it. That is the failure this proxy exists for: on a remote device the ring entry, the write pointer
    and the doorbell are lowered into a compiled write(2) on the socket and the class-level recorder never sees them.

Run: python test/test_wire_proxy.py <path to build/test_pci>
"""
from __future__ import annotations
import os, pathlib, socket, subprocess, sys, tempfile, time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent

def wait_for(path: str, timeout: float = 10.0) -> bool:
  deadline = time.time() + timeout
  while time.time() < deadline:
    if os.path.exists(path): return True
    time.sleep(0.02)
  return False

def run(test_pci: str) -> int:
  fails = 0
  with tempfile.TemporaryDirectory() as d:
    real, front, prefix = f"{d}/tinygpu.sock", f"{d}/proxy.sock", f"{d}/wire"

    server = subprocess.Popen([sys.executable, str(HERE / "mock_server.py"), real, "60"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not wait_for(real):
      print("  the mock server never came up")
      server.kill()
      return 1

    proxy = subprocess.Popen([sys.executable, str(ROOT / "tools" / "nv_wire_proxy.py"),
                              "--trace", prefix, "--connect", real, "--listen", front],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if not wait_for(front):
      print("  the proxy never listened")
      proxy.kill()
      server.kill()
      return 1

    # the same client the backend test uses, pointed at the proxy instead of the server
    client = subprocess.run([test_pci, front], capture_output=True, text=True)
    if client.returncode != 0:
      print(f"  the client failed through the proxy (exit {client.returncode}):")
      for line in (client.stdout + client.stderr).strip().splitlines()[-6:]: print(f"    {line}")
      fails += 1

    # the proxy exits when the client's socket closes; give it a moment to flush
    try: proxy.wait(timeout=10)
    except subprocess.TimeoutExpired:
      proxy.terminate()
      proxy.wait(timeout=5)
    os.path.exists(real) and os.unlink(real)
    server.wait(timeout=10)

    trace = pathlib.Path(f"{prefix}.trace")
    if not trace.exists():
      print("  the proxy wrote no trace")
      return fails + 1
    lines = [l.split() for l in trace.read_text().splitlines() if l.strip()]
    kinds: dict[str, int] = {}
    for f in lines:
      if f[0].startswith("#"): continue
      kinds[f[0]] = kinds.get(f[0], 0) + 1

    # the descriptor relay, which is the one failure that would take a real session down at its first DMA allocation:
    # the reply to a memory mapping carries a file descriptor as ancillary data, and a proxy that recv's instead of
    # recvmsg's drops it silently. The client cannot succeed without it, so this is checked rather than assumed.
    wire = next((l for l in trace.read_text().splitlines() if l.startswith("# wire commands")), "")
    if "map_sysmem_fd=" not in wire:
      print("  the session never passed a file descriptor, so the relay is untested; this must not go near hardware")
      fails += 1
    else:
      print(f"  descriptor relay exercised: {wire.split(':', 1)[1].strip()}")

    if "submission: recorded" not in trace.read_text():
      print("  the trace does not declare that submission is visible to it")
      fails += 1
    # the failure this exists for: writes must be in the trace, because at the wire nothing can route around it
    if not kinds.get("mmiow"):
      print("  no writes in the trace: the proxy is not seeing what it exists to see")
      fails += 1
    if not kinds.get("mmior"):
      print("  no reads in the trace, so the reply stream is not being framed")
      fails += 1
    print(f"  wire trace: " + ", ".join(f"{k}={v}" for k, v in sorted(kinds.items())))

  print("wire proxy: a session through it behaves the same and its writes are recorded" if not fails
        else f"wire proxy: {fails} checks failed")
  return fails

if __name__ == "__main__":
  if len(sys.argv) < 2: raise SystemExit("usage: test_wire_proxy.py <build/test_pci>")
  raise SystemExit(1 if run(sys.argv[1]) else 0)
