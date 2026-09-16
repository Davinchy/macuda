#!/usr/bin/env python3
"""Record a GPU session at the wire, by sitting between the client and the TinyGPU server.

The class-level recorder in nv_trace.py cannot see work submission. On a remote device the driver lowers the ring entry,
the write pointer and the doorbell into a compiled write(2) on the socket, so those three writes never pass through the
interface it patches - and those three writes in that order are the whole submission protocol. A trace taken that way is
evidence about booting and about nothing else, however complete it looks.

Everything reaches the server through the socket. So this records there instead, and submission becomes visible by
construction rather than by patching the right class. (Session A's proposal, 2026-09-14.)

    python tools/nv_wire_proxy.py --trace out --connect /tmp/tinygpu.sock --listen /tmp/proxy.sock
    APL_REMOTE_SOCK=/tmp/proxy.sock python whatever-drives-the-gpu.py

The client takes a custom socket through APL_REMOTE_SOCK, and a custom socket means a custom server, so nothing is
installed or spawned and the real server is untouched.

Two things the protocol forces:

  - A reply carrying a file descriptor arrives as ancillary data, so those are relayed with recvmsg and sendmsg. Plain
    recv and send would drop the descriptor and the session would fail in a way that looks like the server's fault.
  - Writes are fire and forget: the client sends them and never waits. So the reply stream is framed against a queue of
    the requests that do expect one, which is also what makes a read's value available to record.
"""
from __future__ import annotations
import argparse, array, collections, os, socket, struct, sys, threading, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nv_trace import Recorder  # noqa: E402  the same trace vocabulary, so the replay backend needs no change

REQ_FMT, RESP_FMT = "<BIIQQQ", "<BQQ"
REQ_SZ, RESP_SZ = struct.calcsize(REQ_FMT), struct.calcsize(RESP_FMT)

PROBE, MAP_BAR, MAP_SYSMEM_FD, CFG_READ, CFG_WRITE, RESET, MMIO_READ, MMIO_WRITE, MAP_SYSMEM, SYSMEM_READ, \
  SYSMEM_WRITE, RESIZE_BAR, PING = range(13)
NAMES = {PROBE: "probe", MAP_BAR: "map_bar", MAP_SYSMEM_FD: "map_sysmem_fd", CFG_READ: "cfg_read", CFG_WRITE: "cfg_write",
         RESET: "reset", MMIO_READ: "mmio_read", MMIO_WRITE: "mmio_write", MAP_SYSMEM: "map_sysmem",
         SYSMEM_READ: "sysmem_read", SYSMEM_WRITE: "sysmem_write", RESIZE_BAR: "resize_bar", PING: "ping"}

# how many bytes of payload follow a request's header, and whether the client waits for a reply
def req_payload(cmd: int, a0: int, a1: int) -> int: return a1 if cmd in (MMIO_WRITE, SYSMEM_WRITE, PROBE) else 0
def expects_reply(cmd: int) -> bool: return cmd not in (MMIO_WRITE, SYSMEM_WRITE)

class Wire:
  """Frames the two directions and turns each exchange into the same operations the class-level recorder emits."""
  def __init__(self, rec: Recorder):
    self.rec, self.pending, self.lock = rec, collections.deque(), threading.Lock()
    self.counts: dict[str, int] = {}

  def request(self, hdr: bytes, payload: bytes):
    cmd, dev_id, bar, a0, a1, a2 = struct.unpack(REQ_FMT, hdr)
    self.counts[NAMES.get(cmd, str(cmd))] = self.counts.get(NAMES.get(cmd, str(cmd)), 0) + 1
    with self.lock:
      # writes are recorded as they go: nothing comes back to wait for, which is exactly why they were invisible before
      if cmd == MMIO_WRITE: self.rec.write("mmiow", f"bar:{bar}", a0, payload)
      elif cmd == SYSMEM_WRITE: self.rec.write("memw", f"mem:{bar}", a0, payload)
      elif cmd == CFG_WRITE: self.rec.op(f"cfgw {a0:#x} {a1} {a2:#x}")
      elif cmd == RESET: self.rec.op("reset")
      if expects_reply(cmd): self.pending.append((cmd, dev_id, bar, a0, a1, a2))

  def reply(self, hdr: bytes, extra: bytes):
    status, r0, r1 = struct.unpack(RESP_FMT, hdr)
    with self.lock:
      if not self.pending: return
      cmd, _dev, bar, a0, a1, _a2 = self.pending.popleft()
    if status != 0:
      self.rec.note(f"server refused {NAMES.get(cmd, cmd)}: {extra.decode('utf-8', 'replace')}")
      return
    with self.lock:
      if cmd == CFG_READ: self.rec.op(f"cfgr {a0:#x} {a1} {r0:#x}")
      elif cmd == MAP_BAR: self.rec.op(f"bari {bar} {r0:#x} {r1:#x}")
      elif cmd == MMIO_READ: self.rec.read("mmior", f"bar:{bar}", a0, extra)
      elif cmd == SYSMEM_READ: self.rec.read("memr", f"mem:{bar}", a0, extra)
      elif cmd == MAP_SYSMEM:
        paddrs = struct.unpack(f"<{len(extra) // 8}Q", extra) if extra else ()
        self.rec.op(f"dma mem:{r1} {a0:#x} {len(paddrs)} " + " ".join(f"{p:#x}" for p in paddrs[:4]))

  # how many bytes follow a reply header, which depends on what was asked for
  def reply_extra(self, status: int, r0: int, r1: int) -> int:
    with self.lock:
      if not self.pending: return 0
      cmd, _dev, _bar, _a0, a1, _a2 = self.pending[0]
    if status != 0: return r1  # an error carries its message
    if cmd in (MMIO_READ, SYSMEM_READ): return a1
    if cmd == MAP_SYSMEM: return r0   # the page addresses
    if cmd == PROBE: return r0
    return 0

def recv_exactly(sock: socket.socket, n: int) -> bytes:
  out = b""
  while len(out) < n:
    b = sock.recv(n - len(out))
    if not b: raise ConnectionError("the other end went away")
    out += b
  return out

def pump_requests(client: socket.socket, server: socket.socket, wire: Wire):
  while True:
    hdr = recv_exactly(client, REQ_SZ)
    cmd, _dev, _bar, a0, a1, _a2 = struct.unpack(REQ_FMT, hdr)
    payload = recv_exactly(client, req_payload(cmd, a0, a1)) if req_payload(cmd, a0, a1) else b""
    wire.request(hdr, payload)
    server.sendall(hdr + payload)

def pump_replies(server: socket.socket, client: socket.socket, wire: Wire):
  while True:
    # a reply may carry a file descriptor, and dropping it would fail as an obscure server error, so every reply header
    # is taken with recvmsg and relayed with sendmsg whether or not one arrives
    fds = array.array("i")
    msg, anc, _flags, _addr = server.recvmsg(RESP_SZ, socket.CMSG_SPACE(struct.calcsize("i")))
    if not msg: raise ConnectionError("the server closed the connection")
    while len(msg) < RESP_SZ: msg += recv_exactly(server, RESP_SZ - len(msg))
    for level, typ, data in anc:
      if level == socket.SOL_SOCKET and typ == socket.SCM_RIGHTS: fds.frombytes(data[:len(data) - len(data) % 4])

    status, r0, r1 = struct.unpack(RESP_FMT, msg)
    extra = recv_exactly(server, n) if (n := wire.reply_extra(status, r0, r1)) else b""
    wire.reply(msg, extra)

    if fds:
      client.sendmsg([msg + extra], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", list(fds)))])
      for fd in fds: os.close(fd)
    else: client.sendall(msg + extra)

def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--trace", required=True, help="prefix for <trace>.trace and <trace>.blob")
  ap.add_argument("--connect", required=True, help="the real TinyGPU socket")
  ap.add_argument("--listen", required=True, help="the socket to put in APL_REMOTE_SOCK")
  args = ap.parse_args()

  if os.path.exists(args.listen): os.unlink(args.listen)
  listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
  listener.bind(args.listen)
  listener.listen(1)
  print(f"proxy listening on {args.listen}, forwarding to {args.connect}", flush=True)

  rec = Recorder(args.trace)
  rec.note("submission: recorded")  # this recorder sits at the wire, so nothing can route around it
  rec.note(f"wire proxy to {args.connect}")
  wire = Wire(rec)
  client, _ = listener.accept()
  # the real server may still be starting, and connecting eagerly at proxy start would race it, so this waits until a
  # client actually arrives and then retries: a refused connection here is timing, not a failure
  server = None
  deadline = time.time() + 30.0
  while time.time() < deadline:
    try:
      server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
      server.connect(args.connect)
      break
    except (ConnectionRefusedError, FileNotFoundError):
      server.close()
      server = None
      time.sleep(0.1)
  if server is None:
    print(f"nothing is listening on {args.connect}", flush=True)
    rec.note(f"nothing was listening on {args.connect}")
    rec.close()
    return 1

  up = threading.Thread(target=pump_requests, args=(client, server, wire), daemon=True)
  down = threading.Thread(target=pump_replies, args=(server, client, wire), daemon=True)
  up.start()
  down.start()
  try:
    while up.is_alive() and down.is_alive(): up.join(0.2)
  except KeyboardInterrupt: pass
  finally:
    rec.note("wire commands: " + ", ".join(f"{k}={v}" for k, v in sorted(wire.counts.items())))
    rec.close()
    print(f"wrote {args.trace}.trace: " + ", ".join(f"{k}={v}" for k, v in sorted(wire.counts.items())), flush=True)
  return 0

if __name__ == "__main__": raise SystemExit(main())
