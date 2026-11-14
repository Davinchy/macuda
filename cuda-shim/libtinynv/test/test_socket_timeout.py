#!/usr/bin/env python3
"""Minimal TinyGPU wire-protocol mock for test_socket_timeout (no tinygrad checkout required).

Speaks just enough of the protocol in docs/driver/architecture.md §2: a 33-byte packed request, a 17-byte packed
reply. Two servers, run in sequence:

  wedge: answers the first CMD_CFG_READ with the RTX 4090 identity, then reads requests and never replies. The C
         client must time out once (TINYNV_SOCKET_TIMEOUT_S=1) and then refuse the poisoned connection fast.
  eintr: answers every CMD_CFG_READ, but stalls the SECOND reply by 1.2 s. The C client raises SIGALRM mid-recv
         (handler without SA_RESTART); the transfer must be retried, not poisoned (a catchable signal
         must leave the recovery unwind a working socket).
"""
import os
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time

REQ = struct.Struct("<BIIQQQ")
RESP = struct.Struct("<BQQ")
assert REQ.size == 33 and RESP.size == 17


def serve(path, ready, mode):
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(path)
    srv.listen(1)
    ready.set()
    conn, _ = srv.accept()
    answered = 0
    try:
        while True:
            data = b""
            while len(data) < REQ.size:
                chunk = conn.recv(REQ.size - len(data))
                if not chunk:
                    return
                data += chunk
            cmd = data[0]
            if cmd == 3:  # CMD_CFG_READ
                if mode == "wedge":
                    if answered == 0:
                        conn.sendall(RESP.pack(0, 0x268410DE, 0))
                        answered += 1
                    # every later request: read and stay silent (the wedge)
                else:  # eintr
                    answered += 1
                    if answered == 2:
                        time.sleep(1.2)  # the client's alarm fires inside this recv
                    conn.sendall(RESP.pack(0, 0x268410DE, 0))
            # other commands: read and stay silent (the tests only issue config reads)
    except OSError:
        pass
    finally:
        conn.close()
        srv.close()


def run_mode(binary, mode):
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "tinygpu.sock")
        ready = threading.Event()
        t = threading.Thread(target=serve, args=(path, ready, mode), daemon=True)
        t.start()
        ready.wait(5)
        # the eintr stall (1.2 s) must stay UNDER the receive ceiling, or the timeout would poison the socket
        # for the right reason and the test would prove nothing about EINTR
        env = dict(os.environ, TINYNV_SOCKET_TIMEOUT_S="1" if mode == "wedge" else "5")
        return subprocess.call([binary, path, mode], env=env)


def main():
    binary = sys.argv[1]
    rc = run_mode(binary, "wedge")
    rc += run_mode(binary, "eintr")
    return 1 if rc else 0


if __name__ == "__main__":
    sys.exit(main())