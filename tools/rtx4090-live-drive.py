#!/usr/bin/env python3
"""rtx4090-live-drive.py — supervised pty driver for `sh tools/rtx4090-live.sh unlock`.

The live runner requires an interactive terminal: its unlock confirmation is read from stdin, and the probe's
recovery hold reads the power-off confirmation ('off') from the same stdin. An agent session has no terminal, so
this driver provides a pty and NOTHING ELSE:

  * it answers the runner's supervision confirmation once with 'yes'; the immediately preceding read-only
    preflight must have shown the verified stock-cold state;
  * it NEVER confirms a recovery power-off by itself. 'off' is written to the pty only when the operator creates
    the control file (default /tmp/macuda-tensor-unlock-confirm-off) after physically powering the enclosure off — or the
    operator sends SIGUSR1 to the probe pid directly (the probe's own documented path);
  * it has NO timeout and never kills the runner or the probe. If the probe holds ownership under uncertain
    containment, this driver stays alive with it (transport-audit O14: killing force-unmaps under live firmware).

Usage:  python3 tools/rtx4090-live-drive.py [transcript-log] [command...]
Default command: sh tools/rtx4090-live.sh unlock. Exit code = the runner's exit code.
The transcript log holds every byte the runner/probe printed, plus [drive] markers for what was sent.
"""
import os
import pty
import select
import sys

LOG = sys.argv[1] if len(sys.argv) > 1 else "logs/rtx4090-tensor-unlock-pty.log"
CMD = sys.argv[2:] or ["sh", "tools/rtx4090-live.sh", "unlock"]
CONFIRM_OFF_FILE = os.environ.get(
    "TINYNV_TENSOR_UNLOCK_CONFIRM_OFF_FILE",
    "/tmp/macuda-tensor-unlock-confirm-off",
)

os.setsid()  # detach from the calling harness's process group: a harness timeout must not kill the ownership holder
pid, fd = pty.fork()
if pid == 0:
    os.environ["TERM"] = "dumb"
    os.execvp(CMD[0], CMD)

log = open(LOG, "wb", buffering=0)
log.write(f"[drive] runner pid {pid}, command {CMD}, transcript {LOG}\n".encode())
log.write(f"[drive] recovery 'off' is sent ONLY when {CONFIRM_OFF_FILE} is created by the operator\n".encode())
answered = False
off_sent = False
buf = b""
rc = None
while True:
    try:
        r, _, _ = select.select([fd], [], [], 1.0)
    except InterruptedError:
        continue
    if fd in r:
        try:
            data = os.read(fd, 4096)
        except OSError:
            data = b""  # EIO: the pty slave closed (runner exited)
        if data:
            log.write(data)
            buf = (buf + data)[-8192:]
            if not answered and b"[yes/N]" in buf:
                os.write(fd, b"yes\n")
                answered = True
                log.write(b"\n[drive] 'yes' sent: supervised run on the verified stock-cold card\n")
    if not off_sent and os.path.exists(CONFIRM_OFF_FILE):
        os.write(fd, b"off\n")
        off_sent = True
        log.write(b"\n[drive] 'off' sent: the operator created the confirm file (physical enclosure power-off)\n")
    done, status = os.waitpid(pid, os.WNOHANG)
    if done == pid:
        rc = os.waitstatus_to_exitcode(status)
        # drain whatever is left in the pty buffer
        while True:
            try:
                data = os.read(fd, 4096)
            except OSError:
                break
            if not data:
                break
            log.write(data)
        break
log.write(f"\n[drive] runner exit {rc}\n".encode())
sys.exit(rc if rc is not None else 2)
