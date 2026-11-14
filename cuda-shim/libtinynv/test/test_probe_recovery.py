#!/usr/bin/env python3
"""End-to-end recovery-ownership and signal-handling tests for the rtx4090-probe binary.

Drives the probe over a unix socket against mock_ad102_server (the same fake AD102 the unit suite uses), with the
REAL pinned bundle through the production loader. Proves CONTROL FLOW only — no GPU is touched, and nothing here
covers SIGKILL, a client crash or server death: those force-unmap under a possibly-live firmware and are the
documented O14 limitation, not a tested guarantee.

Scenarios:
  preflight            read-only: no config writes, no DMA, no reset; lock released
  happy                full transaction: one FLR, both falcons, buffers accounted; exit 0
  contained            wrong device: exit 1, nothing allocated, no recovery hold
  hold + signals       uncontained failure: process/socket/mappings/lock stay alive with NO timeout; SIGINT/SIGTERM
                       refused; SIGUSR1 (owner power-off confirmation) exits 3 and only then disconnects
  early SIGUSR1        a SIGUSR1 sent BEFORE the hold must not pre-confirm the power-off
  stdin 'off'          exactly 'off' confirms; 'offline' does not
  partial stdin        a SIGUSR1 interrupts a blocked pselect after partial 'of' input and exits immediately
  stdin EOF            non-interactive: still holds, SIGUSR1 still works (no pause() race, no timer exit)
  SIGTERM mid-recv     a catchable signal during a socket round trip aborts into the hold with the socket INTACT
                       (EINTR retried): the best-effort BME clear still reaches the server
  lock exclusivity     a second probe is refused while the first holds recovery

Usage: test_probe_recovery.py <probe-binary> <mock-server-binary> <bundle-dir> [scenario-substring]
"""
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time

FAILS = []


def check(cond, what):
    if cond:
        print(f"  ok: {what}")
    else:
        print(f"  FAIL: {what}")
        FAILS.append(what)


class Run:
    def __init__(self, probe, server, bundle, knob, cmd="unlock", stdin_pipe=True):
        self.td = tempfile.TemporaryDirectory(prefix="tu-probe-")
        self.sock = os.path.join(self.td.name, "s")
        self.lock = os.path.join(self.td.name, "lock")
        self.evidence = os.path.join(self.td.name, "evidence")
        self.server = subprocess.Popen([server, self.sock, knob, self.evidence],
                                       stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        assert self.server.stdout.readline().strip() == b"ready", "mock server did not come up"
        self.out = []
        self.proc = subprocess.Popen(
            [probe, cmd, "--socket", self.sock, "--lock-file", self.lock, "--bundle", bundle],
            stdin=subprocess.PIPE if stdin_pipe else subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        for line in self.proc.stdout:
            self.out.append(line)

    def text(self):
        return "".join(self.out)

    def wait(self, needle, timeout):
        end = time.time() + timeout
        while time.time() < end:
            if needle in self.text():
                return True
            if self.proc.poll() is not None and needle not in self.text():
                return False
            time.sleep(0.05)
        return needle in self.text()

    def alive(self):
        return self.proc.poll() is None

    def write_stdin(self, text):
        try:
            self.proc.stdin.write(text)
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError):
            pass  # the probe already exited (a failing scenario); the checks report it

    def signal(self, sig):
        try:
            self.proc.send_signal(sig)
        except ProcessLookupError:
            pass  # already exited (a failing scenario may have lost the hold); the checks report it

    def wait_rc(self, timeout):
        try:
            return self.proc.wait(timeout)
        except subprocess.TimeoutExpired:
            return None

    def evidence_map(self):
        if not os.path.exists(self.evidence):
            return None
        d = {}
        for line in open(self.evidence):
            k, v = line.strip().split("=", 1)
            d[k] = int(v, 0)
        return d

    def cleanup(self):
        # Mock only: there is no real DMA behind this socket, so killing a misbehaving test probe is safe HERE.
        # The live probe must never be killed (README); this concession does not transfer to hardware.
        if self.alive():
            for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGUSR1):
                self.signal(sig)
                if self.wait_rc(4) is not None:
                    break
            if self.alive():
                self.proc.kill()
                self.wait_rc(5)
        if self.server.poll() is None:
            try:
                self.server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server.kill()
                self.server.wait(timeout=5)
        self.reader.join(timeout=2)
        self.td.cleanup()


def scenario_preflight(probe, server, bundle):
    print("preflight: read-only classification, lock released")
    r = Run(probe, server, bundle, "none", cmd="preflight")
    try:
        rc = r.wait_rc(30)
        check(rc == 0, f"exit 0 (got {rc})")
        check("stock-cold" in r.text(), "classified stock-cold")
        ev = r.evidence_map()
        check(ev is not None and ev["cmd_writes"] == 0 and ev["dma_allocs"] == 0 and ev["resets"] == 0,
              "no config writes, no DMA, no reset")
        check(not os.path.exists(r.lock), "lock released")
    finally:
        r.cleanup()


def scenario_happy(probe, server, bundle):
    print("happy path: the production probe completes the transaction over the wire")
    r = Run(probe, server, bundle, "none")
    try:
        rc = r.wait_rc(90)
        check(rc == 0, f"exit 0 (got {rc})")
        check("applied-and-verified" in r.text(), "applied-and-verified reported")
        check("RECOVERY REQUIRED" not in r.text(), "no recovery hold")
        ev = r.evidence_map()
        check(ev is not None and ev["resets"] == 1 and ev["starts"] == 2 and ev["dma_allocs"] == 7
              and ev["sec2_applied"] == 1, f"one FLR, both falcons, 7 buffers, payload applied (ev={ev})")
        check(ev is not None and (ev["last_command"] & 0x4) == 0, "bus master left cleared")
        check(not os.path.exists(r.lock), "lock released")
    finally:
        r.cleanup()


def scenario_contained(probe, server, bundle):
    print("contained failure: wrong device refuses before any firmware or DMA")
    r = Run(probe, server, bundle, "wrong-device")
    try:
        rc = r.wait_rc(60)
        check(rc == 1, f"exit 1 (got {rc})")
        check("RECOVERY REQUIRED" not in r.text(), "no recovery hold")
        ev = r.evidence_map()
        check(ev is not None and ev["dma_allocs"] == 0 and ev["starts"] == 0, "nothing allocated, no falcon started")
        check(not os.path.exists(r.lock), "lock released")
    finally:
        r.cleanup()


def hold_run(probe, server, bundle, knob="no-perms"):
    r = Run(probe, server, bundle, knob)
    if not r.wait("RECOVERY REQUIRED", 60):
        check(False, f"recovery hold entered (output tail: {r.text()[-400:]})")
    return r


def scenario_hold_signals(probe, server, bundle):
    print("recovery hold: kept alive with no timeout; SIGINT/SIGTERM refused; SIGUSR1 confirms")
    r = hold_run(probe, server, bundle)
    try:
        time.sleep(1.5)
        check(r.alive(), "still holding after 1.5 s (no automatic exit)")
        check(os.path.exists(r.lock) and f"pid={r.proc.pid}" in open(r.lock).read(), "lock still held by the probe")
        check(r.evidence_map() is None and r.server.poll() is None, "socket/mappings still alive (no disconnect)")
        r.signal(signal.SIGTERM)
        time.sleep(0.7)
        check(r.alive(), "SIGTERM refused during the hold")
        check("termination refused" in r.text(), "refusal explained")
        r.signal(signal.SIGINT)
        time.sleep(0.7)
        check(r.alive(), "SIGINT refused during the hold")
        r.signal(signal.SIGUSR1)
        rc = r.wait_rc(10)
        check(rc == 3, f"exit 3 after power-off confirmation (got {rc})")
        ev = r.evidence_map()
        check(ev is not None and ev["dma_live_at_disconnect"] == ev["dma_allocs"] == 7,
              "every buffer retained until the confirmed disconnect (none freed early)")
        check(ev is not None and ev["bme_clears"] >= 2, "best-effort BME clear reached the server")
        check(not os.path.exists(r.lock), "lock released after confirmation")
    finally:
        r.cleanup()


def scenario_early_sigusr1(probe, server, bundle):
    print("early SIGUSR1: a confirmation sent BEFORE the hold must not pre-confirm the power-off")
    r = Run(probe, server, bundle, "no-perms")
    try:
        time.sleep(0.05)  # asset loading: the transaction has not failed yet
        r.signal(signal.SIGUSR1)
        check(r.wait("RECOVERY REQUIRED", 60), "recovery hold entered")
        time.sleep(1.0)
        check(r.alive(), "still holding: the early SIGUSR1 was ignored")
        r.signal(signal.SIGUSR1)
        check(r.wait_rc(10) == 3, "a SIGUSR1 DURING the hold confirms and exits 3")
    finally:
        r.cleanup()


def scenario_stdin_off(probe, server, bundle):
    print("stdin confirmation: exactly 'off' confirms, 'offline' does not")
    r = hold_run(probe, server, bundle)
    try:
        r.write_stdin("offline\n")
        time.sleep(0.5)
        check(r.alive(), "'offline' is not a confirmation")
        check("still holding" in r.text(), "non-confirmation explained")
        r.write_stdin("off\n")
        check(r.wait_rc(10) == 3, "'off' confirms and exits 3")
    finally:
        r.cleanup()


def scenario_partial_stdin_sigusr1(probe, server, bundle):
    print("partial stdin: SIGUSR1 interrupts pselect even when 'off' has not received its newline")
    r = hold_run(probe, server, bundle)
    try:
        r.write_stdin("of")
        time.sleep(0.5)
        check(r.alive(), "partial command is not treated as confirmation")
        r.signal(signal.SIGUSR1)
        check(r.wait_rc(10) == 3, "SIGUSR1 confirms immediately after partial stdin")
    finally:
        r.cleanup()


def scenario_stdin_eof(probe, server, bundle):
    print("stdin EOF: non-interactive hold waits for SIGUSR1, never exits on a timer")
    r = hold_run(probe, server, bundle)
    try:
        r.proc.stdin.close()
        time.sleep(0.7)
        check(r.alive(), "still holding with stdin closed")
        r.signal(signal.SIGUSR1)
        check(r.wait_rc(10) == 3, "SIGUSR1 confirms after EOF (no lost-signal race)")
    finally:
        r.cleanup()


def scenario_sigterm_mid_recv(probe, server, bundle):
    print("SIGTERM mid-recv: abort lands in the hold with the socket INTACT (EINTR retried, quiesce works)")
    r = Run(probe, server, bundle, "stall-sec2")
    try:
        check(r.wait("falcon start: base=0x840000", 60), "SEC2 started")
        time.sleep(0.5)  # the server stalls the next read reply for 2 s: the probe is blocked in recv NOW
        r.signal(signal.SIGTERM)
        check(r.wait("RECOVERY REQUIRED", 30), "catchable termination entered the recovery hold")
        check("connection is broken" not in r.text(), "the healthy socket was NOT poisoned by the signal")
        check("abort requested" in r.text(), "the abort flag, not a transport failure, stopped the transaction")
        ev_before = r.evidence_map()
        check(ev_before is None, "no disconnect during the hold")
        r.signal(signal.SIGUSR1)
        check(r.wait_rc(10) == 3, "exit 3 after confirmation")
        ev = r.evidence_map()
        check(ev is not None and ev["bme_clears_after_sec2"] >= 1,
              "best-effort BME clear AFTER SEC2 start reached the server (recovery quiesce worked)")
    finally:
        r.cleanup()


def scenario_lock_exclusive(probe, server, bundle):
    print("lock exclusivity: a second probe is refused while the first holds recovery")
    r = hold_run(probe, server, bundle)
    try:
        second = subprocess.run(
            [probe, "preflight", "--socket", r.sock, "--lock-file", r.lock, "--bundle", bundle],
            capture_output=True, text=True, timeout=30)
        check(second.returncode == 2, f"second probe exit 2 (got {second.returncode})")
        check("LOCKED" in second.stderr, "second probe reports LOCKED")
        check(r.alive(), "first probe still holding")
        r.signal(signal.SIGUSR1)
        check(r.wait_rc(10) == 3, "first probe exits 3 after confirmation")
    finally:
        r.cleanup()


def main():
    if len(sys.argv) not in (4, 5):
        print(__doc__)
        return 2
    probe, server, bundle = sys.argv[1:4]
    only = sys.argv[4] if len(sys.argv) == 5 else ""
    for path in (probe, server):
        if not os.path.exists(path):
            print(f"  FAIL: missing binary {path}")
            return 1
    if not os.path.isdir(bundle):
        print(f"  SKIPPED: prepared bundle not present at {bundle} (run tools/rtx4090-unlock/prepare_bundle.py)")
        return 0
    for fn in (scenario_preflight, scenario_happy, scenario_contained, scenario_hold_signals,
               scenario_early_sigusr1, scenario_stdin_off, scenario_partial_stdin_sigusr1,
               scenario_stdin_eof, scenario_sigterm_mid_recv,
               scenario_lock_exclusive):
        if only and only not in fn.__name__:
            continue
        fn(probe, server, bundle)
    if FAILS:
        print(f"probe recovery: {len(FAILS)} checks failed")
        return 1
    print("probe recovery: all checks passed (mock only; SIGKILL/crash/server death are NOT covered, by design)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
