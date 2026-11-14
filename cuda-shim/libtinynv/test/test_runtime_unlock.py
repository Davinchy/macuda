#!/usr/bin/env python3
"""End-to-end tests of the runtime integration gate (tinynv_tensor_unlock_gate — the exact function
device_boot calls) against mock_ad102_server, with the REAL pinned bundle through the production loader.

Proves CONTROL FLOW only — no GPU is touched. Nothing here covers SIGKILL, a client crash or server death: those
force-unmap under a possibly-live firmware and are the documented O14 limitation, not a tested guarantee.

Scenarios:
  off-unset / off-zero / off-empty   the default path is inert: no config writes, no DMA, no reset, no falcon,
                                     no unlock output at all (spec: disabled mode changes nothing)
  invalid                            TINYNV_TENSOR_UNLOCK=2 / yes refused by name, no device contact
  wrong-backend                      option 1 on a non-tinygpu backend: unsupported, no device contact
  applied                            option 1, stock mock: applied-and-verified, one FLR, both falcons,
                                     7 buffers accounted, caller's SIGUSR1 disposition restored
  verified-existing                  option 1, retained mock: verified-existing with NO allocation, NO reset,
                                     NO firmware (the retained path must not reapply)
  unsupported-device                 option 1, wrong PCI id: unsupported, nothing allocated, no firmware
  recovery                           option 1, SEC2 stall + SIGTERM: the LIBRARY path enters the same hold —
                                     alive with no timeout, SIGTERM refused, SIGUSR1 exits 3, buffers retained

Usage: test_runtime_unlock.py <harness-binary> <mock-server-binary> <bundle-dir> [scenario-substring]
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
    def __init__(self, harness, server, bundle, knob, env_value=None, arg=None, stdin_pipe=True):
        self.td = tempfile.TemporaryDirectory(prefix="tu-rt-")
        self.sock = os.path.join(self.td.name, "s")
        self.evidence = os.path.join(self.td.name, "evidence")
        self.server = None
        if arg != "--wrong-backend":
            self.server = subprocess.Popen([server, self.sock, knob, self.evidence],
                                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            assert self.server.stdout.readline().strip() == b"ready", "mock server did not come up"
        env = dict(os.environ)
        env.pop("TINYNV_TENSOR_UNLOCK", None)
        env["TINYNV_UNLOCK_BUNDLE"] = bundle
        if env_value is not None:
            env["TINYNV_TENSOR_UNLOCK"] = env_value
        self.out = []
        self.proc = subprocess.Popen(
            [harness, arg if arg else self.sock],
            stdin=subprocess.PIPE if stdin_pipe else subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
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

    def signal(self, sig):
        try:
            self.proc.send_signal(sig)
        except ProcessLookupError:
            pass

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
        # Mock only: there is no real DMA behind this socket, so killing a misbehaving harness is safe HERE.
        # The live path must never be killed (README); this concession does not transfer to hardware.
        if self.alive():
            for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGUSR1):
                self.signal(sig)
                if self.wait_rc(4) is not None:
                    break
            if self.alive():
                self.proc.kill()
                self.wait_rc(5)
        if self.server is not None:
            if self.server.poll() is None:
                try:
                    self.server.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.server.kill()
                    self.server.wait(timeout=5)
        self.reader.join(timeout=2)
        self.td.cleanup()


def inert(r, label):
    rc = r.wait_rc(30)
    check(rc == 0, f"{label}: gate returns 0 (got {rc})")
    check("gate rc=0 active=0" in r.text(), f"{label}: active=0 (normal boot proceeds untouched)")
    check("tensor-unlock" not in r.text(), f"{label}: no unlock output at all")
    ev = r.evidence_map()
    check(ev is not None and ev["cmd_writes"] == 0 and ev["dma_allocs"] == 0 and ev["resets"] == 0
          and ev["starts"] == 0, f"{label}: no config writes, no DMA, no reset, no falcon start")


def scenario_off_unset(h, s, b):
    print("off (unset): the default path is inert")
    r = Run(h, s, b, "none", env_value=None)
    try:
        inert(r, "unset")
    finally:
        r.cleanup()


def scenario_off_zero(h, s, b):
    print("off (=0): explicitly disabled is inert")
    r = Run(h, s, b, "none", env_value="0")
    try:
        inert(r, "zero")
    finally:
        r.cleanup()


def scenario_off_empty(h, s, b):
    print("off (empty): an empty value is absence, like every other knob in this library")
    r = Run(h, s, b, "none", env_value="")
    try:
        inert(r, "empty")
    finally:
        r.cleanup()


def scenario_invalid(h, s, b):
    print("invalid values: refused by name, no device contact")
    for bad in ("2", "yes", "true", "01"):
        r = Run(h, s, b, "none", env_value=bad)
        try:
            rc = r.wait_rc(30)
            check(rc == 1, f"'{bad}': gate returns 1 (got {rc})")
            check("must be 0 or 1" in r.text(), f"'{bad}': refusal names the contract")
            ev = r.evidence_map()
            check(ev is not None and ev["cmd_writes"] == 0 and ev["dma_allocs"] == 0 and ev["resets"] == 0,
                  f"'{bad}': nothing touched on the device")
        finally:
            r.cleanup()


def scenario_wrong_backend(h, s, b):
    print("wrong backend: option 1 is unsupported off the TinyGPU transport")
    r = Run(h, s, b, "none", env_value="1", arg="--wrong-backend")
    try:
        rc = r.wait_rc(30)
        check(rc == 1, f"gate returns 1 (got {rc})")
        check("unsupported" in r.text(), "refusal says unsupported")
        check("gate rc=1 active=0" in r.text(), "active=0")
    finally:
        r.cleanup()


def scenario_applied(h, s, b):
    print("applied: option 1 on a stock mock runs the full transaction through the runtime gate")
    r = Run(h, s, b, "none", env_value="1")
    try:
        rc = r.wait_rc(90)
        check(rc == 0, f"gate returns 0 (got {rc})")
        check("status=applied-and-verified" in r.text(), "status=applied-and-verified reported")
        check("gate rc=0 active=1" in r.text(), "active=1 (post-boot verification is armed)")
        check("sigusr1-restored ok" in r.text(), "the caller's SIGUSR1 disposition was restored after the window")
        ev = r.evidence_map()
        check(ev is not None and ev["resets"] == 1, f"exactly one FLR (got {ev and ev['resets']})")
        check(ev is not None and ev["starts"] == 2, f"both falcons started (got {ev and ev['starts']})")
        check(ev is not None and ev["dma_allocs"] == 7 and "allocated=7 freed=7" in r.text(),
              "7 buffers allocated, all released client-side after verified containment")
        check(ev is not None and (ev["last_command"] & 0x4) == 0, "bus master left cleared")
        check(ev is not None and ev["sec2_applied"] == 1, "the SEC2 privileged writes applied")
    finally:
        r.cleanup()


def scenario_verified_existing(h, s, b):
    print("verified-existing: a retained override is verified, never reapplied")
    r = Run(h, s, b, "retained", env_value="1")
    try:
        rc = r.wait_rc(90)
        check(rc == 0, f"gate returns 0 (got {rc})")
        check("status=verified-existing" in r.text(), "status=verified-existing reported")
        check("gate rc=0 active=1" in r.text(), "active=1")
        check("sigusr1-restored ok" in r.text(), "signal dispositions restored")
        ev = r.evidence_map()
        check(ev is not None and ev["dma_allocs"] == 0 and ev["resets"] == 0 and ev["starts"] == 0,
              "no allocation, no reset, no firmware: the retained path allocates nothing")
    finally:
        r.cleanup()


def scenario_unsupported_device(h, s, b):
    print("unsupported device: wrong PCI identity is refused before any firmware")
    r = Run(h, s, b, "wrong-device", env_value="1")
    try:
        rc = r.wait_rc(60)
        check(rc == 1, f"gate returns 1 (got {rc})")
        check("status=unsupported" in r.text(), "status=unsupported reported")
        check("RECOVERY REQUIRED" not in r.text(), "no recovery hold (nothing was launched)")
        ev = r.evidence_map()
        check(ev is not None and ev["dma_allocs"] == 0 and ev["starts"] == 0,
              "nothing allocated, no firmware started")
    finally:
        r.cleanup()


def scenario_recovery(h, s, b):
    print("recovery: a catchable signal mid-transaction lands in the LIBRARY hold; ownership stays alive")
    r = Run(h, s, b, "stall-sec2", env_value="1")
    try:
        check(r.wait("falcon start: base=0x840000", 60), "SEC2 started")
        time.sleep(0.5)  # the server stalls the next read reply for 2 s: the gate is provably blocked in recv NOW
        r.signal(signal.SIGTERM)
        check(r.wait("RECOVERY REQUIRED", 30), "catchable termination entered the recovery hold")
        time.sleep(1.0)
        check(r.alive(), "still holding after 1 s (no automatic exit, no timeout)")
        r.signal(signal.SIGTERM)
        time.sleep(0.5)
        check(r.alive(), "SIGTERM refused during the hold")
        check("termination refused" in r.text(), "refusal explained")
        ev_before = r.evidence_map()
        check(ev_before is None, "no disconnect yet: the evidence file is written only at disconnect")
        r.signal(signal.SIGUSR1)  # the owner's power-off confirmation
        rc = r.wait_rc(10)
        check(rc == 3, f"gate returns 3 after the confirmed power-off (got {rc})")
        check("gate rc=3 active=0" in r.text(), "active=0: the normal boot must not proceed")
        ev = r.evidence_map()
        check(ev is not None and ev["dma_live_at_disconnect"] == 7,
              f"all 7 buffers retained until the confirmed disconnect (got {ev and ev['dma_live_at_disconnect']})")
        check(ev is not None and ev["bme_clears_after_sec2"] >= 1,
              "the best-effort BME clear AFTER SEC2 start reached the server (socket not poisoned)")
    finally:
        r.cleanup()


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        return 2
    h, s, b = sys.argv[1:4]
    only = sys.argv[4] if len(sys.argv) > 4 else ""
    scenarios = [scenario_off_unset, scenario_off_zero, scenario_off_empty, scenario_invalid,
                 scenario_wrong_backend, scenario_applied, scenario_verified_existing,
                 scenario_unsupported_device, scenario_recovery]
    for sc in scenarios:
        if only and only not in sc.__name__:
            continue
        sc(h, s, b)
    if FAILS:
        print(f"runtime unlock: {len(FAILS)} FAILED check(s)")
        return 1
    print("runtime unlock: all checks passed (mock only; SIGKILL/crash/server death are NOT covered, by design)")
    return 0


if __name__ == "__main__":
    sys.exit(main())