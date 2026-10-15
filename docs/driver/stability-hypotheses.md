# Stability hypotheses and experiment plan (W3)

Written 2026-09-13 by Session B. Inputs: `docs/driver/architecture.md` (§6), Session A's `docs/research/stability-dext.md` (kernel-side
mechanisms) and A's hardware log in `docs/SHARED-STATUS.md` (E1-E3, Run 1-3, 5-cycle soak, 03:12-03:40). Status per hypothesis is what the
evidence supports today; "branch" means `egpu-hcq2-remote` in `tinygrad/`, "stable" means A's patch 5239fc52b in `tinygrad-stable/`.

## 1. Ranked hypotheses

| # | Hypothesis (mechanism) | Evidence | Status | Fix |
|---|---|---|---|---|
| H1 | **GSP unload at exit poisons the next boot** over the TB tunnel (`rpc_unloading_guest_driver`, then next boot falls off the bus) | #16454 same enclosure; A: Run 2/3 + soak 5/5 clean with GSP resident; in-process WPR2 reset works | **confirmed, fixed** | `NV_KEEP_GSP` default 1 on macOS/remote (stable + branch) |
| H2 | **DMA after teardown**: socket close → dext `CompleteDMA` while Bus Master is set and GSP writes host queues/logs/notifier → DART faults, worst case SPTM panic (#16086) | mechanism from source; A: no `pci-dart-error-data` after clean exits with BM cleared | **fixed for clean exits**, open for crashes | BM cleared in `device_fini` `finally` (branch) / `NVDev.fini` (stable). Not covered: SIGKILL, python crash, server `_exit(0)` on unplug (needs server/dext change, W2) |
| H3 | **No timeouts**: server blocks in uninterruptible IOKit RPCs (`U`), client blocked in `recv`; a stalled dext call starves `Stop()` and IOPCIFamily spins | A's report §2; April symptoms | **client side fixed**, server side open | select timeout 60 s (branch), `settimeout` (stable). Server-side watchdog needs W2 |
| H4 | **FLR timing on GB202**: kernel restores config 100 ms after FLR, GFW needs longer → all-ones reads, `KeyError: 63` | A: FLR 0.106 s, GFW ready 1.4 s, `wait_for_reset` (10 s) covers it; all-ones now a clear error | **resolved** | none needed; diagnostic in `_early_ip_init` (both trees) |
| H5 | **Silent MMIO write drops**: `server.c` drops any `MMIO_WRITE` that fails `validate_bar` without a response. Triggers: GPFIFO area above the 256 MB BAR1 window, zeroing of a contiguous VRAM allocation above the window, a view built past a BAR's end. Looks like a GPU hang (30 s `signal wait timed out`) | code reading; not observed | **open, mitigated** | branch: gpfifo area allocated before any tensor; `RemoteMMIOInterface` raises `IndexError` past its view. Server-side logging needs W2 |
| H6 | **Second client**: `listen(1)` queues a second connection which then blocks in `recv` (state `S`); a stale `nv_usb4.lock` or a different `$TMPDIR` bypasses the flock | code reading | open, low risk | keep one venv/`$TMPDIR`; A's `preflight.sh` lists sockets/locks; server-side reject needs W2 |
| H7 | **Sleep / Power Nap with GSP resident**: PCI power transitions or tunnel teardown under a live GSP | untested; A's report H5 | open | experiment E7; session hygiene `caffeinate -dims`, `pmset -a powernap 0` |
| H8 | **DART / mapping budget**: 128 shm slots per server session, ~1.5 GB DART budget, 32-segment `PrepareForDMA` cap | code reading + A's report | mitigated in branch | 64 MB chunks + dedicated big mappings; a failure surfaces as `RuntimeError`, not a wedge |
| H9 | **hcq2-specific**: the compiled submit writes packets with `write(2)`; if the server died, `write` returns `EPIPE` (python ignores SIGPIPE), the doorbell never lands, the batch times out after 30 s, `device_fini`'s BM clear then fails on the dead socket | code reading | open, benign | acceptable: an error, not a wedge; log it clearly |
| H10 | **Cable unplug while running**: nub terminates, server `_exit(0)`s on `kIOMessageServiceIsTerminated` with mappings live (H2 window), python sees all-ones/timeouts | A's report §5 | open | E3-unplug below (Antonio's call, physically) |

Retired: FLR unsupported on this path (E3 showed FLR via the dext works and clears WPR2), Razer dock in the chain (box is on its own bus).

## 2. Experiment plan (each is one GPU step, under the lock, preflight first, `logstream.sh` running)

| Id | Owner | Steps | Expect | Closes |
|---|---|---|---|---|
| E4 | B | branch: `DEBUG=2 DEV=PCI+NV:NAK python tools/nv_boot_smoke.py`, then `test/test_tiny.py`, then `nv_launch_overhead.py` | boot < 5 s, `remote: dma chunk 0: 64 MB`, kernels correct, exit with BM cleared; launch overhead well under hcq1's 14 µs/kernel and no RPC reads per token | W1 acceptance, H5 (gpfifo placement), H8 (chunks) |
| E5 | B | while a kernel loop runs on the branch, `kill -9` the python process; then `preflight.sh` (look for `pci-dart-error-data`), then a normal run | either no DART error (GSP idle between batches) or an error property but a clean next boot after in-process FLR | H2 residual |
| E6 | A or B | two processes: one holding the device, a second `Device["NV"]` | second fails fast with the flock error, never hangs | H6 |
| E7 | needs Antonio | device idle with GSP resident, `pmset sleepnow`, wake, run again | clean boot; if not, document and add "power-cycle the box after sleep" to the protocol | H7 |
| E8 | B | branch with `NV_KEEP_GSP=0` once, then a normal run | reproduces #16454 (device falls off the bus on the second boot) or not; either way the default stays 1 | H1 sharpness |
| E9 | needs Antonio | unplug the cable during an idle session, replug, run | server exits, dext stops, no 100 % CPU spin; next boot works after the in-process FLR | H10, H3 server side |
| E10 | B | allocate host buffers until the DART budget is hit (`Tensor.empty` on CPU mapped into NV, 128 MB steps) | `RuntimeError` from `_map_chunk`/`PrepareDMA`, no wedge; record the budget | H8 |

Rules: one step per process, preflight between steps, never `kill -9` a TinyGPU process, never reset the system extension. E5, E9 are the only
ones that intentionally exercise a failure path; run them last and only with A's agreement, since a wedge costs a reboot.

## 3. Fixes, by where they live

Done in the branch (`egpu-hcq2-remote`): `NV_KEEP_GSP` default, BM clear in a `finally`, select-based RPC timeout, all-ones diagnostic,
dma chunking, eager gpfifo area, `RemoteMMIOInterface` range check.

Client side, still to do: log the 30 s timeout with the last packets sent (H9); a `tools/nv_quiesce.py` that clears BM through a fresh
server connection for the crash case (H2 residual) — this is A's `nv_e3_flr.py` minus the FLR.

Server/dext (W2, needs SIP off or tinygrad's signing): revoke BM in `cleanup()` before `IOServiceClose` and in `Stop_Impl`; respond to
`MMIO_WRITE` failures (or log them); a watchdog thread that reports an RPC older than N seconds; reject a second client; per-buffer unmap;
lift `MAX_SYSMEM` and the 32-segment cap.
