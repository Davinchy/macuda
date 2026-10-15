# TinyGPU on macOS: architecture of the NVIDIA-over-Thunderbolt path (W0 note)

Written 2026-09-13 by Session B from source only (no GPU touched). Sources: `extra/usbgpu/tbgpu/installer/` (dext, server, app),
`extra/hcq1/remote.py` and `git show 33cd373ad:tinygrad/runtime/support/system.py` (the removed client), `tinygrad/runtime/ops_nv.py`,
`tinygrad/runtime/support/{system,hcq2,nv/nvdev,nv/ip}.py` on master (commit 947882518), commits #17984 #18015 #18073 #18085 #16007 #16075.
Session A's `docs/research/stability-dext.md` covers the kernel side (xnu/IOPCIFamily) in depth; this note stays at the code we own.

## 1. The stack, one box per process

```
Python (tinygrad, venv)                      TinyGPU server (C, inside TinyGPU.app, signed by tinygrad Corp)      dext org.tinygrad.tinygpu.driver2
  APLRemotePCIDevice  ── unix socket ──▶  server.c: one client at a time, single thread, no timeouts             TinyGPUDriver (IOService on IOPCIDevice)
  $TMPDIR/tinygpu.sock                     IOServiceOpen("tinygpu") per client  ──── IOKit user client ────▶      TinyGPUDriverUserClient
  flock $TMPDIR/nv_usb4.lock               IOConnectMapMemory64(bar)            ──── CopyClientMemoryForType ─▶   MapBar → _CopyDeviceMemoryWithIndex
                                            IOConnectCallMethod(sel 0/1/2)       ──── ExternalMethod ─────────▶   CfgRead / CfgWrite / Reset(FLR)
                                            IOConnectCallStructMethod(sel 3)     ──── ExternalMethod ─────────▶   PrepareDMA (IODMACommand over server pages)
  mmap(shm fd) ◀── SCM_RIGHTS ──────────    shm_open("/tinygpu_N") + mmap                                          Start(): pci->Open, set IO|MEM|BUSMASTER
  MMIO = RPC; sysmem = local memory         MMIO reads/writes executed HERE on the BAR mapping (32-bit volatile)   Stop(): pci->Close only
```

Facts that shape everything else:

- **Only the signed app can open the dext.** The app carries `com.apple.developer.driverkit.userclient-access = [org.tinygrad.tinygpu.driver2]`
  (`macOS/macOS.entitlements`); the Release dext does not have `allow-any-userclient-access` (only `TinyGPUDriver.NoSIP.entitlements` does).
  An unentitled Python cannot `IOServiceOpen` the dext. So with SIP on, **the wire protocol below is the only door**, and neither `server.c`
  nor the dext can change without rebuilding and re-signing (W2, needs SIP off or tinygrad's profile).
- **Sysmem is shared memory, not RPC.** Every DMA-able host buffer is a POSIX shm object created by the server, DART-mapped by the dext, and
  mmapped directly into Python. Signal polling, GSP status-queue polling, timelines and command buffers are plain memory reads/writes in
  Python and in hcq2's compiled host program. The handoff's "every doorbell/signal poll is a socket round trip" is only half right: **MMIO is
  RPC, sysmem is local.**
- **BAR1 is 256 MB on this Mac** (Session A, E1) and `CMD_RESIZE_BAR` is a no-op in `server.c`, so `NVDev.large_bar` is False: page tables are
  reserved inside the BAR window (`reserve_ptable=True`), GSP boot structures go to sysmem, and `PCIIfaceBase.is_bar_small()` is True, which
  routes every `cpu_access` allocation to sysmem. Only `force_devmem` allocations (the 3 MB GPFIFO area) are CPU-visible VRAM through a BAR1 window.

## 2. Wire protocol (unchanged since March 2026; `server.c` and `extra/hcq1/remote.py` agree)

Request, 33 bytes, `struct.pack('<BIIQQQ', cmd, dev_id, bar, arg0, arg1, arg2)` = `request_t {u8 cmd; u32 dev_id, bar; u64 arg0, arg1, arg2}` packed.
Response, 17 bytes, `'<BQQ'` = `response_t {u8 status; u64 resp0, resp1}`; status 1 = error, then `resp0` bytes of message follow.
`dev_id` is ignored by `server.c` (one device); the client uses 0.

| cmd | name | args | server.c does | dext does | response |
|---|---|---|---|---|---|
| 0 | PROBE | | not implemented → status 1, "unknown error" | | (TCP `serve.py` only; macOS discovery is IOKit `pci_scan_bus`) |
| 1 | MAP_BAR | bar | `IOConnectMapMemory64(conn, bar)` once, cached | `CopyClientMemoryForType(type=bar<6)` → `MapBar` | resp0 = **server VA of the mapping**, resp1 = size |
| 2 | MAP_SYSMEM_FD | arg0=size, arg1=contiguous (ignored) | `shm_open("/tinygpu_i")`, `ftruncate(max(round4K(size),16K))`, mmap, `IOConnectCallStructMethod(3, ptr, sz, out, 8192)`, copy the returned `[iova,len]*,0,0` table to the **start of the shm buffer** | `PrepareDMA`: `IODMACommand::Create(pci, maxAddressBits=40)` + `PrepareForDMA(kIOMemoryDirectionInOut, ≤32 segments)` on the server's pages; keeps the IODMACommand in a per-client array | resp0 = mapped size, resp1 = index, **+ fd via SCM_RIGHTS** |
| 3 | CFG_READ | arg0=off, arg1=size(1/2/4) | `dext_rpc(0, [off,size])` | `ConfigurationRead8/16/32` | resp0 = value |
| 4 | CFG_WRITE | arg0=off, arg1=size, arg2=val | `dext_rpc(1, [off,size,val])` | `ConfigurationWrite8/16/32` | ok |
| 5 | RESET | | `dext_rpc(2)` | installed 1.0.0/3: `pci->Reset(kIOPCIDeviceResetTypeFunctionReset)`; source since #16075 (unreleased): falls back to HotReset on error | ok after the kernel FLR sequence (~150 ms) |
| 6 | MMIO_READ | bar, arg0=off, arg1=len (≤64 MB, inside BAR) | `mmio_copy` BAR→bulk buf (32-bit volatile loads), then `send` | nothing (BAR is mapped in the server) | resp0 = len, then `len` raw bytes |
| 7 | MMIO_WRITE | bar, arg0=off, arg1=len, then `len` payload bytes | `recvall` payload, `mmio_copy` bulk→BAR **if** `validate_bar` passes, else silently dropped | nothing | **no response** (posted; pipelined) |
| 8/9/10 | MAP_SYSMEM / SYSMEM_READ / SYSMEM_WRITE | | not implemented (TCP `serve.py` only) | | |
| 11 | RESIZE_BAR | bar | nothing | | ok (BAR stays 256 MB) |
| 12 | PING | | not implemented | | (client enum only) |

Client conventions (from `extra/hcq1/remote.py`): `read_config(off,size)`, `write_config(off,value,size)` → args `(off,size,value)`; `bar_info(bar)`
= `(resp0, resp1)` cached; `map_bar(bar, off, size, fmt)` returns a `RemoteMMIOInterface` whose `__getitem__`/`__setitem__` issue one
MMIO_READ / MMIO_WRITE each (any width, any length); `alloc_sysmem(size)` = MAP_SYSMEM_FD + `mmap(fd)` + parse the `(iova,len)` pairs from the
mapping head into a 4 KB page list; `reset()` = RESET. `_recvall` has **no timeout**; `REMOTE_TIMEOUT` only guards the TCP connect.

Server lifecycle: `TinyGPU server <sock>` is spawned by the first Python client (`subprocess.Popen`, up to 5 s of connect retries) and outlives
it; `listen(1)`, `accept` loop, `handle_client` is synchronous, so **a second client blocks silently in `recv` until the first disconnects**
(the Python-side `flock` on `nv_usb4.lock` is what actually turns that into an error). On client disconnect: `cleanup()` unmaps BARs, munmaps/
unlinks all shm, `IOServiceClose` → dext `TinyGPUDriverUserClient::Stop_Impl` → `CompleteDMA()` on every IODMACommand (DART unmap, all at once).
On `kIOMessageServiceIsTerminated` (unplug) the server `_exit(0)`s without cleanup. **There is no per-buffer unmap command**: sysmem lives until
the client disconnects, and the server holds at most `MAX_SYSMEM = 128` allocations per client session.

## 3. Where memory lives (master, small BAR)

| What | Allocation path | Physical location | CPU access from Python / host program |
|---|---|---|---|
| BAR0 registers (`NVDev.mmio`, `gpu_mmio` doorbell page at BAR0+0xbb0000) | `map_bar(0)` | GPU | RPC per access |
| BAR1 VRAM window (`NVDev.vram`, page tables, boot images written by CPU) | `map_bar(1)` | VRAM < 256 MB | RPC per access |
| GSP cmd/stat queues, libos args, RM args, WPR meta, FMC boot args, 2 MB log buffer, GSP ELF radix3, bootloader, signature | `_alloc_boot_mem(sysmem=True or not large_bar)` → `alloc_sysmem` | sysmem (DART IOVA) | local (shm mmap) |
| FRTS/booter patched images, gpfifo method buffer (`sysmem=False`) | `mm.palloc` + `vram.view` | VRAM < 256 MB | RPC writes at boot |
| Tensors, kernels' data (no cpu_access) | `mm.valloc` | VRAM anywhere | none |
| Programs (`prog_bufs`, `cpu_access=True`), rt pools (`rt_buffer`, uncached+cpu_access), timelines (`host=True`), notifier (`uncached` only → VRAM) | `PCIIfaceBase.alloc` | sysmem (because `is_bar_small()`) except notifier | local |
| GPFIFO area 3 MB (ring + USERD/GPPut, `force_devmem=True, cpu_access=True`) | `mm.valloc(contiguous)` + `map_bar(1, off=paddr)` | VRAM, must be < 256 MB | RPC (and, in hcq2, the compiled submit program stores into it by raw pointer, see §5) |

DMA addresses handed to the GPU are DART IOVAs (`maxAddressBits=40`), one contiguous segment per shm object in practice; page-table entries
use aperture SYS + snooped + uncached. Nothing in the client checks the 32-segment cap or the DART budget (Session A's report: ~1.5 GB total).
Note `bar_info(bar)[0]` on macOS is the **server's mach VA**, not the PCI bus address; it feeds `GspSystemInfo.gpuPhysAddr/gpuPhysFbAddr/
gpuPhysInstAddr` (`ip.py rpc_set_gsp_system_info`) and `_alloc_boot_mem`'s BAR1 "sysaddr" (unused when the BAR is small). CI passed with it, so
GSP tolerates it, but the port should read the real base from config space (`0x10 + 4*bar`) instead.

## 4. Boot and teardown sequence (what the GPU sees per `Device["NV"]`)

1. `NVDev.__init__`: `map_bar(0)`; `_early_ip_init`: if `NV_PFB_PRI_MMU_WPR2_ADDR_HI != 0` (GSP already booted since power-on): clear Bus
   Master (config write+readback), `RESET` (FLR via dext), sleep 0.1 s; set Bus Master; read `NV_PMC_BOOT_0/42` → chip GB202 → `NV_FLCN_COT`
   (FSP chain-of-trust boot, Blackwell) + `NV_GSP`; `flcn.wait_for_reset()` polls `NV_THERM_I2CS_SCRATCH == 0xff` up to 10 s (GFW boot).
2. `_early_mmu_init`: `map_bar(1)`, VRAM size from `NV_PGC6_AON_SECURE_SCRATCH_GROUP_42`, MMU v3 page tables in the BAR window.
3. `init_sw`: firmware from the cache (`gsp-570.144.bin`, `fmc-570.144.bin`, `bootloader-570.144.bin`), GSP RPC queues + registry
   (`RMForcePcieConfigSave`, `RMSecBusResetEnable`) prefilled in sysmem, WPR meta.
4. `init_hw`: FSP `NVDM_TYPE_COT` message through `NV_PFSP_EMEM` (every word an RPC), wait for `riscv_br_priv_lockdown == 0`;
   GSP: `wait_resp(GSP_INIT_DONE)` (10 s timeout), BAR1 block, golden context image, then the tinygrad client (`NVDevice.__init__`) allocates
   device/subdevice/vaspace/channel group over `rpc_rm_alloc`. Every `wait_cond` is 10 s; RPC waits are 10 s; a `CPU sequencer` event from GSP
   is executed on the Mac via `run_cpu_seq` (more register RPCs).
5. Runtime: hcq2 compiles a host C program per batch that writes the ring entry and GPPut and rings the doorbell, then polls the timeline in
   sysmem. `PCIIface.sleep()` drains the GSP status queue (local memory) and raises on `is_err_state`.
6. Teardown: `atexit` → `Device.finalize()` → `synchronize()` → `PCIIface.device_fini()` → `NVDev.fini()` → `gsp.fini_hw()` =
   `rpc_unloading_guest_driver` (the "clean" GSP unload that issue #16454 blames for poisoning the next boot on this enclosure). Nothing clears
   Bus Master. Then the socket closes as Python exits → server `cleanup()` → dext `CompleteDMA` on every mapping. An abnormal exit (crash,
   SIGKILL, or an exception before the device is fully up) skips the unload but not the DART teardown.

## 5. What hcq2 changed, and why the old client cannot just be re-added (input to W1)

- hcq1 wrote GPFIFO entries, GPPut and the doorbell from Python through `MMIOInterface.__setitem__`, which `RemoteMMIOInterface` overrides
  with RPCs. hcq2 (`NVQueue.submit`) turns those three stores into a **compiled host program** (`HCQ_RUNTIME_DEV=CPU`, clang) that stores by
  raw pointer: `ring`/`gpput` are views of `gpfifo_buf` (BAR1 window) and `doorbell` is `Buffer("CPU", external_ptr=gpu_mmio.addr+0x90)`.
  A socket-backed MMIO object has no address; `RemoteMMIOInterface` lacks `.addr` entirely. This is why #17984 removed the path "for now".
- The USB port (#18015) is the template: keep the host program, but rewrite loads/stores that target remote memory into C calls
  (`ccall(libusb...)`) with `pm_lower`, and rewrite host↔device copies with `pm_batch`. For TinyGPU the equivalent is far simpler because
  writes are posted and pipelined: a store to ring/GPPut/doorbell becomes one `write(sock_fd, packet, 33+len)` of an MMIO_WRITE packet
  (`libc.write` is in the autogen and `hcq2.patch` already supports the unaligned 33-byte header). Three `write`s per batch, ~10 µs.
- Everything the host program *reads* (timelines, signals, slots, cmdbuf, qmd, programs) already lives in sysmem on a small BAR, so those
  need no rewrite. Link-time patching (`fold_binary`/`fold_words`) goes through `Buffer.host`, which the RPC-backed view serves fine.
- Host↔device copies: `PCIIfaceBase.map()` computes physical pages from `/proc/self/pagemap`; there is no macOS equivalent and DART mapping is
  only reachable through the dext's `PrepareDMA` (server-owned pages). So CPU/NPY/PYTHON buffers cannot be mapped into the GPU. hcq2 already
  has a fallback (`stage_copy` → 128 MB `_staging()` CPU buffer), but that staging buffer must itself be GPU-mappable: on macOS it has to be
  carved from dext sysmem and `map()` has to resolve it from a registry instead of pagemap. Same trick USB uses (`USBAllocator.map`).
- Other adaptations: skip `System.reserve_va` (Linux `MAP_FIXED_NOREPLACE`), never `munmap(storage.buf)` on free (GPU VA ≠ CPU VA on macOS),
  give `RemoteMMIOInterface` a BAR-relative `.addr` so `gpu_mmio.addr + 0x90` stays meaningful, sub-allocate sysmem in ≥64 MB chunks (128-slot
  cap, DART budget, one `PrepareDMA` per chunk), and keep the GPFIFO area below 256 MB (allocate it before any tensor).
- Not needed: sysmem USERD (would require `userdMem.addressSpace=1`, untested with GSP), server or dext changes.

## 6. Where the April wedge can sit (code-level view; mechanisms in Session A's report §2-3)

Ranked by how directly the code we own participates:

1. **DMA after teardown (H2).** Socket close → `cleanup()` → `IOServiceClose` → `Stop_Impl` → `CompleteDMA` while Bus Master is set
   (master never clears it on exit). After a clean unload GSP is halted, so the window is small; after an abnormal exit (the common case while
   bring-up is failing) GSP is still running and writing its stat queue, logs and the 48 MB notifier → DART faults. Fix lives in the client:
   quiesce (clear `PCI_COMMAND_MASTER`, optionally FLR) **before** closing the socket, in a `try/finally` that also runs on exceptions.
2. **GSP unload → next boot falls off the bus (H1, #16454, same enclosure).** This is master's default clean-exit path (§4.6). The #16454
   recipe (leave GSP resident, standalone FLR from a separate process before the next run) translates to: make `device_fini` skip
   `gsp.fini_hw()` (env-gated so the experiment is reversible), keep the in-process WPR2 reset as fallback, and use a standalone FLR tool that
   speaks the protocol (Session A's `tools/nv_e3_flr.py`).
3. **Uninterruptible RPC with no timeout on either side.** `server.c` blocks in `IOConnectCallMethod`/`IOConnectMapMemory64` (`THREAD_UNINT`),
   the client blocks in `_recvall`. A dext call that stalls (Reset while the tunnel drops, PrepareDMA under DART pressure) freezes the server in
   `U`, the dext's serial queue cannot deliver `Stop()`, IOPCIFamily spins. The client should at least set `SO_RCVTIMEO`/`settimeout` so Python
   fails instead of joining the wedge, and prefer MMIO (never proxied) over config accesses in polling loops.
4. **FLR timing on GB202 (H3).** `RESET` returns after the kernel's fixed 100 ms; master then polls GFW readiness for 10 s, which April did
   not. Posted MMIO writes to a device still in reset vanish silently; reads return `0xFFFFFFFF` → `KeyError: 63`. Failures, not wedges, but
   they leave Bus Master on and GSP half-booted for hazard 1.
5. **Silent write drops.** `MMIO_WRITE` outside the BAR window (e.g. a GPFIFO area above 256 MB, or `validate_bar` on a BAR that failed to map)
   is dropped without any error; a doorbell that never lands looks like a GPU hang and ends in a 30 s `wait_timeout_ms` and a teardown (1).
6. **Two clients.** A second Python process is queued by `listen(1)` and hangs in `recv` (state `S`, killable); `flock` normally prevents it,
   but a stale lock file or a different `$TMPDIR` (e.g. `sudo`, system python) bypasses it.

## 7. Open items handed to W2/W3

- W2 (dext/server): structured `os_log` per RPC with timings, `SO_RCVTIMEO` on the client socket, Bus-Master revoke in `Stop_Impl`, a
  `QUIESCE` command (clear BM, FLR) usable from a fresh process, per-buffer sysmem unmap, and lifting `MAX_SYSMEM`/32-segment limits.
- W3 (stability): decide the exit protocol (quiesce vs resident GSP), confirm whether `IOPCIDevice::Reset` on this tunnel is FLR or SBR
  (Session A's E3), test config-space BAR base vs the fake `bar_info` addresses, and measure the DART budget on a real run.
