# Launch-overhead prior-art research (2026-09-14, Session A's agent)

Research commissioned by Antonio: how others solve the decode per-launch-overhead problem, with code from
similar Linux/BSD projects. Full agent report; condensed to the actionable core.

## Headline verdict
Our converged diagnosis is correct and matches all prior art. Decode is per-launch-latency bound, and the
dominant cost is the GPU front-end doing **non-posted reads of per-launch descriptors from HOST memory over
Thunderbolt**. Decisive fact: the Windows 92.6 tok/s reference runs on the **same AORUS enclosure over the same
TB link**, so ~5.5 us/launch is provably achievable over TB. **The link is not the floor; our submission memory
model is.** The ~5x is a software fix, not an eGPU tax.

Calibration: Windows 92.6 includes MTP speculative decode (~54% draft accept), so part of that gap is fewer
forward passes, not pure per-launch parity. Agent estimate: descriptor placement alone ~28.7 -> ~6-9 us/launch
(~3-4x decode, ~50-75 tok/s); graph-replay and later spec-decode close the rest. (A's caveat: this 3-4x estimate
is not yet consistent with our own test_hw_chain 8.5 us/launch on the same host-fetch path — see "open wrinkle" —
so measure the payoff before banking it.)

## Mechanism (why ~5.5 us vs our ~28.7)
- PCIe/TB: memory WRITES are posted (fire-and-forget, pipeline at bandwidth); memory READS are NON-POSTED
  (requester waits for a Completion TLP). TB inflates read RTT to ~1-2 us/round-trip.
- Per launch the GPU fetches, from host over TB: the pushbuffer method stream (PBDMA), the QMD (384 B, CWD), and
  constant buffer 0 (~896 B, SM at kernel start). These are DEPENDENT (pointer-chase: ring -> pushbuffer -> QMD
  -> cbuf -> next QMD), so they can't pipeline: ~20 dependent reads x ~1.4 us ~= 28 us. Cost is round trips, not
  bytes (~45 MB/s of control data = 1.3% of TB4).
- In VRAM the same reads are ~free (GPU global-mem latency ~300-600 ns, thousands of outstanding requests, hide
  behind compute) -> healthy ~5.5 us/launch is basically the kernel's own VRAM time.
- Decode is NOT bandwidth-bound over TB: the 15.65 GB of weights are read from VRAM at ~1.79 TB/s, never cross TB.

## Fix hierarchy
1. **PRIME (~3-4x): move pushbuffer + QMD + cbuf0 into CPU-visible VRAM (the reserved BAR pool).** Machinery
   already exists; ~a two-flag change plus a staging refactor.
2. **LARGE (after 1): record-once / replay the per-token QMD chain (CUDA-graph equivalent)**, patching only
   KV-cache pointers -> removes per-token CPU rebuild AND makes the token one continuous dependent-QMD chain
   (no seam drains).
3. **LOW: one-release-per-chain, deeper chains, cutting per-QMD semaphore traffic** — real but second-order
   (releases are posted writes; we're not write-bound). Correctly deprioritized.
- **Do NOT:** deeper chains / less semaphore traffic as primary levers; never route quantized decode via cuBLAS.

## Change sites in our code (agent-located)
- `libtinynv/src/exec.c:657` (and :651) — `tinynv_mm_alloc_buffer(... host=1, force_devmem=0 ...)` for the cmd/
  QMD/cbuf arena = **the bug**. Change to `host=0, cpu_access=1, force_devmem=1`.
- `libtinynv/src/pt.c:399` — sysmem fallback for the small window; :431-444 the bar_pool path (sets `in_bar_pool`).
- `libtinynv/src/mmu.c:159-175` — `tinynv_mm_reserve_bar_pool` (already called on boot, gsp.c:533; 4 MB) -> grow
  to ~32-64 MB (trivial vs 256 MB BAR; CMD_BYTES is already 4 MB).
- Build each batch in a HOST scratch buffer, then one block write (`nv_wr_block`) into the VRAM slot (incremental
  write-combined writes through the window reintroduce many small transactions). Cache a per-kernel VRAM QMD
  template and patch only changed dwords.
- Keep the timeline semaphore in HOST memory (GPU posted-writes it, host reads locally — both cheap). Already right.
- `libtinycudart/cudart_api.c:81` — graph capture currently returns cudaErrorNotSupported; implement replay at the
  libtinynv level (record QMD chain once, replay + patch) rather than the full cudaGraph* API.

## How the others place descriptors (validates the lead)
- **tinygrad (oracle):** GPFIFO ring in CPU-visible VRAM, write-combined (`ops_nv.py:652`, cpu_access+force_devmem+
  WRITECOMBINED); reserved BAR pool for small-BAR (`support/system.py:285` `_reserve_bar_pool`, :309 routing);
  cmdbuf/QMD device buffers.
- **Mesa NVK:** QMD -> VRAM and cbufs -> VRAM (`nvk_cmd_buffer.c:311-322`, :250-272). Pushbuffer -> GART by design
  with the telling comment (`:210-220`): "command buffers tend to be read-once so there's not much benefit to
  putting them in VRAM" — **that assumption is exactly what breaks over TB.** NVK does NOT chain QMDs (we + tinygrad
  are ahead there).
- **nouveau:** ring in GART by default, `nouveau.vram_pushbuf=1` module param exists precisely for expensive PCI reads.

## CUDA Graphs nuance (Q3)
Graphs collapse many launches into one submission, paying validation once at instantiate. Native numbers (V100,
2.9 us kernel): launch+sync 9.6 us, async 3.8, graph 3.4. On native, async->graph saves only ~0.4 us (CPU is the
whole story there, descriptors already VRAM). We already banked the async win, so a HOST-built graph alone won't
fix our 28.7 us (that's GPU-side TB reads). A graph helps US because its command buffer is VRAM-resident and it
removes per-token rebuild + seams. So: **descriptor placement first, graph-replay second.**
- llama.cpp CUDA graphs (PR #6766): H100 Llama-2-7B Q4_K_M 143->164 tok/s (+14%), "graph exec ~40% faster",
  bottleneck was "gaps between kernels... mostly GPU-side launch overheads". Captures whole token, patches KV
  copy dst via cudaGraphExecUpdate; decode-only.
- WARNING: CUDA graphs currently HANG on RTX 5090/sm_120 upstream (#27330, workaround GGML_CUDA_DISABLE_GRAPHS=1).
  A GPU-side ring WE control is safer than mirroring the upstream graph path on Blackwell.

## Our other methods vs practice (validated)
- QMD chaining: matches tinygrad exactly (`ops_nv.py:178-184` vs our `qmd.c:241-258`). Blackwell sm_120 =
  BLACKWELL_COMPUTE_B (0xCEC0), QMDV05_00, dependent fields DEPENDENT_QMD0_ENABLE MW(336)/ACTION MW(339:337)/
  PREFETCH MW(340)/POINTER MW(415:384) in NVIDIA/open-gpu-doc classes/compute/clcec0qmd.h.
- Per-QMD release: tinygrad releases on the LAST QMD only (`ops_nv.py:143-146`); we require it on every link
  (`qmd.c:223-239`). Releases are posted writes -> not our bottleneck. Match the oracle for cleanliness, low priority.
- Timeline semaphores: we converged on the standard (per-queue monotonic slot, cross-queue wait names other
  queue's (slot,value)); Vulkan timeline semantics + NVK (DRM syncobj) + tinygrad all agree. Our shared-slot bug
  was the classic trap; per-queue fix is the standard answer. Copy-engine report semaphore is 32-bit + needs
  FLUSH_ENABLE (we have it, submit.c:82-88).
- Doorbell/GPFIFO: our ring-entry -> GP_PUT -> doorbell with fences matches UVM `internal_channel_submit_work`
  (write entry -> mb -> GP_PUT -> wmb -> doorbell). nouveau adds a USERD readback to flush BAR1->vidmem before the
  doorbell; ensure a barrier between GP_PUT and doorbell on any direct-MMIO path.
- Kernel efficiency is NOT the decode problem: decode uses MMVQ (mul_mat_vec_q, dequant-on-the-fly, bandwidth-
  bound); same cubins as Windows at 5.5 us. cuBLAS decode would be ~3.6x worse (materializes fp16).

## Open wrinkle (A, to resolve before trusting the 3-4x estimate)
The agent attributes ~28.7 us to descriptor fetch, but our test_hw_chain (chained trivial vecadd, host-resident
QMD/cbuf, same fetch path) measured 8.5 us/launch, and the 27B chain-depth sweep 32->128 moved <1%. So either real
decode incurs more/less-hidden TB reads than the trivial chain (why?), or part of the 28.7 is real-kernel-specific
and VRAM placement recovers less than 3-4x. Resolve with a GPU-side per-kernel timestamp (split exec from gap), and
measure test_hw_chain + 27B before/after the VRAM change rather than banking the estimate. Direction (VRAM
placement) is right regardless and cheap with existing machinery; the wrinkle is about payoff size.

## Prior-art projects worth reading
tinygrad (ops_nv.py, support/system.py, support/hcq2.py, engine/jit.py HCQGraph); Mesa NVK (nvk_cmd_buffer.c,
nvk_cmd_dispatch.c, nak/qmd.rs); nouveau (nouveau_chan.c, nvif/chanc36f.c); open-gpu-kernel-modules (clc86f.h,
clc6b5.h, uvm_channel.c); open-gpu-doc (clcec0qmd.h); gdev (USENIX ATC'12); envytools/rnndb; ZLUDA; nova.
CUDA Graphs: developer.nvidia.com/blog/cuda-graphs/, llama.cpp #6763/#6766, sm_120 hang #27330, roofline #28196.
PCIe posted-vs-non-posted: xillybus.com/tutorials/pci-express-tlp-pcie-primer-tutorial-guide-1.
