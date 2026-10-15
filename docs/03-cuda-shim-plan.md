# CUDA-on-Mac plan — native shim over a C userspace driver (chosen 2026-09-13)

Antonio's decision (2026-09-13): stop chasing tinygrad's ~60%-of-llama.cpp ceiling; **get real CUDA working on the Mac** via a
native shim, not VM passthrough. This is Option A from `docs/research/cuda-shim-feasibility.md` (read it for the verified detail).
Effort: **16–26 person-weeks**, decomposable and testable on the Linux box first. Ceiling: **within ~5–15% of Linux CUDA decode**,
prefill identical (GPU-side SASS is byte-identical to a real CUDA run).

## 1. What we are building (three layers)

```
llama.cpp + ggml-cuda  (host code compiled on the Mac by clang; device .cu compiled to sm_120 cubins on the Linux box)
        │ links against ↓                                    (NO NVIDIA binary runs on the Mac; cubins are the user's own)
  libtinycudart.dylib   — cudart ABI: the ~140 calls ggml-cuda uses (vendors/hip.h is the spec); module/cubin loading,
        │                 kernel launch (QMD), memcpy (copy engine), streams (GPFIFO queues), events (timeline signals)
  libtinycublas.dylib   — ~8 entry points (cublasGemmEx/Sgemm + strided/batched) on one sm_120 tensor-core GEMM cubin
        │
  libtinynv.(dylib/a)   — C port of tinygrad's userspace NVIDIA driver: GSP-RM boot (Blackwell FSP/COT, fw 570.144),
        │                 RM object model over RPC, MMU v3 page tables, GPFIFO channels + userspace doorbell, QMD v5 launch,
        │                 copy engine, timeline semaphores, fault reporting. PCI backend: Linux sysfs/vfio OR macOS TinyGPU socket.
        │ TinyGPU dext (macOS) / sysfs+vfio (Linux)
  RTX 5090 (GB202, sm_120)
```

**Build config to start:** `-DGGML_CUDA_FORCE_MMQ=ON -DGGML_CUDA_NO_VMM=ON -DGGML_CUDA_GRAPHS=OFF -DGGML_CUDA_COMPRESSION_MODE=none`.
cuBLAS is always *linked* by ggml but with FORCE_MMQ the quantized path never calls it; only residual F16/F32 GEMMs (mostly
prompt-time) hit our `libtinycublas`. Graphs deferred (a later decode-latency optimization). Flash-attention kernels are pure cubins.

## 2. Why a shim and not the alternatives (settled)
- **Not `libcuda.dylib` alone**: nothing on macOS calls it; ggml links cudart, whose compiler-generated host stubs
  (`__cudaRegisterFatBinary` etc.) are the real ABI. We implement the cudart ABI, compiled by clang on the Mac.
- **Not a ggml backend (Option B)**: same driver cost, lower ceiling, throws away ggml-cuda's host dispatch. We use "launch one
  ggml-cuda cubin from the driver" only as an early *milestone*, not the destination.
- **Not VM passthrough**: rejected (Apple entitlement latency, TB5-enclosure risk). The GPU stays on the host TinyGPU path.

## 3. The north star: `libtinynv` is the reusable asset (Antonio's layering insight, 2026-09-13)
The shim only covers llama.cpp's CUDA surface; PyTorch/vLLM need ~10× more (cuBLASLt, cuDNN, cuSPARSE, NCCL, Triton's runtime PTX
compiler). Widening the shim per-library toward PyTorch is a 1–2 year path — **not** how we get the rest of the ecosystem.

**The path to the full ecosystem without passthrough** (a real *phase 2*, not a free bonus, and not yet feasibility-studied to the
depth of the shim): put a small **paravirtual "nvidia.ko"** kernel module inside a Linux guest (Apple Virtualization / QEMU-HVF, no
PCIe passthrough) that forwards the RM/UVM ioctl surface (`NV_ESC_RM_ALLOC/CONTROL`, `UVM_*`) over vsock/virtio to **`libtinynv` on the
host**. The guest then runs the entire unmodified NVIDIA userspace stack (all CUDA, cuBLAS, PyTorch, vLLM) because it believes it has
a real GPU driver — the ioctl surface *is* the RM API `libtinynv` already speaks. No VFIO dext, no Apple PCI entitlement, no TB5
passthrough risk; the GPU stays on our host path.
- Tractable for **inference**: device memory lives in real VRAM (handles only cross the boundary); per-token H2D/D2H transfers are
  tiny, so a bounce-buffer through host DMA (the dext's `PrepareDMA`) is cheap; unified-memory page-faulting (the hard part) is
  mostly unused by llama.cpp/vLLM/PyTorch-inference, so it can be stubbed.
- Hard parts: the guest fake-driver, cross-VM DMA/memory coherence, object lifetime, and any UVM path. Deserves its own feasibility
  study before commit.
**Design consequence now:** build `libtinynv` as a clean, reusable driver library with the RM/UVM operations as a defined interface
(not entangled with the cudart shim), so phase 2 is a natural extension rather than a rewrite.

## 4. Where the work runs
- **Pop!_OS 3090 box** — primary dev + regression target for `libtinynv` (Ampere GSP path, recoverable hangs, `PCIDevice` sysfs
  backend = the same code), the kernel-compile host (nvcc/clang → sm_120 cubins; the warm compile server on :6481 already exists),
  and the correctness oracle (`test-backend-ops`, `llama-bench` reference numbers). The card runs there natively.
- **The 5090** — validated in the Pop!_OS box (native, recoverable) before ever touching the Mac.
- **Mac (M4 Max)** — the target: host-compile ggml-cuda with clang, link the three libs, run against the eGPU over TinyGPU.
- **Windows box** — reference numbers (done today: 27B Q4_K_M = 92.6 tok/s decode / ~1370 prefill).

## 5. Division of labour (two sessions)
- **Session B → `libtinynv`** (its existing expertise; it already ported the Python driver and knows the RM/QMD/GSP path). Deliver
  the C driver, Linux-first, with a PCI-backend interface (sysfs/vfio + TinyGPU socket) and an RPC-trace recorder to diff C vs the
  Python reference. This subsumes B's earlier hcq2-port work — the Python port stays the reference oracle.
- **Session A → toolchain + shim + integration**: the clang device/host split, `libtinycudart`, `libtinycublas`, ggml/CMake patches,
  `test-backend-ops` and `llama-bench` bring-up, launch-latency tuning, graphs later.

## 6. Milestones (from cuda-shim-feasibility.md §2.4/§8, calibrated to today's numbers)
| # | Milestone | Wk | Gate |
|---|---|---|---|
| M0 | Reference baselines: AORUS box on the Pop!_OS box, `nvidia-driver-590/595-open`, `llama-bench` on Qwen3.8-27B Q4_K_M + a dense model; tinygrad's NV runtime there in native `DEV=PCI+NV` sysfs mode (the C-driver dev target) | ≤1 | numbers recorded; native tinygrad boots on the 3090 |
| M1 | **Toolchain** — ✅ **DONE 2026-09-13, GO — and 68/68 real ggml-cuda TUs host-compile on the Mac (full sweep).** nvcc→sm_120 fatbin on Linux; Homebrew clang 22 `--cuda-host-only -nocudalib -fcuda-include-gpubinary` on the Mac embeds it in the Mach-O; ran the full register→configure→setup-args→launch flow against a logging shim. Targeting the CLASSIC launch ABI (clang 22 ≠ CUDA 13). Artifacts: cuda-shim/spike/. | done | ✅ Mac host-compiles + embeds Linux device code natively; 8-symbol libtinycudart spec pinned |
| M2 | **Launch an external cubin from the Python runtime** — 🟡 **CORE PROVEN 2026-09-13** (external nvcc sm_86 vecadd → tinygrad QMD on the 3090 → 256/256 correct; param ABI at c[0x0][0x160] matches tinygrad packing; 0x17 layout decoded). Remaining: interleaved-arg kernel + one real ggml TU (rms_norm); grid>1 deferred to libtinynv's QMD builder. | ≤1 | ✅ external SASS runs + param ABI pinned |
| M3 | **`libtinynv` C port, Linux-first** | 6–10 | boot GSP on 3090 then 5090; alloc/map memory; submit a QMD; signal/wait; CE copy; fault report |
| M4 | **`libtinycudart` + `libtinycublas`** — 🟢 both skeletons up 2026-09-13 (cudart ABI + fatbin extract + classic-launch marshalling; cuBLAS 8-entry subset over one GEMM cubin; 68/68 ggml TUs host-compile). Remaining: swap stub→libtinynv, `test-backend-ops` on HW. | 3–5 +1–3 | all-pass on Linux with `libtinynv`, then on the Mac |
| M5 | **`llama-bench` + real models** (MoE/FA paths), launch-latency tuning, graphs | 4–6 | 27B Q4_K_M within ~15% of the 92.6 reference on the Mac eGPU |

## 7. Inherited constraints to respect (from the dext/firmware)
- **32 DMA segments per `PrepareDMA`** (silently truncated): `cudaMallocHost` = a pool of ≤32-segment mappings; `cudaHostRegister`
  only for page-aligned modest buffers, else report unsupported and let ggml use unpinned copies. (Session A already hit and worked
  around the related 128-mapping cap in the tinygrad path — same dext limit.)
- **GSP firmware 570.144 vs USB4-eGPU bus type**: a candidate cause of Blackwell-over-TB init flakiness; moving to 580/590 GSP blobs
  means regenerating the RPC structs (autogen `nv_580`/`nv_610` exist) and re-validating boot. Test on the 3090 (fw path differs) first.
- **BAR1 over Thunderbolt** small-BAR (256 MB): page tables in the BAR window, GSP boot structs in sysmem — `libtinynv` must mirror
  tinygrad's small-BAR handling.
- The stability recipe carries over: never unload GSP at exit, clear bus-master, standalone FLR to recover, replug clears a latched DART flag.

## 8. Immediate next actions
1. **M0 setup (needs the box on the Pop!_OS machine — but it has no Thunderbolt).** Correction: the Pop!_OS box cannot host the
   AORUS enclosure (no TB controller). So the C-driver dev target on the 3090 is the **3090 itself in its PCIe slot** (native
   `DEV=PCI+NV` / sysfs), which is the recoverable Ampere GSP path we want for bring-up; the **5090 reference** comes from the
   Windows box (done: 92.6/1370) and, when needed, the 5090 in the Mac over TinyGPU. Re-confirm the plan with this constraint.
2. **M1 toolchain spike** — the cheapest go/no-go for the whole native-compile approach. Start here in parallel with M0.
3. **M2** — launch one ggml cubin from the existing Python NV runtime on the 3090 box (no C yet); this de-risks the kernel ABI.
4. Session B: pivot from the hcq2-remote port to the `libtinynv` C-port design (the Python port remains the reference oracle).
5. Open a dedicated feasibility study for the **phase-2 paravirtual-driver-in-a-guest** path before committing to it.

## 9. Kill / pivot criteria
- If M2 shows unmodified ggml cubins can't be launched via the QMD path (param ABI or smem blocker that isn't ~40 lines) → reconsider
  Option B (ggml backend) vs A.
- If `libtinynv` GSP boot on the **5090 over Thunderbolt** proves unrecoverable in a way the 3090 didn't predict → the C driver is
  still valid for a Linux-hosted 5090; the Mac-native leg may need the 580/590 firmware bump first.
- If, after M4, launch latency can't get under ~10 µs/call → graphs (batch capture → one GPFIFO submit) become mandatory, not optional.
