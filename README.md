# macuda — CUDA inference on Apple Silicon

**macuda runs CUDA workloads on an NVIDIA RTX 5090 connected to an Apple Silicon Mac over Thunderbolt.** A C userspace driver and CUDA compatibility libraries support llama.cpp and stable-diffusion.cpp with their CUDA backends, without an NVIDIA macOS GPU driver.

The project combines GPU driver development, runtime compatibility and inference profiling. In the 19 September 2026 language-model batch, decode throughput reached **91–101% of the Windows reference**, using the same physical GPU and Thunderbolt enclosure. The three dense models reached **97–98%**. Results, configuration differences and validation criteria are documented below.

This is a research implementation for the tested applications and hardware. It does not provide general CUDA compatibility or established support for PyTorch and vLLM.

## Architecture

```text
llama.cpp / stable-diffusion.cpp with ggml-cuda
        │
libtinycudart.a     CUDA runtime compatibility: modules, launches, memory, streams, events and graphs
        │
libtinycublas.a     The cuBLAS entry points used by ggml, backed by tensor-core GEMM kernels
        │
libtinynv.a         C userspace driver: firmware initialisation, memory management and GPU submission
        │
TinyGPU.app        tinygrad's signed DriverKit extension and Unix-socket hardware interface
        │
RTX 5090           GB202, 32 GB VRAM, connected through an AORUS AI BOX over Thunderbolt
```

`libtinynv` is a C port of the userspace NVIDIA driver in [tinygrad](https://github.com/tinygrad/tinygrad), using NVIDIA's published `open-gpu-kernel-modules` definitions. It boots the GPU's signed GSP firmware, manages MMU page tables and submits kernels through hardware queues.

The build separates CPU and GPU compilation. Homebrew clang compiles host code into native macOS objects; Linux `nvcc` compiles GPU kernels for `sm_120a`. Linux compilation can run in a Docker container on the Mac or on a separate machine over SSH. The resulting kernels are embedded in the macOS application. Inference uses macuda's driver and runtime; NVIDIA's firmware executes on the GPU.

For implementation details, see the [architecture and integration plan](docs/03-cuda-shim-plan.md), [driver design record](docs/driver/libtinynv-design.md) and [DriverKit transport](docs/driver/architecture.md).

## Quick start

Run the prerequisite check before installing or building:

```sh
git clone https://github.com/Davinchy/macuda
cd macuda
sh install.sh check
```

Check mode reports missing prerequisites without installing packages or building the project. Install Homebrew and Xcode Command Line Tools first if requested. The full installer prompts before supported dependency installations and then builds the stack:

```sh
DEFS_EXTRA=-DGGML_CUDA_USE_GRAPHS sh install.sh
```

The explicit graph flag matches the published graph-enabled benchmark configuration. The current build script does not enable `GGML_CUDA_USE_GRAPHS` unless it is supplied through `DEFS_EXTRA`.

The installer can install missing CMake and LLVM packages, Docker Desktop and TinyGPU.app. Start Docker Desktop if using the local compiler container. TinyGPU's DriverKit extension still requires user approval through macOS; after the app is installed, run:

```sh
/Applications/TinyGPU.app/Contents/MacOS/TinyGPU install
```

Follow the macOS approval prompt in System Settings. Building does not require the GPU, but inference requires the connected enclosure and an approved, running extension.

Download one of the validated GGUF models. These commands use the `hf` CLI from `huggingface_hub`:

```sh
# Dense model and its multi-token prediction (MTP) head
hf download unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-UD-Q4_K_M.gguf MTP/mtp-Qwen3.8-27B-Q4_0.gguf --local-dir models/

# Additional validated models
hf download unsloth/Qwen3.5-35B-A3B-GGUF Qwen3.5-35B-A3B-Q4_K_M.gguf --local-dir models/
hf download bartowski/Meta-Llama-3.1-8B-Instruct-GGUF Meta-Llama-3.1-8B-Instruct-Q8_0.gguf --local-dir models/
```

Qwen3.8 27B is used by the serving and validation tools. Model files may also be symlinked from an external drive; storage throughput affects load time. See [model storage notes](models/README.md).

Before running a workload, read the [operating protocol](#run-and-operate) and confirm that preflight reports `VERDICT: OK`:

```sh
sh tools/preflight.sh
sh tools/nv_shim_step.sh A bench models/Qwen3.8-27B-UD-Q4_K_M.gguf
```

## What you need

| Component | Tested configuration or requirement |
|---|---|
| Mac | Apple Silicon; development and measurements use an M4 Max MacBook Pro on macOS 27.0 |
| GPU | RTX 5090 in a Thunderbolt enclosure; measurements use an AORUS AI BOX |
| Host toolchain | Xcode Command Line Tools or Xcode, CMake, Python 3 and Homebrew LLVM at `/opt/homebrew/opt/llvm` |
| Hardware access | TinyGPU.app and its approved DriverKit extension, `org.tinygrad.tinygpu.driver2` |
| GPU compiler | Docker with the configured CUDA image, or a Linux machine with CUDA 13 accessible over SSH |
| Storage | Build artifacts, downloaded dependencies and model files; the installer checks available space |

**Compiler options.** With `TINYCC_HOST` unset, `tinycc` uses `nvidia/cuda:13.0.3-devel-ubuntu24.04` through Docker. The container compiles device code only and requires no GPU access or NVIDIA container runtime. To use an SSH compiler host, set `TINYCC_HOST=user@linux-box` before installation or the build; use `TINYCC_KEY` if an explicit SSH identity is required. The published benchmark builds used the SSH path.

**Fetched dependencies.** `setup.sh deps` obtains NVIDIA headers at commit `81fe4fb` (release 570.86.16), hash-pinned GSP firmware 570.144 and CUDA Toolkit headers. Toolkit headers come from the configured Linux host, the Docker image or `CUDA_INCLUDE_SRC`. These dependencies are not committed to this repository. The header and firmware versions are intentionally different and have been validated together on the test GPU.

**Optional development dependencies.** Some offline reference tests and recovery tools use the tinygrad checkouts and Python environment described in [env.sh](env.sh). These are separate from the application build prerequisites.

## Measured performance

### Language models

Measured **19 September 2026, 21:07–21:11**, on driver build `abd0458`, with CUDA graphs enabled, device-to-device copies handled by kernels and ggml-cuda host code compiled at `-O2`.

Settings: `llama-bench -ngl 99 -p 256 -n 128 -r 3`. Decode (`tg128`) measures generation of 128 tokens; prefill (`pp256`) measures processing of a 256-token prompt. Values are tokens per second, with macuda results shown as mean ± standard deviation over three repeats.

| Model | macuda decode | Windows decode | Decode ratio | macuda prefill | Windows prefill |
|---|---:|---:|---:|---:|---:|
| Llama 3.1 8B Instruct Q8_0 | 161.0 ± 1.0 | 164.5 | 98% | 8,927 ± 349 | 11,118 |
| Nemotron Nano 9B v2 Q6_K | 141.5 ± 0.9 | 140.6 | 101% | 6,894 ± 175 | 4,031 |
| Gemma 4 12B it Q4_K_M | 133.4 ± 1.0 | 136.4 | 98% | 5,617 ± 207 | 4,169 |
| Gemma 4 26B-A4B it Q4_K_XL | 225.5 ± 6.5 | 227.8 | 99% | 7,063 ± 207 | 5,478 |
| Qwen3.8 27B UD-Q4_K_M | 73.4 ± 0.1 | 75.3 | 97% | 2,592 ± 69 | 2,447 |
| Qwen3.5 35B-A3B Q4_K_M | 223.0 ± 1.9 | 245.0 | 91% | 5,250 ± 209 | 4,726 |

Ratios are calculated from the displayed means and rounded to the nearest whole percent. The Windows reference was recorded on **14 September 2026** with driver 595.79, using the same GPU and AORUS enclosure over Thunderbolt at PCIe Gen 4 ×4 and byte-identical model files. It used llama.cpp b10970; the macuda tree was b10950 plus one commit. The recorded source comparison found only an AMD-specific change in the intervening ggml-cuda revisions.

These are comparisons of complete host-and-driver configurations. Host load, thermals and run-to-run variance remain relevant. The Mac's load average was 5.9–8.7 during this batch, and some Windows prefill measurements have large standard deviations. Individual optimisation claims use interleaved A/B tests on the same Mac.

Sources: [Mac benchmark batches](docs/bench/mac-batch-20260919.md) and [Windows reference](docs/bench/windows-reference-20260914.md).

### Image generation

The following totals come from the **18:12–18:16 batch on 19 September**, build `05d8b4a`. Total time measures stable-diffusion.cpp's `generate_image` operation. All runs use seed 42 and the reference prompt.

| Model | Settings | macuda total | Windows total |
|---|---|---:|---:|
| SDXL Turbo | 512 × 512, 4 steps, CFG 1.0 | 2.78 s | 4.11 s |
| Stable Diffusion 1.5, fp32 | 512 × 512, 20 steps, CFG 7 | 3.71 s | 3.86 s |
| Z-Image Turbo | 1024 × 1024, 8 steps, CFG 1.0 | 10.06 s | 9.46 s |

The Z-Image result is a warm rerun; the initial run took 16.71 s with a cold encoder and VAE. PNG outputs matched earlier macuda validation runs byte for byte. The Windows log records thermal throttling, which affects timing comparisons. Later image measurements are retained separately in the [batch record](docs/bench/mac-batch-20260919.md).

### Serving and speculative decoding

These measurements are from separate serving tests, not the language-model batch above:

| Configuration | Result | Conditions |
|---|---|---|
| Qwen3.8 27B + MTP, single stream | 124 tok/s on the first request; 112 tok/s mean | 62-minute soak with byte-identical greedy probes; Windows greedy reference: 124.7 tok/s |
| Qwen3.8 27B + MTP, eight slots | Approximately 207 tok/s aggregate | Temperature 0.6; 440 requests and zero errors in a 20-minute soak |

Speculative throughput depends on prompt content and draft acceptance. Single-stream throughput and aggregate throughput across concurrent requests measure different workloads.

## Build details

Each stage can be run separately from the repository root:

```sh
sh setup.sh deps
sh setup.sh llama
sh setup.sh sd
sh setup.sh shim
DEFS_EXTRA=-DGGML_CUDA_USE_GRAPHS sh setup.sh cuda
sh setup.sh link
```

`setup.sh llama` pins llama.cpp to `ad6c668` and applies upstream fix `2f53959`. `setup.sh sd` pins stable-diffusion.cpp to `59c23bc`, uses llama.cpp's ggml and applies the integration patch in [patches/](patches/). `LLAMA_SRC` and `SD_SRC` can identify local source clones. The scripts manage those dependency checkouts; preserve local edits before rebuilding them.

**CUDA archive.** `cuda-shim/build/libggml-cuda.a` and the CUDA-enabled backend registry object are build outputs tied to the pinned llama.cpp source. The build uses `GGML_CUDA_FORCE_MMQ` and `GGML_CUDA_NO_VMM`; graph support is added explicitly above. `JOBS` controls compilation parallelism. For SSH builds, synchronise `llama.cpp/ggml/` to the location used by `TINYCC_GGML_LINUX` (default `ggml` on the remote host) before compiling. Docker builds mount the local tree directly.

`TINYCC_REUSE_FATBIN=1` reuses device fatbins left in `/tmp` for a host-only rebuild. Use it only when the GPU source and device compilation settings are unchanged. The runtime's copy kernels and cuBLAS-compatible GEMM kernel ship as committed cubins; their source files record the compilation commands.

**Application linking.** llama.cpp first builds as a CPU-only static tree in `build-null`. The link scripts add the CUDA backend, compatibility libraries and LLVM demangler. The resulting `*-null` binaries use the driver's null device until `TINYNV_SOCKET` selects a live TinyGPU server. The suffix does not mean they are restricted to offline execution.

**Build provenance.** `libtinynv` embeds its Git commit, with `-dirty` for an uncommitted tree or `nogit` outside a checkout. The runtime prints the ID at startup, and the runner checks it before hardware access. Use that ID to associate a result with its build. Historical results and corrections remain in the [driver design record](docs/driver/libtinynv-design.md) and [19 September handoff](docs/handoff-2026-09-19.md).

## Run and operate

Hardware access follows one sequence: **read-only preflight → acquire the lock → run one GPU process → release the lock**, leaving firmware resident and the card idle. Concurrent clients can hang this setup. Use the supplied runners to coordinate access.

Power the laptop from its own adapter. The project observed Thunderbolt re-enumeration during heavy Metal workloads when the enclosure also supplied laptop power; the same workloads did not reproduce it with the separate adapter. Coordinate jobs that allocate more than approximately 20 GB of host or Apple-GPU memory with whoever owns the GPU lock.

```sh
# Check hardware state without opening the GPU
sh tools/preflight.sh

# Operator validation and a decode benchmark
sh tools/nv_shim_step.sh A opverify
sh tools/nv_shim_step.sh A bench models/Qwen3.8-27B-UD-Q4_K_M.gguf 128

# Greedy generation and speculative decoding
sh tools/nv_shim_step.sh A simple models/Qwen3.8-27B-UD-Q4_K_M.gguf 96 'Explain GPU memory bandwidth.'
sh tools/nv_shim_step.sh A spec models/Qwen3.8-27B-UD-Q4_K_M.gguf models/MTP/mtp-Qwen3.8-27B-Q4_0.gguf 128 4
```

The runner prints the actual driver settings and build ID, tracks abnormal exits and releases its lock. `BIN` selects an alternate binary directory; `DRY=1` uses the null device for host-path diagnostics, not output validation. `QUIESCE=1` requests a cold stop. The `tinynv-smi` tool in `cuda-shim/build/shim/nv/` reads published temperature, power and clock data without opening the GPU.

### Serve an OpenAI-compatible API

The download command above stores the draft head under `models/MTP/`, while `serve.sh` defaults to a top-level model path. Set `MTP` explicitly:

```sh
MTP=models/MTP/mtp-Qwen3.8-27B-Q4_0.gguf sh tools/serve.sh start
sh tools/serve.sh status
sh tools/serve.sh test
```

The API is available at `http://127.0.0.1:8090/v1`. `PARALLEL` selects the number of concurrent slots; `MTP=` disables speculative decoding. To run a two-hour soak with a 256-token limit, then stop the server:

```sh
sh tools/soak.sh 2 256
sh tools/serve.sh stop
```

### Recovery and diagnostics

Use [tools/preflight.sh](tools/preflight.sh) to determine whether a run can proceed. The recovery tools `nv_quiesce.sh`, `nv_e3_flr.py` and `nv_temp.py` use the tinygrad environment configured by `env.sh`. A disconnected GPU, a latched `pci-dart-error-data` flag or a failed link check may require physically reconnecting the enclosure. Do not use `systemextensionsctl reset` as a recovery shortcut.

The current driver defaults include asynchronous submission, a VRAM command arena, chain depth 128, a 64 MB descriptor region, token-aligned delta delivery, tail-only timeline release, inline small uploads and kernel-based device-to-device copies. Inspect the startup line to confirm the settings for each run.

| Setting | Purpose |
|---|---|
| `TINYNV_SYNC=1` | Wait synchronously for each batch |
| `TINYNV_CHAIN_DEPTH=N`, `TINYNV_DESC_MB=N` | Adjust chain depth or descriptor-region size |
| `TINYNV_ARENA_VRAM=0`, `TINYNV_ARENA_DMA=0` | Select diagnostic arena and descriptor-delivery paths |
| `TINYNV_DELTA_DELIVERY=0`, `TINYNV_DELTA_REWIND=0`, `TINYNV_TAIL_RELEASE=0` | Disable the corresponding descriptor optimisations |
| `TINYNV_INLINE_UPLOAD=0`, `TINYNV_QMD_MEMBAR=sys` | Disable inline uploads or restore system-scope barriers on all descriptors |
| `TINYNV_DTOD_VIA_COMPUTE=0` | Route device-to-device copies through the copy engine |
| `TINYNV_GRAPH_RESIDENT=1` | Enable experimental resident graph replay; disabled by default |
| `TINYNV_SENSORS=0` | Disable sensor publication |
| `TINYNV_FW_DIR`, `TINYNV_SOCKET` | Select firmware files or the TinyGPU socket |
| `TINYCUDART_TRACE=1`, `TINYCUDART_STATS=1` | Enable runtime traces or per-kernel statistics |
| `TINYCUBLAS_TC=0` | Use the diagnostic scalar GEMM path |

Disabling per-launch cache invalidation has produced different greedy outputs even when operator tests passed. Keep `TINYNV_QMD_INVALIDATE` as a diagnostic, not a performance setting.

## Validation

Offline checks run without the GPU:

```sh
make -C cuda-shim test
```

The suite covers page tables, descriptors, cubin loading and relocation, resource-management refusals, boot replay and required CUDA/cuBLAS symbols. Some tests need tinygrad reference checkouts, recorded traces or locally built spike cubins. Set `PYTHON` and `TINYGRAD_SRC` as required by those tests. `TINYNV_ALLOW_NO_TRACE=1` and `TINYNV_ALLOW_NO_CUBINS=1` allow explicit skips; a skipped check is not a validation pass.

Hardware validation includes:

- Value-checked `test-backend-ops`: 450/450 at chain depths 32, 64 and 128, plus synchronous mode.
- Dense and MoE decode benchmarks using driver defaults.
- Repeated temperature-zero generation, compared byte for byte with a reference; speculative output is checked against plain greedy output.
- Image-output comparisons and serving soaks for changes affecting those paths.
- Interleaved old/new build measurements for performance comparisons on the shared host.

[tools/wtgate.sh](tools/wtgate.sh) runs the standard gate from a detached `cuda-shim-f/` worktree beside the checkout. [tools/ab_pair.sh](tools/ab_pair.sh) alternates two builds in the same measurement window. Offline checks do not cover every hardware submission-ordering failure; runtime invariants and concurrent serving tests provide additional coverage.

## Disaggregated inference

Models larger than the GPU's memory can use **CUDA for prefill and Metal for generation**, provided the model fits the Mac's available memory. The CUDA server streams host-resident experts, saves the conversation state and transfers it to a Metal server using the same model file. [tools/disagg-router.py](tools/disagg-router.py) exposes the pair through one endpoint; [tools/disagg-serve.sh](tools/disagg-serve.sh) coordinates them under the GPU protocol.

In the 16 September test, Qwen3-Coder-Next 80B-A3B (49.6 GB) processed a 23,692-token prompt in **15.2 s on the RTX 5090**, compared with **35.7 s on Metal**. Saving 662 MB of state took 0.7 s and restoration took 0.1 s; Metal then generated at 54 tok/s.

The 19 September prefill-only measurements extended the comparison:

| Prompt length | RTX 5090 prefill | Metal prefill | Prefill speedup |
|---|---:|---:|---:|
| 23,691 tokens | 15.0 s | 35.2 s | 2.35× |
| 26,156 tokens | 22.0 s | 64.6 s | 2.94× |
| 48,667 tokens | 28.9 s | 160.3 s | 5.55× |

**Re-measured 20 September, same model and machine — the direction replicated, the magnitude did not:**

| Prompt length | RTX 5090 prefill | Metal prefill | Prefill speedup |
|---|---:|---:|---:|
| 23,999 tokens | 13.5 s | 37.5 s | 2.77× |
| 25,999 tokens | 20.4 s | 42.7 s | 2.09× |
| 47,999 tokens | 27.2 s | 104.9 s | **3.86×** |

The RTX 5090 times agree across both runs to within 10%, and Metal agrees at 24k. At 48k Metal went from 160.3 s to
104.9 s, and that single figure is the whole difference between 5.55× and 3.86×. The leading hypothesis is untested:
the first run read the 49.6 GB model from an external SSD that was 100% full and measured at 41 MB/s, the second from
one measured at 398 MB/s. **Quote 3.86× until that is resolved.** A same-night A/B found no disk effect on a
card-resident model, but that case keeps every weight in VRAM after load and does not test this one.

These timings exclude state transfer and generation. The configured microbatch size is 24,576 tokens, and expert data is transferred once per microbatch. A [six-model sweep](docs/bench/crossmodel-draft-20260919.html) since confirmed that the speedup tracks **active parameters** rather than model size, from 2.77× on this model to 10.37× on Mixtral 8x22B at 48k. The screening tool's constant Metal-throughput assumption remains provisional at longer context.

Placement flags and saving state one token early are required by this implementation; see [tools/disagg-prefill.sh](tools/disagg-prefill.sh). Split generation can differ from Metal-only greedy output on near-ties because the backends use different activation numerics. See the [inference state and measurement record](docs/disagg-inference-state.md) for the full configuration and corrected cost model.

## Scope and known limitations

- **Application coverage:** the compatibility libraries implement the interfaces used by the tested applications. GGUF support in upstream llama.cpp alone does not establish compatibility with this stack.
- **Remaining decode overhead:** profiling identifies host turnaround and roughly 1.7 µs of driver work per launch as optimisation targets. Qwen3.5 35B-A3B reaches 91% of reference decode throughput in the latest listed batch.
- **Experimental graph replay:** CUDA runtime graph support is implemented. Resident driver-level replay is a separate, opt-in path; it has not established a performance improvement in the recorded tests.
- **Unsupported operator cases:** `test-backend-ops` MUL_MAT with mxfp4/nvfp4 can hang the card. Treat these as dedicated hardware investigations, not routine validation workloads.
- **Pinned upstream fix:** older llama.cpp revisions can corrupt the arm64 rope work buffer during operator tests. `setup.sh llama` applies the required `2f53959` fix.
- **Memory reporting:** the driver reserves and validates the firmware region at startup. The former 64 MB reservation was corrected to a derived 256 MB reservation; reported free memory still reflects the total usable allocation rather than live remaining capacity.
- **Hardware recovery:** some Thunderbolt and firmware faults require physical reconnection. Halting firmware or sleeping with the card attached can send the enclosure fans to full speed until firmware restarts.

A Linux guest driver forwarding resource-management requests to `libtinynv` over vsock remains a separate research direction. It is not a completed route to general NVIDIA userspace support. See [remaining performance work](docs/driver/moe-next-steps.md) and [resident-chain design](docs/driver/chain-replay-plan.md).

## Repository layout

```text
install.sh          Prerequisite checks and guided installation
setup.sh            Dependency fetching and staged builds
env.sh              Local paths for development and operating tools
cuda-shim/
  libtinynv/        Userspace driver, offline tests and hardware probes
  libtinycudart/    CUDA runtime compatibility and copy kernels
  libtinycublas/    cuBLAS-compatible entry points and GEMM kernel
  build/            Compiler wrapper, build scripts and generated binaries
  spike/            Bring-up experiments and test kernels
  test/             Transfer and launch microbenchmarks
tools/              Preflight, locking, runners, serving, recovery and validation
patches/            stable-diffusion.cpp integration with upstream ggml
docs/               Architecture, design records, benchmarks and research notes
models/             Local model files; weights are not committed
logs/               Local execution records; not committed
```

## Third-party components and licensing

- **tinygrad** (MIT, © tiny corp): the userspace driver reference and signed TinyGPU app/extension. Retain its copyright notice with the driver.
- **NVIDIA open-gpu-kernel-modules headers** (MIT/GPL dual-licensed; used under MIT): fetched, not committed.
- **NVIDIA GSP firmware 570.144**: fetched from linux-firmware with hash verification, not committed; subject to its accompanying NVIDIA licence.
- **CUDA Toolkit headers**: obtained from the configured toolkit or container, not committed; subject to the CUDA Toolkit licence.
- **llama.cpp / ggml** and **stable-diffusion.cpp** (MIT): fetched at pinned commits, with the integration changes described above.
- **macuda code**: [MIT License](LICENSE). [NOTICE](NOTICE) contains third-party attributions.
