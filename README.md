# macuda — CUDA on an Apple Silicon Mac, over Thunderbolt, with no NVIDIA driver

Unmodified **llama.cpp (CUDA backend)** and **stable-diffusion.cpp** run on an **RTX 5090 (GB202, sm_120)** in a Thunderbolt
eGPU enclosure on **macOS**, correct and fast, with no NVIDIA driver on the Mac. Three libraries replace the vendor stack:

```
llama.cpp + ggml-cuda        host code compiled on the Mac by clang (--cuda-host-only); device .cu compiled to sm_120a
        │                    cubins by nvcc on a Linux box, embedded in the host objects (cuda-shim/build/tinycc)
  libtinycudart.a            the cudart ABI ggml-cuda links against: fatbin/module loading, kernel launch, memcpy, streams, events
  libtinycublas.a            the cuBLAS entry points ggml uses, on one tensor-core GEMM cubin (f16 wmma / TF32, strided-batched)
  libtinynv.a                a C userspace NVIDIA driver: GSP-RM boot (firmware 570.144), RM object model over RPC, MMU v3 page
        │                    tables, GPFIFO channels + doorbell, QMD launch with chaining, copy engine, timeline semaphores
  TinyGPU.app + DriverKit dext (tinygrad's)   PCI config/BAR/DMA access over a unix socket
  RTX 5090 over Thunderbolt
```

`libtinynv` is a C port of the userspace driver in [tinygrad](https://github.com/tinygrad/tinygrad) (`tinygrad/runtime/ops_nv.py`),
with NVIDIA's own structure definitions from `open-gpu-kernel-modules` pinned to the release that added the 5090.

## Measured (this Mac, RTX 5090 in an AORUS AI BOX; "native" = the same card in a Windows PC, driver 595.79, same llama.cpp build era, byte-identical models)

| workload | this shim | native | notes |
|---|---|---|---|
| Qwen3.8-27B Q4_K_M, tg128 | **67.8 tok/s** | 75.3 | dense decode, launch-bound for what remains |
| Qwen3.8-27B Q4_K_M, pp256 | ~2550 tok/s | 2447 | prefill at parity (the SASS is identical) |
| Qwen3.8-27B + its MTP draft head, greedy | **124 tok/s** first request, 112 mean over a 62-minute soak | 124.7 | `llama-server`, `--spec-type draft-mtp` |
| `llama-server`, 8 slots, 27B + MTP, temp 0.6 | **~207 tok/s aggregate** | — | 20-minute soaks at eight slots: 440 requests, 0 errors |
| Qwen3.5-35B-A3B MoE Q4_K_M, tg128 | **162.6 tok/s** | 245.0 | ~2000 launches a token; the purest launch-bound case |
| SDXL-Turbo, 4 steps, 512², cfg 1.0 | **2.72 s** | 4.11 s | stable-diffusion.cpp |
| SD 1.5 (fp32), 20 steps, 512² | **3.72 s** | 3.86 s | |
| Z-Image-Turbo, 8 steps, 1024² | **9.67 s** | 9.46 s | Q8 DiT + Qwen3-4B encoder + FLUX VAE |

Every number above was taken with the standard gate: a clean driver build whose build id is verified in each binary, `test-backend-ops`
value-checked 450/450 at three chain depths, both decodes at the driver defaults, and a 96-token greedy text byte-compared against a
reference file. Per-launch host cost is ~1.3 µs at chain depth 128 (the vendor's is ~1). Decode numbers move a few percent with host
load; MoE numbers move more (see `docs/03-cuda-shim-plan.md` and the log excerpts in `docs/bench/`).

**Re-verified 2026-09-18** against `main`'s tip (`a988ec7`) via `tools/wtgate.sh` (dense/MoE tg128) and three repeats of the standard
`sd` step (Z-Image, 9.66/9.68/9.68 s): the three previously-published numbers above (65.3, 134.3, 10.16 s) had all gone stale relative
to what the gated build actually does — MoE tg128 in particular is genuinely ~163 tok/s now (confirmed by a second, independent build),
not 134.3, closing a meaningful chunk of what looked like a driver deficiency; Z-Image's number was also traced to having been captured
on a pre-gate build, and a separate later reading of 16-17 s came from an uncommitted, dirty tree — neither is what `main` does. Always
check `strings <binary> | grep -c '^<expected-id>$'` before trusting any number pulled from an old log.

## What you need

**Hardware.** An Apple Silicon Mac (built and measured on an M4 Max, macOS 27.0) and an RTX 5090 in a Thunderbolt enclosure. The
card is fragile over Thunderbolt: a wedged GSP or a latched DART fault needs a physical replug, and nothing here can do that for you.
Read `tools/preflight.sh` and the protocol in §Run before touching it.

**On the Mac.**
- Xcode / Command Line Tools (the build picks a macOS SDK by *linking a one-line program* — `cuda-shim/build/sdk.sh` — because SDK
  paths moved under the tree more than once), CMake, `python3`.
- Homebrew LLVM at `/opt/homebrew/opt/llvm` (`brew install llvm`): its clang does the CUDA host-only compile and its `libLLVMDemangle`
  is linked into the runtime.
- **TinyGPU.app and its DriverKit dext** (`org.tinygrad.tinygpu.driver2`): tinygrad's signed release,
  `https://github.com/tinygrad/tinygpu_releases/raw/c0d024f9ff0e1dc8fdf217f255da7101d91e8323/TinyGPU.zip`, unzipped to `/Applications`,
  then `/Applications/TinyGPU.app/Contents/MacOS/TinyGPU install` (approve the system extension). `tools/tinygpu-server.sh` starts and
  checks the server that fronts the dext over `$TMPDIR/tinygpu.sock`.
- Optional, for the driver's offline reference tests and the card-recovery tools: tinygrad checkouts (two branches of the fork the
  driver was developed against — see `env.sh` for which tool needs which) in `tinygrad/` and `tinygrad-stable/`, with a Python 3.12
  venv in `venv/` that has tinygrad installed editable.

**A Linux box with CUDA 13 (`nvcc`) reachable over ssh**, for the device side of every CUDA compile (ggml-cuda's ~190 translation
units, and the shim's own three kernels). Nothing NVIDIA runs on the Mac; the cubins are your own build output. Set `TINYCC_HOST`
(and `TINYCC_KEY` if needed) before any step that compiles device code. A prebuilt `libggml-cuda.a` for the pinned llama.cpp commit is
enough to link and run without that box (see §Build).

**Fetched by `setup.sh deps`, not in the repository:** NVIDIA's `open-gpu-kernel-modules` headers at commit `81fe4fb` (release
570.86.16, the one that added the RTX 5090; the build asserts the driver's structure sizes against them), the three signed GSP
firmware images for 570.144 from linux-firmware (hash-pinned), and the CUDA Toolkit headers (copied from the nvcc box's
`/usr/local/cuda/include`). The header/firmware skew (570.86.16 / 570.144) is deliberate and proven by booting the card with it.

## Build

```sh
export TINYCC_HOST=user@linux-box            # the nvcc box (TINYCC_KEY=~/.ssh/... if it needs an identity file)
sh setup.sh deps                            # NVIDIA headers + firmware (hash-checked) + CUDA headers
sh setup.sh llama                           # llama.cpp at ad6c668 + upstream fix 2f53959, CPU-only static build in llama.cpp/build-null
sh setup.sh sd                              # stable-diffusion.cpp at 59c23bc on llama.cpp's ggml (+ patches/), build-null
sh setup.sh shim                            # libtinynv.a + libtinycudart.a + libtinycublas.a (no GPU; ~3 s)
sh setup.sh link                            # the *-null binaries in cuda-shim/build/bin
```

`setup.sh` is the whole sequence; each step is idempotent. `LLAMA_SRC=` / `SD_SRC=` copy from a local clone instead of GitHub.

**The ggml-cuda archive.** `cuda-shim/build/libggml-cuda.a` (plus `ggml-backend-reg.cuda.o`) is every ggml-cuda translation unit
compiled for `sm_120a` with the shim's configuration (`-DGGML_CUDA_FORCE_MMQ -DGGML_CUDA_NO_VMM`, CUDA graphs compiled out) through
`cuda-shim/build/tinycc`: device compile on the Linux box, host compile on the Mac, one Mach-O object each. It is a build output
(gitignored) and it belongs to the pinned llama.cpp commit. To rebuild it, sync `llama.cpp/ggml` to `~/ggml` on the box
(`rsync -a llama.cpp/ggml/ $TINYCC_HOST:ggml/`) and run `JOBS=8 sh cuda-shim/build/build-ggml-cuda.sh` (about an hour). The shim's
own kernels (`libtinycudart/copy1d.cu`, `copy2d.cu`, `libtinycublas/gemm.cu`) ship as committed cubins with the `nvcc -arch=sm_120`
line that built them in each source file.

**How the binaries are made.** llama.cpp is built once as a plain CPU-only *static* tree (`build-null`; no Metal, no CUDA). `build/link-null.sh`
replays the exact link line CMake generated for a target and inserts, ahead of `libggml.a`: the backend registry object compiled with
`GGML_USE_CUDA`, `libggml-cuda.a`, the three shim libraries and LLVM's demangler. The result runs on the driver's **null device**
(the whole host path, no GPU — `TINYCUDART_TRACE=1` / `TINYCUDART_STATS=1` count launches and time calls there) until
`TINYNV_SOCKET` points at a TinyGPU server, when it opens the card. The `-null` suffix is that heritage.

**Build ids.** `libtinynv` bakes `git rev-parse --short HEAD` (plus `-dirty`) into every build, the runtime prints it at init
(`[tinycudart] libtinynv build <id>`), and `tools/nv_shim_step.sh` checks a binary's id against the branch history before touching the
card. Outside a git checkout the id reads `nogit`.

**Offline tests.** `make -C cuda-shim test` runs the driver's suite (page tables, QMD descriptors, cubin/relocation loading, RM
refusals, the replayed boot against a recorded trace — `traces/5090-boot-smoke.{trace,blob}` and `traces/5090-launch2-launch.json`,
recorded from this card, gitignored, `TINYNV_ALLOW_NO_TRACE=1` to skip — and a link smoke test through the real CUDA headers),
then checks that every `cuda*`/`cublas*` symbol `libggml-cuda.a` needs is defined by the shim. Several tests compare against tinygrad's
own driver: run it as `PYTHON=<venv python with tinygrad> TINYGRAD_SRC=<the egpu-hcq2-remote checkout> make -C cuda-shim test`
(green from `make clean` on 2026-09-15 with `traces/` and the spike cubins in place). The small test
cubins in `cuda-shim/spike/` other than `vecadd.*` are gitignored build outputs (`libtinynv/tools/build_spike_cubin.py`, through
tinygrad's CUDA compile path); without them `test_reloc` stops the suite unless `TINYNV_ALLOW_NO_CUBINS=1`.

## Run

The card is operated through one protocol, and the scripts enforce it: **read-only preflight → lock → one step as its own process →
release, leaving the card idle warm** (firmware resident, nothing submitted). Two processes on the card at once wedge it.

**Card-free is not host-free.** On 2026-09-16 the Thunderbolt tunnel dropped and re-enumerated twice with the card idle, both times
under a full Metal prefill (the Mac's own GPU flat out with a 50 GB model mapped) while the laptop was powered by the enclosure's USB-C
power delivery over the same cable; on Apple's own charger the same prefills did not drop it. So: power the laptop from its own adapter,
never from the enclosure, and treat any job that maps or allocates more than ~20 GB of host or Apple-GPU memory as a card-affecting
action while anyone holds the card - it needs the scheduler's word like a slot. Check a staged script with `sh -n` only; never source
or run it to see what it prints (that is how one of those prefills was started without a word).

```sh
sh tools/preflight.sh                                   # VERDICT: OK to proceed — reads PCI/dext/server/lock state, touches nothing
sh tools/nv_shim_step.sh A opverify                     # value-checked test-backend-ops on the card: expect 450/450
sh tools/nv_shim_step.sh A bench models/Qwen3.8-27B-UD-Q4_K_M.gguf 128          # llama-bench -ngl 99 -p 256 -n 128 -r 3
sh tools/nv_shim_step.sh A simple models/Qwen3.8-27B-UD-Q4_K_M.gguf 96 'prompt'  # greedy decode, output kept for byte-compares
sh tools/nv_shim_step.sh A spec models/Qwen3.8-27B-UD-Q4_K_M.gguf models/MTP/mtp-Qwen3.8-27B-Q4_0.gguf 128 4   # speculative decode
sh tools/nv_shim_step.sh A sd models/sd/<model> 4 'prompt' --cfg-scale 1.0       # stable-diffusion.cpp, image in images/
sh tools/serve.sh start|status|test|stop                # llama-server + MTP head, OpenAI-compatible at http://127.0.0.1:8090/v1
sh tools/soak.sh 2 256                                  # hours, max tokens: a soak against the server with a greedy probe every tenth request
```

The runner (`nv_shim_step.sh`) does the preflight, takes the lock with its pid, makes sure the TinyGPU server is up (never replacing a
live one), runs the step, echoes back the driver's startup line (build id, submission mode, arena, chain depth — *trust that line, not
what you exported*), marks a step that died by signal or never had the card to itself, and releases. `BIN=<dir>` picks a binary set,
`DRY=1` runs the step on the null device, `QUIESCE=1` halts the firmware after (cold; the enclosure's fans go to full speed when GSP is
down — that is not heat). `tinynv-smi` (`cuda-shim/build/shim/nv/tinynv-smi`) reads temperature, power and clocks the driver
publishes four times a second, without opening the card.

**Recovery.** `tools/nv_quiesce.sh` (FLR via the dext, halts GSP), `tools/nv_e3_flr.py`, `tools/nv_temp.py` need the tinygrad checkout
(`env.sh`). A card that drops off the bus, a latched `pci-dart-error-data` flag or "link NOT up" means a physical replug. Never
`systemextensionsctl reset`.

**Driver knobs** (each run's startup line says which took effect). Defaults are the fast path: asynchronous submission, the command
arena in VRAM, launch descriptors delivered by the copy engine, chain depth 128, a 64 MB descriptor region, sensors on, small uploads
ridden inline with the next compute batch. `TINYNV_SYNC=1` (synchronous, every batch waited on), `TINYNV_ARENA_VRAM=0` (host arena),
`TINYNV_ARENA_DMA=0` (descriptors written across the link), `TINYNV_CHAIN_DEPTH=N`, `TINYNV_DESC_MB=N`, `TINYNV_SENSORS=0`,
`TINYNV_INLINE_UPLOAD=0`, `TINYNV_FW_DIR=<firmware dir>`, `TINYNV_SOCKET=<tinygpu.sock>` (unset = null device). The runtime's:
`TINYCUDART_TRACE=1` (timestamped call trace), `TINYCUDART_STATS=1` (per-kernel launch counts and times), `TINYCUBLAS_TC=0`
(scalar GEMM instead of tensor cores).

**Validation recipe** (what "validated" means here — value-checked, not just non-crash): op-verify 450/450 at chain depths 32/64/128
plus once under `TINYNV_SYNC=1`; the two decode numbers at the defaults; a temp-0 greedy decode run three times and byte-compared;
for anything speculative, the depth-4 greedy output byte-identical to plain greedy. `tools/wtgate.sh` is that gate as one script
(it expects a detached worktree `cuda-shim-f/` beside the checkout, so a build never runs against a tree someone is editing);
`tools/ab_pair.sh OLD NEW` interleaves two builds within the same minutes, which is the only honest way to compare decode numbers on a
loaded host.

## Disaggregated inference: prefill on the card, decode on the Mac

A model that fits the Mac's unified memory but not the card's 32 GB can still use the card for what it is best at. The card
server keeps every expert tensor mmapped on the host (`--n-cpu-moe 48 --no-host --no-repack`) and ggml's scheduler streams
each layer's experts across the link once per batch, so the card prefills a long prompt at its own speed; the slot's state
(KV cache plus the linear-attention recurrent state, 24 KB/token for Qwen3-Coder-Next) is saved through `--slot-save-path`,
restored into a Metal `llama-server` holding the same file, and Metal decodes. `tools/disagg-router.py` makes that one
endpoint: it reproduces Metal's tokenization, sends cold prompts over a threshold to the card, hands the state over, and
streams from Metal; `tools/disagg-serve.sh` runs the trio as one tenant under the card protocol.

Measured 2026-09-16, Qwen3-Coder-Next 80B-A3B (49.6 GB), a 23,692-token prompt, this M4 Max: card prefill **15.2 s (1558
tok/s)** against 35.7 s (664 tok/s) on Metal alone; the 662 MB state saved in 0.7 s and restored in 0.1 s; Metal then
evaluated one token and decoded at 54 tok/s. Three things are load-bearing and documented in `tools/disagg-prefill.sh`:
the placement flags (without them llama.cpp puts CPU-placed weights in a buffer type that is either not mmappable or not
offloadable), the 24576 ubatch (ggml-cuda's MoE id helper caps a ubatch at 25,088 tokens on sm_120, so longer prompts stream
the experts once per ubatch), and saving the state one token early (the recurrent state cannot be rewound). Greedy text
from the split diverges from Metal-only text on near-ties, since the card's expert matmuls quantize activations and Metal's
do not; it is numerics, not a wrong cache. The Metal half maps the whole model and is a host-rule job (see §Run).

## Layout

```
setup.sh                      fetch + build sequence (deps | llama | sd | shim | link)
env.sh                        paths for the operating tools (tinygrad checkout, venv, nvcc host)
cuda-shim/                    the driver and the shim — snapshot of the working tree's main at 09e9cdb (2026-09-15)
  libtinynv/                  the C driver: src/ (gsp.c boot + RM, mmu.c/pt.c/tlsf.c memory, submit.c ring + doorbell, qmd.c launch +
                              chaining, exec.c the launch path and arena, cubin.c/image.c loading, fw.c/flcn.c firmware, pci_tinygpu.c
                              the socket backend), test/ (the offline suite, hardware probes test_hw_*), tools/ (fetch scripts, lints,
                              reference generators), include/tinynv.h (the A–B interface)
  libtinycudart/              the cudart ABI (cudart.c, cudart_api.c, fatbin.c, hostpool.c, demangle.cpp) + copy1d/copy2d kernels
  libtinycublas/              cublas.c + gemm.cu / gemm.cubin
  build/                      tinycc, build-ggml-cuda.sh, link-null.sh, link-llama-bench.sh, sdk.sh, fetch-cuda-headers.sh
  spike/                      the bring-up spikes and the small cubins the driver tests load
  test/                       microbenchmarks (socket, bandwidth, H2D/D2H, launch+copy)
tools/                        preflight, lock, server, runner, serving, soak, load generator, recovery, gates, Windows reference drivers
patches/                      stable-diffusion.cpp on upstream ggml (its fork-only int8/fp8 paths refuse instead of compiling)
docs/                         03-cuda-shim-plan.md (architecture and decisions), driver/ (the driver's design documents),
                              research/launch-overhead-prior-art.md, bench/ (the Windows reference runs)
models/  logs/                not committed; see models/README.md
```

## Known limits and hazards

- **MoE decode is at ~66% of native** and dense decode at ~90%: what remains is per-launch cost (~1.3 µs vs ~1) and a copy-engine ↔
  compute-engine runlist switch at each token boundary (~0.35 ms a token on this enclosure), measured and documented in the design
  docs. Fewer launches and bytes per token is the next lever, not the link.
- **Latent, untested by design:** the driver reserves a flat 64 MB at the top of VRAM for the firmware, but the firmware's
  write-protected region (WPR2) as read from the chip spans ~203 MB, starting ~161 MB below where the driver's allocator stops. Nothing
  has broken because the allocator has never been half full. Do not raise `gspFwHeapSize` until the reservation is derived from the
  firmware's own numbers (the fix in progress).
- CUDA graphs are compiled out (`GGML_CUDA_USE_GRAPHS` absent); `cudaGraph*` is stubbed. Upstream llama.cpp graphs hang on sm_120 anyway.
- `test-backend-ops` MUL_MAT with mxfp4/nvfp4 hangs the card; probe it last in a session, if at all.
- A llama.cpp checkout older than upstream `2f53959` corrupts its own heap in `test-backend-ops` on arm64 (ggml-cpu's rope work buffer)
  and will look like a driver crash; `setup.sh llama` applies that fix.
- The card runs warm and idle between steps; halting the firmware or sleeping the Mac with the card attached sends the enclosure's fans
  to full speed until the next boot of the firmware.

## Third-party components and licensing

- **tinygrad** (MIT, © tiny corp): `libtinynv` is a port of its userspace NVIDIA driver, and the TinyGPU app/dext is its signed
  release. Keep its copyright notice with the driver.
- **NVIDIA open-gpu-kernel-modules** headers (MIT/GPL dual-licensed; used under MIT): fetched, not committed.
- **NVIDIA GSP firmware** 570.144 (NVIDIA's linux-firmware licence, redistributable unmodified): fetched from linux-firmware, hash-checked,
  not committed.
- **CUDA Toolkit headers** (NVIDIA CUDA EULA): not committed; copied from your own toolkit install by `fetch-cuda-headers.sh`.
- **llama.cpp / ggml** (MIT) and **stable-diffusion.cpp** (MIT): cloned at pinned commits by `setup.sh`; `patches/` carries the only change.
- This repository's own code is under the MIT License (`LICENSE`); `NOTICE` carries the attributions above.

`docs/bench/windows-reference-20260914.md` is the raw record of the native reference runs and contains the Windows box's paths.
