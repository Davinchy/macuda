# MoE decode on the shim: where the remaining third is, and the options (2026-09-19, evening)

Written for Antonio after the day's measurements. Every number is from this tree's logs (`docs/handoff-2026-09-19.md`,
`logs/gate-steps/step2*-step42*`), interleaved where it is a comparison; the model is Qwen3.5-35B-A3B Q4_K_M unless
said otherwise, the reference is the Windows-native 245 tok/s tg128 from `docs/bench/windows-reference-20260914.md`.

## 1. The budget, measured

| | this shim | native |
|---|---|---|
| tg128 | 166 (morning, host load 2.5-4); 140 (afternoon, load 4-7) | 245 |
| token period | 6.0 ms / 7.1 ms | 4.1 ms |
| launches a token | ~1,634 (16 chains of 128 at the default depth) | the same ops |
| engine busy (a kernel executing) | 66% = ~4.0-4.7 ms | - |
| engine dry | 34% = ~2.0-2.4 ms | - |
| memory busy | 6% | - |
| clocks / power / state | 2,862 MHz, 154 W, pstate 1, no throttling | - |

Per launch: 3.7 µs here (6.0 ms / 1,634) against 2.5 µs native (4.1 / 1,634). Of ours, ~2.4-2.9 µs is kernel-executing
and ~1.2-1.5 µs is dry: the ~1 µs inter-kernel dispatch gap every chained descriptor pays, plus the boundary.

Native's own CUDA graphs are worth 10-15% of decode to it (#27330's reporters, PR #6766), so native without graphs is
~215-220 tok/s = 4.6 ms. Our chain has every GPU-side property a graph has except one submission per token; the
depth sweep showed that one property is not on our critical path (deeper chains are slower). So ~0.5 ms of the gap is
"graphs", explained and not reachable by anything at the descriptor layer, and **~1.4 ms is elsewhere**.

## 2. What this week removed, and what it found absent

Removed: the copy engine delivering descriptors (delta delivery, the token-aligned rewind, the tail release: +2-3%
MoE, +4.4% dense, the token-boundary runlist switch for descriptors gone). Measured absent, each by an interleaved
A/B: the seam between chains (depth 1024 is 14% slower, not faster), the first-chain build (short first chain: no
change), the per-descriptor system-scope barrier (MoE: nothing; dense: 3%), gpu-scope barriers (nothing). Measured
load-bearing: the five per-launch cache invalidates (the greedy text differs without them).

## 3. What is still crossing the copy engine every token - the one thing left at this layer

At the current defaults a 96-token MoE decode logs **746 host-to-device copies that went to the copy engine** beside
685 that rode in the pushbuffer: ~6-7 a token above `TINYNV_INLINE_MAX` (4,096 bytes, `submit.h:45`), plus the
logits download every token. §4f measured the compute<->copy runlist handoff at ~0.35 ms a token on average (~0.7 on
half the tokens) while the copy engine delivered descriptors; that delivery left the copy engine today, but these
transfers did not, and each is a switch candidate. §4f also measured that on a bench with one channel active the
handoff never occurs. Two levers already exist and were each measured "no change" alone in §4f - when the descriptor
delivery still forced the switch regardless. Together, with that delivery gone, they can take the copy engine out of
the decode loop entirely.

## 4. The options, ranked by expected MoE gain per cost

**A. Take the copy engine out of the decode loop (do first).** (1) `TINYCUDART_TRACE=1` on a 16-token decode lists
every upload and download a token makes, by size. (2) Raise `TINYNV_INLINE_MAX` so the ≤32 KB uploads ride the
pushbuffer (the 20 KB one costs ~85 µs of register writes at §4d's rate against a channel switch). (3)
`TINYNV_DOWNLOAD_VIA_COMPUTE=1` for the logits: the shim lends `copy1d` at start-up (`cudart.c:72-77`), so the knob
is live. (4) Interleaved tg128 on both models plus the busy counters. **Expected:** up to the handoff, MoE +5-10%,
dense +2-4%. **Refusal reading:** no change means the copy engine is already off the critical path at these sizes.
**Cost:** one constant, two env knobs, one build, ~6 card steps at the standard risk tier (byte-compare gates it).

**B. A per-kernel profile from the card's own clock (measure before guessing further).** Under `TINYNV_PROFILE` the
release payload can carry the GPU clock; today only the last release survives (each overwrites the slot). A diagnostic
mode that releases each descriptor into its own ring slot gives every kernel's completion time, hence per-kernel
duration and the gap before it, by kernel name. That turns "inside busy time" into a list: which of the ~1,634 kernels
account for the 4 ms, and whether any run slower than they should. Paired with a native Nsight Systems trace of the
same decode on the Windows box (`nsys profile llama-bench ...`, which Antonio can capture), it names the kernels
where we lose 1.4 ms. **Cost:** ~50 lines in the driver, one card run, one Windows capture. **Gain:** knowledge;
this is the measurement that decides between C and "the gap is the gaps".

**C. Fewer launches a token via upstream llama.cpp.** The vendored tree is at b906d25. Upstream ggml-cuda keeps
fusing MoE decode ops (router + top-k, gated activation into the expert matmul, norm + scale). Every launch removed
saves ~3.7 µs here against ~2.5 µs native, so a fusion pays us more than it pays native. Rebase, rebuild the fatbins
on the nvcc box (`env.sh`), count launches a token before and after (the driver prints it), re-gate. **Cost:** a
rebuild round trip and the risk of new upstream bugs (graphs stay off, so #27330 is irrelevant). **Gain:** unknown
until the launch count is compared; check upstream's MoE PRs first, then decide.

**D. Speculative decoding for the MoE with a small draft (biggest user-visible win available today, no code).**
The 27B went 68 -> 124 tok/s with its MTP head because a card whose per-token cost is mostly overhead verifies four
tokens for little more than one; the MoE has the same shape (66% busy, tiny kernels). Qwen3.5-0.8B shares the
vocabulary; `sh tools/nv_shim_step.sh A spec models/Qwen3.5-35B-A3B-Q4_K_M.gguf <0.8B draft> 128 4` with
`SPEC_TYPE=draft` measures it in one step. **Expected:** 1.5-2x tokens/s on greedy and low-temperature work,
acceptance-dependent. **Cost:** 2-4 card steps. It does not close the driver's gap; it is the fastest MoE today.

**E. Invalidate subsets.** Keep the texture-data, shader-data and constant-bank invalidates, drop the header and
sampler ones (no textures in ggml). Correctness-gated by the byte-compare like today's A/B. **Expected:** ≤2%.
**Cost:** one knob value, three card steps.

**F. Multi-slot serving for MoE throughput.** The dry 34% is idle engine that another slot's kernels can fill;
the 27B went 124 -> ~207 tok/s aggregate at 8 slots. Zero code; a server soak. Relevant when the goal is served
throughput rather than one stream's latency.

**Closed, not options:** launch-chain replay (the study and its four reads), deeper or shallower chains, the short
first chain, gpu-scope barriers (nothing), dropping invalidates (wrong text).

## 5. Recommendation

A first: it is the last thing at this layer that is known to cost a switch, the levers exist, and it is one
afternoon. Then B, because after A every remaining microsecond is inside kernel time and the next choice (C, or
accepting the ~1 µs gap as the floor) needs the per-kernel list rather than another inference. D in parallel whenever
the card is free: it changes what a user of the MoE sees today. Absolute levels only compare within the same minutes
on this host (the same code paths read 16% apart today under different host load); every A/B stays interleaved.

## 8. B measured (2026-09-19 14:11-14:22, `f562d0b`..`35f17de`): the MoE is host-bound

`TINYNV_KERNEL_PROFILE=1` stamps every descriptor's completion with the engine's clock into a ring slot of its own
(the chain's tail from the command stream), reads the slots once the timeline has passed them, and attributes each
interval to the kernel that ended it. Both greedy texts stay byte-identical under it; zero stamps were missing over
107 windows. Per window on the MoE (~1.16 windows a token in `llama-simple`): 1,476 launches, 5.4 ms in kernels of
which **1.85 ms is the chain's first intervals** (10.4 chains: the engine finished the chain and waited for the
host's next one), 1.24 ms at the token boundary; the mean interval inside a chain is **2.45 us** including the
dispatch gap, so a window's in-chain time (~3.6 ms, ~4.1 ms a token) is about native's whole token. The host side
of the same run (`TINYNV_LAUNCH_PROFILE`, `TINYCUDART_STATS`): **4.1 us a launch** - 2.38 between consecutive launches
in ggml-cuda/llama.cpp, 1.75 inside `tinynv_launch` (1.29 of it the driver's build). The dense 27B is the opposite
case: 12.0 ms of its 15.4 ms token is kernels (mul_mat_vec_q 76%, 1.27 TB/s effective), seams 0.48 ms, boundary
1.3 ms.

**So the ranking changes.** The unattributed 1.4 ms of section 1 was never in the kernels: it is the engine idle
at chain seams because the host cannot build launches as fast as the engine retires them, plus the boundary. The
levers, in order of what they buy the MoE:

- **B'. A CUDA-graph subset in the runtime layer** (capture records kernel, parameters, shape; replay re-issues to
  the driver): removes ggml's 2.38 us a launch, host to ~1.75 us, under the engine's 2.45 - the seams close and the
  boundary's graph-build share goes with them. Estimated 6.6 -> ~4.5-4.8 ms a window (+35-45%). What ggml needs:
  begin/end capture, instantiate, launch, exec update, kernel-node get/set params; capture boundaries are ggml's
  own (`ggml_cuda_graph` re-captures on topology change, updates copy-kernel pointers per token). The engine-side
  replay the chain-replay study declined is not this: the driver still builds every descriptor.
- **The driver's own 1.29 us** (0.62 build + 0.65 flush share a launch): a second lever of up to ~0.5 us; the
  per-kernel spread (rope_multi 8.5 us, k_bin_bcast 3.6, quantize 0.9) says the parameter path is part of it.
- **C, fusion** (fewer launches) helps both sides proportionally and stays worth doing.
- **D, speculative decoding**: unchanged, the visible win with no code.
- The Windows Nsight capture (B's second half) now only refines the per-kernel comparison.

## 9. The host was compiled at -O0 (2026-09-19 14:35-14:53): MoE +53%, dense +4%

Reading the build for the graph work: `cuda-shim/build/tinycc` host-compiled every ggml-cuda TU with no `-O` flag,
so the dispatcher whose 2.38 us a launch section 8 measured ran unoptimised. Rebuilt at -O2 with the device halves
recovered byte-for-byte from the archive (no Linux box needed): the caller's time between launches 2.38 -> 0.42 us,
seams 1.85 -> 0.41 ms a window, MoE tg128 **140 -> 216** interleaved (native 245), dense 71 -> 74 (native 75.3),
every text and image byte-identical, op-verify 450/450 x3, a five-minute MTP soak clean. B' (the CUDA-graph subset)
shrinks to a small item: the host is now ~2.2 us a launch against the engine's 2.45. What remains on both models is
the token boundary (~1.2-1.4 ms of engine idle a token: sampling, the ggml graph build and scheduler, the first
chain) - compare against native's boundary in the Nsight trace - and the driver's own 1.74 us a launch.

## 10. The boundary and the 3090 reference (2026-09-19 15:00-16:30): the lever left is the driver's cost a launch

Native traces on the Pop!_OS 3090 box (Nsight, `logs/native-3090/`): graphs are worth +29% natively on the MoE
(134.6 -> 174.3 tok/s) because native without graphs is host-bound at ~5 us a launch - the state the shim was in this
morning; with graphs the gap between kernels inside a token is 0.14 us and the per-token boundary in the bench loop
~1.1 ms, about ours. Native's dense kernels run at 87% of the 3090's bandwidth; ours at 71% of the 5090's, which is
what native reaches on a 5090 too. On our side, the host's turnaround after the logits is 161 us, the short first
chain and the logits download on the compute queue measured nothing again, and a keep-alive kernel that holds the
engine scheduled across the boundary (`TINYNV_KEEPALIVE`, three attempts to get its semantics right) measured
nothing once it worked: the engine was never asleep. The boundary is the copy, the host's turnaround and the first
chain's build, and the seams are the host barely ahead of the engine (2.2 vs 2.45 us a launch). **What is left is
the driver's own 1.74 us a launch** (1.29 building the descriptor and its constant bank, 0.45 in the API layer's
lock, checks and info copies): at ~1 us the first chain builds in half the time, the seams close, a short first
chain starts paying, and the MoE gains an estimated ~10%, the dense a few percent. The CUDA-graph subset in the
runtime layer would remove ggml's remaining 0.42 us as well, for less than that.

## 11. The last 12% (2026-09-19 16:00-16:20): host levers measured out, the path is a resident-chain replay

The submit's cost was split (announcing = the fence read's round trip, ~25 us a batch) and a token's batches
counted (13 compute chains + 1 copy; every chain after the first carries 29 dwords: the descriptors are identical
token to token). Deeper chains after a short first one lost 4% at 512 and did nothing at 256; the fence read
pipelined (`TINYNV_RING_ASYNC`) cut the host's launch cost 6% and moved nothing. The budget: ~0.4 ms a token of
engine dispatch gaps (2.45 us a launch against native's 0.14 inside a graph), ~0.4 ms of chain seams, a boundary at
parity. **Next: B' as a resident-chain replay** - capture at ggml's CUDA-graph boundary (runtime layer), keep the
token's descriptors resident in the driver, link all chains into one, patch the timeline values and the first
chain's input uploads, one submission a token. Estimated +15% MoE (parity), +3-4% dense. Needs the ggml-cuda host
halves rebuilt with `-DGGML_CUDA_USE_GRAPHS` (host-only rebuild via the recovered fatbins).
