# Disaggregated inference — state and handoff (2026-09-17)

**WHERE A FRESH START BEGINS is the entry point.**

Prefill on the RTX 5090 through the shim, decode on the Mac's GPU, for a model that fits unified memory but not VRAM.
Everything below is in this tree (`tools/disagg-*.{sh,py}`, README §Disaggregated inference). The numbers are from the
real card on this M4 Max (128 GB) on Apple's 140 W charger, Qwen3-Coder-Next-UD-Q4_K_XL (49.6 GB, 43.7 GiB of it expert
tensors). **Every card number below is tagged with the driver build that produced it, because the driver moved twice and
the 09-16 numbers are not comparable with the 09-17 ones.**

## WHERE A FRESH START BEGINS — 2026-09-17. **READ THIS SECTION FIRST.**

There is no other Claude session on this project. Antonio works directly with this assistant; a card step needs
Antonio's own explicit go, proposed with its expected result, refusal readings and risk class, same as any other
hardware step (see README §Disaggregated inference).

**As of 2026-09-17 evening, Antonio confirmed the card is ready.** Window 3 (prepared below, never run) can go ahead
directly: preflight first to anchor the before-readings, then the sequence, with the registered predictions checked
against the observed numbers afterward.

**Confirmed 2026-09-17 evening: this harness's auto-mode classifier refuses the actual window-3 launch too.** The
card-free 0.8B plumbing test was refused earlier that day ("Interfere With Workloads"); starting the window-3 trio
(`tools/disagg-serve.sh start`, card idle and lock free, preflight clean) was refused separately, this time with no
stated reason at all ("judged dangerous," no explanation given). The card side is not the obstacle — preflight was
VERDICT OK, lock free, dext unchanged at pid 659 — this is the permission mode itself blocking any host process
launch. Do not try to route around it (backgrounding, alternate launch methods): report the exact refusal and let
Antonio decide between running the sequence himself in a terminal or adjusting the permission mode for this session.

**DO NOT HALF-START THE WINDOW.** A window that dies with a server refused mid-sequence leaves the lock held and the card
in a state nobody pre-registered. Preflight is read-only and safe to run first; the servers are not.

**THE SEQUENCE, seven foreground commands, no script** (staged scripts are checked with `sh -n` and never run):

    cd /Volumes/512SSD/EGPU_MAC_Nvidia && . ./env.sh
    sh tools/preflight.sh | tail -5
    BIN=/Volumes/512SSD/EGPU/cuda-shim-f/build/bin/llama-server-null NCPUMOE=40 THRESH=512 \
      sh tools/disagg-serve.sh start /Volumes/512SSD/EGPU/models/Qwen3.5-35B-A3B-Q4_K_M.gguf
    # tokenize A and B through the router first (Metal-side, no card work): if they differ by more than ~50 tokens, trim
    sh tools/disagg-serve.sh test logs/disagg/w3-prompt-a.txt 32     # t1, FIRST after load
    sh tools/disagg-serve.sh chat logs/disagg/chat-6k.txt            # t2
    sh tools/disagg-serve.sh test logs/disagg/w3-prompt-b.txt 32     # t3, same length as A
    sh tools/disagg-serve.sh stop && sh tools/nv_shim_step.sh V1 opverify

**WHAT IS PREPARED, AND IT IS NOT IN GIT.** `logs/` is gitignored, so these exist only on this Mac's disk:
`logs/disagg/w3-prompt-a.txt` (75,151 B) and `logs/disagg/w3-prompt-b.txt` (75,162 B) — disjoint slices of the same
driver-source corpus with a 65,000-byte gap, equal to ~3 tokens at the measured 3.534 bytes/token, ~21,300 tokens each:
one ubatch with ~14% margin under the 24,576 ceiling in case `qwen35moe` tokenizes less efficiently than the coder model.
The middle request is the existing `logs/disagg/chat-6k.txt`. **If this tree is ever cloned elsewhere, the prompts must be
rebuilt** — the recipe is two 75,000-byte line-aligned slices of `logs/disagg/prompt-64k.txt` at offsets 0 and 140,000.

**A and B's own length gap** (card-free check, 2026-09-17, arithmetic only — not the real tokenizer): the files are
75,151 B and 75,162 B, an 11-byte difference, ≈3 tokens at the measured 3.534 bytes/token. That is well inside the
~50-token tolerance the sequence calls for, so trimming looks unnecessary — but this is a byte-count proxy, not
`qwen35moe`'s actual tokenization, which the sequence's own tokenize-through-the-router step (Metal-side, no card
work) still needs to confirm at go time, since that step itself starts a host process this session's permission mode
also refuses.

**THE REGISTERED PREDICTIONS STAND AS WRITTEN — do not re-derive them, and do not widen them after seeing a number:**
warm per-ubatch stream **3.0-3.9 s**, **t1 - t3 = 0.6-0.9 s** if the residual is a cold first touch and **≈0** if it is
not, **no band on the marginal rate**. Nobody is touching them before the run.

**BANKING MEANS COPYING THE ARTIFACT, NOT WRITING THE PROSE** (earned on run 87 an hour before: two
invocations shared one output directory and the second replaced the first's daemon log by design, so the prose survived
while the file behind the citation became a different run's output under the same name). In this tooling the three serve
logs are timestamped and safe, but **`logs/disagg/serve/router.port` and `router.log.path` are rewritten by every start,
and the slot state files are overwritten by the next prefill with the same slot id**. So the moment the window ends:
copy the three logs to `logs/disagg/serve/w3-<ts>-{card,metal,router}.log`, sha256 each copy, and **cite the copies**.

**BEFORE-READINGS TO ANCHOR AGAINST** (measured at 08:31, confirmed again at 08:5x; re-read them yourself at go
time rather than quoting these): preflight VERDICT OK, **DART absent with NO snapshot** (cleared 08:35 after the reboot
re-enumerated the nub; both pre-reboot records preserved in ROOT `logs/`), **dext 1 instance, pid 659, started
07:41:33**, lock free. Report LANDED or REFUSED either way, with the three prefill times, the `cache_n` values and the
op-verify result. **A model that has never been loaded failing to load IS a result for this window, not a wasted slot.**

## Where it stands in one line

The feature works as one endpoint and is fast: a 24K-token prefill that Metal does in 35.7 s takes **9.6 s** split, with
decode unmoved on Metal at 52-58 tok/s. What is NOT settled is the shape of the cost model near its fixed term — see
THE OPEN QUESTION.

## The runs, by driver

**`2fe7091` (2026-09-16 16:21-16:24, LANDED, op-verify 450/450 after, dext 15→15, DART absent).** The trio
(`tools/disagg-serve.sh start`, threshold 512) served three requests through the router:

| request | route | card prefill | Metal saw | decode | wall |
|---|---|---|---|---|---|
| 23,692-token prompt, cold (FIRST after load) | card → save 662 MB 0.7 s → restore 0.1 s | 15.3 s (1545 tok/s) | prompt_n 1, cache_n 23691 | 52 tok/s | 17.4 s |
| same context + one question | Metal only (24 cold tokens) | none | cache hit | | ~1 s |
| chat, 6,975-token message (THIRD) | card via /apply-template → 251 MB | 8.5 s (821 tok/s) | prompt_n 1, cache_n 6974 | 58 tok/s | 10.3 s |

Metal alone prefills the 24K prompt in 35.7 s (664 tok/s). Greedy text from the split diverges from Metal-only text on
near-ties (0.2-nat margins, full top-5 ranking reproduced): numerics, not a wrong cache.

**`2fe7091`, window 1 (2026-09-17 07:31).** 64,577 tokens = **3 ubatches** in **40.5 s**, inside the pre-registered
40-50 s band. This is what `dac5549` rests on: **the expert stream is paid PER UBATCH, not once per prompt** — ggml's
scheduler re-copies the host-resident experts for every ubatch it evaluates. Architecture-level, so it carries across
drivers; the seconds do not.

**`ad308bd`, window 2 (2026-09-17 07:34-07:36) — the residency control and treatment, requests paired by position,
lock released between the cycles.** 18 of 48 expert layers resident on the card (`NCPUMOE=30`) against all 48 streamed:

| condition | 7K prefill (first after load) | 24K prefill (second) | decode | op-verify |
|---|---|---|---|---|
| CONTROL `NCPUMOE=48` | 6,974 tok in **11.43 s** (610 tok/s) | 23,691 tok in **12.59 s** (1882 tok/s) | 54-58 tok/s | 450/450 |
| TREATMENT `NCPUMOE=30` | **7.59 s** (918 tok/s, **-33%**) | **9.57 s** (2476 tok/s, **-24%**) | 54-58 tok/s | 450/450 |

`cache_n` 6,974 and 23,691 in both cycles; no allocation trouble at ~29 GB of 31.8.

**THE CONTROL WAS NECESSARY, and the reason is worth keeping:** the driver change ALONE took the 24K prefill from 15.3 s
(`2fe7091`) to 12.59 s, so residency measured against yesterday would have been credited ~18% that belongs to `ad308bd`.

## THE OPEN QUESTION — a residual the published model does not explain

The model reported at the 07:37 stand-down (fixed **8.0 s** per ubatch, marginal **5150 tok/s**, residency scaling the
fixed term by the byte fraction 30/48) was fitted on the **two 24K points** — two unknowns, two equations. **Its exactness
at those points is arithmetic, not a test**, and the stand-down line's "both conditions fit one model exactly" should have
said so. The two 7K prefills were **not** in that fit:

| point | model predicts | observed | residual |
|---|---|---|---|
| 7K control (48 layers streamed) | 9.35 s | 11.43 s | **+2.08 s** |
| 7K treatment (30 streamed) | 6.35 s | 7.59 s | **+1.24 s** |

The residuals are in the ratio **0.60**, against the residency byte fraction 30/48 = **0.625**: the unexplained time
scales with **expert bytes streamed**. A first-touch / cold-mapping cost does that. A compute-efficiency effect at small
batch cannot, because residency does not change the arithmetic, only where the weights live.

**It is a hypothesis, not a finding, and the confound is named:** n=1 per condition, and in BOTH cycles the 7K prefill ran
FIRST and was also SHORTER, so order and length are inseparable in this data. (The 09-16 run had the opposite order — 24K
first, 7K third — but on the other driver, so it cannot arbitrate.) **Window 3 breaks the confound by design.**

## Window 3 — prepared 2026-09-17, ready to run now that Antonio has confirmed the card is free

- **Model:** `/Volumes/512SSD/EGPU/models/Qwen3.5-35B-A3B-Q4_K_M.gguf` — 19.7 GiB, arch `qwen35moe`, **40 blocks, 256
  experts with 8 used**, 18.2 GiB of expert tensors (92%). Against the coder model's 48 layers / 43.7 GiB / few large
  experts, this is a second *architecture*, not a second size. Host peak ~40 GiB (both halves map it), ~40% of the coder runs.
- **Binary:** `BIN=/Volumes/512SSD/EGPU/cuda-shim-f/build/bin/llama-server-null` (`ad308bd`, the window-2 driver). The
  build line must be quoted **from the binary's own first output**, not from notes.
- **Placement:** `NCPUMOE=40` — every expert layer streamed, the conservative arm only. No residency arm, so the allocator
  stays far from the ~30 GB point and the flat-64 MB-vs-~203 MB WPR2 carveout hazard is not touched.
- **The order IS the experiment:** 24K prompt A (first after load) → 7K chat → 24K prompt B, same length, different
  content. `t1 - t3` is read at equal token counts, so the marginal rate cancels and the first-prefill penalty is one number.
- **Registered before the go:** warm per-ubatch stream **3.0-3.9 s** (window 2's own implied link rate, 43.7 GiB / 8.0 s =
  5.46 GiB/s, applied to 18.2 GiB — window 2 predicting window 3, not a fresh fit); **t1 - t3 = 0.6-0.9 s** if the
  cold-first reading is right, **≈0** if it is not; **no band on the marginal rate** — different architecture, no basis.
- **A load failure is a RESULT for this window, not a wasted slot:** the placement flags were found by crashing
  on one model. Report LANDED or REFUSED either way, with the three prefill times and the `cache_n` values.

## What changed in the router today (card-free, 2026-09-17 08:4x)

1. **Cold-start constants are now the `ad308bd` pair:** `--card-fixed` 5.7 → **8.0 s per ubatch**, `--card-rate` 2460 →
   **5150 tok/s**. The old pair was the `2fe7091` fit and was two drivers stale. Break-even moves **~5,300 → 6,197** cold
   tokens (`--selftest`). Against the control prefill they came from: 12.60 s predicted, 12.59 s observed.
2. **A refit now needs THREE observations, not two, and must reproduce every point it was fitted on within 20%.** This is
   a measured defect, not a tidy-up: in window 2 the live router refit itself from the control pair to **11.0 s +
   14,471 tok/s** — nearly three times the marginal rate the same window supports — and then predicted **16.2 s** for the
   prefill that had just taken 12.59 s. Two single-ubatch points fit themselves exactly by construction. Verified in
   process: that real pair now leaves the priors standing, three consistent points still refit, three points on a
   genuinely different line (3.30 s + 9000 tok/s, window 3's predicted shape) are tracked exactly, and one absurd point
   among three is rejected.
3. **The first card prefill after a start is named in the log** as possibly paying a cost the later ones do not, citing
   the two residuals above. It is NOT excluded from the fit — the question is open, and window 3 decides it.

**Not re-run today: the 10/10 plumbing test** (`tools/disagg-router-test.sh`, 0.8B on the null device). It starts three
host processes and this session's permission mode refused to launch them. The changed code paths were exercised in process
instead (above) and `--selftest` is OK, but the end-to-end plumbing has not been re-checked since the change.

4. **`fit_line()`'s refit-robustness is now a persisted test, not a one-off:** `tools/disagg-router-fit-test.py` (pure
   Python, no host process, no card — imports the router module and calls `fit_line` directly) pins the five properties
   this section claimed as "verified in process": two points don't fit, three consistent points do, three points on a
   different line are tracked on their own terms, one absurd point among three is rejected, and a multi-ubatch prompt
   still recovers the per-ubatch fixed cost rather than a flat one. All five PASS as of `35c3aed`+1. Card-free session on
   2026-09-17 (still using the card at the time) also re-hit the same plumbing-test refusal above under a fresh session
   name (`egpu-mac-nvidia-fc`) — confirms it is a permission-mode property, not a one-time fluke of the earlier session.

## The rules the runs obeyed, and why

- **Card protocol**: the tenant takes the shared lock (`. ./env.sh` first), through preflight and the TinyGPU
  server, and holds it while the card server is up; `stop` releases it, card idle warm. **Antonio approves every
  slot** — there is no other Claude session to schedule around.
- **512-token cap on Metal prefills in a card window**: the eGPU tunnel dropped twice on 09-16 under full
  Metal prefills while the laptop drew power through the enclosure's USB-C PD; on Apple's charger it has held, but n is
  small. With the router's threshold at 512, Metal only ever sees short prompts and cache-hit decodes.
- **Card-free is not host-free**: the Metal half maps the whole model; any job over ~20 GB of host or Apple-GPU memory
  needs Antonio's word while anyone holds the card. Small-model plumbing tests (the 0.8B at
  `~/.lmstudio/models/omnirecipes/qwen35-0.8b-GGUF/Qwen3.5-0.8B-Q8_0.gguf`, null device as the card) are allowed at will.
- **`sh -n` only**: never source or run a staged script to see what it prints. One unauthorized run came from that.
- **After the 09-17 reboot the clean card state changed**: DART absent **with no snapshot**, dext **1 instance, pid 659,
  started 07:41:33** (cleared at 08:35, after copying both records to ROOT `logs/`). Anchor before-readings there.

## The slot shape that works (reuse it verbatim)

State to Antonio directly: step, binary + build id, model, duration, expected result, refusal readings, risk class.
Once he gives the go, as separate foreground commands: readings before (anchored dext count, DART line, preflight
OK) → `THRESH=512 sh tools/disagg-serve.sh start` → the requests → `stop` → `sh tools/nv_shim_step.sh V1
opverify` (450/450) → readings after → DONE. Report LANDED/REFUSED with the cache_n values and the card prefill times.

## What is next, after window 3

1. **Residency in the cost model.** The router has no notion of `NCPUMOE` at all: the fixed term is a constant where the
   measurement says it is proportional to the expert bytes actually streamed. Wire it once window 3 says whether the fixed
   term is one number or two (cold and warm).
2. **Prompts past 24,576 tokens on `ad308bd`.** Window 1's 64K point is `2fe7091` and must not be pooled with window 2;
   the per-ubatch structure carries, the seconds do not.
3. **A second model through the same gate** beyond window 3 — the dense 27B, since the placement flags were found by
   crashing on one model and nothing yet says they generalise to a dense architecture.
4. **The 24K state save took 2.5 s in the treatment against 0.7 s in the control** (same 662 MB), unexplained, n=1,
   flagged rather than fitted. Possibly VRAM pressure on the device-to-host read. Watch it in window 3.

Logs: `logs/disagg/serve/{card,metal,router}-20260916-162122.log` (09-16), `…-20260917-073136.log` (window 1),
`…-20260917-073431.log` (control), `…-20260917-073533.log` (treatment), `logs/shim-opverify-20260916-162353.log`.
