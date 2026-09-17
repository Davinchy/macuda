# Disaggregated inference: state at the 2026-09-16 close (session V1)

Prefill on the RTX 5090 through the shim, decode on the Mac's GPU, for a model that fits unified memory but not VRAM.
Everything below is in this tree (`tools/disagg-*.{sh,py}`, README §Disaggregated inference); the numbers are from the
real card, driver build `2fe7091`, this M4 Max (128 GB) on Apple's 140 W charger, Qwen3-Coder-Next-UD-Q4_K_XL (49.6 GB).

## What landed, 16:21-16:24 (A: LANDED, op-verify 450/450 after, dext 15→15, DART absent)

The trio (`tools/disagg-serve.sh start`, threshold 512) served three requests through the router:

| request | route | card prefill | Metal saw | decode | wall |
|---|---|---|---|---|---|
| 23,692-token prompt, cold | card → save 662 MB 0.7 s → restore 0.1 s | 15.3 s (1545 tok/s) | prompt_n 1, cache_n 23691 | 52 tok/s | 17.4 s |
| same context + one question | Metal only (24 cold tokens) | none | cache hit | | ~1 s |
| chat, 6,975-token message | card via /apply-template → 251 MB | 8.5 s (821 tok/s) | prompt_n 1, cache_n 6974 | 58 tok/s | 10.3 s |

Metal alone prefills this prompt in 35.7 s (664 tok/s). Greedy text from the split diverges from Metal-only text on
near-ties (0.2-nat margins, full top-5 ranking reproduced): numerics, not a wrong cache.

## The rules the run obeyed, and why

- **Card protocol**: the tenant takes the shared lock as V1 (`. ./env.sh` first), through preflight and the TinyGPU
  server, and holds it while the card server is up; `stop` releases it, card idle warm. A schedules every slot.
- **512-token cap on Metal prefills in the window** (A's condition): the eGPU tunnel dropped twice that day under full
  Metal prefills while the laptop drew power through the enclosure's USB-C PD; on Apple's charger it has held, but n is
  small. With the router's threshold at 512, Metal only ever sees short prompts and cache-hit decodes; the card does
  every long prefill. `THRESH` in `disagg-serve.sh`.
- **Card-free is not host-free**: the Metal half maps 49.6 GB; any job over ~20 GB of host or Apple-GPU memory needs
  A's word while anyone holds the card. Small-model plumbing tests (the 0.8B at
  `~/.lmstudio/models/omnirecipes/qwen35-0.8b-GGUF/Qwen3.5-0.8B-Q8_0.gguf`, null device as the card) are allowed at will.
- **`sh -n` only**: never source or run a staged script to see what it prints. One unauthorized run came from that.

## The slot shape that worked (reuse it verbatim)

Message A with: step, binary + build id, model, duration, expected result, refusal readings, risk class. Then on OPEN-V1,
as separate foreground commands: readings before (anchored dext count, DART line, preflight OK) → `THRESH=512 sh
tools/disagg-serve.sh start` → the requests (`test`, `chat`) → `stop` → `sh tools/nv_shim_step.sh V1 opverify` (450/450)
→ readings after → DONE. Report LANDED/REFUSED with the cache_n values and the card prefill time.

## What is next

1. **A time-based threshold.** The card pays a fixed ~7-8 s to stream all 47 GB of experts whatever the prompt length,
   so a 7K prompt barely wins (8.5 s vs ~10.5 s on Metal). Route by estimated seconds (fixed stream cost + tokens/rate)
   against Metal's rate, not by a token count. Offline work in `tools/disagg-router.py`.
2. **Partial expert residency.** About 20 of the 48 expert layers fit on the card beside the 11.2 GB compute buffer;
   `NCPUMOE=28` would roughly halve the fixed cost. Needs one measured slot (same shape as above).
3. **Prompts over 24,576 tokens** stream the experts once per ubatch (ggml-cuda's MoE id helper caps a ubatch at 25,088
   tokens on sm_120): measure a 64K prompt before promising numbers there.
4. **A second model** through the same gate (the dense 27B), since the placement flags were found by crashing on one.

Logs of the run: `logs/disagg/serve/{card,metal,router}-20260916-162122.log`, `logs/shim-opverify-20260916-162353.log`.
