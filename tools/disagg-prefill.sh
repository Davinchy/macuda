#!/bin/sh
# disagg-prefill.sh — the CARD half of the disaggregated-inference experiment (V1, 2026-09-16).
#
# Runs ONLY through the runner, which does the protocol (preflight → lock as V1 → TinyGPU server → this → release):
#   sh tools/nv_shim_step.sh V1 probe tools/disagg-prefill.sh [prompt-file] [state-name]
# or on the null device to check the wiring:  DRY=1 sh tools/nv_shim_step.sh V1 probe tools/disagg-prefill.sh
#
# What it does: starts llama-server-null on the Qwen3-Coder-Next 80B-A3B (49.6 GB; the card has 32 GB) with every
# expert tensor in HOST memory (--n-cpu-moe 48) and a ubatch as large as the prompt, so the scheduler offloads the
# prompt's MoE matmuls to the card and streams each layer's experts across the link ONCE per prompt; prefills the
# prompt with cache_prompt so the sequence stays in slot 0; saves slot 0's state (KV cache + the linear-attention
# recurrent state, ~24 KB/token + ~81 MB) to a file the Metal side restores; stops the server. Exit 0 only if the
# prefill and the save both succeeded.
#
#
# --no-host --no-repack are load-bearing (found on the null device, 2026-09-16): with a CUDA device present llama.cpp puts
# CPU-placed weights in the CUDA *host* buffer type (not mmappable: 47 GB copied through the shim's pinned pool) and,
# with --no-host alone, in the CPU *repack* buffer type (47 GB repacked on load, and NOT a host buffer, so the scheduler
# never offloads those matmuls to the card). Only the plain CPU buffer type is mmapped AND offloadable.
#
# UBATCH CAP (found on the null device, 2026-09-16): ggml-cuda's MoE id helper (mmid.cu launch_mm_ids_helper) keeps 4 bytes
# per token of the ubatch in shared memory and asserts it fits the device's opt-in limit; the shim advertises 100,352 B
# for sm_120 (a real driver says the same), so a ubatch holds at most 25,088 tokens. A longer prompt is prefilled in
# ceil(n/24576) ubatches and the experts are streamed once PER UBATCH. 24576 keeps a margin under the cap.
# Both halves MUST use the same flash-attention setting: the state file stores V row-wise with -fa on and transposed
# with -fa off (llama-kv-cache.cpp state_write_data, v_trans), and the reader checks its own setting, not the file's.
set -u
R=${EGPU_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
BIN=${BIN:-$R/cuda-shim/build/bin/llama-server-null}
MODEL=${MODEL:-/Volumes/512SSD/LocalCode/offline-ai-kit/models/Qwen3-Coder-Next-UD-Q4_K_XL.gguf}
PROMPT=${1:-$R/logs/disagg/prompt-24k.txt}; NAME=${2:-coder-next-24k.bin}
STATE=$R/logs/disagg; PORT=${PORT:-8092}; CTX=${CTX:-40960}; UB=${UB:-24576}; NCPUMOE=${NCPUMOE:-48}   # NCPUMOE=0 puts the experts on the device (null-device rehearsals only)
mkdir -p "$STATE"; ts=$(date +%Y%m%d-%H%M%S); SLOG=$STATE/prefill-server-$ts.log
test -f "$PROMPT" || { echo "no prompt at $PROMPT"; exit 2; }; test -x "$BIN" || { echo "no binary at $BIN"; exit 2; }
test -f "$MODEL" || { echo "no model at $MODEL"; exit 2; }
echo "disagg-prefill: $(date '+%F %T') prompt $PROMPT ($(wc -c < "$PROMPT" | tr -d ' ') bytes) -> $STATE/$NAME; server log $SLOG"
echo "   socket: ${TINYNV_SOCKET:-<unset: null device>}"
"$BIN" -m "$MODEL" -ngl 999 --n-cpu-moe "$NCPUMOE" --no-host --no-repack -c "$CTX" -b "$UB" -ub "$UB" -fa on --parallel 1 --no-warmup \
  --host 127.0.0.1 --port "$PORT" --slot-save-path "$STATE/" ${PREFILL_ARGS:-} > "$SLOG" 2>&1 &
pid=$!
cleanup() { kill -TERM $pid 2>/dev/null; i=0; while kill -0 $pid 2>/dev/null && [ $i -lt 120 ]; do i=$((i+1)); sleep 0.5; done
            kill -0 $pid 2>/dev/null && { echo "   server still alive after 60 s, KILL"; kill -KILL $pid; }; }
trap cleanup EXIT
t0=$(date +%s); i=0
until [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null)" = 200 ]; do
  i=$((i+1)); [ $i -gt 1200 ] && { echo "   server not healthy after 10 min; tail:"; tail -5 "$SLOG" | cut -c1-160; exit 1; }
  kill -0 $pid 2>/dev/null || { echo "   server died while loading; tail:"; tail -8 "$SLOG" | cut -c1-160; exit 1; }; sleep 0.5; done
echo "   server healthy after $(( $(date +%s) - t0 )) s"
# the driver's own account of the run, echoed so the runner's checks (build id, VOID) see it in its log
/usr/bin/grep -hE 'libtinynv build|found [0-9]+ CUDA|tinynv: (submitting|sync|async|command arena|chain)|n_ubatch +=|CUDA0 model buffer|CPU_Mapped model buffer|CUDA0 compute buffer|KV buffer size|graph splits' "$SLOG" | cut -c1-160 | sed 's/^/   /'
# Prefill all but the LAST prompt token. This model's linear-attention layers carry a recurrent state that cannot be
# rewound, and the decode side must itself evaluate the last prompt token to get logits: a state saved AFTER that token
# forced a full re-prefill on the Metal side (2026-09-16 rehearsal: cache_n 0, 54 s). Saved one token early, the decode
# side finds N-1 cached tokens and evaluates one. Tokenized here the way /completion would, so both sides agree on N.
python3 - "$PORT" "$PROMPT" <<'PY' || exit 1
import sys, json, time, urllib.request
port, pf = sys.argv[1], sys.argv[2]
prompt = open(pf, encoding='utf-8').read()
def post(path, obj):
    r = urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(obj).encode(), {"Content-Type": "application/json"}), timeout=3600)
    return json.loads(r.read())
ids = post("/tokenize", {"content": prompt, "add_special": True, "parse_special": True})["tokens"]
print(f"   TOKENS {len(ids)}: prefilling {len(ids)-1}, the last one is left for the decode side")
t0 = time.perf_counter()
d = post("/completion", {"prompt": ids[:-1], "n_predict": 1, "temperature": 0, "cache_prompt": True, "id_slot": 0, "return_tokens": False})
t = d.get("timings", {})
print(f"   PREFILL wall {time.perf_counter()-t0:.1f} s: prompt_n {t.get('prompt_n')} tokens in {t.get('prompt_ms',0)/1000:.1f} s = {t.get('prompt_per_second',0):.0f} tok/s; cache_n {t.get('cache_n')}")
PY
# save the slot: filename is a basename under --slot-save-path
python3 - "$PORT" "$NAME" <<'PY' || exit 1
import sys, json, time, urllib.request
port, name = sys.argv[1], sys.argv[2]
t0 = time.perf_counter()
r = urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/slots/0?action=save", json.dumps({"filename": name}).encode(), {"Content-Type": "application/json"}), timeout=3600)
d = json.loads(r.read())
print(f"   SAVE {name}: {d.get('n_saved')} tokens, {d.get('n_written',0)/1e6:.0f} MB in {time.perf_counter()-t0:.1f} s")
PY
ls -la "$STATE/$NAME" | sed 's/^/   /'
# the runtime's teardown line (copies, wraps, warnings) arrives when the server exits; keep it in the runner's log
cleanup; trap - EXIT
/usr/bin/grep -hE 'went backwards|is wrong:|not dispatched|warning|WARN|tinycudart\] .*(copies|wraps|overflow)' "$SLOG" | cut -c1-160 | head -8 | sed 's/^/   /'
echo "disagg-prefill: done $(date '+%F %T')"
