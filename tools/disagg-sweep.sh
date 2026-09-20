#!/bin/sh
# disagg-sweep.sh — prefill the SAME prompts on one half of the disaggregated path, several prompt lengths per model
# load (2026-09-19). The pair of halves is what the experiment compares:
#
#   card:   sh tools/nv_shim_step.sh V1 probe tools/disagg-sweep.sh card  <tokens> [<tokens>...]
#   Metal:  sh tools/disagg-sweep.sh metal <tokens> [<tokens>...]         (no card, no lock, no preflight)
#
# The arguments are TOKEN COUNTS, not files. A prompt file cannot be compared across models: the same 267 KB of text
# is 74k tokens to one tokenizer and 60k to another, so "the same prompt" would land on different sides of the
# 24,576-token ubatch boundary for different models and the step being measured would be an artefact of vocabulary.
# Instead each server tokenizes one base text with ITS OWN tokenizer and the ids are sliced to the requested length,
# so every model is asked for exactly 24,000 tokens of the same text.
#
# Why one server for the whole sweep: disagg-prefill.sh loads the model per prompt, and at 60-80 GB that is minutes
# of page-cache filling per point - on a four-model sweep it dominates the measurement's wall time and nothing is
# learned from it. The slot is ERASED between prompts instead, because these prompts are truncations of one text and
# share long prefixes: without the erase the second and third points would be cache hits and read as absurdly fast.
#
# MODEL is required. NCPUMOE (card) decides how much of the model streams: n_layer puts EVERY expert in host memory,
# which is the configuration the cost model's constants were fitted on and the only one that is identical across
# models, so it is the default here. CTX must cover the longest prompt. Rows are appended to $OUT as TSV.
set -u
R=${EGPU_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
side=${1:?card|metal}; shift
: "${MODEL:?set MODEL to the .gguf (shard 1 of a split is the whole model to llama.cpp)}"
TAG=${TAG:-$(basename "$MODEL" .gguf | cut -c1-32)}
OUT=${OUT:-/Volumes/Crucial_8TB/disagg-bench/results.tsv}
STATE=${STATE:-/Volumes/Crucial_8TB/disagg-bench/state}
CTX=${CTX:-57344}; NCPUMOE=${NCPUMOE:-99}; UB=${UB:-24576}
mkdir -p "$(dirname "$OUT")" "$STATE"
ts=$(date +%Y%m%d-%H%M%S); SLOG=$STATE/../logs/sweep-$side-$TAG-$ts.log; mkdir -p "$(dirname "$SLOG")"
test -f "$MODEL" || { echo "no model at $MODEL"; exit 2; }
BASE=${BASE:-$R/logs/disagg/prompt-74k-3ub.txt}
test -f "$BASE" || { echo "no base text at $BASE"; exit 2; }

if [ "$side" = card ]; then
  BIN=${BIN:-$R/cuda-shim/build/bin/llama-server-null}; PORT=${PORT:-8092}
  test -x "$BIN" || { echo "no binary at $BIN"; exit 2; }
  echo "sweep card: $(date '+%F %T') $TAG ncpumoe=$NCPUMOE ctx=$CTX ub=$UB; server log $SLOG"
  echo "   socket: ${TINYNV_SOCKET:-<unset: null device>}"
  # SERVER_ARGS passes anything else straight to the server, unsplit - e.g. --cache-type-k q8_0 --cache-type-v q8_0.
  # Quantising the KV pays twice on this path: less of the card's memory spent on cache means more resident experts,
  # and the state handed to Metal IS the KV, so the handoff shrinks with it.
  "$BIN" -m "$MODEL" -ngl 999 --n-cpu-moe "$NCPUMOE" --no-host --no-repack -c "$CTX" -b "$UB" -ub "$UB" -fa on \
    --parallel 1 --no-warmup --host 127.0.0.1 --port "$PORT" --slot-save-path "$STATE/" ${SERVER_ARGS:-} > "$SLOG" 2>&1 &
else
  K=/Volumes/512SSD/LocalCode/offline-ai-kit; BIN=${METAL_BIN:-$K/bin/llama-server}; PORT=${PORT:-8091}
  test -x "$BIN" || { echo "no Metal server at $BIN"; exit 2; }
  echo "sweep metal: $(date '+%F %T') $TAG ctx=$CTX; server log $SLOG"
  "$BIN" -m "$MODEL" -ngl 999 -c "$CTX" -b 4096 -ub 512 -fa on --parallel 1 --no-warmup \
    --host 127.0.0.1 --port "$PORT" --slot-save-path "$STATE/" ${SERVER_ARGS:-} > "$SLOG" 2>&1 &
fi
pid=$!
cleanup() { kill -TERM $pid 2>/dev/null; i=0; while kill -0 $pid 2>/dev/null && [ $i -lt 240 ]; do i=$((i+1)); sleep 0.5; done
            kill -0 $pid 2>/dev/null && { echo "   server still alive after 2 min, KILL"; kill -KILL $pid; }; }
trap cleanup EXIT
t0=$(date +%s); i=0
until [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null)" = 200 ]; do
  i=$((i+1)); [ $i -gt 2400 ] && { echo "   server not healthy after 20 min; tail:"; tail -6 "$SLOG" | cut -c1-160; exit 1; }
  kill -0 $pid 2>/dev/null || { echo "   server died while loading; tail:"; tail -12 "$SLOG" | cut -c1-160; exit 1; }; sleep 0.5; done
load=$(( $(date +%s) - t0 )); echo "   server healthy after $load s"
/usr/bin/grep -hE 'libtinynv build|found [0-9]+ CUDA|n_ubatch +=|CUDA0 model buffer|CPU_Mapped model buffer|Metal.*model buffer|CUDA0 compute buffer|KV buffer size|graph splits|GPU name' "$SLOG" | cut -c1-160 | sed 's/^/   /'

LOAD=$load SIDE=$side TAG=$TAG OUT=$OUT PORT=$PORT BASE=$BASE python3 - "$@" <<'PY' || exit 1
import json, os, sys, time, urllib.request, datetime
port, side, tag, out, load = os.environ["PORT"], os.environ["SIDE"], os.environ["TAG"], os.environ["OUT"], os.environ["LOAD"]
def post(path, obj, timeout=7200):
    r = urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}{path}",
        json.dumps(obj).encode(), {"Content-Type": "application/json"}), timeout=timeout)
    return json.loads(r.read())
new = not os.path.exists(out)
with open(out, "a") as fh:
    if new: fh.write("when\tside\tmodel\tprompt\ttokens\tprefill_s\ttok_s\twall_s\tload_s\n")
    text = open(os.environ["BASE"], encoding="utf-8").read()
    all_ids = post("/tokenize", {"content": text, "add_special": True, "parse_special": True})["tokens"]
    print(f"   base text: {len(text)} chars = {len(all_ids)} tokens by this model's own tokenizer")
    for arg in sys.argv[1:]:
        want = int(arg); name = f"{want//1000}k"
        if want > len(all_ids):
            print(f"   {want} tokens asked for, base text only holds {len(all_ids)} - skipped"); continue
        ids = all_ids[:want]
        # erase first: these prompts share prefixes, and a cache hit would be recorded as a fast prefill
        try: urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/slots/0?action=erase", b"{}", {"Content-Type": "application/json"}), timeout=600)
        except Exception as e: print(f"   (erase: {e})")
        t0 = time.perf_counter()
        d = post("/completion", {"prompt": ids[:-1], "n_predict": 1, "temperature": 0, "cache_prompt": True,
                                 "id_slot": 0, "return_tokens": False})
        wall = time.perf_counter() - t0; t = d.get("timings", {})
        n, ms, rate = t.get("prompt_n"), t.get("prompt_ms", 0), t.get("prompt_per_second", 0)
        cache_n = t.get("cache_n")
        flag = "" if not cache_n else f"  !! cache_n {cache_n} - NOT a cold prefill"
        print(f"   {side:5} {name:>12}: {n} tokens in {ms/1000:7.1f} s = {rate:6.0f} tok/s (wall {wall:.1f} s){flag}")
        fh.write(f"{datetime.datetime.now().isoformat(timespec='seconds')}\t{side}\t{tag}\t{name}\t{n}\t{ms/1000:.2f}\t{rate:.1f}\t{wall:.1f}\t{load}\n")
        fh.flush()
PY
cleanup; trap - EXIT
/usr/bin/grep -hE 'went backwards|is wrong:|not dispatched|error|ERROR' "$SLOG" | cut -c1-160 | head -6 | sed 's/^/   /'
echo "sweep $side: done $(date '+%F %T')"
