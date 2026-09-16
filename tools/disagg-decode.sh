#!/bin/sh
# disagg-decode.sh — the METAL half of the disaggregated-inference experiment (V1, 2026-09-16). NO card: this runs the
# offline-ai-kit's own Metal llama-server on the Mac GPU; no lock, no preflight, nothing from A.
#
#   sh tools/disagg-decode.sh baseline [prompt-file] [n]       Metal alone: full prefill + n greedy tokens; the reference
#   sh tools/disagg-decode.sh restore  [prompt-file] [n] [state-name]
#                                                             restore slot 0 from the card's state file, then the same request:
#                                                             prefill should be ~0 tokens (cache hit), decode n greedy tokens,
#                                                             and the text is compared with the baseline's, token by token
#
# Same model file (same tokenizer), same -fa on (see disagg-prefill.sh), same context. Port 8091 so the kit's usual
# server on 8090 is untouched. Timings come from the server's own timings block, not wall clocks.
set -u
R=${EGPU_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
K=/Volumes/512SSD/LocalCode/offline-ai-kit; BIN=${METAL_BIN:-$K/bin/llama-server}
MODEL=${MODEL:-$K/models/Qwen3-Coder-Next-UD-Q4_K_XL.gguf}
mode=${1:?baseline|restore}; PROMPT=${2:-$R/logs/disagg/prompt-24k.txt}; N=${3:-128}; NAME=${4:-coder-next-24k.bin}
STATE=$R/logs/disagg; PORT=${PORT:-8091}; CTX=${CTX:-40960}
mkdir -p "$STATE"; ts=$(date +%Y%m%d-%H%M%S); SLOG=$STATE/metal-$mode-server-$ts.log; OUT=$STATE/metal-$mode-$ts
test -f "$PROMPT" || { echo "no prompt at $PROMPT"; exit 2; }; test -x "$BIN" || { echo "no Metal server at $BIN"; exit 2; }
[ "$mode" = restore ] && { test -f "$STATE/$NAME" || { echo "no state file at $STATE/$NAME (run the prefill half first)"; exit 2; }; }
echo "disagg-decode $mode: $(date '+%F %T') prompt $PROMPT n=$N; server log $SLOG; result $OUT.{txt,json}"
"$BIN" -m "$MODEL" -ngl 999 -c "$CTX" -b 4096 -ub 512 -fa on --parallel 1 --no-warmup \
  --host 127.0.0.1 --port "$PORT" --slot-save-path "$STATE/" > "$SLOG" 2>&1 &
pid=$!
cleanup() { kill -TERM $pid 2>/dev/null; i=0; while kill -0 $pid 2>/dev/null && [ $i -lt 120 ]; do i=$((i+1)); sleep 0.5; done
            kill -0 $pid 2>/dev/null && kill -KILL $pid; }
trap cleanup EXIT
t0=$(date +%s); i=0
until [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null)" = 200 ]; do
  i=$((i+1)); [ $i -gt 1200 ] && { echo "   server not healthy after 10 min; tail:"; tail -5 "$SLOG" | cut -c1-160; exit 1; }
  kill -0 $pid 2>/dev/null || { echo "   server died while loading; tail:"; tail -8 "$SLOG" | cut -c1-160; exit 1; }; sleep 0.5; done
echo "   Metal server healthy after $(( $(date +%s) - t0 )) s ($(/usr/bin/grep -m1 -oE 'GPU name: .*' "$SLOG" | cut -c1-60))"
if [ "$mode" = restore ]; then
  python3 - "$PORT" "$NAME" <<'PY' || exit 1
import sys, json, time, urllib.request
port, name = sys.argv[1], sys.argv[2]
t0 = time.perf_counter()
try:
    r = urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/slots/0?action=restore", json.dumps({"filename": name}).encode(), {"Content-Type": "application/json"}), timeout=3600)
except urllib.error.HTTPError as e:
    print(f"   RESTORE FAILED: HTTP {e.code}: {e.read().decode()[:300]}"); sys.exit(1)
d = json.loads(r.read())
print(f"   RESTORE {name}: {d.get('n_restored')} tokens, {d.get('n_read',0)/1e6:.0f} MB in {time.perf_counter()-t0:.1f} s")
PY
fi
python3 - "$PORT" "$PROMPT" "$N" "$OUT" <<'PY' || exit 1
import sys, json, time, urllib.request
port, pf, n, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
prompt = open(pf, encoding='utf-8').read()
body = json.dumps({"prompt": prompt, "n_predict": n, "temperature": 0, "seed": 7, "cache_prompt": True, "id_slot": 0, "return_tokens": True, "n_probs": int(__import__("os").environ.get("NPROBS", "0"))}).encode()
t0 = time.perf_counter()
r = urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/completion", body, {"Content-Type": "application/json"}), timeout=3600)
d = json.loads(r.read()); wall = time.perf_counter() - t0; t = d.get("timings", {})
open(out + ".txt", "w").write(d.get("content", "")); json.dump({"timings": t, "tokens": d.get("tokens", []), "probs": d.get("completion_probabilities", [])}, open(out + ".json", "w"))
print(f"   PREFILL prompt_n {t.get('prompt_n')} tokens (cache_n {t.get('cache_n')}) in {t.get('prompt_ms',0)/1000:.1f} s = {t.get('prompt_per_second',0):.0f} tok/s")
print(f"   DECODE  {t.get('predicted_n')} tokens in {t.get('predicted_ms',0)/1000:.1f} s = {t.get('predicted_per_second',0):.1f} tok/s; wall {wall:.1f} s")
print("   text: " + d.get("content", "")[:200].replace("\n", " "))
PY
if [ "$mode" = restore ]; then
  base=$(ls -t "$STATE"/metal-baseline-*.json 2>/dev/null | head -1)
  if [ -n "$base" ]; then python3 - "$base" "$OUT.json" <<'PY'
import sys, json
a = json.load(open(sys.argv[1]))["tokens"]; b = json.load(open(sys.argv[2]))["tokens"]
k = 0
while k < min(len(a), len(b)) and a[k] == b[k]: k += 1
print(f"   COMPARE vs baseline {sys.argv[1].split('/')[-1]}: {k} of {min(len(a),len(b))} tokens identical" + (" (ALL)" if k == min(len(a),len(b)) else f"; first difference at token {k}"))
PY
  else echo "   (no baseline result to compare against; run: sh tools/disagg-decode.sh baseline)"; fi
fi
echo "disagg-decode $mode: done $(date '+%F %T')"
