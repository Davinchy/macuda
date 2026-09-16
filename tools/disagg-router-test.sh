#!/bin/sh
# disagg-router-test.sh — plumbing test for tools/disagg-router.py with a SMALL model and NO card (V1, 2026-09-16).
#
#   MODEL=/path/to/small.gguf sh tools/disagg-router-test.sh
#
# Three processes, all host-only and small (A's allowance: under 10 GB mapped, no device opened):
#   Metal server  (the kit's llama-server, port 8091)          decodes; --slot-save-path logs/disagg/rt/
#   "card" server (llama-server-null on the NULL DEVICE, 8092)  TINYNV_SOCKET unset: no card is opened, launches are
#                                                              reported and not executed, so the state it saves is
#                                                              garbage by construction — the PLUMBING is what is tested
#   router        (tools/disagg-router.py, port 8095, threshold 64 tokens)
# Checks, each printed as PASS/FAIL:
#   1. a short /completion (under the threshold) is served by Metal alone
#   2. a long /completion takes the card path: card prefill, save, restore, and Metal's cache_n == N-1
#   3. the same long prompt again is served by Metal alone with cache_n ~ N (the router tracks the slot)
#   4. a long /v1/chat/completions takes the card path with cache_n == N-1 (apply-template + tokenize agree with Metal)
#   5. a streamed request is relayed to the end (a "[DONE]" line arrives)
#   6. /health and /tokenize pass through
# Everything is killed on exit, whatever happens.
set -u
R=${EGPU_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}; cd "$R" || exit 2
MODEL=${MODEL:?set MODEL=/path/to/a/small.gguf}; K=/Volumes/512SSD/LocalCode/offline-ai-kit; METAL_BIN=${METAL_BIN:-$K/bin/llama-server}
NULL_BIN=${NULL_BIN:-$R/cuda-shim/build/bin/llama-server-null}
ST=$R/logs/disagg/rt; mkdir -p "$ST"; ts=$(date +%Y%m%d-%H%M%S); L=$ST/test-$ts
test -f "$MODEL" || { echo "no model at $MODEL"; exit 2; }
sz=$(( $(stat -f %z "$MODEL") / 1000000000 )); [ "$sz" -lt 10 ] || { echo "model is ${sz} GB: over the 10 GB allowance for card-free tests"; exit 2; }
echo "== router plumbing test $(date '+%F %T')  model $(basename "$MODEL") (${sz} GB)  logs $L.*"
pids=""
cleanup() { for p in $pids; do kill -TERM "$p" 2>/dev/null; done; sleep 2; for p in $pids; do kill -KILL "$p" 2>/dev/null; done; }
trap cleanup EXIT
wait_health() { i=0; until [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$1/health" 2>/dev/null)" = 200 ]; do i=$((i+1)); [ $i -gt 600 ] && { echo "   port $1 not healthy after 5 min"; return 1; }; sleep 0.5; done; }
"$METAL_BIN" -m "$MODEL" -ngl 999 -c 8192 -b 2048 -ub 512 -fa on --parallel 1 --no-warmup --host 127.0.0.1 --port 8091 --slot-save-path "$ST/" > "$L.metal.log" 2>&1 & pids="$pids $!"
env -u TINYNV_SOCKET "$NULL_BIN" -m "$MODEL" -ngl 999 -c 8192 -b 2048 -ub 2048 -fa on --parallel 1 --no-warmup --host 127.0.0.1 --port 8092 --slot-save-path "$ST/" > "$L.null.log" 2>&1 & pids="$pids $!"
wait_health 8091 || exit 1; wait_health 8092 || exit 1
/usr/bin/grep -m1 -oE 'tinynv: no gpu opened, running as a null device' "$L.null.log" | sed 's/^/   card side: /' || { echo "   the null server did not announce the null device; stopping"; exit 1; }
python3 tools/disagg-router.py --listen 8095 --threshold 64 --state-dir "$ST" > "$L.router.log" 2>&1 & pids="$pids $!"
wait_health 8095 || exit 1
python3 - "$L.router.log" <<'PY'
import json, sys, time, urllib.request
R = "http://127.0.0.1:8095"; rlog = sys.argv[1]; fails = 0
def post(path, obj, raw=False):
    r = urllib.request.urlopen(urllib.request.Request(R + path, json.dumps(obj).encode(), {"Content-Type": "application/json"}), timeout=600)
    return r.read() if raw else json.loads(r.read())
def check(name, ok, detail=""):
    global fails; fails += 0 if ok else 1; print(f"   {'PASS' if ok else 'FAIL'} {name}{': ' + detail if detail else ''}")
def routes(): return open(rlog).read()
long_text = ("The quick brown fox jumps over the lazy dog. " * 60) + "\nSummarize the above in one sentence:"
# 1. short prompt -> metal only
d = post("/completion", {"prompt": "Say hello.", "n_predict": 8, "temperature": 0}); t = d["timings"]
check("1 short /completion served", "content" in d, f"prompt_n {t.get('prompt_n')}")
check("1 short routed metal-only", "-> metal" in routes().splitlines()[-2] if len(routes().splitlines()) > 1 else False)
# 2. long prompt -> card path, cache_n == N-1
n = len(post("/tokenize", {"content": long_text, "add_special": True, "parse_special": True})["tokens"])
before = routes()
d = post("/completion", {"prompt": long_text, "n_predict": 8, "temperature": 0}); t = d["timings"]
new = routes()[len(before):]
check("2 long /completion took the card path", "-> card+metal" in new and "card: prefill" in new, new.strip().splitlines()[-2][:120] if new.strip() else "no router log")
check("2 Metal saw N-1 cached", t.get("cache_n") == n - 1 and t.get("prompt_n") == 1, f"N {n}, prompt_n {t.get('prompt_n')}, cache_n {t.get('cache_n')}")
# 3. same prompt again -> metal only, cached
before = routes(); d = post("/completion", {"prompt": long_text, "n_predict": 8, "temperature": 0}); t = d["timings"]; new = routes()[len(before):]
check("3 repeat routed metal-only", "-> metal" in new and "card:" not in new)
check("3 repeat hit Metal's cache", (t.get("cache_n") or 0) >= n - 1, f"prompt_n {t.get('prompt_n')}, cache_n {t.get('cache_n')}")
# 4. chat request, long -> card path
msgs = [{"role": "user", "content": ("Tell me about the fox. " * 80) + "Answer in one sentence."}]
tpl = post("/apply-template", {"messages": msgs})["prompt"]; nc = len(post("/tokenize", {"content": tpl, "add_special": True, "parse_special": True})["tokens"])
before = routes(); d = post("/v1/chat/completions", {"messages": msgs, "max_tokens": 8, "temperature": 0}); t = d.get("timings", {}); new = routes()[len(before):]
check("4 chat took the card path", "-> card+metal" in new)
check("4 chat: Metal saw N-1 cached", t.get("cache_n") == nc - 1, f"N {nc}, prompt_n {t.get('prompt_n')}, cache_n {t.get('cache_n')}")
# 5. streaming relay
raw = post("/completion", {"prompt": "Count to five:", "n_predict": 8, "temperature": 0, "stream": True}, raw=True).decode(errors="replace")
check("5 stream relayed to the end", raw.count("data:") >= 2 and ("[DONE]" in raw or '"stop":true' in raw), f"{raw.count('data:')} events")
# 6. passthrough
h = urllib.request.urlopen(R + "/health", timeout=10).status; tk = post("/tokenize", {"content": "abc"})
check("6 /health and /tokenize pass through", h == 200 and "tokens" in tk)
print(f"== {'ALL PASS' if fails == 0 else str(fails) + ' FAILED'}")
sys.exit(1 if fails else 0)
PY
rc=$?
echo "== router log =="; cut -c1-160 "$L.router.log"
exit $rc
