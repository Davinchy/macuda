#!/bin/sh
# disagg-serve.sh — the disaggregated trio live under the card protocol (V1, 2026-09-16): the card server (llama-server-null
# on the 5090 through the shim, experts on the host), the Metal server (the kit's llama-server), and the router, as one tenant.
#
#   sh tools/disagg-serve.sh start [model.gguf] [router-port]   preflight → TinyGPU server → lock as V1 → card server → Metal server → router
#   sh tools/disagg-serve.sh test  [prompt-file] [n]     one /completion through the router; prints the router's own lines for it
#   sh tools/disagg-serve.sh chat  [text]                one /v1/chat/completions through the router
#   sh tools/disagg-serve.sh status                      the three processes, the router's tail, who holds the lock
#   sh tools/disagg-serve.sh stop                        router, Metal, card server (SIGTERM, then KILL), release the lock, card idle warm
#
# This holds the card for as long as it is up (the card server is a tenant like tools/serve.sh) AND maps the 49.6 GB model
# twice (host mmap shared): a card step and a host-rule job at once, so it starts only on A's word and stops before the
# window closes. DRY=1 starts the card server on the null device (no preflight, no lock; the Metal half still maps the model,
# so DRY=1 is still a host-rule job with the big model).
#
# Both halves: the SAME model file, the SAME --slot-save-path, -fa on. The card half: --n-cpu-moe 48 --no-host --no-repack
# and a 24576 ubatch (see tools/disagg-prefill.sh for why each is load-bearing).
set -u
R=${EGPU_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}; cd "$R" || exit 2
K=/Volumes/512SSD/LocalCode/offline-ai-kit; METAL_BIN=${METAL_BIN:-$K/bin/llama-server}; CARD_BIN=${BIN:-$R/cuda-shim/build/bin/llama-server-null}
ST=$R/logs/disagg/serve; mkdir -p "$ST"; PIDF=$ST/pids; CTX=${CTX:-40960}; UB=${UB:-24576}; NCPUMOE=${NCPUMOE:-48}
CARD_PORT=${CARD_PORT:-8092}; METAL_PORT=${METAL_PORT:-8091}; THRESH=${THRESH:-5000}
cmd=${1:-status}
# the router port: the third argument of start, remembered in $ST/router.port for every other subcommand
if [ "$cmd" = start ]; then PORT=${3:-${PORT:-8095}}; else PORT=${PORT:-$(cat "$ST/router.port" 2>/dev/null || echo 8095)}; fi
wait_health() { i=0; until [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$1/health" 2>/dev/null)" = 200 ]; do
  i=$((i+1)); [ $i -gt 1200 ] && { echo "   port $1 not healthy after 10 min"; return 1; }
  kill -0 "$2" 2>/dev/null || { echo "   pid $2 (port $1) died; tail of $3:"; tail -6 "$3" | cut -c1-160; return 1; }; sleep 0.5; done; }
case "$cmd" in
  start)
    M=${2:-/Volumes/512SSD/LocalCode/offline-ai-kit/models/Qwen3-Coder-Next-UD-Q4_K_XL.gguf}
    test -f "$M" || { echo "no model at $M"; exit 2; }; test -x "$CARD_BIN" || { echo "no card binary at $CARD_BIN"; exit 2; }; test -x "$METAL_BIN" || { echo "no Metal server at $METAL_BIN"; exit 2; }
    [ -f "$PIDF" ] && { echo "pids file exists ($PIDF): stop first"; exit 1; }
    ts=$(date +%Y%m%d-%H%M%S); CL=$ST/card-$ts.log; ML=$ST/metal-$ts.log; RL=$ST/router-$ts.log
    if [ "${DRY:-0}" = 1 ]; then unset TINYNV_SOCKET; echo "DRY=1: card server on the null device, no preflight, no lock"
    else
      sh tools/preflight.sh | tail -1 | grep -q "VERDICT: OK" || { echo "preflight ABORT — not touching the GPU"; exit 1; }
      sh tools/tinygpu-server.sh ensure || exit 1
      sh tools/gpu-lock.sh acquire V1 "disagg-serve $(basename "$M" .gguf) :$PORT" "$$" || exit 1
      export TINYNV_SOCKET=${TINYNV_SOCKET:-${TMPDIR}tinygpu.sock}
    fi
    echo "== disagg-serve start $(date '+%F %T')  model $(basename "$M")  card $CARD_PORT  metal $METAL_PORT  router $PORT  logs $ST/*-$ts.log"
    "$CARD_BIN" -m "$M" -ngl 999 --n-cpu-moe "$NCPUMOE" --no-host --no-repack -c "$CTX" -b "$UB" -ub "$UB" -fa on --parallel 1 --no-warmup \
      --host 127.0.0.1 --port "$CARD_PORT" --slot-save-path "$ST/" > "$CL" 2>&1 & cpid=$!
    echo "card=$cpid" > "$PIDF"
    wait_health "$CARD_PORT" "$cpid" "$CL" || { sh "$0" stop; exit 1; }
    /usr/bin/grep -hE 'libtinynv build|tinynv: (submitting|no gpu)|found [0-9]+ CUDA' "$CL" | head -3 | cut -c1-140 | sed 's/^/   card: /'
    "$METAL_BIN" -m "$M" -ngl 999 -c "$CTX" -b 4096 -ub 512 -fa on --parallel 1 --no-warmup --host 127.0.0.1 --port "$METAL_PORT" --slot-save-path "$ST/" > "$ML" 2>&1 & mpid=$!
    echo "metal=$mpid" >> "$PIDF"
    wait_health "$METAL_PORT" "$mpid" "$ML" || { sh "$0" stop; exit 1; }
    python3 tools/disagg-router.py --listen "$PORT" --metal "http://127.0.0.1:$METAL_PORT" --card "http://127.0.0.1:$CARD_PORT" --threshold "$THRESH" --state-dir "$ST" > "$RL" 2>&1 & rpid=$!
    echo "router=$rpid" >> "$PIDF"; echo "$RL" > "$ST/router.log.path"; echo "$PORT" > "$ST/router.port"
    wait_health "$PORT" "$rpid" "$RL" || { sh "$0" stop; exit 1; }
    echo "   UP: http://127.0.0.1:$PORT (router), threshold $THRESH cold tokens; $(sh tools/gpu-lock.sh status)" ;;
  stop)
    [ -f "$PIDF" ] || { echo "not running (no $PIDF)"; }
    if [ -f "$PIDF" ]; then
      for who in router metal card; do p=$(sed -n "s/^$who=//p" "$PIDF"); [ -n "$p" ] || continue
        if kill -0 "$p" 2>/dev/null; then kill -TERM "$p"; i=0; while kill -0 "$p" 2>/dev/null && [ $i -lt 120 ]; do i=$((i+1)); sleep 0.5; done
          kill -0 "$p" 2>/dev/null && { echo "   $who still alive after 60 s, KILL"; kill -KILL "$p"; }; echo "   stopped $who (pid $p)"; fi; done
      rm -f "$PIDF"
    fi
    if grep -q "session=V1 " "${GPU_LOCK:-$R/.gpu-lock}" 2>/dev/null; then sh tools/gpu-lock.sh release V1 >/dev/null; echo "   card left idle warm, lock released"; fi ;;
  status)
    if [ -f "$PIDF" ]; then while IFS== read -r who p; do printf '   %-6s pid %-6s %s\n' "$who" "$p" "$(kill -0 "$p" 2>/dev/null && echo alive || echo DEAD)"; done < "$PIDF"
      [ -f "$ST/router.log.path" ] && tail -5 "$(cat "$ST/router.log.path")" | cut -c1-160 | sed 's/^/   /'
    else echo "   not running"; fi; sh tools/gpu-lock.sh status ;;
  test)
    P=${2:-$R/logs/disagg/prompt-24k.txt}; N=${3:-64}; test -f "$P" || { echo "no prompt at $P"; exit 2; }
    python3 - "$PORT" "$P" "$N" <<'PY'
import sys, json, time, urllib.request
port, pf, n = sys.argv[1], sys.argv[2], int(sys.argv[3]); prompt = open(pf, encoding="utf-8").read()
t0 = time.perf_counter()
r = urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/completion", json.dumps({"prompt": prompt, "n_predict": n, "temperature": 0}).encode(), {"Content-Type": "application/json"}), timeout=3600)
d = json.loads(r.read()); t = d.get("timings", {})
print(f"   TEST wall {time.perf_counter()-t0:.1f} s: prompt_n {t.get('prompt_n')} cache_n {t.get('cache_n')} prompt {t.get('prompt_ms',0)/1000:.1f} s; decode {t.get('predicted_n')} tok at {t.get('predicted_per_second',0):.1f} tok/s")
print("   text: " + d.get("content", "")[:160].replace("\n", " "))
PY
    [ -f "$ST/router.log.path" ] && tail -4 "$(cat "$ST/router.log.path")" | cut -c1-170 | sed 's/^/   /' ;;
  chat)
    T=${2:-"In one paragraph, what does this driver do?"}
    python3 - "$PORT" "$T" <<'PY'
import sys, json, time, urllib.request
port, text = sys.argv[1], sys.argv[2]; t0 = time.perf_counter()
r = urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps({"messages": [{"role": "user", "content": text}], "max_tokens": 64, "temperature": 0}).encode(), {"Content-Type": "application/json"}), timeout=3600)
d = json.loads(r.read()); t = d.get("timings", {})
print(f"   CHAT wall {time.perf_counter()-t0:.1f} s: prompt_n {t.get('prompt_n')} cache_n {t.get('cache_n')}; decode {t.get('predicted_n')} tok at {t.get('predicted_per_second',0):.1f} tok/s")
print("   text: " + d["choices"][0]["message"]["content"][:160].replace("\n", " "))
PY
    [ -f "$ST/router.log.path" ] && tail -3 "$(cat "$ST/router.log.path")" | cut -c1-170 | sed 's/^/   /' ;;
  *) echo "usage: $0 start [model] [port] | test [prompt-file] [n] | chat [text] | status | stop"; exit 2 ;;
esac
