#!/bin/sh
# serve.sh — the 27B (or any GGUF) live on the eGPU through the CUDA shim, OpenAI-compatible, MTP on by default.
#
#   sh tools/serve.sh start [model.gguf] [port]   preflight → TinyGPU server → lock → llama-server-null in the background
#   sh tools/serve.sh stop                        stop the server, leave the card idle warm (QUIESCE=1 halts it cold), release the lock
#   sh tools/serve.sh status                      is it up, is it healthy, who holds the lock
#   sh tools/serve.sh test [prompt]               one chat completion: prints the text, tok/s and the draft accept rate
#   sh tools/serve.sh restart                     stop + start with the same model and port (resets the driver's timeline)
#
# KNOWN LIMIT (B, 2026-09-14): the driver's timeline is a 32-bit counter that advances once per launch (~100k/s while
# generating), so ~10–13 HOURS OF CONTINUOUS GENERATION makes requests fail with "the timeline has run past 32 bits" —
# a refusal, not a wrong answer. The counter is per process: `restart` resets it. Restart once a day if it runs
# overnight under load until the driver rebases the counter at an arena wrap (B's fix, hardware-validated separately).
#
# Defaults: the 27B with unsloth's MTP head (MTP=<file> to change, MTP= to disable), draft depth 4, chat-style sampler
# defaults (clients override per request), 8192 context, one slot (PARALLEL=N for batching; decode is one stream per slot).
# Every driver knob passes through the environment exactly as the binary reads it (TINYNV_SYNC=1, TINYNV_CHAIN_DEPTH …).
# Defaults since 2026-09-15 00:20 (B's word, after the one-hour soak: 1,876 requests, 0 errors, 188 probes identical):
# TINYNV_ARENA_DMA=1 (descriptors delivered by the copy engine) and TINYNV_CHAIN_DEPTH=128. Set either explicitly to
# override (TINYNV_ARENA_DMA=0 turns the delivery off). The DRIVER's own default stays off until the region change is soaked.
# DRY=1 serves on the null device: no preflight, no lock, garbage tokens, useful for the API plumbing only.
set -u
R=${EGPU_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}; S=$R/cuda-shim; BIN=${BIN:-$S/build/bin/llama-server-null}
PIDF=$R/logs/serve.pid; URLF=$R/logs/serve.url
cmd=${1:-status}; shift 2>/dev/null || true
case "$cmd" in
  start)
    M=${1:-$R/models/Qwen3.8-27B-UD-Q4_K_M.gguf}; PORT=${2:-8090}
    MTP=${MTP-$R/models/mtp-Qwen3.8-27B-Q4_0.gguf}
    test -f "$M" || { echo "no model at $M"; exit 2; }; test -x "$BIN" || { echo "no server binary at $BIN"; exit 2; }
    [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null && { echo "already running (pid $(cat "$PIDF")); stop it first"; exit 1; }
    spec=""; if [ -n "$MTP" ]; then test -f "$MTP" || { echo "no MTP head at $MTP (MTP= to serve without one)"; exit 2; }
      spec="-md $MTP -ngld 99 --spec-type draft-mtp --spec-draft-n-max ${DRAFT:-4}"; fi
    if [ "${DRY:-0}" = 1 ]; then unset TINYNV_SOCKET; echo "DRY=1: null device, no preflight, no lock"
    else
      sh $R/tools/preflight.sh | tail -1 | grep -q "VERDICT: OK" || { echo "preflight ABORT — not touching the GPU"; exit 1; }
      sh $R/tools/tinygpu-server.sh ensure || exit 1
      sh $R/tools/gpu-lock.sh acquire A "serve $(basename "$M" .gguf) :$PORT" || exit 1
      export TINYNV_SOCKET=${TINYNV_SOCKET:-${TMPDIR}tinygpu.sock}
    fi
# export TINYNV_ARENA_DMA=${TINYNV_ARENA_DMA-1} TINYNV_CHAIN_DEPTH=${TINYNV_CHAIN_DEPTH-128}   # retired 2026-09-15 10:53: the driver defaults to these now (copy-engine delivery, depth 128, sensors on); set a knob explicitly to override
    # 2026-09-15 05:40: sensors (911e68e) off while serving until B has understood two crashes seen with them publishing
# export TINYNV_SENSORS=${TINYNV_SENSORS-0}   # retired 2026-09-15 10:53: the driver defaults to these now (copy-engine delivery, depth 128, sensors on); set a knob explicitly to override
    mkdir -p $R/logs; ts=$(date +%Y%m%d-%H%M%S); log=$R/logs/serve-$ts.log
    echo "== serve  $(date '+%F %T')  $BIN (build id $(strings "$BIN" | grep -c "^$(git -C $S rev-parse --short main)$" | sed 's/^1$/main tip/; s/^0$/NOT main tip/'))  log $log"
    env | grep -E '^TINYNV_' | sed 's/^/   env /'; true
    nohup "$BIN" -m "$M" -ngl 99 $spec -c "${CTX:-8192}" --host 127.0.0.1 --port "$PORT" --parallel "${PARALLEL:-1}" --jinja \
      --temp "${TEMP:-0.6}" --top-p 0.95 --top-k 20 --min-p 0 ${SERVE_ARGS:-} > "$log" 2>&1 &
    echo $! > "$PIDF"; echo "http://127.0.0.1:$PORT" > "$URLF"; printf '%s\n%s\n' "$M" "$PORT" > $R/logs/serve.args
    # /health answers 503 "Loading model" while the weights load; ready means HTTP 200
    i=0; until [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null)" = 200 ]; do i=$((i+1)); [ $i -gt 1200 ] && { echo "server did not come up in 10 min; tail of $log:"; tail -5 "$log"; exit 1; }
      kill -0 "$(cat "$PIDF")" 2>/dev/null || { echo "server died; tail of $log:"; tail -8 "$log" | cut -c1-160; exit 1; }; sleep 0.5; done
    grep -h -E 'libtinynv build|tinynv: (submitting|sync|async|command arena)|loading draft model|spec' "$log" | head -5 | cut -c1-160 | sed 's/^/   /'
    echo "   UP: http://127.0.0.1:$PORT/v1  (pid $(cat "$PIDF"), $(( i / 2 )) s to healthy)  — sh tools/serve.sh test" ;;
  stop)
    [ -f "$PIDF" ] || { echo "not running (no pidfile)"; }
    if [ -f "$PIDF" ]; then pid=$(cat "$PIDF"); if kill -0 "$pid" 2>/dev/null; then kill -TERM "$pid"; i=0; while kill -0 "$pid" 2>/dev/null && [ $i -lt 60 ]; do i=$((i+1)); sleep 0.5; done
        kill -0 "$pid" 2>/dev/null && { echo "still alive after 30 s, sending KILL"; kill -KILL "$pid"; }; echo "stopped pid $pid"; else echo "pid $pid was not running"; fi; rm -f "$PIDF" "$URLF"; fi
    # the lock lives where gpu-lock.sh puts it (the shared EGPU tree when present), whether or not env.sh was sourced
    # by this shell: on 2026-09-19 a stop from a shell without GPU_LOCK looked at the checkout's own .gpu-lock, found no
    # session there, and left the serve lock held after the server was gone.
    LOCKF=${GPU_LOCK:-$([ -d /Volumes/512SSD/EGPU ] && echo /Volumes/512SSD/EGPU/.gpu-lock || echo $R/.gpu-lock)}
    if grep -q "session=A " "$LOCKF" 2>/dev/null; then [ "${QUIESCE:-0}" = 1 ] && sh $R/tools/nv_quiesce.sh 2>&1 | sed 's/^/   /'; sh $R/tools/gpu-lock.sh release A >/dev/null; echo "   card $([ "${QUIESCE:-0}" = 1 ] && echo "quiesced cold" || echo "left idle warm (GSP resident)"), lock released"; fi ;;
  restart)
    test -f $R/logs/serve.args || { echo "nothing to restart from (no logs/serve.args)"; exit 1; }
    M=$(sed -n 1p $R/logs/serve.args); PORT=$(sed -n 2p $R/logs/serve.args); sh "$0" stop; exec sh "$0" start "$M" "$PORT" ;;
  status)
    if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then url=$(cat "$URLF" 2>/dev/null); printf 'serve: pid %s  %s  health: %s\n' "$(cat "$PIDF")" "$url" "$(curl -s "$url/health" 2>/dev/null || echo unreachable)"; else echo "serve: not running"; fi
    sh $R/tools/gpu-lock.sh status ;;
  test)
    url=$(cat "$URLF" 2>/dev/null || echo "http://127.0.0.1:${PORT:-8090}"); P=${1:-"Explain in three sentences why the sky is blue."}
    python3 - "$url" "$P" "${N:-128}" <<'PY'
import sys, json, time, urllib.request
url, prompt, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
req = {"messages": [{"role": "user", "content": prompt}], "max_tokens": n, "stream": False, "timings_per_token": True}
t0 = time.perf_counter()
try:
    r = urllib.request.urlopen(urllib.request.Request(f"{url}/v1/chat/completions", data=json.dumps(req).encode(), headers={"Content-Type": "application/json"}), timeout=600)
except urllib.error.HTTPError as e:
    print(f"server answered HTTP {e.code}: {e.read().decode()[:200]}"); sys.exit(1)
d = json.loads(r.read()); t1 = time.perf_counter()
text = d["choices"][0]["message"]["content"]; print(text.strip()[:500]); t = d.get("timings", {})
pred = t.get("predicted_n"); pps = t.get("predicted_per_second"); dn = t.get("draft_n"); da = t.get("draft_n_accepted")
line = f"--- wall {t1-t0:.2f}s; predicted {pred} tokens at {pps:.1f} tok/s" if pps else f"--- wall {t1-t0:.2f}s; usage {d.get('usage')}"
if dn: line += f"; drafted {dn}, accepted {da} ({100*da/dn:.1f}%)"
print(line); print("    timings:", {k: (round(v,1) if isinstance(v,float) else v) for k,v in t.items()} if t else "none in response")
PY
    ;;
  *) echo "usage: $0 start [model] [port] | stop | restart | status | test [prompt]"; exit 2 ;;
esac
