#!/bin/sh
# disagg-batch.sh — batch serving: prefill a shared prompt ONCE, broadcast it to N parallel slots via
# save/restore, then serve N independent per-job completions extending that shared prefix (V1, 2026-09-18).
#
# Confirmed on real hardware: an 11,461-token shared prompt across 4 jobs landed in 16.2s total (11.5s card
# prefill + 0.4s save + 0.09s for 4 restores + 4.2s for 4 seeded queries) against 66.7s for 4 independent full
# prefills on fresh slots — 4.1x, and the win only grows with more jobs (each one past the first costs ~1s
# instead of a full prefill). Also confirmed working entirely on the card alone, no Metal at all, via
# `card-only` below (same mechanism, same-process multi-slot restore).
#
# CRITICAL, the one thing that makes this work: every seeded request MUST be a TOKEN-LIST prompt (pre-tokenized
# ids), never a raw string. A string re-tokenized against a restored slot fails to register a cache hit past a
# trivial length (~1-2 tokens) — confirmed on real hardware, direction- and parallel-count-independent (see
# docs/disagg-inference-state.md, "reverse sync ... RESOLVED"). This tool always tokenizes client-side and sends
# ids; never change `job` to send a raw string to a seeded slot.
#
#   sh tools/disagg-batch.sh start <model.gguf> <shared-prompt-file> <n-slots> [ncpumoe=48]
#       preflight -> lock as V1 -> card server -> seed the shared prompt on the card, save it -> Metal server
#       (--parallel n-slots, same n-stream so the restore isn't rejected) -> restore the saved prefix onto every
#       Metal slot. Ready for `job` calls against slots 0..n-slots-1 after.
#   sh tools/disagg-batch.sh card-only <model.gguf> <shared-prompt-file> <n-slots> [ncpumoe=48]
#       the same idea entirely on the card, no Metal: seed slot 0, save, restore onto slots 1..n-slots-1, all in
#       one process. `job` against slots 1..n-slots-1 afterward (slot 0 already holds the bare shared prompt).
#   sh tools/disagg-batch.sh job <slot> <job-text-file-or-string> [n_predict=64]
#       one seeded completion against the given slot. Prints wall time and cache_n; cache_n should equal the
#       shared prompt's token count exactly — if it's lower, the restore didn't take for this request and it
#       silently ran a full prefill instead (see the reverse-sync postmortem in the state doc for why that can
#       happen: the seeded slot's real content must actually match this job's prefix, which it does whenever the
#       shared prompt itself never changes underneath it, i.e. always, in this tool's own usage).
#   sh tools/disagg-batch.sh status | stop
#
# DRY=1 is not supported: the null device cannot exercise the expert-streaming path this depends on.
set -u
R=${EGPU_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}; cd "$R" || exit 2
K=/Volumes/512SSD/LocalCode/offline-ai-kit; METAL_BIN=${METAL_BIN:-$K/bin/llama-server}
CARD_BIN=${BIN:-$R/cuda-shim/build/bin/llama-server-null}
ST=$R/logs/disagg/batch; mkdir -p "$ST"; PIDF=$ST/pids; META=$ST/meta
CARD_PORT=${CARD_PORT:-8092}; METAL_PORT=${METAL_PORT:-8091}
NCPUMOE_DEFAULT=48; CTX=${CTX:-200000}; UB=${UB:-24576}

wait_health() { i=0; until [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$1/health" 2>/dev/null)" = 200 ]; do
  i=$((i+1)); [ $i -gt 1200 ] && { echo "   port $1 not healthy after 10 min"; return 1; }
  kill -0 "$2" 2>/dev/null || { echo "   pid $2 (port $1) died; tail of $3:"; tail -8 "$3" | cut -c1-160; return 1; }; sleep 0.5; done; }

post() { # port path json-body(already valid JSON, e.g. from a file) -> prints response body
  # python/urllib, not curl -d: a large body (a shared prompt's full token-id array can be 50-100 KB+) passed as
  # a single curl command-line argument was found 2026-09-18 to silently fail (empty stdout, no error) - urllib
  # posting the same bytes as a request body works at any size already exercised in this project.
  python3 - "$1" "$2" "$3" <<'PY'
import sys, urllib.request
port, path, body = sys.argv[1], sys.argv[2], sys.argv[3]
r = urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}{path}", body.encode(),
    {"Content-Type": "application/json"}), timeout=3600)
sys.stdout.write(r.read().decode())
PY
}

tokenize_file() { # port file -> prints a JSON array of token ids
  python3 - "$1" "$2" <<'PY'
import sys, json, urllib.request
port, path = sys.argv[1], sys.argv[2]
text = open(path, encoding="utf-8").read()
body = json.dumps({"content": text, "add_special": True, "parse_special": True}).encode()
r = urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/tokenize", body, {"Content-Type": "application/json"}))
print(json.dumps(json.loads(r.read())["tokens"]))
PY
}

cmd=${1:-status}
case "$cmd" in
  start|card-only)
    M=${2:?model.gguf}; SHARED=${3:?shared-prompt-file}; NSLOTS=${4:?n-slots}; NCPUMOE=${5:-$NCPUMOE_DEFAULT}
    test -f "$M" || { echo "no model at $M"; exit 2; }; test -f "$SHARED" || { echo "no shared prompt at $SHARED"; exit 2; }
    test -x "$CARD_BIN" || { echo "no card binary at $CARD_BIN"; exit 2; }
    [ -f "$PIDF" ] && { echo "pids file exists ($PIDF): stop first"; exit 1; }
    card_only=0; [ "$cmd" = card-only ] && card_only=1
    sh tools/preflight.sh | tail -1 | grep -q "VERDICT: OK" || { echo "preflight ABORT — not touching the GPU"; exit 1; }
    sh tools/tinygpu-server.sh ensure || exit 1
    sh tools/gpu-lock.sh acquire V1 "disagg-batch $(basename "$M" .gguf) x$NSLOTS" "$$" || exit 1
    export TINYNV_SOCKET=${TINYNV_SOCKET:-${TMPDIR}tinygpu.sock}
    ts=$(date +%Y%m%d-%H%M%S); CL=$ST/card-$ts.log; ML=$ST/metal-$ts.log
    # the card and Metal must share the SAME --parallel count even when only the card ever prefills, because the
    # save file's n_stream dimension is checked on restore and a mismatch is rejected outright (found 2026-09-18)
    "$CARD_BIN" -m "$M" -ngl 999 --n-cpu-moe "$NCPUMOE" --no-host --no-repack -c "$CTX" -b "$UB" -ub "$UB" -fa on \
      --parallel "$NSLOTS" --no-warmup --host 127.0.0.1 --port "$CARD_PORT" --slot-save-path "$ST/" > "$CL" 2>&1 &
    cpid=$!; echo "card=$cpid" > "$PIDF"
    wait_health "$CARD_PORT" "$cpid" "$CL" || { sh "$0" stop; exit 1; }
    /usr/bin/grep -hE 'libtinynv build' "$CL" | head -1 | sed 's/^/   /'

    if [ "$card_only" = 0 ]; then
      test -x "$METAL_BIN" || { echo "no Metal server at $METAL_BIN"; sh "$0" stop; exit 1; }
      "$METAL_BIN" -m "$M" -ngl 999 -c "$CTX" -b 4096 -ub 512 -fa on --parallel "$NSLOTS" --no-warmup \
        --host 127.0.0.1 --port "$METAL_PORT" --slot-save-path "$ST/" > "$ML" 2>&1 &
      mpid=$!; echo "metal=$mpid" >> "$PIDF"
      wait_health "$METAL_PORT" "$mpid" "$ML" || { sh "$0" stop; exit 1; }
      TOK_PORT=$METAL_PORT
    else
      TOK_PORT=$CARD_PORT
    fi

    echo "== seeding the shared prompt on the card =="
    SHARED_IDS=$(tokenize_file "$TOK_PORT" "$SHARED")
    echo "$SHARED_IDS" > "$ST/shared-ids.json"
    N_SHARED=$(python3 -c "import json,sys; print(len(json.loads(sys.argv[1])))" "$SHARED_IDS")
    r=$(post "$CARD_PORT" /completion "{\"prompt\": $SHARED_IDS, \"n_predict\": 1, \"temperature\": 0, \"cache_prompt\": true, \"id_slot\": 0, \"return_tokens\": false}")
    # NOTE: pass data via argv, never pipe+heredoc into the same python3 call - a heredoc IS the program's stdin
    # (its source, since no -c/file was given), so a pipe feeding that same stdin is invisible to the program;
    # json.load(sys.stdin) inside then sees nothing and raises "Expecting value" (found writing this tool).
    python3 - "$r" <<'PY'
import json, sys
t = json.loads(sys.argv[1]).get("timings", {})
print(f"   prefill {t.get('prompt_n')} tok in {t.get('prompt_ms', 0)/1000:.1f}s ({t.get('prompt_per_second', 0):.0f} tok/s)")
PY
    s=$(post "$CARD_PORT" "/slots/0?action=save" '{"filename": "batch-shared.bin"}')
    python3 - "$s" <<'PY'
import json, sys
d = json.loads(sys.argv[1])
print(f"   save: {d.get('n_saved')} tok, {d.get('n_written', 0)/1e6:.0f} MB")
PY

    if [ "$card_only" = 0 ]; then
      echo "== restoring onto $NSLOTS Metal slots =="
      i=0; while [ "$i" -lt "$NSLOTS" ]; do
        m=$(post "$METAL_PORT" "/slots/$i?action=restore" '{"filename": "batch-shared.bin"}')
        echo "   slot $i: $(echo "$m" | python3 -c "import json,sys; print(json.load(sys.stdin).get('n_restored'))") tok restored"
        i=$((i+1))
      done
      echo "$N_SHARED $NSLOTS $METAL_PORT $METAL_PORT 0" > "$META"
    else
      echo "== restoring onto card slots 1..$((NSLOTS-1)) (same process) =="
      i=1; while [ "$i" -lt "$NSLOTS" ]; do
        m=$(post "$CARD_PORT" "/slots/$i?action=restore" '{"filename": "batch-shared.bin"}')
        echo "   slot $i: $(echo "$m" | python3 -c "import json,sys; print(json.load(sys.stdin).get('n_restored'))") tok restored"
        i=$((i+1))
      done
      echo "$N_SHARED $NSLOTS $CARD_PORT $CARD_PORT 1" > "$META"
    fi
    echo "== ready: $NSLOTS slots seeded with $N_SHARED shared tokens; sh $0 job <slot> <text-or-file> [n_predict] ==" ;;

  job)
    [ -f "$META" ] || { echo "not started (no $META): run start/card-only first"; exit 1; }
    read -r N_SHARED NSLOTS QPORT CPORT CARD_ONLY < "$META"
    SLOT=${2:?slot 0..$((NSLOTS-1))}; JOB=${3:?job text or file}; NPRED=${4:-64}
    [ -f "$JOB" ] && JOBFILE="$JOB" || { JOBFILE=$(mktemp); printf '%s' "$JOB" > "$JOBFILE"; }
    # one process end to end (tokenize the job, build shared_ids+job_ids, time the completion, report) so the
    # timing reported is the real request cost, not shell-loop/python-startup overhead added around it. The
    # shared prompt's own ids are never re-tokenized here (that would risk a boundary mismatch against what's
    # actually seeded) - they're read from the sidecar start/card-only wrote once.
    python3 - "$QPORT" "$CPORT" "$SLOT" "$NPRED" "$N_SHARED" "$ST/shared-ids.json" "$JOBFILE" <<'PY'
import sys, json, time, urllib.request
tok_port, c_port, slot, npred, n_shared, shared_path, job_path = sys.argv[1:8]
def post(port, path, body):
    r = urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}{path}",
        json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=3600)
    return json.loads(r.read())
job_text = open(job_path, encoding="utf-8").read()
job_ids = post(tok_port, "/tokenize", {"content": job_text, "add_special": False, "parse_special": True})["tokens"]
shared_ids = json.load(open(shared_path))
prompt = shared_ids + job_ids
t0 = time.perf_counter()
d = post(c_port, "/completion", {"prompt": prompt, "n_predict": int(npred), "temperature": 0,
                                  "cache_prompt": True, "id_slot": int(slot), "return_tokens": False})
wall = time.perf_counter() - t0
t = d.get("timings", {})
hit = t.get("cache_n") == int(n_shared)
print(f"slot {slot}: wall {wall:.2f}s  cache_n={t.get('cache_n')} ({'HIT' if hit else 'MISS - see header comment'})  "
      f"prompt_n={t.get('prompt_n')}  decode {t.get('predicted_n')}@{t.get('predicted_per_second',0):.0f}tok/s")
print("  text:", d.get("content", "")[:200].replace("\n", " "))
PY
    ;;

  status)
    [ -f "$PIDF" ] && while IFS== read -r who p; do printf '   %-6s pid %-6s %s\n' "$who" "$p" "$(kill -0 "$p" 2>/dev/null && echo alive || echo DEAD)"; done < "$PIDF" || echo "   not running"
    sh tools/gpu-lock.sh status ;;

  stop)
    if [ -f "$PIDF" ]; then
      for who in metal card; do p=$(sed -n "s/^$who=//p" "$PIDF"); [ -n "$p" ] || continue
        if kill -0 "$p" 2>/dev/null; then kill -TERM "$p"; i=0; while kill -0 "$p" 2>/dev/null && [ $i -lt 120 ]; do i=$((i+1)); sleep 0.5; done
          kill -0 "$p" 2>/dev/null && { echo "   $who still alive after 60s, KILL"; kill -KILL "$p"; }; echo "   stopped $who (pid $p)"; fi; done
      rm -f "$PIDF" "$META" "$ST/shared-ids.json"
    fi
    if grep -q "session=V1 " "${GPU_LOCK:-$R/.gpu-lock}" 2>/dev/null; then sh tools/gpu-lock.sh release V1 >/dev/null; echo "   card left idle warm, lock released"; fi ;;

  *) echo "usage: $0 start <model> <shared-prompt-file> <n-slots> [ncpumoe] | card-only ... | job <slot> <text|file> [n_predict] | status | stop"; exit 2 ;;
esac
