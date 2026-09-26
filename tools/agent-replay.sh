#!/bin/sh
# agent-replay.sh — start a llama-server, replay the coding-agent conversation against it, stop it. One process from the
# card's point of view, which is what lets nv_shim_step.sh run it as a step (`probe` takes any executable):
#
#   sh tools/agent-replay.sh <label> <llama-server binary> <model.gguf> <ctx> [port]
#   sh tools/nv_shim_step.sh A probe tools/agent-replay.sh 3060 cuda-shim/build/bin/llama-server-null models/X.gguf 65536
#
# cache_prompt is on (the server's default) and the context is the caller's; -fa 1 and -ngl 99 as the sweep uses.
# The per-turn table and total go to logs/replay-<label>-<ts>.json and .log.
set -eu
label=${1:?label}; bin=${2:?llama-server binary}; model=${3:?model}; ctx=${4:?ctx}; port=${5:-8097}
R=${EGPU_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
ts=$(date +%Y%m%d-%H%M%S); out=$R/logs/replay-$label-$ts.json; slog=$R/logs/replay-$label-$ts.server.log
mkdir -p "$R/logs"
"$bin" -m "$model" -ngl "${NGL:-99}" -fa 1 -c "$ctx" -np 1 --port "$port" --host 127.0.0.1 ${SERVER_ARGS:-} > "$slog" 2>&1 &
spid=$!
trap 'kill $spid 2>/dev/null; wait $spid 2>/dev/null' EXIT INT TERM
i=0; until curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; do i=$((i+1)); [ $i -gt 600 ] && { echo "server did not come up; see $slog"; exit 1; }; kill -0 $spid 2>/dev/null || { echo "server died; see $slog"; tail -5 "$slog"; exit 1; }; sleep 1; done
echo "server up after $i s ($label, ctx $ctx)"
# The link the replay was made on, sampled from the host every 5 s while the server runs (the firmware brings the link
# to the driver's cap a few seconds into the boot, so the link before the run says nothing), written beside the JSON
# as <out>.env.txt in the sweep's format so the report reads both the same way.
link_samples="$out.link.txt"; : > "$link_samples"
( while :; do system_profiler SPPCIDataType 2>/dev/null | awk '/Vendor ID: 0x10de/{nv=1} nv&&/Link Width/{w=$3} nv&&/Link Speed/{print w, $3, $4; exit}' >> "$link_samples"; sleep 5; done ) & sampler=$!
python3 "$R/tools/agent-replay.py" --url "http://127.0.0.1:$port" --label "$label" --out "$out" ${REPLAY_ARGS:-}
kill $sampler 2>/dev/null; wait $sampler 2>/dev/null || true
{ echo "label=$label"; echo "date=$(date -Iseconds)"; echo "model=$model"; echo "binary=$bin"; echo "ctx=$ctx"; echo "server_up_s=$i"
  if [ -s "$link_samples" ]; then
    echo "pcie_link_width=$(sort "$link_samples" | uniq -c | sort -rn | head -1 | awk '{print $2}')"; echo "pcie_link_speed=$(sort "$link_samples" | uniq -c | sort -rn | head -1 | awk '{print $3" "$4}')"
    echo "pcie_link_seen=$(sort "$link_samples" | uniq -c | awk '{printf "%s%s at %s %s (%d samples)", (NR>1?"; ":""), $2, $3, $4, $1}')"
  fi; } > "$out.env.txt"
rm -f "$link_samples"
echo "wrote $out"
