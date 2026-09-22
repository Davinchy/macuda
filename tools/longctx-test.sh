#!/bin/sh
# longctx-test.sh — start a llama-server, run the long-context test against it, stop it. One process from the card's
# point of view, so nv_shim_step.sh runs it as a step (`probe` takes any executable):
#
#   sh tools/longctx-test.sh <label> <llama-server binary> <model.gguf> <ctx> <budget> [port]
#   QUIESCE=1 sh tools/nv_shim_step.sh A probe tools/longctx-test.sh lfm-3060 cuda-shim/build/bin/llama-server-null models/LFM2.5-8B-A1B-Q8_0.gguf 131072 115000
#
# The cache is q8_0 on both sides (-ctk/-ctv), flash attention on, full offload unless NGL says otherwise; SERVER_ARGS adds
# (e.g. --n-cpu-moe 40 to keep every expert on the CPU). The link the run was made on is sampled from the host.
set -eu
label=${1:?label}; bin=${2:?llama-server binary}; model=${3:?model}; ctx=${4:?ctx}; budget=${5:?token budget}; port=${6:-8099}
R=${EGPU_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
ts=$(date +%Y%m%d-%H%M%S); out=$R/logs/longctx-$label-$ts.json; slog=$R/logs/longctx-$label-$ts.server.log
mkdir -p "$R/logs"
"$bin" -m "$model" -ngl "${NGL:-99}" -fa 1 -c "$ctx" -ctk q8_0 -ctv q8_0 -b 2048 -ub 512 -np 1 --port "$port" --host 127.0.0.1 ${SERVER_ARGS:-} > "$slog" 2>&1 &
spid=$!
trap 'kill $spid 2>/dev/null; wait $spid 2>/dev/null' EXIT INT TERM
i=0; until curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; do i=$((i+1)); [ $i -gt 1200 ] && { echo "server did not come up; see $slog"; exit 1; }; kill -0 $spid 2>/dev/null || { echo "server died; see $slog"; tail -5 "$slog"; exit 1; }; sleep 1; done
echo "server up after $i s ($label, ctx $ctx)"
link_samples="$out.link.txt"; : > "$link_samples"
( while :; do system_profiler SPPCIDataType 2>/dev/null | awk '/Vendor ID: 0x10de/{nv=1} nv&&/Link Width/{w=$3} nv&&/Link Speed/{print w, $3, $4; exit}' >> "$link_samples"; sleep 10; done ) & sampler=$!
python3 "$R/tools/longctx-test.py" --url "http://127.0.0.1:$port" --label "$label" --files tools/longctx-files.txt --budget "$budget" --out "$out"
kill $sampler 2>/dev/null; wait $sampler 2>/dev/null || true
{ echo "label=$label"; echo "date=$(date -Iseconds)"; echo "model=$model"; echo "binary=$bin"; echo "ctx=$ctx"; echo "budget=$budget"; echo "server_up_s=$i"; echo "server_args=${SERVER_ARGS:-}"
  if [ -s "$link_samples" ]; then
    echo "pcie_link_width=$(sort "$link_samples" | uniq -c | sort -rn | head -1 | awk '{print $2}')"; echo "pcie_link_speed=$(sort "$link_samples" | uniq -c | sort -rn | head -1 | awk '{print $3" "$4}')"
    echo "pcie_link_seen=$(sort "$link_samples" | uniq -c | awk '{printf "%s%s at %s %s (%d samples)", (NR>1?"; ":""), $2, $3, $4, $1}')"
  fi; } > "$out.env.txt"
rm -f "$link_samples"
echo "wrote $out"
