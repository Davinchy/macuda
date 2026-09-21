#!/bin/sh
# agentbench.sh — llama-bench at the depths a coding agent actually runs at, one JSON per hardware config.
#
#   sh tools/agentbench.sh <label> <llama-bench binary> <model.gguf> [out.json]
#
# Prompt processing: -p 2048 at depths 0, 16384, 32768, 65536. Generation: -n 256 at the same depths. Combined: -pg
# 16384,512. Five repetitions, JSON out, full offload and flash attention - every flag identical across configs, which
# is the whole point. DEPTHS= overrides the depth list when a config cannot hold the biggest one. The JSON is what the
# report is built from; the build commit is in it (llama-bench writes build_commit), and the environment the run was
# made in is written beside it as <out>.env.txt: the driver's build id for the card, the link the run was made on
# (sampled from the host while llama-bench runs; the link before the run is recorded too), macOS.
set -eu
label=${1:?label}; bin=${2:?llama-bench binary}; model=${3:?model}; out=${4:-logs/agentbench-$label-$(date +%Y%m%d-%H%M%S).json}
R=${EGPU_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
depths=${DEPTHS:-0,16384,32768,65536}
mkdir -p "$(dirname "$out")"
{
  echo "label=$label"; echo "date=$(date -Iseconds)"; echo "model=$model"; echo "binary=$bin"
  echo "macos=$(sw_vers -productVersion) $(uname -m)"; echo "depths=$depths"
  # the driver's build id only means something in a binary that carries the driver; the strings of a Metal build have none
  # the id is the string build_id.c carries, and the archive the binary was linked from carries the same one alone
  if strings "$bin" | grep -q 'libtinynv build'; then echo "driver_build_id=$(strings "$R/cuda-shim/build/shim/nv/libtinynv.a" 2>/dev/null | grep -oE '^[0-9a-f]{7}(-dirty)?$|^nogit(-dirty)?$' | head -1)"; else echo "driver_build_id=none (not a driver build)"; fi
  system_profiler SPPCIDataType 2>/dev/null | awk '/Vendor ID: 0x10de/{nv=1} nv&&/Link Width|Link Speed/{k=tolower($1"_"$2); sub(/:$/, "", k); print "pcie_"k"_before="$3" "$4} nv&&/Link Status/{exit}'
} > "$out.env.txt"
# The link the card is on BEFORE the run is not the link the run is made on: the firmware idles the link down to Gen1
# after every unload and brings it to the driver's cap (Gen3 on this enclosure) a few seconds into the next boot. So
# the link is sampled from the host every 5 s while llama-bench runs, and the width and speed the run settled on -
# the last sample - are what the report reads as pcie_link_width / pcie_link_speed, with every distinct sample listed.
link_samples="$out.link.txt"; : > "$link_samples"
( while :; do sleep 5; system_profiler SPPCIDataType 2>/dev/null | awk '/Vendor ID: 0x10de/{nv=1} nv&&/Link Width/{w=$3} nv&&/Link Speed/{print w, $3, $4; exit}' >> "$link_samples"; done ) & sampler=$!
# one llama-bench invocation: the tool crosses -p/-n with every -d and adds the -pg rows
"$bin" -m "$model" -ngl "${NGL:-99}" -fa 1 -p 2048 -n 256 -d "$depths" -pg 16384,512 -r 5 -o json ${BENCH_ARGS:-} > "$out"
kill $sampler 2>/dev/null; wait $sampler 2>/dev/null || true
if [ -s "$link_samples" ]; then
  { echo "pcie_link_width=$(tail -1 "$link_samples" | awk '{print $1}')"; echo "pcie_link_speed=$(tail -1 "$link_samples" | awk '{print $2" "$3}')"
    echo "pcie_link_seen=$(sort "$link_samples" | uniq -c | awk '{printf "%s%s at %s %s (%d samples)", (NR>1?"; ":""), $2, $3, $4, $1}')"; } >> "$out.env.txt"
fi
rm -f "$link_samples"
echo "wrote $out ($(python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))), 'rows')" "$out"))"
