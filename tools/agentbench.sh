#!/bin/sh
# agentbench.sh — llama-bench at the depths a coding agent actually runs at, one JSON per hardware config.
#
#   sh tools/agentbench.sh <label> <llama-bench binary> <model.gguf> [out.json]
#
# Prompt processing: -p 2048 at depths 0, 16384, 32768, 65536. Generation: -n 256 at the same depths. Combined: -pg
# 16384,512. Five repetitions, JSON out, full offload and flash attention - every flag identical across configs, which
# is the whole point. DEPTHS= overrides the depth list when a config cannot hold the biggest one. The JSON is what the
# report is built from; the build commit is in it (llama-bench writes build_commit), and the environment the run was
# made in is written beside it as <out>.env.txt: the driver's build id for the card, the link the card is on, macOS.
set -eu
label=${1:?label}; bin=${2:?llama-bench binary}; model=${3:?model}; out=${4:-logs/agentbench-$label-$(date +%Y%m%d-%H%M%S).json}
R=${EGPU_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
depths=${DEPTHS:-0,16384,32768,65536}
mkdir -p "$(dirname "$out")"
{
  echo "label=$label"; echo "date=$(date -Iseconds)"; echo "model=$model"; echo "binary=$bin"
  echo "macos=$(sw_vers -productVersion) $(uname -m)"; echo "depths=$depths"
  echo "driver_build_id=$(strings "$bin" | grep -xE '[0-9a-f]{7}(-dirty)?' | grep -v -e 59c23bc -e b906d25 | head -1)"
  system_profiler SPPCIDataType 2>/dev/null | awk '/Vendor ID: 0x10de/{nv=1} nv&&/Link Width|Link Speed/{print "pcie_"tolower($1)"_"tolower($2)"="$3" "$4} nv&&/Link Status/{exit}'
} > "$out.env.txt"
# one llama-bench invocation: the tool crosses -p/-n with every -d and adds the -pg rows
"$bin" -m "$model" -ngl "${NGL:-99}" -fa 1 -p 2048 -n 256 -d "$depths" -pg 16384,512 -r 5 -o json ${BENCH_ARGS:-} > "$out"
echo "wrote $out ($(python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))), 'rows')" "$out"))"
