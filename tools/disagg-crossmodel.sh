#!/bin/sh
# disagg-crossmodel.sh — the cross-model disaggregation sweep (2026-09-19). Four MoE models spanning the one variable
# the page at docs/bench/disaggregation-draft-20260919.html says is the lever - ACTIVE parameters - at roughly fixed
# total size, each prefilled on both halves at the same token counts:
#
#   Qwen3-Coder-Next UD-Q4_K_XL  49.6 GB    3B active   the anchor: every constant in the cost model was fitted on it
#   gpt-oss-120b MXFP4           63.4 GB  5.1B active
#   Qwen3.5-122B-A10B Q4_K_M     69.1 GB   10B active
#   DeepSeek-V4-Flash UD-Q2_K_XL 96.8 GB  ~10B active   MLA: ~1 KB of KV a token, where Mixtral pays 229
#   Llama-4-Scout Q4_K_M         65.4 GB   17B active
#   Mixtral-8x22B Q4_K_S         80.5 GB   39B active
#
# Coder-Next is in the set to be re-measured, not for a new number: if this rig reproduces the 2.35x at 24k and
# 5.55x at 48k that were measured on it directly on 2026-09-19, the other five rows can be believed.
#
# Every model runs with EVERY expert in host memory (--n-cpu-moe n_layer). That is the configuration the cost model's
# constants were fitted on, and the only one identical across models: tuning residency per model would make each
# model's stream a different fraction of itself and the comparison would measure the tuning, not the model.
#
#   sh tools/disagg-crossmodel.sh [token counts...]      default: 12000 24000 26000 48000
#
# 24000 and 26000 straddle the 24,576-token ubatch boundary, which is where the expert stream is paid a second time;
# 12000 and 48000 give the context curve two more points. Rows land in $OUT as TSV, one per (model, side, length).
set -u
R=${EGPU_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
M=${MODELS_DIR:-/Volumes/Crucial_8TB/Models}
B=${BENCH_DIR:-/Volumes/Crucial_8TB/disagg-bench}
OUT=${OUT:-$B/results.tsv}; export OUT STATE=${STATE:-$B/state}
TOKENS=${*:-12000 24000 26000 48000}
mkdir -p "$B/logs" "$STATE"
ts=$(date +%Y%m%d-%H%M%S); LOG=$B/logs/crossmodel-$ts.log
SIDES=${SIDES:-card metal}

models="coder-next-80b-a3b|$M/Qwen3-Coder-Next-GGUF/Qwen3-Coder-Next-UD-Q4_K_XL.gguf
gpt-oss-120b|$M/gpt-oss-120b-GGUF/gpt-oss-120b-MXFP4.gguf
qwen3.5-122b-a10b|$M/Qwen3.5-122B-A10B-GGUF/Qwen3.5-122B-A10B-Q4_K_M-00001-of-00002.gguf
llama4-scout-109b-a17b|$M/Llama-4-Scout-17B-16E-Instruct-GGUF/Llama-4-Scout-17B-16E-Instruct-Q4_K_M-00001-of-00002.gguf
deepseek-v4-flash-q2|$M/DeepSeek-V4-Flash-0731-GGUF/DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00001-of-00003.gguf
mixtral-8x22b-a39b|$M/Mixtral-8x22B-Instruct-v0.1-GGUF/Mixtral-8x22B-Instruct-v0.1.Q4_K_S-00001-of-00002.gguf"

echo "disagg-crossmodel: $(date '+%F %T') tokens: $TOKENS; results $OUT; log $LOG" | tee -a "$LOG"
echo "$models" | while IFS='|' read -r tag model; do
  [ -f "$model" ] || { echo "== $tag: NO FILE at $model - skipped" | tee -a "$LOG"; continue; }
  # --n-cpu-moe wants a layer count; read it out of the header rather than carrying a table that can go stale
  nl=$(python3 -c "
import importlib.util, sys
spec = importlib.util.spec_from_file_location('s', '$R/tools/disagg-screen.py'); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
kv = m.read_kv('$model'); a = kv.get('general.architecture', '?')
print(kv.get(a + '.block_count', 99))" 2>/dev/null) || nl=99
  echo "" | tee -a "$LOG"
  echo "===== $tag  $(basename "$model")  n_layer=$nl  $(date '+%F %T')" | tee -a "$LOG"
  for side in $SIDES; do
    if [ "$side" = card ]; then
      MODEL=$model TAG=$tag NCPUMOE=$nl sh "$R/tools/nv_shim_step.sh" V1 probe "$R/tools/disagg-sweep.sh" card $TOKENS >> "$LOG" 2>&1 \
        || echo "   card sweep FAILED for $tag (exit $?) - continuing" | tee -a "$LOG"
    else
      MODEL=$model TAG=$tag sh "$R/tools/disagg-sweep.sh" metal $TOKENS >> "$LOG" 2>&1 \
        || echo "   metal sweep FAILED for $tag (exit $?) - continuing" | tee -a "$LOG"
    fi
    tail -8 "$LOG" | sed 's/^/     /'
  done
done
echo "" | tee -a "$LOG"; echo "disagg-crossmodel: done $(date '+%F %T')" | tee -a "$LOG"
column -t -s "$(printf '\t')" "$OUT" 2>/dev/null | tail -40 | tee -a "$LOG"
