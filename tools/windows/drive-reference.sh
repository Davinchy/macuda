#!/bin/sh
# drive-reference.sh <llm-small|llm-big|mtp|sd> — run the Windows reference benchmarks over ssh one at a time (Windows OpenSSH
# kills detached children when the session ends, so the session must stay open for each run) and append to a local report.
K=${WIN_KEY:-}; H=${WIN_HOST:?set WIN_HOST=user@windows-box}; W=${WIN_DIR:?set WIN_DIR to the staging directory on the Windows box, e.g. C:\\Users\\you\\egpu-reference}
R=${EGPU_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}; OUT=$R/bench/windows-reference-$(date +%Y%m%d).md; mkdir -p $R/bench
run() { ssh ${K:+-i $K} -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 $H "$1" 2>&1 | tr -d '\r'; }
[ -f "$OUT" ] || { echo "# Windows reference run $(date '+%F %T') — RTX 5090 in the AORUS box on the Windows PC, real NVIDIA driver" > $OUT
  echo '```' >> $OUT; run 'nvidia-smi --query-gpu=name,driver_version,pcie.link.gen.current,pcie.link.width.current,clocks.max.sm,memory.total --format=csv' >> $OUT; run "Get-Content $W\\..\\egpu-reference\\prepared.txt" >> $OUT; echo '```' >> $OUT; }
case "$1" in
  llm-small) list="gemma-4-12B-it-Q4_K_M.gguf nvidia_NVIDIA-Nemotron-Nano-9B-v2-Q6_K.gguf Meta-Llama-3.1-8B-Instruct-Q8_0.gguf" ;;
  llm-big)   list="Qwen3.8-27B-UD-Q4_K_M.gguf gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf Qwen3.5-35B-A3B-Q4_K_M.gguf" ;;
  mtp)       list="" ;;
  sd)        list="" ;;
  *) echo "usage: $0 llm-small|llm-big|mtp|sd"; exit 2 ;;
esac
for m in $list; do
  echo "## llama-bench $m (-ngl 99 -p 256 -n 128 -r 3)" >> $OUT
  run "& $W\\llama.cpp\\llama-bench.exe -m $W\\models\\$m -ngl 99 -p 256 -n 128 -r 3 -o md 2>\$null" | grep -E '^\|' >> $OUT
  grep -E '\| *(pp|tg)[0-9]+ *\|' $OUT | tail -2
done
if [ "$1" = mtp ]; then
  echo "## MTP: llama-speculative-simple, draft-mtp depth 4, greedy, n=256, same prompt as the Mac" >> $OUT; echo '```' >> $OUT
  run "& $W\\llama.cpp\\llama-speculative-simple.exe -m $W\\models\\Qwen3.8-27B-UD-Q4_K_M.gguf -md $W\\models\\mtp-Qwen3.8-27B-Q4_0.gguf -ngl 99 -ngld 99 --spec-type draft-mtp --spec-draft-n-max 4 -n 256 --temp 0 -p 'The three most important things to know about the Thunderbolt bus are' 2>&1 | Select-String -Pattern 'decoded|n_accept|accept |n_drafted' | ForEach-Object { \$_.Line }" >> $OUT
  echo '```' >> $OUT; tail -5 $OUT
fi
if [ "$1" = sd ]; then
  P='a photograph of a red fox sitting on a mossy log in a sunlit forest, detailed fur'
  echo "## image generation (sd-cli, same seeds/steps/prompt as the Mac)" >> $OUT; echo '```' >> $OUT
  for spec in "SDXL-Turbo 4 steps 512|-m $W\\models\\sd\\sd_xl_turbo_1.0_fp16.safetensors --steps 4 --cfg-scale 1.0 --sampling-method euler -W 512 -H 512" \
              "SD 1.5 20 steps 512|-m $W\\models\\sd\\v1-5-pruned-emaonly.safetensors --steps 20 -W 512 -H 512" \
              "Z-Image-Turbo 8 steps 1024|--diffusion-model $W\\models\\sd\\z_image_turbo-Q8_0.gguf --vae $W\\models\\sd\\flux-ae.safetensors --llm $W\\models\\sd\\Qwen3-4B-Instruct-2507-Q8_0.gguf --steps 8 --cfg-scale 1.0 --sampling-method euler -W 1024 -H 1024"; do
    name=${spec%%|*}; args=${spec#*|}; png="$W\\results\\$(echo $name | tr ' ' '_').png"
    lines=$(run "& $W\\sd.cpp\\sd-cli.exe $args -p '$P' -s 42 -o $png 2>&1 | Select-String -Pattern 'sampling completed|decode_first_stage|generate_image completed' | ForEach-Object { \$_.Line }" | sed -E 's/.*- //' | tr '\n' ' ')
    echo "$name: $lines" | tee -a $OUT
  done
  echo '```' >> $OUT
fi
