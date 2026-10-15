#!/bin/sh
# drive-batched.sh — batched-serving reference on the Windows box (the same 5090 in the same enclosure, NVIDIA's driver),
# driven from the Mac with the SAME load generator, prompts, token count and sampler as the Mac runs (tools/loadgen.py),
# so the aggregate tok/s compare directly with docs/SHARED-STATUS.md 2026-09-15 02:05.
#   sh tools/windows/drive-batched.sh llama            llama-server.exe b10970: 27B p1+MTP, 27B p8, 27B p16, MoE p1, MoE p8
#   sh tools/windows/drive-batched.sh vllm             vLLM in WSL docker (the user's own image + args): 27B NVFP4 and the 35B-A3B GPTQ, c=1/8/16/32
#   sh tools/windows/drive-batched.sh llama-one <gguf> <parallel> <ctx> <mtp 0|1> <concurrency> <requests>
# The Windows server runs in the FOREGROUND of an ssh session held open from here (Windows OpenSSH kills a session's
# children when it ends, which is exactly the teardown wanted); the API is reached through that session's -L tunnel.
K=${WIN_KEY:-}; H=${WIN_HOST:?set WIN_HOST=user@windows-box}; W=${WIN_DIR:?set WIN_DIR to the staging directory on the Windows box, e.g. C:\\Users\\you\\egpu-reference}; R=${EGPU_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
OUT=$R/bench/windows-batched-$(date +%Y%m%d).md; mkdir -p $R/bench $R/logs
PORT=8099; TUN=18099; VPORT=8000; VURL=http://${WIN_HOST#*@}:8000
ssh_q() { ssh ${K:+-i $K} -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 $H "$1" 2>&1 | tr -d '\r'; }
health() { for i in $(seq 1 "$2"); do [ "$(curl -s -o /dev/null -w '%{http_code}' "$1" 2>/dev/null)" = 200 ] && return 0; sleep 2; done; return 1; }
note() { echo "$1" | tee -a "$OUT"; }
llama_one() { # gguf parallel ctx mtp concurrency requests
  m=$1; par=$2; ctx=$3; mtp=$4; conc=$5; reqs=$6; spec=""
  [ "$mtp" = 1 ] && spec="-md '$W\\models\\mtp-Qwen3.8-27B-Q4_0.gguf' -ngld 99 --spec-type draft-mtp --spec-draft-n-max 4"
  ts=$(date +%Y%m%d-%H%M%S); log=$R/logs/win-llama-$ts.log
  ssh ${K:+-i $K} -o BatchMode=yes -o ServerAliveInterval=15 -o ExitOnForwardFailure=yes -L $TUN:127.0.0.1:$PORT $H \
    "& '$W\\llama.cpp\\llama-server.exe' -m '$W\\models\\$m' -ngl 99 $spec -c $ctx --host 127.0.0.1 --port $PORT --parallel $par --jinja --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0" > "$log" 2>&1 &
  pid=$!
  if ! health "http://127.0.0.1:$TUN/health" 150; then note "FAILED to come up: $m parallel $par (see $log)"; kill $pid 2>/dev/null; ssh_q 'Stop-Process -Name llama-server -Force -ErrorAction SilentlyContinue' >/dev/null; return 1; fi
  note "## llama-server b10970 (Windows, driver $(ssh_q 'nvidia-smi --query-gpu=driver_version --format=csv,noheader' | head -1)): $m parallel=$par ctx=$ctx mtp=$mtp"
  LOADGEN_URL=http://127.0.0.1:$TUN python3 $R/tools/loadgen.py "$conc" "$reqs" 256 0.6 | tee -a "$OUT"
  kill $pid 2>/dev/null; sleep 2; ssh_q 'Stop-Process -Name llama-server -Force -ErrorAction SilentlyContinue' >/dev/null; sleep 3
}
vllm_one() { # model-dir served-name max-num-seqs max-model-len extra-args concurrency requests
  m=$1; name=$2; seqs=$3; mlen=$4; extra=$5; conc=$6; reqs=$7
  ts=$(date +%Y%m%d-%H%M%S); log=$R/logs/win-vllm-$ts.log
  printf 'docker rm -f bench-vllm >/dev/null 2>&1; docker run -d --name bench-vllm --gpus all --ipc=host --network host -v /root/models:/models vllm/vllm-openai:latest --model /models/%s --served-model-name %s --host 0.0.0.0 --port %s --gpu-memory-utilization 0.90 --max-model-len %s --max-num-seqs %s --trust-remote-code %s\n' "$m" "$name" "$VPORT" "$mlen" "$seqs" "$extra" \
    | ssh ${K:+-i $K} -o BatchMode=yes -o ConnectTimeout=10 $H 'wsl -d Ubuntu-24.04 -- bash -s' > "$log" 2>&1
  if ! health "$VURL/health" 900; then note "FAILED to come up: vLLM $m (see $log)"; printf 'docker inspect bench-vllm --format "oom={{.State.OOMKilled}} exit={{.State.ExitCode}} started={{.State.StartedAt}} finished={{.State.FinishedAt}}"; docker logs bench-vllm --tail 200 2>&1; docker rm -f bench-vllm\n' | ssh ${K:+-i $K} -o BatchMode=yes $H 'wsl -d Ubuntu-24.04 -- bash -s' >> "$log" 2>&1; return 1; fi
  note "## vLLM (WSL docker vllm/vllm-openai:latest): $m max-num-seqs=$seqs max-model-len=$mlen"
  LOADGEN_URL=$VURL LOADGEN_MODEL=$name python3 $R/tools/loadgen.py "$conc" "$reqs" 256 0.6 | tee -a "$OUT"
}
vllm_down() { printf 'docker rm -f bench-vllm >/dev/null 2>&1; echo down\n' | ssh ${K:+-i $K} -o BatchMode=yes $H 'wsl -d Ubuntu-24.04 -- bash -s' 2>&1 | tr -d '\r' | tail -1; }
case "${1:-}" in
  llama)
    note "# Windows batched reference, llama.cpp, $(date '+%F %T') — loadgen: 256 tokens, temp 0.6, eight rotating prompts, N threads x 3 requests"
    llama_one Qwen3.8-27B-UD-Q4_K_M.gguf 1 8192 1 1 6
    llama_one Qwen3.8-27B-UD-Q4_K_M.gguf 8 32768 0 8 3
    llama_one Qwen3.8-27B-UD-Q4_K_M.gguf 16 65536 0 16 3
    llama_one Qwen3.5-35B-A3B-Q4_K_M.gguf 1 8192 0 1 4
    llama_one Qwen3.5-35B-A3B-Q4_K_M.gguf 8 32768 0 8 3 ;;
  llama-one) shift; llama_one "$@" ;;
  vllm)
    note "# Windows batched reference, vLLM in WSL, $(date '+%F %T') — same loadgen"
    for c in 1 8 16 32; do vllm_one Qwen3.8-27B-NVFP4-RTX5090 qwen3.8-27b 32 16384 "--reasoning-parser qwen3" $c $([ $c = 1 ] && echo 6 || echo 3) || break; done; vllm_down
    for c in 1 8 24; do vllm_one Qwen3.6-35B-A3B-GPTQ-Int4 qwen3.6-35b 24 12288 "" $c $([ $c = 1 ] && echo 4 || echo 3) || break; done; vllm_down ;;
  vllm-27b) note "# Windows vLLM 27B NVFP4, $(date '+%F %T')"; for c in 1 8 16 32; do vllm_one Qwen3.8-27B-NVFP4-RTX5090 qwen3.8-27b 32 16384 "--reasoning-parser qwen3" $c $([ $c = 1 ] && echo 6 || echo 3) || break; done; vllm_down ;;
  vllm-moe) note "# Windows vLLM Qwen3.6-35B-A3B GPTQ-Int4 (the production config), $(date '+%F %T')"; for c in 1 8 24; do vllm_one Qwen3.6-35B-A3B-GPTQ-Int4 qwen3.6-35b 24 12288 "" $c $([ $c = 1 ] && echo 4 || echo 3) || break; done; vllm_down ;;
  vllm-down) vllm_down ;;
  *) sed -n '2,9p' "$0"; exit 2 ;;
esac
