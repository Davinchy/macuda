# Windows batched reference, llama.cpp, 2026-09-15 02:05:25 — loadgen: 256 tokens, temp 0.6, eight rotating prompts, N threads x 3 requests
## llama-server b10970 (Windows, driver 595.79): Qwen3.8-27B-UD-Q4_K_M.gguf parallel=1 ctx=8192 mtp=1
concurrency 1 x 6 requests, max_tokens 256, temp 0.6: 6/6 ok, 1281 tokens in 14.8 s = AGGREGATE 86.8 tok/s; per-request mean 113.0 tok/s (min 99.5, max 128.7); draft accept 45%
## llama-server b10970 (Windows, driver 595.79): Qwen3.8-27B-UD-Q4_K_M.gguf parallel=8 ctx=32768 mtp=0
concurrency 8 x 3 requests, max_tokens 256, temp 0.6: 24/24 ok, 4706 tokens in 32.3 s = AGGREGATE 145.8 tok/s; per-request mean 24.2 tok/s (min 19.4, max 37.6); draft accept 0%
## llama-server b10970 (Windows, driver 595.79): Qwen3.8-27B-UD-Q4_K_M.gguf parallel=16 ctx=65536 mtp=0
concurrency 16 x 3 requests, max_tokens 256, temp 0.6: 48/48 ok, 9619 tokens in 67.6 s = AGGREGATE 142.3 tok/s; per-request mean 13.4 tok/s (min 8.3, max 25.5); draft accept 0%
## llama-server b10970 (Windows, driver 595.79): Qwen3.5-35B-A3B-Q4_K_M.gguf parallel=1 ctx=8192 mtp=0
concurrency 1 x 4 requests, max_tokens 256, temp 0.6: 4/4 ok, 1024 tokens in 9.7 s = AGGREGATE 105.3 tok/s; per-request mean 215.3 tok/s (min 213.6, max 217.6); draft accept 0%
## llama-server b10970 (Windows, driver 595.79): Qwen3.5-35B-A3B-Q4_K_M.gguf parallel=8 ctx=32768 mtp=0
concurrency 8 x 3 requests, max_tokens 256, temp 0.6: 24/24 ok, 6144 tokens in 22.7 s = AGGREGATE 270.2 tok/s; per-request mean 54.0 tok/s (min 34.8, max 63.7); draft accept 0%
# Windows batched reference, vLLM in WSL, 2026-09-15 02:09:40 — same loadgen
FAILED to come up: vLLM Qwen3.8-27B-NVFP4-RTX5090 (see /Volumes/512SSD/EGPU/logs/win-vllm-20260915-020940.log)
# Windows vLLM Qwen3.6-35B-A3B GPTQ-Int4 (the production config), 2026-09-15 02:31:13
## vLLM 0.23.0 docker (the user's vllm-q38 container): Qwen3.8-27B-NVFP4-RTX5090, max-num-seqs 8, max-model-len 16384, gpu-mem 0.90, 2026-09-15 03:33:27
concurrency 1 x 6 requests, max_tokens 256, temp 0.6: 6/6 ok, 1317 tokens in 19.4 s = AGGREGATE 67.8 tok/s; per-request mean 70.5 tok/s (min 47.8, max 77.2); draft accept 0%
concurrency 8 x 3 requests, max_tokens 256, temp 0.6: 24/24 ok, 4704 tokens in 17.5 s = AGGREGATE 269.6 tok/s; per-request mean 37.0 tok/s (min 18.6, max 64.5); draft accept 0%
concurrency 16 x 3 requests, max_tokens 256, temp 0.6: 48/48 ok, 9607 tokens in 31.1 s = AGGREGATE 309.2 tok/s; per-request mean 21.9 tok/s (min 8.4, max 33.8); draft accept 0%
