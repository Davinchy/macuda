# Models

Not in the repository. Every number in the top-level README was measured with these GGUF files in this directory:

| file | used for |
|---|---|
| `Qwen3.8-27B-UD-Q4_K_M.gguf` | the dense decode number, the greedy byte-compare text, serving (unsloth's UD quant) |
| `MTP/mtp-Qwen3.8-27B-Q4_0.gguf` | its MTP draft head for speculative decode (`--spec-type draft-mtp -md <this> -ngld 99`; unsloth ships it beside the model) |
| `Qwen3.5-35B-A3B-Q4_K_M.gguf` | the MoE decode number |
| `Meta-Llama-3.1-8B-Instruct-Q8_0.gguf` | the dense Q8 reference row |
| `sd/` | stable-diffusion.cpp models: SDXL-Turbo, SD 1.5 (+ LCM LoRA), Z-Image-Turbo (Q8 DiT + Qwen3-4B encoder + FLUX VAE) |

`tools/serve.sh` and `tools/soak.sh` look for the 27B and its MTP head here by these names; `tools/nv_shim_step.sh bench|simple|spec|sd`
take any path.
