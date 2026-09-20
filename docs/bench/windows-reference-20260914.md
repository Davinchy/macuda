# Windows reference run 2026-09-14 21:11:26 — RTX 5090 in the AORUS box on the Windows PC, real NVIDIA driver

**The configuration this was taken in, stated because it has already been mis-remembered once (2026-09-19: a peer
session recorded it as an x16 slot capture and concluded a fresh reference would not be comparable).** The card was
in the AORUS enclosure, on the Windows PC, over Thunderbolt - not in a PCIe slot. The nvidia-smi line below records
it: `pcie.link.gen.current` 4 and `pcie.link.width.current` 4, i.e. gen 4 x4, which is the enclosure's link and not
a slot's x16. That is the same physical path the Mac numbers are taken over, so these ratios isolate the driver and
the host, not the link. A fresh capture in the enclosure IS comparable to this file.
```
name, driver_version, pcie.link.gen.current, pcie.link.width.current, clocks.max.sm [MHz], memory.total [MiB]
NVIDIA GeForce RTX 5090, 595.79, 4, 4, 3090 MHz, 32607 MiB
prepared: llama.cpp b10970 (llama-b10970-bin-win-cuda-12.4-x64.zip), stable-diffusion.cpp master-869-07a85c7 (sd-master-07a85c7-bin-win-cuda12-x64.zip)
```
## llama-bench gemma-4-12B-it-Q4_K_M.gguf (-ngl 99 -p 256 -n 128 -r 3)
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| gemma4 ?B Q4_K - Medium        |   6.86 GiB |    11.91 B | CUDA       |  99 |           pp256 |    4168.86 ± 2771.20 |
| gemma4 ?B Q4_K - Medium        |   6.86 GiB |    11.91 B | CUDA       |  99 |           tg128 |        136.35 ± 2.71 |
## llama-bench nvidia_NVIDIA-Nemotron-Nano-9B-v2-Q6_K.gguf (-ngl 99 -p 256 -n 128 -r 3)
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| nemotron_h 9B Q6_K             |   8.50 GiB |     8.89 B | CUDA       |  99 |           pp256 |     4030.81 ± 383.78 |
| nemotron_h 9B Q6_K             |   8.50 GiB |     8.89 B | CUDA       |  99 |           tg128 |        140.61 ± 0.54 |
## llama-bench Meta-Llama-3.1-8B-Instruct-Q8_0.gguf (-ngl 99 -p 256 -n 128 -r 3)
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| llama 8B Q8_0                  |   7.95 GiB |     8.03 B | CUDA       |  99 |           pp256 |   11117.59 ± 1979.33 |
| llama 8B Q8_0                  |   7.95 GiB |     8.03 B | CUDA       |  99 |           tg128 |        164.45 ± 0.65 |
## llama-bench Qwen3.8-27B-UD-Q4_K_M.gguf (-ngl 99 -p 256 -n 128 -r 3)
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | CUDA       |  99 |           pp256 |     2446.98 ± 326.18 |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | CUDA       |  99 |           tg128 |         75.25 ± 0.26 |
## llama-bench gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf (-ngl 99 -p 256 -n 128 -r 3)
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| gemma4 26B.A4B Q4_0            |  13.26 GiB |    25.23 B | CUDA       |  99 |           pp256 |    5477.69 ± 3845.38 |
| gemma4 26B.A4B Q4_0            |  13.26 GiB |    25.23 B | CUDA       |  99 |           tg128 |       227.82 ± 35.15 |
## llama-bench Qwen3.5-35B-A3B-Q4_K_M.gguf (-ngl 99 -p 256 -n 128 -r 3)
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35moe 35B.A3B Q4_K - Medium |  19.71 GiB |    34.66 B | CUDA       |  99 |           pp256 |    4726.40 ± 1090.83 |
| qwen35moe 35B.A3B Q4_K - Medium |  19.71 GiB |    34.66 B | CUDA       |  99 |           tg128 |        245.00 ± 0.96 |
## MTP: llama-speculative-simple, draft-mtp depth 4, greedy, n=256, same prompt as the Mac
```
& : The term 'C:\Users\davinchy\egpu-reference\llama.cpp\llama-speculative-simple.exe' is not 
recognized as the name of a cmdlet, function, script file, or operable program. Check the 
spelling of the name, or if a path was included, verify that the path is correct and try again.
At line:1 char:3
+ & C:\Users\davinchy\egpu-reference\llama.cpp\llama-speculative-simple ...
+   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : ObjectNotFound: (C:\Users\davinc...tive-simple.exe:String) [], Comm 
   andNotFoundException
    + FullyQualifiedErrorId : CommandNotFoundException
 
```
## image generation (sd-cli, same seeds/steps/prompt as the Mac)
```
SDXL-Turbo 4 steps 512: sampling completed, taking 2.45s decode_first_stage completed, taking 0.32s generate_image completed in 4.11s 
SD 1.5 20 steps 512: sampling completed, taking 3.18s decode_first_stage completed, taking 0.31s generate_image completed in 3.86s 
Z-Image-Turbo 8 steps 1024: sampling completed, taking 7.18s decode_first_stage completed, taking 0.62s generate_image completed in 9.46s 
```
## MTP via llama-server (draft-mtp depth 4, chat template, sampler temp 0.6 / top-p 0.95 / top-k 20, max_tokens 256)
```
prose  (sky/sunsets):        114.1 tok/s, 43.9% draft accept
code   (ISO-8601 parser):    140.7 tok/s, 57.7% accept
list   (planets):            153.1 tok/s, 62.3% accept
greedy (Thunderbolt prompt): 124.7 tok/s, 48.6% accept
```
