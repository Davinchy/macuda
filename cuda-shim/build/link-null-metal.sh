#!/bin/sh
# One process, two GPUs: link a llama.cpp target with BOTH the Metal backend (Apple GPU) and the CUDA backend through the
# shim (the eGPU), so `-dev CUDA0,MTL0 -sm layer -ts <card>,<mac>` splits a model's layers across the two inside one
# llama-server - the card's layers and KV in VRAM, the rest in unified memory. This is how models larger than Metal's
# working set run at all (GLM-5.3-Flash UD-IQ3_XXS, 112 GiB, with 11 of 46 layers on a 5090), and how the split
# prefill/decode numbers in the README were taken.
#   sh setup.sh split          (configures llama.cpp/build-metal-null, builds the registry object, links the binaries)
#   sh build/link-null-metal.sh <cmake target dir> <target name> <output binary>      (one target, by hand)
# The Metal tree is static (BUILD_SHARED_LIBS=OFF) and from the SAME llama.cpp commit as libggml-cuda.a: ggml's backend
# interface is not stable across commits, and a mismatched archive links but crashes or computes garbage.
# Every crossing between the two devices costs ~0.3 ms per decode step (measured on a 5090 over Thunderbolt), so a few
# large contiguous layer ranges beat many small ones; -ts 27,21 (Qwen3.8-Flash-Next) and 11,35 (GLM-5.3-Flash) were best.
set -eu
R=${EGPU_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}; G=${GGML_BUILD:-$R/cuda-shim/build}
export L=${L:-$R/llama.cpp/build-metal-null}
test -f "$G/ggml-backend-reg.cuda-metal.o" || sh "$(dirname "$0")/build-backend-reg.sh" cuda-metal
# CUDA graphs default OFF in these binaries (MACUDA_SPLIT_GRAPHS=1 opts in): see mc-defaults.c for the fault that is why
cc -O2 -arch arm64 -c "$(dirname "$0")/mc-defaults.c" -o "$G/mc-defaults.o"
REG="$G/ggml-backend-reg.cuda-metal.o $G/mc-defaults.o" exec sh "$(dirname "$0")/link-null.sh" "$@"
