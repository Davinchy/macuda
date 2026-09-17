#!/bin/sh
# Compile ALL of ggml's CUDA backend (top-level TUs + template-instances) through tinycc (device on the Linux box, host on
# the Mac) with the shim's final build configuration, in parallel, and archive to libggml-cuda.a.
#   JOBS=8 sh build/build-ggml-cuda.sh            # everything, for sm_120a (ggml's own CMake ships 120a-real on Blackwell)
#   ONLY="mmq-instance-nvfp4 mmq-instance-mxfp4" sh build/build-ggml-cuda.sh   # a subset, into the existing objects
set -u
R=${EGPU_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}; G=$R/llama.cpp/ggml
export TINYCC=$R/cuda-shim/build/tinycc OBJ=$R/cuda-shim/build/obj-${ARCH:-sm_120a} LOGD=$R/cuda-shim/build/logs-${ARCH:-sm_120a}
JOBS=${JOBS:-8}
# the architecture is passed to tinycc EXPLICITLY: the FP4 MMQ instances use Blackwell's block-scaled tensor-core MMA,
# which ptxas only accepts for the architecture-specific target, and an inherited environment variable proved too easy to lose
export TINYCC_ARCH=${ARCH:-sm_120a}
ONLY=${ONLY:-}
if [ -z "$ONLY" ]; then rm -rf "$OBJ" "$LOGD"; fi; mkdir -p "$OBJ" "$LOGD"
# resolved once here rather than by each of the 188 tinycc invocations, each of which would otherwise link a probe
export TINYCC_SYSROOT=${TINYCC_SYSROOT:-$(sh "$R/cuda-shim/build/sdk.sh")}
export INC="-I$G/include -I$G/src -I$G/src/ggml-cuda"
# docs/03-cuda-shim-plan.md §1: quantized matmul forced onto MMQ (cuBLAS only for residual GEMMs), no virtual memory
# management, and CUDA graphs compiled OUT (USE_CUDA_GRAPH needs GGML_CUDA_USE_GRAPHS, which is deliberately absent:
# ggml wraps cudaStreamBeginCapture in CUDA_CHECK, so a refused capture would abort rather than fall back)
export DEFS="-DGGML_CUDA_FORCE_MMQ -DGGML_CUDA_NO_VMM -DNDEBUG"
FA=""; for k in F16 Q4_0 Q4_1 Q5_0 Q5_1 Q8_0 BF16; do for v in F16 Q4_0 Q4_1 Q5_0 Q5_1 Q8_0 BF16; do FA="$FA -DGGML_CUDA_FA_${k}_${v}=1"; done; done
export FA
ls "$G"/src/ggml-cuda/*.cu "$G"/src/ggml-cuda/template-instances/*.cu > "$LOGD/tus.txt"
if [ -n "$ONLY" ]; then for b in $ONLY; do grep "/$b\.cu$" "$LOGD/tus.txt"; done > "$LOGD/tus.only"; mv "$LOGD/tus.only" "$LOGD/tus.txt"; fi
total=$(wc -l < "$LOGD/tus.txt" | tr -d ' ')
echo "compiling $total TUs for $TINYCC_ARCH with $JOBS jobs: $DEFS"
start=$(date +%s)
xargs -n1 -P"$JOBS" sh "$R/cuda-shim/build/cc-one.sh" < "$LOGD/tus.txt" | tee "$LOGD/results.txt" | grep --line-buffered FAIL
ok=$(grep -c '^ok' "$LOGD/results.txt"); fail=$(grep -c '^FAIL' "$LOGD/results.txt")
echo "compiled $ok/$total ggml-cuda TUs in $(( $(date +%s) - start )) s; failed: $(grep '^FAIL' "$LOGD/results.txt" | cut -d' ' -f2 | tr '\n' ' ')"
if [ "$ok" -gt 0 ]; then
  # ONE ARCHIVE PER ARCHITECTURE, and the plain name is a symlink to whichever one the tree links against. This used to
  # be a single libggml-cuda.a, so building for a second card silently destroyed the first card's archive - an hour of
  # compiling, gone, and nothing to say which arch the surviving file held. Now `ls -l` answers that.
  A="$R/cuda-shim/build/libggml-cuda.$TINYCC_ARCH.a"
  rm -f "$A"
  /opt/homebrew/opt/llvm/bin/llvm-ar rcs "$A" "$OBJ"/*.o && {
    ln -sfn "libggml-cuda.$TINYCC_ARCH.a" "$R/cuda-shim/build/libggml-cuda.a"
    echo "archived libggml-cuda.$TINYCC_ARCH.a = $(stat -f%z "$A") bytes ($(ls "$OBJ"/*.o | wc -l | tr -d ' ') objects); libggml-cuda.a now points at it"
  }
fi
[ "$fail" -eq 0 ]
