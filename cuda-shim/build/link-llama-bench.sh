#!/bin/sh
# Link llama-bench against the CUDA backend built through the shim, on top of the CPU-only static CMake build.
# The one CPU-side object that changes under GGML_USE_CUDA is the backend registry; it is compiled separately with the
# switch on (build/ggml-backend-reg.cuda.o) and placed BEFORE libggml.a so the archive's copy is never pulled.
# Order matters for static archives: libggml-cuda.a references ggml core symbols, so it precedes libggml*.a, and the
# shim libraries follow it. Output: build/bin/llama-bench-null (runs on libtinynv's null device until a GPU is opened).
set -eu
# S selects the shim tree whose libraries (and, for llama-bench, whose build/bin) are used: the main checkout by default,
# or a branch worktree such as $R/cuda-shim-b. The ggml-cuda archive is branch-independent and expensive, so it always
# comes from GGML_BUILD (the main checkout's build/) unless said otherwise.
R=${EGPU_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}; L=$R/llama.cpp/build-null; S=${S:-$R/cuda-shim}; G=${GGML_BUILD:-$R/cuda-shim/build}
# the macOS SDK to link against, chosen by a link test (this script's own copy of sdk.sh, whichever tree S names)
SDKROOT=$(sh "$(dirname "$0")/sdk.sh") || exit 1
mkdir -p $S/build/bin
/usr/bin/c++ -isysroot $SDKROOT -O3 -DNDEBUG -arch arm64 \
  $L/tools/llama-bench/CMakeFiles/llama-bench.dir/main.cpp.o \
  -o $S/build/bin/llama-bench-null \
  $L/tools/llama-bench/libllama-bench-impl.a $L/common/libllama-common.a $L/common/libllama-common-base.a \
  $L/vendor/cpp-httplib/libcpp-httplib.a /opt/homebrew/Cellar/openssl@3/3.6.4/lib/libssl.dylib /opt/homebrew/Cellar/openssl@3/3.6.4/lib/libcrypto.dylib \
  -framework CoreFoundation -framework Security \
  $L/src/libllama.a \
  $G/ggml-backend-reg.cuda.o \
  $G/libggml-cuda.a \
  $S/build/shim/libtinycudart.a $S/build/shim/libtinycublas.a $S/build/shim/nv/libtinynv.a /opt/homebrew/opt/llvm/lib/libLLVMDemangle.a \
  $L/ggml/src/libggml.a $L/ggml/src/libggml-cpu.a $L/ggml/src/libggml-base.a \
  -lm -framework Accelerate
echo "linked $S/build/bin/llama-bench-null ($(stat -f%z $S/build/bin/llama-bench-null) bytes)"
