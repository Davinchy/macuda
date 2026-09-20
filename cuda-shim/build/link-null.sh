#!/bin/sh
# Link any llama.cpp CMake executable target against the CUDA backend built through the shim, reusing the exact link
# line CMake generated for the CPU-only static build and inserting, ahead of libggml.a: the registry object compiled
# with GGML_USE_CUDA, libggml-cuda.a, the three shim libraries and LLVM's demangler.
#   sh build/link-null.sh <cmake target dir, e.g. tools/llama-bench> <target name, e.g. llama-bench> <output binary>
set -eu
# S selects the shim tree whose libraries (and, for llama-bench, whose build/bin) are used: the main checkout by default,
# or a branch worktree such as $R/cuda-shim-b. The ggml-cuda archive is branch-independent and expensive, so it always
# comes from GGML_BUILD (the main checkout's build/) unless said otherwise.
R=${EGPU_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}; L=${L:-$R/llama.cpp/build-null}; S=${S:-$R/cuda-shim}; G=${GGML_BUILD:-$R/cuda-shim/build}   # L: the CMake build whose link line is replayed (llama.cpp by default; stable-diffusion.cpp's works the same way)
# the macOS SDK to link against, chosen by a link test (this script's own copy of sdk.sh, whichever tree S names)
SDKROOT=$(sh "$(dirname "$0")/sdk.sh") || exit 1
tdir="$1"; name="$2"; out="$3"; linktxt="$L/$tdir/CMakeFiles/$name.dir/link.txt"; test -f "$linktxt" || { echo "no link line at $linktxt (build the target first)"; exit 1; }
extra="$G/ggml-backend-reg.cuda.o $G/libggml-cuda.a $S/build/shim/libtinycudart.a $S/build/shim/libtinycublas.a $S/build/shim/nv/libtinynv.a /opt/homebrew/opt/llvm/lib/libLLVMDemangle.a"
cmd=$(sed -E "s#^([^ ]+) #\\1 -isysroot $SDKROOT #; s#-o [^ ]+#-o $out#; s#([^ ]*/libggml\.a)#$extra \\1#" "$linktxt")
mkdir -p "$(dirname "$out")"; cd "$L/$tdir" && eval "$cmd" && test -f "$out" && echo "linked $out ($(stat -f%z "$out") bytes)"
