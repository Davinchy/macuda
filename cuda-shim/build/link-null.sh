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
# GGML_ARCHIVE and SHIM_DIR name one architecture's ggml-cuda archive and shim libraries explicitly (setup.sh sets them
# from ARCH); unset, the plain libggml-cuda.a symlink and build/shim are used, as before.
GA=${GGML_ARCHIVE:-$G/libggml-cuda.a}; SD=${SHIM_DIR:-$S/build/shim}
extra="$G/ggml-backend-reg.cuda.o $GA $SD/libtinycudart.a $SD/libtinycublas.a $SD/nv/libtinynv.a /opt/homebrew/opt/llvm/lib/libLLVMDemangle.a"
# The link line is CMake's, recorded when llama.cpp was configured - which means it carries the EXACT Homebrew Cellar
# paths of the machine that configured it. On this machine openssl@3 is 3.6.2 and the recorded line wants 3.6.4, so four
# of the five targets failed with "no such file or directory: .../3.6.4/lib/libcrypto.dylib" and the fifth, which does
# not link openssl, went through - which reads like a broken build rather than a moved dependency. Any Cellar path that
# no longer exists is repointed at whatever version of that formula is installed; one that cannot be found at all is
# left alone so the linker still names it.
retarget() {
  echo "$1" | tr ' ' '\n' | while IFS= read -r tok; do
    case "$tok" in
      /opt/homebrew/Cellar/*)
        if [ -e "$tok" ]; then printf '%s\n' "$tok"; continue; fi
        formula=$(echo "$tok" | cut -d/ -f5); rest=$(echo "$tok" | cut -d/ -f7-)
        found=""
        for v in /opt/homebrew/Cellar/"$formula"/*; do [ -e "$v/$rest" ] && found="$v/$rest"; done
        printf '%s\n' "${found:-$tok}" ;;
      *) printf '%s\n' "$tok" ;;
    esac
  done | tr '\n' ' '
}
cmd=$(sed -E "s#^([^ ]+) #\\1 -isysroot $SDKROOT #; s#-o [^ ]+#-o $out#; s#([^ ]*/libggml\.a)#$extra \\1#" "$linktxt")
cmd=$(retarget "$cmd")
mkdir -p "$(dirname "$out")"; cd "$L/$tdir" && eval "$cmd" && test -f "$out" && echo "linked $out ($(stat -f%z "$out") bytes)"
