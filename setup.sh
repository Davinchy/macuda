#!/bin/sh
# setup.sh — fetch and build everything the shim needs that is not in this repository, in order.
#   sh setup.sh            every step below
#   sh setup.sh deps       NVIDIA open-gpu-kernel-modules headers + GSP firmware (pinned, hash-checked) + CUDA headers
#   sh setup.sh llama      clone llama.cpp at the pinned commit and build its CPU-only static tree (llama.cpp/build-null)
#   sh setup.sh sd         clone stable-diffusion.cpp at the pinned commit, point it at llama.cpp's ggml, apply the patch, build
#   sh setup.sh shim       make libtinynv.a + libtinycudart.a + libtinycublas.a (no GPU needed)
#   sh setup.sh cuda       compile ggml's CUDA backend: device code through nvcc, host code through clang
#   sh setup.sh link       link the *-null binaries against the shim (needs cuda-shim/build/libggml-cuda.a from the step above)
# Environment: LLAMA_SRC / SD_SRC name a local clone to copy from instead of GitHub; JOBS caps the parallel build (default:
# all cores). The device compile needs nvcc, which is Linux-only: set TINYCC_HOST / TINYCC_KEY to use a Linux box over ssh,
# or leave them unset and it runs in a CUDA container on this Mac (see cuda-shim/build/tinycc, and install.sh).
set -eu
R=$(cd "$(dirname "$0")" && pwd); cd "$R"
LLAMA_URL=https://github.com/ggml-org/llama.cpp.git
LLAMA_BASE=ad6c66839af3c5646fba8c6c2e2087a1e4e38948   # ad6c668: the tree the ggml-cuda archive and every number in the README were built from
LLAMA_FIX=2f539596c6e9a977e91b6bc6344650422c6bc3b0    # 2f53959 (#28882): ggml-cpu rope work-buffer fix; without it test-backend-ops corrupts its heap on arm64
SD_URL=https://github.com/leejet/stable-diffusion.cpp.git
SD_COMMIT=59c23bce0d82be3a922023ab811194f05b3e2faa      # 59c23bc
JOBS=${JOBS:-$(sysctl -n hw.ncpu)}
step=${1:-all}

deps() {
  sh cuda-shim/libtinynv/tools/fetch_nv_headers.sh
  sh cuda-shim/libtinynv/tools/fetch_firmware.sh
  sh cuda-shim/build/fetch-cuda-headers.sh
}
clone_at() { # clone_at <dir> <url> <commit> [local source]
  d=$1; url=$2; c=$3; src=${4:-}
  if [ ! -d "$d/.git" ]; then
    if [ -n "$src" ]; then git clone -q "$src" "$d" && git -C "$d" remote set-url origin "$url"; else git clone -q "$url" "$d"; fi
  fi
  git -C "$d" cat-file -e "$c^{commit}" 2>/dev/null || git -C "$d" fetch -q origin "$c"
  git -C "$d" checkout -q --detach "$c"
}
llama() {
  # already base + the fix (a cherry-pick on top of LLAMA_BASE)? then leave the tree alone
  if [ "$(git -C llama.cpp rev-parse HEAD~1 2>/dev/null)" != "$LLAMA_BASE" ] || ! git -C llama.cpp log --format=%s -1 | grep -q 'CACHE_LINE_SIZE'; then
    clone_at llama.cpp "$LLAMA_URL" "$LLAMA_BASE" "${LLAMA_SRC:-}"
    git -C llama.cpp cat-file -e "$LLAMA_FIX^{commit}" 2>/dev/null || git -C llama.cpp fetch -q origin "$LLAMA_FIX"
    git -C llama.cpp -c user.name=setup -c user.email=setup@localhost cherry-pick -x "$LLAMA_FIX"
  fi
  echo "llama.cpp at $(git -C llama.cpp rev-parse --short HEAD): $(git -C llama.cpp log -1 --format=%s | cut -c1-70)"
  # CPU-only, static, no Metal: the shim replaces the CUDA backend at link time, and every other backend must be absent
  mkdir -p llama.cpp/build-null
  cmake -S llama.cpp -B llama.cpp/build-null -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
    -DGGML_METAL=OFF -DGGML_BLAS=OFF -DGGML_ACCELERATE=ON -DGGML_NATIVE=ON -DLLAMA_CURL=OFF \
    -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_OPENSSL=ON > llama.cpp/build-null/cmake.log 2>&1 || { tail -20 llama.cpp/build-null/cmake.log; exit 1; }
  cmake --build llama.cpp/build-null -j"$JOBS" --target test-backend-ops llama-bench llama-simple llama-speculative-simple llama-server llama-mtmd-cli \
    > llama.cpp/build-null/build.log 2>&1 || { tail -30 llama.cpp/build-null/build.log; exit 1; }
  echo "llama.cpp/build-null built"
}
sd() {
  clone_at stable-diffusion.cpp "$SD_URL" "$SD_COMMIT" "${SD_SRC:-}"
  # upstream ggml (llama.cpp's), not leejet's fork: the shim's ggml-cuda archive is built from llama.cpp's tree
  if [ ! -L stable-diffusion.cpp/ggml ]; then rm -rf stable-diffusion.cpp/ggml; ln -s ../llama.cpp/ggml stable-diffusion.cpp/ggml; fi
  git -C stable-diffusion.cpp apply --check ../patches/stable-diffusion.cpp-upstream-ggml.patch 2>/dev/null \
    && git -C stable-diffusion.cpp apply ../patches/stable-diffusion.cpp-upstream-ggml.patch && echo "patch applied" || echo "patch already applied (or does not apply — check git -C stable-diffusion.cpp status)"
  mkdir -p stable-diffusion.cpp/build-null
  cmake -S stable-diffusion.cpp -B stable-diffusion.cpp/build-null -DCMAKE_BUILD_TYPE=Release -DSD_BUILD_SHARED_LIBS=OFF \
    -DGGML_METAL=OFF -DGGML_NATIVE=ON > stable-diffusion.cpp/build-null/cmake.log 2>&1 || { tail -20 stable-diffusion.cpp/build-null/cmake.log; exit 1; }
  cmake --build stable-diffusion.cpp/build-null -j"$JOBS" --target sd-cli > stable-diffusion.cpp/build-null/build.log 2>&1 || { tail -30 stable-diffusion.cpp/build-null/build.log; exit 1; }
  echo "stable-diffusion.cpp/build-null built"
}
shim() { ( cd cuda-shim && rm -rf build/shim/nv && make -s ) && echo "shim built: libtinynv build id $(strings cuda-shim/build/shim/nv/libtinynv.a | grep -oE '^[0-9a-f]{7}(-dirty)?$|^nogit(-dirty)?$' | head -1)"; }
cuda() { sh cuda-shim/build/build-ggml-cuda.sh; }
link() {
  test -f cuda-shim/build/libggml-cuda.a || { echo "no cuda-shim/build/libggml-cuda.a: build it with  sh setup.sh cuda  (nvcc in a container here, or on TINYCC_HOST)"; exit 1; }
  cd cuda-shim
  for t in 'tests test-backend-ops' 'examples/simple llama-simple' 'examples/speculative-simple llama-speculative-simple' 'tools/server llama-server' 'tools/mtmd llama-mtmd-cli' 'tools/llama-bench llama-bench'; do
    d=${t%% *}; n=${t##* }; sh build/link-null.sh "$d" "$n" "$R/cuda-shim/build/bin/$n-null" 2>&1 | grep -vE 'duplicate|warning' | tail -1
  done
  if [ -f "$R/stable-diffusion.cpp/build-null/examples/cli/CMakeFiles/sd-cli.dir/link.txt" ]; then
    L=$R/stable-diffusion.cpp/build-null sh build/link-null.sh examples/cli sd-cli "$R/cuda-shim/build/bin/sd-cli-null" 2>&1 | grep -vE 'duplicate|warning' | tail -1
  fi
  ls -1 build/bin
}
case "$step" in
  all)   deps; llama; sd; shim; cuda; link ;;
  deps)  deps ;; llama) llama ;; sd) sd ;; shim) shim ;; cuda) cuda ;; link) link ;;
  *) echo "usage: sh setup.sh [all|deps|llama|sd|shim|cuda|link]"; exit 2 ;;
esac
