#!/bin/sh
# Copy the CUDA Toolkit headers the host-side compile needs into cuda-shim/cuda-13/include: the shim's cudart/cublas
# prototypes (libtinycudart/cudart_api.c, libtinycublas/cublas.c) and clang's --cuda-host-only pass over ggml-cuda (tinycc).
# They are NVIDIA's, under the CUDA Toolkit EULA, so they are not in this repository (cuda-13 is gitignored). They come
# from the SAME CONTAINER IMAGE that runs nvcc for the device compile, which means a Mac with docker needs nothing else -
# no second machine and no CUDA install. In preference order: CUDA_INCLUDE_SRC=<.../include> for a local toolkit,
# TINYCC_HOST for the original ssh path to a Linux box, and otherwise the container.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd); DEST=$ROOT/cuda-13
if [ -d "$DEST/include/crt" ] && [ -f "$DEST/include/cublas_v2.h" ] && [ -d "$DEST/include/cccl" ]; then echo "have $DEST/include"; exit 0; fi
mkdir -p "$DEST/bin"
if [ -n "${CUDA_INCLUDE_SRC:-}" ]; then
  cp -R "$CUDA_INCLUDE_SRC" "$DEST/include"
elif [ -n "${TINYCC_HOST:-}" ]; then
  scp -q -r ${TINYCC_KEY:+-i "$TINYCC_KEY"} "$TINYCC_HOST:/usr/local/cuda/include" "$DEST/"
else
  # No Linux box: take them out of the same CUDA container that compiles the device code, so there is exactly one
  # source of CUDA on the machine and the headers match the nvcc that will be used. Nothing here executes CUDA or
  # needs a GPU. The bind-mount destination works on both runtimes - Apple's `container` has no cp subcommand.
  #
  # The copy is a tar pipeline INSIDE the container, and -h MATTERS: in these images /usr/local/cuda/include is a
  # SYMLINK to targets/sbsa-linux/include, so an archive of the plain path lands a dangling symlink. The failure is
  # quiet - the directory appears to exist and contains one entry - and the first sign of it is the host compile not
  # finding cublas_v2.h. tar also keeps the extraction the container's own, so nothing about the host's filesystem
  # (case folding, xattrs, a noowners volume) can alter what lands.
  IMAGE=${TINYCC_CUDA_IMAGE:-nvidia/cuda:13.0.3-devel-ubuntu24.04}
  RT=$(sh "$(dirname "$0")/container-runtime.sh")
  command -v "$RT" > /dev/null 2>&1 || { echo "no CUDA headers source: set CUDA_INCLUDE_SRC=<local .../include>, or TINYCC_HOST=user@box, or install a container runtime - Apple's (brew install container) or Docker (see install.sh)"; exit 1; }
  echo "taking the CUDA headers out of $IMAGE with $RT (no GPU involved)"
  mkdir -p "$DEST"
  "$RT" run --rm --network=none -v "$DEST:/dest" "$IMAGE" \
    sh -c 'cd /usr/local/cuda && tar cfh - include | (cd /dest && tar xf -)'
fi
# clang reads version.txt/version.json to decide whether it knows this CUDA; 12.8 is the newest the pinned Homebrew clang
# accepts, the headers are whatever the box has (13.0), and tinycc passes -Wno-unknown-cuda-version for the difference.
printf 'CUDA Version 12.8.0\n' > "$DEST/version.txt"
printf '{"cuda":{"name":"CUDA SDK","version":"12.8.0"}}' > "$DEST/version.json"
echo "CUDA headers at $DEST/include ($(ls "$DEST/include" | wc -l | tr -d ' ') entries)"
