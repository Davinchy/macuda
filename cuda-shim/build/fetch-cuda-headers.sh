#!/bin/sh
# Copy the CUDA Toolkit headers the host-side compile needs into cuda-shim/cuda-13/include: the shim's cudart/cublas
# prototypes (libtinycudart/cudart_api.c, libtinycublas/cublas.c) and clang's --cuda-host-only pass over ggml-cuda (tinycc).
# They are NVIDIA's, under the CUDA Toolkit EULA, so they are not in this repository (cuda-13 is gitignored). They come from
# the same Linux box that runs nvcc for the device compile (TINYCC_HOST, /usr/local/cuda — CUDA 13.0 there), or from any local
# CUDA 12.8+/13.x install with CUDA_INCLUDE_SRC=<.../include>.
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
  # source of CUDA on the machine and the headers match the nvcc that will be used. `docker create` makes a container
  # without running it; nothing here executes CUDA or needs a GPU.
  IMAGE=${TINYCC_CUDA_IMAGE:-nvidia/cuda:13.0.3-devel-ubuntu24.04}
  command -v ${DOCKER:-docker} > /dev/null 2>&1 || { echo "no CUDA headers source: set CUDA_INCLUDE_SRC=<local .../include>, or TINYCC_HOST=user@box, or install Docker (see install.sh)"; exit 1; }
  echo "taking the CUDA headers out of $IMAGE (no GPU involved)"
  cid=$(${DOCKER:-docker} create "$IMAGE" /bin/true)
  ${DOCKER:-docker} cp "$cid:/usr/local/cuda/include" "$DEST/" && ${DOCKER:-docker} rm -f "$cid" > /dev/null
fi
# clang reads version.txt/version.json to decide whether it knows this CUDA; 12.8 is the newest the pinned Homebrew clang
# accepts, the headers are whatever the box has (13.0), and tinycc passes -Wno-unknown-cuda-version for the difference.
printf 'CUDA Version 12.8.0\n' > "$DEST/version.txt"
printf '{"cuda":{"name":"CUDA SDK","version":"12.8.0"}}' > "$DEST/version.json"
echo "CUDA headers at $DEST/include ($(ls "$DEST/include" | wc -l | tr -d ' ') entries)"
