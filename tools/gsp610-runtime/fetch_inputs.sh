#!/bin/sh
# : fetch and hash-verify the two pinned NVIDIA header/source trees the ABI audit compares.
#
#   570 side: open-gpu-kernel-modules commit 81fe4fb417c8ac3b9bdcc1d56827d116743892a5 (release 570.86.16),
#             the exact pin tools/fetch_nv_headers.sh and the Makefile's NVSRC already use.
#   610 side: NVIDIA-kernel-module-source-610.43.03.tar.xz from download.nvidia.com.
#
# Everything lands in this worktree's cuda-shim/libtinynv/third_party/ (gitignored).
# No GPU is contacted. Re-running reproduces the same verified trees.
set -e
HERE=$(cd "$(dirname "$0")/../.." && pwd)   # repo root
TP=$HERE/cuda-shim/libtinynv/third_party
mkdir -p "$TP"

OGKM_COMMIT=81fe4fb417c8ac3b9bdcc1d56827d116743892a5
OGKM_TARBALL=ogkm-$OGKM_COMMIT.tar.gz
KMS_NAME=NVIDIA-kernel-module-source-610.43.03
KMS_TARBALL=$KMS_NAME.tar.xz
KMS_SHA=7e118923c7a23edc36114d63273a46e3e04e9af98695a42203e7ac2dfe9fc1dc

sha_of() { shasum -a 256 "$1" | cut -d' ' -f1; }

# --- 570 header tree (same pin as fetch_nv_headers.sh; hash expectation from tools/SHA256SUMS) ---
if [ ! -d "$TP/open-gpu-kernel-modules-$OGKM_COMMIT/src" ]; then
  if [ ! -f "$TP/$OGKM_TARBALL" ]; then
    echo "fetching open-gpu-kernel-modules $OGKM_COMMIT"
    curl -fL --retry 3 -o "$TP/$OGKM_TARBALL.part" \
      "https://github.com/NVIDIA/open-gpu-kernel-modules/archive/$OGKM_COMMIT.tar.gz"
    mv "$TP/$OGKM_TARBALL.part" "$TP/$OGKM_TARBALL"
  fi
  WANT=$(grep "$OGKM_COMMIT" "$HERE/cuda-shim/libtinynv/tools/SHA256SUMS" | cut -d' ' -f1)
  GOT=$(sha_of "$TP/$OGKM_TARBALL")
  [ "$GOT" = "$WANT" ] || { echo "HASH MISMATCH $OGKM_TARBALL: got $GOT want $WANT"; exit 1; }
  echo "570 tarball verified: $GOT"
  tar xf "$TP/$OGKM_TARBALL" -C "$TP"
fi
echo "570 headers: $TP/open-gpu-kernel-modules-$OGKM_COMMIT"

# --- 610.43.03 kernel module source (contains the same sdk/nvidia header layout) ---
if [ ! -d "$TP/$KMS_NAME/src" ]; then
  if [ ! -f "$TP/$KMS_TARBALL" ]; then
    echo "fetching $KMS_TARBALL (26 MB)"
    curl -fL --retry 3 -o "$TP/$KMS_TARBALL.part" \
      "https://download.nvidia.com/XFree86/NVIDIA-kernel-module-source/$KMS_TARBALL"
    mv "$TP/$KMS_TARBALL.part" "$TP/$KMS_TARBALL"
  fi
  GOT=$(sha_of "$TP/$KMS_TARBALL")
  [ "$GOT" = "$KMS_SHA" ] || { echo "HASH MISMATCH $KMS_TARBALL: got $GOT want $KMS_SHA"; exit 1; }
  echo "610 tarball verified: $GOT"
  tar xf "$TP/$KMS_TARBALL" -C "$TP"
fi
echo "610 headers: $TP/$KMS_NAME"
echo "done"
