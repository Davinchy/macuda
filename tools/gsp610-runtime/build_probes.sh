#!/bin/sh
# : compile the ABI probe against each pinned NVIDIA tree and emit the raw S/F/C records.
# Unique build dir under cuda-shim/build so no active training build is touched. No GPU contact.
set -e
HERE=$(cd "$(dirname "$0")/../.." && pwd)          # repo root
TP=$HERE/cuda-shim/libtinynv/third_party
OUT=$HERE/cuda-shim/build/gsp610-abi
mkdir -p "$OUT"

OGKM=$TP/open-gpu-kernel-modules-81fe4fb417c8ac3b9bdcc1d56827d116743892a5
KMS=$TP/NVIDIA-kernel-module-source-610.43.03

inc() {
  T=$1
  echo "-I$T/src/nvidia/arch/nvalloc/common/inc -I$T/src/common/sdk/nvidia/inc -I$T/src/common/inc -I$T/src/nvidia/inc -I$T/src/nvidia/inc/kernel -I$T/src/common/shared/msgq/inc -I$T/src/nvidia/inc/libraries -I$T/src/common/uproc/os/common/include -I$T/src/nvidia/generated"
}

CC=${CC:-cc}
for pair in "570:$OGKM" "610:$KMS"; do
  tag=${pair%%:*}; tree=${pair#*:}
  [ -d "$tree/src" ] || { echo "missing tree for $tag: $tree (run fetch_inputs.sh)"; exit 1; }
  echo "== building probe against $tag ($tree)"
  $CC -std=c11 -O0 -w -DGSP610_ABI_TREE=$tag $(inc "$tree") "$HERE/tools/gsp610-runtime/abi_probe.c" -o "$OUT/probe_$tag"
  "$OUT/probe_$tag" > "$OUT/abi_$tag.txt"
  echo "   $(wc -l < "$OUT/abi_$tag.txt") records -> $OUT/abi_$tag.txt"
done
echo "done"