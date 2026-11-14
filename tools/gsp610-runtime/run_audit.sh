#!/bin/sh
# : reproduce the full offline ABI audit + input manifest. No GPU contact.
#   1. fetch/verify the two pinned NVIDIA header trees (idempotent)
#   2. compile the ABI probe against each tree into cuda-shim/build/gsp610-abi
#   3. source-level compare the port-layer/.c definitions
#   4. build the comparison matrix
#   5. record immutable input hashes -> docs/driver/gsp-610-runtime/inputs.json
set -e
HERE=$(cd "$(dirname "$0")/../.." && pwd)
T=$HERE/tools/gsp610-runtime
TP=$HERE/cuda-shim/libtinynv/third_party
OUT=$HERE/cuda-shim/build/gsp610-abi

sh "$T/fetch_inputs.sh"
sh "$T/build_probes.sh"
python3 "$T/abi_text.py" \
  "$TP/open-gpu-kernel-modules-81fe4fb417c8ac3b9bdcc1d56827d116743892a5" \
  "$TP/NVIDIA-kernel-module-source-610.43.03" > "$OUT/text_diff.json"
python3 "$T/abi_diff.py" "$OUT" > "$OUT/abi_matrix.json"
python3 "$T/make_inputs.py" --repo "$HERE"
echo "audit artifacts in $OUT"