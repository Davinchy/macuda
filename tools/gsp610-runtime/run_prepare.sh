#!/bin/sh
# : reproduce the complete AD102 610.43.03 runtime bundle + validation, offline.
#   1. fetch/verify the pinned 610.43.03 source tree and official release assets
#   2. prepare the runtime bundle deterministically (boot AND unload assets, manifest, notices)
#   3. validate the bundle against the frozen pins
#   4. run the unittest suite (valid/corrupt/truncated/wrong-chip + real-artifact integration)
# No GPU is contacted.
set -e
HERE=$(cd "$(dirname "$0")/../.." && pwd)
T=$HERE/tools/gsp610-runtime
OUT=$HERE/cuda-shim/build/gsp610-runtime/bundle

sh "$T/fetch_inputs.sh"
sh "$T/fetch_release_fw.sh"
python3 "$T/prepare_runtime.py" --out "$OUT" "$@"
python3 "$T/validate_runtime.py" "$OUT"
python3 -m unittest discover -s "$T" -p 'test_*.py' -v
echo "runtime bundle: $OUT"
