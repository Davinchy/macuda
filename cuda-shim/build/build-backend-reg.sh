#!/bin/sh
# The one CPU-side object that changes when the CUDA backend is linked in: ggml's backend REGISTRY, compiled with
# GGML_USE_CUDA on. The link scripts place it before libggml.a so the archive's own (registry-less) copy is never
# pulled - see build/link-null.sh.
#
#   sh build/build-backend-reg.sh              -> build/ggml-backend-reg.cuda.o       (-DGGML_USE_CUDA)
#   sh build/build-backend-reg.sh cuda-metal   -> build/ggml-backend-reg.cuda-metal.o (+ -DGGML_USE_METAL, for
#                                                 build/link-null-metal.sh; needs the Metal build tree as well)
#
# WHY THIS SCRIPT EXISTS: it did not, and nothing else built this object. Every link script consumed it, the README
# called it a build output, and on the machines where the project grew up it was simply lying in build/ from an
# earlier by-hand compile. A fresh clone therefore linked NOTHING - clang++ said "no such file or directory" seven
# times, setup.sh's link step swallowed it (2026-09-22), and the install reported success with an empty build/bin.
#
# The flags are taken from the CMake build's own compile_commands.json rather than copied here, because this object
# is linked beside objects that CMake compiled: if their flags drift and this one's are pinned in a script, the two
# disagree silently, which is the same class of bug as the missing object itself.
set -eu
R=${EGPU_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
L=${L:-$R/llama.cpp/build-null}; G=${GGML_BUILD:-$R/cuda-shim/build}
mode=${1:-cuda}
case "$mode" in
  cuda)       defs="-DGGML_USE_CUDA"; out=$G/ggml-backend-reg.cuda.o ;;
  cuda-metal) defs="-DGGML_USE_CUDA -DGGML_USE_METAL"; out=$G/ggml-backend-reg.cuda-metal.o ;;
  *) echo "usage: $0 [cuda|cuda-metal]" >&2; exit 2 ;;
esac
cc="$L/compile_commands.json"
test -f "$cc" || { echo "no $cc: build llama.cpp first (sh setup.sh llama)"; exit 1; }
mkdir -p "$G"
cmd=$(DEFS="$defs" OUT="$out" python3 - "$cc" <<'PY'
import json, os, shlex, sys
entries = json.load(open(sys.argv[1]))
for e in entries:
    if os.path.basename(e["file"]) == "ggml-backend-reg.cpp":
        argv = shlex.split(e["command"])
        # replace CMake's object path with ours; everything else - warnings, standard, arch, includes - is kept as
        # compiled, so this object and the ones it links beside were built the same way
        argv[argv.index("-o") + 1] = os.environ["OUT"]
        print(shlex.join(argv[:1] + shlex.split(os.environ["DEFS"]) + argv[1:]))
        break
else:
    sys.exit("ggml-backend-reg.cpp is not in compile_commands.json")
PY
)
eval "$cmd"
echo "compiled $(basename "$out") ($(stat -f%z "$out") bytes, $defs)"
