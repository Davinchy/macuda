#!/bin/sh
set -e
CC=/opt/homebrew/opt/llvm/bin/clang; CXX=/opt/homebrew/opt/llvm/bin/clang++; ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT/libtinycudart"
$CC -std=c11 -I"$ROOT/include" -c cudart.c fatbin.c tinynv_stub.c   # C linkage: matches cuda_runtime.h extern "C"
$CXX "$ROOT/spike/vecadd.o" cudart.o fatbin.o tinynv_stub.o -o "$ROOT/spike/spike_real"
echo "built $ROOT/spike/spike_real"
