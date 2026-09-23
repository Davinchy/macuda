#!/bin/sh
# run-3090.sh - build gemm.cu for sm_86 and run test_splitk on the 3090 box, then the three mutants (B, 2026-09-23).
#   sh run-3090.sh <dir holding gemm.cu and test_splitk.cu>
# The reference arm must exit 0; every mutant must exit nonzero, and each mutant's sed must change exactly one line.
set -u
D=$1; cd "$D" || exit 99
NVCC=/usr/local/cuda-13.0/bin/nvcc
echo "=== host: $(hostname) $(date) up: $(uptime)"
nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total,utilization.gpu --format=csv,noheader
echo "toolkit: $($NVCC --version | tail -1)"
echo "gemm.cu sha256: $(sha256sum gemm.cu | cut -c1-16)  test sha256: $(sha256sum test_splitk.cu | cut -c1-16)"
$NVCC -O2 -arch=sm_86 -o test_splitk test_splitk.cu -lcuda || { echo "BUILD FAIL test"; exit 99; }
mk(){ $NVCC -cubin -arch=sm_86 -O3 -o "$2" "$1" || { echo "BUILD FAIL $1"; exit 99; }; }
mk gemm.cu ref.cubin
echo "=== REFERENCE ARM"; ./test_splitk ref.cubin; rc=$?; echo "REFERENCE rc=$rc (must be 0)"
mut(){ # name, sed expression
  sed "$2" gemm.cu > "m_$1.cu"
  nl=$(diff gemm.cu "m_$1.cu" | grep -c '^>'); echo "=== MUTANT $1: $nl line(s) changed (must be 1)"
  [ "$nl" = 1 ] || { echo "MUTANT $1 VOID: the sed did not change exactly one line"; return; }
  mk "m_$1.cu" "m_$1.cubin"; ./test_splitk "m_$1.cubin" | grep -E 'FAIL|RESULT|ERROR' | head -8; r=$?
  echo "MUTANT $1 exit: see RESULT above (must be >= 1 FAIL)"
}
mut drop_slice 's/for (int q = 0; q < S; q++) sum +=/for (int q = 0; q < S - 1; q++) sum +=/'
mut reverse_order 's/for (int q = 0; q < S; q++) sum += __ldcg(base + (long long)q \*/for (int q = S - 1; q >= 0; q--) sum += __ldcg(base + (long long)q */'
mut no_reset 's/    if (tid == 0) cnt\[bz \* tiles + tile\] = 0u;/    ;/'
echo "=== done $(date)"
