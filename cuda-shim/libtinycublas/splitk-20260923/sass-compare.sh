#!/bin/sh
# sass-compare.sh <old.cu> <new.cu> - per-function sm_120 SASS, addresses stripped, old vs new (B, 2026-09-23).
# Every function in old must read SAME; functions only in new are listed. Control: the untouched f32/trsm/bw kernels.
set -u
N=/usr/local/cuda-13.0/bin
$N/nvcc -cubin -arch=sm_120 -O3 -o old120.cubin "$1" || exit 99
$N/nvcc -cubin -arch=sm_120 -O3 -o new120.cubin "$2" || exit 99
fns(){ $N/cuobjdump -sass "$1" | grep -o 'Function : [a-z0-9_]*' | awk '{print $3}' | sort; }
h(){ $N/cuobjdump -sass -fun "$2" "$1" | grep -E '^ +/\*[0-9a-f]{4}\*/' | sed 's#/\*[0-9a-f]*\*/##g' | sha256sum | cut -c1-12; }
fns old120.cubin > o.lst; fns new120.cubin > n.lst; d=0
for f in $(cat o.lst); do a=$(h old120.cubin $f); b=$(h new120.cubin $f); [ "$a" = "$b" ] && r=SAME || { r=DIFF; d=$((d+1)); }; echo "$f $a $b $r"; done
echo "new only: $(comm -13 o.lst n.lst | tr '\n' ' ')"
echo "SASS DIFF COUNT: $d"
