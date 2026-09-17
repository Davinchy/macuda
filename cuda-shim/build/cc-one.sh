#!/bin/sh
# one ggml-cuda TU through tinycc; env: TINYCC OBJ LOGD INC DEFS FA. prints ok/FAIL <basename>
f="$1"; b=$(basename "$f" .cu)
if $TINYCC "$f" -o "$OBJ/$b.o" $INC $DEFS $FA > "$LOGD/$b.log" 2>&1; then echo "ok $b"; else echo "FAIL $b"; fi
