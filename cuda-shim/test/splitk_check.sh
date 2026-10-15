#!/bin/sh
# Rows for split-K dispatch, read from the launch trace of test/splitk_dispatch.c in BOTH arms (B, 2026-09-23).
# Card-free: null device. Each row is ACCEPT-OR-REFUSE against the printed trace, as stats_grid_check.sh does.
set -u
BIN="${1:?usage: splitk_check.sh <path to splitk_dispatch binary>}"
ON=$(TINYCUBLAS_SPLITK=1 TINYCUDART_TRACE=1 TINYNV_SOCKET= "$BIN" 2>&1); rc_on=$?
OFF=$(TINYCUBLAS_SPLITK=0 TINYCUDART_TRACE=1 TINYNV_SOCKET= "$BIN" 2>&1); rc_off=$?
# THE DEFAULT (split-K ON since 09-23 07:15): the knob unset must behave exactly as =1
DEF=$(env -u TINYCUBLAS_SPLITK TINYCUDART_TRACE=1 TINYNV_SOCKET= "$BIN" 2>&1); rc_def=$?
fails=0
row() { if [ "$2" = 0 ]; then echo "   ok:   $1"; else echo "   FAIL: $1"; fails=$((fails+1)); fi; }
has() { printf '%s\n' "$1" | grep -cF -- "$2" || true; }
row "both arms ran to the end (rc $rc_on and $rc_off, plan rows included)" "$([ $rc_on = 0 ] && [ $rc_off = 0 ] && echo 0 || echo 1)"
# ON arm: the design's grids, and the entry print
row "ON: split-K announces itself on entry" "$([ "$(has "$ON" 'split-K ON (TINYCUBLAS_SPLITK=1)')" = 1 ] && echo 0 || echo 1)"
row "ON: k/v -> tc2k grid (4,1,16)" "$([ "$(has "$ON" 'launch tinyblas_gemm_bf16_tc2k_tn grid=(4,1,16)')" = 1 ] && echo 0 || echo 1)"
row "ON: q/o and down -> tc2k grid (32,1,8), twice" "$([ "$(has "$ON" 'launch tinyblas_gemm_bf16_tc2k_tn grid=(32,1,8)')" = 2 ] && echo 0 || echo 1)"
row "ON: gate/up unchanged -> tc2s grid (172,1,1)" "$([ "$(has "$ON" 'launch tinyblas_gemm_bf16_tc2s_tn grid=(172,1,1)')" = 1 ] && echo 0 || echo 1)"
row "ON: diffusion mix unchanged -> tc2s grid (20,4,1)" "$([ "$(has "$ON" 'launch tinyblas_gemm_bf16_tc2s_tn grid=(20,4,1)')" = 1 ] && echo 0 || echo 1)"
row "ON: exactly 3 tc2k launches" "$([ "$(has "$ON" '[trace] launch tinyblas_gemm_bf16_tc2k')" = 3 ] && echo 0 || echo 1)"
# OFF arm: the reference - no split anywhere, and every shape on the kernel it took before
row "OFF: no tc2k launch and no entry print" "$([ "$(has "$OFF" 'tc2k')" = 0 ] && [ "$(has "$OFF" 'split-K ON')" = 0 ] && echo 0 || echo 1)"
row "OFF: k/v -> tc2s grid (4,1,1)" "$([ "$(has "$OFF" 'launch tinyblas_gemm_bf16_tc2s_tn grid=(4,1,1)')" = 1 ] && echo 0 || echo 1)"
row "OFF: q/o and down -> tc2s grid (32,1,1), twice" "$([ "$(has "$OFF" 'launch tinyblas_gemm_bf16_tc2s_tn grid=(32,1,1)')" = 2 ] && echo 0 || echo 1)"
row "OFF: diffusion mix -> tc2s grid (20,4,1)" "$([ "$(has "$OFF" 'launch tinyblas_gemm_bf16_tc2s_tn grid=(20,4,1)')" = 1 ] && echo 0 || echo 1)"
# D's unit row (lazy workspace): ON - 0 at create, 0 after the non-splitting handle, exactly 1 after the splitting one; OFF - 0 throughout
row "ON: workspace allocations 0 / 0 / 1 (create, non-splitting handle, splitting handle)" "$([ "$(has "$ON" 'ws allocs after cublasCreate x2: 0')" = 1 ] && [ "$(has "$ON" "ws allocs after the non-splitting handle's GEMMs: 0")" = 1 ] && [ "$(has "$ON" "ws allocs after the splitting handle's 3 GEMMs: 1")" = 1 ] && echo 0 || echo 1)"
row "OFF: workspace allocations 0 / 0 / 0" "$([ "$(has "$OFF" 'ws allocs after cublasCreate x2: 0')" = 1 ] && [ "$(has "$OFF" "ws allocs after the non-splitting handle's GEMMs: 0")" = 1 ] && [ "$(has "$OFF" "ws allocs after the splitting handle's 3 GEMMs: 0")" = 1 ] && echo 0 || echo 1)"
row "DEFAULT (knob unset): ran, and its launch trace equals the ON arm's" "$([ $rc_def = 0 ] && [ "$(printf '%s\n' "$DEF" | grep '^\[trace\] launch' | sed 's/ -> 0x[0-9a-f]*//')" = "$(printf '%s\n' "$ON" | grep '^\[trace\] launch' | sed 's/ -> 0x[0-9a-f]*//')" ] && [ "$(has "$DEF" '[trace] launch tinyblas_gemm_bf16_tc2k')" = 3 ] && echo 0 || echo 1)"
[ $fails = 0 ] || { printf '%s\n' "--- ON arm output:" "$ON" "--- OFF arm output:" "$OFF" | head -60; }
echo "splitk_check: $fails failure(s)"
exit $fails
