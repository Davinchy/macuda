#!/bin/sh
# tools/ab_pair.sh OLD NEW — interleaved tg128 A/B of two cuda-shim commits, one binary each, same minutes, same host load.
#   The worktree cuda-shim-f must be at NEW (already gated); OLD is built clean in the worktree, NEW's bench binary is kept as a
#   copy, then A/B/A/B per model over two rounds; the worktree ends back at NEW, rebuilt clean. This is how "did commit X move the
#   decodes" is answered — never by comparing runs an hour apart (HANDOFF §1 precision note).
R=${EGPU_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}; S=${S:-$R/logs/gate-steps}; mkdir -p "$S"
cd "$R"; W=$R/cuda-shim-f; OLD=$1; NEW=$2
i=0; until sh tools/gpu-lock.sh status | grep -q 'GPU lock: free'; do sleep 20; i=$((i+1)); [ $i -ge 90 ] && exit 1; done
[ "$(git -C $W rev-parse --short HEAD)" = "$NEW" ] || { echo "worktree is not at $NEW"; exit 1; }
mkdir -p $S/bin_$NEW; cp $W/build/bin/llama-bench-null $S/bin_$NEW/; [ "$(strings $S/bin_$NEW/llama-bench-null | grep -c "^$NEW$")" = 1 ] || { echo "copied binary is not $NEW"; exit 1; }
git -C $W checkout -q --detach $OLD || exit 1
( cd $W && rm -rf build/shim/nv && make -s 2>&1 | grep -E 'error' | head -3; S=$W sh build/link-llama-bench.sh 2>&1 | tail -1 ) | grep -vE '^cc '
[ "$(strings $W/build/bin/llama-bench-null | grep -c "^$OLD$")" = 1 ] || { echo "worktree bench is not $OLD"; exit 1; }
echo "$(date +%H:%M:%S): binaries ready: A=$OLD (worktree) B=$NEW (copy); load $(sysctl -n vm.loadavg | awk '{print $2}')"
run() { BIN=$1 sh tools/nv_shim_step.sh A bench models/$2.gguf 128 > $S/w15.step 2>&1; L=$(ls -t logs/shim-bench-*.log | head -1); echo "$(date +%H:%M:%S) $3 $2: $(grep -E '\| *tg[0-9]+ *\|' "$L" | sed -E 's#^\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|##' | tr -s ' ') $(grep -oE 'exit [0-9]+ after [0-9]+ s|CRASHED[^ ]*|VOID:[^—]*' $S/w15.step | head -1) load $(sysctl -n vm.loadavg | awk '{print $2}')"; }
for r in 1 2; do for m in Qwen3.8-27B-UD-Q4_K_M Qwen3.5-35B-A3B-Q4_K_M; do run $W/build/bin $m "round $r A=$OLD"; run $S/bin_$NEW $m "round $r B=$NEW"; done; done
git -C $W checkout -q --detach $NEW; ( cd $W && rm -rf build/shim/nv && make -s 2>&1 | grep -E 'error' | head -3; S=$W sh build/link-llama-bench.sh 2>&1 | tail -1 ) | grep -vE '^cc '
echo "$(date +%H:%M:%S): AB_PAIR DONE ($OLD vs $NEW); worktree back at $(git -C $W rev-parse --short HEAD); $(sh tools/gpu-lock.sh status)"
