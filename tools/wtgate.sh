#!/bin/sh
# tools/wtgate.sh — THE STANDARD GATE of cuda-shim main, run by Session A after every commit that lands there.
#   sh tools/wtgate.sh            gates main's tip: worktree cuda-shim-f checked out detached at main, driver built CLEAN
#                                 (rm -rf build/shim/nv), every null binary relinked and its build id checked (x1), B's
#                                 stale-object check, op-verify 450/450 at chain depths 32/64/128, 27B and MoE tg128 at the
#                                 defaults (empty environment), the 96-token greedy text byte-compared to logs/shim-simple-20260914-225831.txt.
#   Prints one line per step with the runner's exit code; read every line — a count of successes alone cannot tell "passed"
#   from "died". Step outputs go to $S (default: logs/gate-steps). Decode numbers carry the host state: see HANDOFF §1 precision note.
R=${EGPU_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}; S=${S:-$R/logs/gate-steps}; mkdir -p "$S"
cd "$R"
W=$R/cuda-shim-f; H=$(git -C cuda-shim rev-parse --short main)
git -C $W checkout -q --detach $H || { echo "checkout failed"; exit 1; }
echo "$(date '+%H:%M:%S'): worktree at $(git -C $W rev-parse --short HEAD) (main), dirty=$(git -C $W status --porcelain | wc -l | tr -d ' ')"
( cd $W && rm -rf build/shim/nv && make -s 2>&1 | grep -E 'error' | head -3
  for t in 'tests test-backend-ops' 'examples/simple llama-simple' 'examples/speculative-simple llama-speculative-simple' 'tools/server llama-server' 'tools/mtmd llama-mtmd-cli'; do d=${t%% *}; n=${t##* }; S=$W sh build/link-null.sh $d $n $W/build/bin/$n-null 2>&1 | grep -vE 'duplicate|warning' | tail -1; done
  S=$W sh build/link-llama-bench.sh 2>&1 | tail -1
  L=$R/stable-diffusion.cpp/build-null S=$W sh build/link-null.sh examples/cli sd-cli $W/build/bin/sd-cli-null 2>&1 | grep -vE 'duplicate|warning' | tail -1
  bad=0; for b in test-backend-ops llama-bench llama-simple llama-speculative-simple llama-server llama-mtmd-cli sd-cli; do c=$(strings build/bin/$b-null 2>/dev/null | grep -c "^$H$"); printf '%s x%s  ' "$b" "$c"; [ "$c" = 1 ] || bad=1; done; echo
  [ $bad = 0 ] && echo "BUILD OK $H (worktree)" || echo "BUILD ID MISMATCH (worktree)" ) 2>&1 | grep -vE '^cc '
strings $W/build/bin/test-backend-ops-null | grep -c "^$H\$" | grep -q '^1$' || { echo "BUILD NOT OK — stopping"; exit 1; }
# B's guard (3ae1b7f): GNU Make 3.81 compares mtimes at one-second granularity, so an object can be stale behind a fresh build id;
# the driver is built clean above (rm -rf build/shim/nv) and this check says so for the record
( cd $W && out=$(python3 libtinynv/tools/check_stale_objects.py build/shim/nv libtinynv 2>&1); rc=$?; echo "   stale-check (driver objects vs sources): exit $rc${out:+ — $(echo "$out" | tail -1)}" )
B=$W/build/bin
step() { TINYNV_CHAIN_DEPTH=$1 BIN=$B sh tools/nv_shim_step.sh A opverify > $S/w.step 2>&1
         grep -oE 'exit [0-9]+ after [0-9]+ s|CRASHED: signal [0-9]+|VOID:[^—]*|[0-9]+/[0-9]+ tests passed|preflight ABORT|LOCKED[^:]*' $S/w.step | tr '\n' ' '; echo; }
for d in 32 64 128; do printf '%s op-verify depth %-3s: ' "$(date '+%H:%M:%S')" "$d"; step $d; done
for m in Qwen3.8-27B-UD-Q4_K_M Qwen3.5-35B-A3B-Q4_K_M; do BIN=$B sh tools/nv_shim_step.sh A bench models/$m.gguf 128 > $S/w.step 2>&1; grep -E '\| *tg[0-9]+ *\|' "$(ls -t logs/shim-bench-*.log | head -1)" | sed -E "s#^\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|#   $m#" | cut -c1-80; grep -oE 'sensors[^.]{0,80}' "$(ls -t logs/shim-bench-*.log | head -1)" | head -1 | sed 's/^/      startup: /'; done
BIN=$B sh tools/nv_shim_step.sh A simple models/Qwen3.8-27B-UD-Q4_K_M.gguf 96 'The three laws of thermodynamics, explained for a bright twelve-year-old, are:' > $S/w.step 2>&1
t=$(ls -t logs/shim-simple-*.txt | head -1); cmp -s "$t" logs/shim-simple-20260914-225831.txt && echo "   text IDENTICAL to the 22:58 file" || echo "   text DIFFERS from the 22:58 file ($t)"
echo "$(date +%H:%M:%S): GATE DONE ($H)"; f=${TMPDIR}tinynv-sensors; echo "   sensor file after a default-env decode: $(stat -f %Sm -t %H:%M:%S $f 2>/dev/null) pid $(awk "/^pid/{print \$2}" $f 2>/dev/null)"; echo "$(date +%H:%M:%S): WTGATE DONE ($H)"; exit 0
