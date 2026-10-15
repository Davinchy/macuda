#!/bin/sh
# disagg-slot.sh — V1's whole card window for the disaggregated-inference experiment, as ONE command, run only on A's
# word (A, 2026-09-16 ~14:40: effective when B's last probe read is logged AND the lock reads free; A confirms).
#
#   . ./env.sh && sh tools/disagg-slot.sh          (from the macuda root)
#
# Sequence, each card step its own process through the runner (preflight → lock as V1 → server → step → release):
#   0. refuse to start unless the shared lock reads free (never on a lock that is not ours; never a model load before it)
#   1. dext instance count + DART line, before
#   2. PREFILL on the card: 23,691 of 23,692 tokens, experts streamed from host mmap, state saved (tools/disagg-prefill.sh)
#   3. OP-VERIFY under V1: the health check A asked for after a first-of-its-kind 24,576-token ubatch
#   4. dext instance count + DART line, after
#   5. Metal restore + decode + token-by-token comparison against the Metal baseline (no card, but maps 49.6 GB of host
#      memory, so it stays inside the window: the window closes on the DONE line, not on the op-verify's release)
# Everything the report needs is printed with a leading tag (BEFORE/AFTER/PREFILL/OPVERIFY/RESTORE/COMPARE/DONE).
set -u
R=${EGPU_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}; cd "$R" || exit 2
ts=$(date +%Y%m%d-%H%M%S); LOG=$R/logs/disagg/slot-$ts.log; mkdir -p "$R/logs/disagg"
[ -n "${GPU_LOCK:-}" ] || { echo "GPU_LOCK is not exported: source env.sh first so the SHARED lock is used"; exit 2; }
dext() { pgrep -f '^/Library/SystemExtensions/[^ ]*/org\.tinygrad\.tinygpu\.driver2' | wc -l | tr -d ' '; }   # anchored to the executable: a loose -f match counts the counter itself (A, 15:25)
dart() { ioreg -l -w0 2>/dev/null | grep -q 'pci-dart-error-data' && echo "present" || echo "absent"; }
{
echo "== V1 slot $(date '+%F %T')  lock=$GPU_LOCK  log $LOG"
sh tools/gpu-lock.sh status | grep -q 'GPU lock: free' || { echo "ABORT: the lock is not free: $(sh tools/gpu-lock.sh status)"; exit 1; }
echo "BEFORE: dext instances $(dext), DART error data $(dart), $(date '+%T')"
echo "== step 2: prefill on the card =="
sh tools/nv_shim_step.sh V1 probe "$R/tools/disagg-prefill.sh" "$R/logs/disagg/prompt-24k.txt" coder-next-24k.bin; rc1=$?
echo "PREFILL step exit $rc1"
echo "== step 3: op-verify under V1 =="
sh tools/nv_shim_step.sh V1 opverify; rc2=$?
echo "OPVERIFY step exit $rc2"
echo "AFTER: dext instances $(dext), DART error data $(dart), $(date '+%T'); lock: $(sh tools/gpu-lock.sh status)"
if [ $rc1 -eq 0 ] && [ -f "$R/logs/disagg/coder-next-24k.bin" ]; then
  echo "== step 5: Metal restore + decode + compare (host only, 49.6 GB mapped) =="
  sh tools/disagg-decode.sh restore "$R/logs/disagg/prompt-24k.txt" 128 coder-next-24k.bin; rc3=$?
  echo "RESTORE step exit $rc3"
else echo "RESTORE skipped: prefill step failed or no state file"; rc3=1; fi
echo "DONE $(date '+%F %T'): prefill $rc1, opverify $rc2, restore $rc3; window closed"
} 2>&1 | tee "$LOG"
