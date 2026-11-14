#!/bin/sh
# rtx4090-live.sh — supervised live runner for the AD102 tensor-unlock experiment.
#
#   sh tools/rtx4090-live.sh preflight   # read-only: system preflight + server ensure + probe preflight
#   sh tools/rtx4090-live.sh unlock      # the ONE supervised transaction (plan Phase 4), after a power cycle
#   sh tools/rtx4090-live.sh gemm [stock|unlocked|any]   # runtime handoff + BF16 GEMM
#
# Read docs/driver/rtx4090-tensor-unlock/spec.md and tools/rtx4090-probe/README.md
# first. The installed TinyGPU app/server/dext are used unchanged.
#
# Protocol enforced here: read-only preflight -> exclusive gpu-lock (the probe acquires it; this runner refuses
# to start if it is already held) -> ONE GPU process -> controlled release. The enclosure must be power-cycled
# (physical off >= 30 s) before `unlock`; the runner asks the operator to confirm that, because no software
# operation clears a retained override (FLR keeps it).
#
# NO KILLING TIMEOUT: the probe is NEVER wrapped in timeout(1) or a harness kill. If firmware may still DMA, the
# probe keeps the process/socket/mappings alive until the operator powers the enclosure off and confirms
# ('off' + Enter, or SIGUSR1 to the pid it prints). Killing it force-unmaps under a possibly-live firmware
# (transport-audit O14 — the documented crash limitation). Run `unlock` from an interactive terminal.
R=${EGPU_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
cd "$R" || exit 2
MODE=${1:-}
case "$MODE" in preflight|unlock|gemm) ;; *) echo "usage: $0 preflight|unlock|gemm [stock|unlocked|any]"; exit 2 ;; esac
EXPECT=${2:-any}
if [ "$MODE" = gemm ]; then case "$EXPECT" in stock|unlocked|any) ;; *) echo "usage: $0 gemm [stock|unlocked|any]"; exit 2 ;; esac; fi

PROBE=tools/rtx4090-probe/build/rtx4090-probe
BUNDLE=${TINYNV_UNLOCK_BUNDLE:-$R/cuda-shim/libtinynv/third_party/unlock-610.43.03}
LOGDIR=$R/logs; mkdir -p "$LOGDIR"
LOG=$LOGDIR/rtx4090-live-$MODE-$(date +%Y%m%d-%H%M%S).log

echo "== rtx4090-live $MODE: $(date '+%F %T') =="
if [ "$MODE" = gemm ]; then
  echo "-- build the BF16 handoff test (shim libraries must already be built for this card) --"
  make -C cuda-shim/libtinycublas/test || { echo "ABORT: test build failed; build the shim first:"; \
    echo "  cd cuda-shim && make BUILD=build/shim-sm_89 CUBIN_ARCH=sm_86"; exit 2; }
else
  echo "-- build + pin check --"
  make -C tools/rtx4090-probe >/dev/null || { echo "ABORT: probe build failed"; exit 2; }
  make -C tools/rtx4090-probe check || { echo "ABORT: C asset pins drifted from the manifest"; exit 2; }

  echo "-- bundle validation (offline, fail-closed) --"
  python3 tools/rtx4090-unlock/validate_bundle.py "$BUNDLE" --quiet || {
    echo "ABORT: bundle invalid or missing; prepare it: python3 tools/rtx4090-unlock/prepare_bundle.py"; exit 2; }
fi

echo "-- system preflight (read-only) --"
sh tools/preflight.sh || { echo "ABORT: tools/preflight.sh said do not touch the GPU"; exit 2; }

echo "-- TinyGPU server --"
sh tools/tinygpu-server.sh ensure || { echo "ABORT: no usable TinyGPU server (busy or wedged; NOT replaced)"; exit 2; }

echo "-- exclusive lock --"
sh tools/gpu-lock.sh status
L=${GPU_LOCK:-$([ -n "${MACUDA_SHARED_ROOT:-}" ] && echo "$MACUDA_SHARED_ROOT/.gpu-lock" || echo "$R/.gpu-lock")}
if [ -f "$L" ]; then echo "ABORT: the GPU lock is held; one card, one client. Release it first (tools/gpu-lock.sh)."; exit 2; fi

if [ "$MODE" = unlock ]; then
  [ -t 0 ] || { echo "ABORT: unlock needs an interactive terminal (the recovery hold reads 'off' from stdin)"; exit 2; }
  echo
  echo "Before the transaction:"
  echo "  * the enclosure was POWER-CYCLED (physical off >= 30 s) — the only verified stock-clearing operation;"
  echo "  * the laptop is on its own power adapter (project README);"
  echo "  * no other client owns the card; the probe takes the exclusive lock itself."
  printf "Confirm the enclosure was power-cycled and you are supervising this run [yes/N]: "
  read -r answer
  [ "$answer" = yes ] || { echo "not confirmed; nothing was launched"; exit 2; }
fi

echo "-- $MODE (log: $LOG) --"
if [ "$MODE" = unlock ]; then
  echo "If a RECOVERY REQUIRED banner appears: POWER OFF THE ENCLOSURE, then type 'off' + Enter (or SIGUSR1)."
  echo "Do NOT kill the probe; there is no timeout here by design."
fi
RCFILE=$(mktemp)
if [ "$MODE" = gemm ]; then
  # The gemm test is a normal macuda client: it boots the UNCHANGED runtime firmware, verifies the unlock registers
  # post-boot and post-workload (read-only), runs the BF16 correctness matrix and one timed GEMM, and unloads cold
  # (TINYNV_UNLOAD=1, the chain-of-trust default). The runner holds the project lock around it; one GPU process.
  sh tools/gpu-lock.sh acquire rtx4090-gemm "bf16 handoff gemm (expect $EXPECT)" $$ || { echo "ABORT: could not take the GPU lock"; exit 2; }
  { TINYNV_SOCKET=${TINYNV_SOCKET:-${TMPDIR:-/tmp/}tinygpu.sock} TINYNV_UNLOAD=1 \
      cuda-shim/libtinycublas/test/bf16_unlock_gemm --expect "$EXPECT"; echo $? > "$RCFILE"; } 2>&1 | tee "$LOG"
  sh tools/gpu-lock.sh release rtx4090-gemm
else
  # the probe runs in the foreground with stdin/stdout on this terminal; tee only copies the transcript.
  # The rc is captured inside the pipeline group so it is the PROBE's exit code, not tee's.
  { "$PROBE" "$MODE" --bundle "$BUNDLE"; echo $? > "$RCFILE"; } 2>&1 | tee "$LOG"
fi
rc=$(cat "$RCFILE"); rm -f "$RCFILE"
echo "== $MODE exit $rc; transcript: $LOG =="
case "$rc" in
  0) if [ "$MODE" = preflight ]; then
       echo "read-only preflight done; nothing was applied. For the transaction: power-cycle the enclosure,"
       echo "then run: sh tools/rtx4090-live.sh unlock"
     elif [ "$MODE" = gemm ]; then
       echo "gemm: correctness and the expected register state verified; the card was unloaded cold."
     else
       echo "success (applied-and-verified or verified-existing). Next: the handoff card boots normal macuda"
       echo "firmware, re-verifies the override post-boot, then runs the known-correct BF16 GEMM."
     fi ;;
  1) if [ "$MODE" = gemm ]; then
       echo "gemm FAILED: a correctness case exceeded tolerance or the register state disagreed. The card was"
       echo "unloaded cold by the driver's normal exit path. Read the log; do not attribute it to the unlock"
       echo "until the stock baseline says otherwise."
     else
       echo "contained failure: buffers released, PCI command restored. Read the first-error line; do not loop"
       echo "the privileged transaction. A fresh attempt needs a fresh power cycle."
     fi ;;
  2) echo "setup/lock failure: nothing was launched." ;;
  3) echo "recovery was required and the power-off was confirmed. Keep the log; the next session starts from"
     echo "a fresh enclosure power cycle." ;;
esac
exit "$rc"
