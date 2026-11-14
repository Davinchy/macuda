#!/bin/sh
# Supervised GSP 610 runtime validation on an RTX 4090.
#
#   sh tools/gsp610-live.sh preflight
#   sh tools/gsp610-live.sh boot610
#   sh tools/gsp610-live.sh lifecycle610
#   sh tools/gsp610-live.sh lifecycle570
#
# The script uses one GPU process at a time, holds an exclusive lock, and never
# retries a failed 610 transaction or falls back to 570 behind the caller's back.
set -eu

R=${EGPU_ROOT:-$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)}
cd "$R"

MODE=${1:-}
case "$MODE" in
    preflight|boot610|lifecycle610|lifecycle570) ;;
    *)
        echo "usage: $0 preflight|boot610|lifecycle610|lifecycle570" >&2
        exit 2
        ;;
esac

BUNDLE=${GSP610_BUNDLE:-$R/cuda-shim/build/gsp610-runtime/bundle}
FW570=${TINYNV_FW570_DIR:-$R/cuda-shim/libtinynv/third_party/firmware}
NVBUILD=${TINYNV_NV_BUILD:-$R/cuda-shim/build/gsp610-nv}
SHIMBUILD=${TINYNV_SHIM_BUILD:-$R/cuda-shim/build/gsp610-shim}
WORKLOAD=${TINYNV_WORKLOAD:-$NVBUILD/test_hw}
LOGDIR=${GSP610_LOG_DIR:-$SHIMBUILD/live}
TS=$(date +%Y%m%d-%H%M%S)
SOCK=${TINYNV_SOCKET:-${TMPDIR:-/tmp/}tinygpu.sock}
SESSION=${GSP610_SESSION:-gsp610-live}
REPS=${REPS:-5}
LOCK=${GPU_LOCK:-$R/.gpu-lock}

mkdir -p "$LOGDIR"

echo "== gsp610-live $MODE: $(date '+%F %T') =="
echo "   nvbuild=$NVBUILD"
echo "   shim=$SHIMBUILD"
echo "   reps=$REPS"

echo "-- runtime bundle validation --"
python3 "$R/tools/gsp610-runtime/validate_runtime.py" "$BUNDLE"

echo "-- read-only system preflight --"
sh "$R/tools/preflight.sh"

if [ "$MODE" = preflight ]; then
    echo "-- TinyGPU server status --"
    sh "$R/tools/tinygpu-server.sh" status
    echo "-- GPU lock status --"
    sh "$R/tools/gpu-lock.sh" status
    echo "preflight complete; no GPU work was submitted"
    exit 0
fi

[ -d "$FW570/nvidia" ] || {
    echo "missing 570 firmware root: $FW570" >&2
    exit 2
}
[ -x "$NVBUILD/test_hw" ] || {
    echo "missing driver test: $NVBUILD/test_hw" >&2
    exit 2
}
if [ "$MODE" != boot610 ]; then
    [ -x "$WORKLOAD" ] || {
        echo "missing workload: $WORKLOAD" >&2
        exit 2
    }
fi

echo "-- TinyGPU server --"
sh "$R/tools/tinygpu-server.sh" ensure

echo "-- acquire exclusive GPU lock --"
if [ -f "$LOCK" ]; then
    echo "GPU lock is already held: $LOCK" >&2
    exit 2
fi
sh "$R/tools/gpu-lock.sh" acquire "$SESSION" "supervised GSP 610 $MODE" $$
release_lock() {
    sh "$R/tools/gpu-lock.sh" release "$SESSION" >/dev/null 2>&1 || true
}
trap release_lock EXIT
trap 'exit 130' INT TERM

run_step() {
    name=$1
    log=$2
    shift 2
    echo "-- $name --"
    start=$(date +%s)
    set +e
    "$@" >"$log" 2>&1
    rc=$?
    set -e
    end=$(date +%s)
    cat "$log"
    echo "-- $name exit $rc, wall $((end - start)) s, log $log --"
    return "$rc"
}

accept_boot_exit() {
    log=$1
    rc=$2
    grep -q 'GSP-RM is up' "$log" || {
        echo "boot did not reach GSP-RM; see $log" >&2
        return 1
    }
    grep -q 'the card is cold' "$log" || {
        echo "boot did not prove a cold unload; see $log" >&2
        return 1
    }
    if [ "$rc" -ne 0 ]; then
        failures=$(grep -c 'FAIL' "$log" || true)
        bus_only=$(grep 'FAIL' "$log" | grep -c 'bus mastering already off' || true)
        [ "$rc" -eq 1 ] && [ "$failures" -gt 0 ] && [ "$failures" -eq "$bus_only" ] || {
            echo "test_hw failed for a reason other than the documented bus-mastering evidence check; see $log" >&2
            return 1
        }
        echo "note: test_hw exit 1 is limited to its bus-mastering evidence check"
    fi
}

run_init() {
    profile=$1
    log="$LOGDIR/$TS-$profile-init.log"
    if [ "$profile" = 610 ]; then
        set +e
        run_step "$profile init probe" "$log" \
            env -u TINYNV_FAILPOINT -u TINYNV_UNLOCK_BUNDLE \
                TINYNV_TENSOR_UNLOCK=0 TINYNV_SOCKET="$SOCK" \
                TINYNV_HW=1 TINYNV_UNLOAD=1 TINYNV_GSP_RUNTIME=610.43.03 \
                TINYNV_FW_DIR="$BUNDLE" "$NVBUILD/test_hw" "$SOCK"
        rc=$?
        set -e
    else
        set +e
        run_step "$profile init probe" "$log" \
            env -u TINYNV_GSP_RUNTIME -u TINYNV_FAILPOINT -u TINYNV_UNLOCK_BUNDLE \
                TINYNV_TENSOR_UNLOCK=0 TINYNV_SOCKET="$SOCK" \
                TINYNV_HW=1 TINYNV_UNLOAD=1 TINYNV_FW_DIR="$FW570" \
                "$NVBUILD/test_hw" "$SOCK"
        rc=$?
        set -e
    fi
    accept_boot_exit "$log" "$rc"
}

run_cycle() {
    profile=$1
    cycle=$2
    log="$LOGDIR/$TS-$profile-cycle$cycle.log"
    if [ "$profile" = 610 ]; then
        run_step "$profile cycle $cycle/$REPS" "$log" \
            env -u TINYNV_FAILPOINT -u TINYNV_UNLOCK_BUNDLE \
                TINYNV_TENSOR_UNLOCK=0 TINYNV_SOCKET="$SOCK" \
                TINYNV_HW=1 TINYNV_UNLOAD=1 TINYNV_GSP_RUNTIME=610.43.03 \
                TINYNV_FW_DIR="$BUNDLE" "$WORKLOAD" "$SOCK"
    else
        run_step "$profile cycle $cycle/$REPS" "$log" \
            env -u TINYNV_GSP_RUNTIME -u TINYNV_FAILPOINT -u TINYNV_UNLOCK_BUNDLE \
                TINYNV_TENSOR_UNLOCK=0 TINYNV_SOCKET="$SOCK" \
                TINYNV_HW=1 TINYNV_UNLOAD=1 TINYNV_FW_DIR="$FW570" \
                "$WORKLOAD" "$SOCK"
    fi
    grep -q 'the card is cold' "$log" || {
        echo "cycle did not prove a cold unload; see $log" >&2
        return 1
    }
}

if [ "$MODE" = boot610 ]; then
    log="$LOGDIR/$TS-610-boot.log"
    set +e
    run_step "610 boot probe" "$log" \
        env -u TINYNV_FAILPOINT -u TINYNV_UNLOCK_BUNDLE \
            TINYNV_TENSOR_UNLOCK=0 TINYNV_SOCKET="$SOCK" \
            TINYNV_HW=1 TINYNV_UNLOAD=1 TINYNV_GSP_RUNTIME=610.43.03 \
            TINYNV_FW_DIR="$BUNDLE" "$NVBUILD/test_hw" "$SOCK"
    rc=$?
    set -e
    accept_boot_exit "$log" "$rc"
    echo "610 boot and cold unload passed"
    exit 0
fi

case "$MODE" in
    lifecycle610)
        run_init 610
        profile=610
        ;;
    lifecycle570)
        run_init 570
        profile=570
        ;;
esac

cycle=1
while [ "$cycle" -le "$REPS" ]; do
    run_cycle "$profile" "$cycle"
    cycle=$((cycle + 1))
done

echo "-- post-session system preflight --"
sh "$R/tools/preflight.sh" | grep -iE 'DART|mmu|fault|VERDICT' || true
echo "$MODE passed $REPS complete cycles; logs are in $LOGDIR"
