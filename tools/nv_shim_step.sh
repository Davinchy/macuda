#!/bin/sh
# nv_shim_step.sh — ONE hardware step through the CUDA shim, under the protocol in docs/HANDOFF.md §3:
#   preflight (read-only) → gpu-lock acquire → the step, its own process, log captured → quiesce cold → release.
#
#   sh tools/nv_shim_step.sh <A|B> opverify                      value-checked op-verify, expect 450/450
#   sh tools/nv_shim_step.sh <A|B> bench   <model.gguf> [n=128]  llama-bench: -ngl 99 -p 256 -n <n> -r 3
#   sh tools/nv_shim_step.sh <A|B> ops     <OP[,OP..]|all> [params regex]   test-backend-ops on those ops (edge-case sweeps);
#                                                   prints pass counts and the first failing tests
#   sh tools/nv_shim_step.sh <A|B> simple  <model.gguf> [n=64] [prompt]   greedy decode; output kept for a byte-compare
#   sh tools/nv_shim_step.sh <A|B> spec    <model.gguf> <draft.gguf> [n=128] [draft_n_max=3] [prompt]   speculative decode
#                                                   (llama-speculative-simple, --spec-type ${SPEC_TYPE:-draft-mtp}, extra flags in SPEC_ARGS); prints accept rate + t/s
#   sh tools/nv_shim_step.sh <A|B> sd      <model> [steps=20] [prompt] [extra sd-cli args...]   image generation (sd-cli-null);
#                                                   writes images/sd-<ts>.png and prints the stage timings; SD_ARGS for more flags;
#                                                   SD_MODEL_FLAG=--diffusion-model for split models (then pass --vae/--llm as extra args)
#   sh tools/nv_shim_step.sh <A|B> probe   <binary> [args...]          any driver-level probe binary, under the protocol
#   sh tools/nv_shim_step.sh <A|B> gap     [cubin]                    B's test_hw_gap: the space between kernels; knobs
#                                                   TINYNV_GAP_MODULES=N (distinct modules) TINYNV_GAP_RUN=R (launches per module)
#   sh tools/nv_shim_step.sh <A|B> fault   unmapped|trap [gap] [cubin]  DELIBERATELY faults the card (B's test_hw_fault);
#                                                   last slot of a session, Antonio's go for it by name, quiesced after
#
# Knobs pass through the environment exactly as the binaries read them (TINYNV_ASYNC, TINYNV_SYNC, TINYNV_ARENA_VRAM,
# TINYNV_CHAIN_DEPTH); the startup line that records what a run actually used is echoed back, so the report never
# depends on what anyone remembers exporting. BIN=<dir> picks the binary set (default build/bin, the validated one);
# the build id baked into the binary is printed BEFORE the card is touched. QUIESCE=1 quiesces cold after the step (default: idle warm);
# DRY=1 runs on the null device with no preflight/lock/quiesce, to test this script itself.
# Driver test steps (gap, fault) take their binary from TREE=<worktree> (default: the main checkout) or TESTBIN=<path>.
set -u
R=${EGPU_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}; S=$R/cuda-shim; T=${TREE:-$S}; BIN=${BIN:-$S/build/bin}; SOCK=${TINYNV_SOCKET:-${TMPDIR}tinygpu.sock}
who=${1:?A|B}; step=${2:?opverify|bench|simple}; shift 2
mkdir -p $R/logs; ts=$(date +%Y%m%d-%H%M%S); log=$R/logs/shim-$step-$ts.log
out=
case "$step" in
  opverify) bin=$BIN/test-backend-ops-null; set -- -b CUDA0 -o GATED_DELTA_NET -o GATED_LINEAR_ATTN -o SSM_SCAN -o SSM_CONV -o SOLVE_TRI -o RMS_NORM -o ROPE ;;
  bench)    bin=$BIN/llama-bench-null;      m=${1:?model}; n=${2:-128}; set -- -m "$m" -ngl 99 -p 256 -n "$n" -r 3 ;;
  ops)      bin=$BIN/test-backend-ops-null; ops=${1:?ops}; rx=${2:-}; set -- ${OPS_MODE:-test} -b CUDA0   # OPS_MODE=perf for throughput
            [ "$ops" != all ] && set -- "$@" -o "$ops"        # one comma-separated argument, as the suite wants it
            [ -n "$rx" ] && set -- "$@" -p "$rx"; m="ops-$(echo "$ops" | cut -c1-24 | tr ',' '+')" ;;
  simple)   bin=$BIN/llama-simple-null;     m=${1:?model}; n=${2:-64}; p=${3:-"The three most important things to know about the Thunderbolt bus are"}; set -- -m "$m" -ngl 99 -n "$n" "$p" ;;
  spec)     bin=$BIN/llama-speculative-simple-null; m=${1:?model}; md=${2:?draft model}; n=${3:-128}; dn=${4:-3}
            p=${5:-"The three most important things to know about the Thunderbolt bus are"}
            set -- -m "$m" -md "$md" -ngl 99 -ngld 99 --spec-type "${SPEC_TYPE:-draft-mtp}" --spec-draft-n-max "$dn" -n "$n" -p "$p" ${SPEC_ARGS:-} ;;
  sd)       bin=$BIN/sd-cli-null; m=${1:?model}; st=${2:-20};
            # the first sampling step uploads the mmapped weights, so a model file that has dropped out of the page cache makes
            # that step run at SSD speed (7-12 s instead of 1-2, seen 2026-09-15 00:26-01:05); read it here first and say how long it took,
            # so the number can be read for what it is (under ~2 s = it was cached; the run itself starts warm either way)
            t0=$(date +%s); cat "$m" > /dev/null 2>&1; echo "   pre-read $(basename "$m") ($(du -h "$m" | cut -f1)) in $(( $(date +%s) - t0 )) s before the run"
 p=${3:-"a photograph of a red fox sitting on a mossy log in a sunlit forest, detailed fur"}; shift 3 2>/dev/null || shift $#
            mkdir -p $R/images; png=$R/images/sd-$ts.png; set -- ${SD_MODEL_FLAG:--m} "$m" -p "$p" --steps "$st" -o "$png" -W "${SD_W:-512}" -H "${SD_H:-512}" -s "${SD_SEED:-42}" ${SD_ARGS:-} "$@" ;;
  probe)    bin=${1:?binary}; shift; test -x "$bin" || { echo "no binary at $bin"; exit 2; }   # any driver-level probe (C's DMA-segment probe etc.)
            export TINYNV_HW=1 ;;
  gap)      bin=${TESTBIN:-$T/build/shim/nv/test_hw_gap}
            tree=$(cd "$(dirname "$bin")/../../.." && pwd); cubin=${1:-$tree/spike/vecadd.sm120.cubin}
            test -f "$cubin" || { echo "no cubin at $cubin"; exit 2; }
            export TINYNV_HW=1; set -- "$cubin" ;;
  fault)    bin=${TESTBIN:-$T/build/shim/nv/test_hw_fault}; mode=${1:?unmapped|trap}; gap=${2:-67108864}
            tree=$(cd "$(dirname "$bin")/../../.." && pwd)     # the worktree that built it owns the spike cubins
            cubin=${3:-$tree/spike/$([ "$mode" = trap ] && echo trap_kernel || echo vecadd).sm120.cubin}
            test -f "$cubin" || { echo "no cubin at $cubin (the trap one is build output: python tools/build_spike_cubin.py ../spike/trap_kernel.cu in libtinynv)"; exit 2; }
            echo "   NOTE: this step faults the card on purpose ($mode); it needs Antonio's go for it by name"
            export TINYNV_HW=1; set -- "$mode" "$gap" "$cubin" ;;
  *) echo "unknown step $step"; exit 2 ;;
esac
test -x "$bin" || { echo "no binary at $bin"; exit 2; }
# which commit the binary carries, checked against the branch tips rather than guessed from any 7-hex string in it
id=unknown; for br in main async-default fault-report phase2-vm sigbus; do n=0
  for h in $(git -C $S log --format=%h -8 "$br" 2>/dev/null); do
    for v in "$h" "$h-dirty"; do [ "$id" = unknown ] && strings "$bin" | grep -qx "$v" && { [ $n = 0 ] && id="$v ($br tip)" || id="$v ($br~$n)"; }; done
    n=$((n+1)); done; done
sdk=$(otool -l "$bin" 2>/dev/null | awk '/LC_BUILD_VERSION/{f=1} f&&$1=="sdk"{print $2; exit}')
echo "== $step  $(date '+%F %T')  binary $bin  build id $id  sdk ${sdk:-?}  log $log"
case "$id" in unknown|*dirty*) echo "   WARNING: binary is not a clean build of a recent main or async-default commit"; esac
env | grep -E '^TINYNV_' | sed 's/^/   env /' ; true
if [ "${DRY:-0}" = 1 ]; then
  echo "   DRY=1: null device, no preflight, no lock, no quiesce$([ "$step" = fault ] && echo '; test_hw_fault refuses to run without a socket, so this only checks the wiring')"
  unset TINYNV_SOCKET
else
  sh $R/tools/preflight.sh | tail -1 | grep -q "VERDICT: OK" || { echo "preflight ABORT — not touching the GPU"; exit 1; }
  # the lock first (held by THIS runner's pid, refused to everyone else including this session's other runs), the server second
  sh $R/tools/gpu-lock.sh acquire "$who" "$step $(basename "${m:-}" .gguf) $ts" "$$" || exit 1
  sh $R/tools/tinygpu-server.sh ensure || { sh $R/tools/gpu-lock.sh release "$who" >/dev/null; exit 1; }
  export TINYNV_SOCKET=$SOCK
fi
start=$(date +%s)
# GMALLOC=1 runs the step under guard malloc (every allocation on its own page, a guard page after it, freed pages unmapped:
# a processor write past a buffer or into a freed one faults AT THE WRITER). It has to be set HERE: dyld strips DYLD_* from
# the environment of platform binaries such as /bin/sh, so a value exported by the caller never reaches the step.
if [ "${GMALLOC:-0}" = 1 ]; then export DYLD_INSERT_LIBRARIES=/usr/lib/libgmalloc.dylib; echo "   GMALLOC=1: guard malloc inserted for the step (expect it 10-50x slower)"; fi
if [ "$step" = simple ]; then
  # llama-simple writes the prompt and the generated tokens to stdout and everything else to stderr: keep them apart,
  # so the text file is the text and nothing else, and the byte-compare between runs compares generations only
  out=$R/logs/shim-simple-$ts.txt; "$bin" "$@" > "$out" 2> "$log"; rc=$?
else
  "$bin" "$@" > "$log" 2>&1; rc=$?
fi
echo "   exit $rc after $(( $(date +%s) - start )) s"
if [ $rc -gt 128 ]; then
  echo "   CRASHED: signal $((rc-128)); newest crash report: $(ls -t ~/Library/Logs/DiagnosticReports/$(basename "$bin")-*.ips 2>/dev/null | head -1)"
fi
# a step that never reached the card is not a result: say so and fail it, rather than let a CPU-only "pass" count
v=$(grep -m1 -oE 'found 0 CUDA devices|no server on [^ ]*: [A-Za-z ]+|is already mapped|Connection refused' "$log" ${out:+"$out"} 2>/dev/null | head -1)
if [ -n "$v" ]; then echo "   VOID: the step did not have the card to itself ($v) — result discarded"; [ $rc -eq 0 ] && rc=97; fi
# what the run says it was, and what it did
grep -h -E 'libtinynv build|tinynv: (submitting|sync|async|command arena|chain)|went backwards|is wrong:|not dispatched|write pointer read back' "$log" ${out:+"$out"} 2>/dev/null | head -12 | cut -c1-200 | sed 's/^/   /'
case "$step" in
  opverify) grep -E 'tests passed|Backend CUDA0|FAIL' "$log" | grep -v '^\[' | tail -4 | sed 's/^/   /' ;;
  bench)    grep -E '\| *(pp|tg)[0-9]+ *\|' "$log" | sed 's/^/   /' ;;
  ops)      grep -E 'tests passed|Backend CUDA0' "$log" | grep -v '^\[' | tail -3 | sed 's/^/   /'
            [ "${OPS_MODE:-test}" = perf ] && grep -E 'FLOPS|GB/s|us/run' "$log" | sed -E 's/\x1b\[[0-9;]*m//g' | cut -c1-170 | sed 's/^/   /' | head -40
            nf=$(grep -c 'FAIL' "$log"); echo "   failing lines: $nf"; grep -E 'FAIL|not supported|unsupported|NMSE|ERR' "$log" | grep -v -E '^\s*$' | head -25 | cut -c1-170 | sed 's/^/   /' ;;
  spec)     grep -E 'n_draft|n_predict|n_drafted|n_accept|accept|drafted|decoded|t/s|tokens per second|eval time|speculative' "$log" | grep -v -E '^(llama_model_loader|load_tensors|print_info)' | tail -14 | cut -c1-160 | sed 's/^/   /' ;;
  sd)       grep -E 'completed|taking|loading|error|unsupported|NaN|nan|lora|LoRA' "$log" | grep -v -E '^\[tinycudart\]' | tail -14 | cut -c1-160 | sed 's/^/   /'
            [ -f "$png" ] && echo "   image -> $png ($(stat -f %z "$png") bytes)" || echo "   NO IMAGE written" ;;
  probe)    grep -v -E '^(libtinynv|tinynv|\[tinycudart\])' "$log" | tail -20 | cut -c1-160 | sed 's/^/   /' ;;
  gap)      grep -E 'launching|us/launch|^ +[0-9]+ +[0-9]+ +[0-9.]+ +[0-9.]+|fitted|gap between|memory rate|cannot|: [A-Z_]+$' "$log" | head -16 | cut -c1-160 | sed 's/^/   /' ;;
  fault)    grep -E 'storing to|mmu fault|faulted on|SM [0-9]+:|no mmu fault|did not answer|PASS|FAIL|pass|fail|completed|try a larger gap|cannot read' "$log" | head -14 | cut -c1-200 | sed 's/^/   /' ;;
  simple)   grep -E 'decoded [0-9]+ tokens' "$log" "$out" | sed -E 's/^[^:]*://; s/^/   /' | head -2
            echo "   text -> $out ($(wc -c < "$out" | tr -d ' ') bytes, sha $(shasum "$out" | cut -c1-12))"
            echo "   $(head -c 300 "$out" | tr '\n' ' ')" ;;
esac
if [ "${DRY:-0}" != 1 ]; then
  # Default since 2026-09-14 evening: leave the firmware resident ("idle warm"). With GSP halted the AORUS box runs its fans at
  # fail-safe full speed (Antonio), and the driver resets a warm card itself on open (B). QUIESCE=1 restores the cold state.
  [ "${QUIESCE:-0}" = 1 ] && sh $R/tools/nv_quiesce.sh 2>&1 | sed 's/^/   /'
  sh $R/tools/gpu-lock.sh release "$who" >/dev/null
fi
exit $rc
