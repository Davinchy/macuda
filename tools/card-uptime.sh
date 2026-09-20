#!/bin/sh
# card-uptime.sh — how long the card has stayed up, sampled while it is WORKING (2026-09-19).
#
#   sh tools/card-uptime.sh sample            one row into $OUT
#   sh tools/card-uptime.sh watch [seconds]   sample forever (default every 300 s)
#   sh tools/card-uptime.sh report            what the rows add up to
#
# Everything here is read-only and card-free: ps, ioreg and the lock file. Nothing opens the device, because the
# point is to measure a card that is busy with something else - a sampler that had to open it would be measuring
# its own interference, and on this stack a second opener is how you get two drivers on one card.
#
# The clock is the DEXT INSTANCE's start time. The system extension process is restarted when the card is
# re-enumerated - a replug, a sleep/wake, a bus reset - so "same dext pid, same start time" is exactly the claim
# "the card has not gone away since", which is the thing worth timing. A changed start time is a re-enumeration and
# the clock restarts; preflight refuses on the same signal, for the same reason.
set -u
R=${EGPU_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
OUT=${OUT:-/Volumes/Crucial_8TB/disagg-bench/card-uptime.tsv}
L=${GPU_LOCK:-$([ -d /Volumes/512SSD/EGPU ] && echo /Volumes/512SSD/EGPU/.gpu-lock || echo $R/.gpu-lock)}
mkdir -p "$(dirname "$OUT")"

dext_started() {   # epoch seconds of the newest dext instance, 0 if the driver is not loaded
  newest=0
  for dp in $(pgrep -f '^/Library/SystemExtensions/[^ ]*/org\.tinygrad\.tinygpu\.driver2' 2>/dev/null); do
    ls_=$(ps -p "$dp" -o lstart= 2>/dev/null | sed 's/^ *//'); [ -n "$ls_" ] || continue
    ep=$(date -j -f "%a %b %d %H:%M:%S %Y" "$ls_" +%s 2>/dev/null || echo 0)
    [ "$ep" -gt "$newest" ] && { newest=$ep; echo "$dp" > /tmp/.card-uptime-pid; }
  done
  echo "$newest"
}

sample() {
  now=$(date +%s); started=$(dext_started); pid=$(cat /tmp/.card-uptime-pid 2>/dev/null || echo 0)
  up=0; [ "$started" -gt 0 ] && up=$((now - started))
  dart=clean; ioreg -l -w0 2>/dev/null | grep -q 'pci-dart-error-data' && dart=PRESENT
  if [ -f "$L" ]; then lock=$(sed -n 's/.*what=\([^ ]*\).*/\1/p' "$L" | tr -d '\n'); lock=${lock:-held}; else lock=free; fi
  busy=0; pgrep -qf 'llama-server-null|llama-bench-null|sd-cli-null|llama-speculative-simple-null' && busy=1
  [ -f "$OUT" ] || printf 'when\tuptime_s\tdext_pid\tdext_started\tdart\tlock\tcard_busy\n' > "$OUT"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -Iseconds)" "$up" "$pid" "$started" "$dart" "$lock" "$busy" >> "$OUT"
}

report() {
  [ -f "$OUT" ] || { echo "no samples at $OUT"; exit 1; }
  # macOS awk has no strftime, so the one date it needs is formatted here and passed in
  st=$(tail -1 "$OUT" | cut -f4); st_h=$(date -r "${st:-0}" '+%F %T' 2>/dev/null || echo "?")
  awk -F'\t' -v st_h="$st_h" 'NR>1 {
      n++; if (!first) { first=$1; start0=$4 } last=$1; up=$2; startN=$4; pid=$3
      if ($4 != start0 && $4 != 0 && start0 != 0) reenum++
      if ($5 != "clean") faults++
      if ($7 == 1) busy++
    } END {
      if (!n) { print "no samples"; exit }
      h = int(up/3600); m = int((up%3600)/60)
      printf "card up %d h %02d min, unbroken (dext pid %s since %s)\n", h, m, pid, st_h
      printf "%d samples between %s and %s\n", n, first, last
      printf "%d re-enumerations, %d DART faults, %d samples (%.0f%%) with a card process live\n",
             reenum+0, faults+0, busy+0, 100*busy/n
    }' "$OUT"
}

case "${1:-report}" in
  sample) sample ;;
  watch)  iv=${2:-300}; echo "card-uptime: sampling every $iv s into $OUT"; while :; do sample; sleep "$iv"; done ;;
  report) report ;;
  *) echo "usage: $0 sample|watch [seconds]|report"; exit 2 ;;
esac
