#!/bin/sh
# disagg-pdtest.sh — the discriminating run for the Thunderbolt-drop mechanism (V1, 2026-09-16). Runs ONLY on A's word.
#
# Both re-enumerations on 2026-09-16 (13:46:24.6, 15:29:05.5) sat on a full Metal prefill of this prompt while the
# MacBook was powered by the eGPU box's USB-C PD (96 W, unnamed) on the cable that carries the tunnel. The Mac is now on
# Apple's 140 W adapter on a separate port. This repeats the exact shape that dropped the link, with the card idle:
#   no re-enumeration -> the PD hypothesis holds (A relaxes the host rule to "no Metal prefills on the box's PD")
#   a re-enumeration  -> back to the unfiltered kernel log
# Two runs (A: n=2 against the two events), readings before, between and after. Each maps 49.6 GB and holds the Apple
# GPU at full load for ~35 s: a >20 GB job under A's rule, so: A's word first.
#   sh tools/disagg-pdtest.sh            (from the macuda root; no card is opened; nothing needs env.sh)
set -u
R=${EGPU_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}; cd "$R" || exit 2
ts=$(date +%Y%m%d-%H%M%S); LOG=$R/logs/disagg/pdtest-$ts.log; mkdir -p "$R/logs/disagg"
dext()  { pgrep -f '^/Library/SystemExtensions/[^ ]*/org\.tinygrad\.tinygpu\.driver2' | wc -l | tr -d ' '; }
newest(){ ps -axo lstart=,command= | /usr/bin/grep '/Library/SystemExtensions/[^ ]*/org.tinygrad.tinygpu.driver2' | /usr/bin/grep -v grep | python3 -c 'import sys,time
b=None
for l in sys.stdin:
    t=time.strptime(" ".join(l.split()[:5]),"%a %b %d %H:%M:%S %Y"); b=t if b is None or t>b else b
print(time.strftime("%Y-%m-%d %H:%M:%S",b) if b else "none")'; }   # youngest instance by lstart, parsed (a text sort is wrong across days)
dart()  { ioreg -l -w0 2>/dev/null | grep -q 'pci-dart-error-data' && echo "present" || echo "absent"; }
power() { pmset -g ac 2>/dev/null | awk -F'= ' '/Wattage|Adapter Name|Manufacturer/{printf "%s; ", $2}'; }
{
echo "== V1 PD test $(date '+%F %T')  log $LOG"
echo "POWER: $(power)"
echo "BEFORE: dext instances $(dext) (newest started $(newest)), DART error data $(dart), $(date '+%T')"
for run in 1 2; do
  echo "== run $run of 2: one full Metal prefill of prompt-24k (baseline shape), card idle =="
  NPROBS=0 sh tools/disagg-decode.sh baseline "$R/logs/disagg/prompt-24k.txt" 16; rc=$?
  echo "METAL run $run exit $rc"
  sleep 5   # the 13:46 event fired the second the server exited; give a post-teardown drop time to show
  [ $run = 1 ] && echo "BETWEEN: dext instances $(dext) (newest started $(newest)), DART error data $(dart), $(date '+%T')"
done
echo "AFTER: dext instances $(dext) (newest started $(newest)), DART error data $(dart), $(date '+%T')"
echo "POWER: $(power)"
echo "DONE-PDTEST $(date '+%F %T')"
} 2>&1 | tee "$LOG"
