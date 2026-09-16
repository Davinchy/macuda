#!/bin/sh
R=${EGPU_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
# Read-only pre-flight for the AORUS RTX5090 AI BOX + TinyGPU dext. Prints a verdict. Touches nothing.
# An ABORT names every rule that tripped on the verdict line, so a check that aborts and later clears on its own leaves
# a reason behind rather than only a verdict (B, 2026-09-14: one that did not cost a minute; the next could cost a session).
abort=0; why=""
fail() { abort=1; why="${why:+$why; }$1"; }
echo "== $(date '+%F %T')  preflight =="
echo "-- Thunderbolt --"; system_profiler SPThunderboltDataType 2>/dev/null | awk '/AORUS/{f=1} f&&/Device Name|Mode|Speed|Link Status/{print "   "$0} f&&NR>200{exit}' | head -6
system_profiler SPThunderboltDataType 2>/dev/null | grep -q "AORUS" || { echo "   AORUS box NOT on the Thunderbolt bus"; fail "box not on the Thunderbolt bus"; }
echo "-- PCI --"; system_profiler SPPCIDataType 2>/dev/null | awk '/0x2b85/{f=1} f&&/Link|Driver Installed|Tunnel/{print "   "$0} /pci10de,22e8/{exit}'
system_profiler SPPCIDataType 2>/dev/null | grep -A12 "0x2b85" | grep -q "Link up" || { echo "   5090 link NOT up"; fail "PCI link not up"; }
echo "-- IOKit nub --"
nub=$(ioreg -l -w0 2>/dev/null | awk '/"device-id" = <852b0000>/{f=1} f{print; if(++n>70) exit}')
# the dart flag is thousands of hex digits and is reported below instead; dumping it here buries every other line
echo "$nub" | grep -E 'IOPCIResourced|IODEXTMatchCount' | sed 's/^ *[| ]*/   /'
echo "$nub" | grep -q '"IOPCIResourced" = Yes' || { echo "   IOPCIResourced missing"; fail "IOPCIResourced missing"; }
# The flag is latched until the device is re-enumerated, so an FLR leaves a reset, usable card still carrying it. A
# check that then aborts forever says the same thing whatever the truth is. tools/dart-stale.sh snapshots the exact bytes
# with a reason; a byte-identical flag is the one already accounted for, and anything else aborts as before.
if echo "$nub" | grep -q 'pci-dart-error-data'; then
  cur=$(echo "$nub" | grep -o '"pci-dart-error-data" = <[0-9a-f]*>' | head -1)
  if [ -f $R/.dart-stale ] && grep -qxF "$cur" $R/.dart-stale; then
    echo "   DART error data present, byte-identical to the snapshot: stale, not a new fault"
    sed -n 's/^why=/   reason: /p; s/^when=/   recorded: /p' $R/.dart-stale
  else
    echo "   DART ERROR DATA PRESENT"; fail "new DART error data"
  fi
fi
echo "-- dext / server processes --"
for p in $(pgrep -f "tinygpu.driver2|org.tinygrad.tinygpu"); do ps -p $p -o pid,state,%cpu,time,command | tail -1 | cut -c1-100 | sed 's/^/   /'; done
ps -axo pid,state,%cpu,time,command | grep -E "TinyGPU( |$)|TinyGPU server" | grep -v grep | cut -c1-100 | sed 's/^/   /'
# Wedged means stuck: a dext or server in uninterruptible wait, or spinning, and still so two seconds later. One sample
# alone also catches the server in the middle of a legitimate kernel call right after a step (it cleans a session up for
# a moment after the client disconnects), which is what aborted a step for about a minute on 2026-09-14 22:22.
stuck() { ps -axo state,%cpu,command | grep -E "tinygpu|TinyGPU" | grep -v grep | awk '$1 ~ /U/ || $2+0 > 90 {print "   " $0}'; }
s1=$(stuck)
if [ -n "$s1" ]; then
  sleep 2; s2=$(stuck)
  if [ -n "$s2" ]; then echo "$s2" | sed 's/^   /   WEDGED (still, after 2 s): /'; fail "dext or server wedged"
  else echo "$s1" | sed 's/^   /   busy for a moment, clear 2 s later: /'; fi
fi
echo "-- socket / lock --"; ls -la "$TMPDIR/tinygpu.sock" "$TMPDIR/nv_usb4.lock" 2>&1 | sed 's/^/   /' | cut -c1-90
echo "-- power --"; pmset -g 2>/dev/null | grep -E "powermode|lowpowermode" | sed 's/^/   /'
echo "-- gpu lock --"; $R/tools/gpu-lock.sh status | sed 's/^/   /'
[ $abort -eq 0 ] && echo "VERDICT: OK to proceed (one GPU step at a time)" || echo "VERDICT: ABORT - do not touch the GPU [$why]"
exit $abort
