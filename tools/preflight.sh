#!/bin/sh
R=${EGPU_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
DART_STALE=${DART_STALE:-$([ -d /Volumes/512SSD/EGPU ] && echo /Volumes/512SSD/EGPU/.dart-stale || echo $R/.dart-stale)}   # shared with the EGPU tree when it exists
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
# THE DART DECISION IS A PURE FUNCTION so the absent branch can be SEEN to refuse on a machine where the property is
# present - on the machine where it is absent, "absent refuses" passes whether or not the code does anything (B, 13:5x).
#   dart_verdict <present 0|1> <snapshot 0|1> <matches 0|1>  ->  stale | new-fault | reenum | clean
dart_verdict() { if [ "$1" = 1 ]; then [ "$2" = 1 ] && [ "$3" = 1 ] && echo stale || echo new-fault; else [ "$2" = 1 ] && echo reenum || echo clean; fi; }
# THE SECOND RE-ENUMERATION DETECTOR, independent of the DART property (V1, 14:4x): every re-enumeration since boot starts a
# NEW org.tinygrad.tinygpu.driver2 instance and the old ones linger, so the newest instance's start time moving forward IS
# a re-enumeration, whether or not the property ever existed. The newest start seen at the last clean step is recorded in
# .dext-seen; a newer one refuses until a human runs tools/dart-stale.sh --clear (which forgets both records).
#   dext_verdict <recorded_epoch|0> <newest_epoch>  ->  first | same | newer
dext_verdict() { if [ "$1" = 0 ]; then echo first; elif [ "$2" -gt "$1" ]; then echo newer; else echo same; fi; }
if [ "${1:-}" = --selftest ]; then
  ok=1; chk() { r=$(dart_verdict $1 $2 $3); [ "$r" = "$4" ] || { echo "SELFTEST FAIL: dart_verdict $1 $2 $3 = $r, expected $4"; ok=0; }; }
  chk 1 1 1 stale; chk 1 1 0 new-fault; chk 1 0 0 new-fault; chk 1 0 1 new-fault; chk 0 1 0 reenum; chk 0 1 1 reenum; chk 0 0 0 clean; chk 0 0 1 clean
  chd() { r=$(dext_verdict $1 $2); [ "$r" = "$3" ] || { echo "SELFTEST FAIL: dext_verdict $1 $2 = $r, expected $3"; ok=0; }; }
  chd 0 1000 first; chd 1000 1000 same; chd 1000 999 same; chd 1000 1001 newer
  [ $ok = 1 ] && { echo "preflight --selftest: dart_verdict OK on all 8 combinations (absent+snapshot REFUSES); dext_verdict OK on 4 (a newer instance REFUSES)"; exit 0; } || exit 1
fi
# FOUR STATES, not two. The property is written by the DART when a fault latches and lives with the NUB: only a
# re-enumeration removes it. So: present and byte-identical to the snapshot = stale, proceed; present and different = a
# new fault, refuse; ABSENT while a snapshot exists = the nub was re-enumerated since the snapshot was taken (run 41,
# 13:5x: the card re-enumerated under a live daemon with no replug) - REFUSE until a human has looked and cleared the
# snapshot, because the old gate read this state as clean and it is the one state that needs a human; absent with no
# snapshot = clean. A gate that cannot fail is not a gate.
SNAP=${DART_STALE:-$R/.dart-stale}
present=0; matches=0; snapshot=0; [ -f "$SNAP" ] && snapshot=1
# THE READER'S SCOPE IS AT LEAST THE WRITER'S: tools/dart-stale.sh records the property from the WHOLE ioreg, while $nub is
# the 70 lines after the device-id match - the property sat at line ~52 all day, which is the only reason this worked. A
# fault whose property lands past line 70 would read as absent, and absent-with-no-snapshot is "clean" (B, 14:3x). So the
# property is searched where the writer searches it; a false positive refuses, a false negative is the one that matters.
dart_all=$(ioreg -l -w0 2>/dev/null | grep -o '"pci-dart-error-data" = <[0-9a-f]*>' | head -1)
if [ -n "$dart_all" ]; then
  present=1; cur=$dart_all
  [ $snapshot = 1 ] && grep -qxF "$cur" "$SNAP" && matches=1
fi
case $(dart_verdict $present $snapshot $matches) in
  stale)     echo "   DART error data present, byte-identical to the snapshot: stale, not a new fault"
             sed -n 's/^why=/   reason: /p; s/^when=/   recorded: /p' "$SNAP" ;;
  new-fault) echo "   DART ERROR DATA PRESENT"; fail "new DART error data" ;;
  reenum)    echo "   DART error data ABSENT but a snapshot exists ($(sed -n 's/^when=//p' "$SNAP")): the nub has been RE-ENUMERATED since"
             echo "   it was taken. That is not a clean card, it is a changed one. A human looks first; then tools/dart-stale.sh --clear."
             fail "nub re-enumerated since the DART snapshot" ;;
  clean)     echo "   DART error data absent, no snapshot: clean" ;;
esac
DEXT_SEEN=${DEXT_SEEN:-$(dirname "$SNAP")/.dext-seen}
dext_n=0; dext_newest=0; dext_pid=0
# ANCHORED TO THE EXECUTABLE PATH: `pgrep -f <name>` matches any process whose COMMAND LINE carries the name - including
# another session's own pgrep/grep for it. At 15:25:24 B's gate counted a fifteenth "instance" that was V1's slot script
# counting instances at the same moment; it was gone a minute later and preflight (which prints each path) never saw it.
for dp in $(pgrep -f '^/Library/SystemExtensions/[^ ]*/org\.tinygrad\.tinygpu\.driver2' 2>/dev/null); do
  dext_n=$((dext_n+1)); ls_=$(ps -p "$dp" -o lstart= 2>/dev/null | sed 's/^ *//'); [ -n "$ls_" ] || continue
  ep=$(date -j -f "%a %b %d %H:%M:%S %Y" "$ls_" +%s 2>/dev/null || echo 0)
  [ "$ep" -gt "$dext_newest" ] && { dext_newest=$ep; dext_pid=$dp; }
done
rec=0; [ -f "$DEXT_SEEN" ] && rec=$(sed -n 's/^newest=//p' "$DEXT_SEEN" | head -1); rec=${rec:-0}
case $(dext_verdict "$rec" "$dext_newest") in
  first) printf 'newest=%s\npid=%s\ncount=%s\nwhen=%s\n' "$dext_newest" "$dext_pid" "$dext_n" "$(date -Iseconds)" > "$DEXT_SEEN"
         echo "   dext instances: $dext_n, newest pid $dext_pid started $(date -r "$dext_newest" '+%F %T') - recorded (first sighting)" ;;
  same)  echo "   dext instances: $dext_n, newest pid $dext_pid started $(date -r "$dext_newest" '+%F %T') - unchanged since the record" ;;
  newer) echo "   dext instances: $dext_n, NEWEST pid $dext_pid started $(date -r "$dext_newest" '+%F %T') is NEWER than the record ($(date -r "$rec" '+%F %T')):"
         echo "   the card has been RE-ENUMERATED since the last clean step. A human looks first; then tools/dart-stale.sh --clear."
         fail "a new dext instance since the record" ;;
esac
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
