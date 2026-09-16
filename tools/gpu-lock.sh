#!/bin/sh
# gpu-lock.sh acquire <A|B|C> "<what>" [holder-pid] | release <A|B|C> | status
#
# A held lock refuses EVERYONE, the holder's own session included. Until 2026-09-15 06:53 acquire only refused OTHER
# sessions, and two runs from the same session collided on the card (a foreground op-verify and a background bench,
# both "A"): the second acquire succeeded, its release freed the first run's lock, and a third run then drove the card
# alongside the first — two drivers, one card, "0x1020000000 is already mapped" and a wedged process.
#
# A lock may record the pid it belongs to (the runner passes its own): a lock whose pid is dead is stale and is taken
# over with a note. A lock written without a pid (a session holding a window by hand across several commands, as C
# does) is held until it is released, exactly as before. The line still starts "session=X " for everyone who greps it.
R=${EGPU_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}; L=$R/.gpu-lock
case "${1:-status}" in
  acquire) if [ -f "$L" ]; then
             hp=$(sed -n 's/.* pid=\([0-9][0-9]*\).*/\1/p' "$L" | head -1)
             if [ -n "$hp" ] && ! kill -0 "$hp" 2>/dev/null; then
               echo "stale lock (holder pid $hp is dead), taking it over: $(cat "$L")"
             else
               echo "LOCKED${hp:+ (holder pid $hp is alive)}:"; cat "$L"; exit 1
             fi
           fi
           printf 'session=%s%s since=%s what=%s\n' "$2" "${4:+ pid=$4}" "$(date -Iseconds)" "$3" > "$L"; echo "acquired:"; cat "$L" ;;
  release) [ -f "$L" ] && ! grep -q "session=$2 " "$L" && { echo "not your lock:"; cat "$L"; exit 1; }; rm -f "$L"; echo "released" ;;
  status)  [ -f "$L" ] && cat "$L" || echo "GPU lock: free" ;;
  *) echo "usage: $0 acquire <session> \"<what>\" [holder-pid] | release <session> | status"; exit 2 ;;
esac
