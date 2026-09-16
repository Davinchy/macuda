#!/bin/sh
# The TinyGPU server: is it there, and start it if not.
#
# A replug takes it with it. That has happened three times now and cost a confused minute each time, because the socket
# file survives and a client gets "connection refused" rather than "nothing is listening" - which reads like a driver
# failure rather than a missing daemon. So: ask, and start it the way it was running.
#
# But "connection refused" is NOT proof the server is gone: on macOS a listening unix socket refuses connects while its
# backlog is full, and on 2026-09-15 06:53 a live server refused one client for a moment, `ensure` took that for death,
# unlinked its socket and started a second server on the same card. Now: a live server PROCESS for this socket is never
# replaced. A refusal with the process alive is retried for two seconds and then reported as busy, exit 1, so a runner
# aborts its step rather than run against a replacement.
#
#   tools/tinygpu-server.sh status | start | ensure
SOCK=${TMPDIR}tinygpu.sock
APP=/Applications/TinyGPU.app/Contents/MacOS/TinyGPU
R=${EGPU_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}; LOG=$R/logs/tinygpu-server.log; mkdir -p "$R/logs"

alive() { python3 -c "
import socket, sys
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
try: s.connect('$SOCK')
except OSError: sys.exit(1)
finally: s.close()
"; }
serving() { pgrep -f "TinyGPU server $SOCK" | tr '\n' ' '; }   # every server process bound to this socket path (should be one)

case "${1:-status}" in
  status) p=$(serving); if alive; then echo "tinygpu server: listening on $SOCK (pid ${p:-?})"; else
            echo "tinygpu server: NOT listening on $SOCK$([ -S "$SOCK" ] && echo ' (the socket file is stale)')${p:+; server process(es) $p exist but refuse connections}"; exit 1; fi ;;
  start|ensure)
    if alive; then [ "$1" = ensure ] && { echo "tinygpu server: already listening (pid $(serving))"; exit 0; }
                   echo "tinygpu server: already listening; stop it first to restart"; exit 1; fi
    p=$(serving)
    if [ -n "$p" ]; then
      i=0; while [ $i -lt 20 ]; do i=$((i+1)); sleep 0.1; alive && { echo "tinygpu server: listening on $SOCK (pid $p) after a refused connect — it was busy"; exit 0; }; done
      echo "tinygpu server: process(es) $p exist for $SOCK but refuse connections after 2 s — busy or wedged, NOT starting another"; exit 1
    fi
    rm -f "$SOCK"
    nohup "$APP" server "$SOCK" >> "$LOG" 2>&1 &
    echo "started $!, waiting for the socket"
    i=0; while [ ! -S "$SOCK" ] && [ $i -lt 100 ]; do i=$((i+1)); sleep 0.1; done
    if alive; then echo "tinygpu server: listening on $SOCK"; else echo "tinygpu server: did not come up; see $LOG"; exit 1; fi ;;
  *) echo "usage: $0 status|start|ensure"; exit 2 ;;
esac
