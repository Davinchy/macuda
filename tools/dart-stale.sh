#!/bin/sh
# Record the current pci-dart-error-data as known-stale, so preflight stops reading it as a fresh fault.
#
# The flag is latched by IOPCIFamily and an FLR does not clear it - only re-enumeration does, which means a replug, a
# sleep/wake or a reboot. After a recovery FLR the card is reset and usable while the flag still says otherwise, and a
# preflight that aborts on it forever stops being a check: it says the same thing whatever the truth is.
#
# So the flag's exact bytes are snapshotted here, with a reason, and preflight treats a byte-identical flag as the one
# already accounted for. Anything else - one more fault record, a different address - does not match and aborts as before.
#
#   tools/dart-stale.sh "why this is known stale"      record
#   tools/dart-stale.sh --clear                        forget it (after a replug)
ROOT=${EGPU_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
SNAP=${DART_STALE:-$([ -d /Volumes/512SSD/EGPU ] && echo /Volumes/512SSD/EGPU/.dart-stale || echo $ROOT/.dart-stale)}
[ "${1:-}" = "--clear" ] && { rm -f "$SNAP" "$(dirname "$SNAP")/.dext-seen"; echo "cleared: the next dart error data will abort; the dext-instance record is forgotten and re-recorded at the next clean preflight"; exit 0; }
[ -n "${1:-}" ] || { echo "say why it is stale: tools/dart-stale.sh \"<reason>\""; exit 1; }
cur=$(ioreg -l -w0 2>/dev/null | grep -o '"pci-dart-error-data" = <[0-9a-f]*>' | head -1)
[ -n "$cur" ] || { echo "no pci-dart-error-data on the nub: nothing to record"; exit 1; }
{ printf 'when=%s\nwhy=%s\n' "$(date -Iseconds)" "$1"; echo "$cur"; } > "$SNAP"
echo "recorded as stale ($(printf '%s' "$cur" | wc -c | tr -d ' ') bytes of flag): $1"
