#!/bin/sh
# nv_quiesce.sh — drop the GPU to its coolest plugged-in state: standalone FLR halts GSP (WPR2=0) and clears bus master.
# Safe, ~1.5 s, no boot. Use after any session, or any time the box is hot/loud. Read-only preflight first.
R=${EGPU_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}; . $R/env.sh >/dev/null
if system_profiler SPThunderboltDataType 2>/dev/null | grep -q AORUS; then
  python $R/tools/nv_e3_flr.py 30 2>&1 | tail -3
  python $R/tools/nv_e2_regs.py 2>&1 | grep -E "^verdict" | sed 's/^/quiesced: /'
else
  echo "AORUS box not on the Thunderbolt bus (already powered off / unplugged) — nothing to quiesce."
fi
