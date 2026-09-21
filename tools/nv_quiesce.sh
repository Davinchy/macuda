#!/bin/sh
# nv_quiesce.sh — drop the GPU to its coolest plugged-in state: standalone FLR halts GSP (WPR2=0) and clears bus master.
# Safe, ~1.5 s, no boot. Use after any session, or any time the box is hot/loud. Read-only preflight first.
R=${EGPU_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}; . $R/env.sh >/dev/null
# The card is looked for, not the enclosure. This asked for an AORUS box by name and so refused to quiesce a 3060 in a
# Razer Core X V2 that was sitting on the bus reading all ones and needed exactly this - the same mistake preflight made.
if system_profiler SPPCIDataType 2>/dev/null | awk '
      /^ *Type: / { vga = ($0 ~ /VGA|Display|3D/) ? 1 : 0; nv = 0 }
      /^ *Vendor ID: 0x10de/ { nv = 1 }
      /^ *Device ID: / { if (vga && nv) { found = 1 } }
      END { exit !found }'; then
  python $R/tools/nv_e3_flr.py 30 2>&1 | tail -3
  python $R/tools/nv_e2_regs.py 2>&1 | grep -E "^verdict" | sed 's/^/quiesced: /'
else
  echo "no NVIDIA card on the PCI bus (powered off, unplugged, or fully dropped) — nothing to quiesce."
fi
