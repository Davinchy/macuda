#!/bin/sh
# Print the Metal-side llama-server: the one that runs on the Mac's own GPU, which every disaggregation tool here
# needs beside the card's *-null binary.
#
# Searched in order: $METAL_BIN, this tree's own Metal build, a llama-server on PATH, then the offline-ai-kit's
# ($OFFLINE_AI_KIT overrides where that is). The kit path used to be written into six scripts as their default, which
# is fine on the machine that has it and leaves every other machine with a tool that cannot start (2026-09-22).
#
# Nothing found: the tree-local path is printed anyway, so the caller's own "no Metal server at ..." message names a
# path that belongs to this project, and where it was looked for goes to stderr rather than into that message.
set -u
R=${EGPU_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
KIT=${OFFLINE_AI_KIT:-/Volumes/512SSD/LocalCode/offline-ai-kit}
LOCAL=$R/llama.cpp/build-metal/bin/llama-server
for b in ${METAL_BIN:-} "$LOCAL" "$KIT/bin/llama-server"; do
  [ -x "$b" ] && { echo "$b"; exit 0; }
done
onpath=$(command -v llama-server 2>/dev/null) && [ -n "$onpath" ] && { echo "$onpath"; exit 0; }
echo "no Metal llama-server found: looked at ${METAL_BIN:+$METAL_BIN, }$LOCAL, llama-server on PATH, $KIT/bin/llama-server." >&2
echo "Build one (cmake -B llama.cpp/build-metal -DGGML_METAL=ON llama.cpp && cmake --build llama.cpp/build-metal -j --target llama-server)," >&2
echo "or set METAL_BIN=/path/to/llama-server. This is llama.cpp's ordinary Metal build, not one of the shim's binaries." >&2
echo "$LOCAL"
