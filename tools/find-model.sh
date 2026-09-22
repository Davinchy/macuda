#!/bin/sh
# Print the first of the named model files that exists, so a default model path is a SEARCH rather than one machine's
# layout. Names are relative to a models directory; an absolute name is taken as it stands.
#
#   sh tools/find-model.sh mtp-Qwen3.8-27B-Q4_0.gguf MTP/mtp-Qwen3.8-27B-Q4_0.gguf
#
# Searched in order: $MODELS_DIR if set, this tree's models/, then the offline-ai-kit's models directory when that is
# present ($OFFLINE_AI_KIT overrides where it is). The kit is where this machine keeps the large shared files; it does
# not exist anywhere else, which is exactly why it cannot be a default on its own.
#
# WHY: the README's `hf download ... --local-dir models/` writes the MTP head to models/MTP/, while tools/serve.sh had
# models/mtp-....gguf compiled into it. Both paths exist on the machines this grew up on, one of them by a symlink
# made by hand in September, so a fresh install was the first place the disagreement could be seen - and there it
# refused to serve at all (2026-09-22).
#
# When nothing is found the FIRST name is printed under this tree's models/, so the caller's own "no model at ..."
# message names the file a reader is most likely to go and fetch.
set -u
R=${EGPU_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
KIT=${OFFLINE_AI_KIT:-/Volumes/512SSD/LocalCode/offline-ai-kit}
for n in "$@"; do
  case $n in
    /*) [ -f "$n" ] && { echo "$n"; exit 0; }; continue ;;
  esac
  for d in ${MODELS_DIR:-} "$R/models" "$KIT/models"; do
    [ -f "$d/$n" ] && { echo "$d/$n"; exit 0; }
  done
done
echo "$R/models/$1"
