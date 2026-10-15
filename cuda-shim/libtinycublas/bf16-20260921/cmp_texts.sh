#!/bin/sh
# the correctness pair's comparator: the two llama-simple texts (prompt + 32 generated tokens) must be byte-identical.
#   sh cmp_texts.sh <scalar-path text> <fixed-path text>      exit 0 IDENTICAL, 1 DIFFER (with the first differing byte)
# Its control, run before the slot: the same compare against a copy with ONE byte changed past the prompt must say DIFFER.
if cmp -s "$1" "$2"; then echo "IDENTICAL: $(wc -c < "$1" | tr -d ' ') bytes, sha $(shasum "$1" | cut -c1-12)"; exit 0; fi
echo "DIFFER: $(cmp "$1" "$2" 2>&1 | head -1)"; exit 1
