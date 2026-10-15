#!/bin/sh
# Fetch the NVIDIA open-gpu-kernel-modules tree the driver's struct definitions come from.
#
# The commit is pinned to exactly what tinygrad's autogen was generated from, because the build asserts our sizeof and
# offsetof against tinygrad's structs: any other tag would fire those assertions on innocent upstream drift. That commit
# is release 570.86.16, the one that added RTX 5090 support. Note the deliberate skew — headers 570.86.16, firmware
# 570.144 — which the python oracle proves compatible by booting the card with exactly that pairing.
#
# The tree is large and never committed: it lands in third_party/, which .gitignore excludes.
set -e
COMMIT=81fe4fb417c8ac3b9bdcc1d56827d116743892a5
HERE=$(cd "$(dirname "$0")/.." && pwd)
DEST=$HERE/third_party
TARBALL=$DEST/ogkm-$COMMIT.tar.gz
SUMS=$HERE/tools/SHA256SUMS

mkdir -p "$DEST"
if [ ! -f "$TARBALL" ]; then
  echo "fetching open-gpu-kernel-modules $COMMIT (about 200 MB)"
  curl -fL --retry 3 -o "$TARBALL.part" \
    "https://github.com/NVIDIA/open-gpu-kernel-modules/archive/$COMMIT.tar.gz"
  mv "$TARBALL.part" "$TARBALL"
fi

got=$(shasum -a 256 "$TARBALL" | cut -d' ' -f1)
if [ -f "$SUMS" ] && grep -q "$COMMIT" "$SUMS"; then
  want=$(grep "$COMMIT" "$SUMS" | cut -d' ' -f1)
  [ "$got" = "$want" ] || { echo "HASH MISMATCH: got $got, expected $want. Refusing to extract."; exit 1; }
  echo "hash verified: $got"
else
  # first fetch on this machine: record what we got, and say plainly that this one is trust-on-first-use
  echo "$got  ogkm-$COMMIT.tar.gz" > "$SUMS"
  echo "recorded hash $got (trust on first use: cross-check it against another machine before relying on it)"
fi

if [ ! -d "$DEST/open-gpu-kernel-modules-$COMMIT/src" ]; then
  echo "extracting"
  tar xf "$TARBALL" -C "$DEST"
fi
echo "headers at $DEST/open-gpu-kernel-modules-$COMMIT"
du -sh "$DEST/open-gpu-kernel-modules-$COMMIT" 2>/dev/null || true
