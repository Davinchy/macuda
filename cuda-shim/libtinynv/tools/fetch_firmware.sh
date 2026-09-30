#!/bin/sh
# Fetch the signed NVIDIA firmware images a card needs to boot, and refuse anything that hashes wrong.
#
# Blackwell (gb202) boots through the FSP chain of trust: fmc + bootloader. Ampere (ga102) boots its falcon from the
# VBIOS instead - FWSEC/FRTS out of the card's own ROM, then booter_load over SEC2 - so it needs booter_load, its own
# bootloader, and booter_unload, which tears the write-protected region down again when the driver exits so the next
# open finds a cold card. GSP-RM itself (gsp-570.144.bin) ships once, under ga102, and serves both.
#
# These are code the GPU's secure boot executes, so the version is pinned by content, not by filename: the hashes below
# are exactly the ones tinygrad's driver asserts, and the boot the replay trace records was taken with these bytes.
# Note the deliberate skew documented in fetch_nv_headers.sh — headers 570.86.16, firmware 570.144.
#
# The images land in third_party/firmware/, which .gitignore excludes: 63 MB of GSP-RM is not something to commit.
# If tinygrad has already downloaded them on this machine they are copied from its cache instead of fetched again.
set -e
HERE=$(cd "$(dirname "$0")/.." && pwd)
DEST=$HERE/third_party/firmware
# linux-firmware at the commit tinygrad pins in fetch_fw()
BASE=https://gitlab.com/kernel-firmware/linux-firmware/-/raw/0a6871b19abf5d6e024b5d208b101ae53e7fa0de
CACHE=${XDG_CACHE_HOME:-$HOME/.cache}/tinygrad/downloads/fw

# chip-dir  name  sha256
set -- \
  "gb202 fmc-570.144.bin         cb59a35c1d4bd1274d7267fd10243c29f843ff41c851b9cbd59f5af2ddd7fece" \
  "gb202 bootloader-570.144.bin  d40b48e431d1707dc77af3605db358ed7a32ebfc2830eb74de2eddb4d3025071" \
  "ga102 gsp-570.144.bin         a8c3ebeed280323aedb51c061f321e73379cce7a9ae643a33dd03915df027f7f" \
  "ga102 booter_load-570.144.bin 4497e3eff7e95c774b8a569d17b27c08c9650158d10b229d2be81cdcad9a085b" \
  "ga102 booter_unload-570.144.bin 8e63db5b78d7d3e349f20a2d11099c3d7109081393cb09ffc0a28133324ae009" \
  "ga102 bootloader-570.144.bin  82428f532240727e95bb3083fbaaba9b2cc7b937314323f2d546ce7245f27fad" \
  "ad102 booter_load-570.144.bin 8b293e19b637c5e22c87a2428d1c71bb13e0904e8a88ac6b3c6c1f2679c6e37a" \
  "ad102 booter_unload-570.144.bin 975b85a14ded8e430d30f000c3c1afdd55c15dee04f35ff9dfd876acd7e67186" \
  "ad102 bootloader-570.144.bin  65ab2e6b6e0fca95365c4deac79a34582abcfeb15b6ae234138f22e7183118a8" \
  "ad102 scrubber-570.144.bin    c397358e5c4258dab070966589c7b172eb00771fd15d0bb74b8e22f327c1909b"

sha_of() { shasum -a 256 "$1" 2>/dev/null | cut -d' ' -f1 || sha256sum "$1" | cut -d' ' -f1; }

for entry in "$@"; do
  # shellcheck disable=SC2086
  set -- $entry
  chip=$1 name=$2 want=$3
  out=$DEST/nvidia/$chip/gsp/$name
  mkdir -p "$(dirname "$out")"

  if [ -f "$out" ] && [ "$(sha_of "$out")" = "$want" ]; then echo "have $chip/$name"; continue; fi

  # tinygrad names its downloads by a hash of the url, so find ours by content rather than by name
  found=
  if [ -d "$CACHE" ]; then
    for f in "$CACHE"/*; do
      [ -f "$f" ] || continue
      if [ "$(sha_of "$f")" = "$want" ]; then found=$f; break; fi
    done
  fi
  if [ -n "$found" ]; then
    echo "copying $chip/$name from tinygrad's cache"
    cp "$found" "$out"
  else
    echo "fetching $chip/$name"
    curl -fL --retry 3 -o "$out.part" "$BASE/nvidia/$chip/gsp/$name"
    mv "$out.part" "$out"
  fi

  got=$(sha_of "$out")
  [ "$got" = "$want" ] || { rm -f "$out"; echo "HASH MISMATCH for $name: got $got, expected $want"; exit 1; }
  echo "  verified $got"
done
echo "firmware at $DEST"
du -sh "$DEST" 2>/dev/null || true
