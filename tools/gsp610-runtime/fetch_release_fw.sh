#!/bin/sh
# Fetch and verify the GSP ELF and license from the official NVIDIA 610.43.03
# Linux driver package. No GPU is contacted.
#
# The release is large, so an already-extracted pair can be supplied for offline
# builds with:
#   GSP610_RELEASE_GSP=/path/gsp_ga10x.bin
#   GSP610_RELEASE_LICENSE=/path/LICENSE
#
# Output:
#   cuda-shim/libtinynv/third_party/gsp610-runtime/release/gsp_ga10x.bin
#   cuda-shim/libtinynv/third_party/gsp610-runtime/release/NVIDIA-LICENSE
set -eu

R=$(CDPATH= cd -- "$(dirname "$0")/../.." && pwd)
TP=$R/cuda-shim/libtinynv/third_party
BASE=${GSP610_RELEASE_DIR:-$TP/gsp610-runtime}
OUT=$BASE/release
RUN=NVIDIA-Linux-x86_64-610.43.03.run
RUN_SHA=45e2d4c134a23c35e50f253a4aa63e7e5e8d17e3d185d4a07c8a58e9612ed392
GSP_SHA=572373620a37418f24dc16b5031c39338778c3257e48e8408de9a57291b24f3a
LICENSE_SHA=c692fb14ad4f499c6aee1b9e144e6905148c420e6225b17cb2937366c1e507ad
URL=https://download.nvidia.com/XFree86/Linux-x86_64/610.43.03/$RUN

sha_of() {
    shasum -a 256 "$1" | cut -d' ' -f1
}

mkdir -p "$OUT"

if [ -f "$OUT/gsp_ga10x.bin" ] && [ -f "$OUT/NVIDIA-LICENSE" ] \
   && [ "$(sha_of "$OUT/gsp_ga10x.bin")" = "$GSP_SHA" ] \
   && [ "$(sha_of "$OUT/NVIDIA-LICENSE")" = "$LICENSE_SHA" ]; then
    echo "610.43.03 release assets verified: $OUT"
    exit 0
fi

if [ -n "${GSP610_RELEASE_GSP:-}" ] || [ -n "${GSP610_RELEASE_LICENSE:-}" ]; then
    [ -f "${GSP610_RELEASE_GSP:-}" ] || {
        echo "GSP610_RELEASE_GSP must name an existing gsp_ga10x.bin" >&2
        exit 1
    }
    [ -f "${GSP610_RELEASE_LICENSE:-}" ] || {
        echo "GSP610_RELEASE_LICENSE must name an existing NVIDIA LICENSE" >&2
        exit 1
    }
    cp "$GSP610_RELEASE_GSP" "$OUT/gsp_ga10x.bin"
    cp "$GSP610_RELEASE_LICENSE" "$OUT/NVIDIA-LICENSE"
else
    DOWNLOADS=$BASE/downloads
    EXTRACTED=$BASE/extracted
    mkdir -p "$DOWNLOADS"
    if [ ! -f "$DOWNLOADS/$RUN" ] || [ "$(sha_of "$DOWNLOADS/$RUN")" != "$RUN_SHA" ]; then
        echo "fetching $RUN (440 MiB)"
        curl -fL --retry 3 -o "$DOWNLOADS/$RUN.part" "$URL"
        [ "$(sha_of "$DOWNLOADS/$RUN.part")" = "$RUN_SHA" ] || {
            echo "HASH MISMATCH $RUN" >&2
            exit 1
        }
        mv "$DOWNLOADS/$RUN.part" "$DOWNLOADS/$RUN"
    fi
    echo "release package verified: $RUN_SHA"
    if [ ! -f "$EXTRACTED/firmware/gsp_ga10x.bin" ]; then
        mkdir -p "$EXTRACTED"
        (
            cd "$DOWNLOADS"
            sh "$RUN" --extract-only --target "$EXTRACTED"
        )
    fi
    cp "$EXTRACTED/firmware/gsp_ga10x.bin" "$OUT/gsp_ga10x.bin"
    cp "$EXTRACTED/LICENSE" "$OUT/NVIDIA-LICENSE"
fi

[ "$(sha_of "$OUT/gsp_ga10x.bin")" = "$GSP_SHA" ] || {
    echo "HASH MISMATCH gsp_ga10x.bin" >&2
    exit 1
}
[ "$(sha_of "$OUT/NVIDIA-LICENSE")" = "$LICENSE_SHA" ] || {
    echo "HASH MISMATCH NVIDIA license" >&2
    exit 1
}
echo "610.43.03 release assets verified: $OUT"
