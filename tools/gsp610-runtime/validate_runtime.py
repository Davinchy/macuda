#!/usr/bin/env python3
""": validate a prepared AD102 610.43.03 runtime bundle (offline, read-only).

Checks the immutable manifest against the frozen pins, every asset's size/sha256, the BTL1
booter/scrubber containers (bounds, alignment, patch ranges, engine/ucode ids, pinned AD102
geometry), the v5 RISC-V boot descriptor, the GSP ELF sections (.fwimage/.fwsignature_ad10x/
.fwversion), cross-asset lifecycle constraints and the required notices.

Usage: python3 validate_runtime.py [BUNDLE_DIR]   (default: cuda-shim/build/gsp610-runtime/bundle)
Exit status 0 = valid; 1 = first concrete failure printed to stderr.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))

import runtime_assets as ra  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bundle", nargs="?", type=Path,
                        default=REPO / "cuda-shim/build/gsp610-runtime/bundle")
    args = parser.parse_args()
    try:
        ra.validate_bundle(args.bundle)
    except (ra.AssetError, OSError, ValueError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
