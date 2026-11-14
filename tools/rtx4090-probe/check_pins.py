#!/usr/bin/env python3
"""The C transaction's embedded asset pins must equal tools/rtx4090-unlock/manifest.json.

The manifest is the single pin source; tensor_unlock.c embeds the same five (name, size, sha256) triples for
deterministic, download-free discovery at open time. This check makes a drift between the two a loud failure.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
MANIFEST = os.path.join(ROOT, "tools", "rtx4090-unlock", "manifest.json")
SOURCE = os.path.join(ROOT, "cuda-shim", "libtinynv", "src", "tensor_unlock.c")


def main():
    m = json.load(open(MANIFEST))
    assets = m["assets"]
    src = open(SOURCE).read()
    block = src.split("tensor_unlock_pins = {", 1)[1].split("};", 1)[0]
    pins = {}
    for key, name, size, sha in re.findall(r'\.(\w+)\s*=\s*\{"([^"]+)",\s*(\d+),\s*"([0-9a-f]{64})"\}', block):
        pins[key] = (name, int(size), sha)
    want = {
        "booter": "booter_load_ad102.bin",
        "payload": "v67-ad102-610.43.03-four-plm.bin",
        "gsp": "gsp_ga10x.bin",
        "boot": "gsp_boot_ad102.bin",
        "desc": "gsp_boot_desc_ad102.bin",
    }
    bad = 0
    for key, name in want.items():
        if key not in pins:
            print(f"FAIL: pin '{key}' missing from tensor_unlock.c")
            bad += 1
            continue
        pname, psize, psha = pins[key]
        a = assets[name]
        if pname != name or psize != a["size"] or psha != a["sha256"]:
            print(f"FAIL: pin '{key}' = {pins[key]} but manifest {name} = ({a['size']}, {a['sha256']})")
            bad += 1
        else:
            print(f"ok: {key} {name} {psize} bytes {psha[:16]}…")
    # the recipe gates the C code implements must still be the manifest's
    recipe = m["recipe"]
    for lit, why in [
        (recipe["mailbox0_marker"], "SEC2 completion marker"),
        (recipe["selectors"]["SS0"]["value"], "SS0 unlock value"),
        (recipe["selectors"]["SS1"]["value"], "SS1 unlock value"),
        (recipe["effective_readout"]["mask"], "effective readout mask"),
    ]:
        v = int(lit, 16)
        if not re.search(r"0x0*%xu?\b" % v, src, re.I):
            print(f"FAIL: {why} {lit} not found in tensor_unlock.c")
            bad += 1
        else:
            print(f"ok: {why} {lit}")
    allow = m["device_allowlist"]
    assert len(allow) == 1 and allow[0]["device"] == "0x2684", "manifest allowlist changed; review the C gate"
    if "0x268410de" not in src:
        print("FAIL: the C identity gate is not 10de:2684")
        bad += 1
    print("FAIL" if bad else "PASS: C pins equal the manifest")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
