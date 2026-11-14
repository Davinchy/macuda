#!/usr/bin/env python3
""": build the same/changed/removed/added ABI comparison matrix from the two compiled
probe outputs (abi_570.txt / abi_610.txt) plus the source-level text_diff.json.

Each record is keyed by its symbol. A symbol present in both with equal value is 'same';
present in both with different value is 'changed'; only in 570 is 'removed'; only in 610 is
'added'. Field-level (F) records are grouped under their struct so a struct whose size or any
field offset moved is reported as 'changed' with the specific deltas.

Usage: python3 abi_diff.py <build-dir> > abi_matrix.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path


def parse(path: Path) -> dict:
    rec = {}
    for line in path.read_text().splitlines():
        parts = line.split(" ", 2)
        if len(parts) < 3:
            continue
        kind, name, val = parts
        rec[(kind, name)] = val
    return rec


def main() -> None:
    build = Path(sys.argv[1])
    a = parse(build / "abi_570.txt")
    b = parse(build / "abi_610.txt")
    keys = sorted(set(a) | set(b))
    rows = []
    for kind, name in keys:
        va, vb = a.get((kind, name)), b.get((kind, name))
        if va is not None and vb is not None:
            status = "same" if va == vb else "changed"
        elif va is not None:
            status = "removed"
        else:
            status = "added"
        rows.append({"kind": kind, "symbol": name, "v570": va, "v610": vb, "status": status})

    # group field deltas under their struct for readability
    structs = {}
    for r in rows:
        if r["kind"] == "S":
            structs.setdefault(r["symbol"], {"size": r, "fields": []})
        elif r["kind"] == "F":
            base = r["symbol"].split(".")[0]
            structs.setdefault(base, {"size": None, "fields": []})["fields"].append(r)

    text = json.loads((build / "text_diff.json").read_text())
    out = {
        "records": rows,
        "structs": structs,
        "text": text,
        "summary": {
            "compiled_records": len(rows),
            "same": sum(1 for r in rows if r["status"] == "same"),
            "changed": sum(1 for r in rows if r["status"] == "changed"),
            "removed": sum(1 for r in rows if r["status"] == "removed"),
            "added": sum(1 for r in rows if r["status"] == "added"),
        },
    }
    json.dump(out, sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main()