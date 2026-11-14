#!/usr/bin/env python3
"""Read unwrapped packed nvlog records from gsp_silent_boot_diag transcripts.

The ring stores arguments, packed metadata, then timestamp, not metadata first.
This decodes record boundaries, not stripped format strings. Refuse truncated or
wrapped dumps rather than guessing missing arguments. No GPU access.
"""
import argparse
import json
import re
from pathlib import Path


def records(words):
    """Word zero is the producer count; follow committed records backwards."""
    if 0 not in words:
        raise ValueError("missing producer count")
    end = words[0]
    if end >= 96:
        raise ValueError("wrapped or capped transcript needs a full binary ring dump")
    # Diagnostic output omits trailing zero words. A missing timestamp is not
    # recoverable; zero-valued arguments inside a committed record are valid.
    result = []
    while end:
        if end not in words or end - 1 not in words:
            raise ValueError("missing committed timestamp/metadata")
        meta = words[end - 1]
        count, task = (meta >> 48) & 255, meta >> 56
        if not 2 <= count <= 20 or not 1 <= task <= 9 or count > end:
            raise ValueError(f"invalid packed record ending at word {end}")
        start = end - count + 1
        if any(i not in words for i in range(start, end + 1)):
            raise ValueError("missing committed record words")
        result.append(dict(task=task, meta=hex(meta & 0xffffffffffff),
                           args=[hex(words[i]) for i in range(start, end - 1)],
                           timestamp=hex(words[end]), start=start, end=end))
        end -= count
    return list(reversed(result))


def transcript(text):
    logs = []
    name, words = None, {}
    for line in text.splitlines() + ["tinynv: gsp log END: empty"]:
        header = re.match(r"tinynv: gsp log (\w+):", line)
        if header:
            if name and words:
                logs.append(dict(log=name, records=records(words)))
            name, words = header[1], {}
        elif name:
            word = re.match(r"\s*\[(\d+)\] ([0-9a-fA-F]{16})\b", line)
            if word:
                words[int(word[1])] = int(word[2], 16)
    return logs


def pc_relative_references(text, targets):
    """Resolve adjacent AUIPC/ADDI pairs, including signed 20-bit immediates.

    A stripped metadata pointer need not appear as an immediate/literal. LLVM
    objdump prints the split PC-relative construction used by this firmware.
    Only exact adjacent same-register pairs are recognized; this is not a
    general RISC-V data-flow engine.
    """
    lines = text.splitlines()
    found = []
    for first, second in zip(lines, lines[1:]):
        upper = re.search(r"^\s*([0-9a-f]+):.*\bauipc\s+(\w+), (0x[0-9a-f]+)", first)
        if not upper:
            continue
        lower = re.search(r"\baddi\s+" + upper[2] + ", " + upper[2]
                          + r", (-?0x[0-9a-f]+)", second)
        if not lower:
            continue
        hi = int(upper[3], 16)
        if hi & (1 << 19):
            hi -= 1 << 20
        address = (int(upper[1], 16) + (hi << 12) + int(lower[1], 16)) & ((1 << 64) - 1)
        if address in targets:
            found.append(dict(pc=hex(int(upper[1], 16)), target=hex(address)))
    return found


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("transcript", type=Path)
    parser.add_argument("--init-disassembly", type=Path,
                        help="optional INIT disassembly; resolve metadata relative to 0x20000000")
    args = parser.parse_args()
    logs = transcript(args.transcript.read_text())
    if args.init_disassembly:
        targets = {0x20000000 + int(r["meta"], 16)
                   for log in logs for r in log["records"] if r["task"] == 2}
        print(json.dumps(dict(logs=logs, init_references=pc_relative_references(
            args.init_disassembly.read_text(), targets)), indent=2))
    else:
        print(json.dumps(logs, indent=2))
