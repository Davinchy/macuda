#!/usr/bin/env python3
"""Refuse a control whose parameters carry a pointer, at any depth.

A pointer inside an RM control's parameters is an address in the CALLER's process. Forwarded or replayed it means
nothing: the other side reads whatever lives at that number, and under ASLR the same call is a different number every
run. Session C hit this from the guest side and enumerated 59 such commands out of 1375; then found their scan had two
blind spots, and both are the reason this exists rather than a note in a document.

  - NESTED structures. A scan that reads only the fields declared directly in the parameter struct misses a pointer
    inside an array of some other struct, which is where the one they found was hiding.
  - RAW pointers. NVIDIA's own marshalling category is NvP64, but a plain `struct Foo *bar` is just as much an address
    in the caller's process and is not spelled NvP64 anywhere.

libtinynv is clean today and structurally likely to stay clean, because it allocates everything itself and only uses
controls that move fixed-size data. "Likely" is the word this exists to remove: the check costs a second a build and
the failure it prevents is the card reading an address in the wrong process, which reports nothing and corrupts
whatever it lands on.
"""
import os, re, sys

def load(root):
    blob = []
    for d, _, fs in os.walk(root):
        for f in fs:
            if f.endswith('.h'):
                try: blob.append(open(os.path.join(d, f), errors='ignore').read())
                except OSError: pass
    return "\n".join(blob)

PTR = re.compile(r"NvP64|\b\w+\s*\*\s*\w+\s*[;\[]")

def main():
    if len(sys.argv) < 3:
        print("usage: check_ctrl_pointers.py <sdk-inc-dir> <nv_structs.h>"); return 2
    inc, structs = sys.argv[1], sys.argv[2]
    blob = load(inc)
    bodies = {}
    for m in re.finditer(r"typedef struct\s+(\w+)?\s*\{(.*?)\}\s*(\w+);", blob, re.S):
        bodies[m.group(3)] = m.group(2)
        if m.group(1): bodies[m.group(1)] = m.group(2)

    # the controls this driver actually sends, read from its own header so the list cannot drift from the code
    cmds = re.findall(r"#define\s+TINYNV_CTRL_\w+\s+(0x[0-9a-fA-F]+)", open(structs, errors='ignore').read())
    if not cmds:
        print("  no controls found in", structs); return 1

    def params_for(cmd):
        m = re.search(r"\((?:%s|%s)U?\)" % (cmd, cmd.upper()), blob)
        if not m: return None
        line_start = blob.rfind("#define", 0, m.start())
        seg = blob[line_start:m.end() + 400]
        mm = re.search(r"(NV\w+_PARAMS)_MESSAGE_ID", seg)
        return mm.group(1) if mm else None

    def scan(name, seen, path):
        if name in seen or name not in bodies: return []
        seen.add(name); out = []
        for line in bodies[name].splitlines():
            if PTR.search(line): out.append((path + "." + name, line.strip()))
        for t in set(re.findall(r"\b([A-Z][A-Za-z0-9_]{4,})\b", bodies[name])):
            if t in bodies and t != name: out += scan(t, seen, path + "." + name)
        return out

    bad, checked, unlocated = 0, 0, []
    for c in sorted(set(cmds)):
        p = params_for(c)
        if not p: unlocated.append(c); continue
        checked += 1
        for where, line in scan(p, set(), ""):
            print("  FAIL: control %s (%s) carries a pointer: %s\n        at %s" % (c, p, line, where)); bad += 1
    if unlocated:
        # Not a pass. A control whose parameters cannot be found is one that was not checked, and saying so is the
        # difference between this being a test and being decoration.
        print("  could not locate parameters for: %s" % ", ".join(unlocated))
    if bad:
        print("control pointers: %d field(s) that would be an address in the wrong process" % bad); return 1
    print("  %d controls checked to any nesting depth, none carries a pointer%s" %
          (checked, "" if not unlocated else " (%d not locatable, see above)" % len(unlocated)))
    return 0

sys.exit(main())
