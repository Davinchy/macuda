#!/usr/bin/env python3
"""No object may be older than, or the same age as, the source it was built from.

WHY THIS EXISTS. The make on this machine is GNU Make 3.81 - Apple's 2006 build - which compares timestamps at ONE
SECOND granularity. APFS records nanoseconds. So a source edited within the same second that its object was compiled
is "not newer" to make, and the object is NOT rebuilt. No warning, no error: make prints nothing and the test that
runs next is the OLD binary.

This bit twice on 2026-09-15, both times during a negative control - edit, build, test, restore, build, test. On the
restore the object did not rebuild, so the test reported the PLANTED defect's result against the restored source.
Both times it failed in the safe direction and looked like a real failure of correct code. The dangerous direction is
the same mechanism: make a fix, rebuild inside the same second, and the suite passes on a binary that never contained
the fix. Nothing distinguishes that from a fix that works.

IT DEFEATS THE ONE PRACTICE EVERYTHING ELSE RESTS ON. "A new check is not trusted until it has been seen to fail" is
worth nothing if the run that was supposed to fail, and the run that was supposed to pass, were the same binary. That
makes this a check about whether the other checks mean anything, which is why it runs with them rather than by hand.

AND IT IS INVISIBLE TO A BUILD ID. build_id.c is recompiled every time, so the reported build id is always fresh even
when every other object is stale. A gate that checks the id sees a correct, current-looking binary.
"""
import os, sys

def main(build, src):
    bad = []
    for root, _, files in os.walk(build):
        for f in files:
            if not f.endswith(".o"):
                continue
            obj = os.path.join(root, f)
            for ext in (".c", ".S"):
                s = os.path.join(src, os.path.relpath(obj, build)[:-2] + ext)
                if os.path.exists(s):
                    if os.stat(s).st_mtime_ns >= os.stat(obj).st_mtime_ns:
                        bad.append((s, obj))
                    break
    if bad:
        print("STALE OBJECTS - make did not rebuild these, and whatever ran next was the OLD code:")
        for s, o in bad:
            print("  %s is newer than (or the same age as) %s" % (s, o))
        print("\nGNU Make 3.81 compares mtimes to the SECOND. Run `touch` on the sources above, or `make clean`,")
        print("and re-run - any result measured before now was taken from a binary that does not match the tree.")
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "build/src", sys.argv[2] if len(sys.argv) > 2 else "src"))
