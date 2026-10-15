// What this build actually is.
//
// Three times in this session, on both sides of the work, a run has been diagnosed against code it was not running: a
// stale object file, a stale archive member, and a binary linked before a fix landed. The last one cost a hardware run
// and a replug, and was only caught because the driver's error message had been reworded, so the old wording was an
// accidental version marker.
//
// Making that deliberate is a few lines. The commit and the working-tree state are baked in at build time, so anything
// that reports them - a test, the shim's log, a bug report - says which code produced it, and "are you running the fix?"
// is a question with an answer rather than a thing to relink and hope about.
#include "tinynv.h"

#ifndef TINYNV_BUILD_ID
#define TINYNV_BUILD_ID "unknown (built outside the makefile)"
#endif

const char *tinynv_build_id(void) { return TINYNV_BUILD_ID; }
