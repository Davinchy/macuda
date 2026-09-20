#!/bin/sh
# Print the macOS SDK this tree builds against. The machine changed under the tree three times on 2026-09-14 (Command
# Line Tools 27.0 at 09:37, macOS 27 and a reboot, Xcode 27 at 15:32, which took away the 26.2 SDK that built the
# validated binaries), so a single hardcoded path is a guard that fires rather than a build, and a version heuristic is
# a guess. A link is a test: the first candidate that links a one-line program with this machine's linker is the answer.
#   SDKROOT, if set, is honoured or refused - never silently replaced. Otherwise, in order: the SDK that built the
#   validated set (26.2, should it come back), the oldest SDK still on the box below 27 (the Command Line Tools' 26.5,
#   by its resolved path, since the symlinks beside it moved today), then whatever the selected toolchain calls current.
t=${TMPDIR:-/tmp}/sdk-probe-$$; trap 'rm -f "$t.c" "$t.out"' EXIT; printf 'int main(void){return 0;}\n' > "$t.c"
links() { [ -f "$1/usr/lib/libSystem.B.tbd" ] && cc -isysroot "$1" "$t.c" -o "$t.out" >/dev/null 2>&1; }
if [ -n "${SDKROOT:-}" ]; then
  links "$SDKROOT" && { echo "$SDKROOT"; exit 0; }
  echo "sdk.sh: SDKROOT=$SDKROOT does not link a one-line program on this machine" >&2; exit 1
fi
for sdk in /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX26.2.sdk \
           /Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk \
           "$(xcrun --sdk macosx --show-sdk-path 2>/dev/null)"; do
  [ -n "$sdk" ] && links "$sdk" && { echo "$sdk"; exit 0; }
done
echo "sdk.sh: no macOS SDK on this machine links a one-line program; set SDKROOT to one that does" >&2; exit 1
