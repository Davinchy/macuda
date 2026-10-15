# private/

Anything that must never reach the public repository lives here: NVIDIA-licensed material, captured ROM/VBIOS images,
model weights, notes, and any local tooling you do not intend to publish.

Three things keep it out of git:
1. `.gitignore` ignores `private/` (this README is the one exception, so the convention travels with the repo).
2. `.git/info/exclude` ignores it again, locally, so editing `.gitignore` cannot un-ignore it.
3. `.git/hooks/pre-commit` refuses any staged path under `private/`, `cuda-shim/libtinynv/third_party/`, the CUDA header
   trees, `traces/`, `models/`, and any `*.rom *.blob *.bin *.gguf *.trace` — even after `git add -f`.

The hook is local (`.git/hooks` is not versioned); a fresh clone gets 1 and 2 only.
