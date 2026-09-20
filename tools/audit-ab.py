#!/usr/bin/env python3
"""audit-ab.py - the same real audit run against two backends, for a controlled prefill comparison.

    python3 tools/audit-ab.py <arm-name> <endpoint> <out.jsonl> [min_tokens] [max_tokens]

The task is real: for each VLC source file that touches paths, config or the environment, ask whether it writes
outside the application folder. Big cold input, short structured answer, nothing cached between files - the shape
the disaggregated path is for, and the opposite shape to the agentic build measured earlier tonight.

Identical file list and identical prompts in both arms, capped at the same max_tokens, so the only difference is
which GPU processes the prompt. Timings come from the server's own block, not a wall clock.
"""
import json, os, sys, time, urllib.request

ARM, ENDPOINT, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
LO, HI = int(sys.argv[4]) if len(sys.argv) > 4 else 5300, int(sys.argv[5]) if len(sys.argv) > 5 else 44000
ASK = ("You are auditing this source file for PORTABILITY. A portable build must keep every file it writes inside "
       "its own application folder. List every place this file reads or writes a path OUTSIDE that folder - the "
       "user's home directory, system config, caches, temp, or the registry. For each: the function, the API used, "
       "and the path. If there are none, reply exactly NONE. Be terse; no preamble.\n\n=== %s ===\n%s")

files = [f.strip() for f in open("/tmp/cand.txt") if f.strip()]
sel = []
for f in files:
    p = os.path.join("/Volumes/Crucial_8TB/vlc-latest", f.lstrip("./"))
    if not os.path.exists(p): continue
    t = int(os.path.getsize(p) * 0.28)
    if LO <= t <= HI: sel.append((p, t))
sel.sort(key=lambda x: -x[1])
print(f"{ARM}: {len(sel)} files, ~{sum(t for _, t in sel):,} tokens estimated", flush=True)

with open(OUT, "a") as fh:
    for i, (path, est) in enumerate(sel, 1):
        src = open(path, encoding="utf-8", errors="replace").read()
        body = {"model": "qwen3-coder-next", "messages": [{"role": "user", "content": ASK % (os.path.basename(path), src)}],
                "max_tokens": 400, "temperature": 0.2, "seed": 20260920, "stream": False}
        t0 = time.perf_counter()
        try:
            r = urllib.request.urlopen(urllib.request.Request(ENDPOINT, json.dumps(body).encode(),
                {"Content-Type": "application/json"}), timeout=3600)
            d = json.loads(r.read()); wall = time.perf_counter() - t0
            tm = d.get("timings", {}) or {}
            rec = {"arm": ARM, "file": os.path.relpath(path, "/Volumes/Crucial_8TB/vlc-latest"), "wall_s": round(wall, 2),
                   "prompt_n": tm.get("prompt_n"), "prompt_ms": tm.get("prompt_ms"), "predicted_n": tm.get("predicted_n"),
                   "predicted_ms": tm.get("predicted_ms"), "cache_n": tm.get("cache_n"),
                   "answer": (d.get("choices") or [{}])[0].get("message", {}).get("content", "")[:2000]}
        except Exception as e:
            rec = {"arm": ARM, "file": os.path.relpath(path, "/Volumes/Crucial_8TB/vlc-latest"), "error": f"{type(e).__name__}: {e}"[:300]}
        fh.write(json.dumps(rec) + "\n"); fh.flush()
        pn, pm = rec.get("prompt_n"), rec.get("prompt_ms")
        print(f"  [{i}/{len(sel)}] {rec['file'][:52]:<52} " +
              (f"{pn:>6} tok prefill {pm/1000:6.1f}s = {pn/(pm/1000):6.0f} tok/s" if pn and pm else rec.get("error", "?")), flush=True)
