#!/usr/bin/env python3
"""loadgen.py — concurrent chat completions against tools/serve.sh's server; prints aggregate and per-request tok/s.
   python3 tools/loadgen.py <concurrency> <requests_per_thread> [max_tokens] [temperature]"""
import sys, json, time, threading, urllib.request, os
# LOADGEN_URL overrides the server (default: tools/serve.sh's); LOADGEN_MODEL adds a model name to each request (vLLM needs one).
url = (os.environ.get('LOADGEN_URL') or open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs', 'serve.url')).read().strip()) + '/v1/chat/completions'
model = os.environ.get('LOADGEN_MODEL')
C = int(sys.argv[1]); K = int(sys.argv[2]); MAXT = int(sys.argv[3]) if len(sys.argv) > 3 else 256; TEMP = float(sys.argv[4]) if len(sys.argv) > 4 else 0.6
prompts = ["Explain why the sky is blue to a curious ten-year-old.", "Write a short story, under 200 words, about a lighthouse keeper who finds a message in a bottle.",
           "List the steps to make sourdough bread from a starter, briefly.", "What are the main differences between TCP and UDP? Keep it concise.",
           "Summarize the plot of Romeo and Juliet in five sentences.", "Give me a Python function that checks whether a string is a palindrome, with a docstring.",
           "Describe the water cycle.", "What is the difference between weather and climate?"]
res = []; lock = threading.Lock()
def worker(i):
    for k in range(K):
        p = prompts[(i + k) % len(prompts)]
        req = {"messages": [{"role": "user", "content": p}], "temperature": TEMP, "max_tokens": MAXT, "seed": 7 + i}
        if model: req["model"] = model
        body = json.dumps(req).encode()
        t0 = time.time()
        try:
            j = json.loads(urllib.request.urlopen(urllib.request.Request(url, body, {"Content-Type": "application/json"}), timeout=900).read())
            t = j.get("timings", {}); n = t.get("predicted_n", 0); ms = t.get("predicted_ms", 0.0)
            if not n:  # vLLM: no timings block; count completion tokens, time the whole request
                n = j.get("usage", {}).get("completion_tokens", 0); ms = (time.time() - t0) * 1000.0
            with lock: res.append((n, ms, time.time() - t0, t.get("draft_n", 0), t.get("draft_n_accepted", 0)))
        except Exception as e:
            with lock: res.append((0, 0, time.time() - t0, 0, 0)); print("ERROR", str(e)[:120], file=sys.stderr)
T0 = time.time(); th = [threading.Thread(target=worker, args=(i,)) for i in range(C)]
[t.start() for t in th]; [t.join() for t in th]; wall = time.time() - T0
tok = sum(r[0] for r in res); ok = [r for r in res if r[0] > 0]
per = [r[0] / (r[1] / 1000.0) for r in ok if r[1] > 0]
dn = sum(r[3] for r in ok); da = sum(r[4] for r in ok)
print(f"concurrency {C} x {K} requests, max_tokens {MAXT}, temp {TEMP}: {len(ok)}/{len(res)} ok, {tok} tokens in {wall:.1f} s = "
      f"AGGREGATE {tok / wall:.1f} tok/s; per-request mean {sum(per) / max(len(per), 1):.1f} tok/s (min {min(per) if per else 0:.1f}, max {max(per) if per else 0:.1f}); "
      f"draft accept {100.0 * da / dn if dn else 0:.0f}%")
