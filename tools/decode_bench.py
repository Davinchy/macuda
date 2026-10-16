#!/usr/bin/env python3
"""decode_bench.py: decode tok/s against a running llama-server, greedy, fixed prompts.
  python3 tools/decode_bench.py <port> <label> <out.jsonl> [prompts: short,8k,32k,edit] [n_predict=256] [reps=3]
Each prompt: one warm-up (fills the prompt cache), then `reps` timed generations of n_predict tokens with cache_prompt
(prefill is cached, so the number is decode). Records predicted_per_second, draft stats and the token ids (for exactness:
two builds that should compute the same thing must produce the same ids).
  short  a 40-word question         8k / 32k  that many tokens of C++ source + a coding task (ignore_eos: exactly n_predict)
  edit   rewrite one real function with a small change - the case a drafter (MTP, n-gram) is best at; a different
         function per rep, so a drafter that remembers earlier outputs gets no free hits
The C++ source is llama.cpp's tools/server/server-context.cpp from the checkout beside this repo (CODE=<file> overrides).
The numbers in the README were taken with upstream llama.cpp d812350's copy (2026-10-05):
  git -C llama.cpp fetch origin d812350 && git -C llama.cpp show d812350:tools/server/server-context.cpp > /tmp/sc.cpp; CODE=/tmp/sc.cpp ..."""
import json, os, statistics, sys, time, urllib.request

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = os.environ.get("CODE", os.path.join(R, "llama.cpp", "tools", "server", "server-context.cpp"))
SHORT = ("Write a short explanation of how a hash table handles collisions, then give a small Python example that implements "
         "separate chaining with insert, get and delete, and explain its average and worst-case time complexity.\n")

port, label, out = sys.argv[1], sys.argv[2], sys.argv[3]
ctxs = (sys.argv[4] if len(sys.argv) > 4 else "short,8k").split(",")
NP = int(sys.argv[5]) if len(sys.argv) > 5 else 256
REPS = int(sys.argv[6]) if len(sys.argv) > 6 else 3
base = f"http://127.0.0.1:{port}"

def post(path, obj):
    r = urllib.request.urlopen(urllib.request.Request(base + path, json.dumps(obj).encode(),
                               {"Content-Type": "application/json"}), timeout=3600)
    return json.loads(r.read())

code = open(CODE).read()
short = SHORT
ask = ("\n\n// Task: add a function `int64_t server_slot_count_idle(const server_context & ctx)` that returns the number "
       "of idle slots, with a doc comment, then write a unit test for it.\n")

def edit_prompt(k):
    # a coding-agent edit: one real function from the file, returned whole with a small change (k picks the function)
    starts = [i for i in range(len(code)) if code.startswith("\nstatic ", i) or code.startswith("\n    void ", i)]
    s = starts[(k * 7 + 3) % len(starts)]; body = code[s:s + 2400]
    return ("Here is a C++ function from llama.cpp's server:\n\n```cpp" + body + "\n```\n\nRewrite it in full, unchanged "
            "except: rename every local variable `i` to `idx` and add a one-line comment above the first statement. "
            "Output only the code.\n\n```cpp")

def prompt_ids(c, k=0):
    if c == "edit":
        return post("/tokenize", {"content": edit_prompt(k)})["tokens"]
    if c == "short":
        return post("/tokenize", {"content": short})["tokens"]
    n = {"8k": 8192, "32k": 32768}[c]
    body = post("/tokenize", {"content": code})["tokens"]
    tail = post("/tokenize", {"content": ask})["tokens"]
    while len(body) < n:
        body = body + body
    return body[: n - len(tail)] + tail

with open(out, "a") as f:
    for c in ctxs:
        ids = prompt_ids(c)
        req = {"prompt": ids, "n_predict": NP, "temperature": 0, "top_k": 1, "seed": 1, "cache_prompt": True,
               "return_tokens": True, "stream": False, "id_slot": 0, "ignore_eos": c == "8k"}
        t0 = time.time(); post("/completion", dict(req, n_predict=8)); warm = time.time() - t0
        tps, toks, drafts = [], None, []
        for r in range(REPS):
            if c == "edit":   # a DIFFERENT edit task per rep: a drafter that remembers earlier outputs gets no free hits
                ids = prompt_ids(c, r + 1); post("/completion", dict(req, prompt=ids, n_predict=1))
                req = dict(req, prompt=ids)
            res = post("/completion", req)
            t = res.get("timings", {})
            tps.append(t.get("predicted_per_second") or 0)
            drafts.append((t.get("draft_n"), t.get("draft_n_accepted")))
            if toks is None:
                toks = res.get("tokens")
        row = {"label": label, "ctx": c, "prompt_n": len(ids), "n_predict": NP, "warm_s": round(warm, 1),
               "tps": [round(x, 2) for x in tps], "tps_median": round(statistics.median(tps), 2),
               "draft": drafts, "tokens": toks}
        f.write(json.dumps(row) + "\n"); f.flush()
        acc = ""
        if drafts[0][0]:
            dn, da = drafts[-1]; acc = f" draft {da}/{dn} accepted ({100*da/max(dn,1):.0f}%)"
        print(f"{label:28s} {c:6s} decode {statistics.median(tps):6.2f} tok/s  reps {[round(x,1) for x in tps]}{acc}", flush=True)
