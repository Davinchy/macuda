#!/usr/bin/env python3
"""A coding-agent conversation replayed against a running llama-server, to measure what llama-bench cannot: prompt-cache reuse.

    python tools/agent-replay.py --url http://127.0.0.1:8090 --label 3060 --out logs/replay-3060.json

Turn 1 is a ~20k-token prompt (system prompt, tool definitions, a few source files). Every later turn appends ~3k tokens of
fake tool output and asks for ~500 tokens back. The whole conversation so far is resent every turn, byte for byte, which is
what a coding agent does and what cache_prompt exists for. Per turn: prompt tokens total, tokens the server actually
processed (its own timings), time to first token measured here, generation tokens per second, and the turn's wall time. A
turn whose processed count is close to its total is flagged: the cache was not reused and the whole context was recomputed.
Sizes are hit by tokenizing through the server itself, so they are right for whatever model is loaded.
"""
import argparse, json, random, sys, time, urllib.request

def post(url, path, body, stream=False):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=3600)

def ntok(url, text):
    with post(url, "/tokenize", {"content": text}) as r: return len(json.load(r)["tokens"])

def fill_to(url, text, target, gen):
    """append generated lines until the text is at least `target` tokens (measured by the server)"""
    while ntok(url, text) < target: text += gen()
    return text

def source_file(rng, name, n):
    out = [f"// ---- {name} ----"]
    for i in range(n):
        k = rng.randrange(4)
        if k == 0: out.append(f"static int handle_{name.split('.')[0]}_{i}(struct ctx *c, uint32_t v{i}) {{ if (!c) return -EINVAL; c->slot[{i % 16}] = v{i} ^ 0x{rng.randrange(1<<24):06x}; return c->slot[{i % 16}] & 0x{rng.randrange(1<<8):02x}; }}")
        elif k == 1: out.append(f"#define CFG_{name.split('.')[0].upper()}_{i} 0x{rng.randrange(1<<16):04x}  // {rng.choice(['read at open','written by the firmware','a knob','never set'])}")
        elif k == 2: out.append(f"  // {rng.choice(['TODO','NOTE','FIXME'])}: {rng.choice(['the reply carries the function it answers and nothing else','bounded by the ring size','only on the live path','replay would diverge here'])} ({rng.randrange(100, 999)})")
        else: out.append(f"  if (q->tx.msgCount <= {rng.randrange(1, 64)}) return tinynv_fail(\"{rng.choice(['queue too small','no free slot','pointer outside the ring'])}: %u\", q->tx.msgCount);")
    return "\n".join(out) + "\n"

def tool_output(rng, turn):
    lines = [f"[tool: {rng.choice(['grep','read_file','run_tests','git diff'])}] turn {turn}"]
    for i in range(rng.randrange(60, 90)):
        lines.append(f"src/{rng.choice(['gsp','exec','dev','mmu','flcn'])}.c:{rng.randrange(1, 2400)}: {rng.choice(['tinynv_fail(','if (rc) return','memcpy(','nv_wr32(','// ','static int '])}{rng.choice(['queue','slot','desc','window','region'])}_{rng.randrange(1000)} {rng.choice(['ok','FAIL','skipped','0x%08x' % rng.randrange(1<<32)])}")
    return "\n".join(lines) + "\n"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--label", default="run")
    ap.add_argument("--turns", type=int, default=10)
    ap.add_argument("--first-tokens", type=int, default=20000)
    ap.add_argument("--turn-tokens", type=int, default=3000)
    ap.add_argument("--out-tokens", type=int, default=500)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    url = a.url.rstrip("/")

    system = ("You are a coding agent working in a C driver repository. Use the tools below. Think step by step, cite file:line, "
              "and never guess at hardware state you have not read.\n")
    tools = json.dumps([{"name": t, "description": f"{t} over the repository", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "query": {"type": "string"}}}}
                        for t in ["read_file", "grep", "list_dir", "run_tests", "git_diff", "apply_patch"]], indent=1)
    prompt = system + "\n## Tools\n" + tools + "\n\n## Files\n"
    files = ["gsp.c", "exec.c", "dev.c", "mmu.c", "flcn.c", "submit.c"]
    per_file = max(40, a.first_tokens // (len(files) * 28))
    for f in files: prompt += source_file(rng, f, per_file)
    prompt = fill_to(url, prompt, a.first_tokens, lambda: source_file(rng, "extra.c", 30))
    prompt += "\n## Task\nFind why the second boot in one enumeration loses the card's windows, and propose a fix with file:line references.\n"

    rows = []; t_all = time.time()
    for turn in range(1, a.turns + 1):
        if turn > 1:
            add = fill_to(url, "", a.turn_tokens, lambda: tool_output(rng, turn))
            prompt += "\n### Tool result\n" + add + "\n### Continue. Explain what this shows and what to run next.\n"
        total = ntok(url, prompt)
        body = {"prompt": prompt, "n_predict": a.out_tokens, "ignore_eos": True, "temperature": 0, "cache_prompt": True, "stream": True}
        t0 = time.time(); ttft = None; text = ""; timings = {}
        with post(url, "/completion", body, stream=True) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data: "): continue
                ev = json.loads(line[6:])
                if ttft is None and ev.get("content"): ttft = time.time() - t0
                text += ev.get("content", "")
                if ev.get("timings"): timings = ev["timings"]
        wall = time.time() - t0
        pn = timings.get("prompt_n", -1); pms = timings.get("prompt_ms", 0.0)
        gn = timings.get("predicted_n", -1); gms = timings.get("predicted_ms", 0.0)
        cached = timings.get("cache_n", timings.get("tokens_cached", total - pn if pn >= 0 else -1))
        reprocessed = pn >= 0 and pn >= 0.9 * total
        rows.append({"turn": turn, "prompt_tokens": total, "processed": pn, "cached": cached, "ttft_s": round(ttft or 0, 3),
                     "prompt_tps": round(pn / pms * 1000, 1) if pms else None, "gen_tokens": gn,
                     "gen_tps": round(gn / gms * 1000, 2) if gms else None, "turn_s": round(wall, 2), "cache_miss": reprocessed})
        print(f"turn {turn:2d}: prompt {total:6d} processed {pn:6d} cached {cached:6d} ttft {ttft or 0:6.2f}s gen {gn} @ {rows[-1]['gen_tps']} t/s  turn {wall:6.2f}s{'  <-- CACHE MISS: whole context reprocessed' if reprocessed else ''}", flush=True)
        prompt += text
    total_s = time.time() - t_all
    print(f"total wall time for {a.turns} turns: {total_s:.1f} s ({a.label})")
    if a.out:
        with open(a.out, "w") as f: json.dump({"label": a.label, "turns": rows, "total_s": round(total_s, 2), "args": vars(a)}, f, indent=1)

if __name__ == "__main__": main()
