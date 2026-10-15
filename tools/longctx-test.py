#!/usr/bin/env python3
"""A long-context test over this repository's own source: the driver's files fill the context, ten questions whose answers
are in them are asked one at a time against a running llama-server, and the answers are written out for a judge.

    python tools/longctx-test.py --url http://127.0.0.1:8090 --label lfm-3060 --files tools/longctx-files.txt --budget 115000 --out logs/longctx-lfm-3060.json

The prompt is the same preamble and the same files in the same order every time, cut at --budget tokens as the server
tokenizes them, so every method and every side sees the same context. Each question is one chat request with cache_prompt, thinking off where the template allows:
the first pays for the whole context (its time to first token is the prefill), the rest pay for their own tokens only,
which is what an agent's follow-up questions cost. Per question: prompt tokens, tokens processed, time to first token,
generation tokens/s, wall time, and the answer verbatim. Nothing here judges; the judge reads the answers against the code.
"""
import argparse, json, os, sys, time, urllib.request

QUESTIONS = [
    "How many boot attempts does tinynv_gpu_boot_firmware make by default on a live card that boots through the VBIOS path, and which environment variable overrides that number?",
    "What value does the driver send for the RMPcieLinkSpeed registry entry when the requested link speed is gen3, and what does bit 31 of that value mean?",
    "Which function writes the BARs and the command register back when they are found cleared, and in which file is it defined?",
    "What is TINYNV_FW_RESERVE_TOP, in megabytes?",
    "How many host mappings will the broker ever hand out, and how many bytes is the user shared data (RUSD) buffer that costs one of them?",
    "What is the numeric value of TINYNV_CTRL_GET_GSP_RM_FREE_HEAP?",
    "What are the two FWSEC command ids the driver builds images for, and what does each command do?",
    "While waiting for the firmware's start-up notice, how often does the driver look at the card's PCI configuration, and what does it do if ten seconds after the windows were put back the notice still has not come?",
    "Which environment variable, set to 0, restores the old behaviour of leaving the card idle-warm instead of unloading the firmware at exit?",
    "Which environment variable setting stops the driver from sending the RMPcieLinkSpeed entry at all, and what does the driver send to the firmware's registry in that case?",
]

def post(url, path, body):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=7200)

def ntok(url, text):
    with post(url, "/tokenize", {"content": text}) as r: return len(json.load(r)["tokens"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--label", default="run")
    ap.add_argument("--files", required=True, help="one path per line, in the order they fill the context")
    ap.add_argument("--budget", type=int, default=115000, help="tokens of source to include")
    ap.add_argument("--max-answer", type=int, default=1000)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    url = a.url.rstrip("/")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    preamble = ("You are a coding agent working in a C driver repository. The repository's source follows, file by file. "
                "Answer each question from the source alone, briefly and precisely, and name the file the answer comes from.\n\n")
    corpus = ""; used = ntok(url, preamble); placed = []
    for line in open(os.path.join(root, a.files)):
        path = line.strip()
        if not path or path.startswith("#"): continue
        text = open(os.path.join(root, path), errors="replace").read()
        block = f"\n// ==== {path} ====\n" + text
        n = ntok(url, block)
        if used + n > a.budget:
            # cut the last file by lines to what fits, measured by the server, then stop
            lines = block.split("\n"); lo, hi = 0, len(lines)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if used + ntok(url, "\n".join(lines[:mid])) <= a.budget: lo = mid
                else: hi = mid - 1
            block = "\n".join(lines[:lo]) + "\n// (file cut here to fit the context)\n"; n = ntok(url, block)
            corpus += block; placed.append({"path": path, "start": used, "tokens": n, "cut": True}); used += n
            break
        corpus += block; placed.append({"path": path, "start": used, "tokens": n, "cut": False}); used += n
    print(f"context: {used} tokens of source in {len(placed)} files (last {'cut' if placed[-1]['cut'] else 'whole'}: {placed[-1]['path']})", flush=True)
    for p in placed: print(f"  {p['start']:7d} {p['tokens']:6d} {p['path']}{' (cut)' if p['cut'] else ''}", flush=True)
    rows = []; t_all = time.time()
    for i, q in enumerate(QUESTIONS, 1):
        # The model's own chat template, through the OpenAI-style endpoint: the source is the user message and the question
        # is its last paragraph, so the prefix up to the question is identical every time and the server's cache reuses it.
        # Thinking is switched off where the template has the switch; where it does not, the server splits what the model
        # thinks into reasoning_content and the judge reads content. max_tokens leaves room for either.
        user = corpus + f"\n\n## Question {i}\n{q}\n"
        total = ntok(url, preamble + user)
        body = {"messages": [{"role": "system", "content": preamble.strip()}, {"role": "user", "content": user}],
                "max_tokens": a.max_answer, "temperature": 0, "cache_prompt": True, "stream": True,
                "chat_template_kwargs": {"enable_thinking": False}}
        t0 = time.time(); ttft = None; text = ""; think = ""; timings = {}; finish = None
        with post(url, "/v1/chat/completions", body) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data: ") or line == "data: [DONE]": continue
                ev = json.loads(line[6:])
                for ch in ev.get("choices", []):
                    d = ch.get("delta", {}) or {}
                    if d.get("content"):
                        if ttft is None: ttft = time.time() - t0
                        text += d["content"]
                    if d.get("reasoning_content"):
                        if ttft is None: ttft = time.time() - t0
                        think += d["reasoning_content"]
                    if ch.get("finish_reason"): finish = ch["finish_reason"]
                if ev.get("timings"): timings = ev["timings"]
        wall = time.time() - t0
        pn = timings.get("prompt_n", -1); pms = timings.get("prompt_ms", 0.0); gn = timings.get("predicted_n", -1); gms = timings.get("predicted_ms", 0.0)
        rows.append({"q": i, "question": q, "prompt_tokens": total, "processed": pn, "ttft_s": round(ttft or 0, 2),
                     "prompt_tps": round(pn / pms * 1000, 1) if pms else None, "gen_tokens": gn,
                     "gen_tps": round(gn / gms * 1000, 2) if gms else None, "wall_s": round(wall, 2), "finish": finish,
                     "answer": text.strip(), "thinking_chars": len(think), "thinking": think.strip()[:4000]})
        print(f"q{i:2d}: prompt {total:6d} processed {pn:6d} ttft {ttft or 0:7.1f}s gen {gn} @ {rows[-1]['gen_tps']} t/s wall {wall:7.1f}s finish {finish} thinking {len(think)} chars", flush=True)
        print("     " + text.strip().replace("\n", "\n     ")[:600], flush=True)
    total_s = time.time() - t_all
    print(f"total wall time for {len(QUESTIONS)} questions: {total_s:.1f} s ({a.label}); prefill {rows[0]['ttft_s']} s for {rows[0]['processed']} tokens", flush=True)
    if a.out:
        with open(a.out, "w") as f: json.dump({"label": a.label, "context_tokens": used, "files": placed, "questions": rows, "total_s": round(total_s, 2), "args": vars(a)}, f, indent=1)
        with open(a.out.rsplit(".", 1)[0] + ".answers.md", "w") as f:
            f.write(f"# {a.label}: {used} tokens of context, {total_s:.1f} s\n\n")
            for r in rows: f.write(f"## Q{r['q']}: {r['question']}\n\n{r['answer']}\n\n(ttft {r['ttft_s']} s, {r['gen_tokens']} tokens at {r['gen_tps']} t/s, wall {r['wall_s']} s, finish {r['finish']}, {r['thinking_chars']} chars of thinking)\n\n")

if __name__ == "__main__": main()
