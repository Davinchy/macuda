#!/usr/bin/env python3
"""disagg-analyse.py - turn tools/disagg-sweep.sh's TSV into the two tables the experiment is for (2026-09-19).

    python3 tools/disagg-analyse.py [results.tsv]

Table 1, per model: what each prompt length cost on each half, and the advantage. Tests the claim that the advantage
RISES with context (measured 2.35x -> 5.55x on one model on 2026-09-19) rather than decaying.

Table 2, across models: the advantage against ACTIVE BYTES A TOKEN, read from each model's own header. That is the
claim the four models were chosen to test - that active parameters, not total size, decide whether this path pays.

Where a model has four lengths, the card's own cost model is refit to them:  ceil(n/ub)*stream + n/rate + fixed.
Three free parameters on four points is a weak fit and is reported with its residual, not asserted.
"""
import glob, importlib.util, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
UB = 24576


def screen_mod():
    spec = importlib.util.spec_from_file_location("disagg_screen", os.path.join(HERE, "disagg-screen.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


COLS = ["when", "side", "model", "prompt", "tokens", "prefill_s", "tok_s", "wall_s", "load_s"]


def read_rows(path):
    """Rows by SHAPE, not by trusting line 1 to be the header. Another tool writing its own header into this file
    (tools/card-uptime.sh did, when it inherited an exported OUT) used to make the whole parse read the wrong
    columns and fail on a KeyError far from the cause."""
    rows = []
    with open(path) as fh:
        head = COLS
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) != len(head) or f[0] == "when" or f[1] not in ("card", "metal"): continue
            d = dict(zip(head, f))
            for k in ("tokens", "prefill_s", "tok_s", "wall_s", "load_s"):
                try: d[k] = float(d[k])
                except (KeyError, ValueError): d[k] = float("nan")
            rows.append(d)
    return rows


def fit_card(points):
    """Least squares for (stream, 1/rate, fixed) in ceil(n/UB)*stream + n/rate + fixed, by normal equations on a
    3-column design matrix. Needs 4+ points to say anything; returns None below that."""
    if len(points) < 4: return None
    X = [[math.ceil(n / UB), n, 1.0] for n, _ in points]; y = [t for _, t in points]
    A = [[sum(X[k][i] * X[k][j] for k in range(len(X))) for j in range(3)] for i in range(3)]
    b = [sum(X[k][i] * y[k] for k in range(len(X))) for i in range(3)]
    for i in range(3):                                   # Gauss-Jordan, 3x3
        p = max(range(i, 3), key=lambda r: abs(A[r][i]))
        if abs(A[p][i]) < 1e-12: return None
        A[i], A[p] = A[p], A[i]; b[i], b[p] = b[p], b[i]
        d = A[i][i]; A[i] = [v / d for v in A[i]]; b[i] /= d
        for r in range(3):
            if r == i: continue
            f = A[r][i]; A[r] = [A[r][c] - f * A[i][c] for c in range(3)]; b[r] -= f * b[i]
    stream, inv_rate, fixed = b
    rms = math.sqrt(sum((math.ceil(n / UB) * stream + n * inv_rate + fixed - t) ** 2 for n, t in points) / len(points))
    return stream, (1 / inv_rate if inv_rate > 1e-9 else float("inf")), fixed, rms


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/Volumes/Crucial_8TB/disagg-bench/results.tsv"
    rows = read_rows(path)
    if not rows: print(f"no rows in {path}"); return 2
    models = []
    for r in rows:
        if r["model"] not in models: models.append(r["model"])

    print(f"\n{len(rows)} measurements from {path}\n")
    adv = {}
    for mdl in models:
        mine = [r for r in rows if r["model"] == mdl]
        lens = sorted({int(r["tokens"]) for r in mine})
        print(f"== {mdl}")
        print(f"   {'tokens':>8}{'ubatches':>10}{'card s':>9}{'Metal s':>9}{'card tok/s':>12}{'Metal tok/s':>13}{'advantage':>11}")
        for n in lens:
            c = [r for r in mine if int(r["tokens"]) == n and r["side"] == "card"]
            m = [r for r in mine if int(r["tokens"]) == n and r["side"] == "metal"]
            cs = min(r["prefill_s"] for r in c) if c else float("nan")
            ms = min(r["prefill_s"] for r in m) if m else float("nan")
            a = ms / cs if c and m and cs > 0 else float("nan")
            # bucket to the nearest thousand: a slice asked for as 24,000 tokens is recorded as 23,999 (the last
            # token is left for the decode side), and keying on the exact count silently empties the table
            if a == a: adv.setdefault(mdl, {})[round(n / 1000) * 1000] = a
            f = lambda v, w, p=1: (f"{v:>{w}.{p}f}" if v == v else f"{'-':>{w}}")
            print(f"   {n:>8}{math.ceil(n/UB):>10}{f(cs,9)}{f(ms,9)}{f(n/cs if cs==cs and cs>0 else float('nan'),12,0)}"
                  f"{f(n/ms if ms==ms and ms>0 else float('nan'),13,0)}{(f'{a:.2f}x' if a==a else '-'):>11}")
        pts = [(int(r["tokens"]), r["prefill_s"]) for r in mine if r["side"] == "card" and r["prefill_s"] == r["prefill_s"]]
        fit = fit_card(sorted(set(pts)))
        if fit:
            s, rate, fixed, rms = fit
            print(f"   card fit: ceil(n/{UB})*{s:.2f}s + n/{rate:,.0f} + {fixed:.2f}s   (rms {rms:.2f} s)")
        print()

    # Across models: the lever the four were chosen to separate.
    sc = screen_mod(); router = sc.load_router()
    print("== across models: advantage against what each model actually touches per token")
    print(f"   {'model':<26}{'size':>8}{'active':>9}{'stream':>9}" + "".join(f"{str(n//1000)+'k':>9}" for n in (12000, 24000, 26000, 48000)))
    for mdl in models:
        path_row = MODEL_FILES.get(mdl)
        size = active = stream = float("nan")
        if path_row and os.path.exists(path_row):
            # a shard-1 path names the whole split: glob its siblings, or the model reads as its first shard
            # (and for a metadata-only first shard, as 10 MB of nothing)
            mm = sc.SPLIT_RE.match(os.path.basename(path_row))
            shards = sorted(glob.glob(os.path.join(os.path.dirname(path_row),
                       mm.group("base") + "-*-of-" + mm.group("count") + ".gguf"))) if mm else [path_row]
            row = sc.screen(router, path_row, 30.0, shards or [path_row])
            size, active = row.get("total_gib", float("nan")), row.get("active", float("nan"))
            stream = row.get("streamed", float("nan"))
        cells = "".join((f"{adv.get(mdl, {}).get(n, float('nan')):>8.2f}x" if adv.get(mdl, {}).get(n) else f"{'-':>9}")
                        for n in (12000, 24000, 26000, 48000))
        g = lambda v: f"{v:>7.1f}G" if v == v else f"{'-':>8}"
        print(f"   {mdl[:25]:<26}{g(size)}{g(active):>9}{g(stream):>9}{cells}")
    print("\nactive = GiB a token touches (non-expert weight + the routed fraction of the experts), from the header.")
    print("If active parameters are the lever, the advantage column rises with it across models at the same length.")
    return 0


MODEL_FILES = {
    "coder-next-80b-a3b": "/Volumes/Crucial_8TB/Models/Qwen3-Coder-Next-GGUF/Qwen3-Coder-Next-UD-Q4_K_XL.gguf",
    "gpt-oss-120b": "/Volumes/Crucial_8TB/Models/gpt-oss-120b-GGUF/gpt-oss-120b-MXFP4.gguf",
    "deepseek-v4-flash-q2": "/Volumes/Crucial_8TB/Models/DeepSeek-V4-Flash-0731-GGUF/DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00001-of-00003.gguf",
    "qwen3.5-122b-a10b": "/Volumes/Crucial_8TB/Models/Qwen3.5-122B-A10B-GGUF/Qwen3.5-122B-A10B-Q4_K_M-00001-of-00002.gguf",
    "llama4-scout-109b-a17b": "/Volumes/Crucial_8TB/Models/Llama-4-Scout-17B-16E-Instruct-GGUF/Llama-4-Scout-17B-16E-Instruct-Q4_K_M-00001-of-00002.gguf",
    "mixtral-8x22b-a39b": "/Volumes/Crucial_8TB/Models/Mixtral-8x22B-Instruct-v0.1-GGUF/Mixtral-8x22B-Instruct-v0.1.Q4_K_S-00001-of-00002.gguf",
}

if __name__ == "__main__":
    sys.exit(main())
