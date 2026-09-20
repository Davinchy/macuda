#!/usr/bin/env python3
"""disagg-screen.py - rank GGUF models for disaggregated inference WITHOUT touching the card or reading any weights.

Disaggregated inference here means: the 5090 prefills, the Mac's own GPU decodes, and the slot state is handed over
between them. Whether that is worth doing for a given model is decided by one number, and this reads it out of the
file's header in milliseconds.

The arithmetic, from tools/disagg-router.py's cost model (fitted to six measured prefills, windows 2 and 3):

    metal_seconds = tokens / metal_rate
    card_seconds  = ceil(tokens / card_ubatch) * (streamed_gib / link_rate) + tokens / card_rate

The part people get wrong is the ceil(): ggml re-copies every host-resident expert FOR EVERY UBATCH it evaluates, so
the expert stream is paid once per 24,576 tokens, not once per prompt. That means the speedup does NOT keep growing
with prompt length - it climbs to a plateau and stays there, and the plateau is set almost entirely by how many GiB
have to cross the link:

    plateau = (card_ubatch / metal_rate) / (streamed_gib / link_rate + card_ubatch / card_rate)

So the whole game is minimising streamed_gib - the expert bytes that do NOT stay resident on the card's 32 GB. A
model small enough to live on the card entirely does not want this path at all: card-only measured 5.5-7.3x over
Metal, which beats any disaggregated figure here.

    python3 tools/disagg-screen.py /Volumes/Models              # rank every .gguf found
    python3 tools/disagg-screen.py a.gguf b.gguf --resident 30  # how much fits on the card, GiB (default 28)
"""
import argparse, importlib.util, math, os, sys

GIB = float(1 << 30)
CARD_VRAM_GIB = 30.0          # what a 32 GB card can actually hold once the driver's own reservations are out
METAL_RATE = 660.0            # Metal prefill tok/s, measured
CARD_RATE = 5150.0            # card marginal prefill tok/s, measured (driver ad308bd)
LINK_RATE = 5.46              # GiB/s the expert stream crosses Thunderbolt at, measured
CARD_UBATCH = 24576           # the card server's -ub: the stream is paid once per ubatch


def load_router():
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location("disagg_router", os.path.join(here, "disagg-router.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def card_seconds(tokens, streamed):
    return math.ceil(tokens / CARD_UBATCH) * (streamed / LINK_RATE) + tokens / CARD_RATE


def screen(router, path, resident_gib):
    """Header only: the key/value table and the tensor directory, a few MB at the front of the file."""
    try:
        lay = router.expert_layout(path)
    except Exception as e:
        return {"path": path, "error": str(e)[:60]}
    total_bytes = os.path.getsize(path)
    total_gib = total_bytes / GIB
    expert_gib = lay["total"] / GIB if lay.get("total") else 0.0
    nblocks = len(lay.get("blocks") or [])
    # What must stream: whatever does not fit on the card. Non-expert weight stays resident by preference, so the
    # card holds (resident - non_expert) GiB of experts and the rest crosses the link every ubatch.
    streamed = max(total_gib - resident_gib, 0.0)
    fits = total_gib <= resident_gib
    row = {"path": path, "name": os.path.basename(path), "total_gib": total_gib, "expert_gib": expert_gib,
           "blocks": nblocks, "n_expert": lay.get("n_expert") or 0, "used": lay.get("n_expert_used") or 0,
           "arch": lay.get("arch") or "?", "streamed": streamed, "fits": fits}
    if fits:
        row["verdict"] = "card-only (5.5-7.3x); do not disaggregate"
        row["plateau"] = float("nan")
        return row
    # A DENSE model larger than the card is not a candidate at all, and saying "1.0x" would flatter it. What streams
    # under this path is expert layers (llama.cpp's --n-cpu-moe); a dense model has none, so the overflow cannot be
    # streamed - it would have to be offloaded to the CPU and computed there, which is a different and much worse
    # arrangement than the one these numbers describe. Caught 2026-09-19 by asking what the screener would say about a
    # 70B dense: the first version answered 7.8x, because it measured only expert bytes and a dense model has zero.
    if row["n_expert"] == 0:
        row["plateau"] = float("nan")
        row["verdict"] = "DENSE and bigger than the card: this path does not apply"
        return row
    row["plateau"] = (CARD_UBATCH / METAL_RATE) / (streamed / LINK_RATE + CARD_UBATCH / CARD_RATE)
    # Break-even prompt: below this, Metal alone is faster because the stream is not yet amortised.
    per_tok = 1.0 / METAL_RATE - 1.0 / CARD_RATE
    row["breakeven"] = (streamed / LINK_RATE) / per_tok
    row["verdict"] = ("strong" if row["plateau"] >= 3.0 else "useful" if row["plateau"] >= 1.8 else "marginal")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="GGUF files, or directories to search")
    ap.add_argument("--resident", type=float, default=CARD_VRAM_GIB, help="GiB the card can hold (default %(default)s)")
    a = ap.parse_args()

    files = []
    for p in a.paths:
        if os.path.isdir(p):
            for root, _, names in os.walk(p):
                files += [os.path.join(root, n) for n in names if n.endswith(".gguf")]
        elif p.endswith(".gguf"):
            files.append(p)
    files = sorted(set(files))
    if not files:
        print("no .gguf files found"); return 2

    router = load_router()
    rows = [screen(router, f, a.resident) for f in files]
    ok = [r for r in rows if "error" not in r]
    for r in rows:
        if "error" in r:
            print(f"  could not read {os.path.basename(r['path'])}: {r['error']}")

    ok.sort(key=lambda r: (-(r["plateau"] if r["plateau"] == r["plateau"] else -1)))
    print(f"\ncard holds {a.resident:.0f} GiB; Metal prefill {METAL_RATE:.0f} tok/s, card {CARD_RATE:.0f} tok/s marginal,")
    print(f"link {LINK_RATE:.2f} GiB/s, expert stream paid once per {CARD_UBATCH} tokens.\n")
    print(f"{'model':<44}{'size':>8}{'experts':>9}{'stream':>8}{'plateau':>9}{'break-even':>11}  verdict")
    for r in ok:
        pl = "-" if r["plateau"] != r["plateau"] else f"{r['plateau']:.1f}x"
        be = f"{r['breakeven']:,.0f} tok" if "breakeven" in r else "-"
        print(f"{r['name'][:43]:<44}{r['total_gib']:>7.1f}G{r['expert_gib']:>8.1f}G{r['streamed']:>7.1f}G{pl:>9}{be:>11}  {r['verdict']}")
    print("\nplateau = the speedup a long prompt converges to; break-even = the prompt length below which Metal alone wins.")
    print("A model that fits the card entirely is not a disaggregation candidate - run it card-only instead.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
