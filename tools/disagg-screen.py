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

A model small enough to live on the card entirely does not want this path at all: card-only measured 5.5-7.3x over
Metal, which beats any disaggregated figure here.

The trap, and it reverses the obvious conclusion: metal_rate and card_rate were BOTH measured on one model
(Coder-Next 80B-A3B, ~3.35 GB of active bytes a token). Prefill is compute-bound, so both scale down together as a
model's ACTIVE parameters rise - but the expert stream is fixed seconds and does not scale at all. More active
parameters therefore amortise the stream against more compute, and the speedup climbs toward the raw card:Metal
compute ratio of ~7.8x. Judging candidates by total size alone says "smaller is better"; judging them properly says
the opposite, because active parameters are the lever. Two 60 GB models, one with 8 GB active and one with 17 GB,
screen at 5.3x and 6.4x. So the models a Mac is WORST at - high active-parameter mixtures - are the best ones to put
on this path, which is also where a user actually feels the pain.

    python3 tools/disagg-screen.py /Volumes/Models              # rank every .gguf found
    python3 tools/disagg-screen.py a.gguf b.gguf --resident 30  # how much fits on the card, GiB (default 28)
"""
import argparse, importlib.util, math, os, re, struct, sys


def read_kv(path):
    """The GGUF key/value table alone, for the fields the router's expert reader does not return. Header only, like
    everything else here: a few MB at the front of the file, no weight mapped."""
    scalar = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
    with open(path, "rb") as f:
        def rd(n):
            b = f.read(n)
            if len(b) != n: raise ValueError("header ends early")
            return b
        u32 = lambda: struct.unpack("<I", rd(4))[0]
        u64 = lambda: struct.unpack("<Q", rd(8))[0]
        def string(): return rd(u64()).decode("utf-8", "replace")
        def value(vt):
            if vt == 8: return string()
            if vt == 9:
                et = u32(); n = u64()
                if et == 8: return [string() for _ in range(n)]
                if et == 9: return [value(9) for _ in range(n)]
                fmt = scalar[et]; return list(struct.unpack("<" + fmt * n, rd(struct.calcsize(fmt) * n)))
            fmt = scalar[vt]; return struct.unpack("<" + fmt, rd(struct.calcsize(fmt)))[0]
        if rd(4) != b"GGUF": raise ValueError("not a GGUF file")
        u32(); u64(); n_kv = u64()
        return {(k := string()): value(u32()) for _ in range(n_kv)}


def kv_bytes_per_token(kv, arch):
    """What one token of context costs in KV cache, at f16. Layers x KV heads x (key + value) head dims x 2 bytes.
    Returns None when the header does not say, which is reported rather than guessed around."""
    n_layer = kv.get(f"{arch}.block_count")
    n_kv_head = kv.get(f"{arch}.attention.head_count_kv")
    if isinstance(n_kv_head, list): n_kv_head = max(n_kv_head) if n_kv_head else None
    n_head = kv.get(f"{arch}.attention.head_count")
    if isinstance(n_head, list): n_head = max(n_head) if n_head else None
    n_embd = kv.get(f"{arch}.embedding_length")
    k_len = kv.get(f"{arch}.attention.key_length")
    v_len = kv.get(f"{arch}.attention.value_length")
    if k_len is None and n_embd and n_head: k_len = n_embd // n_head
    if v_len is None: v_len = k_len
    if not (n_layer and n_kv_head and k_len and v_len): return None
    return n_layer * n_kv_head * (k_len + v_len) * 2      # 2 bytes an element, f16 cache

GIB = float(1 << 30)
# What is left for RESIDENT EXPERTS, which is not the card's size. Measured from a real prefill server's own memory
# breakdown (logs/disagg/prefill-server-20260916-134123.log): of 32,607 MiB the compute buffer alone took 11,169 at
# ub=32768 (~8.4 GiB at the 24,576 this path uses), non-expert weight 2,249 and the KV reservation ~1,000. Screening
# against the full 30 GiB credited every candidate with ~10 GiB of residency that does not exist, which flatters the
# plateau of every model ranked so far. --resident overrides for a different ubatch or context.
CARD_VRAM_GIB = 21.0
METAL_RATE = 660.0            # Metal prefill tok/s at 24k. It does NOT hold: 673 at 24k, 405 at 26k, 304 at 49k
                              # (measured 2026-09-19). Prefill is compute-bound and Apple has no matrix hardware, so
                              # the quadratic term bites Metal far harder than the card. Every number below that uses
                              # a constant Metal rate therefore reads LOW at long context - the real advantage RISES.
CARD_RATE = 3300.0            # card marginal prefill tok/s, refit 2026-09-19 on four fresh prefills (was 5150)
LINK_RATE = 7.04              # GiB/s the expert stream crosses Thunderbolt at: 43.69 GiB in 6.21 s (was 5.46)
CARD_FIXED = 1.65             # seconds a prefill pays regardless of length, from the same fit
CARD_UBATCH = 24576           # the card server's -ub: the stream is paid once per ubatch
# The model both rates were measured on: Coder-Next 80B-A3B, whose active bytes a token work out at ~3.35 GiB. Every
# other model's rates are scaled from these by its own active bytes, because prefill is compute-bound.
BASE_ACTIVE_GIB = 3.35
HANDOFF_PER_TOKEN = 3e-5     # seconds a token to save the slot state off the card and restore it into Metal


def load_router():
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location("disagg_router", os.path.join(here, "disagg-router.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def card_seconds(tokens, streamed):
    return math.ceil(tokens / CARD_UBATCH) * (streamed / LINK_RATE) + tokens / CARD_RATE + CARD_FIXED


def plateau_at(streamed, active, tokens=None):
    """Speedup at a given prompt length, or the long-prompt plateau when tokens is None."""
    r = max(active / BASE_ACTIVE_GIB, 1e-6)
    n = CARD_UBATCH if tokens is None else tokens
    metal = n / (METAL_RATE / r)
    card = math.ceil(n / CARD_UBATCH) * (streamed / LINK_RATE) + n / (CARD_RATE / r) + n * HANDOFF_PER_TOKEN
    return metal / card


SPLIT_RE = re.compile(r"^(?P<base>.*)-(?P<no>\d{5})-of-(?P<count>\d{5})\.gguf$")


def shard_sets(files):
    """Group a GGUF split (`name-00001-of-00003.gguf`) back into the one model llama.cpp loads. Screening a shard on
    its own is not a small error: it reads the first shard's bytes as the whole model, so the expert stream - the
    fixed cost that sets the plateau - comes out several times too small and every verdict is flattering. Every model
    in the size band this path is for ships sharded, so this is the common case, not the corner. Returns
    [(representative, [shards in order]), ...]; the representative is shard 1, which carries the full KV table."""
    groups, singles = {}, []
    for f in files:
        m = SPLIT_RE.match(os.path.basename(f))
        if m: groups.setdefault(os.path.join(os.path.dirname(f), m.group("base")), []).append((int(m.group("no")), f))
        else: singles.append((f, [f]))
    out = [(paths[0][1], [p for _, p in paths]) for paths in
           (sorted(v) for v in groups.values())]
    return sorted(out + singles, key=lambda t: t[0])


def merged_layout(router, shards):
    """expert_layout over every shard of one model, added up. Only shard 1 carries the architecture and expert counts;
    the others carry their share of the expert tensors, which is what has to be summed."""
    lay = {"arch": "?", "n_expert": None, "n_expert_used": None, "blocks": [], "total": 0}
    for i, sh in enumerate(shards):
        one = router.expert_layout(sh)
        if i == 0 or lay["arch"] == "?":
            for k in ("arch", "n_expert", "n_expert_used"):
                if one.get(k) not in (None, "?"): lay[k] = one[k]
        lay["blocks"] += one.get("blocks") or []
        lay["total"] += one.get("total") or 0
    lay["blocks"].sort()
    return lay


def screen(router, path, resident_gib, shards=None):
    """Header only: the key/value table and the tensor directory, a few MB at the front of each shard."""
    shards = shards or [path]
    try:
        lay = merged_layout(router, shards)
    except Exception as e:
        return {"path": path, "error": str(e)[:60]}
    total_bytes = sum(os.path.getsize(s) for s in shards)
    total_gib = total_bytes / GIB
    expert_gib = lay["total"] / GIB if lay.get("total") else 0.0
    nblocks = len(lay.get("blocks") or [])
    # What must stream: whatever does not fit on the card. Non-expert weight stays resident by preference, so the
    # card holds (resident - non_expert) GiB of experts and the rest crosses the link every ubatch.
    streamed = max(total_gib - resident_gib, 0.0)
    fits = total_gib <= resident_gib
    name = os.path.basename(path)
    if len(shards) > 1: name = SPLIT_RE.match(name).group("base") + f" ({len(shards)} shards)"
    row = {"path": path, "name": name, "shards": shards, "total_gib": total_gib, "expert_gib": expert_gib,
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
    # Active bytes a token: everything that is not an expert, plus the fraction of experts actually routed to. This is
    # what both prefill rates scale with, and it is readable straight out of the header.
    non_expert = max(total_gib - expert_gib, 0.0)
    frac = (row["used"] / row["n_expert"]) if row["n_expert"] else 1.0
    active = non_expert + expert_gib * frac
    row["active"] = active
    r = max(active / BASE_ACTIVE_GIB, 1e-6)
    row["plateau"] = plateau_at(streamed, active)
    try:
        row["kv_per_tok"] = kv_bytes_per_token(read_kv(path), row["arch"])
    except Exception:
        row["kv_per_tok"] = None
    # Break-even prompt: below this, Metal alone is faster because the stream is not yet amortised.
    per_tok = r / METAL_RATE - r / CARD_RATE
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
                # ._name is macOS's AppleDouble sidecar, written on exFAT drives: a few KB of resource fork that
                # starts with the wrong magic and is not a model. Skipping them is not cosmetic - every external
                # drive holding GGUFs has one per file, and each becomes a "could not read" line.
                files += [os.path.join(root, n) for n in names if n.endswith(".gguf") and not n.startswith("._")]
        elif p.endswith(".gguf"):
            files.append(p)
    files = sorted(set(files))
    if not files:
        print("no .gguf files found"); return 2

    router = load_router()
    rows = [screen(router, rep, a.resident, sh) for rep, sh in shard_sets(files)]
    ok = [r for r in rows if "error" not in r]
    for r in rows:
        if "error" in r:
            print(f"  could not read {os.path.basename(r['path'])}: {r['error']}")

    resident_g = a.resident
    ok.sort(key=lambda r: (-(r["plateau"] if r["plateau"] == r["plateau"] else -1)))
    print(f"\ncard holds {a.resident:.0f} GiB; Metal prefill {METAL_RATE:.0f} tok/s, card {CARD_RATE:.0f} tok/s marginal,")
    print(f"link {LINK_RATE:.2f} GiB/s, expert stream paid once per {CARD_UBATCH} tokens.\n")
    print(f"{'model':<40}{'size':>8}{'active':>8}{'stream':>8}{'plateau':>9}{'break-even':>11}  verdict")
    for r in ok:
        pl = "-" if r["plateau"] != r["plateau"] else f"{r['plateau']:.1f}x"
        be = f"{r['breakeven']:,.0f} tok" if "breakeven" in r else "-"
        ac = f"{r['active']:.1f}G" if "active" in r else "-"
        print(f"{r['name'][:39]:<40}{r['total_gib']:>7.1f}G{ac:>8}{r['streamed']:>7.1f}G{pl:>9}{be:>11}  {r['verdict']}")
    # Does context erode the advantage? Yes, and not through the attention maths - through the KV cache, which lives
    # on the CARD during prefill and takes its room from the resident experts. Every GiB of cache is a GiB that has
    # to stream instead, once per ubatch. This prints the curve rather than asserting the shape.
    # Contexts chosen as exact multiples of the ubatch on purpose. Between multiples the ratio saw-tooths - 32,768
    # tokens pays the expert stream TWICE for 1.33 ubatches of work - and that artefact would otherwise be mistaken
    # for the effect being measured. At multiples the only thing changing is how much cache is displacing experts.
    ctx_list = [CARD_UBATCH, CARD_UBATCH * 2, CARD_UBATCH * 4, CARD_UBATCH * 8]
    cands = [r for r in ok if not r["fits"] and r.get("kv_per_tok") and r["n_expert"]]
    if cands:
        print("\ncontext sweep - the KV cache lives on the card and displaces resident experts:")
        print(f"{'model':<30}{'KV/tok':>8}" + "".join(f"{c//1024:>5}k{'cache':>7}{'ratio':>7}" for c in ctx_list))
        for r in cands:
            cells = ""
            for c in ctx_list:
                kv_gib = r["kv_per_tok"] * c / GIB
                room = max(r["total_gib"] - r["expert_gib"], 0.0)          # non-expert weight, always resident
                expert_room = resident_g - room - kv_gib
                if expert_room < 0:
                    cells += f"{'':>6}{kv_gib:>6.0f}G{'oom':>7}"
                    continue
                streamed = min(max(r["expert_gib"] - expert_room, 0.0), r["expert_gib"])
                cells += f"{'':>6}{kv_gib:>6.1f}G{plateau_at(streamed, r['active'], c):>6.1f}x"
            print(f"{r['name'][:29]:<30}{r['kv_per_tok']/1024:>7.0f}K{cells}")
        print("'oom' = the cache alone no longer leaves room on the card at that context. Falling ratios are the premise")
        print("confirmed: context eats residency, and residency is the only thing keeping experts off the link.")

    print("\nplateau = the speedup a long prompt converges to; break-even = the prompt length below which Metal alone wins.")
    print("active = bytes a token actually touches, which is the lever: more active parameters amortise the fixed expert")
    print("stream against more compute, so the speedup rises toward the ~7.8x card:Metal compute ratio. Total size alone")
    print("is misleading. A model that fits the card entirely is not a candidate - run it card-only instead.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
