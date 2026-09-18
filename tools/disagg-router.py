#!/usr/bin/env python3
"""disagg-router.py — the disaggregated feature as ONE endpoint (V1, 2026-09-16).

Fronts two llama-servers that serve the SAME model file with the SAME --slot-save-path directory and -fa on:
  Metal  (the Mac's GPU, all layers resident, decodes)            default http://127.0.0.1:8091
  card   (the 5090 through the shim, experts on the host, prefills)  default http://127.0.0.1:8092, optional
and speaks the llama-server API a client already uses (/completion, /v1/completions, /v1/chat/completions; everything
else is passed straight to Metal). Per request:
  1. work out the token ids Metal will see: a token-array prompt as is; a string through Metal's /tokenize with
     add_special and parse_special on, which is how the completion endpoints tokenize; a chat request through Metal's
     /apply-template first, which renders the same prompt string the chat endpoint renders;
  2. the cold part is what Metal's slot does not already hold (the router is the slot's only client, so it tracks the
     prompt it last sent there). The card gets it when the operator's cap says so (--threshold: cold tokens at or above
     it ALWAYS go to the card, which keeps a Metal server off long prefills) or when the cost model predicts the card is
     faster: Metal costs cold/metal_rate; the card costs a stream of every expert across the link (~5.7 s here) ONCE PER
     UBATCH - ggml copies the host-resident experts for every ubatch it evaluates - plus cold/card_rate plus the handoff. Fitted from two measured points, break-even
     ~5,300 cold tokens for Qwen3-Coder-Next on this Mac; --selftest prints the curve. The constants are only defaults:
     the router refits them from what it observes (Metal's prefill rate from its timings, the card's fixed cost and
     rate from a rolling fit of its prefills, the handoff from save+restore), --no-learn freezes them. No card: Metal;
  3. otherwise the card prefills all but the LAST token (the recurrent state cannot be rewound, so the decode side must
     evaluate one token itself), saves the slot's state under a name derived from the tokens, Metal restores it, and the
     original request is forwarded to Metal with cache_prompt on: Metal finds N-1 cached tokens, evaluates one, decodes.
     Any failure on the card path falls back to Metal alone, loudly.
Streams are relayed byte for byte. One request at a time through the router: both servers have one slot.

The Metal response's timings.cache_n is the self-check: after a card prefill of N-1 tokens it must read N-1. It is
logged on every non-streamed request; a smaller number means the two servers tokenized differently, and the card's
work was wasted rather than wrong.

  python3 tools/disagg-router.py [--listen 8095] [--metal URL] [--card URL] [--state-dir DIR] [--slot 0]
      [--threshold N: the cap] [--no-auto] [--metal-rate T/S] [--card-rate T/S] [--card-fixed S] [--handoff-per-token S] [--selftest]

The card server holds the card for as long as it is up: it is a scheduled tenant under tools/nv_shim_step.sh and the
lock, exactly like tools/serve.sh, and starting it is a card step. The Metal server maps the whole model: under the
host rule in the README that is a card-affecting job while anyone holds the card.
"""
import argparse, hashlib, http.client, json, os, sys, threading, time, urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ARGS = None
LOCK = threading.Lock()            # one request at a time: one slot on each server
STATE = {"metal_tokens": [], "metal_text": "", "card_tokens": [], "card_obs": [], "metal_obs": [], "card_synced_len": 0}
# each side's own slot contents as last sent; observed (tokens, seconds) per side; card_synced_len is what a cache
# hit should be after the last sync_metal_to_card(), to detect a mismatched-tail miss (see its docstring).
# metal_text is the exact text (prompt + Metal's own real decoded content) that produced metal_tokens - used by
# prompt_tokens() to detect a literal continuation and avoid retokenizing the whole growing string every time


def log(msg):
    sys.stderr.write(f"[router] {time.strftime('%H:%M:%S')} {msg}\n"); sys.stderr.flush()


def split_url(url):
    u = urllib.parse.urlsplit(url); return u.hostname, u.port or 80


def request(base, method, path, body=None, timeout=3600):
    """One upstream call; returns (status, headers, bytes). body is a dict (JSON) or None."""
    host, port = split_url(base)
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    data = json.dumps(body).encode() if body is not None else None
    conn.request(method, path, data, {"Content-Type": "application/json"} if data else {})
    r = conn.getresponse(); out = r.read(); hdrs = r.getheaders(); conn.close()
    return r.status, hdrs, out


def post_json(base, path, body, timeout=3600):
    st, _, out = request(base, "POST", path, body, timeout)
    if st != 200: raise RuntimeError(f"{base}{path} -> HTTP {st}: {out[:200]!r}")
    return json.loads(out)


def healthy(base, timeout=2):
    try: return request(base, "GET", "/health", None, timeout)[0] == 200
    except OSError: return False


def prompt_tokens(path, body):
    """The (token ids, rendered text) Metal will use for this request; (None, None) when the shape is one the
    router does not handle, or text is None when the client sent pre-tokenized ids directly (nothing to compare
    against Metal's tracked text in that case - the caller already manages its own tokens).

    When the rendered text is confirmed - by a literal Python string prefix check, not a retokenize - to extend
    STATE["metal_text"] exactly, reuses STATE["metal_tokens"] for that whole prefix and tokenizes ONLY the new
    suffix (add_special=False: this is a continuation, never a fresh BOS). This avoids retokenizing the whole
    growing string every request, which a BPE tokenizer does not always do losslessly at a boundary - found
    2026-09-18: a real 27-token generation, once detokenized to text and reprocessed as part of a longer string,
    retokenized as 26 tokens. A generic BPE property (multiple token sequences can represent the same text), not
    a bug here, but it meant sync_safe in do_POST could rarely confirm true even for a genuinely live, real
    conversation. This makes sync_safe true BY CONSTRUCTION for any well-behaved client (one that resends exactly
    what it was shown, which is how virtually every real chat client works) instead of by chance, and it's
    cheaper too - only the new suffix gets tokenized, not the whole history every time.

    KNOWN REMAINING GAP, found 2026-09-18 chasing this fix's first confirmed-safe sync attempt failing with
    cache_n=0 on real hardware: splitting a tokenize() call at this prefix/suffix boundary is not always
    equivalent to one continuous tokenize() call over the same text, because a BPE merge can span the boundary.
    Caught directly: Metal's own continuous tokenization encoded ",\n\n" (a comma the model generated, followed
    by the next turn's blank lines) as ONE token (3554); tokenizing the same text as a fresh suffix, split right
    after the comma, produced "," and "\n\n" as TWO tokens (11, 271) - same text, different token ids. So
    STATE["metal_tokens"] + suffix_ids can, at a merge boundary, diverge from what Metal's real cache actually
    holds even though the string-prefix check above is textually correct. Confirmed SAFE, not just theorized: the
    card's own cache-prefix match simply misses (cache_n=0) and disagg-router falls back to a full reprocess (see
    the "sync mismatch" log line in card_prefill), so this costs a wasted sync attempt's latency, never a wrong
    answer. The real fix would re-tokenize a small overlap (the last few known-good characters plus the new
    suffix) and only trust the reuse when that overlap's tokens match STATE["metal_tokens"] exactly - not done
    here, since the failure mode is a latency cost on an already-optimization-only path, not a correctness bug."""
    if path in ("/completion", "/completions", "/v1/completions"):
        p = body.get("prompt")
        if isinstance(p, list) and p and all(isinstance(t, int) for t in p): return p, None
        if not isinstance(p, str): return None, None
        text = p
    elif path == "/v1/chat/completions":
        if not isinstance(body.get("messages"), list): return None, None
        tpl = {k: v for k, v in body.items() if k in ("messages", "tools", "tool_choice", "chat_template_kwargs", "reasoning_format", "add_generation_prompt")}
        text = post_json(ARGS.metal, "/apply-template", tpl).get("prompt")
        if not isinstance(text, str): return None, None
    else:
        return None, None
    if STATE["metal_text"] and text.startswith(STATE["metal_text"]):
        suffix = text[len(STATE["metal_text"]):]
        suffix_ids = post_json(ARGS.metal, "/tokenize", {"content": suffix, "add_special": False, "parse_special": True})["tokens"] if suffix else []
        return STATE["metal_tokens"] + suffix_ids, text
    ids = post_json(ARGS.metal, "/tokenize", {"content": text, "add_special": True, "parse_special": True})["tokens"]
    return ids, text


def common_prefix(a, b):
    n = min(len(a), len(b)); i = 0
    while i < n and a[i] == b[i]: i += 1
    return i


def sync_metal_to_card():
    """Push Metal's current slot state onto the card's slot, when the card has fallen behind: llama-server's slot
    save/restore format is symmetric (same code compiled into both binaries, confirmed by reading
    llama-kv-cache.cpp/llama-memory-hybrid.cpp 2026-09-18 - card->metal already relies on this working one way),
    so after this the card recognizes the same shared prefix Metal does and only prefills the true delta, instead
    of redoing the whole conversation (the 09-18 finding: a stale card redid 21,343 tokens for a 6,573-token ask).

    Real risk, confirmed 2026-09-18, not hypothetical: Metal's actual cache is usually LONGER than the prompt
    tokens tracked in STATE["metal_tokens"], because decoding a reply extends it too, and the identity of those
    extra tokens isn't known here. In ordinary use this is harmless - a real client resends the model's own
    reply as history, so the untracked tail matches whatever comes next anyway. But if it does NOT match (proven
    directly: restore a state with an untracked decoded tail, then query with real content that diverges from
    that tail), llama-server's cache-prefix check does not gracefully trim to the last good match - it reports
    cache_n: 0 and reprocesses the ENTIRE restored prefix from scratch, not just the tokens past the mismatch.
    This is a real llama-server robustness gap (confirmed on Metal, not a card-specific thing - see
    docs/disagg-inference-state.md), not this project's bug to fix, but card_prefill() below checks for it so a
    production run reports a much-slower-than-predicted prefill honestly instead of silently eating the cost."""
    name = "router-sync-" + hashlib.sha1(json.dumps(STATE["metal_tokens"]).encode()).hexdigest()[:16] + ".bin"
    t0 = time.perf_counter()
    s = post_json(ARGS.metal, f"/slots/{ARGS.slot}?action=save", {"filename": name}); t1 = time.perf_counter()
    r = post_json(ARGS.card, f"/slots/{ARGS.slot}?action=restore", {"filename": name}); t2 = time.perf_counter()
    if ARGS.state_dir:
        try: os.unlink(os.path.join(ARGS.state_dir, name))
        except OSError: pass
    log(f"sync: metal->card {s.get('n_saved')} tok {s.get('n_written', 0)/1e6:.0f} MB, save {t1-t0:.1f} s, restore {t2-t1:.1f} s")
    if r.get("n_restored") != s.get("n_saved"):
        # a length check against the round trip itself, not against STATE["metal_tokens"]: the save/restore round
        # trip not preserving its own count is the actual fault condition (an untracked decoded tail is not).
        raise RuntimeError(f"card restore returned {r.get('n_restored')} tokens, save reported {s.get('n_saved')}")
    STATE["card_tokens"] = list(STATE["metal_tokens"])
    STATE["card_synced_len"] = len(STATE["metal_tokens"])   # what a hit SHOULD look like; card_prefill checks this


def card_prefill(ids, sync_needed):
    """Prefill ids[:-1] on the card, hand the state to Metal. Raises on any failure; the caller falls back.
    sync_needed: the card's own cache is behind Metal's for this request - pull Metal's state onto the card
    first, so the card's cache_prompt match below sees the same prefix Metal does."""
    if sync_needed: sync_metal_to_card()
    head = ids[:-1]
    name = "router-" + hashlib.sha1(json.dumps(head).encode()).hexdigest()[:16] + ".bin"
    t0 = time.perf_counter()
    r = post_json(ARGS.card, "/completion", {"prompt": head, "n_predict": 1, "temperature": 0, "cache_prompt": True, "id_slot": ARGS.slot, "return_tokens": False})
    t = r.get("timings", {}); t1 = time.perf_counter()
    if sync_needed and t.get("cache_n", 0) < STATE.get("card_synced_len", 0):
        # the sync's restored tail didn't match this request's real content at that position - llama-server's
        # cache check doesn't trim to the last good match, it misses entirely (confirmed 2026-09-18, not this
        # project's bug: see the sync_metal_to_card() docstring). Costs a full reprocess instead of the predicted
        # delta; surfaced here so that shows up as a clear log line instead of a silent, unexplained slow prefill.
        log(f"sync mismatch: expected a >= {STATE.get('card_synced_len', 0)}-token cache hit after sync, got "
            f"cache_n={t.get('cache_n', 0)} - the restored tail didn't match this request, full reprocess paid")
    s = post_json(ARGS.card, f"/slots/{ARGS.slot}?action=save", {"filename": name}); t2 = time.perf_counter()
    m = post_json(ARGS.metal, f"/slots/{ARGS.slot}?action=restore", {"filename": name}); t3 = time.perf_counter()
    if ARGS.state_dir:
        try: os.unlink(os.path.join(ARGS.state_dir, name))
        except OSError: pass
    log(f"card: prefill {t.get('prompt_n')} tok in {t.get('prompt_ms', 0)/1000:.1f} s ({t.get('prompt_per_second', 0):.0f} tok/s, wall {t1-t0:.1f}); "
        f"save {s.get('n_saved')} tok {s.get('n_written', 0)/1e6:.0f} MB {t2-t1:.1f} s; restore {m.get('n_restored')} tok {t3-t2:.1f} s")
    if m.get("n_restored") != len(head): raise RuntimeError(f"restore returned {m.get('n_restored')} tokens for {len(head)}")
    STATE["metal_tokens"] = list(head)
    STATE["card_tokens"] = list(head)
    if t.get("prompt_n") and t.get("prompt_ms"): observe_card(int(t["prompt_n"]), t["prompt_ms"] / 1000.0, (t2 - t1) + (t3 - t2))


def ubatches(cold):
    """How many ubatches the card needs for `cold` tokens. The expert stream is paid ONCE PER UBATCH, not once per
    prompt: ggml's scheduler copies each layer's host-resident experts to the card for every ubatch it evaluates."""
    return max(1, -(-cold // ARGS.card_ubatch))


def fit_line(obs):
    """Least squares over (tokens, seconds): seconds = fixed*ubatches(tokens) + tokens/rate, no intercept beyond the
    per-ubatch fixed cost. Returns (fixed_seconds, tokens_per_second), or None when the observations cannot support a
    fit: fewer than THREE, spanning under 1000 tokens, a singular design, a non-positive fixed cost or rate, or a fit
    that misses ANY observation it was fitted on by more than 20%.

    Three, not two, and it was measured: two single-ubatch points fit themselves exactly by construction, and on
    2026-09-17 the pair (6,974 tok/11.4 s, 23,691 tok/12.6 s) returned 11.0 s + 14,471 tok/s - nearly three times the
    marginal rate the same window supports - and then predicted 16.2 s for the very prefill that had just taken 12.6 s.
    A model that misses the data it was fitted on is rejected here rather than shipped into a routing decision."""
    if len(obs) < 3 or max(n for n, _ in obs) - min(n for n, _ in obs) < 1000: return None
    suu = sun = snn = sut = snt = 0.0
    for n, t in obs:
        u = ubatches(n)
        suu += u * u; sun += u * n; snn += n * n; sut += u * t; snt += n * t
    den = suu * snn - sun * sun
    if den <= 0: return None
    a = (sut * snn - snt * sun) / den      # seconds per ubatch (the expert stream)
    b = (suu * snt - sun * sut) / den      # seconds per token (1/rate)
    if a <= 0 or b <= 0: return None
    if any(abs((ubatches(n) * a + n * b) - t) > 0.2 * t for n, t in obs): return None
    return a, 1.0 / b


def observe_card(tokens, seconds, handoff_seconds):
    """A card prefill happened: refit the fixed cost and rate from the last 8, and the handoff per token."""
    if not ARGS.learn or tokens < 256: return
    if not STATE["card_obs"]:
        log(f"first card prefill since start: {tokens} tok in {seconds:.1f} s - a first prefill may pay a cost the later "
            "ones do not (unresolved: 2026-09-17 window 2's 7K prefills ran 2.08 s and 1.24 s over the model fitted on "
            "its 24K pair, and those residuals scale with expert bytes streamed)")
    STATE["card_obs"] = (STATE["card_obs"] + [(tokens, seconds)])[-8:]
    fit = fit_line(STATE["card_obs"])
    hp = handoff_seconds / tokens
    changed = abs(hp - ARGS.handoff_per_token) > 0.1 * ARGS.handoff_per_token
    ARGS.handoff_per_token = 0.5 * ARGS.handoff_per_token + 0.5 * hp
    if fit:
        fixed, rate = fit
        changed |= abs(fixed - ARGS.card_fixed) > 0.05 * ARGS.card_fixed or abs(rate - ARGS.card_rate) > 0.05 * ARGS.card_rate
        ARGS.card_fixed, ARGS.card_rate = fixed, rate
    if changed: log(f"calibration: card fixed {ARGS.card_fixed:.1f} s + {ARGS.card_rate:.0f} tok/s, handoff {ARGS.handoff_per_token*1e6:.0f} us/token ({len(STATE['card_obs'])} observations)")


def observe_metal(tokens, seconds):
    """Metal prefilled `tokens` itself: track its rate (only prefills long enough to be a rate, not an overhead)."""
    if not ARGS.learn or tokens < 256 or seconds <= 0: return
    rate = tokens / seconds; old = ARGS.metal_rate
    ARGS.metal_rate = 0.5 * ARGS.metal_rate + 0.5 * rate
    if abs(ARGS.metal_rate - old) > 0.05 * old: log(f"calibration: metal {ARGS.metal_rate:.0f} tok/s (observed {rate:.0f} over {tokens} tokens)")


def predict(cold_metal, cold_card, sync_tokens):
    """Seconds of prompt work: Metal on its own cold count; the card on cold_card (which the CALLER has already
    decided is either cold_metal, when a sync is provably safe, or the card's own raw mismatch against its own
    history, when it isn't - see decide()'s docstring), plus a one-time sync tax (0 when not syncing). Before
    2026-09-18 the card's own cold count was blindly used, and a stale card had to redo everything since ITS own
    last use - found the hard way (a 6,573-cold-token turn, after 6 metal-only turns, actually cost a
    21,343-token prefill: the whole conversation). Fixed in two stages: first by syncing Metal's slot state onto
    the card (`sync_metal_to_card()`; the format is symmetric, confirmed by reading
    llama-kv-cache.cpp/llama-memory-hybrid.cpp), then - because a sync can itself be unsafe, found the same day: a
    restored state whose real content (prompt + what Metal actually decoded) diverges from the new request past
    the tracked prefix causes llama-server's cache check to miss ENTIRELY, not just past the divergence point -
    by only ever calling for a sync when Metal's full tracked state (now accurate, see do_POST's return_tokens
    capture) is confirmed to be a genuine prefix of the new request. The card's expert stream is charged PER
    UBATCH, so a prompt past the ubatch pays it again - the single-ubatch form under-predicted a 62,000-token
    prompt by 11 s (found while preparing that run, 2026-09-17)."""
    metal = cold_metal / ARGS.metal_rate
    card = ubatches(cold_card) * ARGS.card_fixed + cold_card / ARGS.card_rate + cold_card * ARGS.handoff_per_token
    card += sync_tokens * ARGS.handoff_per_token
    return metal, card


def decide(cold_metal, cold_card, sync_tokens, n_total, card_up):
    """Where the cold part of a prompt goes. cold_metal is what's new since Metal's own cache (what Metal would
    have to prefill). cold_card is what the CARD would have to prefill if chosen: cold_metal itself when the
    caller has confirmed a sync would be safe (Metal's whole tracked state, prompt plus its real decoded output,
    is a genuine prefix of this request), otherwise the card's own raw mismatch against ITS OWN history (a sync
    is never attempted on unconfirmed content - see the predict() docstring for why). sync_tokens is the one-time
    cost of catching the card's slot up first (0 when already in sync or when sync isn't safe to attempt). The
    operator's cap wins on cold_metal (cold_metal >= --threshold always goes to the card, so a Metal server can
    be kept off long prefills); otherwise the cost model picks the faster side using each side's own real cost.
    card_up is a callable, consulted only when the card would be chosen: a health probe is a round trip."""
    if n_total <= 1 or cold_metal <= 0: return "metal", "nothing cold" if cold_metal <= 0 else "one token"
    m, c = predict(cold_metal, cold_card, sync_tokens)
    if ARGS.threshold and cold_metal >= ARGS.threshold:
        want, why = "card", f"cold {cold_metal} >= cap {ARGS.threshold}; predicted card {c:.1f} s (card {cold_card} tok, sync {sync_tokens} tok), metal {m:.1f} s"
    elif ARGS.auto and c < m:
        want, why = "card", f"predicted card {c:.1f} s (card {cold_card} tok, sync {sync_tokens} tok) < metal {m:.1f} s"
    else:
        return "metal", f"predicted metal {m:.1f} s <= card {c:.1f} s (card {cold_card} tok, sync {sync_tokens} tok)" if ARGS.auto else "auto off, under the cap"
    if not card_up(): return "metal", why + "; card server not healthy"
    return want, why


def selftest():
    """The cost model at a few sizes, the break-even, and the two properties the rule must have."""
    lo, hi = 1, 200000
    while hi - lo > 1:
        mid = (lo + hi) // 2; m, c = predict(mid, mid, 0)
        if c < m: hi = mid
        else: lo = mid
    print(f"cost model: metal {ARGS.metal_rate:.0f} tok/s; card {ARGS.card_fixed:.1f} s per ubatch of {ARGS.card_ubatch} + {ARGS.card_rate:.0f} tok/s + handoff {ARGS.handoff_per_token*1e6:.0f} us/token; break-even {hi} cold tokens (card already in sync)")
    ok = True
    for cold in (512, 2000, 5000, 8000, 24000, 64000):
        m, c = predict(cold, cold, 0); want, why = decide(cold, cold, 0, cold + 1, lambda: True)
        print(f"   {cold:>6} cold: metal {m:6.1f} s  card {c:6.1f} s ({ubatches(cold)} ubatch{'es' if ubatches(cold) > 1 else ''})  -> {want}")
        ok &= (want == "card") == (c < m or (ARGS.threshold and cold >= ARGS.threshold))
    ok &= decide(24000, 24000, 0, 24001, lambda: False)[0] == "metal" # a card that is down never wins
    ok &= decide(0, 0, 0, 5000, lambda: True)[0] == "metal"           # nothing cold: nothing to route
    # a sync that is NOT confirmed safe must fall back to the card's own raw mismatch, never to cold_metal - this
    # is the exact 2026-09-18 catastrophic-miss shape (6,573 cold to Metal, but the card's own real history is
    # 21,344 behind): decide() must be told cold_card=21344, sync=0 in this case, and correctly prefer metal
    ok &= decide(6573, 21344, 0, 21345, lambda: True)[0] == "metal"
    ok &= decide(6573, 6573, 0, 6574, lambda: True)[0] == "card"      # same request, card confirmed in sync: card wins
    # a CONFIRMED-safe sync still isn't free: it costs more than being already in sync, but far less than the
    # card's raw-mismatch fallback for the same underlying gap (both properties must hold)
    m_fresh, c_fresh = predict(6573, 6573, 0)                        # card already in sync: cheapest case
    m_sync, c_sync = predict(6573, 6573, 14771)                      # confirmed-safe sync, 14,771 tokens to catch up
    c_reprocess = ubatches(21344) * ARGS.card_fixed + 21344 / ARGS.card_rate + 21344 * ARGS.handoff_per_token
    ok &= c_sync > c_fresh                                           # syncing costs something over being in sync...
    ok &= c_sync < c_reprocess                                       # ...but far less than the raw-mismatch fallback
    known = [(6974, 8.0 + 6974 / 5150), (23691, 8.0 + 23691 / 5150), (64577, 3 * 8.0 + 64577 / 5150)]
    fit = fit_line(known)                                             # the estimator recovers a line it was given
    ok &= fit is not None and abs(fit[0] - 8.0) < 0.05 and abs(fit[1] - 5150) < 25
    obs24 = 12.588                                                    # the control prefill the constants were fitted on
    ok &= abs((1 * ARGS.card_fixed + 23691 / ARGS.card_rate) - obs24) < 0.3
    ok &= ubatches(24576) == 1 and ubatches(24577) == 2 and ubatches(62031) == 3      # the stream is per ubatch
    m3, c3 = predict(62031, 62031, 0)                                 # and a 3-ubatch prompt pays it three times
    ok &= abs(c3 - (3 * ARGS.card_fixed + 62031 / ARGS.card_rate + 62031 * ARGS.handoff_per_token)) < 0.01
    ok &= fit_line([(6974, 8.5)]) is None and fit_line([(1000, 5.0), (1500, 5.2), (1800, 5.3)]) is None  # too few, too narrow
    ok &= fit_line([(1000, 9.0), (5000, 7.0), (9000, 5.0)]) is None   # a negative slope is not a rate
    ok &= fit_line([(6974, 11.433), (23691, 12.588)]) is None         # the real pair that returned 14,471 tok/s
    ok &= fit_line(known[:2] + [(64577, 36.5 * 2)]) is None           # one point no line through the others explains
    print(f"estimator on a known 8.00 s + 5150 tok/s line: fixed {fit[0]:.2f} s, {fit[1]:.0f} tok/s" if fit else "fit: FAILED")
    print(f"shipped constants against the 23,691-token control prefill they came from: "
          f"{1 * ARGS.card_fixed + 23691 / ARGS.card_rate:.2f} s predicted, {obs24:.2f} s observed")
    print("selftest:", "OK" if ok else "FAIL"); return 0 if ok else 1


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a): pass   # our own log() instead

    def _relay(self, method, path, body_bytes):
        """Forward to Metal and stream the reply back byte for byte (chunked), so SSE streams behave."""
        host, port = split_url(ARGS.metal)
        conn = http.client.HTTPConnection(host, port, timeout=3600)
        hdrs = {"Content-Type": self.headers.get("Content-Type", "application/json")}
        conn.request(method, path, body_bytes, hdrs if body_bytes else {})
        r = conn.getresponse()
        self.send_response(r.status)
        for k, v in r.getheaders():
            if k.lower() in ("content-type", "cache-control", "access-control-allow-origin"): self.send_header(k, v)
        self.send_header("Transfer-Encoding", "chunked"); self.end_headers()
        tail = b""
        while True:
            chunk = r.read(65536)
            if not chunk: break
            tail = (tail + chunk)[-65536:]   # was 4096: too small to reliably hold return_tokens' ids array
            # for a real decode length, which silently defeated the self-check for longer replies
            self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
        self.wfile.write(b"0\r\n\r\n"); self.wfile.flush(); conn.close()
        return r.status, tail

    def do_GET(self):
        with LOCK: self._relay("GET", self.path, None)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0); raw = self.rfile.read(n) if n else b""
        path = urllib.parse.urlsplit(self.path).path
        if path not in ("/completion", "/completions", "/v1/completions", "/v1/chat/completions"):
            with LOCK: self._relay("POST", self.path, raw); return
        try: body = json.loads(raw) if raw else {}
        except ValueError: body = None
        with LOCK:
            route = "metal"; ids = None; text = None
            if isinstance(body, dict):
                try: ids, text = prompt_tokens(path, body)
                except Exception as e: log(f"tokenize failed ({e}); metal-only")
            if ids:
                cached = common_prefix(STATE["metal_tokens"], ids); cold = len(ids) - cached
                cached_card = common_prefix(STATE["card_tokens"], ids)
                # a sync is only ever offered when Metal's WHOLE tracked state (prompt + what it actually
                # decoded, captured below via return_tokens) is confirmed to be a genuine prefix of this
                # request - anything less risks the 2026-09-18 catastrophic miss (a restored tail that
                # disagrees with the new request makes llama-server's cache check fail ENTIRELY, not just past
                # the disagreement). When it's not confirmed, the card - if chosen at all - is left to its own
                # raw history, exactly like before reverse sync existed: slower, but provably never wrong.
                sync_safe = cached == len(STATE["metal_tokens"])
                if sync_safe:
                    sync_tokens = max(0, cached - cached_card); cold_card = cold
                else:
                    sync_tokens = 0; cold_card = len(ids) - cached_card
                want, why = decide(cold, cold_card, sync_tokens, len(ids), lambda: healthy(ARGS.card))
                if want == "card":
                    try: card_prefill(ids, sync_tokens > 0); route = "card+metal"
                    except Exception as e: log(f"card path failed ({e}); metal-only")
                stale = f", card {cold_card} tok behind{'' if sync_safe else ' (sync unconfirmed)'}" if cold_card != cold else ""
                log(f"{path}: {len(ids)} tokens, {cached} cached at metal, {cold} cold{stale} -> {route} ({why})")
                body["cache_prompt"] = True; body["id_slot"] = ARGS.slot; body["return_tokens"] = True
                raw = json.dumps(body).encode()
            t0 = time.perf_counter(); status, tail = self._relay("POST", self.path, raw)
            metal_tokens_next = list(ids) if ids else None
            metal_text_next = text
            # the self-check, when the reply was not a stream: Metal's own prompt_n / cache_n for this request,
            # and - the 2026-09-18 fix - the ACTUAL decoded token ids AND text, so next turn's sync_safe check
            # (and prompt_tokens()'s literal-continuation check) are verified facts about Metal's real content,
            # not an assumption that a client resent it faithfully
            try:
                d = json.loads(tail) if tail.strip().startswith(b"{") else None
                t = (d or {}).get("timings") or {}
                if t:
                    log(f"metal: prompt_n {t.get('prompt_n')} cache_n {t.get('cache_n')} prompt {t.get('prompt_ms', 0)/1000:.1f} s, "
                        f"decode {t.get('predicted_n')} tok at {t.get('predicted_per_second', 0):.1f} tok/s (wall {time.perf_counter()-t0:.1f} s)")
                    if route == "metal" and t.get("prompt_n") and t.get("prompt_ms"): observe_metal(int(t["prompt_n"]), t["prompt_ms"] / 1000.0)
                if ids and isinstance((d or {}).get("tokens"), list) and t.get("predicted_n") == len(d["tokens"]):
                    metal_tokens_next = list(ids) + list(d["tokens"])
                if text is not None and isinstance((d or {}).get("content"), str):
                    metal_text_next = text + d["content"]
            except ValueError: pass
            if ids and status == 200:
                STATE["metal_tokens"] = metal_tokens_next
                STATE["metal_text"] = metal_text_next or ""


def main():
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--listen", type=int, default=8095); ap.add_argument("--metal", default="http://127.0.0.1:8091")
    ap.add_argument("--card", default="http://127.0.0.1:8092")
    ap.add_argument("--threshold", type=int, default=0, help="operator's cap: cold tokens at or above which the card ALWAYS prefills (0 = off); keeps a Metal server off long prefills")
    ap.add_argument("--no-auto", dest="auto", action="store_false", help="disable the cost model (then only --threshold routes to the card)")
    ap.add_argument("--metal-rate", type=float, default=660.0, help="Metal prefill tok/s for this model (measured 664 on the M4 Max, Qwen3-Coder-Next)")
    ap.add_argument("--card-rate", type=float, default=5150.0, help="card marginal prefill tok/s (driver ad308bd, NCPUMOE=48, 2026-09-17 window 2; the pre-ad308bd fit was 2460 and is superseded)")
    ap.add_argument("--card-fixed", type=float, default=8.0, help="card seconds per UBATCH: 43.7 GiB of experts across the link at 5.46 GiB/s (same window, the two 24K prefills under the byte-fraction model). The first prefill after a load looks MORE expensive than this; window 3 measures it")
    ap.add_argument("--card-ubatch", type=int, default=24576, help="the card server's -ub: the expert stream is paid once per ubatch, so this shapes the prediction for prompts past it")
    ap.add_argument("--handoff-per-token", type=float, default=3e-5, help="seconds per token to save and restore the state (662 MB / 23,691 tok in 0.8 s)")
    ap.add_argument("--no-learn", dest="learn", action="store_false", help="freeze the cost model: do not refit it from observed prefills")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--state-dir", default=None, help="the --slot-save-path both servers share, to delete handed-over files"); ap.add_argument("--slot", type=int, default=0)
    ARGS = ap.parse_args()
    if ARGS.selftest: sys.exit(selftest())
    if not healthy(ARGS.metal, 5): log(f"Metal server at {ARGS.metal} is not healthy; refusing to start"); sys.exit(1)
    log(f"metal {ARGS.metal} up; card {ARGS.card} {'up' if healthy(ARGS.card) else 'down (metal-only until it appears)'}; cap {ARGS.threshold or 'off'}; cost model {'on' if ARGS.auto else 'off'} (metal {ARGS.metal_rate:.0f} tok/s, card {ARGS.card_fixed:.1f} s + {ARGS.card_rate:.0f} tok/s); listening on 127.0.0.1:{ARGS.listen}")
    ThreadingHTTPServer(("127.0.0.1", ARGS.listen), Handler).serve_forever()


if __name__ == "__main__":
    main()
