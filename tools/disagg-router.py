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
STATE = {"metal_tokens": [], "card_obs": [], "metal_obs": []}   # Metal's slot contents as sent; observed (tokens, seconds) per side


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
    """The token ids Metal will use for this request, or None when the shape is one the router does not handle."""
    if path in ("/completion", "/completions", "/v1/completions"):
        p = body.get("prompt")
        if isinstance(p, list) and p and all(isinstance(t, int) for t in p): return p
        if not isinstance(p, str): return None
        text = p
    elif path == "/v1/chat/completions":
        if not isinstance(body.get("messages"), list): return None
        tpl = {k: v for k, v in body.items() if k in ("messages", "tools", "tool_choice", "chat_template_kwargs", "reasoning_format", "add_generation_prompt")}
        text = post_json(ARGS.metal, "/apply-template", tpl).get("prompt")
        if not isinstance(text, str): return None
    else:
        return None
    return post_json(ARGS.metal, "/tokenize", {"content": text, "add_special": True, "parse_special": True})["tokens"]


def common_prefix(a, b):
    n = min(len(a), len(b)); i = 0
    while i < n and a[i] == b[i]: i += 1
    return i


def card_prefill(ids):
    """Prefill ids[:-1] on the card, hand the state to Metal. Raises on any failure; the caller falls back."""
    head = ids[:-1]
    name = "router-" + hashlib.sha1(json.dumps(head).encode()).hexdigest()[:16] + ".bin"
    t0 = time.perf_counter()
    r = post_json(ARGS.card, "/completion", {"prompt": head, "n_predict": 1, "temperature": 0, "cache_prompt": True, "id_slot": ARGS.slot, "return_tokens": False})
    t = r.get("timings", {}); t1 = time.perf_counter()
    s = post_json(ARGS.card, f"/slots/{ARGS.slot}?action=save", {"filename": name}); t2 = time.perf_counter()
    m = post_json(ARGS.metal, f"/slots/{ARGS.slot}?action=restore", {"filename": name}); t3 = time.perf_counter()
    if ARGS.state_dir:
        try: os.unlink(os.path.join(ARGS.state_dir, name))
        except OSError: pass
    log(f"card: prefill {t.get('prompt_n')} tok in {t.get('prompt_ms', 0)/1000:.1f} s ({t.get('prompt_per_second', 0):.0f} tok/s, wall {t1-t0:.1f}); "
        f"save {s.get('n_saved')} tok {s.get('n_written', 0)/1e6:.0f} MB {t2-t1:.1f} s; restore {m.get('n_restored')} tok {t3-t2:.1f} s")
    if m.get("n_restored") != len(head): raise RuntimeError(f"restore returned {m.get('n_restored')} tokens for {len(head)}")
    STATE["metal_tokens"] = list(head)
    if t.get("prompt_n") and t.get("prompt_ms"): observe_card(int(t["prompt_n"]), t["prompt_ms"] / 1000.0, (t2 - t1) + (t3 - t2))


def ubatches(cold):
    """How many ubatches the card needs for `cold` tokens. The expert stream is paid ONCE PER UBATCH, not once per
    prompt: ggml's scheduler copies each layer's host-resident experts to the card for every ubatch it evaluates."""
    return max(1, -(-cold // ARGS.card_ubatch))


def fit_line(obs):
    """Least squares over (tokens, seconds): seconds = fixed*ubatches(tokens) + tokens/rate, no intercept beyond the
    per-ubatch fixed cost. Returns (fixed_seconds, tokens_per_second), or None when the observations cannot support a
    fit: fewer than two, spanning under 1000 tokens, a singular design, or a non-positive fixed cost or rate."""
    if len(obs) < 2 or max(n for n, _ in obs) - min(n for n, _ in obs) < 1000: return None
    suu = sun = snn = sut = snt = 0.0
    for n, t in obs:
        u = ubatches(n)
        suu += u * u; sun += u * n; snn += n * n; sut += u * t; snt += n * t
    den = suu * snn - sun * sun
    if den <= 0: return None
    a = (sut * snn - snt * sun) / den      # seconds per ubatch (the expert stream)
    b = (suu * snt - sun * sut) / den      # seconds per token (1/rate)
    if a <= 0 or b <= 0: return None
    return a, 1.0 / b


def observe_card(tokens, seconds, handoff_seconds):
    """A card prefill happened: refit the fixed cost and rate from the last 8, and the handoff per token."""
    if not ARGS.learn or tokens < 256: return
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


def predict(cold):
    """Seconds of prompt work for `cold` uncached tokens: on Metal, and on the card including the handoff. The card's
    expert stream is charged PER UBATCH, so a prompt past the ubatch pays it again - the single-ubatch form
    under-predicted a 62,000-token prompt by 11 s (found while preparing that run, 2026-09-17)."""
    metal = cold / ARGS.metal_rate
    card = ubatches(cold) * ARGS.card_fixed + cold / ARGS.card_rate + cold * ARGS.handoff_per_token
    return metal, card


def decide(cold, n_total, card_up):
    """Where the cold part of a prompt goes. The operator's cap wins (cold >= --threshold always goes to the card, so a
    Metal server can be kept off long prefills); otherwise the cost model picks the faster side. card_up is a callable,
    consulted only when the card would be chosen: a health probe is a round trip."""
    if n_total <= 1 or cold <= 0: return "metal", "nothing cold" if cold <= 0 else "one token"
    m, c = predict(cold)
    if ARGS.threshold and cold >= ARGS.threshold:
        want, why = "card", f"cold {cold} >= cap {ARGS.threshold}; predicted card {c:.1f} s, metal {m:.1f} s"
    elif ARGS.auto and c < m:
        want, why = "card", f"predicted card {c:.1f} s < metal {m:.1f} s"
    else:
        return "metal", f"predicted metal {m:.1f} s <= card {c:.1f} s" if ARGS.auto else "auto off, under the cap"
    if not card_up(): return "metal", why + "; card server not healthy"
    return want, why


def selftest():
    """The cost model at a few sizes, the break-even, and the two properties the rule must have."""
    lo, hi = 1, 200000
    while hi - lo > 1:
        mid = (lo + hi) // 2; m, c = predict(mid)
        if c < m: hi = mid
        else: lo = mid
    print(f"cost model: metal {ARGS.metal_rate:.0f} tok/s; card {ARGS.card_fixed:.1f} s per ubatch of {ARGS.card_ubatch} + {ARGS.card_rate:.0f} tok/s + handoff {ARGS.handoff_per_token*1e6:.0f} us/token; break-even {hi} cold tokens")
    ok = True
    for cold in (512, 2000, 5000, 8000, 24000, 64000):
        m, c = predict(cold); want, why = decide(cold, cold + 1, lambda: True)
        print(f"   {cold:>6} cold: metal {m:6.1f} s  card {c:6.1f} s ({ubatches(cold)} ubatch{'es' if ubatches(cold) > 1 else ''})  -> {want}")
        ok &= (want == "card") == (c < m or (ARGS.threshold and cold >= ARGS.threshold))
    ok &= decide(24000, 24001, lambda: False)[0] == "metal"          # a card that is down never wins
    ok &= decide(0, 5000, lambda: True)[0] == "metal"                # nothing cold: nothing to route
    fit = fit_line([(6974, 8.5), (23691, 15.3)])                      # the two measured points give back the defaults
    ok &= fit is not None and abs(fit[0] - 5.66) < 0.05 and abs(fit[1] - 2458) < 5
    ok &= ubatches(24576) == 1 and ubatches(24577) == 2 and ubatches(62031) == 3      # the stream is per ubatch
    m3, c3 = predict(62031)                                           # and a 3-ubatch prompt pays it three times
    ok &= abs(c3 - (3 * ARGS.card_fixed + 62031 / ARGS.card_rate + 62031 * ARGS.handoff_per_token)) < 0.01
    ok &= fit_line([(6974, 8.5)]) is None and fit_line([(1000, 5.0), (1500, 5.2)]) is None   # too few, too narrow
    ok &= fit_line([(1000, 9.0), (9000, 5.0)]) is None                # a negative slope is not a rate
    print(f"fit of the two measured points: fixed {fit[0]:.2f} s, {fit[1]:.0f} tok/s" if fit else "fit: FAILED")
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
            tail = (tail + chunk)[-4096:]
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
            route = "metal"; ids = None
            if isinstance(body, dict):
                try: ids = prompt_tokens(path, body)
                except Exception as e: log(f"tokenize failed ({e}); metal-only")
            if ids:
                cached = common_prefix(STATE["metal_tokens"], ids); cold = len(ids) - cached
                want, why = decide(cold, len(ids), lambda: healthy(ARGS.card))
                if want == "card":
                    try: card_prefill(ids); route = "card+metal"
                    except Exception as e: log(f"card path failed ({e}); metal-only")
                log(f"{path}: {len(ids)} tokens, {cached} cached at metal, {cold} cold -> {route} ({why})")
                body["cache_prompt"] = True; body["id_slot"] = ARGS.slot; raw = json.dumps(body).encode()
            t0 = time.perf_counter(); status, tail = self._relay("POST", self.path, raw)
            if ids and status == 200: STATE["metal_tokens"] = list(ids)
            # the self-check, when the reply was not a stream: Metal's own prompt_n / cache_n for this request
            try:
                d = json.loads(tail) if tail.strip().startswith(b"{") else None
                t = (d or {}).get("timings") or {}
                if t:
                    log(f"metal: prompt_n {t.get('prompt_n')} cache_n {t.get('cache_n')} prompt {t.get('prompt_ms', 0)/1000:.1f} s, "
                        f"decode {t.get('predicted_n')} tok at {t.get('predicted_per_second', 0):.1f} tok/s (wall {time.perf_counter()-t0:.1f} s)")
                    if route == "metal" and t.get("prompt_n") and t.get("prompt_ms"): observe_metal(int(t["prompt_n"]), t["prompt_ms"] / 1000.0)
            except ValueError: pass


def main():
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--listen", type=int, default=8095); ap.add_argument("--metal", default="http://127.0.0.1:8091")
    ap.add_argument("--card", default="http://127.0.0.1:8092")
    ap.add_argument("--threshold", type=int, default=0, help="operator's cap: cold tokens at or above which the card ALWAYS prefills (0 = off); keeps a Metal server off long prefills")
    ap.add_argument("--no-auto", dest="auto", action="store_false", help="disable the cost model (then only --threshold routes to the card)")
    ap.add_argument("--metal-rate", type=float, default=660.0, help="Metal prefill tok/s for this model (measured 664 on the M4 Max, Qwen3-Coder-Next)")
    ap.add_argument("--card-rate", type=float, default=2460.0, help="card marginal prefill tok/s (fit of 23,691 tok/15.3 s and 6,974 tok/8.5 s)")
    ap.add_argument("--card-fixed", type=float, default=5.7, help="card fixed seconds per prompt: the expert stream across the link (same fit)")
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
