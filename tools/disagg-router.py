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
     prompt it last sent there); under the threshold, or with no healthy card server, the request goes to Metal as is;
  3. otherwise the card prefills all but the LAST token (the recurrent state cannot be rewound, so the decode side must
     evaluate one token itself), saves the slot's state under a name derived from the tokens, Metal restores it, and the
     original request is forwarded to Metal with cache_prompt on: Metal finds N-1 cached tokens, evaluates one, decodes.
     Any failure on the card path falls back to Metal alone, loudly.
Streams are relayed byte for byte. One request at a time through the router: both servers have one slot.

The Metal response's timings.cache_n is the self-check: after a card prefill of N-1 tokens it must read N-1. It is
logged on every non-streamed request; a smaller number means the two servers tokenized differently, and the card's
work was wasted rather than wrong.

  python3 tools/disagg-router.py [--listen 8095] [--metal URL] [--card URL] [--threshold 5000] [--state-dir DIR] [--slot 0]

The card server holds the card for as long as it is up: it is a scheduled tenant under tools/nv_shim_step.sh and the
lock, exactly like tools/serve.sh, and starting it is a card step. The Metal server maps the whole model: under the
host rule in the README that is a card-affecting job while anyone holds the card.
"""
import argparse, hashlib, http.client, json, os, sys, threading, time, urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ARGS = None
LOCK = threading.Lock()            # one request at a time: one slot on each server
STATE = {"metal_tokens": []}       # the prompt tokens Metal's slot holds, as far as the router has sent them


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
                if cold >= ARGS.threshold and len(ids) > 1 and healthy(ARGS.card):
                    try: card_prefill(ids); route = "card+metal"
                    except Exception as e: log(f"card path failed ({e}); metal-only")
                log(f"{path}: {len(ids)} tokens, {cached} cached at metal, {cold} cold -> {route}")
                body["cache_prompt"] = True; body["id_slot"] = ARGS.slot; raw = json.dumps(body).encode()
            t0 = time.perf_counter(); status, tail = self._relay("POST", self.path, raw)
            if ids and status == 200: STATE["metal_tokens"] = list(ids)
            # the self-check, when the reply was not a stream: Metal's own prompt_n / cache_n for this request
            try:
                d = json.loads(tail) if tail.strip().startswith(b"{") else None
                t = (d or {}).get("timings") or {}
                if t: log(f"metal: prompt_n {t.get('prompt_n')} cache_n {t.get('cache_n')} prompt {t.get('prompt_ms', 0)/1000:.1f} s, "
                          f"decode {t.get('predicted_n')} tok at {t.get('predicted_per_second', 0):.1f} tok/s (wall {time.perf_counter()-t0:.1f} s)")
            except ValueError: pass


def main():
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--listen", type=int, default=8095); ap.add_argument("--metal", default="http://127.0.0.1:8091")
    ap.add_argument("--card", default="http://127.0.0.1:8092"); ap.add_argument("--threshold", type=int, default=5000, help="cold tokens at or above which the card prefills")
    ap.add_argument("--state-dir", default=None, help="the --slot-save-path both servers share, to delete handed-over files"); ap.add_argument("--slot", type=int, default=0)
    ARGS = ap.parse_args()
    if not healthy(ARGS.metal, 5): log(f"Metal server at {ARGS.metal} is not healthy; refusing to start"); sys.exit(1)
    log(f"metal {ARGS.metal} up; card {ARGS.card} {'up' if healthy(ARGS.card) else 'down (metal-only until it appears)'}; threshold {ARGS.threshold} cold tokens; listening on 127.0.0.1:{ARGS.listen}")
    ThreadingHTTPServer(("127.0.0.1", ARGS.listen), Handler).serve_forever()


if __name__ == "__main__":
    main()
