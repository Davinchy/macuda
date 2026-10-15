# send the soak's greedy probe every 12 s while batch6's loadgen runs; keep content + reasoning of every reply;
# at the end: how many replies are identical to the majority, how the divergent ones differ, and whether they are coherent text
import json, urllib.request, hashlib, time, sys, collections, os
url="http://127.0.0.1:8090/v1/chat/completions"; out=sys.argv[1]; secs=int(sys.argv[2])
prompt="The three laws of thermodynamics, explained for a bright twelve-year-old, are:"
rows=[]; t0=time.time()
while time.time()-t0 < secs:
    body=json.dumps({"messages":[{"role":"user","content":prompt}],"temperature":0,"max_tokens":128,"seed":7}).encode()
    try:
        r=urllib.request.urlopen(urllib.request.Request(url, body, {"Content-Type":"application/json"}), timeout=300)
        j=json.loads(r.read()); m=j["choices"][0]["message"]; t=j.get("timings",{})
        rows.append({"t":time.strftime("%H:%M:%S"),"content":m.get("content") or "","reasoning":m.get("reasoning_content") or "","n":t.get("predicted_n"),"draft_acc":t.get("draft_n_accepted"),"draft_n":t.get("draft_n")})
    except Exception as e:
        rows.append({"t":time.strftime("%H:%M:%S"),"err":str(e)[:120]})
    time.sleep(12)
json.dump(rows, open(out,"w"), indent=1)
full=[(r["reasoning"]+"\n----\n"+r["content"]) for r in rows if "err" not in r]
c=collections.Counter(hashlib.sha256(x.encode()).hexdigest()[:12] for x in full)
maj,majn=c.most_common(1)[0]
print(f"probes {len(rows)}, errors {sum('err' in r for r in rows)}, distinct full texts {len(c)}, majority {majn}/{len(full)}")
majtext=next(x for x in full if hashlib.sha256(x.encode()).hexdigest()[:12]==maj)
for x in full:
    h=hashlib.sha256(x.encode()).hexdigest()[:12]
    if h==maj: continue
    # first divergence point against the majority text
    i=next((k for k in range(min(len(x),len(majtext))) if x[k]!=majtext[k]), min(len(x),len(majtext)))
    print(f"divergent {h}: diverges at char {i}/{len(majtext)}: ...{majtext[max(0,i-40):i]!r} | majority next {majtext[i:i+40]!r} | this next {x[i:i+40]!r}")
    print(f"   its tail: {x[-120:]!r}")
