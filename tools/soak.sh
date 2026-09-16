#!/bin/sh
# soak.sh — hours of real traffic through the live server (tools/serve.sh), with the checks that make it a soak rather
# than a demo: every request's tok/s is logged; every PROBE_EVERY requests a fixed greedy probe is sent and its text
# must be byte-identical to the first probe's (a silent wrong number shows up here, not as an error); any HTTP error,
# any refusal from the driver, or the server dying ends the run with a line that says so.
#   sh tools/soak.sh [hours] [max_tokens]      log: logs/soak-<ts>.log ; touch logs/soak.stop to end it early
# Lines that matter start with SUMMARY, PROBE, MISMATCH, ERROR, DEAD or DONE (grep them; the Monitor does).
R=${EGPU_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}; H=${1:-2}; MAXT=${2:-256}; EVERY=${PROBE_EVERY:-10}
# hours may be fractional (0.33 = 20 minutes): the shell's arithmetic is integer-only, so the seconds are computed once by awk
SECS=$(awk -v h="$H" 'BEGIN{printf "%d", h*3600}')
URL=$(cat $R/logs/serve.url 2>/dev/null)/v1/chat/completions; [ -n "$URL" ] || { echo "ERROR no logs/serve.url — is the server up?"; exit 2; }
ts=$(date +%Y%m%d-%H%M%S); log=$R/logs/soak-$ts.log; rm -f $R/logs/soak.stop; end=$(( $(date +%s) + SECS ))
echo "== soak $(date '+%F %T')  $URL  ${H}h  max_tokens $MAXT  probe every $EVERY  log $log" | tee "$log"
P1="Explain why the sky is blue to a curious ten-year-old."; P2="Write a short story, under 200 words, about a lighthouse keeper who finds a message in a bottle."
P3="List the steps to make sourdough bread from a starter, briefly."; P4="What are the main differences between TCP and UDP? Keep it concise."
P5="Summarize the plot of Romeo and Juliet in five sentences."; P6="Give me a Python function that checks whether a string is a palindrome, with a docstring."
P7="Describe the water cycle."; P8="What is the difference between weather and climate?"
PROBE="The three laws of thermodynamics, explained for a bright twelve-year-old, are:"
n=0; fails=0; dead=0; first=""; sum=0; cnt=0; tsum=$(date +%s)
# every 30 s, the card's own sensors as the running server publishes them (tinynv-smi reads the publication; it never opens the
# card while someone holds it). One line per sample: SENSOR <time> <temp C> <power W> <graphics MHz>. Blank when sensors are off.
SMI=$R/cuda-shim/build/shim/nv/tinynv-smi
( while [ ! -f $R/logs/soak.stop ]; do r=$(TINYNV_SOCKET=${TINYNV_SOCKET:-${TMPDIR}tinygpu.sock} $SMI 2>/dev/null | awk '/gpu_temp_c/{t=$2} /gpu_power_w/{p=$2} /graphics_mhz/{m=$2} END{if(t!="") printf "%s %s %s", t, p, m}'); [ -n "$r" ] && echo "SENSOR $(date +%T) $r" >> "$log"; sleep 30; done ) &
smipid=$!
req() { # $1 prompt, $2 temp, $3 max_tokens ; prints: tokens ms text-sha
  python3 - "$URL" "$1" "$2" "$3" <<'PY'
import sys, json, hashlib, urllib.request
url, prompt, temp, maxt = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
body = json.dumps({"messages":[{"role":"user","content":prompt}],"temperature":temp,"max_tokens":maxt,"seed":7}).encode()
try:
    r = urllib.request.urlopen(urllib.request.Request(url, body, {"Content-Type":"application/json"}), timeout=600)
    j = json.loads(r.read())
except Exception as e:
    print("ERR", 0, 0, str(e).replace("\n"," ")[:160]); sys.exit(0)
t = j.get("timings", {}); txt = j["choices"][0]["message"]["content"]
print("OK", t.get("predicted_n", 0), round(t.get("predicted_ms", 0), 1), hashlib.sha256(txt.encode()).hexdigest()[:16], t.get("draft_n", ""), t.get("draft_n_accepted", ""))
PY
}
while [ "$(date +%s)" -lt "$end" ] && [ ! -f $R/logs/soak.stop ]; do
  n=$((n+1))
  if [ $((n % EVERY)) -eq 1 ]; then
    set -- $(req "$PROBE" 0 128); st=$1; tok=$2; ms=$3; sha=$4
    if [ "$st" = OK ]; then
      if [ -z "$first" ]; then first=$sha; echo "PROBE #$n first hash $sha ($tok tokens)" >> "$log"
      elif [ "$sha" != "$first" ]; then echo "MISMATCH #$n probe hash $sha differs from first $first ($tok tokens) $(date '+%T')" >> "$log"
      else echo "PROBE #$n ok $sha $(date '+%T')" >> "$log"; fi
    else echo "ERROR #$n probe: $5 $(date '+%T')" >> "$log"; fails=$((fails+1)); fi
  else
    i=$(( (n % 8) + 1 )); eval "p=\$P$i"
    set -- $(req "$p" 0.6 "$MAXT"); st=$1; tok=$2; ms=$3
    if [ "$st" = OK ] && [ "$tok" -gt 0 ] 2>/dev/null; then
      tps=$(python3 -c "print(round($tok/($ms/1000.0),1))"); echo "#$n p$i $tok tok $ms ms $tps tok/s draft $5/$6 $(date '+%T')" >> "$log"
      sum=$(python3 -c "print($sum+$tps)"); cnt=$((cnt+1))
    else echo "ERROR #$n p$i: ${5:-$st} $(date '+%T')" >> "$log"; fails=$((fails+1)); fi
  fi
  pid=$(cat $R/logs/serve.pid 2>/dev/null); kill -0 "$pid" 2>/dev/null || { echo "DEAD server pid $pid gone at request #$n $(date '+%T')" >> "$log"; dead=1; break; }
  [ $fails -ge 5 ] && { echo "DONE stopping: $fails errors" >> "$log"; break; }
  if [ $(( $(date +%s) - tsum )) -ge 600 ]; then mean=$(python3 -c "print(round($sum/max($cnt,1),1))"); echo "SUMMARY $(date '+%T') requests $n, errors $fails, mean $mean tok/s over the last $cnt sampled, probes $( [ -n "$first" ] && echo consistent || echo none)" >> "$log"; sum=0; cnt=0; tsum=$(date +%s); fi
done
kill $smipid 2>/dev/null; mean=$(python3 -c "print(round($sum/max($cnt,1),1))"); echo "DONE $(date '+%F %T') requests $n, errors $fails, dead $dead, last-window mean $mean tok/s" >> "$log"; grep -cE '^MISMATCH' "$log" | sed 's/^/mismatches: /' >> "$log"; tail -3 "$log"
