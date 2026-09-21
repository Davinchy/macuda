#!/usr/bin/env python3
"""The coding-agent benchmark report: two hardware configs, the sweep and the replay, the differences, the anomalies.

    python tools/agent-report.py --a 3060:logs/agentbench-3060.json:logs/replay-3060-X.json \
                                 --b metal:logs/agentbench-metal.json:logs/replay-metal-Y.json [--md out.md]

Each config is label:sweep.json:replay.json (the replay may be omitted). The sweep JSON is llama-bench's -o json; the
environment file written beside it (<sweep>.env.txt) supplies the build id and the PCIe link. Anomalies flagged: a
repetition spread over 5% of the mean, a replay turn that reprocessed its whole context, a link narrower than x4 or
slower than 8 GT/s during the run (what a Thunderbolt enclosure holds), a link that changed during the run, and any
throttle reason the driver's sensors published.
"""
import argparse, json, os, sys

def label_of(r):
    if r["n_gen"] == 0: return f"pp{r['n_prompt']}"
    if r["n_prompt"] == 0: return f"tg{r['n_gen']}"
    return f"pp{r['n_prompt']}+tg{r['n_gen']}"

def load_sweep(path):
    rows = json.load(open(path)); env = {}
    ep = path + ".env.txt"
    if os.path.exists(ep):
        for line in open(ep):
            if "=" in line: k, v = line.rstrip("\n").split("=", 1); env[k] = v
    return rows, env

def sweep_table(rows):
    out = ["| test | depth | tok/s | ± | spread |", "|---|---:|---:|---:|---:|"]
    anomalies = []
    for r in rows:
        cv = r["stddev_ts"] / r["avg_ts"] * 100 if r["avg_ts"] else 0
        flag = " **>5%**" if cv > 5 else ""
        out.append(f"| {label_of(r)} | {r['n_depth']} | {r['avg_ts']:.1f} | {r['stddev_ts']:.1f} | {cv:.1f}%{flag} |")
        if cv > 5: anomalies.append(f"{label_of(r)} at depth {r['n_depth']}: repetitions spread {cv:.1f}% of the mean")
    return out, anomalies

def replay_table(rep):
    out = ["| turn | prompt tokens | processed | cached | TTFT s | prompt tok/s | gen tok/s | turn s | cache |", "|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    anomalies = []
    for t in rep["turns"]:
        miss = "first" if t["turn"] == 1 else "MISS" if t["cache_miss"] else "hit"
        out.append(f"| {t['turn']} | {t['prompt_tokens']} | {t['processed']} | {t['cached']} | {t['ttft_s']:.2f} | {t['prompt_tps'] or '-'} | {t['gen_tps'] or '-'} | {t['turn_s']:.2f} | {miss} |")
        if t["cache_miss"] and t["turn"] > 1: anomalies.append(f"turn {t['turn']}: cache reuse failed, {t['processed']} of {t['prompt_tokens']} tokens reprocessed")
    out.append(f"| **total** | | | | | | | **{rep['total_s']:.1f}** | |")
    return out, anomalies

def pct(a, b):  # b relative to a
    return (b - a) / a * 100 if a else float("nan")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True); ap.add_argument("--md", default="")
    a = ap.parse_args()
    cfgs = []
    for spec in (a.a, a.b):
        parts = spec.split(":"); label, sweep = parts[0], parts[1]; replay = parts[2] if len(parts) > 2 and parts[2] else None
        rows, env = load_sweep(sweep)
        rep = json.load(open(replay)) if replay and os.path.exists(replay) else None
        renv = {}
        if replay and os.path.exists(replay + ".env.txt"):
            for line in open(replay + ".env.txt"):
                if "=" in line: k, v = line.rstrip("\n").split("=", 1); renv[k] = v
        cfgs.append({"label": label, "rows": rows, "env": env, "replay": rep, "replay_env": renv, "sweep_path": sweep})
    md = ["# Coding-agent benchmark: " + " vs ".join(c["label"] for c in cfgs), ""]
    for c in cfgs:
        e = c["env"]; r0 = c["rows"][0]
        md += [f"## {c['label']}", "",
               f"- model: `{os.path.basename(r0['model_filename'])}` ({r0['model_type']}, {r0['model_size']/2**30:.2f} GiB)",
               f"- llama.cpp build: `{r0['build_commit']}` (#{r0['build_number']}), backend {r0['backends']}, ngl {r0['n_gpu_layers']}, flash attention {r0['flash_attn']}, {r0['n_threads']} threads",
               f"- driver build id: `{e.get('driver_build_id') if e.get('driver_build_id') not in (None, '', 'fffffff') else '-'}`; PCIe link during the run: {e.get('pcie_link_width', '-')} at {e.get('pcie_link_speed', '-')}"
               + (f" (before the run: {e['pcie_link_width_before']} at {e['pcie_link_speed_before']})" if e.get('pcie_link_speed_before') else "") + f"; {e.get('macos', '')}",
               f"- date: {e.get('date', '-')}", "", "### Sweep", ""]
        t, an = sweep_table(c["rows"]); md += t; c["anomalies"] = an
        if c["replay"]:
            re_ = c["replay_env"]
            md += ["", "### Replay, ten turns", ""]
            if re_.get("pcie_link_speed"): md += [f"- PCIe link during the replay: {re_.get('pcie_link_width', '-')} at {re_['pcie_link_speed']}; context {re_.get('ctx', '-')}; server up in {re_.get('server_up_s', '-')} s", ""]
            t, an2 = replay_table(c["replay"]); md += t; c["anomalies"] += an2
            rs = re_.get("pcie_link_speed", "")
            try: rg = float(rs.split()[0]) if rs else None
            except ValueError: rg = None
            if rg is not None and rg < 8.0: c["anomalies"].append(f"replay ran on a PCIe link at {rs} - not the 8 GT/s the enclosure holds")
            if ";" in re_.get("pcie_link_seen", ""): c["anomalies"].append(f"the link changed during the replay: {re_['pcie_link_seen']}")
        md += [""]
    # differences, b relative to a, matched by (label, depth)
    A, B = cfgs
    md += [f"## {B['label']} relative to {A['label']}", "", "| test | depth | " + A["label"] + " | " + B["label"] + " | difference |", "|---|---:|---:|---:|---:|"]
    ia = {(label_of(r), r["n_depth"]): r for r in A["rows"]}
    for r in B["rows"]:
        k = (label_of(r), r["n_depth"])
        if k in ia: md.append(f"| {k[0]} | {k[1]} | {ia[k]['avg_ts']:.1f} | {r['avg_ts']:.1f} | {pct(ia[k]['avg_ts'], r['avg_ts']):+.1f}% |")
    if A["replay"] and B["replay"]:
        md.append(f"| replay total | | {A['replay']['total_s']:.1f} s | {B['replay']['total_s']:.1f} s | {pct(A['replay']['total_s'], B['replay']['total_s']):+.1f}% (time: negative is faster) |")
    md += ["", "## Anomalies", ""]
    any_flag = False
    for c in cfgs:
        for x in c["anomalies"]: md.append(f"- {c['label']}: {x}"); any_flag = True
        w = c["env"].get("pcie_link_width", ""); s = c["env"].get("pcie_link_speed", "")
        # the link a Thunderbolt enclosure holds is x4 at 8 GT/s (Gen3); the driver caps the firmware there, and anything
        # narrower or slower during the run (a Gen1 link the firmware never brought up, a lane dropped) is an anomaly
        try: gts = float(s.split()[0]) if s else None
        except ValueError: gts = None
        if w and w != "x4" or gts is not None and gts < 8.0:
            md.append(f"- {c['label']}: PCIe link {w} at {s} during the run - not the x4 at 8 GT/s the enclosure holds"); any_flag = True
        seen = c["env"].get("pcie_link_seen", "")
        if seen and ";" in seen:
            md.append(f"- {c['label']}: the link changed during the run: {seen}"); any_flag = True
    if not any_flag: md.append("- none")
    text = "\n".join(md)
    print(text)
    if a.md: open(a.md, "w").write(text + "\n")

if __name__ == "__main__": main()
