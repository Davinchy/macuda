#!/usr/bin/env python3
"""disagg-router-model-test.py — the expert-GiB cost model in tools/disagg-router.py, held against every card
prefill recorded in docs/disagg-inference-state.md. Card-free: the two GGUFs are opened for their HEADERS only (the
key/value table and the tensor directory, a few MB at the front of the file); no weight is mapped or read, no host
process, no network.

The model: a card prefill costs, per ubatch, the expert bytes streamed across the link divided by the link rate,
plus the tokens over a marginal rate, plus - for the FIRST prefill after a server start - a cold-mapping cost
proportional to the same streamed bytes. Windows 2 and 3 (2026-09-17) measured six prefills across two models,
two residency settings and two driver builds; every one of them is a check here, with the tolerance the state doc
recorded for it. A refit that cannot reproduce the points it came from is the bug this project already had once
(9574c6b), so the shipped constants are held to the same rule.

  python3 tools/disagg-router-model-test.py       exit 0 = every check passed; 2 = a model file is not on this Mac
"""
import importlib.util, os, sys, types

CODER = "/Volumes/512SSD/LocalCode/offline-ai-kit/models/Qwen3-Coder-Next-UD-Q4_K_XL.gguf"
MOE35 = "/Volumes/512SSD/EGPU_MAC_Nvidia/models/Qwen3.5-35B-A3B-Q4_K_M.gguf"
GIB = float(1 << 30)


def load_router():
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location("disagg_router", os.path.join(here, "disagg-router.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def model_args(r, layout, ncpumoe, card_rate, link_rate=5.46, cold_rate=0.0465):
    """ARGS as main() would build them for this model and residency, with the shipped physical constants."""
    s = r.streamed_gib(layout, ncpumoe)
    return types.SimpleNamespace(card_ubatch=24576, metal_rate=660.0, card_rate=card_rate, handoff_per_token=3e-5,
                                 streamed_gib=s, link_rate=link_rate, cold_rate=cold_rate, card_fixed=s / link_rate,
                                 threshold=0, auto=True, learn=True)


def main():
    for p in (CODER, MOE35):
        if not os.path.exists(p):
            print(f"  (skipped: {p} is not on this machine, so nothing was checked)"); return 2
    r = load_router()
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
        ok = ok and cond

    def predict_card(r, tokens, first):
        r.STATE["card_prefills"] = 0 if first else 1
        return r.predict(tokens, tokens, 0)[1] - tokens * r.ARGS.handoff_per_token   # prompt_ms excludes the handoff

    # --- the layouts, against what was read off the GGUF metadata by hand on 2026-09-18 ---
    coder = r.expert_layout(CODER)
    check("Coder-Next: arch qwen3next, 48 MoE layers, 512 experts, 10 used",
          coder["arch"] == "qwen3next" and len(coder["blocks"]) == 48 and coder["n_expert"] == 512 and coder["n_expert_used"] == 10,
          f"{coder['arch']} {len(coder['blocks'])} {coder['n_expert']}/{coder['n_expert_used']}")
    # 43.69 GiB from the file's own tensor directory (the Unsloth dynamic quant keeps some expert layers above Q4_K); it
    # is the "43.7 GiB" window 2's 5.46 GiB/s was derived from, and the 43.50 figure in this project's notes was the
    # stray one of the two (a Q4_K/Q5_K hand estimate) - corrected 2026-09-19 when this reader first counted them.
    check("Coder-Next: experts total 43.69 GiB (+-0.05)", abs(coder["total"] / GIB - 43.69) < 0.05, f"{coder['total'] / GIB:.3f} GiB")
    moe = r.expert_layout(MOE35)
    check("35B-A3B: 40 MoE layers, 256 experts, 8 used", len(moe["blocks"]) == 40 and moe["n_expert"] == 256 and moe["n_expert_used"] == 8,
          f"{moe['arch']} {len(moe['blocks'])} {moe['n_expert']}/{moe['n_expert_used']}")
    check("35B-A3B: experts total 18.2 GiB (+-0.2)", abs(moe["total"] / GIB - 18.2) < 0.2, f"{moe['total'] / GIB:.3f} GiB")

    # --- window 2 (driver ad308bd, Coder-Next, card rate 5150 tok/s): the control at NCPUMOE=48 ---
    r.ARGS = model_args(r, coder, 48, 5150.0)
    check("NCPUMOE=48 streams 43.69 GiB: fixed 8.00 s (the shipped 8.0 was this number)", abs(r.ARGS.card_fixed - 8.00) < 0.05, f"{r.ARGS.card_fixed:.2f} s")
    c24 = predict_card(r, 23691, first=False)
    check("control 24K, warm (23,691 tok): 12.59 s observed", abs(c24 - 12.588) < 0.3, f"predicted {c24:.2f} s")
    c7 = predict_card(r, 6974, first=True)
    check("control 7K, FIRST after load (6,974 tok): 11.4 s observed", abs(c7 - 11.4) < 0.3, f"predicted {c7:.2f} s = warm {c7 - 43.69 * 0.0465:.2f} + first {43.69 * 0.0465:.2f}")

    # --- window 2's treatment at NCPUMOE=30: the byte fraction, not a new constant ---
    r.ARGS = model_args(r, coder, 30, 5150.0)
    check("NCPUMOE=30 streams 30/48 of the bytes: fixed 4.98 s", abs(r.ARGS.card_fixed - 4.98) < 0.05, f"{r.ARGS.streamed_gib:.2f} GiB, {r.ARGS.card_fixed:.2f} s")
    t24 = predict_card(r, 23691, first=False)
    check("treatment 24K, warm: 9.6 s observed", abs(t24 - 9.6) < 0.3, f"predicted {t24:.2f} s")
    t7 = predict_card(r, 6974, first=True)
    check("treatment 7K, FIRST after load: 7.6 s observed", abs(t7 - 7.6) < 0.3, f"predicted {t7:.2f} s")

    # --- window 3 (driver a988ec7, 35B-A3B at NCPUMOE=40, warm line fitted 3.226 s + 7,442 tok/s) ---
    r.ARGS = model_args(r, moe, 40, 7442.0)
    check("35B-A3B streams 18.2 GiB: fixed 3.33 s vs the warm line's 3.23", abs(r.ARGS.card_fixed - 3.226) < 0.15, f"{r.ARGS.card_fixed:.2f} s")
    w2 = predict_card(r, 7250, first=False)
    check("t2 chat (7,250 tok, second): 4.2 s observed", abs(w2 - 4.2) < 0.3, f"predicted {w2:.2f} s")
    w3 = predict_card(r, 22879, first=False)
    check("t3 prompt B (22,879 tok, third): 6.3 s observed", abs(w3 - 6.3) < 0.3, f"predicted {w3:.2f} s")
    w1 = predict_card(r, 21233, first=True)
    check("t1 prompt A (21,233 tok, FIRST): 7.0 s observed", abs(w1 - 7.0) < 0.3, f"predicted {w1:.2f} s")

    # --- and the shape: more residency, strictly cheaper, never below the marginal cost alone ---
    fixed = [model_args(r, coder, k, 5150.0).card_fixed for k in (48, 40, 30, 20, 0)]
    check("fixed cost falls monotonically with residency and reaches 0 at NCPUMOE=0", all(a > b for a, b in zip(fixed, fixed[1:])) and fixed[-1] == 0.0,
          " ".join(f"{x:.2f}" for x in fixed))
    check("-1 means every MoE layer on the host", model_args(r, coder, -1, 5150.0).card_fixed == model_args(r, coder, 48, 5150.0).card_fixed)
    check("a request past 48 is clamped, not extrapolated", model_args(r, coder, 99, 5150.0).card_fixed == model_args(r, coder, 48, 5150.0).card_fixed)

    print("model test:", "OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
