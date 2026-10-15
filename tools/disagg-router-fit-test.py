#!/usr/bin/env python3
"""disagg-router-fit-test.py — unit test for fit_line() in tools/disagg-router.py. Pure computation: no host
processes, no model, no card, no network (V1, 2026-09-17).

docs/disagg-inference-state.md records that the refit-robustness fix (9574c6b, after the 2026-09-17 window-2 bug
where two single-ubatch points fit themselves exactly and then mispredicted the very prefill they came from by
3.6 s) was "verified in process" during development and never persisted. House style: a check is not trusted
until it has been seen to fail. This pins the properties down as PASS/FAIL so the next change to fit_line can be
checked against them instead of re-deriving them by hand:

  1. fewer than three observations: no fit (the two-point self-fit that caused the bug)
  2. three points that lie on a line the model can represent: recovered
  3. three points on a genuinely DIFFERENT line (a smaller model's shape): tracked, not pulled toward whatever
     constants happen to be loaded
  4. one absurd point among three otherwise-consistent ones: rejected (>20% miss on data it was fitted on)
  5. a prompt spanning multiple ubatches: the per-ubatch fixed cost, not a flat one, is what the fit recovers
     (this is what dac5549 rests on: window 1's 64,577-token / 3-ubatch point)
"""
import importlib.util, sys, types


def load_router():
    spec = importlib.util.spec_from_file_location("disagg_router", "tools/disagg-router.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # module-level code only defines functions; main() runs only under __main__
    mod.ARGS = types.SimpleNamespace(card_ubatch=24576)
    return mod


def line(fixed, rate, ub, tokens):
    """Synthetic (tokens, seconds) observations that sit exactly on seconds = fixed*ubatches + tokens/rate."""
    return [(n, fixed * -(-n // ub) + n / rate) for n in tokens]


def close(a, b, tol):
    return abs(a - b) <= tol * b


def main():
    r = load_router()
    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"  {'PASS' if cond else 'FAIL'}  {name}")
        ok = ok and cond

    two = line(8.0, 5150, 24576, [6974, 23691])
    check("two points -> no fit (the len<3 guard the 09-17 bug needed)", r.fit_line(two) is None)

    three = line(8.0, 5150, 24576, [3000, 12000, 23691])
    fit = r.fit_line(three)
    check("three consistent points refit the line they're on",
          fit is not None and close(fit[0], 8.0, 0.01) and close(fit[1], 5150, 0.01))

    other = line(3.30, 9000, 24576, [3000, 12000, 21300])
    fit2 = r.fit_line(other)
    check("three points on a genuinely different line are tracked exactly",
          fit2 is not None and close(fit2[0], 3.30, 0.01) and close(fit2[1], 9000, 0.01))

    absurd = line(8.0, 5150, 24576, [3000, 12000]) + [(23691, 30.0)]
    check("one absurd point among three is rejected", r.fit_line(absurd) is None)

    multi = line(8.0, 5150, 24576, [10000, 30000, 64577])  # 1, 2, 3 ubatches — window 1's real point is the last
    fit3 = r.fit_line(multi)
    check("a multi-ubatch prompt still recovers the per-ubatch fixed cost",
          fit3 is not None and close(fit3[0], 8.0, 0.01) and close(fit3[1], 5150, 0.01))

    print("fit_line test:", "OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
