#!/usr/bin/env python3
"""A test's pass/fail decision must be the last thing it does.

WHY. On 2026-09-15 Session B found that test_rm_free.c EXITED 0 WITH "FAIL" PRINTED. It accumulated failures into
`fails` and consulted that counter at line 117 of a 318-line main. The guard was CORRECT WHEN WRITTEN — line 117 was
the end of main then. The file grew: a mapping record, a writability check and the entire C4b decision were appended
BELOW a guard that had already returned, and `make test` reported green over all of it.

What it cost, concretely: the four rows proving the READ_ONLY=3 fix — the fix for the inverted guarantee 7 — sat
below the guard and never ran. "Suite green" was true and meant nothing.

This is the worst of the silent-check shapes collected here. The other four DEGRADE evidence: a check declines to run
and says nothing, so you learn less than you think. This one MANUFACTURES it: the run prints FAIL and reports
success, so you learn the opposite of what happened.

THE TELL IS NOT A BAD CHECK. Every CHECK site in that file read correctly. The tell is structural — a pass/fail
decision with code after it — and the defence has to be structural too, because "remember to append above the guard"
is a rule that holds until somebody is in a hurry.

    python3 libtinynv/tools/exit_status_lint.py [files...]   # default: every test/bench C file in the tree
    python3 libtinynv/tools/exit_status_lint.py --selftest    # plant both defects and require them to be caught
    python3 libtinynv/tools/exit_status_lint.py --counters    # also print what each file was judged to count
"""
import os
import re
import sys

# Two levels up from libtinynv/tools/ is the repository root. It lives here rather than in vm/trace/ because this
# guards libtinynv's OWN tests as much as the VM track's, and a guard that only runs from another branch's
# checkout protects main only for as long as somebody remembers to run it for you — Session B's point, and right.
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DIRS = ["libtinynv/test", "vm/guest/bench", "vm/guest/aptest", "vm/tinynvd"]

# COUNTERS ARE DERIVED FROM THE FILE, NOT FROM A LIST OF NAMES I HAPPEN TO KNOW.
#
# They were a list — fails|failures|broken|nfail|nerr — and Session B broke the lint with a two-line negative
# control: a file counting into `bad`, with a defect planted below its verdict, was reported CLEAN. The list did not
# have `bad`. In B's tree alone, 6 of the 13 files this tool claimed "no opinion" about DID count failures, under
# `bad`, `wrong`, `missing`, `unmatched` and `oom` — and five of those are the card tests A's gating leans on.
#
# THAT IS THE FOURTH TIME THIS FILE HAS HAD THE SAME DEFECT, after prose matched as code, the counter searched only
# in main's body, and examined conflated with checked. Every one put a file in a bucket that READS AS COVERAGE while
# nothing looked at it. Widening the regex would have moved the edge; it would not have removed it, and the next
# suite to use `errs` or `problems` would land in the same silent bucket.
#
# So the question is asked of the code instead: an identifier that is INCREMENTED somewhere and RETURNED from main
# is that file's failure count, whatever it is called. B's `bad`, the hardware tests' `wrong`, anything.
INCR = re.compile(r"\b([A-Za-z_]\w*)\s*(?:\+\+|\+=)")
RET = re.compile(r"\breturn\b([^;]*);")
IDENT = re.compile(r"\b([A-Za-z_]\w*)\b")


# Bounded on purpose. This was `if\s*\(([^)]*)\)\s*(?:\{\s*)?(?:[^;{}]*;\s*)*?return\b`, whose nested
# quantifier backtracks catastrophically — the sweep ran for five minutes on real sources and had to be killed. A
# lint nobody can afford to run is not a lint. `[^()]*` inside the parens and a short fixed window after them cover
# `if (fails) return 1;` and `if (fails) { return 1; }`, which is every spelling these suites actually use.
IF_COND = re.compile(r"\bif\s*\(([^()]{0,200})\)")


def if_blocks(body):
    """Every `if (COND)` in the body with the span its block covers: (start, end, cond).

    Brace-matched, so the span is the statements the `if` actually guards. Needed twice — to decide what counts as a
    failure counter, and to decide whether a return is guarded by one — and the second use is where approximating it
    went wrong twice: a fixed character window took an unrelated `return` as proof, and "the nearest preceding if"
    took `if (fails) return 1;` as guarding a `return 0;` three statements BELOW it, which is the exact shape this
    lint exists to catch. A span either contains the return or it does not.
    """
    out = []
    for m in IF_COND.finditer(body):
        j = m.end()
        while j < len(body) and body[j] in " \t\r\n":
            j += 1
        if j >= len(body):
            continue
        if body[j] == "{":
            depth, k = 0, j
            while k < len(body):
                if body[k] == "{":
                    depth += 1
                elif body[k] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                k += 1
            end = k
        else:
            e = body.find(";", j)
            end = e if e > 0 else j
        out.append((j, end, m.group(1)))
    return out


def counters(text, body):
    """Identifiers incremented anywhere in the file and used to DECIDE a return in main.

    Both spellings count, and missing the second regressed this tool the moment the first was fixed:

        return fails ? 1 : 0;    the counter is in the return EXPRESSION
        if (wrong) { ...; return 1; }   the counter is in the CONDITION and the expression is a bare constant

    The second is the commoner spelling in these suites and is the shape of B's test_rm_free.c defect, so a
    definition that only saw the first would decline to judge the very files this exists for — while reporting them
    as having no counter, which is the bucket-that-reads-as-coverage again.
    """
    incremented = set(INCR.findall(text))
    used = set()
    for m in RET.finditer(body):
        used |= set(IDENT.findall(m.group(1)))
    for start, end, cond in if_blocks(body):
        if re.search(r"\breturn\b", body[start:end]):
            used |= set(IDENT.findall(cond))
    return incremented & used

# An assertion site: something that can still discover a failure and so must not sit after the verdict.
SITE = re.compile(r"\b(CHECK|EXPECT|REQUIRE|ASSERT|assert|check_refused|expect_\w+)\s*\(")


def strip_noise(text):
    """Blank out comments and string literals, preserving line structure.

    The first version of this did not, and it immediately produced nine confident false findings against
    test_rm_free.c by matching the PROSE "return an error, it tears down the client..." inside a comment. Which is
    the same defect this file's own author had just removed from header_audit.py for the same reason: a checker
    whose findings are mostly false is worse than no checker, because the next real one lands in a list the reader
    has learned to skim. Newlines are kept so every reported line number still points at the real line.
    """
    out, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            j = text.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i)); i = j
        elif c == "/" and i + 1 < n and text[i + 1] == "*":
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append("".join(ch if ch == "\n" else " " for ch in text[i:j])); i = j
        elif c in "\"'":
            q, j = c, i + 1
            while j < n and text[j] != q:
                j += 2 if text[j] == "\\" else 1
            j = min(j + 1, n)
            out.append("".join(ch if ch == "\n" else " " for ch in text[i:j])); i = j
        else:
            out.append(c); i += 1
    return "".join(out)


def main_span(text):
    """(start, end) character offsets of main()'s body, by brace balance."""
    m = re.search(r"\bint\s+main\s*\([^)]*\)\s*\{", text)
    if not m:
        return None
    i = text.index("{", m.start())
    depth, j = 0, i
    while j < len(text):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return (i, j)
        j += 1
    return None


def check(path):
    """The precise defect, which is narrower than 'a return with code after it'.

    A CONDITIONAL BAIL-OUT IS FINE. `if (fails) return fails;` mid-file only fires when the run is ALREADY failing,
    so nothing that would have failed later is lost — it is a guard against proceeding on a broken premise, and
    test_hw_kernel.c uses three of them correctly. Flagging those was this lint's first behaviour and it was wrong.

    THE DEFECT IS THAT MAIN'S LAST RETURN DOES NOT CONSULT A COUNTER. If the final statement is `return 0;` then
    every failure accumulated after the last counter-consulting return is discarded — the run prints FAIL and exits
    0. That is what B found in test_rm_free.c, where `if (fails) return 1;` at line 117 lost nothing and the
    `return 0;` at line 317 lost everything.

    Returns (findings, counter_names): the names are reported even when there is no finding, so a reader can see
    WHAT this tool thought the file counted rather than only whether it approved.
    """
    text = strip_noise(open(path, errors="ignore").read())
    span = main_span(text)
    if not span:
        return [], set()
    a, b = span
    body = text[a:b]
    base = text[:a].count("\n") + 1
    cnt = counters(text, body)
    if not cnt:
        return [], set()
    verdict = re.compile(r"\breturn\b[^;]*\b(?:%s)\b[^;]*;" % "|".join(re.escape(c) for c in sorted(cnt)))
    cnt_re = re.compile(r"\b(?:%s)\b" % "|".join(re.escape(c) for c in sorted(cnt)))
    blocks = if_blocks(body)

    def consults(m):
        """Does this return decide on the counter — in its own expression, or via the `if` guarding it?

        `if (wrong) { printf(...); return 1; }` consults the counter and the return statement does not mention it.
        Matching only the return text classified every hardware test as having NO verdict at all, which turned
        "correct today but unguarded" into "loses evidence now" — a true finding reported at the wrong severity,
        which is its own way of being useless.
        """
        if verdict.search(m.group(0)):
            return True
        # Guarded means INSIDE the if's block, not merely after some `if` that mentions the counter. Two earlier
        # spellings both got this wrong in the dangerous direction: `back.rfind("if")` matched the letters i-f
        # inside words like "specific" in a FAIL message, and "the nearest preceding if" treated
        # `if (fails) return 1;` as guarding a `return 0;` three statements below it — declaring the planted
        # defect safe. Containment is the property; anything weaker approves the thing being looked for.
        return any(a <= m.start() < b and cnt_re.search(cond) for a, b, cond in blocks)

    returns = list(RET.finditer(body))
    if not returns:
        return [("LOSING", base + body.count("\n"), "main counts failures and never returns at all")], cnt
    last = returns[-1]
    if consults(last):
        return [], cnt
    verdicts = [m for m in returns if consults(m)]
    names = "/".join(sorted(cnt))
    if not verdicts:
        return [("LOSING", base + body[:last.start()].count("\n"),
                 "main's last statement is `%s` and NOTHING anywhere consults %s"
                 % (last.group(0).strip(), names))], cnt
    after = [m for m in SITE.finditer(body) if m.start() > verdicts[-1].end()]
    vline = base + body[:verdicts[-1].start()].count("\n")
    if after:
        return [("LOSING", base + body[:last.start()].count("\n"),
                 "%d assertion site(s) run after the last %s-consulting return at line %d and cannot affect the "
                 "exit status; main ends `%s`" % (len(after), names, vline, last.group(0).strip()))], cnt
    # TWO SEVERITIES, because they are different facts and a tool that cannot gate is not used.
    #
    # LOSING: evidence is being discarded right now. UNGUARDED: the file is correct today — the counter is consulted
    # at `vline` and only non-failing work follows — but the last statement does not consult it, so a check appended
    # later is lost silently. That is precisely B's test_rm_free.c before it grew: correct when written, and the
    # growing is what nobody notices. Reported, not failed, or every clean tree exits 1 and the tool gets ignored.
    # ACCUMULATING vs FAIL-FAST — Session B's distinction, and it is what makes UNGUARDED enforceable rather than
    # merely interesting. Both shapes end without consulting the count, but the risk is not comparable:
    #
    #   ACCUMULATING  the file has assertion macros that increment a counter consulted once, somewhere. The natural
    #                 way to add a test is to append a CHECK — and it then silently does not count. This is the
    #                 trap, and it is exactly what bit B's test_rm_free.c.
    #   FAIL-FAST     every failure site returns 1 itself. The natural way to add a check is to copy the
    #                 neighbouring return, so the last statement not consulting the count is nearly harmless.
    #
    # Measured across both trees when B proposed it: all the UNGUARDED files here have ZERO assertion-macro sites;
    # B's two genuinely broken ones had 66 and 18. So the split is sharp in practice, not just in principle, and a
    # tree can be held at the stricter setting without churn.
    nsites = len(list(SITE.finditer(body)))
    kind = "UNGUARDED-ACCUMULATING" if nsites else "UNGUARDED-FAILFAST"
    detail = ("%d assertion site(s) increment a counter consulted only at line %d, so a CHECK appended below it "
              "would not count" % (nsites, vline)) if nsites else (
              "every failure site returns directly (no assertion macros), so this is nearly harmless — a new check "
              "copied from its neighbours returns too")
    return [(kind, base + body[:last.start()].count("\n"),
             "consults %s at line %d and then ends `%s`; %s" % (names, vline, last.group(0).strip(), detail))], cnt


SELFTEST = """
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("FAIL\\n"); } } while (0)

int main(void) {
  int fails = 0;
  CHECK(1 == 1, "a passing check");
  if (fails) return 1;
  CHECK(2 == 2, "a check appended below the guard, months later");
  return 0;
}
"""

# UNGUARDED-ACCUMULATING: the counter IS consulted and no site follows it, so nothing is lost today — but the file
# has an assertion macro, so appending a CHECK below the consult would be. This is the class --strict fails on, and
# a selftest that only planted LOSING cases could never tell whether that flag worked.
SELFTEST_ACCUM = """
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("FAIL\\n"); } } while (0)

int main(void) {
  int fails = 0;
  CHECK(1 == 1, "a passing check");
  if (fails) { printf("some failed\\n"); return 1; }
  cleanup_that_cannot_fail();
  return 0;
}
"""

# B's negative control, kept verbatim as a second case: a counter under a name no list contained. This is the one
# that proved a name list cannot be the coverage.
SELFTEST_ODD_NAME = """
#define CHECK(c, ...) do { if (!(c)) { bad++; printf("FAIL\\n"); } } while (0)

int main(void) {
  int bad = 0;
  CHECK(1 == 1, "a passing check");
  return bad ? 1 : 0;
  CHECK(0, "a planted defect below the verdict");
  return 0;
}
"""


def selftest():
    """Build the defect and require it to be caught. A lint never seen to fail is not known to work.

    This is Session B's standing practice, adopted after both of 2026-09-15's defects turned out to be invisible to
    every green run and obvious to one deliberate break. It costs a temp file.
    """
    import tempfile
    rc = 0
    cases = (("known counter name", SELFTEST, "LOSING"),
             ("counter named `bad`, from B", SELFTEST_ODD_NAME, "LOSING"),
             ("accumulating but not yet losing", SELFTEST_ACCUM, "UNGUARDED-ACCUMULATING"))
    for name, src, want in cases:
        with tempfile.NamedTemporaryFile("w", suffix=".c", delete=False) as f:
            f.write(src)
            path = f.name
        try:
            found, cnt = check(path)
        finally:
            os.unlink(path)
        if not found or found[0][0] != want:
            print("SELFTEST FAILED (%s): expected %s, got %s.\n"
                  "Everything else this tool reports is worthless until this passes."
                  % (name, want, found[0][0] if found else "nothing"), file=sys.stderr)
            rc = 1
        else:
            print("selftest ok  [%s] counter=%s severity=%s — %s"
                  % (name, "/".join(sorted(cnt)), found[0][0], found[0][2][:64]))
    return rc


def main(argv):
    if "--selftest" in argv:
        return selftest()
    verbose_names = "--counters" in argv
    strict = "--strict" in argv
    argv = [a for a in argv if a not in ("--counters", "--strict")]
    files = argv or [os.path.join(ROOT, d, f)
                     for d in DIRS if os.path.isdir(os.path.join(ROOT, d))
                     for f in sorted(os.listdir(os.path.join(ROOT, d))) if f.endswith(".c")]
    n_checked, n_subject, findings = 0, 0, []
    subjects, unjudged = [], []
    for p in files:
        if not os.path.exists(p):
            continue
        n_checked += 1
        # EXAMINED IS NOT CHECKED. A file with no failure counter is not subject to this rule, and counting it in
        # with the rest would report reassurance for files the lint never had an opinion about — which is the same
        # conflation the header audit reports separately and the same one that let B's suite read as green.
        f_found, f_cnt = check(p)
        if f_cnt:
            n_subject += 1
            subjects.append((os.path.relpath(p, ROOT), sorted(f_cnt)))
        elif main_span(strip_noise(open(p, errors="ignore").read())):
            unjudged.append(os.path.relpath(p, ROOT))
        for sev, line, why in f_found:
            findings.append((sev, os.path.relpath(p, ROOT), line, why))

    if not n_checked:
        print("exit_status_lint: NO FILES EXAMINED. This is not a pass — check the paths.", file=sys.stderr)
        return 2
    print("exit_status_lint: %d file(s) examined; %d of them count failures and are subject to this rule.\n"
          "                  The other %d have no failure counter and this lint has no opinion about them."
          % (n_checked, n_subject, len(unjudged)))
    # NAMED, not counted. B's point, and it is the better half of the fix: widening the counter regex moves the
    # edge, listing what fell outside it makes the edge VISIBLE. Had these been named, B would have spotted `wrong`
    # in five card tests in about ten seconds instead of building a negative control to prove they were invisible.
    if unjudged:
        print("\n  No failure counter found in (so NOT judged — check by eye if one of these should be):")
        for u in unjudged:
            print("      %s" % u)
    if verbose_names:
        print("\n  Counters found:")
        for path, names in subjects:
            print("      %-44s %s" % (path, ", ".join(names)))
    if not findings:
        print("Every file that counts failures ends by consulting the count. A block appended to the end of any of\n"
              "them will still be reached, and its failures will still reach the exit status.")
        return 0
    losing = [f for f in findings if f[0] == "LOSING"]
    accum = [f for f in findings if f[0] == "UNGUARDED-ACCUMULATING"]
    unguarded = [f for f in findings if f[0] == "UNGUARDED-FAILFAST"]
    if losing:
        print("\nLOSING EVIDENCE NOW — these can print FAIL and exit 0:")
        for _sev, path, line, why in losing:
            print("  %s:%d  %s" % (path, line, why))
    if accum:
        print("\nUNGUARDED AND ACCUMULATING — correct today, and the natural way to add a test breaks them:")
        for _sev, path, line, why in accum:
            print("  %s:%d  %s" % (path, line, why))
    if unguarded:
        print("\nUNGUARDED BUT FAIL-FAST — correct today, and nearly harmless:")
        for _sev, path, line, why in unguarded:
            print("  %s:%d  %s" % (path, line, why))
    print("\nMove the verdict to the LAST statement of main. Not 'move this block up' — the guard has already been\n"
          "correct once and grown stale; only being last is a property a later edit cannot quietly break.")
    # ONLY "LOSING" FAILS. UNGUARDED files are correct today; failing on them means a clean tree exits 1, and a gate
    # that is red when nothing is wrong gets disabled within a week — after which the LOSING findings it exists for
    # go unseen too. The severity split is only worth having if the exit status respects it.
    # --strict also fails on UNGUARDED-ACCUMULATING, which is the class where appending a test silently loses it.
    # FAIL-FAST files never fail the build: there is no realistic edit that breaks them, and failing on them is how
    # a gate gets disabled.
    if losing:
        return 1
    return 1 if (strict and accum) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
