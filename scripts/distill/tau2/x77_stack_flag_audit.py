"""Which implemented levers cannot fire, because the stack the driver reads never sets them?

Every live driver sources `go_stack.sh` and nothing else, so a flag the production
code reads but that stack never exports does not exist in a live run — however well
it was implemented, unit-tested, or even proven in a replay. Two prescriptions were
found in exactly that state on 2026-08-05 by hand (`T2_QUOTE_PIN`, whose live effect
had already been demonstrated in C282, and `T2_MATCH_COUNT`), and the only reason
they were found is that a read-through happened to walk past them. This asks the
question exhaustively instead, so the next one is found before a run pays for it.

The answer is deliberately not "these flags are missing, turn them on". Three kinds
share the same silence and only the third is a loss:

  parameter   a threshold or cap whose default is the intended value
  off-by-intent  retired, or reserved for an isolated arm script
  unregistered   built and verified, then never put on the stack

Sorting them is a judgement about intent, which this script does not make. It prints
the evidence a person needs to make it: who reads the flag, who (if anyone) sets it,
and the newest line in the authority documents that mentions the flag next to a
verdict word. Read that line before concluding anything.

Usage:
  python x77_stack_flag_audit.py            # full report
  python x77_stack_flag_audit.py --unset    # only flags no driver sets
"""

import argparse
import collections
import glob
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
DOCS = os.path.join(REPO, "reports", "facet_rft_2026")
STACK = "go_stack.sh"

READ = re.compile(r'environ(?:\.get\(|\[)["\'](T2_[A-Z0-9_]+)["\']')
SET = re.compile(r'^\s*export\s+(T2_[A-Z0-9_]+)=', re.M)
# Verdict words, not sentiment: each one marks a line where a document decided
# something about this flag. The line is printed verbatim so the reader judges.
VERDICT = re.compile(r"(폐기|철회|기각|보류|미구현|재설계|등재|승격|GO 조건|GO|검증|PASS|실증|arm 전용|기본 OFF)")


def scan_code():
    reads, sets = collections.defaultdict(set), collections.defaultdict(set)
    for dirpath, _dirs, files in os.walk(HERE):
        if os.sep + "a2" in dirpath:
            continue
        for fn in files:
            if not (fn.endswith(".py") or fn.endswith(".sh")):
                continue
            path = os.path.join(dirpath, fn)
            rel = os.path.relpath(path, HERE)
            try:
                txt = open(path, encoding="utf-8").read()
            except Exception:
                continue
            # An instrument reading a flag is not evidence the lever is live —
            # replays and tests set flags themselves to compare arms.
            instrument = fn.startswith("test_") or (fn.startswith("x") and fn[1:3].isdigit())
            for m in READ.finditer(txt):
                reads[m.group(1)].add((rel, "instrument" if instrument else "code"))
            for m in SET.finditer(txt):
                sets[m.group(1)].add(rel)
    return reads, sets


def doc_verdict(flag, blobs):
    for name, blob in blobs:
        for ln in blob.splitlines():
            if flag in ln and VERDICT.search(ln):
                return name, " ".join(ln.split())[:180]
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unset", action="store_true",
                    help="only flags that no driver sets at all")
    A = ap.parse_args()

    reads, sets = scan_code()
    blobs = []
    for p in sorted(glob.glob(os.path.join(DOCS, "*.md")), key=os.path.getmtime, reverse=True):
        try:
            blobs.append((os.path.basename(p), open(p, encoding="utf-8").read()))
        except Exception:
            pass

    on_stack = {f for f, w in sets.items() if STACK in w}
    prod = {f for f, w in reads.items() if any(role == "code" for _n, role in w)}
    print("go_stack이 세우는 플래그 %d종 · 프로덕션 코드가 읽는 플래그 %d종\n"
          % (len(on_stack), len(prod)))

    unset = sorted(f for f in prod if not sets.get(f))
    arm_only = sorted(f for f in prod if sets.get(f) and f not in on_stack)

    print("── ① 어떤 드라이버도 세우지 않는다 (%d종) — 라이브 발화 원천 불가 ──" % len(unset))
    for f in unset:
        where = sorted(n for n, role in reads[f] if role == "code")
        name, ln = doc_verdict(f, blobs)
        print("  %-30s %s" % (f, ", ".join(where[:2])))
        print("      %s" % (("%s :: %s" % (name[:38], ln)) if ln else "(판정 문장 없음 — 문서에 근거가 없다)"))
    if A.unset:
        return
    print("\n── ② arm 스크립트만 세운다 (%d종) — 의도적 arm일 수 있다 ──" % len(arm_only))
    for f in arm_only:
        print("  %-30s set_by=%s" % (f, ", ".join(sorted(sets[f])[:3])))


if __name__ == "__main__":
    main()
