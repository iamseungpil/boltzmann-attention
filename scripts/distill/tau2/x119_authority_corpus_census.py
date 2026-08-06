# -*- coding: utf-8 -*-
"""Which arguments may we declare the records authoritative for?

The rule is settled (user instruction, 2026-08-06): when the customer talks about record
contents or policy, they are not the authority. The mechanism is settled too — `corpus_roles`
already exists per argument and the engine already reads it. What is not settled is *where* we
may declare it, and getting that wrong blocks a gold call.

So this counts the only thing that decides it. For every gold-matched call in the sweep, take
each argument a write spec grounds, and ask where its value actually appears in that trajectory:
in a tool output, or only in what the customer said. An argument whose gold value ever lives
only in customer speech must keep the default corpus — declaring `["tool"]` there would refuse
a call the task wanted.

Pass condition per argument: **user-only occurrences among gold-matched calls = 0**.

usage: x119_authority_corpus_census.py [--tag 20260806]
"""

import collections
import glob
import gzip
import io
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
TAU2 = os.environ.get("GO_TAU2", "/home/woori/scratch/tau2-bench")
TAG = "20260806"
if "--tag" in sys.argv:
    TAG = sys.argv[sys.argv.index("--tag") + 1]

A2 = json.load(io.open(os.path.join(HERE, "a2", "banking_knowledge.specific.json"),
                       encoding="utf-8"))
SPECS = A2.get("write_arg_grounding") or []
WANT = set()
for sp in SPECS:
    for ga in (sp.get("grounded_args") or []):
        WANT.add(ga)
print("write_arg_grounding이 접지하는 인자 %d개: %s" % (len(WANT), sorted(WANT)))


def jopen(p):
    op = gzip.open if p.endswith(".gz") else io.open
    with op(p, "rt", encoding="utf-8", errors="replace") as fh:
        return json.load(fh)


sims = []
for base in (os.path.join(TAU2, "data", "simulations"),
             os.path.join(REPO, "reports", "facet_rft_2026", "sim_results")):
    for pat in (os.path.join(base, "bank_*" + TAG + "*", "results.json"),
                os.path.join(base, "bank_*" + TAG + "*.results.json.gz")):
        for p in sorted(glob.glob(pat)):
            try:
                d = jopen(p)
            except Exception:
                continue
            sims.extend(d.get("simulations") or [])
print("sim %d개" % len(sims))

tool_only = collections.Counter()
user_only = collections.Counter()
both = collections.Counter()
absent = collections.Counter()
examples = collections.defaultdict(list)

for s in sims:
    msgs = s.get("messages") or []
    tool_text = " \n".join(str(m.get("content") or "") for m in msgs
                           if m.get("role") == "tool" and not m.get("error"))
    user_text = " \n".join(str(m.get("content") or "") for m in msgs
                           if m.get("role") == "user")
    for a in ((s.get("reward_info") or {}).get("action_checks") or []):
        if not a.get("action_match"):
            continue                       # gold이 실제로 맞다고 본 호출만 센다
        args = ((a.get("action") or {}).get("arguments") or {})
        for k, v in args.items():
            if k not in WANT or not isinstance(v, (str, int, float)):
                continue
            vs = str(v)
            if not vs or len(vs) < 3:
                continue
            it, iu = (vs in tool_text), (vs in user_text)
            if it and iu:
                both[k] += 1
            elif it:
                tool_only[k] += 1
            elif iu:
                user_only[k] += 1
                if len(examples[k]) < 3:
                    examples[k].append((a.get("action") or {}).get("name"))
            else:
                absent[k] += 1

print("\n%-26s %7s %7s %8s %7s   판정" % ("인자", "tool만", "양쪽", "user만", "부재"))
verdict = {}
for k in sorted(WANT):
    n = tool_only[k] + both[k] + user_only[k] + absent[k]
    if not n:
        print("%-26s %7s %7s %8s %7s   (gold 관측 0 — 판정 불가)" % (k, "-", "-", "-", "-"))
        verdict[k] = None
        continue
    ok = user_only[k] == 0
    verdict[k] = ok
    print("%-26s %7d %7d %8d %7d   %s"
          % (k, tool_only[k], both[k], user_only[k], absent[k],
             "선언 가능 [tool]" if ok else "★선언 금지 — user만 %s" % (examples[k] or "")))

safe = [k for k, v in verdict.items() if v]
print("\n선언 가능(레코드 권위): %s" % (sorted(safe) or "없음"))
print("선언 금지(손님도 권위): %s" % sorted([k for k, v in verdict.items() if v is False]))
print("판정 불가(관측 0): %s" % sorted([k for k, v in verdict.items() if v is None]))
