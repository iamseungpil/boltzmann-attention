# -*- coding: utf-8 -*-
"""Would the procedure check have blocked a call that gold actually wanted?

Enforcement is the expensive kind of lever: a wrong block costs a run that would
otherwise have succeeded, and this repo has shipped one before (C279 ②: two of the
guard's three live firings were mis-blocks). So before the flag goes on a live run,
the declaration is replayed over every persisted simulation and each would-be deny is
labelled against that simulation's own gold.

  legit     the blocked call was not in gold, or gold has the missing step before it
  OVERBLOCK gold contains this call and does NOT contain the step we call missing

An overblock count above zero is a stop, not a tuning parameter.

  usage: x80_procedure_overblock.py [--arm N97B] [--tag bank_qpmc_gpu*_20260805]
"""

import argparse
import collections
import glob
import gzip
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_procedure as P  # noqa: E402
from x50_says_not_does import ARMS, SIM  # noqa: E402

A2 = json.load(open(os.path.join(HERE, "a2", "banking_knowledge.specific.json"), encoding="utf-8"))
PROCS = A2.get("procedures") or []


def inner(a):
    a = a if isinstance(a, dict) else {}
    nm = a.get("agent_tool_name") or a.get("discoverable_tool_name") or a.get("user_tool_name")
    sub = a.get("arguments")
    if isinstance(sub, str):
        try:
            sub = json.loads(sub)
        except Exception:
            sub = None
    return (nm, sub if isinstance(sub, dict) else {}) if nm else (None, a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="N97B")
    ap.add_argument("--tag", default=None)
    A = ap.parse_args()
    pattern = A.tag or (ARMS[A.arm] + "*")
    files = sorted(glob.glob(os.path.join(SIM, pattern + ".results.json.gz")))
    if not files:
        files = sorted(glob.glob(os.path.join(SIM, pattern + "*.results.json.gz")))

    denies, over, by_task = [], [], collections.Counter()
    sims = 0
    for p in files:
        with gzip.open(p, "rt", encoding="utf-8") as f:
            d = json.load(f)
        for s in (d.get("simulations") if isinstance(d, dict) else d):
            sims += 1
            gold_tools = set()
            for c in (s.get("reward_info") or {}).get("action_checks") or []:
                g = c.get("action") or {}
                nm, _ = inner(g.get("arguments"))
                gold_tools.add(nm or g.get("name"))
            executed = set()
            for m in s.get("messages") or []:
                for tc in m.get("tool_calls") or []:
                    nm, args = inner(tc.get("arguments"))
                    name = nm or tc.get("name")
                    verdict = P.decide(PROCS, name, args, set(executed))
                    if verdict["verdict"] == "deny":
                        rec = (s["task_id"], s.get("trial"), name, tuple(verdict["missing"]))
                        denies.append(rec)
                        by_task[s["task_id"]] += 1
                        # gold이 이 호출을 원했고, 우리가 '누락'이라 부른 단계는 gold에 없다면 오차단
                        missing_tools = set()
                        for nid in verdict["missing"]:
                            _pp = P.find_procedure(PROCS, name, executed) or {}
                            for n in (_pp.get("nodes") or []):
                                if n.get("id") == nid:
                                    missing_tools |= set(P._tools_of(n))
                        if name in gold_tools and not (missing_tools & gold_tools):
                            over.append(rec)
                    executed.add(name)

    print("검사한 sim %d · deny 후보 %d건" % (sims, len(denies)))
    print("태스크별:", dict(by_task.most_common(10)))
    print("\n★OVERBLOCK(= gold이 원한 호출인데 우리가 요구한 선행이 gold에 없음): %d건" % len(over))
    for r in over[:10]:
        print("   ", r)
    print("\n예시 deny 10건:")
    for r in denies[:10]:
        print("   ", r)


if __name__ == "__main__":
    main()
