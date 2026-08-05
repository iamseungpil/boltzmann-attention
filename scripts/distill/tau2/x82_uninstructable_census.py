# -*- coding: utf-8 -*-
"""How often does the agent tell the customer to run something it has not handed over?

task_012 ends with the agent instructing a tool call the customer cannot make — the
named tool does not exist anywhere in the corpus (grep: 0) and nothing had been handed
over. The check is a nudge, not a block, so the risk is noise rather than lost runs;
what matters before wiring it into a run is how often it would speak and whether it
would speak on turns where a hand-over was already under way.

  fires        prose carries a declared hand-over-instruction token, nothing given yet
  gave-later   the same simulation hands something over afterwards → the nudge was early
               but not wrong (it says "give it first"), reported apart so the count is honest

  usage: x82_uninstructable_census.py [--arm N97B]
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

from x50_says_not_does import ARMS, SIM  # noqa: E402

A2 = json.load(open(os.path.join(HERE, "a2", "base", "shared.json"), encoding="utf-8"))
AX = A2.get("axis_notes") or {}
TOKENS = AX.get("user_exec_tokens") or []
MARK = AX.get("given_marker") or ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="N97B")
    A = ap.parse_args()
    print("선언 토큰 %s · 전달 마커 %r\n" % (TOKENS, MARK))

    fires, gave_later, sims_fired = 0, 0, set()
    per_task = collections.Counter()
    for p in sorted(glob.glob(os.path.join(SIM, ARMS[A.arm] + "*.results.json.gz"))):
        with gzip.open(p, "rt", encoding="utf-8") as f:
            d = json.load(f)
        for s in (d.get("simulations") if isinstance(d, dict) else d):
            msgs = s.get("messages") or []
            given_at = [i for i, m in enumerate(msgs)
                        if m.get("role") == "tool" and MARK in str(m.get("content") or "")]
            first_given = given_at[0] if given_at else None
            fired_here = False
            for i, m in enumerate(msgs):
                if m.get("role") != "assistant" or m.get("tool_calls"):
                    continue
                said = str(m.get("content") or "")
                if not any(t in said for t in TOKENS):
                    continue
                if first_given is not None and first_given < i:
                    continue                       # 이미 전달된 뒤 = 정당한 안내
                if fired_here:
                    continue                       # sim당 1회(엔진과 같은 규칙)
                fired_here = True
                fires += 1
                per_task[s["task_id"]] += 1
                sims_fired.add((s["task_id"], s.get("trial")))
                if first_given is not None:
                    gave_later += 1
    print("발화 %d회 · %d sim" % (fires, len(sims_fired)))
    print("그 중 나중에 전달이 실제로 일어난 경우: %d (넛지가 이르지만 틀리지 않음)" % gave_later)
    print("태스크 상위:", dict(per_task.most_common(12)))


if __name__ == "__main__":
    main()
