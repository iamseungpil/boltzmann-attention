# -*- coding: utf-8 -*-
"""Rows the engine settled, that nobody then submitted — and whether saying so would ever be wrong.

`task_019` computed four discrepancies and submitted two: one row the engine had settled
was dropped after the customer argued about it in prose. [[21]] says the agent has to be
right whatever the simulated customer does, and the closed half of that is countable —
the engine's own output named the row, and the call history says whether it was ever
submitted. Nothing here reads the customer's sentences.

  settled     transaction ids the discrepancy tool returned
  submitted   ids that a dispute call actually carried
  dropped     settled minus submitted

The gate is the usual one, in its surfacing form: if gold itself leaves settled rows
unsubmitted, then "you dropped a row" is not a defect and must not be said as one.

Also reports the second question in the same pass — whether any protocol was used
outside the time window its document declares — so that a lever with no target is not
built. Free: persisted trajectories.

  usage: x94_withdrawn_row_census.py [arm]
"""

import collections
import glob
import gzip
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from x50_says_not_does import ARMS, SIM  # noqa: E402

ARM = sys.argv[1] if len(sys.argv) > 1 else "N97B"
SETTLE_TOOL = "get_reward_discrepancies"
SUBMIT_TOOL = "submit_cash_back_dispute_0589"
IDPAT = re.compile(r"\btxn_[0-9a-f_]+\b")


def inner(a):
    a = a if isinstance(a, dict) else {}
    return (a.get("agent_tool_name") or a.get("discoverable_tool_name")
            or a.get("user_tool_name"))


def args_of(x):
    a = x.get("arguments") if isinstance(x, dict) else None
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except Exception:
            a = {}
    return a if isinstance(a, dict) else {}


def submitted_ids(calls):
    out = set()
    for tc in calls:
        if (inner(args_of(tc)) or tc.get("name")) != SUBMIT_TOOL:
            continue
        a = args_of(tc)
        sub = a.get("arguments")
        if isinstance(sub, str):
            try:
                sub = json.loads(sub)
            except Exception:
                sub = {}
        for src in (a, sub if isinstance(sub, dict) else {}):
            for v in src.values():
                if isinstance(v, str) and v.startswith("txn_"):
                    out.add(v)
    return out


tally = collections.Counter()
ex, gex = [], []
times = collections.Counter()
for p in sorted(glob.glob(os.path.join(SIM, ARMS[ARM] + "*.results.json.gz"))):
    with gzip.open(p, "rt", encoding="utf-8") as f:
        d = json.load(f)
    for s in (d.get("simulations") if isinstance(d, dict) else d):
        tally["sim"] += 1
        sid = "%s/t%s" % (s.get("task_id"), s.get("trial"))
        settled, pend, calls = set(), {}, []
        for m in s.get("messages") or []:
            for tc in (m.get("tool_calls") or []):
                pend[tc.get("id")] = inner(args_of(tc)) or tc.get("name")
                calls.append(tc)
            if m.get("role") == "tool":
                nm = pend.get(m.get("id") or m.get("tool_call_id"))
                c = str(m.get("content") or "")
                if nm == SETTLE_TOOL and not c.lstrip().startswith("Error:"):
                    settled |= set(IDPAT.findall(c))
                # 시간 창 표적 확인용: 대화가 본 현재 시각
                for mt in re.findall(r"\b(20\d\d-\d\d-\d\d)\b", c):
                    times[mt] += 1
        if not settled:
            continue
        tally["settled_sim"] += 1
        got = submitted_ids(calls)
        drop = settled - got
        if drop:
            tally["dropped_sim"] += 1
            tally["dropped_rows"] += len(drop)
            if len(ex) < 10:
                ex.append((sid, len(settled), len(got), sorted(drop)[:3]))
        gold_ids = submitted_ids([(c.get("action") or {})
                                  for c in ((s.get("reward_info") or {}).get("action_checks") or [])])
        if gold_ids and (settled - gold_ids):
            tally["gold_drops"] += 1
            if len(gex) < 8:
                gex.append((sid, sorted(settled - gold_ids)[:3]))

print("arm %s · sim %d · 엔진이 행을 낸 sim %d" % (ARM, tally["sim"], tally["settled_sim"]))
print("  **확정된 행이 제출되지 않은 sim %d · 행 %d개**"
      % (tally["dropped_sim"], tally["dropped_rows"]))
for sid, ns, ng, dr in ex:
    print("    %-16s 확정 %2d · 제출 %2d · 누락 %s" % (sid, ns, ng, dr))
print()
print("  ★게이트 — **gold 자신도 확정 행을 남기는 sim = %d**" % tally["gold_drops"])
for sid, dr in gex:
    print("    %-16s %s" % (sid, dr))
print("  판정: %s" % ("표면화 등재 가능" if tally["dropped_sim"] and not tally["gold_drops"]
                      else "**주의** — gold도 남긴다면 '누락'이라 말할 수 없다"))
print()
print("  [G1 표적 확인] 대화에 등장한 날짜 상위: %s" % times.most_common(6))
