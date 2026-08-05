# -*- coding: utf-8 -*-
"""If the agent did exactly what gold does, would our own declarations stop it?

x80 asks whether a deny ever landed on a call gold wanted, over the calls the agent
actually made. That population has a hole: a step the agent always takes but gold never
does will never appear in it. `task_048` is that hole — the closure declaration requires
the eligibility check before the later steps, gold never calls it, and gold's own
trajectory is therefore denied twelve times. The measurement said zero because every
recorded run happened to call it.

So this replays gold's action sequence against the declaration with an empty history and
asks the question directly: walking gold, does anything block? A declaration that blocks
gold is wrong no matter how well it scores, because the trajectory we are trying to
produce is the one it forbids.

A hit is a stop. The fix has to come from the policy document, never from this output —
reading the shape of gold and copying it into A2 is exactly what [[23]] forbids. This
tells you *that* a declaration is wrong, not *what* to write instead.

Free: persisted trajectories and the declaration.

  usage: x91_gold_path_gate.py [arm]
"""

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

import t2_procedure as P                 # noqa: E402
from x50_says_not_does import ARMS, SIM  # noqa: E402

ARM = sys.argv[1] if len(sys.argv) > 1 else "N97B"


def a2_procedures(domain="banking_knowledge"):
    import gate_interpreter as GI
    return (GI.load_domain_a2(domain) or {}).get("procedures") or []


def inner(a):
    a = a if isinstance(a, dict) else {}
    return (a.get("agent_tool_name") or a.get("discoverable_tool_name")
            or a.get("user_tool_name"))


def gold_sequence(sim):
    out = []
    for c in (sim.get("reward_info") or {}).get("action_checks") or []:
        act = c.get("action") or {}
        nm = inner(act.get("arguments")) or act.get("name")
        if nm:
            out.append((nm, act.get("arguments") if isinstance(act.get("arguments"), dict) else {}))
    return out


procs = a2_procedures()
tally = collections.Counter()
hits = collections.Counter()
detail = {}
for p in sorted(glob.glob(os.path.join(SIM, ARMS[ARM] + "*.results.json.gz"))):
    with gzip.open(p, "rt", encoding="utf-8") as f:
        d = json.load(f)
    for s in (d.get("simulations") if isinstance(d, dict) else d):
        tid = s.get("task_id")
        if tid in detail:
            continue                      # 태스크당 1회면 충분하다(gold은 trial 불변)
        seq = gold_sequence(s)
        if not seq:
            continue
        tally["task"] += 1
        ex, blocked = [], []
        for nm, args in seq:
            dc = P.decide(procs, nm, args, ex)
            if dc.get("verdict") == "deny":
                blocked.append((nm, tuple(dc.get("missing") or []) or ("prohibited",)))
            ex.append(nm)
        detail[tid] = blocked
        if blocked:
            tally["blocked_task"] += 1
            for nm, miss in blocked:
                hits[(nm, miss)] += 1

print("arm %s · gold 경로를 가진 태스크 %d" % (ARM, tally["task"]))
print("  ★**gold이 막히는 태스크 = %d**" % tally["blocked_task"])
print()
for (nm, miss), c in hits.most_common(20):
    print("    %-46s missing=%-42s %d회" % (nm, ",".join(miss), c))
print()
worst = sorted(((len(v), k) for k, v in detail.items() if v), reverse=True)[:8]
for n, tid in worst:
    print("    %-12s %d곳" % (tid, n))
print()
print("  판정: %s" % ("통과 — 선언이 gold 경로를 막지 않는다" if not tally["blocked_task"]
                      else "**정지** — 선언이 gold이 밟는 길을 막는다. 정책 원문으로 되돌아가 "
                           "선언을 고칠 것(gold 모양을 베끼는 것은 [[23]] 위반)"))
