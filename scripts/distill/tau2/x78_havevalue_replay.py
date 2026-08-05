"""Why did the have-value nudge stay silent on the trajectory it was written for?

`T2_HAVE_VALUE=1` is on the stack, the A2 spec names this exact write, this exact
producer and these exact re-ask phrases, and the marker `[T2_HAVE_VALUE]` appears
zero times in all 194 simulations of the rerun. task_040/t0 is the case that
should have fired it: the producer really executed ("Executed: get_card_last_4_digits
… 1652"), and the agent then asked the customer to re-fetch that value six times.

This replays the predicate itself — no model, no environment — over every assistant
turn of a persisted trajectory, and reports for each turn which of its four
conditions held. A predicate that returns None on the turn it was written for is a
defect in the predicate; a predicate that would have fired means the call site never
reached it, and the fix is in the chain above, not in the predicate.

  usage: x78_havevalue_replay.py --task task_040 --trial 0 [--arm N97B]
"""

import argparse
import glob
import gzip
import json
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="task_040")
ap.add_argument("--trial", type=int, default=0)
ap.add_argument("--arm", default="N97B")
A = ap.parse_args()

from x50_says_not_does import ARMS, SIM  # noqa: E402
import t2_gate_patch as GP  # noqa: E402

A2 = json.load(open(os.path.join(HERE, "a2", "banking_knowledge.specific.json"), encoding="utf-8"))


def find_specs(o, key="have_value_reask"):
    if isinstance(o, dict):
        if key in o:
            return o[key]
        for v in o.values():
            r = find_specs(v, key)
            if r:
                return r
    elif isinstance(o, list):
        for v in o:
            r = find_specs(v, key)
            if r:
                return r
    return None


SPECS = find_specs(A2)
print("A2 have_value_reask specs: %d" % len(SPECS or []))
for sp in SPECS or []:
    print("  write=%s producer=%s arg=%s" % (sp.get("write"), sp.get("producer"), sp.get("arg")))


def load(task, trial):
    for p in sorted(glob.glob(os.path.join(SIM, ARMS[A.arm] + "*.results.json.gz"))):
        with gzip.open(p, "rt", encoding="utf-8") as f:
            d = json.load(f)
        for s in (d.get("simulations") if isinstance(d, dict) else d):
            if s.get("task_id") == task and s.get("trial") == trial:
                return s
    return None


sim = load(A.task, A.trial)
if sim is None:
    print("no such sim"); sys.exit(1)


def ns(m):
    """The predicate reads messages by attribute, the persisted form is a dict."""
    o = types.SimpleNamespace(**m)
    o.tool_calls = [types.SimpleNamespace(**tc) for tc in (m.get("tool_calls") or [])] or None
    return o


msgs = [ns(m) for m in sim["messages"]]
print("\n%s/t%s · %d messages\n" % (A.task, A.trial, len(msgs)))

fired = 0
for i, m in enumerate(msgs):
    if getattr(m, "role", None) != "assistant":
        continue
    prior = msgs[:i]                     # the predicate only sees committed history
    out = GP._have_value_reask_fb(m, prior, SPECS)
    # Re-derive each condition so a None can be attributed rather than guessed.
    detail = []
    for sp in SPECS or []:
        W, producer = sp.get("write"), sp.get("producer")
        marker = sp.get("producer_marker") or producer
        sigs = [str(s).lower() for s in (sp.get("reask_signals") or [])]
        cur = {GP._eff_tool_name(tc) for tc in (getattr(m, "tool_calls", None) or [])}
        txt = str(getattr(m, "content", "") or "").lower()
        outs = GP._producer_outputs(prior, marker)
        detail.append({
            "W_not_called": W not in cur,
            "producer_out": len(outs),
            "prior_reask": any(getattr(x, "role", None) == "assistant"
                               and any(s in str(getattr(x, "content", "") or "").lower() for s in sigs)
                               for x in prior),
            "now_reask": any(s in txt for s in sigs) or (producer in cur),
        })
    if out or any(d["producer_out"] and d["now_reask"] for d in detail):
        fired += 1 if out else 0
        print("msg %3d  %s" % (i, "★FIRES" if out else "조건 일부만"))
        print("        %s" % json.dumps(detail, ensure_ascii=False))
        if out:
            print("        → %s" % " ".join(out.split())[:200])

print("\n총 발화 %d회 (라이브 로그의 실측은 0회)" % fired)
