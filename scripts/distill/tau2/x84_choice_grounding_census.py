# -*- coding: utf-8 -*-
"""Is the account class the agent chooses ever grounded in what it actually read?

The account-opening cluster fails 47 times out of 52 gold demands, split about evenly
between choosing the wrong class and never calling the tool at all. A deterministic
lever needs a closed test, and the candidate one is: the class name must appear in some
document the conversation retrieved before the call. That is only enforceable if gold's
own choice always passes it — otherwise the gate would block correct answers.

  gold-grounded      gold's class appears in retrieved text before the gold call
  gold-ungrounded    it does not → a deny gate would be wrong
  agent-ungrounded   the class the agent actually used was not in retrieved text → gain

  usage: x84_choice_grounding_census.py [tool] [arg]
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

from x50_says_not_does import ARMS, SIM  # noqa: E402

TOOL = sys.argv[1] if len(sys.argv) > 1 else "open_bank_account_4821"
ARG = sys.argv[2] if len(sys.argv) > 2 else "account_class"


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


tally = collections.Counter()
examples = collections.defaultdict(list)
for p in sorted(glob.glob(os.path.join(SIM, ARMS["N97B"] + "*.results.json.gz"))):
    with gzip.open(p, "rt", encoding="utf-8") as f:
        d = json.load(f)
    for s in (d.get("simulations") if isinstance(d, dict) else d):
        gold_vals = set()
        for c in (s.get("reward_info") or {}).get("action_checks") or []:
            g = c.get("action") or {}
            nm, ga = inner(g.get("arguments"))
            if nm == TOOL and ga.get(ARG):
                gold_vals.add(ga[ARG])
        if not gold_vals:
            continue
        seen = []
        agent_vals = []
        for m in s.get("messages") or []:
            for tc in m.get("tool_calls") or []:
                nm, a = inner(tc.get("arguments"))
                if (nm or tc.get("name")) == TOOL and a.get(ARG):
                    agent_vals.append((a[ARG], " ".join(seen)))
            if m.get("role") == "tool":
                seen.append(str(m.get("content") or ""))
        blob_all = " ".join(seen)
        for gv in gold_vals:
            k = "gold-grounded" if gv in blob_all else "gold-ungrounded"
            tally[k] += 1
            if k == "gold-ungrounded" and len(examples[k]) < 8:
                examples[k].append("%s %s" % (s["task_id"], gv))
        for av, blob_at_call in agent_vals:
            k = "agent-grounded" if av in blob_at_call else "agent-ungrounded"
            tally[k] += 1
            if k == "agent-ungrounded" and len(examples[k]) < 8:
                examples[k].append("%s %s" % (s["task_id"], av))

print("도구 %s · 인자 %s\n" % (TOOL, ARG))
for k in ("gold-grounded", "gold-ungrounded", "agent-grounded", "agent-ungrounded"):
    print("  %-18s %d" % (k, tally[k]))
for k in ("gold-ungrounded", "agent-ungrounded"):
    if examples[k]:
        print("\n  [%s]" % k)
        for e in examples[k]:
            print("     %s" % e)
