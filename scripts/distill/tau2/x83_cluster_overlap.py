# -*- coding: utf-8 -*-
"""Do the card-closure tasks share one procedure, or ten of them?

The read-through said 043 through 054 fail in the same three places, which is an
argument for declaring one procedure rather than fixing tasks. That argument is only
worth acting on if the tasks really do demand the same steps, so this counts it: the
gold tool set per task, what fraction of each task's gold is inside the shared core,
and what each task asks for that no other does.

If the shared core covers most of every task, then declaring it once covers the
cluster and picking three tasks is a sampling decision, not a scoping one. If tasks
carry large private remainders, fixing three fixes three.

  usage: x83_cluster_overlap.py [task_043 task_044 ...]
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

DEFAULT = ["task_043", "task_044", "task_046", "task_047", "task_048",
           "task_049", "task_050", "task_052", "task_053", "task_054"]
WANT = sys.argv[1:] or DEFAULT
WRAP = {"unlock_discoverable_agent_tool", "give_discoverable_user_tool",
        "call_discoverable_agent_tool", "call_discoverable_user_tool"}


def inner(a):
    a = a if isinstance(a, dict) else {}
    return (a.get("agent_tool_name") or a.get("discoverable_tool_name")
            or a.get("user_tool_name"))


gold = collections.defaultdict(set)
for p in sorted(glob.glob(os.path.join(SIM, ARMS["N97B"] + "*.results.json.gz"))):
    with gzip.open(p, "rt", encoding="utf-8") as f:
        d = json.load(f)
    for s in (d.get("simulations") if isinstance(d, dict) else d):
        if s["task_id"] not in WANT:
            continue
        for c in (s.get("reward_info") or {}).get("action_checks") or []:
            g = c.get("action") or {}
            nm = inner(g.get("arguments")) if g.get("name") in WRAP else g.get("name")
            if nm:
                gold[s["task_id"]].add(nm)

present = collections.Counter()
for t, tools in gold.items():
    for x in tools:
        present[x] += 1

n = len(gold)
core = {x for x, c in present.items() if c >= max(2, n // 2)}       # 과반 태스크가 요구
print("태스크 %d개 · 공유 코어(과반이 요구하는 도구) %d종\n" % (n, len(core)))
for x, c in present.most_common():
    print("  %-42s %d/%d %s" % (x, c, n, "★core" if x in core else ""))

print("\n태스크별: gold 중 코어가 덮는 비율 · 그 태스크만의 잔여")
for t in sorted(gold):
    tools = gold[t]
    cov = len(tools & core)
    private = sorted(x for x in tools if present[x] == 1)
    print("  %-10s %2d/%2d (%3.0f%%)  단독요구: %s"
          % (t, cov, len(tools), 100 * cov / max(1, len(tools)), ", ".join(private) or "-"))
