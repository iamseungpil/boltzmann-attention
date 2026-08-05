# -*- coding: utf-8 -*-
"""Is "hand over only a tool some retrieved policy names" a predicate we can enforce?

Handing a tool to the customer writes a row (`user_discoverable_tools`), and so does
their call of it (`user_discoverable_tool_calls`) — both are in the database the run is
scored against. So a handover gold did not ask for is not a stylistic detour, it is a
scored difference. In the cash-back flow the policy even says not to collect card
details, and the eight runs that did it all failed while eight of twelve that did not
passed.

The rule has to be closed and domain-general, so the candidate is: hand over X only if
the name X appears in something the conversation actually retrieved (a document, a tool
result). This measures that candidate before it is built:

  blocked-and-not-gold   the handover is unlicensed and gold did not want it   → gain
  blocked-but-gold       gold wanted this handover and no retrieved text names it → OVERBLOCK
  allowed-and-gold       licensed, gold wanted it                              → untouched
  allowed-not-gold       licensed, gold did not want it  → the rule cannot see this one

  usage: x81_give_grounding_census.py [--arm N97B]
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

GIVE = "give_discoverable_user_tool"


def inner(a):
    a = a if isinstance(a, dict) else {}
    return (a.get("agent_tool_name") or a.get("discoverable_tool_name")
            or a.get("user_tool_name"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="N97B")
    A = ap.parse_args()

    tally = collections.Counter()
    overblocks, gains = [], []
    for p in sorted(glob.glob(os.path.join(SIM, ARMS[A.arm] + "*.results.json.gz"))):
        with gzip.open(p, "rt", encoding="utf-8") as f:
            d = json.load(f)
        for s in (d.get("simulations") if isinstance(d, dict) else d):
            gold_gives = set()
            for c in (s.get("reward_info") or {}).get("action_checks") or []:
                g = c.get("action") or {}
                if g.get("name") == GIVE:
                    gold_gives.add(inner(g.get("arguments")))
            seen_text = []          # tool outputs so far — what the conversation retrieved
            for m in s.get("messages") or []:
                for tc in m.get("tool_calls") or []:
                    if tc.get("name") != GIVE:
                        continue
                    name = inner(tc.get("arguments"))
                    if not name:
                        continue
                    licensed = any(name in t for t in seen_text)
                    in_gold = name in gold_gives
                    key = ("allowed" if licensed else "blocked") + ("-gold" if in_gold else "-notgold")
                    tally[key] += 1
                    rec = (s["task_id"], s.get("trial"), name)
                    if not licensed and in_gold:
                        overblocks.append(rec)
                    elif not licensed and not in_gold:
                        gains.append(rec)
                if m.get("role") == "tool":
                    seen_text.append(str(m.get("content") or ""))

    tot = sum(tally.values())
    print("give 호출 %d건 (arm %s)\n" % (tot, A.arm))
    for k in ("blocked-notgold", "blocked-gold", "allowed-gold", "allowed-notgold"):
        print("  %-18s %4d  (%.0f%%)" % (k, tally[k], 100 * tally[k] / max(1, tot)))
    print("\n★OVERBLOCK(blocked-gold) = %d건" % len(overblocks))
    for r in sorted(set(overblocks))[:12]:
        print("   ", r)
    print("\n차단됐을 비-gold give(=이득 후보) 상위:")
    for r, n in collections.Counter(x[2] for x in gains).most_common(8):
        print("   %-34s %d" % (r, n))


if __name__ == "__main__":
    main()
