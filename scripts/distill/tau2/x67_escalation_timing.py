"""When the agent hands the case to a human, how much of the case had it done first?

Reading task_078 showed the shape: the agent offers a transfer in its *first* reply, the
simulated customer accepts, two searches come back irrelevant, and it escalates — with
22 gold actions untouched. The 2026-08-04 arc already named this (C292: the transfer got
cheaper, so it got bought more often; hand-off avoidance is the dominant cause class).
This measures how far it goes on the full task set rather than the 32-task front.

Per simulation:

  transferred        `transfer_to_human_agents` was called
  gold-wanted        gold has a transfer action too — escalating was the right move
  first-turn offer   the agent's opening reply already proposes the transfer, before it
                     has read anything about the customer's case
  work before        how many non-search tool calls happened before the transfer

The number that matters is the last one: an escalation after the work is a hand-off, an
escalation before it is an exit.
"""

import argparse
import collections
import glob
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from x50_says_not_does import ARMS, SIM  # noqa: E402

TRANSFER = {"transfer_to_human_agents", "request_human_agent_transfer",
            "initial_transfer_to_human_agent"}
SEARCH_PREFIX = ("KB_search", "get_current_time", "think")


def load(pattern):
    out = []
    for p in sorted(glob.glob(f"{SIM}/{pattern}.results.json.gz")):
        print(f"  read {os.path.basename(p)}")
        out.extend(json.load(gzip.open(p, "rt", encoding="utf-8")).get("simulations") or [])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="N97,A,B4")
    args = ap.parse_args()

    for a in args.arms.split(","):
        sims = load(ARMS[a])
        n = collections.Counter()
        work_before = []
        rows = []
        for s in sims:
            passed = (s.get("reward_info") or {}).get("reward") == 1.0
            n["pass" if passed else "fail"] += 1
            gold_transfer = any(
                (c.get("action") or {}).get("name") in TRANSFER
                for c in (s.get("reward_info") or {}).get("action_checks") or [])

            first_offer = False
            xfer_at = None
            work = 0
            for i, m in enumerate(s.get("messages") or []):
                if m.get("role") != "assistant":
                    continue
                if xfer_at is None and "TRANSFER NOTICE" in (m.get("content") or ""):
                    # the opening greeting is message 0; an offer in the next assistant
                    # turn means it was proposed before any account work
                    first_offer = first_offer or work == 0
                for tc in m.get("tool_calls") or []:
                    name = tc.get("name") or (tc.get("function") or {}).get("name") or ""
                    if name in TRANSFER:
                        xfer_at = xfer_at if xfer_at is not None else work
                    elif not name.startswith(SEARCH_PREFIX):
                        work += 1

            if xfer_at is not None:
                n["transferred"] += 1
                n["transferred·gold wanted it" if gold_transfer
                  else "transferred·gold did NOT"] += 1
                if not passed:
                    work_before.append(xfer_at)
                if first_offer:
                    n["offered in first reply"] += 1
                if xfer_at == 0:
                    n["transferred with 0 case actions"] += 1
                    if not gold_transfer:
                        rows.append((s["task_id"], s.get("trial"),
                                     len((s.get("reward_info") or {}).get("action_checks") or [])))
            elif gold_transfer:
                n["gold wanted transfer·did not"] += 1

        print(f"\n=== {a}: escalation ===")
        tot = n["pass"] + n["fail"]
        for k in ["transferred", "transferred·gold wanted it", "transferred·gold did NOT",
                  "offered in first reply", "transferred with 0 case actions",
                  "gold wanted transfer·did not"]:
            print(f"  {k:34} {n[k]:3} / {tot}")
        if work_before:
            work_before.sort()
            mid = work_before[len(work_before) // 2]
            print(f"  case actions before escalating (failed sims): median {mid}, "
                  f"zero in {work_before.count(0)}/{len(work_before)}")
        if rows:
            print("  bailed with nothing done, and gold never asked for a transfer:")
            for t, tr, g in sorted(rows):
                print(f"    {t}/t{tr}  gold actions abandoned = {g}")
        print()


if __name__ == "__main__":
    main()
