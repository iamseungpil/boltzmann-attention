"""Did the information the run needed ever reach the model's context?

Before investing in a richer retrieval structure, this bounds how much retrieval
could buy at all. For every gold action a run missed, it takes the action's key
token — the discoverable tool's name, or the enum value the action required — and
asks whether that token ever appeared in a KB_search result, or anywhere in the
context, before the run ended.

  RETRIEVED  the token came back from a KB search
  IN_CONTEXT the token appeared somewhere the model could read (engine message,
             tool listing, its own earlier text) even if no search returned it
  ABSENT     the token never appeared; retrieval genuinely could not have helped
             the model act on it

A miss whose token was RETRIEVED is not a retrieval failure. A richer index
cannot recover it.
"""

import argparse
import collections
import glob
import gzip
import json

SIM_DIR = (
    "/home/woori/workspace_common/boltzmann-attention-pi/reports/facet_rft_2026/sim_results"
)


def load(tag):
    sims = []
    for path in sorted(glob.glob(f"{SIM_DIR}/bank_ax33n_gpu*_{tag}.results.json.gz")):
        sims.extend(json.load(gzip.open(path, "rt", encoding="utf-8")).get("simulations") or [])
    return sims


def norm(a):
    if isinstance(a, str):
        try:
            return json.loads(a)
        except Exception:
            return {}
    return a if isinstance(a, dict) else {}


def key_token(action):
    """The one string the run had to know to perform this action."""
    args = norm(action.get("arguments") or {})
    for field in ("discoverable_tool_name", "agent_tool_name"):
        if args.get(field):
            return args[field], field
    if action.get("name") == "transfer_to_human_agents" and args.get("reason"):
        return args["reason"], "reason"
    if action.get("name") == "apply_for_credit_card" and args.get("card_type"):
        return args["card_type"], "card_type"
    return action.get("name"), "tool_name"


def channels(sim):
    """(text returned by KB searches, all other text the model could read)."""
    retrieved, other = [], []
    msgs = sim.get("messages") or []
    pending_kb = False
    for m in msgs:
        if m.get("role") == "assistant":
            names = [
                (tc.get("name") or (tc.get("function") or {}).get("name"))
                for tc in m.get("tool_calls") or []
            ]
            pending_kb = any(n and n.startswith("KB_search") for n in names)
            if isinstance(m.get("content"), str):
                other.append(m["content"])
        elif m.get("role") == "tool" and isinstance(m.get("content"), str):
            (retrieved if pending_kb else other).append(m["content"])
            pending_kb = False
        elif isinstance(m.get("content"), str):
            other.append(m["content"])
    return "\n".join(retrieved), "\n".join(other)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="20260803g")
    args = ap.parse_args()

    tally = collections.Counter()
    per_task = collections.defaultdict(collections.Counter)
    rows = []

    for sim in load(args.tag):
        ri = sim.get("reward_info") or {}
        if (ri.get("reward") or 0.0) == 1.0:
            continue
        kb, rest = channels(sim)
        misses = [c for c in (ri.get("action_checks") or []) if not c.get("action_match")]
        seen = set()
        for c in misses:
            a = c.get("action") or {}
            tok, field = key_token(a)
            if not tok or (a.get("name"), tok) in seen:
                continue
            seen.add((a.get("name"), tok))
            if tok in kb:
                verdict = "RETRIEVED"
            elif tok in rest:
                verdict = "IN_CONTEXT"
            else:
                verdict = "ABSENT"
            tally[verdict] += 1
            per_task[sim.get("task_id")][verdict] += 1
            rows.append(
                (sim.get("task_id"), sim.get("trial"), a.get("requestor"), a.get("name"),
                 field, str(tok)[:40], verdict)
            )

    rows.sort()
    print(f"{'task':10s} {'t':>2} {'who':>9} {'gold action':32s} {'token':42s} verdict")
    for t, tr, who, name, field, tok, v in rows:
        print(f"{t:10s} {tr:2d} {str(who):>9} {str(name):32s} {field}={tok:38s} {v}")

    total = sum(tally.values())
    print(f"\n=== missed gold actions: {total} ===")
    for k in ("RETRIEVED", "IN_CONTEXT", "ABSENT"):
        n = tally[k]
        print(f"  {k:11s} {n:4d}  ({n / total:.1%})" if total else f"  {k}: 0")
    print("\n=== per task ===")
    for t in sorted(per_task):
        c = per_task[t]
        print(f"  {t:10s} " + " ".join(f"{k}={c[k]}" for k in ("RETRIEVED", "IN_CONTEXT", "ABSENT") if c[k]))


if __name__ == "__main__":
    main()
