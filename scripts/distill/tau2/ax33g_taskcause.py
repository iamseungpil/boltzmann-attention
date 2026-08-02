"""Per-task, per-step cause extraction for AX33 run-g.

For every simulation this locates, for each gold action the run failed to match,
whether the agent never tried that tool at all, tried it with different arguments
(and which argument differs), or tried it and the engine turned it away. It then
prints the surrounding steps so the cause can be read rather than inferred.

No aggregation. One block per task, both trials side by side.

Usage:
    python ax33g_taskcause.py                 # every task
    python ax33g_taskcause.py --task task_005
    python ax33g_taskcause.py --only-failing
"""

import argparse
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


def norm(args):
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            return {"_raw": args}
    return args if isinstance(args, dict) else {"_raw": args}


def emitted(sim):
    """[(msg_index, tool_name, args_dict, say_text_before)]"""
    out = []
    say = ""
    for i, m in enumerate(sim.get("messages") or []):
        if m.get("role") != "assistant":
            continue
        if isinstance(m.get("content"), str) and m["content"].strip():
            say = m["content"].strip()
        for tc in m.get("tool_calls") or []:
            name = tc.get("name") or (tc.get("function") or {}).get("name")
            a = tc.get("arguments")
            if a is None:
                a = (tc.get("function") or {}).get("arguments")
            out.append((i, name, norm(a), say))
    return out


def result_after(sim, idx):
    for m in (sim.get("messages") or [])[idx + 1 : idx + 3]:
        if m.get("role") == "tool" and isinstance(m.get("content"), str):
            return m["content"]
    return ""


def argdiff(gold, got):
    keys = set(gold) | set(got)
    return [
        (k, gold.get(k, "<absent>"), got.get(k, "<absent>"))
        for k in sorted(keys)
        if gold.get(k) != got.get(k)
    ]


def one(sim, width):
    ri = sim.get("reward_info") or {}
    calls = emitted(sim)
    print(
        f"\n  -- trial {sim.get('trial')}  reward={ri.get('reward')}  "
        f"term={sim.get('termination_reason')}  calls={len(calls)}  "
        f"basis={ri.get('reward_basis')}  db={(ri.get('db_check') or {}).get('db_match')}"
    )
    misses = [c for c in (ri.get("action_checks") or []) if not c.get("action_match")]
    if not misses:
        print("     no missed gold action")
    # A gold action carries a requestor. Only requestor=assistant is the agent's to
    # emit; requestor=user actions are executed by the customer after the agent hands
    # the tool over, so "the agent never called it" is the expected shape there and
    # says nothing about the failure.
    for c in misses:
        g = c.get("action") or {}
        who = g.get("requestor")
        gname, gargs = g.get("name"), norm(g.get("arguments") or {})
        if who != "assistant":
            print(f"     [requestor={who}] {gname} — agent-side non-call is expected; "
                  f"the agent's obligation is the hand-off, not the call")
            continue
        same = [e for e in calls if e[1] == gname]
        print(f"     GOLD MISS  {gname}  {json.dumps(gargs, ensure_ascii=False)[:width]}")
        if not same:
            print("        -> NEVER ATTEMPTED this tool")
            continue
        best, bestd = None, None
        for e in same:
            d = argdiff(gargs, e[2])
            if bestd is None or len(d) < len(bestd):
                best, bestd = e, d
        print(f"        -> attempted {len(same)}x; closest at msg{best[0]}")
        for k, gv, av in bestd[:6]:
            print(f"           arg '{k}': gold={json.dumps(gv, ensure_ascii=False)[:120]}")
            print(f"           {'':>{len(k) + 6}} got ={json.dumps(av, ensure_ascii=False)[:120]}")
        res = result_after(sim, best[0]).strip().replace("\n", " ")
        if res:
            print(f"           engine said: {res[:width]}")
        if best[3]:
            print(f"           agent had said: {' '.join(best[3].split())[:width]}")
    tail = [m for m in (sim.get("messages") or []) if m.get("role") == "assistant"]
    if tail and isinstance(tail[-1].get("content"), str) and tail[-1]["content"]:
        print(f"     last words: {' '.join(tail[-1]['content'].split())[:width]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="20260803g")
    ap.add_argument("--task")
    ap.add_argument("--only-failing", action="store_true")
    ap.add_argument("--width", type=int, default=320)
    args = ap.parse_args()

    sims = load(args.tag)
    by_task = {}
    for s in sims:
        by_task.setdefault(s.get("task_id"), []).append(s)

    for task in sorted(by_task):
        if args.task and task != args.task:
            continue
        pair = sorted(by_task[task], key=lambda s: s.get("trial") or 0)
        rewards = [(s.get("reward_info") or {}).get("reward") for s in pair]
        if args.only_failing and all(r == 1.0 for r in rewards):
            continue
        kind = (
            "BOTH PASS" if all(r == 1.0 for r in rewards)
            else "BOTH FAIL" if all(r != 1.0 for r in rewards)
            else "SPLIT"
        )
        print("=" * 96)
        print(f"## {task}   {kind}   rewards={rewards}")
        for s in pair:
            one(s, args.width)


if __name__ == "__main__":
    main()
