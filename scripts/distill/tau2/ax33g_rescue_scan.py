"""Who solved the task — the agent, or the customer?

Reading 018/020/021/003/034/035 by hand showed passing trials where the customer
named the step the agent was missing. [[21]] says a run the user-sim rescued is not
ours to count, so this measures the rescue rather than assuming it.

user-taught-tool event (decidable, no prose interpretation):
    a user message names tool X (X drawn from the env's observed tool set)
  & the agent has not called X at any earlier point
  & the agent calls X afterwards

Also counts, per simulation, the engine messages that the hand-off reading found
to be load-bearing, so the same close read extends to every failing run:
    GRANT_ERR   "has not been given to you by the agent"
    UNKNOWN     "Unknown discoverable tool"
    ARGERR      "Unexpected parameter" / "Missing required parameter"
    ASKED       agent emitted "Would you like to be transferred"
    EXECOK      a discoverable tool actually executed
    DUPWRITE    a repeated write whose result says the record already exists
"""

import argparse
import collections
import glob
import gzip
import json
import re

SIM_DIR = (
    "/home/woori/workspace_common/boltzmann-attention-pi/reports/facet_rft_2026/sim_results"
)


def load(tag):
    sims = []
    for path in sorted(glob.glob(f"{SIM_DIR}/bank_ax33n_gpu*_{tag}.results.json.gz")):
        sims.extend(json.load(gzip.open(path, "rt", encoding="utf-8")).get("simulations") or [])
    return sims


def tool_universe(sims):
    names = set()
    for s in sims:
        for m in s.get("messages") or []:
            for tc in m.get("tool_calls") or []:
                n = tc.get("name") or (tc.get("function") or {}).get("name")
                if n:
                    names.add(n)
            c = m.get("content")
            if isinstance(c, str):
                for hit in re.findall(r"\b([a-z][a-z0-9_]{6,})\b", c):
                    if hit.endswith(tuple("0123456789")) and "_" in hit:
                        names.add(hit)  # discoverable tools carry a numeric suffix
    return names


def scan(sim, names):
    msgs = sim.get("messages") or []
    called_at = collections.defaultdict(list)
    for i, m in enumerate(msgs):
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            n = tc.get("name") or (tc.get("function") or {}).get("name")
            if n:
                called_at[n].append(i)

    taught = []
    for i, m in enumerate(msgs):
        if m.get("role") != "user" or not isinstance(m.get("content"), str):
            continue
        for n in names:
            if n not in m["content"]:
                continue
            before = [j for j in called_at.get(n, []) if j < i]
            after = [j for j in called_at.get(n, []) if j > i]
            if not before and after:
                taught.append((i, n, after[0]))
    seen = set()
    taught = [t for t in taught if not (t[1] in seen or seen.add(t[1]))]

    counts = collections.Counter()
    for m in msgs:
        c = m.get("content")
        if not isinstance(c, str):
            continue
        if m.get("role") == "tool":
            if "has not been given to you by the agent" in c:
                counts["GRANT_ERR"] += 1
            if "Unknown discoverable tool" in c:
                counts["UNKNOWN"] += 1
            if "Unexpected parameter" in c or "Missing required parameter" in c:
                counts["ARGERR"] += 1
            if "Executed:" in c:
                counts["EXECOK"] += 1
            if "already exist" in c:
                counts["DUPWRITE"] += 1
        elif m.get("role") == "assistant":
            if "Would you like to be transferred" in c:
                counts["ASKED"] += 1
    return taught, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="20260803g")
    args = ap.parse_args()
    sims = load(args.tag)
    names = tool_universe(sims)

    rows = []
    for s in sims:
        taught, counts = scan(s, names)
        rows.append(
            (
                s.get("task_id"),
                s.get("trial"),
                (s.get("reward_info") or {}).get("reward") or 0.0,
                taught,
                counts,
            )
        )
    rows.sort()

    print(f"tool universe: {len(names)}\n")
    print(f"{'task':10s} {'t':>2} {'rew':>4}  {'GRANT':>5} {'UNK':>3} {'ARG':>3} {'ASK':>3} "
          f"{'EXEC':>4} {'DUP':>3}  user-taught")
    for t, tr, r, taught, c in rows:
        tt = ", ".join(f"{n}@{i}" for i, n, _ in taught) or "-"
        print(f"{t:10s} {tr:2d} {r:4.1f}  {c['GRANT_ERR']:5d} {c['UNKNOWN']:3d} {c['ARGERR']:3d} "
              f"{c['ASKED']:3d} {c['EXECOK']:4d} {c['DUPWRITE']:3d}  {tt[:70]}")

    def rate(sel):
        sub = [x for x in rows if sel(x)]
        return len(sub), (sum(1 for x in sub if x[2] == 1.0) / len(sub) if sub else float("nan"))

    print("\n=== user rescue ===")
    n, p = rate(lambda x: bool(x[3]))
    print(f"  user-taught >=1 : n={n:3d} pass={p:.3f}")
    n, p = rate(lambda x: not x[3])
    print(f"  user-taught  0  : n={n:3d} pass={p:.3f}")
    passed = [x for x in rows if x[2] == 1.0]
    print(f"  of {len(passed)} passing sims, {sum(1 for x in passed if x[3])} had a user-taught tool")

    print("\n=== engine-message events ===")
    for key in ["GRANT_ERR", "UNKNOWN", "ARGERR", "ASKED", "DUPWRITE"]:
        n, p = rate(lambda x, k=key: x[4][k] > 0)
        m, q = rate(lambda x, k=key: x[4][k] == 0)
        tot = sum(x[4][key] for x in rows)
        print(f"  {key:10s} total={tot:4d}  present n={n:3d} pass={p:.3f} | absent n={m:3d} pass={q:.3f}")


if __name__ == "__main__":
    main()
