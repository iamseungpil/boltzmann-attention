"""Which walls each task set actually hits, counted over a whole arm rather than a sample.

The 2026-08-04 read-through built its §2 table while the sweep was still draining the
reserve queue, so it counted 142 of the 194 simulations and the split it reported
(`[READ-FIRST]` 34, zero-score search 380, shell 173) is a subset, not the arm. The
claim the table carries — that front-32 never walks the paths the new tasks die on — is
strong enough to be worth measuring on the finished arm, and cheap to get wrong by
sampling the half that finished first.

So each signal here is a decidable predicate over the transcript, counted twice: how many
times it fired, and how many simulations it touched at all. Both matter — 68 warnings in
one simulation and 68 across 68 are opposite findings ([[08]]).

  read-first        the engine blocked a call for a missing prerequisite read
  zero-score        a search came back with `Score: 0.0000` — retrieval found nothing
  shell             calls to the filesystem shell, and how many returned no matches
  shell-echo-term   a shell search whose quoted term came out of an earlier tool output
  shell-db-id       a shell search carrying a record identifier — the filesystem cannot
                    hold one, so the call cannot hit
  duplicate-read    the engine warned that a read was already done
  grounding-warning the grounding guard dropped or flagged an operand
  coverage          the completion lever fired

The front-32 / new split is derived, not listed: front-32 is exactly the task set of the
A arm that ran the same stack three days earlier, so the boundary moves with the data
instead of with a constant someone has to remember to update.
"""

import argparse
import collections
import glob
import gzip
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from x50_says_not_does import ARMS, SIM, fam, norm_args  # noqa: E402
from x66_effective_tool_miss import agent_actions  # noqa: E402

# The one tool that hands back a checking-account id, and so the entry point of the whole
# debit chain. Suffix-stripped: the transcript writes it `..._3847`.
GATE = "get_all_user_accounts_by_user_id"


def gold_name(check):
    """The tool a gold action is really about, unwrapped from its dispatcher."""
    g = check.get("action") or {}
    a = norm_args(g.get("arguments"))
    inner = (a.get("agent_tool_name") or a.get("discoverable_tool_name")
             or a.get("user_tool_name"))
    return fam(inner or g.get("name") or "")

# Markers the engine writes into tool output. Counted as occurrences of the tag, so a
# single message carrying three of them counts three times.
TAGS = {
    "read-first": "[READ-FIRST]",
    "zero-score": "Score: 0.0000",
    "duplicate-read": "[DUPLICATE-READ]",
    "duplicate-compute": "[DUPLICATE-COMPUTE]",
    "grounding-warning": "[GROUNDING WARNING]",
    "coverage": "[coverage]",
}

NO_MATCH = re.compile(r"no matches found", re.I)
# Quoted search terms: `grep -r 'rp65a7b3c4' .` — the term is what the agent believes the
# filesystem holds. Short terms are dropped because two- and three-character strings hit
# by accident.
QUOTED = re.compile(r"""['"]([^'"]{4,})['"]""")
# Record identifiers as the environment writes them: `rp65a7b3c4`, `chk_cr89a2b3c4_2`,
# `txn_a8f1c2d3e403`, `dbc_28110cb53a43`, bare `755bcb4d5d`. These exist only in the
# database, so a filesystem search for one can never hit — which is the point.
DB_ID = re.compile(r"\b(?:[a-z]{2,4}_[0-9a-z_]{6,}|[a-z]{2}[0-9a-f]{8}|[0-9a-f]{10})\b")


def load(pattern):
    files = sorted(glob.glob(f"{SIM}/{pattern}.results.json.gz"))
    if not files:
        raise SystemExit(f"no runs matched {SIM}/{pattern}.results.json.gz")
    sims = []
    for p in files:
        sims.extend(json.load(gzip.open(p, "rt", encoding="utf-8")).get("simulations") or [])
    return files, sims


def tool_outputs(sim):
    return [m.get("content") for m in sim.get("messages") or []
            if m.get("role") == "tool" and isinstance(m.get("content"), str)]


def shell_calls(sim):
    """Every shell call with the tool output that answered it.

    Pairing is by `tool_call_id` where the transcript carries one and by position
    otherwise, because a mis-paired output would make a hit look like a miss.
    """
    msgs = sim.get("messages") or []
    out = []
    for i, m in enumerate(msgs):
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            name = tc.get("name") or (tc.get("function") or {}).get("name")
            if name != "shell":
                continue
            args = tc.get("arguments")
            if args is None:
                args = (tc.get("function") or {}).get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {"command": args}
            cmd = (args or {}).get("command") or ""
            cid = tc.get("id")
            result = ""
            for mm in msgs[i + 1:i + 8]:
                if mm.get("role") != "tool":
                    continue
                if cid and mm.get("tool_call_id") not in (None, cid):
                    continue
                result = mm.get("content") or ""
                break
            out.append((i, cmd, result))
    return out


def seen_before(sim, upto):
    """Everything a tool had already returned before message index `upto`."""
    parts = []
    for m in (sim.get("messages") or [])[:upto]:
        if m.get("role") == "tool" and isinstance(m.get("content"), str):
            parts.append(m["content"])
    return "\n".join(parts)


def census(sim):
    c = collections.Counter()
    body = "\n".join(tool_outputs(sim))
    for key, tag in TAGS.items():
        n = body.count(tag)
        if n:
            c[key] = n
    for idx, cmd, result in shell_calls(sim):
        c["shell"] += 1
        if NO_MATCH.search(result or ""):
            c["shell-no-match"] += 1
        prior = seen_before(sim, idx)
        terms = [t for t in QUOTED.findall(cmd)]
        if any(t in prior for t in terms):
            c["shell-echo-term"] += 1
        if any(DB_ID.search(t) for t in terms) or DB_ID.search(cmd):
            c["shell-db-id"] += 1
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="N97", help="arm key from x50.ARMS")
    ap.add_argument("--glob", help="explicit glob, overrides --arm")
    ap.add_argument("--front", default="A",
                    help="arm whose task set defines front-32 (empty = no split)")
    args = ap.parse_args()

    pattern = args.glob or ARMS[args.arm]
    files, sims = load(pattern)

    front = set()
    if args.front:
        _, fs = load(ARMS[args.front])
        front = {s["task_id"] for s in fs}

    groups = {"front": [], "new": []}
    for s in sims:
        groups["front" if s["task_id"] in front else "new"].append(s)

    print(f"arm {args.arm} · {pattern}")
    for p in files:
        print(f"  {os.path.basename(p)}")
    print(f"\nsim {len(sims)} · task {len({s['task_id'] for s in sims})} · "
          f"front-32 기준 arm {args.front or '-'} ({len(front)} task)\n")

    keys = list(TAGS) + ["shell", "shell-no-match", "shell-echo-term", "shell-db-id"]
    tot = {g: collections.Counter() for g in groups}
    touched = {g: collections.Counter() for g in groups}
    passed = {}
    for g, ss in groups.items():
        passed[g] = sum(1 for s in ss if (s.get("reward_info") or {}).get("reward") == 1)
        for s in ss:
            c = census(s)
            for k, v in c.items():
                tot[g][k] += v
                touched[g][k] += 1

    w = max(len(k) for k in keys) + 1
    print(f"{'신호':<{w}}  {'front32 (n=%d)' % len(groups['front']):>22}   "
          f"{'신규 (n=%d)' % len(groups['new']):>22}")
    for k in keys:
        f = f"{tot['front'][k]}회 / {touched['front'][k]} sim"
        n = f"{tot['new'][k]}회 / {touched['new'][k]} sim"
        print(f"{k:<{w}}  {f:>22}   {n:>22}")
    print(f"\n{'pass':<{w}}  {passed['front']}/{len(groups['front'])}"
          f" = {passed['front'] / max(1, len(groups['front'])):.3f}"
          f"   {passed['new']}/{len(groups['new'])}"
          f" = {passed['new'] / max(1, len(groups['new'])):.3f}")
    tp = passed["front"] + passed["new"]
    print(f"{'pass 전체':<{w}}  {tp}/{len(sims)} = {tp / max(1, len(sims)):.4f}")

    term = collections.Counter(s.get("termination_reason") for s in sims)
    print("\n종료사유: " + " · ".join(f"{k}={v}" for k, v in term.most_common()))

    gate_table(groups)


def gate_table(groups):
    """Does the gateway tool separate the two task sets, and does calling it help?

    The read-through's answer was that gold requires `get_all_user_accounts_by_user_id`
    in 31 of the new tasks and none of front-32, and that calling it is necessary but not
    sufficient. Both halves are re-counted here on the finished arm, because the second
    half is what decides whether the entry lever can be expected to move pass at all.
    """
    print(f"\n관문 `{GATE}` — gold 요구 × 실호출")
    for g, ss in groups.items():
        rows = collections.Counter()
        gold_tasks = set()
        for s in ss:
            gold = {gold_name(c) for c in
                    ((s.get("reward_info") or {}).get("action_checks") or [])}
            req = GATE in gold
            if req:
                gold_tasks.add(s["task_id"])
            called = any(e == GATE for _, e, _ in agent_actions(s))
            ok = (s.get("reward_info") or {}).get("reward") == 1
            rows[(req, called, ok)] += 1
        print(f"  [{g}] gold가 관문을 요구하는 태스크 {len(gold_tasks)}개")
        for req in (True, False):
            for called in (True, False):
                p = rows[(req, called, True)]
                f = rows[(req, called, False)]
                if not (p + f):
                    continue
                print(f"    요구={'Y' if req else 'N'} 호출={'Y' if called else 'N'}"
                      f"  pass {p} / fail {f}  = {p / (p + f):.0%}")


if __name__ == "__main__":
    main()
