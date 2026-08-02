"""Per-trajectory forensic for AX33 run-g split tasks (1-of-2 trials passed).

Aggregate pass^2 said variance is worse than frontier. Aggregates do not say why.
For every task whose two trials disagree, this prints the two trajectories side by
side: the failing check, the tool-call sequence, the first point where the two
runs diverge, and every gate/lever utterance in each run.

Read-only. Usage:
    python ax33g_split_forensic.py [--tag 20260803g] [--task task_003]
"""

import argparse
import glob
import gzip
import json
import re
from collections import defaultdict

SIM_DIR = (
    "/home/woori/workspace_common/boltzmann-attention-pi/reports/facet_rft_2026/sim_results"
)
LEVER_RE = re.compile(
    r"\[(T2_[A-Z_]+|coverage|checks)\]|final-word deny|no verbatim customer span|retract=1"
)


def load(tag):
    sims = []
    for path in sorted(glob.glob(f"{SIM_DIR}/bank_ax33n_gpu*_{tag}.results.json.gz")):
        data = json.load(gzip.open(path, "rt", encoding="utf-8"))
        sims.extend(data.get("simulations") or [])
    return sims


def calls(sim):
    """Ordered (tool_name, args) actually proposed by the agent."""
    out = []
    for m in sim.get("messages") or []:
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            name = tc.get("name") or (tc.get("function") or {}).get("name")
            args = tc.get("arguments")
            if args is None:
                args = (tc.get("function") or {}).get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    pass
            out.append((name, args))
    return out


def levers(sim):
    """Every lever utterance in the trajectory, with the turn it landed on."""
    hits = []
    for i, m in enumerate(sim.get("messages") or []):
        content = m.get("content")
        if not isinstance(content, str):
            continue
        for match in LEVER_RE.finditer(content):
            frag = content[max(0, match.start() - 40) : match.start() + 160]
            hits.append((i, m.get("role"), " ".join(frag.split())))
    return hits


def failed_checks(sim):
    ri = sim.get("reward_info") or {}
    out = []
    db = ri.get("db_check") or {}
    if db and not db.get("db_match", True):
        out.append("DB mismatch")
    for ac in ri.get("action_checks") or []:
        if not ac.get("action_match", True):
            act = ac.get("action") or {}
            out.append(f"action MISS {act.get('name')} args={act.get('arguments')}")
    for ea in ri.get("env_assertions") or []:
        if not ea.get("met", True):
            out.append(f"env assertion MISS {ea.get('env_assertion')}")
    return out


def first_divergence(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x[0] != y[0]:
            return i, f"tool name {x[0]} vs {y[0]}"
        if x[1] != y[1]:
            return i, f"same tool {x[0]}, args differ"
    if len(a) != len(b):
        return min(len(a), len(b)), f"length {len(a)} vs {len(b)}"
    return None, "identical call sequences"


def brief(args, width=110):
    s = json.dumps(args, ensure_ascii=False, sort_keys=True) if args else "{}"
    return s if len(s) <= width else s[:width] + "…"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="20260803g")
    ap.add_argument("--task", default=None, help="restrict to one task id")
    ap.add_argument("--full", action="store_true", help="print every call, not just to divergence+6")
    args = ap.parse_args()

    by_task = defaultdict(list)
    for sim in load(args.tag):
        by_task[sim.get("task_id")].append(sim)

    split = {
        t: sorted(v, key=lambda s: s.get("trial") or 0)
        for t, v in by_task.items()
        if len(v) >= 2 and len({(s.get("reward_info") or {}).get("reward") for s in v}) > 1
    }
    if args.task:
        split = {t: v for t, v in split.items() if t == args.task}

    print(f"# AX33 {args.tag} — split tasks: {len(split)}  ({', '.join(sorted(split))})\n")

    for task in sorted(split):
        pair = split[task]
        print("=" * 100)
        print(f"## {task}")
        seqs = []
        for sim in pair:
            ri = sim.get("reward_info") or {}
            seq = calls(sim)
            seqs.append(seq)
            lv = levers(sim)
            print(
                f"\n  trial {sim.get('trial')}  reward={ri.get('reward')}  "
                f"term={sim.get('termination_reason')}  msgs={len(sim.get('messages') or [])}  "
                f"calls={len(seq)}  levers={len(lv)}  dur={round(sim.get('duration') or 0)}s"
            )
            fc = failed_checks(sim)
            if fc:
                for f in fc:
                    print(f"     FAIL: {f}")
            else:
                print("     (all checks passed)")
            for turn, role, frag in lv:
                print(f"     lever @msg{turn} ({role}): {frag}")

        idx, why = first_divergence(*seqs)
        print(f"\n  >>> first divergence: {why}" + (f" at call #{idx}" if idx is not None else ""))
        lo = 0 if args.full or idx is None else max(0, idx - 2)
        hi = max(len(seqs[0]), len(seqs[1])) if args.full else (
            len(seqs[0]) if idx is None else idx + 7
        )
        print(f"  call sequences [{lo}:{hi}]  (t{pair[0].get('trial')} | t{pair[1].get('trial')})")
        for i in range(lo, hi):
            a = seqs[0][i] if i < len(seqs[0]) else None
            b = seqs[1][i] if i < len(seqs[1]) else None
            mark = "  " if a and b and a == b else "**"
            print(f"   {mark}#{i:<3} A {a[0] if a else '-'}  {brief(a[1]) if a else ''}")
            print(f"       {'':<3} B {b[0] if b else '-'}  {brief(b[1]) if b else ''}")
        print()


if __name__ == "__main__":
    main()
