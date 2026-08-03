"""Plan a two-server split of the 97 banking tasks so both sides finish together.

A round-robin split leaves one GPU idle for an hour: per-task cost here spans two orders
of magnitude (90s to 5000s), so who gets the heavy tasks decides the makespan, not how
many tasks each side got.

Two facts shape the plan.

*The weights are only partly known.* The one pass that touched all 97 tasks
(2026-07-18) lost 25 of them to infrastructure errors, and the recent alltools arms
cover 32 — the front subset — so 21 tasks have no timing at all. Scheduling those on a
guessed weight is what unbalances a split.

*Per-sim duration is not a stack constant.* Measured effective concurrency was ~2.0 in
the recent arms and ~6.6 in the full-97 pass, and durations inflate with it. So weights
are only ever used as *relative* sizes within one source, and the second source is
rescaled onto the first by the median ratio over the tasks they share.

So: LPT-schedule what has been measured, and leave what has not to a shared reserve the
drivers drain by claiming batches — whichever side finishes its block first takes the
next one. That is what makes the finish times converge rather than the estimate.
"""

import argparse
import collections
import glob
import gzip
import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
SIM_REMOTE = ("/home/woori/workspace_common/boltzmann-attention-pi/"
              "reports/facet_rft_2026/sim_results")
SIM_LOCAL = os.path.abspath(os.path.join(HERE, "..", "..", "..",
                                         "reports", "facet_rft_2026", "sim_results"))
SIM = SIM_REMOTE if os.path.isdir(SIM_REMOTE) else SIM_LOCAL

TASKS_JSON = "/home/woori/scratch/tau2-bench/data/tau2/domains/banking_knowledge/tasks.json"

# The one pass over all 97 tasks — the only internally consistent full weight scale.
BASE = "bank_all97_nt1_v2_20260718"
# The arms on the current stack (alltools, gate=1, 32B front, concurrency 4).
RECENT = "bank_ax33n_gpu*_20260803g bank_b4_gpu*_20260803h"


def durations(patterns):
    """task_id -> [duration, ...] over every persisted sim matching the glob patterns."""
    out = collections.defaultdict(list)
    for pat in patterns.split():
        for p in sorted(glob.glob(f"{SIM}/{pat}.results.json.gz")):
            d = json.load(gzip.open(p, "rt", encoding="utf-8"))
            for s in d.get("simulations") or []:
                if (s.get("duration") or 0) > 0:      # 0 == infrastructure_error
                    out[s["task_id"]].append(s["duration"])
    return out


def all_task_ids():
    if os.path.isfile(TASKS_JSON):
        return [t["id"] for t in json.load(open(TASKS_JSON, encoding="utf-8"))]
    return sorted(durations(BASE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--recent", default=RECENT)
    ap.add_argument("--ways", type=int, default=2)
    ap.add_argument("--trials", type=int, default=2)
    ap.add_argument("--batch", type=int, default=3, help="tasks per reserve batch")
    ap.add_argument("--emit", help="directory to write the plan into")
    args = ap.parse_args()

    base = {t: statistics.median(v) for t, v in durations(args.base).items()}
    rec = {t: statistics.median(v) for t, v in durations(args.recent).items()}

    shared = [t for t in rec if t in base]
    ratio = statistics.median([rec[t] / base[t] for t in shared]) if shared else 1.0

    tasks = all_task_ids()
    weight, source = {}, {}
    for t in tasks:
        if t in base:
            weight[t], source[t] = base[t], "base"
        elif t in rec:
            weight[t], source[t] = rec[t] / ratio, "rescaled"   # onto the base scale

    known = [t for t in tasks if t in weight]
    unknown = [t for t in tasks if t not in weight]

    print(f"tasks={len(tasks)}  known={len(known)} "
          f"(base={sum(1 for t in known if source[t]=='base')}, "
          f"rescaled={sum(1 for t in known if source[t]=='rescaled')})  "
          f"unknown={len(unknown)}  ratio(recent/base)={ratio:.2f}")

    groups = [[] for _ in range(args.ways)]
    load = [0.0] * args.ways
    for t in sorted(known, key=lambda x: -weight[x]):
        i = load.index(min(load))
        groups[i].append(t)                       # heaviest first == run order
        load[i] += weight[t] * args.trials

    reserve = [unknown[i:i + args.batch] for i in range(0, len(unknown), args.batch)]

    print()
    for i, g in enumerate(groups):
        print(f"# GPU{i} block: {len(g)} tasks x nt{args.trials} = {len(g)*args.trials} sims  "
              f"relative load {load[i]/3600:.1f}")
        print(f"G{i}=" + ",".join(g))
        print()
    spread = (max(load) - min(load)) / max(load) * 100 if max(load) else 0
    print(f"block imbalance {spread:.2f}%  "
          f"(the {len(unknown)} unweighted tasks go to the shared reserve, "
          f"{len(reserve)} batches of <={args.batch})")
    for j, b in enumerate(reserve):
        print(f"R{j:02d}=" + ",".join(b))

    if args.emit:
        os.makedirs(args.emit, exist_ok=True)
        os.makedirs(f"{args.emit}/reserve", exist_ok=True)
        for i, g in enumerate(groups):
            open(f"{args.emit}/gpu{i}.tasks", "w").write(",".join(g) + "\n")
        for j, b in enumerate(reserve):
            open(f"{args.emit}/reserve/batch_{j:02d}", "w").write(",".join(b) + "\n")
        print(f"\nwrote plan to {args.emit}")


if __name__ == "__main__":
    main()
