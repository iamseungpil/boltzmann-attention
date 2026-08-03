"""Print one simulation the way a person reads it: turns, calls, and what gold wanted.

Aggregates keep pointing at classes; deciding what a class actually is requires the
trajectory. This renders a single sim compactly enough to read end to end — customer
turns, agent text, every tool call with its arguments, tool output truncated — with
the gold action list and which of them matched printed first.
"""

import argparse
import glob
import gzip
import json

SIM = ("/home/woori/workspace_common/boltzmann-attention-pi/"
       "reports/facet_rft_2026/sim_results")

ARMS = {
    "A":  "bank_ax33n_gpu*_20260803g",
    "B4": "bank_b4_gpu*_20260803h",
}


def norm(a):
    if isinstance(a, str):
        try:
            return json.loads(a)
        except Exception:
            return {"_raw": a}
    return a if isinstance(a, dict) else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sim", help="task_003/t1")
    ap.add_argument("--arm", default="B4", choices=sorted(ARMS))
    ap.add_argument("--cut", type=int, default=300, help="tool output truncation")
    args = ap.parse_args()

    task, _, trial = args.sim.partition("/")
    trial = int(trial.lstrip("t") or 0)

    target = None
    for p in sorted(glob.glob(f"{SIM}/{ARMS[args.arm]}.results.json.gz")):
        for s in json.load(gzip.open(p, "rt", encoding="utf-8")).get("simulations") or []:
            if s.get("task_id") == task and s.get("trial") == trial:
                target = s
    if target is None:
        raise SystemExit(f"{args.sim} not found in arm {args.arm}")

    ri = target.get("reward_info") or {}
    print(f"=== {args.arm} {args.sim}  reward={ri.get('reward')} "
          f"basis={ri.get('reward_basis')} db={(ri.get('db_check') or {}).get('db_match')} "
          f"term={target.get('termination_reason')} ===\n")

    print("--- gold actions ---")
    for c in ri.get("action_checks") or []:
        a = c.get("action") or {}
        print(f"  [{'OK ' if c.get('action_match') else 'MISS'}] {a.get('requestor'):9s} "
              f"{a.get('name')}  {json.dumps(a.get('arguments'), ensure_ascii=False)[:160]}")

    print("\n--- trajectory ---")
    for i, m in enumerate(target.get("messages") or []):
        role = m.get("role")
        c = m.get("content")
        if role == "user":
            print(f"\n[{i}] USER: {' '.join((c or '').split())[:400]}")
        elif role == "assistant":
            if isinstance(c, str) and c.strip():
                print(f"[{i}] AGENT: {' '.join(c.split())[:400]}")
            for tc in m.get("tool_calls") or []:
                n = tc.get("name") or (tc.get("function") or {}).get("name")
                a = tc.get("arguments")
                if a is None:
                    a = (tc.get("function") or {}).get("arguments")
                print(f"[{i}]   -> {n}({json.dumps(norm(a), ensure_ascii=False)[:220]})")
        elif role == "tool":
            print(f"[{i}]   <- {' '.join((c or '').split())[:args.cut]}")


if __name__ == "__main__":
    main()
