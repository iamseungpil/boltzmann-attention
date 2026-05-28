"""Generate stratified random task subset for Phase 2a/2b α grid.

Picks 10 tasks each from telecom's 3 domains (mobile_data_issue, mms_issue,
service_issue) and saves to JSON. Uses the SAME tasks across all alphas + models
for fair within-subject comparison.

Usage:
  python _phase2a_stratified_task_ids.py \
    --baseline-json reports/facet_rft_2026/phase1_baseline/base_n114_qwen_openrouter_mini/B0_telecom_base.json/results.json \
    --per-domain 10 \
    --seed 42 \
    --out reports/facet_rft_2026/phase2_steering/stratified_task_ids.json
"""
import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path


DOMAINS = ["mobile_data_issue", "mms_issue", "service_issue"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-json", required=True,
                    help="Source of task IDs (results.json from any baseline run)")
    ap.add_argument("--per-domain", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = json.load(open(args.baseline_json))
    sims = d.get("simulations", [])
    # collect unique task IDs grouped by domain prefix
    by_dom = defaultdict(set)
    pat = re.compile(r"^\[(?P<dom>[a-z_]+)\]")
    for s in sims:
        tid = s.get("task_id") or ""
        m = pat.match(tid)
        if m:
            by_dom[m.group("dom")].add(tid)

    rng = random.Random(args.seed)
    chosen = []
    for dom in DOMAINS:
        ids = sorted(by_dom.get(dom, []))
        if not ids:
            print(f"[strat] WARN no tasks for domain '{dom}'")
            continue
        n = min(args.per_domain, len(ids))
        pick = rng.sample(ids, n)
        chosen.extend(pick)
        print(f"[strat] {dom}: picked {n}/{len(ids)} tasks")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(chosen, f, indent=2, ensure_ascii=False)
    print(f"[strat] wrote {len(chosen)} task IDs to {args.out}")


if __name__ == "__main__":
    main()
