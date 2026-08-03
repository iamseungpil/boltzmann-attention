"""How is a transfer's `summary` graded — and does touching it risk a passing run?

Every hand-off lever in the design writes into or conditions on the transfer call's
`summary`. If the evaluator compares that argument exactly, then a lever that makes the
agent rewrite its summary can turn a matched action into a missed one, and the "cost on
passing sims" column of `x61` is not a nuisance but a regression channel. This prints
what gold asks for on transfer actions and whether the runs that matched them wrote the
same string.
"""

import collections
import glob
import gzip
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SIM = os.path.join(REPO, "reports", "facet_rft_2026", "sim_results")
ARMS = {"A": "bank_ax33n_gpu*_20260803g", "B4": "bank_b4_gpu*_20260803h"}
TRANSFER = "transfer_to_human_agents"


def main():
    keys = collections.Counter()
    shown = 0
    matched = collections.Counter()
    for arm, pat in ARMS.items():
        for p in sorted(glob.glob(os.path.join(SIM, f"{pat}.results.json.gz"))):
            for s in json.load(gzip.open(p, "rt", encoding="utf-8")).get("simulations") or []:
                for c in (s.get("reward_info") or {}).get("action_checks") or []:
                    a = c.get("action") or {}
                    if a.get("name") != TRANSFER:
                        continue
                    args = a.get("arguments") or {}
                    keys[tuple(sorted(args))] += 1
                    matched["match" if c.get("action_match") else "miss"] += 1
                    if shown < 6:
                        shown += 1
                        print(f"[{arm}] {s['task_id']}/t{s.get('trial')} "
                              f"{'MATCH' if c.get('action_match') else 'MISS '} "
                              f"{json.dumps(args, ensure_ascii=False)[:220]}")
    print("\n--- gold transfer 인자 키 조합 ---")
    for k, v in keys.most_common():
        print(f"  {v:4d}  {list(k)}")
    print(f"\n--- 대조 결과 --- {dict(matched)}")


if __name__ == "__main__":
    main()
