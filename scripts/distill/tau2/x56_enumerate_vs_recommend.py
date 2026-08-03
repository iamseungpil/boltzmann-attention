"""Did the agent present the eligible set, or pick one card for the customer?

task_003 passes on one trial and fails on the other with the same stack, so the card
choice cannot be a capability limit. Reading both: the passing run listed the eligible
cards with their fees and rates and the customer chose; the failing run named a single
card and the customer followed it into the wrong one.

That suggests the agent's job here is not to choose. The gold action is the *customer*
applying, and the customer holds a preference the agent cannot see. Narrowing to one
card substitutes the agent's judgement for theirs.

This checks the pattern against every simulation that reaches a card application, in
both arms: how many of the eligible cards did the agent name before the customer acted,
and did the run pass. Counting names in the agent's own prose against the eligible set
the tool returned is decidable — no semantic judgement.
"""

import argparse
import collections
import glob
import gzip
import json
import re

SIM = ("/home/woori/workspace_common/boltzmann-attention-pi/"
       "reports/facet_rft_2026/sim_results")
ARMS = {"A": "bank_ax33n_gpu*_20260803g", "B4": "bank_b4_gpu*_20260803h"}


def norm(a):
    if isinstance(a, str):
        try:
            return json.loads(a)
        except Exception:
            return {}
    return a if isinstance(a, dict) else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="A,B4")
    args = ap.parse_args()

    rows = []
    for arm in args.arms.split(","):
        for p in sorted(glob.glob(f"{SIM}/{ARMS[arm]}.results.json.gz")):
            for s in json.load(gzip.open(p, "rt", encoding="utf-8")).get("simulations") or []:
                msgs = s.get("messages") or []
                elig, applied, gold = [], None, None
                for c in (s.get("reward_info") or {}).get("action_checks") or []:
                    a = c.get("action") or {}
                    if a.get("name") == "apply_for_credit_card":
                        gold = norm(a.get("arguments")).get("card_type")
                # Where the customer acted, and what the tool had offered by then.
                act = len(msgs)
                for i, m in enumerate(msgs):
                    if m.get("role") == "user":
                        for tc in m.get("tool_calls") or []:
                            if (tc.get("name") or "") == "apply_for_credit_card":
                                applied = norm(tc.get("arguments")).get("card_type")
                                act = min(act, i)
                    if m.get("role") == "tool" and isinstance(m.get("content"), str) \
                            and "'eligible'" in m["content"]:
                        elig = re.findall(r"'card': '([^']+)'", m["content"])
                if not gold or not elig:
                    continue
                said = " ".join(m.get("content") or "" for m in msgs[:act]
                                if m.get("role") == "assistant" and isinstance(m.get("content"), str))
                named = [c for c in set(elig) if c.lower() in said.lower()]
                rows.append({
                    "arm": arm, "sim": f"{s['task_id']}/t{s.get('trial')}",
                    "pass": ((s.get("reward_info") or {}).get("reward") or 0.0) == 1.0,
                    "elig": len(set(elig)), "named": len(named),
                    "gold_named": gold in named, "gold": gold, "applied": applied,
                })

    print(f"{'arm':4s} {'sim':16s} {'pass':5s} {'적격':4s} {'언급':4s} {'gold언급':7s} "
          f"gold / 신청")
    for r in sorted(rows, key=lambda x: (x["sim"], x["arm"])):
        print(f"{r['arm']:4s} {r['sim']:16s} {str(r['pass']):5s} {r['elig']:4d} {r['named']:4d} "
              f"{str(r['gold_named']):7s} {r['gold']} / {r['applied']}")

    print("\n=== 열거 수 × 통과 교차표 ===")
    tab = collections.Counter()
    for r in rows:
        band = "1장 이하" if r["named"] <= 1 else "2장 이상"
        tab[(band, r["pass"])] += 1
    for band in ("1장 이하", "2장 이상"):
        p, f = tab[(band, True)], tab[(band, False)]
        n = p + f
        print(f"  {band}: 통과 {p} / 실패 {f}" + (f"  = {p / n:.0%}" if n else ""))

    print("\n=== gold를 언급했는가 × 통과 ===")
    tab2 = collections.Counter()
    for r in rows:
        tab2[(r["gold_named"], r["pass"])] += 1
    for g in (True, False):
        p, f = tab2[(g, True)], tab2[(g, False)]
        n = p + f
        print(f"  gold 언급={g}: 통과 {p} / 실패 {f}" + (f"  = {p / n:.0%}" if n else ""))


if __name__ == "__main__":
    main()
