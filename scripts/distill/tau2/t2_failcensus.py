#!/usr/bin/env python
"""궤적 전수 실패-원인 census — 모델간 pass 격차의 *진짜 원인* 확정.

selection-formalize(resolve_selection)는 소수 경로(~5%)임이 드러남. pass는 전체 궤적이 결정.
이 census는 각 sim을 reward 컴포넌트(DB-state / NL-assertion)·놓친 gold 행동·종료사유·날조로 분해해
여러 모델(7B vs 32B 등)의 실패분포를 *비교* → "32B가 무엇을 더 해서 0.19→0.55인가" 확정.

Run: PY t2_failcensus.py <sim_dir1> <sim_dir2> ...   (각 = data/simulations/<save>)
"""
import json
import os
import sys
from collections import Counter


def census(sim_dir):
    with open(os.path.join(sim_dir, "results.json"), encoding="utf-8") as f:
        sims = json.load(f).get("simulations", [])
    n = len(sims)
    npass = 0
    comp_fail = Counter()       # 실패 sim의 컴포넌트: DB_only / NL_only / DB+NL / OTHER
    missed_actions = Counter()  # 놓친 gold 행동(action_match=False)의 도구명
    term = Counter()            # 종료사유
    halluc_tasks = 0
    nl_fail_tasks = 0
    db_fail_tasks = 0
    for s in sims:
        ri = s.get("reward_info") or {}
        r = ri.get("reward", 0)
        if r is not None and r >= 1:
            npass += 1
        else:
            basis = ri.get("reward_basis", [])
            rb = ri.get("reward_breakdown", {})
            db_ok = (ri.get("db_check") or {}).get("db_match", True)
            nl = ri.get("nl_assertions") or []
            nl_unmet = any(not a.get("met", True) for a in nl)
            db_bad = ("DB" in basis) and not db_ok
            nl_bad = ("NL_ASSERTION" in basis) and nl_unmet
            if db_bad: db_fail_tasks += 1
            if nl_bad: nl_fail_tasks += 1
            if db_bad and nl_bad: comp_fail["DB+NL"] += 1
            elif db_bad: comp_fail["DB_only"] += 1
            elif nl_bad: comp_fail["NL_only"] += 1
            else: comp_fail["OTHER"] += 1
            for a in (ri.get("action_checks") or []):
                if not a.get("action_match", True):
                    missed_actions[(a.get("action") or {}).get("name", "?")] += 1
        term[s.get("termination_reason", "?")] += 1
        if s.get("hallucination_retries_used", 0):
            halluc_tasks += 1
    return {"dir": os.path.basename(sim_dir), "n": n, "pass": npass, "pass_rate": npass / max(n, 1),
            "comp_fail": dict(comp_fail.most_common()), "db_fail": db_fail_tasks, "nl_fail": nl_fail_tasks,
            "missed_actions": dict(missed_actions.most_common(10)), "term": dict(term.most_common()),
            "halluc_tasks": halluc_tasks}


def main():
    rows = [census(d) for d in sys.argv[1:]]
    print("=== 궤적 전수 실패-원인 census ===\n")
    for r in rows:
        print(f"## {r['dir']}  (n={r['n']}·pass {r['pass']}/{r['n']}={r['pass_rate']:.3f})")
        print(f"   실패 컴포넌트: {r['comp_fail']}")
        print(f"   DB-state 실패 task: {r['db_fail']}  ·  NL-assertion 실패 task: {r['nl_fail']}  ·  날조-retry task: {r['halluc_tasks']}")
        print(f"   놓친 gold 행동 top: {r['missed_actions']}")
        print(f"   종료사유: {r['term']}\n")
    # 비교 한 줄
    if len(rows) >= 2:
        print("── 비교(pass 격차 원인) ──")
        for r in rows:
            db_share = r['db_fail'] / max(r['n'], 1)
            nl_share = r['nl_fail'] / max(r['n'], 1)
            print(f"  {r['dir']:32s} pass={r['pass_rate']:.3f}  DB실패율={db_share:.2f}  NL실패율={nl_share:.2f}")
        print("  → 모델간 DB실패율 차이가 크면 *행동실행*(능력), NL실패율 차이가 크면 *통신/정보전달*이 격차 원인.")


if __name__ == "__main__":
    main()
