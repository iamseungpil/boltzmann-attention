# -*- coding: utf-8 -*-
"""[[08]] 포렌식 재정량 — "transaction_id ⋈ 지배" 오염 교정 (2026-07-14·forensic-guard 촉발).
bank_keystone_replay/extract의 ⋈ 케이스(853)가 **미제출(coverage) gold를 agent 첫 호출에 페어링**해
오염됐음을 발각(task_086 정독: 16 case 전부 동일 chosen·gold5·"한도로 다 못 냄"). 진짜 정량 =
agent가 *실제 제출한* transaction_id 집합 vs gold 집합 비교.

분류(sim별 집합 연산·결정론):
  correct = agent_ids ∩ gold_ids   (제출·맞음)
  wrong   = agent_ids − gold_ids   (제출·틀림 = 진짜 ⋈ 오선택)
  missed  = gold_ids − agent_ids   (미제출 = COVERAGE 실패)
결과: ⋈(222)는 소수·지배는 coverage(1121·27%). C77 "⋈ 지배" 철회 근거."""
import json, glob
from collections import Counter
import bank_filter_repro as B


def main():
    per = {}; data = {}
    for f in glob.glob("C:/tmp/traj/*_banking.json"):
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        data[f] = d
        for s in d["simulations"]:
            r = (s.get("reward_info") or {}).get("reward")
            if r is None:
                continue
            t = str(s["task_id"]); per.setdefault(t, [0, 0]); per[t][1] += 1
            if r == 1.0:
                per[t][0] += 1
    hard = {t for t, p in per.items() if p[1] >= 10 and p[0] / p[1] <= 0.10}

    agg = Counter()
    for f, d in data.items():
        for s in d["simulations"]:
            if str(s["task_id"]) not in hard:
                continue
            ri = s.get("reward_info") or {}
            if ri.get("reward") in (None, 1.0):
                continue
            msgs = s.get("messages") or []
            agent_ids = set()
            for m in msgs:
                for tc in (m.get("tool_calls") or []):
                    if tc.get("name") == "call_discoverable_agent_tool":
                        tid = B.nd(B.nd(tc.get("arguments")).get("arguments")).get("transaction_id")
                        if tid:
                            agent_ids.add(str(tid))
            gold_ids = set()
            for ac in (ri.get("action_checks") or []):
                a = ac.get("action") or {}
                if a.get("name") != "call_discoverable_agent_tool":
                    continue
                gt = B.nd(B.nd(a.get("arguments")).get("arguments")).get("transaction_id")
                if gt:
                    gold_ids.add(str(gt))
            if not gold_ids:
                continue
            agg["sims"] += 1
            agg["gold_total"] += len(gold_ids)
            agg["correct"] += len(agent_ids & gold_ids)
            agg["wrong(진짜⋈)"] += len(agent_ids - gold_ids)
            agg["missed(coverage)"] += len(gold_ids - agent_ids)

    g = agg["gold_total"]
    print("hard 실패 sims: %d · gold 요구 총: %d" % (agg["sims"], g))
    print("  correct(제출·맞음)      : %4d (%.1f%%)" % (agg["correct"], 100 * agg["correct"] / max(g, 1)))
    print("  ★wrong(제출·틀림=진짜⋈) : %4d (%.1f%% of gold)" % (agg["wrong(진짜⋈)"], 100 * agg["wrong(진짜⋈)"] / max(g, 1)))
    print("  ★missed(미제출=coverage): %4d (%.1f%% of gold) ← 지배" % (agg["missed(coverage)"], 100 * agg["missed(coverage)"] / max(g, 1)))
    print("∴ transaction_id ⋈ 비지배. 지배 = coverage(미제출). C77 '⋈ 지배' 철회.")


if __name__ == "__main__":
    main()
