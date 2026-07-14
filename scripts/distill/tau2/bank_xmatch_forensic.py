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

    import re
    fam = lambda nm: re.sub(r"_\d+$", "", str(nm))     # 도구명 숫자접미 제거 = family
    LIM = ["there is a limit", "only file", "one dispute", "limit of", "maximum of",
           "can only file", "per session", "disputes at a time", "file one",
           "one at a time", "can't file all", "cannot file all"]
    agg = Counter(); mdec = Counter()
    for f, d in data.items():
        for s in d["simulations"]:
            if str(s["task_id"]) not in hard:
                continue
            ri = s.get("reward_info") or {}
            if ri.get("reward") in (None, 1.0):
                continue
            msgs = s.get("messages") or []
            agent = set()
            for m in msgs:
                for tc in (m.get("tool_calls") or []):
                    if tc.get("name") == "call_discoverable_agent_tool":
                        a = B.nd(tc.get("arguments"))
                        if "transaction_dispute" in fam(a.get("agent_tool_name")):
                            t = B.nd(a.get("arguments")).get("transaction_id")
                            if t:
                                agent.add(str(t))
            gold = set()
            for ac in (ri.get("action_checks") or []):
                a = ac.get("action") or {}
                if a.get("name") != "call_discoverable_agent_tool":
                    continue
                if "transaction_dispute" in fam(B.nd(a.get("arguments")).get("agent_tool_name")):
                    t = B.nd(B.nd(a.get("arguments")).get("arguments")).get("transaction_id")
                    if t:
                        gold.add(str(t))
            if not gold:
                continue
            agg["sims"] += 1
            agg["gold"] += len(gold)
            agg["correct"] += len(agent & gold)
            agg["wrong"] += len(agent - gold)
            missed = gold - agent
            agg["missed"] += len(missed)
            if missed:
                txt = " ".join(str(m.get("content") or "").lower() for m in msgs)
                c = "A.0제출(미착수·reach/discovery)" if len(agent) == 0 \
                    else ("B.한도언급" if any(k in txt for k in LIM) else "C.부분제출후 미완(F4/F5)")
                mdec[c] += len(missed)

    g = agg["gold"]
    print("=== disputes(credit+debit) id-level · hard 실패 %d sim · gold %d ===" % (agg["sims"], g))
    print("  correct(제출·맞음)      : %4d (%.1f%%)" % (agg["correct"], 100 * agg["correct"] / max(g, 1)))
    print("  ★wrong(제출·틀림=진짜⋈) : %4d (%.1f%%)" % (agg["wrong"], 100 * agg["wrong"] / max(g, 1)))
    print("  ★missed(미제출=coverage): %4d (%.1f%%) ← 지배" % (agg["missed"], 100 * agg["missed"] / max(g, 1)))
    print("  missed 분해:")
    for c in ["A.0제출(미착수·reach/discovery)", "B.한도언급", "C.부분제출후 미완(F4/F5)"]:
        print("    %-32s %4d (%.0f%% of missed)" % (c, mdec[c], 100 * mdec[c] / max(agg["missed"], 1)))
    print("∴ ⋈ 비지배(4%). 지배=coverage(26%)·그중 80%=미착수(reach/discovery)=handoff§1·C52. C77 '⋈ 지배' 철회.")


if __name__ == "__main__":
    main()
