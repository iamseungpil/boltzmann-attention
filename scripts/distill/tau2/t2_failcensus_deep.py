#!/usr/bin/env python
"""실패 궤적 *구체 원인* 전수 분류 — "정확히 뭘 틀리나" 케이스별. (집계 t2_failcensus의 심화)

retail write 과제(이미 학습으로 통과시켰던 것)에서 base가 어디서 틀리는지 구체 범주로:
  A_notfound_gaveup : 틀린 id→"not found"→복구 못하고 포기(write 미도달) = order_id 오류 + P7 복구결함
  B_wrong_write_args: write(exchange/modify/return/cancel) 도달했으나 인자 틀림 = operand/selection
  D_no_attempt      : write 시도 안 함(에러도 아님·미완)
  F_errorloop       : too_many_errors 종료(실패호출 반복)
  E_partial_or_NL   : 일부 write 맞았으나 미충족(2번째 행동 누락 or NL 통신 실패)
  INFO_only         : gold에 write 없음(정보전달 과제)·NL 실패
각 범주 count + 예시 task_id + "not found" 에러율.

Run: PY t2_failcensus_deep.py <sim_dir1> [sim_dir2 ...]
"""
import json
import os
import sys
from collections import Counter, defaultdict

WRITE = {"modify_pending_order_items", "modify_pending_order_address", "modify_pending_order_payment",
         "exchange_delivered_order_items", "return_delivered_order_items", "cancel_pending_order",
         "modify_user_address", "book_reservation", "update_reservation_flights",
         "update_reservation_passengers", "update_reservation_baggages"}


def classify(s):
    ri = s.get("reward_info") or {}
    acs = ri.get("action_checks") or []
    gold_writes = [a for a in acs if (a.get("action") or {}).get("name") in WRITE]
    write_matched = any(a.get("action_match") for a in gold_writes)
    # agent 실제 호출 + 에러
    agent_writes, notfound, n_err = [], 0, 0
    for m in s.get("messages", []):
        if m.get("role") == "assistant":
            for tc in (m.get("tool_calls") or []):
                if tc.get("name") in WRITE:
                    agent_writes.append(tc.get("name"))
        if m.get("role") == "tool":
            c = str(m.get("content") or "")
            if m.get("error") or "rror" in c:
                n_err += 1
            if "not found" in c.lower():
                notfound += 1
    term = s.get("termination_reason")
    nl = ri.get("nl_assertions") or []
    nl_unmet = any(not a.get("met", True) for a in nl)

    if not gold_writes:
        return "INFO_only", notfound
    if not agent_writes:
        if term == "too_many_errors":
            return "F_errorloop", notfound
        if notfound:
            return "A_notfound_gaveup", notfound
        return "D_no_attempt", notfound
    # write 호출함
    if not write_matched:
        if term == "too_many_errors":
            return "F_errorloop", notfound
        return "B_wrong_write_args", notfound
    return "E_partial_or_NL", notfound


def census(sim_dir):
    with open(os.path.join(sim_dir, "results.json"), encoding="utf-8") as f:
        sims = json.load(f).get("simulations", [])
    cat = Counter(); ex = defaultdict(list); nf_total = 0; nfail = 0
    for s in sims:
        if (s.get("reward_info") or {}).get("reward", 1) >= 1:
            continue
        nfail += 1
        c, nf = classify(s)
        cat[c] += 1; nf_total += (1 if nf else 0)
        if len(ex[c]) < 3:
            ex[c].append(s.get("task_id"))
    return {"dir": os.path.basename(sim_dir), "nfail": nfail, "cat": cat, "ex": ex,
            "notfound_tasks": nf_total}


def main():
    for d in sys.argv[1:]:
        r = census(d)
        print(f"\n## {r['dir']}  (실패 {r['nfail']}건)")
        tot = sum(r['cat'].values())
        for c, n in r['cat'].most_common():
            print(f"   {c:20s} {n:4d} ({n/max(tot,1):.2f})  예: {r['ex'][c]}")
        print(f"   ('not found' 에러 겪은 실패 task: {r['notfound_tasks']}/{r['nfail']})")
    print("\n해석: A=order_id오류+복구결함 / B=operand(write 인자) / D=미시도 / F=에러루프 / E=부분·NL / INFO=정보전달")


if __name__ == "__main__":
    main()
