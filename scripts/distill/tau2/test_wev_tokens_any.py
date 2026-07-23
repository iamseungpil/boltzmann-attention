#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""무료 오프라인 selftest (모델 불요) — WEV `require_tokens_any`(라벨-값 인접·{id} 치환) 검증.
C121(rall19 031.1): 구 AND-substring이 KB doc PIN 예시 "(e.g., 1234, 4321)"와 충돌해 날조값
통과 → 라벨-값 인접 토큰으로 교정. shipped A2 spec으로 fire/no-fire 전수.
Run: py -3 test_wev_tokens_any.py"""
import json
import os
import sys
from types import SimpleNamespace as NS

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t2_gate_patch as G  # noqa: E402

A2 = json.load(open(os.path.join(HERE, "a2", "banking_knowledge.gate.json"), encoding="utf-8"))


def _find_specs(o):
    if isinstance(o, dict):
        if "write_evidence_specs" in o:
            return o["write_evidence_specs"]
        for v in o.values():
            r = _find_specs(v)
            if r is not None:
                return r
    elif isinstance(o, list):
        for v in o:
            r = _find_specs(v)
            if r is not None:
                return r


SPECS = _find_specs(A2)


def tm(c):
    return NS(role="tool", content=c, tool_calls=None, error=False, id=None)


def file_tc(inner):
    return NS(name="call_discoverable_agent_tool", id="x",
              arguments={"agent_tool_name": "file_credit_card_transaction_dispute_4829",
                         "arguments": json.dumps(inner)})


# rall19 실측 문자열 포맷 3종
DOC_COLLISION = ("1. How to Find Your Rho-Bank Credit Card Numbers Online\nID: doc_013\n"
                 "Content: ... use the get_card_last_4_digits tool ... PIN meets security "
                 "requirements:\n- Must be exactly 4 digits\n- Cannot be sequential "
                 "(e.g., 1234, 4321)\n- Cannot be all the same")
USER_TOOL_OUT = ("Card information retrieved successfully.\n\nExecuted: get_card_last_4_digits\n"
                 "Arguments: {\n  \"credit_card_account_id\": \"cc_890389b165_silver\"\n}\n"
                 "Last 4 digits of card: 5320")
RECORD_OUT = ("Found 1 record(s) in 'credit_cards':\n\n1. Record ID: cc_x\n"
              "   credit_card_type: Platinum\n   card_last_4_digits: 4821\n   status: active")


def deny(msgs, digits, omit_key=False):
    inner = {"transaction_id": "txn_1", "card_action": "keep_active"}
    if not omit_key:
        inner["card_last_4_digits"] = digits
    return G._wev_deny_msgs(msgs, file_tc(inner), SPECS)


CASES = [
    # (이름, 기대 deny?, msgs, digits, omit_key)
    ("collision_1234_doc_only(031.1 재현)", True, [tm(DOC_COLLISION)], "1234", False),
    ("legit_user_tool_5320(031.0 [34])", False, [tm(DOC_COLLISION), tm(USER_TOOL_OUT)], "5320", False),
    ("legit_record_4821", False, [tm(RECORD_OUT)], "4821", False),
    ("fabricated_9999_despite_sources", True, [tm(USER_TOOL_OUT), tm(RECORD_OUT)], "9999", False),
    ("empty_value_deny(§2bc 유지)", True, [tm(USER_TOOL_OUT)], "", False),
    ("key_absent_skip(변형 오차단 회피 유지)", False, [tm(DOC_COLLISION)], None, True),
]


def synthetic():
    ok = True
    for name, want, msgs, digits, omit in CASES:
        fb = deny(msgs, digits, omit)
        got = fb is not None
        st = "PASS" if got == want else "FAIL"
        ok &= (got == want)
        print("[%s] %s deny=%s%s" % (st, name, got, ("  fb=" + str(fb)[:80]) if fb and got != want else ""))
    return ok


def evidence_quote():
    """C122 {evidence} 인용: 031.0 재현 — 정답(5320)이 도구출력 실재·오값/빈값 시도 → deny
    피드백에 정답 라인 축자 인용. 정답 부재 시엔 빈 인용(날조 유도 금지)."""
    ok = True
    fb = deny([tm(USER_TOOL_OUT)], "9999")
    good = fb is not None and "Last 4 digits of card: 5320" in fb
    print("[%s] evidence_quote_wrong_value(031.0 재현): 정답 라인 인용=%s" % ("PASS" if good else "FAIL", good))
    ok &= good
    fb = deny([tm(USER_TOOL_OUT)], "")
    good = fb is not None and "5320" in fb
    print("[%s] evidence_quote_empty_value: 정답 라인 인용=%s" % ("PASS" if good else "FAIL", good))
    ok &= good
    fb = deny([tm(DOC_COLLISION)], "1234")
    good = fb is not None and "{evidence}" not in fb and "5320" not in fb
    print("[%s] evidence_quote_no_source: 빈 인용·플레이스홀더 잔존 없음=%s" % ("PASS" if good else "FAIL", good))
    ok &= good
    return ok


def replay():
    """rall19 실궤적 재검 — sim_results gz가 있으면 실행([M] 승급 근거)."""
    import gzip
    p = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "sim_results",
                     "bank_rall19_treat_20260723.results.json.gz")
    if not os.path.exists(p):
        print("[skip] replay: gz 없음")
        return True
    data = json.load(gzip.open(p))
    by = {}
    for s in data.get("simulations", []):
        msgs = [NS(role=m.get("role"), content=m.get("content"), error=m.get("error"))
                for m in (s.get("messages") or [])]
        by[(str(s.get("task_id")), s.get("trial"))] = msgs
    checks = [("task_031", 1, "1234", True, "031.1 날조 1234 → 이제 deny"),
              ("task_031", 0, "5320", False, "031.0 진짜 5320 → 통과(무회귀)"),
              ("task_039", 0, "1652", False, "039.0 진짜 1652 → 통과(무회귀)")]
    ok = True
    for tid, tr, digits, want, label in checks:
        msgs = by.get((tid, tr))
        if msgs is None:
            print("[skip] replay %s.%s 없음" % (tid, tr))
            continue
        got = deny(msgs, digits) is not None
        st = "PASS" if got == want else "FAIL"
        ok &= (got == want)
        print("[%s] replay %s" % (st, label))
    return ok


if __name__ == "__main__":
    ok = synthetic() & evidence_quote() & replay()
    print("ALL PASS" if ok else "FAILURES")
    sys.exit(0 if ok else 1)
