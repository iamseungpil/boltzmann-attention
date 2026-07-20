# -*- coding: utf-8 -*-
"""관문5 (038 transfer-escape·§2ad) 오프라인 검증 — CLAIM_PROV 미래형(pending) + transfer-window.

라이브 함수(_claim_unbacked·_is_transfer_call)와 실제 A2 claim_prov를 그대로 잰다([[03b]]).
038 재현: "I will file 3 disputes"(SAY·미실행) + transfer 호출 → 구판(과거형만·resign창만)=사각 →
신판: transfer창 열림 + pending 원장대조 → 미이행 적발. 정당 즉시-transfer(약속 0)=무간섭.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from t2_gate_patch import _claim_unbacked, _is_transfer_call  # noqa: E402

A2 = json.load(open(os.path.join(HERE, "a2", "banking_knowledge.gate.json"), encoding="utf-8"))
CPV = A2["claim_prov"]
EMAP = CPV["event_map"]

PASS, FAIL = [], []


def check(label, cond):
    (PASS if cond else FAIL).append(label)
    print(("  PASS " if cond else "  FAIL ") + label)


class _TC:
    def __init__(self, name, arguments=None):
        self.name, self.arguments = name, (arguments or {})


class _Msg:
    def __init__(self, tool_calls=None, role="assistant", content=""):
        self.tool_calls, self.role, self.content = (tool_calls or []), role, content


def main():
    check("A2: question v2에 pending 요구", "pending" in CPV["question"] and "WILL do" in CPV["question"])
    check("A2: feedback_pending 선언", bool(CPV.get("feedback_pending"))
          and "{claims}" in CPV["feedback_pending"])

    # ── transfer-window 판정 (_is_transfer_call·event_map.transfer 재사용) ──
    am_tr = _Msg(tool_calls=[_TC("transfer_to_human_agents", {"summary": "dispute"})])
    check("transfer 호출 → 창 열림", _is_transfer_call(am_tr, EMAP))
    am_no = _Msg(tool_calls=[_TC("get_all_accounts")])
    check("일반 도구 호출 → 창 안 열림", not _is_transfer_call(am_no, EMAP))
    check("호출 없음(사임) → 이 판정은 False(창은 _resign이 엶)", not _is_transfer_call(_Msg(), EMAP))

    # ── 038 재현: pending "file disputes"(kind=write)·원장에 실효 write 0 → 미입증 ──
    msgs_038 = [_Msg(tool_calls=[_TC("KB_search_bm25", {"q": "dispute"})]),
                _Msg(tool_calls=[_TC("get_credit_card_transactions_by_user")])]
    evs = {"KB_search_bm25", "get_credit_card_transactions_by_user"}
    pend = [{"kind": "write", "what": "file 3 transaction disputes"},
            {"kind": "write", "what": "cancel the credit card"}]
    unb = _claim_unbacked(pend, EMAP, evs, msgs_038)
    check("038: 미이행 약속 2건 전부 적발", len(unb) == 2)

    # ── 약속 이행됨: 실효 write(디스패처 file_dispute) 존재 → 입증 ──
    msgs_done = msgs_038 + [_Msg(tool_calls=[
        _TC("call_discoverable_agent_tool",
            {"agent_tool_name": "file_credit_card_transaction_dispute_4829"})])]
    unb = _claim_unbacked(pend, EMAP, evs, msgs_done)
    check("이행후: __effective_write__ 대조 통과", len(unb) == 0)

    # ── 정당 즉시-transfer: pending 없음 → 무간섭 ──
    check("정당 transfer: pending 0 → 미입증 0", _claim_unbacked([], EMAP, evs, msgs_038) == [])

    # ── 과거형 claims 회귀 (기존 경로 보존) ──
    cl = [{"kind": "search", "what": "checked the knowledge base"}]
    check("회귀: search 주장·KB_ 원장 실재 → 입증", _claim_unbacked(cl, EMAP, evs, msgs_038) == [])
    check("회귀: search 주장·원장 0 → 미입증",
          len(_claim_unbacked(cl, EMAP, {"get_all_accounts"}, msgs_038)) == 1)
    check("회귀: 미등재 kind=skip",
          _claim_unbacked([{"kind": "meditate", "what": "?"}], EMAP, set(), msgs_038) == [])

    # ── give 약속(031 인접): give_ 패턴 대조 ──
    pend_g = [{"kind": "give", "what": "give customer the last-4 tool"}]
    check("give 약속·원장 0 → 미입증", len(_claim_unbacked(pend_g, EMAP, evs, msgs_038)) == 1)
    check("give 약속·give_ 실재 → 입증",
          _claim_unbacked(pend_g, EMAP, evs | {"give_discoverable_user_tool"}, msgs_038) == [])

    print("\n== 결과: %d PASS / %d FAIL ==" % (len(PASS), len(FAIL)))
    if FAIL:
        for f in FAIL:
            print("  - FAILED: " + f)
        sys.exit(1)
    print("ALL PASS — 관문5: transfer창+pending 원장대조·정당 transfer 무간섭·과거형 회귀 보존.")


if __name__ == "__main__":
    main()
