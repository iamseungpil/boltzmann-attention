# -*- coding: utf-8 -*-
"""follow_up_chains 관문2 확장 오프라인 검증 (2026-07-20·§2aa — requires 전량화+종단결정 nudge).

라이브 디스패치 함수(`t2_gate_patch._chain_dispatch`)와 실제 A2 선언을 그대로 잰다([[03b]] 별도구현 금지).
포렌식 재현: 050(=history 호출·pending 미호출→구판 단일 requires는 미발화·신판은 누락 나열)·
054(query-gap=체크 0)·052-보호(decline 정답: 전체크 후 approve 미호출→decision nudge가 양방향 문구).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from t2_gate_patch import _chain_dispatch  # noqa: E402

A2 = json.load(open(os.path.join(HERE, "a2", "banking_knowledge.gate.json"), encoding="utf-8"))
CHAIN = A2["follow_up_chains"][0]
REQ = CHAIN["requires"]

PASS, FAIL = [], []


def check(label, cond):
    (PASS if cond else FAIL).append(label)
    print(("  PASS " if cond else "  FAIL ") + label)


def main():
    # ★§2bv 강건화: requires=submit + 4체크(5)·after=[submit, check_cli_eligibility] 리스트
    check("A2: requires=submit+4체크(5)", isinstance(REQ, list) and len(REQ) == 5
          and REQ[0] == "submit_credit_limit_increase_request")
    check("A2: after=리스트[submit, check_cli](강건 anchor)",
          isinstance(CHAIN.get("after"), list) and "check_cli_eligibility" in CHAIN["after"])
    check("A2: decision_tools=approve+deny",
          CHAIN.get("decision_tools") == ["approve_credit_limit_increase", "deny_credit_limit_increase"])
    check("A2: decision 문구=deny 도구+DISCOVERABLE 명시",
          "deny_credit_limit_increase" in CHAIN["decision_feedback"]
          and "DISCOVERABLE" in CHAIN["decision_feedback"])
    CHK = ["get_user_dispute_history", "get_pending_replacement_orders",
           "get_credit_limit_increase_history", "get_payment_history"]

    # ── 052 강건화: check_cli anchor·submit 스킵 경로 → 발화·submit이 missing ──
    hit = _chain_dispatch(CHAIN, {"check_cli_eligibility"})
    check("052강건: check_cli anchor·submit 스킵 → 발화", hit is not None and hit[1] == "followup_chain")
    check("052강건: submit 미실행이 missing에 뜸(절차 대체 방지)",
          hit is not None and "submit_credit_limit_increase_request" in hit[0])
    check("052강건: 문구=check_cli는 판정보조·절차대체 아님 명시",
          hit is not None and "check_cli_eligibility" in hit[0] and "replace" in hit[0].lower())

    # ── 050 재현: submit + history만·pending 등 3체크 건너뜀 → 발화·누락 나열 ──
    eff = {"submit_credit_limit_increase_request", "get_credit_limit_increase_history"}
    hit = _chain_dispatch(CHAIN, eff)
    _missing_part = hit[0].split("missing:")[-1].split(". The customer")[0] if hit else ""
    check("050: 부분체크 → 발화·누락 3종 나열·기호출 제외", hit is not None and hit[1] == "followup_chain"
          and all(t in _missing_part for t in ("get_user_dispute_history",
                                               "get_pending_replacement_orders", "get_payment_history"))
          and "get_credit_limit_increase_history" not in _missing_part)

    # ── 전체크 완료(submit+4체크)·decision 미호출 → decision nudge(양방향) ──
    eff = set(REQ)
    hit = _chain_dispatch(CHAIN, eff)
    check("전체크 완료·decision 미호출 → nudge", hit is not None and hit[1] == "followup_decision")
    check("decision 문구 양방향(approve|deny)", hit is not None
          and "deny_credit_limit_increase" in hit[0]
          and "approve_credit_limit_increase" in hit[0]
          and "NOT_ELIGIBLE" in hit[0])

    # ── 완료 케이스 무간섭 ──
    check("전체크+approve → 미발화", _chain_dispatch(CHAIN, eff | {"approve_credit_limit_increase"}) is None)
    check("전체크+deny → 미발화(052 decline-정답)", _chain_dispatch(CHAIN, eff | {"deny_credit_limit_increase"}) is None)

    # ── anchor(submit·check_cli) 둘 다 미호출 → 미발화 (4체크만·간섭 0) ──
    hit = _chain_dispatch(CHAIN, set(CHK))
    check("anchor 둘 다 미호출 → 미발화", hit is None)

    # ── 하위호환: 문자열 requires (구판 A2 선언 형태) ──
    old = {"after": "A", "requires": "B", "feedback": "missing: {missing}"}
    check("하위호환: 문자열 requires 발화", _chain_dispatch(old, {"A"}) == ("missing: B", "followup_chain"))
    check("하위호환: 충족 시 미발화(decision 미선언)", _chain_dispatch(old, {"A", "B"}) is None)

    # ══ closure chain 재설계 (2026-07-22 §2bs·rall10 043+KB 리텐션 프로토콜 재정독) ══
    CH2 = A2["follow_up_chains"][1]
    R2 = CH2["requires"]
    check("closure: requires에 dispute-체크 포함(logistics_003 Step1.1)",
          "get_user_dispute_history" in R2)
    check("closure: close는 requires에서 제거(리텐션 분기 보존)",
          "close_credit_card_account" not in R2)
    check("closure: 종단=decision_tools 양방향(waive|close)",
          CH2.get("decision_tools") == ["apply_credit_card_account_flag", "close_credit_card_account"])
    # 043 재현: eligibility 후 all_accounts+pay만 완료(rall10 도달점) → 잔여 나열
    eff2 = {"check_card_closure_eligibility", "get_all_user_accounts_by_user_id",
            "pay_credit_card_from_checking"}
    hit = _chain_dispatch(CH2, eff2)
    check("closure-043: 부분진행 → 발화·잔여 나열", hit is not None and hit[1] == "followup_chain"
          and all(t in hit[0] for t in ("get_user_dispute_history", "get_pending_replacement_orders",
                                        "get_closure_reason_history", "log_credit_card_closure_reason")))
    # 전체크 완료·종단 미실행 → decision nudge(양방향 문구)
    hit = _chain_dispatch(CH2, {"check_card_closure_eligibility"} | set(R2))
    check("closure: 전체크·종단 미실행 → decision nudge", hit is not None and hit[1] == "followup_decision"
          and "apply_credit_card_account_flag" in hit[0] and "close_credit_card_account" in hit[0])
    # 종단 실행(waive 또는 close) → 미발화
    check("closure: waive 실행 → 미발화",
          _chain_dispatch(CH2, {"check_card_closure_eligibility"} | set(R2)
                          | {"apply_credit_card_account_flag"}) is None)
    check("closure: close 실행 → 미발화",
          _chain_dispatch(CH2, {"check_card_closure_eligibility"} | set(R2)
                          | {"close_credit_card_account"}) is None)

    print("\n== 결과: %d PASS / %d FAIL ==" % (len(PASS), len(FAIL)))
    if FAIL:
        for f in FAIL:
            print("  - FAILED: " + f)
        sys.exit(1)
    print("ALL PASS — 관문2: full required-set 나열 + 종단결정 nudge + 무간섭 케이스 보존.")


if __name__ == "__main__":
    main()
