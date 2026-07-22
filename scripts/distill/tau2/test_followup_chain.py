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
    check("A2: requires=full set 4체크", isinstance(REQ, list) and len(REQ) == 4)
    check("A2: decision_tools=approve+deny(§2au·052 gold=deny 도구 호출)",
          CHAIN.get("decision_tools") == ["approve_credit_limit_increase", "deny_credit_limit_increase"])
    check("A2: decision 문구=도구-호출 명시+unlock 프로토콜",
          "deny_credit_limit_increase" in CHAIN["decision_feedback"]
          and "unlock_discoverable_agent_tool" in CHAIN["decision_feedback"])

    # ── 050 재현: submit + history만 호출·pending 등 3체크 건너뜀 (구판=미발화·신판=발화) ──
    eff = {"submit_credit_limit_increase_request", "get_credit_limit_increase_history"}
    hit = _chain_dispatch(CHAIN, eff)
    check("050: 부분체크(구판 사각) → 발화", hit is not None and hit[1] == "followup_chain")
    # 누락 나열부 = "missing for this account: <나열>. These checks..." 사이 (신 문구 앵커·e2e9 050 보강판)
    _missing_part = hit[0].split("missing for this account:")[-1].split(". These checks")[0] if hit else ""
    check("050: 누락 3종 전량 나열·기호출 history는 누락 아님", hit is not None
          and all(t in _missing_part for t in ("get_user_dispute_history",
                                               "get_pending_replacement_orders", "get_payment_history"))
          and "get_credit_limit_increase_history" not in _missing_part)
    check("050 보강: 피드백이 unlock 프로토콜 명시(직호출 거부 루프 차단)",
          hit is not None and "unlock_discoverable_agent_tool" in hit[0])

    # ── 054 재현: submit만·체크 0 (query-gap) ──
    hit = _chain_dispatch(CHAIN, {"submit_credit_limit_increase_request"})
    check("054: 체크 0 → 발화·4종 전량 나열", hit is not None
          and all(t in hit[0].split("Per the CLI")[0] for t in REQ))

    # ── suffix-strip 대조: effective-name(_NNNN 제거)은 호출부(_eff_tool_name) 몫 — 여기선 이미 strip된 셋 ──
    eff = {"submit_credit_limit_increase_request"} | set(REQ)
    hit = _chain_dispatch(CHAIN, eff)
    check("전체크 완료·approve 미호출 → decision nudge", hit is not None and hit[1] == "followup_decision")
    check("decision 문구 양방향 도구-호출(approve|deny·§2au)", hit is not None
          and "deny_credit_limit_increase" in hit[0]
          and "approve_credit_limit_increase" in hit[0]
          and "ANY" in hit[0])          # deny 조건(어느 하나라도 실패) 명시

    # ── 전체크+approve 호출됨 → 미발화 (완료 케이스 무간섭) ──
    hit = _chain_dispatch(CHAIN, eff | {"approve_credit_limit_increase"})
    check("전체크+approve → 미발화(무간섭)", hit is None)
    # ── 전체크+deny 호출됨 → 미발화 (052 decline-정답 케이스 무간섭·§2au) ──
    hit = _chain_dispatch(CHAIN, eff | {"deny_credit_limit_increase"})
    check("전체크+deny → 미발화(무간섭)", hit is None)

    # ── after 미호출 → 미발화 (submit 전 사임에 간섭 0) ──
    hit = _chain_dispatch(CHAIN, set(REQ))
    check("after(submit) 미호출 → 미발화", hit is None)

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
