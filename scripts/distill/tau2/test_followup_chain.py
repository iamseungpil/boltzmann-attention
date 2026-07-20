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
    check("A2: decision_tools 선언", CHAIN.get("decision_tools") == ["approve_credit_limit_increase"])

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
    check("decision 문구 양방향(approve|decline 명시·052 보호)", hit is not None
          and "declined" in hit[0] and "Declining is a valid outcome" in hit[0]
          and "approve_credit_limit_increase" in hit[0])

    # ── 전체크+approve 호출됨 → 미발화 (완료 케이스 무간섭) ──
    hit = _chain_dispatch(CHAIN, eff | {"approve_credit_limit_increase"})
    check("전체크+approve → 미발화(무간섭)", hit is None)

    # ── after 미호출 → 미발화 (submit 전 사임에 간섭 0) ──
    hit = _chain_dispatch(CHAIN, set(REQ))
    check("after(submit) 미호출 → 미발화", hit is None)

    # ── 하위호환: 문자열 requires (구판 A2 선언 형태) ──
    old = {"after": "A", "requires": "B", "feedback": "missing: {missing}"}
    check("하위호환: 문자열 requires 발화", _chain_dispatch(old, {"A"}) == ("missing: B", "followup_chain"))
    check("하위호환: 충족 시 미발화(decision 미선언)", _chain_dispatch(old, {"A", "B"}) is None)

    print("\n== 결과: %d PASS / %d FAIL ==" % (len(PASS), len(FAIL)))
    if FAIL:
        for f in FAIL:
            print("  - FAILED: " + f)
        sys.exit(1)
    print("ALL PASS — 관문2: full required-set 나열 + 종단결정 nudge + 무간섭 케이스 보존.")


if __name__ == "__main__":
    main()
