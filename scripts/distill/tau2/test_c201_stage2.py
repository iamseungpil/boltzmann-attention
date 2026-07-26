#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C201 2단 처방 오프라인 검증 (무료·모델 불요·2026-07-26).
설계서 `STAGE2_GATE_DESIGN_2026_07_26.md` §7 검증 계획 1~4·6.
D1 claim_prov 세분 kind / D2 에스컬→transfer chain / D3 예비-창 / D4 fit grounding / A2 로드.
⚠단위통과≠라이브발화([[30]])."""
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_gate_patch as GP  # noqa: E402

A2 = json.load(io.open(os.path.join(HERE, "a2", "banking_knowledge.gate.json"), encoding="utf-8"))
CP = A2["claim_prov"]
EMAP = CP["event_map"]

OK = True


def chk(c, m):
    global OK
    OK &= bool(c)
    print(("  ✓ " if c else "  ✗ ") + m)


class TC:
    def __init__(self, name, args=None):
        self.name, self.arguments = name, (args or {})


class MSG:
    def __init__(self, tcs):
        self.role, self.tool_calls = "assistant", tcs


def ledger(names):
    """도구명 목록 → (evs 집합, messages) — 라이브와 같은 _eff_tool_name 경로."""
    msgs, evs = [], set()
    for n in names:
        tc = TC(n)
        msgs.append(MSG([tc]))
        evs.add(n)
        evs.add(GP._eff_tool_name(tc))
    return evs, msgs


def d1_granular_kinds():
    print("D1 claim_prov 세분 kind (026/028 실측형: 유저 dispute write가 원장에 실재):")
    evs, msgs = ledger(["verify_identity", "log_verification",
                        "give_discoverable_user_tool", "submit_cash_back_dispute_0589"])
    generic = [{"kind": "write", "what": "updated the transaction records"}]
    gran = [{"kind": "record_update", "what": "updated the transaction records"}]
    chk(len(GP._claim_unbacked(generic, EMAP, evs, msgs)) == 0,
        "구판 재현: generic write 주장 = 무관한 dispute write로 backed(=구멍)")
    chk(len(GP._claim_unbacked(gran, EMAP, evs, msgs)) == 1,
        "신판: record_update 주장 = unbacked로 검출")
    evs2, msgs2 = ledger(["verify_identity", "update_transaction_rewards_3847"])
    chk(len(GP._claim_unbacked(gran, EMAP, evs2, msgs2)) == 0,
        "정당 갱신 후 같은 주장 = backed(오탐 0·Δspurious 방어)")
    d_claim = [{"kind": "dispute_file", "what": "filed the disputes"}]
    evs3, msgs3 = ledger(["verify_identity", "log_verification"])
    chk(len(GP._claim_unbacked(d_claim, EMAP, evs3, msgs3)) == 1,
        "041 실측형: 분쟁 도구 0회인데 '접수했다' = unbacked")
    evs4, msgs4 = ledger(["call_discoverable_agent_tool"])
    evs4.add("file_credit_card_transaction_dispute")
    chk(len(GP._claim_unbacked(d_claim, EMAP, evs4, msgs4)) == 0,
        "실제 파일링 후 = backed")
    print("D1 A2 선언:")
    chk("record_update" in EMAP and "dispute_file" in EMAP, "event_map에 세분 kind 등록")
    chk("record_update|dispute_file" in CP["question"], "question kind 열거에 반영")
    chk("MOST SPECIFIC" in CP["question"], "구체성 지시 문구 존재")


def d2_escalation_chain():
    print("D2 에스컬→transfer follow_up_chain:")
    ch = next((c for c in A2["follow_up_chains"]
               if "initial_transfer_to_human_agent" in (c.get("after") or [])), None)
    chk(ch is not None, "chain 선언 존재")
    if ch is None:
        return
    # 035 실측형: 에스컬 실행됨 · transfer 미호출
    eff = {"KB_search", "unlock_discoverable_agent_tool",
           "emergency_credit_bureau_incident_transfer"}
    r = GP._chain_dispatch(ch, eff)
    chk(r is not None and "transfer_to_human_agents" in r[0],
        "에스컬 후 transfer 미호출 → feedback 발화({missing} 치환)")
    chk(r is not None and "escalation protocol still requires further steps" in r[0],
        "문구 양방향(프로토콜 잔여 단계 우선 허용) = 조기 transfer Δspurious 완화")
    eff2 = set(eff) | {"transfer_to_human_agents"}
    chk(GP._chain_dispatch(ch, eff2) is None, "transfer 실호출됨 → 무발화(오탐 0)")
    chk(GP._chain_dispatch(ch, {"KB_search"}) is None, "에스컬 미실행 → 무발화")
    chk(ch.get("resign_th") == 1,
        "rev2 결함2: per-chain resign_th=1 선언(전역 2로는 035형 구조적 미발화)")
    others = [c for c in A2["follow_up_chains"] if c is not ch]
    chk(all(c.get("resign_th") is None for c in others),
        "기존 체인은 resign_th 미선언 = env 기본 유지(Δspurious 국소화)")


def d3_reserve_window():
    print("D3 예비-창 순수함수 (_cpv_window):")
    cap = 3
    chk(GP._cpv_window(True, False, 0, cap, 0, 0, True) == "resign", "cap 여유 → resign 창")
    chk(GP._cpv_window(False, True, 3, cap, 0, 0, True) == "transfer", "transfer 창은 cap과 독립")
    chk(GP._cpv_window(True, False, 3, cap, 0, 0, True) == "reserve",
        "cap 소진 + 예비 미사용 → reserve 창(032 표적)")
    chk(GP._cpv_window(True, False, 3, cap, 0, 1, True) is None, "예비 소진 후 = 무발화(상한 유지)")
    chk(GP._cpv_window(True, False, 3, cap, 0, 0, False) is None,
        "A2 reserve_kinds 미선언 = 구거동 보존(발화 0)")
    chk(GP._cpv_window(False, False, 0, cap, 0, 0, True) is None, "창 이벤트 없음 = 무발화")
    print("D3 행동-kind 판정 (_claim_has_kind):")
    rk = CP.get("reserve_kinds") or []
    chk(GP._claim_has_kind([{"kind": "transfer", "what": "x"}], rk), "transfer=행동-kind")
    chk(GP._claim_has_kind([{"kind": "record_update", "what": "x"}], rk), "record_update=행동-kind")
    chk(not GP._claim_has_kind([{"kind": "search", "what": "x"}], rk),
        "search=비행동 → 예비 소진 안 함")
    chk(not GP._claim_has_kind([], rk), "빈 목록 안전")


def d4_fit_grounding():
    print("D4 fit-도구 operand grounding 선언:")
    fit = next((t for t in A2["scaffold_get_tools"]
                if t.get("name") == "check_card_application_fit"), None)
    chk(fit is not None and fit.get("ground"), "check_card_application_fit에 ground 선언")
    if not (fit and fit.get("ground")):
        return
    params = {s["param"] for s in fit["ground"]["scalar_fields"]}
    chk({"max_annual_fee", "min_cashback", "credit_score"} <= params, "발명 위험 제약 전부 포함")
    chk(all(s.get("on_fail") == "drop" for s in fit["ground"]["scalar_fields"]),
        "on_fail=drop(제약 소멸=안전 방향·값 생성 0)")
    chk(all(s.get("corpus") == ["ledger"] for s in fit["ground"]["scalar_fields"]),
        "corpus=ledger(도구 출력+손님 발화)")
    import t2_scaffold_get as SG
    chk(SG._val_grounded("180000", ["I make about $180,000/year"], "number"),
        "손님이 말한 값 = 통과(오탐 0)")
    chk(not SG._val_grounded("700", ["My annual income is $95,000. I want a simpler card."], "number"),
        "코퍼스에 전혀 없는 발명값(credit_score=700) = 드롭 ← D4의 실제 커버")
    chk(SG._val_grounded("95000", ["My annual income is $95,000. I want a simpler card."], "number"),
        "⚠rev2 결함4 한계 박제: 023의 max_annual_fee=95000(소득 오전사)은 값이 실재해 **통과**한다 "
        "— 의미-오전사는 D4로 못 잡는다(효능 부분 [M])")


def d5_wording():
    print("D5(a) 종단 문구 강화:")
    chk("NOT a substitute for the tool call" in CP["feedback"], "feedback: 말-대신 금지 명시")
    chk("has NOT been done" in CP["feedback"], "feedback: 못 하면 미완료를 밝히라(정정 경로 보존)")
    chk("NOT been performed" in CP["feedback_pending"], "feedback_pending 동형 강화")


def a2_schema():
    print("A2 스키마 로드:")
    chk(isinstance(A2.get("claim_prov", {}).get("event_map"), dict), "event_map dict")
    chk(isinstance(A2.get("follow_up_chains"), list) and len(A2["follow_up_chains"]) >= 3,
        "follow_up_chains 로드")
    for c in A2["follow_up_chains"]:
        chk(bool(c.get("after")) and bool(c.get("requires")), "chain 필수 키(after/requires)")


if __name__ == "__main__":
    d1_granular_kinds()
    d2_escalation_chain()
    d3_reserve_window()
    d4_fit_grounding()
    d5_wording()
    a2_schema()
    print("\n%s" % ("PASS — C201 2단 배선 정상 (라이브 발화는 별도·[[30]])" if OK else "FAIL"))
    sys.exit(0 if OK else 1)
