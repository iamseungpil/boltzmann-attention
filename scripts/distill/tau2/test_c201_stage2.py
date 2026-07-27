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
    """D4' 재설계(C203): 값-존재 검사 폐기 → **제약 의도(주제어) 실재** 검사."""
    print("D4' intent-field grounding (구 D4 폐기·값-존재 검사 제거):")
    fit = next((t for t in A2["scaffold_get_tools"]
                if t.get("name") == "check_card_application_fit"), None)
    chk(fit is not None and fit.get("ground"), "check_card_application_fit에 ground 선언")
    if not (fit and fit.get("ground")):
        return
    gr = fit["ground"]
    chk("scalar_fields" not in gr, "구 D4(scalar_fields 값-존재 검사) 제거됨")
    itf = {x["param"]: x for x in gr.get("intent_fields", [])}
    chk({"max_annual_fee", "max_fx_fee", "min_credit_limit", "credit_score"} <= set(itf),
        "발명 위험 제약 전부 intent_fields로 전환")
    chk(all(x.get("corpus") == ["user"] for x in itf.values()),
        "corpus=user(도구 출력 제외) — 003 자기-그라운딩 봉쇄")
    chk(all(x.get("on_fail") == "drop" for x in itf.values()), "불성립=드롭(안전 방향)")

    import t2_scaffold_get as SG

    def run(user_text, ctx):
        """_ground_operands를 손님 발화 코퍼스로 실행(엔진 경로 그대로)."""
        orig = SG._corpus_texts
        SG._corpus_texts = lambda orch, which: [user_text]
        try:
            return SG._ground_operands(None, fit, ctx), ctx
        finally:
            SG._corpus_texts = orig

    # 006 실측: 손님은 소득만 말했는데 모델이 min_credit_limit=95000 발명
    u006 = ("Foreign transaction fee: 1.5% or lower. Minimum payment: 1.5% or lower. "
            "Must include virtual card management. My credit score is 540. "
            "My annual income is $95,000.")
    f, c = run(u006, {"min_credit_limit": "95000", "max_fx_fee": "1.5",
                      "max_min_payment_pct": "1.5", "credit_score": "540"})
    chk(c["min_credit_limit"] is None,
        "★006 표적 포착: 손님이 한도 얘기를 한 적 없음 → min_credit_limit 드롭 (구 D4는 통과시켰음)")
    chk(c["max_fx_fee"] == "1.5" and c["credit_score"] == "540",
        "손님이 말한 제약(fx·score)은 보존 = 오탐 0")
    chk(any("min_credit_limit" in x for x in f), "드롭 사유가 플래그로 반환")

    # 003 실측: 정성 표현 → 모델이 0으로 수치화. 구 D4는 '0이 원장에 없다'며 드롭(오탐)
    u003 = ("No foreign transaction fees. Purchase protection. A credit limit that could "
            "potentially be at least $100,000. I have a Rho-Bank+ subscription.")
    f2, c2 = run(u003, {"max_fx_fee": "0", "min_credit_limit": "100000"})
    chk(c2["max_fx_fee"] == "0",
        "★003 오탐 해소: 정성 표현('no foreign transaction fees')도 주제어가 있으므로 보존")
    chk(c2["min_credit_limit"] == "100000", "명시된 한도 요구는 보존")

    # 023 실측: 소득 95000 → max_annual_fee 오전사
    u023 = ("I'd rather switch to something simpler without complicated rebate rules. "
            "My annual income is $95,000 and I have no Rho-Bank+ subscription.")
    f3, c3 = run(u023, {"max_annual_fee": "95000", "credit_score": "700"})
    chk(c3["max_annual_fee"] is None, "★023 표적 포착: 연회비 얘기 없음 → 드롭")
    chk(c3["credit_score"] is None, "코퍼스에 없는 발명값(신용점수 언급 0) → 드롭")


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
