#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C201 2단 처방 오프라인 검증 (무료·모델 불요·2026-07-26).
설계서 `STAGE2_GATE_DESIGN_2026_07_26.md` §7 검증 계획 1~4·6.
D1 claim_prov 세분 kind / D2 에스컬→transfer chain / D3 예비-창 / D4 fit grounding / A2 로드.
⚠단위통과≠라이브발화([[30]]).
★2026-08-21 수리(낡은 기대 3건 갱신·본문 주석에 박제): ①D1 record_update unbacked 기대는
  C341 센티널(2026-08-08)+완결 저작 submit_ 등재(2026-08-21)로 **의도 반전** ②D2 chain 탐색·
  원장은 98e5efe2(2026-08-05)의 정확명 이행에 정합화 ③resign_th 국소화 기대에 C214(cash-back
  체인 선언) 반영. 이 파일은 t7335 프리플라이트 배터리 밖 — 단독 실행 검정."""
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
        "generic write 주장 = 실효 write(유저 dispute)로 backed (write 센티널 경로)")
    # ★기대 반전 박제(2026-08-21 수리): 원판(C201·2026-07-26)은 "무관한 dispute write가
    #   record_update 주장을 입증하면 안 된다"(unbacked=1)를 세분-kind의 존재 이유로 걸었다.
    #   이후 두 델타가 이 방향을 **의도적으로** 뒤집었다 —
    #   ① C341(2026-08-08)·재판정런 070 t3(2026-08-13): record_update에 `__effective_write__`
    #      센티널 등재. 모델의 kind 라벨이 우리 패턴과 어긋날 때 **실행된 행동을 "안 했다"고
    #      단정**하는 거짓 발화가 DUP 재호출을 제조했다(test_claim_promotion.py DELTA·
    #      t2_gate_patch._claim_unbacked 주석).
    #   ② 완결 저작(2026-08-21·T7335 halfB 050): submit_ 접두를 record_update에 이중 등재
    #      (A2 _note_event_map_completion_2026_08_21 ④ — 백킹 맵이지 분할이 아니다).
    #   ⇒ 이 원장(submit_cash_back_dispute_0589 실재)에서 record_update 주장은 이제 **backed가
    #     설계 정답**이다. 무지목-날조 탐지(해당 계열 실행이 0인 경우)의 현행 커버는
    #     test_claim_backed_write.py·test_claim_promotion.py(DELTA 동결)이고, 이 파일 안에서도
    #     아래 041형(dispute_file·분쟁 도구 실행 0)이 unbacked 검출 생존을 계속 잰다.
    chk(len(GP._claim_unbacked(gran, EMAP, evs, msgs)) == 0,
        "현행(C341 센티널+submit_ 등재): record_update 주장 = backed (구 기대 unbacked=1 폐기·위 주석)")
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
    # ★탐색 갱신(2026-08-21 수리): 98e5efe2(2026-08-05 "Stop inferring which tool a
    #   declaration means")가 after를 무접미 계열명 → **레지스트리 정확명**(initial_transfer_
    #   to_human_agent_0218/_1822·emergency_credit_bureau_incident_transfer_1114)으로 이행했고,
    #   라이브 원장도 _eff_tool_name(접미 제거) → _exact_tool_name 집합이 됐다. 구판의
    #   무접미 정확일치 탐색은 그래서 영구 미스. 여기 startswith는 체인을 **찾는** 용도일 뿐 —
    #   발화 판정(_chain_dispatch)은 라이브와 같은 정확명 집합 대조로 잰다(아래 eff).
    ch = next((c for c in A2["follow_up_chains"]
               if any(str(a).startswith("initial_transfer_to_human_agent")
                      for a in (c.get("after") or []))), None)
    chk(ch is not None, "chain 선언 존재 (정확명 anchor·98e5efe2 이행)")
    if ch is None:
        return
    # 035 실측형: 에스컬 실행됨 · transfer 미호출 — 라이브(_exact_tool_name)와 같은 정확명 원장
    eff = {"KB_search_bm25", "unlock_discoverable_agent_tool",
           "emergency_credit_bureau_incident_transfer_1114"}
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
    # ★기대 갱신(2026-08-21 수리): 원판(C201 rev2)은 "resign_th=1은 이 체인뿐"으로 국소화를
    #   쟀다. C204/C214(2026-07-27·day7 028 실측)가 cash-back 체인에도 같은 사유(사임 턴 부족
    #   → 전역 임계 2로 구조적 미발화)로 resign_th=1을 **의도 선언**했다. 국소화 원칙의 현행
    #   형태 = per-chain 하향은 실측 근거가 있는 체인에만, 나머지는 미선언(전역 기본).
    others = [c for c in A2["follow_up_chains"] if c is not ch]
    declared = [c for c in others if c.get("resign_th") is not None]
    chk(len(declared) == 1 and declared[0].get("resign_th") == 1
        and "submit_cash_back_dispute_0589" in (declared[0].get("after") or []),
        "per-chain resign_th 추가 선언은 cash-back 체인(C214)뿐 — 나머지 미선언(전역 기본·Δspurious 국소화 유지)")


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
