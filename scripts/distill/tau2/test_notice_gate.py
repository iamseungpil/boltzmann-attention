#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""notice-kind 단위 프로브 — NOTICE-PERGATE (레버2·NEXT_LEVER_GEN_DESIGN §1) 검증.

★2026-07-11 갱신: 엔진이 per-gate notice로 바뀜(check()의 transfer_msg_sent가
callable이면 게이트별 notice_text로 평가·스칼라면 현행과 바이트-동일). 이에 따라
구 [B] "한계 실증"(B1-B4·2-notice 배선 불가)을 **반전** — 2-notice 공존이 PASS 기대.

  [A] 현행 notice 의미론(단일 게이트·스칼라) — 하위호환 불변:
      check(tool, args, transfer_msg_sent=False) → deny / True → allow / None → skip(replay).
  [B] per-gate 의미론(callable 커링 = t2_gate_patch 호출부 동형):
      B2′ 환불 고지 송신 → cancel allow(교착 해소) · B3′ 순서-무관 ·
      B5 G4/G8 독립 판정 · B6 스칼라 2-notice 하위호환 · B4′ callable 통로 실증.
  [C] compliance per-gate 측정 leg(t2_compliance.violations_of_sim) 단위.

주의: G8은 **A2 미부착**(notice_pergate_census 2026-07-11: passing-cancel 가상-deny
16.6% > 2% 상한 → 설계 §1.4 착수 게이트 판정 = 보고만). 이 파일의 G8 = 프로브 전용 초안.

usage: PYTHONIOENCODING=utf-8 py -3 scripts/distill/tau2/test_notice_gate.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate_interpreter import GateInterpreter, load_domain_a2  # noqa: E402

FAILS = []


def ok(name, cond):
    print("  %-64s %s" % (name, "PASS" if cond else "FAIL"))
    if not cond:
        FAILS.append(name)


G4 = {
    "id": "G4_TRANSFER_MSG", "kind": "notice",
    "predicate": "the mandatory transfer notice has been communicated to the user",
    "notice_text": "YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON.",
    "ask": "send the user exactly this message first: \"YOU ARE BEING TRANSFERRED TO A "
           "HUMAN AGENT. PLEASE HOLD ON.\" — then retry the transfer",
    "applies_to": ["transfer_to_human_agents"],
}
# §3b 후보 게이트(A2 미부착·프로브 전용) — 정책-사실 정적 문구(동적 값 없음)
G8 = {
    "id": "G8_REFUND_NOTICE", "kind": "notice",
    "predicate": "the refund-destination policy notice has been communicated to the user",
    "notice_text": "Per policy, the refund for a cancelled order always goes back to the "
                   "original payment method used for that order.",
    "ask": "send the user exactly this message first: \"Per policy, the refund for a "
           "cancelled order always goes back to the original payment method used for that "
           "order.\" — then retry the cancellation",
    "applies_to": ["cancel_pending_order"],
}


def sent_fn(assistant_texts):
    """t2_gate_patch 커링 동형: text -> 송신 여부 callable (게이트별 평가 통로)."""
    return lambda text: (None if not text else any(text in t for t in assistant_texts))


def main():
    print("[A] 스칼라 notice 의미론 (단일 게이트 = G4) — 하위호환 불변")
    gi = GateInterpreter([G4])
    allowed, gid, why = gi.check("transfer_to_human_agents", {}, transfer_msg_sent=False)
    ok("A1 문구 미송신(False) → deny", allowed is False and gid == "G4_TRANSFER_MSG")
    ok("A1b deny reason이 ask 복구절 포함", why is not None and "send the user exactly" in why)
    allowed, _, _ = gi.check("transfer_to_human_agents", {}, transfer_msg_sent=True)
    ok("A2 문구 송신(True) → allow", allowed is True)
    allowed, _, _ = gi.check("transfer_to_human_agents", {}, transfer_msg_sent=None)
    ok("A3 판정불가(None·replay) → skip=allow", allowed is True)
    allowed, _, _ = gi.check("cancel_pending_order", {}, transfer_msg_sent=False)
    ok("A4 applies_to 밖 도구엔 미적용", allowed is True)
    # 실제 A2(retail)의 notice 게이트로도 동일 의미론 확인 (G8=미부착이라 여전히 1개)
    a2 = load_domain_a2("retail")
    ok("A5 retail A2 로드·notice 게이트 정확히 1개(G8 미부착)",
       a2 is not None and sum(1 for g in a2["gates"] if g.get("kind") == "notice") == 1)
    gi = GateInterpreter([g for g in a2["gates"] if g.get("kind") == "notice"])
    allowed, gid, _ = gi.check("transfer_to_human_agents", {}, transfer_msg_sent=False)
    ok("A6 retail G4 문구 미송신 → deny", allowed is False and gid == "G4_TRANSFER_MSG")

    print("[B] per-gate 의미론 — notice 2개(문구 상이) 공존 (구 B1-B4 반전)")
    gates = [G4, G8]
    gi = GateInterpreter(gates)
    # B2′: 환불 고지를 송신했으면 cancel allow (구 B2 교착 해소)
    f = sent_fn([G8["notice_text"]])
    allowed, gid, _ = gi.check("cancel_pending_order", {}, transfer_msg_sent=f)
    ok("B2' 환불 고지 송신 → cancel allow (교착 해소)", allowed is True)
    # 환불 고지만 송신·transfer는 여전히 G4 문구 기준 deny
    allowed, gid, _ = gi.check("transfer_to_human_agents", {}, transfer_msg_sent=f)
    ok("B2'b 환불 고지만으론 transfer 여전히 deny(G4)",
       allowed is False and gid == "G4_TRANSFER_MSG")
    # B3′: 게이트 순서-무관 (G8을 앞에 둬도 G4 판정은 G4 문구 기준 = 구 B3 파괴 해소)
    gi2 = GateInterpreter([G8, G4])
    f2 = sent_fn([G4["notice_text"]])
    allowed, _, _ = gi2.check("transfer_to_human_agents", {}, transfer_msg_sent=f2)
    ok("B3' (역순) transfer 문구 송신 → G4 allow (순서-무관)", allowed is True)
    allowed, gid, _ = gi2.check("cancel_pending_order", {}, transfer_msg_sent=f2)
    ok("B3'b (역순) transfer 문구만으론 cancel deny(G8)",
       allowed is False and gid == "G8_REFUND_NOTICE")
    # B5: 독립 판정 — 두 문구 다 송신하면 둘 다 allow / 둘 다 미송신이면 각자 deny
    fboth = sent_fn([G4["notice_text"], G8["notice_text"]])
    a1, _, _ = gi.check("transfer_to_human_agents", {}, transfer_msg_sent=fboth)
    a2ok, _, _ = gi.check("cancel_pending_order", {}, transfer_msg_sent=fboth)
    ok("B5 두 문구 송신 → transfer/cancel 모두 allow", a1 is True and a2ok is True)
    fnone = sent_fn([])
    a1, g1, _ = gi.check("transfer_to_human_agents", {}, transfer_msg_sent=fnone)
    a2v, g2, _ = gi.check("cancel_pending_order", {}, transfer_msg_sent=fnone)
    ok("B5b 둘 다 미송신 → 각자 자기 게이트로 deny",
       a1 is False and g1 == "G4_TRANSFER_MSG" and a2v is False and g2 == "G8_REFUND_NOTICE")
    # B6: 스칼라 2-notice 하위호환 (False=전 게이트 deny·True=allow·None=skip — 구 동작 보존)
    a1, g1, _ = gi.check("cancel_pending_order", {}, transfer_msg_sent=False)
    ok("B6 스칼라 False → cancel deny(G8·바이트-동일 경로)", a1 is False and g1 == "G8_REFUND_NOTICE")
    a1, _, _ = gi.check("cancel_pending_order", {}, transfer_msg_sent=True)
    a2v, _, _ = gi.check("cancel_pending_order", {}, transfer_msg_sent=None)
    ok("B6b 스칼라 True → allow · None → skip", a1 is True and a2v is True)
    # B4′: callable이 게이트별 notice_text로 호출되는지 (게이트별 통로 실증)
    seen = []
    def probe(text):
        seen.append(text)
        return False
    gi.check("cancel_pending_order", {}, transfer_msg_sent=probe)
    ok("B4' callable에 그 게이트의 notice_text 전달(per-gate 통로)",
       seen == [G8["notice_text"]])

    print("[C] compliance 측정 leg — per-gate 위반 검출 (t2_compliance)")
    from t2_compliance import violations_of_sim, domain_constants
    C = {"AUTH_TOOLS": set(), "AUTH_GATES": [], "WRITE_TOOLS": set(), "USER_SCOPED": set(),
         "TRANSFER_MSG": G4["notice_text"], "NOTICE_GATES": [G4, G8]}
    def sim(msgs):
        return {"messages": msgs}
    def tcall(name, cid):
        return {"role": "assistant", "content": None,
                "tool_calls": [{"id": cid, "name": name, "arguments": {}}]}
    def tres(cid):
        return {"role": "tool", "id": cid, "content": "{}", "error": False}
    # C1: cancel 실행 + 환불 문구 부재 → G8 위반·G4 비위반
    v = violations_of_sim(sim([tcall("cancel_pending_order", "c1"), tres("c1")]), C)
    ok("C1 cancel 실행·환불문구 부재 → G8 위반만",
       v["notice_by_gate"] == {"G4_TRANSFER_MSG": False, "G8_REFUND_NOTICE": True}
       and v["g4"] is True)
    # C2: cancel 실행 + 환불 문구 송신 → 위반 없음
    v = violations_of_sim(sim([
        {"role": "assistant", "content": G8["notice_text"]},
        tcall("cancel_pending_order", "c2"), tres("c2")]), C)
    ok("C2 환불문구 송신 후 cancel → 위반 0", v["g4"] is False)
    # C3: transfer 실행 + G4 문구 부재 → G4 위반 (기존 의미론 연속)
    v = violations_of_sim(sim([tcall("transfer_to_human_agents", "c3"), tres("c3")]), C)
    ok("C3 transfer 실행·G4 문구 부재 → G4 위반",
       v["notice_by_gate"]["G4_TRANSFER_MSG"] is True
       and v["notice_by_gate"]["G8_REFUND_NOTICE"] is False)
    # C4: deny된 호출(POLICY GATE)은 미실행 = 비위반 (기존 의미론 승계)
    v = violations_of_sim(sim([
        tcall("transfer_to_human_agents", "c4"),
        {"role": "tool", "id": "c4", "content": "Error: [POLICY GATE G4] blocked", "error": True}]), C)
    ok("C4 gate-deny된 transfer = 미실행 = 비위반", v["g4"] is False)
    # C5: retail A2 도출 상수 — NOTICE_GATES 1개·구 TRANSFER_MSG export 보존
    Cr = domain_constants("retail")
    ok("C5 retail 상수: NOTICE_GATES 1개 + TRANSFER_MSG 호환 보존",
       len(Cr["NOTICE_GATES"]) == 1
       and Cr["TRANSFER_MSG"] == Cr["NOTICE_GATES"][0]["notice_text"])

    if FAILS:
        print("FAILED: %s" % FAILS)
        sys.exit(1)
    print("ALL PASS — [A] 스칼라 하위호환 불변, [B] per-gate 2-notice 공존 동작, "
          "[C] compliance per-gate 검출. G8 자체는 A2 미부착(census over-block 16.6%>2%).")


if __name__ == "__main__":
    main()
