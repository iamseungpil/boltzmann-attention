#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""notice-kind 단위 프로브 — NOTICE-REFUND(§3b CENSUS_LEVERS_DESIGN_2026_07_11) 지원성 판정.

gate_interpreter를 직접 임포트해 두 가지를 고정한다:
  [A] 현행 notice 의미론(단일 게이트) — "지정 문구를 행동 전에 송신해야 통과":
      check(tool, args, transfer_msg_sent=False) → deny / True → allow / None → skip(replay).
  [B] ★한계 실증: notice 판정 입력이 *게이트별*이 아니라 스칼라 1개 — 호출부
      (t2_gate_patch.py:59-60 `_notice_text` = gates 중 *첫* notice의 notice_text,
       :210/:1196/:1314 = 그 한 문구의 송신 여부만 계산 → check()에 스칼라로 전달;
       t2_compliance.py:35·t2_gate.py:40 도 동일한 first-notice 가정)라서,
      notice_text가 다른 2번째 notice 게이트(G8 환불-목적지, applies_to=cancel)는
      A2만으로는 배선 불가:
        · G4 앞(JSON 순서상 G4가 첫 notice): cancel이 G8에 걸리는데 unblock 조건은
          *G4 transfer 문구* 송신 → 환불 고지를 보내도 영구 deny (교착).
        · G8을 첫 notice로: G4 transfer가 *환불 문구* 기준으로 판정 + compliance의
          TRANSFER_MSG가 환불 문구로 바뀜 → G4/위반계측 파괴.
      [B]의 체크는 "현행 코드가 이렇게 동작한다"의 고정 — 엔진이 per-gate notice로
      바뀌면 이 프로브를 갱신하고 G8을 A2에 실제 추가할 것.

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
# §3b 후보 게이트(미채택·프로브 전용) — 정책-사실 정적 문구(동적 값 없음)
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


def caller_notice_text(gates):
    """t2_gate_patch.py:59-60 동형: 호출부가 쓰는 notice_text = *첫* notice 게이트 것."""
    return next((g.get("notice_text") for g in gates if g.get("kind") == "notice"), "")


def caller_sent(assistant_texts, notice_text):
    """t2_gate_patch._transfer_msg_sent(:337)/_regen_transfer_sent(:1061) 동형."""
    if not notice_text:
        return None
    return any(notice_text in t for t in assistant_texts)


def main():
    print("[A] 현행 notice 의미론 (단일 게이트 = G4)")
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
    # 실제 A2(retail)의 notice 게이트로도 동일 의미론 확인
    a2 = load_domain_a2("retail")
    ok("A5 retail A2 로드·notice 게이트 정확히 1개",
       a2 is not None and sum(1 for g in a2["gates"] if g.get("kind") == "notice") == 1)
    gi = GateInterpreter([g for g in a2["gates"] if g.get("kind") == "notice"])
    allowed, gid, _ = gi.check("transfer_to_human_agents", {}, transfer_msg_sent=False)
    ok("A6 retail G4 문구 미송신 → deny", allowed is False and gid == "G4_TRANSFER_MSG")

    print("[B] 한계 실증 — notice 2개(문구 상이)는 현행 배선으로 불가")
    # (B-1) G4 먼저(현 JSON 순서에 G8을 뒤에 append한 경우)
    gates = [G4, G8]
    nt = caller_notice_text(gates)
    ok("B1 호출부 _notice_text = 첫 notice(G4 transfer 문구)만", nt == G4["notice_text"])
    # 에이전트가 §3b 기대 행동을 완수: 환불 고지문을 이미 송신
    sent = caller_sent([G8["notice_text"]], nt)  # 호출부는 G4 문구 기준으로만 판정
    gi = GateInterpreter(gates)
    allowed, gid, _ = gi.check("cancel_pending_order", {}, transfer_msg_sent=sent)
    ok("B2 환불 고지 송신 후에도 cancel 영구 deny (교착·over-block)",
       allowed is False and gid == "G8_REFUND_NOTICE")
    # (B-2) G8을 앞에 두면 이번엔 G4가 파괴됨
    nt2 = caller_notice_text([G8, G4])
    sent2 = caller_sent([G4["notice_text"]], nt2)  # transfer 문구는 보냈지만 판정 기준=환불 문구
    gi2 = GateInterpreter([G8, G4])
    allowed, gid, _ = gi2.check("transfer_to_human_agents", {}, transfer_msg_sent=sent2)
    ok("B3 (역순) transfer 문구 송신에도 G4 deny = 기존 게이트 파괴",
       allowed is False and gid == "G4_TRANSFER_MSG")
    # (B-3) 엔진 check() 자체가 게이트별 문구를 받을 통로가 없음(스칼라 1개)
    import inspect
    sig = inspect.signature(GateInterpreter.check)
    ok("B4 check() notice 입력 = 스칼라 transfer_msg_sent 1개(게이트별 통로 없음)",
       "transfer_msg_sent" in sig.parameters and len(sig.parameters) == 5)

    if FAILS:
        print("FAILED: %s" % FAILS)
        sys.exit(1)
    print("ALL PASS — [A] 현행 의미론 고정, [B] NOTICE-REFUND는 A2-only로 배선 불가(엔진 "
          "per-gate notice 필요). 결론: G8 미추가 (CENSUS §3b 보고 참조).")


if __name__ == "__main__":
    main()
