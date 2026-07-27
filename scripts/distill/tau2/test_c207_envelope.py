#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C207 폭주-방어 레버 오프라인 검증 (2026-07-27·무료·모델 불요).
`RUNAWAY_CONVERSION_DESIGN_2026_07_27` §8-6: A2 봉투-퇴화 / A4 절단 미커밋 / B1 chain 예비 /
C2-a 미보유 기능 약속. ⚠단위통과≠라이브발화([[30]])."""
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_gate_patch as GP  # noqa: E402

A2 = json.load(io.open(os.path.join(HERE, "a2", "banking_knowledge.gate.json"), encoding="utf-8"))
OK = True


def chk(c, m):
    global OK
    OK &= bool(c)
    print(("  ✓ " if c else "  ✗ ") + m)


# ── 픽스처: 006 m4 폭주의 축약 재현(구조 동일: 유효 블록 N개 + 미종결 1개 + CR 퇴화) ──
BLOCK = ('<tool_call>\n{"name": "give_discoverable_user_tool", "arguments": '
         '{"discoverable_tool_name": "apply_for_credit_card", "arguments": '
         '"{\\"card_type\\": \\"EcoCard\\"}"}}</tool_call>\n')
RUNAWAY = BLOCK + (('<tool_call>' + ' ' * 95 + '\n' + BLOCK.split('\n', 1)[1]) * 6) \
          + '<tool_call>\n{"name": "give_discoverable_user_tool", "arg' + ('\r\r\r\n' * 7800)


class TC:
    def __init__(self, name, args=None):
        self.name, self.arguments = name, (args or {})
        self.id = "x"


class AM:
    """AssistantMessage 최소 대역(엔진이 보는 필드만)."""
    def __init__(self, content="", tool_calls=None, finish_reason="stop"):
        self.role, self.content, self.tool_calls = "assistant", content, tool_calls
        self.raw_data = {"choices": [{"finish_reason": finish_reason}]}


def a2_envelope():
    print("A2 봉투-퇴화 게이트 술어:")
    tag = os.environ.get("T2_ENVELOPE_TAG", "<tool_call>")

    def fires(am):
        return (not getattr(am, "tool_calls", None)) and tag in str(getattr(am, "content", "") or "")
    chk(fires(AM(content=RUNAWAY)), "★006 실측형 폭주(봉투 있음·파싱 0) → 발화")
    chk(not fires(AM(content="", tool_calls=[TC("KB_search")])), "정상 도구 호출 응답 → 무발화(오탐 0)")
    chk(not fires(AM(content="Your balance is $1,038.52.")), "봉투 없는 정상 산문 → 무발화")
    chk(not fires(AM(content=RUNAWAY, tool_calls=[TC("KB_search")])),
        "봉투 텍스트가 있어도 파싱된 호출이 있으면 → 무발화(부분 파싱 보호)")
    # 픽스처 자체가 실측 구조와 동형인지
    blocks = re.findall(r"<tool_call>(.*?)</tool_call>", RUNAWAY, re.S)
    ok = 0
    for b in blocks:
        try:
            json.loads(b.strip())
            ok += 1
        except Exception:
            pass
    chk(len(blocks) == 7 and ok == 7,
        "픽스처 동형성: 닫힌 블록 7/7 JSON 유효(=형식 위반이 아니라 정지 실패)")
    chk(RUNAWAY.count("<tool_call>") == 8, "미종결 8번째 블록 존재")


def a4_truncation():
    print("A4 절단 미커밋 술어:")
    def fr(am):
        try:
            return ((getattr(am, "raw_data", None) or {}).get("choices") or [{}])[0].get("finish_reason")
        except Exception:
            return None
    chk(fr(AM(content=RUNAWAY, finish_reason="length")) == "length", "length 절단 → 발화 조건 성립")
    chk(fr(AM(content="ok")) == "stop", "정상 종료 → 무발화")
    chk(fr(AM(content="ok", finish_reason=None)) is None, "finish_reason 부재 → 무발화(안전)")
    # 절단본 대체(리뷰 필수1): 프롬프트에 blob을 싣지 않는다
    lim = 1200
    trunc = RUNAWAY[:lim]
    chk(len(RUNAWAY) > 30000 and len(trunc) <= lim and len(trunc) < len(RUNAWAY) / 20,
        "regen 프롬프트용 절단본 ≤1,200자 (실측 33k 대비 1/25 — 창 초과 방지)")


def b1_chain_reserve():
    print("B1 chain 예비-예산 (_fu_window·반환형 None|normal|reserve):")
    chk(GP._fu_window(0, 3, True, 0, True) == "normal", "cap 여유 → normal")
    chk(GP._fu_window(3, 3, True, 0, True) == "reserve", "★035형: cap 소진+예비 선언+진성 사임 → reserve")
    chk(GP._fu_window(3, 3, True, 0, False) is None,
        "★리뷰 필수2: readloop 변환 턴에서는 예비 소비 안 함")
    chk(GP._fu_window(3, 3, True, 1, True) is None, "예비 소진 후 → 무발화(상한 유지)")
    chk(GP._fu_window(3, 3, False, 0, True) is None, "reserve 미선언 체인만 있으면 → 구거동 보존")
    ch = next((c for c in A2["follow_up_chains"]
               if "initial_transfer_to_human_agent" in (c.get("after") or [])), None)
    chk(ch is not None and ch.get("reserve") is True, "A2: 에스컬→transfer 체인에 reserve 선언")
    chk(sum(1 for c in A2["follow_up_chains"] if c.get("reserve")) == 1,
        "예비 선언은 그 체인 하나뿐(다른 체인 거동 보존)")


def c2a_unavailable():
    print("C2-a 미보유 기능 약속 (집합 대조):")

    class T:
        def __init__(self, n):
            self.name = n
    msgs = [type("M", (), {"tool_calls": [TC("unlock_discoverable_agent_tool",
                                             {"agent_tool_name": "update_transaction_rewards_3847"})]})()]

    class Env:
        class user_tools:
            @staticmethod
            def get_discoverable_tools():
                return ["submit_cash_back_dispute_0589", "get_referral_link"]
    known = GP._known_tool_names([T("KB_search"), T("transfer_to_human_agents")], Env, msgs)
    chk("kb_search" in {k.lower() for k in known}, "도구목록 이름 포함")
    chk("submit_cash_back_dispute" in known,
        "★리뷰 필수3: user-side discoverable(접미사 strip) 포함 → 022/019 오탐 0")
    chk("update_transaction_rewards" in known, "대화서 unlock된 이름 포함")
    up = GP._unavailable_promises(
        [{"kind": "verify", "what": "send a one-time passcode by SMS", "tool": "send_sms_otp"}], known)
    chk(len(up) == 1, "★004/035형: 존재하지 않는 OTP 도구 약속 → 검출")
    chk(not GP._unavailable_promises(
        [{"kind": "dispute_file", "what": "file the dispute", "tool": "submit_cash_back_dispute_0589"}], known),
        "정당한 discoverable 도구 약속 → 무발화")
    chk(not GP._unavailable_promises([{"kind": "verify", "what": "verify you"}], known),
        "tool 미선언 pending → 판정 안 함(하위호환)")
    cp = A2["claim_prov"]
    chk('"tool":' in cp["question"], "A2 question에 pending.tool 스키마 반영")
    chk("UNAVAILABLE-CAPABILITY" in cp.get("feedback_unavailable", ""), "A2 feedback_unavailable 선언")


def wiring():
    print("배선·순서:")
    src = io.open(os.path.join(HERE, "t2_gate_patch.py"), encoding="utf-8").read()
    i_env = src.index("C207/A2·A4")
    chk(i_env < src.index('_fu_cap = int(os.environ.get("T2_FOLLOWUP_CAP"'),
        "★§8-5: A2/A4가 chains보다 먼저(오염된 응답을 뒤 게이트가 판정하지 않게)")
    chk(i_env < src.index("(a1) write-provenance"), "A2/A4가 WRITEPROV보다 먼저")
    chk(i_env < src.index("C193 notice-재발화"), "A2/A4가 NOTICEREP보다 먼저")
    chk("am_override" in src and "regen failed (keeping original)" in src,
        "★리뷰 필수1: regen 프롬프트 절단본 대체 + 예외 흡수(크래시 승격 차단)")
    chk("_t2_fu_readloop_turn" in src, "★리뷰 필수2: readloop 턴 표시 플래그")
    go = io.open(os.path.join(HERE, "go_stack.sh"), encoding="utf-8").read()
    for e in ("T2_ENVELOPE_GUARD=1", "T2_TRUNC_GUARD=1", "T2_UNAVAIL_PROMISE=1"):
        chk(e in go, "go_stack: %s" % e)


if __name__ == "__main__":
    a2_envelope()
    a4_truncation()
    b1_chain_reserve()
    c2a_unavailable()
    wiring()
    print("\n%s" % ("PASS — C207 폭주-방어 배선 정상 (라이브 발화는 별도·[[30]])" if OK else "FAIL"))
    sys.exit(0 if OK else 1)
