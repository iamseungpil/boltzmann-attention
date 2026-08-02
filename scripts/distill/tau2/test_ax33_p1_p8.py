#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AX33 P1(give-인용)·P8(제출-완결 카운터) 오프라인 검증 (2026-08-03·무료·모델 불요).

P1 = `AX32_MIDRUN_PRESCRIPTIONS_DESIGN_2026_08_02` §P1 (010 표적·재현 2/2).
     술어 = 응답 본문에 **손님 발화의 토큰-연속 부분열**이 실재하는가(닫힘·[[22]]).
     ☠채널: 인용을 give **인자에 얹지 않는다**(여분 키가 evaluator exact-match를 깨뜨린 실측).
P8 = 같은 문서 §P8 (020/027 표적). 대상 집합(엔진 산출 ids) vs 실효-write 인자 leaf 대조.

⚠단위통과≠라이브발화([[30]])."""
import io
import json
import os
import sys
from types import SimpleNamespace as NS

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_gate_patch as GP        # noqa: E402
import t2_eplan_patch as EP       # noqa: E402

OK = True


def chk(c, m):
    global OK
    OK &= bool(c)
    print(("  ✓ " if c else "  ✗ ") + m)


A2 = json.load(io.open(os.path.join(HERE, "a2", "base", "shared.json"), encoding="utf-8"))
NOTES = A2["axis_notes"]


# ── P1 ───────────────────────────────────────────────────────────────────────
def test_p1_predicate():
    print("[test_p1] give-인용 실재성 술어(토큰-연속 부분열·닫힘)")
    user = "Hi, can you give me the tool to file a dispute for that charge? thanks"
    chk(GP._shared_span("Sure — you said you want the tool to file a dispute, here it is.",
                        user, 4),
        "축자 4토큰 인용 → 성립")
    chk(not GP._shared_span("I have prepared a referral link for you, it may be useful.",
                            user, 4),
        "손님이 말한 적 없는 것을 건네는 발화 → 불성립(010 기전)")
    chk(not GP._shared_span("the tool to", user, 4), "min_tokens 미만 부분열 → 불성립(우연 차단)")
    chk(GP._shared_span("...THE TOOL, TO FILE A DISPUTE!!!", user, 4),
        "정규화(대소문자·문장부호)는 닫힌 연산으로만 흡수")
    chk(not GP._shared_span("dispute file to tool the", user, 4),
        "토큰 뒤섞기(순서 파괴)는 인용이 아니다 — 연속성 요구")
    chk(isinstance(NOTES.get("give_quote"), str) and "{tool}" in NOTES["give_quote"]
        and "{min}" in NOTES["give_quote"], "A2 문구 실재(도구명·최소토큰 치환자)")
    chk(int(NOTES.get("give_quote_min_tokens") or 0) >= 3, "A2 min_tokens 선언")


def test_p1_install_path():
    """★死코드 사고 재발 방지(설계서 §2.0 규약): 새 생성-레벨 레버는 **승자 설치자
    `unified()` 내부**여야 한다(patched/gen_gated는 라이브서 설치되지 않는다)."""
    print("[test_p1] 설치 경로 = unified() 내부")
    src = io.open(os.path.join(HERE, "t2_gate_patch.py"), encoding="utf-8").read()
    i_gq = src.find('T2_GIVE_QUOTE')
    i_uni = src.find("    def unified(self, message, state):")
    i_end = src.find("    LLMAgent._generate_next_message = unified")
    chk(i_uni < i_gq < i_end, "T2_GIVE_QUOTE 분기가 unified() span 안에 있다")
    chk("_args_dict(_giv1)" in src and "arguments" not in src[i_gq:i_gq + 1200].split(
        "_ap_regen")[0].replace("_args_dict", ""), "인용을 도구 인자에 얹지 않는다(채널 제약)")


# ── P8 ───────────────────────────────────────────────────────────────────────
class M:
    def __init__(self, role, content=None, tool_calls=None, mid=None, error=False):
        self.role, self.content, self.tool_calls = role, content, tool_calls
        self.id, self.error = mid, error


class TC:
    def __init__(self, name, args, cid="c1"):
        self.name, self.arguments, self.id = name, args, cid


class Orch:
    def __init__(self, msgs, ledger):
        self._m, self._t2_dispatch_ledger = msgs, ledger
        self.environment = NS(domain_name="banking_knowledge")

    def get_messages(self):
        return self._m


def _sub(txn, cid="w1"):
    return TC("call_discoverable_agent_tool",
              {"agent_tool_name": "file_credit_card_transaction_dispute_6281",
               "arguments": {"transaction_id": txn, "user_id": "u"}}, cid)


def test_p8():
    print("[test_p8] 제출-완결 대조(대상 집합 vs 실효-write 인자 leaf)")
    led = {"get_reward_discrepancies": ["txn_a", "txn_b", "txn_c"]}
    # 부분 제출(2/3) → 잔여 지목
    msgs = [M("assistant", None, tool_calls=[_sub("txn_a", "w1"), _sub("txn_b", "w2")]),
            M("tool", "ok", mid="w1"), M("tool", "ok", mid="w2")]
    got = EP._dispatch_ledger_check(Orch(msgs, led))
    chk(got is not None and got[1] == 2 and got[2] == ["txn_c"],
        "부분 제출 감지: submitted=%s remaining=%s" % (got[1] if got else None,
                                                      got[2] if got else None))
    # 전량 제출 → 무발화(Δspurious 0)
    msgs2 = msgs + [M("assistant", None, tool_calls=[_sub("txn_c", "w3")]),
                    M("tool", "ok", mid="w3")]
    chk(EP._dispatch_ledger_check(Orch(msgs2, led)) is None, "전량 제출 → 무발화")
    # 읽기 도구에만 등장한 id는 '제출'이 아니다
    msgs3 = [M("assistant", None,
               tool_calls=[TC("get_credit_card_transactions_by_user", {"user_id": "u"}, "r1")]),
             M("tool", "txn_a txn_b txn_c", mid="r1")]
    got3 = EP._dispatch_ledger_check(Orch(msgs3, led))
    chk(got3 is not None and len(got3[2]) == 3, "read 경유 등장 ≠ 제출(실효-write만 계상)")
    # 에러로 끝난 write는 제출이 아니다(F2 규율 동형)
    msgs4 = [M("assistant", None, tool_calls=[_sub("txn_a", "w1")]),
             M("tool", "denied", mid="w1", error=True)]
    got4 = EP._dispatch_ledger_check(Orch(msgs4, led))
    chk(got4 is not None and "txn_a" in got4[2], "에러 write → 미제출로 계상")
    # 원장 없음 = 무발화(미선언 도메인 거동보존)
    chk(EP._dispatch_ledger_check(Orch(msgs, {})) is None, "원장 미등재 → 무발화(거동보존)")


def test_p8_a2():
    print("[test_p8] A2 선언·터미널 훅 배선")
    a2 = json.load(io.open(os.path.join(HERE, "a2", "banking_knowledge.gate.json"),
                           encoding="utf-8"))
    rd = next(t for t in a2["scaffold_get_tools"] if t["name"] == "get_reward_discrepancies")
    chk(rd.get("dispatch_targets") is True, "A2 dispatch_targets 선언")
    ep = io.open(os.path.join(HERE, "t2_eplan_patch.py"), encoding="utf-8").read()
    chk(ep.index("T2_DISPATCH_LEDGER") < ep.index('os.environ.get("T2_TERM_GRANT") == "1"\n'
                                                  '                        and not getattr'),
        "P8이 TERM_GRANT보다 먼저 판정(이관 전에 미제출을 알린다)")
    chk('os.environ.get("T2_DISPATCH_LEDGER") != "1"' in ep, "apply() 게이트에 P8 조건 포함")
    sg = io.open(os.path.join(HERE, "t2_scaffold_get.py"), encoding="utf-8").read()
    chk("_t2_dispatch_ledger" in sg, "scaffold가 대상 집합을 등재")


# ── P2/P10 (alltools 재설계판·2026-08-03) ────────────────────────────────────
def test_p2_p10():
    print("[test_p2/p10] bm25 전-0점 신호(닫힘·env 기계 포맷) + 주장-부분열 라우팅")
    z = ("1. Internal: X\n   ID: d1\n   Score: 0.0000\n   Content: a\n"
         "2. Internal: Y\n   ID: d2\n   Score: 0.0000\n   Content: b")
    chk(GP._kb_zero_hit(z) is True, "전-0점 → True(무의미 질의 실측 포맷)")
    chk(GP._kb_zero_hit(z.replace("0.0000", "0.2119", 1)) is False, "일부 득점 → False")
    chk(GP._kb_zero_hit("Found 3 record(s) in 'x'") is None,
        "점수 행 없는 채널(레코드 덤프·dense 아님) → 판정 불가(None)=오판 안 함")
    chk(GP._kb_zero_hit(None) is None, "비-문자열 → None")
    chk(isinstance(NOTES.get("kb_nohit"), str) and "{n}" in NOTES["kb_nohit"], "P2 A2 문구")
    chk("{query}" in (NOTES.get("kb_claim_nohit") or ""), "P10 A2 문구(질의 에코)")
    # 라우팅: 질의가 손님 발화의 축자 부분열이면 P10 문구, 아니면 P2 문구
    user = "my card was charged a foreign transaction fee twice on the same purchase"
    chk(GP._shared_span("foreign transaction fee twice", user, 4), "주장-질의 → P10 라우팅 성립")
    chk(not GP._shared_span("annual fee rebate threshold", user, 4), "무관 질의 → P2 라우팅")
    src = io.open(os.path.join(HERE, "t2_gate_patch.py"), encoding="utf-8").read()
    i = src.find("T2_KB_NOHIT_SURFACE")
    chk(src.find("    def unified(self, message, state):") < i
        < src.find("    LLMAgent._generate_next_message = unified"),
        "설치 경로 = unified() 내부(死코드 규약)")
    chk("_ap_regen" in src[i:i + 3000], "채널 = 생성-레벨 regen(KB_search는 mutating이라 출력-부착 금지)")


for fn in (test_p1_predicate, test_p1_install_path, test_p8, test_p8_a2, test_p2_p10):
    fn()
print(chr(10) + ("ALL PASS" if OK else "FAIL"))
sys.exit(0 if OK else 1)
