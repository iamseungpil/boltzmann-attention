#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""축-레버 단위 검정 — `FAILURE_AXES_REDESIGN` §5 케이스 매트릭스 사전 등록분."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import t2_axis_levers as AX      # noqa: E402

TPL = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "a2", "base", "shared.json"), encoding="utf-8"))["axis_notes"]
SC = {"get_reward_discrepancies", "check_card_application_fit"}
AG = {"update_transaction_rewards_3847", "get_user_dispute_history_7291"}
US = {"submit_cash_back_dispute_0589", "get_referral_link"}
ok = fail = 0


def chk(name, cond):
    global ok, fail
    if cond:
        ok += 1
        print("  ✓ %s" % name)
    else:
        fail += 1
        print("  ✗ %s" % name)


print("§5-5/6 채널 오분류")
chk("스캐폴드를 unlock → 안내", AX.channel_note(
    "unlock_discoverable_agent_tool", {"agent_tool_name": "get_reward_discrepancies"},
    SC, AG, US, set(), TPL) is not None)
chk("user 도구를 unlock → 안내", AX.channel_note(
    "unlock_discoverable_agent_tool", {"agent_tool_name": "submit_cash_back_dispute_0589"},
    SC, AG, US, set(), TPL) is not None)
chk("agent 도구를 give → 안내", AX.channel_note(
    "give_discoverable_user_tool", {"discoverable_tool_name": "update_transaction_rewards_3847"},
    SC, AG, US, set(), TPL) is not None)
print("§5-7 과잉 방지")
chk("정상 agent unlock → 무개입", AX.channel_note(
    "unlock_discoverable_agent_tool", {"agent_tool_name": "update_transaction_rewards_3847"},
    SC, AG, US, set(), TPL) is None)
chk("정상 user give → 무개입", AX.channel_note(
    "give_discoverable_user_tool", {"discoverable_tool_name": "get_referral_link"},
    SC, AG, US, set(), TPL) is None)

print("폭주⒝ 본문-언급(026/027형)")
said = "I will use the `get_user_dispute_history_7291` tool to retrieve the dispute history."
chk("미-unlock 언급 → 안내", len(AX.mention_note(
    said, {"get_user_information_by_id"}, AG, US, set(), TPL)) == 1)
chk("이미 unlock이면 무개입", AX.mention_note(
    said, {"get_user_information_by_id"}, AG, US, {"get_user_dispute_history_7291"}, TPL) == [])
chk("이미 호출했으면 무개입", AX.mention_note(
    said, {"get_user_dispute_history_7291"}, AG, US, set(), TPL) == [])
chk("정규식 추출 아님(레지스트리 밖 이름은 무시)", AX.mention_note(
    "use `totally_made_up_9999` now", set(), AG, US, set(), TPL) == [])

print("§5-1~3 fit 차이 표면화")
two = ("{'eligible': [{'card': 'A', 'facts': {'annual_fee': 0.0, 'fx_fee': 0.0}}, "
       "{'card': 'B', 'facts': {'annual_fee': 200.0, 'fx_fee': 0.0}}]}")
one = "{'eligible': [{'card': 'A', 'facts': {'annual_fee': 0.0}}]}"
same = ("{'eligible': [{'card': 'A', 'facts': {'annual_fee': 0.0}}, "
        "{'card': 'B', 'facts': {'annual_fee': 0.0}}]}")
n2 = AX.fit_diff_note(two, TPL)
chk("≥2장 → 갈리는 필드만", n2 is not None and "annual_fee" in n2 and "fx_fee" not in n2)
chk("1장 → 무개입", AX.fit_diff_note(one, TPL) is None)
chk("전 필드 동일 → '구분 불가'", (AX.fit_diff_note(same, TPL) or "").startswith("2 options"))
chk("추천/순위 문구 없음", n2 is not None and "recommend" not in n2.lower()
    and "best" not in n2.lower())

print("§배치화(029형)")
chk("단수명에 배열 → 안내", AX.scalar_array_note(
    {"arguments": json.dumps({"user_id": "u", "transaction_id": ["a", "b", "c"]})}, TPL) is not None)
chk("복수명 배열은 정상(item_ids)", AX.scalar_array_note({"item_ids": ["a", "b"]}, TPL) is None)
chk("단일 값은 무개입", AX.scalar_array_note({"transaction_id": "a"}, TPL) is None)

print("§5-8/9 터미널-턴")
chk("토큰 있고 미호출 → 안내", AX.terminal_turn_note(
    "Yes please transfer me ###TRANSFER###", TPL["transfer_tokens"], False, TPL) is not None)
chk("이미 호출했으면 무개입", AX.terminal_turn_note(
    "Yes ###TRANSFER###", TPL["transfer_tokens"], True, TPL) is None)
chk("토큰 없으면 무개입", AX.terminal_turn_note(
    "thanks, bye", TPL["transfer_tokens"], False, TPL) is None)

print("§formalize_arg_axis 창 (2026-08-12·j런 071t0 t38 오탐 회귀)")
# 과제 진술(두 축 요청)은 **첫 발화**에 있는데 구판 창은 [-6:] 뿐이라 종반 턴에서 그것이
# 빠져 한 축만 형식화 → 정답 인자를 오판 지적했다. 첫 발화가 창에 포함되는지를 검정한다.
import t2_resolve as RZ


class _CapSub:
    content = "alpha_kind and beta_kind"


class _CapLA:
    last_prompt = None

    @staticmethod
    def generate(model=None, tools=None, messages=None, call_name=None, **kw):
        _CapLA.last_prompt = getattr(messages[0], "content", None) or messages[0].content
        return _CapSub()


class _CapUM:
    def __init__(self, role=None, content=None):
        self.role, self.content = role or "user", content


class _CapAgent:
    llm = "stub"
    llm_args = {}


class _UMsg:
    def __init__(self, c):
        self.role, self.content = "user", c


msgs_w = [_UMsg("FIRST: I want one alpha_kind and one beta_kind please")] + \
         [_UMsg("later turn %d about alpha only" % i) for i in range(9)]
got = RZ.formalize_arg_axis(_CapAgent(), _CapLA, _CapUM, msgs_w,
                            "kind", ["alpha_kind", "beta_kind"], "arg={arg} choices={choices} text={text}")
chk("첫 발화가 창에 포함된다", _CapLA.last_prompt is not None and "FIRST:" in _CapLA.last_prompt)
chk("두 축 모두 형식화된다: %r" % (got,), got == {"alpha_kind", "beta_kind"})

print("\n%d PASS · %d FAIL" % (ok, fail))
sys.exit(1 if fail else 0)
