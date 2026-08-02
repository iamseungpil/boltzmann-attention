#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P13 회귀: CHANNEL 표면화 채널 이설 (2026-08-02·승인).

계약 2건만 검증한다(엔진 전체 기동 불필요):
  ⑴ **출력-부착 경로에서 CHANNEL이 제거**됐다 — mutating 도구(unlock/give/call)에 [axis] 노트가
     생성되지 않아야 한다(041 사고 원인).
  ⑵ `channel_note` 술어 자체는 **불변**(이설이지 재설계가 아니다) — 같은 입력에 같은 문구.

배경: `_is_mutating_tool` 실측 = give/call/unlock/KB_search **True**, get_* False.
channel_note는 unlock·give·call에서만 발화하므로 출력-부착으로는 항상 replay를 깬다 ⇒ 생성-레벨 이설.

Run: python test_p13_channel_channel.py
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def main():
    fails = []
    src = open(os.path.join(HERE, "t2_gate_patch.py"), encoding="utf-8").read()

    # ⑴ 출력-부착 블록(add 리스트 구성부)에 CHANNEL 호출이 남아 있으면 안 된다.
    m = re.search(r"\n        add = \[\]\n(.*?)\n        if add:", src, re.S)
    if not m:
        fails.append("출력-부착 블록(add=[] … if add:)을 찾지 못함 — 테스트 앵커 갱신 필요")
    else:
        block = m.group(1)
        if "channel_note" in block:
            fails.append("출력-부착 블록에 channel_note가 남아 있다(041 사고 재발 경로)")
        if "T2_TOOL_CHANNEL" in block and "부착하지 않는다" not in block:
            fails.append("출력-부착 블록이 여전히 T2_TOOL_CHANNEL로 노트를 만든다")

    # ⑵ 생성-레벨(unified) 이설 블록이 존재해야 한다.
    if "pre-call regen" not in src or "channel_pre" not in src:
        fails.append("unified 예방형 CHANNEL 블록(pre-call regen)이 없다")

    # ⑶ 술어 불변: channel_note 계약 확인 (이설이지 재설계 아님)
    import t2_axis_levers as AX
    tpl = {
        "is_user_tool": "`{tool}` is a USER tool: hand it over.",
        "is_agent_tool": "`{tool}` is an AGENT tool: unlock it.",
        "is_scaffold": "`{tool}` is NOT discoverable.",
        "not_unlocked": "`{tool}` has not been unlocked yet.",
    }
    agent_d, user_d, scaffold = {"atool"}, {"utool"}, {"stool"}
    cases = [
        ("call_discoverable_agent_tool", {"agent_tool_name": "utool"}, "USER tool"),
        ("give_discoverable_user_tool", {"discoverable_tool_name": "atool"}, "AGENT tool"),
        ("unlock_discoverable_agent_tool", {"agent_tool_name": "stool"}, "NOT discoverable"),
    ]
    for name, args, want in cases:
        got = AX.channel_note(name, args, scaffold, agent_d, user_d, set(), tpl)
        if not got or want not in got:
            fails.append("channel_note 술어 변경됨: %s(%s) -> %r" % (name, args, got))

    # ⑷ 정상 채널은 무발화
    if AX.channel_note("call_discoverable_agent_tool", {"agent_tool_name": "atool"},
                       scaffold, agent_d, user_d, {"atool"}, tpl):
        fails.append("정상 채널인데 발화(과차단)")

    print("P13 회귀 — 출력-부착 제거 / 생성-레벨 존재 / 술어 불변 3케이스 / 과차단 0")
    if fails:
        for f in fails:
            print("  ❌ %s" % f)
        print("FAIL %d건" % len(fails))
        return 1
    print("  ✅ ALL PASS — CHANNEL 이설 완료(replay 안전·술어 불변)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
