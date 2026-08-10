# -*- coding: utf-8 -*-
"""회귀 검정: **R8c 잠금-미호출 침묵** (`T2_UNLOCK_QUIET`·2026-08-11·C408).

무엇을 막는 검정인가 —
 ⒜ **끈 상태에서 거동이 바뀌는 것**. 기본은 OFF 여야 하고 OFF 면 종전과 한 글자도 달라선 안 된다.
 ⒝ **영원히 조용해지는 것**. 침묵은 *그 도구를 부를 때까지*만이다 — 부르면 즉시 원상복귀.
 ⒞ **잠근 적 없는 대화까지 조용해지는 것**. 조건은 `잠금됨 − 호출됨` 이 비지 않을 때뿐이다.
 ⒟ **디스패처 호출을 못 알아보는 것**. 호출은 `call_*` 로 들어온다(잠긴 이름은 인자에 있다).

오프라인 전용(LLM·서버 불요). 실행: py -3 test_unlock_quiet.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

FAILED = []


def chk(c, label):
    print(("  OK   " if c else "  FAIL ") + label)
    if not c:
        FAILED.append(label)


class _TC(object):
    def __init__(self, name, args=None):
        self.name = name
        self.arguments = args or {}


class _M(object):
    def __init__(self, role, content="", tool_calls=None):
        self.role = role
        self.content = content
        self.tool_calls = tool_calls or []


import t2_gate_patch as GP                                        # noqa: E402

# 술어는 A2 선언(`dispatcher_role_check`)에서 나온다 — 도구 이름도 인자 이름도 엔진이 모른다.
# 그래서 검정도 **정본 A2** 를 읽는다(지어낸 스펙으로 통과시키면 배선을 안 지킨다·[[24]]).
import json                                                       # noqa: E402
A2 = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "a2", "banking_knowledge.gate.json"), encoding="utf-8"))
UNLOCK_TOOL = ((A2.get("dispatcher_role_check") or {}).get("unlock_tool"))
UNLOCK_ARG = ((A2.get("dispatcher_role_check") or {}).get("name_args") or {}).get(UNLOCK_TOOL)
LOCKED = "get_all_user_accounts_by_user_id_3847"


def quiet(msgs):
    """운영 코드와 **같은 술어**로 계산한다 — 재구현하면 두 벌이 되어 갈린다([[03b]])."""
    unl = GP._unlocked_names(msgs, A2)
    cal = {GP._exact_tool_name(t) for m in msgs for t in (m.tool_calls or [])
           if str(getattr(t, "name", "") or "").startswith("call_")}
    return sorted(unl - cal)


print("\n§1 잠근 적 없으면 조용해지지 않는다")
base = [_M("user", "hi"), _M("assistant", "", [_TC("KB_search_dense", {"query": "x"})])]
chk(quiet(base) == [], "잠금 0 → 침묵 조건 없음 (%s)" % quiet(base))

print("\n§2 잠그고 안 부르면 조건이 선다")
unlocked = base + [_M("assistant", "", [_TC(UNLOCK_TOOL, {UNLOCK_ARG: LOCKED})]),
                   _M("tool", "Tool unlocked: %s Description: …" % LOCKED)]
print("  (A2 선언: unlock_tool=%s · name_arg=%s)" % (UNLOCK_TOOL, UNLOCK_ARG))
chk(quiet(unlocked) == [LOCKED], "잠금-미호출이 잡힌다 (%s)" % quiet(unlocked))

print("\n§3 부르면 즉시 원상복귀 — 영원한 침묵이 아니다")
called = unlocked + [_M("assistant", "", [_TC("call_discoverable_agent_tool",
                                              {UNLOCK_ARG: LOCKED,
                                               "arguments": '{"user_id": "u1"}'})])]
chk(quiet(called) == [], "호출 후 침묵 조건 해제 (%s)" % quiet(called))

print("\n§4 기본은 OFF")
chk(os.environ.get("T2_UNLOCK_QUIET") != "1", "환경에 기본 ON 이 박혀 있지 않다")
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "t2_gate_patch.py"),
           encoding="utf-8").read()
chk('os.environ.get("T2_UNLOCK_QUIET") == "1"' in src, "플래그 뒤에 있다")
chk(src.count("[T2_UNLOCK_QUIET]") >= 2, "억제·오류 둘 다 마크를 남긴다")
i = src.find('if os.environ.get("T2_UNLOCK_QUIET") == "1":')
j = src.find('_beat("T2_GATE_REGEN"', i)
chk(0 < i < j, "빈-문구 차단 뒤·`_beat` 앞의 단일 진입점에 있다 (i=%d j=%d)" % (i, j))

print("\n%s  (%d/%d)" % ("FAIL" if FAILED else "ALL PASS", 7 - len(FAILED), 7))
sys.exit(1 if FAILED else 0)
