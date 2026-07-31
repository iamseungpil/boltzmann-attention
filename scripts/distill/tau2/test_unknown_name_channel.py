# -*- coding: utf-8 -*-
"""`T2_UNKNOWN_NAME_BL` 채널 인식 회귀 (2026-07-31 — gold 차단 사고의 재발 방지).

사고(Y2-B 26 sim 포렌식·Y1에도 46회 있던 선재 버그): 블랙리스트가 **이름만** 보고 채널을 무시했다.
  task_017 실제 순서 —
    ① `unlock_discoverable_agent_tool(submit_cash_back_dispute_0589)`
       → env: `Unknown **agent** tool 'submit_cash_back_dispute_0589'`   (user 도구를 agent로 부름)
    ② 블랙리스트에 이름만 등록
    ③ `give_discoverable_user_tool(discoverable_tool_name=submit_cash_back_dispute_0589)`
       = **gold 액션**인데 우리 레버가 **18회 차단**하고 "그 이름은 없으니 쓰지 말라"고 지시
⇒ 스캐폴드가 정답을 막았다. false-block = 0이 우리 안전 게이트인데 그게 깨져 있었다.

이 테스트가 고정하는 계약:
  A. 다른 채널에서 난 거부는 **이 채널을 막지 않는다**(gold 통과)
  B. 같은 채널의 동일 이름 재시도는 **여전히 막는다**(원래 목적·C154 050형)
  C. 채널을 모르는 도구는 **막지 않는다**(보수적)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t2_gate_patch import unknown_bl_collect, unknown_bl_hit   # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

NAME = "submit_cash_back_dispute_0589"


class TC:
    def __init__(self, id_, name):
        self.id, self.name = id_, name


class M:
    def __init__(self, role, content="", tool_calls=None, id_=None):
        self.role, self.content, self.tool_calls, self.id = role, content, tool_calls, id_


# task_017 재현: agent 채널로 unlock → env가 'Unknown agent tool'로 거부
TRACE_017 = [
    M("assistant", tool_calls=[TC("c1", "unlock_discoverable_agent_tool")]),
    M("tool", "Error: Unknown agent tool '%s'" % NAME, id_="c1"),
]

OK = True


def chk(label, got, want):
    global OK
    good = (got == want)
    OK = OK and good
    print("  %-58s %s (got=%s)" % (label, "PASS" if good else "★FAIL", got))


print("[채널 인식 블랙리스트]")
bl, kind = unknown_bl_collect(TRACE_017)
chk("수집: (kind,name) 쌍으로 들어갔나", ("agent", NAME) in bl, True)
chk("수집: 거부를 낸 도구의 채널을 관측했나", kind.get("unlock_discoverable_agent_tool"), "agent")

# A. gold 액션(다른 채널)은 통과해야 한다 — 이 사고의 본체
chk("A. gold give(user 채널)를 막지 않는다",
    unknown_bl_hit(bl, kind, "give_discoverable_user_tool", NAME), False)

# B. 같은 채널 재시도는 계속 막는다 (레버의 원래 목적)
chk("B. 같은 도구로 같은 이름 재시도는 막는다",
    unknown_bl_hit(bl, kind, "unlock_discoverable_agent_tool", NAME), True)

# C. 채널 미상 도구는 보수적으로 통과
chk("C. 거부 이력 없는 도구는 막지 않는다",
    unknown_bl_hit(bl, kind, "call_discoverable_agent_tool", NAME), False)

# D. user 채널에서 난 거부는 user 채널을 막는다 (task_018형: 접미사 없는 추측 이름)
TRACE_018 = [
    M("assistant", tool_calls=[TC("d1", "give_discoverable_user_tool")]),
    M("tool", "Error: Unknown discoverable tool 'submit_cash_back_dispute'", id_="d1"),
]
bl2, kind2 = unknown_bl_collect(TRACE_018)
chk("D. user 채널 거부 → 같은 채널 재시도 차단",
    unknown_bl_hit(bl2, kind2, "give_discoverable_user_tool", "submit_cash_back_dispute"), True)
chk("D'. 그 거부가 agent 채널로 새지 않는다",
    unknown_bl_hit(bl2, kind2, "unlock_discoverable_agent_tool", "submit_cash_back_dispute"), False)

# E. 빈 값·미상 이름은 무해
chk("E. 빈 값은 막지 않는다", unknown_bl_hit(bl, kind, "unlock_discoverable_agent_tool", ""), False)

print("RESULT: %s" % ("ALL PASS" if OK else "FAIL"))
sys.exit(0 if OK else 1)
