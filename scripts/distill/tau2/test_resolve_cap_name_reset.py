# -*- coding: utf-8 -*-
"""정체 카운터 리셋 축 검정 — **새 이름 회수 = 인자 변화 = 진행** (x305 근거·[[57]]).

087 실측: 이름 노출 전 구간에서 캡 3회 소진(그때 formalize 는 정직하게 none) → 옳은 이름이
KB 로 도착한 자리에서 `resolve_cap` 침묵. 그 이름이 후보에 들면 formalize 8/8 선택(x305 POST)·
그 문면은 컷을 6/8 로 연다(x304). 그래서 **회수 집합이 커지면** 카운터를 되돌린다.

검정 4칸:
  T1  캡 소진 상태 + 새 이름 회수  → 리셋(진입 허용)
  T2  Δspurious: 캡 소진 + 회수 집합 **불변**(같은 요구 반복) → 침묵 유지 (캡 목적 보존)
  T3  Δspurious: 회수 집합이 **줄어도**(unlock 시도로 후보에서 빠짐) 리셋 없음
  T4  실행 진행 축(기존 규칙)은 그대로 작동
"""
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("T2_GATE_KINDS", "auth,confirm")
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

import t2_resolve as R                                            # noqa: E402

A2 = {"eplan": {"unlock_tool": "unlock_x", "dispatch_tool": "call_x", "list_tool": "list_x"}}
REG = {"get_alpha_1111", "get_beta_2222", "get_gamma_3333"}
FAILS = []


class M(object):
    # ⚠tool 결과는 **id 로** 호출과 이어진다(`_executed_tool_names`) — id 없는 픽스처는
    #   실행을 0건으로 만든다(초판 T4 실패의 원인·계기 결함이었지 레버 결함이 아니었다).
    def __init__(self, role, content="", tool_calls=None, error=False, id=None):
        self.role, self.content, self.tool_calls = role, content, tool_calls
        self.error, self.id = error, id


class TC(object):
    def __init__(self, name, arguments, id="t1"):
        self.name, self.arguments, self.id = name, arguments, id


class Agent(object):
    """`_resolve_cap_ok` 의 self — 레지스트리는 라이브와 같은 접근자로 읽힌다."""
    tools = []
    def __init__(self):
        self._t2_orch = None


def patch_registry():
    """agent_discoverable_names 를 테스트 레지스트리로 (전송·환경 스텁만·판정 로직 불변)."""
    R.agent_discoverable_names = lambda agent: set(REG)


def chk(name, cond, extra=""):
    print("%-4s %s %s" % ("PASS" if cond else "FAIL", name, extra))
    if not cond:
        FAILS.append(name)


patch_registry()
import t2_gate_patch as G                                          # noqa: E402

CAP = G._resolve_cap_ok


def state(msgs, deny=3, names=None, done=None):
    a = Agent()
    a._t2_resolve_deny = deny
    if names is not None:
        a._t2_resolve_names = set(names)
    if done is not None:
        a._t2_resolve_done = set(done)
    return a


# T1: 캡 소진 + 새 이름(get_beta_2222) 회수 → 리셋
msgs = [M("user", "help"), M("tool", "doc says use get_alpha_1111 and get_beta_2222")]
a = state(msgs, deny=3, names={"get_alpha_1111"})
ok = CAP(a, msgs, A2)
chk("T1_new_name_resets", ok and a._t2_resolve_deny == 0, "deny=%s" % a._t2_resolve_deny)

# T2: 캡 소진 + 회수 집합 불변 → 침묵 유지
msgs2 = [M("user", "help"), M("tool", "doc says use get_alpha_1111")]
a2 = state(msgs2, deny=3, names={"get_alpha_1111"})
ok2 = CAP(a2, msgs2, A2)
chk("T2_same_names_stay_silent", (not ok2) and a2._t2_resolve_deny == 3,
    "ok=%s deny=%s" % (ok2, a2._t2_resolve_deny))

# T3: 후보가 줄어든 경우(unlock 시도로 제외) → 리셋 없음
msgs3 = [M("user", "help"), M("tool", "doc says use get_alpha_1111 and get_beta_2222"),
         M("assistant", "", [TC("unlock_x", {"agent_tool_name": "get_beta_2222"})])]
a3 = state(msgs3, deny=3, names={"get_alpha_1111", "get_beta_2222"})
ok3 = CAP(a3, msgs3, A2)
chk("T3_shrink_no_reset", (not ok3) and a3._t2_resolve_deny == 3,
    "ok=%s deny=%s" % (ok3, a3._t2_resolve_deny))

# T4: 기존 실행-진행 축 보존 (이름 축과 독립)
msgs4 = [M("user", "help"),
         M("assistant", "", [TC("call_x", {"agent_tool_name": "get_alpha_1111"}, id="c4")]),
         M("tool", "Executed: get_alpha_1111", id="c4")]
a4 = state(msgs4, deny=3, names=None, done=set())
ok4 = CAP(a4, msgs4, A2)
chk("T4_execution_axis_intact", ok4 and a4._t2_resolve_deny == 0, "deny=%s" % a4._t2_resolve_deny)

print("=" * 60)
print("FAILS:", FAILS or "none")
sys.exit(1 if FAILS else 0)
