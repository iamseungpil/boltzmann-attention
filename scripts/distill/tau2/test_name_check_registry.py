# -*- coding: utf-8 -*-
"""Do we still tell the model that a real tool does not exist?

`task_022` ran ninety minutes and seventy-six turns and was killed in a loop. Reading the
sidecar, the loop is two of our own sentences, alternating:

    'submit_cash_back_dispute_0589' is missing its numeric suffix … search the knowledge base NOW
    'submit_cash_back_dispute_0589' … that exact name does not exist and retrying it will fail

The name is complete and the tool is real — it is a **user**-side discoverable tool. Two
defects put those sentences on the screen, and both are decidable from the registry:

  ① 레지스트리 반쪽    the name check compared against the agent-side set only
  ② 정규화 불일치      `_known_tool_names` strips the numeric suffix, the probe did not, so a
                       suffixed name was permanently "unknown" and fell to the suffix branch

This checks the predicates rather than the wording, and the negative controls are the point:
a genuinely unknown name must still be caught, or the fix is just a silencer.
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

import t2_gate_patch as G          # noqa: E402

fails = []


def check(name, got, want):
    ok = got == want
    print("  %-62s %s%s" % (name, "PASS" if ok else "FAIL",
                            "" if ok else " — got %r want %r" % (got, want)))
    if not ok:
        fails.append(name)


class _Tools(object):
    def __init__(self, agent, user):
        self._a, self._u = agent, user

    def get_discoverable_tools(self):
        return list(self._a)


class _UserTools(object):
    def __init__(self, user):
        self._u = user

    def get_discoverable_tools(self):
        return list(self._u)


class _Env(object):
    """env 실물의 최소 대역 — 두 레지스트리를 각각 노출한다(banking 실측 이름 사용)."""

    def __init__(self):
        self.tools = _Tools({"update_transaction_rewards_3847", "get_user_dispute_history_7291"},
                            set())
        self.user_tools = _UserTools({"submit_cash_back_dispute_0589", "get_card_last_4_digits",
                                      "get_referral_link", "deposit_check_3847"})


env = _Env()
AGENT = G._agent_discoverable(env)
USER = G._user_discoverable(env)

print("== 레지스트리가 두 쪽 다 읽히는가 ==")
check("agent 측 집합 비어있지 않다", bool(AGENT), True)
check("user 측 집합에 022의 도구가 있다", "submit_cash_back_dispute_0589" in USER, True)

print("\n== ① 레지스트리 반쪽 — 이 이름에 대해 레버가 할 말이 있는가 ==")
# 레버의 발화 조건: 이름이 agent 집합 밖 **이고** user 집합 밖일 때만.
def speaks(name):
    return bool(name and AGENT and not G._in_registry(name, AGENT)
                and not G._in_registry(name, USER))


check("user 도구(022의 그 이름) → 침묵", speaks("submit_cash_back_dispute_0589"), False)
check("user 도구(last-4) → 침묵", speaks("get_card_last_4_digits"), False)
check("agent 도구 → 침묵(원래 정상)", speaks("update_transaction_rewards_3847"), False)
check("부정통제: 진짜 미상 이름 → 여전히 발화", speaks("submit_cash_back_dispute_8374"), True)
check("부정통제: 접미사 없는 bare name → 여전히 발화", speaks("submit_cash_back_dispute"), True)

print("\n== ② 정규화 불일치 — 알려진 이름을 알려진 것으로 판정하는가 ==")
known = G._known_tool_names(None, env, [])
probe = "submit_cash_back_dispute_0589"
check("구판 탐침(원형)은 미상으로 판정됐다(회귀 재현)", probe in known, False)
check("정규화 맞춘 탐침은 알려진 것으로 판정된다",
      re.sub(r"_\d+$", "", probe) in known, True)
check("부정통제: 실재하지 않는 이름은 정규화해도 미상",
      re.sub(r"_\d+$", "", "totally_made_up_9999") in known, False)

print("\n== 배선(엔진이 실제로 두 집합과 정규화를 쓰는가) ==")
import inspect                                                            # noqa: E402
src = inspect.getsource(G)
check("UNLOCK_NAME이 user 레지스트리도 본다",
      "_regu2 = _user_discoverable(_env2)" in src, True)
check("UNLOCK_NAME 조건에 user 집합이 들어간다",
      "and not _in_registry(_uval, _regu2)" in src, True)
check("_known 탐침이 정규화된다",
      '_known = re.sub(r"_' + chr(92) + 'd+$", "", _uval) in _known_tool_names(' in src, True)
check("UNKNOWN_NAME_BL이 user 도구에 '없는 이름'이라 말하지 않는다",
      "_in_registry(_uv3, _user_discoverable(" in src, True)

print("\n== 035 死설정 — 거동 불변 ==")
import t2_phase as PH                                                     # noqa: E402
import gate_interpreter as GI                                             # noqa: E402
a2 = GI.load_domain_a2("banking_knowledge") or {}
import json                                                               # noqa: E402
check("A2에서 verify_gather_prefix가 제거됐다",
      "verify_gather_prefix" in json.dumps(a2, ensure_ascii=False), False)
check("phase_of가 문자 집합을 만들지 않는다",
      'set(g.get("verify_gather_prefix")' in inspect.getsource(PH), False)


class _M(object):
    def __init__(self, role, calls=()):
        self.role, self.tool_calls = role, [_TC(c) for c in calls]


class _TC(object):
    def __init__(self, n):
        self.name, self.arguments = n, {}


def unwrap(tc):
    return tc.name


# 실효 술어(구판과 동일해야 한다): verify_identity 호출 ∧ log_verification 미실행
check("verify_identity 호출 → verify",
      PH.phase_of(a2, [_M("assistant", ["verify_identity"])], unwrap, executed=set())[0], "verify")
check("gather 접두사 도구만 호출 → open(구판과 동일)",
      PH.phase_of(a2, [_M("assistant", ["get_user_information_by_name"])], unwrap,
                  executed=set())[0], "open")
check("satisfier 실행됨 → open",
      PH.phase_of(a2, [_M("assistant", ["verify_identity"])], unwrap,
                  executed={"log_verification"})[0], "open")

print("\n결과: %s" % ("ALL PASS" if not fails else "FAIL %d — %s" % (len(fails), fails)))
sys.exit(1 if fails else 0)
