# -*- coding: utf-8 -*-
"""회귀 — C418 `T2_CALL_FORM` · C419 `T2_ARG_EMPTY` (오프라인·모델 0·env 0).

측정 근거는 프로브에 있다(여기서 재지 않는다):
  x249_order_callform_probe.py  — 099 실패 궤적 2개 · B_CALLFORM 8/8·8/8 ↔ A_LIVE 5/8·3/8
  x250_empty_required_arg_probe.py — 010 t2 · B_NAME 8/8 ↔ A_LIVE 0/8 · C_GENERIC 0/8

이 파일이 지키는 것은 **불변식**뿐이다: 도메인 리터럴 0(전부 env 스키마/레지스트리 도출) ·
발견형이 아닌 이름은 안 건드림 · 빈 필수 인자만 거부 · 실패 시 종전 거동.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_gate_patch as G

OK = []


def chk(name, cond, extra=""):
    OK.append(bool(cond))
    print("  %s %s%s" % ("PASS" if cond else "FAIL", name, (" — " + str(extra)) if extra else ""))


class _T(object):
    def __init__(self, name, props, required=()):
        self.openai_schema = {"function": {
            "name": name,
            "parameters": {"type": "object",
                           "properties": {p: {"type": "string"} for p in props},
                           "required": list(required)}}}


class _Agent(object):
    def __init__(self, tools):
        self.tools = tools


class _Toolkit(object):
    def __init__(self, names):
        self._n = names

    def get_discoverable_tools(self):
        return set(self._n)


class _Env(object):
    def __init__(self, names):
        self.tools = _Toolkit(names)


class _TC(object):
    def __init__(self, name, args):
        self.name, self.arguments = name, args


AGENT = _Agent([
    _T("unlock_discoverable_agent_tool", ["agent_tool_name"], ["agent_tool_name"]),
    _T("call_discoverable_agent_tool", ["agent_tool_name", "arguments"], ["agent_tool_name"]),
    _T("get_referrals_by_user", ["user_id"], ["user_id"]),
    _T("log_verification", ["name", "user_id", "address", "email", "phone_number",
                            "date_of_birth", "time_verified"],
       ["name", "user_id", "address", "email", "phone_number", "date_of_birth", "time_verified"]),
])
ENV = _Env({"get_all_user_accounts_by_user_id_3847"})

print("── C418 T2_CALL_FORM ──")
u, c = G._dispatch_tools(AGENT)
chk("디스패처를 스키마 구조로 가른다", (u, c) == ("unlock_discoverable_agent_tool",
                                                  "call_discoverable_agent_tool"), (u, c))

m = G._call_form_map(AGENT, ENV, ["get_all_user_accounts_by_user_id", "get_referrals_by_user"])
chk("발견형 bare 이름 → 접미사 실명 + 호출 형식",
    m.get("get_all_user_accounts_by_user_id")
    == 'get_all_user_accounts_by_user_id_3847 (not in your tool list - it is a discoverable tool; '
       'the way to run it is call_discoverable_agent_tool'
       '(agent_tool_name="get_all_user_accounts_by_user_id_3847"))', m)
chk("발견형이 아닌 도구는 손대지 않는다", "get_referrals_by_user" not in m, m)

m2 = G._call_form_map(AGENT, ENV, ["get_all_user_accounts_by_user_id_3847"])
chk("이미 실명이어도 호출 형식을 붙인다", "get_all_user_accounts_by_user_id_3847" in m2, m2)

chk("레지스트리가 비면 치환 0", G._call_form_map(AGENT, _Env(set()), ["x"]) == {})
chk("디스패처가 없으면 치환 0",
    G._call_form_map(_Agent([_T("f", ["a"])]), ENV, ["get_all_user_accounts_by_user_id"]) == {})

tpl_a2 = {"call_form": {"agent_discoverable": 'first {unlock}("{tool}"), then {call}(...)'}}
m3 = G._call_form_map(AGENT, ENV, ["get_all_user_accounts_by_user_id"], tpl_a2)
chk("문구는 A2 가 갈아 끼울 수 있다(슬롯 셋)",
    m3["get_all_user_accounts_by_user_id"]
    == 'first unlock_discoverable_agent_tool("get_all_user_accounts_by_user_id_3847"), '
       'then call_discoverable_agent_tool(...)', m3)

print("── C419 T2_ARG_EMPTY ──")
full = {"name": "Wei Chen", "user_id": "76ad9cc60e", "address": "88 Harbor View Court",
        "email": "w@example.com", "phone_number": "617-555-0834",
        "date_of_birth": "04/17/1979", "time_verified": "2025-11-14 03:40:00 EST"}
empty = dict(full, date_of_birth="")

d = G._arg_empty_deny(AGENT, _TC("log_verification", empty))
chk("빈 필수 인자를 **이름으로** 짚는다", d and "'date_of_birth'" in d, d)
chk("값을 주지 않는다", d and "04/17/1979" not in d, d)
chk("채워져 있으면 침묵", G._arg_empty_deny(AGENT, _TC("log_verification", full)) is None)
chk("공백만 있어도 빈 값", G._arg_empty_deny(AGENT, _TC("log_verification",
                                                        dict(full, date_of_birth="   "))))
chk("키 부재는 건드리지 않는다(false-block 회피)",
    G._arg_empty_deny(AGENT, _TC("log_verification",
                                 {k: v for k, v in full.items() if k != "date_of_birth"})) is None)
chk("required 선언이 없는 도구는 통과",
    G._arg_empty_deny(_Agent([_T("f", ["a"])]), _TC("f", {"a": ""})) is None)
chk("applies_to 밖이면 침묵",
    G._arg_empty_deny(AGENT, _TC("log_verification", empty), None, {"other_tool"}) is None)

nested = _TC("call_discoverable_agent_tool",
             {"agent_tool_name": "log_verification", "arguments": '{"user_id": ""}'})
chk("중첩 디스패처 인자도 unwrap 한다(WAG 동형)",
    "user_id" in (G._arg_empty_deny(
        _Agent([_T("call_discoverable_agent_tool", ["agent_tool_name", "arguments"]),
                _T("log_verification", ["user_id"], ["user_id"])]), nested) or ""))

a2 = {"arg_empty": {"feedback": "custom {tool}/{args}"}}
chk("문구는 A2 가 갈아 끼울 수 있다",
    G._arg_empty_deny(AGENT, _TC("log_verification", empty), a2)
    == "custom log_verification/'date_of_birth'")

print("\n%d/%d" % (sum(OK), len(OK)))
sys.exit(0 if all(OK) else 1)
