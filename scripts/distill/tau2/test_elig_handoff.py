# -*- coding: utf-8 -*-
"""`T2_ELIG_LINE`(자격 한 줄)과 `T2_HANDOFF_PREDICATE`(G2 인계 술어) 계약 검정.

박는 못:
  ⑴ 자격: 값이 A2 선언 집합 밖이면 **침묵** · 인용이 원문에 없으면 **침묵**(fail-safe·[[25]])
  ⑵ 자격: 두 조건이 다 맞을 때만 줄을 만들고, 줄은 A2 템플릿 그대로다(엔진 문장 저작 0)
  ⑶ 인계: 손님-측 레지스트리가 비면 **빈 목록**(다른 도메인 = 침묵)
  ⑷ 인계: 이미 `give` 한 도구는 위반이 아니다
  ⑸ 인계: 에이전트-측 도구 이름은 이 술어의 대상이 **아니다**(x368 1차 계수가 여기서 틀렸다)
  ⑹ A2 두 층 동기화 + `axis_notes` 는 base 층에 있고 도메인 층이 **덮지 않는다**([[24]] 실화)
"""
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_search as TS  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
bad = []


def chk(cond, name):
    print("%s %s" % ("OK  " if cond else "FAIL", name))
    if not cond:
        bad.append(name)


class _TC(object):
    def __init__(self, name, args):
        self.name, self._a = name, args


class _M(object):
    def __init__(self, calls=()):
        self.tool_calls = list(calls)


class _UT(object):
    def __init__(self, names):
        self._n = names

    def get_discoverable_tools(self):
        return list(self._n)


class _Env(object):
    def __init__(self, names):
        self.user_tools = _UT(names)


def _args(tc):
    return getattr(tc, "_a", {})


def _content(m):
    return getattr(m, "content", "")


# ── ⑶⑷⑸ 인계 술어
NAMES = ["deposit_check_3847", "submit_cash_back_dispute_0589"]
env = _Env(NAMES)
say = "Please use submit_cash_back_dispute_0589 to file each dispute."
chk(TS.handoff_missing(env, [], say, _content, _args) == ["submit_cash_back_dispute_0589"],
    "⑸ 이름을 말했고 give 없음 → 위반으로 잡힌다")
gave = [_M([_TC("give_discoverable_user_tool",
                {"discoverable_tool_name": "submit_cash_back_dispute_0589"})])]
chk(TS.handoff_missing(env, gave, say, _content, _args) == [],
    "⑷ 이미 건넨 도구는 위반이 아니다")
chk(TS.handoff_missing(_Env([]), [], say, _content, _args) == [],
    "⑶ 레지스트리가 비면 침묵(도메인 일반)")
agent_side = "I will call unlock_discoverable_agent_tool for close_debit_card_4721."
chk(TS.handoff_missing(env, [], agent_side, _content, _args) == [],
    "⑸b 에이전트-측 도구 이름은 이 술어 대상이 아니다")

# ── ⑴⑵ 자격 한 줄 (LLM 은 가짜로 갈아끼운다 — 엔진 계약만 본다)
a2 = {}
for name in ("banking_knowledge.settings.json", "banking_knowledge.specific.json"):
    p = os.path.join(HERE, "a2", name)
    if os.path.exists(p):
        a2.update(json.load(io.open(p, encoding="utf-8")))
po = a2.get("policy_ontology") or {}
chk(bool(po.get("eligibility_prompt")) and bool(po.get("eligibility_values"))
    and bool(po.get("eligibility_line_template")), "A2 자격 키 3종 존재")

REC = "I run a landscaping company and want a business card."


class _FakeSC(object):
    out = ""

    @staticmethod
    def sub_generate(agent, la, UM, prompt, tag):
        return _FakeSC.out


_real = TS.SC
TS.SC = _FakeSC
try:
    _FakeSC.out = "BUSINESS\nI run a landscaping company"
    line = TS.eligibility_line(1, 1, 1, po, REC)
    chk("BUSINESS" in line and "landscaping" in line, "⑵ 값+인용 둘 다 맞으면 줄이 나온다")
    chk(line == str(po["eligibility_line_template"]).format(
        v="BUSINESS", q="I run a landscaping company"), "⑵b 줄은 A2 템플릿 축자(엔진 저작 0)")
    _FakeSC.out = "MAYBE\nI run a landscaping company"
    chk(TS.eligibility_line(1, 1, 1, po, REC) == "", "⑴ 값이 선언 집합 밖 → 침묵")
    _FakeSC.out = "BUSINESS\nI fly airplanes for a living"
    chk(TS.eligibility_line(1, 1, 1, po, REC) == "", "⑴b 인용이 원문에 없다 → 침묵")
    _FakeSC.out = "BUSINESS"
    chk(TS.eligibility_line(1, 1, 1, po, REC) == "", "⑴c 근거 줄이 없으면 침묵")
finally:
    TS.SC = _real

# ── ⑹ A2 층 규율
base = json.load(io.open(os.path.join(HERE, "a2", "base", "shared.json"), encoding="utf-8"))
chk("handoff_missing" in (base.get("axis_notes") or {}), "⑹ handoff 문면은 base 층에 있다")
chk("user_tool_channel" in (base.get("axis_notes") or {}), "⑹b 형제 키가 살아 있다")
for f in ("banking_knowledge.gate.json", "banking_knowledge.specific.json"):
    d = json.load(io.open(os.path.join(HERE, "a2", f), encoding="utf-8"))
    chk("axis_notes" not in d, "⑹c %s 가 base 의 axis_notes 를 덮지 않는다" % f)
g = json.load(io.open(os.path.join(HERE, "a2", "banking_knowledge.gate.json"), encoding="utf-8"))
s = json.load(io.open(os.path.join(HERE, "a2", "banking_knowledge.specific.json"), encoding="utf-8"))
for k in ("eligibility_prompt", "eligibility_values", "eligibility_line_template"):
    chk(g["policy_ontology"][k] == s["policy_ontology"][k], "⑹d 두 층 동기 %s" % k)

# ── 배선: 엔진이 상류에서만 부르고 하류에는 안 싣는다
src = io.open(os.path.join(HERE, "t2_gate_patch.py"), encoding="utf-8").read()
chk(src.count("eligibility_line(") == 1, "자격 줄은 **한 자리**(군 선택 상류)에서만 불린다")
chk("T2_HANDOFF_PREDICATE" in src and "give_discoverable_user_tool" in src, "인계 술어 배선 존재")
chk("_missh[0]" not in src, "엔진이 후보 중 하나를 고르지 않는다")

print("\n%s" % ("test_elig_handoff PASS" if not bad else
                "test_elig_handoff FAIL %d건: %s" % (len(bad), ", ".join(bad))))
sys.exit(1 if bad else 0)
