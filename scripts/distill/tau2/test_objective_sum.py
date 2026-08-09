# -*- coding: utf-8 -*-
"""합-목적 회귀 — 손님이 두 축의 **합**을 원할 때 (순수함수·GPU 0).

근거 원장: C377. 표적 = task_098 축자 *"the best combined referral bonus - the total of what
I get plus what she gets"*. 실측(런 s): 두 sim 다 지목 단계에서 `[T2_REDERIVE] raw='NONE'` 이라
결정 블록이 아예 안 만들어졌고 `[T2_OBJ_AXIS]` 는 호출조차 되지 않았다.

⚠고정물은 **도메인 무관 합성**이다 — 엔진만 시험한다.

실행: py -3 test_objective_sum.py
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

import t2_ledger as LG                                          # noqa: E402
from gate_interpreter import load_domain_a2                      # noqa: E402

FAIL = []
HERE = os.path.dirname(os.path.abspath(__file__))


def ok(cond, label, extra=""):
    print(("  OK   " if cond else "  FAIL ") + label + (("  — " + str(extra)) if extra else ""))
    if not cond:
        FAIL.append(label)


# ── 합성 고정물: 어느 축 단독으로도 1위가 아닌데 **합으로는 1위**인 주어를 심는다 ──
#    이것이 이 설계의 핵심 주장을 시험한다.
MINE = {"Alpha": (100, "docA1"), "Bravo": (10, "docB1"), "Charlie": (60, "docC1")}
THEIRS = {"Alpha": (10, "docA2"), "Bravo": (100, "docB2"), "Charlie": (65, "docC2")}
MAPS = {"mine": MINE, "theirs": THEIRS}
ROWS = [("Alpha", ["mine=100"]), ("Bravo", ["mine=10"]), ("Charlie", ["mine=60"])]
SPEC = {"objective_axes_prompt": "axes:\n{axes}\nconv:\n{text}"}


class _Sub(object):
    def __init__(self, content):
        self.content = content


class _LA(object):
    def __init__(self, content):
        self.content, self.seen = content, None

    def generate(self, model=None, tools=None, messages=None, **kw):
        self.seen = messages[0].content
        return _Sub(self.content)


class _LAErr(object):
    def generate(self, **kw):
        raise RuntimeError("transient")


class _UM(object):
    def __init__(self, role=None, content=None):
        self.content = content


class _Agent(object):
    def __init__(self):
        self.llm = None
        self.llm_args = {}


print("\n§1 왜 '지목을 두 번 해서 더한다'가 안 되는가 (설계 주장의 시험)")
top_mine = LG._rank_by(ROWS, MINE)[0][0]
top_theirs = LG._rank_by(ROWS, THEIRS)[0][0]
lab, both = LG.combine_axes(MAPS, ["mine", "theirs"])
top_sum = LG._rank_by(ROWS, both)[0][0]
ok(top_mine == "Alpha" and top_theirs == "Bravo", "축별 1위는 서로 다르다",
   (top_mine, top_theirs))
ok(top_sum == "Charlie", "**합의 1위는 어느 쪽 1위도 아니다**", top_sum)
ok(top_sum not in (top_mine, top_theirs),
   "⇒ argmax(A)·argmax(B) 를 알아도 argmax(A+B) 는 알 수 없다 — 덧셈은 값 층위에서만")

print("\n§2 combine_axes")
ok(lab == "mine + theirs", "라벨이 두 축을 다 말한다", lab)
ok(dict((k, v[0]) for k, v in both.items()) == {"Alpha": 110, "Bravo": 110, "Charlie": 125},
   "값이 합산된다", dict((k, v[0]) for k, v in both.items()))
one_lab, one_map = LG.combine_axes(MAPS, ["mine"])
ok(one_lab == "mine" and one_map is MINE, "축이 하나면 그 지도를 **그대로** 돌려준다(거동 불변)")
ok(LG.combine_axes(MAPS, []) == (None, {}), "축이 없으면 침묵")
part = dict(MAPS, theirs={"Alpha": (10, "d")})          # Bravo·Charlie 는 한 축이 없다
_, pm = LG.combine_axes(part, ["mine", "theirs"])
ok(set(pm) == {"Alpha"},
   "한 축이 빈 주어는 **뺀다** — 부분합은 작은 수가 아니라 틀린 수다([[25]])", sorted(pm))

print("\n§3 formalize_objective_axes — 엔진은 집합 검사만 한다 ([[22]])")
ag = _Agent()
la = _LA("mine\ntheirs")
got = LG.formalize_objective_axes(ag, la, _UM, SPEC, ["conversation"], MAPS)
ok(got == ["theirs", "mine"] or sorted(got) == ["mine", "theirs"], "두 축을 다 받는다", got)
ag2 = _Agent()
ok(LG.formalize_objective_axes(ag2, _LA("profit_margin"), _UM, SPEC, ["c"], MAPS) == [],
   "축 집합 밖 이름은 버린다")
ag3 = _Agent()
ok(LG.formalize_objective_axes(ag3, _LA("NONE"), _UM, SPEC, ["c"], MAPS) == [],
   "NONE 이면 빈 목록")
ag4 = _Agent()
SUB = {"objective_axes_prompt": SPEC["objective_axes_prompt"]}
ok(LG.formalize_objective_axes(ag4, _LA("gain_total"), _UM, SUB, ["c"],
                               {"gain": "g", "gain_total": "gt"}) == ["gain_total"],
   "부분 문자열이 다른 이름에 먹히지 않는다", "gain 은 세지 않는다")
ag5 = _Agent()
ok(LG.formalize_objective_axes(ag5, _LAErr(), _UM, SPEC, ["c"], MAPS) == [],
   "호출이 죽으면 빈 목록")
la5 = _LA("mine")
LG.formalize_objective_axes(ag5, la5, _UM, SPEC, ["c"], MAPS)
ok(la5.seen is not None, "**실패는 기억하지 않는다** — 다음 턴에 다시 묻는다")
ag6 = _Agent()
la6 = _LA("mine")
LG.formalize_objective_axes(ag6, la6, _UM, SPEC, ["c"], MAPS)
la6.seen = None
LG.formalize_objective_axes(ag6, la6, _UM, SPEC, ["c"], MAPS)
ok(la6.seen is None, "성공은 기억한다 (호출 예산)")
ok(LG.formalize_objective_axes(_Agent(), _LA("mine"), _UM, {}, ["c"], MAPS) == [],
   "선언이 없으면 아예 묻지 않는다")

print("\n§4 A2 두 층 + 라이브 spec (死코드 방지·[[24]])")


def grab(fn, k):
    d = json.load(io.open(os.path.join(HERE, "a2", fn), encoding="utf-8"))
    for x in d.get("ledger_metrics", []):
        if x.get("objective_axis_prompt"):
            return x.get(k)
    return None


for k in ("objective_axes_prompt", "objective_hint_text"):
    s = grab("banking_knowledge.settings.json", k)
    g = grab("banking_knowledge.gate.json", k)
    ok(bool(s), "정본에 %s 가 있다" % k)
    ok(s == g, "%s 가 정본·gate 바이트 동일" % k)
spec = None
for x in (load_domain_a2("banking_knowledge") or {}).get("ledger_metrics", []):
    if x.get("objective_axis_prompt"):
        spec = x
ok(bool(spec and spec.get("objective_axes_prompt")), "병합된 라이브 spec 이 들고 있다")
ok(all(("{%s}" % k) in (spec or {}).get("objective_axes_prompt", "")
       for k in ("axes", "text")), "axes/text 자리표시자가 있다")
ok(all(("{%s}" % k) in (spec or {}).get("objective_hint_text", "")
       for k in ("axis", "best")), "axis/best 자리표시자가 있다")
ok("{choice}" not in (spec or {}).get("objective_hint_text", "")
   and "name" not in (spec or {}).get("objective_hint_text", "").lower(),
   "★되돌리는 것은 **값**이지 이름이 아니다 ([[05]] Q2 보존)")

print("\n§5 기존 단일-축 경로는 건드리지 않았다 (099/100 3/3 을 세운 구성)")
ok("ONE of the names" in (spec or {}).get("objective_axis_prompt", ""),
   "단일 축 문구가 그대로다", (spec or {}).get("objective_axis_prompt", "")[:60])

print("\n" + ("FAIL %d: %s" % (len(FAIL), FAIL) if FAIL else "PASS  (0 실패)"))
sys.exit(1 if FAIL else 0)
