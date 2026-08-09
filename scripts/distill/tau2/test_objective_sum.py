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


for k in ("objective_axes_prompt", "rederive_by_axis_prompt", "_note_objective_axes"):
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
# ★C382: 되묻는 문구는 `rederive_prompt` 가 아니라 **자기 문구**다. 그 문구는 *"손님의 말이
#   하나를 짚지 못하면 NONE"* 인데 `asked` 가 의도적으로 비어 있어(x158) NONE 이 옳은 답이 된다
#   — 실측 y098: 값은 `best=65`(gold)로 맞았는데 세 번 다 무응답.
byax = (spec or {}).get("rederive_by_axis_prompt", "")
ok(bool(byax), "되묻는 자기 문구가 선언돼 있다")
ok(all(("{%s}" % k) in byax for k in ("table", "facts", "axis", "ranking")),
   "table/facts/axis/**ranking** 자리표시자가 있다 (합계는 표에 없으므로 엔진이 실어 준다)")
ok("customer's words" not in byax,
   "★손님의 말에 기대지 않는다 (그 지시가 빈 `asked` 에서 NONE 을 강제했다)")
ok("{choice}" not in byax, "우리가 이름을 넣어 주지 않는다 — 표에서 찾는 것은 서브다([[05]] Q2)")

print("\n§5 기존 단일-축 경로는 건드리지 않았다 (099/100 3/3 을 세운 구성)")
ok("ONE of the names" in (spec or {}).get("objective_axis_prompt", ""),
   "단일 축 문구가 그대로다", (spec or {}).get("objective_axis_prompt", "")[:60])

print("\n§6 라이브 부검이 잡은 결함 — 물어본 축을 엔진이 풀 수 있어야 한다 (C377b)")
# 첫 라이브 런에서 098 은 정답 쌍을 냈는데 `referred_bonus_usd` 에 `a3_map` 노드가 없어
# 지도가 비었고, 순위가 0 이 되어 재질의가 **아무 흔적 없이** 안 나갔다. 010 은 아예 상한
# 축을 답했다 — 13축을 다 보여 주면서 5축만 풀 수 있었기 때문이다.
A2 = load_domain_a2("banking_knowledge") or {}
resolvable = {(n.get("params") or {}).get("axis")
              for n in (A2.get("derived") or ()) if n.get("op") == "a3_map"}
elig = next((x.get("eligible") or {}) for x in A2.get("ledger_metrics", [])
            if x.get("eligible_text"))
shown = list(elig.get("show_axes") or [])
missing = [a for a in shown if a not in resolvable]
ok(not missing, "★표에 보여 주는 축은 전부 조회 가능해야 한다 (순위를 못 매기면 죽는다)",
   missing or sorted(resolvable))
ok("referred_bonus_usd" in resolvable,
   "098 이 필요로 한 축에 a3_map 노드가 있다 (첫 런에서 없어서 침묵했다)")

# ★A2 가 후보 축을 선언한다 (사용자 지적: "왜 13축을 다 묻나"). 목록이 둘로 갈리면 이번
#   결함이 그대로 재발하므로, 선언과 조회 가능성의 **일치**를 여기서 강제한다.
sp2 = next(x for x in A2.get("ledger_metrics", []) if x.get("objective_axis_prompt"))
cand = sp2.get("objective_axes")
ok(bool(cand), "A2 가 목적 후보 축을 선언한다", cand)
ok(not [a for a in (cand or []) if a not in resolvable],
   "★선언된 후보 축은 전부 조회 가능하다 (두 목록이 갈리면 조용히 죽는다)",
   [a for a in (cand or []) if a not in resolvable])
axes_all = set((A2.get("policy_ontology") or {}).get("axes") or {})
ok(not [a for a in (cand or []) if a not in axes_all],
   "선언된 후보 축은 전부 A3 축 집합의 원소다 ([[22]])")
ok(len(cand or []) < len(axes_all),
   "후보가 전체보다 좁다 — 제약 축은 목적이 될 수 없다 (010 이 한도 축을 답했다)",
   "%d / %d" % (len(cand or []), len(axes_all)))
# 한 축이라도 못 풀면 합은 성립하지 않는다 — 조용히가 아니라 **빈 지도**로 드러난다
_, gone = LG.combine_axes({"mine": MINE}, ["mine", "theirs"])
ok(gone == {}, "지도 없는 축이 섞이면 합은 빈다 (부분합을 만들지 않는다)", gone)

print("\n§7 최댓값은 **자격이 대조된** 주어에서만 (C381·라이브 부검)")
# 라이브 실측: x098 `best=125` · x010 `best=850` 이 나왔고 서브는 무응답이었다. 그 값을 만든
# 주어는 그 기준 값이 A3 에 없어 **걸러지지 않았을 뿐** 자격이 확인된 적이 없다.
VSPEC = {"eligible": {"criteria": [{"axis": "floor", "operand": "stated", "compare": "ge"}],
                      "show_axes": ["mine", "theirs", "floor"]},
         "eligible_text": "x{eligible}"}
VMAPS = {"mine": {"Alpha": (10, "d"), "Ghost": (900, "d")},
         "theirs": {"Alpha": (10, "d"), "Ghost": (900, "d")},
         "floor": {"Alpha": (5, "d")}}          # Ghost 는 기준 값이 **없다**
ver = LG.verified_subjects(None, None, VMAPS, VSPEC, {"floor": 100})
ok(ver == {"Alpha"}, "기준 값이 없는 주어는 **대조된 것이 아니다**", sorted(ver))
ok("Ghost" not in ver, "★거르지 않은 것과 확인한 것은 다르다 (Ghost 는 표엔 남지만 최댓값을 못 만든다)")
rows_v = [("Alpha", ["mine=10"]), ("Ghost", ["mine=900"])]
_l, cmb = LG.combine_axes(VMAPS, ["mine", "theirs"])
ok(LG._rank_by(rows_v, cmb)[0][0] == "Ghost", "구판 최댓값은 미대조 주어가 만든다 (=결함 재현)")
ok(LG._rank_by([r for r in rows_v if r[0] in ver], cmb)[0][0] == "Alpha",
   "신판 최댓값은 대조된 주어에서 나온다")
ok(LG.verified_subjects(None, None, VMAPS, {"eligible": {}}, {}) == set(),
   "걸 수 있는 기준이 하나도 없으면 **빈 집합** (통과 도장을 찍지 않는다)")

print("\n" + ("FAIL %d: %s" % (len(FAIL), FAIL) if FAIL else "PASS  (0 실패)"))
sys.exit(1 if FAIL else 0)
