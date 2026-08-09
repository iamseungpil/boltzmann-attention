# -*- coding: utf-8 -*-
r"""회귀 — 종류 필터: **LLM 이 고르고 엔진은 뺀다** (C389 · GPU 0 · 순수함수).

근거(x201·격리·n=8·32B): `A_iso` 0/8(카드를 집는다) · **`E_hint` 0/8**(한 줄 전달로는 안 된다)
· `F_kind` 8/8 · `G_llm` 8/8(종류 선택 8/8 정확). 전달 팔을 **먼저 재서 실패**했기에 필터가
정당하다(⛔0 ②).

무엇을 막는가 —
 ⒜ **엔진이 종류를 정하는 것.** 종류는 LLM 이 고르고, 엔진은 그 답이 A3 종류 집합의 원소인지만
    본다. 못 고르면 **아무것도 안 거른다**(종전 거동).
 ⒝ **모름을 탈락으로 바꾸는 것.** 종류가 없는(또는 갈리는) 주어는 표에 남는다([[25]]).
 ⒞ **지어낸 종류.** 모든 종류 값은 그 행이 인용한 출처 문서군과 일치해야 한다([[23]]).
 ⒟ **조용한 죽음.** A2 선언(`kind_field`·`kind_prompt`)이 두 층에 다 있고 라이브 병합본이
    들고 있어야 한다([[24]]).

실행: py -3 test_kind_filter.py
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
import t2_factdag as FD                                         # noqa: E402
from gate_interpreter import load_domain_a2                      # noqa: E402

FAIL = []
HERE = os.path.dirname(os.path.abspath(__file__))


def ok(cond, label, extra=""):
    print(("  OK   " if cond else "  FAIL ") + label + (("  — " + str(extra)) if extra else ""))
    if not cond:
        FAIL.append(label)


class _Sub(object):
    def __init__(self, c):
        self.content = c


class _LA(object):
    def __init__(self, c):
        self.content, self.seen = c, None

    def generate(self, model=None, tools=None, messages=None, **kw):
        self.seen = messages[0].content
        return _Sub(self.content)


class _Agent(object):
    def __init__(self):
        self.llm, self.llm_args = None, {}


class _UM(object):
    def __init__(self, role=None, content=None):
        self.content = content


print("\n§1 순수함수 — 도메인 무관 고정물")
ROWS = [{"subject": "A", "k": "fruit"}, {"subject": "B", "k": "fruit"},
        {"subject": "C", "k": "metal"}, {"subject": "D"},
        {"subject": "E", "k": "fruit"}, {"subject": "E", "k": "metal"}]
kb = LG.subject_kinds(ROWS, "k")
ok(kb == {"A": "fruit", "B": "fruit", "C": "metal"}, "행의 선언된 필드로 주어별 종류를 모은다", kb)
ok("D" not in kb, "종류가 없는 주어는 지도에 없다")
ok("E" not in kb, "종류가 **갈리는** 주어는 강제하지 않는다 (빼지도 넣지도 않는다)")
MAPS = {"x": {"A": (1, "d"), "B": (2, "d"), "C": (3, "d"), "D": (4, "d"), "E": (5, "d")}}
got, drop = LG.restrict_to_kind(MAPS, kb, "fruit")
ok(sorted(got["x"]) == ["A", "B", "D", "E"], "그 종류가 아닌 주어만 빠진다", sorted(got["x"]))
ok(drop == ["C"], "뺀 것을 말한다 (조용히 빼지 않는다)", drop)
ok(LG.restrict_to_kind(MAPS, kb, None) == (MAPS, []),
   "★종류를 못 고르면 **아무것도 안 거른다** (종전 거동)")
ok(LG.restrict_to_kind(MAPS, {}, "fruit") == (MAPS, []), "종류 지도가 비면 안 거른다")

print("\n§2 formalize_kind — 엔진은 집합 검사만 ([[22]])")
SPEC = {"kind_prompt": "kinds:\n{kinds}\nconv:\n{text}"}
ag, la = _Agent(), _LA("fruit")
ok(LG.formalize_kind(ag, la, _UM, SPEC, ["conversation"], ["fruit", "metal"]) == "fruit",
   "집합 안의 답은 받는다")
ok("fruit" in (la.seen or "") and "metal" in (la.seen or ""), "후보를 다 보여 준다")
ok(LG.formalize_kind(_Agent(), _LA("plastic"), _UM, SPEC, ["c"], ["fruit", "metal"]) is None,
   "집합 밖 답은 버린다 (거르지 않는다)")
ok(LG.formalize_kind(_Agent(), _LA("fruit"), _UM, {}, ["c"], ["fruit"]) is None,
   "선언이 없으면 아예 묻지 않는다")
ok(LG.formalize_kind(_Agent(), _LA("fruit"), _UM, SPEC, ["c"], []) is None,
   "후보가 없으면 묻지 않는다")
ag2, la2 = _Agent(), _LA("fruit")
LG.formalize_kind(ag2, la2, _UM, SPEC, ["c"], ["fruit"])
la2.seen = None
LG.formalize_kind(ag2, la2, _UM, SPEC, ["c"], ["fruit"])
ok(la2.seen is None, "성공은 기억한다 (호출 예산)")

print("\n§3 라이브 A2/A3 — 선언과 데이터가 실재한다 ([[24]])")
a2 = load_domain_a2("banking_knowledge") or {}
rows = (a2.get("policy_ontology") or {}).get("rows") or []
spec = next((s for s in (a2.get("ledger_metrics") or []) if s.get("eligible_text")), None)
cfg = (spec or {}).get("eligible") or {}
ok(cfg.get("kind_field") == "kind", "라이브 병합본이 kind_field 를 들고 있다", cfg.get("kind_field"))
ok("{kinds}" in (cfg.get("kind_prompt") or "") and "{text}" in (cfg.get("kind_prompt") or ""),
   "kind_prompt 자리표시자가 있다")
for rel in ("a2/banking_knowledge.settings.json", "a2/banking_knowledge.gate.json"):
    d = json.load(io.open(os.path.join(HERE, rel), encoding="utf-8"))
    g = [(s.get("eligible") or {}).get("kind_prompt") for s in (d.get("ledger_metrics") or [])
         if (s.get("eligible") or {}).get("kind_prompt")]
    ok(g and g[0] == cfg.get("kind_prompt"), "%s 가 같은 문구를 들고 있다" % rel)

kb2 = LG.subject_kinds(rows, "kind")
ok(len(kb2) >= 30, "A3 주어 대부분에 종류가 붙어 있다 (%d)" % len(kb2))

print("\n§4 ⛔종류는 **출처에서** 나왔다 — 지어낸 값이 없다 ([[23]])")
bad = []
for r in rows:
    k = r.get("kind")
    doc = (r.get("source") or {}).get("doc") or ""
    if k and not doc.startswith("doc_" + k + "_"):
        bad.append((r.get("subject"), k, doc))
ok(not bad, "모든 종류 값이 그 행이 인용한 문서군과 일치한다", bad[:3] or "일치")
sc = {r.get("scope") for r in rows if r.get("scope")}
ok(sc <= {"product", "general"}, "범위 값은 둘뿐이다", sorted(sc))
for r in rows:
    if r.get("scope") == "general":
        ok("_(general)_" in ((r.get("source") or {}).get("doc") or ""),
           "범위 주어 %s 는 (general) 문서에서 왔다" % r.get("subject"))
        break

print("\n§5 098 의 자리 — 종류로 거르면 표에 무엇이 남나")
axes = cfg.get("show_axes") or []
maps = {ax: FD._a3_map(rows, {"axis": ax}) for ax in axes}
tbl = LG.eligible_text(400, {}, maps, spec, {"qualifying_deposit_usd": 600}) or ""
named = {l.strip().split(":")[0].strip() for l in tbl.splitlines() if l.startswith("  ")}
ok(any(kb2.get(s) == "business_credit_cards" for s in named),
   "거르기 전에는 카드가 표에 있다 (=라이브 오답의 자리)")
kept, dropped = LG.restrict_to_kind(maps, kb2, "checking_accounts")
tbl2 = LG.eligible_text(400, {}, kept, spec, {"qualifying_deposit_usd": 600}) or ""
named2 = {l.strip().split(":")[0].strip() for l in tbl2.splitlines() if l.startswith("  ")}
ok("Blue" in named2, "gold 인 `Blue` 는 남는다")
ok(not any(kb2.get(s) in ("credit_cards", "business_credit_cards") for s in named2),
   "카드는 하나도 안 남는다", sorted(s for s in named2 if "Card" in s))
ok(all(kb2.get(s) in (None, "checking_accounts") for s in named2),
   "남은 것은 그 종류이거나 종류를 모르는 것뿐이다", sorted(named2))
# 고르는 일은 여전히 모델 몫 — 표에 후보가 **여럿** 남아야 한다
ok(len(named2) >= 3, "거른 뒤에도 후보가 여럿이다 (%d) — 엔진이 답을 좁혀 주지 않는다" % len(named2))

print("\n%s  (%d 실패)" % ("PASS" if not FAIL else "FAIL", len(FAIL)))
sys.exit(1 if FAIL else 0)
