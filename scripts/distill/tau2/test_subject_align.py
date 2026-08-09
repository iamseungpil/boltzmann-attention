# -*- coding: utf-8 -*-
"""주어 정합 회귀 — 원장 표기와 A3 주어를 잇는 자리가 살아 있는지 (순수함수·GPU 0).

근거 원장: C376. 고치는 결함 = 라이브 전수에서 `unmatched_text` 77회 발화 중 A3 주어와
정확일치 **0**, `exhausted_text` 발화 **0회**. 둘 다 원장 표기(`… Account`)와 A3 주어
(접미사 없음)를 **정확 일치**로 맞대서 생긴 것이다.

⚠이 파일의 고정물은 **도메인 무관 합성**이다 — 엔진만 시험한다. 라이브 이름은
A3·사이드카에서 **읽어서** 확인한다(테스트가 도메인 리터럴을 저작하지 않는다).

실행: py -3 test_subject_align.py
"""
import glob
import gzip
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
SIMS = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "sim_results")


def ok(cond, label, extra=""):
    print(("  OK   " if cond else "  FAIL ") + label + (("  — " + str(extra)) if extra else ""))
    if not cond:
        FAIL.append(label)


# ── 합성 고정물: 원장은 접미사를 달고, 상한은 안 단다 (라이브와 같은 *형태*·도메인 무관) ──
TALLY = {"Alpha Unit": 3, "Bravo Unit": 9, "Zulu Unit": 1}
LIMITS = {"Alpha": (5, "docA"), "Bravo": (9, "docB")}
SPEC = {"exhausted_text": "none left for {exhausted}. room: {remaining}.",
        "unmatched_text": "not checked: {unmatched}.",
        "subject_align_prompt": "subjects:\n{subjects}\ngroups:\n{groups}"}


class _Sub(object):
    def __init__(self, content):
        self.content = content


class _LA(object):
    """la.generate 대역 — 정해진 답만 돌려준다(GPU 0)."""

    def __init__(self, content):
        self.content = content
        self.seen = None

    def generate(self, model=None, tools=None, messages=None, **kw):
        self.seen = messages[0].content
        return _Sub(self.content)


class _UM(object):
    def __init__(self, role=None, content=None):
        self.content = content


class _Agent(object):
    def __init__(self):
        self.llm = None
        self.llm_args = {}


print("\n§1 부정 통제 — 정합이 없으면 라이브 결함이 그대로 재현된다")
ex0 = LG.exhausted_text(TALLY, LIMITS, SPEC)
un0 = LG.unmatched_text(TALLY, LIMITS, SPEC)
ok(ex0 == "", "정합 없으면 exhausted 가 침묵한다 (라이브 0회의 원인)", repr(ex0[:40]))
ok("Alpha Unit" in un0 and "Bravo Unit" in un0,
   "정합 없으면 unmatched 가 전 그룹을 이름으로 부른다 (라이브 77회의 원인)")

print("\n§2 align_tally — 정렬된 것은 주어 이름으로, 못 고른 것은 원래 이름으로")
al, left = LG.align_tally(TALLY, {"Alpha Unit": "Alpha", "Bravo Unit": "Bravo"})
ok(al == {"Alpha": 3, "Bravo": 9}, "정렬된 그룹이 A3 주어 키로 옮겨진다", al)
ok(left == {"Zulu Unit": 1}, "못 고른 그룹만 남는다", left)
al2, _ = LG.align_tally({"A x": 2, "A y": 3}, {"A x": "Alpha", "A y": "Alpha"})
ok(al2 == {"Alpha": 5}, "같은 주어로 정렬된 둘은 합산된다", al2)
ok(LG.align_tally({}, {}) == ({}, {}), "빈 누계는 빈 결과")

print("\n§3 정합 후에는 두 소비자가 제대로 말한다")
ex1 = LG.exhausted_text(al, LIMITS, SPEC)
un1 = LG.unmatched_text(left, LIMITS, SPEC)
ok("Bravo 9/9" in ex1, "한도에 닿은 그룹을 소진으로 말한다 (전에는 불가능했다)", ex1)
ok("Alpha 3/5" in ex1, "남은 그룹은 남은 것으로 말한다")
ok("Zulu Unit" in un1 and "Alpha" not in un1,
   "unmatched 는 **정말 못 고른 것만** 부른다", un1)

print("\n§4 엔진의 몫은 집합 검사뿐 — 값이 A3 주어가 아니면 버린다 ([[22]])")
ag = _Agent()
la = _LA(json.dumps({"Alpha Unit": "Alpha", "Bravo Unit": "Not A Subject",
                     "Ghost Unit": "Alpha"}))
got = LG.formalize_subject_align(ag, la, _UM, SPEC, list(TALLY), list(LIMITS))
ok(got == {"Alpha Unit": "Alpha"}, "주어 집합 밖 값과 그룹 집합 밖 키를 둘 다 버린다", got)
ok("Alpha" in (la.seen or "") and "Alpha Unit" in (la.seen or ""),
   "프롬프트에 두 목록이 다 실린다")
ag2 = _Agent()
ok(LG.formalize_subject_align(ag2, _LA("I cannot answer that."), _UM, SPEC,
                              list(TALLY), list(LIMITS)) == {},
   "JSON 이 아니면 빈 정합 (무발화)")
ag3 = _Agent()
ok(LG.formalize_subject_align(ag3, _LA("{}"), _UM, {}, list(TALLY), list(LIMITS)) == {},
   "선언이 없으면 아예 묻지 않는다")
ag4 = _Agent()
la4 = _LA(json.dumps({"Alpha Unit": "Alpha"}))
LG.formalize_subject_align(ag4, la4, _UM, SPEC, list(TALLY), list(LIMITS))
la4.seen = None
LG.formalize_subject_align(ag4, la4, _UM, SPEC, list(TALLY), list(LIMITS))
ok(la4.seen is None, "한 번 물으면 그 sim 내내 재사용한다 (호출 예산)")

print("\n§5 A2 정본·gate 두 층이 같은 선언을 갖는다 ([[24]])")


def grab(fn):
    d = json.load(io.open(os.path.join(HERE, "a2", fn), encoding="utf-8"))
    for x in d.get("ledger_metrics", []):
        if x.get("unmatched_text"):
            return x.get("subject_align_prompt")
    return None


s1, g1 = grab("banking_knowledge.settings.json"), grab("banking_knowledge.gate.json")
ok(bool(s1), "정본에 subject_align_prompt 가 있다")
ok(s1 == g1, "정본과 gate 가 바이트 동일")
spec = None
for x in (load_domain_a2("banking_knowledge") or {}).get("ledger_metrics", []):
    if x.get("unmatched_text"):
        spec = x
ok(bool(spec and spec.get("subject_align_prompt")),
   "병합된 라이브 spec 이 그것을 들고 있다 (死코드 방지)")
ok(all(("{%s}" % k) in (spec or {}).get("subject_align_prompt", "")
       for k in ("subjects", "groups")), "필요한 자리표시자가 있다")

print("\n§6 라이브 이름으로 확인 — 고쳐야 할 그룹이 실재하는가 (읽기만·저작 0)")
a2 = load_domain_a2("banking_knowledge")
subj = {str(r.get("subject", "")).strip()
        for r in ((a2.get("policy_ontology") or {}).get("rows") or [])
        if r.get("axis") == "annual_referral_limit"}
mark = str((spec or {}).get("unmatched_text") or "").split("{")[0].strip()
named = set()
for f in glob.glob(os.path.join(SIMS, "fb_bank_*.jsonl.gz")):
    for line in gzip.open(f, "rt", encoding="utf-8"):
        if not line.strip():
            continue
        t = str(json.loads(line).get("text") or "")
        i = t.find(mark) if mark else -1
        if i < 0:
            continue
        seg = t[i + len(mark):].split(", which was NOT")[0]
        for g in seg.split(";"):
            g = " ".join(g.split())
            if g:
                named.add(g.rsplit(" ", 1)[0] if g.rsplit(" ", 1)[-1].isdigit() else g)
ok(bool(subj), "A3 에 상한 주어가 있다", len(subj))
if named:
    exact = sorted(n for n in named if n in subj)
    ok(not exact, "라이브에서 지목된 그룹 중 A3 주어와 정확일치한 것은 없었다 (=결함 재현)",
       "%d개 지목 / 정확일치 %d" % (len(named), len(exact)))
else:
    print("  SKIP  사이드카에 unmatched 발화가 없다 (계량 불가)")

print("\n" + ("FAIL %d: %s" % (len(FAIL), FAIL) if FAIL else "PASS  (0 실패)"))
sys.exit(1 if FAIL else 0)
