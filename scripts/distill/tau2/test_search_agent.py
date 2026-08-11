# -*- coding: utf-8 -*-
"""회귀 검정: **검색 에이전트의 결정론 부분** (`t2_search`·2026-08-10·원장 C405).

무엇을 막는 검정인가 —
 ⒜ **모르는 것을 빼는 것**. 유효 구간이 없는 문서를 만료로 몰면 살아 있는 근거가 사라진다([[25]]).
 ⒝ **조용히 빼는 것**. 뺀 것은 이유와 함께 재료에 남아야 한다(C327).
 ⒞ **없는 문서를 조용히 넘기는 것**. 링크가 가리키는데 파일이 없으면 **표시**돼야 한다.
 ⒟ **시야를 모르는 것**. 링크 커버리지는 이 에이전트가 못 보는 범위를 뜻한다([[50]] ADB).

오프라인 전용(LLM·서버 불요). 실행: py -3 test_search_agent.py
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_search as S                                            # noqa: E402

FAILED = []


def chk(c, label):
    print(("  OK   " if c else "  FAIL ") + label)
    if not c:
        FAILED.append(label)


A2 = {"policy_ontology": {"rows": [
    {"subject": "Sky Blue", "axis": "apy_pct", "source": {"doc": "doc_x_sky_blue_007"}},
    {"subject": "Sky Blue", "axis": "atm", "source": {"doc": "doc_x_sky_blue_007"}},
    {"subject": "Lime Green", "axis": "apy_pct", "source": {"doc": "doc_x_lime_green_001"}},
    {"subject": "Beige", "axis": "apy_pct", "source": {"doc": "doc_x_beige_missing"}},
]}}

print("\n§1 링크 — 중복 없이, 주어·축으로 좁혀진다")
all_docs = S.linked_docs(A2)
chk(all_docs == ["doc_x_sky_blue_007", "doc_x_lime_green_001", "doc_x_beige_missing"],
    "링크 3건·중복 제거 (%s)" % all_docs)
chk(S.linked_docs(A2, subjects={"Sky Blue"}) == ["doc_x_sky_blue_007"], "주어로 좁힌다")
chk(S.linked_docs(A2, axes={"atm"}) == ["doc_x_sky_blue_007"], "축으로 좁힌다")

d = tempfile.mkdtemp()
for name, body in (("doc_x_sky_blue_007", "APY 1.25%. Promotion runs 11/01-11/30."),
                   ("doc_x_lime_green_001", "APY 1.5%. Promotion ran 10/12-11/12.")):
    json.dump({"id": name, "content": body}, open(os.path.join(d, name + ".json"), "w",
                                                  encoding="utf-8"))

print("\n§2 읽기 — 없는 링크는 **표시**된다 (조용히 넘기지 않는다)")
docs, missing = S.read_docs(all_docs, d)
chk(set(docs) == {"doc_x_sky_blue_007", "doc_x_lime_green_001"}, "있는 것만 읽는다")
chk(missing == ["doc_x_beige_missing"], "없는 링크가 표시된다 (%s)" % missing)

print("\n§3 만료 — 준 구간으로만 거르고, 모르면 안 뺀다")
spans = {"doc_x_sky_blue_007": ("2025-11-01", "2025-11-30"),
         "doc_x_lime_green_001": ("2025-10-12", "2025-11-12")}
keep, dropped = S.drop_expired(docs, spans, "2025-11-14")
chk(set(keep) == {"doc_x_sky_blue_007"}, "만료된 것만 빠진다")
chk(dropped and dropped[0][0] == "doc_x_lime_green_001", "뺀 것이 기록된다")
keep2, dropped2 = S.drop_expired(docs, {}, "2025-11-14")
chk(set(keep2) == set(docs) and not dropped2, "구간을 모르면 **아무것도 안 뺀다**")
keep3, _ = S.drop_expired(docs, spans, None)
chk(set(keep3) == set(docs), "현재 시각을 모르면 안 뺀다")

print("\n§4 재료 — 뺀 것은 **이유와 함께** 남는다")
mat = S.as_material(keep, dropped)
chk("doc_x_sky_blue_007" in mat, "남은 문서가 축자로 실린다")
chk("Excluded as out of date" in mat and "doc_x_lime_green_001" in mat,
    "뺀 것과 그 구간이 재료에 남는다")
chk("APY 1.5%" not in mat, "뺀 문서의 본문은 실리지 않는다")

print("\n§5 시야 — 링크 커버리지를 셀 수 있다")
n_linked, n_total, ratio = S.coverage(A2, d)
chk(n_linked == 3 and n_total == 2, "링크 %d · 코퍼스 %d" % (n_linked, n_total))

print("\n§6 색인 — 빌드가 적어 둔 것을 **읽기만** 한다 (파일명 해석 0)")
A2X = {"policy_ontology": {
    "doc_index": {"business_checking_accounts": {
        "sky_blue": ["doc_business_checking_accounts_sky_blue_001",
                     "doc_business_checking_accounts_sky_blue_002"],
        "lime_green": ["doc_business_checking_accounts_lime_green_001"],
        "_general_": []},
        "bank_accounts": {"_general_": ["doc_bank_accounts_bank_accounts_(general)_013"]}},
    "doc_windows": [
        {"doc": "doc_bank_accounts_bank_accounts_(general)_013", "from": "2025-11-01",
         "to": "2025-11-30", "quote": "ACTIVE FROM 11/01/2025 TO 11/30/2025"},
        {"doc": "doc_bank_accounts_bank_accounts_(general)_014", "from": "2025-10-12",
         "to": "2025-11-12", "quote": "ACTIVE FROM 10/12/2025 TO 11/12/2025"}]}}
allb = S.docs_for(A2X, "business_checking_accounts")
chk(len(allb) == 3, "문서군 전체 3건 (%d)" % len(allb))
chk(S.docs_for(A2X, "business_checking_accounts", {"lime_green"}) ==
    ["doc_business_checking_accounts_lime_green_001"], "주어로 좁힌다")
chk(S.docs_for(A2X, "bank_accounts") ==
    ["doc_bank_accounts_bank_accounts_(general)_013"], "주어 없는 공통 문서가 딸려 온다")
chk(S.docs_for(A2X, "bank_accounts", general=False) == [], "공통을 끄면 안 딸려 온다")
chk(S.docs_for(A2X, "없는_문서군") == [], "모르는 문서군은 빈 목록(예외 아님)")

print("\n§7 선언된 유효 구간 — 조회만 하고, 없는 문서는 안 들어온다")
w = S.declared_windows(A2X)
chk(len(w) == 2 and w["doc_bank_accounts_bank_accounts_(general)_013"] ==
    ("2025-11-01", "2025-11-30"), "두 구간을 그대로 읽는다 (%s)" % len(w))
chk(list(S.declared_windows(A2X, {"doc_bank_accounts_bank_accounts_(general)_013"})) ==
    ["doc_bank_accounts_bank_accounts_(general)_013"], "문서로 좁힌다")
chk(S.declared_windows({"policy_ontology": {}}) == {}, "선언이 없으면 빈 dict")
# 색인 + 구간 + 제거가 **한 줄로 이어지는가** (071 이 요구하는 그 사슬)
docs2 = {"doc_bank_accounts_bank_accounts_(general)_013": "active promo",
         "doc_bank_accounts_bank_accounts_(general)_014": "expired promo"}
keep4, drop4 = S.drop_expired(docs2, S.declared_windows(A2X, docs2), "2025-11-14")
chk(list(keep4) == ["doc_bank_accounts_bank_accounts_(general)_013"] and len(drop4) == 1,
    "선언→비교→제거가 이어진다 (남음 %s · 뺀 것 %s)" % (list(keep4), [d for d, _f, _t in drop4]))

print("\n§8 체인 — 색인 → 읽기 → 만료 제거 → 재료 (한 함수·x243 이 정한 모양)")
d2 = tempfile.mkdtemp()
for name, body in (("doc_g_sky_001", "Sky Blue: APY 1.25%."),
                   ("doc_g_lime_001", "Lime Green: APY 1.5%."),
                   ("doc_bank_accounts_bank_accounts_(general)_013",
                    "PROMOTION ACTIVE: prefer Sky Blue."),
                   ("doc_bank_accounts_bank_accounts_(general)_014",
                    "PROMOTION: prefer Lime Green.")):
    json.dump({"id": name, "content": body},
              open(os.path.join(d2, name + ".json"), "w", encoding="utf-8"))
A2C = {"policy_ontology": {
    "doc_index": {"g": {"sky": ["doc_g_sky_001"], "lime": ["doc_g_lime_001"], "_general_": []},
                  "bank_accounts_bank_accounts": {"_general_": [
                      "doc_bank_accounts_bank_accounts_(general)_013",
                      "doc_bank_accounts_bank_accounts_(general)_014"]}},
    "doc_windows": [
        {"doc": "doc_bank_accounts_bank_accounts_(general)_013",
         "from": "2025-11-01", "to": "2025-11-30"},
        {"doc": "doc_bank_accounts_bank_accounts_(general)_014",
         "from": "2025-10-12", "to": "2025-11-12"}]}}
mat, info = S.material_for(A2C, "g", d2, "2025-11-14")
chk(info["kept"] == 3 and info["dropped"] == ["doc_bank_accounts_bank_accounts_(general)_014"],
    "제품 2 + 효력 있는 고지 1 이 남고 만료 1 이 빠진다 (%s)" % info)
chk("prefer Sky Blue" in mat and "prefer Lime Green" not in mat, "만료 고지의 본문은 재료에 없다")
chk("Excluded as out of date" in mat, "뺀 것은 이유와 함께 남는다")
mat0, info0 = S.material_for(A2C, "g", d2, "2025-11-14", windowed="none")
chk(info0["kept"] == 2 and "PROMOTION" not in mat0, "부정 통제 — 유효창을 안 실으면 고지가 없다")
chk(S.material_for(A2C, "g", d2, None)[1]["kept"] == 4, "현재 시각을 모르면 아무것도 안 뺀다")

print("\n§9 환경 어댑터 — 경로를 박지 않고 **환경이 든 문서**를 읽는다 ([[05]])")


class _Doc(object):
    def __init__(self, c):
        self.content = c


class _KB(object):
    documents = {"doc_g_sky_001": _Doc("Sky Blue: APY 1.25%.")}


class _Tools(object):
    knowledge_base = _KB()


class _Env(object):
    tools = _Tools()


cor = S.corpus_from_env(_Env())
chk(cor == {"doc_g_sky_001": "Sky Blue: APY 1.25%."},
    "도구가 든 KB 에서 문서를 꺼낸다 (%s)" % list(cor))
chk(S.corpus_from_env(object()) == {}, "못 찾으면 빈 dict — 조용한 성공보다 낫다")
got9, miss9 = S.read_docs(["doc_g_sky_001", "doc_none"], corpus=cor)
chk(list(got9) == ["doc_g_sky_001"] and miss9 == ["doc_none"],
    "corpus 경로도 없는 id 를 표시한다")
mat9, info9 = S.material_for(A2C, "g", corpus={
    "doc_g_sky_001": "Sky Blue.", "doc_g_lime_001": "Lime Green.",
    "doc_bank_accounts_bank_accounts_(general)_013": "PROMOTION ACTIVE: prefer Sky Blue.",
    "doc_bank_accounts_bank_accounts_(general)_014": "PROMOTION: prefer Lime Green."},
    now="2025-11-14")
chk(info9["kept"] == 3 and "prefer Lime Green" not in mat9,
    "체인이 corpus 로도 그대로 돈다 (%s)" % info9)

print("\n%s  (%d/%d)" % ("FAIL" if FAILED else "ALL PASS", 31 - len(FAILED), 31))
sys.exit(1 if FAILED else 0)
