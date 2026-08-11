# -*- coding: utf-8 -*-
"""회귀 검정: **검색 에이전트 배선** (`T2_SEARCH_AGENT`·2026-08-11·C418).

무엇을 막는 검정인가 —
 ⒜ **끈 상태에서 도는 것**. 기본 OFF 이고, OFF 면 진입점이 아예 호출되지 않아야 한다.
 ⒝ **원장이 있는 계열을 가로채는 것**. 문서 소스는 `_limit_reduce_text` 가 **침묵할 때만** 붙는다.
 ⒞ **선언 없이 도는 것**. A2 문구 세 개가 없으면 침묵한다(엔진이 문장을 짓지 않는다·[[16]]).
 ⒟ **경로를 박는 것**. 코퍼스는 **환경이 든 것**을 읽는다 — 파일 경로 상수가 없어야 한다([[05]]).
 ⒠ **한 sim 에 여러 번 도는 것**. 재료는 대화와 무관한 정책 상수라 1회다([[57]]).

오프라인 전용(LLM·서버 불요). 실행: py -3 test_search_agent_wiring.py
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

FAILED = []


def chk(c, label):
    print(("  OK   " if c else "  FAIL ") + label)
    if not c:
        FAILED.append(label)


HERE = os.path.dirname(os.path.abspath(__file__))
SRC = open(os.path.join(HERE, "t2_gate_patch.py"), encoding="utf-8").read()

print("\n§1 플래그 · 기본 OFF · 원장 우선")
chk(os.environ.get("T2_SEARCH_AGENT") != "1", "환경에 기본 ON 이 박혀 있지 않다")
chk('os.environ.get("T2_SEARCH_AGENT") == "1"' in SRC, "플래그 뒤에 있다")
chk(re.search(r"if \(not _m3 and os\.environ\.get\(\"T2_SEARCH_AGENT\"\)", SRC) is not None,
    "`_limit_reduce_text` 가 **침묵할 때만**(`not _m3`) 붙는다 — 원장 계열을 가로채지 않는다")
chk("_t2_searchagent_fired" in SRC and SRC.count("_t2_searchagent_fired") >= 2,
    "한 sim 1회 가드가 있다")

print("\n§2 진입점 — 선언이 없으면 침묵한다")
import t2_gate_patch as GP                                        # noqa: E402
chk(callable(getattr(GP, "_search_material", None)), "진입점이 모듈 수준에 있다")
chk(GP._search_material(None, None, []) == "", "A2 가 없으면 빈 문자열")
chk(GP._search_material(None, {"policy_ontology": {"doc_index": {"g": {}}}}, []) == "",
    "문구 선언이 없으면 빈 문자열(엔진이 문장을 짓지 않는다)")

print("\n§3 A2 선언 — 두 층에 같은 문구가 있다 ([[24]])")
keys = ("group_prompt", "doc_decide_prompt", "decided_by_docs_text")
layers = []
for rel in ("a2/banking_knowledge.specific.json", "a2/banking_knowledge.gate.json"):
    po = json.load(open(os.path.join(HERE, rel), encoding="utf-8"))["policy_ontology"]
    layers.append({k: po.get(k) for k in keys})
    chk(all(po.get(k) for k in keys), "%s 에 세 문구가 있다" % rel.split("/")[-1])
chk(layers[0] == layers[1], "두 층의 문구가 **바이트 동일**하다")

print("\n§4 도메인 어휘 0 — 문구가 은행을 모른다")
blob = " ".join(str(layers[0][k]) for k in keys).lower()
bank = [w for w in ("account", "referral", "card", "bank", "checking", "savings")
        if re.search(r"\b%s\b" % w, blob)]
chk(not bank, "은행 어휘가 없다 (%s)" % (bank or "없음"))
chk("{groups}" in layers[0]["group_prompt"] and "{material}" in layers[0]["doc_decide_prompt"]
    and "{choice}" in layers[0]["decided_by_docs_text"], "슬롯이 런타임에 채워진다")

print("\n§5 경로 하드코딩 0 — 코퍼스는 환경에서 온다 ([[05]])")
i = SRC.find("def _search_material(")
seg = SRC[i:i + 2600]
chk("corpus_from_env" in seg, "환경 어댑터를 쓴다")
chk("domains/" not in seg and "/home/" not in seg, "진입점에 파일 경로 상수가 없다")
chk("decide_from_docs" in seg and "formalize_group" in seg, "고르는 일은 두 번 다 LLM 이다")
chk(not re.search(r"\bsorted\(.*reverse=True\)\[0\]|\bmax\(", seg),
    "엔진이 최댓값·순위로 답을 집지 않는다 (⛔0 ④)")

print("\n%s  (%d/%d)" % ("FAIL" if FAILED else "ALL PASS", 16 - len(FAILED), 16))
sys.exit(1 if FAILED else 0)
