#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""`T2_CARD_DOCS` 래칫 — 실제 선언·실제 문서로 초 단위 검정.

## 무엇을 잠그나 (측정 정본 = `x574_subject_docs_subagent_iso.py`)

사용자 지시: *"격리 서브에이전트는 자신에게 관계된 문서만 받고 그것만 읽고 결정해야 한다.
그러기 위해서 A3 에 관련 문서들을 index 로 정의한 거다."*

⑴ 이름 → 색인 소속이 **닫힌 집합 대조**로 갈린다 — `Silver Rewards Card` 는
   `credit_cards/silver_rewards_card` 이고, 헷갈리는 이웃(`silver_zoom_card` ·
   `business_silver_rewards_card`)은 **다른 주어**다.
⑵ 그 주어의 문서 집합에 A3 가 `qualifying_spend` 의 출처로 대는 문서가 들어 있다.
⑶ 서브의 답이 **우리가 준 문서 id 를 인용하지 않으면 침묵**한다([[22]]·[[25]]).
⑷ 선언 두 칸이 **양 층**에 있다([[24]]).
⑸ 플래그로만 켜진다.

## ⛔여기서 판정하지 않는 것
*"배달하면 통과하는가"* 는 런이 잰다([[69]]).

실행: PYTHONIOENCODING=utf-8 py -3 test_card_docs.py
"""
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import gate_interpreter as GI                                       # noqa: E402
import t2_gate_patch as G                                           # noqa: E402
import t2_ledger as LG                                              # noqa: E402

FAIL = []
A2 = GI.load_domain_a2("banking_knowledge") or {}
IDX = (A2.get("policy_ontology") or {}).get("doc_index") or {}


def chk(c, ok, extra=""):
    print(("  OK   " if ok else "  FAIL ") + c + (("  — " + extra) if extra else ""))
    if not ok:
        FAIL.append(c)


def subject_of(name):
    for g, subs in IDX.items():
        for s in G._subject_keys(subs):
            if G._slug_disp(s).strip().lower() == name.strip().lower():
                return (g, s)
    return None


print("## ⑴ 이름 → 주어 (닫힌 집합)")
got = subject_of("Silver Rewards Card")
chk("Silver Rewards Card → credit_cards/silver_rewards_card",
    got == ("credit_cards", "silver_rewards_card"), str(got))
zoom = subject_of("Silver Zoom Card")
chk("Silver Zoom Card 는 **다른 주어**", zoom is not None and zoom != got, str(zoom))

print()
print("## ⑵ 그 주어의 문서에 A3 가 대는 출처가 있다")
ids = list((IDX.get("credit_cards") or {}).get("silver_rewards_card") or ())
facts = [r for r in G._policy_facts(A2)
         if str(r.get("subject") or "") == "credit_cards_silver_rewards_card"
         and "qualifying" in str(r.get("axis") or "")]
srcs = sorted({s.get("doc") for r in facts for s in (r.get("sources") or []) if s.get("doc")})
chk("A3 가 qualifying_spend 의 출처를 댄다", bool(srcs), str(srcs))
chk("그 출처가 색인의 그 주어 안에 있다", all(d in ids for d in srcs), "%d/%d" % (len(ids), len(srcs)))

print()
print("## ⑶ 인용 없으면 침묵")
spec = next(s for s in (A2.get("ledger_metrics") or []) if s.get("diagnose_prompt"))


class _Ag(object):
    pass


class _FakeSC(object):
    def __init__(self, reply):
        self.reply = reply

    def sub_generate(self, *a, **k):
        return self.reply


import t2_subcall as SC                                              # noqa: E402
_orig = SC.sub_generate
try:
    SC.sub_generate = lambda *a, **k: "The referred person must spend at least $750."
    out = LG.requirement_choice(_Ag(), None, None, spec, "ID: doc_x\nbody", "Silver Rewards Card",
                                ["doc_x"])
    chk("인용 없는 답 → None", out is None, repr(out))
    SC.sub_generate = lambda *a, **k: "Spend $750 (doc_x)."
    out2 = LG.requirement_choice(_Ag(), None, None, spec, "ID: doc_x\nbody", "Silver Rewards Card",
                                 ["doc_x"])
    chk("인용 있는 답 → 그대로", out2 is not None and "750" in str(out2), repr(out2))
finally:
    SC.sub_generate = _orig

print()
print("## ⑷ 선언이 양 층에 · ⑸ 플래그로만")
for p in ("a2/banking_knowledge.settings.json", "a2/banking_knowledge.gate.json"):
    d = json.load(io.open(os.path.join(HERE, p), encoding="utf-8"))
    has = any(s.get("requirement_prompt") and s.get("requirement_text")
              for s in (d.get("ledger_metrics") or []))
    chk("%s 에 선언 두 칸" % p.split("/")[-1][:28], has)
SRC = io.open(os.path.join(HERE, "t2_gate_patch.py"), encoding="utf-8").read()
chk("플래그 검사가 소스에 있다", 'os.environ.get("T2_CARD_DOCS") == "1"' in SRC)
chk("엔진이 색인을 **읽기만** 한다(검색 호출 없음)",
    "corpus_from_env" in SRC and "KB_search" not in SRC.split("T2_CARD_DOCS")[1][:1500])

print()
print("결과: %s" % ("모두 통과" if not FAIL else "실패 %d — %s" % (len(FAIL), FAIL)))
sys.exit(1 if FAIL else 0)
