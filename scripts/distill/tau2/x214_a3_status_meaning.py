# -*- coding: utf-8 -*-
r"""x214 — 상태값의 **뜻**을 A3 로 옮긴다: 결정점에서 retrieval 의존 0 (유료 0 · 데이터만).

## 왜 (사용자 지시 2026-08-10)

> *"BM25 나 embedding 이나 grep 이나 모두 불완전하다. 결정점에서의 정책 값들은 A2 A3 로
>  격리해서 서브에이전트에서 정하게 하라. **retrieval 사용하지 말라**."*
> *"혼잡 제거를 위해서 불확정성이 있는 KB 말고, 결정점에서는 A2 A3 온톨로지를 사용하게 하는 게
>  기본 설계다."*

010 이 결정점에서 KB 에 의존하던 조각은 **하나뿐**이다 —

  어느 행이 미지급인가 → **원장**(도구 출력·모델 전사)      … 온톨로지 쪽
  창 상수(9일에 2건)   → **A3 에 이미 있음**
  날짜 산수            → **엔진**
  **상태값의 뜻**      → **KB 문서뿐** ← 유일한 불확정 의존

x211 실측이 그 불확정성을 보여 준다: 같은 문서를 두고 에이전트 질의 24개 중 **12개만** 그것을
냈다(1위도 있고 아예 없는 것도 있다). 뜻은 대화마다 달라지는 값이 아니라 **고정된 정책 상수**이므로
A3 가 들 자리다.

## 지어내지 않는다

여섯 정의는 `doc_credit_cards_credit_cards_(general)_001` 의 **한 줄씩 그대로**다(EXACT 6/6 대조).
⚠문서는 하이픈이 아니라 **em-dash(—)** 를 쓴다 — 축자 인용은 그 글자까지 같아야 한다.

## 무엇을 넣지 않나

`kind`/`scope` 는 **붙이지 않는다**. 이 주어들은 제품이 아니라 상태값이고, 종류 필터의 대상이
아니다(붙이면 제품 종류 집합을 오염시킨다).

실행: py -3 x214_a3_status_meaning.py [--apply]
"""
import argparse
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
LAYERS = ["a2/banking_knowledge.specific.json", "a2/banking_knowledge.gate.json"]
DOC = "doc_credit_cards_credit_cards_(general)_001"
AXIS = "status_meaning"
AXIS_DESC = ("what a referral record's status value means, stated verbatim by the policy "
             "document that defines the statuses (a fixed policy constant, not a per-conversation "
             "value)")
DEFS = [
    ("COMPLETE", "COMPLETE — the referred person has successfully opened a new account and "
                 "met the criteria to get the referral bonus"),
    ("IN_PROGRESS", "IN_PROGRESS — the referred person has successfully opened a new account "
                    "and is in progress to meet the criteria for the referral bonus"),
    ("NO_PROGRESS", "NO_PROGRESS — the referred person has not applied yet, no progress has "
                    "been made"),
    ("APPLIED", "APPLIED — the referred person has sent in the application and is waiting "
                "for a decision"),
    ("REJECTED", "REJECTED — the user has too many referral processes going on"),
    ("ERROR", "ERROR — an error has occurred throughout the process"),
]
BASIS = ("**도구 경계에서 판정되지 않는다** — 원장 행의 상태값을 읽을 때 쓰는 고정 상수다. "
         "값은 정책 문서가 정의로 못박은 것이라 대화마다 달라지지 않는다.")


def row(subject, text):
    return {"applies_to": {"consumers": [], "basis": BASIS},
            "subject": subject, "axis": AXIS, "value": text,
            "against": None, "compare": None, "when": [],
            "source": {"doc": DOC, "quote": text, "quote_match": "exact"}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    staged = {}
    for rel in LAYERS:
        p = os.path.join(HERE, rel)
        txt = io.open(p, encoding="utf-8").read()
        doc = json.loads(txt)
        if json.dumps(doc, ensure_ascii=False, indent=1) + ("\n" if txt.endswith("\n") else "") != txt:
            print("  중단: %s 재직렬화가 바이트 동일하지 않다" % rel)
            return 1
        po = doc["policy_ontology"]
        po.setdefault("axes", {})
        if po["axes"].get(AXIS) != AXIS_DESC:
            po["axes"][AXIS] = AXIS_DESC
        rows = po["rows"]
        have = {(r.get("subject"), r.get("axis")) for r in rows}
        n = 0
        for subj, text in DEFS:
            if (subj, AXIS) in have:
                print("  건너뜀 (이미 있다): %s" % subj)
                continue
            rows.append(row(subj, text))
            n += 1
        staged[rel] = (doc, n, txt.endswith("\n"))
        print("  %-38s +%d행 → %d행" % (rel, n, len(rows)))

    a_rows = staged[LAYERS[0]][0]["policy_ontology"]["rows"]
    b_rows = staged[LAYERS[1]][0]["policy_ontology"]["rows"]
    same = json.dumps(a_rows, ensure_ascii=False, sort_keys=True) == \
        json.dumps(b_rows, ensure_ascii=False, sort_keys=True)
    print("  두 층 rows 동일: %s" % ("OK" if same else "**불일치**"))
    if not same:
        return 1
    # 종류 집합을 오염시키지 않았는지 확인 (상태 주어에는 kind 가 없어야 한다)
    bad = [r.get("subject") for r in a_rows if r.get("axis") == AXIS and r.get("kind")]
    print("  상태 주어에 kind 가 붙지 않았다: %s" % ("OK" if not bad else bad))
    if bad:
        return 1
    if not a.apply:
        print("\n(미적용 — 쓰려면 --apply)")
        return 0
    for rel, (doc, _n, nl) in staged.items():
        io.open(os.path.join(HERE, rel), "w", encoding="utf-8", newline="").write(
            json.dumps(doc, ensure_ascii=False, indent=1) + ("\n" if nl else ""))
        print("  wrote %s" % rel)
    return 0


if __name__ == "__main__":
    sys.exit(main())
