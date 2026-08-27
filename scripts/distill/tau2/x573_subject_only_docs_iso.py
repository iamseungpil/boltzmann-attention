# -*- coding: utf-8 -*-
r"""x573 — 그 주체의 문서**만** 남기면 갈리는가 (유료 0 · 회수된 라이브 프롬프트 위에서).

## 왜 (사용자 지시 2026-08-27: *"A3 로 정의하라. 문서들 헷갈리게 하지 말고, 인용할 문서를
   정확하게 명시하라."*)

016 의 마지막 칸은 금액이다. 틀린 수들의 출처가 전부 **다른 제품의 문서**다:

    $750 → doc_credit_cards_silver_rewards_card_011            ← 이 카드
    $150 → doc_business_credit_cards_silver_zoom_card_011      ← Silver **Zoom** Card(법인)
    $500 → doc_credit_cards_ecocard_011
    $350 → doc_business_checking_accounts_beige_012            ← 법인 당좌

**전달 결손이 아니다** — 이 카드의 문서는 이미 도착했다(t7365 s1567 msg[37]). 문제는 **같은
봉투에 다른 제품 문서 넷이 함께** 왔다는 것이다. A3 `policy_ontology.doc_index` 가
`{군: {주어: [doc id]}}` 로 그 경계를 **이미 선언**하고 있다(698 문서 전수·엔진은 읽기만 한다).

## 팔 — A3 가 **지목한 문서 하나**를 배달한다 (사용자 지시)

회수된 프롬프트를 보니 t7366 스모크에서는 이 카드의 문서가 **아예 안 왔다**(실린 10 문서 중 0).
t7365 s1567 에서는 왔지만 **다른 제품 문서 넷과 한 봉투**였다. 둘 다 같은 처방을 가리킨다 —
**주체가 정해지면 A3 가 지목하는 그 문서를 배달한다.**

    A_asis     회수된 라이브 프롬프트 그대로
    B_doc      + A3 가 그 주체·그 축에 대해 대는 문서 **축자**(`t2_search.read_docs`)
    N_len      같은 길이의 무관 문장(길이 통제·[[57]])

⛔ASK 를 붙이지 않는다. ⛔엔진이 숫자를 말하지 않는다 — **문서를 놓을 뿐**이다([[62]] ③④).

## 채점 — 닫힌 술어

답에 나온 통화 수치가 **어느 제품 문서의 것인가**를 doc_index 소속으로 가른다.
엔진은 어느 수가 정답인지 모른다 — 소속만 센다.

사용: PYTHONPATH=. py -3 x573_subject_only_docs_iso.py --port 8140 [--wiring-only]
"""
import argparse
import collections
import gzip
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import gate_interpreter as GI                                       # noqa: E402
import x559_016_row_pick_iso as X559                                # noqa: E402
import x567_numeric_arg_census as X567                              # noqa: E402
import t2_search as SRCH                                            # noqa: E402
import t2_gate_patch as G                                           # noqa: E402
import x572_anchor_on_live_prompt_iso as X572                       # noqa: E402

NL = chr(10)
SUBJECT = "silver_rewards_card"
FAMILY = "credit_cards"


def doc_ids(a2, family=FAMILY, subject=SUBJECT):
    di = (a2.get("policy_ontology") or {}).get("doc_index") or {}
    return set((di.get(family) or {}).get(subject) or ())


def split_blocks(text):
    """`ID: doc_…` 경계로 자른다 — env 가 찍는 표지다([[59]] 고정 포맷 전사)."""
    return re.split(r"(?=ID: doc_)", text)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8140)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--wiring-only", action="store_true")
    a = ap.parse_args(argv)

    a2 = GI.load_domain_a2("banking_knowledge") or {}
    # ★A3 가 이 주체의 **어느 축을 어느 문서로** 대는지 — 저작 0, 읽기만 한다.
    #   축 선별은 **프로브의 편의**다(추천 관련 축). 라이브에서 축을 고르는 것은 모델 몫이다([[22]]).
    facts = [r for r in G._policy_facts(a2) if SUBJECT in str(r.get("subject") or "")
             and str(r.get("subject") or "").startswith(FAMILY)]
    want = sorted({src.get("doc") for r in facts
                   if any(k in str(r.get("axis") or "") for k in ("referral", "qualifying"))
                   for src in (r.get("sources") or []) if src.get("doc")})
    best, best_n = None, -1
    for r in X572.prompts():
        t = str(r.get("text") or "")
        n = len(set(re.findall(r"ID: (doc_\S+)", t)))
        if n > best_n:
            best, best_n = r, n
    base = str(best.get("text"))
    have = set(re.findall(r"ID: (doc_\S+)", base))
    print("# x573 — 프롬프트 turn=%s · %d자 · 실린 문서 %d" % (best.get("turn"), len(base), len(have)))
    print("   A3 가 이 주체·추천 축에 대해 대는 문서: %s" % want)
    print("   그 문서가 프롬프트에 이미 있나: %s" % {d: (d in have) for d in want})
    if not want:
        print("A3 가 그 축의 문서를 안 댄다 — 판정하지 않는다([[25]])", file=sys.stderr)
        return 2
    try:
        docs = SRCH.read_docs(want)
    except Exception as e:
        print("문서 읽기 실패: %r" % (e,), file=sys.stderr)
        return 2
    body = NL.join("ID: %s%s%s" % (d, NL, (docs.get(d) if isinstance(docs, dict) else str(docs)))
                   for d in want)
    deliver = (NL + NL + "[KB DELIVERY] Read the following before choosing your next action. "
               "This is, in full and verbatim, the knowledge-base document the records for this "
               "card rest on." + NL + body)
    pad = (NL + NL + ("[note] the details gathered so far in this conversation remain current. "
                      * max(1, len(deliver) // 70))[:len(deliver)])
    print("   배달 %d자 · 그 안에 $750: %s" % (len(deliver), "있음" if "750" in deliver else "없음"))
    if a.wiring_only:
        print("--- 배달 앞부분 ---")
        print("   " + " ".join(deliver.split())[:300])
        return 0

    arms = {"A_asis": base, "B_doc": base + deliver, "N_len": base + pad}
    print()
    print("%-8s %-5s %-52s %s" % ("팔", "temp", "답 앞부분", "답 속 금액"))
    print("-" * 104)
    tally = collections.defaultdict(collections.Counter)
    for nm in ("A_asis", "B_doc", "N_len"):
        for tp, cnt in ((0.0, 1), (a.temp, a.n)):
            for _ in range(cnt):
                try:
                    rep = " ".join(str(X559.gen(a.port, arms[nm], 200, tp)).split())
                except Exception as e:
                    print("%-8s %-5s 호출 실패: %r" % (nm, tp, e))
                    continue
                figs = [X567.digits(g) for g in re.findall(r"\$(\d[\d,]*(?:\.\d\d)?)", rep)]
                for g in (figs or ["없음"]):
                    tally[nm][g] += 1
                print("%-8s %-5s %-52s %s" % (nm, tp, rep[:52], ", ".join("$" + f for f in figs) or "없음"))
    print()
    print("## 답에 나온 금액 분포")
    for nm in ("A_asis", "B_doc", "N_len"):
        print("   %-8s %s" % (nm, dict(tally[nm])))
    print()
    print("⚠A_asis 가 이미 이 카드의 수만 대면 결손이 아니다([[62]] 2b).")
    print("⚠N_len 이 B_doc 와 같으면 그 이득은 **길이**다([[57]]).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
