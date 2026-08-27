# -*- coding: utf-8 -*-
r"""x574 — 격리 서브가 **자기 카드 문서만** 받으면 요건을 옳게 내는가 (유료 0).

## 설계 (사용자 지시 2026-08-27)

> *"격리 서브에이전트는 자신에게 관계된 문서만 받고 그것만 읽고 결정해야 한다.
>   그러기 위해서 A3 에 관련 문서들을 index 로 정의한 거다."*

A3 `policy_ontology.doc_index` = `{군: {주어: [doc id …]}}` · 698 문서 전수 · 엔진은 읽기만 한다.
그 색인이 이 자리를 정확히 가른다:

    credit_cards / silver_rewards_card              → …silver_rewards_card_001..011  ($750 은 _011)
    business_credit_cards / silver_zoom_card        → …silver_zoom_card_*            ($150)
    business_credit_cards / business_silver_rewards_card → …_*                       ($3,000)
    credit_cards / ecocard                          → …ecocard_*                     ($500)

016 실측: 틀린 수 넷이 **전부 다른 주어의 문서**에서 왔다(t7365 s1567 · x570·x573).
그리고 라이브는 이 문서들을 **한 봉투에 섞어** 받는다(회수 프롬프트 turn=50: 문서 10개 중
이 카드 것 **0개**).

## 팔

    A_only    이 주어의 문서**만**                    ← 설계가 말하는 상태
    B_mixed   같은 문서 + 라이브가 실제로 준 다른 주어 문서들   ← 지금 상태
    N_none    문서 없이 물음만                        ← 부정통제(모델 사전지식 확인)

## 채점 — 닫힌 술어 · gold 무참조([[23]])

답이 댄 수와 **문서 id**를 본다. 정답 후보는 A3 가 그 주어·추천 축에 대해 대는 값이다
(`qualifying_spend = $750` · 출처 `doc_credit_cards_silver_rewards_card_011`).

사용: (리모트) PYTHONPATH=. py -3 x574_subject_docs_subagent_iso.py --port 8140 [--wiring-only]
"""
import argparse
import collections
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
import t2_gate_patch as G                                           # noqa: E402
import t2_search as SRCH                                            # noqa: E402
import x559_016_row_pick_iso as X559                                # noqa: E402
import x567_numeric_arg_census as X567                              # noqa: E402
import x572_anchor_on_live_prompt_iso as X572                       # noqa: E402

NL = chr(10)
DOCS = "/home/woori/scratch/tau2-bench/data/tau2/domains/banking_knowledge/documents"
FAMILY, SUBJECT = "credit_cards", "silver_rewards_card"
CARD = "Silver Rewards Card"
ASK = (NL + NL + "The customer referred someone who was approved for the " + CARD + " and has "
       "not received the referral bonus. Using only the document(s) above, what must the referred "
       "person do for that bonus to be paid? Answer in one sentence, giving the figure and the "
       "document id it comes from.")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8140)
    ap.add_argument("--docs", default=DOCS)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--wiring-only", action="store_true")
    a = ap.parse_args(argv)

    a2 = GI.load_domain_a2("banking_knowledge") or {}
    idx = (a2.get("policy_ontology") or {}).get("doc_index") or {}
    mine_ids = list((idx.get(FAMILY) or {}).get(SUBJECT) or ())
    if not mine_ids:
        print("색인에 그 주어가 없다", file=sys.stderr)
        return 2
    # A3 가 그 주어·추천 축에 대해 대는 값과 문서 (채점용 · 프롬프트에 안 들어간다)
    facts = [r for r in G._policy_facts(a2)
             if str(r.get("subject") or "") == "%s_%s" % (FAMILY, SUBJECT)
             and any(k in str(r.get("axis") or "") for k in ("qualifying", "referral"))]
    want_fig = sorted({X567.digits(m) for r in facts
                       for m in re.findall(r"\$(\d[\d,]*)", str(r.get("value")))})
    want_doc = sorted({s.get("doc") for r in facts for s in (r.get("sources") or []) if s.get("doc")})

    mine, miss1 = SRCH.read_docs(mine_ids, doc_dir=a.docs)
    # 라이브가 실제로 준 다른 주어 문서 (회수 프롬프트에서 id 를 읽는다)
    best = max(X572.prompts(), key=lambda r: len(set(re.findall(r"ID: (doc_\S+)", str(r.get("text") or "")))))
    live_ids = sorted(set(re.findall(r"ID: (doc_\S+)", str(best.get("text")))))
    other_ids = [d for d in live_ids if d not in set(mine_ids)]
    other, miss2 = SRCH.read_docs(other_ids, doc_dir=a.docs)

    def block(d):
        return NL.join("ID: %s%s%s" % (k, NL, d[k]) for k in sorted(d))

    only = block(mine)
    mixed = block(dict(list(other.items()) + list(mine.items())))
    print("# x574 — 색인이 준 이 주어 문서 %d(읽음 %d) · 라이브가 준 다른 주어 문서 %d(읽음 %d)"
          % (len(mine_ids), len(mine), len(other_ids), len(other)))
    print("   A_only %d자 · B_mixed %d자" % (len(only), len(mixed)))
    print("   채점 기대(프롬프트 밖): 수 %s · 문서 %s" % (want_fig, want_doc))
    print("   A_only 안에 그 수: %s" % {f: (f in only) for f in want_fig})
    if miss1 or miss2:
        print("   ⚠못 읽은 문서: %s" % ((miss1 + miss2)[:4]))
    if a.wiring_only:
        print("--- 물음 ---%s" % ASK)
        return 0

    arms = {"A_only": only + ASK, "B_mixed": mixed + ASK, "N_none": ASK.strip()}
    print()
    print("%-8s %-5s %-64s %s" % ("팔", "temp", "답", "판정"))
    print("-" * 108)
    tally = collections.defaultdict(lambda: [0, 0])
    for nm in ("A_only", "B_mixed", "N_none"):
        for tp, cnt in ((0.0, 1), (a.temp, a.n)):
            for _ in range(cnt):
                try:
                    rep = " ".join(str(X559.gen(a.port, arms[nm], 140, tp)).split())
                except Exception as e:
                    print("%-8s %-5s 호출 실패: %r" % (nm, tp, e))
                    continue
                figs = [X567.digits(g) for g in re.findall(r"\$?(\d[\d,]{2,})", rep)]
                ok = any(f in want_fig for f in figs)
                cite = [d for d in want_doc if d in rep]
                tally[nm][1] += 1
                if ok:
                    tally[nm][0] += 1
                print("%-8s %-5s %-64s %s%s"
                      % (nm, tp, rep[:64], "맞는 수" if ok else ("다른 수 %s" % (figs[:2] or "없음")),
                         " +문서인용" if cite else ""))
    print()
    print("## 판정 (A3 가 그 주어에 대해 대는 수를 쓴 비율)")
    for nm in ("A_only", "B_mixed", "N_none"):
        print("   %-8s %d/%d" % (nm, tally[nm][0], tally[nm][1]))
    print()
    print("⚠N_none 이 이미 맞으면 그것은 모델의 사전지식이지 배달의 공이 아니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
