# -*- coding: utf-8 -*-
r"""x269 — 닫힌 후보 집합 제시가 MULTI 와 접미사를 닫는가 (유료 0 · 엔진 0).

## 왜 (x268 이 남긴 두 구멍)

x268: ⓐ 071 라이브 창 MULTI 의 범인 = **msg 22**(손님의 두-계좌 질문 — 서브가 우리 질문 대신
그 질문에 답한다) ⓑ 접미사는 인자까지 생존한다(`S_SUFF` 0/8 — 모델이 정규화 안 함).

닫힌 집합이 실재한다: A3 `doc_index` 안쪽 키(주어 슬러그). **출처 = env 파일명뿐**(x244·저작 0)
이라 [[23]] 무해이고, 기계 변환형(`sky_blue`→`Sky Blue`)이 정확히 채점 형태다.

⚠[[53]]: enum 제약 자체가 병목일 수 있다(선례 82%→자유서술 98%). 그래서 **측정이 배선보다
먼저다** — 문구는 A2 `decide_candidates_text`(출시할 그 문자열)를 쓴다([[03b]]).
⚠엔진은 **선별하지 않는다**: 군의 주어 전부를 제시한다(비-계좌 주어 포함 — 거르면 [[59]]).

## 팔 (n=8 · 두 축 · 라이브 창 = x267/x268 과 동일)

  A_LIVE       라이브 ask · 후보 없음            ← 0/8 기준선(x268 재현)
  B_ENUM       라이브 ask + 후보 집합 한 줄       ← 처방. MULTI·접미사 동시 표적
  C_ENUM_NO22  msg 22 뺀 ask + 후보 집합          ← 최선의 경우(둘 다 적용)
  D_ENUM_ONLY  후보 집합만·ask 없음 (부정 통제)    ← 이것이 살면 ask 는 기여 0

계기: EXACT(채점 형태 축자) · SUFFIX · MULTI · OTHER.

실행(리모트·GPU1): T2_PROBE_URL=http://localhost:8141/v1/chat/completions python3 x269_enum_candidates.py [N]
"""
import collections
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_search as S                                               # noqa: E402
from x216_read_and_offset import chat                               # noqa: E402
from x266_decide_ask_axis import a2, classify, GOLD, DOCS           # noqa: E402

RES = os.environ.get(
    "X269_RESULTS",
    "/home/woori/scratch/tau2-bench/data/simulations/bank_all6b_20260811/results.json")
NOW = "2025-11-14"


def display(slug):
    """slug → 표기. 기계 변환뿐(도메인 선별 0) — `sky_blue`→`Sky Blue`."""
    return " ".join(w.capitalize() for w in str(slug).split("_"))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    A = a2()
    po = A["policy_ontology"]
    tpl = po.get("decide_candidates_text")
    if not tpl:
        print("A2 에 `decide_candidates_text` 가 없다 — 문구 신설 금지([[03b]]). 중단.")
        return 2
    d = json.load(io.open(RES, encoding="utf-8"))
    sim = [x for x in d["simulations"] if x["task_id"] == "task_071"][0]
    us = [(i, str(m.get("content") or "")) for i, m in enumerate(sim["messages"])
          if m.get("role") == "user"]
    live = [u for u in us if u[0] < 47][-4:]
    ask_all = " --- ".join(c for _, c in live)[-6000:]
    ask_no22 = " --- ".join(c for i, c in live if i != 22)[-6000:]

    for g in ("business_checking_accounts", "business_savings_accounts"):
        gold = GOLD[g]
        mat, info = S.material_for(A, g, DOCS, NOW)
        cands = ", ".join(display(k) for k in sorted(po["doc_index"][g]))
        cline = tpl.format(candidates=cands)
        allnames = sorted(set(GOLD.values()))
        print("축 %s · gold=%r · 후보 %d개 · 문서 %d(뺀 것 %d)"
              % (g, gold, len(po["doc_index"][g]), info["kept"], len(info["dropped"])))
        arms = (("A_LIVE", ask_all, ""),
                ("B_ENUM", ask_all, cline),
                ("C_ENUM_NO22", ask_no22, cline),
                ("D_ENUM_ONLY", "", cline))
        for label, ask, cl in arms:
            body_ask = (ask[:3000] + ("\n\n" + cl if cl else "")) if ask else cl
            c = collections.Counter()
            for _i in range(n):
                try:
                    body = po["doc_decide_prompt"].format(ask=body_ask, material=mat)
                    r = chat(body, None, 0.0 if _i == 0 else 0.7, 40).get("content")
                except Exception as e:
                    r = "ERR %s" % type(e).__name__
                c[classify(r, gold, allnames)] += 1
            print("  %-12s EXACT %d/%d  %s" % (label, c["EXACT"], n, c.most_common(3)))
        print()
    print("※ B_ENUM 이 EXACT 로 서면 후보 한 줄이 MULTI 와 접미사를 동시에 닫는다."
          "\n  B 가 낮고 C 만 높으면 msg 22 는 별도 표적으로 남는다."
          "\n  D 가 B 만큼 높으면 ask 는 이 결정에 기여하지 않는다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
