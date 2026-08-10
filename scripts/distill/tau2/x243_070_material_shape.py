# -*- coding: utf-8 -*-
r"""x243 — **070 의 재료는 어디까지 결정론이어야 하는가** (격리 사다리 · 유료 0 · 로컬 LLM).

## 왜 (사용자 지시 2026-08-11 *"070 1번부터"* · ⛔0 ③ *결정론은 최소한*)

070 의 1차 결손은 **검색**이다(라이브: 프로모션 문서 4개 전부 미회수·질의에 고유명 0). 처방은
정해져 있다 — **shell + A3 로 문서 결정**(C405⒟). 남은 것은 **A3 링크의 모양**인데, 여기서
결정론의 양이 갈린다:

  ⒜ `(주어) → 문서 id`      … 파일명 규칙 하나로 끝난다(x203 과 같은 규칙·**축 형식화 불요**)
  ⒝ `(주어, 축) → 문서 id`  … 축 이름 어휘를 코퍼스에서 뽑고 **축 형식화 슬롯**도 지어야 한다

x235 의 `R7`(8/8)이 실었던 것은 **축별 문장 2개씩**이라 ⒝처럼 보인다. 그러나 그것은 프로브가
그렇게 만들었을 뿐이고, **제품 문서를 그냥 다 주면 되는지는 재본 적이 없다**. 되면 ⒜로 끝나고
우리가 짓는 결정론이 한 층 줄어든다. 그래서 **짓기 전에 잰다**.

## 팔 (n=8 · gold = 정확 일치)

  S0_NAMES      요구 + 후보 이름                       ← 부정 통제
  S1_AXIS_SENT  + 축별 문장 2개씩 + 활성 프로모션      ← `R7` 재현(대조 기준)
  S2_FULLDOCS   + **제품 문서 전문** + 활성 프로모션    ← ⒜가 되는가
  S3_TRUNC      + 제품 문서 **앞 400자** + 활성 프로모션 ← 예산을 줄여도 되는가
  S4_NOPROMO    제품 문서 전문만(프로모션 없음)         ← 프로모션 필요성 재확인(R1 형)

읽는 법 — S2/S3 가 S1 만큼 나오면 **축 링크·축 형식화를 짓지 않는다**. S1 만 높으면 축이 필요하다.
S4 가 낮아야 유효창 처방(x242)이 여전히 본체다.

⚠값은 전부 문서에서 온다. gold 는 채점에만 쓴다([[23]]).
⚠축 문장 추출은 **프로브 안에서만** 한다 — 엔진에 옮기면 [[59]] 위반이다.

실행(리모트·GPU1): T2_PROBE_URL=http://localhost:8141/v1/chat/completions \
                   python x243_070_material_shape.py [N]
"""
import collections
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from x216_read_and_offset import chat                             # noqa: E402
from x235_070_staircase import AX, DOM, GOLD, ASK, docs, axis_sentences, promos, \
    task_requirements                                             # noqa: E402


def product_docs():
    """제품별 문서 **전문** — 파일명이 제품을 말한다(빌드 시점 규칙·x203 과 같은 것)."""
    per = collections.defaultdict(list)
    for d in docs("doc_business_checking_accounts_*.json"):
        m = re.match(r"doc_business_checking_accounts_(.+?)_\d+$", d.get("id", ""))
        if m:
            per[m.group(1)].append((d["id"], " ".join(str(d.get("content") or "").split())))
    return per


def block(per, cap):
    out = []
    for p in sorted(per):
        lines = ["%s:" % p.replace("_", " ").title()]
        for i, c in per[p]:
            lines.append("  [%s] %s" % (i, c[:cap]))
        out.append("\n".join(lines))
    return "Policy documents on record:\n" + "\n".join(out)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    per_ax = axis_sentences()
    per_doc = product_docs()
    act, exp = promos()
    req = task_requirements()
    prods = sorted(per_doc)
    names = "Candidate business checking accounts: " + ", ".join(
        p.replace("_", " ").title() for p in prods)
    doc_block = []
    for p in sorted(per_ax):
        lines = []
        for ax in AX:
            for s, i in per_ax[p][ax][:2]:
                lines.append("  - %s  [%s]" % (s, i))
        doc_block.append("%s:\n%s" % (p.replace("_", " ").title(), "\n".join(lines)))
    doc_block = "Policy document sentences on record:\n" + "\n".join(doc_block)
    pr_act = "\n".join("[%s] %s" % (i, c) for i, c in act)
    print("제품 %d · 문서 %d · 활성 프로모션 %d · 만료 %d"
          % (len(prods), sum(len(v) for v in per_doc.values()), len(act), len(exp)))

    arms = [("S0_NAMES", "\n\n".join([req, names])),
            ("S1_AXIS_SENT", "\n\n".join([req, names, doc_block, pr_act])),
            ("S2_FULLDOCS", "\n\n".join([req, names, block(per_doc, 100000), pr_act])),
            ("S3_TRUNC", "\n\n".join([req, names, block(per_doc, 400), pr_act])),
            ("S4_NOPROMO", "\n\n".join([req, names, block(per_doc, 100000)]))]
    for name, body in arms:
        print("   %-13s %7d자" % (name, len(body)))
    for name, body in arms:
        c = collections.Counter()
        for i in range(n):
            try:
                t = chat(body + "\n\n" + ASK, None, 0.0 if i == 0 else 0.7, 24).get("content", "")
            except Exception as e:
                t = "ERR %s" % type(e).__name__
            c[" ".join(str(t).split())[:40]] += 1
        hit = sum(v for k, v in c.items()
                  if re.fullmatch(r"\**%s( Account)?\**\.?" % re.escape(GOLD), str(k).strip(), re.I))
        print("  %-13s gold %d/%d   %s" % (name, hit, n, c.most_common(3)))
    print("\n※ S2/S3 ≈ S1 이면 A3 링크는 **(주어 → 문서 id)** 로 끝나고 축 슬롯을 짓지 않는다."
          "\n  S1 만 높으면 축 링크가 필요하다. S4 는 낮아야 유효창이 본체로 남는다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
