# -*- coding: utf-8 -*-
r"""x449 — dense 검색이 **선언된 그 문서를 돌려주는가** (2026-08-21·격리·무료·LLM 0)

## 왜 (사용자 물음)
*"dense 와 A2 선언은 같은 성과인가?"* — `x448`(n=4)에서 세 지표가 **동일**하게 나왔다
(참조일치 4/4 · 인용실재 2/4 · 손님인용 0/4). 점수가 같다고 **같은 것을 하는 것은 아니다**.
여기서는 LLM 을 한 번도 부르지 않고 **재료 자체**를 비교한다.

## 무엇을 재나 (전부 닫힌 술어·[[59]])
    hit_rate      dense/bm25 가 돌려준 문서 id 중 **선언된 12편**에 든 비율
    cover         **선언된 각 문서**가 몇 %의 사례에서 회수되는가 (빠지는 문서가 있나)
    churn         사례마다 돌려주는 집합이 얼마나 흔들리는가 — 선언은 **정의상 0**
    defining      그 태스크의 범주를 **정의하는 문서**(A2 색인의 첫 항목)가 들어왔나
⇒ 겹치면 *\"같은 재료를 덜 안정적인 경로로\"* 이고, 안 겹치면 x448 의 4/4 는 **그 재료 덕이 아니다**.

사용: (리모트·cwd=tau2 · PYTHONPATH=src:...) py x449_dense_vs_declared_overlap.py [--k 12]
"""
import argparse
import collections
import io
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

import x447_indexed_category_iso as IX   # noqa: E402  A2 선언 읽기(사본 금지·[[67]])
import x448_index_vs_all_iso as V        # noqa: E402  샌드박스·사례(사본 금지·[[67]])

REP = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026")
_ID = re.compile(r"^\s*ID:\s*(\S+)\s*$", re.M)


def ids_in(text):
    """검색 출력에서 문서 id 만 뽑는다 — 형식이 주는 `ID:` 줄뿐이다(뜻 해석 0)."""
    return [m.group(1) for m in _ID.finditer(str(text or ""))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=0, help="0 = 선언 편수와 같게(예산 일치)")
    ap.add_argument("--tag", default="ov1")
    a = ap.parse_args()

    sb = V.Sandbox()
    declared = [d[0] for d in IX.index_docs()]
    K = a.k or len(declared)
    dset = set(declared)
    cases = V.wide_cases()
    print("=" * 100)
    print("x449 · 사례 %d · 선언 %d편 · k=%d · LLM 호출 0" % (len(cases), len(declared), K))
    print("=" * 100)

    rows = []
    cov = {t: collections.Counter() for t in ("bm25", "dense")}
    sets = {t: [] for t in ("bm25", "dense")}
    for c in cases:
        r = {"task": c["task"], "trial": c["trial"]}
        for tool, key in (("KB_search_bm25", "bm25"), ("KB_search_dense", "dense")):
            got = ids_in(sb.search(tool, c["said"], K))
            inter = [g for g in got if g in dset]
            r[key + "_n"] = len(got)
            r[key + "_hit"] = len(inter)
            r[key + "_ids"] = got
            for g in inter:
                cov[key][g] += 1
            sets[key].append(frozenset(got))
        rows.append(r)
        print("  %-9s t%-3s bm25 %2d/%-2d 선언  dense %2d/%-2d 선언"
              % (r["task"], r["trial"], r["bm25_hit"], r["bm25_n"], r["dense_hit"], r["dense_n"]))

    p = os.path.abspath(os.path.join(REP, "x449_%s.json" % a.tag))
    with io.open(p, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)

    n = len(rows)
    print("\n" + "=" * 100)
    for key in ("bm25", "dense"):
        hit = sum(r[key + "_hit"] for r in rows)
        tot = sum(r[key + "_n"] for r in rows)
        uniq = len(set(sets[key]))
        allids = set()
        for s in sets[key]:
            allids |= s
        print("%-6s 선언과 겹침 %d/%d (%.0f%%) · 서로 다른 결과집합 %d/%d · 등장한 문서 %d종"
              % (key, hit, tot, 100.0 * hit / max(1, tot), uniq, n, len(allids)))
        miss = [d for d in declared if cov[key][d] == 0]
        print("       선언 문서 회수율: " + ", ".join(
            "%s %d%%" % (d.split("_")[-2][:12] + "_" + d.split("_")[-1], 100 * cov[key][d] // max(1, n))
            for d in declared))
        if miss:
            print("       ⚠한 번도 안 돌아온 선언 문서 %d편: %s" % (len(miss), ", ".join(miss)))
    print("선언(우리 방식): 같은 %d편 · 서로 다른 결과집합 1/%d · 회수율 전부 100%%" % (len(declared), n))
    print("→ %s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
