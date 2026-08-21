# -*- coding: utf-8 -*-
r"""x453 샤드 합치기 (2026-08-21·오프라인·LLM 0)

`x453_attr_coverage_audit.py --shard i --of N` 은 **정렬된 클래스 키를 나머지 연산으로** 갈라
서로 겹치지 않는 부분만 본다. 그래서 합집합이 곧 전수이고, 합치기는 사전 union 하나다.

엔진은 **더하기만** 한다 — 재계산은 채택선뿐이고 그 규칙은 감사와 같다(정책이 요건이라 말함
∪ 값이 `minclasses` 이상 클래스에서 명시됨·결과 보기 전에 고정·[[23]]).

검산: 샤드들의 `classes_in_shard` 가 **서로소**여야 하고 파일 수가 선언된 `of` 와 같아야 한다.
아니면 중단한다 — 조용히 일부만 합치면 그게 색인의 구멍이 된다.

사용: py x453_merge_shards.py --out x453_attr_coverage_all.json x453_s0.json x453_s1.json
"""
import argparse
import collections
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

import x430_account_facts as FT         # noqa: E402  현행 선언(비교용)

REP = os.path.abspath(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026"))
YIELD_KEY = "per_doc_yield"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shards", nargs="+")
    ap.add_argument("--out", default="x453_attr_coverage_all.json")
    ap.add_argument("--minclasses", type=int, default=5)
    a = ap.parse_args()

    parts = []
    for f in a.shards:
        with io.open(os.path.join(REP, f), encoding="utf-8") as fh:
            parts.append((f, json.load(fh)))
    print("샤드 %d개" % len(parts))

    # ── 검산: 서로소 + 개수 일치 ────────────────────────────────────────────
    seen, dup = set(), []
    for f, d in parts:
        cs = set(d.get("classes_in_shard") or [])
        dup += sorted(seen & cs)
        seen |= cs
        print("  %-36s 클래스 %3d · 문서 %3d · 인용 %s"
              % (f, len(cs), d.get("n_docs"), d.get("n_cites")))
    if dup:
        raise SystemExit("샤드가 겹친다: %r" % dup[:10])
    ofs = {int(d.get("of") or 1) for _f, d in parts}
    if len(ofs) != 1 or len(parts) != list(ofs)[0]:
        raise SystemExit("샤드 수 불일치: 파일 %d ↔ 선언 of=%r" % (len(parts), ofs))

    obs = collections.defaultdict(set)
    req = collections.defaultdict(set)
    adocs = collections.defaultdict(set)
    cites = collections.defaultdict(list)
    cdocs = collections.defaultdict(set)
    example, req_example, yields = {}, {}, {}
    rejected = ndocs = retried = 0
    rej_detail, empty_after = [], []
    for _f, d in parts:
        for n, v in (d.get("observed") or {}).items():
            obs[n] |= set(v)
        for n, v in (d.get("requirements") or {}).items():
            req[n] |= set(v)
        for n, v in (d.get("attr_docs") or {}).items():
            adocs[n] |= set(v)
        for n, v in (d.get("cites") or {}).items():
            cites[n].extend(v)
        for c, v in (d.get("class_docs") or {}).items():
            cdocs[c] |= set(v)
        for n, v in (d.get("example") or {}).items():
            example.setdefault(n, v)
        for n, v in (d.get("req_example") or {}).items():
            req_example.setdefault(n, v)
        yields.update(d.get(YIELD_KEY) or {})
        rej_detail += (d.get("rejected_detail") or [])
        empty_after += (d.get("empty_after_retry") or [])
        rejected += int(d.get("rejected") or 0)
        ndocs += int(d.get("n_docs") or 0)
        retried += int(d.get("n_retried") or 0)

    adopt = sorted(set(req) | {n for n, s in obs.items() if len(s) >= a.minclasses})
    have = {n for n, _al in FT.ATTRS}
    payload = {"merged_from": a.shards, "minclasses": a.minclasses,
               "n_docs": ndocs, "rejected": rejected, "n_retried": retried,
               "observed": {n: sorted(v) for n, v in obs.items()},
               "requirements": {n: sorted(v) for n, v in req.items()},
               "attr_docs": {n: sorted(v) for n, v in adocs.items()},
               "cites": dict(cites),
               "n_cites": sum(len(v) for v in cites.values()),
               "n_cites_unlocated": sum(1 for v in cites.values()
                                        for c in v if c.get("line") is None),
               "n_cites_span_bad": sum(1 for v in cites.values()
                                       for c in v if c.get("span_ok") is False),
               "example": example, "req_example": req_example,
               "class_docs": {c: sorted(v) for c, v in cdocs.items()},
               YIELD_KEY: yields,
               "docs_with_no_attrs": sorted(k for k, v in yields.items() if not v),
               "rejected_detail": rej_detail, "empty_after_retry": empty_after,
               "adopt": adopt, "new_vs_declared": [n for n in adopt if n not in have],
               "declared_never_seen": sorted(n for n in have if n not in obs)}
    p = os.path.join(REP, a.out)
    with io.open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)

    print("\n" + "=" * 96)
    print("문서 %d편 · 클래스 %d · 관측 축 %d · 요건 축 %d · **채택 %d**(현행에 없는 것 %d)"
          % (ndocs, len(cdocs), len(obs), len(req), len(adopt),
             len(payload["new_vs_declared"])))
    print("인용 %d건 · 위치 못 잡음 %d · 절 범위 검산 실패 %d · 검산 탈락 %d · 재질의 %d"
          % (payload["n_cites"], payload["n_cites_unlocated"], payload["n_cites_span_bad"],
             rejected, retried))
    print("아무 축도 못 건진 문서 %d / %d편 — 여기가 색인의 구멍 후보다"
          % (len(payload["docs_with_no_attrs"]), len(yields)))
    print("→ %s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
