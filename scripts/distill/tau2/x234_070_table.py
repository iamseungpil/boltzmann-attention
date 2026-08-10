# -*- coding: utf-8 -*-
r"""x234 — task_070 표 재료를 **문서 축자로** 뽑는다 (설계서 §4 단계 1의 앞단 · 모델 호출 0).

## 왜

`TASK_070_071_DESIGN_2026_08_09.md` §4: **단계 1 = "표를 손으로 만들어 모델이 고를 수 있는가"**
이고, 그것이 가장 비싼 항목(M1 = A3 빌드) 앞의 관문이다. 표를 만들려면 먼저 재료가 실재하는지
봐야 한다. 여기서는 **뽑아서 인쇄만** 한다 — 판단도 선택도 없다.

## 070 이 요구하는 축 (태스크 정의 축자 · 설계서 §1)

  · ATM 리베이트 **월 $15 이상**   · 당좌대월 수수료 **0**
  · 최소잔액 **$10,000 미만**      · APY **1% 이상**

## 규율

- 값은 **문서 문장에서** 나오고, 각 값에 **문서 id + 그 문장**을 함께 남긴다([[23]]).
- 못 찾은 축은 **빈칸으로 남긴다** — 지어내지 않는다([[25]]). 티어별 값(예: APY 2.5/3.0/3.25%)은
  단일 스칼라로 접지 않는다(설계서 §3-M1 경고).
- 이건 **분석 스크립트**다. 엔진은 런타임에 문서를 이렇게 뜯지 않는다([[59]]).

실행: python x234_070_table.py [출력.json]
"""
import collections
import glob
import json
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DOCS = ("/home/woori/scratch/tau2-bench/data/tau2/domains/banking_knowledge/documents/"
        "doc_business_checking_accounts_*.json")

AXES = [
    ("atm_rebate_usd_month", re.compile(
        r"[^.\n]*ATM fees are rebated up to \$?([0-9,.]+)[^.\n]*", re.I)),
    ("overdraft_fee_usd", re.compile(
        r"[^.\n]*overdraft fee is \$?([0-9,.]+)[^.\n]*", re.I)),
    ("minimum_balance_usd", re.compile(
        r"[^.\n]*minimum (?:daily )?balance (?:requirement )?(?:is|of) \$?([0-9,.]+)[^.\n]*", re.I)),
    ("apy_pct", re.compile(
        r"[^.\n]*account earns an APY of ([0-9.]+)%[^.\n]*", re.I)),
]


def product_of(doc_id):
    m = re.match(r"doc_business_checking_accounts_(.+?)_\d+$", doc_id)
    return m.group(1) if m else None


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "x234_table.json"
    table = collections.defaultdict(dict)
    ndocs = 0
    for p in sorted(glob.glob(DOCS)):
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        ndocs += 1
        prod = product_of(d.get("id", ""))
        if not prod or prod == "(general)":
            continue
        text = str(d.get("content") or "")
        for axis, rx in AXES:
            for m in rx.finditer(text):
                val, sent = m.group(1), " ".join(m.group(0).split())
                cur = table[prod].get(axis)
                if cur and cur["value"] != val:
                    cur.setdefault("conflict", []).append((val, d["id"]))
                    continue
                if not cur:
                    table[prod][axis] = {"value": val, "doc": d["id"], "quote": sent[:160]}
    print("business checking 문서 %d개 · 제품 %d개" % (ndocs, len(table)))
    names = [a for a, _ in AXES]
    print("\n%-16s %10s %10s %12s %8s  결측" % ("제품", "ATM리베이트", "당좌대월", "최소잔액", "APY"))
    for prod in sorted(table):
        row = table[prod]
        miss = [a for a in names if a not in row]
        print("%-16s %10s %10s %12s %8s  %s"
              % (prod[:16],
                 row.get("atm_rebate_usd_month", {}).get("value", "-"),
                 row.get("overdraft_fee_usd", {}).get("value", "-"),
                 row.get("minimum_balance_usd", {}).get("value", "-"),
                 row.get("apy_pct", {}).get("value", "-"),
                 ",".join(a.split("_")[0] for a in miss) or "-"))
    conf = [(p, a, v["conflict"]) for p, r in table.items() for a, v in r.items() if "conflict" in v]
    print("\n충돌(같은 축에 다른 값) %d건" % len(conf))
    for p, a, c in conf[:8]:
        print("  %s · %s ← %s" % (p, a, c[:3]))
    print("\n인용 표본 (축마다 하나)")
    for axis, _ in AXES:
        for prod in sorted(table):
            if axis in table[prod]:
                v = table[prod][axis]
                print("  %-22s %s = %s  [%s]\n      \"%s\""
                      % (axis, prod, v["value"], v["doc"], v["quote"]))
                break
    json.dump({p: {a: v for a, v in r.items()} for p, r in table.items()},
              open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n저장: %s" % out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
