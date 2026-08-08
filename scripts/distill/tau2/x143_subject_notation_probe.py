# -*- coding: utf-8 -*-
"""x143 — **주어 표기 실태**: 온톨로지 주어 ↔ env가 실제로 쓰는 문자열 (유료 0 · LLM 0).

왜: 결정점은 상품 이름으로 조회한다. 온톨로지는 **문서가 쓴 이름**(`Navy Blue Business Checking`)
으로 주어를 적고, env는 **자기 이름**(`Navy Blue`)으로 계좌를 적는다. 둘이 어긋나는 **비율과 형태**를
모르면 *"조회가 된다"* 도 *"안 된다"* 도 근거가 없다(C316이 `Light Green ↔ Light Blue`로 지목한 축).

기준은 **env에서 기계로 뽑는다** — 태스크 지문에 적힌 *"정확한 값 목록"* 은 쓰지 않는다.
그건 user-sim에게 준 지시이지 env의 제약이 아니고(`submit_referral`엔 enum 검증이 **없다**),
gold 경유 저작 금지([[23]])에도 걸린다.

⚠**정규화·유사도 매핑은 하지 않는다**(설계서 §9-4·C8). 이 프로브는 **재기만** 한다.
⚠필드명을 찍지 않는다 — 레코드 **키를 전수 열거**해서 문자열을 모은다.
  (1차판이 `class`만 보고 *"env엔 상품명이 없다"* 는 오진을 냈다. 실제 이름은 `level`에 있다.)
⚠**분석 도구이지 엔진이 아니다**([[59]]는 엔진을 규율한다).

usage: x143_subject_notation_probe.py --db <db.json> --ontology <ontology.json[.gz]>
"""

import argparse
import collections
import gzip
import io
import json
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# 상품명처럼 보이는 문자열 — 대문자로 시작하는 짧은 어구. 상태값(ACTIVE 등)도 걸리므로 함께 인쇄해
# 사람이 가른다(자동으로 거르면 그 필터가 또 하나의 숨은 판단이 된다).
NAMEISH = re.compile(r"^[A-Z][A-Za-z]+(?: [A-Za-z][A-Za-z-]*){0,4}$")


def load(path):
    op = gzip.open if str(path).endswith(".gz") else io.open
    with op(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def table(db, name):
    v = db.get(name) or {}
    d = v.get("data") if isinstance(v, dict) else None
    return list((d or {}).values()) if isinstance(d, dict) else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--ontology", required=True)
    a = ap.parse_args()

    db = load(a.db)
    found = collections.Counter()
    for name in db:
        rs = table(db, name)
        if not rs:
            continue
        keys = collections.Counter()
        for r in rs:
            if isinstance(r, dict):
                keys.update(r.keys())
        print("%-26s 레코드 %3d · 키: %s" % (name, len(rs), ", ".join(k for k, _ in keys.most_common())))
        for r in rs:
            if not isinstance(r, dict):
                continue
            for k, v in r.items():
                if isinstance(v, str) and len(v) > 4 and NAMEISH.match(v):
                    found[(name, k, v)] += 1

    print("\n=== env 안의 상품명형 문자열 (상태값도 섞인다 — 사람이 가른다) ===")
    for (tbl, k, v), n in sorted(found.items()):
        print("   %-24s %-24s %-38s %d" % (tbl, k, v, n))

    subj = sorted({r["subject"] for r in (load(a.ontology).get("rows") or [])})
    names = {v for (_, _, v) in found}
    hit = [s for s in subj if s in names]
    print("\n=== 대조 (정확 일치만) ===")
    print("온톨로지 주어 %d · env 문자열 %d · **정확 일치 %d**" % (len(subj), len(names), len(hit)))
    for s in subj:
        print("   %s %s" % ("✓" if s in names else "✗", s))
    print("\nenv에만 있는 것:")
    for n in sorted(names - set(subj)):
        print("   · %s" % n)
    print("\n⚠불일치는 **두 부류가 섞여 있다** — ⓐ표기 차이 ⓑ그 상품이 이 DB에 아예 없음(모집단 차이).")
    print("  섞어 세면 오진한다. per-case로 가른다([[08]]).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
