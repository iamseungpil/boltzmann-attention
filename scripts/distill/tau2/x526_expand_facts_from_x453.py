# -*- coding: utf-8 -*-
r"""x526 — `x453` 감사 산출을 **사실표로 변환**한다 (2026-08-24·무료·결정론·LLM 0)

## 왜
`x451`(계좌 클래스 선택 격리)의 `F_facts 1/4` 는 능력 판정이 아니다 — 실측:

    표 평균 채움 6.2/16
    055 checking  gold purple_account  10/16  → F_facts 가 **유일하게 맞힌 칸**
    055/063 sav   gold silver_plus      5/16  ↔ 오답 diamond_elite 5 · gold 6 · platinum 5
    070 business  gold sky_blue         9/16  ↔ 오답 navy_blue 13 · true_blue 10  ← gold 가 더 얇다

그리고 `x453`(표적 계열 감사)가 이유를 냈다: **현행 16축 중 12축이 savings·business 계열에서
한 번도 관측되지 않는다**(`declared_never_seen`). 즉 checking 용 표를 다른 계열에 갖다 댄 것이다.

## 무엇을 하나
`x453` 의 `cites`(문서 축자 인용 · 클래스 · 값 · 오프셋 검산 통과분)를 **x430 사실표 형식**으로
옮긴다. LLM 호출 0 · 판단 0 · 새 추출 0 — 이미 검산된 인용을 형식만 바꿔 싣는다.

## 채택 규칙 = x453 이 **결과 보기 전에** 못박아 둔 것을 그대로 쓴다
    · `adopt` (= 서로 다른 클래스 5개 이상에서 관측된 속성) 만 싣는다 — 빈도뿐
    · 값마다 문서 id + 축자 인용을 함께 싣는다 (없으면 안 싣는다)
    · 같은 (클래스, 속성)에 서로 다른 값이 오면 `conflict: true` 로 **둘 다 남긴다**(우리가 안 고른다)
    ⛔gold·태스크·실패 사례를 보고 축을 고르지 않는다([[23]]). 어느 축이 어느 태스크를 살리는지는
      **재측정 후에** 본다.

## 병합
기존 `x430_account_facts_llm_filled.json` 을 **바탕으로 두고 그 위에 얹는다**(합집합) —
checking 계열이 이미 갖고 있는 값을 잃지 않기 위해서다.

사용: py -3 x526_expand_facts_from_x453.py [--audit ...] [--base ...] [--out ...]
"""
import argparse
import io
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
REP = os.path.normpath(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", default=os.path.join(REP, "x453_attr_coverage_targeted_2026_08_24.json"))
    ap.add_argument("--base", default=os.path.join(REP, "x430_account_facts_llm_filled.json"))
    ap.add_argument("--out", default=os.path.join(REP, "x430_account_facts_expanded_2026_08_24.json"))
    a = ap.parse_args()

    with io.open(a.audit, encoding="utf-8") as f:
        aud = json.load(f)
    adopt = set(aud.get("adopt") or [])
    cites = aud.get("cites") or {}
    if not adopt or not cites:
        raise SystemExit("x453 산출에 adopt/cites 가 없다 — 중단")

    table = {}
    if os.path.exists(a.base):
        with io.open(a.base, encoding="utf-8") as f:
            table = json.load(f)
    base_cells = sum(1 for c, row in table.items() if isinstance(row, dict)
                     for k, v in row.items() if isinstance(v, dict) and v.get("values"))

    added, skipped_attr, conflicts = 0, 0, 0
    for attr, rows in cites.items():
        if attr not in adopt:
            skipped_attr += 1
            continue
        for r in rows:
            cls = r.get("class")
            val = r.get("value")
            quote = r.get("quote")
            doc = r.get("doc")
            if not cls or val in (None, "") or not quote or not doc:
                continue
            cell = table.setdefault(cls, {}).setdefault(
                attr, {"values": [], "conflict": False, "unit": None, "cap": None, "evidence": []})
            if not isinstance(cell, dict):
                continue
            cell.setdefault("values", [])
            cell.setdefault("evidence", [])
            if val not in cell["values"]:
                cell["values"].append(val)
            cell["evidence"].append({"value": val, "doc": doc, "quote": quote})
            added += 1
        # 충돌 표시 — 우리가 고르지 않는다
        for cls, row in table.items():
            v = row.get(attr) if isinstance(row, dict) else None
            if isinstance(v, dict) and len(v.get("values") or []) > 1 and not v.get("conflict"):
                v["conflict"] = True
                conflicts += 1

    with io.open(a.out, "w", encoding="utf-8") as f:
        json.dump(table, f, ensure_ascii=False, indent=1)

    def n_filled(row):
        return sum(1 for k, v in row.items()
                   if isinstance(v, dict) and v.get("values") and not k.startswith("_"))

    rows = [(c, n_filled(r)) for c, r in table.items() if isinstance(r, dict)]
    print("[x526] 채택 속성 %d종 · 실은 인용 %d건 · 충돌 표시 %d · 비채택 속성 %d종 건너뜀"
          % (len(adopt), added, conflicts, skipped_attr))
    print("[x526] 채움 칸: 이전 %d → 이후 %d" % (base_cells, sum(n for _, n in rows)))
    print("[x526] 클래스 %d · 평균 %.1f칸" % (len(rows), sum(n for _, n in rows) / max(1, len(rows))))
    for t in ("purple_account", "silver_plus_account", "green_account_(savings)",
              "diamond_elite_account", "gold_account", "platinum_account",
              "sky_blue", "navy_blue", "true_blue", "hunter_green"):
        if t in table:
            print("    %-28s %3d칸" % (t, n_filled(table[t])))
    print("[x526] wrote %s" % a.out)


if __name__ == "__main__":
    main()
