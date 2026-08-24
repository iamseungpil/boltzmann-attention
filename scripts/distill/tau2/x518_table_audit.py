# -*- coding: utf-8 -*-
r"""x518 — 사실표 축-배정 감사 (x509 S4 선행 · 사용자 지시 2026-08-24 *"B 부터 가라"*).

## 왜

`x502` 가 `_filled` 의 **빈 9 칸**만 손으로 감사해 **ACCEPT 2 · REJECT 7** 을 냈다. 병의 이름은
*"값을 못 찾은 것이 아니라 **다른 축의 값을 이 축에 넣었다**"* 이고, x502 자신이 이렇게 적어 뒀다:

    "조건부 추출기의 축 배정이 이 표본에서 2/9 다. **다른 346 칸도 같은 병일 수 있으나
     그것은 안 쟀다** — 여기서 감사한 것은 `_filled` 가 빈 9 칸뿐이다."

그 346 칸이 이 프로브의 대상이다. 왜 지금인가 — `x513`(S2c)의 `C_filtered` 가 **순증 0** 으로
읽혔는데, 그 필터가 쥔 술어의 사실이 이 표에서 나온다. [[63]] 이 여섯 축에서 실측한
*"제거만 닫는다"*(0/8 → 8/8)가 ②범주에서만 재현되지 않은 유일한 후보 설명이 **술어가 틀린 것**이다.
표를 감사하기 전에는 x513 의 음성을 빼기에 대한 증거로 읽을 수 없다.

## 무엇을 보나 — **값의 참·거짓이 아니라 배정** (x502 와 같은 물음)

닫힌 술어만 쓴다([[59]]). 판정하지 않고 **깃발만 든다** — 사람이 읽을 칸을 좁히는 것이 목적이다.

    F_unit    속성 이름이 요구하는 단위 ↔ 값의 단위가 다르다   (횟수 자리에 달러 등)
    F_period  속성의 기간어 ↔ 축자의 기간어가 다르다            (monthly 자리에 daily)
    F_delta   값·축자가 **가산분**이다                          (`+0.3%` · additional · bonus)
    F_scope   축자가 **다른 범주**를 말한다                      (foreign/international ↔ 국내 OON)
    F_bound   축자가 **한계어**를 쓴다                          (up to · minimum · at least)
    F_name    축자에 그 상품 이름 토큰이 하나도 없다             (정밀도 낮음 · 따로 보고)

## 계기의 검정 — x502 의 9 칸이 **라벨 집합**이다

기계 깃발이 REJECT 7 을 얼마나 회수하고 ACCEPT 2 를 얼마나 오탐하는지 **먼저** 인쇄한다.
회수가 낮으면 그 사실을 적고 결과를 그만큼 약하게 쓴다 — 계기를 믿기 전에 잰다([[25]]·[[67]]).

실행: PYTHONIOENCODING=utf-8 python x518_table_audit.py
"""
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

OUT = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026")
TABLE = os.path.join(OUT, "x430_account_facts_conditional.json")
AUDIT9 = os.path.join(OUT, "x502_conditional_cell_audit_2026_08_24.json")

PERIODS = ("daily", "monthly", "weekly", "annual", "yearly", "per month", "per year",
           "per day", "per statement cycle", "calendar month")
CURRENCY = re.compile(r"\$\s?\d")
PERCENT = re.compile(r"\d\s?%")
BARE = re.compile(r"(?<![\$\d.])\b\d[\d,]*(?:\.\d+)?\b(?!\s?%)")
DELTA = re.compile(r"(^\s*[+\-]\s*\d)|(\badditional\b|\bbonus\b|\bextra\b|\brebate\b|\bboost\b)", re.I)
BOUND = re.compile(r"\bup to\b|\bminimum fee\b|\bat least\b|\bmaximum\b|\bno more than\b", re.I)
FOREIGN = re.compile(r"\bforeign\b|\binternational\b", re.I)
DOMESTIC = re.compile(r"\bdomestic\b|\bout-of-network\b|\bout of network\b", re.I)


def expected_kind(attr):
    a = attr.lower()
    if "apy" in a:
        return "PERCENT"
    if a.startswith("free_") or a.endswith("_per_month") and a.startswith("free"):
        return "COUNT"
    if "age" in a:
        return "COUNT"
    if ("fee" in a or "limit" in a or "balance" in a or "reimbursement" in a
            or "holding" in a):
        return "CURRENCY"
    return None


def period_of(text):
    t = (text or "").lower()
    return {p for p in PERIODS if p in t}


def kind_of(text):
    if CURRENCY.search(text or ""):
        return "CURRENCY"
    if PERCENT.search(text or ""):
        return "PERCENT"
    if BARE.search(text or ""):
        return "COUNT"
    return None


def name_tokens(cls):
    return [t for t in re.split(r"[^a-z0-9]+", cls.lower()) if len(t) >= 4]


# ★기간어 동의 (2026-08-24 자기검산): `per month` ≡ `monthly` 다. 이 동치를 안 넣으면
#   `automatic_sweep` 같은 칸이 **옳은데도** F_period 로 잡힌다(초판이 그랬다).
PERIOD_EQ = {"monthly": {"monthly", "per month", "calendar month", "per statement cycle"},
             "daily": {"daily", "per day"},
             "annual": {"annual", "yearly", "per year"}}
# 스스로 **한계**인 속성 — 축자의 `at least`/`up to` 는 여기서 정상이다.
BOUND_ATTR = re.compile(r"(^|_)(min|max|waiver|limit|free|cap|reimbursement)(_|$)")


def unit_modal(tbl):
    """속성별 **선언 단위의 최빈값** — 표 자신에서 유도한다(손 선언 0)."""
    seen = collections.defaultdict(collections.Counter)
    for cls, attrs in tbl.items():
        for attr, cell in attrs.items():
            if isinstance(cell, dict) and cell.get("values") and cell.get("unit"):
                seen[attr][cell["unit"]] += 1
    return {a: c.most_common(1)[0][0] for a, c in seen.items() if c}


def flags_for(cls, attr, cell, modal=None):
    """깃발만 든다 — 판정하지 않는다. 반환 = set(플래그).

    ★F_unit 정정: 값 문자열에서 단위를 **추정하지 않는다**. 표에 `unit` 이 선언돼 있고
      값은 맨숫자로 저장되므로(`'20.00'` + `unit='USD'`), 추정하면 거의 전량이 오탐이다
      (초판 355 중 157). 선언 단위를 그 속성의 **최빈 단위**와 비교한다.
    """
    fl = set()
    vals = [str(v) for v in (cell.get("values") or [])]
    evs = cell.get("evidence") or []
    quotes = " ".join(str(e.get("quote") or "") for e in evs)
    if not vals or not quotes:
        return fl
    u = cell.get("unit") or ""
    m = (modal or {}).get(attr)
    if m and u and u != m:
        fl.add("F_unit")
    elif m and not u:
        fl.add("F_unit_undeclared")
    ap = period_of(attr.replace("_", " "))
    qp = period_of(quotes)
    if ap and qp:
        eq = set()
        for k, grp in PERIOD_EQ.items():
            if ap & grp:
                eq |= grp
        if eq and not (qp & eq):
            fl.add("F_period")
    elif ap and not qp:
        fl.add("F_period_silent")
    if any(re.match(r"^\s*[+\-]\s*\d", v) for v in vals):
        fl.add("F_delta")
    if BOUND.search(quotes) and not BOUND_ATTR.search(attr.lower()):
        fl.add("F_bound")
    a = attr.lower()
    if ("oon" in a or "out_of_network" in a) and FOREIGN.search(quotes) \
            and not DOMESTIC.search(quotes):
        fl.add("F_scope")
    toks = name_tokens(cls)
    if toks and not any(t in quotes.lower() for t in toks):
        fl.add("F_name")
    return fl


WEAK = {"F_name", "F_unit_undeclared", "F_period_silent"}


def main():
    tbl = json.load(open(TABLE, encoding="utf-8"))
    lab = json.load(open(AUDIT9, encoding="utf-8"))
    modal = unit_modal(tbl)

    # ── 0. 계기 검정 — x502 의 9 칸을 라벨로 쓴다
    print("=" * 100)
    print("(0) 계기 검정 — x502 의 손-감사 9 칸(ACCEPT 2 · REJECT 7)을 기계 깃발이 얼마나 잡나")
    print("=" * 100)
    hit = collections.Counter()
    for c in (lab.get("cells") or []):
        cls, attr, want = c.get("class"), c.get("attr"), c.get("verdict")
        cell = (tbl.get(cls) or {}).get(attr) or {}
        fl = flags_for(cls, attr, cell, modal) if cell else set()
        strong = fl - WEAK
        got = "FLAG" if strong else "clean"
        ok = (want == "REJECT" and strong) or (want == "ACCEPT" and not strong)
        hit[("recall" if want == "REJECT" else "specificity", bool(ok))] += 1
        print("  %-9s %-26s %-28s 기계=%-5s %s  %s"
              % (want, cls[:26], attr[:28], got, "OK" if ok else "MISS",
                 ",".join(sorted(fl)) or "-"))
    rec = hit[("recall", True)]
    spec = hit[("specificity", True)]
    print("")
    print("  회수(REJECT 를 깃발로 잡음) %d/7 · 오탐없음(ACCEPT 를 clean 으로 둠) %d/2"
          % (rec, spec))
    print("  ※ 약한 깃발(F_name·F_unit_undeclared·F_period_silent)은 강한 깃발에서 제외한다.")

    # ── 1. 전수 스윕
    print("")
    print("=" * 100)
    print("(1) 전수 스윕 — 값이 있는 칸 전량")
    print("=" * 100)
    per_flag = collections.Counter()
    per_attr = collections.defaultdict(collections.Counter)
    flagged = []
    n_filled = 0
    audited9 = {(c.get("class"), c.get("attr")) for c in (lab.get("cells") or [])}
    for cls, attrs in tbl.items():
        for attr, cell in attrs.items():
            if not isinstance(cell, dict) or not cell.get("values"):
                continue
            n_filled += 1
            fl = flags_for(cls, attr, cell, modal)
            for f in fl:
                per_flag[f] += 1
            strong = fl - WEAK
            per_attr[attr]["cells"] += 1
            if strong:
                per_attr[attr]["flag"] += 1
                flagged.append({"class": cls, "attr": attr, "flags": sorted(fl),
                                "values": cell.get("values"),
                                "quote": ((cell.get("evidence") or [{}])[0].get("quote") or "")[:200],
                                "already_audited": (cls, attr) in audited9})
    strong_n = sum(1 for r in flagged if set(r["flags"]) - WEAK)
    print("  값 있는 칸 %d · **강한 깃발 %d (%.0f%%)** · 이미 감사된 9 칸 중 %d"
          % (n_filled, strong_n, 100.0 * strong_n / max(n_filled, 1),
             sum(1 for r in flagged if r["already_audited"])))
    print("")
    print("  깃발별: " + " · ".join("%s %d" % kv for kv in per_flag.most_common()))
    print("")
    print("  %-32s %6s %6s %6s" % ("속성", "칸", "깃발", "비율"))
    print("  " + "-" * 56)
    for attr, c in sorted(per_attr.items(), key=lambda kv: -kv[1]["flag"]):
        if not c["flag"]:
            continue
        print("  %-32s %6d %6d %5.0f%%"
              % (attr, c["cells"], c["flag"], 100.0 * c["flag"] / max(c["cells"], 1)))

    print("")
    print("=" * 100)
    print("(2) 강한 깃발 표본 — 사람이 읽을 칸")
    print("=" * 100)
    for r in [x for x in flagged if set(x["flags"]) - WEAK][:14]:
        print("  %-24s %-30s %-24s %s"
              % (r["class"][:24], r["attr"][:30], ",".join(r["flags"])[:24],
                 str(r["values"])[:26]))
        print("      %s" % r["quote"][:150])

    # ── 3. ★희소성 — 필터는 **없는 칸 위에서** 순위를 못 매긴다
    #    이것이 x513 의 `C_filtered` 순증 0 을 설명하는 두 번째 후보다(첫째는 오배정).
    print("")
    print("=" * 100)
    print("(3) 희소성 — x513 이 실제로 비교한 클래스들의 채워진 칸 수 (16 중)")
    print("=" * 100)
    CASES = {"055": ["purple_account", "green_fee-free_account"],
             "057": ["blue_account", "evergreen_account", "light_blue_account"],
             "063": ["silver_plus_account", "diamond_elite_account", "silver_account",
                     "gold_account", "green_account_(savings)"]}
    sparse = {}
    for t, cls_list in CASES.items():
        row = {}
        for c in cls_list:
            n = sum(1 for at, cell in (tbl.get(c) or {}).items()
                    if isinstance(cell, dict) and cell.get("values"))
            row[c] = n
        sparse[t] = row
        avg = sum(row.values()) / max(len(row), 1)
        print("  %-5s 평균 %.1f/16   %s" % (t, avg,
              " · ".join("%s %d" % (k.split("_")[0][:12], v) for k, v in row.items())))
    print("")
    print("  판독: 055(x513 에서 **6/6 로 닫힌** 사례)는 평균이 가장 높고, 063(0/6)은 가장 낮다.")
    print("        필터가 정보를 못 나른 이유의 후보는 **오배정이 아니라 빈칸**일 수 있다.")

    out = {"probe": "x518_table_audit", "date": "2026-08-24",
           "table": os.path.basename(TABLE),
           "calibration": {"recall_reject": "%d/7" % rec, "specificity_accept": "%d/2" % spec,
                           "note": "x502 의 손-감사 9 칸이 라벨. `F_name` 은 강한 깃발에서 제외."},
           "n_filled": n_filled, "n_strong_flag": strong_n,
           "flags": dict(per_flag),
           "per_attr": {k: dict(v) for k, v in per_attr.items()},
           "flagged": flagged, "sparsity": sparse,
           "limits": ["깃발은 **판정이 아니다** — 사람이 읽을 칸을 좁힐 뿐이다(x502 가 판정의 형식).",
                      "닫힌 술어만 썼다([[59]]). 의미 판단 0.",
                      "회수율이 7/7 이 아니면 **깃발 없는 칸도 깨끗하다는 뜻이 아니다**([[25]]).",
                      "이 표는 `--conditional overlay` 가 얹힌 판본이다."]}
    dst = os.path.join(OUT, "x518_table_audit_2026_08_24.json")
    with io.open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("")
    print("-> %s" % os.path.normpath(dst))
    return 0


if __name__ == "__main__":
    sys.exit(main())
