# -*- coding: utf-8 -*-
r"""후보별 값 주석(`op.value_formula`) 회귀 검정 — 엔진은 **곱셈·뺄셈만** 한다

근거: C562(같은 48 문항에서 값 없이 0.38 → 값 주면 **0.98**) · 부작용 선계량 C563.
여기서 고정하는 계약 네 가지:
    ⒜ 값이 **결정론 참조와 축자 일치**한다(요율 × 금액 − 연회비)
    ⒝ 금액을 손님이 말하지 않았으면 **아무것도 붙이지 않는다**(추측 금지)
    ⒞ 그 카드에 적용될 요율이 문서에 없으면 **unverified** 라고 말한다(날조 금지)
    ⒟ 엔진은 **정렬하지 않고 지목하지 않는다** — 산출은 후보별 값의 표뿐([[62]])
"""
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

# ★플래그 기본값은 **OFF** 다(2026-08-20 밤·A/B 준비) — 검정은 켠 상태의 계약을 고정한다.
#   `base` = ⒟안(범주 분기 없음). `full`(⒜안)은 별도 항목에서 본다.
os.environ.setdefault("T2_VALUE_FORMULA", "base")

import t2_compute as C  # noqa: E402

LABEL = "documented_return_for_stated_spend"


def spec():
    with io.open(os.path.join(HERE, "a2", "banking_knowledge.gate.json"), encoding="utf-8") as f:
        d = json.load(f)
    return [x for x in d["scaffold_get_tools"]
            if x["name"] == "check_card_application_fit"][0]["op"]


def values(ctx):
    out = C.apply_op(spec(), ctx)
    got = {}
    for grp in ("eligible", "unverified", "excluded"):
        for it in (out.get(grp) or []):
            v = ((it.get("facts") or {}).get(LABEL))
            if v is not None:
                got[it["card"]] = v
    return got, out


def num(s):
    return float(str(s).split("=")[0].strip())


def main():
    ok = True

    # ⒜ 결정론 참조와 일치 — personal · $8,000 · travel (기본 요율만·⒟안)
    got, _ = values({"business": False, "spend_amount": 8000, "spend_category": "travel"})
    want = {"Platinum Rewards Card": 600.0,      # 10.0% × 8000 − 200
            "Gold Rewards Card": 200.0,          # 2.5% × 8000
            "Silver Rewards Card": 80.0,         # ⒟: 기본 1.0% × 8000 (범주 4% 를 안 쓴다)
            "Bronze Rewards Card": 80.0,         # 1.0% × 8000
            "Crypto-Cash Back": 85.0}            # 2.0% × 8000 − 75
    for k, v in want.items():
        if k not in got or abs(num(got[k]) - v) > 0.005:
            print("  ✗ %s: %s (기대 %.2f)" % (k, got.get(k), v))
            ok = False
    print("  %s ⒜ personal·travel 5행이 참조와 일치" % ("✓" if ok else "✗"))

    # ⒜' business · $40,000 · 범주 없음 (024 형)
    got2, _ = values({"business": True, "spend_amount": 40000})
    want2 = {"Business Bronze Rewards Card": 400.0, "Business Silver Rewards Card": 277.5,
             "Business Gold Rewards Card": 200.0, "Business Platinum Rewards Card": 150.0,
             "Green Rewards Card": 300.0}
    bad = [k for k, v in want2.items() if k not in got2 or abs(num(got2[k]) - v) > 0.005]
    print("  %s ⒜' business·$40,000 5행이 참조와 일치%s"
          % ("✓" if not bad else "✗", "" if not bad else " — " + ", ".join(bad)))
    ok = ok and not bad

    # ⒝ 금액 미제공 → 주석 0
    got3, _ = values({"business": True})
    print("  %s ⒝ 금액 미제공이면 주석이 없다 (붙은 행 %d)" % ("✓" if not got3 else "✗", len(got3)))
    ok = ok and not got3

    # ⒞ 요율 없는 행은 unverified
    u = [k for k, v in got2.items() if "unverified" in v]
    print("  %s ⒞ 요율 미문서 행은 unverified (%s)" % ("✓" if u else "✗", ", ".join(u) or "없음"))
    ok = ok and bool(u)

    # ⒟ 엔진이 순위·지목을 내지 않는다
    _g, out = values({"business": True, "spend_amount": 40000})
    banned = [k for k in out if k in ("winner", "best", "recommended", "ranking", "sorted")]
    print("  %s ⒟ 산출에 순위·지목 키가 없다%s"
          % ("✓" if not banned else "✗", "" if not banned else " — " + ",".join(banned)))
    ok = ok and not banned

    print("\nRESULT: %s" % ("ALL PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
