# -*- coding: utf-8 -*-
r"""x444 — **범주가 틀렸을 때 두 값 병기가 막아주나** (C565 ⒝ 안 격리 측정 · 2026-08-20 밤)

## 왜 (사용자 지시 *"B안으로 가라"*)
C565 스모크: 024 는 트럭 구매를 `spend_category=operations` 로 매핑했고, 값을 **하나만** 보이면 엔진이
**틀린 범주 위에서 자신 있게 곱셈**을 해 준다(Business Gold **800** > gold Bronze 400). 범주 소속은
**열린 술어**라 엔진이 판정할 수 없다([[22]]·[[66]]·[[23]]) ⇒ ⒝안 = **산술만 둘 다** 하고 판단은 모델에.

## 설계 — 일부러 틀린 범주로 주석을 만든다
    손님이 말한 것(참) : `cat_true` 에 $amt 를 쓴다
    주석이 계산한 것    : **`cat_wrong`**(문서화된 다른 범주)로 계산 = 오매핑 재현
    팔 `E_one`  : 값 하나(오매핑 값만)
    팔 `F_two`  : 값 둘 — 오매핑 값 **|** *"그 지출이 이 카드의 보너스 범주가 아니면"* 기본 요율 값
                  (엔진이 실제로 내는 문면 그대로)

## 채점 (gold 미등장 · 전부 결정론 참조)
    `ok_true`   = 손님이 말한 **참 범주** 기준 argmax 를 골랐나            ← 정직한 답
    `followed`  = **오매핑 값** 기준 argmax 를 골랐나                       ← 부풀린 수를 따라갔나
판정(사전 고정): `followed(F_two) < followed(E_one)` 이어야 병기가 일한다. 차 8/48 이상만 읽는다.

사용: py -3 x444_mismap_hedge_iso.py [--port 8141]
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

import x431_spec_selects as X  # noqa: E402
import x441_objective_setup_iso as M  # noqa: E402
import t2_probe as PR  # noqa: E402

REP = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026")
SYS = M.SYS


def rate_of(r, cat):
    cr = r.get("category_rates") or {}
    return (cr.get(cat), "category rate") if cat in cr else (r.get("base_cashback"), "base rate")


def val(r, rate):
    return 40000 * 0 + (rate is not None) and None  # placeholder replaced below


def value_text(r, cat, amt, two):
    """엔진(`t2_compute`)이 실제로 내는 문면 그대로 — 프로브가 다른 말을 쓰면 측정이 딴것이 된다."""
    rate, basis = rate_of(r, cat)
    if rate is None:
        return "unverified — this card documents no rate that applies to that spending"
    fee = float(r.get("annual_fee") or 0.0)
    v = amt * float(rate) / 100.0 - fee
    txt = "%.2f = %s%% (%s) x %s%s" % (v, rate, basis, amt, (" minus annual_fee %s" % fee) if fee else "")
    if two and basis == "category rate":
        b = r.get("base_cashback")
        if b is not None and abs(float(b) - float(rate)) > 1e-9:
            bv = amt * float(b) / 100.0 - fee
            txt += ("  |  if that spending is NOT in this card's bonus category: "
                    "%.2f = %s%% (base rate) x %s%s" % (bv, b, amt, (" minus annual_fee %s" % fee) if fee else ""))
    return txt


def block(cards, cat, amt, two):
    out = []
    for r in cards:
        bits = ["annual fee $%s" % (r.get("annual_fee") or 0.0), "base rate %s%%" % r.get("base_cashback")]
        cr = r.get("category_rates") or {}
        if cr:
            bits.append("category rates: " + ", ".join("%s %s%%" % (k, v) for k, v in sorted(cr.items())))
        bits.append("estimated return for this customer: " + value_text(r, cat, amt, two))
        out.append("- %s: %s" % (r["card"], " · ".join(bits)))
    return "\n".join(out)


def argmax(cards, cat, amt):
    sc = sorted(((M.net(r, cat, amt), r["card"]) for r in cards if r.get("base_cashback") is not None),
                reverse=True)
    return sc[0][1] if sc and (len(sc) < 2 or sc[0][0] > sc[1][0] + 1e-9) else None


def items():
    """참 범주 ≠ 주석 범주 인 문항만 — 둘의 argmax 가 **갈릴 때만** 채택(안 갈리면 잴 것이 없다)."""
    rows = M.rows_a2()
    out = []
    for fam, sel in (("personal", lambda r: not r.get("business")),
                     ("business", lambda r: bool(r.get("business")))):
        cards = [r for r in rows if sel(r) and r.get("base_cashback") is not None]
        docs = sorted({k for r in cards for k in (r.get("category_rates") or {})})
        for cat_true in M.CATS:
            for wrong in docs:
                if wrong == cat_true:
                    continue
                for amt in M.AMTS:
                    a_true, a_wrong = argmax(cards, cat_true, amt), argmax(cards, wrong, amt)
                    if not a_true or not a_wrong or a_true == a_wrong:
                        continue
                    out.append({"family": fam, "cat_true": cat_true, "cat_wrong": wrong, "amt": amt,
                                "cards": cards, "true": a_true, "wrong": a_wrong})
    return out


def ask(port, it, two):
    cat = ("purchases that are not in any bonus category" if it["cat_true"] == "other"
           else it["cat_true"])
    body = ("# Cards\n%s\n\n# This customer\nOver the next year they will spend $%d on %s, and little "
            "else.\n\n# Note\nThe estimated returns above were computed for the category '%s'.\n"
            "\n# Question\nWhich single card returns the most to this customer?\n"
            % (block(it["cards"], it["cat_wrong"], it["amt"], two), it["amt"], cat, it["cat_wrong"]))
    return X.ask(port, SYS, body, maxtok=300)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8141)
    ap.add_argument("--tag", default="mismap1")
    ap.add_argument("--limit", type=int, default=48)
    a = ap.parse_args()
    its = items()[:a.limit]
    print("=" * 104)
    print("x444 · 오매핑 대치 · 문항 %d (참 범주 argmax ≠ 오매핑 argmax 인 것만)" % len(its))
    print("판정: followed(F_two) < followed(E_one) 이어야 병기가 일한다 · 차 8 이상만 읽는다")
    print("=" * 104)
    rows, cnt = [], collections.Counter()
    for it in its:
        r = {k: it[k] for k in ("family", "cat_true", "cat_wrong", "amt", "true", "wrong")}
        for arm, two in (("E_one", False), ("F_two", True)):
            ans = ask(a.port, it, two) or {}
            w = str(ans.get("winner") or "")
            hit_t = PR._word_in(w, it["true"]) and PR._word_in(it["true"], w)
            hit_w = PR._word_in(w, it["wrong"]) and PR._word_in(it["wrong"], w)
            r[arm] = {"winner": w, "ok_true": bool(hit_t), "followed": bool(hit_w),
                      "why": str(ans.get("why") or "")[:140]}
            cnt[(arm, "ok_true")] += bool(hit_t)
            cnt[(arm, "followed")] += bool(hit_w)
        rows.append(r)
        print("  %-8s 참 %-14s 주석 %-16s $%-6d | E=%s F=%s"
              % (it["family"], it["cat_true"], it["cat_wrong"], it["amt"],
                 ("따라감" if r["E_one"]["followed"] else "정직" if r["E_one"]["ok_true"] else "기타"),
                 ("따라감" if r["F_two"]["followed"] else "정직" if r["F_two"]["ok_true"] else "기타")))
    n = len(rows)
    print(chr(10) + "  ★%d 문항 — 따라감 E_one %d · F_two %d   |   정직 E_one %d · F_two %d"
          % (n, cnt[("E_one", "followed")], cnt[("F_two", "followed")],
             cnt[("E_one", "ok_true")], cnt[("F_two", "ok_true")]))
    p = os.path.abspath(os.path.join(REP, "x444_%s.json" % a.tag))
    with io.open(p, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print("→ %s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
