# -*- coding: utf-8 -*-
r"""x441 — **목적함수를 세우는가**: 결정론 참조로 채점하는 재측정 (2026-08-20 밤 · 사용자 지시 *"재측정하라"*)

## 왜 다시 재나 (C560 의 계기 결함)
`x440` 은 **gold 로 채점**했는데 표본 4 중 3 의 gold 가 같은 카드라, 손님 정보를 **뺀** 부정통제가
*"항상 Silver"* 라는 **상수 답**만으로 3/4 를 얻었다 ⇒ 판정 불능. 여기서는 세 가지를 고친다:

    ⒜ **채점 기준을 gold 에서 떼어낸다** — 정답 = 우리 A2 표의 수로 **결정론적으로 계산**한 argmax.
       `net = (해당 범주 요율 ∨ 기본 요율)/100 × 금액 − 연회비`. gold 는 여기 **안 들어온다**([[23]]).
    ⒝ **정답이 갈리는 표본**을 만든다 — 범주 6 × 금액 4 × 계열 2 = 48 문항, 유일 argmax 만 채택.
       상수 답으로 얻을 수 있는 상한(최빈 정답 비율)을 **같이 인쇄**한다.
    ⒞ **부정통제 유지** — 그것만이 x440 의 오독을 막았다([[57]]).

## 팔 (사전 고정)
    A_task   후보표(계열 부분집합·**범주별 요율 포함**) + *"이 손님은 올해 <범주>에 $<금액> 쓴다"*
    C_nopat  같은 표 · **금액·범주 없음** — 부정통제(상수 답의 실측 상한)
    D_hint   A_task + **목적식 한 문장**(*"돌아오는 값 = 해당 요율 × 금액 − 연회비"*)
             ⇒ 케이스 열거가 아니라 **일반식 진술**이다([[66]] 경계 준수)

## 판정 (사전 고정)
    A ≈ C            → 손님 패턴을 **아예 안 쓴다**
    D ≫ A            → 결손은 **목적함수 설정**이다(무엇을 계산할지 말해주면 한다) ⇒ 레버는 격리 서브콜
    D ≈ A 이고 둘 다 낮음 → 피연산자·산술 경계
    ★차이는 **8/48(≈17pp) 이상**일 때만 읽는다. 파이프는 greedy(temp 0)이고 무처치 폭 0 은 C554·C555 에서 확인.

## 규율
★표 블록은 **A2 원행 그대로**(범주 요율 포함) — `x440` 은 `card_table()` 이 dict 값을 버려서
  `category_rates` 를 **모델에게 안 보여줬다**(라이브 도구는 `rate_for('travel')` 를 준다·계기 결함).
★적격·자격은 이 문항의 범위 밖이라고 **문항에 명시**한다 — 여기서 재는 것은 **비교**뿐이다.

사용: py -3 x441_objective_setup_iso.py [--port 8141] [--tag obj1]
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
import t2_probe as PR  # noqa: E402

REP = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026")
CATS = ("travel", "software", "operations", "green", "media_advertising", "other")
AMTS = (500, 2000, 8000, 40000)

SYS = ("You compare payment cards for one customer. Reply with ONE JSON object only: "
       "{\"winner\": \"<exact card name from the table>\", \"amount_back\": <number or null>, "
       "\"why\": \"<one short sentence>\"}. Eligibility is out of scope — compare only what the table says.")


def rows_a2():
    with io.open(os.path.join(HERE, "a2", "banking_knowledge.gate.json"), encoding="utf-8") as f:
        d = json.load(f)
    t = [x for x in (d.get("scaffold_get_tools") or [])
         if x.get("name") == "check_card_application_fit"][0]
    return ((t.get("op") or {}).get("table")) or []


def net(r, cat, amt):
    """결정론 참조 — 표의 수만 쓴다. 이 함수가 **정답의 정의**이고 gold 는 등장하지 않는다."""
    rate = (r.get("category_rates") or {}).get(cat, r.get("base_cashback"))
    return amt * float(rate) / 100.0 - float(r.get("annual_fee") or 0.0)


def block(cards, it=None):
    """★`it` 이 주어지면 각 행 끝에 **그 손님에 대한 값**을 붙인다(팔 `E_values`).

    이것이 곧 *"최소 결정론 단계"* 의 모사다 — 엔진은 **주어진 수의 곱셈·뺄셈**만 하고 산출은
    **후보별 값의 표**다(정렬도 안 한다·엔진이 *"정답은 X"* 를 내면 측정 대상이 사라진다·[[62]]).
    라이브 도구는 지금 **요율 조회까지만** 한다(`t2_compute.py:415-436` 축자 *"조회만 한다
    (값 생성·선택·순위 0)"*) — 그 경계를 한 칸 옮기면 무엇이 달라지는지를 여기서 먼저 잰다.
    """
    out = []
    for r in cards:
        bits = ["annual fee $%s" % (r.get("annual_fee") or 0.0),
                "base rate %s%%" % r.get("base_cashback")]
        cr = r.get("category_rates") or {}
        if cr:
            bits.append("category rates: " + ", ".join("%s %s%%" % (k, v) for k, v in sorted(cr.items())))
        if r.get("cashback_scope"):
            bits.append("scope %s" % r["cashback_scope"])
        if it is not None:
            bits.append("estimated return for this customer: $%.2f" % net(r, it["cat"], it["amt"]))
        out.append("- %s: %s" % (r["card"], " · ".join(bits)))
    return "\n".join(out)


def items():
    """문항 = (계열, 범주, 금액) 중 **유일 argmax** 인 것만. 정답은 결정론 계산."""
    rows = rows_a2()
    out = []
    for fam, sel in (("personal", lambda r: not r.get("business")),
                     ("business", lambda r: bool(r.get("business")))):
        cards = [r for r in rows if sel(r) and r.get("base_cashback") is not None]
        for cat in CATS:
            for amt in AMTS:
                sc = sorted(((net(r, cat, amt), r["card"]) for r in cards), reverse=True)
                if len(sc) < 2 or sc[0][0] <= sc[1][0] + 1e-9:
                    continue                      # 동률은 안 쓴다(판정 불가)
                out.append({"family": fam, "cat": cat, "amt": amt, "cards": cards,
                            "answer": sc[0][1], "best": round(sc[0][0], 2),
                            "second": sc[1][1], "second_val": round(sc[1][0], 2)})
    return out


def ask(port, it, arm):
    cat = "purchases that are not in any bonus category" if it["cat"] == "other" else it["cat"]
    body = "# Cards" + chr(10) + block(it["cards"], it if arm == "E_values" else None) + chr(10)
    if arm != "C_nopat":
        body += ("\n# This customer\nOver the next year they will spend $%d on %s, and little else.\n"
                 % (it["amt"], cat))
    if arm == "D_hint":
        body += ("\n# How the return is defined\nWhat a card returns = the rate that applies to that "
                 "spending, times the amount spent, minus the annual fee.\n")
    body += "\n# Question\nWhich single card returns the most to this customer?\n"
    return X.ask(port, SYS, body, maxtok=300)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8141)
    ap.add_argument("--tag", default="obj1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--arms", default="A_task,C_nopat,D_hint",
                    help="돌릴 팔 — 이미 잰 팔을 다시 돌리지 않는다(산출 덮어쓰기 방지)")
    a = ap.parse_args()
    its = items()
    if a.limit:
        its = its[:a.limit]
    base = collections.Counter(x["answer"] for x in its)
    ceil = max(base.values()) / float(len(its))
    print("=" * 104)
    print("x441 · 목적함수 설정 격리 · 문항 %d (유일 argmax) · **상수 답 상한 %.2f** (최빈 정답 %s)"
          % (len(its), ceil, base.most_common(1)[0][0]))
    print("판정: A≈C 면 패턴 미사용 · D≫A 면 목적함수 설정 결손 · 둘 다 낮으면 경계 · 차 8/48 이상만 읽음")
    print("=" * 104)
    rows, hit = [], collections.Counter()
    said = collections.defaultdict(collections.Counter)
    for it in its:
        r = {k: it[k] for k in ("family", "cat", "amt", "answer", "best", "second", "second_val")}
        for arm in [x for x in a.arms.split(",") if x]:
            ans = ask(a.port, it, arm) or {}
            win = str(ans.get("winner") or "")
            ok = PR._word_in(win, it["answer"]) and PR._word_in(it["answer"], win)
            r[arm] = {"winner": win, "ok": bool(ok), "amount_back": ans.get("amount_back"),
                      "why": str(ans.get("why") or "")[:160]}
            hit[arm] += bool(ok)
            said[arm][win] += 1
        rows.append(r)
        print("  %-8s %-18s $%-6d 정답 %-30s | %s"
              % (it["family"], it["cat"], it["amt"], it["answer"],
                 " ".join("%s=%s" % (k[0], "O" if r[k]["ok"] else "X")
                          for k in [x for x in a.arms.split(",") if x])))
    n = len(rows)
    arms = [x for x in a.arms.split(",") if x]
    print(chr(10) + "  ★정답률  " + " · ".join("%s %d/%d (%.2f)" % (k, hit[k], n, hit[k] / float(n))
                                              for k in arms))
    for k in arms:
        top, cnt = said[k].most_common(1)[0]
        print("     %-8s 최빈 답 %-30s %d/%d (%.2f) — 상수성" % (k, top, cnt, n, cnt / float(n)))
    p = os.path.abspath(os.path.join(REP, "x441_%s.json" % a.tag))
    with io.open(p, "w", encoding="utf-8") as f:
        json.dump([{k: v for k, v in r.items() if k != "cards"} for r in rows], f,
                  ensure_ascii=False, indent=1)
    print("→ %s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
