# -*- coding: utf-8 -*-
r"""x440 — **선택 근거의 계산**을 격리로 잰다 (2026-08-20 밤 · 사용자 지시 *"진행하라"*)

## 왜
오늘 네 축이 전부 같은 자리를 가리켰다 — C559 의 반례가 그것을 가장 또렷이 말한다: 에이전트가 세 장을
**구별 속성까지 정확히 열거**했는데 손님이 gold 가 아닌 카드를 골랐다. 손님 규칙은 *"최소 연회비 · 동률이면
최고 캐시백"* 이고 후보는 `Gold(전 구매 2.5%)` ↔ `Silver(여행 4%·기본 1%)` 다 ⇒ **손님의 지출 패턴에 맞춘
비교를 해 주지 않으면 그 동률은 손님 쪽에서 임의로 갈린다**. §1-4 도 같은 말을 한다: *"카탈로그 선택의
정체는 필터가 아니라 argmin/argmax"*.

⚠**먼저 재고 나서**다([[62]]). x424 가 이미 *"피연산자 맞음·결과 틀림 **0**/144"* 를 냈다 — 즉 **산술은
  안 틀린다**. 그러므로 여기서 재는 것은 계산기가 아니라 **어느 수를 쓸지 고르는 것**이다.

## 팔 (사전 고정)
    A_min    후보 표(우리 A2 카드표 전 행) + 손님 발화 **축자 인용만** + 물음
    B_full   같은 물음 · 손님 발화 **전문**(부하)
    C_nopat  같은 물음 · **손님 패턴을 뺀** 후보 표만 — 부정통제([[57]])
             → C 에서도 같은 답이 나오면 이 측정은 **계산이 아니라 prior** 를 재고 있는 것이다

## 채점 ([[23]] gold 는 채점에만)
    ⒜ 지목한 후보가 gold 인가
    ⒝ 답이 **우리가 준 수**를 둘 이상 쓰고 있나(존재 확인만 · 산문 파싱 0)
    ⒞ A ↔ B 차 = 부하 · A ↔ C 차 = 그 답이 패턴에서 왔나

## 표본
`card_type` 사례만 쓴다 — 계좌 축에는 **적격 도구가 아예 없어서**(§1-4) 후보 표를 주면 정보-맞춤이
깨진다([[18]]). 그래서 이 프로브의 표본은 **003·024·063 t0** 이다(사례 5).

사용: py -3 x440_selection_calc_iso.py [--port 8141] [--tag calc1]
"""
import argparse
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

import x423_choice_isolation as I  # noqa: E402
import x431_spec_selects as X  # noqa: E402
import x437_declaration_isolation as P  # noqa: E402
import t2_probe as PR  # noqa: E402

REP = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026")

SYS = ("You help a bank customer choose one product. You are given a table of candidates and, unless it is "
       "withheld, what the customer said. Reply with ONE JSON object only: "
       "{\"winner\": \"<exact candidate name from the table>\", "
       "\"why\": \"<one sentence>\", "
       "\"numbers\": [<every number from the table you used, as bare numbers>]}. "
       "Use only numbers that appear in the table.")


def block(tbl):
    """후보 표를 그대로 편다 — 우리 표의 값만, 해석 0."""
    out = []
    for name, row in sorted(tbl.items()):
        bits = []
        for k, c in sorted(row.items()):
            if not isinstance(c, dict) or not c.get("values"):
                continue
            bits.append("%s=%s" % (k, c["values"][0]))
        out.append("- %s: %s" % (name, ", ".join(bits)))
    return "\n".join(out)


def numbers(tbl):
    """표에 실제로 있는 수의 닫힌 집합 — 답이 쓴 수를 **존재로만** 확인한다."""
    ns = set()
    for row in tbl.values():
        for c in row.values():
            if isinstance(c, dict) and c.get("values"):
                v = X.cellval(c)
                if v is not None:
                    ns.add(v)
    return ns


def ask_arm(port, tblblock, said):
    body = "# Candidates\n%s\n" % tblblock
    if said:
        body += "\n# What the customer said\n%s\n" % said[:5000]
    body += "\n# Question\nWhich single candidate is best for this customer, and which numbers decide it?\n"
    return X.ask(port, SYS, body, maxtok=500)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8141)
    ap.add_argument("--tag", default="calc1")
    a = ap.parse_args()

    card, _ = X.card_table()
    blk, pool = block(card), numbers(card)
    seen, cs = set(), []
    for c in I.cases(60):
        if c["arg"] != "card_type":
            continue
        k = (c["task"], c["trial"])
        if k in seen:
            continue
        seen.add(k)
        cs.append(c)

    print("=" * 100)
    print("x440 · 선택 근거의 계산 격리 · 사례 %d · 팔 A_min / B_full / C_nopat(부정통제)" % len(cs))
    print("판정(사전 고정): A 가 되고 B 가 안 되면 **부하** · A 도 안 되면 **경계** · C 가 같으면 **prior**")
    print("=" * 100)
    rows = []
    for c in cs:
        ts = P.turns(c)
        full = " \n".join(ts)
        # A_min 의 '축자 인용' = 손님 발화 중 **수가 든 문장**만 형태로 고른다(뜻 해석 0).
        quotes = " \n".join([s for s in full.split(". ") if any(ch.isdigit() for ch in s)][:12])
        r = {"task": c["task"], "trial": c["trial"], "gold": c["gold"]}
        for tag, said in (("A_min", quotes), ("B_full", full), ("C_nopat", "")):
            ans = ask_arm(a.port, blk, said)
            win = str((ans or {}).get("winner") or "")
            used = [x for x in (ans or {}).get("numbers") or [] if isinstance(x, (int, float))]
            r[tag] = {"winner": win, "hit": PR._word_in(win, c["gold"]) or PR._word_in(c["gold"], win),
                      "n_used": len([x for x in used if float(x) in pool]),
                      "why": str((ans or {}).get("why") or "")[:200]}
        rows.append(r)
        print("  %-9s t%s gold=%s" % (r["task"], r["trial"], r["gold"]))
        for tag in ("A_min", "B_full", "C_nopat"):
            x = r[tag]
            print("      %-8s → %-28s %s · 표의 수 %d개 사용" %
                  (tag, x["winner"][:28], "적중" if x["hit"] else "빗나감", x["n_used"]))
    agg = {t: {"hit": sum(r[t]["hit"] for r in rows), "n": len(rows),
               "used": sum(r[t]["n_used"] for r in rows)} for t in ("A_min", "B_full", "C_nopat")}
    print(chr(10) + "  ★집계 " + json.dumps(agg, ensure_ascii=False))
    p = os.path.abspath(os.path.join(REP, "x440_%s.json" % a.tag))
    with io.open(p, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print("→ %s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
