# -*- coding: utf-8 -*-
r"""x438 — **이접(OR)을 주면 모델이 스스로 쓰는가** (어휘 격리 · 2026-08-20 밤)

## 왜 (사용자 지시 *"1,2,3 순으로 하라"* 의 1)
C555 가 남긴 자리: 격리 전달이 요구를 회수해도 **성적이 안 올랐고**, 그 이유는 손님의 한 마디
*"no foreign transaction fees"* 가 표에 **두 꼴**로 적히기 때문이다 — 어떤 계좌는 값 `0%`, 어떤 계좌는
*"해당 없음"*(문서가 없다고 말함). 우리 op 은 둘 중 하나를 고르게 강제했고 055 는 `absent` 를 골라
**값 0% 를 적어 둔 gold 만** 떨어졌다(생존 0). 오프라인 반사실로 이접이 그 자리를 여는 것은 봤지만
(`x436.relaxed()`), 그것은 **우리가 손으로 푼 것**이다. 여기서 재는 것은 다르다:

    **어휘를 주면 모델이 스스로 쓰는가 · 쓰면 어디에 쓰는가 · 안 써야 할 데 쓰지는 않는가**

## 팔 (사전 고정)
    F_now    전 문맥 · 현행 어휘            ← 라이브 재현 기준선
    F_or     전 문맥 · **이접 어휘**
    T_now    턴 하나씩 · 현행 어휘          ← C555 의 B_turn 재현
    T_or     턴 하나씩 · **이접 어휘**       ← 표적 칸
    T_noise  턴 하나씩 · **길이만 같은 무능력 문장**  ← 부정통제([[57]])

★이접은 **스키마로만** 알린다 — *언제 쓰라*는 규칙은 한 줄도 없다. 케이스를 열거하면 그 열거가
  독립 트리거가 되어 복합 발화를 밀어버린다([[66]]·C453 이 judge6 −6 을 냈다).
★`T_noise` 는 프롬프트가 길어져서 제약이 는 것인지 **능력** 때문인지를 가른다. 길이 실측:
  현행 1238자 · 이접 1446자 · 무능력 1486자.
★엔진은 여전히 **비교와 OR** 만 한다. 대안 목록은 LLM 이 짠다([[10]]·[[22]]).
★gold 는 ④ 채점에만([[23]]). `gold_in`·`unique` 는 [[69]] 상 **진단 지표**지 성적이 아니다.

## 판정 (사전 고정)
    T_or 의 이접 사용이 **0** 이면          → 어휘를 줘도 안 쓴다 = 이 축은 닫힘(전달 레버도 무의미)
    T_or > T_noise (제약·생존 기준)          → 움직인 것은 **능력**이지 길이가 아니다
    T_or ≈ T_noise                          → 판정 불가(길이 효과와 구분 안 됨)
    T_or 가 안 써야 할 데 이접을 쓰면        → 그 비용을 같이 적는다([[70]] 양방향)

사용: py -3 x438_disjunction_vocab.py [--port 8141] [--tag or1]
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

REP = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026")
ARMS = (("F_now", "full", {}),
        ("F_or", "full", {"with_or": True}),
        ("T_now", "turn", {}),
        ("T_or", "turn", {"with_or": True}),
        ("T_noise", "turn", {"filler": True}))


def n_or(cons):
    return sum(1 for c in cons if isinstance(c.get("any_of"), list) and c["any_of"])


def run_case(port, c, table):
    tbl = X.card_table()[0] if c["arg"] == "card_type" else table
    fams = X.arg_families(c["arg"])
    caps = X.table_caps(tbl, fams)
    allow = {k: sorted(v) for k, v in caps.items()}
    menu = X.attr_menu_for(c["arg"], tbl)
    ts = P.turns(c)
    full = " \n".join(ts)

    row = {"task": c["task"], "trial": c["trial"], "gold": c["gold"], "arg": c["arg"], "n_turns": len(ts)}
    for tag, delivery, kw in ARMS:
        sysmsg = X.sysmsg_spec(**kw)
        if delivery == "full":
            cons = P.spec_from(port, sysmsg, full, menu, allow)[0]
        else:
            cons = P.union([P.spec_from(port, sysmsg, t, menu, allow)[0] for t in ts])
        surv = P.survivors(tbl, fams, cons)
        hit = [x for x in surv if X.clsname(x) == X.clsname(c["gold"])]
        row[tag] = {"n_con": len(cons), "n_or": n_or(cons), "n_surv": len(surv),
                    "gold_in": bool(hit), "unique": len(surv) == 1 and bool(hit),
                    "attrs": sorted({x.get("attribute") for x in cons}), "cons": cons}
    return row


def debug_one(port, c, table, kw, label):
    """한 사례·한 어휘로 **원문 출력과 거절 사유**를 그대로 인쇄한다.

    집계는 *"제약이 0개"* 라고만 말한다 — 모델이 아무것도 안 낸 것인지, 냈는데 검산이 **전부 튕긴**
    것인지 구분이 안 된다. 003 t0 이 이접 어휘에서 제약 3 → 0 이 된 자리를 여기서 본다([[08]]).
    """
    tbl = X.card_table()[0] if c["arg"] == "card_type" else table
    fams = X.arg_families(c["arg"])
    allow = {k: sorted(v) for k, v in X.table_caps(tbl, fams).items()}
    menu = X.attr_menu_for(c["arg"], tbl)
    said = " ".join(P.turns(c))
    sysmsg = X.sysmsg_spec(**kw)
    body = ("# Customer's own words" + chr(10) + said[:6000] + chr(10) + chr(10)
            + "# Attribute names you may use" + chr(10) + menu + chr(10))
    raw = X.ask(port, sysmsg, body)
    cons, bad, dropped, _f = X.check_spec(raw, said, allow)
    print("=== %s · %s t%s" % (label, c["task"], c["trial"]))
    print("  원문: %s" % json.dumps(raw, ensure_ascii=False)[:1200])
    print("  통과 %d · 거절 %d · 선언-기각 %d" % (len(cons), len(bad), len(dropped)))
    for b in bad:
        print("    거절: %s" % b[:160])
    for d in dropped:
        print("    기각(%s): %s" % (d[1], str(d[2])[:80]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8141)
    ap.add_argument("--tag", default="or1")
    ap.add_argument("--only", default="")
    ap.add_argument("--debug", action="store_true",
                    help="--only 로 고른 사례를 세 어휘로 한 번씩 돌려 원문·거절을 인쇄")
    a = ap.parse_args()

    table = P.account_table()
    seen, cs = set(), []
    for c in I.cases(60):
        if c["arg"] not in P.ARGS:
            continue
        if a.only and c["task"] not in a.only.split(","):
            continue
        k = (c["task"], c["trial"])
        if k in seen:
            continue
        seen.add(k)
        cs.append(c)

    if a.debug:
        for c in cs:
            for kw, label in (({}, "현행"), ({"with_or": True}, "이접"), ({"filler": True}, "무능력")):
                debug_one(a.port, c, table, kw, label)
        return 0
    print("=" * 104)
    print("x438 · 이접 어휘 격리 · 사례 %d · 팔 %s" % (len(cs), " / ".join(t for t, _d, _k in ARMS)))
    print("판정(사전 고정): 이접 사용 0 = 축 닫힘 · T_or > T_noise 여야 능력 · T_or ≈ T_noise = 판정 불가")
    print("=" * 104)
    rows = []
    for c in cs:
        r = run_case(a.port, c, table)
        rows.append(r)
        print("  %-9s t%s (턴 %d) gold=%s" % (r["task"], r["trial"], r["n_turns"], r["gold"][:26]))
        for tag, _d, _k in ARMS:
            x = r[tag]
            print("      %-8s 제약 %2d (이접 %d) · 생존 %3d · %s%s  %s"
                  % (tag, x["n_con"], x["n_or"], x["n_surv"],
                     "gold O" if x["gold_in"] else "gold X", " 유일" if x["unique"] else "",
                     ",".join(x["attrs"])[:60]))
    agg = {}
    for tag, _d, _k in ARMS:
        agg[tag] = {"gold_in": sum(r[tag]["gold_in"] for r in rows),
                    "unique": sum(r[tag]["unique"] for r in rows),
                    "n_con": sum(r[tag]["n_con"] for r in rows),
                    "n_or": sum(r[tag]["n_or"] for r in rows),
                    "wipe": sum(1 for r in rows if r[tag]["n_surv"] == 0 and r[tag]["n_con"] > 0)}
    print(chr(10) + "  ★집계 " + json.dumps(agg, ensure_ascii=False))
    p = os.path.abspath(os.path.join(REP, "x438_%s.json" % a.tag))
    with io.open(p, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print("→ %s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
