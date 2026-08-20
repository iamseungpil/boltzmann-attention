# -*- coding: utf-8 -*-
r"""x437 — **요구를 질문으로 선언해 버리는 것**이 능력 결손인가 부하인가 (격리 측정)

## 왜 (사용자 지시 2026-08-20 밤 *"C 부터 하라"*)
확대 표본 8 사례에서 요구로 센 것 16 : 질문/배경으로 버린 것 22(**58%**)였고, `task_055` 는 손님의
요구 셋이 **전부 `question`** 으로 선언돼 제약이 0 이었다(`CONSTRAINT_ATTRIBUTION_2026_08_20_NIGHT` §4).
그런데 그 태스크의 뒤쪽 턴에는 요구가 **평서문으로 다시 진술**돼 있다 — 축자:
    *"...which ONE checking account is the best fit for what I said (international ATM fee rebates,
      no foreign transaction fees, and ideally the ability to hold yen/euros)"*
⇒ 가설은 **말투 판단**이 아니라 **정박**일 수 있다: 모델이 **처음 나온 의문문**을 근거로 잡고 뒤의
   평서문 재진술을 안 본다. 그렇다면 결손은 능력이 아니라 **부하**이고, 레버는 [[62]] 가 허용하는
   **전달(부하 축소)** 뿐이다.

## 팔 ([[18]] 격리 프로브 · [[57]] 부정통제)
    A_full   전 문맥 1회                     — 현행 파이프 그대로
    B_turn   손님 **턴 하나씩** N회 → 합집합  — 격리(부하 축소). 문맥 총량은 같고 **한 번에 보는 양**만 준다
    C_chunk  같은 글을 **N등분**(문자 수 기준·경계 아무데나) N회 → 합집합 — 부정통제
⇒ 부정통제의 자리: B 가 이기면 그것이 **턴 경계 때문인지 그냥 짧아서인지**를 갈라야 한다([[57]]).
  ⚠*"전 문맥을 N번 반복"* 팔은 **두지 않는다** — 디코딩이 greedy(`temperature=0.0`)라 같은 입력이면
    같은 답이 나오고 합집합은 A 와 같아진다. 즉 그 팔은 **구성상 null** 이라 정보가 0 이다
    (실측으로도 이 파이프의 무처치 3회 폭은 0 이다·C554).

## 규율
★프롬프트는 `x431.sysmsg_spec()` **그대로**다 — 팔 사이 유일한 차이는 *무엇을 보여주는가*.
★서법 판단은 **LLM 이** 한다. 엔진은 라벨을 읽기만 하고 `?`·would 따위 어휘로 갈음하지 않는다([[59]]·[[66]]).
★합집합은 `(attribute, op, value)` 동일성으로만 합친다 — 화해·중재 없음(판단 0·[[62]]).
★gold 는 ④ 채점에만 등장한다([[23]]).
★필터·표·능력 유도·검산은 전부 `x431` 정본을 import 한다([[67]] 사본 금지).

사용: py -3 x437_declaration_isolation.py [--port 8141] [--repeat 1] [--tag iso]
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

import x423_choice_isolation as I  # noqa: E402
import x431_spec_selects as X  # noqa: E402

REP = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026")
ARGS = ("account_class", "card_type")


def turns(case):
    """손님 턴만 **형태로** 자른다(뜻 해석 0) — `customer_said` 와 같은 원천·같은 정규화."""
    out = []
    for i, m in enumerate(case["sim"].get("messages") or []):
        if i >= case["msg_i"] or m.get("role") != "user":
            continue
        t = " ".join(str(m.get("content") or "").split())
        if t:
            out.append(t)
    return out


def chunks(text, n):
    """같은 글을 **n 등분** — 경계는 문자 수로만 정한다(뜻 해석 0). 부정통제의 몫이다.

    턴 경계에서 자른 B 와 **호출 수·총 내용·조각 크기**가 같고 **경계의 뜻만 없다**. B 가 이기고
    C 가 못 이기면 그 이득은 *턴 경계* 때문이고, 둘 다 이기면 **그냥 한 번에 보는 양** 때문이다.
    """
    n = max(1, n)
    w = max(1, (len(text) + n - 1) // n)
    return [text[i:i + w] for i in range(0, len(text), w)][:n] or [text]


def account_table():
    p = os.path.abspath(os.path.join(REP, "x430_account_facts_llm_filled.json"))
    with io.open(p, encoding="utf-8") as f:
        return json.load(f)


def spec_from(port, sysmsg, said, menu, allow):
    """한 덩어리의 말에서 제약을 뽑는다 — x431 과 **같은 프롬프트·같은 검산**."""
    body = ("# Customer's own words\n%s\n\n# Attribute names you may use\n%s\n" % (said[:6000], menu))
    spec = X.ask(port, sysmsg, body)
    cons, bad, dropped, dropped_full = X.check_spec(spec, said, allow)
    return cons, dropped, bad


def union(specs):
    """합집합 — 동일성은 `(attribute, op, value)` 뿐이다. 충돌은 **화해하지 않고 둘 다 남긴다**.

    ⚠둘 다 남기면 AND 라서 더 좁아진다. 그것이 이 팔의 **비용**이고, 그 비용을 감추지 않으려고
      화해기를 두지 않는다([[62]]: 엔진이 최종 판단을 하면 측정 대상이 사라진다).
    """
    seen, out = set(), []
    for cons in specs:
        for c in cons:
            k = (c.get("attribute"), c.get("op"), c.get("value"))
            if k in seen:
                continue
            seen.add(k)
            out.append(c)
    return out


def survivors(tbl, fams, cons):
    out = []
    for cls, row in tbl.items():
        if not isinstance(row, dict):
            continue
        if fams and (row.get("_family") not in fams):
            continue
        if all(X.passes(row, c) for c in cons):
            out.append(cls)
    return out


def run_case(a, c, table, sysmsg):
    tbl = X.card_table()[0] if c["arg"] == "card_type" else table
    fams = X.arg_families(c["arg"])
    caps = X.table_caps(tbl, fams)
    allow = {k: sorted(v) for k, v in caps.items()}
    menu = X.attr_menu_for(c["arg"], tbl)
    ts = turns(c)
    full = " \n".join(ts)
    n = len(ts)

    a_cons, a_drop, _ = spec_from(a.port, sysmsg, full, menu, allow)
    b_cons = union([spec_from(a.port, sysmsg, t, menu, allow)[0] for t in ts])
    c_cons = union([spec_from(a.port, sysmsg, t, menu, allow)[0] for t in chunks(full, n)])

    row = {"task": c["task"], "trial": c["trial"], "gold": c["gold"], "arg": c["arg"], "n_turns": n}
    for tag, cons in (("A_full", a_cons), ("B_turn", b_cons), ("C_chunk", c_cons)):
        surv = survivors(tbl, fams, cons)
        hit = [x for x in surv if X.clsname(x) == X.clsname(c["gold"])]
        row[tag] = {"n_con": len(cons), "attrs": sorted({x.get("attribute") for x in cons}),
                    "n_surv": len(surv), "gold_in": bool(hit), "unique": len(surv) == 1 and bool(hit),
                    "cons": cons}
    row["A_dropped"] = [(x[0], x[1]) for x in a_drop]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8141)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--tag", default="iso")
    ap.add_argument("--only", default="", help="쉼표로 태스크 제한(예: task_055)")
    a = ap.parse_args()

    table = account_table()
    sysmsg = X.sysmsg_spec()
    seen, cs = set(), []
    for c in I.cases(60):
        if c["arg"] not in ARGS:
            continue
        if a.only and c["task"] not in a.only.split(","):
            continue
        k = (c["task"], c["trial"])
        if k in seen:
            continue
        seen.add(k)
        cs.append(c)

    print("=" * 100)
    print("x437 · 요구/질문 선언 격리 · 사례 %d · 팔 A_full / B_turn / C_chunk(부정통제)" % len(cs))
    print("=" * 100)
    for rep in range(max(1, a.repeat)):
        rows = []
        for c in cs:
            r = run_case(a, c, table, sysmsg)
            rows.append(r)
            print("  %-9s t%s (턴 %d) gold=%s" % (r["task"], r["trial"], r["n_turns"], r["gold"][:26]))
            for tag in ("A_full", "B_turn", "C_chunk"):
                x = r[tag]
                print("      %-7s 제약 %2d · 생존 %3d · %s%s  %s"
                      % (tag, x["n_con"], x["n_surv"],
                         "gold O" if x["gold_in"] else "gold X",
                         " 유일" if x["unique"] else "", ",".join(x["attrs"])[:70]))
        agg = collections.OrderedDict()
        for tag in ("A_full", "B_turn", "C_chunk"):
            agg[tag] = {"gold_in": sum(r[tag]["gold_in"] for r in rows),
                        "unique": sum(r[tag]["unique"] for r in rows),
                        "n_con": sum(r[tag]["n_con"] for r in rows),
                        "wipe": sum(1 for r in rows if r[tag]["n_surv"] == 0 and r[tag]["n_con"] > 0)}
        print(chr(10) + "  ★[%d회차] %s" % (rep + 1, json.dumps(agg, ensure_ascii=False)))
        p = os.path.abspath(os.path.join(REP, "x437_%s%d.json" % (a.tag, rep + 1)))
        with io.open(p, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=1)
        print("→ %s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
