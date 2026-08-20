# -*- coding: utf-8 -*-
r"""x443 — **모델이 `spend_amount` 를 채우는가** (배선 생존 스모크 · 2026-08-20 밤)

## 왜 (사용자 지시 *"스모크 하라"*)
C564 가 후보별 값 주석을 배선했다. 그런데 그 주석은 **모델이 `spend_amount` 를 채울 때만** 붙는다 —
안 채우면 배선은 **아무 일도 안 한다**. 유료 런을 태우기 전에 그 발화율부터 잰다([[67]] `t2_liveness`
0단계 동형 · [[30]] *"스모크 없이 full-run 금지"* · [[09]] 무료 선행).

## 두 방향을 같이 잰다 ([[70]])
    ⒜ **발화**   손님이 금액을 말한 사례에서 `spend_amount` 를 채우나
    ⒝ **오발화** 손님이 **지출 금액을 말하지 않은** 사례에서 다른 수(연소득·희망 한도)를 넣지는 않나
       ⇒ 003 이 그 사례다: *"annual income is $180,000"* · *"credit limit … at least $100,000"* 는
         **지출 금액이 아니다**. 여기에 무엇을 넣는지가 이 파라미터의 위험이다.

## 규율
★도구 스키마는 A2 **축자 그대로** 준다(우리가 새로 설명하지 않는다). 손님 발화도 축자.
★엔진은 아무 판단도 안 한다 — 모델이 낸 JSON 의 **키 존재와 값**만 본다([[59]]).
★gold 는 등장하지 않는다.

사용: py -3 x443_spend_amount_smoke.py [--port 8141]
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
TOOL = "check_card_application_fit"


def tool_spec():
    with io.open(os.path.join(HERE, "a2", "banking_knowledge.gate.json"), encoding="utf-8") as f:
        d = json.load(f)
    t = [x for x in d["scaffold_get_tools"] if x.get("name") == TOOL][0]
    return t.get("description") or "", t.get("params") or {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8141)
    ap.add_argument("--tag", default="smoke1")
    a = ap.parse_args()
    desc, params = tool_spec()
    sysmsg = ("You are calling one tool. Reply with ONE JSON object only: the arguments for the call. "
              "Include a parameter ONLY if the customer's own words support it — omit anything they did "
              "not state. Do not invent numbers.")
    body_tool = ("# Tool\n%s: %s\n\n# Parameters\n%s\n"
                 % (TOOL, desc, "\n".join("- %s: %s" % (k, v) for k, v in sorted(params.items()))))

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
    print("x443 · `spend_amount` 발화 스모크 · 사례 %d" % len(cs))
    print("⒜ 금액을 말한 사례에서 채우나 · ⒝ 말하지 않은 사례에 **다른 수**를 넣지는 않나")
    print("=" * 100)
    rows = []
    for c in cs:
        said = " \n".join(P.turns(c))
        ans = X.ask(a.port, sysmsg, body_tool + "\n# What the customer said\n%s\n" % said[:6000],
                    maxtok=400) or {}
        amt = ans.get("spend_amount")
        cat = ans.get("spend_category")
        rows.append({"task": c["task"], "trial": c["trial"], "spend_amount": amt,
                     "spend_category": cat, "args": ans})
        print("  %-9s t%s  spend_amount=%-10s spend_category=%-12s | 전체 인자 %s"
              % (c["task"], c["trial"], amt, cat, ",".join(sorted(ans.keys()))[:70]))
    n = len(rows)
    filled = sum(1 for r in rows if r["spend_amount"] not in (None, "", 0))
    print(chr(10) + "  ★`spend_amount` 를 채운 사례 %d/%d" % (filled, n))
    p = os.path.abspath(os.path.join(REP, "x443_%s.json" % a.tag))
    with io.open(p, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print("→ %s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
