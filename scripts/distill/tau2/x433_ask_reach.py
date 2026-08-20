# -*- coding: utf-8 -*-
r"""x433 — **되묻기의 사정거리**: 남은 후보를 가르는 속성을 손님이 말했나 (사용자 물음 2026-08-20)

## 물음
*"요구 누락된 건 손님에게 다시 물어보면 되는 거 아닌가?"* — 맞는 경우와 틀린 경우가 갈린다.
057 t1 은 손님이 **이미 말했는데**(*"I want a safety net: can I link a savings account …"*) 스펙이
못 건진 경우다. 거기서 되묻는 것은 turn 을 팔 뿐이다([[70]]) — 그리고 우리 원장 **C100** 은
H_min 결정기가 `dispute_reason` 에서 **58% ASK** 로 흐른 것을 이미 쟀다(정답 28%).
**ASK 는 기권을 사지 정답을 못 산다.**

    ⒜ 말했는데 스펙이 놓쳤다  → 회수(recall) 문제 · 되묻기는 오답
    ⒝ 말한 적 없다            → ASK 가 정답 · 없는 정보는 만들 수 없다

## 재는 것 (분담은 [[10]] 그대로)
    엔진: 생존 후보를 **실제로 가르는 속성**을 표 산술로 뽑는다(값이 서로 다른 칸·판단 0)
    LLM : 그 속성마다 *"손님이 이것에 대해 말했나"* 를 선언하고 **축자 인용**을 댄다
    엔진: 그 인용이 손님 발화에 실재하는지만 검산한다(cite_norm·[[66]] 가리키기+substring)

⇒ ⒜/⒝ 의 비율이 곧 **되묻기로 살 수 있는 몫**이다. ⒜ 가 크면 표적은 ASK 가 아니라 회수다.

★엔진은 낱말을 안 본다. 서법도 안 본다. gold 는 이 프로브에 **등장하지 않는다**.

사용: py -3 x433_ask_reach.py [--port 8141]
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
import x426_free_gates as G  # noqa: E402
import x431_spec_selects as S  # noqa: E402

SYS = ("You are given a customer's own words and one account attribute. "
       "Reply with ONE JSON object only: {\"mentioned\": true, \"quote\": \"<verbatim span from the "
       "customer's words>\"} if the customer said anything about that attribute - whether as a "
       "requirement, a question, or a preference - otherwise {\"mentioned\": false}. "
       "The quote must be copied character-for-character.")


def discriminating(table, survivors):
    """생존 후보를 **실제로 가르는** 속성 = 값이 서로 다른 칸(엔진 산술·판단 0)."""
    out = []
    for at in S.ATTRS:
        vals = []
        for c in survivors:
            v = ((table.get(c) or {}).get(at) or {}).get("values") or []
            vals.append(v[0] if v else None)
        known = [v for v in vals if v is not None]
        if len(known) >= 2 and len(set(known)) >= 2:
            out.append((at, dict(zip(survivors, vals))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8141)
    ap.add_argument("--max-surv", type=int, default=8, help="후보가 이보다 많으면 건너뛴다(가름이 무의미)")
    a = ap.parse_args()

    with io.open(os.path.abspath(S.TBL), encoding="utf-8") as f:
        table = json.load(f)
    p431 = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "x431_spec_selects.json")
    with io.open(os.path.abspath(p431), encoding="utf-8") as f:
        prev = {(r["task"], r["trial"]): r for r in json.load(f)}

    seen, cs = set(), []
    for c in I.cases(60):
        if c["arg"] != "account_class":
            continue
        k = (c["task"], c["trial"])
        if k in seen:
            continue
        seen.add(k)
        cs.append(c)

    print("=" * 100)
    print("x433 · 되묻기의 사정거리 · 사례 %d" % len(cs))
    print("=" * 100)
    tal = collections.Counter()
    out = []
    for c in cs:
        r = prev.get((c["task"], c["trial"]))
        surv = (r or {}).get("surv") or []
        if not surv or len(surv) > a.max_surv:
            print("  %-9s t%s 생존 %s — 건너뜀(가름 무의미)" % (c["task"], c["trial"], len(surv)))
            continue
        said = G.customer_said(c["sim"], c["msg_i"])
        disc = discriminating(table, surv)
        print("\n### %s t%s · 생존 %d · 가르는 속성 %d" % (c["task"], c["trial"], len(surv), len(disc)))
        for at, vals in disc:
            got = S.ask(a.port, SYS, "# Customer's own words\n%s\n\n# Attribute\n%s\n"
                        % (said[:6000], at.replace("_", " ")), maxtok=200)
            said_it = bool(got.get("mentioned"))
            q = " ".join(str(got.get("quote") or "").split())
            verified = bool(q) and S.cite_norm(q) in S.cite_norm(said)
            kind = ("⒜말했다(스펙이 놓침)" if (said_it and verified)
                    else ("⛔인용 미검증" if said_it else "⒝안 말했다 → ASK"))
            tal[kind] += 1
            out.append({"task": c["task"], "trial": c["trial"], "attr": at, "kind": kind,
                        "quote": q[:120], "values": vals})
            print("   %-34s %s %s" % (at, kind, ("← " + q[:70]) if verified else ""))
    n = sum(tal.values())
    print("\n%-26s %s" % ("판정", "수"))
    for k, v in tal.most_common():
        print("%-26s %d/%d (%.0f%%)" % (k, v, n, 100.0 * v / n if n else 0))
    if n:
        a_cnt = tal["⒜말했다(스펙이 놓침)"]
        print("\n  ★되묻기로 살 수 있는 몫 = ⒝ **%d/%d**. ⒜ **%d/%d** 는 되물어도 안 산다 — 회수 문제다."
              % (tal["⒝안 말했다 → ASK"], n, a_cnt, n))
    p = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "x433_ask_reach.json")
    with io.open(os.path.abspath(p), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n→ %s" % os.path.abspath(p))
    return 0


if __name__ == "__main__":
    sys.exit(main())
