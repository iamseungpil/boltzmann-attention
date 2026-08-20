# -*- coding: utf-8 -*-
r"""x432 — **제약 집합이 판마다 흔들리나** + **축자 인용 실패의 정체** (사용자 지시 2026-08-20)

## 왜
x431 실측: 같은 손님 발화·같은 `temperature=0.0` 인데 057 t1 의 제약이 한 판은 4개(안전망 포함),
다음 판은 3개(안전망 빠짐)로 달라졌다. 안전망 제약이 빠지자 목적함수가 `green_fee-free`(ATM $0)를
골라 gold(`blue`)를 놓쳤다. ⇒ **목적함수가 틀린 게 아니라 그 앞의 제약 집합이 불안정**할 수 있다.
그리고 축자 인용 실패가 3 사례에서 남았다 — 무엇을 어떻게 바꿔 쓰는지 봐야 고칠 수 있다.

## 재는 것
    ⒜ 안정성   같은 입력 n회 → 제약 집합 (속성,op,값) 의 **Jaccard 평균**과 **속성 등장률**
    ⒝ 인용 실패 거절된 `because` 마다 손님 발화에서 **가장 가까운 실제 조각**을 찾아 나란히 찍는다
                (최장 공통 부분열 기준 — 판단 0·형태만)

★엔진은 여기서도 뜻을 해석하지 않는다. 붙여 놓고 **사람이 읽는다**([[08]]).

사용: py -3 x432_spec_stability.py [--n 5] [--port 8141]
"""
import argparse
import collections
import difflib
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


def key_of(con):
    return (con.get("attribute"), con.get("op"), str(con.get("value")))


def jaccard(a, b):
    a, b = set(a), set(b)
    return (len(a & b) / float(len(a | b))) if (a or b) else 1.0


def closest_span(why, said):
    """손님 발화에서 `why` 와 가장 가까운 **실제 조각**(형태만·판단 0)."""
    words = said.split()
    n = max(4, len(why.split()))
    best, score = "", 0.0
    for i in range(0, max(1, len(words) - n + 1)):
        cand = " ".join(words[i:i + n + 4])
        r = difflib.SequenceMatcher(None, why.lower(), cand.lower()).ratio()
        if r > score:
            best, score = cand, r
    return best, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--port", type=int, default=8141)
    ap.add_argument("--temp", type=float, default=0.0)
    a = ap.parse_args()

    seen, cs = set(), []
    for c in I.cases(60):
        if c["arg"] != "account_class":
            continue
        k = (c["task"], c["trial"])
        if k in seen:
            continue
        seen.add(k)
        cs.append(c)

    print("=" * 104)
    print("x432 · 제약 집합 안정성 · 사례 %d × n=%d · temp=%.1f" % (len(cs), a.n, a.temp))
    print("=" * 104)
    out = []
    for c in cs:
        said = G.customer_said(c["sim"], c["msg_i"])
        body = ("# Customer's own words\n%s\n\n# Attribute names you may use\n%s\n"
                % (said[:6000], ", ".join(S.ATTRS)))
        runs, badspans = [], []
        for _k in range(a.n):
            spec = S.ask(a.port, S.sysmsg_spec(), body)
            cons, bad = S.check_spec(spec, said)
            runs.append([key_of(x) for x in cons])
            for x in (spec.get("constraints") or []):
                why = " ".join(str(x.get("because") or "").split())
                if why and why.lower() not in " ".join(said.split()).lower():
                    badspans.append((x.get("attribute"), why))
        pairs = [jaccard(runs[i], runs[j]) for i in range(len(runs)) for j in range(i + 1, len(runs))]
        jac = sum(pairs) / len(pairs) if pairs else 1.0
        freq = collections.Counter(k[0] for r in runs for k in set(r))
        print("\n### %s t%s  gold=%s" % (c["task"], c["trial"], c["gold"]))
        print("   제약 수 %s · **Jaccard 평균 %.2f**" % ([len(r) for r in runs], jac))
        print("   속성 등장률: %s" % ", ".join("%s %d/%d" % (k, v, a.n) for k, v in freq.most_common(8)))
        out.append({"task": c["task"], "trial": c["trial"], "gold": c["gold"],
                    "sizes": [len(r) for r in runs], "jaccard": round(jac, 3),
                    "freq": dict(freq), "bad_spans": badspans[:6]})
        for at, why in badspans[:3]:
            near, r = closest_span(why, said)
            print("   ⛔인용 실패 %-28s r=%.2f" % (at, r))
            print("      모델: %s" % why[:150])
            print("      실제: %s" % near[:150])
    print("\n%-9s %-3s %-10s %s" % ("task", "t", "Jaccard", "제약 수"))
    for r in out:
        print("%-9s %-3s %-10.2f %s" % (r["task"], r["trial"], r["jaccard"], r["sizes"]))
    p = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "x432_spec_stability.json")
    with io.open(os.path.abspath(p), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n→ %s" % os.path.abspath(p))
    return 0


if __name__ == "__main__":
    sys.exit(main())
