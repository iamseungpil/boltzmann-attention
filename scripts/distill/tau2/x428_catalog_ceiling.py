# -*- coding: utf-8 -*-
r"""x428 — **카탈로그 신탁 상한**: 손님 제약 + 문서 사실로 gold 클래스가 유일하게 정해지나 (LLM 0)

## 왜 (사용자 지시 2026-08-20 · [[62]] ③ 앞의 마지막 무료 게이트)
x427 로 *"조건과 사실만 남겨도 안 풀린다"*(0/30)까지 왔다. 그러면 다음 물음은 하나뿐이다 —
**애초에 정해지기는 하는가.** 정해지지 않는 자리에 결정기를 지으면 그것은 gold 프로그램
재작성이다([[23]]·[[62]]). KB 에 *"eligible 중 무엇을 권하라"* 는 규칙은 **0건**이었다.

## 자 (게이트 A 와 같은 꼴 · id → 카탈로그 이름)
후보마다 문서 사실 슬라이스를 세우고, **손님이 실제로 말한 낱말** 중 gold 의 사실에 등장하는 것을
술어로 삼는다(= 최선의 형식화 = 신탁). 그 술어로 후보를 거른 뒤 몇이 남나.

    생존 1 = gold   ⇒ 제약이 gold 를 유일하게 정한다 → 결정론 필터가 정당한 자리
    생존 ≥ 2        ⇒ **최선의 형식화로도 안 갈린다** → 그 자리는 결정기 밖(미결정 후보)

두 판본을 함께 낸다 — 후보 **이름 낱말을 포함**한 것과 **뺀** 것. 손님이 오답 이름을 말하는 일이
5/10 이라(x427 §), 이름 낱말이 들어가면 상한이 부풀 수 있다.

★gold 는 술어 선택(신탁 정의)과 채점에만 쓴다. 어떤 처방도 여기서 나오지 않는다.

사용: py -3 x428_catalog_ceiling.py
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

import x423_choice_isolation as I  # noqa: E402
import x426_free_gates as G  # noqa: E402
import x427_catalog_minimal as CM  # noqa: E402

STOP = set("""the a an and or of for to in on at is are was were be been with without you your i my me we our
this that these those it its as by from not no do does did can could would should will shall have has had
if then than so such about into over under out up down more most less least other another any some all both
each few many much very just only also there here what which who whom when where why how am pm please
account accounts card cards bank rho""".split())
RE_W = re.compile(r"[A-Za-z][A-Za-z\-]{3,}")


def words(txt):
    return {w.lower() for w in RE_W.findall(txt or "") if w.lower() not in STOP}


def main():
    seen, cs = set(), []
    for c in I.cases(60):
        if c["arg"] not in CM.ARGS:
            continue
        k = (c["task"], c["trial"], c["arg"])
        if k in seen:
            continue
        seen.add(k)
        cs.append(c)

    print("=" * 104)
    print("x428 · 카탈로그 신탁 상한 (LLM 0) · 사례 %d" % len(cs))
    print("=" * 104)
    rows = []
    tal = collections.Counter()
    for c in cs:
        docs = G.delivered(c["sim"], c["msg_i"])
        names = CM.catalog_names(docs, c["gold"])
        facts = {n: CM.fact_slice(n, docs) for n in names}
        facts = {n: v for n, v in facts.items() if v}
        if c["gold"] not in facts:
            facts[c["gold"]] = CM.fact_slice(c["gold"], docs)
        cust = words(G.customer_said(c["sim"], c["msg_i"]))
        namewords = words(" ".join(names))
        out = {}
        for variant, cw in (("이름낱말 포함", cust), ("이름낱말 제외", cust - namewords)):
            gw = {t for t in cw if t in (facts.get(c["gold"]) or "").lower()}
            surv = [n for n, v in facts.items() if all(t in v.lower() for t in gw)] if gw else list(facts)
            uniq = len(surv) == 1 and surv[0] == c["gold"]
            out[variant] = (len(gw), len(surv), uniq)
            tal[(variant, "유일=gold" if uniq else "복수 생존")] += 1
        rows.append({"task": c["task"], "trial": c["trial"], "arg": c["arg"], "gold": c["gold"],
                     "n_cand": len(facts), "with_names": out["이름낱말 포함"],
                     "no_names": out["이름낱말 제외"]})
        print("  %-9s t%s %-14s 후보 %2d | 포함: 술어 %2d → 생존 %2d %s | 제외: 술어 %2d → 생존 %2d %s"
              % (c["task"], c["trial"], c["arg"], len(facts),
                 out["이름낱말 포함"][0], out["이름낱말 포함"][1],
                 "✅" if out["이름낱말 포함"][2] else "  ",
                 out["이름낱말 제외"][0], out["이름낱말 제외"][1],
                 "✅" if out["이름낱말 제외"][2] else ""))
    n = len(rows)
    print()
    for v in ("이름낱말 포함", "이름낱말 제외"):
        u = tal[(v, "유일=gold")]
        print("  ★신탁 상한(%s): **%d/%d = %.0f%%**" % (v, u, n, 100.0 * u / n if n else 0))
    print("  (생존 ≥2 인 사례는 최선의 형식화로도 안 갈린다 ⇒ 결정론 필터의 자리가 아니다)")
    p = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "x428_catalog_ceiling.json")
    with io.open(os.path.abspath(p), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print("\n→ %s" % os.path.abspath(p))
    return 0


if __name__ == "__main__":
    sys.exit(main())
