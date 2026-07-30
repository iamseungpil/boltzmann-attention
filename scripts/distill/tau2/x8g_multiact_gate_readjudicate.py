#!/usr/bin/env python3
"""X8-(g) multi-act 아키텍처 착수 게이트 **재판정** (2026-07-30).

배경: `MULTIACT_DECOMPOSITION_REVIEW_2026_07_30.md` §6이 착수 게이트 3개를 걸었고, v1에서
게이트1(multi-act가 실패를 유발하나)을 **CI 비중첩**으로 통과시켰다. 그런데 v2 arm Actx에서
CI가 **중첩**해 판정이 흔들렸다.

★먼저 **판정 기준을 고친다**: CI 중첩/비중첩은 약한 기준이다 — 비중첩은 유의를 함의하지만
**중첩은 무의를 함의하지 않는다**(널리 알려진 오용). v1에서 비중첩으로 통과시킨 것도, v2에서
중첩으로 뒤집는 것도 **같은 오류**다. 직접 검정(Fisher exact)과 짝지은 검정(McNemar)으로 간다.

재판정이 답할 3문:
  G1 multi-act 페널티가 **실재**하나 (arm별 Fisher exact · 효과크기)
  G2 **문맥이 그 페널티를 줄이나** (동일 항목 짝지은 McNemar: A vs Actx, 다중-act만)
  G3 문맥 적용 **후에도** 남는 페널티가 아키텍처를 정당화할 만큼 큰가
     — 아키텍처는 문맥보다 비싸므로, 문맥이 닫는 몫을 뺀 **잔여**가 표적이다.

용법: py -3 x8g_multiact_gate_readjudicate.py [rows.jsonl]
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from math import comb

_HERE = os.path.dirname(os.path.abspath(__file__))
_SIM = os.path.abspath(os.path.join(_HERE, "..", "..", "..",
                                    "reports", "facet_rft_2026", "sim_results"))


def fisher_exact_2x2(a, b, c, d):
    """양측 Fisher exact p. 표 = [[a,b],[c,d]] (a=단일-정답, b=단일-오답, c=다중-정답, d=다중-오답)."""
    n = a + b + c + d
    r1, r2 = a + b, c + d
    c1 = a + c

    def p_of(x):
        # 초기하: 단일 그룹에서 정답 x개
        if x < 0 or x > c1 or (r1 - x) < 0 or (c1 - x) > r2:
            return 0.0
        return comb(c1, x) * comb(n - c1, r1 - x) / comb(n, r1)

    p_obs = p_of(a)
    tot = 0.0
    for x in range(0, min(c1, r1) + 1):
        px = p_of(x)
        if px <= p_obs + 1e-12:
            tot += px
    return min(1.0, tot)


def mcnemar(b, c):
    """짝지은 이분 결과의 정확 McNemar (양측). b,c = 불일치 쌍 개수."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n) * 2
    return min(1.0, p)


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, ctr - h), min(1.0, ctr + h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rows", nargs="?",
                    default=os.path.join(_SIM, "x8v2_rows.jsonl"))
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    rows = [json.loads(l) for l in open(args.rows, encoding="utf-8") if "error" not in l]
    # 시드가 여럿이면 **항목별 다수결**로 접는다(단일 시드 점추정 회피·[[08]])
    agg = defaultdict(list)
    for r in rows:
        agg[(r["arm"], r["sample_id"])].append(r)
    cell = {}
    for k, v in agg.items():
        ok = sum(1 for x in v if x["acts_exact"])
        cell[k] = {"maj": ok * 2 > len(v), "rate": ok / len(v), "n_seed": len(v),
                   "gold_acts": v[0]["gold_acts"]}
    arms = sorted({a for a, _ in cell})
    ids = sorted({s for _, s in cell})
    multi = [s for s in ids if len(cell[(arms[0], s)]["gold_acts"]) > 1]
    single = [s for s in ids if s not in multi]
    print(f"항목 {len(ids)} (단일 {len(single)} · 다중 {len(multi)}) · arm {arms} · "
          f"시드 {cell[(arms[0], ids[0])]['n_seed']} → 항목별 다수결")
    print()

    print("=" * 78)
    print("G1 — multi-act 페널티가 실재하나 (Fisher exact · CI 중첩 기준 폐기)")
    print("=" * 78)
    g1 = {}
    for arm in arms:
        sa = sum(1 for s in single if cell[(arm, s)]["maj"])
        ma = sum(1 for s in multi if cell[(arm, s)]["maj"])
        sb, mb = len(single) - sa, len(multi) - ma
        p = fisher_exact_2x2(sa, sb, ma, mb)
        ps, pm = sa / len(single), ma / len(multi)
        sl, sh = wilson(sa, len(single))
        ml, mh = wilson(ma, len(multi))
        ovl = not (sl > mh or ml > sh)
        g1[arm] = {"p": p, "single": ps, "multi": pm}
        print(f"{arm:5s} 단일 {sa}/{len(single)}={ps:.2f}  다중 {ma}/{len(multi)}={pm:.2f}  "
              f"차 {ps - pm:+.2f} · 비 {ps / pm if pm else float('inf'):.1f}x  "
              f"**Fisher p={p:.4f}** (CI중첩={'예' if ovl else '아니오'})")
    print()
    print("★CI 중첩과 Fisher p가 엇갈리는 arm이 있으면, 중첩은 **검정력 부족**의 표현이지")
    print("  효과 부재의 증거가 아니다 — 그게 v1 판정 기준의 결함이었다.")
    print()

    print("=" * 78)
    print("G2 — 문맥이 그 페널티를 줄이나 (동일 항목 짝지은 McNemar · 다중-act만)")
    print("=" * 78)
    if "Actx" in arms and "A" in arms:
        b = [s for s in multi if cell[("A", s)]["maj"] and not cell[("Actx", s)]["maj"]]
        c = [s for s in multi if not cell[("A", s)]["maj"] and cell[("Actx", s)]["maj"]]
        p = mcnemar(len(b), len(c))
        print(f"다중-act {len(multi)}항목: A만 정답 {len(b)}건 {b} · Actx만 정답 {len(c)}건 {c}")
        print(f"  순이득 {len(c) - len(b):+d} · **McNemar p={p:.4f}**")
        print(f"  다중-act 정확도: A {sum(1 for s in multi if cell[('A', s)]['maj'])}/{len(multi)}"
              f" → Actx {sum(1 for s in multi if cell[('Actx', s)]['maj'])}/{len(multi)}")
        # 단일-act 부작용도 같이 본다(제1원리: 하나 사면 하나 판다)
        sb = [s for s in single if cell[("A", s)]["maj"] and not cell[("Actx", s)]["maj"]]
        sc = [s for s in single if not cell[("A", s)]["maj"] and cell[("Actx", s)]["maj"]]
        print(f"  ⚖단일-act 부작용: A만 {len(sb)}건 {sb} · Actx만 {len(sc)}건 {sc} "
              f"(순 {len(sc) - len(sb):+d})")
    print()

    print("=" * 78)
    print("G3 — 문맥 적용 후 **잔여** 페널티가 아키텍처를 정당화하나")
    print("=" * 78)
    if "Actx" in arms:
        sa = sum(1 for s in single if cell[("Actx", s)]["maj"])
        ma = sum(1 for s in multi if cell[("Actx", s)]["maj"])
        ps, pm = sa / len(single), ma / len(multi)
        p = fisher_exact_2x2(sa, len(single) - sa, ma, len(multi) - ma)
        print(f"Actx 하에서: 단일 {ps:.2f} vs 다중 {pm:.2f} · 차 {ps - pm:+.2f} · Fisher p={p:.4f}")
        base_gap = g1["A"]["single"] - g1["A"]["multi"]
        res_gap = ps - pm
        closed = (base_gap - res_gap) / base_gap if base_gap else 0
        print(f"문맥이 닫은 몫 = ({base_gap:.2f} − {res_gap:.2f})/{base_gap:.2f} = **{100 * closed:.0f}%**")
        print(f"⇒ 아키텍처가 겨냥할 **잔여 = 격차의 {100 * (1 - closed):.0f}%**")
    print()
    print("=" * 78)
    print("판정")
    print("=" * 78)
    print("· G1이 유의(p<0.05)면 multi-act 페널티는 실재 = 게이트1 통과 유지.")
    print("· G2가 유의하고 순이득>0이면 **문맥이 더 싼 대안**이므로, 아키텍처는 문맥을 켠 뒤의")
    print("  **잔여**에 대해서만 정당화돼야 한다(G3).")
    print("· G2에서 단일-act 부작용이 순-음이면 제1원리대로 '하나 사고 하나 판' 것 —")
    print("  아키텍처 판정 전 그 상쇄를 계측에 넣어야 한다.")
    print("⚠표본 48(단일 14·다중 34)은 작다. 유의/무의 둘 다 검정력 한계 안에서 읽을 것.")


if __name__ == "__main__":
    main()
