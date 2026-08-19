# -*- coding: utf-8 -*-
r"""x425 — **선택 오류를 미리 잴 수 있나**: 후보 집합의 에너지 간격(modern Hopfield) · H_min · 예측 정확도

## 왜 (사용자 물음 2026-08-20)
*"선택오류의 근본원인을 모던 홉필드 이론으로 재조명할 수 있나? 애매모호함 H_min 으로 미리 계산할 수
있나? 홉필드의 구멍 패턴에 우리 질의가 얼마나 가깝고 애매모호한지 미리 측정하고 예측 정확도를
정량적으로 미리 알 수 있나?"*

## 우리 repo 가 이미 갖고 있는 것 (재유도 금지 · [[03]])
- **C87 [M]** (`PRIORWORK_SYNTHESIS_4AXIS_2026_07_14` §축4·verify 21/21): Ramsauer 2008.02217 —
  어텐션 ≡ modern Hopfield 업데이트 · **분리 잘 된 패턴 = 지수 용량 · 상관 패턴 = 준안정 혼합(검색 실패)**.
  그 위에 우리 §17-18 = **H_min = log₂ k_eff** · **ASK = 에너지 갭** · conformal 보정 · VOI 순서.
- **Track B 축퇴-단서 논문**(`PAPER_TRACKB_DEGENERATE_CUE_MECHANISM_2026_07_19`): 같은-조항 선행 항목이
  준-축퇴 패턴을 이뤄 결합 odds 를 βΔE − log g 로 깎는다. 행동 계단 **k\*=2** 재현 · B2 knockout 인과 확인 ·
  단 **readout-희석(P3)은 기각**돼 locus = target-행 표상 구축으로 정제됐다.
- ⚠**기각된 판본**(`RETRIEVAL_FORMAT_THEORY_2026_08_19` §2-A): *"문맥 질량↑ → 간섭↑ → 검색↓"*.
  실측 부호가 반대였다(창 4→60 에서 exact 0.08→0.39). **이 프로브는 그 판본을 되살리지 않는다** —
  여기서 재는 것은 문맥 질량이 아니라 **후보 사이의 분리 Δ** 다(축퇴 판본).

## 그래서 이 프로브가 새로 재는 것
결정점마다 **닫힌 후보 집합** C 를 세우고, 각 후보를 teacher-forced 로 채점해 모델 자신의 에너지 지형을 읽는다:

    lp(c) = log p( {"value": "<c>"} | 프롬프트 )        ← 생성 안 함 · 채점만
    p     = softmax(lp)                                  ← 이 자리의 검색 분포
    Δ     = lp(1위) − lp(2위)                            ← **Hopfield 분리**(패턴 간 간격)
    H     = −Σ p log₂ p  ·  k_eff = 2^H                  ← 유효 후보 수
    H_min = −log₂ max p                                  ← 최소엔트로피(우리 §17-18 축)

Hopfield 예측: Δ 가 작으면 준안정 혼합 → **이웃 멤버로 미끄러진다**. 그 예측을 x423 의 실제 적중과
맞춰 **사전 예측력**(AUROC·구간별 적중)을 낸다.

## 사전 고정 해석 (결과 보기 전)
- `p̂(gold)` 가 x423 실측 적중을 **단조**로 예측하고 AUROC ≥ 0.75 ⇒ **미리 잴 수 있다**. 그러면 H_min 은
  ASK/라우팅 문턱으로 쓸 수 있는 gold-free 신호다(C87 §17-18 이 예고한 자리).
- AUROC ≈ 0.5 ⇒ 에너지 간격은 이 축을 예측 못 한다 ⇒ **홉필드 축퇴 판본도 이 자리에서는 기각**.
- 어휘 거리(모델 호출 0)가 p̂ 만큼 예측하면 ⇒ 비싼 측정이 불필요하다(어휘 근접이 곧 축퇴).
- ⚠어느 경우에도 이것은 **관측**이다. 처방은 [[62]] ③ 순서를 따로 밟는다.

사용: py -3 x425_hopfield_margin.py [--port 8141] [--max-cases 40] [--arms A_schema,B_policy,D_live]
"""
import argparse
import collections
import io
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_forensic as F  # noqa: E402
import x395_compliance_iso as X  # noqa: E402
import x397_pvi_channel as P  # noqa: E402
import x423_choice_isolation as I  # noqa: E402


def candidate_pool():
    """(task, tool, arg) → 후보 집합. **두 런에서 실제로 관측된 값의 합집합**(gold + 우리가 낸 값).

    gold 를 후보에 넣는 것은 채점이 아니라 *분모*다 — 정답이 후보에 없으면 분리 Δ 라는 물음 자체가
    성립하지 않는다. 후보를 우리가 지어내지 않는다는 점이 중요하다([[23]]): 전부 궤적·채점표에
    **실재한 문자열**이다.
    """
    pool = collections.defaultdict(set)
    for tag in I.RUNS:
        for sim in F.sims(tag, I.SUF):
            d = F.mutation_diff(sim)
            gold_by = {}
            for g in d["gold"]:
                gold_by.setdefault(g["name"], []).append(g)
            for src in (d["wrongarg"] + d["done"]):
                if src["name"] not in gold_by:
                    continue
                for k, v in (src["args"] or {}).items():
                    if I.is_choice(v):
                        pool[(F.task_id(sim), src["name"], k)].add(str(v))
            for g in d["gold"]:
                for k, v in (g["args"] or {}).items():
                    if I.is_choice(v):
                        pool[(F.task_id(sim), g["name"], k)].add(str(v))
    return pool


def lev(a, b):
    """정규화 편집거리 — 모델 호출 0 의 어휘 분리 대용치."""
    a, b = a.lower(), b.lower()
    if a == b:
        return 0.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / float(max(len(a), len(b)) or 1)


def softmax(xs):
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    s = sum(e) or 1.0
    return [x / s for x in e]


def auroc(scores, labels):
    """정답 1 · 오답 0 에 대한 AUROC(동점 0.5 처리). 표본이 한쪽뿐이면 None."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    tot = 0.0
    for p in pos:
        for n in neg:
            tot += 1.0 if p > n else (0.5 if p == n else 0.0)
    return tot / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8141)
    ap.add_argument("--max-cases", type=int, default=40)
    ap.add_argument("--arms", default="A_schema,B_policy,D_live")
    ap.add_argument("--norm", action="store_true", help="길이 정규화 logp 를 쓴다")
    a = ap.parse_args()
    arms = [x for x in a.arms.split(",") if x]

    pool = candidate_pool()
    cs = I.cases(a.max_cases)
    print("=" * 104)
    print("x425 · 에너지 간격 · 사례 %d · 팔 %s · 포트 %d" % (len(cs), ",".join(arms), a.port))
    print("후보 집합 = 두 런에서 관측된 값의 합집합(gold 포함)")
    print("=" * 104)

    out = []
    for ci, c in enumerate(cs):
        cands = sorted(pool.get((c["task"], c["tool"], c["arg"])) or set())
        if c["gold"] not in cands:
            cands.append(c["gold"])
        if len(cands) < 2:
            continue
        near = min((lev(c["gold"], x) for x in cands if x != c["gold"]), default=1.0)
        bodies = I.build_arms(c)
        for arm in arms:
            pre = P.TPL % (I.SYS, bodies[arm])
            lps = []
            try:
                for cd in cands:
                    lp, n = P.score_suffix(a.port, pre, '{"value": "%s"}' % cd)
                    lps.append(lp / max(1, n) if a.norm else lp)
            except Exception as e:
                print("  ERROR %s %s: %r" % (c["task"], arm, e))
                continue
            p = softmax(lps)
            order = sorted(range(len(cands)), key=lambda i: -lps[i])
            gi = cands.index(c["gold"])
            H = -sum(x * math.log(x, 2) for x in p if x > 0)
            rec = {"task": c["task"], "trial": c["trial"], "tool": c["tool"], "arg": c["arg"],
                   "arm": arm, "n_cand": len(cands), "gold": c["gold"], "live": c["live"],
                   "top1": cands[order[0]], "top1_is_gold": order[0] == gi,
                   "top1_is_live": cands[order[0]] == c["live"],
                   "p_gold": p[gi], "margin": lps[order[0]] - lps[order[1]],
                   "gold_margin": lps[gi] - max(lps[i] for i in range(len(lps)) if i != gi),
                   "H": H, "k_eff": 2 ** H, "H_min": -math.log(max(p), 2),
                   "lex_nearest": near}
            out.append(rec)
        if (ci + 1) % 5 == 0:
            print("  ... %d/%d 사례" % (ci + 1, len(cs)))

    print("\n## 팔별 — 에너지 top1 이 gold 인가 (생성 없이, 채점만)")
    print("%-10s %5s %9s %9s %8s %8s %8s" % ("arm", "n", "TOP1=GOLD", "TOP1=LIVE", "p_gold", "k_eff", "H_min"))
    for arm in arms:
        r = [x for x in out if x["arm"] == arm]
        if not r:
            continue
        n = float(len(r))
        print("%-10s %5d %9.3f %9.3f %8.3f %8.2f %8.2f"
              % (arm, len(r), sum(x["top1_is_gold"] for x in r) / n,
                 sum(x["top1_is_live"] for x in r) / n, sum(x["p_gold"] for x in r) / n,
                 sum(x["k_eff"] for x in r) / n, sum(x["H_min"] for x in r) / n))

    # x423 실측과 붙인다 — 사전 예측력
    p423 = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026",
                        "x423_choice_isolation.json")
    if os.path.exists(p423):
        with io.open(p423, encoding="utf-8") as f:
            emp = json.load(f)
        hit = collections.defaultdict(list)
        for e in emp:
            hit[(e["task"], e["tool"], e["arg"], e["arm"])].append(bool(e["hit"]))
        rows = []
        for x in out:
            k = (x["task"], x["tool"], x["arg"], x["arm"])
            if k in hit and hit[k]:
                rows.append((x, sum(hit[k]) / float(len(hit[k]))))
        if rows:
            print("\n## 사전 예측력 — 에너지 신호가 x423 실측 적중을 맞히나 (사례 %d)" % len(rows))
            for name, key, sgn in (("p_gold", "p_gold", 1), ("gold_margin", "gold_margin", 1),
                                   ("H_min(낮을수록 확신)", "H_min", -1),
                                   ("어휘 최근접(모델 0)", "lex_nearest", 1)):
                sc = [sgn * x[key] for x, _ in rows]
                lb = [r >= 0.5 for _, r in rows]
                au = auroc(sc, lb)
                print("   %-24s AUROC %s" % (name, "n/a" if au is None else "%.3f" % au))
            print("\n   %-14s %6s %8s" % ("p_gold 구간", "사례", "실측적중"))
            for lo, hi in ((0.0, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 1.01)):
                b = [r for x, r in rows if lo <= x["p_gold"] < hi]
                if b:
                    print("   %-14s %6d %8.2f" % ("[%.1f,%.1f)" % (lo, hi), len(b), sum(b) / len(b)))
    else:
        print("\n(x423 산출물이 없다 — 먼저 x423 을 돌리면 사전 예측력까지 낸다)")

    p = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "x425_hopfield_margin.json")
    with io.open(os.path.abspath(p), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n→ %s" % os.path.abspath(p))
    return 0


if __name__ == "__main__":
    sys.exit(main())
