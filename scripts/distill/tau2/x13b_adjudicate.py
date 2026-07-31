# -*- coding: utf-8 -*-
"""X13 판정 — 대본-유저 다중턴 본런 arm 비교 (2026-07-31).

설계 = `MT_PROBE_DESIGN_2026_07_30.md`(rev4). 판정 규율(설계서에 사전 고정):
  · **태스크 pass 채점 금지** — 기전 지표만 본다
  · **ASK를 페널티로 채점 금지** — 대본 유저는 답하지 않으므로 ASK arm이 구조적으로 불리하다
  · **지평 초과율·대본 소비율은 1차 지표**(각주 아님)
  · **CI 중첩으로 유의를 판정하지 말 것**(C242) → 짝지은 **McNemar**와 **Fisher exact**로 간다
  · 케이스 수와 **궤적 수**를 둘 다 보고(X8의 "27케이스가 궤적 4개" 교훈)

★make-or-break(사전 고정): `A_PROMPT`의 봉투 준수가 합성 프로브 천장(32/32=100%)에서 내려와야
  이 조건이 유효하다. 내려오지 않으면 판정 금지.

용법: py -3 x13b_adjudicate.py <rows.jsonl|.gz>
"""
import argparse
import gzip
import json
import math
import sys
from collections import Counter, defaultdict


def _open(p):
    return gzip.open(p, "rt", encoding="utf-8") if p.endswith(".gz") else open(p, encoding="utf-8")


def mcnemar(b, c):
    """짝지은 이항 검정(정확). b,c = 불일치 쌍의 양방향 개수."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n) * 2
    return min(1.0, p)


def fisher(a, b, c, d):
    """2x2 Fisher exact (양측·근사 없이 합산)."""
    def logc(n, k):
        return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
    n = a + b + c + d
    r1, c1 = a + b, a + c
    p0 = math.exp(logc(r1, a) + logc(n - r1, c1 - a) - logc(n, c1))
    tot = 0.0
    lo, hi = max(0, c1 - (n - r1)), min(r1, c1)
    for x in range(lo, hi + 1):
        px = math.exp(logc(r1, x) + logc(n - r1, c1 - x) - logc(n, c1))
        if px <= p0 * (1 + 1e-9):
            tot += px
    return min(1.0, tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rows")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    rows = [json.loads(l) for l in _open(args.rows) if l.strip()]
    ok = [r for r in rows if "error" not in r]
    arms = sorted({r["arm"] for r in ok})
    cases = {r["case"] for r in ok}
    tasks = {r["task_id"] for r in ok}
    print("행 %d · arm %s · 케이스 %d · **서로 다른 task %d**"
          % (len(ok), arms, len(cases), len(tasks)))
    print("⚠케이스 %d개는 task %d개에서 나온다 — 독립 표본이 아니다([[08]])." % (len(cases), len(tasks)))

    print("\n" + "=" * 86)
    print("arm 요약 (봉투는 턴 단위 · 대본/지평은 런 단위)")
    print("=" * 86)
    print("%-13s %7s %7s %8s %8s %7s %7s %7s %7s" %
          ("arm", "봉투율", "무위반", "대본소비", "지평초과", "호출/런", "오류/런", "regen", "드롭의심"))
    per = {}
    for a in arms:
        rs = [r for r in ok if r["arm"] == a]
        turns = sum(r["turns"] for r in rs)
        env = sum(r["envelopes"] for r in rs)
        envok = sum(r["envelope_ok"] for r in rs)
        per[a] = dict(
            rs=rs, turns=turns, env=env, envok=envok,
            script=sum(r["script_rate"] for r in rs) / len(rs),
            horizon=sum(r["horizon_hit"] for r in rs) / len(rs),
            calls=sum(r["tool_calls"] for r in rs) / len(rs),
            errs=sum(r["tool_errors"] for r in rs) / len(rs),
            regen=sum(r["regens"] for r in rs),
            drop=sum(r.get("drop_suspect_turns", 0) for r in rs),
            multi=sum(r.get("multiact_turns", 0) for r in rs),
            maxcalls=max((r.get("max_calls_in_turn", 0) for r in rs), default=0),
            gen=sum(r.get("gen_tokens", 0) for r in rs) / len(rs))
        p = per[a]
        print("%-13s %6.0f%% %6.0f%% %7.0f%% %7.0f%% %7.1f %7.1f %7d %7d"
              % (a, 100 * env / turns, 100 * envok / turns, 100 * p["script"],
                 100 * p["horizon"], p["calls"], p["errs"], p["regen"], p["drop"]))

    print("\nmulti-act(한 턴 다중 호출) 턴 수 · 최대 호출/턴 · 평균 생성토큰:")
    for a in arms:
        p = per[a]
        print("  %-13s multiact %3d · max %2d · gen %.0f" % (a, p["multi"], p["maxcalls"], p["gen"]))

    # ── make-or-break ──
    base = per.get("A_PROMPT")
    if base:
        rate = 100 * base["env"] / base["turns"]
        print("\n★make-or-break: A_PROMPT 봉투율 %.0f%% (합성 프로브 천장 100%%)" % rate)
        print("   ⇒ %s" % ("**충족** — 조건이 결손을 재현한다. 판정 허용."
                           if rate < 95 else "**미충족** — 천장이므로 판정 금지"))

    # ── 짝지은 비교: A vs 각 arm (같은 case+seed에서 봉투 준수 턴 비율) ──
    print("\n" + "=" * 86)
    print("짝지은 비교 (같은 케이스·시드 · McNemar) — 봉투가 **한 턴이라도** 나온 런 기준")
    print("=" * 86)
    key = lambda r: (r["case"], r["seed"])
    idx = defaultdict(dict)
    for r in ok:
        idx[key(r)][r["arm"]] = r
    for a in arms:
        if a == "A_PROMPT" or "A_PROMPT" not in arms:
            continue
        b = c = 0
        for k, d in idx.items():
            if "A_PROMPT" not in d or a not in d:
                continue
            x = d["A_PROMPT"]["envelopes"] > 0
            y = d[a]["envelopes"] > 0
            if x and not y:
                b += 1
            elif y and not x:
                c += 1
        print("  A_PROMPT vs %-13s A만 %2d · %s만 %2d · McNemar p=%.4f"
              % (a, b, a, c, mcnemar(b, c)))

    # ── ★런 단위 짝지은 부호검정 (턴은 독립이 아니다 — 의사반복 회피) ──
    print("\n" + "=" * 86)
    print("★런 단위 짝지은 비교 — 같은 케이스·시드의 **봉투율 차**(부호검정)")
    print("  ⚠턴 단위 Fisher는 한 런의 턴들을 독립으로 취급해 p를 과대평가한다 ⇒ 쓰지 않는다")
    print("=" * 86)
    for a in arms:
        if a == "A_PROMPT" or "A_PROMPT" not in arms:
            continue
        up = dn = tie = 0
        deltas = []
        for k, d in idx.items():
            if "A_PROMPT" not in d or a not in d:
                continue
            ra = d["A_PROMPT"]["envelopes"] / max(1, d["A_PROMPT"]["turns"])
            rb = d[a]["envelopes"] / max(1, d[a]["turns"])
            deltas.append(rb - ra)
            if rb > ra + 1e-9:
                up += 1
            elif ra > rb + 1e-9:
                dn += 1
            else:
                tie += 1
        med = sorted(deltas)[len(deltas) // 2] if deltas else 0
        print("  A_PROMPT → %-13s 상승 %2d · 하락 %2d · 동률 %2d · 중앙 Δ %+.2f · 부호검정 p=%.2e"
              % (a, up, dn, tie, med, mcnemar(up, dn)))

    # ── 드롭-보정 상한 (C248: 호출 뒤에 쓴 봉투는 파서가 버린다) ──
    print("\n★드롭 보정 — 봉투율의 **하한 ~ 상한**(드롭 의심 턴을 전부 봉투로 세면 상한)")
    for a in arms:
        p2 = per[a]
        lo = 100.0 * p2["env"] / p2["turns"]
        hi = 100.0 * min(p2["turns"], p2["env"] + p2["drop"]) / p2["turns"]
        print("  %-13s %3.0f%% ~ %3.0f%%   (드롭 의심 %d턴)" % (a, lo, hi, p2["drop"]))
    print("  ⇒ **상한으로 봐도 서열이 유지되는지**가 판정의 조건이다.")

    # ── 위반 내역 ──
    print("\n위반 내역(§1d 축소판):")
    for a in arms:
        v = Counter()
        for r in per[a]["rs"]:
            for k, n in (r.get("viol_counts") or {}).items():
                v[k] += n
        print("  %-13s %s" % (a, dict(v.most_common())))

    print("\n⚠판정 규율: 태스크 pass 미채점 · ASK 페널티 없음 · 표본 상관(케이스↔task) 감안.")


if __name__ == "__main__":
    main()
