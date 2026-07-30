#!/usr/bin/env python3
"""X8-(c) triage 포렌식 — per-case 교차표 + 3분기 귀속 (2026-07-30).

[[08]] 규율: arm 집계에서 결론 직행 금지. 이 스크립트가 답해야 하는 것:

  Q1 (★아키텍처 착수 게이트) **act 수(1 vs ≥2)가 실패를 예측하나?**
     — multi-act 71%는 *유병률*이다. 해로움이 아니면 multi-act 처리 아키텍처는 비-병목을 겨냥한다.
  Q2 (⑴선언-강제)  arm A→C 에서 **어느 발화가 고쳐졌나/깨졌나** (집계 차 0.25→0.27 뒤의 per-case)
  Q3 (⑵부하)       arm A→Actx(문맥 추가)에서 해소되는 사례가 있나 ([[18]] B_fullctx 근사)
  Q4 (⑶잔여)       세 arm 모두 실패하는 사례 = 능력/경계 후보 → learn 표적
  Q5 (오류 구조)   act별 혼동(어느 라벨을 놓치고 어느 것을 날조하나) · slot 오류 유형

용법: py -3 x8c_triage_forensic.py [rows.jsonl] [--json out.json]
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_SIM = os.path.abspath(os.path.join(_HERE, "..", "..", "..",
                                    "reports", "facet_rft_2026", "sim_results"))


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rows", nargs="?", default=os.path.join(_SIM, "x8_triage_rows.jsonl"))
    ap.add_argument("--json")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    rows = [json.loads(l) for l in open(args.rows, encoding="utf-8")]
    rows = [r for r in rows if "error" not in r]
    samp = {json.loads(l)["sample_id"]: json.loads(l)
            for l in open(os.path.join(_SIM, "x8_sample_utterances.jsonl"), encoding="utf-8")}

    # seed 축 붕괴 확인: temperature 0 이면 seed 무의미 → 중복 계상 방지
    by_as = defaultdict(list)
    for r in rows:
        by_as[(r["arm"], r["sample_id"])].append(r)
    dup_identical = sum(1 for v in by_as.values()
                        if len({(tuple(x["pred_acts"]), tuple(x["pred_slots"])) for x in v}) == 1)
    print(f"행 {len(rows)} · (arm,sample) 조합 {len(by_as)} · "
          f"시드 간 예측 완전동일 {dup_identical}/{len(by_as)}")
    if dup_identical == len(by_as):
        print("⚠**시드가 전부 동일** = temperature 0 하 greedy 결정론 ⇒ 시드는 변동 정보 0. "
              "이하 분석은 (arm,sample) **유일 예측 1개**로 접는다(중복 계상 금지).")
    print()

    # arm × sample 단일화
    cell = {k: v[0] for k, v in by_as.items()}
    arms = sorted({a for a, _ in cell})
    ids = sorted({s for _, s in cell})

    # ── Q1: act 수 vs 실패 (★착수 게이트) ─────────────────────────────────
    print("=" * 74)
    print("Q1 ★착수 게이트 — act 수(1 vs ≥2)가 실패를 예측하나")
    print("=" * 74)
    q1 = {}
    for arm in arms:
        tab = {"single": [0, 0], "multi": [0, 0]}      # [correct, n]
        for sid in ids:
            r = cell.get((arm, sid))
            if not r:
                continue
            k = "single" if len(r["gold_acts"]) == 1 else "multi"
            tab[k][1] += 1
            tab[k][0] += bool(r["acts_exact"])
        s_ok, s_n = tab["single"]
        m_ok, m_n = tab["multi"]
        sl, sh = wilson(s_ok, s_n)
        ml, mh = wilson(m_ok, m_n)
        overlap = not (sl > mh or ml > sh)
        q1[arm] = {"single": tab["single"], "multi": tab["multi"],
                   "single_ci": [sl, sh], "multi_ci": [ml, mh], "ci_overlap": overlap}
        print(f"{arm:5s} 단일-act {s_ok}/{s_n} = {s_ok / s_n if s_n else 0:.2f} "
              f"[{sl:.2f},{sh:.2f}]   다중-act {m_ok}/{m_n} = {m_ok / m_n if m_n else 0:.2f} "
              f"[{ml:.2f},{mh:.2f}]   CI 중첩={'예' if overlap else '아니오'}")
    print()
    print("판정 규칙: 다중-act 정확도가 단일보다 **CI 비중첩으로 낮으면** multi-act가 실패를")
    print("유발한다(게이트 통과). 중첩이면 유병률일 뿐이므로 아키텍처 착수 금지([[08]]).")
    print()

    # ── Q2/Q3/Q4: arm 간 per-case 교차표 ──────────────────────────────────
    print("=" * 74)
    print("Q2/Q3/Q4 — per-case arm 교차표 (acts_exact 기준)")
    print("=" * 74)
    pat = Counter()
    per_pat = defaultdict(list)
    for sid in ids:
        key = tuple("O" if (cell.get((a, sid)) or {}).get("acts_exact") else "x" for a in arms)
        pat[key] += 1
        per_pat[key].append(sid)
    print("패턴(" + "/".join(arms) + ")  건수  sample_id")
    for k, n in sorted(pat.items(), key=lambda x: -x[1]):
        print(f"  {'/'.join(k):16s} {n:3d}  {' '.join(per_pat[k][:14])}")
    allx = per_pat.get(tuple("x" for _ in arms), [])
    allo = per_pat.get(tuple("O" for _ in arms), [])
    print()
    print(f"⑴선언-강제로 **고쳐진** 사례(A=x,C=O): "
          f"{[s for s in ids if not cell[('A', s)]['acts_exact'] and cell[('C', s)]['acts_exact']]}")
    print(f"⑴선언-강제로 **깨진** 사례(A=O,C=x): "
          f"{[s for s in ids if cell[('A', s)]['acts_exact'] and not cell[('C', s)]['acts_exact']]}")
    if "Actx" in arms:
        print(f"⑵문맥으로 **해소된** 사례(A=x,Actx=O): "
              f"{[s for s in ids if not cell[('A', s)]['acts_exact'] and cell[('Actx', s)]['acts_exact']]}")
        print(f"⑵문맥으로 **깨진** 사례(A=O,Actx=x): "
              f"{[s for s in ids if cell[('A', s)]['acts_exact'] and not cell[('Actx', s)]['acts_exact']]}")
    print(f"\n⑶전-arm 실패(능력/경계 후보) {len(allx)}건: {' '.join(allx)}")
    print(f"   전-arm 성공 {len(allo)}건: {' '.join(allo)}")
    print()

    # ── Q5: act 라벨별 혼동 ────────────────────────────────────────────────
    print("=" * 74)
    print("Q5 — act 라벨별 오류 구조 (arm A 기준·놓침 vs 날조)")
    print("=" * 74)
    miss, spur, gold_n, pred_n = Counter(), Counter(), Counter(), Counter()
    for sid in ids:
        r = cell.get(("A", sid))
        if not r:
            continue
        g, p = set(r["gold_acts"]), set(r["pred_acts"])
        for a in g:
            gold_n[a] += 1
        for a in p:
            pred_n[a] += 1
        for a in g - p:
            miss[a] += 1
        for a in p - g:
            spur[a] += 1
    print(f"{'act':10s} {'gold':>5s} {'pred':>5s} {'놓침':>5s} {'날조':>5s}  재현율")
    for a in sorted(gold_n, key=lambda x: -gold_n[x]):
        R = (gold_n[a] - miss[a]) / gold_n[a] if gold_n[a] else 0
        print(f"{a:10s} {gold_n[a]:5d} {pred_n[a]:5d} {miss[a]:5d} {spur[a]:5d}  {R:.2f}")
    only_spur = sorted(set(spur) - set(gold_n))
    if only_spur:
        print(f"gold에 없는데 예측된 라벨: {[(a, spur[a]) for a in only_spur]}")
    print()

    # ── slot 오류 구조 ─────────────────────────────────────────────────────
    print("=" * 74)
    print("slot 오류 구조 (arm 별)")
    print("=" * 74)
    for arm in arms:
        rs = [cell[(arm, s)] for s in ids if (arm, s) in cell]
        tp = sum(r["slots_tp"] for r in rs)
        fp = sum(r["slots_fp"] for r in rs)
        fn = sum(r["slots_fn"] for r in rs)
        fab = sum(r["slot_fabricated"] for r in rs)
        P = tp / (tp + fp) if tp + fp else 0
        R = tp / (tp + fn) if tp + fn else 0
        print(f"{arm:5s} TP={tp:3d} FP={fp:3d} FN={fn:3d} P={P:.2f} R={R:.2f} "
              f"F1={2 * P * R / (P + R) if P + R else 0:.2f}  날조(비-축자)={fab}")
    print("\n볼드-베이스라인(규약 §3): P=0.45 R=0.42 F1=0.43 — 위와 비교할 것")

    if args.json:
        json.dump({"q1": q1, "patterns": {"/".join(k): v for k, v in per_pat.items()},
                   "all_fail": allx, "all_pass": allo,
                   "act_miss": dict(miss), "act_spurious": dict(spur),
                   "seed_collapsed": dup_identical == len(by_as)},
                  open(args.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"\n[saved] {args.json}")


if __name__ == "__main__":
    main()
