# -*- coding: utf-8 -*-
"""Y1 수확 — `results.json` 직독으로 nt=2 편차 폭(판정 임계) 확정 (2026-07-31).

Y1의 목적은 점수가 아니라 **동일 스택 nt=2 flip률**이다(E-MFIX: 시나리오 고정이 기각됐으므로
남은 수단). 지금까지는 진행 로그의 Status 라인에서 **역산**했으나(`y1_partial_reconstruct.py`),
러너가 `results.json`을 증분 기록하므로 **직독이 가능하고 그쪽이 정본**이다.

역산 대비 직독의 이점:
  · 귀속 모호(한 블록에서 2건 동시 종료) **없음**
  · **종료 사유**(termination_reason)를 함께 볼 수 있다 → conc 2 교란 확인([[08]])
  · trial 인덱스가 명시적이라 쌍 구성이 확실하다

⚠[[08]] 규율: avg(=pass율)를 스택 성능으로 읽지 말 것. Y1은 **자를 만드는 런**이다.
용법: py -3 y1_harvest.py <results.json|.gz> [--json out.json]
"""
import argparse
import gzip
import json
import sys
from collections import Counter, defaultdict


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def reward_of(sim):
    ri = sim.get("reward_info") or {}
    r = ri.get("reward")
    return None if r is None else (1 if r >= 1 else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    d = load(args.results)
    sims = d.get("simulations") or []
    print("sim %d개" % len(sims))

    by_task = defaultdict(list)
    term = Counter()
    for s in sims:
        r = reward_of(s)
        ti = s.get("trial")
        by_task[s.get("task_id")].append({
            "trial": ti, "reward": r, "dur": round(s.get("duration") or 0),
            "term": (s.get("termination_reason") or (s.get("info") or {}).get("termination_reason")),
            "id": str(s.get("id"))[:10],
        })
        term[str((s.get("termination_reason")
                  or (s.get("info") or {}).get("termination_reason")))] += 1

    print("\n=== 종료 사유 분포([[08]]: 노이즈·크래시 배제 확인용) ===")
    for k, v in term.most_common():
        print("  %-28s %d" % (k, v))

    pairs = {t: v for t, v in by_task.items() if len(v) >= 2}
    singles = {t: v for t, v in by_task.items() if len(v) == 1}
    print("\n" + "=" * 74)
    print("★nt=2 완결 쌍 %d개 = 편차 폭의 유효 표본 (단일 trial %d개는 표본 아님)"
          % (len(pairs), len(singles)))
    print("=" * 74)
    flip = agree_p = agree_f = unk = 0
    for t in sorted(pairs):
        v = sorted(pairs[t], key=lambda x: (x["trial"] is None, x["trial"]))
        rs = [x["reward"] for x in v]
        ds = [x["dur"] for x in v]
        if None in rs:
            verdict, unk = "?(reward 없음)", unk + 1
        elif len(set(rs)) > 1:
            verdict, flip = "★FLIP", flip + 1
        elif rs[0] == 1:
            verdict, agree_p = "일치(둘 다 PASS)", agree_p + 1
        else:
            verdict, agree_f = "일치(둘 다 fail)", agree_f + 1
        print("  %-10s %s  %-18s dur=%s" % (t, ["PASS" if x == 1 else ("fail" if x == 0 else "?")
                                                for x in rs], verdict, ds))
    n = flip + agree_p + agree_f
    print("\n판정: FLIP %d · 일치 %d(PASS %d · fail %d) · 판정불가 %d"
          % (flip, agree_p + agree_f, agree_p, agree_f, unk))
    if n:
        print("⇒ **동일 스택 nt=2 flip률 = %d/%d = %.0f%%**" % (flip, n, 100.0 * flip / n))
        print("   = 이후 모든 arm 비교의 **판정 임계**. 이보다 작은 차이는 노이즈로 기각한다.")

    # trial별 점수(= 과거 n=1 day 런과 같은 단위)
    print("\n=== trial별 pass (과거 day 런과 같은 단위로 비교하려면 이 값을 쓸 것) ===")
    per_trial = defaultdict(lambda: [0, 0])
    for t, v in by_task.items():
        for x in v:
            k = x["trial"]
            if x["reward"] is None:
                continue
            per_trial[k][0] += x["reward"]
            per_trial[k][1] += 1
    for k in sorted(per_trial, key=lambda z: (z is None, z)):
        p, tot = per_trial[k]
        print("  trial %s: %d/%d = %.2f" % (k, p, tot, p / tot if tot else 0))
    print("⚠avg(전체)는 trial 0과 1이 섞인 값이라 day 런과 직접 비교 금지.")

    if args.json:
        json.dump({"pairs": {t: pairs[t] for t in pairs}, "flip": flip,
                   "agree_pass": agree_p, "agree_fail": agree_f, "unknown": unk,
                   "termination": dict(term)},
                  open(args.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("\n[saved] %s" % args.json)


if __name__ == "__main__":
    main()
