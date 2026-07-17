#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""★arm 페어 비교 (중단·재개 안전) — **양 arm에 다 있는 trial만** 비교한다 (2026-07-18·사용자 지시).

**왜** (`ARM_ALIGNMENT_RESUME_DESIGN_2026_07_18`): 2026-07-18 nt=20 런서 arm이 **서로 다른 진도**로
갈렸다(dreq2 12/20 · ctl2 8/20). 그 시점 평균(0.50 vs 0.38)은 **레버 효과가 아니라 완주-순서 편향**일 수 있다:
  · **완주 순서가 무작위가 아니다** — 빨리 끝난 sim이 먼저 집계되는데 빨리 끝나는 sim은 대개 일찍 실패했거나
    짧게 성공한 것이다.
  · **N이 다르면 부분집합이 다르다** — 두 arm이 *다른 문제들*을 푼 뒤 평균을 견준 셈.
⇒ **어느 시점에 멈추든 유효한 비교의 유일한 방법 = 양쪽에 다 있는 `trial`만 짝지어 비교.**
   tau2가 `(trial, task_id, seed)`로 재개하므로 **양 arm이 같은 seed면 키가 그대로 페어 키**가 된다.

⚠️**[[08]] 잔여 편향(이걸로도 안 없어짐)**: `run_with_retry`는 실패 시도를 **버리고 재실행**한다
(`batch.py`·성공분만 `save_fn`). 컨텍스트 초과로 버려진 궤적은 **저장 자체가 안 된다** → 페어를 맞춰도
**"길어서 죽은 궤적"이 양 arm서 다른 비율로 삭제**돼 있다. ⇒ **재시도 수를 로그서 세어 함께 보고**할 것
(`--retry-log`). 페어링은 *진도 차이*를 없애지 *삭제 편향*을 없애지 못한다.

Run:
  python3 bank_paired_arms.py bank_dreq2_nt20_20260718 bank_ctl2_nt20_20260718 \
      --retry-log /home/woori/scratch/bank_{arm}_nt20_20260718.log
"""
import argparse
import gzip
import json
import os
import re
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SIMDIR = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "sim_results")


def load(tag):
    """영속 gz 우선, 없으면 리모트 save_dir의 results.json(진행 중 런)."""
    p = os.path.join(SIMDIR, f"{tag}.results.json.gz")
    if os.path.exists(p):
        with gzip.open(p, "rt", encoding="utf-8") as f:
            return json.load(f)["simulations"]
    p2 = os.path.join("/home/woori/scratch/tau2-bench/data/simulations", tag, "results.json")
    with open(p2, encoding="utf-8") as f:
        return json.load(f)["simulations"]


def key_of(s):
    """페어 키 = tau2 `done_runs` 키와 동형 (trial, task_id, seed)."""
    return (s.get("trial"), s.get("task_id"), s.get("seed"))


def reward_of(s, infra_as_zero=False):
    """★`--max_retries 0` 런에서는 컨텍스트 초과 sim이 `INFRASTRUCTURE_ERROR`로 남는데
    `reward_info`가 **없다**(`messages=[]`) → 그냥 두면 **평균서 제외** = 삭제 편향이 '결측'으로 형태만 바뀜.
    사용자 지시(2026-07-18): *"32K 넘으면 재시도 하지 말고 **fail 처리**"* ⇒ `infra_as_zero`면 **0으로 센다.**
    ⚠️단 이건 **채점 판단**이므로 반드시 **개수를 함께 보고**한다(아래 main의 `infra=` 출력)."""
    ri = s.get("reward_info") or {}
    r = ri.get("reward")
    if r is None and infra_as_zero:
        return 0.0
    return r


def term_of(s):
    ti = s.get("termination_reason")
    return str(ti) if ti else "?"


def retries_from_log(path):
    if not path or not os.path.exists(path):
        return None
    txt = open(path, encoding="utf-8", errors="replace").read()
    reasons = re.findall(r"Retry \d/\d for task \S+: ([A-Za-z_.]+)", txt)
    return {"retries": len(reasons), "reasons": dict(
        (r, reasons.count(r)) for r in set(reasons))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm_a")
    ap.add_argument("arm_b")
    ap.add_argument("--retry-log", default=None,
                    help="'{arm}' 치환 패턴. 예: /home/woori/scratch/bank_{arm}_nt20_20260718.log")
    ap.add_argument("--infra-as-zero", action="store_true",
                    help="★`--max_retries 0` 런용: reward 없는 sim(=INFRASTRUCTURE_ERROR·컨텍스트 초과)을 "
                         "**fail(0)**로 센다(사용자 지시 2026-07-18). 미지정=제외(구 동작).")
    a = ap.parse_args()

    out = {}
    for tag in (a.arm_a, a.arm_b):
        sims = load(tag)
        d, infra = {}, 0
        for s in sims:
            if (s.get("reward_info") or {}).get("reward") is None:
                infra += 1               # reward 없음 = infra_error(초과) — 세는 건 플래그와 무관
            r = reward_of(s, a.infra_as_zero)
            if r is None:
                continue                 # 미지정 시에만 도달 = 구 동작(제외)
            d[key_of(s)] = {"reward": r, "term": term_of(s)}
        out[tag] = d
        if a.infra_as_zero:
            print(f"{tag}: 채점 대상 {len(d)}건 (그중 **무보상→0(fail)으로 센 것: {infra}건**)")
        else:
            print(f"{tag}: 채점 대상 {len(d)}건" + (f"  ⚠️무보상 {infra}건 **제외됨**" if infra else ""))
            if infra:
                print(f"   ⚠️`--max_retries 0` 런이면 `--infra-as-zero`를 써라 — 안 그러면 "
                      f"삭제 편향이 '결측'으로 형태만 바뀐다.")

    A, B = out[a.arm_a], out[a.arm_b]
    paired = sorted(set(A) & set(B), key=lambda k: (str(k[1]), k[0] if k[0] is not None else -1))
    only_a, only_b = set(A) - set(B), set(B) - set(A)

    print(f"\n★페어(양쪽 다 완주) = {len(paired)}  ·  {a.arm_a}만 {len(only_a)}  ·  {a.arm_b}만 {len(only_b)}")
    if not paired:
        print("페어 0 — 비교 불가(둘 다 같은 trial을 끝낼 때까지 대기).")
        return

    ra = [A[k]["reward"] for k in paired]
    rb = [B[k]["reward"] for k in paired]
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    print(f"\n=== 페어 비교 (N={len(paired)})")
    print(f"  {a.arm_a:34s} mean={ma:.3f}  pass^1={sum(r >= 1 for r in ra)}/{len(ra)}")
    print(f"  {a.arm_b:34s} mean={mb:.3f}  pass^1={sum(r >= 1 for r in rb)}/{len(rb)}")
    print(f"  Δ(mean) = {ma - mb:+.3f}")

    # trial별 짝 (부호 일치/불일치 = 진짜 차이인지 노이즈인지 눈으로)
    win = sum(A[k]["reward"] > B[k]["reward"] for k in paired)
    lose = sum(A[k]["reward"] < B[k]["reward"] for k in paired)
    tie = len(paired) - win - lose
    print(f"  per-trial: {a.arm_a} 우세 {win} · 동률 {tie} · {a.arm_b} 우세 {lose}")

    # ★[[08]] 종료사유 교차표 — infra/crash를 pass 차이로 오인하지 말 것
    print("\n=== 종료사유 (페어 내)")
    tt = defaultdict(lambda: [0, 0])
    for k in paired:
        tt[A[k]["term"]][0] += 1
        tt[B[k]["term"]][1] += 1
    for t, (x, y) in sorted(tt.items(), key=lambda kv: -sum(kv[1])):
        print(f"  {t:32s} {a.arm_a}={x:3d}  {a.arm_b}={y:3d}")

    # ★삭제 편향 공변량 — 페어링으로 안 없어진다
    if a.retry_log:
        print("\n=== ⚠️재시도(=버려진 궤적) — 페어링으로 제거 안 됨")
        for tag, arm in ((a.arm_a, "dreq2"), (a.arm_b, "ctl2")):
            for cand in (a.retry_log.replace("{arm}", arm),
                         a.retry_log.replace("{arm}", tag)):
                info = retries_from_log(cand)
                if info:
                    print(f"  {tag:34s} 재시도 {info['retries']}회  {info['reasons']}")
                    break
            else:
                print(f"  {tag:34s} (로그 없음)")
        print("  → 재시도 수가 arm 간에 다르면 **나쁜 궤적을 더 많이 버린 arm이 유리**하다. Δ 해석 시 필수 고려.")

    print("\n⚠️ per-trial 짝이 있어도 **같은 궤적이 아니다**(temp>0 샘플링). **같은 seed의 같은 문제**를 "
          "풀었다는 뜻이다 — tau2가 배치 seed서 trial별 seed를 파생하고 양 arm이 그걸 공유한다(실측 확인).")
    print("⚠️ 동률이 많다고 '레버 무효'가 아니다 — **seed(문제 인스턴스)가 결과를 지배**할 수 있다. "
          "n이 작으면 pass^1 점추정으로 결론 금지([[08]]).")


if __name__ == "__main__":
    main()
