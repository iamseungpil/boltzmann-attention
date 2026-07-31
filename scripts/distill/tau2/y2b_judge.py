# -*- coding: utf-8 -*-
"""Y2-B 사전등록 판정 (2026-07-31·무료·읽기 전용).

정본 = `Y2_DESIGN_2026_07_31.md` §6. **이 파일은 Y2-B가 도는 중에, 결과를 보기 전에 작성됐다**
(git 로그가 증인). 완주 후 수치를 보고 규칙을 고르는 일을 막는 게 목적이다.

사전등록 규칙(§6-1·§6-3) 축자:
  1차 ①  Y1 대비 **짝지은 McNemar** — 불일치쌍에서 (Y1 fail→Y2 pass) − (Y1 pass→Y2 fail) ≥ **+4**
          ∧ 부호검정 p<.05 = 개선 / 반대 방향 동일 기준 = 악화 / 사이 = 무결론
  1차 ②  **과행동 지표** O1(실효 write 수)·O3(중복 행동) 감소 = 개선 방향 · O4는 **태스크별**
  2차     총점 차분(중단 판정에만)
  부수     V7 재발행률 · 관찰자 무위반률 · forbid over-block 0

★짝짓기 단위 (설계가 한 줄로 못 박지 않은 부분 — **결과 보기 전에** 여기 고정한다):
  Y1은 nt=2, Y2-B는 nt=3이다. 그래서 **주 단위 = 공통 trial {0,1}의 (task,trial) 쌍**으로 한다
  (집계 선택이 개입하지 않는 유일한 단위). 태스크-수준 집계(pass-any / 다수결)는 **부차 표기**로
  같이 출력하되, 셋 중 유리한 걸 고르는 짓을 막기 위해 **항상 셋 다** 찍는다.
  §6-2의 +4는 "32태스크·1 trial 기준"이므로 trial별로도 분해해 보여준다.

★[[08]] 의무: 종료사유 분포를 먼저 찍고, 간섭 구간 sim 제외 민감도를 함께 낸다
  (`Y2B_RUN_PROVENANCE_2026_07_31.md` — 프로브가 같은 서버에 부하를 넣은 구간이 있다).

용법:
  py -3 y2b_judge.py --y1 <y1 results.json> --y2 <dir1/results.json> [<dir2/results.json> ...]
      [--exclude task_003:1 task_026:0 ...]   # 간섭 sim 민감도
"""
import argparse
import gzip
import json
import math
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

INTERFERED = ["task_003:1", "task_026:0", "task_035:0", "task_025:0", "task_027:0"]


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        return json.load(f)["simulations"]


def passed(s):
    return float(((s.get("reward_info") or {}).get("reward") or 0.0)) >= 1.0


def key(s):
    return (str(s.get("task_id")), int(s.get("trial") or 0))


def sign_test(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))


def mcnemar(y1, y2, keys):
    """b = Y1 pass→Y2 fail · c = Y1 fail→Y2 pass. 개선 = c − b."""
    b = sum(1 for k in keys if passed(y1[k]) and not passed(y2[k]))
    c = sum(1 for k in keys if passed(y2[k]) and not passed(y1[k]))
    return b, c, sign_test(b, c)


def verdict(b, c, p):
    d = c - b
    if d >= 4 and p < .05:
        return "개선 (사전등록 기준 충족)"
    if d <= -4 and p < .05:
        return "악화 (사전등록 기준 충족)"
    return "무결론 (Δ=%+d · p=%.3f — 기준 ≥|4| ∧ p<.05)" % (d, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--y1", required=True)
    ap.add_argument("--y2", nargs="+", required=True)
    ap.add_argument("--exclude", nargs="*", default=None,
                    help="민감도용 제외 sim (task:trial). 미지정 시 provenance 기본값")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    Y1 = {key(s): s for s in load(a.y1)}
    Y2 = {}
    for p in a.y2:
        Y2.update({key(s): s for s in load(p)})
    print("Y1 sim %d · Y2-B sim %d" % (len(Y1), len(Y2)))

    # ── [[08]] ① 종료사유 먼저 ─────────────────────────────────────────────
    print("\n=== 종료사유 (비-모델 실패가 분모를 왜곡하는지 먼저 본다)")
    for nm, D in (("Y1", Y1), ("Y2-B", Y2)):
        print("  %-5s %s" % (nm, dict(Counter(s.get("termination_reason") for s in D.values()))))

    common = sorted(set(Y1) & set(Y2))
    print("\n짝지은 쌍 %d (공통 trial %s)"
          % (len(common), sorted(set(t for _, t in common))))

    # ── 1차 ① McNemar ────────────────────────────────────────────────────
    print("\n=== 1차 ① 짝지은 McNemar (주 단위 = (task,trial))")
    b, c, p = mcnemar(Y1, Y2, common)
    print("  Y1 pass→Y2 fail %d · Y1 fail→Y2 pass %d · Δ=%+d · p=%.3f" % (b, c, c - b, p))
    print("  ★판정: %s" % verdict(b, c, p))
    # ★검출 하한 — 결과를 보기 **전에** 계산해 둔 규칙의 실제 문턱.
    #   사전등록은 "Δ≥4 ∧ p<.05"인데, 부호검정에서 p<.05가 되려면 **한쪽으로 최소 6건**이 필요하다
    #   (b=0,c=4 → p=.125 / b=0,c=6 → p=.031). 즉 **묶이는 제약은 Δ가 아니라 p**다.
    nd = b + c
    need = next((d for d in range(1, nd + 2)
                 if d >= 4 and sign_test(max((nd - d) // 2, 0), max((nd - d) // 2, 0) + d) < .05), None)
    print("  [검출 하한] 불일치 %d쌍에서 기준 충족에 필요한 Δ = %s"
          % (nd, ("≥+%d" % need) if need else "이 불일치 수로는 불가"))
    print("             (설계 §6-2가 예상한 효과는 +0~2 — 즉 1차 판정은 구조상 '무결론'이 기본값이다)")
    # ★이진 판정만 내면 정보가 버려진다. 같은 데이터로 **구간**을 함께 낸다(사전 선언·결과 무관).
    if nd:
        ph, z = c / nd, 1.96
        den = 1 + z * z / nd
        ctr = (ph + z * z / (2 * nd)) / den
        hw = z * math.sqrt(ph * (1 - ph) / nd + z * z / (4 * nd * nd)) / den
        lo, hi = (2 * (ctr - hw) - 1) * nd, (2 * (ctr + hw) - 1) * nd
        print("  [구간] Δ 점추정 %+d · 95%% CI [%+.1f, %+.1f] (Wilson·불일치 %d쌍 기준)"
              % (c - b, lo, hi, nd))

    print("\n  [trial별 분해 — §6-2의 +4는 '32태스크·1 trial' 기준이다]")
    for t in sorted(set(t for _, t in common)):
        ks = [k for k in common if k[1] == t]
        bb, cc, pp = mcnemar(Y1, Y2, ks)
        print("   trial %d (n=%2d): Δ=%+d · p=%.3f" % (t, len(ks), cc - bb, pp))

    # 부차 집계 — 항상 셋 다 찍는다(유리한 것 고르기 방지)
    print("\n  [부차: 태스크-수준 집계 — 참고만]")
    tasks = sorted(set(k[0] for k in common))
    for label, agg in (("pass-any", any), ("pass-all", all)):
        b2 = c2 = 0
        for t in tasks:
            ks = [k for k in common if k[0] == t]
            p1, p2 = agg(passed(Y1[k]) for k in ks), agg(passed(Y2[k]) for k in ks)
            b2 += (p1 and not p2)
            c2 += (p2 and not p1)
        print("   %-9s (n=%d 태스크): Δ=%+d · p=%.3f" % (label, len(tasks), c2 - b2, sign_test(b2, c2)))

    # ── 민감도: 간섭 구간 제외 ────────────────────────────────────────────
    exc = set(a.exclude if a.exclude is not None else INTERFERED)
    ex_keys = set()
    for e in exc:
        t, _, tr = e.partition(":")
        ex_keys |= {k for k in common if k[0] == t and (tr == "" or k[1] == int(tr))}
    if ex_keys:
        kk = [k for k in common if k not in ex_keys]
        bb, cc, pp = mcnemar(Y1, Y2, kk)
        print("\n=== 민감도: 프로브 간섭 구간 sim %d개 제외 (provenance 문서)" % len(ex_keys))
        print("  제외: %s" % ", ".join("%s:%d" % k for k in sorted(ex_keys)))
        print("  Δ=%+d · p=%.3f → %s" % (cc - bb, pp, verdict(bb, cc, pp)))
        print("  ⚠방향이 뒤집히면 간섭이 결론을 만든 것 = 재측정 대상")

    # ── 1차 ② 과행동 ─────────────────────────────────────────────────────
    print("\n=== 1차 ② 과행동 (O1 실효write · O3 중복 · O4 등록집합 · O5 미실행)")
    try:
        import x16_overaction as X16
        a2 = X16.load_a2()
        rows = {}
        for nm, D in (("Y1", Y1), ("Y2-B", Y2)):
            ms = [X16.sim_metrics(D[k], a2) for k in common]
            rows[nm] = {o: sum(m.get(o, 0) for m in ms) / max(len(ms), 1)
                        for o in ("O1", "O3", "O4", "O5")}
        print("  %-6s %8s %8s %8s %8s" % ("", "O1", "O3", "O4", "O5"))
        for nm in ("Y1", "Y2-B"):
            print("  %-6s %8.2f %8.2f %8.2f %8.2f"
                  % (nm, rows[nm]["O1"], rows[nm]["O3"], rows[nm]["O4"], rows[nm]["O5"]))
        d = {o: rows["Y2-B"][o] - rows["Y1"][o] for o in ("O1", "O3", "O4", "O5")}
        print("  Δ(Y2−Y1): " + " ".join("%s %+.2f" % (o, d[o]) for o in ("O1", "O3", "O4", "O5")))
        print("  ★방향: O1·O3 **감소**가 개선(§6-3). O4는 태스크별로 봐야 한다(flip 3건의 기전).")
        print("\n  [O4 태스크별 — flip 기전 확인용]")
        for t in tasks:
            ks = [k for k in common if k[0] == t]
            v1 = sum(X16.sim_metrics(Y1[k], a2)["O4"] for k in ks) / len(ks)
            v2 = sum(X16.sim_metrics(Y2[k], a2)["O4"] for k in ks) / len(ks)
            if abs(v2 - v1) >= 1:
                print("   %-10s O4 %.1f → %.1f (%+.1f)" % (t, v1, v2, v2 - v1))
    except Exception as e:
        print("  ⚠과행동 계측 실패(%r) — x16_overaction 사용 가능한 환경에서 재실행" % (e,))

    # ── 2차 총점 ─────────────────────────────────────────────────────────
    print("\n=== 2차 (참고) 총점")
    for nm, D in (("Y1", Y1), ("Y2-B", Y2)):
        ks = [k for k in common]
        print("  %-6s 짝 구간 pass %d/%d = %.3f"
              % (nm, sum(1 for k in ks if passed(D[k])), len(ks),
                 sum(1 for k in ks if passed(D[k])) / max(len(ks), 1)))
    print("\n⚠[[08]]: 여기서 멈추지 말 것. 판정이 무엇이든 **기전 원장**(V7 deny→재발행→give 성공)과")
    print("  per-step 포렌식(x21 --args)이 산출물이다(설계 §6-2: 'Y2는 pass 올리는 실험이 아니다').")


if __name__ == "__main__":
    main()
