#!/usr/bin/env python3
"""x23 — retry-controller 판정 재분석 (C262 후속·무료·읽기 전용).

배경: §1.3이 retry를 "실측 해로움 = 죽은 레버"로 적었으나, 그 근거(FLOW_DISCIPLINE_RESULTS
2026-06-23 §핵심결론3)는 **nt=1 단일시행 pass^1 점추정**이고 같은 문서가 32B 행을 nt=3에서
철회했다(C262). 여기서는 원자료(2026-06-22~23 retail 런)로 다시 잰다 — 집계 직행 금지([[08]]):

  ① 종료사유 분포(throttle/infra 오염 확인)   ② 짝지은 (task_id, trial) 비교 + 부호검정
  ③ **레버 실발화 층화** — 발화한 sim만이 인과 후보. 미발화 sim의 차이는 정의상 노이즈.
  ④ 발화 sim의 전이표(fire × pass)

사용: python x23_retry_reanalysis.py [--root DIR]
출력: stdout 표(원격 실행·리포 트리에 파일 안 씀).
"""
import argparse
import json
import math
import os
from collections import Counter, defaultdict

FIRE_MARKERS = ("[POLICY GATE RETRY_LOOP]", "[POLICY GATE RETRY_ESCALATE]")


def load(path):
    with open(path) as fh:
        return json.load(fh)["simulations"]


def fired(sim):
    """이 궤적에서 retry deny가 실제로 발화한 횟수(규칙별)."""
    n = Counter()
    for m in sim.get("messages") or []:
        if m.get("role") != "tool":
            continue
        c = m.get("content") or ""
        for mark in FIRE_MARKERS:
            if mark in c:
                n[mark.split()[-1].rstrip("]")] += 1
    return n


def passed(sim):
    return float(((sim.get("reward_info") or {}).get("reward") or 0.0)) >= 1.0


def triggered(sim, k=3):
    """★대칭 탐지기 — `gated()`의 두 규칙을 **양 arm 궤적에 동일하게** 재현한다.

    발화(fired)는 처치-후 변수라 그것만으로 층화하면 처치군의 매개변수에 조건화하는
    편향이 된다. 그래서 *대조군에도* 같은 판정을 적용해 '루프에 빠진 궤적'을 대칭으로
    식별한다: ①직전에 실패한 (name,args) 동일 재호출 ②연속 실패 K회.
    반환 = (rule1_hit, rule2_hit) 횟수.
    """
    failed_keys, consec = set(), 0
    r1 = r2 = 0
    res = {}   # tool_call_id -> error?
    for m in sim.get("messages") or []:
        if m.get("role") == "tool":
            res[m.get("id")] = bool(m.get("error"))
    for m in sim.get("messages") or []:
        if m.get("role") != "assistant":
            continue
        for tc in (m.get("tool_calls") or []):
            args = tc.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {"_raw": args}
            key = (tc.get("name") or "") + "::" + json.dumps(args or {}, sort_keys=True,
                                                             ensure_ascii=False)
            if key in failed_keys:
                r1 += 1
            if consec >= k:
                r2 += 1
                consec = 0
            err = res.get(tc.get("id"), False)
            if err:
                failed_keys.add(key)
                consec += 1
            else:
                consec = 0
    return r1, r2


def summarize(sims):
    out = {
        "n": len(sims),
        "pass": sum(1 for s in sims if passed(s)),
        "term": Counter(s.get("termination_reason") for s in sims),
        "fire_sims": 0,
        "fire_calls": Counter(),
    }
    for s in sims:
        f = fired(s)
        if f:
            out["fire_sims"] += 1
            out["fire_calls"].update(f)
    return out


def sign_test(b, c):
    """부호검정(양측·정확) — b=base만 pass, c=retry만 pass."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/woori/scratch/tau2-bench/data/simulations")
    a = ap.parse_args()

    pairs = []
    for model in ("n7b", "n14b", "n32int8"):
        for suf in ("", "_t3"):
            base = os.path.join(a.root, "ours_%s_g15_retail%s" % (model, suf), "results.json")
            retry = os.path.join(a.root, "ours_%s_g15retry_retail%s" % (model, suf), "results.json")
            if os.path.exists(base) and os.path.exists(retry):
                pairs.append(("%s%s" % (model, suf or "_nt1"), base, retry))
    # nt=3 denoise 계열(on_n32int8_*_retail_t3)도 있으면 포함
    b3 = os.path.join(a.root, "on_n32int8_g15_retail_t3", "results.json")
    r3 = os.path.join(a.root, "on_n32int8_g15retry_retail_t3", "results.json")
    if os.path.exists(b3) and os.path.exists(r3):
        pairs.append(("n32int8_DENOISE_t3", b3, r3))

    for label, bp, rp in pairs:
        B, R = load(bp), load(rp)
        sb, sr = summarize(B), summarize(R)
        print("=" * 78)
        print("ARM PAIR: %s" % label)
        print("  g15      n=%3d pass=%3d (%.3f)  term=%s"
              % (sb["n"], sb["pass"], sb["pass"] / max(sb["n"], 1), dict(sb["term"])))
        print("  g15retry n=%3d pass=%3d (%.3f)  term=%s"
              % (sr["n"], sr["pass"], sr["pass"] / max(sr["n"], 1), dict(sr["term"])))
        print("  ★발화: g15retry sims=%d/%d  calls=%s   |  g15(대조) sims=%d"
              % (sr["fire_sims"], sr["n"], dict(sr["fire_calls"]), sb["fire_sims"]))

        # 짝짓기 = (task_id, trial)
        bi = {(s.get("task_id"), s.get("trial")): s for s in B}
        ri = {(s.get("task_id"), s.get("trial")): s for s in R}
        keys = sorted(set(bi) & set(ri), key=lambda k: (str(k[0]), k[1]))
        b_only = c_only = both = neither = 0
        fire_keys = []
        for k in keys:
            pb, pr = passed(bi[k]), passed(ri[k])
            if pb and not pr:
                b_only += 1
            elif pr and not pb:
                c_only += 1
            elif pb and pr:
                both += 1
            else:
                neither += 1
            if fired(ri[k]):
                fire_keys.append(k)
        print("  짝지은 N=%d  (both %d · g15만 %d · retry만 %d · 둘다실패 %d)"
              % (len(keys), both, b_only, c_only, neither))
        print("  부호검정 p=%.3f   Δpass(짝지은)=%+.3f"
              % (sign_test(b_only, c_only), (c_only - b_only) / max(len(keys), 1)))

        # ★발화-층화: 레버가 실제로 발화한 짝만
        if fire_keys:
            fb = sum(1 for k in fire_keys if passed(bi[k]) and not passed(ri[k]))
            fc = sum(1 for k in fire_keys if passed(ri[k]) and not passed(bi[k]))
            fboth = sum(1 for k in fire_keys if passed(ri[k]) and passed(bi[k]))
            print("  ★발화-층화 N=%d: both %d · g15만 %d · retry만 %d · 둘다실패 %d  → p=%.3f"
                  % (len(fire_keys), fboth, fb, fc,
                     len(fire_keys) - fboth - fb - fc, sign_test(fb, fc)))
            print("     (미발화 짝 %d개의 차이는 정의상 레버와 무관 = 노이즈)"
                  % (len(keys) - len(fire_keys)))
        else:
            print("  ★발화-층화: 발화 0 — 이 arm에서 레버는 아무것도 하지 않았다.")

        # ★대칭 층화 — 층을 **대조군 궤적**의 트리거로 정의(처치군 매개변수 조건화 회피).
        ctrl_trig = [k for k in keys if sum(triggered(bi[k])) > 0]
        b2 = sum(1 for k in ctrl_trig if passed(bi[k]) and not passed(ri[k]))
        c2 = sum(1 for k in ctrl_trig if passed(ri[k]) and not passed(bi[k]))
        bp = sum(1 for k in ctrl_trig if passed(bi[k]))
        rp = sum(1 for k in ctrl_trig if passed(ri[k]))
        print("  ★★대조군-트리거 층 N=%d: g15 pass %d(%.3f) vs retry pass %d(%.3f) "
              "· 불일치 %d/%d → p=%.3f"
              % (len(ctrl_trig), bp, bp / max(len(ctrl_trig), 1), rp,
                 rp / max(len(ctrl_trig), 1), b2, c2, sign_test(b2, c2)))
        # 교차표: 대조군 트리거 × 처치군 실발화 (탐지기 타당성 점검)
        xt = Counter((sum(triggered(bi[k])) > 0, bool(fired(ri[k]))) for k in keys)
        print("     교차표(ctrl_trigger, retry_fired): %s"
              % {("T" if a_ else "F") + ("T" if b_ else "F"): v for (a_, b_), v in sorted(xt.items())})


if __name__ == "__main__":
    main()
