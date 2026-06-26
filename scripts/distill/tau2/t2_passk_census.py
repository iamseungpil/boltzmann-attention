#!/usr/bin/env python
"""τ² ±게이트 판정 배터리: pass_hat_k(1..4) + 게이트-deny 궤적 census + 실패 귀속.

ⓟ1 (마스터 §1.6 사전등록): 게이트 해석 = Δpass_hat_4 > Δpass_hat_1.
게이트 음성 시 귀속: G1/G2/G3 deny 빈도·deny가 성공/실패와 어떻게 공변하는지.

Usage: t2_passk_census.py --simdir /home/woori/scratch/tau2-bench/data/simulations
"""
import argparse, json, math
from collections import Counter

GATES = ["G1_AUTH_FIRST", "G2_CONFIRM_WRITE", "G3_SINGLE_USER"]


def load(simdir, arm):
    d = json.load(open(f"{simdir}/retail_7b_{arm}/results.json"))
    return d["simulations"]


def pass_hat_k(per_task, k):
    tot, n = 0.0, 0
    for rs in per_task.values():
        c, nn = sum(rs), len(rs)
        if nn < k:
            continue
        tot += math.comb(c, k) / math.comb(nn, k)
        n += 1
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--simdir", required=True)
    a = ap.parse_args()

    stats = {}
    for arm in ("nogate", "gate"):
        sims = load(a.simdir, arm)
        per_task = {}
        denies = Counter()
        deny_and_fail = deny_and_pass = nodeny_fail = nodeny_pass = 0
        comp = Counter()  # reward_breakdown 실패 컴포넌트
        for s in sims:
            ri = s.get("reward_info") or {}
            r = ri.get("reward", 0) or 0
            ok = r >= 1
            per_task.setdefault(s["task_id"], []).append(1 if ok else 0)
            had = False
            for m in (s.get("messages") or []):
                c = m.get("content") or ""
                if isinstance(c, str) and "POLICY GATE" in c:
                    had = True
                    for g in GATES:
                        if g in c:
                            denies[g] += 1
            if had and ok:
                deny_and_pass += 1
            elif had:
                deny_and_fail += 1
            elif ok:
                nodeny_pass += 1
            else:
                nodeny_fail += 1
            if not ok:
                rb = ri.get("reward_breakdown") or {}
                for kk, v in rb.items():
                    if v is not None and v < 1:
                        comp[kk] += 1
        row = {f"pass^{k}": round(pass_hat_k(per_task, k), 4) for k in (1, 2, 3, 4)}
        stats[arm] = row
        print(f"{arm}: tasks={len(per_task)} {row}")
        print(f"  denies={dict(denies)} | sims deny&fail={deny_and_fail} deny&pass={deny_and_pass} "
              f"nodeny_fail={nodeny_fail} nodeny_pass={nodeny_pass}")
        print(f"  fail components: {dict(comp.most_common(6))}")
    d1 = stats["gate"]["pass^1"] - stats["nogate"]["pass^1"]
    d4 = stats["gate"]["pass^4"] - stats["nogate"]["pass^4"]
    print(f"[p1 verdict] dpass^1={d1:+.4f} dpass^4={d4:+.4f} -> "
          f"{'CONSISTENT with gate=consistency-lever' if d4 > d1 else 'NOT supported'}")


if __name__ == "__main__":
    main()
