# -*- coding: utf-8 -*-
"""x495 — 레버 x 태스크 x 스텝 부호표와 길항 쌍 지도 (2026-08-23 - 사용자 지시).

사용자 지시(축자 취지):
  - "어떤 레버가 어떤 원인을 해결하고 반대로 어떤 원인을 악화시키는지 확인하고,
     최대 pass 를 얻기 위한 절충안을 만들어야 한다."
  - "실패 자체가 돌아다니는 이유는 한 원인을 제거하기 위한 레버가 태스크에 종속되면서,
     다른 태스크에는 적용 안 되거나, 한 원인을 제거하면 다른 원인을 강화하기 때문이므로,
     정확하게 상대되는 원인별 태스크를 규정하고 쌍으로 실험하여 원인과 레버가 떠도는 것을 막아야 한다."

핵심: 새 8시간 런이 필요 없다. 코퍼스에 **ctl/treat 라이브 A/B 가 11쌍** 이미 있고,
우리는 그것을 총점으로만 읽었다. 여기서는 짝마다

    (1) 태스크별 pass 차이   ctl -> treat
    (2) gold 스텝별 miss율 차이 (원인 단위 - 태스크보다 일반적이다)

를 내고, 한 레버 아래에서 **부호가 반대로 갈리는 태스크 쌍 = 길항 쌍**을 뽑는다.
길항 쌍이 확정되면 다음 실험은 그 쌍을 **함께** 태워야 하고, 그때만 절충이 측정된다.

경고와 한계:
  - 짝당 n 이 작다(팔당 sim 4~20). 개별 태스크 차이는 대부분 잡음 바닥(C483: +-4/40) 안이다.
    그래서 여기서 나오는 것은 **확정이 아니라 길항 후보 지도**다. 확정은 그 쌍만 태운 재실험이 한다.
  - 짝마다 sha 와 나머지 스택이 다르다. 같은 짝 안의 ctl<->treat 비교만 유효하고,
    짝을 가로지르는 절대 비교는 하지 않는다.
  - reward 는 벤치가 매긴다([[69]]). 스텝 miss 는 진단 보조이지 성적이 아니다.
  - gold 는 어느 스텝이 빠졌나를 세는 진단으로만 쓴다([[23]]).

실행: PYTHONIOENCODING=utf-8 python x495_lever_antagonist_map.py
"""
import collections
import glob
import gzip
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_forensic as F          # noqa: E402

SIMS = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "sim_results")

# 짝 = (레버 라벨, ctl 파일 접두, treat 파일 접두). 처치 변수는 런 스크립트 축자에서 읽었다.
PAIRS = [
    ("T2_NOW_SELFCALL + T2_SEARCH_ON_PROCEED",
     "bank_t7296_ctl_20260815p", "bank_t7296_treat_20260815p"),
    ("T2_ACT_DEMAND",
     "bank_t7297_ctl_20260815q", "bank_t7297_treat_20260815q"),
    ("T2_MATERIAL_RESERVE",
     "bank_t7299_ctl_20260816b", "bank_t7299_treat_20260816b"),
    ("T2_DELIVER_PRECOMMIT",
     "bank_t7303_ctl_20260816h", "bank_t7303_treat_20260816h"),
    ("T2_PROCEED_DOCBODY",
     "bank_t7304_ctl_20260816j", "bank_t7304_treat_20260816j"),
    ("T2_SUB_REQUIREMENT",
     "bank_t7305_ctl_20260817a", "bank_t7305_treat_20260817a"),
    ("T2_HANDOFF_PREDICATE (a)",
     "bank_t7307_ctl_20260818b", "bank_t7307_treat_20260818b"),
    ("T2_HANDOFF_PREDICATE (b)",
     "bank_t7308_ctl_20260818c", "bank_t7308_treat_20260818c"),
    ("T2_VERDICT_CARRY + T2_ELIG_LINE (a)",
     "bank_t7310_ctl_20260818e", "bank_t7310_treat_20260818e"),
    ("T2_VERDICT_CARRY + T2_ELIG_LINE (b)",
     "bank_t7312_ctl_20260818g", "bank_t7312_treat_20260818g"),
    ("T2_VERDICT_CARRY + T2_ELIG_LINE (c)",
     "bank_t7313_ctl_20260818h", "bank_t7313_treat_20260818h"),
    ("T2_ARG_DOC_SUB + T2_VALUE_FORMULA=full (hot)",
     "bank_t7333_ctl_hot_20260821c", "bank_t7333_treat_hot_20260821c"),
    ("T2_ARG_DOC_SUB + T2_VALUE_FORMULA=full (rest)",
     "bank_t7333_ctl_rest_20260821c", "bank_t7333_treat_rest_20260821c"),
]


def load(prefix, MUT):
    """접두 -> {task: {"n":n, "pass":n, "steps":{aid:[miss,tot]}, "tool":{aid:name}}}"""
    out = collections.defaultdict(
        lambda: {"n": 0, "pass": 0, "steps": collections.defaultdict(lambda: [0, 0]),
                 "tool": {}})
    # 파일명 규칙이 두 갈래다: `<접두>.results.json.gz` 와 `<접두>_results.json.gz`.
    # 둘 다 잡되 `ctlaux/treataux` 같은 보조 팔은 접두가 다르므로 자동으로 제외된다.
    cand = sorted(set(glob.glob(os.path.join(SIMS, prefix + ".results.json.gz")))
                  | set(glob.glob(os.path.join(SIMS, prefix + "_results.json.gz")))
                  | set(glob.glob(os.path.join(SIMS, prefix + "_half*.results.json.gz"))))
    for fp in cand:
        try:
            with gzip.open(fp, "rt", encoding="utf-8", errors="replace") as f:
                d = json.load(f)
        except Exception:
            continue
        for s in (d.get("simulations") or []):
            t = s.get("task_id")
            if not t:
                continue
            rec = out[t]
            rec["n"] += 1
            if (s.get("reward_info") or {}).get("reward"):
                rec["pass"] += 1
            try:
                md = F.mutation_diff(s, MUT)
            except Exception:
                continue
            miss = {e.get("aid") for e in (md.get("missing") or [])}
            for e in (md.get("gold") or []):
                aid = e.get("aid")
                if not aid:
                    continue
                rec["tool"][aid] = e.get("name")
                cell = rec["steps"][aid]
                cell[1] += 1
                if aid in miss:
                    cell[0] += 1
    return out


def main():
    MUT = F.mutating_tools()
    lever_task = {}          # lever -> {task: (d_pass, n_ctl, n_treat, p_ctl, p_treat)}
    lever_step = {}          # lever -> [(task, aid, tool, d_missrate, n)]
    print("A/B 짝 %d개 - 짝 안의 ctl <-> treat 만 비교한다\n" % len(PAIRS))
    for lever, cp, tp in PAIRS:
        C, T = load(cp, MUT), load(tp, MUT)
        tasks = sorted(set(C) & set(T))
        if not tasks:
            print("  [skip] %-46s (겹치는 태스크 없음)" % lever)
            continue
        tt, ss = {}, []
        for t in tasks:
            c, x = C[t], T[t]
            if not c["n"] or not x["n"]:
                continue
            pc = c["pass"] / c["n"]
            px = x["pass"] / x["n"]
            tt[t] = (px - pc, c["n"], x["n"], c["pass"], x["pass"])
            for aid in sorted(set(c["steps"]) & set(x["steps"])):
                cm, ct = c["steps"][aid]
                xm, xt = x["steps"][aid]
                if not ct or not xt:
                    continue
                d = (xm / xt) - (cm / ct)
                if abs(d) > 1e-9:
                    ss.append((t, aid, c["tool"].get(aid) or x["tool"].get(aid),
                               d, min(ct, xt)))
        lever_task[lever] = tt
        lever_step[lever] = ss

    # ── (1) 레버 x 태스크 부호표
    print("=" * 100)
    print("(1) 레버 x 태스크 부호표 - + = treat 가 더 통과 / - = treat 가 덜 통과")
    print("=" * 100)
    for lever in [p[0] for p in PAIRS]:
        tt = lever_task.get(lever)
        if not tt:
            continue
        up = sorted([(d, t, a) for t, (d, cn, xn, cp2, xp) in tt.items()
                     for a in [(cp2, xp, cn, xn)] if d > 0], reverse=True)
        dn = sorted([(d, t, a) for t, (d, cn, xn, cp2, xp) in tt.items()
                     for a in [(cp2, xp, cn, xn)] if d < 0])
        flat = [t for t, (d, *_r) in tt.items() if d == 0]
        tot_c = sum(v[3] for v in tt.values())
        tot_x = sum(v[4] for v in tt.values())
        print("\n%s" % lever)
        print("   총점 ctl %d -> treat %d  (태스크 %d)" % (tot_c, tot_x, len(tt)))
        if up:
            print("   샀다 : " + " · ".join(
                "%s %d/%d->%d/%d" % (t, a[0], a[2], a[1], a[3]) for d, t, a in up))
        if dn:
            print("   팔았다: " + " · ".join(
                "%s %d/%d->%d/%d" % (t, a[0], a[2], a[1], a[3]) for d, t, a in dn))
        if not up and not dn:
            print("   변화 없음 (%d 태스크 전부 동일)" % len(flat))

    # ── (2) 길항 쌍
    print("\n" + "=" * 100)
    print("(2) 길항 쌍 - 같은 레버 아래에서 부호가 반대로 갈린 태스크 쌍")
    print("    이 쌍을 함께 태워야 절충이 측정된다. 한쪽만 보면 레버가 떠돈다.")
    print("=" * 100)
    anta = []
    for lever, tt in lever_task.items():
        up = [t for t, v in tt.items() if v[0] > 0]
        dn = [t for t, v in tt.items() if v[0] < 0]
        for a in up:
            for b in dn:
                anta.append((lever, a, b, tt[a][0], tt[b][0]))
    if not anta:
        print("   (없음)")
    for lever, a, b, da, db in anta:
        print("   %-46s  %s (+%.2f)  <->  %s (%.2f)" % (lever[:46], a, da, b, db))

    # ── (3) 태스크 횡단: 같은 태스크가 여러 레버에서 사는 쪽인가 파는 쪽인가
    print("\n" + "=" * 100)
    print("(3) 태스크별 성향 - 여러 레버에 걸쳐 사는 쪽/파는 쪽 누적")
    print("=" * 100)
    tally = collections.defaultdict(lambda: [0, 0, []])
    for lever, tt in lever_task.items():
        for t, v in tt.items():
            if v[0] > 0:
                tally[t][0] += 1
                tally[t][2].append("+" + lever.split()[0])
            elif v[0] < 0:
                tally[t][1] += 1
                tally[t][2].append("-" + lever.split()[0])
    print("%-10s %6s %6s  %s" % ("task", "샀다", "팔았다", "레버"))
    print("-" * 90)
    for t in sorted(tally, key=lambda x: -(tally[x][0] + tally[x][1])):
        u, d, ls = tally[t]
        if u + d == 0:
            continue
        print("%-10s %6d %6d  %s" % (t, u, d, " ".join(ls)[:64]))

    # ── (4) 스텝(원인) 단위 - 태스크보다 일반적인 층
    print("\n" + "=" * 100)
    print("(4) 스텝 단위 효과 - 같은 gold 스텝이 레버 아래에서 어떻게 움직였나")
    print("    (miss율 차이. - 가 좋다 = treat 에서 덜 빠졌다)")
    print("=" * 100)
    for lever in [p[0] for p in PAIRS]:
        ss = lever_step.get(lever) or []
        ss = [r for r in ss if r[4] >= 2]
        if not ss:
            continue
        best = sorted(ss, key=lambda r: r[3])[:3]
        worst = sorted(ss, key=lambda r: -r[3])[:3]
        print("\n%s" % lever)
        for t, aid, tool, d, n in best:
            print("   좋아짐 %-9s %-9s %-38s %+.2f (n>=%d)" % (t, aid, (tool or "?")[:38], d, n))
        for t, aid, tool, d, n in worst:
            if d > 0:
                print("   나빠짐 %-9s %-9s %-38s %+.2f (n>=%d)" % (t, aid, (tool or "?")[:38], d, n))

    out = {
        "pairs": [{"lever": l, "ctl": c, "treat": t} for l, c, t in PAIRS],
        "lever_task": {l: {t: {"d_pass": v[0], "n_ctl": v[1], "n_treat": v[2],
                               "pass_ctl": v[3], "pass_treat": v[4]}
                           for t, v in tt.items()}
                       for l, tt in lever_task.items()},
        "antagonists": [{"lever": l, "bought": a, "sold": b,
                         "d_bought": da, "d_sold": db}
                        for l, a, b, da, db in anta],
        "lever_step": {l: [{"task": t, "aid": aid, "tool": tool, "d_missrate": d, "n": n}
                           for t, aid, tool, d, n in (lever_step.get(l) or [])]
                       for l in lever_step},
    }
    dst = os.path.join(SIMS, "..", "x495_lever_antagonist_map.json")
    with io.open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n-> %s" % os.path.normpath(dst))
    return 0


if __name__ == "__main__":
    sys.exit(main())
