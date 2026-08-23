# -*- coding: utf-8 -*-
"""x492 — 중재자를 처음으로 잰다 (2026-08-23 · 사용자 물음 *"이걸 측정할 방법이 없나"*).

t7346 40 sim 에서 `[T2_MATERIAL_GATE] stop=` 이 **439회**(sim당 11회) 찍혔다. 우리 층이 우리
층을 막은 기록인데, **몇 개가 결정점 앞이었고 몇 개가 옳았는지 한 번도 안 쟀다.**

술어·집계는 전부 정본 `t2_liveness.arbitration/report_arbitration` 에 있다([[67]] — 여기에
사본을 짜지 않는다). 이 파일은 **드라이버**일 뿐이다: 로그·결과를 물려 주고, 태스크별 표를
찍고, JSON 으로 영속한다.

★[[57]] 부정통제의 관찰 형태 = **pass sim 과 fail sim 의 정지율을 나란히 놓는 것**이다.
  같으면 그 정지는 실패의 원인이 아니다. 이건 상관이지 인과가 아니고, 인과는 라이브 A/B 나
  환경-롤아웃만 준다(결정점 재생은 못 한다 — C596).

실행: PYTHONIOENCODING=utf-8 python x492_arbitration_ledger.py [런태그 ...]   (기본 t7346)
"""
import collections
import glob
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

import t2_liveness as L          # noqa: E402

SIMS = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "sim_results")


def main(argv):
    tags = argv or ["bank_t7346_half*_20260822"]
    logs, res = [], []
    for t in tags:
        logs += sorted(glob.glob(os.path.join(SIMS, t + ".log.gz")))
        res += sorted(glob.glob(os.path.join(SIMS, t + ".results.json.gz")))
    if not logs:
        print("로그 없음: %s" % tags)
        return 1
    print("로그 %d · 결과 %d\n" % (len(logs), len(res)))

    rewards = L.rewards_from_results(res)
    per = L.arbitration(logs, rewards)

    groups = L.report_arbitration(per)

    # ── 태스크별: 정지가 몰린 곳과 배달이 0인 곳
    print("\n태스크별 (정지 / 배달 / reward)")
    print("%-10s %6s %7s %8s %9s" % ("task", "sim", "정지", "배달", "pass"))
    print("-" * 46)
    bytask = collections.defaultdict(lambda: [0, 0, 0, 0])
    for k, v in per.items():
        t = k.split("#")[0]
        row = bytask[t]
        row[0] += 1
        row[1] += sum(v["stops"].values())
        row[2] += v["deliveries"]
        row[3] += 1 if v.get("reward") else 0
    for t in sorted(bytask, key=lambda x: -bytask[x][1]):
        n, st, dl, ps = bytask[t]
        print("%-10s %6d %7d %8d %9d" % (t, n, st, dl, ps))

    # ── 정지 0배달 sim: 관문이 그 sim 에서 채널을 영구히 닫았나
    closed = sorted(k for k, v in per.items()
                    if v["deliveries"] == 0 and sum(v["stops"].values()) > 0)
    print("\n★정지는 있는데 **배달이 한 번도 없던 sim** %d개:" % len(closed))
    for k in closed:
        v = per[k]
        print("   %-24s 정지 %3d (%s) reward=%s"
              % (k, sum(v["stops"].values()),
                 ",".join("%s:%d" % kv for kv in v["stops"].most_common()), v.get("reward")))

    # ── 승자 분포
    win = collections.Counter()
    for v in per.values():
        win.update(v["winners"])
    print("\n`other_lever` 정지에서 **이긴 레버**:")
    for k, n in win.most_common():
        print("   %-10s %d" % (k, n))


    # ── ★방향성 칸 (2026-08-23): 전체 정지율은 **교란된다** — 실패 sim 은 더 길고 더 오래
    #    버티므로 정지가 자연히 쌓인다(실측 로그줄 PASS 202 ↔ FAIL 401 = 1.99×). 그래서
    #    **분기 이전 구간**(이른 메시지)만 따로 센다. 그 구간의 정지는 아직 일어나지 않은
    #    실패의 결과일 수 없다. 여기서 갈리지 않으면 **전체 상관은 증상이지 원인이 아니다**.
    def early(v, cut):
        return sum(1 for _, t in v["stop_turns"] if t <= cut)

    print("\n★분기-이전 구간의 정지 (여기서 갈려야 원인 후보다)")
    print("%-10s %28s %28s %7s" % ("구간", "PASS 평균(합)", "FAIL 평균(합)", "비"))
    print("-" * 78)
    early_rows = {}
    for cut in (12, 20, 30):
        P = [early(v, cut) for v in per.values() if v.get("reward")]
        F = [early(v, cut) for v in per.values() if not v.get("reward")]
        mp = (sum(P) / len(P)) if P else 0.0
        mf = (sum(F) / len(F)) if F else 0.0
        early_rows["msg<=%d" % cut] = {"pass_mean": mp, "fail_mean": mf,
                                       "pass_sum": sum(P), "fail_sum": sum(F)}
        print("%-10s %28s %28s %7s"
              % ("msg<=%d" % cut, "%.2f (%d)" % (mp, sum(P)), "%.2f (%d)" % (mf, sum(F)),
                 ("%.2fx" % (mf / mp)) if mp else "-"))
    lp = [v["lines"] for v in per.values() if v.get("reward")]
    lf = [v["lines"] for v in per.values() if not v.get("reward")]
    if lp and lf:
        print("(교란 확인) 로그줄 평균 PASS %.0f ↔ FAIL %.0f = %.2fx"
              % (sum(lp) / len(lp), sum(lf) / len(lf), (sum(lf) / len(lf)) / (sum(lp) / len(lp))))

    # ── 짝 대조: 같은 태스크에서 한쪽만 pass 한 경우. 태스크 난이도가 통제된다.
    print("\n★짝 대조(같은 태스크·한쪽만 pass) — 태스크 난이도 통제")
    print("%-10s %24s %24s" % ("task", "PASS 정지/이른12/줄", "FAIL 정지/이른12/줄"))
    pairs = []
    byt2 = collections.defaultdict(list)
    for k, v in per.items():
        byt2[k.split("#")[0]].append(v)
    for t in sorted(byt2):
        ps = [v for v in byt2[t] if v.get("reward")]
        fs = [v for v in byt2[t] if not v.get("reward")]
        if len(ps) == 1 and len(fs) == 1:
            p, f = ps[0], fs[0]
            row = {"task": t,
                   "pass": [sum(p["stops"].values()), early(p, 12), p["lines"]],
                   "fail": [sum(f["stops"].values()), early(f, 12), f["lines"]]}
            pairs.append(row)
            print("%-10s %24s %24s" % (t, "%d / %d / %d" % tuple(row["pass"]),
                                       "%d / %d / %d" % tuple(row["fail"])))
    print("짝 %d개" % len(pairs))
    print("\n★판독 규칙: 분기-이전 구간에서 비가 ~1.0 이면 **총 정지 수는 증상이다** — 실패한")
    print("  sim 이 더 오래 버텨서 쌓인 것이지 정지가 실패를 만든 것이 아니다. 그래도 **개별**")
    print("  정지가 결정점을 막았을 가능성은 남는다(016 의 turn 40·42·44 가 그 후보) —")
    print("  그건 사건별로 *그때 무엇이 대기 중이었나*를 봐야 하고, 이 집계는 못 본다([[08]]).")

    out = {"logs": [os.path.basename(p) for p in logs],
           "n_sim": len(per),
           "totals": {g: {"sims": len(rows),
                          "stops": sum(sum(r["stops"].values()) for r in rows),
                          "lines": sum(r["lines"] for r in rows),
                          "deliveries": sum(r["deliveries"] for r in rows),
                          "zero_delivery_sims": sum(1 for r in rows if r["deliveries"] == 0)}
                      for g, rows in groups.items() if rows},
           "winners": dict(win),
           "channel_closed_sims": closed,
           "early_window": early_rows,
           "matched_pairs": pairs,
           "per_sim": {k: {"reward": v.get("reward"), "lines": v["lines"],
                           "stops": dict(v["stops"]), "deliveries": v["deliveries"],
                           "clobbered_bytes": v["clobbered_bytes"],
                           "suppressed": v["suppressed"],
                           "stop_turns": v["stop_turns"]}
                       for k, v in per.items()}}
    dst = os.path.join(SIMS, "..", "x492_arbitration_ledger.json")
    with io.open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n→ %s" % os.path.normpath(dst))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
