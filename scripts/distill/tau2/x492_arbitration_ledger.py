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
        t = L.task_of(k)
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
        byt2[L.task_of(k)].append(v)
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


    # ── ★6단계: 사건별 — 이 정지는 **어떤 턴**에 떨어졌고, 그 뒤에 재료가 오긴 했나
    #    `turn=len(state.messages)` 는 그 턴에 커밋된 assistant 메시지의 인덱스와 **정확히**
    #    일치한다(016#s626729 의 turn 38·40·42·44·46 ↔ messages[38·40·42·44·46] 로 검증).
    #    그래서 옛 런에도 패자 기록 없이 이 한 칸은 갈 수 있다: 그 턴의 초안이 **도구를 하나도
    #    안 불렀나**(=산문 턴). 산문 턴은 모델이 다음 수를 못 정한 자리이고, 재료가 바로 그때
    #    막혔다면 그것이 최대 혐의다. 도구 턴이면 모델은 어차피 행동 중이었다.
    #    ⚠[[23]]: gold 는 **진단**으로만 읽는다(어느 sim 이 실패했나). 레버 선택에 안 쓴다.
    msgs = {}
    for rp in res:
        opener2 = gzip.open if rp.endswith(".gz") else io.open
        with opener2(rp, "rt", encoding="utf-8", errors="replace") as f:
            dd = json.load(f)
        for sm in (dd.get("simulations") or []):
            msgs["%s|%s#s%s" % (L.run_tag(rp), sm.get("task_id"), sm.get("seed"))] = sm.get("messages") or []

    BUCKETS = ("결정이후_행동턴", "행동턴", "산문턴_뒤에배달있음", "★산문턴_뒤에배달없음", "정렬불가")
    tally = {b: collections.Counter() for b in BUCKETS}       # bucket -> {PASS/FAIL: n}
    suspects = collections.Counter()                          # sim -> 혐의 정지 수
    for k, v in per.items():
        grp = "PASS" if v.get("reward") else ("FAIL" if v.get("reward") is not None else None)
        if grp is None:
            continue
        mm = msgs.get(k) or []
        dls = v.get("delivery_lines") or []
        for kind, turn, at in (v.get("stop_lines") or []):
            if turn >= len(mm):
                b = "정렬불가"
            else:
                m = mm[turn]
                if m.get("role") != "assistant":
                    b = "정렬불가"
                elif (m.get("tool_calls") or []):
                    b = "행동턴"
                elif any(n > at for n in dls):
                    b = "산문턴_뒤에배달있음"
                else:
                    b = "★산문턴_뒤에배달없음"
                    suspects[k] += 1
            tally[b][grp] += 1

    print("\n★6단계 — 정지 %d건을 **떨어진 턴의 성격**으로 가른다"
          % sum(sum(c.values()) for c in tally.values()))
    print("%-26s %8s %8s %10s %12s" % ("버킷", "PASS", "FAIL", "합", "FAIL 비중"))
    print("-" * 70)
    for b in BUCKETS:
        pz, fz = tally[b].get("PASS", 0), tally[b].get("FAIL", 0)
        if pz + fz == 0:
            continue
        print("%-26s %8d %8d %10d %11.0f%%" % (b, pz, fz, pz + fz, 100.0 * fz / (pz + fz)))
    npass = sum(1 for v in per.values() if v.get("reward"))
    nfail = sum(1 for v in per.values() if v.get("reward") is not None and not v.get("reward"))
    sp_ = tally["★산문턴_뒤에배달없음"]
    print("\n혐의 버킷의 **sim 당** 비율 — 여기가 갈려야 원인 후보다([[57]])")
    print("   PASS %.2f/sim (%d sim)   FAIL %.2f/sim (%d sim)   비 %s"
          % ((sp_.get("PASS", 0) / npass) if npass else 0, npass,
             (sp_.get("FAIL", 0) / nfail) if nfail else 0, nfail,
             ("%.2fx" % (((sp_.get("FAIL", 0) / nfail) / (sp_.get("PASS", 0) / npass))
                         if npass and sp_.get("PASS", 0) else 0)) if npass else "-"))
    if suspects:
        print("\n혐의 정지가 가장 많은 sim:")
        for k, n in suspects.most_common(8):
            print("   %-24s %2d건  reward=%s" % (k, n, per[k].get("reward")))
    step6 = {"buckets": {b: dict(tally[b]) for b in BUCKETS},
             "suspects": dict(suspects), "n_pass": npass, "n_fail": nfail}


    # ── ★6단계의 부정통제: 혐의 버킷에도 **분기-이전 창**을 건다.
    #    전체 정지에서 1.04x 였던 것이 혐의 버킷에서 1.81x 로 갈렸다. 그런데 그 1.81x 역시
    #    실패 sim 이 더 길어서 생긴 것일 수 있다 — 실제로 pass 인 017#s373753 에 혐의 정지가
    #    10건 있다. 그래서 **아직 결과가 안 갈린 구간**만 다시 센다.
    def _suspect_turns(k, v):
        mm = msgs.get(k) or []
        dls = v.get("delivery_lines") or []
        out = []
        for kind, turn, at in (v.get("stop_lines") or []):
            if turn >= len(mm):
                continue
            m = mm[turn]
            if m.get("role") != "assistant" or (m.get("tool_calls") or []):
                continue
            if any(n > at for n in dls):
                continue
            out.append(turn)
        return out

    print("\n★혐의 버킷 × 분기-이전 창 (여기서도 갈려야 인과 후보다)")
    print("%-10s %22s %22s %8s" % ("구간", "PASS 평균(합)", "FAIL 평균(합)", "비"))
    print("-" * 66)
    susp_early = {}
    for cut in (12, 20, 30, 10 ** 9):
        P, F = [], []
        for k, v in per.items():
            if v.get("reward") is None:
                continue
            n = sum(1 for t in _suspect_turns(k, v) if t <= cut)
            (P if v.get("reward") else F).append(n)
        mp = (sum(P) / len(P)) if P else 0.0
        mf = (sum(F) / len(F)) if F else 0.0
        lbl = "전체" if cut > 10 ** 8 else "msg<=%d" % cut
        susp_early[lbl] = {"pass_mean": mp, "fail_mean": mf,
                           "pass_sum": sum(P), "fail_sum": sum(F)}
        print("%-10s %22s %22s %8s"
              % (lbl, "%.2f (%d)" % (mp, sum(P)), "%.2f (%d)" % (mf, sum(F)),
                 ("%.2fx" % (mf / mp)) if mp else "-"))

    # 040 이 혐의 95건 중 41건이다 — 한 태스크가 결론을 끌고 가지 않는지 본다.
    P2, F2 = [], []
    for k, v in per.items():
        if v.get("reward") is None or L.task_of(k) == "task_040":
            continue
        (P2 if v.get("reward") else F2).append(len(_suspect_turns(k, v)))
    m2p = (sum(P2) / len(P2)) if P2 else 0.0
    m2f = (sum(F2) / len(F2)) if F2 else 0.0
    print("040 제외(혐의 95 중 41 이 040) → PASS %.2f/sim ↔ FAIL %.2f/sim = %s"
          % (m2p, m2f, ("%.2fx" % (m2f / m2p)) if m2p else "-"))
    susp_early["_excl_040"] = {"pass_mean": m2p, "fail_mean": m2f}
    step6["suspect_early"] = susp_early


    # ═════════════════════════════════════════════════════════════════════════
    # ★★태스크별 귀속 (2026-08-23 · 사용자 지시 *"통계로 하면 한 태스크의 원인이 과하게
    #   반영된다. 통계에서 태스크별 원인 귀속을 다시 하라"*).
    #
    #   근거는 바로 위 집계 자신이다: 혐의 95건 중 **41건이 040 하나**였고 그 하나가 1.81x 를
    #   만들었다. 040 을 빼면 0.88x 로 무너진다. **풀링은 정지 수로 가중되므로, 오래 헤매는
    #   태스크 하나가 코퍼스의 결론을 산다.** 통계는 *분포*를 말하지 *귀속*을 못 한다([[08]]).
    #
    #   그래서 여기서는 태스크마다 **그 태스크 자신의 실패 창** 안에서만 센다:
    #     실패 창 = `last_matched_msg`(gold 변이를 마지막으로 맞힌 메시지) 이후 ~ 궤적 끝.
    #     그 구간은 **빚진 변이가 아직 안 갚아진 구간**이고, 정지가 해를 끼칠 수 있는 유일한 곳이다.
    #     창 앞의 정지는 이미 맞힌 것을 못 막고, 창은 태스크마다 길이가 달라 풀링이 안 된다.
    #   ⚠[[23]]: gold 는 **창을 찾는 진단**으로만 쓴다(어디까지 맞혔나). 레버 선택에 안 쓴다 —
    #     이건 기존 포렌식이 gold 를 읽는 것과 같은 용법이다.
    #
    #   판정은 태스크 **안에서** 닫는다(닫힌 술어·풀링 0):
    #     무관        그 태스크의 모든 sim 에서 창내 혐의 0 — 중재는 이 실패를 안 건드렸다
    #     자체통제탈락 같은 태스크의 **통과한 sim** 이 실패 sim 만큼(이상) 창내 혐의를 받았다
    #                 ⇒ 그 패턴은 실패와 무관하다([[57]] 부정통제가 태스크 안에 있다)
    #     후보        위 둘이 아니다 — 창내 혐의가 실패 sim 에만 있다. **정독 대상**
    #     통제불가    그 태스크에 통과 sim 이 없어 태스크-내 대조가 불가능(n 부족을 숨기지 않는다)
    print("\n" + "═" * 78)
    print("★★태스크별 귀속 — 각 태스크의 **자기 실패 창** 안에서만 센다")
    print("   (위 집계는 분포이지 귀속이 아니다: 혐의 95 중 41 이 040 하나였다)")
    print("═" * 78)

    import t2_forensic as _F
    _MUT = _F.mutating_tools()
    sims_by_key = {}
    for rp in res:
        opener3 = gzip.open if rp.endswith(".gz") else io.open
        with opener3(rp, "rt", encoding="utf-8", errors="replace") as f:
            dd = json.load(f)
        for sm in (dd.get("simulations") or []):
            sims_by_key["%s|%s#s%s" % (L.run_tag(rp), sm.get("task_id"), sm.get("seed"))] = sm

    def _window_start(sim):
        """빚이 아직 남은 구간의 시작 = gold 변이를 마지막으로 맞힌 메시지 인덱스."""
        md = _F.mutation_diff(sim, _MUT)
        idx = [e.get("msg_i") for e in (md.get("matched") or []) if e.get("msg_i") is not None]
        return (max(idx) if idx else 0), md

    per_task = {}
    for k, v in per.items():
        if v.get("reward") is None or k not in sims_by_key:
            continue
        t = L.task_of(k)
        w0, md = _window_start(sims_by_key[k])
        sus = _suspect_turns(k, v)
        row = {"sim": k, "reward": v["reward"], "window_start": w0,
               "n_msgs": len(msgs.get(k) or []),
               "stops": sum(v["stops"].values()),
               "suspects": len(sus), "in_window": sorted(t2 for t2 in sus if t2 > w0),
               "missing": len(md.get("missing") or []),
               "wrongarg": len(md.get("wrongarg") or []),
               "extra": len(md.get("extra") or [])}
        per_task.setdefault(t, []).append(row)

    # ── 판정: **태스크 안에서** pass ↔ fail 을 평균으로 대조한다.
    #    ⚠`max` 로 판정하면 n 이 커질수록 이상치 하나가 태스크를 후보로 만든다(코퍼스 실행에서
    #      실제로 12 후보가 그렇게 나왔다). 평균으로 대조하고 **양 팔 최소 n**을 요구한다.
    #    ⚠창 길이가 태스크마다 다르므로 **창 100 메시지당**으로도 함께 낸다.
    #    ⚠⚠**구조적 교란**(2026-08-23 발견): 창은 *마지막으로 맞힌 gold 변이 이후*다. 통과 sim 은
    #      전부 맞혔으므로 창이 **궤적 끝에 붙어 짧다**(017: pass 창 3~15 메시지 ↔ fail 30+).
    #      그래서 **창내 건수**로 비교하면 통과 쪽이 구조적으로 0 이 나온다 — 8/10 후보의 pass 평균이
    #      정확히 0.00 이었던 것이 그 인공물이다. ⇒ 판정은 **창 100 메시지당 비율**로만 한다.
    MIN_ARM = 2
    print("%-10s %5s %5s %9s %9s %9s %9s %7s  %s"
          % ("task", "pass", "fail", "창len P", "창len F", "창100P", "창100F", "비", "판정"))
    print("-" * 96)
    verdicts = {}
    for t in sorted(per_task):
        rows = per_task[t]
        ps = [r for r in rows if r["reward"]]
        fs = [r for r in rows if not r["reward"]]

        def _mean(rs):
            return (sum(len(r["in_window"]) for r in rs) / len(rs)) if rs else 0.0

        def _per100(rs):
            w = sum(max(r["n_msgs"] - r["window_start"], 1) for r in rs)
            return 100.0 * sum(len(r["in_window"]) for r in rs) / w if rs else 0.0

        def _wlen(rs):
            return (sum(max(r["n_msgs"] - r["window_start"], 1) for r in rs) / len(rs)) if rs else 0.0

        mp, mf = _mean(ps), _mean(fs)
        rp, rf = _per100(ps), _per100(fs)
        # ★창 비교가능성 (2026-08-23): 정규화해도 **창 3 메시지 ↔ 창 45 메시지**는 못 비교한다.
        #   짧은 창은 정지를 담을 기회 자체가 없어 0 이 나오고, 그 0 이 `pass=0` 판정을 만든다
        #   (10 후보 중 8 이 그 인공물이었다). 두 팔의 평균 창이 **2배 안**일 때만 판정한다.
        wp, wf = _wlen(ps), _wlen(fs)
        _comparable = bool(wp and wf and 0.5 <= (wf / wp) <= 2.0)
        if not fs:
            vd = "전부통과"
        elif len(ps) < MIN_ARM or len(fs) < MIN_ARM:
            vd = "통제불가(n)"
        elif not _comparable:
            vd = "창비교불가"
        elif rf == 0:
            vd = "무관"
        elif rp >= rf:
            vd = "자체통제탈락"
        else:
            vd = "★후보"
        verdicts[t] = {"verdict": vd, "n_pass": len(ps), "n_fail": len(fs),
                       "mean_pass": mp, "mean_fail": mf,
                       "per100_pass": rp, "per100_fail": rf,
                       "win_pass": wp, "win_fail": wf, "comparable": _comparable,
                       "rows": rows}
        print("%-10s %5d %5d %9.1f %9.1f %9.2f %9.2f %7s  %s"
              % (t, len(ps), len(fs), wp, wf, rp, rf,
                 ("%.2fx" % (rf / rp)) if rp else "-", vd))

    order = ("★후보", "자체통제탈락", "무관", "창비교불가", "통제불가(n)", "전부통과")
    print("\n판정 집계: " + " · ".join(
        "%s %d" % (k, sum(1 for v in verdicts.values() if v["verdict"] == k)) for k in order))
    cand = [t for t, v in verdicts.items() if v["verdict"] == "★후보"]
    print("\n★후보 = 같은 태스크에서 **창 길이로 정규화한** 창내 혐의율이 실패 쪽에서 더 높은 태스크")
    if not cand:
        print("   (없음)")
    for t in cand:
        v = verdicts[t]
        print("   %-10s 창100당  pass %.2f (창 %.0f msg ×%d) ↔ fail %.2f (창 %.0f msg ×%d)  = %s"
              % (t, v["per100_pass"], v["win_pass"], v["n_pass"],
                 v["per100_fail"], v["win_fail"], v["n_fail"],
                 ("%.2fx" % (v["per100_fail"] / v["per100_pass"])) if v["per100_pass"] else "pass=0"))
    inc = [(t, v) for t, v in verdicts.items() if v["verdict"] == "창비교불가"]
    if inc:
        print("")
        print("창비교불가 %d개 — 두 팔의 창 길이가 2배를 넘어 비교가 성립하지 않는다." % len(inc))
        print("   (통과 sim 은 gold 변이를 다 맞혀 창이 궤적 끝에 붙는다 = 구조적으로 짧다)")
        for t, v in sorted(inc, key=lambda kv: -(kv[1]["per100_fail"] or 0)):
            print("   %-10s 창 P %.0f ↔ F %.0f (%.2fx) · 창100당 P %.2f ↔ F %.2f%s"
                  % (t, v["win_pass"], v["win_fail"],
                     (v["win_fail"] / v["win_pass"]) if v["win_pass"] else 0,
                     v["per100_pass"], v["per100_fail"],
                     "   ←경계(2배 문턱 바로 밖)" if v["win_pass"] and
                     0.4 <= (v["win_fail"] / v["win_pass"]) <= 2.5 else ""))
    unc = [t for t, v in verdicts.items() if v["verdict"] == "통제불가(n)"]
    if unc:
        print("\n통제불가(n) %d개 — 이 코퍼스에 한쪽 팔이 %d개 미만이라 태스크-내 대조가 안 된다: %s"
              % (len(unc), MIN_ARM, ", ".join(unc)))


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
           "step6": step6,
           "per_task_attribution": {t: {k2: v2 for k2, v2 in v.items() if k2 != "rows"}
                                    for t, v in verdicts.items()},
           "per_sim": {k: {"reward": v.get("reward"), "lines": v["lines"],
                           "stops": dict(v["stops"]), "deliveries": v["deliveries"],
                           "clobbered_bytes": v["clobbered_bytes"],
                           "suppressed": v["suppressed"],
                           "stop_turns": v["stop_turns"]}
                       for k, v in per.items()}}
    # 태그를 주면 결과 파일도 갈라 쓴다(단일 런 결과를 코퍼스 실행이 덮지 않게)
    _sfx = "" if not argv else "_corpus"
    dst = os.path.join(SIMS, "..", "x492_arbitration_ledger%s.json" % _sfx)
    with io.open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n→ %s" % os.path.normpath(dst))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
