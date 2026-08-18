#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""x375 — **단계 전수 포렌식**(staged census 공용·사용자 지시 2026-08-18 *"단계별로 전수 포렌식"*).

집계에서 결론 직행 금지([[08]]). 이 스크립트가 매 단계 내는 것:

  ① sim 별 표 — 태스크 · **W그룹**(재현 가능 축) · reward · 종료사유 · steps · 최대 turn
  ② **W그룹 × 팔** 성적 교차표
  ③ **레버 × 팔** 발화 — 각 레버를 **자기 트리거 자리 대비**로(이웃 마커 분모 금지·C532⒝)
  ④ 종료사유 분포 · 크래시/CWE
  ⑤ **turn 단위**: 레버가 언제 떴는가 · 첫 write 시도는 언제인가(순서 채널)
  ⑥ 요구-인용 검증(통과/기각·기각 축자) — C533 이 연 자리

⚠판정 규율: pass 는 `reward` 로만(C486) · 발화율 분모는 **술어의 자리**에서 · sim·turn 은
  `t2_forensic.trace`/`turns_of`(2026-08-18 정본화: trace 가 turn 을 99.3% 갖고 있다).

사용: python x375_stage_forensic.py <ctl_tag> <treat_tag>
"""
import collections
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t2_forensic as F  # noqa: E402

SUF = ".results.json.gz"
CTL, TREAT = (sys.argv[1:3] + [None, None])[:2]
GRP = json.load(io.open(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026",
                                     "TASK_GROUPS_W_2026_08_18.json"), encoding="utf-8"))
T2G = GRP["task_to_group"]
GLBL = {g: "+".join(d["acts"]) or "(없음)" for g, d in GRP["groups"].items()}


def rows(tag):
    out = []
    for s in F.scored(tag, SUF):
        k = F.simtag(s)
        out.append(dict(task=F.task_id(s), key=k, grp=T2G.get(F.task_id(s), "?"),
                        reward=(s.get("reward_info") or {}).get("reward"),
                        term=F.term_reason(s), steps=len(s.get("messages") or []), sim=s))
    return sorted(out, key=lambda r: r["task"])


def maxturn(tag):
    mx = collections.Counter()
    for d in F.trace(tag):
        s, t = d.get("sim"), d.get("turn")
        if s and isinstance(t, int) and t > mx[s]:
            mx[s] = t
    return mx


def main():
    print("=" * 92)
    print("x375 — 단계 전수 포렌식  ctl=%s  treat=%s" % (CTL, TREAT))
    print("=" * 92)

    data = {a: rows(t) for a, t in (("ctl", CTL), ("treat", TREAT))}
    mts = {a: maxturn(t) for a, t in (("ctl", CTL), ("treat", TREAT))}

    print("\n① sim 별 표")
    print("  %-4s %-9s %-5s %-42s %-7s %-18s %6s %6s" %
          ("팔", "태스크", "W그룹", "그룹 정의(gold write 집합)", "reward", "종료사유", "steps", "maxturn"))
    for a in ("ctl", "treat"):
        for r in data[a]:
            print("  %-4s %-9s %-5s %-42s %-7s %-18s %6d %6d" %
                  (a, r["task"], r["grp"], GLBL.get(r["grp"], "?")[:42], r["reward"],
                   r["term"], r["steps"], mts[a].get(r["key"], 0)))

    print("\n② W그룹 × 팔 성적")
    gs = sorted({r["grp"] for a in data for r in data[a]})
    print("  %-5s %-42s %10s %10s" % ("W그룹", "정의", "ctl", "treat"))
    for g in gs:
        c = [r for r in data["ctl"] if r["grp"] == g]
        t = [r for r in data["treat"] if r["grp"] == g]
        print("  %-5s %-42s %10s %10s" % (g, GLBL.get(g, "?")[:42],
              "%d/%d" % (sum(1 for r in c if r["reward"]), len(c)),
              "%d/%d" % (sum(1 for r in t if r["reward"]), len(t))))

    print("\n③ 레버 × 팔 발화 (분모 = **그 술어의 자리**)")
    for a, tag in (("ctl", CTL), ("treat", TREAT)):
        subreq = F.by_sim(tag, r"\[T2_SUB_REQUIREMENT\] 인용 \d+개 중 원문 검증 통과 \d+개")
        trig = set()
        for k, hits in subreq.items():
            for _i, ln in hits:
                m = re.search(r"통과 (\d+)개", ln if isinstance(ln, str) else "")
                if m and int(m.group(1)) > 0:
                    trig.add(k)
        vc = set(F.by_sim(tag, r"\[T2_VERDICT\]").keys())
        el = set(F.by_sim(tag, r"\[T2_ELIG\]").keys())
        go = set(F.by_sim(tag, r"\[T2_GROUPORDER\]").keys())
        keys = {r["key"] for r in data[a]}
        print("  [%s] VERDICT %d/%d(트리거=요구인용 통과 sim) · ELIG %d/%d(트리거=군 결정 sim) · "
              "GROUPORDER %d" % (a, len(vc & trig), len(trig), len(el & keys), len(go & keys),
                                 len(go & keys)))

    print("\n④ 종료사유 · 부작용")
    for a, tag in (("ctl", CTL), ("treat", TREAT)):
        term = collections.Counter(r["term"] for r in data[a])
        cwe = len(F.by_sim(tag, r"context_window_exceeded"))
        tb = len(F.by_sim(tag, r"Traceback"))
        print("  [%s] %s · CWE sim %d · Traceback sim %d" % (a, dict(term), cwe, tb))

    print("\n⑤ turn 단위 — 레버 발화 turn ↔ 첫 write 시도 turn")
    for a, tag in (("ctl", CTL), ("treat", TREAT)):
        vt = F.turns_of(tag, r"\[T2_VERDICT\]")
        et = F.turns_of(tag, r"\[T2_ELIG\]")
        wt = F.turns_of(tag, r"\[T2_WRITE_SUB\]")
        ks = sorted({r["key"] for r in data[a]})
        for k in ks:
            v = sorted(x for x in (vt.get(k) or []) if x is not None)
            e = sorted(x for x in (et.get(k) or []) if x is not None)
            w = sorted(x for x in (wt.get(k) or []) if x is not None)
            if v or e or w:
                print("  [%s] %-22s ELIG@%s · VERDICT@%s · WRITE_SUB@%s"
                      % (a, k, e[:4] or "-", v[:4] or "-", w[:4] or "-"))

    print("\n⑥ 요구-인용 검증 (C533 이 연 자리)")
    for a, tag in (("ctl", CTL), ("treat", TREAT)):
        hits = F.by_sim(tag, r"\[T2_SUB_REQUIREMENT\]")
        tot = ok = rej = 0
        rejtxt = []
        for k, hs in hits.items():
            for _i, ln in hs:
                m = re.search(r"인용 (\d+)개 중 원문 검증 통과 (\d+)개", ln if isinstance(ln, str) else "")
                if not m:
                    continue
                tot += int(m.group(1)); ok += int(m.group(2))
                r2 = re.search(r"기각 \d+: (.+)$", ln)
                if r2:
                    rej += 1; rejtxt.append((k, r2.group(1)[:90]))
        print("  [%s] 인용 %d · 통과 %d · 기각줄 %d" % (a, tot, ok, rej))
        for k, t in rejtxt[:6]:
            print("        %s | %s" % (k, t))

    print("\n" + "=" * 92)
    print("⚠pass 는 `reward` 로만(C486) · 발화율 분모는 술어의 자리(C532⒝) · 결론 전 궤적 정독([[08]])")
    print("=" * 92)


if __name__ == "__main__":
    main()
