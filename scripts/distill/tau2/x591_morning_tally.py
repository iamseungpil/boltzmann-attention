# -*- coding: utf-8 -*-
r"""x591 - 밤샘 결과를 **아침에 한 장으로** (모델 0 · 무료 · 계수만).

## 무엇을 내나

  (1) 태스크별 pass  - 새 런 전부를 합쳐 n 을 키우고, **런별로도 따로** 찍는다.
      seed 는 재현 단위가 아니다(2026-08-28 실측: 074 s626729 가 t7376 1.0 -> t7378 0.0).
      그래서 seed 가 아니라 **런의 비율**로 읽어야 하고, 같은 반쪽을 두 번 돌린 이유가 그것이다.
  (2) 기준선 대비 - t7376(같은 10 태스크·nt2·2/20) 과 맞대어 **0->1 / 1->0** 을 갈라 찍는다.
  (3) 실패의 per-step 재료가 준비됐나 - 프롬프트 덤프가 태스크별로 몇 건 회수됐나.
      내일 격리(x575/x585/x590 계열)를 새 유료 런 없이 돌리려면 이게 있어야 한다.

⛔집계로 원인을 말하지 않는다([[08]]). 이 표는 **어디를 팔지 고르는 데만** 쓴다 -
   원인은 x587/x588/x589 가 sim#msg 로 짚는다.
"""
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_forensic as F

BASE = "/home/woori/scratch/logs"


def tally(tags):
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    for t in tags:
        try:
            sims = F.sims(t)
        except Exception as e:
            print("(못 읽음) %s : %r" % (t, e))
            continue
        for s in sims:
            per[s.get("task_id")][t].append((s.get("reward_info") or {}).get("reward"))
    return per


def main(argv=None):
    a = (argv or sys.argv[1:])
    if not a:
        print("사용: x591_morning_tally.py <새 런 태그...> [--base <기준선 태그>]")
        return 2
    base = "bank_t7376_treat_20260828"
    if "--base" in a:
        i = a.index("--base")
        base = a[i + 1]
        a = a[:i] + a[i + 2:]
    new = a
    pn = tally(new)
    pb = tally([base])

    tasks = sorted(set(pn) | set(pb))
    print("=" * 104)
    print("태스크별 pass — 새 런 %d개 ↔ 기준선 %s" % (len(new), base))
    print("=" * 104)
    print("%-11s %-12s %-12s %-9s %s" % ("태스크", "기준선", "새 런 합계", "판정", "런별"))
    up = down = flat = 0
    for t in tasks:
        b = [x for v in pb[t].values() for x in v]
        n = [x for v in pn[t].values() for x in v]
        bp, np_ = sum(1 for x in b if x == 1.0), sum(1 for x in n if x == 1.0)
        br = (bp / len(b)) if b else None
        nr = (np_ / len(n)) if n else None
        mark = "-"
        if br is not None and nr is not None:
            if nr > br: mark = "★0→1"; up += 1
            elif nr < br: mark = "⛔1→0"; down += 1
            else: mark = "불변"; flat += 1
        detail = " ".join("%s=%d/%d" % (k.split("_")[1], sum(1 for x in v if x == 1.0), len(v))
                          for k, v in sorted(pn[t].items()))
        print("%-11s %-12s %-12s %-9s %s"
              % (t, ("%d/%d" % (bp, len(b))) if b else "-",
                 ("%d/%d" % (np_, len(n))) if n else "-", mark, detail))
    print("")
    print("  올라간 태스크 %d · 내려간 태스크 %d · 불변 %d" % (up, down, flat))
    allb = [x for t in tasks for v in pb[t].values() for x in v]
    alln = [x for t in tasks for v in pn[t].values() for x in v]
    print("  총계  기준선 %d/%d  ↔  새 런 %d/%d"
          % (sum(1 for x in allb if x == 1.0), len(allb),
             sum(1 for x in alln if x == 1.0), len(alln)))

    print("")
    print("=" * 104)
    print("런-간 변이 — 같은 태스크가 런마다 다른가 (seed 가 아니라 런의 비율로 읽는다)")
    print("=" * 104)
    for t in tasks:
        rates = [(k.split("_")[1], sum(1 for x in v if x == 1.0), len(v))
                 for k, v in sorted(pn[t].items()) if v]
        if len(rates) >= 2 and len(set(r[1] for r in rates)) > 1:
            print("  %-11s %s" % (t, " ↔ ".join("%s %d/%d" % r for r in rates)))

    print("")
    print("=" * 104)
    print("내일 격리에 쓸 재료 — 프롬프트 덤프 회수 현황")
    print("=" * 104)
    for t in new:
        p = os.path.join(BASE, "fb_%s.jsonl" % t)
        if not os.path.exists(p):
            print("  %-34s (사이드카 없음)" % t)
            continue
        c = collections.Counter()
        mx = 0
        for ln in open(p, encoding="utf-8", errors="replace"):
            try:
                r = json.loads(ln)
            except Exception:
                continue
            if r.get("kind") == "prompt":
                c[r.get("simtag")] += 1
                mx = max(mx, len(r.get("text") or ""))
        print("  %-34s prompt %4d건 · sim %2d · 최대 %d자"
              % (t, sum(c.values()), len(c), mx))
        if not c:
            print("     ⚠덤프가 비었다 — T2_PROMPT_DUMP 또는 T2_FB_SIDECAR_TEXT_MAX 를 확인하라")
    print("")
    print("[읽기] 이 표는 **어디를 팔지** 고르는 데만 쓴다. 원인은 x587/x588/x589 로 sim#msg 까지 내려가라.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
