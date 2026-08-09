# -*- coding: utf-8 -*-
r"""x205 — 한 런을 **집계가 아니라 기전으로** 읽는다 ([[08]] 포렌식 · 유료 0).

`x196` 은 *발화* 를 센다. 이 파일은 그 위에 이번 변경이 걸린 자리를 얹는다 —

  ⒜ 태스크별 보상 (표적 098·010 / **회귀** 099·100)
  ⒝ `T2_KIND` 발화·고른 종류·제외 수  ← 종류 필터가 살아 있나
  ⒞ **`value: … accepted/rejected`** ← 손님이 말한 예치액이 실제로 잡히나. 스모크 2 에서
     `model gave 1, accepted 0, rejected 1` 이라 예치 기준이 못 걸렀고 100 이 그 자리에서 졌다.
  ⒟ **계좌 원장 전사 여부** ← 안 읽으면 통과 표 자체가 없다. 스모크 1 은 도구를 unlock 만 하고
     호출하지 않은 채 KB 1위를 답으로 냈다.
  ⒠ 요구된 단계(`T2_DEMANDED_STEP`)의 머리 ← `GB3` 가 계좌 읽기를 요구한 적이 있나(지금까지 0회)

⚠**발화 ≠ 전달 ≠ 효과.** 이 표는 앞의 둘과 결과를 나란히 놓을 뿐이고, 인과는 궤적 정독이다.

실행 (리모트): python x205_run_report.py <tag>
"""
import collections
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

LOGS = os.environ.get("T2_LOGS", "/home/woori/scratch/logs")
SIMS = os.environ.get("T2_SIMS", "/home/woori/scratch/tau2-bench/data/simulations")


def main():
    if len(sys.argv) < 2:
        print("usage: x205_run_report.py <tag>")
        return 2
    tag = sys.argv[1]
    res = os.path.join(SIMS, tag, "results.json")
    tr = os.path.join(LOGS, "trace_%s.jsonl" % tag)
    fb = os.path.join(LOGS, "fb_%s.jsonl" % tag)

    print("=" * 92)
    print("%s" % tag)
    print("=" * 92)

    print("\n§1 결과")
    by = collections.defaultdict(list)
    if os.path.exists(res):
        d = json.load(open(res, encoding="utf-8"))
        for s in d.get("simulations") or []:
            by[s.get("task_id")].append(((s.get("reward_info") or {}).get("reward"),
                                         s.get("termination_reason")))
        for t in sorted(by):
            rs = by[t]
            ok = sum(1 for r, _x in rs if r == 1)
            print("  %-10s %d/%d   %s" % (t, ok, len(rs),
                                          ", ".join("%s/%s" % (r, x) for r, x in rs)))
        tot = [r for rs in by.values() for r, _x in rs]
        print("  ── 전체 %d/%d" % (sum(1 for r in tot if r == 1), len(tot)))
    else:
        print("  (결과 파일 없음: %s)" % res)

    if not os.path.exists(tr):
        print("\n⚠ trace 없음 — 기구 발화를 확인할 수 없다: %s" % tr)
        return 0
    lines = []
    for ln in open(tr, encoding="utf-8", errors="replace"):
        try:
            lines.append(json.loads(ln))
        except Exception:
            pass

    def grep(sub):
        return [x for x in lines if sub in str(x.get("line") or "")]

    print("\n§2 종류 필터 (`T2_KIND`)")
    picks = collections.Counter()
    for x in grep("[T2_KIND]"):
        s, l = x.get("sim"), str(x.get("line"))
        if "→" in l:
            picks[(s, l.split("→")[-1].strip())] += 1
    drops = collections.Counter((x.get("sim"), str(x.get("line")).split("아닌 주어")[0].split("]")[-1].strip())
                                for x in grep("아닌 주어"))
    print("  발화 %d줄 · 선택: %s" % (len(grep("[T2_KIND]")), dict(picks) or "없음"))
    for (s, k), n in sorted(drops.items()):
        print("    %-10s %-28s 제외 발화 %d회" % (s, k, n))

    print("\n§3 대화-피연산자 (손님이 말한 값)")
    acc = collections.Counter()
    for x in grep("value: model gave"):
        l = str(x.get("line"))
        acc[(x.get("sim"), l.split("value:")[-1].strip()[:70])] += 1
    if not acc:
        print("  (발화 0 — 물어본 적이 없다)")
    for (s, l), n in sorted(acc.items()):
        print("  %-10s x%d  %s" % (s, n, l))

    print("\n§4 원장 전사 (표가 만들어질 조건)")
    seen = collections.Counter()
    for x in grep("queued to view"):
        seen[(x.get("sim"), str(x.get("line")).split("]")[-1].strip().split()[0])] += 1
    for x in grep("transcription returned 0 rows"):
        seen[(x.get("sim"), str(x.get("line")).split("]")[-1].strip().split(":")[0] + " (0행)")] += 1
    for (s, t), n in sorted(seen.items()):
        print("  %-10s %-46s x%d" % (s, t, n))
    tasks = sorted(by) or sorted({x.get("sim") for x in lines if x.get("sim")})
    miss = [t for t in tasks
            if not any(s == t and "get_all_user_accounts" in k for (s, k) in seen)]
    print("  ⚠계좌 원장이 한 번도 전사되지 않은 태스크: %s" % (miss or "없음"))

    print("\n§5 요구된 단계 (`T2_DEMANDED_STEP`)")
    heads = collections.Counter(str(x.get("line")).split("head=")[-1].strip()
                                for x in grep("[T2_DEMANDED_STEP] head="))
    for h, n in heads.most_common(10):
        print("  %-70s x%d" % (h[:70], n))
    print("  ⚠계좌 읽기를 요구한 적: %d회"
          % sum(n for h, n in heads.items() if "get_all_user_accounts" in h))

    print("\n§6 우리가 **실제로 보낸** 문장 (사이드카)")
    if os.path.exists(fb):
        kinds = collections.Counter()
        for ln in open(fb, encoding="utf-8", errors="replace"):
            try:
                o = json.loads(ln)
            except Exception:
                continue
            t = str(o.get("text") or o.get("body") or "")
            for name, sig in (("통과표", "Policy constants on record"),
                              ("결정블록", "decided"), ("상태별세기", "grouped by the status"),
                              ("창산수", "Date arithmetic on the records"),
                              ("소진", "no room left this year"),
                              ("미대조", "was NOT checked against any allowance")):
                if sig in t:
                    kinds[(o.get("sim") or o.get("task") or "?", name)] += 1
        for (s, n0), n in sorted(kinds.items()):
            print("  %-10s %-10s x%d" % (s, n0, n))
        if not kinds:
            print("  (해당 문구 0 — 채널을 확인하라)")
    else:
        print("  (사이드카 없음: %s)" % fb)

    print("\n※ 발화 ≠ 전달 ≠ 효과. 원인은 궤적 정독으로만 확정한다([[08]]).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
