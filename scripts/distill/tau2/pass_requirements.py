# -*- coding: utf-8 -*-
r"""**pass 요건 전수 분해** — 태스크마다 *무엇을 다 맞춰야 pass 인가* (2026-08-21·오프라인·LLM 0)

사용자 지시 축자: *"8개 태스크 정밀 포렌식해서 pass 를 위해 뭘해야 하는지 확정하고, 다음 단계로
1단계 20 태스크를 다시 정밀 포렌식해서 뭘 해야 하는지 확정하고 P1해야 한다."*

## 왜 이 모양인가
reward 는 **궤적 재실행 후 DB 해시 비교**라 gold 변이가 **전부** 맞아야 1.0 이다([[69]]). 그래서
*"이 레버가 이 변이를 고친다"* 만으로는 pass 를 못 산다 — **그 sim 의 남은 변이까지** 봐야 한다.
이 스크립트는 태스크별로 gold 변이를 펼치고, 각 변이가 팔마다 맞았는지 표시한 뒤, 안 맞은 것을
**축**으로 분류한다. 축은 우리가 가진 레버의 주소다.

## 축 분류 (엔진이 판단하지 않는다 — 인자 이름과 도구 이름만 본다·[[59]])
    CATALOG   문서가 닫힌 목록을 정의하는 인자(account_class·card_type·…) → **배달 선언**이 닿는다
    RECORD    고객 DB 레코드를 가리키는 인자(*_id·last_4·date) → 참조-격리 축(고객 DB=모델 몫·C405ⓔ)
    COMPUTE   수치 계산 결과(amount·apy·total·fee) → 값 레버 축(C562)
    DUP       같은 변이를 두 번 성공시킴 → 중복 실행 축(050 형)
    EXTRA     gold 에 없는 변이를 성공시킴 → 월권 축
    MISSING_CALL 그 도구를 아예 안 불렀다 → 완결/커버리지 축
    BLOCKED   시도했으나 거절당함 → deny 주체를 함께 본다(env 인지 우리 층인지)
"""
import collections
import glob
import gzip
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_forensic as F   # noqa: E402

BASE = os.path.abspath(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "sim_results"))

_RECORD = re.compile(r"(_id$|^id$|last_4|last4|date$|_date|email|phone|address)", re.I)
_COMPUTE = re.compile(r"(amount|apy|total|fee|balance|rate|difference|income|limit)", re.I)


def axis_of(tool, arg):
    if arg is None:
        return "MISSING_CALL"
    if _RECORD.search(arg):
        return "RECORD"
    if _COMPUTE.search(arg):
        return "COMPUTE"
    return "CATALOG"


def sims_of(patterns):
    out = []
    for pat in patterns:
        for p in sorted(glob.glob(os.path.join(BASE, pat))):
            try:
                d = json.load(gzip.open(p, "rt", encoding="utf-8"))
            except Exception:
                continue
            for s in (d.get("simulations") or []):
                out.append((os.path.basename(p), s))
    return out


def analyse(sim, mut):
    """이 sim 이 pass 하려면 **무엇이 더 맞아야 했나** — (축, 도구, 인자) 목록."""
    d = F.mutation_diff(sim, mut)
    need = []
    gold_by = {}
    for g in (d.get("gold") or []):
        gold_by.setdefault(g.get("name"), []).append(g.get("args") or {})
    wrong_by = collections.defaultdict(list)
    for w in (d.get("wrongarg") or []):
        wrong_by[w.get("name")].append(w.get("args") or {})
    for m in (d.get("missing") or []):
        nm = m.get("name")
        ws = wrong_by.get(nm)
        if not ws:                      # 도구 자체를 안 불렀다
            need.append(("MISSING_CALL", nm, None))
            continue
        ga = m.get("args") or {}
        for k in sorted(set(ga) | set(ws[0])):
            if str(ga.get(k)) != str(ws[0].get(k)):
                need.append((axis_of(nm, k), nm, k))
    for x in (d.get("extra") or []):
        need.append(("EXTRA", x.get("name"), None))
    for x in (d.get("dup") or []):
        need.append(("DUP", x.get("name"), None))
    for b in (d.get("blocked") or []):
        need.append(("BLOCKED:%s" % str(b.get("deny") or "?")[:12], b.get("name"), None))
    return need, len(d.get("gold") or [])


def main():
    pats = sys.argv[1:] or ["bank_t7333_*_20260821c.results.json.gz"]
    rows = sims_of(pats)
    mut = F.mutating_tools()
    by_task = collections.defaultdict(list)
    for src, s in rows:
        by_task[str(s.get("task_id") or "")].append((src, s))

    axis_tally = collections.Counter()
    task_axes = collections.defaultdict(collections.Counter)
    print("=" * 100)
    print("pass 요건 분해 · sim %d · 태스크 %d" % (len(rows), len(by_task)))
    print("=" * 100)
    for t in sorted(by_task):
        group = by_task[t]
        npass = sum(1 for _s, x in group if ((x.get("reward_info") or {}).get("reward") or 0) >= 1.0)
        print("\n### %s   pass %d/%d" % (t, npass, len(group)))
        for src, s in group:
            r = (s.get("reward_info") or {}).get("reward") or 0.0
            arm = "treat" if "_treat_" in src else ("ctl" if "_ctl_" in src else "val")
            if r >= 1.0:
                print("   %-6s t%-2s ✓" % (arm, s.get("trial")))
                continue
            need, ngold = analyse(s, mut)
            seen = []
            for ax, nm, arg in need:
                key = (ax, nm, arg)
                if key in seen:
                    continue
                seen.append(key)
                axis_tally[ax] += 1
                task_axes[t][ax] += 1
            print("   %-6s t%-2s ✗ gold변이 %d · 남은 요건 %d: %s"
                  % (arm, s.get("trial"), ngold, len(seen),
                     " | ".join("%s:%s%s" % (a, (n or "")[:26], ("." + g) if g else "")
                                for a, n, g in seen[:5])))
    print("\n" + "=" * 100)
    print("[축별 총계] — 우리 레버의 주소")
    for ax, n in axis_tally.most_common():
        print("  %-22s %d" % (ax, n))
    print("\n[태스크별 축]")
    for t in sorted(task_axes):
        print("  %-10s %s" % (t, " ".join("%s=%d" % (a, n) for a, n in task_axes[t].most_common())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
