# -*- coding: utf-8 -*-
r"""1단계 20 태스크 **실패 단위 전수 census** — 다음 레버가 어디에 있나 (2026-08-21·오프라인·LLM 0)

사용자 물음 둘에 답하기 위한 계량:
  ⑴ *"레버가 발화했는데도 003·047·055·063·070 은 왜 실패했나 · pass 를 올릴 방법이 없나"*
  ⑵ *"원래 96 태스크의 1단계 20 태스크 돌리기로 한 것 아닌가 · 1단계 전체를 돌릴 계획을 준비하라"*

t7333 포렌식이 보인 것: 실패의 상당수가 `open_bank_account_4821` 의 **`account_class` 선택**이다.
그것은 `spend_category` 와 **같은 모양의 결정**(문서가 정의하는 목록에서 하나 고르기)인데
A2 `catalog_arg_docs` 는 **`spend_category` 하나만** 선언한다 — 엔진 루프는 이미 일반이다.
⇒ 이 census 는 **선언을 늘리기 전에 상금을 잰다**([[62]] 순서).

세는 단위 = `t2_forensic.mutation_diff` (MISSING·WRONGARG·EXTRA·DUP·BLOCKED) = reward 의 실패 단위([[69]]).
"""
import collections
import glob
import gzip
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_forensic as F   # noqa: E402

BASE = os.path.abspath(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "sim_results"))
STAGE1 = ["task_003", "task_004", "task_016", "task_017", "task_024", "task_033", "task_040",
          "task_050", "task_055", "task_057", "task_063", "task_072", "task_073", "task_074",
          "task_079", "task_085", "task_093", "task_094", "task_098", "task_100"]


def sims_of(patterns):
    out = []
    for pat in patterns:
        for p in sorted(glob.glob(os.path.join(BASE, pat))):
            try:
                d = json.load(gzip.open(p, "rt", encoding="utf-8"))
            except Exception:
                continue
            out.extend(d.get("simulations") or [])
    return out


def main():
    pats = sys.argv[1:] or ["bank_t7328_half*.results.json.gz"]
    sims = [s for s in sims_of(pats) if str(s.get("task_id") or "") in STAGE1]
    mut = F.mutating_tools()
    print("=" * 96)
    print("1단계 census · sim %d · 태스크 %d" % (len(sims), len({s.get("task_id") for s in sims})))
    print("=" * 96)

    npass = collections.Counter()
    ntot = collections.Counter()
    by_tool = collections.Counter()          # (도구, 실패종류) → 건수
    by_arg = collections.Counter()           # (도구, 인자) → gold 와 다른 값을 넣은 건수
    tasks_of_tool = collections.defaultdict(set)
    for s in sims:
        t = str(s.get("task_id") or "")
        ntot[t] += 1
        r = (s.get("reward_info") or {}).get("reward") or 0.0
        if r >= 1.0:
            npass[t] += 1
            continue
        d = F.mutation_diff(s, mut)
        for kind in ("missing", "wrongarg", "extra", "dup"):
            for x in (d.get(kind) or []):
                nm = x.get("name") or "?"
                by_tool[(nm, kind)] += 1
                tasks_of_tool[nm].add(t)
        # WRONGARG 는 **어느 인자**가 틀렸는지가 레버의 주소다
        gold_by = {}
        for g in (d.get("gold") or []):
            gold_by.setdefault(g.get("name"), []).append(g.get("args") or {})
        for w in (d.get("wrongarg") or []):
            gs = gold_by.get(w.get("name")) or [{}]
            wa = w.get("args") or {}
            for k in sorted(set(gs[0]) | set(wa)):
                if str(gs[0].get(k)) != str(wa.get(k)):
                    by_arg[(w.get("name"), k)] += 1

    print("\n[태스크별 pass]")
    for t in STAGE1:
        if ntot[t]:
            print("  %-10s %d/%d" % (t, npass[t], ntot[t]))
    print("  ── 합계 %d/%d" % (sum(npass.values()), sum(ntot.values())))

    print("\n[실패한 도구 · 종류] (reward<1 sim 만·상위 12)")
    for (nm, kind), n in by_tool.most_common(12):
        print("  %-42s %-9s %3d   태스크 %s"
              % (nm[:42], kind.upper(), n, ",".join(sorted(x[-3:] for x in tasks_of_tool[nm]))[:40]))

    print("\n★[WRONGARG 가 걸린 인자] — 이것이 다음 배달 선언의 주소다")
    for (nm, k), n in by_arg.most_common(12):
        print("  %-42s %-22s %3d" % (nm[:42], k, n))
    return 0


if __name__ == "__main__":
    sys.exit(main())
