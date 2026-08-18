# -*- coding: utf-8 -*-
r"""x391 — **우리 가드가 무엇을 막았나** 전수 census (사용자 지시 2026-08-19: 자기차단 먼저).

## 왜
t7326 에서 `operator-fab` 43회 · `T2_UNLOCK_PROV deny` 3회가 찍혔다. 어제 실측으로 그 계열이
**우리가 지목한 이름을 우리가 막는** 형태였음이 확인됐고(050 이 그렇게 갈렸다), 지금 0/2 실패가
14 태스크다. 그 43회가 **무엇을 막았고 그 뒤 어떻게 됐는지**를 궤적과 맞대야 표적이 정해진다.

## 무엇을 세나 (결정론·LLM 0)
로그의 deny 줄에서 `sim` · 도구 · 값을 뽑고, 같은 sim 의 궤적에서
  ⒜ 그 값이 **나중에 실행됐는가**(막혔어도 스스로 회복했나)
  ⒝ 그 값이 **gold 액션에 있는가**(막은 것이 정답이었나 = 오차단)
  ⒞ 그 sim 의 reward
를 붙인다. ⚠판정이 아니라 **표**다 — 오차단 여부는 ⒝ 가 말하고, 비용은 ⒜ 가 말한다.

사용: py -3 x391_guard_selfblock_census.py <tag> [<tag> …]
"""
import collections
import io
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

LOGDIR = "/home/woori/scratch/logs"
SIMDIR = "/home/woori/scratch/tau2-bench/data/simulations"

SIM_RE = re.compile(r"\[sim=(task_\d+)#s(\d+)\]")
FAB_RE = re.compile(r"\[T2_RESOLVE\] deny tool=(\S+) arg=(\S+) reason=(\S+)")
UNL_RE = re.compile(r"\[T2_UNLOCK_PROV\] deny unprovenanced name \(followup-regen\) "
                    r"tool=(\S+) val=(\S+)")


def gold_tools(sim):
    out = set()
    for a in ((sim.get("reward_info") or {}).get("action_checks") or []):
        act = a.get("action") or {}
        args = act.get("arguments") or {}
        out.add(str(args.get("agent_tool_name") or act.get("name") or ""))
    return {x for x in out if x}


def called_tools(sim):
    out = collections.Counter()
    for m in sim.get("messages") or []:
        for tc in (m.get("tool_calls") or []):
            a = tc.get("arguments") or {}
            if isinstance(a, str):
                try:
                    a = json.loads(a)
                except Exception:
                    a = {}
            out[str((a or {}).get("agent_tool_name") or tc.get("name") or "")] += 1
    return out


def main():
    tags = sys.argv[1:]
    if not tags:
        print("usage: x391_guard_selfblock_census.py <tag> ...")
        return 2

    sims = {}
    for t in tags:
        p = os.path.join(SIMDIR, t, "results.json")
        if not os.path.exists(p):
            continue
        d = json.load(io.open(p, encoding="utf-8", errors="replace"))
        for s in d.get("simulations") or []:
            sims[(t, str(s.get("task_id")), str(s.get("trial")))] = s

    rows = []
    for t in tags:
        lp = os.path.join(LOGDIR, t + ".log")
        if not os.path.exists(lp):
            continue
        cur = None
        for line in io.open(lp, encoding="utf-8", errors="replace"):
            m = SIM_RE.search(line)
            if m:
                cur = m.group(1)
            f = FAB_RE.search(line)
            u = UNL_RE.search(line)
            if f and cur:
                rows.append((t, cur, "operator-" + f.group(3).split("-")[-1], f.group(1), ""))
            elif u and cur:
                rows.append((t, cur, "unlock_prov", u.group(1), u.group(2)))

    print("자기차단 %d건 · 런 %d개" % (len(rows), len(tags)))
    by_task = collections.Counter(r[1] for r in rows)
    by_kind = collections.Counter(r[2] for r in rows)
    print("종류별:", dict(by_kind))
    print("태스크별:", dict(by_task.most_common(12)))

    print("\n=== 값이 잡힌 건(unlock_prov) — 그 이름이 gold 인가 / 나중에 실행됐나 ===")
    seen = 0
    for t, task, kind, tool, val in rows:
        if not val:
            continue
        seen += 1
        for (tg, tk, tr), s in sims.items():
            if tg != t or tk != task:
                continue
            g = gold_tools(s)
            c = called_tools(s)
            print("  %-8s %-12s val=%-40s gold=%-5s 이후실행=%-5s reward=%s"
                  % (kind, task, val[:40], val in g, bool(c.get(val)),
                     (s.get("reward_info") or {}).get("reward")))
            break
    if not seen:
        print("  (값이 인쇄되는 종류의 deny 는 없었다 — operator-fab 줄은 값을 안 찍는다)")

    print("\n=== operator-fab 이 뜬 sim 의 결과 분포 ===")
    fab_tasks = {r[1] for r in rows if r[2].startswith("operator")}
    dist = collections.Counter()
    for (tg, tk, tr), s in sims.items():
        if tk in fab_tasks:
            dist[(s.get("reward_info") or {}).get("reward")] += 1
    print("  ", dict(dist), "· 해당 태스크", sorted(x.replace("task_", "") for x in fab_tasks))
    return 0


if __name__ == "__main__":
    sys.exit(main())
