# -*- coding: utf-8 -*-
r"""x460 — **채점 축을 먼저 읽고** 실패를 분해한다 (2026-08-21·오프라인·LLM 0·무료)

## 왜 (내 계기가 한 축만 보고 있었다)
`x458`/`x459` 는 `t2_forensic.mutation_diff` 로 실패를 읽었다. 그것은 **DB 변이 축**의 도구다 —
`gold_mutations` 가 GRANTS(unlock·give)와 비-변이 도구를 **설계상 걸러낸다**([[69]] 재실행 해시).

그런데 `task_033` 은 gold 액션이 `unlock_discoverable_agent_tool`×2 · `call_discoverable_agent_tool`×2
· `transfer_to_human_agents` 이고 **`reward_basis = ["ACTION"]`** 이다. 전부 DB 를 안 바꾸므로
변이 축에서는 **"clean"** 으로 보이고, 실제로는 `action_checks` 5개 중 4~5개가 False 다.
⇒ *"변이 집합이 맞는데 reward 0"* 은 모델 현상이 아니라 **내가 다른 자를 댄 것**이었다([[55]]).

## 무엇을 하나
sim 마다 **`reward_basis` 를 먼저 읽고** 그 축으로 분해한다. 축은 벤치가 선언한 것이고
우리가 고르지 않는다.

    ACTION       `action_checks` 를 항목별로 — 어느 `action_id` 가 False 인가, 래퍼면 대상 도구까지
    DB           `db_check.db_match` + 변이 diff(`mutation_diff`)
    NL           `nl_assertions` 항목별
    COMMUNICATE  `communicate_checks` 항목별
    ENV          `env_assertions`

## 채점 (닫힌 술어만)
벤치가 남긴 `action_match` · `db_match` 를 **그대로 읽는다**. 우리가 판정하지 않는다.
gold 인자 대조는 **진단 라벨**로만 쓴다([[23]]·`x451` 동형).

사용: py x460_reward_basis_census.py [--tags ...]
"""
import argparse
import collections
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

import t2_forensic as F                 # noqa: E402  정본(사본 금지·[[67]])

REP = os.path.abspath(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026"))
NLC = chr(10)


def action_rows(sim):
    """`action_checks` 를 항목별로 — 래퍼면 대상 도구 이름까지 편다."""
    out = []
    for ck in ((sim.get("reward_info") or {}).get("action_checks") or []):
        a = ck.get("action") or {}
        ar = a.get("arguments") or {}
        outer = str(a.get("name") or "")
        inner = str(F.inner_name(ar) or "")
        out.append({"id": str(a.get("action_id") or ""), "match": bool(ck.get("action_match")),
                    "tool": outer, "inner": inner,
                    "args": json.dumps(F.flat_args(ar), ensure_ascii=False)[:120]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="")
    ap.add_argument("--out", default="x460_reward_basis.json")
    a = ap.parse_args()

    tags = [t for t in a.tags.split(",") if t.strip()] or \
        [F.tag_of_file(p) for p in F.all_result_files() if "t7328" in os.path.basename(p)]
    sims = []
    for t in sorted(set(tags)):
        try:
            sims += list(F.sims(t))
        except Exception:
            pass
    print("=" * 100)
    print("x460 · sim %d · 태그 %s" % (len(sims), ", ".join(sorted(set(tags)))))
    print("=" * 100)

    per = collections.defaultdict(lambda: {"n": 0, "pass": 0, "basis": collections.Counter(),
                                           "dbfail": 0, "afail": collections.Counter(),
                                           "atot": 0, "amiss": 0, "rows": []})
    for s in sims:
        ri = s.get("reward_info") or {}
        tid = F.task_id(s)
        p = per[tid]
        p["n"] += 1
        rw = ri.get("reward")
        p["pass"] += 1 if rw in (1, 1.0) else 0
        for b in (ri.get("reward_basis") or []):
            p["basis"][str(b)] += 1
        if (ri.get("db_check") or {}).get("db_match") is False:
            p["dbfail"] += 1
        ars = action_rows(s)
        p["atot"] += len(ars)
        for r in ars:
            if not r["match"]:
                p["amiss"] += 1
                p["afail"][r["inner"] or r["tool"]] += 1
        p["rows"].append({"sim": F.sim_key(s), "reward": rw,
                          "basis": ri.get("reward_basis"),
                          "breakdown": ri.get("reward_breakdown"),
                          "db_match": (ri.get("db_check") or {}).get("db_match"),
                          "n_nl": len(ri.get("nl_assertions") or []),
                          "n_comm": len(ri.get("communicate_checks") or []),
                          "actions": ars})

    print("%-10s %-5s %-16s %-6s %-8s %s"
          % ("task", "pass", "reward_basis", "db✗", "action✗", "실패한 gold 액션(대상 도구)"))
    for tid in sorted(per):
        p = per[tid]
        b = ",".join(sorted(p["basis"]))
        f = ", ".join("%s×%d" % (k[:26] or "?", v) for k, v in p["afail"].most_common(3))
        print("%-10s %d/%-3d %-16s %-6s %-8s %s"
              % (tid, p["pass"], p["n"], b[:16], "%d/%d" % (p["dbfail"], p["n"]),
                 "%d/%d" % (p["amiss"], p["atot"]), f[:64]))

    print(NLC + "[채점 축 분포] " + ", ".join(
        "%s=%d태스크" % (b, sum(1 for p in per.values() if b in p["basis"]))
        for b in sorted({b for p in per.values() for b in p["basis"]})))

    # ★변이 축으로는 안 보이던 태스크 — ACTION 축인데 gold 가 전부 비-변이
    mut = F.mutating_tools()
    blind = []
    for tid, p in sorted(per.items()):
        names = {r["inner"] or r["tool"] for row in p["rows"] for r in row["actions"]}
        if names and not (names & mut) and p["pass"] < p["n"]:
            blind.append((tid, sorted(names)))
    print(NLC + "★변이 축 계기가 놓치는 태스크(gold 액션에 상태변경 도구가 하나도 없다) %d개" % len(blind))
    for tid, names in blind:
        print("   %-10s %s" % (tid, ", ".join(n[:34] for n in names)[:88]))

    p2 = os.path.join(REP, a.out)
    with io.open(p2, "w", encoding="utf-8") as f:
        json.dump({"tags": sorted(set(tags)),
                   "per_task": {k: {kk: (dict(vv) if isinstance(vv, collections.Counter) else vv)
                                    for kk, vv in v.items()} for k, v in per.items()}},
                  f, ensure_ascii=False, indent=1)
    print(NLC + "→ %s" % p2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
