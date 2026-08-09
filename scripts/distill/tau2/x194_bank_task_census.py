# -*- coding: utf-8 -*-
r"""x194 — banking 97 태스크 전수 census: 통과 현황 × 관문 × 11셀 귀속 (유료 0).

## 왜

다음 표적을 **인상이 아니라 census 로** 고른다. 지금 손에 있는 사실 셋을 한 표에 놓는다:

  ⒜ 태스크별 통과 현황 — 최신 전수 런(N97·2026-08-06)과 이후의 표적 런들
  ⒝ **관문 요구 여부** — `get_all_user_accounts_by_user_id_3847`(신규 65 중 41이 요구·N97 §1).
     N97 시점에 이 계열은 **호출한 26 sim 전부 실패**였다. 이번 정박 슬롯 런에서 099·100 이
     그 사슬을 지나 통과했으므로, **그 0%가 아직 0% 인지**가 표적 선정의 첫 질문이다.
  ⒞ **11셀 귀속** — `t2_stack.MECHANISMS` 의 근거 칸에 적힌 태스크 번호를 모아 결손별로 센다.
     ⚠이 귀속은 **레버를 만들 때 남긴 기록**이지 전수 재부검이 아니다. 근거 칸에 안 적힌
     태스크는 `미귀속` 으로 남긴다 — 침묵을 "결손 없음"으로 읽지 않는다([[08]]).

## 무엇이 아닌가

이 스크립트는 **실패 원인을 판정하지 않는다**. 판정은 궤적 정독이고([[08]]), 여기서는
이미 판정된 것을 모으고 아직 안 된 것을 **미귀속으로 드러낼 뿐**이다.
gold 는 **진단·표적 선정에만** 쓴다 — A2 에 넣지 않는다([[23]]).

실행: python x194_bank_task_census.py [--tasks <tasks.json>] [--runs <glob> ...]
"""
import argparse
import collections
import glob
import gzip
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_stack as STACK                                        # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "..", ".."))
SIMDIR = os.path.join(REPO, "reports", "facet_rft_2026", "sim_results")
TASKS_DEFAULT = "/home/woori/scratch/tau2-bench/data/tau2/domains/banking_knowledge/tasks.json"
BASE_RUNS = ["bank_n97_gpu*_main_20260806*.results.json.gz"]
GATE_TOOL = "get_all_user_accounts_by_user_id_3847"


def load_runs(patterns):
    """{task: [reward, ...]} — 같은 태스크의 여러 런/시행을 모은다."""
    per, files = collections.defaultdict(list), []
    for pat in patterns:
        for f in sorted(glob.glob(os.path.join(SIMDIR, pat))):
            files.append(os.path.basename(f))
            d = json.load(gzip.open(f, "rt", encoding="utf-8"))
            for s in d.get("simulations", ()):
                per[s["task_id"]].append(float(s["reward_info"]["reward"]))
    return per, files


def gold_shape(task):
    """gold 가 요구하는 행동의 모양 — 도구 이름 집합과 관문 요구 여부."""
    actions = ((task.get("evaluation_criteria") or {}).get("actions")
               or task.get("actions") or ())
    names, args = [], []
    for a in actions:
        names.append(a.get("name"))
        args.append(json.dumps(a.get("arguments") or {}, ensure_ascii=False))
    blob = " ".join(n or "" for n in names) + " " + " ".join(args)
    basis = (task.get("evaluation_criteria") or {}).get("reward_basis") or []
    return names, (GATE_TOOL in blob), list(basis)


def mechanism_index(ids):
    """`t2_stack.MECHANISMS` 근거 칸에 적힌 태스크 번호 → 결손 이름 집합."""
    idx = collections.defaultdict(set)
    for cause, _mech, evid, _flag, _method in STACK.MECHANISMS:
        for num in set(re.findall(r"(?<![\w/])(\d{3})(?![\w])", evid or "")):
            tid = "task_" + num
            if tid in ids:
                idx[tid].add(cause)
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=TASKS_DEFAULT)
    ap.add_argument("--runs", nargs="*", default=BASE_RUNS)
    a = ap.parse_args()

    tasks = {t["id"]: t for t in json.load(open(a.tasks, encoding="utf-8"))}
    ids = set(tasks)
    per, files = load_runs(a.runs)
    idx = mechanism_index(ids)

    print("태스크 %d · 결과파일 %d · 궤적 %d"
          % (len(tasks), len(files), sum(len(v) for v in per.values())))
    missing = sorted(ids - set(per))
    if missing:
        print("⚠런에 없는 태스크 %d개: %s" % (len(missing), ", ".join(missing[:8])))

    rows = []
    for tid in sorted(ids):
        rw = per.get(tid, [])
        names, gate, basis = gold_shape(tasks[tid])
        n_pass = sum(1 for x in rw if x >= 1.0)
        rows.append({"id": tid, "n": len(rw), "pass": n_pass, "gate": gate,
                     "basis": basis, "tools": names,
                     "causes": sorted(idx.get(tid, ()))})

    def bucket(r):
        if not r["n"]:
            return "미측정"
        return "전패" if r["pass"] == 0 else ("전승" if r["pass"] == r["n"] else "혼합")

    print("\n§1 통과 현황 × 관문 요구")
    tab = collections.Counter((bucket(r), "관문" if r["gate"] else "비관문") for r in rows)
    for b in ("전승", "혼합", "전패", "미측정"):
        line = "  %-5s" % b
        for g in ("관문", "비관문"):
            line += "  %s %-3d" % (g, tab.get((b, g), 0))
        print(line)

    print("\n§2 11셀 귀속 (근거 칸에 적힌 태스크만 · 나머지는 미귀속)")
    cc = collections.Counter()
    for r in rows:
        if not r["causes"]:
            cc["미귀속"] += 1
        for c in r["causes"]:
            cc[c] += 1
    for c, n in cc.most_common():
        print("  %-24s %d" % (c, n))

    print("\n§3 전패 태스크 — 관문 계열 (N97 시점 이 계열은 호출해도 0%)")
    fam = [r for r in rows if bucket(r) == "전패" and r["gate"]]
    print("  n=%d: %s" % (len(fam), ", ".join(r["id"].replace("task_", "") for r in fam)))

    print("\n§4 전패 태스크 — 비관문")
    fam2 = [r for r in rows if bucket(r) == "전패" and not r["gate"]]
    print("  n=%d: %s" % (len(fam2), ", ".join(r["id"].replace("task_", "") for r in fam2)))

    print("\n§5 reward_basis 분포 (채점이 무엇을 보는가)")
    for b, n in collections.Counter("+".join(r["basis"]) or "?" for r in rows).most_common():
        print("  %-16s %d" % (b, n))

    out = os.path.join(SIMDIR, "..", "bank_task_census.json")
    json.dump(rows, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n표 저장: %s" % os.path.normpath(out))


if __name__ == "__main__":
    main()
