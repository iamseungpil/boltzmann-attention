# -*- coding: utf-8 -*-
"""예산을 없애면 명령이 실제로 전진하는가 — 저장된 궤적으로, 라이브 없이.

`requirements_for`는 호출될 때마다 `done`을 다시 계산하고 `first_step`으로 뿌리까지 내려간다.
즉 **프런티어 재계산은 이미 구현돼 있다.** 101에서 그것이 한 번도 다음 층으로 올라가지 못한 이유는
발화가 turn 4~8에서 cap으로 끝나 **재계산이 다시 실행되지 않았기** 때문이다(부검 §7b).

그러면 cap을 없앤 것만으로 충분한가, 아니면 루프를 따로 만들어야 하는가? 그 물음은 유료 런 없이
답할 수 있다 — 저장된 궤적의 **각 assistant 턴에서** 같은 함수를 불러 보면 된다([[09]]).

각 턴마다 찍는다:
  done        그 시점까지 성공한 호출 (엔진과 같은 규칙: A2 `failure_markers` 접두 = 실패)
  NOW         그 시점에 `requirements_for`가 "지금 하라"로 낼 단계
  전진 여부   NOW가 이전 턴과 달라졌는가

기대(부검 §7b의 처방이 맞다면): 신원 확인이 성공한 턴 **이후**에 NOW가 `get_referrals_by_user`로
바뀌는 턴이 존재한다. 현재 라이브에서 그 명령의 실측 횟수는 **0**이다.

usage: x123_frontier_replay.py --dirs a,b --tasks task_101 [--target submit_referral] [--show]
"""

import glob
import io
import json
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
TAU2 = os.environ.get("GO_TAU2", "/home/woori/scratch/tau2-bench")
SIMBASES = [os.path.join(TAU2, "data", "simulations"),
            os.path.join(REPO, "reports", "facet_rft_2026", "sim_results")]

import t2_dominance as DOM                                             # noqa: E402
from gate_interpreter import load_domain_a2                            # noqa: E402


def arg(n, d=None):
    return sys.argv[sys.argv.index(n) + 1] if n in sys.argv else d


DIRS = [d for d in (arg("--dirs") or "").split(",") if d]
TASKS = [t for t in (arg("--tasks") or "task_101,task_102").split(",") if t]
TARGET = arg("--target", "submit_referral")
SHOW = "--show" in sys.argv


def load_sims():
    out = []
    for base in SIMBASES:
        for d in DIRS:
            for p in glob.glob(os.path.join(base, d, "results.json")):
                with io.open(p, encoding="utf-8", errors="replace") as fh:
                    for s in (json.load(fh).get("simulations") or []):
                        s["_src"] = d
                        out.append(s)
    return out


def eff_name(tc):
    n = tc.get("name") or (tc.get("function") or {}).get("name") or ""
    return str(n)


def replay(sim, a2, marks):
    """엔진과 같은 규칙으로 done을 키우며, 각 assistant 턴의 NOW를 낸다."""
    done, pending, rows = set(), {}, []
    turn = 0
    for m in sim.get("messages") or []:
        role = m.get("role")
        if role == "tool":
            nm = pending.get(m.get("id") or m.get("tool_call_id")
                             or m.get("requestor_tool_call_id"))
            txt = str(m.get("content") or "").lstrip()
            failed = (m.get("error") or txt.startswith("Error:")
                      or any(txt.startswith(k) for k in marks))
            if nm and not failed:
                done.add(nm)
            continue
        for tc in (m.get("tool_calls") or []):
            pending[tc.get("id")] = eff_name(tc)
        if role != "assistant":
            continue
        turn += 1
        reqs = DOM.requirements_for(a2, [], TARGET, executed=set(done))
        now = (reqs[0]["satisfiers"][0] if reqs and reqs[0].get("satisfiers") else None)
        rows.append((turn, now, [r["id"] for r in reqs], sorted(done)))
    return rows


def main():
    a2 = load_domain_a2("banking_knowledge") or {}
    marks = tuple(a2.get("failure_markers") or ())
    print("A2 gates=%d · failure_markers=%d · prereq edges=%d"
          % (len(a2.get("gates") or []), len(marks), len(DOM.prereq_map(a2))))
    if not marks:
        print("⚠failure_markers 0 — 실패 판정이 엔진과 달라진다. A2 로드를 먼저 확인하라.")
    sims = load_sims()
    tot = {"sim": 0, "원장이_NOW가_된_sim": 0, "전진횟수": 0}
    for tid in TASKS:
        for s in sorted([x for x in sims if x.get("task_id") == tid],
                        key=lambda x: (x["_src"], x.get("trial") or 0)):
            rows = replay(s, a2, marks)
            seq, first_ledger = [], None
            prev = object()
            for turn, now, ids, done in rows:
                if now != prev:
                    seq.append((turn, now))
                    prev = now
                if now == "get_referrals_by_user" and first_ledger is None:
                    first_ledger = turn
            tot["sim"] += 1
            tot["전진횟수"] += max(0, len(seq) - 1)
            if first_ledger:
                tot["원장이_NOW가_된_sim"] += 1
            print("  [%-22s t%s] NOW 변화: %s%s"
                  % (s["_src"], s.get("trial"),
                     " → ".join("턴%d:%s" % (t, n or "(없음=표적 개방)") for t, n in seq),
                     ("   ★원장 명령 가능 턴=%d" % first_ledger) if first_ledger else ""))
            if SHOW:
                for turn, now, ids, done in rows:
                    print("      턴%-3d NOW=%-26s reqs=%s done=%s"
                          % (turn, now, ",".join(ids), ",".join(done)))
    print("  ── %s" % tot)


if __name__ == "__main__":
    main()
