"""Does the give-execution nudge mistake a customer who ran the tool for one who did not?

`T2_GIVE_EXEC_NUDGE` (t2_gate_patch.py:5907) decides "we handed it over and they never
ran it" from two sets: tools handed over, and tools run. It builds the second one by
looking for calls named `dispatcher_role_check.user_call` and reading their
`discoverable_tool_name` argument.

But a customer execution does not always look like that. In the persisted runs the
customer's calls sit in `role=user` messages carrying the tool's own name directly —
the same shape that made an analyzer report 18/18 sims as "never handed over" on
2026-08-04 by reading the wrong key. If the engine has the same blind spot, then every
sim where the customer actually ran the tool still counts as idle, and the nudge tells
them to do what they already did.

This counts both shapes over both arms and reports how many nudge decisions would flip
once direct executions are counted. Predicate only — no fix is applied here.
"""

import argparse
import collections
import glob
import gzip
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SIM_LOCAL = os.path.join(REPO, "reports", "facet_rft_2026", "sim_results")
SIM_REMOTE = ("/home/woori/workspace_common/boltzmann-attention-pi/"
              "reports/facet_rft_2026/sim_results")
ARMS = {"A": "bank_ax33n_gpu*_20260803g", "B4": "bank_b4_gpu*_20260803h"}


def norm(a):
    if isinstance(a, str):
        try:
            return json.loads(a)
        except Exception:
            return {}
    return a if isinstance(a, dict) else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="A,B4")
    ap.add_argument("--a2", default=os.path.join(HERE, "a2", "banking_knowledge.gate.json"))
    ap.add_argument("--sim", default="")
    ap.add_argument("--detail", action="store_true")
    args = ap.parse_args()

    sim_dir = args.sim or (SIM_LOCAL if os.path.isdir(SIM_LOCAL) else SIM_REMOTE)
    a2 = json.load(open(args.a2, encoding="utf-8")) if os.path.exists(args.a2) else {}
    drc = (a2.get("dispatcher_role_check") or {})
    give_tool, user_call = drc.get("give_tool"), drc.get("user_call")
    print(f"sim_dir = {sim_dir}")
    print(f"A2 give_tool={give_tool!r}  user_call={user_call!r}")

    for arm in args.arms.split(","):
        print(f"\n{'=' * 78}\n[{arm}]  {ARMS[arm]}\n{'=' * 78}")
        sims = []
        for p in sorted(glob.glob(os.path.join(sim_dir, f"{ARMS[arm]}.results.json.gz"))):
            sims.extend(json.load(gzip.open(p, "rt", encoding="utf-8")).get("simulations") or [])

        t = collections.Counter()
        rows = []
        for s in sorted(sims, key=lambda x: (x["task_id"], x.get("trial") or 0)):
            ok = ((s.get("reward_info") or {}).get("reward") or 0.0) == 1.0
            side = "pass" if ok else "fail"
            given, g2n, ran_dispatch, ran_direct = set(), {}, set(), set()
            for m in s.get("messages") or []:
                role = m.get("role")
                for tc in m.get("tool_calls") or []:
                    nm = tc.get("name") or (tc.get("function") or {}).get("name") or ""
                    a = tc.get("arguments")
                    if a is None:
                        a = (tc.get("function") or {}).get("arguments")
                    inner = str(norm(a).get("discoverable_tool_name") or "")
                    if nm == give_tool:
                        g2n[tc.get("id")] = inner
                    elif nm == user_call and inner:
                        ran_dispatch.add(inner)
                    elif role == "user" and nm:
                        # The shape the engine does not look for: the customer running
                        # the handed-over tool under its own name.
                        ran_direct.add(nm)
                if role == "tool" and not m.get("error"):
                    n = g2n.get(m.get("id"))
                    if n:
                        given.add(n)

            if not given:
                continue
            t[f"give 성사 sim·{side}"] += 1
            idle_now = given - ran_dispatch                 # 엔진의 현 판정
            idle_fixed = given - ran_dispatch - ran_direct  # 직접 실행까지 센 판정
            if idle_now:
                t[f"현행 넛지 발화·{side}"] += 1
            if idle_now and not idle_fixed:
                t[f"★오발화(손님은 실행했다)·{side}"] += 1
                rows.append((f"{s['task_id']}/t{s.get('trial')}", side,
                             sorted(given), sorted(ran_direct)))

        print(f"\n  {'':34s} {'fail':>6s} {'pass':>6s}")
        for k in ("give 성사 sim", "현행 넛지 발화", "★오발화(손님은 실행했다)"):
            print(f"  {k:34s} {t[k + '·fail']:6d} {t[k + '·pass']:6d}")
        n_bad = t["★오발화(손님은 실행했다)·fail"] + t["★오발화(손님은 실행했다)·pass"]
        n_fire = t["현행 넛지 발화·fail"] + t["현행 넛지 발화·pass"]
        print(f"\n  발화 {n_fire} 중 오발화 {n_bad}"
              + ("  ⇒ ★술어 결함 확정" if n_bad else "  ⇒ 이 축의 결함은 재현 안 됨"))
        if args.detail:
            for key, side, g, r in rows:
                print(f"    {key:16s} {side:4s} given={g} user_ran_direct={r[:4]}")


if __name__ == "__main__":
    main()
