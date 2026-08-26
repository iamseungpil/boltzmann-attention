# -*- coding: utf-8 -*-
r"""x544 - 반증: 074 의 `db_match=False` 가 **중복 실행** 때문인가. 중복만 빼고 다시 채점한다.

## 왜 (2026-08-26 · t7358 074)

`reward_info` 는 `action_checks` **13/13 일치** 인데 `db_check.db_match=False` 다(basis=['DB']).
궤적을 보면 같은 크레딧 4건이 **두 번** 실행됐다(msg 82/85/87/89 → msg 97 에 4건 한꺼번에).
env 구현(`tools.py:2730 apply_checking_account_credit_5829`)은

    new_balance = current_balance + amount          # 부를 때마다 **더한다**
    transaction_id = txn_{_deterministic_id(...)}   # 인자로부터 결정되는 id
    history.data[transaction_id] = record           # 두 번째가 첫 번째를 **덮는다**

이라서 잔액은 두 번 오르고 거래 기록은 하나다 ⇒ 궤적에도 `mutation_diff` 의 EXTRA 에도
안 남고 오직 DB 해시로만 드러난다. **그러나 이것은 아직 가설이다** — 중복을 빼면 정말
`db_match` 가 뒤집히는지 확인하지 않았다([[77]] ③).

## 팔 ([[57]] 부정통제 포함)

    A_full     궤적 전량            <- 기록된 False 를 **재현해야** 한다. 아니면 판정 불가([[62]] 2b)
    B_nodup    **중복 변이 전부 제거** <- 이 팔만 True 로 뒤집히면 가설이 산다
    C_one      중복 **하나만** 제거    <- 남은 셋이 여전히 어긋나므로 False 여야 한다
    N_reads    성공한 **읽기** 4건 제거 <- 메시지를 자르는 행위 자체가 뒤집는 게 아님을 가른다

## 중복 판정은 닫힌 술어다 ([[22]]·[[59]])

`t2_forensic.mut_key(name, args)` 로 (도구·인자) 키를 만들고, **같은 키가 앞서 성공한 적이
있으면** 뒤엣것을 중복으로 센다. 도메인 낱말·태스크 id·gold 미접촉([[23]]).

실행 (리모트·cwd=scripts/distill/tau2):
    PYTHONIOENCODING=utf-8 /home/woori/venvs/seka_env/bin/python x544_dup_credit_regrade.py
"""
import argparse
import copy
import inspect
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t2_forensic as F                                              # noqa: E402

from tau2.data_model.simulation import SimulationRun                 # noqa: E402
from tau2.evaluator.evaluator_env import EnvironmentEvaluator        # noqa: E402
from tau2.registry import registry                                   # noqa: E402
from tau2.domains.banking_knowledge.environment import get_tasks     # noqa: E402

DOMAIN = "banking_knowledge"
RETRIEVAL = "alltools"   # t7358 러너 인자와 동일 (`--retrieval_config alltools`)


def result_id(m):
    """tool 메시지가 **어느 호출의 결과인지**. 이 궤적의 키는 `id` 다(`tool_call_id` 아님)."""
    return str(m.get("id") or m.get("tool_call_id") or "")


def tool_result_ok(ms, i, tcid):
    """이 호출의 결과 메시지가 오류가 아닌가 — `error` 플래그와 'Error:' 접두만 본다."""
    for j in range(i + 1, len(ms)):
        m = ms[j]
        if str(m.get("role")) != "tool":
            continue
        if result_id(m) != tcid:
            continue
        if m.get("error"):
            return False
        return not str(m.get("content") or "").lstrip().startswith("Error:")
    return False


def scan(sim, mut):
    """(중복 변이, 성공 읽기) 각각의 (msg_idx, tool_call_id) 목록."""
    ms = sim.get("messages") or []
    seen, dups, reads = set(), [], []
    for i, m in enumerate(ms):
        if str(m.get("role")) != "assistant":
            continue
        for tc in (m.get("tool_calls") or []):
            a = F.argsof(tc)
            nm = str(F.nameof(tc))
            tcid = str((tc.get("id") if isinstance(tc, dict) else "") or "")
            if not tool_result_ok(ms, i, tcid):
                continue
            inner = F.inner_name(a)
            target = str(inner or nm)
            if target not in mut:
                reads.append((i, tcid, target))
                continue
            key = F.mut_key(nm, a)
            if key in seen:
                dups.append((i, tcid, target))
            else:
                seen.add(key)
    return dups, reads


def prune(sim, drop):
    """(msg_idx, tool_call_id) 를 궤적에서 제거 — 호출과 그 결과 메시지 둘 다."""
    d = copy.deepcopy(sim)
    ids = {t for _, t, *_ in drop}
    out = []
    for m in (d.get("messages") or []):
        if str(m.get("role")) == "tool" and result_id(m) in ids:
            continue
        if str(m.get("role")) == "assistant" and (m.get("tool_calls") or []):
            keep = [tc for tc in m["tool_calls"]
                    if str((tc.get("id") if isinstance(tc, dict) else "") or "") not in ids]
            if not keep and not str(m.get("content") or "").strip():
                continue
            m = dict(m)
            m["tool_calls"] = keep or None
        out.append(m)
    d["messages"] = out
    return d


def grade(sim_dict, task):
    run = SimulationRun.model_validate(sim_dict)
    # ★env 는 런과 **같은 모양**이어야 한다 — t7358 은 `--retrieval_config alltools` 로 돌았고,
    #   그것 없이 재생하면 `KB_search_bm25` 를 모르는 env 가 되어 재생 자체가 죽는다(실측).
    kw = dict(environment_constructor=registry.get_env_constructor(DOMAIN),
              task=task, full_trajectory=run.messages,
              env_kwargs={"retrieval_config": RETRIEVAL})
    # 설치된 tau2 에 `strict_replay` 가 없는 판본이 있다 — 있을 때만 준다(추정 금지).
    if "strict_replay" in inspect.signature(
            EnvironmentEvaluator.calculate_reward).parameters:
        kw["strict_replay"] = False
    ri = EnvironmentEvaluator.calculate_reward(**kw)
    dbc = ri.db_check
    return (None if dbc is None else dbc.db_match), ri.reward


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="bank_t7358_d074_20260826")
    ap.add_argument("--task", default="task_074")
    a = ap.parse_args(argv)

    sims = [s for s in F.sims(a.tag, ".results.json.gz")
            if str(s.get("task_id")) == a.task]
    if not sims:
        print("판정 불가: %s 에 %s 가 없다" % (a.tag, a.task))
        return 2
    sim = sims[0]
    task = next((t for t in get_tasks() if t.id == a.task), None)
    if task is None:
        print("판정 불가: 태스크 선언을 못 읽었다")
        return 2

    mut = F.mutating_tools(DOMAIN)
    dups, reads = scan(sim, mut)
    print("=" * 96)
    print("x544 반증 — %s / %s" % (a.tag, a.task))
    print("=" * 96)
    print("중복 변이 %d 건: %s" % (len(dups), [(i, t) for i, _, t in dups]))
    print("성공 읽기 %d 건 (부정통제용 4건만 뺀다)" % len(reads))
    print("기록된 성적: reward=%s db_match=%s"
          % ((sim.get("reward_info") or {}).get("reward"),
             ((sim.get("reward_info") or {}).get("db_check") or {}).get("db_match")))

    arms = [("A_full", []), ("B_nodup", dups), ("C_one", dups[:1]),
            ("N_reads", reads[:4])]
    out = {}
    for name, drop in arms:
        if name != "A_full" and not drop:
            print("  %-9s 건너뜀 (뺄 것이 없다)" % name)
            continue
        m, rw = grade(prune(sim, drop) if drop else sim, task)
        out[name] = {"db_match": m, "reward": rw, "dropped": len(drop)}
        print("  %-9s db_match=%-6s reward=%-5s (뺀 호출 %d)" % (name, m, rw, len(drop)))

    p = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026",
                     "x544_dup_credit_regrade_2026_08_26.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump({"tag": a.tag, "task": a.task, "arms": out,
                   "dups": [(i, t) for i, _, t in dups]}, fh,
                  ensure_ascii=False, indent=2)
    print("\n산출: %s" % os.path.abspath(p))

    A, B = out.get("A_full", {}), out.get("B_nodup", {})
    if A.get("db_match") is not False:
        print("⛔A_full 이 기록된 False 를 재현하지 못했다 — **판정하지 않는다**([[62]] 2b)")
    elif B.get("db_match") is True:
        print("★가설이 산다: 중복만 빼면 db_match 가 True 로 뒤집힌다")
    else:
        print("★가설이 죽는다: 중복을 다 빼도 db_match 는 %s — 원인은 다른 칸이다"
              % (B.get("db_match"),))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
