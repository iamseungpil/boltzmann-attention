#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""claude_user_batch.py — Claude-as-user-sim 배치 (n건·gpt-4.1 0원·operand 안정성).
turns 파일 = {task_id: [user turn strings]}. 각 task를 ScriptedUser(내 턴)로 풀-플로우 구동,
compact 출력: model writes vs gold writes · reward · operand-match(gold write 집합 일치). gpt-4.1 0.
사용: env(게이트들) python claude_user_batch.py --turns batch.json --agent_base ... --agent_model ...
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

WRITES = {"modify_pending_order_items", "exchange_delivered_order_items", "return_delivered_order_items",
          "modify_pending_order_address", "cancel_pending_order", "modify_pending_order_payment"}


def norm(v):
    return tuple(sorted(str(x) for x in v)) if isinstance(v, list) else str(v)


def gold_writes(task):
    d = task.model_dump() if hasattr(task, "model_dump") else task
    out = []
    for a in ((d.get("evaluation_criteria") or {}).get("actions") or []):
        if a.get("name") in WRITES:
            ar = a.get("arguments") or {}
            out.append((a["name"], {k: norm(ar[k]) for k in ("order_id", "item_ids", "new_item_ids", "address1") if k in ar}))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", required=True)
    ap.add_argument("--agent_base", default="http://localhost:8360/v1")
    ap.add_argument("--agent_model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    a = ap.parse_args()

    if os.environ.get("T2_GATE") != "0":
        import t2_gate_patch
        t2_gate_patch.apply()

    batch = json.load(open(a.turns, encoding="utf-8"))
    cur = {"turns": [], "i": 0}

    from tau2.user.user_simulator import UserSimulator
    from tau2.data_model.message import UserMessage

    def _gen(self, message, state):
        if cur["i"] >= len(cur["turns"]):
            msg = UserMessage(role="user", content="###STOP###")
        else:
            msg = UserMessage(role="user", content=cur["turns"][cur["i"]])
        cur["i"] += 1
        state.messages.append(msg)
        return msg, state
    UserSimulator.generate_next_message = _gen

    import tau2.evaluator.evaluator_nl_assertions as _nle
    _nle.DEFAULT_LLM_NL_ASSERTIONS = f"openai/{a.agent_model}"
    _nle.DEFAULT_LLM_NL_ASSERTIONS_ARGS = {"api_base": a.agent_base, "api_key": "dummy",
                                           "temperature": 0.0, "response_format": {"type": "json_object"}}

    from tau2.run import get_tasks, run_single_task
    from tau2.data_model.simulation import TextRunConfig

    n_opmatch = 0
    n_done = 0
    for tid, turns in batch.items():
        cur["turns"], cur["i"] = turns, 0
        tasks = get_tasks("retail", task_ids=[tid])
        if not tasks:
            print(f"t{tid}: NO TASK"); continue
        task = tasks[0]
        cfg = TextRunConfig(domain="retail", agent="llm_agent", user="user_simulator",
                            llm_agent=f"openai/{a.agent_model}",
                            llm_args_agent={"api_base": a.agent_base, "api_key": "dummy", "temperature": 0.0},
                            llm_user=f"openai/{a.agent_model}",
                            llm_args_user={"api_base": a.agent_base, "api_key": "dummy", "temperature": 0.0},
                            num_trials=1, max_concurrency=1, save_to="claude_batch_tmp")
        try:
            sim = run_single_task(cfg, task)
        except Exception as e:
            print(f"t{tid}: RUN_ERR {type(e).__name__}: {str(e)[:80]}"); continue
        # model writes
        mw = []
        for m in (sim.messages if hasattr(sim, "messages") else []):
            for tc in (getattr(m, "tool_calls", None) or []):
                if getattr(tc, "name", None) in WRITES:
                    ar = getattr(tc, "arguments", {}) or {}
                    mw.append((tc.name, {k: norm(ar[k]) for k in ("order_id", "item_ids", "new_item_ids", "address1") if k in ar}))
        gw = gold_writes(task)
        ri = getattr(sim, "reward_info", None)
        rew = getattr(ri, "reward", None) if ri else None
        db = getattr(getattr(ri, "db_check", None), "db_match", None) if ri else None
        term = getattr(sim, "termination_reason", None)
        # operand-match: 모델 write 집합이 gold write 집합을 포함(operand 정확)?
        opmatch = all(g in mw for g in gw) if gw else (len(mw) == 0)
        n_done += 1
        n_opmatch += bool(opmatch)
        print(f"t{tid}: reward={rew} db={db} op_match_gold={opmatch} term={str(term)[-12:]}")
        print(f"   GOLD: {gw}")
        print(f"   MODEL: {mw}")
    print(f"\n=== SUMMARY: operand-match-gold {n_opmatch}/{n_done} (faithful Claude-user full-flow·gpt-4.1 0) ===")


if __name__ == "__main__":
    main()
