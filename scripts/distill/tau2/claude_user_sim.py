#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""claude_user_sim.py — Claude-as-user-sim (gpt-4.1 0원·full-flow orchestration 확인).
사용자 제안(2026-06-26 예산소진): gpt-4.1 user-sim 대신 *내가(Claude)* user 턴 공급.
agent=로컬 Qwen·env+gate=그대로·user=스크립트(내 턴)·NL judge=로컬(gpt-4.1 0).

턴별 구동(agent temp=0 deterministic replay): turns 파일에 user 턴 누적 → 매 run마다
처음부터 재생 → turns 소진 시 ###STOP### → 전체 transcript 출력 → 내가 다음 턴 추가 → 재run.

사용: env(T2_GATE_KINDS=... T2_PRESENT_READS=1 T2_PRESENT_NESTED=1 T2_CALC=1) \
  python claude_user_sim.py --task <id> --turns turns.json --agent_base http://localhost:8360/v1 \
  --agent_model Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--turns", required=True, help="json list of user turn strings")
    ap.add_argument("--agent_base", default="http://localhost:8360/v1")
    ap.add_argument("--agent_model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    a = ap.parse_args()

    if os.environ.get("T2_GATE") != "0":
        import t2_gate_patch
        t2_gate_patch.apply()

    turns = json.load(open(a.turns, encoding="utf-8"))
    _state = {"i": 0, "exhausted": False}

    # ── ScriptedUser: gpt-4.1 대신 내 턴 반환 (LLM 호출 0) ──
    from tau2.user.user_simulator import UserSimulator
    from tau2.data_model.message import UserMessage

    def _gen(self, message, state):
        if _state["i"] >= len(turns):
            _state["exhausted"] = True
            msg = UserMessage(role="user", content="###STOP###")
        else:
            msg = UserMessage(role="user", content=turns[_state["i"]])
        _state["i"] += 1
        state.messages.append(msg)
        return msg, state
    UserSimulator.generate_next_message = _gen

    # NL judge → 로컬 Qwen (gpt-4.1 0). db_match/action은 judge 불요.
    import tau2.evaluator.evaluator_nl_assertions as _nle
    _nle.DEFAULT_LLM_NL_ASSERTIONS = f"openai/{a.agent_model}"
    _nle.DEFAULT_LLM_NL_ASSERTIONS_ARGS = {"api_base": a.agent_base, "api_key": "dummy",
                                           "temperature": 0.0, "response_format": {"type": "json_object"}}

    from tau2.run import get_tasks, run_single_task
    from tau2.data_model.simulation import TextRunConfig
    tasks = get_tasks(a.task_domain if hasattr(a, "task_domain") else "retail", task_ids=[a.task])
    if not tasks:
        print(f"task {a.task} 없음"); return
    task = tasks[0]
    cfg = TextRunConfig(
        domain="retail", agent="llm_agent", user="user_simulator",
        llm_agent=f"openai/{a.agent_model}",
        llm_args_agent={"api_base": a.agent_base, "api_key": "dummy", "temperature": 0.0},
        llm_user=f"openai/{a.agent_model}",  # 패치돼서 호출 안 됨(placeholder)
        llm_args_user={"api_base": a.agent_base, "api_key": "dummy", "temperature": 0.0},
        num_trials=1, max_concurrency=1, save_to="claude_user_tmp")
    sim = run_single_task(cfg, task)

    # ── transcript 출력 ──
    print("=" * 70)
    print(f"TASK {a.task} · 공급한 user턴={len(turns)} · exhausted={_state['exhausted']} · term={getattr(sim,'termination_reason',None)}")
    msgs = sim.messages if hasattr(sim, "messages") else []
    for m in msgs:
        role = getattr(m, "role", "?")
        content = getattr(m, "content", None)
        tcs = getattr(m, "tool_calls", None) or []
        if role == "user":
            print(f"\n[USER] {content}")
        elif role == "assistant":
            if content: print(f"[AGENT] {content}")
            for tc in tcs:
                print(f"   →CALL {getattr(tc,'name',None)}({json.dumps(getattr(tc,'arguments',{}) or {}, ensure_ascii=False)[:200]})")
        elif role == "tool":
            err = "ERR! " if getattr(m, "error", False) else ""
            c = str(content)
            # present/calc 마커는 짧게
            head = c.split("[OPERAND")[0].split("[DISAMBIGUATION")[0].split("[COMPUTED")[0]
            tags = []
            if "DISAMBIGUATION" in c: tags.append("+present")
            if "COMPUTED FACTS" in c: tags.append("+calc")
            print(f"   ←{err}{head[:160]}  {' '.join(tags)}")
    ri = getattr(sim, "reward_info", None)
    if ri:
        print(f"\nREWARD={getattr(ri,'reward',None)} db_match={getattr(getattr(ri,'db_check',None),'db_match',None)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
