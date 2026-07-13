# -*- coding: utf-8 -*-
"""action-required 레버가 banking 실패를 얼마나 겨냥하나(상한 추정·gold로 target 근사).
실패 sim 중: gold가 action_tool write 요구 ∧ 에이전트가 그 action_tool을 *하나도* 안 부르고
회피(마지막 턴 조언/transfer) = action-required 발화 후보. (실제 폐쇄는 32B 준수+타 blocker 별개.)"""
import gzip, json, os
HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.abspath(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026",
                                 "sim_results", "bankxfer_floor_bank_t4.results.json.gz"))
A2 = json.load(open(os.path.join(HERE, "a2", "banking_knowledge.gate.json"), encoding="utf-8"))
ACTION = set(A2["action_tools"])
XFER = {"transfer_to_human_agents", "request_human_agent_transfer"}


def main():
    d = json.load(gzip.open(P))
    tasks = {str(t.get("id")): t for t in (d.get("tasks") or [])}
    fails = target = deflect_advice = deflect_xfer = 0
    for x in d["simulations"]:
        if ((x.get("reward_info") or {}).get("db_check") or {}).get("db_match"):
            continue
        fails += 1
        tid = str(x["task_id"])
        gold = (tasks.get(tid, {}).get("evaluation_criteria") or {}).get("actions") or []
        gold_actions = {a.get("name") for a in gold} & ACTION
        if not gold_actions:
            continue
        msgs = x.get("messages") or []
        called = {tc.get("name") for m in msgs if m.get("role") == "assistant" and m.get("tool_calls")
                  for tc in m["tool_calls"]}
        if called & gold_actions:
            continue      # 이미 그 action 부름 = action-required 대상 아님
        # 마지막 assistant 턴이 회피(조언 or transfer)?
        last_asst = [m for m in msgs if m.get("role") == "assistant"][-1:] or [{}]
        la = last_asst[0]
        lcalls = {tc.get("name") for tc in (la.get("tool_calls") or [])}
        if not lcalls:
            target += 1; deflect_advice += 1
        elif lcalls <= XFER:
            target += 1; deflect_xfer += 1
    print("banking floor 실패 %d개:" % fails)
    print("  action-required 발화후보(gold action 도구 미호출+회피종결): %d (%.0f%%)"
          % (target, 100 * target / fails))
    print("    ├ 조언 회피(도구호출 0):  %d" % deflect_advice)
    print("    └ transfer 포기:          %d" % deflect_xfer)
    print("  ★상한: 이 %d개가 action-required 최대 타깃 (단 32B 준수+타 blocker는 별개)" % target)


if __name__ == "__main__":
    main()
