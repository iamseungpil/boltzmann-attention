"""
sopbench_reward.py — §3 Rung2 ③: GRPO reward from the SOPBench rule evaluator (deterministic,
no LLM judge, free). One reward per rollout trajectory:

    r = w_pass * pass_term            # should_T: BOTH(dirgraph_satisfied AND action_called_correctly)
                                      # should_F: correct refusal (success)
      + w_proc * dirgraph_coverage    # dense process: fraction of GT dirgraph nodes actually called
      - w_early * early_act_frac       # ★ordering penalty: required dirgraph nodes NOT gathered
                                      #   before the goal was called (premature ACT)

Why these terms (design §3): pass=BOTH directly rewards gather-AND-act (the competition);
the early-act penalty directly punishes acting before gathering (learns the ordering);
dirgraph_coverage is dense so all-fail groups still get a gradient (GRPO advantage != 0).
Reward is GT-grounded -> no reward hacking / judge noise (cf. GRPO_REWARD_DESIGN §4).

Used by grpo_train_sopbench.py. `ev` = a SOPBench evaluation dict (the per-task entry's
evaluations[0]); `tool_seq` = ordered executed tool names; `task` = the task row (has
user_goal + directed_action_graph).
"""

def dirgraph_nodes(task):
    nodes = task.get("directed_action_graph", {}).get("nodes", []) or []
    return [n[0] for n in nodes if isinstance(n, (list, tuple)) and n]


def reward(ev, tool_seq, task, w_pass=1.0, w_proc=0.3, w_early=0.5):
    should = ev.get("action_should_succeed", task.get("action_should_succeed", True))
    if should:                                   # should_T: BOTH (gather AND act) is the bar
        pass_term = 1.0 if (ev.get("dirgraph_satisfied") and ev.get("action_called_correctly")) else 0.0
    else:                                        # should_F: correct refusal
        pass_term = 1.0 if ev.get("success") else 0.0
    goal = task.get("user_goal")
    nodes = dirgraph_nodes(task)
    required = [n for n in nodes if n != goal]           # precondition nodes (must gather before goal)
    called = list(tool_seq or [])
    proc = (sum(1 for n in nodes if n in set(called)) / len(nodes)) if nodes else 0.0
    # ordering penalty: how many required nodes were NOT gathered before the goal was called
    early_frac = 0.0
    if goal in called and required:
        idx = called.index(goal)
        before = set(called[:idx])
        early_frac = sum(1 for n in required if n not in before) / len(required)
    return {
        "reward": w_pass * pass_term + w_proc * proc - w_early * early_frac,
        "pass": pass_term, "proc": proc, "early_frac": early_frac, "should_T": bool(should),
    }


def group_advantages(rewards):
    """GRPO advantage = (r - mean)/std over the group (per prompt). Returns list aligned to rewards."""
    import statistics
    if not rewards:
        return []
    m = statistics.mean(rewards)
    sd = statistics.pstdev(rewards) or 1.0
    return [(r - m) / sd for r in rewards]


if __name__ == "__main__":   # self-test
    task = {"user_goal": "apply_credit_card",
            "directed_action_graph": {"nodes": [["internal_check_username_exist"],
                                                 ["login_user"], ["apply_credit_card"]]},
            "action_should_succeed": True}
    # gather-then-act (correct) > act-without-gather > gather-then-stop
    good = reward({"action_should_succeed": True, "dirgraph_satisfied": True, "action_called_correctly": True},
                  ["internal_check_username_exist", "login_user", "apply_credit_card"], task)
    early = reward({"action_should_succeed": True, "dirgraph_satisfied": False, "action_called_correctly": True},
                   ["apply_credit_card"], task)
    refuse = reward({"action_should_succeed": True, "dirgraph_satisfied": True, "action_called_correctly": False},
                    ["internal_check_username_exist", "login_user"], task)
    print("good  ", good)
    print("early ", early)
    print("refuse", refuse)
    assert good["reward"] > early["reward"], "gather-then-act must beat premature act"
    assert good["reward"] > refuse["reward"], "gather-then-act must beat gather-then-stop"
    assert early["early_frac"] > 0, "premature act must be penalized"
    print("adv:", group_advantages([good["reward"], early["reward"], refuse["reward"]]))
    print("OK")
