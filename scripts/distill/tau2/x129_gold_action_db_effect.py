# -*- coding: utf-8 -*-
"""x129 — gold 액션 하나하나가 **DB 해시를 바꾸는가** (유료 0·모델 0).

왜 필요한가: `db_match`는 해시 동등성이다(`evaluator_env.py:118-126`) — gold 환경은
`evaluation_criteria.actions`를 **실제로 실행해서** 만들어진다. 그래서 gold 목록에 있는 호출이
DB를 바꾸는 종류라면, 그 호출을 빠뜨린 궤적은 **제출이 완벽해도 영원히 0점**이다.

led_j task_100이 정확히 그 모양이었다: 제출 유형 집합이 gold과 **완전 일치**(초과 0·누락 0)인데
reward 0. 남은 차이는 `unlock_discoverable_agent_tool` + `call_discoverable_agent_tool` 뿐이었다.

이 도구는 그 물음만 답한다 — 빈 환경에서 gold 액션을 **하나씩** 실행하며 해시를 찍는다.
바뀌면 그 호출은 **필수 상태 변경**이고, 안 바뀌면 채점과 무관한 절차다. 추측이 아니라 해시다.

usage: x129_gold_action_db_effect.py --domain banking_knowledge --tasks task_100,task_101
"""

import argparse
import io
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="banking_knowledge")
    ap.add_argument("--tasks", default="task_100,task_101")
    a = ap.parse_args()

    from tau2.registry import registry

    get_env = registry.get_env_constructor(a.domain)
    tasks = {t.id: t for t in registry.get_tasks_loader(a.domain)()}

    for tid in [t.strip() for t in a.tasks.split(",") if t.strip()]:
        task = tasks.get(tid)
        if task is None:
            print("== %s == (없음)" % tid)
            continue
        actions = (task.evaluation_criteria.actions or []) if task.evaluation_criteria else []
        print("=" * 96)
        print("== %s ==  reward_basis=%s · gold 액션 %d개"
              % (tid, getattr(task.evaluation_criteria, "reward_basis", None), len(actions)))

        env = get_env()
        init = task.initial_state
        if init is not None:
            env.set_state(initialization_data=getattr(init, "initialization_data", None),
                          initialization_actions=getattr(init, "initialization_actions", None),
                          message_history=getattr(init, "message_history", None) or [])
        prev = (env.get_db_hash(), env.get_user_db_hash())
        print("   시작 해시  agent=%s user=%s" % (prev[0][:12], prev[1][:12]))
        for i, act in enumerate(actions, 1):
            try:
                env.make_tool_call(tool_name=act.name, requestor=act.requestor, **act.arguments)
                err = None
            except Exception as e:
                err = repr(e)[:90]
            cur = (env.get_db_hash(), env.get_user_db_hash())
            changed = "★바뀜" if cur != prev else "  그대로"
            print("   %d. %-34s %s  %s%s"
                  % (i, act.name, changed,
                     json.dumps(act.arguments, ensure_ascii=False)[:60],
                     ("   ⚠%s" % err) if err else ""))
            prev = cur
        print("   ⇒ ★바뀜 표시가 붙은 호출은 **빠뜨리면 db_match 불가**다.")


if __name__ == "__main__":
    main()
