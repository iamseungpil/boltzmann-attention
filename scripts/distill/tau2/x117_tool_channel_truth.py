# -*- coding: utf-8 -*-
"""한 도구의 **채널**을 우리가 몇 곳에서, 서로 다르게 판정하는가.

102 실측(2026-08-06): 같은 sim 안에서 우리 두 문구가 `submit_referral`을 두고 반대로 말했다 —
`[ACTION] 'submit_referral' is run by the CUSTOMER, not by you`(3회) ↔
`Error: 'submit_referral' is one of YOUR OWN agent tools - it cannot be given to the customer`.
gold는 **에이전트가 직접** 부르는 것이다. 두 문구의 출처가 다르기 때문에 생긴 모순이다:
`_exec_side`는 **에이전트에게 노출된 도구 목록**(`self.tools`)을, give-가드는 **환경 레지스트리**
(`env.tools`)를 본다. 어느 쪽이 사실인지는 도메인 환경에 물어야 한다 — 그것이 이 프로브다.

  usage:  x117_tool_channel_truth.py [task_102] [도구이름 ...]
"""

import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

TASK = next((a for a in sys.argv[1:] if a.startswith("task_")), "task_102")
NAMES = [a for a in sys.argv[1:] if not a.startswith("task_") and not a.startswith("-")]
DOMAIN = os.environ.get("X117_DOMAIN", "banking_knowledge")


def main():
    from tau2.registry import registry
    env = registry.get_env_constructor(DOMAIN)()
    tasks = registry.get_tasks_loader(DOMAIN)()
    t = next((x for x in tasks if getattr(x, "id", None) == TASK), None)

    at = getattr(env, "tools", None)
    reg_names = sorted({getattr(x, "name", None) for x in (getattr(at, "get_tools", lambda: [])() or [])
                        if getattr(x, "name", None)}) if at is not None else []
    print("== env 레지스트리 ==")
    print("  도구 %d개" % len(reg_names))
    disc = []
    for n in reg_names:
        try:
            if hasattr(at, "has_discoverable_tool") and at.has_discoverable_tool(n):
                disc.append(n)
        except Exception:
            pass
    print("  discoverable %d개: %s" % (len(disc), ", ".join(disc[:10])))

    ut = list(getattr(t, "user_tools", None) or []) if t is not None else []
    print("\n== 태스크 %s ==" % TASK)
    print("  user_tools: %s" % ut)

    targets = NAMES or [n for n in set(reg_names) | set(ut) if "referral" in n]
    print("\n== 도구별 판정 ==")
    print("  %-34s %-10s %-12s %-12s" % ("도구", "레지스트리", "discoverable", "task.user_tools"))
    for n in sorted(targets):
        in_reg = n in reg_names
        is_disc = n in disc
        print("  %-34s %-10s %-12s %-12s"
              % (n, "O" if in_reg else "X", "O" if is_disc else "X", "O" if n in ut else "X"))
    print("\n  ⇒ 세 열이 어긋나는 도구가 **우리 문구가 갈라지는 자리**다.")


if __name__ == "__main__":
    main()
