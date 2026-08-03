"""Regression: the give-execution nudge must not fire at a customer who already ran it.

`T2_GIVE_EXEC_NUDGE` decides "handed over, never run" from the trajectory. Until
2026-08-05 it recognised only one shape of customer execution — a
`call_discoverable_user_tool` wrapper carrying `discoverable_tool_name`. Real runs use
the other shape: the customer calls the tool under its own name from a `role=user`
message. Measured over both persisted arms (`x64_give_exec_predicate.py`), 5 of 17
firings were at customers who had already run the tool, and 3 of those were on
*passing* trajectories — the engine told a correct run to go do what it had done.

Four cases, one per branch of the predicate.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t2_gate_patch import give_exec_idle  # noqa: E402

GIVE = "give_discoverable_user_tool"
UCALL = "call_discoverable_user_tool"
TOOL = "apply_for_credit_card"

_fails = []


def chk(cond, what):
    print(("  ok   " if cond else "  FAIL ") + what)
    if not cond:
        _fails.append(what)


class TC:
    def __init__(self, name, args=None, id=None, requestor="assistant"):
        self.name, self.arguments, self.id, self.requestor = name, args or {}, id, requestor


class M:
    def __init__(self, role, tool_calls=None, id=None, error=False):
        self.role, self.tool_calls, self.id, self.error = role, tool_calls or [], id, error


def handed_over():
    """give issued and the environment accepted it."""
    return [M("assistant", [TC(GIVE, {"discoverable_tool_name": TOOL}, id="g1")]),
            M("tool", id="g1")]


def main():
    print("test_give_exec_user_direct")

    # 1. Handed over, never run — the nudge's actual target.
    idle = give_exec_idle(handed_over(), GIVE, UCALL)
    chk(idle == [TOOL], "인계 후 미실행 → 표적 (%r)" % (idle,))

    # 2. Customer ran it through the dispatcher — the shape the old code knew.
    msgs = handed_over() + [M("user", [TC(UCALL, {"discoverable_tool_name": TOOL})])]
    chk(give_exec_idle(msgs, GIVE, UCALL) == [], "디스패처 경유 실행 → 무발화")

    # 3. ★The regression: customer ran it directly, under the tool's own name.
    msgs = handed_over() + [M("user", [TC(TOOL, {"card_type": "Platinum"})])]
    chk(give_exec_idle(msgs, GIVE, UCALL) == [], "손님 직접 실행(role=user) → 무발화")

    # 3b. Same, keyed off requestor rather than message role.
    msgs = handed_over() + [M("assistant", [TC(TOOL, {}, requestor="user")])]
    chk(give_exec_idle(msgs, GIVE, UCALL) == [], "손님 직접 실행(requestor=user) → 무발화")

    # 4. The hand-over itself errored — nothing was ever given, so nothing is idle.
    msgs = [M("assistant", [TC(GIVE, {"discoverable_tool_name": TOOL}, id="g1")]),
            M("tool", id="g1", error=True)]
    chk(give_exec_idle(msgs, GIVE, UCALL) == [], "give 실패 → 인계 성사 아님 → 무발화")

    # 5. The agent calling the tool is not the customer running it.
    msgs = handed_over() + [M("assistant", [TC(TOOL, {})])]
    chk(give_exec_idle(msgs, GIVE, UCALL) == [TOOL],
        "에이전트 자신의 호출은 손님 실행이 아니다")

    print("\n%s (%d/%d)" % ("PASS" if not _fails else "FAIL", 6 - len(_fails), 6))
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
