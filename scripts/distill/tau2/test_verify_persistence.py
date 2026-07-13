# -*- coding: utf-8 -*-
"""Lever 3: verify-persistence 유닛 (task_023형·신원수집+검증미완+포기·2026-07-13).
resolve_verify_persistence 순수함수 — 발동/미발동 경계."""
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.environ.setdefault("T2_GATE_KINDS", "auth")
import t2_resolve as R

BANK = json.load(open(os.path.join(HERE, "a2", "banking_knowledge.gate.json"), encoding="utf-8"))
XFER = {"transfer_to_human_agents"}


class TC:
    def __init__(self, name): self.name = name
class AM:
    def __init__(self, tool_calls=None, content=None):
        self.role, self.tool_calls, self.content = "assistant", tool_calls, content
class M:
    def __init__(self, role, tool_calls=None):
        self.role, self.tool_calls = role, tool_calls or []


def gathered_hist():
    return [M("assistant", [TC("get_user_information_by_date_of_birth")]),
            M("tool"), M("assistant", [TC("get_user_information_by_email")])]


FAILS = []
def ck(n, c, d=""):
    print(("PASS " if c else "FAIL ") + n + ("" if c else " | " + str(d)))
    if not c: FAILS.append(n)


# 1) 신원수집 + 검증미완 + 조언종료 → deny
r = R.resolve_verify_persistence(AM(content="I'll transfer you to a human."), gathered_hist(),
                                 BANK, transfer_tools=XFER)
ck("gathered_incomplete_giveup_deny", r["status"] == "deny" and r["reason"] == "verify-persistence", r)
ck("names_satisfier", "log_verification" in r.get("feedback", ""), r)

# 2) 신원수집 + 검증완료(log_verification) → ok
h = gathered_hist() + [M("assistant", [TC("log_verification")])]
r = R.resolve_verify_persistence(AM(content="bye"), h, BANK, transfer_tools=XFER)
ck("verified_ok", r["status"] == "ok", r)

# 3) 신원수집 + 포기아님(다른 도구 호출 중) → ok
r = R.resolve_verify_persistence(AM(tool_calls=[TC("get_current_time")]), gathered_hist(),
                                 BANK, transfer_tools=XFER)
ck("still_working_ok", r["status"] == "ok", r)

# 4) 신원 미수집 + 포기 → ok (스퓨리어스 방지)
r = R.resolve_verify_persistence(AM(content="I can't help."), [M("user")], BANK, transfer_tools=XFER)
ck("not_gathered_ok", r["status"] == "ok", r)

# 5) transfer 종료 + 미완 → deny (포기=transfer)
r = R.resolve_verify_persistence(AM(tool_calls=[TC("transfer_to_human_agents")]), gathered_hist(),
                                 BANK, transfer_tools=XFER)
ck("transfer_giveup_deny", r["status"] == "deny", r)

# 6) verify 게이트 config 없음(retail류) → ok
r = R.resolve_verify_persistence(AM(content="bye"), gathered_hist(),
                                 {"gates": [{"kind": "auth", "satisfiers": {"x": []}}]}, transfer_tools=XFER)
ck("no_gather_prefix_ok", r["status"] == "ok", r)

# 7) a2 없음 → ok
r = R.resolve_verify_persistence(AM(content="bye"), gathered_hist(), None)
ck("no_a2_ok", r["status"] == "ok", r)

print("\n%d FAIL" % len(FAILS) if FAILS else "\nALL PASS (Lever 3 verify-persistence)")
sys.exit(1 if FAILS else 0)
