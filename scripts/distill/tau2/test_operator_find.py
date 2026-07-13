# -*- coding: utf-8 -*-
"""Lever 1: operator FIND(⋈) 유닛 — 발견 후보 2+ 중 의도-매칭 도구 선택 검증 (2026-07-13).
resolve_operator 순수함수: FAB(미발견)·FIND(발견됐으나 틀린 선택)·정답·1후보·find_intent-off."""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.environ.setdefault("T2_GATE_KINDS", "auth")
import t2_resolve as R

PAT = "[a-z_]+_[0-9]{4}"
SPEC = {"kind": "operator", "arg": "agent_tool_name", "name_pattern": PAT,
        "operator_resolution": "discoverable", "find_intent": True}
SPEC_NOFIND = dict(SPEC); SPEC_NOFIND.pop("find_intent")


class TMsg:
    def __init__(self, role, content, error=False):
        self.role, self.content, self.error = role, content, error


def ctx(*names):
    """discovered 후보를 담은 tool-result 메시지 + user 발화."""
    return [TMsg("user", "I need help with a credit bureau incident."),
            TMsg("tool", "discovered tools: " + ", ".join(names))]


# formalize 서브콜을 결정론으로 목킹: 반환할 도구명을 지정
class FakeSub:
    def __init__(self, tool): self.content = '{"tool": "%s"}' % tool
class FakeLA:
    def __init__(self, tool): self._t = tool
    def generate(self, **kw): return FakeSub(self._t)
class FakeAgent:
    llm = "m"; llm_args = {}; tools = []
class FakeUM:
    def __init__(self, role=None, content=None): self.role, self.content = role or "user", content


FAILS = []
def ck(n, c, d=""):
    print(("PASS " if c else "FAIL ") + n + ("" if c else " | " + str(d)));
    if not c: FAILS.append(n)


TWO = ["emergency_credit_bureau_incident_transfer_1114", "reset_pin_tool_2020"]

# 1) FIND: 후보 2개·틀린 선택(reset_pin) → formalize=emergency → deny operator-find
r = R.resolve_operator(SPEC, {"agent_tool_name": "reset_pin_tool_2020"}, ctx(*TWO),
                       FakeAgent(), FakeLA(TWO[0]), FakeUM)
ck("find_wrong_deny", r["status"] == "deny" and r["reason"] == "operator-find", r)
ck("find_names_want", "emergency_credit_bureau_incident_transfer_1114" in r.get("feedback", ""), r)

# 2) 정답 선택 → ok (formalize=chosen)
r = R.resolve_operator(SPEC, {"agent_tool_name": TWO[0]}, ctx(*TWO),
                       FakeAgent(), FakeLA(TWO[0]), FakeUM)
ck("find_right_ok", r["status"] == "ok", r)

# 3) 후보 1개 → FIND 미발동(≥2 필요) → ok
r = R.resolve_operator(SPEC, {"agent_tool_name": TWO[0]}, ctx(TWO[0]),
                       FakeAgent(), FakeLA("whatever"), FakeUM)
ck("single_cand_noop", r["status"] == "ok", r)

# 4) FAB: 미발견 도구명 → operator-fab (FIND 이전 단계)
r = R.resolve_operator(SPEC, {"agent_tool_name": "invented_tool_9999"}, ctx(*TWO),
                       FakeAgent(), FakeLA(TWO[0]), FakeUM)
ck("fab_deny", r["status"] == "deny" and r["reason"] == "operator-fab", r)

# 5) find_intent off → FIND 미발동(틀린 선택도 ok)
r = R.resolve_operator(SPEC_NOFIND, {"agent_tool_name": "reset_pin_tool_2020"}, ctx(*TWO),
                       FakeAgent(), FakeLA(TWO[0]), FakeUM)
ck("find_off_noop", r["status"] == "ok", r)

# 6) agent 없음(오프라인 격리) → FIND 미발동 (우아한 강등)
r = R.resolve_operator(SPEC, {"agent_tool_name": "reset_pin_tool_2020"}, ctx(*TWO))
ck("no_agent_noop", r["status"] == "ok", r)

# 7) formalize=none → 발화 안 함
r = R.resolve_operator(SPEC, {"agent_tool_name": "reset_pin_tool_2020"}, ctx(*TWO),
                       FakeAgent(), FakeLA("none"), FakeUM)
ck("formalize_none_ok", r["status"] == "ok", r)

# 8) direct-dispatch(operator_resolution≠discoverable) → no-op (retail·U3)
r = R.resolve_operator({"kind": "operator", "arg": "x", "operator_resolution": "direct",
                        "find_intent": True}, {"x": "a"}, ctx(*TWO), FakeAgent(), FakeLA("b"), FakeUM)
ck("direct_dispatch_noop", r["status"] == "ok", r)

print("\n%d FAIL" % len(FAILS) if FAILS else "\nALL PASS (Lever 1 operator FIND)")
sys.exit(1 if FAILS else 0)
