# -*- coding: utf-8 -*-
"""U1 가드 offline 검정 — 완료된 디스패치 후 재-발견 강요 금지 (FORENSIC_SYNTHESIS §2-A).

실측 대상(073 t0 `bank_t7285_b`): msg57 성공 write 3건 → 보고 턴이 회피로 오판 →
[DISCOVERY-REQUIRED] → 같은 3건 재실행(61·67·73). 가드 = `_dispatch_since_last_user`.

검정 3칸:
  T1  손님 발화 후 성공 디스패치 → 순수-텍스트 턴 = ok (가드 발화)
  T2  Δspurious: 디스패치가 **거부**(Error:)였으면 종전대로 deny (과침묵 방지)
  T3  Δspurious: 디스패치가 손님 발화 **이전**(이번 요청과 무관)이었으면 종전대로 deny
"""
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

import t2_resolve as R                                             # noqa: E402


class M(object):
    def __init__(self, role, content="", tool_calls=None, error=False):
        self.role, self.content, self.tool_calls, self.error = role, content, tool_calls, error


class TC(object):
    _n = 0
    def __init__(self, name, arguments):
        TC._n += 1
        self.id = "tc%d" % TC._n
        self.name, self.arguments, self.requestor = name, arguments, "assistant"


A2 = {"action_tools": ["call_dispatch", "unlock_x", "give_x"],
      "eplan": {"unlock_tool": "unlock_x", "dispatch_tool": "call_dispatch",
                "list_tool": "list_x"},
      "operands": {"call_dispatch": {"agent_tool_name": {
          "operator_resolution": "discoverable", "getter": "KB_search"}}}}
OPS = {"action_tools": A2["action_tools"]}
ADVICE = M("assistant", "I have applied the corrections to your accounts.", None)
FAILS = []


def chk(name, cond, extra=""):
    print("%-4s %s %s" % ("PASS" if cond else "FAIL", name, extra))
    if not cond:
        FAILS.append(name)


def run(msgs):
    return R.resolve_action_operator(OPS, ADVICE, msgs, A2, target_tool="call_dispatch",
                                     transfer_tools={"transfer_to_human_agents"})


# T1: 손님 발화 → 성공 디스패치 → 보고 턴 = ok
call = TC("call_dispatch", {"agent_tool_name": "apply_credit_1234", "arguments": "{}"})
msgs_t1 = [M("user", "please fix the fees"),
           M("assistant", "", [call]),
           M("tool", "Credit applied successfully!")]
r = run(msgs_t1)
chk("T1_executed_silences", r.get("status") == "ok", str(r)[:90])

# T2: 디스패치 **거부** → 종전대로 deny (가드가 과침묵하지 않는다)
call2 = TC("call_dispatch", {"agent_tool_name": "apply_credit_1234", "arguments": "{}"})
msgs_t2 = [M("user", "please fix the fees"),
           M("assistant", "", [call2]),
           M("tool", "Error: not unlocked", error=True)]
r = run(msgs_t2)
chk("T2_denied_still_fires", r.get("status") == "deny", str(r)[:90])

# T3: 성공 디스패치가 손님 발화 **이전** → 이번 요청에 대한 회피는 종전대로 deny
call3 = TC("call_dispatch", {"agent_tool_name": "get_something_5678", "arguments": "{}"})
msgs_t3 = [M("user", "earlier request"),
           M("assistant", "", [call3]),
           M("tool", "Executed: get_something_5678"),
           M("user", "now do the NEW thing")]
r = run(msgs_t3)
chk("T3_stale_dispatch_still_fires", r.get("status") == "deny", str(r)[:90])

print("=" * 60)
print("FAILS:", FAILS or "none")
sys.exit(1 if FAILS else 0)
