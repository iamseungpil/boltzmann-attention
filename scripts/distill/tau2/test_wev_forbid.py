# -*- coding: utf-8 -*-
"""WEV `forbid_tokens` 회귀 — 자격 없는 dispute 상태를 증거로 인정하지 않는가 (2026-07-31).

배경([[23]] 감사): 구판은 `require_tokens=["RESOLVED"]` substring이라 KB가 열거한
`RESOLVED_BANK_FAVOR`("Investigation found transaction was valid, no credit issued")까지
증거로 통과시켰다 — 막으려던 오염을 한 갈래 열어둔 것이다. 긍정 토큰을 좁히면 도구 출력 형식
(`Status: RESOLVED - approved`)이 막혀 회귀하므로, 자격 없는 상태명을 A2가 선언해 배제한다.

라이브 `_wev_deny_msgs` + 실제 A2 스펙을 그대로 잰다([[03b]] — 테스트용 사본 금지).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from t2_gate_patch import _wev_deny_msgs  # noqa: E402

A2 = json.load(open(os.path.join(HERE, "a2", "banking_knowledge.gate.json"), encoding="utf-8"))
SPECS = [s for s in A2["write_evidence_specs"]
         if (s.get("applies_when") or {}).get("prefix") == "update_transaction_rewards"]
TXN = "txn_7f3a91"


class _TC:
    def __init__(self, name, arguments):
        self.name, self.arguments = name, arguments


class _Msg:
    def __init__(self, role, content):
        self.role, self.content = role, content


def _call():
    return _TC("call_discoverable_agent_tool",
               {"agent_tool_name": "update_transaction_rewards_3847",
                "arguments": json.dumps({"transaction_id": TXN,
                                         "new_rewards_earned": "120 points"})})


def _run(output):
    msgs = [_Msg("tool", output)] if output is not None else []
    return _wev_deny_msgs(msgs, _call(), SPECS)


CASES = [
    # (이름, 도구 출력, deny 되어야 하나)
    ("고객 승소(KB 상태명)", "Dispute %s status: RESOLVED_CUSTOMER_FAVOR" % TXN, False),
    ("도구 출력 형식(승인)", "Status: RESOLVED - approved for %s" % TXN, False),
    ("★은행 승소 = 크레딧 없음", "Dispute %s status: RESOLVED_BANK_FAVOR" % TXN, True),
    ("★부분 크레딧 = 보수적 차단", "Dispute %s status: RESOLVED_PARTIAL" % TXN, True),
    ("미해결(제출만)", "Status: SUBMITTED for %s" % TXN, True),
    ("증거 없음", None, True),
]

if __name__ == "__main__":
    assert SPECS, "update_transaction_rewards 스펙이 A2에 없다"
    assert SPECS[0].get("forbid_tokens"), "forbid_tokens 선언이 없다"
    ok = 0
    for name, out, want_deny in CASES:
        got = _run(out)
        hit = bool(got) == want_deny
        ok += hit
        print("  %-24s deny=%-5s want=%-5s %s" % (name, bool(got), want_deny, "OK" if hit else "FAIL"))
    print("RESULT: %d/%d %s" % (ok, len(CASES), "ALL PASS" if ok == len(CASES) else "FAIL"))
    sys.exit(0 if ok == len(CASES) else 1)
