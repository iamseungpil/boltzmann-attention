# -*- coding: utf-8 -*-
"""reference_filter credit 변형 + 복수-스펙 지원 (2026-07-22 §2bj·031 정박-치환 재현) 오프라인 테스트.
검정: ①031 재현(유저가 Marriott/167.34/11-07 지목·에이전트가 Amazon id 기입) → deny+correct id
②정답 id면 무개입 ③debit 스펙(구판) 회귀 무영향(리스트化 후에도 단일-dict 하위호환).
⚠️단위통과≠라이브발화([[30]])."""
import json
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_resolve as R  # noqa: E402

A2 = json.load(open(os.path.join(HERE, "a2", "banking_knowledge.gate.json"), encoding="utf-8"))

class M:
    def __init__(self, role, content, error=False):
        self.role, self.content, self.error = role, content, error

class TC:
    def __init__(self, name, arguments):
        self.name, self.arguments = name, arguments

class AM:
    def __init__(self, tcs):
        self.tool_calls = tcs

RECS = ("Found 2 record(s) in 'credit_card_transaction_history':\n"
        "1. Record ID: txn_2017c3b2b119\n   transaction_id: txn_2017c3b2b119\n"
        "   merchant_name: Amazon\n   transaction_amount: $89.99\n"
        "   transaction_date: 10/10/2025\n   status: COMPLETED\n"
        "2. Record ID: txn_adea68821a1d\n   transaction_id: txn_adea68821a1d\n"
        "   merchant_name: Marriott Hotels\n   transaction_amount: $167.34\n"
        "   transaction_date: 11/07/2025\n   status: COMPLETED\n")
MSGS = [M("user", "Please file the dispute for the Marriott Hotels $167.34 (11/07/2025) charge."),
        M("tool", RECS)]

class LA:
    @staticmethod
    def generate(model=None, tools=None, messages=None, call_name=None, **kw):
        return types.SimpleNamespace(content=json.dumps(
            {"date": "11/07/2025", "merchant": "Marriott", "amount": "167.34"}))

AG = types.SimpleNamespace(llm="m", llm_args={})
class UM:
    def __init__(self, role="user", content=""):
        self.role, self.content = role, content

def dispute(txn):
    return AM([TC("call_discoverable_agent_tool",
                  {"agent_tool_name": "file_credit_card_transaction_dispute_4829",
                   "arguments": json.dumps({"transaction_id": txn, "card_action": "keep_active"})})])

ok = True
def chk(c, m):
    global ok
    ok &= bool(c)
    print(("  ✓ " if c else "  ✗ ") + m)

print("① 031 재현: Amazon id 오기입 → deny+correct:")
r = R.resolve_reference_filter(dispute("txn_2017c3b2b119"), MSGS, A2, AG, LA, UM)
chk(r.get("status") == "deny", "deny 발화")
chk(r.get("correct") == "txn_adea68821a1d", "correct = Marriott txn (실제 gold)")

print("② 정답 id → 무개입:")
r2 = R.resolve_reference_filter(dispute("txn_adea68821a1d"), MSGS, A2, AG, LA, UM)
chk(r2.get("status") == "ok", "정답이면 ok")

print("③ 하위호환(단일-dict 스펙):")
a2s = dict(A2); a2s["reference_filter"] = A2["reference_filter"][0]   # debit 단일
r3 = R.resolve_reference_filter(dispute("txn_2017c3b2b119"), MSGS, a2s, AG, LA, UM)
chk(r3.get("status") == "ok", "debit-전용 스펙은 credit 호출 무개입(구판 거동 보존)")

print("\n%s" % ("PASS — credit reference_filter 배선 정상 (라이브 발화는 별도 검증·[[30]])" if ok else "FAIL"))
sys.exit(0 if ok else 1)
