# -*- coding: utf-8 -*-
"""T2_WRITE_ARG_GROUND(_write_arg_ground_deny·2026-07-22 §2bs) 오프라인 검증.

rall10 031 실측: WRITE_EVIDENCE(선행-read)는 통과했으나 뷰에 실재한 last4(5320) 대신
'1234'를 기입해 dispute 제출 — read-강제와 값-전사는 별개 구멍. 라이브 함수+실제 A2
선언을 그대로 잰다([[03b]] 별도구현 금지). 검정: ①031 재현(미근거 1234) → deny
②도구출력 근거(5320) → 통과 ③user 발화 근거 → 통과 ④prefix 불일치 도구 → 무간섭
⑤키 부재 → skip ⑥미근거 transaction_id → deny.
⚠️단위통과≠라이브발화([[30]]) — 라이브 검정=rall11 로그 [T2_WRITE_ARG_GROUND] 태그.
"""
import json
import os
import sys
import types
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8"); sys.stderr.reconfigure(encoding="utf-8")
except Exception: pass
def mkmod(n):
    m = types.ModuleType(n); sys.modules[n] = m; return m
mkmod("tau2"); mkmod("tau2.agent"); la = mkmod("tau2.agent.llm_agent")
la.generate = lambda **kw: None
class LLMAgent: pass
la.LLMAgent = LLMAgent
mkmod("tau2.data_model"); msgmod = mkmod("tau2.data_model.message")
class _M:
    def __init__(self, **kw): self.__dict__.update(kw)
msgmod.ToolMessage = msgmod.UserMessage = msgmod.MultiToolMessage = _M
msgmod.ToolCall = None
mkmod("tau2.orchestrator"); oo = mkmod("tau2.orchestrator.orchestrator")
class BaseOrchestrator: pass
oo.BaseOrchestrator = BaseOrchestrator
from t2_gate_patch import _write_arg_ground_deny  # noqa: E402

A2 = json.load(open(os.path.join(HERE, "a2", "banking_knowledge.gate.json"), encoding="utf-8"))
SPECS = A2["write_arg_grounding"]

class Msg:
    def __init__(self, role, content):
        self.role, self.content = role, content
class TC:
    def __init__(self, name, arguments):
        self.name, self.arguments = name, arguments

VIEW = ("Found 4 record(s) in 'credit_card_accounts':\n"
        "   account_id: cc_890389b165_silver\n   card_last_4_digits: 5320\n")
TXNS = "Found 47 record(s):\n   transaction_id: txn_adea68821a1d\n   merchant: Marriott"
HIST = [Msg("tool", VIEW), Msg("tool", TXNS)]

def dispute(last4, txn):
    return TC("call_discoverable_agent_tool",
              {"agent_tool_name": "file_credit_card_transaction_dispute_4829",
               "arguments": json.dumps({"transaction_id": txn,
                                        "card_last_4_digits": last4,
                                        "card_action": "keep_active"})})

FAILS = []
def check(n, c):
    print(("PASS " if c else "FAIL ") + n)
    if not c: FAILS.append(n)

# ① 031 재현: 미근거 '1234' → deny·값 명시
fb = _write_arg_ground_deny(HIST, dispute("1234", "txn_adea68821a1d"), SPECS)
check("G1_fabricated_last4_denied", fb is not None and "1234" in fb and "card_last_4_digits" in fb)
# ② 도구출력 근거(5320) → 통과
check("G2_grounded_pass", _write_arg_ground_deny(HIST, dispute("5320", "txn_adea68821a1d"), SPECS) is None)
# ③ user 발화 근거 → 통과(고객이 직접 준 값=정당)
H3 = HIST + [Msg("user", "my card ends in 9999")]
check("G3_user_stated_pass", _write_arg_ground_deny(H3, dispute("9999", "txn_adea68821a1d"), SPECS) is None)
# ④ prefix 불일치 도구 → 무간섭
other = TC("call_discoverable_agent_tool",
           {"agent_tool_name": "order_replacement_credit_card_7291",
            "arguments": json.dumps({"card_last_4_digits": "1234"})})
check("G4_other_tool_untouched", _write_arg_ground_deny(HIST, other, SPECS) is None)
# ⑤ 키 부재 → skip(변형 오차단 회피)
nokey = TC("call_discoverable_agent_tool",
           {"agent_tool_name": "file_credit_card_transaction_dispute_4829",
            "arguments": json.dumps({"card_action": "keep_active"})})
check("G5_missing_key_skip", _write_arg_ground_deny(HIST, nokey, SPECS) is None)
# ⑥ 미근거 transaction_id → deny
fb = _write_arg_ground_deny(HIST, dispute("5320", "txn_2017c3b2b119"), SPECS)
check("G6_fabricated_txn_denied", fb is not None and "txn_2017c3b2b119" in fb)

print("\n%s" % ("ALL PASS" if not FAILS else "FAILS: %s" % FAILS))
sys.exit(1 if FAILS else 0)
