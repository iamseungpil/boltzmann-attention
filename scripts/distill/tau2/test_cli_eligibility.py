# -*- coding: utf-8 -*-
"""check_cli_eligibility(§2bu·rall11 052 실측) + close-WEV(§2bu·rall11 038 실측) 오프라인 검증.

라이브 코드(t2_compute.apply_op·t2_gate_patch._wev_deny_msgs)+실제 A2 선언을 그대로 잰다
([[03b]] 별도구현 금지). CLI-자격: KB 규칙(logistics_005/006)의 결정론 판정 —
①052 픽스처(bronze·최근 승인 09-15·as_of 11-14=60d<120d) → COOLDOWN deny 방향
②승인 이력 無 → ELIGIBLE ③이용률 초과 → UTILIZATION ④계좌 연령 미달 → ACCOUNT_AGE
⑤납부 미달 → PAYMENT_HISTORY ⑥premium 완화 기준 ⑦미지 카드 → None(abstain)
⑧입력 결손 → None(abstain). close-WEV: ⑨CLOSURE_OK 증거 없이 close → deny
⑩CLOSURE_OK 공존 → 통과. ⚠️단위통과≠라이브발화([[30]])."""
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
from t2_compute import apply_op  # noqa: E402
from t2_gate_patch import _wev_deny_msgs  # noqa: E402

A2 = json.load(open(os.path.join(HERE, "a2", "banking_knowledge.gate.json"), encoding="utf-8"))
TOOL = [t for t in A2["scaffold_get_tools"] if t["name"] == "check_cli_eligibility"][0]
OP = TOOL["op"]
WEV = A2["write_evidence_specs"]

FAILS = []
def check(n, c):
    print(("PASS " if c else "FAIL ") + n)
    if not c: FAILS.append(n)

BASE = {"credit_card_account_id": "cc_5e4c1a83b0_bronze", "card_type": "Bronze Rewards Card",
        "date_of_account_open": "05/10/2023", "as_of_date": "2025-11-14",
        "current_balance": 1500.0, "credit_limit": 4000.0,
        "consecutive_on_time_payments": 6,
        "last_approved_cli_submitted_date": "2025-09-15"}

# ① 052 픽스처 → COOLDOWN (60d < 120d·entry)
r = apply_op(OP, dict(BASE))
check("C1_052_cooldown_deny", isinstance(r, str) and r.startswith("NOT_ELIGIBLE_COOLDOWN")
      and "deny_credit_limit_increase" in r)
# ② 승인 이력 無 → ELIGIBLE
r = apply_op(OP, dict(BASE, last_approved_cli_submitted_date="none"))
check("C2_no_prior_eligible", isinstance(r, str) and r.startswith("ELIGIBLE"))
# ③ 이용률 초과(3900/4000=97.5% > 70%·쿨다운 無) → UTILIZATION
r = apply_op(OP, dict(BASE, last_approved_cli_submitted_date="none", current_balance=3900.0))
check("C3_utilization", isinstance(r, str) and r.startswith("NOT_ELIGIBLE_UTILIZATION"))
# ④ 계좌 연령 미달(개설 2025-10-01→44d < 120d) → ACCOUNT_AGE (첫 관문)
r = apply_op(OP, dict(BASE, date_of_account_open="2025-10-01"))
check("C4_account_age", isinstance(r, str) and r.startswith("NOT_ELIGIBLE_ACCOUNT_AGE"))
# ⑤ 납부 미달(entry 6 요구·5) → PAYMENT_HISTORY
r = apply_op(OP, dict(BASE, last_approved_cli_submitted_date="none", consecutive_on_time_payments=5))
check("C5_payment_history", isinstance(r, str) and r.startswith("NOT_ELIGIBLE_PAYMENT_HISTORY"))
# ⑥ premium 완화(gold: 60d 쿨다운→60d 경과=미달 아님·pay 3 요구) → ELIGIBLE
r = apply_op(OP, dict(BASE, card_type="Gold Rewards Card", consecutive_on_time_payments=3,
                      last_approved_cli_submitted_date="2025-09-15"))
check("C6_premium_thresholds", isinstance(r, str) and r.startswith("ELIGIBLE"))
# ⑦ 미지 카드타입 → None(abstain·[[05]] 3-값)
check("C7_unknown_card_abstain", apply_op(OP, dict(BASE, card_type="Mystery Card")) is None)
# ⑧ 입력 결손(balance 없음) → None(abstain)
b8 = dict(BASE, last_approved_cli_submitted_date="none"); b8.pop("current_balance")
check("C8_missing_input_abstain", apply_op(OP, b8) is None)

# ── close-WEV (fix 8·038) ──
class Msg:
    def __init__(self, role, content):
        self.role, self.content = role, content
class TC:
    def __init__(self, name, arguments):
        self.name, self.arguments = name, arguments
CLOSE = TC("call_discoverable_agent_tool",
           {"agent_tool_name": "close_credit_card_account_7834",
            "arguments": json.dumps({"credit_card_account_id": "cc_890389b165_silver",
                                     "user_id": "890389b165"})})
H_NO = [Msg("tool", "Found 4 record(s) in 'credit_card_accounts': cc_890389b165_silver current_balance: $1,173.27")]
fb = _wev_deny_msgs(H_NO, CLOSE, WEV)
# first-match = 절차 선행 스펙(closure reason history)·§2bu 질의-가이드 주입 확인
check("W9_close_no_evidence_denied", fb is not None and "cc_890389b165_silver" in fb
      and "plain words" in fb and '"closure reason history"' in fb)
H_OK = H_NO + [
    Msg("tool", "Executed: get_closure_reason_history_8293 - no closure reason records for cc_890389b165_silver"),
    Msg("tool", "Executed: log_credit_card_closure_reason_4521 - reason logged for cc_890389b165_silver"),
    Msg("tool", "Executed: get_pending_replacement_orders_5765 - no pending orders for cc_890389b165_silver"),
    Msg("tool", "Closure eligibility for credit card account cc_890389b165_silver (outstanding balance 0.0): CLOSURE_OK - no outstanding balance; the balance requirement of the closure protocol is satisfied.")]
check("W10_close_with_full_evidence_pass", _wev_deny_msgs(H_OK, CLOSE, WEV) is None)
# CLOSURE_OK 스펙(잔액-관문)이 마지막 방벽: 절차 3종 있어도 verdict 없으면 deny + keep-open 경고
H_PART = H_OK[:-1]
fb2 = _wev_deny_msgs(H_PART, CLOSE, WEV)
check("W11_verdict_is_last_barrier", fb2 is not None and "CLOSURE_OK" in fb2
      and "KEEP the account open" in fb2)

# ── fix 10 (054·§2bu): approve WEV의 payment 증거 술어 = 실제 출력 포맷 정합 ──
APPROVE = TC("call_discoverable_agent_tool",
             {"agent_tool_name": "approve_credit_limit_increase_5847",
              "arguments": json.dumps({"credit_card_account_id": "cc_584f9c5d00_gold",
                                       "user_id": "584f9c5d00", "new_credit_limit": 7500})})
PAY_OUT = ("Payment history for account 'cc_584f9c5d00_gold' (last 3 months):\n"
           "Consecutive on-time payments: 3\n  - Payment Date: 10/15/2025")
H_EV = [Msg("tool", PAY_OUT),
        Msg("tool", "Credit limit increase history retrieved.\nExecuted: get_credit_limit_increase_history_4829\nfor cc_584f9c5d00_gold"),
        Msg("tool", "Pending replacement orders check completed.\nExecuted: get_pending_replacement_orders_5765\nfor cc_584f9c5d00_gold")]
check("W12_approve_with_real_payment_output_pass", _wev_deny_msgs(H_EV, APPROVE, WEV) is None)
check("W13_approve_without_payment_denied",
      _wev_deny_msgs(H_EV[1:], APPROVE, WEV) is not None)

print("\n%s" % ("ALL PASS" if not FAILS else "FAILS: %s" % FAILS))
sys.exit(1 if FAILS else 0)
