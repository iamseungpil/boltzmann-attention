# -*- coding: utf-8 -*-
"""write-착수 격리 서브 + 근거 검산 검정 (x307~x310 근거·[[25]] 집행 장치).

계약: 판단은 서브(LLM)가 하고, 엔진은 **닫힌 술어 둘**만 본다 —
  ① 도구명이 이 대화의 실재 이름 집합 원소인가   ② 제안의 모든 값이 근거에 substring 실재하는가
실패하면 **전달하지 않고** 종전 경로로 폴백한다(조용한 거동 변경 금지).

검정 6칸:
  T1 접지된 제안 → 전달 문구 생성(근거 포함)
  T2 Δspurious: 근거에 없는 금액(날조) → 전달 안 함(폴백)      ← FIX-11 재발 방지
  T3 Δspurious: 레지스트리 밖 도구명 → 전달 안 함
  T4 서브가 빈 calls → 전달 안 함
  T5 근거(직전 손님 발화 이후 성공 도구결과)가 없으면 서브 자체를 안 부른다
  T6 T2_WRITE_SUB 미설정이면 호출부가 종전 경로 그대로
"""
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

import t2_resolve as R                                            # noqa: E402

BASIS = ("ATM fee lines whose charged amount does NOT match the documented fee schedule: "
         "btxn_kj07s5t6u7v9 (charged $3.00, documented fee $0.00, difference $3.00). "
         "the credit policy requires ONE fee_refund credit for the net correction of THIS "
         "account = $9.50 for account chk_kj93a7b2e1_1")
NAMES = {"apply_checking_account_credit_5829", "get_bank_account_transactions_9173"}
A2 = {"write_initiation": {"instructions": "INSTR", "answer_format": "FMT",
                           "delivery_template": "CALLS:\n{calls}\nBASIS:\n{basis}",
                           "basis_max_chars": 4000, "temperature": 0},
      "eplan": {"unlock_tool": "unlock_x", "dispatch_tool": "call_x", "list_tool": "list_x"}}
FAILS = []


class M(object):
    def __init__(self, role, content="", tool_calls=None, error=False, id=None):
        self.role, self.content, self.tool_calls = role, content, tool_calls
        self.error, self.id = error, id


class UserMessage(object):
    def __init__(self, content="", role="user"):
        self.role, self.content = role, content


class LA(object):
    def __init__(self, payload):
        self.payload = payload
    def generate(self, model=None, tools=None, messages=None, call_name=None, **kw):
        return type("S", (), {"content": self.payload})()


class Agent(object):
    llm, llm_args, tools = "m", {}, []


MSGS = [M("user", "please fix the ATM fees"), M("tool", BASIS, id="t1")]


def chk(name, cond, extra=""):
    print("%-4s %s %s" % ("PASS" if cond else "FAIL", name, extra))
    if not cond:
        FAILS.append(name)


def run(payload, msgs=MSGS, names=NAMES):
    return R.sub_write_proposal(Agent(), LA(payload), UserMessage, msgs, A2, names)


good = json.dumps({"calls": [{"tool": "apply_checking_account_credit_5829",
                              "account_id": "chk_kj93a7b2e1_1", "amount": 9.50,
                              "credit_type": "fee_refund"}]})
r = run(good)
chk("T1_grounded_delivered", bool(r) and "apply_checking_account_credit_5829" in r
    and "BASIS:" in r, (r or "")[:60])

bad_amt = json.dumps({"calls": [{"tool": "apply_checking_account_credit_5829",
                                 "account_id": "chk_kj93a7b2e1_1", "amount": 77.77,
                                 "credit_type": "fee_refund"}]})
chk("T2_ungrounded_value_blocked", run(bad_amt) is None)

bad_tool = json.dumps({"calls": [{"tool": "not_a_real_tool_0000",
                                  "account_id": "chk_kj93a7b2e1_1", "amount": 9.50}]})
chk("T3_offregistry_tool_blocked", run(bad_tool) is None)

chk("T4_empty_calls_blocked", run(json.dumps({"calls": []})) is None)

no_basis = [M("user", "please fix the ATM fees")]
chk("T5_no_basis_no_sub", run(good, msgs=no_basis) is None)

# T6: 플래그 없으면 호출부가 종전 경로(문면)로 간다
os.environ.pop("T2_WRITE_SUB", None)
OPS = {"action_tools": ["call_x", "unlock_x"]}
AM = M("assistant", "I will guide you through it.", None)
out = R.resolve_action_operator(OPS, AM, MSGS, A2, target_tool="call_x",
                                transfer_tools={"transfer_to_human_agents"})
chk("T6_flag_off_keeps_old_path", out.get("reason") != "write-initiation-sub",
    str(out.get("reason")))

print("=" * 62)
print("FAILS:", FAILS or "none")
sys.exit(1 if FAILS else 0)
