#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fetch-first 격리 서브(`mode=fetch_formalize`) 오프라인 배선 테스트 (2026-07-20 isolate-승격·023).
검정: ①A2 선언(check_rebate) ②1라운드 required·getter만 노출 ③getter 결과 서브 문맥 되먹임
④operand_keys 필터 파싱(top-level) ⑤ref/keys/getter 부재 → None 폴백(거동보존).
⚠️단위통과≠라이브발화([[30]]) — 배선만 본다.
"""
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

_msg = types.ModuleType("tau2.data_model.message")


class UserMessage:
    def __init__(self, role="user", content=""):
        self.role, self.content = role, content


class ToolMessage:
    def __init__(self, id=None, role="tool", requestor="assistant", content="", error=False):
        self.id, self.role, self.requestor, self.content, self.error = id, role, requestor, content, error


_msg.UserMessage, _msg.ToolMessage = UserMessage, ToolMessage
_msg.MultiToolMessage = type("MultiToolMessage", (), {})
sys.modules.setdefault("tau2", types.ModuleType("tau2"))
sys.modules.setdefault("tau2.data_model", types.ModuleType("tau2.data_model"))
sys.modules["tau2.data_model.message"] = _msg
_la = types.ModuleType("tau2.agent.llm_agent")
sys.modules.setdefault("tau2.agent", types.ModuleType("tau2.agent"))
sys.modules["tau2.agent.llm_agent"] = _la

import t2_scaffold_get as SG  # noqa: E402

A2 = json.load(open(os.path.join(HERE, "a2", "banking_knowledge.gate.json"), encoding="utf-8"))
REBATE = next(x for x in A2["scaffold_get_tools"] if x["name"] == "check_rebate_qualification")
ISO = REBATE.get("isolate") or {}

CALLS = []


class _TC:
    def __init__(self, name, arguments):
        self.id, self.name, self.arguments, self.requestor = "c1", name, arguments, "assistant"


class _Resp:
    def __init__(self, content=None, tool_calls=None):
        self.role, self.content, self.tool_calls = "assistant", content, tool_calls


def fake_generate(model=None, tools=None, messages=None, call_name=None, **kw):
    CALLS.append({"round": len(CALLS), "tool_choice": kw.get("tool_choice"),
                  "tools": [getattr(t, "name", None) for t in (tools or [])],
                  "n_msgs": len(messages or [])})
    if len(CALLS) == 1:
        return _Resp(tool_calls=[_TC("get_credit_card_transactions_by_user", {"user_id": "u1"})])
    return _Resp(content='here:\n{"transactions": [{"date": "01/15/2024", "amount": 520.0},'
                         ' {"date": "02/03/2024", "amount": 610.5}], "noise": 1}\nend.')


_la.generate = fake_generate


def main():
    tool = types.SimpleNamespace(name="get_credit_card_transactions_by_user")
    orch = types.SimpleNamespace(agent=types.SimpleNamespace(
        tools=[tool, types.SimpleNamespace(name="close_credit_card_account")],
        llm="fake-model", llm_args={"temperature": 0.0}))
    ctx = {"user_id": "u1", "credit_card_account_id": "acc_9",
           "account_opening_date": "01/10/2024", "monthly_threshold": 500}
    ran = []

    def run_env(tcs):
        ran.extend(tcs)
        return [ToolMessage(id=t.id, content='[{"date":"01/15/2024","amount":520.0}]') for t in tcs]

    ok = True

    def chk(cond, msg):
        nonlocal ok
        ok &= bool(cond)
        print(("  ✓ " if cond else "  ✗ ") + msg)

    print("① A2 선언:")
    chk(ISO.get("mode") == "fetch_formalize", "check_rebate isolate.mode=fetch_formalize")
    chk(ISO.get("ref_params") == ["user_id", "credit_card_account_id"], "ref_params 선언")
    chk(ISO.get("operand_keys") == ["transactions"], "operand_keys=transactions만(최소 offload)")
    chk("user_id" in REBATE["params"] and "credit_card_account_id" in REBATE["params"],
        "ref 파라미터가 도구 스키마에 존재")
    # §2aq: get_correct_savings_apy도 fetch-iso(097 컨텍스트 레버)
    APY = next(x for x in A2["scaffold_get_tools"] if x["name"] == "get_correct_savings_apy")
    chk((APY.get("isolate") or {}).get("mode") == "fetch_formalize"
        and (APY["isolate"].get("operand_keys") == ["components"])
        and set(APY["isolate"].get("ref_params") or []) <= set(APY["params"]),
        "APY: fetch_formalize 선언·ref가 스키마에 존재")
    chk(bool(APY.get("ground")), "APY: 서브 산출도 기존 ground가 검증(심층방어)")

    sub = SG._sub_fetch_formalize(orch, REBATE, ISO, ctx, run_env)
    print("② 서브 흐름:")
    chk(CALLS[0]["tool_choice"] == "required", "1라운드 required(봉투 드롭 차단)")
    chk(CALLS[0]["tools"] == ["get_credit_card_transactions_by_user"],
        "A2 선언 getter만 노출(write 도구 제외)")
    chk(len(ran) == 1 and ran[0].name == "get_credit_card_transactions_by_user",
        "env가 결정론 실행(off-ledger)")
    chk(CALLS[1]["n_msgs"] == 3, "getter 결과 서브 문맥 되먹임(user+assistant+tool)")
    print("③ 파싱:")
    chk(isinstance(sub, dict) and len(sub.get("transactions") or []) == 2,
        "operand_keys 필터로 transactions 2건 추출")
    chk("noise" not in (sub or {}), "operand_keys 밖 키는 배제")
    print("④ 폴백(거동보존):")
    chk(SG._sub_fetch_formalize(orch, REBATE, dict(ISO, ref_params=["없는참조"]), ctx, run_env) is None,
        "ref 부재 → None(에이전트 인자 폴백)")
    chk(SG._sub_fetch_formalize(orch, REBATE, dict(ISO, operand_keys=[]), ctx, run_env) is None,
        "operand_keys 미선언 → None")
    chk(SG._sub_fetch_formalize(orch, REBATE, dict(ISO, getter_tools=["없는도구"]), ctx, run_env) is None,
        "getter 부재 → None")
    print("\n%s" % ("PASS — fetch-first 배선 정상 (라이브 발화는 별도 검증·[[30]])" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
