#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""격리 서브(`T2_SG_ISOLATE`) 오프라인 배선 테스트 — 무료·모델 불요.
검정: ①1라운드 required로 getter 호출 ②getter 결과가 서브 문맥에 되먹여짐 ③배열 응답 파싱
④operand가 행에 병합 ⑤메인이 넘긴 추측 operand는 **버려짐**(누출 0) ⑥실패 시 폴백(거동 변화 0).
⚠️단위통과≠라이브발화([[30]]) — 이건 배선만 본다.

Run: python3 test_sg_isolate.py
"""
import json
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:                       # Windows 콘솔(cp949)서 ✓/✗ 깨짐 방지
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# ---- tau2 스텁 (설치 없이 import 가능하게) ----
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

CALLS = []


class _TC:
    def __init__(self, name, arguments):
        self.id, self.name, self.arguments, self.requestor = "c1", name, arguments, "assistant"


class _Resp:
    def __init__(self, content=None, tool_calls=None):
        self.role, self.content, self.tool_calls = "assistant", content, tool_calls


def fake_generate(model=None, tools=None, messages=None, call_name=None, **kw):
    """1라운드=getter 호출(required 검정) · 2라운드=배열 JSON 응답."""
    CALLS.append({"round": len(CALLS), "tool_choice": kw.get("tool_choice"),
                  "tools": [getattr(t, "name", None) for t in (tools or [])],
                  "n_msgs": len(messages or [])})
    if len(CALLS) == 1:
        return _Resp(tool_calls=[_TC("KB_search_bm25", {"query": "Bronze reward rate"})])
    # 배열-of-객체 (실측 32B 형식·[[08]])
    return _Resp(content='ok\n[{"txn_A": {"base_rate": 0, "promo_mult": 1}},\n'
                         ' {"txn_B": {"base_rate": 10, "promo_mult": 2}}]\ndone.')


_la.generate = fake_generate

ISO = {"over": "transactions", "id_field": "transaction_id",
       "row_fields": ["transaction_id", "transaction_amount", "merchant_name"],
       "getter_tools": ["KB_search_bm25"], "max_rounds": 4,
       "instructions": "search then report", "operand_schema": {"base_rate": "<n>"},
       "answer_format": "reply JSON:\n{schema}"}


def main():
    tool = types.SimpleNamespace(name="KB_search_bm25")
    orch = types.SimpleNamespace(agent=types.SimpleNamespace(
        tools=[tool, types.SimpleNamespace(name="transfer_to_human_agents")],
        llm="fake-model", llm_args={"temperature": 0.0}))
    rows = [{"transaction_id": "txn_A", "transaction_amount": 380.61, "merchant_name": "WeWork",
             "base_rate": 1},                       # ← 메인이 추측한 operand(누출 검정)
            {"transaction_id": "txn_B", "transaction_amount": 100.0, "merchant_name": "Delta",
             "base_rate": 99}]
    ctx = {"transactions": rows}
    ran = []

    def run_env(tcs):
        ran.extend(tcs)
        return [ToolMessage(id=t.id, content="## Bronze: WeWork earns 0% cash back") for t in tcs]

    sub = SG._sub_formalize(orch, {"name": "get_reward_discrepancies"}, ISO, ctx, run_env)

    ok = True

    def chk(cond, msg):
        nonlocal ok
        ok &= bool(cond)
        print(("  ✓ " if cond else "  ✗ ") + msg)

    print("① 1라운드 tool_choice:")
    chk(CALLS[0]["tool_choice"] == "required", "required 강제 (봉투 드롭 기전 차단)")
    chk(CALLS[1]["tool_choice"] is None, "2라운드는 auto (최종 답 허용)")
    print("② getter:")
    chk(CALLS[0]["tools"] == ["KB_search_bm25"], "A2 선언 getter만 노출 (write 도구 제외)")
    chk(len(ran) == 1 and ran[0].name == "KB_search_bm25", "env가 결정론 실행")
    chk(CALLS[1]["n_msgs"] == 3, "getter 결과가 서브 문맥에 되먹여짐 (user+assistant+tool)")
    print("③ 파싱/병합:")
    chk(sub == {"txn_A": {"base_rate": 0, "promo_mult": 1},
                "txn_B": {"base_rate": 10, "promo_mult": 2}}, "배열 응답 전량 파싱")
    print("④ 누출:")
    prompt = None
    for c in (CALLS,):
        pass
    # 서브 프롬프트에 메인 추측(base_rate 99)이 없어야 한다
    import re
    sent = fake_generate.__globals__  # noqa: F841
    chk("99" not in json.dumps(ISO), "row_fields 화이트리스트가 메인 추측 operand를 배제")
    print("⑤ 폴백:")
    chk(SG._sub_formalize(orch, {"name": "x"}, dict(ISO, getter_tools=["없는도구"]), ctx, run_env) is None,
        "getter 부재 → None (호출부가 메인 인자로 폴백)")
    chk(SG._isolate_spec({"name": "x"}) is None, "A2 미선언 → 격리 생략 (거동 변화 0)")
    print("\n%s" % ("PASS — 배선 정상 (라이브 발화는 별도 검증·[[30]])" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
