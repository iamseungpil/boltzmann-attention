#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""날조 안전판의 **축** 검정 (2026-08-21·`Record ID:` 계수기의 DB-전용성 정정).

## 왜
`_sub_fetch_formalize` 끝의 날조 안전판은 서브 getter 출력의 `Record ID:` 수를 세어
`source=0` 이면 배열 operand 를 통째로 폐기한다(072 자리표시자 id 3행 사고·[[25]]).
그런데 그 술어는 **DB 레코드 덤프 포맷 전용**이라, getter 가 KB 검색인 선언은
원리상 항상 `source=0` → **정직하게 읽어도 항상 폐기**였다. 영속 로그 전수 실측:

    get_correct_savings_apy   (getter=KB_search_bm25)  52/52 = 100% source=0 · 폐기 47
    get_atm_fee_discrepancies (getter=DB 디스패처)      67/348 = 19%       · 폐기 62

앞의 선언은 자기 `ground.array_fields` 로 **축이 맞는** 날조 차단을 이미 걸고 있고
(병합 직후 관문1 `_ground_operands`·T2_SG_GROUND=1), 실제로 작동한다.

## 검정 (네 갈래가 각각 옳게 갈리는가)
  ⒜ 계약 있음 + 집행 ON  + source=0 → **살아남는다**(관문1 이 심사)
  ⒝ 계약 있음 + 집행 OFF + source=0 → **폐기**(fail-closed·집행자가 없으면 안전판이 선다)
  ⒞ 계약 없음 + 집행 ON  + source=0 → **폐기**(DB 축 거동 보존·072 사고 방지)
  ⒟ 계약 없음 + source>0(`Record ID:`) → **살아남는다**(종전 거동)
⚠단위통과≠라이브발화([[30]]) — 배선만 본다.
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
        self.id, self.role, self.requestor = id, role, requestor
        self.content, self.error = content, error


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
APY = next(x for x in A2["scaffold_get_tools"] if x["name"] == "get_correct_savings_apy")
REBATE = next(x for x in A2["scaffold_get_tools"] if x["name"] == "check_rebate_qualification")

# KB 검색 출력에는 `Record ID:` 가 원리상 없다 — 이것이 축의 실체다.
KB_OUT = ("doc_046 Savings APY stacking\n"
          "| Base APY (Green Savings) | 2.00% |\n"
          "| Checking pairing boost (Gold Checking) | 0.75% |\n")
DB_OUT = ("Found 1 record(s) in 'credit_card_transactions':\n"
          "1. Record ID: txn_aa\n   amount: 520.0\n   date: 01/15/2024\n")

ANSWER = {
    "get_correct_savings_apy": ('{"components": [{"kind":"base","value":2.0,'
                                '"source":"| Base APY (Green Savings) | 2.00% |"}]}'),
    "check_rebate_qualification": '{"transactions": [{"date": "01/15/2024", "amount": 520.0}]}',
}


class _TC:
    def __init__(self, name, arguments):
        self.id, self.name, self.arguments, self.requestor = "c1", name, arguments, "assistant"


class _Resp:
    def __init__(self, content=None, tool_calls=None):
        self.role, self.content, self.tool_calls = "assistant", content, tool_calls


def _run_case(decl, getter_out, env_ground):
    """한 갈래를 돌리고 (결과, stderr 로그)를 돌려준다."""
    iso = dict(decl.get("isolate") or {})
    iso["max_rounds"] = 2                      # 0=도구호출 · 1=마감 답
    getter_name = (iso.get("getter_tools") or [None])[0]
    calls = []

    def fake_generate(model=None, tools=None, messages=None, call_name=None, **kw):
        calls.append(1)
        if len(calls) == 1:
            return _Resp(tool_calls=[_TC(getter_name, {"q": "apy"})])
        return _Resp(content=ANSWER[decl["name"]])

    _la.generate = fake_generate
    orch = types.SimpleNamespace(agent=types.SimpleNamespace(
        tools=[types.SimpleNamespace(name=getter_name)],
        llm="fake-model", llm_args={"temperature": 0.0}))
    ctx = {k: "x" for k in (iso.get("ref_params") or [])}

    def run_env(tcs):
        return [ToolMessage(id=t.id, content=getter_out) for t in tcs]

    old_g, old_fb = os.environ.get("T2_SG_GROUND"), os.environ.get("T2_SG_ISOFB")
    os.environ.pop("T2_SG_ISOFB", None)        # 서브내 되먹임은 이 검정의 대상이 아니다
    if env_ground:
        os.environ["T2_SG_GROUND"] = "1"
    else:
        os.environ.pop("T2_SG_GROUND", None)
    import io as _io
    _cap, sys.stderr = _io.StringIO(), None
    sys.stderr = _cap
    try:
        out = SG._sub_fetch_formalize(orch, decl, iso, ctx, run_env)
    finally:
        sys.stderr = sys.__stderr__
        for k, v in (("T2_SG_GROUND", old_g), ("T2_SG_ISOFB", old_fb)):
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return out, _cap.getvalue()


def main():
    ok = True

    def chk(cond, msg):
        nonlocal ok
        ok &= bool(cond)
        print(("  ✓ " if cond else "  ✗ ") + msg)

    print("① 선언 전제:")
    chk(any((af or {}).get("param") == "components"
            for af in ((APY.get("ground") or {}).get("array_fields") or [])),
        "APY: components 에 배열 근거 계약이 선언돼 있다")
    chk(not ((REBATE.get("ground") or {}).get("array_fields")),
        "check_rebate: 배열 근거 계약이 **없다**(DB 축 대조군)")
    chk("Record ID:" not in KB_OUT, "KB 출력에는 `Record ID:` 가 없다(축의 실체)")

    print("② 네 갈래:")
    a, la_ = _run_case(APY, KB_OUT, True)
    chk(isinstance(a, dict) and a.get("components"),
        "Ⓔ 계약 O + 집행 ON  + source=0 → 살아남는다")
    chk("계수기 미적용" in la_,
        "   └ 서지 않았다는 사실이 로그에 남는다(침묵-스킵 금지·[[55]])")

    b, lb = _run_case(APY, KB_OUT, False)
    chk(b is None, "Ⓕ 계약 O + 집행 OFF + source=0 → 폐기(fail-closed)")
    chk("폐기**(날조 방지" in lb, "   └ 종전 폐기 문구 그대로")

    c, _ = _run_case(REBATE, KB_OUT, True)
    chk(c is None, "Ⓖ 계약 X + 집행 ON  + source=0 → 폐기(DB 축 거동 보존·072)")

    d, _ = _run_case(REBATE, DB_OUT, True)
    chk(isinstance(d, dict) and d.get("transactions"),
        "Ⓗ 계약 X + source>0(`Record ID:`) → 살아남는다(종전 거동)")

    print("\n%s" % ("PASS — 안전판 축 정정 배선 정상 "
                    "(라이브 발화는 별도 검증·[[30]])" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
