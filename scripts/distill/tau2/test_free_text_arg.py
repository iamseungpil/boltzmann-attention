# -*- coding: utf-8 -*-
"""★`T2_FREE_TEXT_ARG` **실행** 검정 (2026-09-01 재작성).

⛔이 파일의 이전 판은 `SRC = open(...).read()` 로 **소스 문자열만** 봤다. 그래서 정본에
  `_json` 미바인딩(NameError)이 있는 채로 **통과**했고, 라이브에서 그 술어는 **기회 8회 전부
  도달하고도 발화 0** 이었다(밤샘런 x712/x713). 계기가 아니라 **거동**을 검정한다([[07]]).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t2_gate_patch as G


class TC(object):
    """tau2 ToolCall 의 최소 대역 — 우리가 읽는 세 칸만 가진다."""

    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments
        self.id = "tc_1"


A2 = {"free_text_defaults": {"close_bank_account_7392": ["reason"]}}


def _dispatch_call(reason="Customer requested closure"):
    """065 실물 모양: 디스패처 경유 + 내부 도구 + 인자 봉투(문자열 JSON)."""
    inner = {"account_id": "chk_9f2", "reason": reason}
    return TC("call_discoverable_agent_tool",
              {"agent_tool_name": "close_bank_account_7392",
               "arguments": json.dumps(inner, ensure_ascii=False)})


def t_drops_when_ungrounded():
    tc = _dispatch_call()
    out = G.free_text_drop([tc], "손님은 계좌를 닫고 싶다고만 말했다", A2)
    bag = json.loads(tc.arguments["arguments"])
    assert out and out[0][1] == "reason", out
    assert "reason" not in bag, bag
    assert bag.get("account_id") == "chk_9f2", "대상 id 는 건드리지 않는다"


def t_keeps_when_grounded():
    tc = _dispatch_call("moving abroad")
    out = G.free_text_drop([tc], "I am MOVING ABROAD next month", A2)
    bag = json.loads(tc.arguments["arguments"])
    assert not out and bag.get("reason") == "moving abroad", (out, bag)


def t_untargeted_tool_untouched():
    tc = TC("close_other_account", {"reason": "whatever"})
    out = G.free_text_drop([tc], "", A2)
    assert not out and tc.arguments.get("reason") == "whatever"


def t_direct_call_shape():
    """디스패처를 안 타는 직접 호출도 같은 선언으로 잡힌다."""
    tc = TC("close_bank_account_7392", {"account_id": "chk_1", "reason": "made up"})
    out = G.free_text_drop([tc], "관련 없는 문맥", A2)
    assert out and "reason" not in tc.arguments


def t_no_declaration_is_noop():
    tc = _dispatch_call()
    before = json.dumps(tc.arguments, sort_keys=True)
    assert G.free_text_drop([tc], "", {}) == []
    assert json.dumps(tc.arguments, sort_keys=True) == before, "선언 없으면 바이트 동일"


def t_kb_corpus_arm():
    """⚠§S-9-4: KB 문서를 코퍼스에 넣어도 모델 작문은 접지되지 않아야 한다."""
    tc = _dispatch_call("Customer requested closure")
    kb = "Accounts may be closed at the customer's request. Fees are refunded."
    out = G.free_text_drop([tc], kb, A2)
    assert out, "KB 에 그 문장이 축자로 없으면 여전히 제거된다"


def t_empty_and_blank_values():
    for v in ("", "   "):
        tc = _dispatch_call(v)
        assert G.free_text_drop([tc], "", A2) == [], "빈 값은 건드리지 않는다"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("t_")]
    for f in fns:
        f()
        print("ok %s" % f.__name__)
    print("PASS %d/%d" % (len(fns), len(fns)))
