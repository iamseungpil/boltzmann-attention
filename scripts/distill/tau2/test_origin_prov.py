# -*- coding: utf-8 -*-
"""Unit tests for L3 origin-prov (v3.2·A1_V3_PROBE_FORENSIC §1 t97 / A1_V3_DESIGN §2-L3).

오프라인·tau2 실임포트 0 (순수 함수만: _origin_role / _first_origin_fab).
Covers:
  1) t97 재현(reject): 주소가 assistant-first(제안)→user 복창(세탁)·tool-never → origin-fab.
  2) t43 재현(allow): user-first 명시 주소 → None (날조 아님).
  3) getter-확인 allow: assistant-first라도 tool 출력에 등장(tool-ever) → None (리뷰 caveat a).
  4) 스코프 가드: 주소류 아닌 인자(item_ids)는 origin 검사 밖 → None (Δspurious 보수).
  5) '#'-정규화 상속: _ctx_has 재사용 경로.
"""
import sys, os

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("T2_GATE_KINDS", "auth,confirm")

from t2_gate_patch import _origin_role, _first_origin_fab


class Msg:
    def __init__(self, role, content, error=False):
        self.role, self.content, self.error = role, content, error


class ToolCall:
    _n = 0
    def __init__(self, name, arguments):
        ToolCall._n += 1
        self.id = "tc%d" % ToolCall._n
        self.name, self.arguments, self.requestor = name, arguments, "assistant"


class AM:
    def __init__(self, tool_calls):
        self.role, self.tool_calls, self.content = "assistant", tool_calls, None


def t97_msgs():
    """t97 실측 형상: 재공유 거부 → assistant가 123 Broadway 최초 제안 → user yes 복창."""
    return [
        Msg("user", "I need to change the delivery address to my NYC address - "
                    "the one you have on file from my other order."),
        Msg("tool", '{"user_id": "yusuf_li_7255", "orders": ["#W6750959", "#W3407479"]}'),
        Msg("user", "my NYC address should already be on file from my previous order. "
                    "Could you use the same address as that order?"),
        Msg("assistant", "I encountered a policy requirement... The new shipping address will be: "
                         "123 Broadway / New York / NY / 10007"),
        Msg("user", "Yes, that's correct. Please go ahead and update the shipping address to "
                    "123 Broadway, New York, NY 10007."),
    ]


def test_t97_reject():
    msgs = t97_msgs()
    first, tool_ever = _origin_role("123 Broadway", msgs)
    assert first == "assistant" and not tool_ever, (first, tool_ever)
    am = AM([ToolCall("modify_pending_order_address",
                      {"order_id": "#W6750959", "address1": "123 Broadway",
                       "city": "New York", "zip": "10007"})])
    hit = _first_origin_fab(am, msgs)
    assert hit is not None, "t97 laundered address must be origin-fab"
    _, k, s = hit
    assert k == "address1" and s == "123 Broadway", (k, s)
    print("PASS test_t97_reject")


def test_t43_user_first_allow():
    msgs = [
        Msg("user", "Please ship it to 1234 S Michigan Ave, Chicago IL 60605."),
        Msg("assistant", "Sure - I will update the address to 1234 S Michigan Ave."),
    ]
    first, tool_ever = _origin_role("1234 S Michigan Ave", msgs)
    assert first == "user" and not tool_ever, (first, tool_ever)
    am = AM([ToolCall("modify_pending_order_address",
                      {"order_id": "#W1", "address1": "1234 S Michigan Ave"})])
    assert _first_origin_fab(am, msgs) is None, "user-first address must be allowed"
    print("PASS test_t43_user_first_allow")


def test_tool_ever_allow():
    """assistant가 먼저 말했더라도 getter 출력에 존재(재진술)면 grounded → allow (caveat a)."""
    msgs = [
        Msg("assistant", "Your other order ships to 476 Maple Drive, Suite 432."),
        Msg("tool", '{"order_id": "#W3407479", "address": {"address1": "476 Maple Drive", '
                    '"address2": "Suite 432", "city": "New York", "zip": "10093"}}'),
    ]
    first, tool_ever = _origin_role("476 Maple Drive", msgs)
    assert first == "assistant" and tool_ever, (first, tool_ever)
    am = AM([ToolCall("modify_pending_order_address",
                      {"order_id": "#W6750959", "address1": "476 Maple Drive"})])
    assert _first_origin_fab(am, msgs) is None, "tool-confirmed value must be allowed"
    print("PASS test_tool_ever_allow")


def test_non_addr_scope_guard():
    msgs = [Msg("assistant", "I will use item 9635758562 for you."),
            Msg("user", "yes 9635758562 sounds good.")]
    am = AM([ToolCall("modify_pending_order_items",
                      {"order_id": "#W1", "item_ids": ["9635758562"]})])
    assert _first_origin_fab(am, msgs) is None, "non-address args are out of origin scope"
    print("PASS test_non_addr_scope_guard")


def test_error_tool_ignored():
    """error=True tool 메시지는 tool_ever 근거가 아님."""
    msgs = [
        Msg("assistant", "address is 99 Fake Street"),
        Msg("tool", "Error: not found 99 Fake Street", error=True),
        Msg("user", "yes use 99 Fake Street"),
    ]
    first, tool_ever = _origin_role("99 Fake Street", msgs)
    assert first == "assistant" and not tool_ever, (first, tool_ever)
    am = AM([ToolCall("modify_pending_order_address",
                      {"order_id": "#W1", "address1": "99 Fake Street"})])
    assert _first_origin_fab(am, msgs) is not None
    print("PASS test_error_tool_ignored")


if __name__ == "__main__":
    test_t97_reject()
    test_t43_user_first_allow()
    test_tool_ever_allow()
    test_non_addr_scope_guard()
    test_error_tool_ignored()
    print("ALL PASS (5/5)")
