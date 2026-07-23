#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""무료 selftest — T2_COV_MIDDRIVE drift 판정(_last_assistant_did_write) 결정론 검증."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t2_gate_patch as G
from types import SimpleNamespace as NS
def tc(name, **a): return NS(name=name, arguments=a, id="x")
def am(tcs): return NS(role="assistant", content="", tool_calls=tcs, id=None)
def um(): return NS(role="user", content="hi", tool_calls=None, id=None)
def tm(): return NS(role="tool", content="ok", tool_calls=None, id=None)
W = {"file_credit_card_transaction_dispute"}
assert G._last_assistant_did_write([um(), am([tc("get_credit_card_transactions_by_user", user_id="u")]), tm(), um()], W) is False
assert G._last_assistant_did_write([um(), am([tc("call_discoverable_agent_tool", agent_tool_name="file_credit_card_transaction_dispute_4829", arguments="{}")]), tm()], W) is True
assert G._last_assistant_did_write([um(), am(None)], W) is False
print("PASS: mid-drive drift 판정(read/prose=drift·write(디스패처 unwrap)=진행)")
