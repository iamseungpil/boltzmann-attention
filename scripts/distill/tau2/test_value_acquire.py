#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""무료 selftest — T2_VALUE_ACQUIRE(give 표면화) 검출기. 실제 031hv/031base 궤적 검증.
031hv(값 미획득·give 없음)=발화 / 031base(give함·값 획득)=미발화(have-value 관할)."""
import gzip, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("T2_A2_VARIANT", "ledger,ratefix")
import t2_gate_patch as G
from gate_interpreter import load_domain_a2
from types import SimpleNamespace as NS
specs = load_domain_a2("banking_knowledge")["value_acquisition"]
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
                 "reports", "facet_rft_2026", "sim_results")
def cvt(m):
    tcs = []
    for tc in (m.get("tool_calls") or []):
        fn = tc.get("function", {}) if "function" in tc else tc
        a = fn.get("arguments"); a = json.loads(a) if isinstance(a, str) else (a or {})
        tcs.append(NS(name=fn.get("name"), arguments=a, id=tc.get("id")))
    return NS(role=m.get("role"), content=m.get("content") or "",
              tool_calls=tcs or None, error=bool(m.get("error")), id=m.get("id"))
def sims(arm, tid):
    f = os.path.join(D, "bank_hve2e9_%s_20260723.results.json.gz" % arm)
    for s in json.load(gzip.open(f, "rt", encoding="utf-8"))["simulations"]:
        if s["task_id"] == tid:
            return [cvt(m) for m in s["messages"]]
def reask_fire(msgs):
    return any(m.role == "assistant" and ("last 4" in (m.content or "").lower())
               and G._value_acquire_fb(m, msgs[:i], specs) for i, m in enumerate(msgs))
h = sims("hv", "task_031"); b = sims("base", "task_031")
assert G._tool_given(b, "give_discoverable_user_tool", "get_card_last_4_digits") is True
assert G._tool_given(h, "give_discoverable_user_tool", "get_card_last_4_digits") is False
assert reask_fire(h) is True, "031hv(give 없음) 발화 실패"
# base: give-후(producer 출력 존재) 미발화
gi = next(i for i, m in enumerate(b) if G._tool_given([m], "give_discoverable_user_tool", "get_card_last_4_digits"))
post = any(m.role == "assistant" and ("last 4" in (m.content or "").lower())
           and G._value_acquire_fb(m, b[:i], specs) for i, m in enumerate(b) if i > gi)
assert post is False, "031base give-후 발화(over-fire)"
print("PASS: give 없음(031hv) 발화·give함(031base) give-후 미발화")
