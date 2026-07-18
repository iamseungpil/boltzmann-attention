#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""_sub_inject 오프라인 배선 테스트 (무료·모델 불요·2026-07-18 재설계 §2e).
검정: ①카드별 그룹핑(레코드 필드) ②제목접두 문서주입 ③grounding(quote∈문서→0유지·아니면 default 백필)
④엔진 리터럴 0(값·인용 LLM·엔진 substring만) ⑤operand를 rows에 병합.
⚠️단위통과≠라이브발화([[30]])."""
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

# tau2 스텁
_msg = types.ModuleType("tau2.data_model.message")


class UserMessage:
    def __init__(self, role="user", content=""):
        self.role, self.content = role, content


_msg.UserMessage = UserMessage
_msg.ToolMessage = type("ToolMessage", (), {})
_msg.MultiToolMessage = type("MultiToolMessage", (), {})
sys.modules.setdefault("tau2", types.ModuleType("tau2"))
sys.modules.setdefault("tau2.data_model", types.ModuleType("tau2.data_model"))
sys.modules["tau2.data_model.message"] = _msg
_la = types.ModuleType("tau2.agent.llm_agent")
sys.modules.setdefault("tau2.agent", types.ModuleType("tau2.agent"))
sys.modules["tau2.agent.llm_agent"] = _la

import t2_scaffold_get as SG  # noqa: E402

# 가짜 도메인 문서 2카드(제목접두 필터·grounding 검정)
DOCS = [
    {"title": "Silver Rewards Card: How to Earn 4% Back", "content": "Earn 4.0% on travel and software. All other purchases earn 1.0%."},
    {"title": "Business Silver Rewards Card: How to Earn 10% Back", "content": "Earn 10.0% on travel and software. All other purchases earn 1.0%."},
    {"title": "Business Silver Rewards Card: Exceptions", "content": "The following merchants earn 0% cash back: WeWork, Regus."},
]
SG._DOC_CACHE["banking_knowledge"] = DOCS

# 가짜 서브 응답: 카드별로 다른 답. base_rate + exclusion_quote.
RESP = {
    # Silver: Travel=4 +promo2, Software 과엄격 0 (근거없음 → 백필돼야)
    "Silver Rewards Card": '{"t_s1": {"base_rate": 4, "exclusion_quote": "", "promo_mult": 2, "promo_window_months": 6, "promo_start": "01/01/2025", "promo_end": "12/31/2025"}, "t_s2": {"base_rate": 0, "exclusion_quote": "", "promo_mult": 1}}',
    # Business Silver: WeWork=0 (근거 있음 → 유지)
    "Business Silver Rewards Card": '{"t_b1": {"base_rate": 0, "exclusion_quote": "The following merchants earn 0% cash back: WeWork, Regus.", "promo_mult": 1}}',
}
DEFAULTS = {"Silver Rewards Card": '{"base_default": 1}', "Business Silver Rewards Card": '{"base_default": 1}'}
CALLS = []


class _Resp:
    def __init__(self, content):
        self.content, self.tool_calls = content, None


def fake_generate(model=None, tools=None, messages=None, call_name=None, **kw):
    CALLS.append(call_name)
    txt = messages[0].content
    card = "Business Silver Rewards Card" if "Business Silver" in txt else "Silver Rewards Card"
    if call_name == "sg_default":
        return _Resp(DEFAULTS[card])
    return _Resp(RESP[card])


_la.generate = fake_generate

ISO = {
    "over": "transactions", "id_field": "transaction_id", "group_by": "credit_card_type",
    "row_fields": ["transaction_id", "credit_card_type", "category", "merchant_name"],
    "inject_docs": True, "rate_field": "base_rate", "quote_field": "exclusion_quote", "quote_min": 8,
    "operand_schema": {"base_rate": "<n>", "exclusion_quote": "<s>"},
    "inject_instructions": "{group}\n{docs}\n{items}\n{schema}",
    "base_default_prompt": "{group}\n{docs}",
}


def main():
    orch = types.SimpleNamespace(
        agent=types.SimpleNamespace(llm="fake", llm_args={"temperature": 0.0}),
        environment=types.SimpleNamespace(domain_name="banking_knowledge"))
    rows = [
        {"transaction_id": "t_s1", "credit_card_type": "Silver Rewards Card", "category": "Travel", "merchant_name": "Delta", "amount": 100},
        {"transaction_id": "t_s2", "credit_card_type": "Silver Rewards Card", "category": "Software", "merchant_name": "Coursera", "amount": 50},
        {"transaction_id": "t_b1", "credit_card_type": "Business Silver Rewards Card", "category": "Other", "merchant_name": "WeWork", "amount": 380},
    ]
    ctx = {"transactions": rows}
    ISO["base_default_prompt"] = ISO["base_default_prompt"]  # keep
    out = SG._sub_inject(orch, {"name": "get_reward_discrepancies"}, ISO, ctx, _la, UserMessage)

    ok = True

    def chk(c, m):
        nonlocal ok
        ok &= bool(c)
        print(("  ✓ " if c else "  ✗ ") + m)

    print("① 그룹핑(카드별 서브 호출):")
    chk(CALLS.count("sg_inject") == 2, "카드 2종 → sg_inject 2회 (Silver·Business Silver 분리)")
    print("② operand 병합 + grounding:")
    r = {x["transaction_id"]: x for x in rows}
    chk(r["t_s1"].get("base_rate") == 4, "Silver Travel = 4 (정상)")
    chk(r["t_s2"].get("base_rate") == 1, "Silver Software 과엄격0 → 기본율 1 백필 (근거없음)")
    chk(r["t_b1"].get("base_rate") == 0, "WeWork = 0 유지 (근거 문서에 실재)")
    print("③ promo 파라미터 병합(엔진 op가 읽음):")
    chk(r["t_s1"].get("promo_mult") == 2 and r["t_s1"].get("promo_start") == "01/01/2025",
        "Silver Travel promo_mult=2·기간 병합됨")
    chk("exclusion_quote" not in r["t_s1"], "exclusion_quote는 병합서 제외(grounding 전용)")
    print("④ 엔진 리터럴 0:")
    chk("WeWork" not in open(os.path.join(HERE, "t2_scaffold_get.py"), encoding="utf-8").read(),
        "엔진 코드에 도메인 리터럴(WeWork 등) 없음")
    print("\n%s" % ("PASS — 재설계 배선 정상 (라이브 발화는 별도·[[30]])" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
