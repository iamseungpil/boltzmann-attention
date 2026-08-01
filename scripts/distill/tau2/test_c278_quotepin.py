#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C278 quote-pin 라우팅 오프라인 테스트 (무료·모델 불요).
정본 = `QUOTE_GROUND_PINKIND_REDESIGN_2026_08_01.md` §5 케이스 매트릭스(1~11).
검정: ①named 통과(ba8b형 회수) ②비-포함/비-앵커 드롭 ③category 통과+마크 ④kind 결측·열거밖=재질의→abstain
⑤quote 날조 드롭 ⑥토큰-경계(부분-단어 매칭 차단) ⑦§2c 결핍 필드 출처 분리 ⑧OFF=C197 경로 불변 ⑨엔진 리터럴 0.
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

OK = True


def chk(c, m):
    global OK
    OK &= bool(c)
    print(("  ✓ " if c else "  ✗ ") + m)


# ---------------------------------------------------------------- 순수함수 계층
# 실제 KB 축자(C275/C276 실측)를 그대로 픽스처로 — 케이스 1·3·5·6·11이 전부 실사례다.
DOC = ("what is excluded from the higher sustainability points rate on ecocard "
       "general retailers even for eco friendly product purchases target walmart amazon "
       "thrift and resale markets thredup "
       "gaming subscription merchants software exclusion xbox game pass playstation plus "
       "online learning platform merchants software exclusion linkedin learning skillshare pluralsight "
       "hardware electronics merchants software exclusion apple microsoft dell "
       "grocery markets are excluded from the bonus rate "
       "vacation rentals or home sharing platforms coded under real estate do not qualify")
# rev2(C279): 식별표 멤버십. 표 내용은 A2 저작분의 부분집합(케이스 검정용).
TBL = {"Target": ["Target", "Target - Eco Collection"], "ThredUp": ["ThredUp"],
       "Microsoft": ["Microsoft 365"], "Dell": ["Dell", "Dell Technologies"],
       "LinkedIn Learning": [],                       # 판단된 무대응(LinkedIn Ads=광고·비-학습플랫폼)
       "Thrift and Resale Markets": ["ThredUp"],
       "General Retailers": ["Target", "Target - Eco Collection", "Amazon"]}
QP = {"policy_field": "exclusion_policy_merchant", "kind_field": "exclusion_pin_kind",
      "row_field": "merchant_name", "policy_group_rows": TBL,
      "reject_note": "the policy text you quoted names '{pin}', but this row's merchant is '{merchant}'.",
      "member_note": "the policy group '{pin}' does not list '{merchant}' among its members.",
      "lookup_note": "'{pin}' is not a name this policy lists.",
      "category_note": "category-based exclusion ('{pin}') applied to '{merchant}' — membership unverified."}


def V(quote, pin, kind, merchant):
    return SG._quote_pin_check(
        QP, {"exclusion_quote": quote, "exclusion_policy_merchant": pin, "exclusion_pin_kind": kind},
        {"merchant_name": merchant}, "exclusion_quote", 8, DOC)


def main():
    print("① 케이스 매트릭스 (설계서 §5 — 순수함수 `_quote_pin_check`):")
    v, _ = V("General Retailers (even for eco-friendly product purchases) Target",
             "Target", "named_merchant", "Target - Eco Collection")
    chk(v == "pass", "C1 ba8b: 정책 'Target' ↔ 행 'Target - Eco Collection' → pass (C275 회수)")

    v, _ = V("Hardware Electronics Merchants Software Exclusion Apple Microsoft Dell",
             "Microsoft", "named_merchant", "Microsoft 365")
    chk(v == "pass", "C2 Microsoft 365 → pass")

    v, i = V("Thrift and Resale Markets ThredUp", "ThredUp", "named_merchant", "Thrive Market")
    chk(v == "reject_member", "C3 019형(ThredUp → Thrive Market): 표에 비-구성원 → 드롭 유지")
    v, _ = V("Thrift and Resale Markets ThredUp", "Thrift and Resale Markets", "category", "Thrive Market")
    chk(v == "reject_member",
        "C3b ★범주로 선언해도 막힌다 — 그 그룹 구성원은 {ThredUp}뿐(⒞ 그룹-키 표)")

    v, i = V("Hardware Electronics Merchants Software Exclusion Apple Microsoft Dell",
             "Dell", "named_merchant", "Delta Sky Club")
    chk(v == "reject_member", "C4 Delta: 'Dell' 구성원 집합에 Delta Sky Club 없음 → 드롭 (다른 회사)")

    v, i = V("grocery markets are excluded from the bonus rate",
             "markets", "named_merchant", "Thrive Market")
    chk(v == "lookup_missing",
        "C5 범주어 핀 'markets': 표에 없는 키 → 조회 실패 → 재질의→abstain (false-apply 없음)")

    v, i = V("Gaming Subscription Merchants (Software Exclusion)",
             "Gaming Subscription Merchants", "category", "Xbox Game Pass")
    chk(v == "category" and i["pin"] == "Gaming Subscription Merchants",
        "C6 정직한 범주 주장 → 통과+마크 (R2)")

    v, _ = V("General Retailers even for eco-friendly product purchases", "Target", "", "Target - Eco Collection")
    chk(v == "kind_missing", "C7 kind 결측 → 재질의→abstain")
    v, _ = V("General Retailers even for eco-friendly product purchases", "Target", "named", "Target - Eco Collection")
    chk(v == "kind_missing", "C7b kind 열거 밖('named' 오타) = 결측 동일 (발견 6)")

    v, i = V("Notion purchases never earn cash back at this bank.",
             "Notion", "named_merchant", "Notion")
    chk(v == "reject" and i["why"] == "quote_unverbatim", "C8 날조 인용(문서 밖) → 드롭")

    v, _ = V("General Retailers even for eco-friendly product purchases Target",
             "Target", "named_merchant", "Targeting Solutions Inc")
    chk(v == "reject_member",
        "C6b ★'Targeting Solutions'는 Target 구성원 집합에 없음 (앵커 시절 오통과했던 케이스)")

    v, _ = V("", "", "", "Target - Eco Collection")
    chk(v == "pass", "quote 없음(비-강등형) → 라우팅 대상 아님")

    v, _ = V("Online Learning Platform Merchants (Software Exclusion) LinkedIn Learning Skillshare Pluralsight",
             "LinkedIn Learning", "named_merchant", "LinkedIn Ads")
    chk(v == "reject_member",
        "C11 ★rev2서 닫힘: 'LinkedIn Learning'→[] (판단된 무대응) ⇒ LinkedIn Ads 차단")
    v, _ = V("Online Learning Platform Merchants (Software Exclusion) LinkedIn Learning Skillshare Pluralsight",
             "LinkedIn", "named_merchant", "LinkedIn Ads")
    chk(v == "lookup_missing",
        "C11b 조각-복사 'LinkedIn': 표에 없는 키 → 재질의(‘WHOLE name’)→abstain")

    v, _ = V("Vacation rentals or home-sharing platforms coded under real estate do not qualify",
             "home-sharing platforms", "category", "Airbnb Stay")
    chk(v == "category",
        "C12 ★열거 없는 산문 범주 = 표 키 없음 → category 경로 유지(R2·열린 잔여 정직 보존)")

    print("\n② §2c 결핍 필드 출처 분리 (R3 버그픽스·발견 5 엣지):")
    iso = {"row_fields": ["account_open", "transaction_amount"],
           "operand_schema": {"base_rate": "", "promo_start": ""}}
    rec, sub = SG._split_missing_fields({"account_open": 3, "base_rate": 1, "promo_start": 1}, iso)
    chk(rec == {"account_open": 3} and sub == {"base_rate": 1, "promo_start": 1},
        "record-유래/sub-유래 분리 (C275 모순 지시 제거)")
    rec, sub = SG._split_missing_fields({"transaction_amount": 1}, {"row_fields": ["transaction_amount"],
                                                                   "operand_schema": {"transaction_amount": ""}})
    chk(rec and not sub, "양쪽 소속 → row_fields 우선 (발견 5)")
    rec, sub = SG._split_missing_fields({"mystery": 2}, iso)
    chk(not rec and sub == {"mystery": 2}, "어느 쪽도 아님 → sub-유래(안전측·이행-불가 지시 금지)")

    # ---------------------------------------------------------------- 배선 계층
    print("\n③ `_sub_inject` 배선 (ON: 라우팅·재질의 / OFF: C197 경로 불변):")
    SG._DOC_CACHE["banking_knowledge"] = [
        {"title": "EcoCard: Exceptions", "content": "General Retailers (even for eco-friendly product purchases): Target, Walmart. Grocery markets are excluded."}]
    ISO = {
        "over": "transactions", "id_field": "transaction_id",
        "group_by": ["credit_card_type"], "doc_key": "credit_card_type",
        "row_fields": ["transaction_id", "credit_card_type", "merchant_name"],
        "inject_docs": True, "rate_field": "base_rate", "quote_field": "exclusion_quote", "quote_min": 8,
        "quote_must_contain_field": "merchant_name",
        "quote_pin": dict(QP, lookup_retry_prompt="Copy the WHOLE name.", retry_prompt="Re-check and declare kind."),
        "operand_schema": {"base_rate": "<n>", "exclusion_quote": "<s>",
                           "exclusion_policy_merchant": "<s>", "exclusion_pin_kind": "<s>"},
        "inject_instructions": "{group}\n{docs}\n{items}\n{schema}",
    }
    # t1=ba8b형(named 정상) · t2=범주어 핀(재질의 후 category로 고침) · t3=019형(재질의해도 못 고침)
    FIRST = ('{"t1": {"base_rate": 1, "exclusion_quote": "General Retailers (even for eco-friendly product purchases): Target, Walmart.", "exclusion_policy_merchant": "Target", "exclusion_pin_kind": "named_merchant"}, '
             '"t2": {"base_rate": 1, "exclusion_quote": "Grocery markets are excluded.", "exclusion_policy_merchant": "markets", "exclusion_pin_kind": "named_merchant"}, '
             '"t3": {"base_rate": 0, "exclusion_quote": "General Retailers (even for eco-friendly product purchases): Target, Walmart.", "exclusion_policy_merchant": "Target", "exclusion_pin_kind": "named_merchant"}}')
    RETRY = ('{"t2": {"base_rate": 1, "exclusion_quote": "Grocery markets are excluded.", "exclusion_policy_merchant": "Grocery markets", "exclusion_pin_kind": "category"}, '
             '"t3": {"base_rate": 0, "exclusion_quote": "General Retailers (even for eco-friendly product purchases): Target, Walmart.", "exclusion_policy_merchant": "Target", "exclusion_pin_kind": "named_merchant"}}')
    calls = []

    class _R:
        def __init__(self, c):
            self.content, self.tool_calls = c, None

    def gen(model=None, tools=None, messages=None, call_name=None, **kw):
        calls.append(call_name)
        return _R(RETRY if call_name == "sg_inject_retry" else FIRST)

    _la.generate = gen

    def run(flag):
        calls.clear()
        os.environ["T2_QUOTE_PIN"] = flag
        orch = types.SimpleNamespace(
            agent=types.SimpleNamespace(llm="fake", llm_args={"temperature": 0.0}),
            environment=types.SimpleNamespace(domain_name="banking_knowledge"))
        rows = [{"transaction_id": "t1", "credit_card_type": "EcoCard", "merchant_name": "Target - Eco Collection"},
                {"transaction_id": "t2", "credit_card_type": "EcoCard", "merchant_name": "Thrive Market"},
                {"transaction_id": "t3", "credit_card_type": "EcoCard", "merchant_name": "Whole Foods Market"}]
        SG._sub_inject(orch, {"name": "get_reward_discrepancies"}, ISO, {"transactions": rows}, _la, UserMessage)
        return {r["transaction_id"]: r for r in rows}, orch

    r, orch = run("1")
    chk(r["t1"].get("base_rate") == 1, "ON C1: ba8b형 rate 생존 (현행 가드였다면 드롭)")
    chk("exclusion_policy_merchant" not in r["t1"] and "exclusion_pin_kind" not in r["t1"],
        "ON: 핀/종류는 op operand서 제외(grounding 전용)")
    chk("sg_inject_retry" in calls, "ON: guard-불성립 행 → 재질의 1회 발화 (R4)")
    chk(r["t2"].get("base_rate") == 1, "ON C5→C6: 조회 실패 핀이 재질의 후 category 재선언 → 회수")
    chk(any("category-based" in n for n in orch._t2_qp_notes) if orch._t2_qp_notes else False,
        "ON: category 마크 표면화됨")
    chk("base_rate" not in r["t3"], "ON C3: 표 비-구성원은 재질의 후에도 드롭 유지 (false-apply 차단)")

    r2, _ = run("0")
    chk("base_rate" not in r2["t1"], "OFF: C197 경로 불변(ba8b형 드롭 = 현행 거동 보존)")
    os.environ.pop("T2_QUOTE_PIN", None)

    print("\n④ [[05]] 엔진 리터럴 0:")
    src = open(os.path.join(HERE, "t2_scaffold_get.py"), encoding="utf-8").read()
    body = "\n".join(l for l in src.split("\n") if not l.strip().startswith("#"))
    for lit in ("merchant_name", "exclusion_pin_kind", "exclusion_policy_merchant", "Target", "named_merchant"):
        n = body.count(lit)
        # named_merchant/category = 열거 *값*이라 엔진이 안다(A2가 정하는 건 필드명·문구). 설계서 §2b 승인 범위.
        allow = 1 if lit in ("named_merchant",) else 0
        chk(n <= allow, "엔진 코드에 '%s' %d회 (허용 %d)" % (lit, n, allow))

    print("\n%s" % ("PASS — C278 배선 정상 (라이브 발화는 별도·[[30]])" if OK else "FAIL"))
    return 0 if OK else 1


if __name__ == "__main__":
    sys.exit(main())
