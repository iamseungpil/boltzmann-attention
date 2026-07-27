#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C204 다음-런 레버 오프라인 검증 (2026-07-27·무료·모델 불요).
D6 카테고리 요율 주석(`D6_CATEGORY_RATE_DESIGN` rev2 §7) / D7 계산도구 dedup /
D8 dispute→update chain / D9 정규화 도구명 매칭. ⚠단위통과≠라이브발화([[30]])."""
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_compute as C
import t2_gate_patch as GP
import t2_scaffold_get as SG

A2 = json.load(io.open(os.path.join(HERE, "a2", "banking_knowledge.gate.json"), encoding="utf-8"))
FIT = next(t for t in A2["scaffold_get_tools"] if t["name"] == "check_card_application_fit")


def find_table(o):
    if isinstance(o, dict):
        for k, v in o.items():
            if k == "table" and isinstance(v, list) and v and isinstance(v[0], dict) and "card" in v[0]:
                return v
            r = find_table(v)
            if r is not None:
                return r
    if isinstance(o, list):
        for v in o:
            r = find_table(v)
            if r is not None:
                return r
    return None


def find_op(o):
    if isinstance(o, dict):
        if o.get("op") == "catalog_filter":
            return o
        for v in o.values():
            r = find_op(v)
            if r is not None:
                return r
    if isinstance(o, list):
        for v in o:
            r = find_op(v)
            if r is not None:
                return r
    return None


SPEC = find_op(FIT)
OK = True


def chk(c, m):
    global OK
    OK &= bool(c)
    print(("  ✓ " if c else "  ✗ ") + m)


def entry(res, card):
    for b in ("eligible", "unverified", "excluded"):
        for e in res[b]:
            if e["card"] == card:
                return b, e
    return None, None


def d6_annotation():
    print("D6 카테고리 요율 주석:")
    # §7-1 회귀 0: spend_category 미제공 = 재구조화 전과 동일 판정(003 실인자)
    base_ctx = {"max_fx_fee": "0", "needs_purchase_protection": "true",
                "min_credit_limit": "100000", "premium_subscriber": "true"}
    r0 = C.apply_op(SPEC, dict(base_ctx))
    chk("rate_for" not in json.dumps(r0), "미제공 시 주석 0(거동 보존)")
    names0 = sorted(e["card"] for e in r0["eligible"])
    # §7-2/§7-6: 003 리플레이 + travel
    r1 = C.apply_op(SPEC, dict(base_ctx, spend_category="travel"))
    chk(sorted(e["card"] for e in r1["eligible"]) == names0, "주석은 eligible 집합을 바꾸지 않음")
    _, sil = entry(r1, "Silver Rewards Card")
    _, gld = entry(r1, "Gold Rewards Card")
    chk(sil and "4.0% (other categories: 1.0%)" in str(sil["facts"].get("rate_for('travel')")),
        "003 리플레이: Silver = travel 4.0%(그외 1.0%) 주석")
    chk(gld and "2.5%" in str(gld["facts"].get("rate_for('travel')"))
        and "no documented bonus" not in str(gld["facts"].get("rate_for('travel')")),
        "Gold = 2.5%(all→base로 선언됨) 주석")
    r2 = C.apply_op(SPEC, {"business": "true", "spend_category": "travel"})
    _, bs = entry(r2, "Business Silver Rewards Card")
    chk(bs and "10.0%" in str(bs["facts"].get("rate_for('travel')")), "Business Silver = travel 10.0%")
    _, bg = entry(r2, "Business Gold Rewards Card")
    chk(bg and "1.0%" in str(bg["facts"].get("rate_for('travel')")),
        "Business Gold = travel엔 base 1.0%(operations만 2.5)")
    # §7-4: 미문서 카드
    r3 = C.apply_op(SPEC, {"spend_category": "travel"})
    _, eco = entry(r3, "EcoCard")
    chk(eco and "unverified" in str(eco["facts"].get("rate_for('travel')", "unverified")),
        "EcoCard = unverified(추측 0)")
    # §7-2b: 동의어 토큰 = 행별 주석 생략 + 정직 안내(rev2 결함1)
    r4 = C.apply_op(SPEC, dict(base_ctx, spend_category="flights"))
    chk("rate_for" not in json.dumps(r4), "미등재 토큰('flights') = 행별 주석 생략(오표기 0)")
    chk("no documented category-specific rate" in r4["note"] and "flights" in r4["note"],
        "미등재 토큰 = note로 정직 안내+재호출 지시")
    # 표 재구조화 사실
    rows = find_table(FIT)
    pl = next(r for r in rows if r["card"] == "Platinum Rewards Card")
    chk(pl.get("base_cashback") == 10.0 and not pl.get("category_rates"),
        "Platinum = all 10%(KB 본문 축자·모순 해소·category_rates 없음)")
    chk(sum(1 for r in rows if r.get("base_cashback") is not None) == 11,
        "재구조화 11행(9+BizGold/BizPlat 신규 문서화 발견)")


def d7_dedup():
    print("D7 계산도구 dedup (환경 스위치·순수 로직 부분):")
    # exec2 내장 로직이라 여기서는 제외 조건의 선언 정합만 검증
    vi = next(t for t in A2["scaffold_get_tools"] if t["name"] == "verify_identity")
    chk((vi.get("variants", {}).get("ledger", {}).get("op") or {}).get("evidence_from"),
        "verify_identity(ledger)=evidence_from 선언 → dedup 제외 조건 성립(005 재호출 보호)")
    rd = next(t for t in A2["scaffold_get_tools"] if t["name"] == "get_reward_discrepancies")
    chk(not (rd.get("op") or {}).get("evidence_from"), "rate 도구=순수 인자 op → dedup 대상(022 표적)")
    chk(not (FIT.get("op") or {}).get("evidence_from"), "fit 도구=순수 인자 op → dedup 대상(003 표적)")


def d8_update_chain():
    print("D8 dispute→update chain:")
    ch = next((c for c in A2["follow_up_chains"]
               if "submit_cash_back_dispute" in (c.get("after") or [])), None)
    chk(ch is not None, "chain 선언 존재")
    if ch is None:
        return
    r = GP._chain_dispatch(ch, {"submit_cash_back_dispute", "give_discoverable_user_tool"})
    chk(r is not None and "update_transaction_rewards" in r[0],
        "028 실측형: 분쟁 제출됨·갱신 미호출 → feedback({missing})")
    chk("If resolution has NOT been confirmed yet, do not update" in r[0],
        "문구 양방향(미해결 시 갱신 금지) = 조기-갱신 Δspurious 완화")
    chk(GP._chain_dispatch(ch, {"submit_cash_back_dispute", "update_transaction_rewards"}) is None,
        "갱신 실행됨 → 무발화(오탐 0)")
    chk(GP._chain_dispatch(ch, {"KB_search"}) is None, "분쟁 미제출 → 무발화")
    chk(ch.get("resign_th") is None, "resign_th 미선언=전역 기본(D2와 달리 장문 대화라 조기 발화 불필요)")


def d9_normalized_name():
    print("D9 정규화 도구명 매칭:")
    decls = {"get_reward_discrepancies": {}, "check_card_application_fit": {}}

    class TC:
        def __init__(self, args):
            self.arguments = args
    chk(SG._a2_named_in_args(TC({"query": "get reward discrepancies"}), decls)
        == "get_reward_discrepancies", "019 실측형: 공백형 질의 → 정규화-동등으로 포착")
    chk(SG._a2_named_in_args(TC({"query": "Get-Reward-Discrepancies"}), decls)
        == "get_reward_discrepancies", "하이픈/대문자형도 포착")
    chk(SG._a2_named_in_args(TC({"query": "how to get reward discrepancies resolved"}), decls) is None,
        "산문(부분일치) = 미포착(기존 오탐 방어 유지)")
    chk(SG._a2_named_in_args(TC({"query": "reward discrepancies"}), decls) is None,
        "부분 이름 = 미포착(동등만)")


if __name__ == "__main__":
    d6_annotation()
    d7_dedup()
    d8_update_chain()
    d9_normalized_name()
    print("\n%s" % ("PASS — C204 다음-런 레버 배선 정상 (라이브 발화는 별도·[[30]])" if OK else "FAIL"))
    sys.exit(0 if OK else 1)
