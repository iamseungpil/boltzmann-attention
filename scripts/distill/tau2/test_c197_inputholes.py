# -*- coding: utf-8 -*-
"""C197 입력-결함 침묵 경로 봉쇄 테스트 (2026-07-26 · 019/020 [S] 포렌식 기반).
오프라인·무료. 근거: RESEARCH_MASTER C197 · day2 리플레이(019 str-인자 침묵 "(none)"·020 account_open
누락 promo 강제 해제). 픽스처 수치 = bank_day2frontB 020 실측(operands=day2B_operands.jsonl 축자)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t2_compute as C

RATEFIX_OP = {
    "op": "select_discrepant", "over": "transactions", "id_field": "transaction_id",
    "actual_field": "rewards_earned", "tolerance": 1,
    "steps": {
        "promo_elig": {"op": "date_between", "x": "r.account_open",
                       "lo": "r.promo_start", "hi": "r.promo_end"},
        "promo_active": {"op": "date_in_window", "anchor": "r.account_open",
                         "target": "r.transaction_date", "months": "r.promo_window_months"},
        "mult": {"op": "if_then",
                 "cond": {"op": "bool_expr", "all": [{"ref": "steps.promo_elig"},
                                                     {"ref": "steps.promo_active"}]},
                 "then": {"op": "ref", "path": "r.promo_mult"},
                 "else": {"op": "const", "value": 1}},
        "rate": {"op": "multiply", "a": "r.base_rate", "b": "steps.mult"},
        "expected": {"op": "multiply", "a": "r.transaction_amount", "b": "steps.rate"},
    },
    "expected_ref": "steps.expected",
}

PROMO = {"promo_mult": 2, "promo_window_months": 6,
         "promo_start": "2024-11-14", "promo_end": "2025-11-14"}


def row(tid, amt, rew, date, rate, promo=False, account_open=None):
    r = {"transaction_id": tid, "transaction_amount": amt, "rewards_earned": rew,
         "transaction_date": date, "base_rate": rate,
         "promo_mult": 1, "promo_window_months": 0, "promo_start": "", "promo_end": ""}
    if promo:
        r.update(PROMO)
    if account_open:
        r["account_open"] = account_open
    return r


def t1_date_fns_none_propagation():
    assert C._date_between(None, "2024-11-14", "2025-11-14") is None
    assert C._date_between("", "2024-11-14", "2025-11-14") is None
    assert C._date_between("02/13/2025", "2024-11-14", "2025-11-14") is True
    assert C._date_in_window(None, "02/20/2025", 6) is None
    assert C._date_in_window("02/13/2025", "02/20/2025", 6) is True
    assert C._date_in_window("02/13/2025", "10/20/2025", 6) is False  # 창 밖=확정 False 유지
    print("t1 date-None-전파 OK")


def t2_ifthen_equal_branches():
    # cond 미확정 + 양분기 동일 → 그 값 (무-프로모 행 보존)
    spec = {"op": "if_then", "cond": {"op": "date_between", "x": "r.account_open",
                                     "lo": "r.promo_start", "hi": "r.promo_end"},
            "then": {"op": "ref", "path": "r.promo_mult"}, "else": {"op": "const", "value": 1}}
    assert C.apply_op(spec, {"r": {"promo_mult": 1, "promo_start": "", "promo_end": ""}}) == 1
    # cond 미확정 + 분기 상이 → None(abstain)
    assert C.apply_op(spec, {"r": dict(PROMO)}) is None
    print("t2 if_then 동일분기 OK")


def t3_notalist_stats():
    ctx = {"transactions": "[{'transaction_id': 'x', 'rewards_earned': 0223}]"}  # 019 실측형
    res = C.apply_op(RATEFIX_OP, ctx)
    assert res == [] and ctx.get("_sg_stats") == {"judged": 0, "skipped": 0, "total": 0}, \
        (res, ctx.get("_sg_stats"))
    print("t3 not-a-list stats OK")


def t4_missing_account_open_abstains():
    """020 실측 축소판: promo 행은 account_open 없으면 skipped(오판정 아님)·무-프로모 행은 판정."""
    rows = [
        # 020 실측형: 401 recorded 9000 = 450*10*2(프로모 정당·rate=points/$) — 구판 FP(4500), 신판 판정불가
        row("t401", 450.0, 9000, "02/20/2025", 10.0, promo=True),
        # 020 실측형: 403 recorded 3150 = 프로모 누락 기록(정답 315*20=6300) — 구판 FN, 신판 판정불가
        row("t403", 315.0, 3150, "02/25/2025", 10.0, promo=True),
        # 무-프로모(Silver 506형): recorded 2550 vs 정답 255*4=1020 → account_open 없어도 판정·플래그
        row("t506", 255.0, 2550, "03/01/2025", 4.0),
    ]
    ctx = {"transactions": rows}
    res = C.apply_op(RATEFIX_OP, ctx)
    st = ctx["_sg_stats"]
    assert sorted(res) == ["t506"], res
    # ★P4(C208④·2026-07-28): _sg_stats에 missing_fields 키 추가(결핍-필드 지목) — 부분집합 대조로 갱신.
    assert {k: st[k] for k in ("judged", "skipped", "total")} == \
        {"judged": 1, "skipped": 2, "total": 3}, st
    assert (st.get("missing_fields") or {}).get("account_open") == 2, st
    print("t4 account_open 누락=abstain(판정 1·불가 2·missing_fields 지목) OK")


def t5_with_account_open_full_judgment():
    """반사실(리모트 [S] 재현과 동형): 개설일 주입 시 프로모 정합/부정합 정확 판정."""
    rows = [
        row("t401", 450.0, 9000, "02/20/2025", 10.0, promo=True, account_open="02/13/2025"),
        row("t403", 315.0, 3150, "02/25/2025", 10.0, promo=True, account_open="02/13/2025"),
        row("t506", 255.0, 2550, "03/01/2025", 4.0, account_open="01/20/2025"),
        # 프로모 창 밖(개설+6mo 초과·확정 False) → mult 1: recorded=10*10=100 일치 → 비플래그
        row("tOUT", 10.0, 100, "10/20/2025", 10.0, promo=True, account_open="02/13/2025"),
    ]
    ctx = {"transactions": rows}
    res = C.apply_op(RATEFIX_OP, ctx)
    st = ctx["_sg_stats"]
    assert sorted(res) == ["t403", "t506"], res            # 401=프로모 정당(9000=9000)·403=FN 해소
    assert {k: st[k] for k in ("judged", "skipped", "total")} == \
        {"judged": 4, "skipped": 0, "total": 4}, st        # P4: missing_fields 키 추가(완전입력={})
    assert st.get("missing_fields") == {}, st
    print("t5 개설일 주입=전수 판정·403 검출 OK")


def t6_no_regression_017_shape():
    """017형(정상 JSON·무-프로모·Silver): 기존 판정 거동 보존."""
    rows = [row("a", 100.0, 400, "02/01/2025", 4.0),      # 400=100*4 → 비플래그
            row("b", 100.0, 100, "02/02/2025", 4.0)]      # 100≠400 → 플래그
    ctx = {"transactions": rows}
    res = C.apply_op(RATEFIX_OP, ctx)
    st = ctx["_sg_stats"]
    assert res == ["b"] and {k: st[k] for k in ("judged", "skipped", "total")} == \
        {"judged": 2, "skipped": 0, "total": 2}            # P4: missing_fields 키 추가
    print("t6 무-프로모 회귀보존 OK")


if __name__ == "__main__":
    t1_date_fns_none_propagation()
    t2_ifthen_equal_branches()
    t3_notalist_stats()
    t4_missing_account_open_abstains()
    t5_with_account_open_full_judgment()
    t6_no_regression_017_shape()
    print("ALL C197 t2_compute tests PASS")
