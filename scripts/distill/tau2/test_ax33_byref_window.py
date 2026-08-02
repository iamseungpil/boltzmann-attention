#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AX33 §4-1/§4-2 오프라인 검증 (2026-08-03·무료·모델 불요).

정본 = `HANDOFF_2026_08_02_NIGHT.md` §4-1(byref 구조 결함)·§4-2(표본 충분성 미검) +
`AX32_MIDRUN_PRESCRIPTIONS_DESIGN_2026_08_02` P4ⓐ.

§4-1: `_byref_resolve`가 **최상위 op의 `over`만** 읽어 중첩 op 트리(rebate: `op.cond.a.over`)에서
      항상 `over=None` ⇒ 에이전트의 옳은 by-ref 시도를 "이 도구엔 by-ref 인자가 없다"로 차단 →
      손-전사 1건 폴백(028 사슬과 공통 원인).
§4-2: 그 1건으로 `across=min`이 **부정 판정을 확정 발급**(정답 QUALIFIES). 결여 윈도=0이 아니라
      미측정 ⇒ abstain + 표면화.

⚠단위통과≠라이브발화([[30]])."""
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

import t2_compute as TC           # noqa: E402
import t2_scaffold_get as SG      # noqa: E402

OK = True


def chk(c, m):
    global OK
    OK &= bool(c)
    print(("  ✓ " if c else "  ✗ ") + m)


class M:
    def __init__(self, role, content=None, tool_calls=None, mid=None, error=False):
        self.role, self.content, self.tool_calls = role, content, tool_calls
        self.id, self.error = mid, error


class TCall:
    def __init__(self, name, cid="c1"):
        self.name, self.id = name, cid


class FakeOrch:
    def __init__(self, msgs):
        self._msgs = msgs

    def get_messages(self):
        return self._msgs


A2 = json.load(io.open(os.path.join(HERE, "a2", "banking_knowledge.gate.json"), encoding="utf-8"))
RB = next(t for t in A2["scaffold_get_tools"] if t["name"] == "check_rebate_qualification")
RD = next(t for t in A2["scaffold_get_tools"] if t["name"] == "get_reward_discrepancies")

GETTER = "get_credit_card_transactions_by_user"
ANCHOR = "11/10/2022"          # 개설일(합성값·gold 무관)
ASOF = "12/01/2025"            # 마지막 완결 기념년 = 2024-11-10 ~ 2025-11-09


def _dump(rows):
    """env 기계 포맷 record dump 재현(`_parse_record_dump` 계약)."""
    out = ["Found %d record(s) in 'credit_card_transaction_history':\n" % len(rows)]
    for i, r in enumerate(rows, 1):
        out.append("%d. Record ID: %s" % (i, r["transaction_id"]))
        for k, v in r.items():
            out.append("   %s: %s" % (k, v))
        out.append("")
    return "\n".join(out)


def _txn(i, date, amt):
    return {"transaction_id": "txn_%02d" % i, "transaction_amount": "$%,.2f".replace(",", "")
            % amt if False else "$%.2f" % amt, "transaction_date": date,
            "credit_card_type": "Platinum Rewards Card"}


def _full_year(per_month=800.0):
    """마지막 완결 기념년(2024-11-10~2025-11-09)의 12 윈도를 매달 1건씩 채운다."""
    rows, i = [], 0
    for k in range(12):
        mo = (11 + k - 1) % 12 + 1
        yr = 2024 + (11 + k - 1) // 12
        i += 1
        rows.append(_txn(i, "%02d/15/%d" % (mo, yr), per_month))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
def test_over_params():
    print("[test_over_params] §4-1 중첩 op 트리에서 배열 파라미터 도출")
    chk(SG._over_params(RB["op"]) == ["transactions"],
        "rebate(op.cond.a.over 중첩) → ['transactions'] (구판=None): %s" % SG._over_params(RB["op"]))
    rf = RD["variants"]["ratefix"]
    chk(SG._over_params(rf["op"])[:1] == [rf["op"]["over"]],
        "최상위 over 스펙은 구판과 동일값(거동보존): %s" % SG._over_params(rf["op"])[:1])
    chk(SG._primary_over(RB, {"transactions": []}) == "transactions", "_primary_over = ctx 실재 배열")


def test_byref_open():
    print("[test_byref_open] §4-1 rebate byref 개통 + 컬럼명 매핑 + 필요 컬럼 지목")
    rows = _full_year()
    orch = FakeOrch([M("assistant", None, tool_calls=[TCall(GETTER, "cT")]),
                     M("tool", _dump(rows), mid="cT")])
    ctx = {"transactions": "@last:%s" % GETTER, "account_opening_date": ANCHOR,
           "monthly_threshold": 500, "as_of_date": ASOF}
    SG._byref_resolve(orch, RB, ctx)
    got = ctx["transactions"]
    chk(isinstance(got, list) and len(got) == 12, "참조 해석 → 12행(구판=_ByrefError): %s" % type(got))
    chk(all(("date" in r and "amount" in r) for r in got),
        "A2 byref_field_map: transaction_date/amount → date/amount 복사")
    chk(TC.apply_op(RB["op"], dict(ctx)) == "QUALIFIES",
        "byref 행으로 판정 성립(엔진 산식 정상·P4ⓐ 조사 결론과 정합)")
    # 비-over 인자 참조 = 정확한 허용목록 문구(구판은 "only the 'None' argument…"로 깨졌다)
    ctx2 = dict(ctx, transactions=rows, monthly_threshold="@last:%s" % GETTER)
    try:
        SG._byref_resolve(orch, RB, ctx2)
        chk(False, "비-over 인자 거부")
    except SG._ByrefError as e:
        chk("'transactions'" in str(e) and "None" not in str(e),
            "허용 인자를 정확히 지목('transactions')·'None' 문구 소멸: %s" % str(e)[:90])
    # 필요 컬럼이 없는 덤프 참조 → 침묵 아니라 지목
    bad = [{"transaction_id": "txn_x", "merchant_name": "M"}]
    orch3 = FakeOrch([M("assistant", None, tool_calls=[TCall(GETTER, "cB")]),
                      M("tool", _dump(bad), mid="cB")])
    try:
        SG._byref_resolve(orch3, RB, {"transactions": "@last:%s" % GETTER})
        chk(False, "필요 컬럼 부재 지목")
    except SG._ByrefError as e:
        chk("'date'" in str(e) and "'amount'" in str(e) and "merchant_name" in str(e),
            "부재 컬럼 + 실재 컬럼 병기(빈 집계 침묵 차단): %s" % str(e)[:100])


def test_window_abstain():
    print("[test_window_abstain] §4-2 미측정 윈도 = abstain + 표면화")
    one = [dict(_txn(1, "11/15/2024", 800.0), date="11/15/2024", amount=800.0)]
    ctx1 = {"transactions": one, "account_opening_date": ANCHOR,
            "monthly_threshold": 500, "as_of_date": ASOF}
    os.environ.pop("T2_SG_WINDOW_ABSTAIN", None)
    chk(TC.apply_op(RB["op"], dict(ctx1)) == "QUALIFIES",
        "플래그 OFF = 구 거동 보존(1건만으로 판정 발급 — 이것이 023 결함)")
    os.environ["T2_SG_WINDOW_ABSTAIN"] = "1"
    c1 = dict(ctx1)
    chk(TC.apply_op(RB["op"], c1) is None, "플래그 ON + 11칸 공백 → abstain(None)")
    grm = c1.get("_gr_missing")
    chk(isinstance(grm, dict) and len(grm.get("missing") or []) == 11 and grm.get("expected") == 12,
        "_gr_missing = 11/12 지목: %s" % (grm or {}).get("missing"))
    note = SG._window_coverage_note(RB, c1, None)
    chk("11 of 12 windows" in note and "#2" in note, "표면화 문구에 결여 윈도 수·번호: %s" % note[:70])
    chk(RB.get("incomplete_hint") and RB["incomplete_hint"] in note,
        "A2 incomplete_hint 부착(도메인 문구=A2·엔진 리터럴 0)")
    chk(SG._window_coverage_note(RB, {}, "QUALIFIES") == "", "정상 판정 시 문구 없음(거동보존)")
    # 12칸 전부 채워지면 판정 정상 발급(과잉 abstain 없음 = Δspurious 0 확인)
    full = [dict(r, date=r["transaction_date"], amount=float(r["transaction_amount"][1:]))
            for r in _full_year()]
    c2 = {"transactions": full, "account_opening_date": ANCHOR,
          "monthly_threshold": 500, "as_of_date": ASOF}
    chk(TC.apply_op(RB["op"], c2) == "QUALIFIES", "완결 12윈도 → 정상 판정(과잉 abstain 없음)")
    c3 = dict(c2, monthly_threshold=1000)
    chk(TC.apply_op(RB["op"], c3) == "DOES NOT QUALIFY", "완결 12윈도 + 미달 → 정상 부정 판정")
    os.environ.pop("T2_SG_WINDOW_ABSTAIN", None)


def test_no_regression_top_level():
    print("[test_no_regression] 최상위-over 도구(get_reward_discrepancies) 거동 불변")
    rf = RD["variants"]["ratefix"]
    d = {"name": "t", "op": rf["op"], "byref_join": rf.get("byref_join")}
    dump = ("Found 1 record(s) in 'credit_card_transaction_history':\n\n"
            "1. Record ID: txn_aa\n   transaction_id: txn_aa\n   transaction_amount: $100.00\n"
            "   rewards_earned: 400 points\n   transaction_date: 03/01/2025\n"
            "   credit_card_type: Silver Rewards Card\n   category: Y\n")
    orch = FakeOrch([M("assistant", None, tool_calls=[TCall("get_txns", "cT")]),
                     M("tool", dump, mid="cT")])
    ctx = {"transactions": "@last:get_txns"}
    SG._byref_resolve(orch, d, ctx)
    chk(isinstance(ctx["transactions"], list) and len(ctx["transactions"]) == 1,
        "기존 byref 경로 정상(필요 컬럼 id/actual 실재)")


for fn in (test_over_params, test_byref_open, test_window_abstain, test_no_regression_top_level):
    fn()
print("\n%s" % ("ALL PASS" if OK else "FAIL"))
sys.exit(0 if OK else 1)
