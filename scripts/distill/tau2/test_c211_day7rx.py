#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C211/day7 처방(F6·F7·F8) 오프라인 검증 (2026-07-28·무료·모델 불요).
`DAY7_PRESCRIPTIONS_DESIGN_2026_07_28` §6 배터리 + 리뷰 반영 케이스:
- [가드레일] 공유 술어 **의도** 불변: banking에서 give_/unlock_ 은 실효 write가 아니다.
  ⚠2026-07-30 C241 U1': 이 술어는 도메인 어휘를 **A2에서** 읽도록 바뀌었다(엔진 리터럴 제거·
  전역 상태 없이 a2 명시 전달). 따라서 a2 없이 호출하면 도메인 어휘를 알 수 없어 write로
  판정된다 — 그게 **설계된 동작**이다(교차-도메인 누출 방지). 검사는 banking A2를 넘겨
  **의도**를 확인하도록 갱신했다. 도메인별 등가성 전수는 `test_c241_u1_predicate.py`.
- [보강1] F7b 복수-매칭=불성립(abstain 안전측)
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

import t2_gate_patch as GP        # noqa: E402
import t2_prekb_patch as PK       # noqa: E402
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
    def __init__(self, name, cid="c1", args=None):
        self.name, self.id, self.arguments = name, cid, (args or {})


class FakeOrch:
    def __init__(self, msgs):
        self._msgs = msgs

    def get_messages(self):
        return self._msgs


A2 = json.load(io.open(os.path.join(HERE, "a2", "banking_knowledge.gate.json"), encoding="utf-8"))
RD = next(t for t in A2["scaffold_get_tools"] if t["name"] == "get_reward_discrepancies")
RF = RD["variants"]["ratefix"]


def _row(tid, amt, rew, date="03/01/2025", rate=4.0, card="X", promo=False, open_=None):
    r = {"transaction_id": tid, "transaction_amount": amt, "rewards_earned": rew,
         "transaction_date": date, "credit_card_type": card, "category": "Y",
         "base_rate": rate, "promo_mult": 2 if promo else 1,
         "promo_window_months": 6 if promo else 0,
         "promo_start": "2024-11-14" if promo else "", "promo_end": "2025-11-14" if promo else ""}
    if open_ is not None:
        r["account_open"] = open_
    return r


def test_f6a():
    print("[test_f6a] 발명-행 비-에코 (id_field 결핍=판정 제외+지목)")
    chk((RF.get("grounded_params") or {}).get("transaction_id", {}).get("producer_contains"),
        "A2 grounded_params.transaction_id 선언 실재")
    # P4b 강등 재현: 발명 행의 id=None → 판정 제외·지목·비-에코
    rows = [_row("txn_real", 100.0, 100, open_="01/01/2025"),          # 100≠400 → discrepant
            dict(_row("gone", 50.0, 10, open_="01/01/2025"), transaction_id=None)]
    ctx = {"transactions": rows}
    res = TC.apply_op(RF["op"], ctx)
    st = ctx["_sg_stats"]
    chk(res == ["txn_real"] and None not in res,
        "id-결핍 행 → out_ids 비-에코(day6 020/028 기전 봉쇄): %s" % res)
    chk(st["skipped"] == 1 and (st["missing_fields"] or {}).get("transaction_id") == 1,
        "skipped+missing_fields['transaction_id'] 계상: %s" % st["missing_fields"])
    # 정합: 발명 id 값 자체는 producer 출력에 없음(강등 술어 동형 확인)
    outs = {"get_credit_card_transactions_by_user":
            "found 2 record(s) in 'credit_card_transaction_history': transaction_id: txn_real"}
    sels = [s.lower() for s in RF["grounded_params"]["transaction_id"]["producer_contains"]]
    cands = [t for t in outs.values() if any(s in t for s in sels)]
    chk(any("txn_real" in t for t in cands) and not any("txn_e0h5i0j0k004" in t for t in cands),
        "실재 id=통과·발명 id(020 실측값)=강등 대상")


def test_f6b():
    print("[test_f6b] FAB_STRIP give-구멍 수리 (국소·공유 술어 불변)")
    # [가드레일] 공유 술어 회귀: give는 여전히 비-실효-write
    from gate_interpreter import load_domain_a2 as _lda
    _a2b = _lda("banking_knowledge")
    chk(GP._is_effective_write("give_discoverable_user_tool", _a2b) is False,
        "공유 _is_effective_write('give_…', bankingA2)=False 유지(claim_prov/WRITEPROV 기반 불변)")
    chk(GP._is_effective_write("unlock_discoverable_agent_tool", _a2b) is False,
        "공유 _is_effective_write('unlock_…', bankingA2)=False 유지")
    # ★C241 U1' 신설 가드레일: a2 없이는 도메인 어휘를 모른다 = 누출 방지의 설계된 결과
    chk(GP._is_effective_write("give_discoverable_user_tool", None) is True,
        "a2 미전달 시 도메인 어휘 미적용(전역 누출 부재의 대우)")
    # 국소 술어 동형 재현(수리 후 로직): inner=discoverable_tool_name까지 unwrap·디스패처는 inner 판정
    import re as _re
    RDP, PRC = GP._READ_PREFIX_RE, GP._PROCEDURAL_RE

    def pred(nm, ar, ctx_lc):
        inner = (ar.get("agent_tool_name") or ar.get("user_tool_name")
                 or ar.get("discoverable_tool_name") or "")
        eff = _re.sub(r"_\d+$", "", str(inner or nm))
        if inner and _re.match(r"^(give|call)_", str(nm), _re.I):
            if RDP.match(eff) or PRC.search(eff):
                return False
        elif not eff or RDP.match(eff) or PRC.search(eff):
            return False
        sub = ar.get("arguments")
        if isinstance(sub, str):
            try:
                sub = json.loads(sub)
            except Exception:
                sub = {}
        d = sub if isinstance(sub, dict) else ar
        for k, v in (d or {}).items():
            if "id" not in k.lower():
                continue
            s = str(v).strip()
            if len(s) >= 4 and s.lower() not in ctx_lc:
                return True
        return False

    ctx_lc = "record txn_real appears here"
    give_fab = {"discoverable_tool_name": "submit_cash_back_dispute_0589",
                "arguments": '{"user_id": "u1", "transaction_id": "txn_e0h5i0j0k004"}'}
    chk(pred("give_discoverable_user_tool", give_fab, ctx_lc) is True,
        "give+발명 txn(중첩) → 검출(day5 022형·구판은 외피 면제로 통과)")
    give_ok = {"discoverable_tool_name": "submit_cash_back_dispute_0589",
               "arguments": '{"user_id": "u1", "transaction_id": "txn_real"}'}
    chk(pred("give_discoverable_user_tool", give_ok, ctx_lc) is False,
        "give+실재 txn → 무개입")
    give_read = {"discoverable_tool_name": "get_card_last_4_digits",
                 "arguments": '{"credit_card_account_id": "cc_x"}'}
    chk(pred("give_discoverable_user_tool", give_read, ctx_lc) is False,
        "give+read-성 inner → 면제(inner 기준 판정)")


ACC_DUMP = ("Found 2 record(s) in 'credit_card_accounts':\n\n"
            "1. Record ID: cc_u_silver\n"
            "   account_id: cc_u_silver\n"
            "   card_type: Silver Rewards Card\n"
            "   date_of_account_open: 01/20/2025\n\n"
            "2. Record ID: cc_u_bsilver\n"
            "   account_id: cc_u_bsilver\n"
            "   card_type: Business Silver Rewards Card\n"
            "   date_of_account_open: 02/13/2025\n")

TXN_DUMP = ("Found 2 record(s) in 'credit_card_transaction_history':\n\n"
            "1. Record ID: txn_aa\n"
            "   transaction_id: txn_aa\n"
            "   transaction_amount: $100.00\n"
            "   rewards_earned: 400 points\n"
            "   transaction_date: 03/01/2025\n"
            "   credit_card_type: Silver Rewards Card\n"
            "   category: Y\n\n"
            "2. Record ID: txn_bb\n"
            "   transaction_id: txn_bb\n"
            "   transaction_amount: $100.00\n"
            "   rewards_earned: 1000 points\n"
            "   transaction_date: 03/05/2025\n"
            "   credit_card_type: Business Silver Rewards Card\n"
            "   category: Y\n")


def test_f7b():
    print("[test_f7b] BYREF equijoin (유일-매칭·복수=불성립·026/027형 재생)")
    chk("byref_join" in RF and "account_open" in RF["byref_join"], "A2 byref_join 선언 실재")
    msgs = [M("assistant", None, tool_calls=[TCall("get_txns", "cT"), TCall("get_accts", "cA")]),
            M("tool", TXN_DUMP, mid="cT"), M("tool", ACC_DUMP, mid="cA")]
    orch = FakeOrch(msgs)
    d = {"name": "t", "op": RF["op"], "byref_join": RF["byref_join"]}
    ctx = {"transactions": "@last:get_txns", "account_open": "@last:get_accts"}
    SG._byref_resolve(orch, d, ctx)
    SG._byref_join(orch, d, ctx)
    rows = ctx["transactions"]
    chk([r.get("account_open") for r in rows] == ["01/20/2025", "02/13/2025"],
        "카드별 유일 매칭 조인(실개설일 — 027의 날조-운 제거 경로): %s"
        % [r.get("account_open") for r in rows])
    chk("account_open" not in ctx, "소비된 참조 파라미터 제거")
    # [보강1] 복수-매칭 = 불성립(안전측)
    dup = ACC_DUMP + ("\n3. Record ID: cc_u_silver2\n   account_id: cc_u_silver2\n"
                      "   card_type: Silver Rewards Card\n   date_of_account_open: 05/05/2025\n")
    msgs2 = [M("assistant", None, tool_calls=[TCall("get_txns", "cT"), TCall("get_accts", "cA")]),
             M("tool", TXN_DUMP, mid="cT"), M("tool", dup, mid="cA")]
    ctx2 = {"transactions": "@last:get_txns", "account_open": "@last:get_accts"}
    SG._byref_resolve(FakeOrch(msgs2), d, ctx2)
    SG._byref_join(FakeOrch(msgs2), d, ctx2)
    r0 = ctx2["transactions"][0]
    chk("account_open" not in r0,
        "복수-매칭(Silver 2계좌) → 불성립=미기입(침묵-오값 차단·D4형 방지)")
    # 잘못된 소스 참조(selector 불일치) → 명확한 에러
    ctx3 = {"transactions": "@last:get_txns", "account_open": "@last:get_txns"}
    try:
        SG._byref_resolve(FakeOrch(msgs), d, ctx3)
        SG._byref_join(FakeOrch(msgs), d, ctx3)
        chk(False, "selector 불일치 거부")
    except SG._ByrefError:
        chk(True, "selector 불일치(거래 덤프를 계좌 소스로) → _ByrefError")
    # e2e: 조인 후 판정 — promo BS행(1000 vs 100×10×2=2000? base 10·×2) 판정 가능해야
    rows_e = ctx["transactions"]
    for r in rows_e:
        bs = "Business" in r["credit_card_type"]
        r.update({"base_rate": 10.0 if bs else 4.0, "promo_mult": 2 if bs else 1,
                  "promo_window_months": 6 if bs else 0,
                  "promo_start": "2024-11-14" if bs else "",
                  "promo_end": "2025-11-14" if bs else ""})
    ctxe = {"transactions": rows_e}
    res = TC.apply_op(RF["op"], ctxe)
    st = ctxe["_sg_stats"]
    chk(st["judged"] == 2 and st["skipped"] == 0,
        "조인된 개설일로 promo 행 포함 전수 판정(judged=%s)" % st["judged"])
    chk(res == ["txn_bb"], "BS promo 오적립 검출(1000≠100×10×2): %s" % res)


def test_f8():
    print("[test_f8] arg_producers 트리거 3분기")
    chk((A2.get("arg_producers") or {}).get("card_last_4_digits", {}).get("user_tool")
        == "get_card_last_4_digits", "A2 arg_producers 선언 실재")
    h = PK._argprod_hits(A2, "Error: Missing required parameter: card_last_4_digits")
    chk(h == [("card_last_4_digits", "get_card_last_4_digits")],
        "필수-인자 에러+선언 인자명 → 트리거: %s" % h)
    chk(PK._argprod_hits(A2, "Error: something else entirely") == [],
        "무관 에러 → 무발화")
    chk(PK._argprod_hits(A2, "card_last_4_digits is a nice field") == [],
        "required/missing 부재 → 무발화(느슨 매칭 방지)")
    chk(PK._argprod_hits({}, "Missing required: card_last_4_digits") == [],
        "선언 없는 도메인(retail형) → 전역 무개입")


if __name__ == "__main__":
    for fn in (test_f6a, test_f6b, test_f7b, test_f8):
        fn()
        print()
    print("RESULT: %s" % ("ALL PASS" if OK else "FAIL"))
    sys.exit(0 if OK else 1)
