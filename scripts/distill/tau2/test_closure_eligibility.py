# -*- coding: utf-8 -*-
"""관문3 (043·§2aa 요건②③) 오프라인 검증 — 종단행동(close)을 실제 실격조건(잔액>0)으로 게이트.

실제 A2 + 라이브 함수(apply_op·_ground_operands·_render_scalar·_wev_deny_msgs)를 그대로 잰다([[03b]]).
043 재현: $75 미납 → 구판 3체크 WEV는 초록불·신판 CLOSURE_OK WEV는 차단. 잔액0 날조=source_param이 차단.
전 체인: 판정도구 반환문(id+토큰 에코) → WEV 증거로 성립 → close 허용.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t2_scaffold_get as sg  # noqa: E402
import t2_compute as _c  # noqa: E402
from t2_gate_patch import _wev_deny_msgs  # noqa: E402

A2 = json.load(open(os.path.join(HERE, "a2", "banking_knowledge.gate.json"), encoding="utf-8"))
TOOL = next(d for d in A2["scaffold_get_tools"] if d["name"] == "check_card_closure_eligibility")
WEV = [s for s in A2["write_evidence_specs"]
       if (s.get("applies_when") or {}).get("prefix") == "close_credit_card_account"]
WEV_ELIG = [s for s in WEV if "CLOSURE_OK" in (s.get("require_tokens") or [])]

LEDGER = ("get_credit_card_account: {credit_card_account_id: acc_cc_43, status: active, "
          "outstanding_balance: $75.00, credit_limit: 5000}")


class _FakeOrch:
    class environment:
        domain_name = "banking"


def _install_corpus():
    sg._DOC_CACHE.clear()
    sg._load_domain_docs = lambda domain: []
    sg._evidence_ctx = lambda orch: {"__tool_outputs": {"ledger": LEDGER.lower()}, "__user_text": ""}


class _TC:
    def __init__(self, name, arguments):
        self.name, self.arguments = name, arguments


class _Msg:
    def __init__(self, role, content):
        self.role, self.content = role, content


PASS, FAIL = [], []


def check(label, cond):
    (PASS if cond else FAIL).append(label)
    print(("  PASS " if cond else "  FAIL ") + label)


def main():
    _install_corpus()
    check("A2: 판정도구 선언 존재", TOOL is not None)
    check("A2: CLOSURE_OK WEV 스펙 존재", len(WEV_ELIG) == 1)
    check("토큰 비중첩(BLOCKED 문구에 CLOSURE_OK 미포함)",
          "CLOSURE_OK" not in TOOL["op"]["else"]["value"])

    # ── 043 재현: 잔액 75 (원장 실재) → BLOCKED ──
    ctx = {"credit_card_account_id": "acc_cc_43", "outstanding_balance": 75,
           "balance_source": "outstanding_balance: $75.00"}
    flags = sg._ground_operands(_FakeOrch(), TOOL, ctx)
    check("043: 진짜 잔액 75 grounding 통과", not flags and ctx["outstanding_balance"] == 75)
    res = _c.apply_op(TOOL["op"], ctx)
    check("043: 잔액 75 → BLOCKED 판정", res is not None and "CLOSURE_BLOCKED" in str(res))
    txt = sg._render_scalar(TOOL, ctx, res)
    check("043: 반환문에 id+판정 에코", "acc_cc_43" in txt and "CLOSURE_BLOCKED" in txt
          and "CLOSURE_OK" not in txt)

    # ── 잔액 0 날조 (원장은 75) → source_param 축자검증이 드롭 → abstain ──
    ctx = {"credit_card_account_id": "acc_cc_43", "outstanding_balance": 0,
           "balance_source": "outstanding_balance: $0.00"}          # 날조 인용
    flags = sg._ground_operands(_FakeOrch(), TOOL, ctx)
    check("날조0: source 미실재 → 드롭+플래그", len(flags) == 1 and ctx["outstanding_balance"] is None)
    res = _c.apply_op(TOOL["op"], ctx)
    check("날조0: if_then 3-값 → None(abstain·오판 0)", res is None)
    txt = sg._render_scalar(TOOL, ctx, res)
    check("날조0: missing_hint 반환(CLOSURE_OK 미발급)", "COULD NOT VERIFY" in txt
          and "CLOSURE_OK" not in txt)

    # ── 인용 실재·값 불일치(인용은 75인데 balance=0 주장) → 드롭 ──
    ctx = {"credit_card_account_id": "acc_cc_43", "outstanding_balance": 0,
           "balance_source": "outstanding_balance: $75.00"}
    flags = sg._ground_operands(_FakeOrch(), TOOL, ctx)
    check("값불일치: value ∉ source → 드롭", ctx["outstanding_balance"] is None and len(flags) == 1)

    # ── 완납 후: 원장에 0 실재 → CLOSURE_OK ──
    sg._evidence_ctx = lambda orch: {"__tool_outputs": {
        "ledger": (LEDGER + " ... pay_credit_card_balance result: payment accepted; "
                   "outstanding_balance: $0.00").lower()}, "__user_text": ""}
    ctx = {"credit_card_account_id": "acc_cc_43", "outstanding_balance": 0,
           "balance_source": "outstanding_balance: $0.00"}
    flags = sg._ground_operands(_FakeOrch(), TOOL, ctx)
    res = _c.apply_op(TOOL["op"], ctx)
    ok_txt = sg._render_scalar(TOOL, ctx, res)
    check("완납후: 0 grounded → CLOSURE_OK", not flags and res is not None and "CLOSURE_OK" in str(res))
    check("완납후: 반환문 id+CLOSURE_OK 에코", "acc_cc_43" in ok_txt and "CLOSURE_OK" in ok_txt)

    # ── WEV 게이트: close 호출 ──
    close_tc = _TC("call_discoverable_agent_tool",
                   {"agent_tool_name": "close_credit_card_account_9921",
                    "credit_card_account_id": "acc_cc_43"})
    # (a) 구판 3체크만 있는 궤적(043 실제) → 신판 WEV가 차단
    msgs_3checks = [_Msg("tool", "Executed get_closure_reason_history for acc_cc_43: none"),
                    _Msg("tool", "Executed log_credit_card_closure_reason for acc_cc_43: logged"),
                    _Msg("tool", "Executed get_pending_replacement_orders for acc_cc_43: none")]
    deny = _wev_deny_msgs(msgs_3checks, close_tc, WEV_ELIG)
    check("WEV: 3체크만(043 궤적) → close 차단", deny is not None and "acc_cc_43" in deny)
    # (b) BLOCKED 판정만 있는 궤적 → 여전히 차단(BLOCKED에 CLOSURE_OK 미포함)
    deny = _wev_deny_msgs(msgs_3checks + [_Msg("tool", txt)], close_tc, WEV_ELIG)
    check("WEV: BLOCKED 판정 → 여전히 차단", deny is not None)
    # (c) CLOSURE_OK 판정(우리 반환문) → 허용
    deny = _wev_deny_msgs(msgs_3checks + [_Msg("tool", ok_txt)], close_tc, WEV_ELIG)
    check("WEV: CLOSURE_OK 반환문=증거 → close 허용", deny is None)
    # (d) 다른 계좌의 CLOSURE_OK → id-공존 불성립 → 차단
    other = ok_txt.replace("acc_cc_43", "acc_cc_99")
    deny = _wev_deny_msgs(msgs_3checks + [_Msg("tool", other)], close_tc, WEV_ELIG)
    check("WEV: 타계좌 CLOSURE_OK → id 불공존 차단", deny is not None)

    print("\n== 결과: %d PASS / %d FAIL ==" % (len(PASS), len(FAIL)))
    if FAIL:
        for f in FAIL:
            print("  - FAILED: " + f)
        sys.exit(1)
    print("ALL PASS — 관문3: close가 실제 실격조건(잔액)으로 게이트·날조0 차단·완납 경로 성립.")


if __name__ == "__main__":
    main()
