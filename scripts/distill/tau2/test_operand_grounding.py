# -*- coding: utf-8 -*-
"""operand grounding 오프라인 검증 (관문1·2026-07-20·`ACCOUNT_APY_OFFLOAD §2a` 리뷰③ 배선).

실제 A2 `ground` 선언(banking_knowledge.gate.json)을 로드해, 포렌식(§2ab/2ac/2ae)의
097(날조)·095(오독)·023(개설일 오독) operand가 **드롭+플래그**되고 정상 operand는 통과하는지 증명.
엔진 검증 로직만 테스트(유료 X·[[09]] 무료검증). 코퍼스(KB/원장)는 통제값으로 주입.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t2_scaffold_get as sg  # noqa: E402

A2 = json.load(open(os.path.join(os.path.dirname(__file__), "a2", "banking_knowledge.gate.json"),
                   encoding="utf-8"))
DECLS = {d["name"]: d for d in A2["scaffold_get_tools"]}

# ── 통제 코퍼스 ────────────────────────────────────────────────────────────────
KB_SAVINGS = ("Gold Saver savings accounts earn a base APY of 5.00%. Pairing an eligible checking "
              "account adds a 0.50% checking APY boost. Holding the Platinum card adds a 0.25% card "
              "APY bonus. A relationship tier bonus of 0.10% applies on top of the base rate.")
KB_REBATE = ("Platinum Rewards Card: the $150 annual fee is rebated only if the cardmember spends at "
             "least $2,000 in every monthly window of the cardmember year; a single month below the "
             "threshold disqualifies, with no proration.")
LEDGER = ("get_savings_account: {account_id: acc_sav_7, balance: 100000.00, applied_apy: 3.35, "
          "period_start: 01/01/2024, period_end: 12/31/2024}. "
          "get_credit_card_account: {account_id: acc_cc_9, opened: 11/05/2024, balance: 4200.00}.")


class _FakeEnv:
    domain_name = "banking"


class _FakeOrch:
    environment = _FakeEnv()


def _install_corpus():
    """엔진 코퍼스 소스를 통제값으로 몽키패치(라이브 KB/원장 대신)."""
    sg._DOC_CACHE.clear()
    sg._load_domain_docs = lambda domain: [
        {"title": "savings", "content": KB_SAVINGS},
        {"title": "rebate", "content": KB_REBATE},
    ]
    sg._evidence_ctx = lambda orch: {
        "__tool_outputs": {"ledger": LEDGER.lower()},
        "__user_text": "",
    }


def _run(name, ctx):
    return sg._ground_operands(_FakeOrch(), DECLS[name], ctx)


def _kinds(arr):
    return [e.get("kind") for e in arr]


PASS, FAIL = [], []


def check(label, cond):
    (PASS if cond else FAIL).append(label)
    print(("  PASS " if cond else "  FAIL ") + label)


def main():
    _install_corpus()

    # ── 097 (날조) get_correct_savings_apy ───────────────────────────────────────
    print("[097 날조] get_correct_savings_apy — source 축자아님 + base 0.05 추측")
    ctx = {"components": [
        {"kind": "base", "value": 0.05, "source": "Base APY for Gold savings"},          # 날조 source
        {"kind": "checking", "value": 0.02, "source": "checking boost estimate"},         # 날조 source
    ]}
    flags = _run("get_correct_savings_apy", ctx)
    check("097: 날조 component 전멸 드롭(2건)", len(ctx["components"]) == 0 and len(flags) == 2)

    # ── 095 (오독) 실제 인용이나 값 오독 + 정상 component 공존 ──────────────────────
    print("[095 오독] get_correct_savings_apy — 실제 KB 인용이나 base 값 오독(0.05 vs 5.0)")
    ctx = {"components": [
        {"kind": "base", "value": 0.05, "source": "earn a base APY of 5.00%"},            # 인용 실재·값 오독
        {"kind": "checking", "value": 0.5, "source": "adds a 0.50% checking APY boost"},   # 정상
        {"kind": "tier", "value": 0.10, "source": "relationship tier bonus of 0.10%"},     # 정상
    ]}
    flags = _run("get_correct_savings_apy", ctx)
    check("095: 오독 base 드롭(값 ∉ 인용)", "base" not in _kinds(ctx["components"]))
    check("095: 정상 checking 유지(false-drop 0)", "checking" in _kinds(ctx["components"]))
    check("095: 정상 tier 유지(false-drop 0)", "tier" in _kinds(ctx["components"]))
    check("095: 플래그 정확히 1건(오독 base만)", len(flags) == 1)

    # ── 전부 정상(회귀: grounding이 정답을 안 죽이나) ────────────────────────────────
    print("[정상] get_correct_savings_apy — 전 component 정상(회귀)")
    ctx = {"components": [
        {"kind": "base", "value": 5.0, "source": "earn a base APY of 5.00%"},
        {"kind": "checking", "value": 0.5, "source": "adds a 0.50% checking APY boost"},
        {"kind": "card", "value": 0.25, "source": "Platinum card adds a 0.25% card APY bonus"},
    ]}
    flags = _run("get_correct_savings_apy", ctx)
    check("정상: 전 component 유지·플래그 0", len(ctx["components"]) == 3 and len(flags) == 0)

    # ── 097 principal 추측 get_interest_correction ──────────────────────────────
    print("[097 principal] get_interest_correction — principal 95000 추측(레코드=100000)")
    ctx = {"expected_apy": 6.85, "actual_apy": 3.35, "principal": 95000,
           "period_start": "01/01/2024", "period_end": "12/31/2024"}
    flags = _run("get_interest_correction", ctx)
    check("097: 추측 principal 드롭(→None)", ctx["principal"] is None)
    check("097: 정상 actual_apy 유지", ctx["actual_apy"] == 3.35)
    check("097: 정상 기간 유지", ctx["period_start"] == "01/01/2024")

    print("[정상] get_interest_correction — principal 레코드값 100000(회귀)")
    ctx = {"expected_apy": 6.85, "actual_apy": 3.35, "principal": 100000,
           "period_start": "01/01/2024", "period_end": "12/31/2024"}
    flags = _run("get_interest_correction", ctx)
    check("정상: 100000 유지·플래그 0", ctx["principal"] == 100000 and len(flags) == 0)

    # ── 023 개설일 오독 check_rebate_qualification ──────────────────────────────
    print("[023 개설일] check_rebate_qualification — opening_date 11/10/2022 오독(레코드=11/05/2024)")
    ctx = {"account_opening_date": "11/10/2022", "monthly_threshold": 2000,
           "transactions": [{"date": "12/01/2024", "amount": 3000}]}
    flags = _run("check_rebate_qualification", ctx)
    check("023: 오독 개설일 드롭(→None·anchor 소실→abstain)", ctx["account_opening_date"] is None)
    check("023: 정상 threshold(2000∈KB) 유지", ctx["monthly_threshold"] == 2000)

    print("[정상] check_rebate_qualification — 개설일 레코드값 11/05/2024(회귀)")
    ctx = {"account_opening_date": "11/05/2024", "monthly_threshold": 2000,
           "transactions": [{"date": "12/01/2024", "amount": 3000}]}
    flags = _run("check_rebate_qualification", ctx)
    check("정상: 11/05/2024 유지·플래그 0", ctx["account_opening_date"] == "11/05/2024" and not flags)

    print("[023 threshold 날조] check_rebate_qualification — threshold 9999 날조")
    ctx = {"account_opening_date": "11/05/2024", "monthly_threshold": 9999,
           "transactions": [{"date": "12/01/2024", "amount": 3000}]}
    flags = _run("check_rebate_qualification", ctx)
    check("023: 날조 threshold 9999 드롭", ctx["monthly_threshold"] is None)

    print("\n== 결과: %d PASS / %d FAIL ==" % (len(PASS), len(FAIL)))
    if FAIL:
        print("FAILED:")
        for f in FAIL:
            print("  - " + f)
        sys.exit(1)
    print("ALL PASS — operand grounding이 097/095/023 날조·오독을 드롭·정상은 유지(false-drop 0).")


if __name__ == "__main__":
    main()
