# -*- coding: utf-8 -*-
"""일회성 — (B2) get_checking_atm_fee_totals A2 항목 3사본 프로그램 삽입.

격리 근거: x291_checking_pick_iso.py (X291_CHECKING_FIT_DESIGN §2 사전등록 매트릭스) —
A_LIVE 0/8 · B_DOCS 0/8 · C_CALC 8/8 · D_NEG 0/8 · E_FRESH 1/8 ⇒ (B) 경로 확정.
요율 출처: 전부 정책 문서 축자(gold 미접촉) — 각 행 source 필드 + _note_ 병기.
스태킹(OON+foreign 동시 부과)은 personal 문서 미규정 ⇒ 축별 2열 분리 반환(단정 0).
"""
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PATHS = ["a2/banking_knowledge.specific.json", "a2/banking_knowledge.gate.json",
         "a2/split/banking_knowledge.core.json"]

PCT = lambda b: {"op": "multiply", "a": "withdrawal_amount", "b": b}  # noqa: E731

ROW = lambda cls, oon, oon_free, forx, forx_free, src: {  # noqa: E731
    "account_class": cls, "oon_fee": oon, "oon_free_per_month": oon_free,
    "forx_fee": forx, "forx_free_per_month": forx_free, "source": src}

TABLE = [
    ROW("Blue Account", {"op": "min", "of": [PCT(0.01), 3.0]}, 0,
        {"op": "max", "of": [PCT(0.03), 5.0]}, 0, "doc blue_account_001/_012"),
    ROW("Bluest Account", 2.0, 0, 0.0, 0, "doc bluest_account_010/_003"),
    ROW("Green Account (checking)", 3.0, 0,
        {"op": "max", "of": [PCT(0.03), 5.0]}, 0, "doc green_account_(checking)_010/_012"),
    ROW("Green Fee-Free Account", 0.0, 0, 0.0, 0, "doc green_fee-free_account_005/_002/_003"),
    ROW("Gold Years Account", 0.0, 0, 3.5, 0, "doc gold_years_account_001/_002"),
    ROW("Light Green Account", 1.5, 4,
        {"op": "lookup_table", "key": "withdrawal_amount", "table": [
            {"cmp": "<=", "thr": 100, "result": 2.0},
            {"cmp": "<=", "thr": 300, "result": 3.5},
            {"cmp": ">", "thr": 300, "result": 5.0}]}, 0,
        "doc light_green_account_001/_013"),
    ROW("Light Blue Account", 2.5, 2, 4.0, 2, "doc light_blue_account_004/_006"),
    ROW("Purple Account", 2.5, 0, 0.0, 0, "doc purple_account_012/_001"),
    ROW("Dark Green Account", {"op": "max", "of": [PCT(0.01), 1.5]}, 0,
        {"op": "min", "of": [PCT(0.025), 6.0]}, 0, "doc dark_green_account_001/_002"),
    ROW("Evergreen Account", {"op": "min", "of": [PCT(0.01), 2.5]}, 0,
        {"op": "max", "of": [PCT(0.02), 3.0]}, 0, "doc evergreen_account_008/_001"),
]

AXIS = lambda ax: {  # noqa: E731  — 축별 총액: max(0, 월횟수−월무료)×개월×건당요율
    "%s_unit" % ax: {"op": "ref_op", "path": "r.%s_fee" % ax},
    "%s_paid_raw" % ax: {"op": "diff", "a": "withdrawals_per_month",
                         "b": "r.%s_free_per_month" % ax},
    "%s_paid" % ax: {"op": "clamp", "value": "steps.%s_paid_raw" % ax, "min": 0},
    "%s_cnt" % ax: {"op": "multiply", "a": "steps.%s_paid" % ax, "b": "months"},
    "%s_total" % ax: {"op": "multiply", "a": "steps.%s_unit" % ax, "b": "steps.%s_cnt" % ax},
}

STEPS = {}
STEPS.update(AXIS("oon"))
STEPS.update(AXIS("forx"))

ENTRY = {
    "name": "get_checking_atm_fee_totals",
    "description": (
        "MANDATORY before recommending which personal checking account class to open (or keep) "
        "when the customer's stated criterion involves ATM fees: formalize the customer's stated "
        "usage pattern and call this ONCE. It computes, from the documented per-class fee "
        "schedules, each class's out-of-network ATM fee total and foreign ATM fee total for that "
        "usage (documented monthly free allowances deducted). It does NOT rank or pick a class - "
        "compare the returned totals yourself against what the customer asked to minimize, check "
        "the candidate class's other documented terms (eligibility, minimum deposits, monthly "
        "fees) in its source docs, and confirm the choice with the customer. Do the fee math "
        "with this tool - do not eyeball it yourself."),
    "params": {
        "months": "number - how many months the customer's usage pattern covers (their stated "
                  "trip/usage duration).",
        "withdrawals_per_month": "number - the customer's stated ATM withdrawals per month.",
        "withdrawal_amount": "number - the customer's stated typical amount per withdrawal in "
                             "USD (no $).",
    },
    "return_template": (
        "Documented ATM fee totals per personal checking account class for the stated usage "
        "(monthly free allowances deducted; out-of-network and foreign totals are shown as "
        "SEPARATE columns - how the two fee types combine for one withdrawal abroad is not "
        "documented, so weigh both columns for foreign out-of-network usage): {result}. This "
        "tool does not pick a class: compare the totals yourself, verify the remaining "
        "candidate's eligibility and non-ATM terms in its cited source docs, and confirm with "
        "the customer. If 'rows' is empty, a numeric parameter was missing or unreadable - call "
        "again with months, withdrawals_per_month and withdrawal_amount as plain numbers."),
    "op": {
        "op": "catalog_compute",
        "label_field": "account_class",
        "table": TABLE,
        "steps": STEPS,
        "value_cols": {"out_of_network_total_usd": "steps.oon_total",
                       "foreign_atm_total_usd": "steps.forx_total"},
    },
    "_note_": (
        "2026-08-13 x291 판정(A_LIVE 0/8·B_DOCS 0/8·C_CALC 8/8·D_NEG 0/8·E_FRESH 1/8 — "
        "X291_CHECKING_FIT_DESIGN §2 사전등록)에 따른 F2b 결정론기. 요율 출처 전부 정책 문서 "
        "축자(ATM_FEE_SCHEDULE_VERBATIM_2026_08_13.md + green_fee-free_005('Out-of-network ATM "
        "withdrawal fee $0.00'·'Foreign ATM withdrawal fee $0.00') + gold_years_002('Out-of-network "
        "ATM withdrawal fee: $0.00'·'Foreign ATM withdrawal fee: $3.50')·gold 미접촉). 무료횟수는 "
        "축별 조항 축자(lb_004 oon 2회/월·lb_006 forx 2회/월·lg_001 oon 4회/월) — 풀 공유 여부는 "
        "미규정(모호점 8·10)이라 축별 독립 차감(문면 그대로). OON+foreign 스태킹은 personal 문서 "
        "미규정(business navy_blue_008 은 either/or 시사) ⇒ 합산하지 않고 축별 2열 반환·결합 판단은 "
        "모델 몫. 클래스명은 general_001 공식 표기(full official name ending with 'Account'). "
        "픽·자격 확인·write 는 모델 몫 불변([[62]] 최소 결정론)."),
}

entries = []
for rel in PATHS:
    p = os.path.join(HERE, rel)
    j = json.load(io.open(p, encoding="utf-8"))
    tools = j.get("scaffold_get_tools")
    if tools is None:
        print("NO scaffold_get_tools in %s" % rel)
        sys.exit(1)
    tools[:] = [t for t in tools if t.get("name") != ENTRY["name"]]
    # 삽입 위치 = get_atm_fee_discrepancies 바로 뒤(가족 인접)
    idx = next((i for i, t in enumerate(tools)
                if t.get("name") == "get_atm_fee_discrepancies"), len(tools) - 1)
    tools.insert(idx + 1, json.loads(json.dumps(ENTRY)))
    with io.open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(j, f, ensure_ascii=False, indent=1)
        f.write("\n")
    entries.append(tools[idx + 1])
    print("inserted into %s" % rel)

print("3사본 json-등가:", entries[0] == entries[1] == entries[2])
