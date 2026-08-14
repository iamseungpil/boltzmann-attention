# -*- coding: utf-8 -*-
"""환급 차감 검정 — 072 원장 실물로 **$14.00** 재현(현행 규칙은 $12.00 이었다).

근거(C487·원장 직독·gold 무접촉): Bluest 계좌 원장에 `ATM FEE REBATE` 5건($2.00×5)이 있다.
    부과  11/20 2.50 · 11/18 2.00 · 11/14 2.00 · 11/10 2.00 · 11/05 2.00 · 11/02 2.00(third-party)
          11/12 3.50 · 11/08 8.00(foreign)                                   합 24.00
    환급  11/20 2.00 · 11/18 2.00 · 11/10 2.00 · 11/05 2.00 · 11/02 2.00     합 10.00
    미환급 = 14.00 ← gold 정확 일치
정책 축자(`bluest_account_003`): *"third-party ATM fees … rebated up to $50 per monthly
statement cycle"* · *"Foreign ATM withdrawal fee: $0.00"* ⇒ Bluest 는 ATM 수수료 **순비용 0**.

우리 op 가 놓친 자리는 **11/14 $2.00**이다 — 환급이 **없는데** 기존 규칙에선 *"기대 2.00 = 부과
2.00"* 이라 정상 판정됐다. 그래서 총액이 12.00 이었다.

측정 근거([[62]] 순서·x323·n=24·블록 8·8·8): 환급 축자 제공 **0/24** · 정책 문면까지 **0/24** ·
엔진이 뺀 값 제공 **24/24** ⇒ 이 한 칸(뺄셈)만 결정론으로 정당화된다.

이 검정이 고정하는 것: 총액 14.00 · 11/14 가 discrepant 로 잡힘 · `rebate_field` 미선언 시 거동 보존.
"""
import io
import os
import sys

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t2_compute as C                                             # noqa: E402

FAIL = []
# 072 Bluest 원장 축자(부과·환급·network) — 서브가 전사할 형태 그대로
ROWS = [
    {"transaction_id": "f_1120", "fee_amount": 2.50, "network": "non_rho", "rebated_amount": 2.00},
    {"transaction_id": "f_1118", "fee_amount": 2.00, "network": "non_rho", "rebated_amount": 2.00},
    {"transaction_id": "f_1114", "fee_amount": 2.00, "network": "non_rho", "rebated_amount": 0.0},
    {"transaction_id": "f_1112", "fee_amount": 3.50, "network": "foreign", "rebated_amount": 0.0},
    {"transaction_id": "f_1110", "fee_amount": 2.00, "network": "non_rho", "rebated_amount": 2.00},
    {"transaction_id": "f_1108", "fee_amount": 8.00, "network": "foreign", "rebated_amount": 0.0},
    {"transaction_id": "f_1105", "fee_amount": 2.00, "network": "non_rho", "rebated_amount": 2.00},
    {"transaction_id": "f_1102", "fee_amount": 2.00, "network": "non_rho", "rebated_amount": 2.00},
]
SPEC = {"op": "select_discrepant", "over": "transactions", "id_field": "transaction_id",
        "actual_field": "fee_amount", "tolerance": 0.01, "rebate_field": "rebated_amount",
        "steps": {"oon": {"op": "case", "key": "account_class",
                          "cases": {"Bluest Account": 0}, "default": None},
                  "expected": {"op": "case", "key": "r.network",
                               "cases": {"rho": 0, "non_rho": {"op": "ref", "path": "steps.oon"},
                                         "foreign": 0}, "default": None}},
        "expected_ref": "steps.expected"}


def chk(c, m):
    if not c:
        FAIL.append(m)
    print("  %s %s" % ("ok  " if c else "FAIL", m))


def run(spec):
    ctx = {"transactions": [dict(r) for r in ROWS], "account_class": "Bluest Account"}
    return C.apply_op(spec, ctx), ctx


def main():
    print("[환급 차감 후 총액]")
    ids, ctx = run(SPEC)
    dets = ctx.get("_sg_details") or []
    total = round(sum(d.get("delta") or 0 for d in dets), 2)
    chk(abs(total - 14.00) < 1e-6, "미환급 합 = 14.00 (실제 %.2f · 현행 규칙은 12.00)" % total)
    chk("f_1114" in [d.get("id") for d in dets],
        "11/14 $2.00(환급 없음)이 discrepant 로 잡힌다 ← 놓쳤던 그 칸")
    chk("f_1118" not in [d.get("id") for d in dets], "전액 환급된 라인은 안 잡힌다")

    print("[거동 보존 — 미선언이면 종전대로]")
    spec2 = dict(SPEC)
    spec2.pop("rebate_field")
    _ids2, ctx2 = run(spec2)
    t2 = round(sum(d.get("delta") or 0 for d in (ctx2.get("_sg_details") or [])), 2)
    chk(abs(t2 - 14.00) > 1e-6, "rebate_field 없으면 값이 다르다(=차감이 선언 의존·%.2f)" % t2)

    print("\n%s (%d fail)" % ("FAIL" if FAIL else "PASS", len(FAIL)))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
