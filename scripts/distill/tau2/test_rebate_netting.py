# -*- coding: utf-8 -*-
"""환급 축 검정 — `op.rebate` 순액 기전 (2026-08-24 재작성 · 정본 A2 op 직접 사용).

**무엇이 바뀌었나.** 2026-08-15 판(`rebate_field`·보류)은 **부과 쪽에서만** 환급을 빼고 Bluest 의
non_rho 기대값을 0 으로 내렸다. 그러면 서브가 환급을 못 뽑는 순간 모든 수수료가 전액 불일치로 잡혀
**과다 환불**($24.00)이 났다. 새 판은 **양쪽을 같이 순액화**한다:

    expected_net = 문서 요율 − min(문서 요율, 남은 월 상한)
    actual_net   = 부과액   − 실제 환급액

환급 채무를 **부과액이 아니라 문서 요율**로 계산하는 것이 이 기전의 핵심이다 — 과부과분은 환급이
아니라 과부과로 돌려주는 것이라 이중 계상하면 안 된다(072 Bluest 11/20 = 2.50 부과·2.00 환급·
문서 2.00 ⇒ 순 +0.50 이지 +1.00 이 아니다).

출처(gold 무접촉·[[23]]): 상한은 KB 등급 문서 축자 — Bluest 월 $50(bluest_001/_003/_007) ·
Purple 월 $30(purple_001/_004/_010). `ATM_FEE_SCHEDULE_VERBATIM_2026_08_13.md` 가 이미 추출해 두고
'1판 범위 밖' 으로 유보했던 축이다.

계좌 전수 대조(072·073·074 아홉 계좌 = gold)는 `test_atm_ledger_close.py` 가 한다. 여기서는
**기전**만 고정한다: 방향 · 상한 · 기권 · 미선언 등급 거동 보존.
"""
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

from t2_compute import apply_op                                    # noqa: E402

A2 = json.load(io.open(os.path.join(HERE, "a2", "banking_knowledge.specific.json"),
                       encoding="utf-8"))
ENTRY = [t for t in A2["scaffold_get_tools"]
         if t.get("name") == "get_atm_fee_discrepancies"][0]
OP = ENTRY["op"]

FAIL = []


def chk(c, m, extra=""):
    if not c:
        FAIL.append(m)
    print("  %s %s%s" % ("ok  " if c else "FAIL", m, (" — %s" % extra) if extra else ""))


def row(i, amt, net, fee, reb=None):
    r = {"transaction_id": "f%02d" % i, "fee_amount": fee, "withdrawal_amount": amt,
         "network": net}
    if reb is not None:
        r["rebate_amount"] = reb
    return r


def run(cls, rows):
    ctx = {"account_class": cls, "transactions": [dict(r) for r in rows]}
    ids = apply_op(OP, ctx)
    return ids, ctx.get("_sg_stats") or {}, ctx.get("_sg_details") or []


def main():
    print("[선언 — 상한은 A2 상수이고 엔진은 뺄셈만 한다]")
    chk(OP.get("rebate", {}).get("field") == "rebate_amount", "op.rebate.field 선언")
    caps = ((OP.get("rebate") or {}).get("cap") or {}).get("cases") or {}
    chk(caps.get("Bluest Account") == 50.0 and caps.get("Purple Account") == 30.0,
        "상한 = KB 축자(Bluest $50 · Purple $30)", caps)
    chk("rebate_field" not in OP, "구판 `rebate_field` 선언은 남아 있지 않다")

    print("\n[방향 — 채무는 **문서 요율**로 계산한다(부과액이 아니라)]")
    #   072 Bluest 11/20 실물: $2.00 인데 2.50 부과 · 환급 2.00.
    _i, _s, det = run("Bluest Account", [row(20, 400, "non_rho", 2.50, reb=2.00)])
    chk(det and det[0]["delta"] == 0.50,
        "2.50 부과·2.00 환급·문서 2.00 ⇒ +0.50 (부과액 기준이면 잘못 +1.00)", det)
    _i, _s, det = run("Bluest Account", [row(14, 100, "non_rho", 2.00, reb=0.0)])
    chk(det and det[0]["delta"] == 2.00, "문서대로 부과됐는데 환급 줄이 없으면 +2.00", det)
    ids, _s, _d = run("Bluest Account", [row(18, 250, "non_rho", 2.00, reb=2.00)])
    chk(ids == [], "부과·환급이 문서대로면 안 잡힌다", ids)
    _i, _s, det = run("Bluest Account", [row(8, 400, "foreign", 8.00, reb=0.0)])
    chk(det and det[0]["delta"] == 8.00,
        "해외 $0 등급의 해외 수수료는 **과부과 전액**이지 미환급이 아니다(이중 계상 금지)", det)

    print("\n[상한 — 월 캡을 넘긴 뒤의 미환급은 채무가 아니다]")
    rows = [row(i, 200, "non_rho", 2.00, reb=2.00) for i in range(1, 26)]     # 2.00 × 25 = 50
    ids, _s, _d = run("Bluest Account", rows + [row(26, 200, "non_rho", 2.00, reb=0.0)])
    chk(ids == [], "캡 소진(25건) 뒤 26번째의 미환급은 잡히지 않는다", ids)
    ids, _s, _d = run("Bluest Account", rows[:24] + [row(25, 200, "non_rho", 2.00, reb=0.0)])
    chk(ids == ["f25"], "캡 이내(25번째)의 미환급은 채무다", ids)
    ids, _s, det = run("Purple Account", [row(i, 200, "non_rho", 2.50, reb=2.50)
                                          for i in range(1, 13)]
                       + [row(13, 200, "non_rho", 2.50, reb=0.0)])
    chk(ids == [], "Purple 은 $30 캡 = $2.50 × 12건까지 (13번째는 채무 아님)", ids)

    print("\n[기권 — 필드가 아예 없으면 판정하지 않는다([[25]])]")
    #   2026-08-15 판이 죽은 자리. 틀린 수를 내는 것보다 coverage 결손을 내보내는 편이 낫다.
    ids, st, det = run("Bluest Account", [{"transaction_id": "f01", "fee_amount": 2.00,
                                           "withdrawal_amount": 200, "network": "non_rho"}])
    chk(ids == [] and not det, "rebate_amount 결핍 행은 discrepant 로 안 잡힌다(과다 환불 방지)")
    chk(st.get("skipped") == 1 and "rebate_amount" in (st.get("missing_fields") or {}),
        "결핍 필드로 지목돼 재요청이 가능하다([[64]])", st)
    _i, _s, det = run("Bluest Account", [row(1, 200, "non_rho", 2.00, reb=0.0)])
    chk(det and det[0]["delta"] == 2.00, "0.0 은 '환급 없음'이라는 **정보**이므로 판정한다", det)

    print("\n[미선언 등급 — 이 축이 통째로 꺼져 거동이 전과 같다]")
    ids, st, _d = run("Dark Green Account", [{"transaction_id": "f01", "fee_amount": 4.00,
                                              "withdrawal_amount": 200, "network": "non_rho"}])
    chk(ids == ["f01"] and st.get("skipped") == 0,
        "Dark Green(캡 미선언)은 rebate_amount 없이도 종전대로 판정된다", st)

    print("\n%s (%d fail)" % ("FAIL" if FAIL else "PASS", len(FAIL)))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
