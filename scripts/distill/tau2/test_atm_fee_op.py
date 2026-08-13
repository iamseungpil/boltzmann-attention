# -*- coding: utf-8 -*-
"""회귀 — get_atm_fee_discrepancies op (오프라인·모델 0·x288→(B)·2026-08-13).

op 는 A2 정본에서 로드(두 벌 금지)·apply_op 직접 실행. 검정 축:
  ⑴ 클래스별 공식(min/max 캡·플랫·tier 경계=하위) ⑵ rho=기대 0(네트워크 모순 검출)
  ⑶ 판정-보류 클래스(light_green oon 등)는 discrepant 가 아니라 skipped 로 계상
  ⑷ 3사본 동일성.
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

OK = []


def chk(name, cond, extra=""):
    OK.append(bool(cond))
    print("  %s %s%s" % ("PASS" if cond else "FAIL", name, (" — " + str(extra)) if extra else ""))


def load_entry(path):
    a = json.load(io.open(os.path.join(HERE, path), encoding="utf-8"))
    for t in a.get("scaffold_get_tools") or []:
        if t.get("name") == "get_atm_fee_discrepancies":
            return t
    return None


E = load_entry("a2/banking_knowledge.specific.json")
chk("A2 항목 존재", E is not None)
OP = (E or {}).get("op") or {}


def run(cls, recs):
    ctx = {"account_class": cls, "transactions": recs}
    ids = apply_op(OP, ctx)
    return ids, ctx.get("_sg_stats") or {}, ctx.get("_sg_details") or []


def fee(i, amt, w, net):
    return {"transaction_id": "t%d" % i, "fee_amount": amt,
            "withdrawal_amount": w, "network": net}


# ⑴ 클래스 공식
ids, st, det = run("Blue Account", [fee(1, 1.00, 100, "non_rho"),   # 1% = 1.00 정상
                                    fee(2, 3.50, 100, "non_rho"),   # 기대 1.00 → discrepant
                                    fee(3, 3.00, 500, "non_rho"),   # min(5,3)=3.00 정상(캡)
                                    fee(4, 5.00, 120, "foreign"),   # max(3.6,5)=5.00 정상
                                    fee(5, 6.00, 300, "foreign")])  # max(9,5)=9.00 → discrepant
chk("blue: 캡·최소 공식", ids == ["t2", "t5"], ids)

ids, st, _ = run("Green Account", [fee(1, 3.00, 80, "non_rho"),     # 플랫 3.00 정상
                                   fee(2, 2.50, 80, "non_rho")])    # → discrepant
chk("green: 플랫", ids == ["t2"], ids)

ids, st, _ = run("Dark Green Account", [fee(1, 1.50, 100, "non_rho"),  # max(1.0,1.5)=1.5 정상
                                        fee(2, 1.00, 100, "non_rho"),  # → discrepant
                                        fee(3, 6.00, 400, "foreign"),  # min(10,6)=6 정상
                                        fee(4, 10.00, 400, "foreign")])
chk("dark_green: min/max", ids == ["t2", "t4"], ids)

ids, st, _ = run("Light Green Account", [fee(1, 2.00, 100, "foreign"),   # 경계=하위 tier 2.00 정상
                                         fee(2, 3.50, 100, "foreign"),   # → discrepant
                                         fee(3, 3.50, 300, "foreign"),   # 경계 300 → 3.50 정상
                                         fee(4, 5.00, 301, "foreign")])  # >300 → 5.00 정상
chk("light_green: tier 경계=하위", ids == ["t2"], ids)

# ⑵ rho 모순(NON-RHO fee 가 RHO 인출에)
ids, st, _ = run("Bluest Account", [fee(1, 2.00, 200, "rho"),        # 기대 0 → discrepant
                                    fee(2, 2.00, 200, "non_rho"),    # 플랫 2.00 정상
                                    fee(3, 4.00, 200, "foreign")])   # 기대 0 → discrepant
chk("bluest: rho/foreign 기대 0", ids == ["t1", "t3"], ids)

# ⑶ 판정-보류(문서 미규정 축) = discrepant 0 + skipped 계상
ids, st, _ = run("Light Green Account", [fee(1, 1.50, 50, "non_rho")])
chk("light_green oon: 보류(skipped)", ids == [] and st.get("skipped") == 1, (ids, st))
ids, st, _ = run("Gold Years Account", [fee(1, 2.00, 50, "non_rho")])
chk("미선언 클래스: 보류", ids == [] and st.get("skipped") == 1, (ids, st))

# ⑷ 3사본 동일
E2 = load_entry("a2/banking_knowledge.gate.json")
E3 = load_entry("a2/split/banking_knowledge.core.json")
chk("3사본 바이트-동일(json 등가)", E == E2 == E3)

print("\n%d/%d" % (sum(OK), len(OK)))
sys.exit(0 if all(OK) else 1)
