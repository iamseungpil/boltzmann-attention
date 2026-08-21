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


def fee(i, amt, w, net, rebated=None):
    r = {"transaction_id": "t%d" % i, "fee_amount": amt,
         "withdrawal_amount": w, "network": net}
    if rebated is not None:
        r["rebated_amount"] = rebated
    return r


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
#   ⚠환급 차감(C487·x323)은 **보류**다 — Bluest 기대값 0 + 미추출이 과다 환불을 만든다.
#     기구와 검정(`test_rebate_netting`)은 남기고 A2 선언만 뺐다. 여기선 종전 규칙을 고정한다.
ids, st, _ = run("Bluest Account", [fee(1, 2.00, 200, "rho"),        # 기대 0 → discrepant
                                    fee(2, 2.00, 200, "non_rho"),    # 플랫 2.00 정상
                                    fee(3, 4.00, 200, "foreign")])   # 기대 0 → discrepant
chk("bluest: rho/foreign 기대 0", ids == ["t1", "t3"], ids)
ids, st, _ = run("Light Green Account", [fee(1, 1.50, 50, "non_rho")])
chk("light_green oon: 보류(skipped)", ids == [] and st.get("skipped") == 1, (ids, st))
ids, st, _ = run("Gold Years Account", [fee(1, 2.00, 50, "non_rho")])
chk("미선언 클래스: 보류", ids == [] and st.get("skipped") == 1, (ids, st))

# ⑹ dup_field(2026-08-14 x301 C_DUP 8/8): 모델이 선언한 duplicate_of → 기대 0(전액 환불)
#    + 073 계좌2 실물 재현: WOORI 초과($3)+RHO 라인($3)+중복($3) = net $9.00 (gold)
ids, st, det = run("Green Account", [
    fee(1, 15.00, 400, "foreign"),                      # WOORI: 기대 max(12,5)=12 → delta 3
    fee(2, 3.00, 300, "rho"),                           # RHO-BANK 인출 위 fee: 기대 0 → delta 3
    dict(fee(3, 3.00, 250, "non_rho"), duplicate_of="tX"),  # 중복: 금액은 공식값이나 기대 0
    fee(4, 3.00, 250, "non_rho")])                      # 원본 라인: 정상
chk("dup 선언=기대 0", ids == ["t1", "t2", "t3"], ids)
_net = round(sum(d2["delta"] for d2 in det), 2)
chk("073 계좌2 net=9.00 재현", _net == 9.00, _net)

# ⑹b 중복 그룹(2026-08-14 t7283 073 계좌2 실물: 서브가 **양쪽 모두** duplicate_of 부착 →
#     구판은 둘 다 기대 0 = $12(정답 $9). 그룹에서 첫 행은 원본으로 남긴다.)
ids2, st2, det2 = run("Green Account", [
    fee(1, 15.00, 400, "foreign"),                          # WOORI 초과 → delta 3
    fee(2, 3.00, 300, "rho"),                               # RHO 라인 → delta 3
    dict(fee(3, 3.00, 250, "non_rho"), duplicate_of="t4"),  # 상호 참조 쌍(앞) = 원본
    dict(fee(4, 3.00, 250, "non_rho"), duplicate_of="t3")]) # 상호 참조 쌍(뒤) = 중복 → delta 3
chk("상호 dup: 하나만 중복", ids2 == ["t1", "t2", "t4"], ids2)
chk("상호 dup net=9.00", round(sum(d2["delta"] for d2 in det2), 2) == 9.00,
    round(sum(d2["delta"] for d2 in det2), 2))

# ⑺ param 문면 = x301 C_DUP 축자([[03b]] 측정한 문구 = 출시 문구)
chk("param C_DUP 축자", "duplicate_of" in (E or {}).get("params", {}).get("transactions", "")
    and "paired withdrawal's description" in (E or {}).get("params", {}).get("transactions", ""))
chk("op dup_field 선언", (E or {}).get("op", {}).get("dup_field") == "duplicate_of")

# ⑸ FIX-5(2026-08-13 t7274w 073 실측): delta 키 + 템플릿 렌더(net correction 표면화)
ids, st, det = run("Blue Account", [fee(1, 3.50, 100, "non_rho"),   # 기대 1.00 → delta 2.50
                                    fee(2, 5.00, 120, "foreign")])  # 기대 5.00 정상
chk("delta 키", len(det) == 1 and det[0].get("delta") == 2.50, det)
_item = (E or {}).get("detail_item_template", "").format(**(det[0] if det else {}))
chk("detail 렌더", "charged $3.50" in _item and "documented fee $1.00" in _item
    and "difference $2.50" in _item, _item)
_dtot = round(sum(d2["delta"] for d2 in det), 2)
_txt = (E or {}).get("return_template", "").format(
    ids=", ".join(ids), delta_total=_dtot,
    details="; ".join((E or {}).get("detail_item_template", "").format(**d2) for d2 in det))
# ★2026-08-19 회귀 검정으로 반전 — 엔진이 **채점되는 값**(계좌별 net = gold `amount`)을
#   문면으로 건네면 formalize→calc 아키텍처가 아니라 그 위조판을 재게 된다([[62]]·[[03b]]).
#   남겨야 하는 것: 정책 축자 문구(net correction)와 라인별 중간 사실({details}).
#   없어야 하는 것: 합계 수치 자체.
chk("REG: 정책 문구 유지", "net correction" in _txt, _txt[:160])
chk("REG: 라인별 사실 유지", "difference $2.50" in _txt, _txt[:200])
chk("REG: 엔진이 net 수치를 안 건넨다", "= $2.50" not in _txt and "delta_total" not in
    (E or {}).get("return_template", ""), _txt[:200])

# ⑻ P5(2026-08-21·t7335 halfA 072): 반환문 완결-인상 제거 + 검사/미검사 축 문면 명시.
#    구 문구 "across all identified fee discrepancies" 가 완결 인상을 줘 모델의 보완
#    rebate 검사를 억제([38] $12.00 write·차액 $2.00=11/14 누락 rebate). 수리는 문면만
#    ([[62]] — 누락-rebate 검사 로직 신설 0·op 불변).
_rt = (E or {}).get("return_template", "")
chk("P5: 완결 인상 제거", "across all identified" not in _rt, _rt[:120])
chk("P5: 검사한 축 명시(전달 라인의 금액만)",
    "fee-line amounts only" in _rt and "you passed in" in _rt, _rt[:250])
chk("P5: 미검사 축 명시(fee_rebate 부재)",
    "did NOT check" in _rt and "fee_rebate" in _rt, _rt[:400])
chk("P5: [[64]] fix-naming(무엇을 하면 풀리나)",
    "check the account's rebate policy against the fee_rebate lines yourself" in _rt)
chk("P5: 렌더에 {details} 유지·다른 자리표시자 없음",
    "{details}" in _rt and "difference $2.50" in _txt
    and not [m for m in __import__("re").findall(r"\{(\w+)", _rt) if m != "details"], _rt[-120:])

# ⑷ 3사본 동일
E2 = load_entry("a2/banking_knowledge.gate.json")
E3 = load_entry("a2/split/banking_knowledge.core.json")
chk("3사본 바이트-동일(json 등가)", E == E2 == E3)

print("\n%d/%d" % (sum(OK), len(OK)))
sys.exit(0 if all(OK) else 1)
