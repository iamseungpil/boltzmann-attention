# -*- coding: utf-8 -*-
"""회귀 — ATM 수수료 환급 규칙이 **072·073·074 아홉 계좌 전수**에서 닫히는가 (2026-08-24).

이 검정이 고정하는 것 = *"우리 도구가 계좌마다 내는 순보정액이 그 계좌의 gold `amount` 와 같다"*.

⚠[[23]] 규율. 규칙의 **모든 수치는 KB 등급 문서 축자**이고 그 추출은 2026-08-13 에 이미 끝나 있었다
   (`reports/facet_rft_2026/ATM_FEE_SCHEDULE_VERBATIM_2026_08_13.md` — 무료 횟수·리베이트 캡은 그
   문서가 '1판 범위 밖' 이라고 유보해 둔 축이다). gold 는 **저작 입력이 아니라 대조 기준**으로만
   쓴다([[69]] reward 가 채점 단위). 아래 원장은 db.json 의 레코드 축자 전사이고, 서브가 형식화할
   형태 그대로다(엔진은 집합 차와 산수만 — [[10]]·[[22]]·[[59]]).

⚠이 파일이 잡는 세 가지 결손(2026-08-24 이전 우리 도구가 다 놓쳤다):
   ① 부과됐어야 하는데 **줄이 아예 없는** 인출  (074 _3 −1.75 · _4 −1.80 · 072 LG −1.50 · 073 LG −1.50)
   ② 등급의 **월 무료 횟수**                     (072 LG 앞 4건 · 073 LG 앞 4건 · 074 LB 타행 2·해외 2)
   ③ 환급 프로그램 등급의 **fee_rebate 부재**    (072 Bluest 11/14 +2.00 · 074 Purple 11/11 +2.50)
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
OP = [t for t in A2["scaffold_get_tools"]
      if t.get("name") == "get_atm_fee_discrepancies"][0]["op"]

FAIL = []


def w(i, amt, net, fee=0.0, reb=None, dup=None):
    """인출 1건 = 행 1개. fee=0 은 **수수료 줄이 없다**는 뜻(부재)."""
    r = {"transaction_id": ("f%02d" % i) if fee else ("w%02d" % i),
         "fee_amount": fee, "withdrawal_amount": amt, "network": net}
    if reb is not None:
        r["rebate_amount"] = reb
    if dup is not None:
        r["duplicate_of"] = dup
    return r


# ─────────────────────────────────────────────────────────────────────────────
# 원장 (db.json `bank_account_transaction_history` 축자 전사 · 날짜순)
# ─────────────────────────────────────────────────────────────────────────────
LEDGERS = [
    # ── task_072 ────────────────────────────────────────────────────────────
    ("072 Bluest      chk_lj82d4f1a9", "Bluest Account", 14.00, [
        w(2, 200, "non_rho", 2.00, reb=2.00),      # 11/02 CHASE
        w(5, 300, "non_rho", 2.00, reb=2.00),      # 11/05 BANK OF AMERICA
        w(8, 400, "foreign", 8.00, reb=0.0),       # 11/08 BARCLAYS LONDON   해외 $0 → +8.00
        w(10, 500, "non_rho", 2.00, reb=2.00),     # 11/10 WELLS FARGO
        w(12, 200, "foreign", 3.50, reb=0.0),      # 11/12 MIZUHO TOKYO      해외 $0 → +3.50
        w(14, 100, "non_rho", 2.00, reb=0.0),      # 11/14 CITIBANK          ★환급 부재 → +2.00
        w(16, 50, "foreign", 0.0, reb=0.0),        # 11/16 BANAMEX           정상(무수수료)
        w(18, 250, "non_rho", 2.00, reb=2.00),     # 11/18 PNC
        w(20, 400, "non_rho", 2.50, reb=2.00)]),   # 11/20 US BANK           $2.00 인데 2.50 → +0.50
    ("072 Light Green chk_538bfb9cba", "Light Green Account", 3.50, [
        w(1, 50, "non_rho"),                       # 11/01 무료 1
        w(3, 75, "non_rho"),                       # 11/03 무료 2
        w(5, 100, "non_rho", 1.50),                # 11/05 무료 3 인데 부과   → +1.50
        w(7, 60, "non_rho"),                       # 11/07 무료 4
        w(9, 80, "non_rho"),                       # 11/09 5번째 = $1.50 인데 줄 없음 → −1.50
        w(11, 40, "non_rho", 1.50),                # 11/11 6번째 정상
        w(13, 80, "foreign", 5.00),                # 11/13 $80 = TIER1 2.00  → +3.00
        w(15, 200, "foreign", 4.00),               # 11/15 $200 = TIER2 3.50 → +0.50
        w(17, 350, "foreign", 5.00),               # 11/17 TIER3 정상
        w(19, 100, "foreign", 2.00)]),             # 11/19 TIER1 정상
    # ── task_073 ────────────────────────────────────────────────────────────
    ("073 Blue        chk_kj93a7b2e1_1", "Blue Account", 9.50, [
        w(1, 200, "non_rho", 2.00),
        w(3, 150, "non_rho", 1.50),
        w(5, 300, "non_rho", 4.50),                # 1% max $3.00        → +1.50
        w(7, 100, "non_rho", 1.00),
        w(9, 500, "foreign", 20.00),               # max(3%,$5) = 15.00  → +5.00
        w(11, 250, "foreign", 7.50),
        w(13, 400, "rho", 3.00),                   # RHO 인출에 수수료   → +3.00
        w(15, 175, "non_rho", 1.75),
        w(17, 100, "foreign", 5.00),
        w(19, 225, "non_rho", 2.25)]),
    ("073 Green       chk_kj93a7b2e1_2", "Green Account", 9.00, [
        w(1, 150, "non_rho", 3.00),
        w(3, 200, "non_rho", 3.00),
        w(5, 100, "non_rho", 3.00),
        w(7, 300, "rho", 3.00),                    # RHO 인출에 수수료   → +3.00
        w(9, 400, "foreign", 15.00),               # max(3%,$5) = 12.00  → +3.00
        w(11, 175, "non_rho", 3.00),
        w(13, 200, "foreign", 6.00),
        w(15, 250, "non_rho", 3.00),               # 11/15 원본
        dict(w(151, 250, "non_rho", 3.00), duplicate_of="f15"),   # 11/15 중복 → +3.00
        w(17, 150, "foreign", 5.00),
        w(19, 125, "non_rho", 3.00)]),
    ("073 Light Green chk_kj93a7b2e1_3", "Light Green Account", 1.50, [
        w(1, 40, "non_rho"),                       # 무료 1
        w(3, 60, "non_rho", 1.50),                 # 무료 2 인데 부과     → +1.50
        w(5, 50, "non_rho"),                       # 무료 3
        w(7, 80, "non_rho"),                       # 무료 4
        w(9, 100, "non_rho", 1.50),                # 5번째 정상
        w(11, 75, "non_rho", 1.50),                # 6번째 정상
        w(13, 80, "foreign", 2.00),                # TIER1 정상
        w(15, 150, "foreign", 5.00),               # $150 = TIER2 3.50    → +1.50
        w(17, 65, "non_rho"),                      # 7번째 = $1.50 인데 줄 없음 → −1.50
        w(19, 350, "foreign", 5.00)]),             # TIER3 정상
    # ── task_074 ────────────────────────────────────────────────────────────
    ("074 Purple      chk_ar72c5d8e3_1", "Purple Account", 27.00, [
        w(2, 200, "rho", 0.0, reb=0.0),
        w(3, 300, "rho", 2.50, reb=0.0),           # RHO 인출에 수수료   → +2.50
        w(4, 150, "rho", 0.0, reb=0.0),
        w(5, 100, "non_rho", 2.50, reb=2.50),
        w(6, 250, "non_rho", 2.50, reb=2.50),
        w(7, 400, "foreign", 8.00, reb=0.0),       # 해외 $0             → +8.00
        w(8, 600, "foreign", 0.0, reb=0.0),
        w(9, 350, "foreign", 10.50, reb=0.0),      # 해외 $0             → +10.50
        w(101, 500, "foreign", 0.0, reb=0.0),
        w(11, 175, "non_rho", 3.50, reb=0.0),      # $2.50 인데 3.50 + ★환급 부재 → +3.50
        w(12, 225, "non_rho", 2.50, reb=2.50),
        w(13, 450, "foreign", 0.0, reb=0.0),
        w(14, 400, "rho", 0.0, reb=0.0),
        w(15, 300, "non_rho", 2.50, reb=2.50),     # 11/15 원본
        dict(w(151, 300, "non_rho", 2.50, reb=0.0), duplicate_of="f15"),   # 중복 → +2.50
        w(16, 275, "non_rho", 2.50, reb=2.50),
        w(17, 550, "foreign", 0.0, reb=0.0),
        w(18, 200, "foreign", 0.0, reb=0.0)]),
    ("074 Light Blue  chk_ar72c5d8e3_2", "Light Blue Account", 14.50, [
        w(2, 150, "rho"),
        w(3, 200, "rho", 2.50),                    # RHO 인출에 수수료   → +2.50
        w(4, 100, "rho"),
        w(5, 120, "non_rho", 2.50),                # 타행 무료 1 인데 부과 → +2.50
        w(6, 180, "non_rho", 2.50),                # 타행 무료 2 인데 부과 → +2.50
        w(7, 200, "non_rho", 2.50),                # 타행 3번째 정상
        w(8, 300, "foreign", 4.00),                # 해외 무료 1 인데 부과 → +4.00
        w(9, 400, "foreign", 4.00),                # 해외 무료 2 인데 부과 → +4.00
        w(101, 150, "non_rho"),                    # 타행 4번째 = $2.50 인데 줄 없음 → −2.50
        w(11, 250, "foreign", 4.00),               # 해외 3번째 정상
        w(12, 175, "non_rho", 2.50),
        w(13, 350, "foreign", 5.50),               # 해외 4번째 = $4.00  → +1.50
        w(14, 225, "non_rho", 2.50),
        w(15, 180, "foreign", 4.00),
        w(16, 300, "non_rho", 2.50),
        w(17, 500, "foreign", 4.00)]),
    ("074 Dark Green  chk_ar72c5d8e3_3", "Dark Green Account", 4.75, [
        w(2, 100, "rho", 1.50),                    # RHO 인출에 수수료   → +1.50
        w(3, 80, "rho"),
        w(4, 60, "rho"),
        w(5, 50, "non_rho", 0.50),                 # min $1.50 미적용    → −1.00
        w(6, 150, "non_rho", 1.50),
        w(7, 200, "non_rho", 4.00),                # 1% = 2.00           → +2.00
        w(8, 300, "non_rho", 3.00),
        w(9, 80, "foreign", 2.00),
        w(101, 160, "foreign", 4.00),
        w(11, 250, "foreign", 6.00),
        w(12, 400, "foreign", 10.00),              # max $6.00 미적용    → +4.00
        w(13, 175, "non_rho"),                     # ★줄 없음 = $1.75    → −1.75
        w(14, 125, "non_rho", 1.50),
        w(15, 300, "foreign", 6.00),
        w(16, 200, "rho"),
        w(17, 225, "non_rho", 2.25)]),
    ("074 Evergreen   chk_ar72c5d8e3_4", "Evergreen Account", 3.70, [
        w(2, 200, "rho"),
        w(3, 150, "rho", 1.50),                    # RHO 인출에 수수료   → +1.50
        w(4, 100, "rho"),
        w(5, 40, "non_rho", 0.40),
        w(6, 150, "non_rho", 1.50),
        w(7, 250, "non_rho", 2.50),
        w(8, 400, "non_rho", 4.00),                # max $2.50 미적용    → +1.50
        w(9, 100, "foreign", 2.00),                # min $3.00 미적용    → −1.00
        w(101, 200, "foreign", 4.00),
        w(11, 350, "foreign", 10.50),              # 2% = 7.00           → +3.50
        w(12, 180, "non_rho"),                     # ★줄 없음 = $1.80    → −1.80
        w(13, 120, "foreign", 3.00),
        w(14, 275, "non_rho", 2.50),
        w(15, 450, "foreign", 9.00),
        w(16, 300, "rho"),
        w(17, 225, "non_rho", 2.25)]),
]


def chk(cond, msg, extra=""):
    if not cond:
        FAIL.append(msg)
    print("  %s %s%s" % ("ok  " if cond else "FAIL", msg, (" — %s" % extra) if extra else ""))


def run(cls, rows):
    ctx = {"account_class": cls, "transactions": [dict(r) for r in rows]}
    ids = apply_op(OP, ctx)
    return ids, ctx.get("_sg_stats") or {}, ctx.get("_sg_details") or []


def main():
    print("[계좌별 순보정액 = gold]")
    for label, cls, gold, rows in LEDGERS:
        ids, st, det = run(cls, rows)
        net = round(sum(d.get("delta") or 0 for d in det), 2)
        chk(abs(net - gold) < 1e-6, "%s  net %.2f = gold %.2f" % (label, net, gold), None if
            abs(net - gold) < 1e-6 else "ids=%s" % ids)
        chk(st.get("skipped") == 0, "%s  판정불가 0행" % label, st)

    print("\n[부재 줄이 실제로 잡히는가 — 음수 delta]")
    for label, cls, _g, rows in LEDGERS:
        ids, _st, det = run(cls, rows)
        neg = [d["id"] for d in det if (d.get("delta") or 0) < 0]
        if label.startswith(("074 Dark Green", "074 Evergreen", "074 Light Blue",
                             "072 Light Green", "073 Light Green")):
            chk(bool(neg), "%s  부재/미달 라인 검출" % label, neg)

    print("\n[무료 횟수 — 앞 N건은 기대 0]")
    ids, _st, det = run("Light Green Account", [w(1, 50, "non_rho", 1.50)])
    chk(ids == ["f01"] and det and det[0]["expected"] == 0,
        "light_green 1번째 타행 = 무료 → 부과는 전액 과부과", det)
    ids, _st, det = run("Light Green Account",
                        [w(i, 50, "non_rho") for i in range(1, 5)] + [w(5, 50, "non_rho")])
    chk(ids == ["w05"] and det and det[0]["delta"] == -1.5,
        "light_green 5번째 = $1.50 인데 줄 없음 → −1.50", det)
    ids, _st, det = run("Light Blue Account",
                        [w(1, 100, "non_rho"), w(2, 100, "non_rho"),
                         w(3, 100, "foreign"), w(4, 100, "foreign")])
    chk(ids == [], "light_blue 타행·해외 무료 풀은 **분리**(각 2건까지 무료)", ids)

    print("\n[환급 축 — 방향·상한·기권은 `test_rebate_netting.py` 가 고정한다(중복 금지·[[67]])]")
    _i, _s, det = run("Bluest Account", [w(14, 100, "non_rho", 2.00, reb=0.0)])
    chk(det and det[0]["delta"] == 2.00, "072 Bluest 11/14 = 우리가 놓쳤던 그 칸(+2.00)", det)

    print("\n%s (%d fail)" % ("FAIL" if FAIL else "PASS", len(FAIL)))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
