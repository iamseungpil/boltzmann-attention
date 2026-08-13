# -*- coding: utf-8 -*-
"""회귀 — get_checking_atm_fee_totals op (오프라인·모델 0·x291→(B2)·2026-08-13).

op 는 A2 정본에서 로드(두 벌 금지)·apply_op 직접 실행. 검정 축:
  ⑴ 075 사용 패턴(3개월·월6회·$350)의 클래스별 축별 총액 — 정책 스케줄 수계산과 대조
  ⑵ 무료횟수 차감(월 경계)·tier 경계=하위 ⑶ 파라미터 결측 = rows 공집합 + not_computable 전원
  ⑷ 반환 템플릿 렌더 ⑸ 3사본 동일성.
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
        if t.get("name") == "get_checking_atm_fee_totals":
            return t
    return None


E = load_entry("a2/banking_knowledge.specific.json")
chk("A2 항목 존재", E is not None)
OP = (E or {}).get("op") or {}


import copy
import re

OP_DICT = copy.deepcopy(OP)
OP_DICT.pop("row_template", None)          # dict-모드(row_template 미선언) 경로 검정용


def run(months, wpm, amt, op=None):
    ctx = {"months": months, "withdrawals_per_month": wpm, "withdrawal_amount": amt}
    res = apply_op(op if op is not None else OP_DICT, ctx)
    rows = {r["account_class"]: (r["out_of_network_total_usd"], r["foreign_atm_total_usd"])
            for r in (res or {}).get("rows", [])}
    return res, rows


# ⑴ 075 사용 패턴 — 정책 스케줄 수계산 대조(축자 출처는 A2 _note_)
res, rows = run(3, 6, 350)
EXP = {"Blue Account": (54.0, 189.0), "Bluest Account": (36.0, 0.0),
       "Green Account (checking)": (54.0, 189.0), "Green Fee-Free Account": (0.0, 0.0),
       "Gold Years Account": (0.0, 63.0), "Light Green Account": (9.0, 90.0),
       "Light Blue Account": (30.0, 48.0), "Purple Account": (45.0, 0.0),
       "Dark Green Account": (63.0, 108.0), "Evergreen Account": (45.0, 126.0)}
chk("클래스 10행 전부 계산", len(rows) == 10 and not res.get("not_computable"),
    (len(rows), res.get("not_computable")))
for cls, exp in sorted(EXP.items()):
    chk("%s" % cls, rows.get(cls) == exp, (rows.get(cls), exp))

# ⑵ 무료횟수·tier 경계
_, r2 = run(3, 2, 350)          # light_blue: 월2회 전액 무료 → 0/0 · light_green: 월2<4 → oon 0
chk("무료 전액 흡수", r2["Light Blue Account"] == (0.0, 0.0)
    and r2["Light Green Account"][0] == 0.0, (r2["Light Blue Account"], r2["Light Green Account"]))
_, r3 = run(1, 1, 300)          # tier 경계 300 = 하위(3.50) · 월1회<무료4 → lg oon 0
chk("tier 경계=하위", r3["Light Green Account"] == (0.0, 3.5)
    and r3["Light Blue Account"] == (0.0, 0.0)
    and r3["Blue Account"] == (3.0, 9.0), (r3["Light Green Account"], r3["Blue Account"]))
_, r4 = run(1, 5, 300)          # light_green forx 경계=3.50×5
chk("tier 경계 값", r4["Light Green Account"][1] == 17.5, r4["Light Green Account"])

# ⑶ 파라미터 결측 → not_computable 전원(rows 공집합)
res5, rows5 = run(None, 6, 350)
chk("결측=전원 보류", rows5 == {} and len(res5.get("not_computable") or []) == 10,
    (len(rows5), len(res5.get("not_computable") or [])))

# ⑷ row_template 컴팩트 렌더(x291b 형식 포렌식판·출시 문면) — 값·행 수·보류 표기
res6 = apply_op(OP, {"months": 3, "withdrawals_per_month": 6, "withdrawal_amount": 350})
chk("row_template=str", isinstance(res6, str) and res6.count("\n") == 9, type(res6).__name__)
_pairs = dict(re.findall(r"- (.+?): out-of-network ATM total \$([\d.]+)", res6))
chk("컴팩트 렌더 값", _pairs.get("Green Fee-Free Account") == "0.00"
    and _pairs.get("Purple Account") == "45.00" and len(_pairs) == 10, _pairs)
res7 = apply_op(OP, {"months": None, "withdrawals_per_month": 6, "withdrawal_amount": 350})
chk("결측 렌더=보류 표기", isinstance(res7, str) and "not computable" in res7
    and "(none computable)" in res7, res7[:80])
_txt = (E or {}).get("return_template", "").format(result=res6)
chk("템플릿 렌더", "SEPARATE columns" in _txt and "Green Fee-Free Account" in _txt
    and "does not pick" in _txt and "OUTSIDE these totals" in _txt, _txt[:120])

# ⑸ 3사본 동일
E2 = load_entry("a2/banking_knowledge.gate.json")
E3 = load_entry("a2/split/banking_knowledge.core.json")
chk("3사본 바이트-동일(json 등가)", E == E2 == E3)

print("\n%d/%d" % (sum(OK), len(OK)))
sys.exit(0 if all(OK) else 1)
