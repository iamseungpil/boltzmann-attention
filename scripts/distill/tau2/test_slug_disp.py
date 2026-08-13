# -*- coding: utf-8 -*-
"""회귀 — _slug_disp 슬러그→표시명 전개 (FIX-6·t7276 075 실측·2026-08-13).

구판 w.capitalize() 는 'fee-free'→'Fee-free' 를 만들어 WRITE_ARG_ENUM 소속 검사가
모델의 오표기 제출과 일치-통과했다(deny 미발화·gold 'Fee-Free' 불일치). env 문서 title
('Green Fee-Free Account: …')이 하이픈 대문자화의 기계 검증원(리모트서 육안 확정 2026-08-13).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from t2_gate_patch import _slug_disp                               # noqa: E402

OK = []


def chk(name, cond, extra=""):
    OK.append(bool(cond))
    print("  %s %s%s" % ("PASS" if cond else "FAIL", name, (" — " + str(extra)) if extra else ""))


CASES = {
    "green_fee-free_account": "Green Fee-Free Account",   # FIX-6 표적(하이픈)
    "gold_years_account": "Gold Years Account",
    "sky_blue": "Sky Blue",                               # 맨이름 군(2561 사이트) 불변
    "blue_account": "Blue Account",
    "light_blue_account": "Light Blue Account",
}
for slug, want in sorted(CASES.items()):
    got = _slug_disp(slug)
    chk(slug, got == want, (got, want))

# 구판 결함 재현 방지(음성 통제): capitalize 산출물과 달라야 한다
old = " ".join(w.capitalize() for w in "green_fee-free_account".split("_"))
chk("구판과 분기(음성 통제)", old == "Green Fee-free Account"
    and _slug_disp("green_fee-free_account") != old, old)

print("\n%d/%d" % (sum(OK), len(OK)))
sys.exit(0 if all(OK) else 1)
