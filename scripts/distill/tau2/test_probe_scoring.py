# -*- coding: utf-8 -*-
"""`t2_probe._count` 채점 검정 — **형제 후보가 gold 를 부분문자열로 품는 자리**(C515·리뷰 지적).

실화(2026-08-17 x357): `EVERGREEN ACCOUNT` 가 `GREEN ACCOUNT` 를 포함해 **오답을 적중으로** 셌고,
`LIGHT BLUE ACCOUNT` 가 `BLUE ACCOUNT` 를 포함해 **없던 적중을 만들었다**. 34축 중 5축이 그 형태다.
C486(소수점 표기)·x347($32,500 ⊃ $2,500)에 이은 **같은 계열 세 번째**라 검정으로 못 박는다.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_probe as P

CHECKING = ["Blue Account", "Light Blue Account", "Green Account (Checking)", "Evergreen Account",
            "Dark Green Account", "Light Green Account", "Purple Account", "Bluest Account"]
SAVINGS = ["Green Account (Savings)", "Silver Plus Account", "Gold Account", "Bronze Account"]

CASES = [
    # (답 본문, marks, names, 기대)
    ("Evergreen Account", {"G": "Green Account"}, CHECKING, 0),          # ★핵심: 형제가 품는다
    ("Green Account (Checking)", {"G": "Green Account (Checking)"}, CHECKING, 1),
    ("Light Blue Account", {"G": "Blue Account"}, CHECKING, 0),          # ★최장매칭이 끊는다
    ("Blue Account", {"G": "Blue Account"}, CHECKING, 1),
    ("Silver Plus Account", {"G": "Silver Plus Account"}, SAVINGS, 1),
    ("Green Account (Savings)", {"G": "Silver Plus Account"}, SAVINGS, 0),
    ("I would recommend the Purple Account for you.", {"G": "Purple Account"}, CHECKING, 1),
    ("none of these fit", {"G": "Purple Account"}, CHECKING, 0),
]

# names 를 안 준 종전 호출도 **경계 검사**는 받아야 한다
LEGACY = [("Evergreen Account", {"G": "Green Account"}, 0),
          ("Green Account is best", {"G": "Green Account"}, 1)]


def main():
    bad = 0
    for text, marks, names, want in CASES:
        got = P._count(text, marks, names)["G"]
        ok = (got == want)
        bad += (not ok)
        print("%s want=%d got=%d · %r vs %r" % ("OK  " if ok else "FAIL", want, got, text[:40],
                                                marks["G"]))
    print("-- names 없이(종전 호출·경계 검사만)")
    for text, marks, want in LEGACY:
        got = P._count(text, marks)["G"]
        ok = (got == want)
        bad += (not ok)
        print("%s want=%d got=%d · %r" % ("OK  " if ok else "FAIL", want, got, text[:40]))
    print("\n%s" % ("test_probe_scoring PASS" if not bad else "test_probe_scoring FAIL %d건" % bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
