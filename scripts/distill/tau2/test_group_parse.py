# -*- coding: utf-8 -*-
"""`formalize_groups` 포함관계 필터 검정 (C516).

실화: 필터가 `business_checking_accounts` 가 있으면 `checking_accounts` 를 **무조건** 지웠다.
라이브 317줄 중 **134줄(42%)** 이 그 경우였고 전부 모델이 **둘 다 따로 나열**한 것이었다
(024 68·098 55·055 11). 098 은 gold 군이 매번 지워졌다.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

NAMES = ["checking_accounts", "business_checking_accounts", "savings_accounts",
         "business_savings_accounts", "credit_cards", "business_credit_cards"]


def parse(raw):
    """엔진 축자 규약(수리판) 재현 — t2_search.formalize_groups 의 파싱부와 같은 코드."""
    import t2_search as TS
    import types
    low = raw.lower()
    out = sorted((g for g in NAMES if g and g.lower() in low), key=lambda g: low.find(g.lower()))
    src = io.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "t2_search.py"),
                  encoding="utf-8").read()
    assert "_outside(" in src, "수리가 없다"
    def outside(low_, g_, longer_):
        i = low_.find(g_)
        while i >= 0:
            cov = False
            for o_ in longer_:
                j = low_.find(o_)
                while j >= 0:
                    if j <= i and i + len(g_) <= j + len(o_):
                        cov = True; break
                    j = low_.find(o_, j + 1)
                if cov: break
            if not cov: return True
            i = low_.find(g_, i + 1)
        return False
    return [g for g in out
            if outside(low, g.lower(), [o.lower() for o in out if o != g and g.lower() in o.lower()])]


import io  # noqa: E402

CASES = [
    ("business_checking_accounts", ["business_checking_accounts"]),          # 포함 artifact 제거
    ("checking_accounts, business_checking_accounts",
     ["checking_accounts", "business_checking_accounts"]),                   # ★둘 다 남는다
    ("The customer asks about checking_accounts and savings_accounts",
     ["checking_accounts", "savings_accounts"]),
    ("business_savings_accounts", ["business_savings_accounts"]),
    ("savings_accounts and business_savings_accounts",
     ["savings_accounts", "business_savings_accounts"]),
    ("none of these", []),
]


def main():
    bad = 0
    for raw, want in CASES:
        got = parse(raw)
        ok = sorted(got) == sorted(want)
        bad += (not ok)
        print("%s raw=%-52r → %s" % ("OK  " if ok else "FAIL", raw[:50], got))
    print("\n%s" % ("test_group_parse PASS" if not bad else "test_group_parse FAIL %d건" % bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
