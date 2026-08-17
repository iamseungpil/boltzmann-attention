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
    """**정본을 그대로 부른다** — 사본 금지([[67]]).

    ★2026-08-17 2차 수리: 이 검정은 원래 정본 파싱부를 **베껴** 갖고 있었다(수리가 정본에만
      들어가면 검정은 옛 거동을 통과시킨다). `t2_search.groups_in` 으로 뺐으니 그것을 부른다.
    """
    import t2_search as TS
    return TS.groups_in(raw, NAMES)


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
