# -*- coding: utf-8 -*-
"""`t2_search.quote_in` 검정 — 강조 표기 때문에 **참인 인용이 탈락**하지 않는가(C510),
그리고 **없는 인용은 여전히 떨어지는가**(부정통제·[[57]]).

실화(2026-08-17): 손님 축자 `- I tap into my savings **3–4 times a week**, sometimes more.` 를
모델이 별표 없이 인용해 `q in text` 가 **제안 7개 → 통과 0개**를 냈고, 그 조용한 0 이
*"모델이 요구가 없다고 답했다"* 로 읽혔다. 그 오독을 코드가 막는다.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_search as TS

SRC = ("Perfect! Now I also need a savings account. Let me tell you what I'm looking for there... "
       "- I tap into my savings **3–4 times a week**, sometimes more. I really don't want fees for "
       "that. - Since I already have checking with you, is there a **bonus rate**?")

DQ = chr(34)          # "  — 보통 따옴표
LQ = chr(0x201C)      # “
RQ = chr(0x201D)      # ”

CASES = [
    # (인용, 기대)
    ("I tap into my savings **3–4 times a week**, sometimes more.", True),   # 축자 그대로
    ("I tap into my savings 3–4 times a week, sometimes more.", True),       # 강조만 뺀 인용 ★
    ("is there a bonus rate?", True),                                       # 강조 안쪽만
    ("I keep fifty thousand dollars in savings.", False),                   # ★없는 말 = 탈락
    ("I want a Silver Plus Account.", False),                               # ★날조 = 탈락
    ("", False),
    # ★감싸는 따옴표(2026-08-18·x374): 프롬프트가 "Copy VERBATIM" 이라 모델이 인용을
    #   따옴표로 **감싸서** 낸다. 내용이 축자면 통과해야 한다 — 실측으로 024 인용 2건이
    #   quote_in(원본)=False ↔ quote_in(벗김)=True 였다(우리 검산이 참인 인용을 떨어뜨림).
    (DQ + "I tap into my savings 3–4 times a week" + DQ, True),
    (LQ + "I tap into my savings 3–4 times a week" + RQ, True),
    # ⚠**벗기는 것은 바깥 구분자뿐** — 감쌀다고 없는 문장이 통과하면 안 된다
    (DQ + "I want a Silver Plus Account." + DQ, False),
    (DQ + DQ, False),
]


def main():
    bad = 0
    for q, want in CASES:
        got = TS.quote_in(q, SRC)
        ok = (got == want)
        bad += (not ok)
        print("%s want=%-5s got=%-5s %r" % ("OK  " if ok else "FAIL", want, got, q[:60]))
    if TS.quote_in("anything", ""):
        print("FAIL 빈 원문에 통과"); bad += 1
    print("\n%s" % ("test_quote_in PASS" if not bad else "test_quote_in FAIL %d건" % bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
