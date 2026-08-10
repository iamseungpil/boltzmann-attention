# -*- coding: utf-8 -*-
r"""x215 — x213 의 **사례 편차**가 사례 성질인가 프로브 결함인가 (유료 0 · LLM 호출 0).

## 왜

x213(42셀)에서 갈렸다 — `E_CLEAN`(천장)이 6/6 인 사례가 있는가 하면 **0/6** 인 사례(x010·day4b)와
2/6(all97)이 있고, 아예 `A_FULL` 이 6/6 인 사례(c13)도 있다. 천장이 무너지면 그 사례의 모든 팔이
무의미하므로 **먼저 구성부터 의심한다**([[55]] 우리 배관 먼저).

## 첫 용의자 (프로브 결함)

`E_CLEAN` 은 *"`Record ID` 가 들어간 **첫** tool 메시지"* 를 원장으로 집는다. 그런데 이 도메인에서
사용자 조회도 `Found 1 record(s) in 'users': … Record ID: …` 형태다. 첫 메시지가 **추천 원장이
아니라 사용자 조회**면 `E_CLEAN` 은 답에 필요한 행을 아예 안 들고 있고, 그러면 0/6 은 사례 성질이
아니라 **내가 만든 통제 오염**이다.

## 인쇄하는 것 (LLM 없이 문자열만)

  · 각 사례의 `E_CLEAN` 이 실제로 무엇을 담았나 — 추천 원장인가, 몇 행인가, 상태값이 있나
  · `probe_point` 가 고른 손님 발화 — 사례마다 같은 종류의 질문인가
  · `A_FULL` 문맥에 정의 문장이 몇 번 나오나 · 이관 압박 문장 수

실행: python x215_case_variance.py
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from x213_congestion_ablation import cases, probe_point, render, SIG   # noqa: E402


def main():
    for tag, trial, msgs in cases():
        i = probe_point(msgs)
        ask = " ".join(str(msgs[i].get("content") or "").split())
        full, _dd, _dx = render(msgs, i, "A")
        # E_CLEAN 이 집는 것과 **똑같이** 집는다
        led = next((" ".join(str(m.get("content") or "").split()) for m in msgs
                    if m.get("role") == "tool" and "Record ID" in str(m.get("content") or "")), "")
        which = ("referrals" if "referral" in led.lower() else
                 ("users" if "'users'" in led or "user_id:" in led else "?"))
        nrec = len(re.findall(r"Record ID:", led))
        has_status = bool(re.search(r"referral_status", led))
        print("\n" + "=" * 96)
        print("%s trial=%s" % (tag, trial))
        print("  E_CLEAN 이 집은 원장: **%s** · Record %d개 · 상태필드 %s"
              % (which, nrec, "있음" if has_status else "**없음**"))
        print("  그 원장 앞 120자: %s" % led[:120])
        print("  probe_point 발화: %s" % ask[:150])
        print("  A_FULL: %d자 · 정의 등장 %d회 · 이관 압박 %d문장"
              % (len(full), full.count(SIG),
                 len(re.findall(r"transfer|human agent", full, re.I))))
    print("\n※ `상태필드 없음` 이거나 `users` 를 집은 사례는 **천장이 오염된 것**이다 —")
    print("  그 사례의 x213 수치는 사례 성질이 아니라 내 구성 결함이고, 합계에서 빼야 한다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
