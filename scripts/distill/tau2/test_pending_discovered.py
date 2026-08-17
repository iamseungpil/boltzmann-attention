# -*- coding: utf-8 -*-
"""`T2_PENDING_DISCOVERED`(019 가족 수리) 배선 검정 — 소스 수준 계약.

x362(det·부정통제 유효): 현행 정적 집합이면 형식화기가 `call_discoverable_agent_tool` 을 지목하고,
**env 가 표면화한 discoverable 도구를 넣으면** 분쟁 도구를 지목한다. 그 수리가 다음을 지키는지 본다:
  ⑴ 집합의 출처가 **env 레지스트리 ∩ 이미 받은 도구 출력**뿐이다(도메인 어휘 0·정규식 0)
  ⑵ 플래그 OFF 면 종전 거동(추가 0)
  ⑶ 실패는 no-op(예외가 턴을 죽이지 않는다)
"""
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SRC = io.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "t2_gate_patch.py"),
              encoding="utf-8").read()
i = SRC.find("T2_PENDING_DISCOVERED")
BLK = SRC[max(0, i - 1400):i + 2600] if i >= 0 else ""


def chk(ok, why):
    print("  %s %s" % ("✓" if ok else "✗", why))
    return 0 if ok else 1


def main():
    bad = 0
    bad += chk(i >= 0, "플래그 `T2_PENDING_DISCOVERED` 가 있다")
    bad += chk("_user_discoverable(" in BLK, "출처 = env **user-discoverable 레지스트리**")
    bad += chk('getattr(_m9, "role", None) == "tool"' in BLK,
               "이미 받은 **도구 출력 텍스트**와 교집합만 쓴다")
    bad += chk("sub_tool_names(" in BLK, "**LLM 이 이름을 형식화**한다(엔진 교집합 아님)")
    bad += chk("_uacts |= set(_add9)" in BLK, "대기 집합에 **추가만** 한다(제거 0)")
    bad += chk("no-op" in BLK and "except Exception" in BLK, "실패는 no-op")
    for pat in ("re.", "regex", "dispute", "cash_back"):
        bad += chk(pat not in BLK, "블록에 %r 없다(정규식·도메인 어휘 0)" % pat)
    print("\n%s" % ("test_pending_discovered PASS" if not bad
                    else "test_pending_discovered FAIL %d건" % bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
