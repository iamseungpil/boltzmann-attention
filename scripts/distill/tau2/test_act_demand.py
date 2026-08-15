# -*- coding: utf-8 -*-
"""`T2_ACT_DEMAND` 검정 — 행동 촉구가 (a) 규율을 지키고 (b) 플래그로만 열리는지.

근거(격리 3런 재현·부정통제 포함):
    x330 `C_EMIT_ASK` 11/24 · x331 `D_ASK` 13/24 · x332 `C_ASK` **16/24**
    ↔ 기준선 2 / 0 / 6 · `D_EARLY` 세 런 모두 **0/24**
반례(같이 박아 둔다): x332 `B_SELFLIST`(세고·체크하고·처리) **0/24** — 기준선보다 낮다.
⇒ **열거를 시키면 묘사가 늘고 실행이 준다.** 문구는 *열거 없는 행동 명령* 하나여야 한다.

못 박는 것:
  ⒜ 문구에 **도구 이름이 없다**(x322: 지목은 24/24 → 0/24 로 파괴)
  ⒝ 문구에 **도메인 어휘가 없다**([[05]] 고정)
  ⒞ 문구가 **열거를 시키지 않는다**(x332 반례)
  ⒟ 게이트는 `T2_ACT_DEMAND` 로만 열린다(기본 OFF = 거동 변화 0)
  ⒠ 격리 컷과 **같은 구조 조건**에서 발화한다(deny 여부와 무관)

실행: seka python test_act_demand.py
"""
import io
import os
import re
import sys

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

FAIL = []
SRC = io.open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "t2_gate_patch.py"), encoding="utf-8").read()
DEMAND = "Carry out the next step of this request now."
# 도메인 어휘(은행) — 하나라도 들어가면 [[05]] 위반
DOMAIN_WORDS = ("account", "credit", "card", "fee", "refund", "transfer", "bank",
                "checking", "savings", "dispute", "transaction")
# 열거를 시키는 말 — x332 가 해롭다고 잰 형태
LIST_WORDS = ("list", "how many", "count", "mark", "check off", "enumerate", "each one")


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAIL.append(msg)


def main():
    check(DEMAND in SRC, "촉구 문구가 소스에 있다")
    low = DEMAND.lower()
    check(not re.search(r"[a-z_]+_\d{4}", DEMAND), "⒜문구에 도구 이름이 없다")
    hit = [w for w in DOMAIN_WORDS if w in low]
    check(not hit, "⒝문구에 도메인 어휘가 없다 (발견: %s)" % (hit or "없음"))
    lhit = [w for w in LIST_WORDS if w in low]
    check(not lhit, "⒞문구가 열거를 시키지 않는다 (발견: %s · x332 반례)" % (lhit or "없음"))
    check('os.environ.get("T2_ACT_DEMAND") == "1"' in SRC, "⒟플래그로만 열린다")
    check(SRC.count("[T2_ACT_DEMAND]") >= 2, "배달·미배달 둘 다 로그를 남긴다([[64]])")

    # ⒠ 발화 자리 = 격리 컷과 같은 구조 조건 = deny 분기 **앞**
    i_dem = SRC.find('os.environ.get("T2_ACT_DEMAND")')
    i_deny = SRC.find('if _ar.get("status") == "deny":\n                                _fb_ar')
    check(i_dem != -1 and i_deny != -1 and i_dem < i_deny,
          "⒠deny 분기 **앞**에서 발화한다(=deny 여부와 무관·격리 컷과 동형)")

    # 재배달 억제([[57]]: 횟수가 아니라 인자 변화로)
    seg = SRC[i_dem: i_dem + 1200] if i_dem != -1 else ""
    check("_t2_cp2_said" in seg, "같은 문자열이면 재배달하지 않는다([[57]])")

    print("\n%s" % ("PASS" if not FAIL else "FAIL: " + " · ".join(FAIL)))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
