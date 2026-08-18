# -*- coding: utf-8 -*-
"""`_resolve_cap_ok` 두 리셋 경로의 **관측 의무** 회귀 (2026-08-18·C530).

## 무엇을 막는가

상한(cap=3)은 *정체에만 과금*하고 **진행이 있으면 되돌린다**. 되돌리는 경로가 둘인데,
관측 의무(C442)가 **한쪽에만** 달려 있었다:

  ⓐ `done - prev` — 새로 **실행된** 도구 이름이 늘면 리셋. ← **마커가 없었다(조용함)**
  ⓑ 새로 **회수된** unlockable 이름이 늘면 리셋. ← `[T2_RESOLVE_CAP] 리셋` 을 찍었다

그래서 t7308 전수(24 sim)에서 **ⓑ 마커가 0 인데 resolve deny 가 sim 당 11~23** 인 상태를
로그만으로는 설명할 수 없었고, `_exact_tool_name` 이 `call_*` 를 내부 이름으로 푼다는 것
(⇒ discoverable 44종이 집합을 계속 키운다)을 **소스를 읽고 프로브(x372)를 짜야** 알 수 있었다.
실효 상한이 3 이 아니라 **≈11/sim** 이었다는 사실이 로그에 한 줄도 없었다는 뜻이다.

## 불변식

  ① 두 경로가 **둘 다** 마커를 찍는다 — 어느 쪽이 리셋했는지 로그로 갈린다.
  ② 둘 다 **실효 리셋일 때만** 찍는다(이미 0 인 카운터를 0 으로 되돌리는 것은 사건이 아니다).
     이 함수는 한 턴에 여러 번 불리고 스냅샷은 **발화 시점에만** 갱신되므로, 조건이 없으면
     한 번의 진행이 마커 수십 줄을 만든다. 두 경로가 **같은 조건**이라야 셈을 비교할 수 있다.
  ③ **거동 불변** — 리셋은 여전히 무조건 일어나고(`= 0`), 반환은 여전히 `< cap` 이다.
     이 검정은 *보이게 만드는 것*까지만 보장한다. 상한 의미를 바꿀지는 별개의 설계 결정이고,
     현재 **손해가 실측되지 않았다**(통과 sim 도 같은 횟수로 순환한다·C530).
  ④ 마커는 **무엇이** 리셋을 유발했는지 남긴다(이름 목록) — C442 가 ⓑ 에 건 요구와 동일.
"""
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
SRC = io.open(os.path.join(HERE, "t2_gate_patch.py"), encoding="utf-8").read()
OK = True


def chk(cond, msg):
    global OK
    OK = OK and bool(cond)
    print("  %s %s" % ("✓" if cond else "✗", msg))


m = re.search(r"def _resolve_cap_ok\(self, messages=None, a2=None\):.*?\n    return getattr"
              r"\(self, \"_t2_resolve_deny\", 0\) < cap", SRC, re.S)
BODY = m.group(0) if m else ""

print("[①] 두 경로가 둘 다 마커를 찍는다")
chk(bool(m), "_resolve_cap_ok 정의를 찾았다")
chk("리셋(실행)" in BODY, "ⓐ 실행-집합 경로가 마커를 찍는다 (이전엔 조용했다)")
chk("리셋(회수)" in BODY, "ⓑ 회수-이름 경로가 마커를 찍는다")
chk(BODY.count("[T2_RESOLVE_CAP]") == 2, "마커는 정확히 두 곳 — 경로마다 하나")

print("[②] 둘 다 실효 리셋일 때만 찍고, **기능이 관측보다 앞선다**(2026-08-18·C538)")
# ★계약이 바뀌었다. 옛 판은 *"마커는 리셋 **전에** 조건을 본다"* 를 요구했고, 그 요구가 `print`
#   를 대입 앞에 두게 만들었다 — 그리고 그 `print` 가 `_sys` 미정의로 NameError 를 던지자
#   바깥 `except: pass` 가 **리셋 대입까지 삼켰다**(x381 줄-추적·상한이 영구 래치가 됐다).
#   ⇒ 조건은 **대입 전 값을 지역에 붙잡아** 유지하고(인쇄되는 숫자는 그대로 0 이 아니다),
#     대입은 **먼저** 한다. 관측이 죽어도 기능은 산다.
guards = re.findall(r"if _was\d?:", BODY)
chk(len(guards) == 2, "두 경로 모두 '되돌린 값이 0 이 아닐 때만' 조건을 갖는다 — 실제 %d"
    % len(guards))
for path, needle in (("ⓐ", "리셋(실행)"), ("ⓑ", "리셋(회수)")):
    seg = BODY[:BODY.find(needle)]
    chk(0 <= seg.rfind("self._t2_resolve_deny = 0") < seg.rfind("print("),
        "%s **대입이 print 보다 앞**이다(관측이 예외를 던져도 리셋은 이미 끝났다)" % path)
chk("_sys." not in BODY,
    "이 함수는 `_sys`(다른 함수의 지역 별칭)를 쓰지 않는다 — 모듈-레벨은 `sys`")

print("[③] 거동 불변")
chk(BODY.count("self._t2_resolve_deny = 0") == 2, "리셋 대입은 여전히 두 경로 각 1회 — 조건부가 아니다")
chk("return getattr(self, \"_t2_resolve_deny\", 0) < cap" in BODY, "반환 술어 불변(< cap)")
chk('cap = int(_c) if (_c or "").strip().isdigit() else 3' in BODY, "기본 상한 3 불변")
chk("T2_RESOLVE_CAP" in BODY, "환경변수 노브 유지")

print("[④] 무엇이 리셋을 유발했는지 남긴다")
chk("sorted(done - prev)[:3]" in BODY, "ⓐ 는 새로 실행된 이름을 남긴다")
chk("sorted(cur - pvn)[:3]" in BODY, "ⓑ 는 새로 회수된 이름을 남긴다")
chk(BODY.count("정체 %d회 → 0") == 2, "두 마커 다 **되돌린 정체 횟수**를 남긴다 — 실효 상한을 셀 수 있다")

print("\n%s" % ("PASS" if OK else "FAIL"))
sys.exit(0 if OK else 1)
