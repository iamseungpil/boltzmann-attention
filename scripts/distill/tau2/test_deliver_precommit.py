# -*- coding: utf-8 -*-
"""T2_DELIVER_PRECOMMIT 회귀 — **시점만 옮겼는가**(2026-08-16·특허 B §B2-6 실시예용).

불변식:
  ① 기본 OFF (플래그 없으면 거동 불변)
  ② sim 당 **1회**(`_t2_precommit_done`)
  ③ **예산 총량 불변** — 같은 카운터(`_t2_searchagent_fired`)를 증가시킨다
  ④ 재료 생성은 **정본 진입점**(`_search_material`) — 사본·새 판단 0
  ⑤ 로그가 **turn 을 찍는다**(1차 종점 = 첫 지목 이전 도달률을 기계로 세려면 필수)
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


m = re.search(r'if \(os\.environ\.get\("T2_DELIVER_PRECOMMIT"\).{0,1200}?재료 %d자"', SRC, re.S)
body = m.group(0) if m else ""

print("[①] 기본 OFF")
chk(bool(m), "선-배달 블록이 존재한다")
chk('os.environ.get("T2_DELIVER_PRECOMMIT") == "1"' in body, "플래그로만 켜진다")
chk('os.environ.get("T2_SEARCH_AGENT") == "1"' in body, "검색 에이전트가 켜진 경우만")

print("[②] sim 당 1회")
chk('_t2_precommit_done' in body and 'not getattr(self, "_t2_precommit_done", False)' in body,
    "1회 플래그로 잠근다")

print("[③] 예산 총량 불변")
chk('_t2_searchagent_fired' in body, "같은 예산 카운터를 증가시킨다(새 예산 아님)")
chk(SRC.count('_t2_searchagent_fired", 0) < 3') >= 2, "총 상한 3 이 다른 자리에 그대로 있다")

print("[④] 정본 진입점")
chk('_search_material(self, a2, state.messages)' in body, "정본 함수를 쓴다(사본 0)")
chk('argmax' not in body and '정답' not in body, "선택·순위 문장 없음([[62]] ④)")

print("[⑤] 계기")
chk('turn=%d' in body, "turn 을 찍는다(첫 지목 이전 도달률 계산의 전제)")

print("\nRESULT: %s" % ("ALL PASS" if OK else "FAIL"))
sys.exit(0 if OK else 1)
