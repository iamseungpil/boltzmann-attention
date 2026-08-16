# -*- coding: utf-8 -*-
"""t2_probe/t2_gap 회귀 — **규율이 코드에 박혀 있는가**(2026-08-16).

강제하는 것:
  ① 잡음 규율이 상수·함수로 존재한다(`NOISE=4` · `cite()` = 차 ≥5 만 참)
  ② `A_REF` 없는 프로브는 **거절**된다(기준선 없이 귀속 금지)
  ③ 통제 팔이 참조와 안 갈리면 **판정을 인쇄하지 않는다**(오늘의 무효 통제 2건 재발 방지)
  ④ `max_tokens` 가 모듈 상수로 고정돼 있다(팔 간 비교 보존)
  ⑤ 사다리가 라이브 요인을 **한 번에 하나씩** 되돌린다(단 이름이 코드에 있다)
"""
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OK = True


def chk(cond, msg):
    global OK
    OK = OK and bool(cond)
    print("  %s %s" % ("✓" if cond else "✗", msg))


import t2_probe as P                                              # noqa: E402

print("[①] 잡음 규율")
chk(P.NOISE == 4, "NOISE = 4 (C483 잡음 바닥)")
chk(not P.cite(8, 12) and P.cite(8, 13), "차 4는 인용 불가 · 차 5는 인용 가능")
chk(not P.cite(0, 4) and P.cite(0, 24), "0↔4 불가 · 0↔24 가능")

print("[②] A_REF 강제")
out = io.StringIO()
_stdout = sys.stdout
sys.stdout = out
try:
    r = P.run("t", {"tag": "t", "task": "t", "cut": 0, "base": ""},
              [("B_ONLY", "")], {"X": "X"}, "v", "a")
finally:
    sys.stdout = _stdout
chk(r is None, "A_REF 없는 셀 구성은 None 을 돌려주고 멈춘다")
chk("A_REF" in out.getvalue(), "이유를 인쇄한다")

SRC = io.open(os.path.join(HERE, "t2_probe.py"), encoding="utf-8").read()
print("[③] 통제 검사")
chk("통제 무효 가능" in SRC, "통제가 안 갈리면 그렇게 인쇄한다")
chk(SRC.index("통제 무효 가능") < SRC.index('print("\\n측정치'), "판정 인쇄보다 먼저 검사한다")
chk("ok = False" in SRC, "무효면 측정치 요약을 내지 않는다")

print("[④] 표집 고정")
chk(P.MAXTOK == 60, "max_tokens 가 모듈 상수(60)")
# ★2026-08-17 det 모드 추가(사용자 지시 "온도 0 이면 n=1 로 확정"): 표집 모드의 규약은 그대로이고
#   det=True 일 때만 전 표본 온도 0 이다. 두 조건을 함께 잠근다.
chk("0.0 if (det or i == 0) else 0.7" in SRC,
    "표집 모드는 첫 표본만 temp 0 · det 모드는 전부 0")
chk("k, nb = 2, 1" in SRC, "det 모드는 팔당 2회(온도 0)만 뽑는다")
chk("결정론 확인(온도 0 ×2 동일) ⇒ **n=1**" in SRC, "동일하면 n=1 로 확정해 인쇄한다")
chk("결정론 깨짐" in SRC, "갈리면 n=1 확정 불가를 인쇄한다")

print("[⑤] 사다리")
G = io.open(os.path.join(HERE, "t2_gap.py"), encoding="utf-8").read()
for rung in ("I0_CORE", "I1_CTX", "I2_EARLY", "I3_RIVAL", "I4_TOOL"):
    chk(rung in G, "단 %s 이 있다" % rung)
chk("지어내면" in G or "지어 넣지 않는다" in G, "못 찾으면 건너뛴다(대체물 저작 금지)")
chk("openai_schema" in G, "도구 바인딩은 환경 스키마 전체(지목 0)")

print("\nRESULT: %s" % ("ALL PASS" if OK else "FAIL"))
sys.exit(0 if OK else 1)
