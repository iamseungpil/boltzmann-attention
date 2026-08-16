# -*- coding: utf-8 -*-
"""T2_MATERIAL_RESERVE 회귀 — **예산을 결정점에 남기는가**(2026-08-16·C498).

사건: 배달은 그 턴의 **재생성 버퍼**에만 붙는다(비커밋·C298). t7298 의 055 는 sim 당 예산 3회를
`대화텍스트 1`(손님이 요구를 말하기도 전)부터 소진했고, 궤적 재료 표지 **0건**, 선택 **0/4**
↔ 같은 재료로 격리는 **24/24**.

불변식:
  ① 일반(초반) 자리 배달이 예약 모드에서 **1회로 묶인다**
  ② **총 예산 3은 그대로**(사용처만 옮긴다 — 더 주는 것이 아니다)
  ③ 플래그 기본 OFF = 거동 불변
  ④ 결정 자리 경로(`T2_SEARCH_ON_PROCEED`)는 이 상한에 걸리지 않는다
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


gen = re.search(r"if \(not _m3 and os\.environ\.get\(\"T2_SEARCH_AGENT\"\).{0,900}?_t2_sa_early.{0,400}?\n",
                SRC, re.S)
body = gen.group(0) if gen else ""

print("[①] 일반 자리 1회로 묶임")
chk(bool(gen), "일반 배달 자리를 찾았다")
chk('_t2_sa_early", 0) < 1' in body, "예약 모드에서 일반 자리는 1회")
chk('T2_MATERIAL_RESERVE' in body, "플래그로만 켜진다")

print("[②] 총 예산 불변")
chk('_t2_searchagent_fired", 0) < 3' in body, "총 상한 3 이 그대로 남아 있다")
chk(SRC.count('_t2_searchagent_fired", 0) < 3') >= 2, "결정 자리에도 같은 총 상한이 걸려 있다")

print("[③] 기본 OFF")
chk('os.environ.get("T2_MATERIAL_RESERVE") != "1"' in body,
    "플래그가 없으면 종전 거동(무제한 일반 배달, 총 3)")

print("[④] 결정 자리는 안 묶인다")
# ★창 600→2000자 (2026-08-16): T2_PROCEED_DOCBODY 주석+가드가 조건과 호출 사이에 들어와
#   경로 자체는 불변인데 창이 짧아 미검출됐다. 거동 검사 대상은 그대로다.
proc = re.search(r"T2_SEARCH_ON_PROCEED\"\) == \"1\".{0,2000}?_search_material", SRC, re.S)
chk(bool(proc), "결정 자리 경로가 있다")
chk('_t2_sa_early' not in (proc.group(0) if proc else "x_t2_sa_early"),
    "결정 자리는 일반-자리 상한(_t2_sa_early)에 걸리지 않는다")

print("\nRESULT: %s" % ("ALL PASS" if OK else "FAIL"))
sys.exit(0 if OK else 1)
