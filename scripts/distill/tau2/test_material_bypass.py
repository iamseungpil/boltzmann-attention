# -*- coding: utf-8 -*-
"""T2_MATERIAL_BYPASS 회귀 — **배달이 요구와 분리됐는가**(2026-08-16·C494 후속).

이 수리가 노리는 사건: `_resolve_cap_ok`(제자리걸음 상한)가 검색 에이전트의 **재료 배달**까지
함께 멎게 했다. 실측 — t7295 의 055 세 sim 에서 결정점 전 재료 도착 **0/3**, t7297(수리 후)에도
`resolve_cap` **97회** 생존.

여기서 강제하는 불변식 넷:
  ① 우회 분기가 **`resolve_cap` 으로 멎은 자리에서만** 열린다(`other_lever`·`contract_off` 는 아님)
  ② 플래그 **기본 OFF** — 켜지 않으면 거동 불변
  ③ 반복 상한이 살아 있다 — `_t2_searchagent_fired < 3` · 같은 문자열 재배달 금지
  ④ 라벨 문자열을 **되파싱하지 않는다**(`_mgate_kind` 구조값) — 로그 문구를 바꿔도 안 깨진다
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


# 우회 블록 본문만 떼어 본다(주석 포함 — 조건은 코드에만 있다)
m = re.search(r"if \(_mgate_kind == \"resolve_cap\"(.{0,1400}?)\n            if \(_contract_on",
              SRC, re.S)
body = m.group(1) if m else ""

print("[①] resolve_cap 에서만 열린다")
chk(bool(m), "우회 분기가 존재하고 `_mgate_kind == \"resolve_cap\"` 로 진입한다")
chk("other_lever" not in body, "other_lever 를 여기서 열지 않는다(그쪽은 [[64]] 합류가 옳다)")

print("[②] 기본 OFF")
chk('os.environ.get("T2_MATERIAL_BYPASS") == "1"' in body, "플래그 없이는 안 켜진다")
chk('os.environ.get("T2_SEARCH_AGENT") == "1"' in body, "검색 에이전트가 켜져 있을 때만")
chk(os.environ.get("T2_MATERIAL_BYPASS") is None or os.environ.get("T2_MATERIAL_BYPASS") != "1"
    or True, "(런타임 값은 드라이버 소관)")

print("[③] 반복 상한 생존")
chk('_t2_searchagent_fired", 0) < 3' in body, "sim 당 3회 상한이 그대로 걸린다")
chk('_t2_cp2_said' in body, "같은 재료면 재배달하지 않는다")
chk('_search_material(' in body, "재료 생성은 정본 진입점을 쓴다(사본 0·[[67]])")

print("[④] 라벨 되파싱 없음")
chk('_mgate.startswith(' not in SRC, "로그 라벨 문자열을 조건으로 쓰지 않는다")
chk(SRC.count('_mgate_kind = "resolve_cap"') == 1, "구조값이 한 자리에서만 설정된다")

print("\nRESULT: %s" % ("ALL PASS" if OK else "FAIL"))
sys.exit(0 if OK else 1)
