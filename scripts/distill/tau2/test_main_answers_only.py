# -*- coding: utf-8 -*-
"""회귀 검정: **메인은 답만 싣는다** (`T2_MAIN_ANSWERS_ONLY`·2026-08-11·C420·메모리 [[65]]).

사용자 지시(2026-08-11): *"메인 컨텍스트는 서브에이전트 호출과 결과만 실린다. 과정은 서브 안에서
끝난다."*

무엇을 막는 검정인가 —
 ⒜ **끈 상태에서 거동이 바뀌는 것**. 기본 OFF 이고 OFF 면 여덟 조각이 종전대로 메인에 나간다.
 ⒝ **답까지 빼는 것**. 켜도 `diagnosed_text`(=서브의 답)는 메인에 남아야 한다.
 ⒞ **조용한 손실**. 메인에서 뺀 재료는 **서브 문맥으로 옮겨지거나**, 못 옮기면 **인쇄**돼야 한다.
 ⒟ **클로저 사고**. 안쪽에서 `_subonly` 를 재대입하면 `_emit` 가 다른 객체를 쓰게 된다.

오프라인 전용(LLM·서버 불요). 실행: py -3 test_main_answers_only.py
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

FAILED = []


def chk(c, label):
    print(("  OK   " if c else "  FAIL ") + label)
    if not c:
        FAILED.append(label)


HERE = os.path.dirname(os.path.abspath(__file__))
SRC = open(os.path.join(HERE, "t2_gate_patch.py"), encoding="utf-8").read()
i = SRC.find("def _limit_reduce_text(")
j = SRC.find("\ndef ", i + 10)
SEG = SRC[i:j if j > 0 else i + 30000]

print("\n§1 플래그 · 기본 OFF")
chk(os.environ.get("T2_MAIN_ANSWERS_ONLY") != "1", "환경에 기본 ON 이 박혀 있지 않다")
chk('os.environ.get("T2_MAIN_ANSWERS_ONLY") == "1"' in SEG, "플래그 뒤에 있다")
chk(re.search(r"if is_answer or not _answers_only:\s*\n\s*return text", SEG) is not None,
    "OFF 면 조각이 **그대로 메인으로** 간다 (거동 보존)")

print("\n§2 여덟 조각이 전부 `_emit` 를 지난다")
# ★답은 `_emit` 를 안 거쳐도 된다 — 애초에 메인에 남을 것들이다. 어느 것이 답인지를 **여기에
#   적어 둔다**: 그래야 새 조각이 슬쩍 끼어들 때 이 검정이 잡는다(화이트리스트가 곧 명세다).
ANSWERS = ("decided_text", "rederived_text", "diagnosed_text")
raw = [t for t in re.findall(r"_add \+= (?!_emit)([^\n]+)", SEG)
       if not any(a in t for a in ANSWERS)]
chk(not raw, "`_emit` 를 안 거치고 메인에 붙는 **과정** 조각이 없다 (%s)" % (raw[:2] or "없음"))
chk(len(re.findall(r"_add \+= (?!_emit)([^\n]+)", SEG)) == 2,
    "답으로 직행하는 조각은 **둘뿐**이다 (결정 블록·재도출 문구)")
chk(SEG.count("_add += _emit(") >= 8, "여덟 조각이 감싸였다 (%d)" % SEG.count("_add += _emit("))

print("\n§3 답은 남고 과정은 옮겨진다")
chk('_emit(_sp2["diagnosed_text"].format(answer=_dg[1]), is_answer=True)' in SEG,
    "서브의 **답**은 `is_answer=True` 로 메인에 남는다")
for name in ("status_breakdown", "window_history", "_il2", "_elig", "exhausted_text"):
    chk(re.search(r"_add \+= _emit\([^\n]*%s" % re.escape(name), SEG) is not None,
        "%s 는 과정으로 분류된다" % name)

print("\n§4 옮긴 재료가 **서브 문맥**에 실린다 (빼기만 하면 손실이다)")
chk('_blk = "\\n".join([_blk] + _subonly)' in SEG, "진단 서브 문맥에 이어 붙인다")
chk('_tbl5 = "\\n".join([_tbl5] + _subonly)' in SEG, "재도출 서브 문맥에도 이어 붙인다")
chk("del _subonly[:]" in SEG and "_subonly = []" not in SEG.split("def _emit")[1][:4000],
    "안쪽에서 **재대입하지 않는다**(클로저 공유) — 비우기만 한다")

print("\n§5 조용한 손실 금지")
chk("미소비 재료" in SEG, "어느 서브에도 못 실린 재료는 **인쇄**된다 ([[64]] 의 정신)")
k = SEG.find("미소비 재료")
chk(0 < SEG.rfind("if _answers_only and _subonly:", 0, k) < k, "그 인쇄가 플래그 뒤에 있다")

print("\n%s  (%d/%d)" % ("FAIL" if FAILED else "ALL PASS", 16 - len(FAILED), 16))
sys.exit(1 if FAILED else 0)
