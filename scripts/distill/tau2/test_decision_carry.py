# -*- coding: utf-8 -*-
"""회귀 — CP2 DECISION-CARRY (오프라인·모델 0·env 0·설계 v1.5 §4·§7 스텝 3·원장 C435).

CP2 는 **채널 변경 하나**다: 전문가가 이미 낸 결론 문자열을 배타 체인(`rw_fb`=rank 11)에서
빼내 비커밋 생성-뷰로 옮긴다. 그래서 검정할 것은 성적이 아니라 **경계**다 —

  ⑴ 값을 **엔진이 고르지 않는가** (설계 §4.2 의 유일한 위험선·[[62]] ④).
     `account_class` 는 채점 칸 그 자체라 이 선을 넘으면 070/071 실험이 죽는다.
  ⑵ 도메인 텍스트를 **파싱하지 않는가** ([[59]] — 재발화 판정은 문자열 동등성이어야 한다).
  ⑶ 히스토리에 **커밋하지 않는가** (C298 — 커밋하면 `set_state` replay 가 깨진다).
  ⑷ 배타 체인에 **얹지 않는가** (설계 §3.1 — 16번째 경쟁자가 되면 오늘의 0/6 재생).
  ⑸ 재발화가 **횟수가 아니라 인자 변화** 기준인가 ([[57]]).
  ⑹ 기본 **OFF** 인가 (플래그 0 이면 종전과 바이트 동일).
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
OK = []


def chk(name, cond, extra=""):
    OK.append(bool(cond))
    print("  %s %s%s" % ("PASS" if cond else "FAIL", name, (" — " + str(extra)) if extra else ""))


m = re.search(r"if _m3 and os\.environ\.get\(\"T2_DECISION_CARRY\"\).*?"
              r"\n(\s+)if _m3:\n", SRC, re.S)
chk("CP2 블록을 찾았다", m is not None)
body = m.group(0) if m else ""
code = "\n".join(l for l in body.split("\n") if not l.lstrip().startswith("#"))

# ⑴ 엔진이 고르지 않는다 ------------------------------------------------------
for pat, why in ((r"\bargmax\b", "argmax"),
                 (r"\bsorted\s*\([^)]*\)\s*\[\s*0\s*\]", "sorted(...)[0]"),
                 (r"\bmax\s*\(", "max("),
                 (r"\bmin\s*\(", "min("),
                 (r"\.sort\s*\(", ".sort(")):
    chk("선택기가 없다: %s" % why, not re.search(pat, code), why)

# ⑵ 도메인 텍스트를 파싱하지 않는다 ------------------------------------------
for pat, why in ((r"re\.(search|findall|match|sub|split)\s*\(", "정규식"),
                 (r"\.split\s*\(\s*[\"']", "구분자 split"),
                 (r"\.replace\s*\(\s*[\"']", "문자열 치환")):
    chk("도메인 텍스트 파싱 없음: %s" % why, not re.search(pat, code), why)
chk("재발화 판정은 문자열 동등성", "_m3 != " in code, )

# ⑶ 비커밋 --------------------------------------------------------------------
# ★C443: 채널이 **뷰 큐 → 이 턴의 재생성 버퍼**로 바뀌었다(뷰 큐는 다음 턴 소비라 한 턴
#   늦었고 `arrived=False` 로 실측됐다). 불변식은 그대로 **비커밋**이다 — 생성 버퍼(`work`)에
#   붙지 `state.messages` 에는 안 붙는다(C298).
chk("이 턴 재생성 버퍼로 보낸다(_t2_cp2_pending)", "_t2_cp2_pending" in code)
chk("소비처가 재생성 버퍼다(work + UserMessage)",
    re.search(r"_t2_cp2_pending.*?work = work \+ \[UserMessage", SRC, re.S) is not None)
chk("뷰 큐를 더 이상 쓰지 않는다(한 턴 지연 제거)", "_t2_view_fb" not in code)
for pat in ("_append", "state.messages.append", "messages.append"):
    chk("히스토리 커밋 없음: %s" % pat, pat not in code)

# ⑷ 배타 체인 밖 --------------------------------------------------------------
chk("체인에 얹지 않는다(_m3 를 비운다)", re.search(r"_m3 = \"\"", code) is not None)
chk("rw_fb 를 직접 건드리지 않는다", "rw_fb" not in code)
chk("fb 배치를 건드리지 않는다", "fb.append" not in code and "_SRC8" not in code)

# ⑸ [[57]] 인자 변화 기준 -----------------------------------------------------
chk("같은 값이면 재발화 0", "_t2_cp2_said" in code)
chk("횟수 상한으로 막지 않는다(카운터 비교 없음)",
    not re.search(r"_t2_cp2_(fired|count)", code))

# ⑹ 기본 OFF ------------------------------------------------------------------
chk("기본 OFF (== \"1\" 로 열린다)",
    'os.environ.get("T2_DECISION_CARRY") == "1"' in SRC)
chk("기본값을 1 로 두지 않았다",
    'os.environ.get("T2_DECISION_CARRY", "1")' not in SRC)

# 파싱 -------------------------------------------------------------------------
try:
    import ast
    ast.parse(SRC)
    chk("t2_gate_patch.py 파싱", True)
except Exception as e:
    chk("t2_gate_patch.py 파싱", False, e)

print("\n%d/%d" % (sum(OK), len(OK)))
sys.exit(0 if all(OK) else 1)
