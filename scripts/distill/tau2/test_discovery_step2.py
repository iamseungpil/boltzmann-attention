# -*- coding: utf-8 -*-
"""회귀 — 진행-감응 발견 요구 (T2_DISCOVERY_STEP2 · 오프라인 · 모델 0 · 원장 C442).

이 레버가 사는 근거는 x273(n=8): 출시 문구는 이름을 이미 알아도 **UNLOCK 0/8**(전부 재검색),
(2)단계 문구로 바꾸면 **8/8**. 그래서 검정할 것은 그 경계다 —

  ⑴ 기본 OFF · ⑵ 이름은 **레지스트리 ∩ 회수 텍스트**에서만(짓지 않는다)
  ⑶ 이미 unlock 시도한 이름은 다시 말하지 않는다 · ⑷ 못 찾으면 종전 문구(회귀 0)
  ⑸ 출시 문구 = x273 이 이긴 문자열(축자·[[03b]]) · ⑹ 도메인 리터럴 0
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
sys.path.insert(0, HERE)

SRC = io.open(os.path.join(HERE, "t2_resolve.py"), encoding="utf-8").read()
GATE = io.open(os.path.join(HERE, "t2_gate_patch.py"), encoding="utf-8").read()
OK = []


def chk(name, cond, extra=""):
    OK.append(bool(cond))
    print("  %s %s%s" % ("PASS" if cond else "FAIL", name, (" — " + str(extra)) if extra else ""))


chk("기본 OFF", 'os.environ.get("T2_DISCOVERY_STEP2") == "1"' in SRC)
chk("문구 상수가 있다", "DISCOVERY_STEP2_FB" in SRC)
m = re.search(r'DISCOVERY_STEP2_FB = \((.*?)\n\)', SRC, re.S)
body = m.group(1) if m else ""
chk("x273 이 이긴 문자열 축자", "Do not search again" in body and "must be unlocked" in body)
chk("이름·도구는 주입", "{name}" in body and "{unlock}" in body)
chk("도메인 리터럴 0",
    not re.search(r"open_bank_account|account_class|Sky Blue|checking", body))

m2 = re.search(r"def _retrieved_unlockable\(.*?\n(?=\n\ndef |\nDISC|\ndef )", SRC, re.S)
fn = m2.group(0) if m2 else ""
chk("판정 함수를 찾았다", bool(fn))
chk("이름 집합은 호출부가 준다(짓지 않는다)", "known_names" in fn and "if not known_names" in fn)
chk("회수 텍스트 실재를 본다", "in hay" in fn)
chk("이미 시도한 이름은 제외", "tried" in fn and "not in tried" in fn)
chk("못 찾으면 None(종전 경로)", "return None" in fn)
for pat, why in ((r"re\.(search|findall|match)\s*\(", "정규식"),
                 (r"\bmax\s*\(", "max("), (r"\bargmax\b", "argmax")):
    chk("도메인 해석·선택기 없음: %s" % why, not re.search(pat, fn))

chk("호출부가 레지스트리를 넘긴다", "known_names=_rz.registry_names(self)" in GATE)
chk("registry_names 는 프레임워크 출처", "get_discoverable_tools" in SRC)

try:
    import ast
    ast.parse(SRC)
    ast.parse(GATE)
    chk("파싱", True)
except Exception as e:
    chk("파싱", False, e)

print("\n%d/%d" % (sum(OK), len(OK)))
sys.exit(0 if all(OK) else 1)
