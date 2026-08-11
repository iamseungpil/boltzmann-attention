# -*- coding: utf-8 -*-
"""회귀 — 결정-선행 write (T2_DECIDE_BEFORE_WRITE · 오프라인 · 모델 0 · env 0).

규칙 한 줄(사용자 지시 2026-08-12): **이 대화에 결정 재료가 없으면 write 를 1턴 미루고,
그 자리서 서브를 돌려 재료를 담아 돌려준다.** 검정할 것은 경계다 —

  ⑴ 기본 OFF · ⑵ write 강제 아님(지연 1회·cap) · ⑶ 서브 침묵이면 통과(막다른 골목 금지)
  ⑷ 엔진이 고르지 않는다(선택기 0) · ⑸ [[64]] — 왜 막혔고 무엇이 답인지 담는다
  ⑹ 배타 체인·_SRC8 에 등록됐다(계측이 본다)
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


m = re.search(r"dw_fb = None\n(.*?)_FB_GENERIC = ", SRC, re.S)
chk("블록을 찾았다", m is not None)
body = m.group(1) if m else ""
code = "\n".join(l for l in body.split("\n") if not l.lstrip().startswith("#"))

chk("기본 OFF", 'os.environ.get("T2_DECIDE_BEFORE_WRITE") == "1"' in code)
chk("sim 당 1회 cap", "_t2_dwrite_deny" in code)
chk("서브 침묵 = 통과 (if _dmat 안에서만 deny)", re.search(r"if _dmat:", code) is not None)
chk("재료는 서브가 만든다(_search_material)", "_search_material" in code)
for pat, why in ((r"\bargmax\b", "argmax"), (r"\bmax\s*\(", "max("),
                 (r"\bsorted\s*\([^)]*\)\s*\[", "sorted[...]")):
    chk("선택기 없음: %s" % why, not re.search(pat, code))
chk("write 집합은 A2 도출", "_confirm_write_tools(a2)" in code)
chk("[[64]] — 답을 담아 돌려준다", "_dmat +" in code or "+ _dmat" in code)
chk("다른 소스 침묵일 때만 (겹침 회피)", "pr_fb is None" in code and "wev_fb is None" in code)
chk("_SRC8 등록", '("decide_write", dw_fb)' in SRC)
chk("배타 체인 elif 등록",
    re.search(r"elif dw_fb is not None and c is dw_fb\[0\]", SRC) is not None)

try:
    import ast
    ast.parse(SRC)
    chk("파싱", True)
except Exception as e:
    chk("파싱", False, e)

print("\n%d/%d" % (sum(OK), len(OK)))
sys.exit(0 if all(OK) else 1)
