# -*- coding: utf-8 -*-
"""Does every live-run driver record what our layer said?

The 97-task sweep of 2026-08-06 ran for seven hours without `T2_FB_SIDECAR`, because the
smoke driver set it and the sweep driver did not. Nothing in the code said they had to agree,
so the difference survived until the forensic asked a question the data could not answer.
The rule is cheap to check and expensive to forget, so it is checked here rather than
remembered ([[07]]: a constraint that depends on someone noticing is not a constraint).
"""

import glob
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# 라이브 e2e를 띄우는 드라이버 = t2_launch를 부르는 셸 스크립트
drivers = []
for p in sorted(glob.glob(os.path.join(HERE, "*.sh"))):
    src = io.open(p, encoding="utf-8", errors="replace").read()
    if re.search(r"^\s*t2_launch\b", src, re.M):
        drivers.append((os.path.basename(p), src))

fails = []

# ★단일 출처 우선: 개별 드라이버가 각자 기억해서 켜는 방식이 이 사고의 뿌리였다. 그래서 판정은
#   "모든 라이브 런이 통과하는 자리(t2_launch)가 기본값을 두는가"이고, 드라이버별 지정은 선택이다.
go = io.open(os.path.join(HERE, "go_stack.sh"), encoding="utf-8", errors="replace").read()
launch = go.split("t2_launch()", 1)[-1].split(chr(10) + "}", 1)[0] if "t2_launch()" in go else ""
default_ok = "T2_FB_SIDECAR" in launch and ":=" in launch
print("  %-28s %s" % ("go_stack.sh t2_launch 기본값",
                      "PASS" if default_ok else "FAIL — 여기 없으면 드라이버마다 잊는다"))
if not default_ok:
    fails.append("t2_launch-default")

print(chr(10) + "  (참고) 드라이버별 명시 지정:")
for name, src in drivers:
    if name == "go_stack.sh":
        continue
    print("    %-28s %s" % (name, "명시" if "T2_FB_SIDECAR" in src else "기본값 상속"))
if not drivers:
    print("  (드라이버를 못 찾음 — 이 검정 자체가 무효다)")
    fails.append("no-drivers-found")

print("\n결과: %s" % ("ALL PASS" if not fails else "FAIL %d — %s" % (len(fails), fails)))
sys.exit(1 if fails else 0)
