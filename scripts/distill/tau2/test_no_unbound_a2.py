# -*- coding: utf-8 -*-
"""**바인딩 전 `a2` 읽기 금지** 회귀 (2026-08-16).

사건 둘, 같은 뿌리:
  ⑴ 내가 넣은 `T2_DELIVER_PRECOMMIT` 블록이 `a2` 바인딩 **전**에 있었다 →
     `UnboundLocalError` → t7303 treat **12 sim 전부 infrastructure_error**(런 1회 소각).
  ⑵ 그 위의 **declfirst 가이드 주입**도 같은 자리에 있었고, 바로 아래 `except Exception: pass` 가
     예외를 삼켜 **한 번도 주입되지 않은 채** 조용히 지나갔다(같은 코드의 `patched` 경로 사본은
     살아 있어서 더 안 보였다).

⇒ 규칙: `unified()` 안에서 `a2` 는 **바인딩 줄 이후에만** 읽는다. 예외를 삼키는 블록은 그 자체로
   죽은 코드를 숨기므로, 정적 검사로 봉인한다([[55]] 0단계 = 배선 생존).
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

SRC = io.open(os.path.join(HERE, "t2_gate_patch.py"), encoding="utf-8").read().split("\n")
OK = True


def chk(cond, msg):
    global OK
    OK = OK and bool(cond)
    print("  %s %s" % ("✓" if cond else "✗", msg))


start = next(i for i, l in enumerate(SRC) if l.strip().startswith("def unified(self, message, state):"))
end = next((i for i in range(start + 1, len(SRC)) if re.match(r"    def \w+\(", SRC[i])), len(SRC))
bind = next(i for i in range(start, end) if re.match(r'\s*a2 = getattr\(self, "_t2_a2"', SRC[i]))

print("[①] unified() 안에서 바인딩 전 a2 읽기 0")
bad = []
for i in range(start, bind):
    line = SRC[i]
    if line.strip().startswith("#"):
        continue
    if re.search(r"(?<![_\w])a2(?![_\w])", line):
        bad.append((i + 1, line.strip()[:90]))
chk(not bad, "바인딩(%d행) 전 a2 참조 %d개" % (bind + 1, len(bad)))
for n, l in bad:
    print("      ✗ %d: %s" % (n, l))

print("[②] 선-배달 블록은 바인딩 **뒤**에 있다")
pre = next((i for i in range(start, end) if "T2_DELIVER_PRECOMMIT" in SRC[i]), None)
chk(pre is not None and pre > bind, "선-배달 블록 %s > 바인딩 %d" % (pre + 1 if pre else None, bind + 1))

print("[③] 죽은 가이드는 플래그로 분리됐다(조용한 부활 금지)")
seg = "\n".join(SRC[start:end])
chk("T2_DECLFIRST_GUIDE_FIX" in seg, "부활은 `T2_DECLFIRST_GUIDE_FIX=1` 로만")
chk("_df.guide_text(a2)" not in seg, "unified 경로에 바인딩-전 guide_text(a2) 호출이 없다")

print("\nRESULT: %s" % ("ALL PASS" if OK else "FAIL"))
sys.exit(0 if OK else 1)
