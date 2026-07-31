# -*- coding: utf-8 -*-
"""레버가 **설치되는 경로**에 있는가 (2026-07-31 — V7 사망 사고의 재발 방지).

사고: V7(`T2_TOOL_SIGNATURE`)을 `gated`(=`BaseOrchestrator._execute_tool_calls`)에 구현했는데,
`go_stack`은 `T2_GATE_REGEN=1`이라 런처가 `_unified` 분기를 타서 **`t2_gate_patch.apply()`를 아예
호출하지 않는다**(`t2_run_gated.py:196`). 실행 훅은 `exec_augment`("deny 없음")가 차지한다.
⇒ V7은 selftest 4/4를 통과하면서도 **어떤 런에서도 발화할 수 없었다**. Z4·Z5·Y2에서 deny 0.

교훈: **selftest 통과 ≠ 라이브 도달.** 레버는 *설치되는 함수* 안에 있어야 한다.

이 테스트가 강제하는 것: go_stack이 켜는 on/off 레버 중 `t2_gate_patch.py`가 구현하는 것은
**`unified`(설치되는 생성-레벨 경로) 안에서 참조**되어야 한다. `gated`에만 있으면 실패한다.
(`gated`는 `T2_GATE_REGEN` 미사용 스택에서만 설치되므로 거기에만 있는 레버 = 사실상 죽은 코드.)
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
GO = io.open(os.path.join(HERE, "go_stack.sh"), encoding="utf-8").read()


def body_of(func):
    """`def <func>(` 부터 같은 들여쓰기의 다음 def 전까지."""
    m = re.search(r"^(\s*)def %s\(" % re.escape(func), SRC, re.M)
    if not m:
        return ""
    ind, start = m.group(1), m.end()
    nxt = re.search(r"^%sdef " % ind, SRC[start:], re.M)
    return SRC[start:start + (nxt.start() if nxt else len(SRC) - start)]


UNIFIED = body_of("unified")
GATED = body_of("gated")
ON = set(re.findall(r"\b(T2_[A-Z0-9_]+)=1\b", GO))
IMPLEMENTED = set(re.findall(r"environ\.get\(\s*[\'\"](T2_[A-Z0-9_]+)[\'\"]", SRC))

# `gated`에만 있어도 정당한 것: 실행-레벨 read-augment 계열(deny가 아니라 응답 가공)
EXEC_LEVEL_OK = {"T2_PRESENT_READS", "T2_PRESENT_NESTED", "T2_CALC", "T2_GATE_KINDS",
                 "T2_WRITE_CAP", "T2_WRITE_CAP_K", "T2_RETRY_CONTROLLER", "T2_RETRY_K",
                 "T2_PROVENANCE", "T2_AUTOFETCH"}

OK = True
print("[경로 가드] go_stack ON 레버가 **설치되는 경로**(unified)에 있는가")
print("  unified 본문 %d자 · gated 본문 %d자" % (len(UNIFIED), len(GATED)))
dead = []
for f in sorted(ON & IMPLEMENTED):
    if f in EXEC_LEVEL_OK:
        continue
    in_u, in_g = (f in UNIFIED), (f in GATED)
    if in_g and not in_u:
        dead.append(f)
        OK = False
    print("  %-26s unified=%-5s gated=%-5s %s"
          % (f, in_u, in_g, "★죽은 경로(gated 전용)" if (in_g and not in_u) else ""))

if dead:
    print("\n✗ FAIL — 설치되지 않는 경로에만 있는 레버: %s" % ", ".join(dead))
    print("  (selftest는 통과하지만 라이브에서 절대 발화하지 않는다 — V7 사고와 동형)")
else:
    print("\n✓ PASS — gated 전용 레버 없음")

print("RESULT: %s" % ("ALL PASS" if OK else "FAIL"))
sys.exit(0 if OK else 1)
