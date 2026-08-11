# -*- coding: utf-8 -*-
"""회귀 검정: **R9 접힘이 본문을 지우지 않는다** (`T2_KEEP_DENY_BODY`·2026-08-11·C414).

무엇을 막는 검정인가 —
 ⒜ **끈 상태에서 거동이 바뀌는 것**. 기본 OFF 이고 OFF 면 종전대로 일반 문구로 내려간다.
 ⒝ **deny 가 사라지는 것**. 켜도 도구는 여전히 막힌다 — 바뀌는 것은 **본문뿐**이다(fail-closed).
 ⒞ **접힘 자체가 없어지는 것**. `admit()` 는 그대로 부르고 마크도 그대로 남는다(계기 보존).
 ⒟ **일반 문구가 코드에서 사라지는 것**. OFF 경로가 쓰는 문자열은 그대로 있어야 한다.

오프라인 전용(LLM·서버 불요). 실행: py -3 test_keep_deny_body.py
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


SRC = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "t2_gate_patch.py"),
           encoding="utf-8").read()

print("\n§1 플래그 · 기본 OFF")
chk(os.environ.get("T2_KEEP_DENY_BODY") != "1", "환경에 기본 ON 이 박혀 있지 않다")
chk('os.environ.get("T2_KEEP_DENY_BODY") == "1"' in SRC, "플래그 뒤에 있다")

print("\n§2 OFF 경로가 그대로 남아 있다 (거동 보존)")
chk('_FB_GENERIC = "Error: resolve the flagged call(s) first; do not call this tool yet."' in SRC,
    "일반 문구가 코드에 그대로 있다")
chk(re.search(r"if not _keep9:\s*\n\s*content = _FB_GENERIC", SRC) is not None,
    "OFF 일 때만 본문을 일반 문구로 갈아 끼운다")

print("\n§3 deny 는 fail-closed — 켜도 도구는 막힌다")
i = SRC.find("_keep9 = os.environ.get")
seg = SRC[i:i + 1400]
chk("ToolMessage(id=c.id, role=\"tool\", requestor=\"assistant\"," in seg and "error=True" in seg,
    "같은 블록이 여전히 error=True 툴 메시지를 만든다")
chk("content = _FB_GENERIC" in seg, "OFF 분기가 그 안에 있다")

print("\n§4 접힘·계기는 보존된다")
chk("_stk8.admit(self, _fbtag.get(id(c), \"fb\"), content)" in SRC, "`admit()` 호출이 그대로다")
chk("window folded fb tag=" in SRC and "kept (R9)" in SRC,
    "마크가 남고, 켠 상태를 구별해 인쇄한다")

print("\n§5 형제 대기 문구도 이름을 댄다 (C416 · x247 감사가 찾은 네 자리)")
import t2_gate_patch as GP                                        # noqa: E402


class _C(object):
    def __init__(self, name):
        self.name = name


os.environ.pop("T2_KEEP_DENY_BODY", None)
off = GP._sibling_wait("POLICY GATE", _C("submit_referral"), "the policy gate")
chk("resolve the flagged call first" in off, "OFF 면 종전 문구 그대로")
os.environ["T2_KEEP_DENY_BODY"] = "1"
on = GP._sibling_wait("POLICY GATE", _C("submit_referral"), "the policy gate")
chk("submit_referral" in on, "ON 이면 **문제된 호출 이름**을 댄다")
chk("re-issue this call" in on, "ON 이면 **다음 한 수**를 준다")
chk("resolve the flagged call first" in GP._sibling_wait("POLICY GATE", None, "x"),
    "이름을 모르면 종전 문구로 떨어진다(지어내지 않는다)")
os.environ.pop("T2_KEEP_DENY_BODY", None)
chk(SRC.count("_sibling_wait(") >= 5, "네 자리 + 정의가 한 헬퍼를 쓴다(두 벌 금지)")

print("\n%s  (%d/%d)" % ("FAIL" if FAILED else "ALL PASS", 13 - len(FAILED), 13))
sys.exit(1 if FAILED else 0)
