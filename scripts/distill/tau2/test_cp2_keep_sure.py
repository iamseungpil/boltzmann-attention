# -*- coding: utf-8 -*-
"""`T2_CP2_KEEP_SURE` — 확실한 배달을 불확실한 배달과 맞바꾸지 않는다 (2026-08-24).

실물 재생: 057#s373753(t7348) 의 배달 순서 247 → 247 → 87,407 자.
현행(플래그 OFF)은 clobber·clobber 로 앞의 둘을 죽이고, 남은 87,407 자는 소비 지점의
`_ctx_fits` 가드가 `ctx_skip` 으로 버린다 ⇒ **세 배달물 전부 소실**.
플래그 ON 이면 마지막 대형이 소형을 죽이지 못한다.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t2_gate_patch as G                                   # noqa: E402

FAIL = []


def chk(c, msg):
    print(("  OK   " if c else "  FAIL ") + msg)
    if not c:
        FAIL.append(msg)


class _S(object):
    def __init__(self):
        self._t2_cp2_pending = None


def run(seq, keep_sure):
    os.environ["T2_CP2_KEEP_SURE"] = "1" if keep_sure else "0"
    os.environ["T2_CP2_QUEUE"] = "0"
    s = _S()
    for txt, tag in seq:
        G._cp2_assign(s, txt, tag)
    return s._t2_cp2_pending


SMALL_A = "A" * 247
SMALL_B = "B" * 247
HUGE = "H" * 87407
SEQ = [(SMALL_A, "SEARCH_ON_PROCEED"), (SMALL_B, "SEARCH_ON_PROCEED"), (HUGE, "SEARCH_ON_PROCEED")]

print("[1] 현행(OFF) — 실물과 같이 대형만 남는다(그리고 소비 지점서 ctx_skip 된다)")
off = run(SEQ, False)
chk(off == HUGE, "OFF: 남은 것은 87,407자 대형 (실측 재현)")
chk(len(off) >= G._CP2_GUARD_MIN, "OFF: 남은 것이 가드 검사 대상 = 배달이 불확실하다")

print("[2] ON — 확실한 소형이 살아남는다")
on = run(SEQ, True)
chk(on == SMALL_B, "ON: 남은 것은 마지막 소형 247자")
chk(len(on) < G._CP2_GUARD_MIN, "ON: 남은 것이 가드 문턱 아래 = 반드시 배달된다")

print("[3] 부작용 경계 — 소형↔소형·대형↔대형은 종전 그대로")
chk(run([(SMALL_A, "t"), (SMALL_B, "t")], True) == SMALL_B, "소형→소형: 종전대로 덮어씀")
# 구판 `_big` 구제(>=10,000자 미소비)는 **이어붙인다** - 내 가드는 `not _big` 이라 여길 안 건드린다.
_bigres = run([(HUGE, "t"), ("X" * 6000, "t")], True)
_want = HUGE + chr(10) + chr(10) + "X" * 6000
chk(_bigres == _want, "대형->대형: 구판 `_big` 구제대로 이어붙임(내 가드가 안 건드린다)")
chk(run([(SMALL_A, "t"), (SMALL_A, "t")], True) == SMALL_A, "같은 내용 재대입: 무변화")

print("")
print("test_cp2_keep_sure " + ("PASS" if not FAIL else "FAIL %d" % len(FAIL)))
sys.exit(1 if FAIL else 0)
