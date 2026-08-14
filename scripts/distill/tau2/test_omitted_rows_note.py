#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FIX-14/15 검정 — 격리 서브가 원천에서 읽은 행 중 **넘기지 않은 행**의 표면화.

근거(2026-08-14 야간 t7289 074 실물): 원장 33행 → 서브가 8행만 형식화 → 도구는 8행으로 정직하게
net $6.00 산출(gold 27.00). 그런데 반환문은 `8 of 8 rows were checked (0 could not be verified)`
라서 **25행의 손실이 어디에도 보이지 않았다**(stderr `⚠MISMATCH sub=8 · source=33` 뿐).

이 검정이 고정하는 것:
  1. 누락이 있을 때만 문장이 붙고, 그 수가 정확히 source-sub 다 (FIX-14 분모 손실 가시화)
  2. 문장이 **무엇을 하면 되는지**까지 말한다 (FIX-15·[[64]])
  3. 073 형태(진짜 부분집합이 아니라 sub≥source)에서는 **아무 말도 안 한다** = 거동보존
  4. C212 `_COVERAGE_RE` 가 덧붙은 문장 뒤에서도 여전히 매치한다(엔진↔엔진 프로토콜 비파손)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t2_scaffold_get as SG                                        # noqa: E402
import t2_gate_patch as GP                                          # noqa: E402

FAIL = []


def chk(cond, msg):
    if not cond:
        FAIL.append(msg)
        print("  FAIL %s" % msg)
    else:
        print("  ok   %s" % msg)


def test_missing_surfaced():
    """074 형태: 원천 33행 · 서브 8행 → 25행 누락을 말한다."""
    txt = SG._omitted_rows_note({"source": 33, "sub": 8})
    chk("25 further row(s)" in txt, "누락 행수 = source-sub (25)")
    chk("NOT supplied" in txt, "누락 사실이 명시된다")
    chk("call again" in txt and "re-read" in txt, "고칠 방법을 이름으로 댄다([[64]])")


def test_no_note_when_complete():
    """전량 전달·073 형태(부분집합이지만 계기상 sub≥source)·미선언 = 침묵(거동보존)."""
    chk(SG._omitted_rows_note({"source": 33, "sub": 33}) == "", "전량 전달이면 무발화")
    chk(SG._omitted_rows_note({"source": 8, "sub": 8}) == "", "동수면 무발화")
    chk(SG._omitted_rows_note({"source": 0, "sub": 3}) == "",
        "source=0 은 FIX-11 소관 — 여기서 중복 발화 안 함")
    chk(SG._omitted_rows_note(None) == "", "계기 없음 = 무발화")
    chk(SG._omitted_rows_note({}) == "", "빈 dict = 무발화")
    chk(SG._omitted_rows_note({"source": "x", "sub": None}) == "", "비수치 = 무발화(예외 안 냄)")


def test_coverage_regex_intact():
    """덧붙인 문장이 C212 의 `[coverage]` 파싱을 깨지 않는다(같은 줄에 붙는다)."""
    line = ("[coverage] 8 of 8 rows were checked (0 could not be verified)."
            + SG._omitted_rows_note({"source": 33, "sub": 8}))
    m = GP._COVERAGE_RE.search(line)
    chk(m is not None, "_COVERAGE_RE 매치 유지")
    if m:
        chk((m.group(1), m.group(2), m.group(3)) == ("8", "8", "0"),
            "판정/전체/미검증 세 수 파싱 불변")


def main():
    for t in (test_missing_surfaced, test_no_note_when_complete, test_coverage_regex_intact):
        print("[%s]" % t.__name__)
        t()
    print("\n%s (%d fail)" % ("FAIL" if FAIL else "PASS", len(FAIL)))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
