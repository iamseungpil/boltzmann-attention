# -*- coding: utf-8 -*-
"""`T2_PENDING_DISCOVERED` **제거** 계약 검정 (수리 항목 R8-pending-disc-dead·2026-08-23).

## 무엇이 바뀌었나
옛 판(2026-08-18)은 *"레버가 **있는가**"* 를 지켰다. 그 레버는 이제 **없다** — 455 로그 전수에서
`[T2_PENDING_DISC]` 0줄이고, 켰더라도 ⑵종료 술어 부재 ⑶정책 모순 ⑷reward 헤드룸 0 이라
해가 이익보다 크다는 것이 닫힌 계수로 확정됐다(근거 전문은 `t2_gate_patch.py` 의 묘비 주석).
이 파일은 그래서 **반대 방향**을 지킨다: 죽은 레버가 조용히 되살아나지 않고, 그 옆의 살아 있는
배선은 한 글자도 안 바뀌었다는 것.

## 검정 구조 (양성대조·부정통제 짝)
  · **제거 확인**  — 주석을 걷어낸 *실행 텍스트*에 `T2_PENDING_DISCOVERED` · `[T2_PENDING_DISC]` 0회
  · **부정통제**   — 같은 걷어내기 뒤에도 이웃 플래그(`T2_WINDOW` · `T2_ARBITRATE` ·
                     `T2_TOOL_SIGNATURE`)와 `[T2_ACTIONREQ]` 는 **여전히 보인다**
                     ⇒ "0회"가 탐지자가 죽어서 나온 값이 아니다(무내용 통과 방지·[[57]]).
  · **기록 보존**  — 원문(주석 포함)에는 묘비가 **남아 있다** ⇒ 지운 이유가 함께 지워지지 않았다([[64]]).
  · **거동 보존**  — `_uacts` 도출 · `_upending = _uacts - _effall` · `[T2_ACTIONREQ]` 인쇄가 축자 동일.
  · **고아 0**     — 지운 블록이 유일 소비자였던 `sub_tool_names` 호출이 남아 있지 않고,
                     반대로 여러 곳이 쓰는 `_user_discoverable(` 은 살아 있다.

실행: `PYTHONIOENCODING=utf-8 py -3 test_pending_discovered.py`
"""
import io
import os
import sys
import tokenize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ★양성대조 경로(사용자 인자): 수리 **전** 사본을 주면 이 검정이 FAIL 해야 한다.
#   `py -3 test_pending_discovered.py <수리전_t2_gate_patch.py>` — 무내용 통과 방지([[57]]).
TARGET = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 \
    else os.path.join(HERE, "t2_gate_patch.py")
RAW = io.open(TARGET, encoding="utf-8").read()


def code_only(path):
    """주석만 걷어낸 **실행 텍스트**(토큰 1개당 1줄). 문자열 리터럴은 남긴다 —
    지운 블록의 `print("[T2_PENDING_DISC] ...")` 가 리터럴이었기 때문.
    ⚠토큰을 줄바꿈으로 잇는다 ⇒ **여러 토큰에 걸친 조각**(`foo(`)은 여기서 찾지 마라.
      이름은 식별자 토큰 그대로, 문면은 문자열 토큰 그대로 검사한다."""
    out = []
    with io.open(path, encoding="utf-8") as fh:
        for tok in tokenize.generate_tokens(fh.readline):
            if tok.type == tokenize.COMMENT:
                continue
            out.append(tok.string)
    return "\n".join(out)


CODE = code_only(TARGET)

_bad = 0


def chk(ok, why):
    global _bad
    print("  %s %s" % ("O" if ok else "X", why))
    if not ok:
        _bad += 1
    return ok


def main():
    print("[1] 제거 확인 — 실행 텍스트에 죽은 레버가 없다")
    chk("T2_PENDING_DISCOVERED" not in CODE,
        "환경 플래그 `T2_PENDING_DISCOVERED` 를 읽는 코드 0회")
    chk("[T2_PENDING_DISC]" not in CODE,
        "로그 마커 `[T2_PENDING_DISC]` 를 찍는 코드 0회(error no-op 포함)")

    print("[2] 부정통제 — 같은 탐지자로 이웃 배선은 **보인다**(탐지자 무내용 아님)")
    for name in ('"T2_WINDOW"', '"T2_ARBITRATE"', '"T2_TOOL_SIGNATURE"'):
        chk(name in CODE, "이웃 플래그 %s 는 실행 텍스트에 실재" % name)
    chk("[T2_ACTIONREQ] window=open pending_user=%s " in CODE,
        "숙주 계기 `[T2_ACTIONREQ]` 인쇄는 실재")
    chk(len(CODE) > 200000, "걷어낸 텍스트가 통째로 비지 않았다 (%d자)" % len(CODE))

    print("[3] 기록 보존 — 원문에는 묘비가 남아 있다([[64]] 이유를 지우지 않는다)")
    chk("T2_PENDING_DISCOVERED" in RAW, "원문(주석 포함)에 플래그 이름이 남아 있다")
    chk("R8-pending-disc-dead" in RAW, "수리 항목 id 가 묘비에 박혀 있다")
    for frag in ("refute_6.json", "12,323", "give_discoverable_user_tool"):
        chk(frag in RAW, "묘비가 근거 %r 를 축자로 담는다" % frag)

    print("[4] 거동 보존 — 이웃 배선 축자 동일(플래그 OFF 였으므로 라이브 델타 0)")
    for frag in ('_uacts = {t for t in ((a2 or {}).get("action_tools") or [])',
                 'if _exec_side(t) == "user"}',
                 "_upending = sorted(_uacts - _effall)",
                 '_effall = {_eff_tool_name(tc) for m2 in state.messages',
                 'print("[T2_ACTIONREQ] window=open pending_user=%s "'):
        chk(frag in RAW, "축자 보존: %s" % frag)

    print("[5] 고아 0 — 지운 블록의 전유 소비자만 사라졌다")
    _names = CODE.split("\n")
    chk("sub_tool_names" not in _names,
        "식별자 `sub_tool_names` 잔재 0(이 블록이 유일 호출자였다)")
    chk("t2_search" not in _names or "_ts9" not in _names,
        "지운 블록의 지역 별칭 `_ts9` 잔재 0")
    chk("_user_discoverable" in _names,
        "공용 헬퍼 `_user_discoverable` 은 다른 소비자가 있어 살아 있다")
    chk(_names.count("_user_discoverable") >= 5,
        "공용 헬퍼 등장 >=5곳 (%d) — 정의 1 + 소비자 다수"
        % _names.count("_user_discoverable"))

    print("[6] 문법 — 제거 후에도 컴파일된다")
    try:
        compile(RAW, TARGET, "exec")
        chk(True, "compile OK")
    except SyntaxError as e:
        chk(False, "compile FAIL: %r" % (e,))

    print("\n%s" % ("test_pending_discovered PASS"
                    if not _bad else "test_pending_discovered FAIL %d건" % _bad))
    return 1 if _bad else 0


if __name__ == "__main__":
    sys.exit(main())
