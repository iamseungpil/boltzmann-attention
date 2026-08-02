#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""t2_lever_beat: 레버 **동작 증명** 공용 헬퍼 (2026-08-02 · 사용자 지시
*"레버가 동작하는지 안 하는지 판별할 방법이 없나? 무태그도 레버 동작 여부 확인하게 하라"*).

배경 = wrap(`T2_FN_ISOLATE`)이 구현 완료인 채 발화 0이던 사고(x43) + 무태그 레버는 그 감사의
사각이었다. 규약:
  · **켜짐** 증명 = 러너의 env 덤프(`env | grep ^T2_`) — 여기 아님.
  · **동작** 증명 = 레버의 *효과 지점*에서 `beat("T2_FLAG")` 1줄 → stderr `[T2_LEVER] T2_FLAG`.
  · x44가 `[T2_LEVER] <플래그>`를 **이름으로 정확 매핑**해 센다(±40행 휴리스틱 불요).
  · 프로세스당 플래그별 **3회까지만 출력**(로그 스팸 방지·계수는 x44가 로그 전체에서 하므로
    존재 증명엔 1회면 충분하다). 이후는 무음 카운트.
  · print 전용 — 거동 변화 0. 신규 레버는 앞으로 태그 대신 이 헬퍼를 쓴다(표준화).
"""
import sys

_SEEN = {}
_CAP = 3


def beat(flag, detail=""):
    """레버 효과 지점에서 호출 — 동작의 stderr 증거. 실패해도 무해."""
    try:
        n = _SEEN.get(flag, 0) + 1
        _SEEN[flag] = n
        if n <= _CAP:
            print("[T2_LEVER] %s%s%s" % (flag, (" " + detail) if detail else "",
                                         " (이후 무음)" if n == _CAP else ""),
                  file=sys.stderr, flush=True)
    except Exception:
        pass
