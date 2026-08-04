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
_SIM = [None]


def set_sim_from(obj):
    """이 턴이 어느 sim인지 기록한다 — 로그 한 줄이 어느 태스크의 것인지 알기 위해.

    2026-08-05 P1 스모크에서 발화 3건이 **어느 sim의 것인지 말할 수 없었다**(concurrency 4로
    로그가 섞이는데 줄에 식별자가 없었다). 발화×결과 귀속(C294)이 판정의 1차 지표인데 그 귀속이
    로그 수준에서 불가능했던 것이다. orchestrator는 `task.id`를 갖고 에이전트는 `_t2_orch`로
    그것에 닿는다 — 둘 중 어느 쪽으로 불려도 찾는다. 실패하면 종전대로 무기명.
    """
    try:
        for cand in (obj, getattr(obj, "_t2_orch", None)):
            task = getattr(cand, "task", None)
            tid = getattr(task, "id", None)
            if tid:
                _SIM[0] = str(tid)
                return
    except Exception:
        pass


def beat(flag, detail=""):
    """레버 효과 지점에서 호출 — 동작의 stderr 증거. 실패해도 무해.

    캡은 **(레버, sim)별**이다. 프로세스 전체로 세면 한 sim이 캡을 소진해 나머지 sim의 발화가
    통째로 안 보인다 — 존재 증명에는 충분했지만 귀속에는 못 쓴다.
    """
    try:
        sim = _SIM[0]
        key = (flag, sim)
        n = _SEEN.get(key, 0) + 1
        _SEEN[key] = n
        if n <= _CAP:
            print("[T2_LEVER] %s%s%s%s" % (flag, (" sim=" + sim) if sim else "",
                                           (" " + detail) if detail else "",
                                           " (이후 무음)" if n == _CAP else ""),
                  file=sys.stderr, flush=True)
    except Exception:
        pass
