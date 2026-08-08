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
import threading

_SEEN = {}
_CAP = 3
# ★★전역이 아니라 **thread-local**이다 (2026-08-08·C325). 구판은 `_SIM = [None]` 리스트
#   하나를 프로세스 전체가 공유했는데, 러너는 sim을 `ThreadPoolExecutor`로 **동시에** 돌린다
#   (`tau2/runner/batch.py`: `ThreadPoolExecutor(max_workers=config.max_concurrency)`).
#   ⇒ 마지막으로 `set_sim_from`을 부른 스레드의 값이 **다른 sim의 줄에 찍힌다.**
#   실측(2 sim 동시 런): beat 3줄이 전부 한 sim의 이름을 달았는데, 레코드마다 sim을 지니는
#   사이드카로 대조하니 그 문장은 **반대쪽 sim**의 것이었다. 그 잘못된 태그를 근거로 원장
#   등재까지 갔다가 되돌렸다 — 태그는 *있다*는 것보다 *맞다*는 것이 중요하다([[25]]).
_LOCAL = threading.local()


def current_sim():
    """이 스레드가 지금 돌리고 있는 sim id(모르면 None). 태그를 붙이는 모든 자리의 단일 출처."""
    return getattr(_LOCAL, "sim", None)


def set_sim_from(obj):
    """이 턴이 어느 sim인지 기록한다 — 로그 한 줄이 어느 태스크의 것인지 알기 위해.

    2026-08-05 P1 스모크에서 발화 3건이 **어느 sim의 것인지 말할 수 없었다**(concurrency 4로
    로그가 섞이는데 줄에 식별자가 없었다). 발화×결과 귀속(C294)이 판정의 1차 지표인데 그 귀속이
    로그 수준에서 불가능했던 것이다. orchestrator는 `task.id`를 갖고 에이전트는 `_t2_orch`로
    그것에 닿는다 — 둘 중 어느 쪽으로 불려도 찾는다. 실패하면 종전대로 무기명.

    한 sim = 한 워커 스레드이므로 여기서 심은 값은 그 sim의 모든 호출에서 그대로 보인다.
    """
    try:
        for cand in (obj, getattr(obj, "_t2_orch", None)):
            task = getattr(cand, "task", None)
            tid = getattr(task, "id", None)
            if tid:
                _LOCAL.sim = str(tid)
                return
    except Exception:
        pass


class _TaggingStderr(object):
    """stderr의 **모든 줄** 앞에 `[sim=…]`를 붙인다(모르면 종전 그대로).

    왜 줄마다인가: 지금까지 sim을 다는 줄은 `[T2_LEVER]` 하나뿐이었고, 나머지 수백 줄
    (`[T2_ARBITRATE]`·`[T2_RESOLVE]`·`[T2_LEDGER]`·`[T2_PREKB]`…)은 무기명이었다. 동시 런의
    로그는 인터리브라서, 무기명 줄은 **어느 sim의 것인지 원리적으로 복원할 수 없다** — 실제로
    포렌식에서 한 sim의 침묵을 다른 sim의 발화로 읽을 뻔했다. 귀속을 사이드카에만 의존하면
    사이드카가 없는 런은 포렌식 자체가 불가능하다([[08]]).

    ★프리픽스인 이유: 기존 로그 파서는 `search()`로 찾고 일부는 **행말**에 앵커를 건다
      (`x134`: `suppressed=(\\[.*?\\])\\s*$`). 접미사는 그것을 깨고 접두사는 깨지 않는다.
    ★관측 전용이다 — 문자열만 바뀌고 거동은 그대로다(비커밋·모델에 안 보인다).
    """

    def __init__(self, inner):
        self._inner = inner
        self._part = {}                      # thread ident -> 개행 전 잔여

    def write(self, s):
        try:
            if not s:
                return 0
            tid = threading.get_ident()
            buf = self._part.pop(tid, "") + s
            lines = buf.split("\n")
            self._part[tid] = lines.pop()    # 마지막 조각은 아직 줄이 아니다
            if not lines:
                return len(s)
            sim = current_sim()
            pre = ("[sim=%s] " % sim) if sim else ""
            self._inner.write("".join((pre + ln + "\n") if ln else "\n" for ln in lines))
            return len(s)
        except Exception:
            try:
                return self._inner.write(s)
            except Exception:
                return 0

    def flush(self):
        try:
            self._inner.flush()
        except Exception:
            pass

    def __getattr__(self, name):
        return getattr(self._inner, name)


def install_stderr_tagger():
    """한 번만 감싼다. 라이브 런이 **전부 통과하는 한 자리**에서 부른다(드라이버 기억 금지·[[07]])."""
    if isinstance(sys.stderr, _TaggingStderr):
        return False
    sys.stderr = _TaggingStderr(sys.stderr)
    return True


def beat(flag, detail="", orch=None, target=None, fact=None, order=None):
    """레버 효과 지점에서 호출 — 동작의 stderr 증거. 실패해도 무해.

    캡은 **(레버, sim)별**이다. 프로세스 전체로 세면 한 sim이 캡을 소진해 나머지 sim의 발화가
    통째로 안 보인다 — 존재 증명에는 충분했지만 귀속에는 못 쓴다.

    ★등록점 승격 (2026-08-07·정본 §5.5 `register()` 배선의 실제 형태).
    `register()`를 55개 호출부에 따로 넣는 대신 **여기**에 둔다. 이유는 귀속이다: 어느 레버가
    말했는지를 아는 유일한 자리는 그 레버 자신이고, 바깥에서 `*_fb` 변수명으로 플래그를
    되짚는 것은 추측이다(실제로 해 보니 `ep_fb` 위 6줄에 `T2_EPLAN_DENY_CAP`이 걸려 있어
    엉뚱한 이름이 나왔다). 레버가 스스로 이름을 대는 지점이 곧 등록 지점이다.

    `orch`를 주면 스택에 등록한다. 안 주면 종전대로 stderr 한 줄뿐 — **기본 거동 불변**이라
    이미 있는 17개 호출부는 손대지 않아도 된다.
    """
    try:
        sim = current_sim()
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
    if orch is None:
        return
    try:
        import t2_stack as _stk
        _stk.register(orch, flag=flag, target=target, fact=fact, order=order)
    except Exception:
        pass
