# -*- coding: utf-8 -*-
r"""`_resolve_cap_ok` **런타임** 회귀 — 리셋 *대입이 실제로 실행되는가* (2026-08-18·C538).

## 무엇을 막는가 (실제로 일어난 사고)

`a627a18b`("Adds stderr markers only; no behavior change")가 리셋 블록에서 **print 를 대입 앞에**
놓았다. 그 print 는 `_sys`(이 모듈에서는 **함수 안(:5377)에서만** 정의)를 쓰므로 모듈-레벨 함수
`_resolve_cap_ok` 에서는 `NameError` 다. 바깥 `except Exception: pass` 가 그것을 삼키면서
**리셋 대입까지 통째로 건너뛰어졌다** ⇒ 상한이 3회 deny 후 **영구 래치**가 됐다.

기존 검정 둘 다 이것을 못 봤다:
  · `test_resolve_cap_marker.py` — **소스 문자열**만 본다(마커가 코드에 있는가).
  · `test_no_undefined_names.py` — 임포트를 **모듈 전체에서 평평하게** 모아서, 함수 안의
    `import sys as _sys` 가 모듈 전역처럼 보였다(스코프를 안 본다).

⇒ 이 검정은 **함수를 실제로 부르고 상태를 확인한다**. 문자열이 아니라 *대입*이 계약이다.

## 불변식

  ① 진행(새 실행 도구)이 있으면 `_t2_resolve_deny` 가 **0 이 되고** 반환이 True 다.
  ② 그때 마커가 **실제로 stderr 에 찍힌다**(인쇄가 예외를 던지지 않는다).
  ③ 진행이 없으면 리셋도 마커도 없다(반환 False·정체 유지).
  ④ 스냅샷이 없으면(`_t2_resolve_done` 미설정) 리셋하지 않는다 — 첫 발화 전엔 사건이 아니다.

오프라인 전용(LLM·서버 불요). 실행: py -3 test_resolve_cap_runtime.py
"""
import contextlib
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_gate_patch as GP                                   # noqa: E402

FAILED = []


def chk(c, label):
    print(("  OK   " if c else "  FAIL ") + label)
    if not c:
        FAILED.append(label)


class TC(object):
    def __init__(self, name, cid):
        self.name = name
        self.arguments = "{}"
        self.id = cid


class M(object):
    def __init__(self, role, content="", tool_calls=(), mid=None, error=False):
        self.role = role
        self.content = content
        self.tool_calls = list(tool_calls)
        self.id = mid
        self.tool_call_id = mid
        self.error = error


class Agent(object):
    pass


def convo(names):
    """이름마다 (assistant 호출 + 성공한 tool 결과) 한 쌍."""
    out = []
    for i, n in enumerate(names):
        cid = "c%d" % i
        out.append(M("assistant", "", [TC(n, cid)]))
        out.append(M("tool", "done", (), cid))
    return out


def call(self, msgs):
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        ok = GP._resolve_cap_ok(self, msgs, {})
    return ok, buf.getvalue()


def main():
    print("test_resolve_cap_runtime — 리셋 대입이 실행되는가(C538)")

    base = convo(["get_a", "get_b"])
    snap = GP._executed_tool_names(base, {})
    chk(snap == {"get_a", "get_b"}, "① 준비: 실행 집합을 읽는다 (%s)" % sorted(snap))

    # ── 진행 있음: 새 도구가 하나 늘었다
    self = Agent()
    self._t2_resolve_deny = 3
    self._t2_resolve_done = set(snap)
    grown = base + convo(["get_c"])
    ok, err = call(self, grown)
    chk(self._t2_resolve_deny == 0, "② 진행 있으면 **대입이 실행된다**(deny=%s)"
        % self._t2_resolve_deny)
    chk(ok is True, "③ 그때 반환이 True 다(상한 해제)")
    chk("[T2_RESOLVE_CAP]" in err and "get_c" in err,
        "④ 마커가 실제로 stderr 에 찍힌다(예외 없이): %r" % err.strip()[:70])

    # ── 진행 없음: 같은 도구를 또 불러도 집합이 안 는다
    self2 = Agent()
    self2._t2_resolve_deny = 3
    self2._t2_resolve_done = set(snap)
    same = base + convo(["get_a"])
    ok2, err2 = call(self2, same)
    chk(self2._t2_resolve_deny == 3, "⑤ 진행 없으면 정체 유지(deny=%s)" % self2._t2_resolve_deny)
    chk(ok2 is False, "⑥ 그때 반환은 False")
    chk("[T2_RESOLVE_CAP]" not in err2, "⑦ 그때 마커도 없다(실효 리셋일 때만 찍는다)")

    # ── 스냅샷 없음: 첫 발화 전에는 리셋하지 않는다
    self3 = Agent()
    self3._t2_resolve_deny = 3
    ok3, _e3 = call(self3, grown)
    chk(self3._t2_resolve_deny == 3 and ok3 is False, "⑧ 스냅샷 없으면 리셋 안 함")

    # ── 회귀 자체를 다시 심으면 잡히는가(양성 대조·소스 순서)
    import inspect
    src = inspect.getsource(GP._resolve_cap_ok)
    body = [l.strip() for l in src.split("\n")
            if "_t2_resolve_deny = 0" in l or "print(" in l]
    first_assign = next((i for i, l in enumerate(body) if "= 0" in l), None)
    first_print = next((i for i, l in enumerate(body) if l.startswith("print(")), None)
    chk(first_assign is not None and (first_print is None or first_assign < first_print),
        "⑨ 소스 순서: **대입이 print 보다 앞**(관측이 기능을 죽이지 못한다)")
    chk("_sys." not in src, "⑩ 이 함수는 `_sys`(함수-지역 별칭)를 쓰지 않는다")

    print("")
    print("RESULT: %s" % ("ALL PASS" if not FAILED else "FAIL %d" % len(FAILED)))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
