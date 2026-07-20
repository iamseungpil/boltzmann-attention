# -*- coding: utf-8 -*-
"""exec 결과 재조립 크래시 픽스 검증 (2026-07-20·023/031 infrastructure_error 근본).

`_reassemble`은 tool_calls와 1:1·같은 순서를 보장해야 한다 — 안 그러면 full-duplex tick의
call↔result 쌍이 깨져 eval replay가 "Tool call id mismatch" 크래시(비결정론=orig_exec 순서 의존).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t2_scaffold_get as sg  # noqa: E402


class TC:
    def __init__(self, id, name="t", requestor="assistant"):
        self.id, self.name, self.requestor = id, name, requestor


class TM:
    def __init__(self, id=None, role="tool", requestor="assistant", error=False, content=""):
        self.id, self.role, self.requestor, self.error, self.content = id, role, requestor, error, content

    def __repr__(self):
        return "TM(%s%s)" % (self.id, "!" if self.error else "")


PASS, FAIL = [], []


def check(label, cond):
    (PASS if cond else FAIL).append(label)
    print(("  PASS " if cond else "  FAIL ") + label)


def ids(out):
    return [m.id for m in out]


def main():
    # ── 정상 1:1 (rest만·순서 유지) ──
    tcs = [TC("a"), TC("b"), TC("c")]
    out = sg._reassemble(tcs, {}, [TM("a"), TM("b"), TM("c")], TM)
    check("정상: 1:1 순서 유지", ids(out) == ["a", "b", "c"] and len(out) == 3)

    # ── ★핵심: orig_exec가 순서 뒤바꿈 → id로 재정렬(크래시 근본) ──
    out = sg._reassemble(tcs, {}, [TM("c"), TM("a"), TM("b")], TM)
    check("reorder: id 매칭으로 tc 순서 복원", ids(out) == ["a", "b", "c"])

    # ── ours + rest 혼합 (ours=우리가 답한 tc·id 보존) ──
    ours = {id(tcs[1]): TM("b", content="OURS")}
    out = sg._reassemble(tcs, ours, [TM("a"), TM("c")], TM)
    check("혼합: ours 보존 + rest id매칭", ids(out) == ["a", "b", "c"]
          and out[1].content == "OURS")

    # ── ★결과 부족 → 드롭 금지·에러 ToolMessage로 채움(1:1 보장) ──
    out = sg._reassemble(tcs, {}, [TM("a")], TM)
    check("부족: len==len(tool_calls) 유지(드롭 0)", len(out) == 3)
    check("부족: 누락 tc는 에러메시지+tc.id", out[1].id == "b" and out[1].error
          and out[2].id == "c" and out[2].error)

    # ── id 없는 백엔드(하위호환) → 위치 폴백 ──
    out = sg._reassemble(tcs, {}, [TM(None), TM(None), TM(None)], TM)
    check("id없음: 위치 폴백 3개", len(out) == 3)

    # ── 전부 ours (rest 없음) ──
    ours = {id(t): TM(t.id) for t in tcs}
    out = sg._reassemble(tcs, ours, [], TM)
    check("전부 ours: 1:1", ids(out) == ["a", "b", "c"])

    # ── 빈 입력 ──
    check("빈 tool_calls", sg._reassemble([], {}, [], TM) == [])

    # ── requestor 미러링(누락 채움 시) ──
    tcs2 = [TC("x", requestor="user")]
    out = sg._reassemble(tcs2, {}, [], TM)
    check("누락 채움: requestor 미러링", out[0].requestor == "user" and out[0].id == "x")

    print("\n== 결과: %d PASS / %d FAIL ==" % (len(PASS), len(FAIL)))
    if FAIL:
        for f in FAIL:
            print("  - FAILED: " + f)
        sys.exit(1)
    print("ALL PASS — 재조립이 tool_calls와 1:1·순서 보장(reorder/부족서도) = tick 쌍 붕괴 차단.")


if __name__ == "__main__":
    main()
