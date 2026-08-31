# -*- coding: utf-8 -*-
"""회귀 — 이름 없는 거부 문면이 **A2 선언을 가리킨다** (`T2_DENY_HOWTO`·[[64]]).

★결손 (2026-08-31 · x692 `task_094` 라이브 축자):
    [T2_TOOL_OBS] id=chatcmpl-tool- err=True -> Error: resolve the flagged call(s) first;
    do not call this tool yet.
  [[64]] 의 두 칸 중 **뒷칸(무엇을 하면 풀리나)이 비었다**. 그런데 그 답은 A2 에 이미 있다 —
  `relations.by_tool[t].requires`(선행 read)와 `scaffold_get_tools[].params`(인자 계약).
  실패 305 액션 중 **275(90%)가 WRONGARG** 이므로 필요한 것이 정확히 그 두 가지다.

⚠엔진은 조회·나열만 한다(선택·요약·패턴매칭 0·[[59]]/[[10]]). 선언이 없으면 **빈 문자열**이라
  OFF 와 바이트 동일이다(지어내지 않는다·C416 규율).
⚠fail-closed 불변 — 문면만 바뀌고 호출은 여전히 실행되지 않는다.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t2_gate_patch as G

os.environ["T2_DENY_HOWTO_PARAMS"] = "1"    # 인자 계약은 옵트인([[70]] 문맥 중복 회피)

A2 = {
    "relations": {"by_tool": {"do_thing": {"requires": ["read_a", "read_b"]}}},
    "scaffold_get_tools": [
        {"name": "do_thing", "description": "d",
         "params": {"components": "a JSON array you read from the KB", "_hidden": "no"},
         "requires_reads": ["read_b", "read_c"]},
    ],
}


def test_quotes_reads_and_params():
    out = G._decl_howto("do_thing", A2)
    assert "do_thing" in out
    for r in ("read_a", "read_b", "read_c"):
        assert r in out, (r, out)
    assert out.count("read_b") == 1, "중복 나열 금지: " + out       # requires ∪ requires_reads
    assert "components" in out and "a JSON array you read from the KB" in out
    assert "_hidden" not in out                                     # `_` 접두 선언은 내부용


def test_suffixed_call_resolves_to_family():
    """호출은 접미사가 붙고 선언은 base 로 적힌다 — 그 사이를 `_fam` 이 잇는다."""
    out = G._decl_howto("do_thing_4829", A2)
    assert "read_a" in out and "components" in out, out


def test_silent_when_nothing_declared():
    assert G._decl_howto("unknown_tool", A2) == ""
    assert G._decl_howto("do_thing", None) == ""
    assert G._decl_howto("", A2) == ""


def test_params_are_optin():
    """인자 계약은 주입 스키마에 이미 있으므로 기본 OFF — 선행 read 는 그래도 남는다."""
    os.environ.pop("T2_DENY_HOWTO_PARAMS", None)
    try:
        out = G._decl_howto("do_thing", A2)
        assert "components" not in out and "read_a" in out, out
    finally:
        os.environ["T2_DENY_HOWTO_PARAMS"] = "1"


def test_cap_truncates():
    out = G._decl_howto("do_thing", A2, cap=20)
    assert out.endswith("[...]") and len(out) < 200, out


if __name__ == "__main__":
    for n, f in sorted(globals().items()):
        if n.startswith("test_"):
            f(); print("ok", n)
