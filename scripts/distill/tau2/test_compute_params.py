# -*- coding: utf-8 -*-
"""compute op 인터프리터 유닛 + **값 산출 기구 폐기 계약** 검정 (2026-08-19 갱신).

## 무엇이 바뀌었나
전판은 A2 에 실린 `compute_ops` 스펙(책임한도 표·이자 차액 식)이 *정책대로 계산하는지* 를 봤다.
그 두 op 는 **삭제**됐다 — 엔진이 채점되는 gold 인자를 채우고 있었고([[62]]), 그 상수 하나는
gold 재현율로 골라졌다([[23]]·`bank_rule_fit.py` 가 `reward_info.action_checks` 를 훑었다).
⇒ 이 파일은 이제 두 가지를 지킨다:
  ①**일반 op 인터프리터**(`t2_compute.apply_op`)는 그대로 동작한다 — 스펙은 **인라인**으로 준다
    (A2 에서 읽지 않는다. 읽으면 삭제된 상수를 다시 불러오는 통로가 된다).
  ②**A2 는 값 산출 op 를 싣지 않는다** — 회귀 방지. 다시 실리면 여기서 깨진다.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t2_compute import apply_op

A2_PATHS = [os.path.join(os.path.dirname(__file__), "a2", n) for n in
            ("banking_knowledge.gate.json", "banking_knowledge.specific.json")]


def run(spec, params, now="11/14/2025"):
    return apply_op(spec, {"params": params, "now": now})


def test_lookup_table_interpreter():
    """`lookup_table`·`days_between` 해석기 자체(도메인 무관·인라인 스펙)."""
    spec = {"op": "lookup_table",
            "key": {"op": "days_between", "a": "params.d0", "b": "params.d1"},
            "table": [{"cmp": "<=", "thr": 30, "result": "A"},
                      {"cmp": "<=", "thr": 60, "result": "B"},
                      {"result": {"op": "ref", "path": "params.fallback"}}]}
    assert run(spec, {"d0": "11/13/2025", "d1": "11/14/2025", "fallback": 9}) == "A"
    assert run(spec, {"d0": "10/04/2025", "d1": "11/14/2025", "fallback": 9}) == "B"
    assert run(spec, {"d0": "01/01/2025", "d1": "11/14/2025", "fallback": 9} ) == 9
    assert run(spec, {"fallback": 9}) is None                      # 입력 없음 → 기권


def test_bool_expr_engine():
    """bool_expr 3값논리(도메인 무관)."""
    spec = {"op": "bool_expr", "all": [
        {"ref": "params.a", "eq": True},
        {"expr": {"op": "days_between", "a": "params.d1", "b": "params.d2"}, "<=": 60},
        {"ref": "params.c", "in": ["x", "y"]}]}
    assert run(spec, {"a": True, "d1": "11/01/2025", "d2": "11/10/2025", "c": "x"}) is True
    assert run(spec, {"a": False, "d1": "11/01/2025", "d2": "11/10/2025", "c": "x"}) is False
    assert run(spec, {"a": True, "d1": "01/01/2025", "d2": "11/10/2025", "c": "x"}) is False
    assert run(spec, {"a": True, "c": "x"}) is None


def test_a2_ships_no_compute_ops():
    """★회귀 방지 — A2 두 층 모두 `compute_ops` 가 비어 있어야 한다.

    다시 채우려는 사람에게: 정책의 표는 **문면으로 배달**하면 된다. 금지된 것은 엔진이 그 값을
    호출 인자에 **써 넣는** 것이다(그 인자가 채점되면 모델 기여가 0 이 된다).
    """
    for p in A2_PATHS:
        d = json.load(open(p, encoding="utf-8"))
        assert d.get("compute_ops") == {}, (p, d.get("compute_ops"))
        assert "_note_compute_ops_removed_2026_08_19" in d, p


def test_live_path_does_not_substitute():
    """★엔진 경로 계약 — 선언이 있어도 **인자를 덮어쓰지 않는다**.

    `t2_gate_patch` 에서 compute 치환 블록이 삭제됐는지 소스로 확인한다(호출 없이 검정 가능한
    유일한 축자 근거). 문자열 검사인 이유: 그 경로는 라이브 오케스트레이터 없이는 안 돈다.
    """
    src = open(os.path.join(os.path.dirname(__file__), "t2_gate_patch.py"), encoding="utf-8").read()
    # ⚠주석은 그 문자열을 **증거로 인용**하므로 문자열 존재 여부로는 못 잰다 — 코드 형태로 잰다.
    assert "resolve_compute_params(am, state.messages, a2)" not in src, "치환 호출이 살아 있다"
    assert '[T2_RESOLVE] compute silent-repair %s %s->%s' not in src, "치환 인쇄가 살아 있다"
    assert '_nz[_cp["param"]] = _cp["computed"]' not in src, "인자 덮어쓰기가 살아 있다"
    assert '_nested[_rf["param"]] = _rf["correct"]' not in src, "참조 치환이 살아 있다"


def test_all():
    n = 0
    for fn in (test_lookup_table_interpreter, test_bool_expr_engine,
               test_a2_ships_no_compute_ops, test_live_path_does_not_substitute):
        fn(); n += 1
    print("compute unit: %d/%d PASS (op 해석기 + 값-산출 폐기 계약)" % (n, n))


if __name__ == "__main__":
    test_all()
