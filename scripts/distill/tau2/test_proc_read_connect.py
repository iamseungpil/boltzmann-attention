# -*- coding: utf-8 -*-
"""절차 read-강제 **연결** 검정 (2026-08-18·새 레버 0·기존 레버 두 조건 수정).

무엇을 박나:
  ① `tools_with_pin` 이 **다값 enum** 을 만든다(리스트) · 스칼라는 종전과 바이트 동일.
  ② 인자가 스키마에 없으면 여전히 **고정하지 않는다**(날조 유도 방지·종전 규약).
  ③ 호출부 조건이 `전부 read` 로 바뀌었다 — write 가 섞이면 **침묵**(§1.5 Q5 쓰기 강제 금지).
  ④ 표면화 예산이 **ready-서명별**이고, 증가는 **전달 자리**에서 일어난다([[55]]).
  ⑤ 소비자가 집합 핀을 이해한다(하나라도 실행되면 해제).

⚠전부 오프라인·모델 호출 0.
"""
import io
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_pin_read as PR

HERE = os.path.dirname(os.path.abspath(__file__))


class _T(object):
    """도구 대역 — name 과 openai_schema 만 본다."""
    def __init__(self, name, args):
        self.name = name
        self.openai_schema = {"function": {"name": name, "parameters": {
            "properties": {a: {"type": "string"} for a in args}}}}


def _enum_of(tools, name, arg):
    for t in tools or []:
        if getattr(t, "name", None) == name:
            return (((t.openai_schema.get("function") or {}).get("parameters") or {})
                    .get("properties") or {}).get(arg, {}).get("enum")
    return None


def main():
    bad = 0
    tools = [_T("call_discoverable_agent_tool", ["agent_tool_name", "arguments"]),
             _T("KB_search_bm25", ["query"])]

    out = PR.tools_with_pin(tools, "call_discoverable_agent_tool", "agent_tool_name", "a_1")
    e = _enum_of(out, "call_discoverable_agent_tool", "agent_tool_name")
    print("① 스칼라 → enum=%r" % (e,))
    if e != ["a_1"]:
        print("   FAIL — 스칼라 거동이 바뀌었다(종전 런과 비교 불가)"); bad += 1

    out = PR.tools_with_pin(tools, "call_discoverable_agent_tool", "agent_tool_name",
                            ["b_2", "a_1"])
    e = _enum_of(out, "call_discoverable_agent_tool", "agent_tool_name")
    print("② 리스트 → enum=%r" % (e,))
    if e != ["b_2", "a_1"]:
        print("   FAIL — 다값 enum 이 안 만들어졌다(호출부가 준 순서 그대로여야 한다)"); bad += 1
    if len(out) != len(tools):
        print("   FAIL — 다른 도구가 사라졌다"); bad += 1

    out = PR.tools_with_pin(tools, "call_discoverable_agent_tool", "no_such_arg", ["x"])
    print("③ 없는 인자 → %r" % (out,))
    if out is not None:
        print("   FAIL — 인자가 없는데 고정했다(이름만 지목 = 날조 유도)"); bad += 1

    src = io.open(os.path.join(HERE, "t2_gate_patch.py"), encoding="utf-8").read()

    cond = "if _rd15 and len(_rd15) == len(_cand15):" in src
    print("④ 호출부 조건 = 전부 read: %s" % cond)
    if not cond:
        print("   FAIL — `len==1` 로 되돌아갔다(이 절차에선 영영 발화 못 한다)"); bad += 1
    if "if len(_rd15) == 1:" in src:
        print("   FAIL — 구 조건이 남아 있다"); bad += 1

    sig = ("_t2_proc_state_seen" in src and "if _dagk in _abs_seen:" in src
           and "_PROC.checklist(_p, _done2)" in src
           and "T2_PROC_ABSENT_CAP" not in src)
    print("⑤ 예산 없음 · **DAG 상태**로 루핑 판정: %s" % sig)
    if not sig:
        print("   FAIL — 총량 상한이 남았거나, 판정이 우리 문면(문자열)에 걸려 있다"); bad += 1

    # 증가가 **전달 자리**(abs_fb is not None)에서만 일어나는가
    i = src.find("if abs_fb is not None:")
    inc_at_delivery = i > 0 and "_seen6.add(_last6)" in src[i:i + 700]
    j = src.find("self._t2_proc_absent_last = _dagk")
    picks_at_select = j > 0 and "_seen6.add(" not in src[j:j + 200]
    print("⑥ 기억은 전달 자리에서만: %s / 선택 자리에선 예약만: %s"
          % (inc_at_delivery, picks_at_select))
    if not (inc_at_delivery and picks_at_select):
        print("   FAIL — 인쇄를 전달로 셌다([[55]] 위반)"); bad += 1

    # ⑧ ★dedup 의 전제 = **문장이 상태 결정론**이라는 것. 문장에 턴 번호나 카운터가 섞이면
    #    같은 상태에서도 문자열이 달라져 "반복만 막는다"가 무력해진다(무한 표면화).
    import collections
    import json as _json
    import t2_procedure as _P
    _a2 = _json.load(io.open(os.path.join(HERE, "a2", "banking_knowledge.specific.json"),
                             encoding="utf-8"))
    _pr = next(p for p in _a2["procedures"] if p["id"] == "credit_limit_increase")
    _e1 = collections.Counter({"submit_credit_limit_increase_request_7392": 1})
    _e2 = collections.Counter({"submit_credit_limit_increase_request_7392": 1,
                               "get_user_dispute_history_7291": 1})
    _k = lambda ex: frozenset(n for n, _t, d in _P.checklist(_pr, ex) if d)
    _k1a, _k1b, _k2 = _k(_e1), _k(_e1), _k(_e2)
    print("⑧ DAG 상태 — 같은 실행이력 → 같은 키: %s / 한 걸음 나가면 다른 키: %s (%s → %s)"
          % (_k1a == _k1b, _k1a != _k2, sorted(_k1a), sorted(_k2)))
    if _k1a != _k1b:
        print("   FAIL — 상태 키가 결정론이 아니다(루핑 판정이 무력해진다)"); bad += 1
    if _k1a == _k2:
        print("   FAIL — 걸음이 나갔는데 같은 상태다(새 단계를 영영 못 말한다)"); bad += 1
    if not (_k1a < _k2):
        print("   FAIL — 상태가 단조롭지 않다(진동하면 루핑 판정이 성립 안 한다)"); bad += 1

    consumer = ("_pv = list(_pv) if isinstance(_pv, (list, tuple, set)) else [_pv]" in src
                and "any(v in _exec_now for v in _pv)" in src)
    print("⑦ 소비자가 집합 핀을 이해: %s" % consumer)
    if not consumer:
        print("   FAIL — 집합 핀이 해제되지 않아 재무장 로직이 깨진다"); bad += 1

    print("\n%s" % ("test_proc_read_connect PASS" if not bad
                    else "test_proc_read_connect FAIL %d건" % bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
