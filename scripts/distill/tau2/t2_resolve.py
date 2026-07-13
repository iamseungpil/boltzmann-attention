# -*- coding: utf-8 -*-
"""통일 per-operand/operator 해소 인터프리터 (UNIFIED_OPERAND_A2_2026_07_13 §3·§7·§8).

★결정화 논제: tool-use = {operand=값 + operator=도구} 각각의 4지선다 의미해소 루프
  (GET→FIND→INFER/select→ASK/abstain)를 scaffold+A2+learn로 푸는 것.
- scaffold = 이 디스패처(도메인 무수정·리터럴 0) + 기존 primitive.
- A2 = a2["operands"][tool][arg] = {kind, ...} 선언(도메인 정보만).
- learn = FIND의 formalize 정확도(fexec).

한 함수 resolve_operand()가 kind로 라우팅: value(fexec)·operator(도구명 grounding)·
  membership(L10)·provenance(L3). banking(operator)·retail(value)이 같은 코드로 해소 = 전이.

기존 primitive 재사용(단방향 의존): t2_gate_patch·t2_formalize_exec. 이 모듈은 순수(라이브
배선 전 오프라인 검증 가능). 라이브 배선은 unified()가 T2_RESOLVE=1 시 이 함수 호출.
"""
import re
import json

# ── operator(도구명) 해소 — banking이 드러낸 일반화(§8b) ──
# operand에 operator(도구명)가 포함. GET=discovery/KB 출력의 후보 도구명, PROV=선택 도구명이
# 그 후보에 grounded(발명 금지·banking 35.9% 도구명 날조). 로직 일반·패턴은 A2.

def discovered_names(msgs, name_pattern):
    """이전 성공 tool-result에 등장한, name_pattern 매칭 도구명 집합(grounded 후보)."""
    if not name_pattern:
        return set()
    rx = re.compile(name_pattern)
    names = set()
    for m in msgs:
        if getattr(m, "role", None) != "tool" or getattr(m, "error", False):
            continue
        c = getattr(m, "content", None)
        if isinstance(c, str):
            names |= set(rx.findall(c))
    return names


def resolve_operator(opspec, args_dict, msgs):
    """operator(도구명) operand 해소. 반환 {status: ok|deny, reason, feedback}.
    ★리뷰 U3: operator=operand는 discoverable 아키텍처서만 성립(§8b agent_tool_name=명시인자).
      direct-dispatch(retail/airline)는 도구선택이 인자 아님 → operator-해소 없음(GATE/L7 관할).
      opspec.operator_resolution != "discoverable" 이면 no-op(ok).
    kind=operator: opspec={arg, name_pattern, [getter], operator_resolution:discoverable}.
      - PROV: 선택 도구명 ∉ 발견된 후보 → deny(발명·GET 강제)."""
    if opspec.get("operator_resolution") != "discoverable":
        return {"status": "ok"}   # direct-dispatch = operator는 operand 아님(U3)
    arg = opspec.get("arg", "agent_tool_name")
    chosen = args_dict.get(arg)
    if not chosen:
        return {"status": "ok"}
    cands = discovered_names(msgs, opspec.get("name_pattern"))
    if cands and str(chosen) not in cands:
        return {"status": "deny", "reason": "operator-fab",
                "feedback": ("[OPERATOR-PROVENANCE] tool name '%s' was not discovered from any "
                             "prior search/listing result — do NOT invent tool names. Search/list "
                             "the available tools first (getter %s), then use one of the discovered "
                             "names." % (chosen, opspec.get("getter", "")))}
    return {"status": "ok"}


def resolve_operand(opspec, tool, arg, args_dict, msgs, a2,
                    agent=None, la=None, UserMessage=None):
    """★통일 디스패처. opspec.kind로 기존 primitive 라우팅. 반환 {status, ...}.
    kind ∈ {operator, membership, provenance, value}. 미지원/누락 = ok(우아한 강등)."""
    kind = (opspec or {}).get("kind")
    if kind == "operator":
        return resolve_operator(opspec, args_dict, msgs)
    if kind == "membership":
        import t2_gate_patch as _g
        spec = {"entity_key": opspec["entity_key"], "items_key": opspec["items_key"],
                "items_id_path": opspec["items_id_path"]}
        mv = _g.membership_violation(args_dict, spec, msgs)
        if mv:
            bad, oid, hint = mv
            return {"status": "deny", "reason": "membership",
                    "feedback": _g.CONS_MEMBER_FEEDBACK.format(
                        bad=", ".join(bad), ent=spec["entity_key"], oid=oid,
                        hint=(" They appear in %s='%s'." % (spec["entity_key"], hint)) if hint else "")}
        return {"status": "ok"}
    if kind == "provenance":
        # write-인자 값이 assistant-first ∧ tool-never = 확인-세탁(L3). 주소류 등.
        import t2_gate_patch as _g
        val = args_dict.get(arg)
        if val and opspec.get("mode", "grounded") == "grounded":
            first, tool_ever = _g._origin_role(str(val), msgs)
            if first == "assistant" and not tool_ever:
                return {"status": "deny", "reason": "provenance",
                        "feedback": _g.ORIGIN_FEEDBACK.format(k=arg, s=val)}
        return {"status": "ok"}
    if kind == "value" and agent is not None and la is not None:
        # getter+filter(fexec)로 변형/후보 값 해소 → 1?치환후보 : ≥2?ask : 0?fallback
        import t2_formalize_exec as _fx
        v_spec = opspec.get("variant_spec") or (a2 or {}).get("variant_spec")
        req = " ".join(str(getattr(m, "content", "") or "")
                       for m in msgs if getattr(m, "role", None) == "user")
        anchor = None
        anc_op = (v_spec or {}).get("anchor_operand")
        if anc_op:
            av = args_dict.get(anc_op)
            anchor = (av[0] if isinstance(av, list) and av else av)
        cur = args_dict.get(arg)
        cur = (cur[0] if isinstance(cur, list) and cur else cur)
        if cur and v_spec:
            vr = _fx.fexec_variant_decide(agent, la, UserMessage, msgs, arg, str(cur),
                                          v_spec, req, anchor_id=anchor)
            return {"status": "resolved", "decision": vr}
    return {"status": "ok"}


def resolve_write(tool, args_dict, msgs, a2, agent=None, la=None, UserMessage=None,
                  on_error_hit=None):
    """한 write 호출의 전 operand 순차 해소(엔진 고정 순서: PROV→BIND→value→operator).
    반환 첫 deny {status:deny, arg, ...} 또는 {status:ok}. on_error는 호출측(TOOLERR)이 별도."""
    ops = ((a2 or {}).get("operands") or {}).get(tool) or {}
    order = {"provenance": 0, "membership": 1, "value": 2, "operator": 3}
    for arg in sorted(ops, key=lambda a: order.get((ops[a] or {}).get("kind"), 9)):
        r = resolve_operand(ops[arg], tool, arg, args_dict, msgs, a2, agent, la, UserMessage)
        if r.get("status") == "deny":
            r["arg"] = arg
            return r
    return {"status": "ok"}
