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


OPERATOR_FIND_FB = (
    "[OPERATOR-SELECT] you called the discovered tool '{chosen}', but the user's request maps to "
    "'{want}' among the discovered tools. Re-check which discovered tool actually fulfills the "
    "request and call that one."
)


def resolve_operator(opspec, args_dict, msgs, agent=None, la=None, UserMessage=None):
    """operator(도구명) operand 해소. 반환 {status: ok|deny, reason, feedback}.
    ★리뷰 U3: operator=operand는 discoverable 아키텍처서만 성립(§8b agent_tool_name=명시인자).
      direct-dispatch(retail/airline)는 도구선택이 인자 아님 → operator-해소 없음(GATE/L7 관할).
      opspec.operator_resolution != "discoverable" 이면 no-op(ok).
    kind=operator: opspec={arg, name_pattern, [getter], operator_resolution:discoverable, [find_intent]}.
      - PROV(FAB): 선택 도구명 ∉ 발견된 후보 → deny(발명·GET 강제).
      - FIND(⋈·find_intent=true·Lever 1): 발견 후보 ≥2 중 의도-매칭 도구 formalize → 선택≠formalize면 deny.
        (learn 축·formalize 정확도 의존·frame F3 경계 — 확신적 불일치서만 발화)."""
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
    # ★FIND(Lever 1): 발견 후보 2+ 중 의도-매칭 도구 선택 검증.
    if (opspec.get("find_intent") and agent is not None and la is not None
            and len(cands) >= 2 and str(chosen) in cands):
        want = formalize_intent_tool(agent, la, UserMessage, msgs, cands)
        if want and str(want) != str(chosen):
            return {"status": "deny", "reason": "operator-find",
                    "feedback": OPERATOR_FIND_FB.format(chosen=chosen, want=want)}
    return {"status": "ok"}


ACTION_REQUIRED_FB = (
    "[ACTION-REQUIRED] the user's request requires you to CALL the tool '{target}' — do NOT just "
    "explain how to do it, advise self-service, or transfer. Call {target} now to complete it."
)
ACTION_ASK_FB = (
    "[ACTION-ASK] you are ending without completing the request and no available tool matches it. "
    "Do NOT invent a procedure or deflect — ask the user the specific missing detail needed to act, "
    "or state clearly you cannot do this."
)


def _agent_ending(am, transfer_tools):
    """에이전트 이번 턴이 '회피/종결'인가 = 도구호출 0(순수 조언) 또는 transfer만."""
    calls = {getattr(tc, "name", None) for tc in (getattr(am, "tool_calls", None) or [])}
    if not calls:
        return True                       # 순수 텍스트(조언) = 회피
    if calls and calls <= (transfer_tools or set()):
        return True                       # transfer만 = 포기
    return False


def resolve_action_operator(opspec, am, msgs, a2, target_tool=None, transfer_tools=None):
    """★operator 해소 GET→FIND→(execute|ASK) — 행동-vs-조언(사용자 2026-07-13).
    action_tools = A2 선언(요청 성취 도구). target_tool = formalize(의도)→도구(learn·호출측 주입).
      - target ∈ available ∧ 에이전트가 미호출(조언/transfer 회피) → deny(실행 강제·action-required)
      - target 미해소(None) ∧ 회피 → ASK(조언/날조 대신 개방질문)
      - 이미 action_tool 호출 중 → ok."""
    action_tools = set(opspec.get("action_tools") or (a2 or {}).get("action_tools") or [])
    if not action_tools:
        return {"status": "ok"}
    called = {getattr(tc, "name", None) for tc in (getattr(am, "tool_calls", None) or [])}
    if called & action_tools:
        return {"status": "ok"}           # 이미 행동 중
    if not _agent_ending(am, transfer_tools or set()):
        return {"status": "ok"}           # 다른 도구(조회 등) 호출 중 = 진행중
    # 회피(조언/transfer) 확정 → FIND 결과로 분기
    if target_tool and target_tool in action_tools:
        return {"status": "deny", "reason": "action-required",
                "feedback": ACTION_REQUIRED_FB.format(target=target_tool)}
    return {"status": "deny", "reason": "action-ask", "feedback": ACTION_ASK_FB}


def formalize_intent_tool(agent, la, UserMessage, msgs, action_tools):
    """★FIND(의도→operator): 격리 LLM 서브콜 — 사용자 요청이 요구하는 action_tool 1개(or none).
    도메인-일반(intent→operator = 값 formalize의 operator판·learn 정의역). 실패=None(안전)."""
    if not action_tools or agent is None or la is None:
        return None
    users = [str(getattr(m, "content", "") or "") for m in msgs
             if getattr(m, "role", None) == "user"][-6:]
    prompt = ("The user is talking to a customer-service agent. Based ONLY on what the user asked, "
              "which ONE of these tools must the agent CALL to fulfill the request? "
              "Reply with the exact tool name, or 'none' if none applies.\n"
              "Tools: " + ", ".join(sorted(action_tools)) + "\n"
              "User said:\n- " + "\n- ".join(u[:300] for u in users) +
              '\nReply JSON only: {"tool": "<name or none>"}')
    try:
        try:
            um = UserMessage(role="user", content=prompt)
        except TypeError:
            um = UserMessage(content=prompt)
        kw = {k: v for k, v in dict(getattr(agent, "llm_args", None) or {}).items() if "tool" not in k}
        sub = la.generate(model=agent.llm, tools=None, messages=[um],
                          call_name="intent_operator_formalize", **kw)
        txt = getattr(sub, "content", None) or ""
        m = re.search(r'"tool"\s*:\s*"([^"]+)"', txt)
        cand = m.group(1).strip() if m else None
        return cand if cand in action_tools else None
    except Exception:
        return None


def resolve_operand(opspec, tool, arg, args_dict, msgs, a2,
                    agent=None, la=None, UserMessage=None):
    """★통일 디스패처. opspec.kind로 기존 primitive 라우팅. 반환 {status, ...}.
    kind ∈ {operator, membership, provenance, value}. 미지원/누락 = ok(우아한 강등)."""
    kind = (opspec or {}).get("kind")
    if kind == "operator":
        return resolve_operator(opspec, args_dict, msgs, agent, la, UserMessage)
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
