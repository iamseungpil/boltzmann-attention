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
import os
import re
import json
# ★2026-08-12 사고: sys 미임포트인 채 print(file=sys.stderr) 3곳 — T2_ARG_AXIS formalize
#   성공 경로(365행)가 NameError 를 던져 **바깥 ENUM try 까지 통째로 죽였다**(070/071 g런
#   실측: '건너뜀(무발화): NameError sys' · 집합外 'Cobalt Blue Business Checking Account'
#   무검사 통과). 인쇄를 넣으라는 교정(§4-5)이 인쇄 자체의 배선을 안 검사해 생긴 거울상.
import sys

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


def registry_names(agent):
    """이 대화에서 **실재하는** 도구 이름 전부(에이전트 것 + 양측 discoverable).

    출처는 프레임워크 레지스트리뿐이라 도메인 리터럴 0이다. 아래 `stated_names`의 필터로만 쓴다.
    """
    out = {getattr(t, "name", None) for t in (getattr(agent, "tools", None) or [])}
    env = getattr(getattr(agent, "_t2_orch", None), "environment", None)
    for holder in ("tools", "user_tools", "agent_tools"):
        tk = getattr(env, holder, None)
        if tk is None:
            continue
        try:
            out |= set(getattr(tk, "tools", {}) or {})
        except Exception:
            pass
        g = getattr(tk, "get_discoverable_tools", None)
        if callable(g):
            try:
                out |= set(g() or {})
            except Exception:
                pass
    return {n for n in out if n}


def stated_names(msgs, name_pattern, registry):
    """우리 층이 이 대화에서 **이미 이름을 말한** 도구들 — 레지스트리 실재분만.

    구판의 출처 집합은 성공한 tool-result뿐이었다. 우리 피드백은 `role=tool, error=True`로 나가므로
    **구조적으로 제외**됐고, 그래서 절차 체크리스트가 방금 이름을 대 준 도구를 같은 층의 출처 가드가
    "지어낸 이름"이라고 막는 일이 생겼다. 그 턴에 모델이 할 수 있는 행동이 0이 된다.

    우리 scaffold가 정본이므로([[25]]) 우리가 말한 이름은 출처가 있다. 다만 이 확장이 날조 차단을
    약화시키지 않도록 **레지스트리 소속을 교집합으로 건다** — 통과 조건은 '레지스트리 밖 이름 통과 0'
    이고, 이 함수는 그것을 구성으로 보장한다(집합 밖은 애초에 담기지 않는다).
    """
    if not name_pattern or not registry:
        return set()
    rx = re.compile(name_pattern)
    out = set()
    for m in msgs:
        if getattr(m, "role", None) != "tool" or not getattr(m, "error", False):
            continue
        c = getattr(m, "content", None)
        if isinstance(c, str):
            out |= {n for n in rx.findall(c) if n in registry}
    return out


OPERATOR_FIND_FB = (
    "[OPERATOR-SELECT] you called the discovered tool '{chosen}', but the user's request maps to "
    "'{want}' among the discovered tools. Re-check which discovered tool actually fulfills the "
    "request and call that one."
)


def resolve_operator(opspec, args_dict, msgs, agent=None, la=None, UserMessage=None,
                     declared_required=None, a2=None):
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
    # ★T3 출처 집합에 우리 층을 포함 (2026-08-06·정본 `CONFLICT_ARBITRATION_THEORY_2026_08_06` §3-T3).
    #   우리가 방금 이름을 말해 놓고 그 이름을 "발명"이라 막던 자리다. 레지스트리 교집합이라
    #   날조 통과는 구조적으로 불가.
    if agent is not None and os.environ.get("T2_PROV_OURS") == "1":
        try:
            cands = cands | stated_names(msgs, opspec.get("name_pattern"), registry_names(agent))
        except Exception:
            pass
    if cands and str(chosen) not in cands:
        return {"status": "deny", "reason": "operator-fab",
                "feedback": ("[OPERATOR-PROVENANCE] tool name '%s' was not discovered from any "
                             "prior search/listing result — do NOT invent tool names. Search/list "
                             "the available tools first (getter %s), then use one of the discovered "
                             "names." % (chosen, opspec.get("getter", "")))}
    # ★FIND(Lever 1): 발견 후보 2+ 중 의도-매칭 도구 선택 검증.
    # ★C10(2026-08-05·051 실측): 우리 선언이 **이미 요구한 도구**를 "틀린 선택"이라고 말하지 않는다.
    #   051의 gold는 `get_payment_history_6183`과 `get_credit_limit_increase_history_4829`를 **둘 다**
    #   요구하는데, 이 레버는 요청이 후자에 매핑된다며 전자를 거부했다. 하나의 요청에 하나의 도구가
    #   대응한다는 가정이 절차형 태스크에서 깨진다 — 선언된 요구 집합의 원소면 지금 부르는 것이 옳다.
    if str(chosen) in (declared_required or set()):
        return {"status": "ok"}
    if (opspec.get("find_intent") and agent is not None and la is not None
            and len(cands) >= 2 and str(chosen) in cands):
        want = formalize_intent_tool(agent, la, UserMessage, msgs, cands)
        if want and str(want) != str(chosen):
            # ★완료-검사 (2026-08-12·j런 070t0 t48: want=open_4821 이 **이미 실행 성공**한
            #   상태에서 '그것을 호출하라'고 재지시 → 중복 open 2회). 그 행동이 디스패처로
            #   이미 성공 커밋됐으면 재지시는 항상 틀렸다 — 호출 이력 = 닫힌 술어([[22]]).
            #   침묵(ok)한다: 모델의 현재 호출(chosen)은 그대로 진행되고, 새 문구는 없다.
            if str(want) in _executed_dispatch_names(msgs, a2, arg):
                return {"status": "ok"}
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
# ★Lever 2 (discovery controller·BANK_ACTIONREQ_PROBE_FORENSIC §6b (A) REACH 580):
#   target이 discoverable dispatcher면 발견체인(getter→unlock→call)을 명시. 별도 컨트롤러 아님 —
#   action-required 피드백을 발견-인지형으로 특화(A2 operands 참조·도메인-일반).
# ★C241 U3': banking 어휘 제거. 구판은 액션 예시 3종('open bank account'…)과 **실제 도구
#   인스턴스 이름**('open_bank_account_4821')과 디스패처 3종 이름을 산문에 박고 있었다 —
#   페르소나 명사(gate `:2286`)보다 심각한 형태였다. 처방 = Q1 ⓐ(A2 키 순증 0):
#   ①디스패처 이름은 `{unlock}`/`{call}`/`{list}` 플레이스홀더로 A2에서 주입
#   ②구체 예시는 삭제하고 **쿼리 스타일 지침만 남긴다** — "함수명이 아니라 평문으로 행동을
#     지칭하라"는 것은 C114/C139 실측(밑줄 함수명 쿼리는 BM25 점수 0·평문은 절차 문서 1~2위)에서
#     온 **도메인-일반** 사실이므로 유지해도 리터럴이 아니다. 도메인별 예시가 필요하다는
#     반증이 나오면 그때 A2 키를 신설한다(순증 1).
DISCOVERY_REQUIRED_FB = (
    "[DISCOVERY-REQUIRED] this request needs a specialized internal tool whose name is written inside a "
    "knowledge-base document. Do the discovery chain: (1) call {getter} with a query that names the ACTION "
    "you need in PLAIN WORDS (not as a function name with underscores — plain wording is what matches the "
    "documents); the matching KB document states the exact tool name, which carries a numeric suffix; "
    "(2) {unlock} with that name; (3) {call} with that name and its arguments. Do NOT rely on "
    "{list} (it only lists tools you already called) and do NOT transfer — the tool "
    "name is in the knowledge base, search for it."
)


# ★진행-감응 (2026-08-12·C442·`T2_DISCOVERY_STEP2`). 위 문구는 볼 때마다 *"(1) 검색"* 을
#   말한다 — 그래서 이름을 이미 알아도 또 검색한다. x273 실측(n=8·정보-맞춘 격리):
#     출시 문구 + 라이브 검색결과(이름 실재)  UNLOCK **0/8** (전부 재검색)
#     출시 문구 + 이름 문서 통째              UNLOCK **0/8**
#     출시 문구 + 이름 한 줄                  UNLOCK **0/8**
#     ↓ 아래 문구로 바꾸면
#     같은 검색결과 + 이 문구                 UNLOCK **8/8**   · 문맥 없이도 **8/8**
#   라이브의 `KB_search 7회 · unlock 0회` 가 이것으로 설명된다. 진행-무감각한 요구는 자기가
#   만든 상태를 자기가 되감는다([[64]] 정밀화).
#   ⚠**측정한 문구 = 출시할 문구**([[03b]]) — 아래는 x273 `E_STEP2` 축자다({unlock} 만 A2 주입).
DISCOVERY_STEP2_FB = (
    "[DISCOVERY-STEP2] the knowledge base you already searched names the tool for this "
    "action: {name}. It is not in your tool list, so it must be unlocked before it can be "
    "used. Call {unlock} with that name now. Do not search again - the name is already known."
)


def _retrieved_unlockables(messages, known_names, unlock_tool):
    """회수 텍스트에 **실재하고** 아직 unlock 되지 않은 이름 **전부** (닫힌 술어·[[22]]).

    둘 다 문자열/호출-이력 판정이다 — 도메인 산문을 해석하지 않는다([[59]]).
    이름 집합은 **환경 레지스트리**에서 오고(호출부가 넘긴다) 우리가 짓지 않는다.
    ★복수 반환 (2026-08-12·j런 070t0 부검): 구판은 '아무 첫 이름'을 돌려줬고, 그 이름이
      요청과 무관한 도구(transfer_7291)일 때 STEP2 가 5라운드 줄다리기·이관-후 좀비 unlock 을
      만들었다. **어느 이름이 요청을 성취하는지는 LLM 몫**이다 — 호출부가 formalize 로 고른다.
    """
    if not known_names:
        return []
    tried = set()
    seen = []
    for m in (messages or []):
        for tc in (getattr(m, "tool_calls", None) or []):
            if getattr(tc, "name", None) == unlock_tool:
                for v in (getattr(tc, "arguments", None) or {}).values():
                    tried.add(str(v))
        if getattr(m, "role", None) in ("tool", "user"):
            c = getattr(m, "content", None)
            if c:
                seen.append(str(c))
    hay = "\n".join(seen)
    return [nm for nm in sorted(known_names, key=len, reverse=True)
            if nm and nm not in tried and nm in hay]


def _retrieved_unlockable(messages, known_names, unlock_tool):
    """(하위호환 단수형) 첫 후보만 — 신규 경로는 복수형 + formalize 선택을 쓴다."""
    out = _retrieved_unlockables(messages, known_names, unlock_tool)
    return out[0] if out else None


def _discoverable_dispatchers(a2):
    """A2 operands서 operator_resolution=discoverable 인 도구명 → getter 맵 (Lever 2)."""
    out = {}
    for tool, ops in ((a2 or {}).get("operands") or {}).items():
        for _arg, spec in (ops or {}).items():
            if (spec or {}).get("operator_resolution") == "discoverable":
                out[tool] = (spec or {}).get("getter", "the search tool")
    return out


def _agent_ending(am, transfer_tools):
    """에이전트 이번 턴이 '회피/종결'인가 = 도구호출 0(순수 조언) 또는 transfer만."""
    calls = {getattr(tc, "name", None) for tc in (getattr(am, "tool_calls", None) or [])}
    if not calls:
        return True                       # 순수 텍스트(조언) = 회피
    if calls and calls <= (transfer_tools or set()):
        return True                       # transfer만 = 포기
    return False


def resolve_action_operator(opspec, am, msgs, a2, target_tool=None, transfer_tools=None,
                            known_names=None, agent=None, la=None, UserMessage=None):
    """★operator 해소 GET→FIND→(execute|ASK) — 행동-vs-조언(사용자 2026-07-13).
    action_tools = A2 선언(요청 성취 도구). target_tool = formalize(의도)→도구(learn·호출측 주입).
      - target ∈ available ∧ 에이전트가 미호출(조언/transfer 회피) → deny(실행 강제·action-required)
      - target 미해소(None) ∧ 회피 → ASK(조언/날조 대신 개방질문)
      - 이미 action_tool 호출 중 → ok.
    agent/la/UserMessage = STEP2 후보-선택 formalize 용(선택은 LLM 몫·[[62]]·미주입=구판 거동)."""
    action_tools = set(opspec.get("action_tools") or (a2 or {}).get("action_tools") or [])
    if not action_tools:
        return {"status": "ok"}
    called = {getattr(tc, "name", None) for tc in (getattr(am, "tool_calls", None) or [])}
    if called & action_tools:
        return {"status": "ok"}           # 이미 행동 중
    if not _agent_ending(am, transfer_tools or set()):
        return {"status": "ok"}           # 다른 도구(조회 등) 호출 중 = 진행중
    # ★이관-후 침묵 (2026-08-12·j런 070t0 t102: 인간 이관 성공 뒤에도 STEP2 가 unlock 을
    #   재촉해 좀비 unlock 을 만들었다). transfer 가 **실행**됐으면 대화는 종결 국면이다.
    #   ⚠시도가 아니라 실행이다 — 시도만 세면 GB2 에 거부된 transfer 한 번으로 이 레버가
    #     sim 끝까지 침묵한다(과침묵). 실행 판정 = 호출 직후 tool 결과가 우리 deny 채널
    #     (error=True)도 아니고 "Error:" 로 시작하지도 않는 것 — 구조 판정뿐·산문 해석 0.
    if transfer_tools and _transfer_executed(msgs, set(transfer_tools)):
        return {"status": "ok"}
    # 회피(조언/transfer) 확정 → FIND 결과로 분기
    if target_tool and target_tool in action_tools:
        # ★Lever 2: target이 discoverable dispatcher면 발견체인 안내(getter→unlock→call).
        _disc = _discoverable_dispatchers(a2)
        if target_tool in _disc:
            # ★C241 U3': 디스패처 3종 이름을 A2에서 주입. 미선언이면 이 레버를 끈다(B3 교훈 —
            #   플레이스홀더가 None으로 새어 모델에게 "None with that name"이 가면 더 나쁘다).
            _ep3 = ((a2 or {}).get("eplan") or {})
            _u3, _c3, _l3 = (_ep3.get("unlock_tool"), _ep3.get("dispatch_tool"),
                             _ep3.get("list_tool"))
            if not (_u3 and _c3 and _l3):
                return {"status": "ok"}
            # ★진행-감응 분기 (C442·기본 OFF). 이름이 **이미 회수됐고 아직 unlock 안 됐으면**
            #   (1)단계를 다시 시키지 않고 (2)단계를 이름과 함께 말한다. 그 외에는 종전 그대로.
            if os.environ.get("T2_DISCOVERY_STEP2") == "1":
                _cands2 = _retrieved_unlockables(msgs, known_names, _u3)
                _nm2 = None
                if _cands2:
                    # ★후보-정합 (2026-08-12·j런 070t0: '아무 첫 이름'이 요청과 무관한
                    #   transfer_7291 을 5라운드 지목 — OPERATOR-SELECT 와 줄다리기).
                    #   어느 이름이 요청을 성취하는지는 **LLM 이 고른다**(formalize·none 허용).
                    #   후보 1개여도 묻는다 — 070t0 의 오발이 정확히 단일-후보였다.
                    #   formalize 불가(미주입/실패/none)면 침묵이 아니라 **구판 일반문**으로
                    #   내려간다(아래 DISCOVERY_REQUIRED) — 이름 단정만 안 할 뿐이다.
                    _nm2 = formalize_intent_tool(agent, la, UserMessage, msgs, set(_cands2))
                    if _nm2 is None and agent is not None:
                        print("[T2_DISCOVERY_STEP2] 후보 %d개 중 요청-정합 없음(none) — "
                              "이름 단정 없이 일반문으로" % len(_cands2),
                              file=sys.stderr, flush=True)
                if _nm2:
                    # ★로그에 남긴다 (2026-08-12). 초판은 인쇄가 없어 `.log` 를 grep 한 내가
                    #   *"발화 0"* 으로 네 번째 계기 오독을 했다 — 문구는 사이드카로만 나간다.
                    #   [[55]] *로그 마크 ≠ 전달* 의 거울상이라, 두 출처가 **둘 다** 있어야 한다.
                    print("[T2_DISCOVERY_STEP2] deny name=%s (이미 회수·미unlock·formalize 정합)" % _nm2,
                          file=sys.stderr, flush=True)
                    return {"status": "deny", "reason": "discovery-step2",
                            "feedback": DISCOVERY_STEP2_FB.format(name=_nm2, unlock=_u3)}
            return {"status": "deny", "reason": "discovery-required",
                    "feedback": DISCOVERY_REQUIRED_FB.format(
                        target=target_tool, getter=_disc[target_tool],
                        unlock=_u3, call=_c3, list=_l3)}
        return {"status": "deny", "reason": "action-required",
                "feedback": ACTION_REQUIRED_FB.format(target=target_tool)}
    return {"status": "deny", "reason": "action-ask", "feedback": ACTION_ASK_FB}


VERIFY_PERSIST_FB = (
    "[VERIFY-PERSISTENCE] you gathered the customer's identity information but never called "
    "'{satisfier}' to complete/log the verification, and you are now deflecting or transferring. "
    "Do NOT give up: complete the required verification by calling '{satisfier}' with the verified "
    "values (and the current time), then proceed with the request."
)


def _tool_names(msgs):
    return {getattr(tc, "name", None) for m in msgs
            for tc in (getattr(m, "tool_calls", None) or [])}


def _paired_results(msgs):
    """(assistant 호출, 그 호출의 tool 결과) 순차 짝. 구조 판정 전용 — 산문 해석 0([[59]]).

    tau2 대화는 assistant(tool_calls=[c1..cn]) 뒤에 tool-role 결과가 순서대로 따른다.
    개수가 안 맞으면 그 블록은 짝 없음으로 버린다(fail-open).
    """
    ms = list(msgs or [])
    out = []
    for i, m in enumerate(ms):
        tcs = getattr(m, "tool_calls", None) or []
        if getattr(m, "role", None) != "assistant" or not tcs:
            continue
        res = []
        j = i + 1
        while j < len(ms) and getattr(ms[j], "role", None) == "tool":
            res.append(ms[j])
            j += 1
        if len(res) < len(tcs):
            continue
        for k, tc in enumerate(tcs):
            out.append((tc, res[k]))
    return out


def _result_ok(rm):
    """tool 결과가 '실행됨'인가 — 우리 deny 채널(error=True)도, "Error:" 시작도 아님."""
    if getattr(rm, "error", False):
        return False
    return not str(getattr(rm, "content", "") or "").lstrip().startswith("Error")


def _transfer_executed(msgs, transfer_tools):
    """transfer 도구가 실제로 실행(결과 OK)됐는가 — 시도·거부는 세지 않는다."""
    for tc, rm in _paired_results(msgs):
        if getattr(tc, "name", None) in transfer_tools and _result_ok(rm):
            return True
    return False


def _executed_dispatch_names(msgs, a2, arg="agent_tool_name"):
    """디스패처로 **실행 성공한** 발견형 도구 이름 집합 (닫힌 술어·[[22]]).

    2026-08-12 j런 070t0: OPERATOR-SELECT 가 이미 완료된 open 을 '지금 호출하라'고 재지시해
    중복 open 2회를 만들었다 — 완료 여부는 호출 이력이 이미 안다. 이름 = A2 eplan 의
    dispatch_tool 호출 인자(기본 agent_tool_name)·성공 = `_result_ok`.
    """
    d = ((a2 or {}).get("eplan") or {}).get("dispatch_tool")
    if not d:
        return set()
    out = set()
    for tc, rm in _paired_results(msgs):
        if getattr(tc, "name", None) != d or not _result_ok(rm):
            continue
        ar = getattr(tc, "arguments", None) or {}
        v = ar.get(arg) if isinstance(ar, dict) else None
        if v:
            out.add(str(v))
    return out


def resolve_verify_persistence(am, msgs, a2, transfer_tools=None):
    """★Lever 3 (F1/F5·task_023형): 신원 수집(gather_prefix)했으나 검증 satisfier 미호출 상태로
    포기(조언/transfer)하면 완결 리마인더. A2 verify 게이트(kind=auth)의 verify_gather_prefix로 발동.
    satisfier = 게이트 satisfiers 키(예: log_verification). 도메인-일반(로직)·prefix/tool=A2 데이터.
    반환 {status: ok|deny, feedback}."""
    for g in ((a2 or {}).get("gates") or []):
        if g.get("kind") != "auth":
            continue
        prefix = g.get("verify_gather_prefix")
        sats = list((g.get("satisfiers") or {}).keys())
        if not prefix or not sats:
            continue
        satisfier = sats[0]
        called = _tool_names(msgs) | {getattr(tc, "name", None)
                                      for tc in (getattr(am, "tool_calls", None) or [])}
        if satisfier in called:
            return {"status": "ok"}                     # 이미 검증 완료
        gathered = any(nm and str(nm).startswith(prefix) for nm in called)
        if gathered and _agent_ending(am, transfer_tools or set()):
            return {"status": "deny", "reason": "verify-persistence",
                    "feedback": VERIFY_PERSIST_FB.format(satisfier=satisfier)}
    return {"status": "ok"}


# ★축(계열) 표면화 (2026-08-12·C444·`T2_ARG_AXIS`). 실측: 모델은 축을 **정확히 말할 수 있는데**
#   (라이브 11K자 문맥서 8/8 · `business` 낱말을 다 지워도 8/8 · 개인 요청엔 개인으로 0/8 대조)
#   인자를 지을 때 그것을 쓰지 않았다(`account_type="checking"` 로 개인 계좌를 열었다).
#   ⇒ 능력도 부하도 아니라 **적용**이다. 그래서 엔진이 하는 일은 하나 — **LLM 출력 둘을 맞대고
#     다르면 그 사실만 알린다.** 무엇이 옳은지 고르지 않는다([[62]] ③④·[[22]] 닫힌 술어).
#   ⚠선례 `resolve_operator` FIND 와 같은 형태이고 **C10 사고를 상속한다**(정당한 다중을 오차단).
#     그래서 형식화는 **여러 값**을 낼 수 있고, 호출 인자가 그 집합의 **원소이면 통과**한다.
ARG_AXIS_FB = (
    "[ARG-AXIS] you set {arg}='{got}', but this customer's request is for {want}. "
    "Re-check which one this call is for, and call the tool again with the value that "
    "matches the request."
)


def formalize_arg_axis(agent, la, UserMessage, msgs, arg, choices, prompt_tpl):
    """격리 LLM: 손님 발화 → 닫힌 집합(도구 독스트링 enum)의 원소들. 실패=None(안전).

    ⚠집합은 **호출부가 준다**(env 도구 선언 유래) — 엔진이 짓지 않는다.
    ⚠**여러 개**를 허용한다: 요청이 둘인 태스크에서 하나만 받으면 정당한 쪽을 오차단한다(C10).
    """
    if not (choices and agent is not None and la is not None and prompt_tpl):
        return None
    _allu = [str(getattr(m, "content", "") or "") for m in msgs
             if getattr(m, "role", None) == "user"]
    # ★첫 발화 포함 (2026-08-12·j런 071t0 t38 오탐): 과제 진술(두 축 요청)은 첫 발화에
    #   있는데 창이 [-6:] 뿐이라 종반 턴에선 그것이 빠져 한 축만 형식화 → 정답 인자를
    #   오판 지적했다(모델이 무시해 무해했으나 over-block 시도·[[25]]). 첫 발화 + 최근 5.
    users = (_allu[:1] + _allu[-5:]) if len(_allu) > 6 else _allu
    if not users:
        return None
    body = prompt_tpl.format(arg=arg, choices=", ".join(sorted(choices)),
                             text="\n- ".join(u[:400] for u in users))
    try:
        um = UserMessage(role="user", content=body)
    except TypeError:
        um = UserMessage(content=body)
    try:
        kw = {k: v for k, v in dict(getattr(agent, "llm_args", None) or {}).items()
              if "tool" not in k}
        sub = la.generate(model=agent.llm, tools=None, messages=[um],
                          call_name="arg_axis_formalize", **kw)
        raw = " ".join(str(getattr(sub, "content", None) or "").split())
    except Exception as e:
        print("[T2_ARG_AXIS] 호출 실패(무발화): %r" % (e,), file=sys.stderr, flush=True)
        return None
    # 엔진은 **집합 소속만** 본다 — 답에서 원소를 찾는 것뿐이고 도메인 해석 0([[59]]).
    got = {c for c in choices if c and c.lower() in raw.lower()}
    # 긴 이름이 짧은 이름을 포함하면(`business_checking` ⊃ `checking`) 짧은 쪽은 버린다.
    out = {c for c in got if not any(c != o and c in o for o in got)}
    print("[T2_ARG_AXIS] formalize → %s (raw=%r)" % (sorted(out), raw[:60]),
          file=sys.stderr, flush=True)
    return out or None


def formalize_intent_tool(agent, la, UserMessage, msgs, action_tools):
    """★FIND(의도→operator): 격리 LLM 서브콜 — 사용자 요청이 요구하는 action_tool 1개(or none).
    도메인-일반(intent→operator = 값 formalize의 operator판·learn 정의역). 실패=None(안전)."""
    if not action_tools or agent is None or la is None:
        return None
    users = [str(getattr(m, "content", "") or "") for m in msgs
             if getattr(m, "role", None) == "user"][-6:]
    # ⛔의도 분류는 **어디에도 입법하지 않는다** (사용자 지시 2026-08-12: "우리 엔진이나
    #   A2/A3 모두 의도 분류를 하지 않는다"). 이 프롬프트는 순수 질문형으로 남긴다 — 판단은
    #   온전히 격리 LLM 몫이고, 엔진은 답의 집합 소속만 본다([[52]]·[[59]]).
    #   ★사고 기록 (judge6 24 sim 전수 포렌식·-6 중 -6): 여기 "still asking questions,
    #   comparing options ... 이면 none 이라 답하라"는 **케이스 규칙**을 넣었다가(af8c1e21)
    #   의도+질문 복합 발화("I want to refer ... which one?")가 일괄 none 이 되어 [ORDER]/
    #   [ACTION] 푸시가 전면 침묵 — 098 4/4→0/4·099 3/4→1/4. 규칙은 프롬프트에 넣어도
    #   입법이다. 원래 막으려던 무단 개설(071t3)은 이 절로 안 닫혔음도 실측(모델 자발 호출).
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


RECOMMEND_VERIFY_FB = (
    "[RECOMMEND-VERIFY] before offering '{action}' to the user, verify the operand against the "
    "user's stated hard requirements. You are offering '{operand}={chosen}', but checking the "
    "requirements against the available options, '{correct}' is the one that satisfies ALL of them. "
    "Re-offer with '{operand}={correct}' (or, if unsure, ask the user which requirement to prioritize)."
)
# ★오추천(텍스트)·미추천 공통: 항상 offer_tool로 올바른 값 제안하도록 유도(user 이탈도 축소).
RECOMMEND_OFFER_FB = (
    "[RECOMMEND-OFFER] the user wants '{action}', but you are only describing options in text (or "
    "deflecting) instead of formally offering it. Based on the user's hard requirements and the "
    "available options, '{operand}={correct}' is the match. Offer it now by calling '{offer}' with "
    "{name_key}='{action}' and {operand}='{correct}', so the user can act on the correct "
    "option. If no option satisfies all requirements, tell the user that plainly."
)


def _parse_nested_args(v):
    """give_discoverable_user_tool의 arguments(JSON 문자열 or dict) → dict."""
    if isinstance(v, dict):
        return v
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return {}
    return {}


def _formalize_correct_operand(agent, la, UserMessage, msgs, action, operand, chosen):
    """★요구사항→올바른 operand formalize (직접실행 dry-run 동형·learn 정의역).
    사용자 발화(요구) + 문맥의 tool/KB 결과(후보·스펙)를 보고 모든 hard requirement를 만족하는
    단일 값을 고른다. 실패=None(안전·미개입)."""
    if agent is None or la is None:
        return None
    users = [str(getattr(m, "content", "") or "") for m in msgs
             if getattr(m, "role", None) == "user"]
    ctx = [str(getattr(m, "content", "") or "") for m in msgs
           if getattr(m, "role", None) == "tool" and not getattr(m, "error", False)]
    prompt = ("A user is applying via '%s'. Based ONLY on the user's stated HARD requirements and "
              "the option details available below, which single value of '%s' satisfies ALL of the "
              "user's hard requirements? If several qualify, pick the best-fitting; if none clearly "
              "qualifies or info is missing, reply 'none'.\n"
              "User said:\n- %s\n\nOption details (from lookups):\n%s\n"
              'Reply JSON only: {"%s": "<value or none>"}'
              % (action, operand, "\n- ".join(u[:400] for u in users[-8:]),
                 "\n".join(c[:600] for c in ctx[-8:]), operand))
    try:
        try:
            um = UserMessage(role="user", content=prompt)
        except TypeError:
            um = UserMessage(content=prompt)
        kw = {k: v for k, v in dict(getattr(agent, "llm_args", None) or {}).items() if "tool" not in k}
        sub = la.generate(model=agent.llm, tools=None, messages=[um],
                          call_name="recommend_operand_verify", **kw)
        txt = getattr(sub, "content", None) or ""
        m = re.search(r'"%s"\s*:\s*"([^"]+)"' % re.escape(operand), txt)
        cand = m.group(1).strip() if m else None
        if not cand or cand.lower() == "none":
            return None
        return cand
    except Exception:
        return None


def _formalize_recommendation(agent, la, UserMessage, msgs, action, operand):
    """★apply-intent 판별 + 요구→올바른 operand formalize (단일 서브콜). 반환 (applies, correct).
    applies=사용자가 그 user-실행 action을 원하나 · correct=모든 hard requirement 만족 값(불명 none)."""
    if agent is None or la is None:
        return (False, None)
    users = [str(getattr(m, "content", "") or "") for m in msgs
             if getattr(m, "role", None) == "user"]
    ctx = [str(getattr(m, "content", "") or "") for m in msgs
           if getattr(m, "role", None) == "tool" and not getattr(m, "error", False)]
    prompt = ("A user is talking to a bank agent. (1) Based on the user's messages, do they want to "
              "'%s'? (2) If yes, given their stated HARD requirements and the option details below, "
              "which single value of '%s' satisfies ALL hard requirements (pick best-fitting; 'none' "
              "if no option clearly qualifies or info is missing)?\n"
              "User said:\n- %s\n\nOption details (from lookups):\n%s\n"
              'Reply JSON only: {"applies": true/false, "%s": "<value or none>"}'
              % (action, operand, "\n- ".join(u[:400] for u in users[-8:]),
                 "\n".join(c[:600] for c in ctx[-8:]), operand))
    try:
        try:
            um = UserMessage(role="user", content=prompt)
        except TypeError:
            um = UserMessage(content=prompt)
        kw = {k: v for k, v in dict(getattr(agent, "llm_args", None) or {}).items() if "tool" not in k}
        sub = la.generate(model=agent.llm, tools=None, messages=[um],
                          call_name="recommend_formalize", **kw)
        txt = getattr(sub, "content", None) or ""
        applies = bool(re.search(r'"applies"\s*:\s*true', txt, re.I))
        m = re.search(r'"%s"\s*:\s*"([^"]+)"' % re.escape(operand), txt)
        cand = m.group(1).strip() if m else None
        if cand and cand.lower() == "none":
            cand = None
        return (applies, cand)
    except Exception:
        return (False, None)


def _offered_in_history(msgs, offer, action, name_key=None):
    """이전에 offer_tool로 그 action을 이미 제안했나(중복 nag 방지).

    ★C241 U6c: 인자 키(`discoverable_tool_name`)는 **banking 전용**이고 `dispatch_name_key`
    (=`agent_tool_name`·call 디스패처용)와 **다른 키**라 그걸로 대체할 수 없다. A2
    `recommendation_verify.offer_name_key`에서 읽는다(**순증 1키** — Q1 재계산 반영).
    미선언이면 중복-판정을 하지 않는다(=nag 억제 비활성·기능적으로 안전측).
    """
    if not name_key:
        return False
    for m in msgs:
        for tc in (getattr(m, "tool_calls", None) or []):
            if getattr(tc, "name", None) == offer:
                a = getattr(tc, "arguments", None) or {}
                if str(a.get(name_key)) == str(action):
                    return True
    return False


def resolve_recommendation(am, msgs, a2, agent=None, la=None, UserMessage=None, transfer_tools=None):
    """★Lever 4 (사용자 2026-07-13·미추천/오추천 방지): user-실행 action(apply)의 operand(card) 검증.
    user-실행이라 write는 못 게이트하나 *제안(offer)*은 agent 호출=게이트 가능. 두 경로:
      (A) offer_tool로 제안 중 → operand를 요구→formalize 검증·틀리면 교정(오추천-offer).
      (B) 제안 없이 apply-intent로 종결(텍스트추천/미추천) → 올바른 값으로 offer_tool 제안 유도
          (오추천-텍스트·미추천·user 이탈 축소를 한 번에). formalize=learn·직접실행 dry-run 동형."""
    spec = (a2 or {}).get("recommendation_verify")
    if not spec or agent is None or la is None:
        return {"status": "ok"}
    offer = spec.get("offer_tool"); action = spec.get("action_tool"); operand = spec.get("operand")
    # ★C241 U6c: offer 도구의 내부-이름 인자 키를 A2에서 읽는다(신규 키·순증 1).
    name_key = spec.get("offer_name_key")
    if not (offer and action and operand and name_key):
        return {"status": "ok"}
    # (A) 이번 턴에 offer_tool로 제안 중 → operand 검증
    for tc in (getattr(am, "tool_calls", None) or []):
        if getattr(tc, "name", None) != offer:
            continue
        a = getattr(tc, "arguments", None) or {}
        if str(a.get(name_key)) != str(action):
            continue
        chosen = _parse_nested_args(a.get("arguments")).get(operand)
        if not chosen:
            continue
        correct = _formalize_correct_operand(agent, la, UserMessage, msgs, action, operand, chosen)
        if correct and str(correct) != str(chosen):
            return {"status": "deny", "reason": "recommendation-verify", "call": tc,
                    "feedback": RECOMMEND_VERIFY_FB.format(
                        action=action, operand=operand, chosen=chosen, correct=correct)}
        return {"status": "ok"}       # 제안했고 검증 통과
    # (B) 제안 없이 종결(텍스트추천/미추천) → apply-intent면 올바른 값으로 offer 유도
    if not _agent_ending(am, transfer_tools or set()):
        return {"status": "ok"}       # 아직 작업 중(조회 등)
    # ★비용 게이트(Δcost): apply-flow 신호(연구 도구 호출) 있을 때만 formalize 서브콜.
    #   무관한 종결마다 LLM 호출 방지. research_tool 미기재면 항상 허용(하위호환).
    _rt = spec.get("research_tool")
    if _rt and _rt not in _tool_names(msgs):
        return {"status": "ok"}
    if _offered_in_history(msgs, offer, action, name_key):
        return {"status": "ok"}       # 이전에 제안함 → 중복 nag 금지
    applies, correct = _formalize_recommendation(agent, la, UserMessage, msgs, action, operand)
    if applies and correct:
        return {"status": "deny", "reason": "recommendation-offer",
                "feedback": RECOMMEND_OFFER_FB.format(name_key=name_key, 
                    action=action, operand=operand, correct=correct, offer=offer)}
    return {"status": "ok"}


REF_FILTER_FB = (
    "[REFERENCE-FILTER] the request identifies a specific record by {crit}, which uniquely matches "
    "{param}='{correct}' among the retrieved records — but you used '{chosen}'. Use '{correct}'."
)
_RECLINE = re.compile(r"^\s*([a-zA-Z_][a-zA-Z_0-9]*)\s*:\s*(.+?)\s*$")


def parse_records(text, key_field="transaction_id", require=("date", "amount")):
    """★엔진 record 파서(도메인-일반): 'key_field:' 시작마다 record 블록·완전(require 필드有)만.
    field명은 A2가 선언(key_field·require)·로직 일반."""
    recs = []; cur = None
    for line in str(text or "").split("\n"):
        m = _RECLINE.match(line)
        if not m:
            continue
        k, v = m.group(1), m.group(2).strip()
        if k == key_field:
            if cur:
                recs.append(cur)
            cur = {}
        if cur is not None:
            cur[k] = v
    if cur:
        recs.append(cur)
    return [r for r in recs if r.get(key_field) and all(r.get(f) for f in require)]


def _gathered_records(msgs, key_field, require):
    out = []
    for m in msgs:
        if getattr(m, "role", None) == "tool" and not getattr(m, "error", False):
            c = getattr(m, "content", None)
            if isinstance(c, str):
                out += parse_records(c, key_field, require)
    seen = {}
    for r in out:
        seen[r.get(key_field)] = r
    return list(seen.values())


def formalize_reference_criteria(agent, la, UserMessage, msgs, fields):
    """★FIND(learn): user 발화 → 식별기준 dict(fields 예: date·merchant·transaction_type). 실패=None."""
    if agent is None or la is None:
        return {}
    users = [str(getattr(m, "content", "") or "") for m in msgs
             if getattr(m, "role", None) == "user"]
    prompt = ("The user is referencing a specific transaction/record. From their messages, extract "
              "identifying criteria as JSON with keys %s (use null if not stated). Dates as MM/DD/YYYY.\n"
              "User said:\n- %s\nReply JSON only." % (fields, "\n- ".join(u[:400] for u in users[-8:])))
    try:
        try:
            um = UserMessage(role="user", content=prompt)
        except TypeError:
            um = UserMessage(content=prompt)
        kw = {k: v for k, v in dict(getattr(agent, "llm_args", None) or {}).items() if "tool" not in k}
        sub = la.generate(model=agent.llm, tools=None, messages=[um],
                          call_name="reference_criteria_formalize", **kw)
        txt = getattr(sub, "content", None) or ""
        mm = re.search(r"\{.*\}", txt, re.S)
        d = json.loads(mm.group(0)) if mm else {}
        return {k: v for k, v in d.items() if v not in (None, "null", "")}
    except Exception:
        return {}


def resolve_reference_filter(am, msgs, a2, agent=None, la=None, UserMessage=None):
    """★reference-filter 레버(keystone·사용자 참조축): call_discoverable의 참조 id 파라미터를
    formalize(user→기준) → 결정론 filter(수집 record) → 올바른 id로 검증/교정. 반환 {status, ...}.
    ⋈를 LLM 아닌 결정론에 offload(§3 정당·수집사실 위). A2 reference_filter가 도구/필드/매칭 선언."""
    _rfspec = (a2 or {}).get("reference_filter")
    if not _rfspec or agent is None or la is None:
        return {"status": "ok"}
    # ★§2bj: 복수 스펙 지원(A2 리스트·구판 단일 dict 하위호환) — 031 실측: credit-dispute 변형이
    #   미선언이라 debit만 커버(Amazon-id 정박-치환이 무검증 통과). 엔진=순회만·선언=A2.
    for spec in (_rfspec if isinstance(_rfspec, list) else [_rfspec]):
        _r1 = _resolve_one_reference_filter(am, msgs, spec, agent, la, UserMessage)
        if _r1.get("status") != "ok":
            return _r1
    return {"status": "ok"}


def _resolve_one_reference_filter(am, msgs, spec, agent, la, UserMessage):
    # ★C241 U6a: banking 기본값 제거 — A2 `reference_filter[].dispatch_tool`이 이미 선언한다.
    #   미선언이면 dispatcher 개념이 없는 도메인이므로 이 레버를 **끈다**(안전측·B3 교훈).
    offer = spec.get("dispatch_tool")
    if not offer:
        return {"status": "ok"}
    tp = spec.get("tool_prefix"); param = spec.get("param", "transaction_id")
    keyf = spec.get("key_field", param); require = tuple(spec.get("require") or ("date", "amount"))
    match = spec.get("match") or []
    import t2_compute as _c
    for tc in (getattr(am, "tool_calls", None) or []):
        if getattr(tc, "name", None) != offer:
            continue
        a = getattr(tc, "arguments", None) or {}
        gtool = str(a.get("agent_tool_name") or "")
        if tp and not gtool.startswith(tp):
            continue
        nested = _parse_nested_args(a.get("arguments")) or a   # nested JSON or top-level
        chosen = nested.get(param)
        if not chosen:
            continue
        recs = _gathered_records(msgs, keyf, require)
        if len(recs) < 2:
            continue
        crit = formalize_reference_criteria(agent, la, UserMessage, msgs,
                                            spec.get("criteria_fields") or ["date", "merchant", "transaction_type"])
        if not crit:
            continue
        correct = _c.apply_op({"op": "filter", "over": "records", "return": keyf, "match": match,
                               "on_ambiguous": spec.get("on_ambiguous", "none")},
                              {"records": recs, "criteria": crit})
        if correct and str(correct) != str(chosen):
            return {"status": "deny", "reason": "reference-filter", "call": tc,
                    "correct": correct, "param": param, "nested": nested,
                    "feedback": REF_FILTER_FB.format(crit=crit, param=param, correct=correct, chosen=chosen)}
    return {"status": "ok"}


def _current_time_str(msgs):
    """get_current_time 도구결과서 MM/DD/YYYY 추출(compute의 'now' ref). 없으면 None(→abstain)."""
    for m in msgs:
        c = getattr(m, "content", None)
        if c is None and isinstance(m, dict):
            c = m.get("content")
        if isinstance(c, str):
            mm = re.search(r"current time is (\d{4})-(\d{2})-(\d{2})", c)
            if mm:
                return "%s/%s/%s" % (mm.group(2), mm.group(3), mm.group(1))
    return None


def resolve_compute_params(am, msgs, a2):
    """★compute 키스톤(§8·keystone·C81): call_discoverable dispute 호출의 정책-계산 param을 결정론 compute로
    검증·silent-repair. A2 compute_ops[tool_prefix][param]=op-스펙(t2_compute 도메인일반 엔진). 수집사실 위 계산
    (§3 정당·autofetch 아님·[[05]] clean). §8-3: 에이전트가 *제공한* param만·미확정(None)=미개입=Δspurious 최소.
    반환 repair 목록 [{call, param, old, computed, nested}] (호출측이 in-place 치환·reference-filter 동형)."""
    ops = (a2 or {}).get("compute_ops")
    if not ops:
        return []
    import t2_compute as _c
    fam = lambda nm: re.sub(r"_\d+$", "", str(nm))
    now = _current_time_str(msgs)
    recs = _gathered_records(msgs, "transaction_id", ("date", "amount"))
    out = []
    # ★C241 U6b: dispatcher 이름·인자 키를 A2에서 읽는다(구 판은 하드코딩).
    _ep = ((a2 or {}).get("eplan") or {})
    _disp = _ep.get("dispatch_tool")
    _nkey = _ep.get("dispatch_name_key") or "agent_tool_name"
    if not _disp:
        return []                      # dispatcher 미선언 도메인 = 이 경로 없음(안전한 no-op)
    for tc in (getattr(am, "tool_calls", None) or []):
        if getattr(tc, "name", None) != _disp:
            continue
        a = getattr(tc, "arguments", None) or {}
        nm = fam(a.get(_nkey))
        smap = next((m for pref, m in ops.items() if pref in nm), None)
        if not smap:
            continue
        nested = _parse_nested_args(a.get("arguments"))
        if not isinstance(nested, dict):
            continue
        ctx = {"params": nested, "now": now, "records": recs}
        for param, spec in smap.items():
            comp = _c.apply_op(spec, ctx)
            if comp is None:                             # 미확정 → 미개입(안전·§3)
                continue
            if isinstance(comp, float) and comp == int(comp):
                comp = int(comp)                         # 50.0→50 (gold 정수 매칭·라이브 repair 포맷)
            old = nested.get(param)
            if old is None:                              # 에이전트 미기재 → 미개입(§8-3 과도개입 회피)
                continue
            if str(comp) != str(old):
                out.append({"call": tc, "param": param, "old": old, "computed": comp, "nested": nested})
    return out


def resolve_operand(opspec, tool, arg, args_dict, msgs, a2,
                    agent=None, la=None, UserMessage=None):
    """★통일 디스패처. opspec.kind로 기존 primitive 라우팅. 반환 {status, ...}.
    kind ∈ {operator, membership, provenance, value}. 미지원/누락 = ok(우아한 강등)."""
    kind = (opspec or {}).get("kind")
    if kind == "operator":
        # 선언이 요구한 도구 집합(체인 requires·종단결정·절차 노드) — 집합 대조뿐이다.
        _req = set()
        for _fc in ((a2 or {}).get("follow_up_chains") or []):
            _rq = _fc.get("requires")
            _req |= set(_rq if isinstance(_rq, list) else ([_rq] if _rq else []))
            _req |= set(_fc.get("decision_tools") or [])
        for _pr in ((a2 or {}).get("procedures") or []):
            for _nd in (_pr.get("nodes") or []):
                _t = _nd.get("tool") or _nd.get("tools")
                _req |= set(_t if isinstance(_t, list) else ([_t] if _t else []))
        return resolve_operator(opspec, args_dict, msgs, agent, la, UserMessage,
                                declared_required=_req, a2=a2)
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
