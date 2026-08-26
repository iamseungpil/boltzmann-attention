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

import t2_subcall as SC   # 단발-격리 서브 관용구 정본(2026-08-14 리팩토링)

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

# ★★지목 → **범위 표면화** (2026-08-15·x322·n=24·블록 8·8·8·잡음 바닥 ±4 밖):
#     A_REF(개입 없음)          **24/24**  ← 모델은 원래 정답 도구를 고른다
#     B_PINPOINT(위 문구 그대로)  **0/24**  ← 우리 개입이 **완전히 파괴**한다
#     C_SCOPES(선언된 범위 표시) **24/24**  ← 유지된다
#     D_MISMATCH(닫힌 판정 병기) **24/24**
#     E_NEG(가짜 도구 한 줄 추가)  **0/24**  ← 어떤 이름 지목에도 극도로 순응
#   라이브 실물(C485·073): 정답 도구가 **성공한 뒤**에도 5회+ 지목이 갔고, 모델이 재시도해
#   같은 계좌에 두 번 적립 → `db_match=False`. 가드는 *이미 실행한 경우*만 막았는데,
#   x322 는 **지목 자체가 파괴적**이라고 말한다.
#   ⇒ 엔진은 **고르지 않는다**: 후보들의 **선언된 범위**(도구 자기 설명의 첫 문장·기계 추출·
#     저작 0)만 인쇄하고 선택은 LLM 이 한다([[62]] ④ 위반 제거·[[64]] 무엇을 보고 고를지 제공).
OPERATOR_SCOPE_FB = (
    "[OPERATOR-SCOPE] you called '{chosen}'. The declared scope of the candidate tools is: "
    "{scopes}. Check which one is declared for the object this request acts on."
)


def _tool_scope(agent, name, cap=160):
    """도구 **자기 설명의 첫 문장** — 프레임워크 레지스트리에서 그대로 읽는다(저작 0·판단 0).

    엔진은 이 문자열을 **해석하지 않는다**([[59]]) — 인쇄만 하고, 어느 것이 맞는지는 LLM 이 본다.
    못 찾으면 빈 문자열(그 후보는 목록에서 빠진다·거동 안전측).
    """
    for holder in (getattr(agent, "tools", None) or []):
        if getattr(holder, "name", None) == name:
            d = getattr(holder, "description", None) or ""
            return " ".join(str(d).split())[:cap]
    env = getattr(getattr(agent, "_t2_orch", None), "environment", None)
    for h in ("tools", "user_tools", "agent_tools"):
        tk = getattr(env, h, None)
        if tk is None:
            continue
        try:
            f = (getattr(tk, "tools", {}) or {}).get(name)
            d = getattr(f, "__doc__", None) or getattr(f, "description", None) or ""
            if d:
                return " ".join(str(d).split())[:cap]
        except Exception:
            continue
    return ""


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
            _reg = registry_names(agent)
            cands = cands | stated_names(msgs, opspec.get("name_pattern"), _reg)
            # ★우리가 **지목한** 이름도 출처다 (2026-08-19·t7324 050 실측). 핀은 스키마를 좁힐
            #   뿐 말하지 않아 `stated_names` 가 못 찾는다 — 그래서 우리 핀이 이 가드에
            #   `operator-fab` 으로 막혔고, 같은 시드의 다른 런은 문구가 먼저 나간 순서였을 뿐이다.
            #   레지스트리 교집합은 그대로라 날조 통과는 여전히 구조적으로 불가하다.
            cands = cands | ({str(n) for n in
                              (getattr(agent, "_t2_our_names", None) or set())} & set(_reg or ()))
        except Exception:
            pass
    if cands and str(chosen) not in cands:
        # ★검증한 뒤 **참말을 한다** (2026-08-26·x550·t7360 포렌식).
        #   구판은 이 자리에서 언제나 *"네가 지어냈다 · 검색해라"* 라고 했다. 그런데 모델이
        #   **이미 갖고 있는 직접 호출 도구**를 디스패처로 감쌌을 때도 같은 말을 했다 —
        #   074 에서 `get_atm_fee_discrepancies`(우리 A2 스캐폴드 도구·`t2_scaffold_get.py`가
        #   `agent.tools` 에 실제로 주입한다)를 **6회** 그렇게 막았고(t7358·t7360 재현),
        #   모델은 시킨 대로 `KB_search_bm25` 를 3회 돌린 뒤 unlock 까지 시도해 env 오류를
        #   맞았다. env 자신의 문면이 옳게 말한다 — *"If it is a tool you already have,
        #   **call it directly**."* ⇒ 우리 문면이 env 문면보다 나빴다([[25]] 허위·[[64]] 처방).
        #
        #   ⚠**면제가 아니다.** 두 경우 모두 거부한다 — 감싼 호출은 어차피 env 가 죽인다.
        #     바뀌는 것은 [[64]] 가 요구하는 *"무엇을 하면 풀리나"* 한 칸뿐이다.
        #   ⚠권위는 **프레임워크 레지스트리**(`agent.tools`)이지 A2 선언이 아니다. A2 가 낡았거나
        #     주입이 실패했으면 그 이름은 `agent.tools` 에 없고 → 구판 문면이 그대로 나간다.
        #     모델의 주장을 읽는 자리는 없다(날조로 통과할 통로 0). `registry_names` 를 쓰지
        #     않는 이유: 그것은 **discoverable 까지 합집합**이라, 아직 발견 안 된 discoverable
        #     도구에 "직접 부르라"는 **또 다른 거짓말**이 된다.
        _held = set()
        if agent is not None:
            try:
                _held = {getattr(t, "name", None) for t in (getattr(agent, "tools", None) or [])}
            except Exception:
                _held = set()
        if str(chosen) in _held:
            print("[T2_OPERATOR_DIRECT] %s 는 이미 보유한 직접 호출 도구 — '발명' 대신 "
                  "'직접 불러라'로 답한다(x550 §1)" % chosen, file=sys.stderr, flush=True)
            return {"status": "deny", "reason": "operator-direct",
                    "feedback": ("[OPERATOR-DIRECT] '%s' is a tool you already have. It is not a "
                                 "discoverable tool, so it cannot be unlocked or dispatched: "
                                 "passing it as '%s' will not run it. Call '%s' directly, as its "
                                 "own tool call with its own arguments. Do not search for a "
                                 "suffixed version of this name: there is none."
                                 % (chosen, arg, chosen))}
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
            # ★대칭 가드 (2026-08-14 야간·073 실물·이 레버의 **세 번째** 같은 부류 결함):
            #   위 검사는 `want` 만 본다. `chosen` 이 **이미 성공 커밋**된 경우가 빠져 있었고,
            #   그 자리에서 "그건 틀린 도구다"라고 말하면 모델이 할 수 있는 일은 **재시도뿐**이다.
            #   t7292 073 t0 실측: msg45 에서 `apply_checking_account_credit_5829` 가 성공
            #   (잔액 5200.00→5209.50)했는데 우리가 5회 이상 *"요청은 apply_statement_credit_8472
            #   에 매핑된다"* 고 말했고(그건 **신용카드용**이다·체킹 태스크에서 오답),
            #   모델이 재-unlock→재호출해 **같은 계좌에 9.50 을 두 번**(→5219.00) 넣었다.
            #   gold 액션은 8/11 로 통과 때와 같은데 `db_match=False` 가 되어 reward 0.
            #   ⇒ **이미 한 일을 틀렸다고 말하지 않는다**: 되돌릴 수 없는 write 에서 그 문구는
            #     교정이 아니라 **중복 실행 지시**다([[25]] 우리 도구는 100% 정답 의무·
            #     [[64]] 거부는 고칠 방법을 담아야 하는데 여기엔 고칠 방법이 없다).
            #   술어는 닫혀 있다(호출 이력·도메인 판단 0) — 위 `want` 검사와 완전 대칭이다.
            if str(chosen) in _executed_dispatch_names(msgs, a2, arg):
                print("[T2_RESOLVE] operator-find 침묵: chosen=%s 는 이미 성공 실행 — "
                      "재지시는 중복 write 를 만든다" % chosen, file=sys.stderr, flush=True)
                return {"status": "ok"}
            # ★**읽기 선택에는 말하지 않는다** (2026-08-26·x550 §2 실측·`T2_SCOPE_ALL=1` 로 복귀).
            #   이 레버는 *잘못 고른 operator* 를 잡으려고 있는데, 실측이 그 값을 부정한다:
            #   최근 12런에서 `[OPERATOR-SCOPE]` **61회** 중 **46회가 read** 도구였고
            #   (`get_debit_cards_by_account_id` 28 · `get_debit_dispute_status` 9 …),
            #   **61 중 49 는 그 도구가 끝내 실행됐다** — 반려가 선택을 바꾼 게 아니라 **턴만
            #   태웠다**. 태스크는 079(26)·085(25) 에 몰려 있고, 085 는 그 사이 재료 수집이
            #   밀려 gold 4행 중 3행을 놓쳤다.
            #   읽기의 오선택은 회복 가능하다(다시 읽으면 된다). 쓰기의 오선택만 되돌릴 수 없다.
            #   ⇒ [[70]] 의 "끄지 말고 **조건부 발화**" — 조건은 도메인 일반 닫힌 술어
            #     (`_is_effective_write`·A2 파생)라 태스크 id 로 켜는 [[05]] 위반이 아니다.
            #   ⚠파는 것: 읽기 오선택을 **더는 지적하지 않는다**. 그 값이 양수였다는 증거는
            #     아직 없다(49/61 이 그대로 실행됐다). 음성이면 `T2_SCOPE_ALL=1` 로 되돌린다.
            if os.environ.get("T2_SCOPE_ALL") != "1":
                try:
                    import t2_gate_patch as _g
                    if not _g._is_effective_write(_g._SUFFIX_RE.sub("", str(chosen)), a2):
                        print("[T2_RESOLVE] operator-scope 침묵: chosen=%s 는 실효 write 가 "
                              "아니다 — 읽기 오선택은 회복 가능(x550 §2)" % chosen,
                              file=sys.stderr, flush=True)
                        return {"status": "ok"}
                except Exception as _swe:
                    print("[T2_RESOLVE] operator-scope write 판정 건너뜀: %r" % (_swe,),
                          file=sys.stderr, flush=True)
            # ★지목 대신 **범위 표면화**(기본·x322 실측). 지목 문구를 쓰려면 명시적으로
            #   `T2_OPERATOR_PINPOINT=1` 을 켜야 한다 — 되돌릴 길은 남기되 기본은 아니다([[60]]
            #   끄기가 아니라 조정: 발화는 계속 하고 **무엇을 말하는지만** 바꾼다).
            if os.environ.get("T2_OPERATOR_PINPOINT") != "1":
                _sc = [(n, _tool_scope(agent, n)) for n in (str(chosen), str(want))]
                _sc = [(n, d) for n, d in _sc if d]
                if not _sc:
                    return {"status": "ok"}          # 근거를 못 대면 말하지 않는다([[64]])
                print("[T2_RESOLVE] operator-scope: 지목 대신 범위 표면화 (%s)"
                      % ", ".join(n for n, _ in _sc), file=sys.stderr, flush=True)
                return {"status": "deny", "reason": "operator-scope",
                        "feedback": OPERATOR_SCOPE_FB.format(
                            chosen=chosen,
                            scopes="; ".join("'%s' = %s" % (n, d) for n, d in _sc))}
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

# ★레지스트리-폴백 문면 (2026-08-13·x283). 회수 텍스트에 이름이 **없을 때** 위 문구를 쓰면
#   "the knowledge base you already searched names the tool" 이 거짓이 된다([[25]] 우리 층
#   100% 정답 의무). 출처 절만 진실(레지스트리)로 바꾼 변형이고, x283 §E_REG 가 이 축자를
#   쟀다([[03b]] 측정문면=출시문면).
DISCOVERY_STEP2_REG_FB = (
    "[DISCOVERY-STEP2] the tool registry lists the tool for this action: {name}. It is not "
    "in your tool list, so it must be unlocked before it can be used. Call {unlock} with "
    "that name now. Do not search for the name - it is already known."
)


def agent_discoverable_names(agent):
    """env 의 **agent-side discoverable** 이름 집합 (기계 도출·도메인 리터럴 0).

    ★왜 필요한가 (2026-08-13 p런 포렌식): STEP2 후보 필터가 `registry_names`(전 도구 합집합)
    라서 **디스패처 자신**(`call_discoverable_agent_tool`)과 **직접 도구**(`log_verification`)를
    unlock 후보로 공급했다 — 실측 축자: 071 t1 turn60 · t2 turn27/37 의 넌센스 푸시.
    unlock 이 가능한 것은 이 집합의 원소뿐이다.
    """
    try:
        env = getattr(getattr(agent, "_t2_orch", None), "environment", None)
        tk = getattr(env, "tools", None)
        g = getattr(tk, "get_discoverable_tools", None)
        return set(g() or {}) if callable(g) else set()
    except Exception:
        return set()


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
    # ★U1 (2026-08-14·FORENSIC_SYNTHESIS §2-A): 이번 손님 발화 이후 발견형 디스패치가 **성공**
    #   했으면, 결과를 보고하는 순수-텍스트 턴은 회피가 아니다 — 여기서 "처음부터 발견하라"를
    #   내면 완료된 write 를 되감아 중복 실행시킨다(073 t0 잔액 5200→5228.50·write 중복 36건 중
    #   15건이 우리 문구 직후). `_transfer_executed` 와 같은 이력-감응·구조 판정.
    if _dispatch_since_last_user(msgs, a2):
        # 관측 의무(C442·[[55]] 로그 마크 ≠ 전달의 거울상): 침묵도 인쇄가 있어야 라이브에서
        # 발화를 셀 수 있다. 2026-08-14 t7287 모니터링서 이 자리가 **보이지 않아** 가드 작동을
        # 실시간으로 확인할 수 없었다.
        print("[T2_ACTION_HISTORY] 침묵: 이번 손님 발화 이후 디스패치 성공 — 재-발견 요구 안 함",
              file=sys.stderr, flush=True)
        return {"status": "ok"}
    # ★write-착수 전달 (2026-08-14·`T2_WRITE_SUB`·기본 OFF·x307~x310).
    #   여기까지 왔다는 것은 **회피가 확정**된 자리다. 종전에는 문면(발견 체인/실행 촉구)을 냈고,
    #   그 문면은 이 사이트에서 전 팔 0/8 이었다(x302b). 격리 서브는 같은 사실로 7/8~8/8 을 내고,
    #   그 산출을 전달하면 메인이 8/8 로 실행한다(x309). 실패·검산 탈락이면 아래 종전 경로 그대로.
    if os.environ.get("T2_WRITE_SUB") == "1":
        try:
            _names = registry_names(agent) if agent is not None else set()
            _fb = sub_write_proposal(agent, la, UserMessage, msgs, a2, _names)
        except Exception as _we:
            print("[T2_WRITE_SUB] 생략(종전 경로): %r" % (_we,), file=sys.stderr, flush=True)
            _fb = None
        if _fb:
            return {"status": "deny", "reason": "write-initiation-sub", "feedback": _fb}
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
                # ★후보 = unlock 가능한 것만 (2026-08-13 p런 실측: 디스패처 자신·직접 도구가
                #   후보로 새어 "unlock call_discoverable_agent_tool"·"unlock log_verification"
                #   넌센스 푸시 발화 — 071 t1 turn60·t2 turn27/37 축자). 레지스트리 조회가
                #   비면(오프라인 테스트 등) 필터를 걸지 않는다 — 종전 거동 보존.
                _reg2 = agent_discoverable_names(agent)
                if _reg2:
                    _cands2 = [n for n in _cands2 if n in _reg2]
                # ★회수-실패 시 레지스트리 실명 폴백 (x283 C_STEP2: 071 t1/t3 8/8 — 이름이
                #   닿기만 하면 체인이 열린다·070 은 이름을 줘도 0~1/8 = 이 폴백의 한계도
                #   같은 프로브가 쟀다). 후보 = **레지스트리 전체 − unlock 기시도**(기계 도출·
                #   [[25]] 레지스트리=권위). ⚠가족-일치로 좁히지 않는다 — 라이브 A2 의
                #   discoverable 선언은 **디스패처 자신**에 걸려 있어(`call_discoverable_...`)
                #   target 가족 대조는 항상 공집합 = 死코드가 된다(2026-08-13 A2 실측).
                #   어느 이름이 요청을 성취하는지는 아래 formalize(LLM·none 허용)가 고른다.
                _regfb2 = False
                _tried2 = set()
                for _m2 in (msgs or []):
                    for _tc2 in (getattr(_m2, "tool_calls", None) or []):
                        if getattr(_tc2, "name", None) == _u3:
                            for _v2 in (getattr(_tc2, "arguments", None) or {}).values():
                                _tried2.add(str(_v2))
                if not _cands2 and _reg2:
                    _cands2 = sorted(_reg2 - _tried2)
                    _regfb2 = bool(_cands2)
                _nm2 = None
                if _cands2:
                    # ★후보-정합 (2026-08-12·j런 070t0: '아무 첫 이름'이 요청과 무관한
                    #   transfer_7291 을 5라운드 지목 — OPERATOR-SELECT 와 줄다리기).
                    #   어느 이름이 요청을 성취하는지는 **LLM 이 고른다**(formalize·none 허용).
                    #   후보 1개여도 묻는다 — 070t0 의 오발이 정확히 단일-후보였다.
                    #   formalize 불가(미주입/실패/none)면 침묵이 아니라 **구판 일반문**으로
                    #   내려간다(아래 DISCOVERY_REQUIRED) — 이름 단정만 안 할 뿐이다.
                    _nm2 = formalize_intent_tool(agent, la, UserMessage, msgs, set(_cands2))
                    # ★none → 레지스트리 재질의 1회 (2026-08-13 t7273 073t1 turn35/37/45 실측:
                    #   회수 후보 3개가 전부 무관 read 라 formalize 가 **정당하게** none —
                    #   그런데 후보가 비어있지 않아 폴백이 안 돌아 credit 도구가 후보에 든
                    #   적이 없었다. 회수-집합에 정합이 없으면 레지스트리 잔여로 한 번 더
                    #   묻는다 — 같은 formalize·none 허용·신규 판단 0.)
                    if _nm2 is None and not _regfb2 and _reg2:
                        _rest2 = sorted(_reg2 - _tried2 - set(_cands2))
                        if _rest2:
                            _nm2 = formalize_intent_tool(agent, la, UserMessage, msgs,
                                                         set(_rest2))
                            _regfb2 = _nm2 is not None
                    if _nm2 is None and agent is not None:
                        print("[T2_DISCOVERY_STEP2] 후보 %d개 중 요청-정합 없음(none) — "
                              "이름 단정 없이 일반문으로" % len(_cands2),
                              file=sys.stderr, flush=True)
                # ★같은-이름 재푸시 억제 (2026-08-13 t7273 실측: `get_payment_history_6183`
                #   9회 — 인자 변화 없는 반복은 재시도가 아니다[[57]]). sim당 이름별 2회까지,
                #   같은 턴(regen 라운드) 중복은 0회. 상태는 agent 에 두고 대화가 새로 시작되면
                #   (메시지 수 감소) 리셋 — 판정·문면 불변·횟수만 제한.
                if _nm2 and agent is not None:
                    try:
                        _st2 = getattr(agent, "_t2_step2_pushed", None)
                        _now2 = len(msgs or [])
                        if _st2 is None or _st2.get("_hwm", 0) > _now2:
                            _st2 = {"_hwm": 0}
                            agent._t2_step2_pushed = _st2
                        _st2["_hwm"] = _now2
                        _cnt2, _last2 = _st2.get(_nm2, (0, -1))
                        if _last2 == _now2 or _cnt2 >= 2:
                            print("[T2_DISCOVERY_STEP2] 재푸시 억제 name=%s (count=%d)"
                                  % (_nm2, _cnt2), file=sys.stderr, flush=True)
                            _nm2 = None
                        else:
                            _st2[_nm2] = (_cnt2 + 1, _now2)
                    except Exception:
                        pass
                if _nm2:
                    # ★A4 / OL-02 (t7336 마스터 §6.1·2026-08-22): **우리가 지목한 이름을 등재한다.**
                    #   085#0 실측 — 같은 턴 안에서 `[T2_DISCOVERY_STEP2] deny name=
                    #   get_all_user_accounts_by_user_id_3847` 가 나간 직후 모델이 순종했는데
                    #   `[T2_RESOLVE] deny … reason=operator-fab`(*"was not discovered from any prior
                    #   search"*) 이 그 이름을 막았다. STEP2 문구는 **사이드카로만** 나가므로
                    #   `stated_names`(메시지에서 찾는다)로는 안 잡히고, `_t2_our_names` 의 기록자는
                    #   `_read_routine_pin`(`t2_gate_patch.py:2709-2711`) **하나뿐**이었다.
                    #   소비자(`resolve_operator` `:171` · `T2_UNLOCK_PROV`)는 이 집합을 **이미 본다** —
                    #   빠져 있던 것은 등재뿐이다. 레버 신설 0.
                    # ⚠날조 통과는 구조적으로 불가하게 둔다: **레지스트리 교집합**으로만 넣는다
                    #   (`registry_names` = 프레임워크 레지스트리·[[25]]). 레지스트리 조회가 비면
                    #   (오프라인) 아무것도 넣지 않는다 — 종전 거동 보존·fail-closed.
                    # ⚠[[70]] 계측 의무: 이 행이 파는 것 = **`operator-fab` deny 수의 감소**와 맞바꾼
                    #   *"우리가 한 번이라도 지목한 레지스트리 이름은 출처가 있다"* 는 확장이다.
                    #   다음 런 포렌식이 셀 것 = ⑴`[T2_RESOLVE] deny … operator-fab` 건수
                    #   ⑵`[T2_OUR_NAMES] 등재` 건수 ⑶환각 통과(레지스트리 밖 이름 실행) — 구성상 0 이어야.
                    if agent is not None:
                        try:
                            _regn4 = registry_names(agent)
                            if _nm2 in _regn4:
                                _own4 = set(getattr(agent, "_t2_our_names", None) or set())
                                if _nm2 not in _own4:
                                    _own4.add(str(_nm2))
                                    agent._t2_our_names = _own4
                                    print("[T2_OUR_NAMES] 등재 name=%s (출처=T2_DISCOVERY_STEP2 지목)"
                                          % _nm2, file=sys.stderr, flush=True)
                            else:
                                print("[T2_OUR_NAMES] 미등재(레지스트리 밖) name=%s" % _nm2,
                                      file=sys.stderr, flush=True)
                        except Exception as _oe4:
                            print("[T2_OUR_NAMES] 등재 생략: %r" % (_oe4,),
                                  file=sys.stderr, flush=True)
                    # ★로그에 남긴다 (2026-08-12). 초판은 인쇄가 없어 `.log` 를 grep 한 내가
                    #   *"발화 0"* 으로 네 번째 계기 오독을 했다 — 문구는 사이드카로만 나간다.
                    #   [[55]] *로그 마크 ≠ 전달* 의 거울상이라, 두 출처가 **둘 다** 있어야 한다.
                    print("[T2_DISCOVERY_STEP2] deny name=%s (%s·미unlock·formalize 정합)"
                          % (_nm2, "레지스트리 폴백" if _regfb2 else "이미 회수"),
                          file=sys.stderr, flush=True)
                    # 폴백 이름은 회수 텍스트에 없다 — "KB 가 이름을 댔다" 문면은 거짓이 되므로
                    #   출처 절만 레지스트리로 말하는 변형을 쓴다(둘 다 축자 측정본·[[03b]]).
                    _fb2 = DISCOVERY_STEP2_REG_FB if _regfb2 else DISCOVERY_STEP2_FB
                    return {"status": "deny", "reason": "discovery-step2",
                            "feedback": _fb2.format(name=_nm2, unlock=_u3)}
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


def sub_write_proposal(agent, la, UserMessage, msgs, a2, names):
    """★write-착수 격리 서브 (2026-08-14·`T2_WRITE_SUB`·기본 OFF).

    측정 사슬(전부 사전등록·n=8·같은 사이트):
      x307  메인서 실행 **0/8** ↔ 텍스트로 물으면 **knows 7/8**  → knowing-doing
      x308  자리를 옮기면 **7/8**(JSON 요구 시 8/8) · 근거 제거 시 **0/8**(날조 안 함)
      x309  그 산출을 메인에 전달하면 **8/8 실행**(한 건만 전달해도 8/8)
      x310  근거 동봉해도 정답 팔 **8/8**(역효과 0)
    [[62]] ②: 격리에서 되는데 궤적서 못 하면 레버는 **전달뿐**이다 — 계산·선택은 전부 LLM 몫이고
    엔진은 근거 실재만 본다(`t2_subcall.grounded_calls`). [[05]] Q3: 엔진은 **실행하지 않는다** —
    제안을 리마인더로 올릴 뿐이고 호출은 메인이 한다.

    ★관용구는 전부 `t2_subcall` 정본이다(2026-08-14 리팩토링·사용자 지시 "중복을 없애라") —
      1차 구현은 자체 substring 검산이라 `9.50↔9.5` 를 기각했다(정본 `_val_grounded` 는
      형식-불문 수치 매칭). 재구현 금지.

    반환: 전달 문구(str) · 조건 미충족/검산 탈락 = None(종전 경로 폴백).
    """
    spec = (a2 or {}).get("write_initiation") or {}
    if not spec or agent is None or la is None or UserMessage is None:
        return None
    basis = SC.recent_tool_text(msgs, spec.get("basis_max_chars") or 4000,
                                scope=spec.get("basis_scope") or "recent")
    if not basis:
        return None
    users = [str(getattr(m, "content", "") or "") for m in (msgs or [])
             if getattr(m, "role", None) == "user"][-3:]
    prompt = "%s\n\n%s\n\n%s\n\n%s" % (spec.get("instructions", ""),
                                       "\n".join(users), basis,
                                       spec.get("answer_format", ""))
    txt = SC.sub_generate(agent, la, UserMessage, prompt, "write_initiation_formalize",
                          temperature=spec.get("temperature"))
    obj = SC.parse_contract(txt, key="calls")
    calls = obj.get("calls") if obj else None
    good = SC.grounded_calls(calls, [basis], names)
    # ★A-7⑷ (2026-08-23·073): **서브가 본 창**을 여기서 남긴다 — 게이트 쪽 로그의
    #   "트리거 N자" 는 다른 코퍼스라, 두 숫자를 같은 것으로 읽으면 오진한다([[25]]).
    print("[T2_WRITE_SUB] 제안 %d건 → 근거검산 통과 %d건 (서브 창 %d자·scope=%s)"
          % (len(calls or []), len(good), len(basis),
             spec.get("basis_scope") or "recent"), file=sys.stderr, flush=True)
    if not good:
        return None
    return spec.get("delivery_template", "{calls}\n{basis}").format(
        calls="\n".join("  - " + json.dumps(c, ensure_ascii=False) for c in good),
        basis=basis)


def _dispatch_since_last_user(msgs, a2):
    """이번 손님 발화 **이후**에 발견형 도구를 성공 디스패치했는가 (닫힌 술어·구조 판정).

    ★2026-08-14 실측(073 t0 `bank_t7285_b`): msg57 에서 gold credit 3건이 전부 성공했는데,
    그 다음 순수-텍스트 턴(손님에게 결과 보고)이 `_agent_ending` 에 **회피로 판정**돼
    `[DISCOVERY-REQUIRED] (1) 검색 (2) unlock (3) call` 이 나갔고, 모델은 시킨 대로 체인을
    다시 돌아 **같은 3건을 msg61·67·73 에 재실행**했다(잔액 5200→5228.50). 발견 체인을
    방금 완주한 자리에서 *"처음부터 발견하라"* 는 우리 문구는 **사실이 아니다**([[25]]).
    전수 감사(`bank_dup_exec_audit.py`·28 sim): write 중복 36건 중 **15건(42%)** 이 우리
    문구 직후였다(자발 21건 = 부정통제).

    ⚠커버리지(남은 계좌 등)는 이 레버 몫이 아니라 covfollowup 몫이라 침묵해도 안 잃는다.
    이름을 댈 수 있는 경우(STEP2)는 위에서 **이미 실행된 이름만** 후보에서 빼므로 여기 안 온다.
    """
    ms = list(msgs or [])
    last_user = -1
    for i, m in enumerate(ms):
        if getattr(m, "role", None) == "user":
            last_user = i
    if last_user < 0:
        return False
    d = ((a2 or {}).get("eplan") or {}).get("dispatch_tool")
    if not d:
        return False
    for tc, rm in _paired_results(ms[last_user:]):
        if getattr(tc, "name", None) == d and _result_ok(rm):
            return True
    return False


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
    raw = " ".join(SC.sub_generate(agent, la, UserMessage, body,
                                    "arg_axis_formalize").split())
    if not raw:
        return None
    # 엔진은 **집합 소속만** 본다 — 답에서 원소를 찾는 것뿐이고 도메인 해석 0([[59]]).
    got = {c for c in choices if c and c.lower() in raw.lower()}
    # 긴 이름이 짧은 이름을 포함하면(`business_checking` ⊃ `checking`) 짧은 쪽은 버린다.
    out = {c for c in got if not any(c != o and c in o for o in got)}
    print("[T2_ARG_AXIS] formalize → %s (raw=%r)" % (sorted(out), raw[:60]),
          file=sys.stderr, flush=True)
    return out or None


ASK_AGENT_CALL = ("which ONE of these tools must the agent CALL to fulfill the request? ")


def formalize_intent_tool(agent, la, UserMessage, msgs, action_tools, ask=None):
    """★FIND(의도→operator): 격리 LLM 서브콜 — 사용자 요청이 요구하는 action_tool 1개(or none).
    도메인-일반(intent→operator = 값 formalize의 operator판·learn 정의역). 실패=None(안전).

    ★`ask` (2026-08-24·x516 이후 신설): **묻는 문장만** 갈아 끼우는 자리. 기본값은 종전 문장과
      바이트 동일이므로 **라이브 거동은 안 바뀐다** — 프로브가 프롬프트를 베끼지 않게 하려고
      정본에 인자를 낸 것이다([[67]] 사본 금지).
      왜 필요한가: `x516` 이 후보집합 가설을 기각하면서 물음 자체가 **에이전트-프레임**임을
      드러냈다 — *"must the agent CALL"* 은 손님-실행 도구(`submit_transaction` 등)를 옳은
      답으로 **가질 수 없는** 물음이다. 그 프레임이 결손인지를 재려면 문장을 갈아야 한다.
      ⚠[[66]]: 여기에 **케이스 규칙을 넣지 마라**(과거 af8c1e21 이 그러다 098 4/4→0/4).
        일반 어법 교체만 허용된다."""
    if not action_tools or agent is None or la is None:
        return None
    _uall = [i for i, m in enumerate(msgs or [])
             if getattr(m, "role", None) == "user"]
    users = [str(getattr(m, "content", "") or "") for m in msgs
             if getattr(m, "role", None) == "user"][-6:]
    # ★계기 (2026-08-25·거동 변경 0·사용자 지시 *"수리할 방법이 없으면 다음 런을 위해
    #   원인파악을 위한 장치라도 달아두라"*). 016 은 세 격리(x516·x517·x527b)가 전부
    #   gold 0 이었는데, 그 이유가 밝혀진 것은 **창의 범위** 때문이다: 이 서브는 손님 발화
    #   마지막 6개만 보고, 016 이 필요로 하는 자격 요건 축자는 궤적 msg[33]/[45] 로 온다
    #   ⇒ 원리상 창 밖이다. 그 사실이 라이브 로그 어디에도 안 남아서 세 번을 돌아 알았다.
    #   이제 매 호출이 *무엇을 못 봤는지*를 남긴다 — grep 하나로 코퍼스 전체에서 센다.
    #   ⚠판단 0·선택 0: 인덱스 산술과 인쇄뿐이고 프롬프트는 한 글자도 안 바뀐다.
    try:
        print("[T2_SUBWIN] sub=intent_operator_formalize msgs=%d user_msgs=%d used=%d "
              "win_first=%s blind_before=%d"
              % (len(msgs or []), len(_uall), len(users),
                 _uall[-6] if len(_uall) >= 6 else (_uall[0] if _uall else -1),
                 (_uall[-6] if len(_uall) >= 6 else 0)),
              file=sys.stderr, flush=True)
    except Exception:
        pass
    # ⛔의도 분류는 **어디에도 입법하지 않는다** (사용자 지시 2026-08-12: "우리 엔진이나
    #   A2/A3 모두 의도 분류를 하지 않는다"). 이 프롬프트는 순수 질문형으로 남긴다 — 판단은
    #   온전히 격리 LLM 몫이고, 엔진은 답의 집합 소속만 본다([[52]]·[[59]]).
    #   ★사고 기록 (judge6 24 sim 전수 포렌식·-6 중 -6): 여기 "still asking questions,
    #   comparing options ... 이면 none 이라 답하라"는 **케이스 규칙**을 넣었다가(af8c1e21)
    #   의도+질문 복합 발화("I want to refer ... which one?")가 일괄 none 이 되어 [ORDER]/
    #   [ACTION] 푸시가 전면 침묵 — 098 4/4→0/4·099 3/4→1/4. 규칙은 프롬프트에 넣어도
    #   입법이다. 원래 막으려던 무단 개설(071t3)은 이 절로 안 닫혔음도 실측(모델 자발 호출).
    prompt = ("The user is talking to a customer-service agent. Based ONLY on what the user asked, "
              + (ask or ASK_AGENT_CALL) +
              "Reply with the exact tool name, or 'none' if none applies.\n"
              "Tools: " + ", ".join(sorted(action_tools)) + "\n"
              "User said:\n- " + "\n- ".join(u[:300] for u in users) +
              '\nReply JSON only: {"tool": "<name or none>"}')
    try:
        txt = SC.sub_generate(agent, la, UserMessage, prompt, "intent_operator_formalize")
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
        txt = SC.sub_generate(agent, la, UserMessage, prompt, "recommend_operand_verify")
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
        txt = SC.sub_generate(agent, la, UserMessage, prompt, "recommend_formalize")
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
        txt = SC.sub_generate(agent, la, UserMessage, prompt, "reference_criteria_formalize")
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
