#!/usr/bin/env python
"""PRE-ACTION-KB hook (T2_PREKB=1): 종결성 도구 실행 직전 '행동-키' KB 조회 확인.

★표적(C165): 032/033/035 실패 기전 = 에이전트가 KB를 **손님의 문제**로 검색
  ("resolve unpaid statement issue")해 절차 문서를 못 찾고 바로 종결(transfer).
  같은 KB·같은 bm25에서 **행동(도구)-키** 쿼리("transfer to human agent procedure")는
  절차 문서(doc_..._010/011)를 1~2위로 반환한다 — 실측. 즉 정보부재·능력결손이 아니라
  **쿼리-키 실패**이고, 레버는 "행동 이름으로 찾아봤는가"의 결정론 확인이다.

★[[05]]/[[03b]] 감사:
  · 트리거 집합 = tau2 프레임워크 공통 도구(transfer_to_human_agents; 전 도메인 존재)
    ∪ A2 `eplan.finalize_writes`(기존 키 **재사용**·신규 A2 0). 도메인 리터럴 0.
  · 지시하는 쿼리 = 트리거된 도구명에서 기계 유도(suffix 제거→underscore→space+" procedure").
  · 문서를 **찾으라고만** 하고 내용·요구 도구·순서는 일절 제공하지 않는다 — 읽고 따를지,
    어느 프로토콜이 적용되는지는 전부 모델 판단(gold-planting 회피·발견 시험 보존).
  · 검색 실행 주체 = 에이전트(autofetch 금지 선례 준수 — scaffold가 fetch를 대행하지 않음).
★[[10]]: 엔진 몫 = "행동-키 조회 증거가 히스토리에 있는가"의 결정론 판정 + 1회 deny뿐.
★C116/C152: deny 문구 = 복사-가능한 정확한 KB_search 호출 + "Do NOT abandon"(포기 방지).
★cap: fam당 1회(deny 후엔 무조건 통과 — 무한루프·교착 방지. deny=soft임을 인정하는 안전변).

활성화: `import t2_prekb_patch; t2_prekb_patch.apply()` (드라이버: scaffold_get/gate 뒤 최외곽).
"""
import json
import os
import re
import sys

_SUFF = re.compile(r"_\d{3,4}$")
_APPLIED = False
_MARKS = []

# tau2 프레임워크-공통 종결 도구(전 도메인 존재 — 도메인 어휘 아님·DEFAULT_ARG_HINTS 선례)
FRAMEWORK_FINAL = ("transfer_to_human_agents",)

# 증거 스캔에서 무시할 짧은/기능어 토큰(도메인 무관 영어 기능어)
_STOP = {"to", "the", "a", "an", "of", "for", "and", "or", "by", "in", "on", "at", "is"}


def _mark(msg):
    _MARKS.append(msg)
    print("[T2_PREKB] %s" % msg, file=sys.stderr, flush=True)


def marks():
    return list(_MARKS)


def _fam(name):
    return _SUFF.sub("", str(name or ""))


def _trigger_fams(a2):
    """트리거 fam 집합 = 프레임워크 종결 도구 ∪ A2 eplan.finalize_writes ∪ A2 prekb_tools.
    ★C176: `prekb_tools` = finalize 아닌 결과적(consequential) write에 PREKB만 거는 A2 키
    (예: apply_for_credit_card — 클러스터① card_type 오선택: 자격 문서 미열람 채 프리미엄
    카드로 직행. finalize_writes에 넣으면 close-체인 게이트 의미가 오염되므로 별도 키)."""
    fams = {_fam(n) for n in FRAMEWORK_FINAL}
    try:
        for w in ((a2 or {}).get("eplan") or {}).get("finalize_writes") or []:
            fams.add(_fam(w))
    except Exception:
        pass
    try:
        for w in (a2 or {}).get("prekb_tools") or []:
            fams.add(_fam(w))
    except Exception:
        pass
    return fams


def _require_before(a2):
    """★C190 A2 `require_tool_before` = {결과적 도구 base 이름: [선행 필수 도구, ...]}.

    W5 063 실측: 카드 신청 태스크인데 `check_card_application_fit`(자격필터)을 **한 번도
    호출하지 않고** 곧장 Platinum을 권해 신청됐다(gold=Silver). 002/003/006/024는 필터를
    호출했고 **전부 eligible 안에서 골랐다** — 즉 설계는 작동했고 063은 *진입 자체*가 없었다.
    `prekb_tools`는 'KB를 행동-키로 검색했나'만 보므로 이 공백을 못 막는다.
    ★[[05]]: 요구하는 것은 **점검의 실행**이지 결과가 아니다(어느 카드인지 일절 미지정).
    ★[[03b]]: 필터는 에이전트가 호출한다(scaffold가 대행하지 않음)·정답 미주입.
    ★적용 지점: `apply_for_credit_card`는 **손님-측** 도구라 에이전트 호출을 막을 수 없다 —
    C176 `prekb_tools`/C181c utool 피드백과 동일하게 orchestrator 실행 경로(손님 호출 포함)에서
    1회 deny하고 에이전트가 점검 후 재안내하게 한다. cap=fam당 1회(교착 방지·PREKB 선례)."""
    spec = (a2 or {}).get("require_tool_before")
    return spec if isinstance(spec, dict) else {}


def _called_fams(msgs):
    """히스토리에서 실제 호출된 도구 fam 집합(디스패처 내부 이름 포함)."""
    out = set()
    for m in msgs:
        md = m.model_dump() if hasattr(m, "model_dump") else m
        if not isinstance(md, dict):
            continue
        for tc in (md.get("tool_calls") or []):
            nm = _fam(_tc_name(tc))
            if nm:
                out.add(nm)
            a = _tc_args(tc)
            for k in ("agent_tool_name", "discoverable_tool_name"):
                if a.get(k):
                    out.add(_fam(str(a[k])))
    return out


def _tokens(fam):
    return [t for t in fam.split("_") if len(t) >= 4 and t not in _STOP]


def _query_for(fam):
    """지시할 행동-키 쿼리(기계 유도·도메인 리터럴 0)."""
    return fam.replace("_", " ") + " procedure"


def _tc_name(tc):
    return getattr(tc, "name", None) or (tc.get("name") if isinstance(tc, dict) else None)


def _tc_args(tc):
    a = getattr(tc, "arguments", None)
    if a is None and isinstance(tc, dict):
        a = tc.get("arguments")
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except Exception:
            a = {}
    return a if isinstance(a, dict) else {}


def _effective_fams(tc):
    """이 tool_call이 '실행하는' fam들(dispatcher면 내부 이름 포함·unlock은 실행 아님)."""
    nm = _tc_name(tc)
    fam = _fam(nm)
    if fam == "unlock_discoverable_agent_tool":
        return []                                   # 잠금해제=실행 아님(C159 교훈)
    if fam == "call_discoverable_agent_tool":
        inner = _fam(_tc_args(tc).get("agent_tool_name"))
        return [inner] if inner else []
    return [fam]


def _has_evidence(messages, fam):
    """행동-키 조회 증거: ①KB_search 쿼리가 *행동-키*였거나(★C178: fam의 동사 토큰[첫 토큰]
    필수 + 다른 fam 토큰 ≥1 — 구 술어 "아무 토큰 1개"는 credit/card류 범용 토큰이 모든 쿼리에
    있어 W2서 deny 0발화=레버 미검증) ②tool 출력에 fam 리터럴 존재."""
    toks = _tokens(fam)
    verb = (fam.split("_") or [""])[0].lower()          # apply/open/transfer/submit/close…
    rest = [t for t in toks if t != verb]
    for m in messages or []:
        role = getattr(m, "role", None) or (m.get("role") if isinstance(m, dict) else None)
        tcs = getattr(m, "tool_calls", None) or (m.get("tool_calls") if isinstance(m, dict) else None) or []
        for tc in tcs:
            nm = str(_tc_name(tc) or "")
            if nm.startswith("KB_search"):
                q = str(_tc_args(tc).get("query") or "").lower()
                if (len(verb) >= 3 and verb in q
                        and (not rest or any(t in q for t in rest))):
                    return True
        if role == "tool":
            c = getattr(m, "content", None) or (m.get("content") if isinstance(m, dict) else None) or ""
            if fam and fam in str(c):
                return True
    return False


def deny_text(nm, fam):
    q = _query_for(fam)
    return ("Error: [PRE-ACTION-KB] STOP before executing '%s'. Internal procedures sometimes "
            "govern this exact action, and they are documented in the knowledge base under the "
            "ACTION itself, not under the customer's problem. Do this now: "
            "(1) call KB_search(query=\"%s\") and read the results; "
            "(2) if a special procedure applies to the current situation, follow it exactly "
            "(including any internal tools it names) BEFORE '%s'; "
            "(3) if no procedure applies, immediately re-issue this exact same '%s' call. "
            "Do NOT abandon the customer's request." % (nm, q, nm, nm))


def _declared_give_tool(a2):
    """A2가 선언한 '손님에게 도구를 건네는' 디스패처 이름(ABox 유래·엔진 리터럴 0).

    ★[[05]]: `give_discoverable_user_tool`은 banking 전용 이름이라(프레임워크 공통 아님·
    tau2 5도메인 중 banking에만 존재) 엔진에 박으면 도메인 리터럴 위반. A2
    scaffold_get_tools[].follow_up.tool에 이미 선언돼 있으므로 거기서 읽는다.
    미선언 도메인이면 None → 채널 교정 자체를 하지 않음(안전한 no-op)."""
    for t in (a2 or {}).get("scaffold_get_tools") or []:
        if not isinstance(t, dict):
            continue
        nm = (t.get("follow_up") or {}).get("tool")
        if nm:
            return str(nm)
    return None


def _is_agent_regular(env, name):
    """이름이 **에이전트 일반 도구**(직접 호출 가능·discoverable 아님)인지 라이브 레지스트리 판정.

    ★C182c(2026-07-25·W3 099 실측): 에이전트가 `unlock_discoverable_agent_tool
    ('get_user_information_by_name')`을 호출 — 이건 **자기 도구 목록에 이미 있는 일반 도구**다.
    C182 초판은 A2 scaffold 명단만 대조해서 이 경우 일반 분기로 떨어졌고, 그 문구는
    "KB에서 서픽스 포함 정확한 이름을 재확인하라"라 **틀린 처방**이었다(그런 이름은 없다).
    라이브 레지스트리로 판정하면 정확히 "이미 있으니 그냥 호출하라"고 말할 수 있다.
    ★[[05]]: 근거=프레임워크 API(`env.tools`)뿐·도메인 어휘 0."""
    if not env or not name:
        return False
    try:
        at = getattr(env, "tools", None)
        if at is None:
            return False
        if hasattr(at, "has_discoverable_tool") and at.has_discoverable_tool(name):
            return False                      # discoverable = unlock이 정상 경로
        return bool(hasattr(at, "has_tool") and at.has_tool(name))
    except Exception:
        return False


def _is_user_channel(env, name):
    """이름이 **손님-측** discoverable 도구인지 라이브 환경 레지스트리로 판정.

    ★[[05]]: 판정 근거 = tau2 프레임워크 API(`env.user_tools`)뿐 — 도메인 어휘 0.
    guided decoding이 라이브 스키마에서 문법을 뽑는 것과 같은 원리."""
    if not env or not name:
        return False
    try:
        ut = getattr(env, "user_tools", None)
        if ut is None:
            return False
        if hasattr(ut, "has_discoverable_tool"):
            return bool(ut.has_discoverable_tool(name))
        if hasattr(ut, "has_tool"):
            return bool(ut.has_tool(name))
    except Exception:
        return False
    return False


def _notice_done(a2, msgs, fam):
    """★F4(C210): fam의 도구가 A2 notice 게이트(applies_to) 관할이고 notice_text(앞 48자)가
    assistant 발화에 실재하면 True. 판정=A2 데이터 부분문자열만(도메인 리터럴 0)."""
    try:
        for g in (a2 or {}).get("gates") or []:
            if not (isinstance(g, dict) and g.get("kind") == "notice" and g.get("notice_text")):
                continue
            ap = g.get("applies_to")
            tools = ap if isinstance(ap, list) else [ap]
            if not any(_fam(str(t)) == fam for t in tools if t):
                continue
            nt = str(g["notice_text"])[:48]
            for m in msgs:
                if isinstance(m, dict):
                    r, c = m.get("role"), m.get("content")
                else:
                    r, c = getattr(m, "role", None), getattr(m, "content", None)
                if r == "assistant" and nt in str(c or ""):
                    return True
    except Exception:
        return False
    return False


def _is_user_native(env, name):
    """이름이 **손님-측 네이티브** 도구(손님이 직접 실행·discoverable 아님)인지 레지스트리 판정.
    day5 024/010 실측: `apply_for_credit_card`/`submit_referral`이 이 부류 — "Unknown discoverable
    tool" 에러는 이미 비-discoverable을 함의하므로 user_tools.has_tool만으로 충분.
    ★[[05]]: 근거=프레임워크 API(`env.user_tools`)뿐·도메인 어휘 0."""
    if not env or not name:
        return False
    try:
        ut = getattr(env, "user_tools", None)
        return bool(ut is not None and hasattr(ut, "has_tool") and ut.has_tool(name))
    except Exception:
        return False


def _replay_compared(env, name):
    """★P2(C208②·DAY5_PRESCRIPTIONS §P2) 불변식 판정: 이 도구의 ToolMessage.content를 변형하면
    tau2 평가 replay가 깨지는가. replay(`environment.py:374~390`)는 **env 등록 ∧ mutating** 도구만
    깨끗한 env에서 재실행해 content를 정확 비교한다 — 그 대상의 content는 절대 불변이어야 한다
    (day5 024/010 = give_discoverable_user_tool 결과에 GUIDANCE append → ValueError → sim 사망).
    env 미등록(우리 주입 도구·환각 이름)·비-mutating(read)은 replay가 스킵 = append 무해.
    판정불가 시 True(안전측=불변). 판정=레지스트리 질의만(도메인 리터럴 0)."""
    if not env or not name:
        return False
    for tk in (getattr(env, "tools", None), getattr(env, "user_tools", None)):
        if tk is None:
            continue
        try:
            if hasattr(tk, "has_tool") and tk.has_tool(name):
                try:
                    return bool(tk.tool_mutates_state(name))
                except Exception:
                    return True
        except Exception:
            continue
    return False


def _utool_guidance_txt(c, env=None):
    """★P2 사실화(C208② 오진 교정): 'Unknown discoverable tool' 피드백 3-분기 — 구판 단일 문구
    ("you invented it")는 실재 유저-네이티브 도구에 **거짓 피드백**이었다(진짜 오류=채널 오분류).
    반환=덧붙일 문구만(부착 위치는 호출부가 replay-안전성에 따라 결정). 판정=레지스트리·리터럴 0."""
    m = re.search(r"Unknown discoverable tool '([^']+)'", c or "")
    bad = m.group(1) if m else None
    if bad and _is_user_native(env, bad):
        return (" [GUIDANCE] '%s' DOES exist — but it is a tool the customer runs directly "
                "on their side, not a discoverable tool, so it cannot be handed over or "
                "dispatched by you. Do not retry give/call: instruct the customer to run "
                "'%s' themselves with the required arguments." % (bad, bad))
    if bad and _is_agent_regular(env, bad):
        return (" [GUIDANCE] '%s' is already in your own tool list — call it directly "
                "instead of dispatching it." % bad)
    return (" [GUIDANCE] No tool with that exact name exists. Do NOT fall back to verbal "
            "instructions: search the knowledge base for the documented tool that serves "
            "this customer request (query by the capability, e.g. by what the customer is "
            "trying to do), then re-issue give/call with the EXACT documented name.")


def _view_fb(orch, txt, mark):
    """★P2: replay-비교 대상 도구의 피드백은 히스토리 대신 **생성-레벨(뷰 전용) 채널**로 —
    다음 생성의 프롬프트 뷰에만 주입되고 커밋 안 됨(repo 원칙 "생성-레벨=작업버퍼=replay-clean"·
    `_t2_eplan_reminder` 선례 동형). 소비=t2_gate_patch unified."""
    try:
        ag = getattr(orch, "agent", None)
        if ag is None:
            return False
        lst = list(getattr(ag, "_t2_view_fb", None) or [])
        # ★F5(C210·day6 033 [S]): 1회 노출은 in-history append보다 약하다 — 033서 view-fb 1회
        #   직후에도 말-안내 후퇴(day5의 GUIDANCE는 히스토리에 남아 준수됨). K회(기본 2) 연속
        #   생성에 재노출. 항목=[텍스트, 잔여횟수].
        lst.append([txt, int(os.environ.get("T2_FB_VIEW_K", "2"))])
        ag._t2_view_fb = lst
        _mark("view-fb queued (%s)" % mark)
        return True
    except Exception as e:
        _mark("view-fb failed (no-op): %r" % (e,))
        return False


def _atool_guidance(c, a2, env=None):
    """★C182 'Unknown agent tool' 에러에 붙일 처방 문구(순수 함수·selftest 대상).

    017/019 회귀 기전: scaffold-주입 도구(get_reward_discrepancies)를 에이전트가
    unlock_discoverable_agent_tool로 오주소 → env 에러가 '없는 도구'라고만 말해
    직접-호출 재시도 없이 포기. 이름이 A2 scaffold_get_tools 명단에 있으면
    '직접 호출하라', 아니면 'KB에서 정확한(서픽스 포함) 이름을 재확인하라'.

    ★C182b 채널 교정(018/020/022 실측): 반대 방향 오주소 — **손님-측** 도구
    (submit_cash_back_dispute_0589 등)를 agent 채널로 unlock/call. 이 경우
    "그 도구는 손님 것이니 <A2 선언 give-도구>로 건네라"고 채널만 교정한다.
    ★[[03b]] 자기감사: 도구 **선택은 이미 모델이 한 것**(KB서 스스로 발굴한 이름)이고
    엔진은 그 이름이 어느 채널 소속인지만 알려준다 — 어떤 도구를 쓸지·인자·수행 여부는
    전부 모델 몫이므로 gold-planting 아님(C182 직접-호출 분기와 동형).
    엔진은 이름 대조·문자열 덧붙임뿐([[05]] 리터럴 0)."""
    m = re.search(r"Unknown agent tool '([^']+)'", c or "")
    bad = m.group(1) if m else None
    sg = {str(t.get("name")) for t in (a2 or {}).get("scaffold_get_tools") or []
          if isinstance(t, dict)}
    if bad and (bad in sg or _is_agent_regular(env, bad)):
        return (c + " [GUIDANCE] '%s' is NOT a discoverable tool — it is already "
                "available in your regular tool list. Do NOT unlock or dispatch it: "
                "call '%s' directly with its documented parameters." % (bad, bad))
    give = _declared_give_tool(a2)
    if bad and give and _is_user_channel(env, bad):
        return (c + " [GUIDANCE] '%s' is a CUSTOMER-side tool, not an agent tool — "
                "that is why unlocking it fails. You cannot execute it yourself: hand "
                "it to the customer by calling '%s' with discoverable_tool_name='%s' "
                "and the arguments it requires." % (bad, give, bad))
    return (c + " [GUIDANCE] No agent tool with that exact name exists. If you saw "
            "this capability in the knowledge base, re-read the document for the "
            "EXACT documented name (documented names carry a numeric suffix) and "
            "unlock that exact name. If the tool is already in your regular tool "
            "list, call it directly instead of unlocking.")


def apply():
    """BaseOrchestrator._execute_tool_calls 체인-랩(최외곽). T2_PREKB=1일 때만 개입."""
    global _APPLIED
    if _APPLIED:
        return
    from tau2.orchestrator.orchestrator import BaseOrchestrator
    from tau2.data_model.message import ToolMessage
    import t2_gate_patch as _g

    orig_exec = BaseOrchestrator._execute_tool_calls

    def exec2(self, tool_calls):
        if os.environ.get("T2_PREKB") != "1" or not tool_calls:
            return orig_exec(self, tool_calls)
        env = getattr(self, "environment", None)
        a2 = _g._domain_a2(getattr(env, "domain_name", None)) if env is not None else None
        fams = _trigger_fams(a2)
        denied = getattr(self, "_t2_prekb_denied", None)
        if denied is None:
            denied = set()
            self._t2_prekb_denied = denied
        try:
            msgs = self.get_messages() if hasattr(self, "get_messages") else []
        except Exception:
            msgs = []
        # ★C190 선행-도구 요구(063 표적): KB 검색 증거와 **다른 술어** = "그 점검을 실행했나".
        rb = _require_before(a2)
        if rb:
            rbd = getattr(self, "_t2_prekb_rb", None)
            if rbd is None:
                rbd = set()
                self._t2_prekb_rb = rbd
            called = _called_fams(msgs)
            for tc in tool_calls:
                for fam in _effective_fams(tc):
                    need = rb.get(fam)
                    if not need or fam in rbd:
                        continue
                    miss = [n for n in need if _fam(n) not in called]
                    if not miss:
                        continue
                    # ★F1(C210·day6 [S]): 대상 호출이 replay-비교 도구(env 등록∧mutating —
                    #   apply_for_credit_card/submit_referral 등 rb 표적 전부)면 **미실행 deny 스텁을
                    #   커밋할 수 없다** — tau2 replay가 깨끗한 env에서 재실행해 content 정확비교
                    #   (day6 007/010/025 ValueError 3건·재시도 소각 실측). ⇒ 실행은 통과시키고
                    #   점검-지시는 생성-레벨 채널(view-fb)로: "방금 X가 점검 Y 없이 실행됐다 —
                    #   지금 Y를 실행해 결과와 대조·필요시 교정 안내하라". 레버는 사후-점검으로
                    #   약화되지만 측정 정합성이 우선([[19]] 조정=끄기 아님).
                    if _replay_compared(env, getattr(tc, "name", None)):
                        rbd.add(fam)
                        _view_fb(self,
                                 "NOTE: the tool '%s' was just executed WITHOUT the required "
                                 "pre-check %s. Run %s NOW and compare its output with what was "
                                 "just done; if they disagree, tell the customer and correct "
                                 "course before going further."
                                 % (getattr(tc, "name", fam), ", ".join(miss), ", ".join(miss)),
                                 "rb-postcheck")
                        _mark("require_before post-check (replay-safe) fam=%s (missing %s)"
                              % (fam, ",".join(miss)))
                        continue
                    rbd.add(fam)
                    _mark("require_before deny fam=%s (missing %s)" % (fam, ",".join(miss)))
                    # ★F1: 형제-호출 중 replay-비교 대상은 스텁 대신 실행(메인 deny 경로와 동일 규율).
                    _sib2 = [t2 for t2 in tool_calls
                             if t2 is not tc and _replay_compared(env, getattr(t2, "name", None))]
                    _sibres2 = {}
                    if _sib2:
                        try:
                            for _r3 in (orig_exec(self, _sib2) or []):
                                _sibres2[getattr(_r3, "id", None)] = _r3
                        except Exception:
                            _sibres2 = {}
                    out = []
                    for t2 in tool_calls:
                        if t2 is tc:
                            out.append(ToolMessage(
                                id=t2.id, role="tool", requestor=getattr(t2, "requestor", "assistant"),
                                error=True,
                                content=("Error: [CHECK-FIRST] '%s' was not carried out yet. Before "
                                         "this action is finalized you must run %s and base your "
                                         "recommendation on what it returns — it filters the "
                                         "documented catalog against the requirements the customer "
                                         "actually stated. Call %s now, tell the customer which "
                                         "options it reports, and only then have this action "
                                         "re-issued. Do NOT abandon the customer's request."
                                         % (", ".join(miss), ", ".join(miss), ", ".join(miss)))))
                        elif getattr(t2, "id", None) in _sibres2:
                            out.append(_sibres2[t2.id])
                        else:
                            out.append(ToolMessage(
                                id=t2.id, role="tool", requestor=getattr(t2, "requestor", "assistant"),
                                error=True,
                                content="Error: [CHECK-FIRST] deferred: resolve the check above "
                                        "first, then re-issue this call."))
                    return out
        hit = None                                   # (tc, nm, fam) 최초 트리거
        for tc in tool_calls:
            for fam in _effective_fams(tc):
                if fam in fams and fam not in denied and not _has_evidence(msgs, fam):
                    # ★F4(C210·day6 004 [S]): A2 notice 게이트가 이 도구를 관장하고 **notice가
                    #   이미 공표**됐다면 deny 면제 — notice 프로토콜(KB 확인·전 절차 시도 후에만
                    #   공표)이 PREKB의 확인 의무를 포섭하며, 특히 동의-터미널 직후의 deny는
                    #   마지막 행동 턴을 소각한다(004: 호출→deny→종료·notice-레이스의 공범).
                    #   판정=A2 notice_text 부분문자열(도메인 리터럴 0).
                    if _notice_done(a2, msgs, fam):
                        _mark("deny waived fam=%s (notice already announced)" % fam)
                        denied.add(fam)
                        continue
                    hit = (tc, _tc_name(tc), fam)
                    break
            if hit:
                break
        if hit is None:
            out = orig_exec(self, tool_calls)
            # ★C181b: 가공 user-도구 피드백(012/075/015 패밀리 — navigate_to_section 등 반복 발명
            #   →env 에러 후 모델이 말-안내로 후퇴). env 자체 에러 문구(도메인-일반)를 트리거로
            #   처방적 지시를 덧붙임(C116). 목록 제공=spoon-feeding이라 금지·KB 재검색 지시만.
            #   cap 2/sim. 엔진은 문자열 덧붙임뿐(도구 목록·이름 미주입).
            try:
                nfb = getattr(self, "_t2_utool_fb", 0)
                afb = getattr(self, "_t2_atool_fb", 0)
                _id2nm = {getattr(t, "id", None): getattr(t, "name", None)
                          for t in (tool_calls or [])}
                for r in (out or []):
                    c = getattr(r, "content", "") or ""
                    _rnm = _id2nm.get(getattr(r, "id", None))
                    # ★P2(C208②): give/call 디스패처는 mutating=replay-비교 대상 — content 불변.
                    #   피드백은 사실화 3-분기 문구(_utool_guidance_txt)로 생성-레벨 채널에 싣는다.
                    #   비-비교 도구(이론상)면 구판대로 append(거동 최소 변화).
                    if nfb < 2 and "Unknown discoverable tool" in c:
                        _txt = _utool_guidance_txt(c, env)
                        if _replay_compared(env, _rnm):
                            _view_fb(self, "About your last '%s' call, which failed:%s"
                                     % (_rnm, _txt), "utool")
                        else:
                            r.content = c + _txt
                        self._t2_utool_fb = nfb + 1
                        _mark("utool feedback (n=%d)" % (nfb + 1))
                        break
                    # ★C182: scaffold-주입 도구를 unlock/dispatcher 채널로 오주소(017/019
                    #   회귀 기전 — dispatcher-규율이 주입 도구까지 discoverable로 오분류하게
                    #   유도·에러 후 직접 호출 미시도·포기). 트리거=env 에러 문구(도메인-일반)·
                    #   이름 판정=A2 scaffold_get_tools 명단 대조만(엔진 리터럴 0). cap 2/sim.
                    if afb < 2 and "Unknown agent tool" in c:
                        newc = _atool_guidance(c, a2, env)
                        if newc:
                            if _replay_compared(env, _rnm):
                                _view_fb(self, "About your last '%s' call, which failed:%s"
                                         % (_rnm, newc[len(c):]), "atool")
                            else:
                                r.content = newc
                            self._t2_atool_fb = afb + 1
                            _mark("atool feedback (n=%d)" % (afb + 1))
                            break
            except Exception:
                pass
            return out
        tc0, nm0, fam0 = hit
        denied.add(fam0)                             # cap: fam당 1회
        _mark("deny fam=%s (no action-keyed KB evidence) — instructing search" % fam0)
        # ★P2(C208② 잠재위험 봉합): 배치의 **비-hit mutating(=replay-비교 대상)** 호출을 미실행
        #   deferred-스텁으로 남기면 replay가 그 호출을 재실행해 content 불일치 = 같은 사망 계열.
        #   → 그런 형제-호출은 정상 실행하고 결과를 그대로 배치한다(비-비교 형제만 스텁 유지).
        _sib = [tc for tc in tool_calls
                if tc is not tc0 and _replay_compared(env, getattr(tc, "name", None))]
        _sibres = {}
        if _sib:
            try:
                for _r2 in (orig_exec(self, _sib) or []):
                    _sibres[getattr(_r2, "id", None)] = _r2
                _mark("deny: executed %d replay-compared sibling call(s) (no stub)" % len(_sib))
            except Exception as _se:
                _sibres = {}
                _mark("deny sibling exec failed (no-op): %r" % (_se,))
        out = []
        for tc in tool_calls:
            if tc is tc0:
                out.append(ToolMessage(id=tc.id, role="tool", requestor="assistant",
                                       error=True, content=deny_text(nm0, fam0)))
            elif getattr(tc, "id", None) in _sibres:
                out.append(_sibres[tc.id])
            else:
                out.append(ToolMessage(
                    id=tc.id, role="tool", requestor="assistant", error=True,
                    content="Error: [PRE-ACTION-KB] deferred: resolve the check above first, "
                            "then re-issue this call."))
        return out

    BaseOrchestrator._execute_tool_calls = exec2
    _APPLIED = True
    _mark("patch applied (outermost exec wrap)")


if __name__ == "__main__":
    # 오프라인 selftest(서버·tau2 불요 — 순수 함수만)
    a2 = {"eplan": {"finalize_writes": ["close_credit_card_account"]},
          "prekb_tools": ["apply_for_credit_card"]}
    fams = _trigger_fams(a2)
    assert "transfer_to_human_agents" in fams and "close_credit_card_account" in fams
    assert "apply_for_credit_card" in fams          # C176: prekb_tools 키
    assert _fam("close_credit_card_account_7834") == "close_credit_card_account"
    # dispatcher 내부 이름·unlock 제외
    tc_call = {"name": "call_discoverable_agent_tool",
               "arguments": {"agent_tool_name": "close_credit_card_account_7834"}}
    assert _effective_fams(tc_call) == ["close_credit_card_account"]
    assert _effective_fams({"name": "unlock_discoverable_agent_tool",
                            "arguments": {"agent_tool_name": "close_credit_card_account_7834"}}) == []
    # 증거: 행동-키 쿼리
    msgs_q = [{"role": "assistant",
               "tool_calls": [{"name": "KB_search_bm25",
                               "arguments": {"query": "transfer to human agent procedure"}}]}]
    assert _has_evidence(msgs_q, "transfer_to_human_agents")
    # 증거: tool 출력에 리터럴
    msgs_doc = [{"role": "tool", "content": "... the regular transfer_to_human_agents tool ..."}]
    assert _has_evidence(msgs_doc, "transfer_to_human_agents")
    # 무증거: 문제-기반 쿼리(033 실제)
    msgs_bad = [{"role": "assistant",
                 "tool_calls": [{"name": "KB_search",
                                 "arguments": {"query": "resolve unpaid statement issue"}}]},
                {"role": "tool", "content": "1. True Blue Account: Dedicated Support ..."}]
    assert not _has_evidence(msgs_bad, "transfer_to_human_agents")
    # ★C178 회귀: 범용 토큰(credit/card)만 있는 쿼리는 증거 아님(W2 002 실제 쿼리)
    msgs_generic = [{"role": "assistant",
                     "tool_calls": [{"name": "KB_search",
                                     "arguments": {"query": "Rho-Bank credit cards with highest cash back rewards"}}]}]
    assert not _has_evidence(msgs_generic, "apply_for_credit_card")
    # 행동-키 쿼리(동사+타 토큰)는 증거 맞음
    msgs_act = [{"role": "assistant",
                 "tool_calls": [{"name": "KB_search",
                                 "arguments": {"query": "apply for credit card procedure"}}]}]
    assert _has_evidence(msgs_act, "apply_for_credit_card")
    # deny 문구: 복사-가능 쿼리 포함·절차 내용(정답) 미포함
    t = deny_text("transfer_to_human_agents", "transfer_to_human_agents")
    assert 'KB_search(query="transfer to human agents procedure")' in t
    assert "initial_transfer" not in t and "Do NOT abandon" in t
    # ★C182: Unknown-agent-tool 처방 문구
    a2sg = {"scaffold_get_tools": [{"name": "get_reward_discrepancies"},
                                   {"name": "check_cli_eligibility"}]}
    err = "Error: Unknown agent tool 'get_reward_discrepancies'. This tool is not available."
    g1 = _atool_guidance(err, a2sg)
    assert "call 'get_reward_discrepancies' directly" in g1 and "Do NOT unlock" in g1
    err2 = "Error: Unknown agent tool 'submit_cash_back_dispute'. This tool is not available."
    g2 = _atool_guidance(err2, a2sg)
    assert "numeric suffix" in g2 and "directly" in g2 and "call 'submit" not in g2
    g3 = _atool_guidance("Error: something else entirely", a2sg)
    assert "numeric suffix" in g3          # 이름 미추출도 일반 처방은 붙음
    # scaffold 명단 비어도 크래시 없음
    assert _atool_guidance(err, {}) and "numeric suffix" in _atool_guidance(err, {})

    # ★C182b 채널 교정(018/020/022): 손님-측 도구를 agent 채널로 오주소
    class _FakeUT:
        def __init__(self, names): self._n = set(names)
        def has_discoverable_tool(self, n): return n in self._n

    class _FakeEnv:
        def __init__(self, names): self.user_tools = _FakeUT(names)

    a2fu = dict(a2sg)
    a2fu["scaffold_get_tools"] = [
        {"name": "get_reward_discrepancies",
         "follow_up": {"tool": "give_discoverable_user_tool"}},
        {"name": "check_cli_eligibility"}]
    assert _declared_give_tool(a2fu) == "give_discoverable_user_tool"
    assert _declared_give_tool(a2sg) is None           # follow_up 미선언 → None
    envu = _FakeEnv(["submit_cash_back_dispute_0589"])
    assert _is_user_channel(envu, "submit_cash_back_dispute_0589")
    assert not _is_user_channel(envu, "file_credit_card_transaction_dispute_4829")
    assert not _is_user_channel(None, "submit_cash_back_dispute_0589")
    erru = "Error: Unknown agent tool 'submit_cash_back_dispute_0589'. This tool is not available."
    g4 = _atool_guidance(erru, a2fu, envu)
    assert "CUSTOMER-side tool" in g4 and "give_discoverable_user_tool" in g4
    assert "discoverable_tool_name='submit_cash_back_dispute_0589'" in g4
    # 손님-측 아님(진짜 agent 도구 오타) → 일반 처방 유지
    erra = "Error: Unknown agent tool 'file_credit_card_transaction_dispute_482'. This tool is not available."
    assert "numeric suffix" in _atool_guidance(erra, a2fu, envu)
    # scaffold 분기가 채널 분기보다 우선(주입 도구는 직접 호출)
    assert "call 'get_reward_discrepancies' directly" in _atool_guidance(
        err, a2fu, _FakeEnv(["get_reward_discrepancies"]))
    # A2가 give-도구 미선언이면 채널 교정 안 함(도메인 리터럴 0의 대가·안전 no-op)
    assert "numeric suffix" in _atool_guidance(erru, a2sg, envu)
    # env 없어도 크래시 없음
    assert "numeric suffix" in _atool_guidance(erru, a2fu, None)
    # ★C182c 에이전트 일반 도구 오주소(099 실측: unlock('get_user_information_by_name'))
    class _FakeAT:
        def __init__(self, reg, disc): self._r=set(reg); self._d=set(disc)
        def has_tool(self, n): return n in self._r or n in self._d
        def has_discoverable_tool(self, n): return n in self._d
    class _FakeEnv2:
        def __init__(self, reg, disc, user=()):
            self.tools=_FakeAT(reg, disc); self.user_tools=_FakeAT((), user)
    env2=_FakeEnv2(reg=["get_user_information_by_name"], disc=["open_bank_account_4821"],
                   user=["submit_cash_back_dispute_0589"])
    assert _is_agent_regular(env2, "get_user_information_by_name")
    assert not _is_agent_regular(env2, "open_bank_account_4821")   # discoverable=unlock이 정상
    assert not _is_agent_regular(env2, "submit_cash_back_dispute_0589")
    assert not _is_agent_regular(None, "x")
    err099="Error: Unknown agent tool 'get_user_information_by_name'. This tool is not available."
    g=_atool_guidance(err099, {}, env2)
    assert "already" in g and "call 'get_user_information_by_name' directly" in g
    assert "numeric suffix" not in g          # 구판의 틀린 처방이 안 나와야
    # 손님-측 분기가 여전히 우선 동작
    a2g={"scaffold_get_tools":[{"name":"x","follow_up":{"tool":"give_discoverable_user_tool"}}]}
    gu=_atool_guidance("Error: Unknown agent tool 'submit_cash_back_dispute_0589'. x", a2g, env2)
    assert "CUSTOMER-side tool" in gu
    # ★C190 require_tool_before(063 표적)
    a2rb = {"require_tool_before": {"apply_for_credit_card": ["check_card_application_fit"]}}
    assert _require_before(a2rb) == {"apply_for_credit_card": ["check_card_application_fit"]}
    assert _require_before({}) == {}
    assert _require_before({"require_tool_before": "nope"}) == {}
    # 호출 이력 수집: 평문 도구 + dispatcher 내부 이름 + 서픽스 제거
    hist = [{"role": "assistant", "tool_calls": [{"name": "check_card_application_fit"}]},
            {"role": "assistant", "tool_calls": [
                {"name": "call_discoverable_agent_tool",
                 "arguments": {"agent_tool_name": "open_bank_account_4821"}}]}]
    cf = _called_fams(hist)
    assert "check_card_application_fit" in cf and "open_bank_account" in cf
    assert _called_fams([]) == set()
    # 손님-측 apply도 effective fam으로 잡혀야(게이트 대상)
    assert _effective_fams({"name": "apply_for_credit_card"}) == ["apply_for_credit_card"]
    # 미호출이면 결핍 검출 · 호출했으면 통과
    need = _require_before(a2rb)["apply_for_credit_card"]
    assert [n for n in need if _fam(n) not in _called_fams([])] == ["check_card_application_fit"]
    assert [n for n in need if _fam(n) not in cf] == []
    # ★C191 디스패처-중첩 write에도 걸려야(070/099/046: open/submit/pay는 call_discoverable 경유)
    a2c = {"require_tool_before": {"open_bank_account": ["get_all_user_accounts_by_user_id"]}}
    tc_open = {"name": "call_discoverable_agent_tool",
               "arguments": {"agent_tool_name": "open_bank_account_4821"}}
    assert _effective_fams(tc_open) == ["open_bank_account"]          # 게이트 대상으로 인식
    hist_no = [{"role": "assistant", "tool_calls": [{"name": "KB_search"}]}]
    need = _require_before(a2c)["open_bank_account"]
    assert [n for n in need if _fam(n) not in _called_fams(hist_no)] == ["get_all_user_accounts_by_user_id"]
    # 디스패처로 실행한 read도 충족으로 인정(서픽스 제거)
    hist_yes = [{"role": "assistant", "tool_calls": [
        {"name": "call_discoverable_agent_tool",
         "arguments": {"agent_tool_name": "get_all_user_accounts_by_user_id_3847"}}]}]
    assert [n for n in need if _fam(n) not in _called_fams(hist_yes)] == []
    # unlock만 한 경우는 '실행'이 아니므로 여전히 결핍(C159 규율)
    hist_unlock = [{"role": "assistant", "tool_calls": [
        {"name": "unlock_discoverable_agent_tool",
         "arguments": {"agent_tool_name": "get_all_user_accounts_by_user_id_3847"}}]}]
    assert _called_fams(hist_unlock) >= {"get_all_user_accounts_by_user_id"}   # 이름은 수집됨(관대)
    print("selftest OK · trigger_fams=%s" % sorted(fams))
