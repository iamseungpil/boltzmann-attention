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


def _atool_guidance(c, a2):
    """★C182 'Unknown agent tool' 에러에 붙일 처방 문구(순수 함수·selftest 대상).

    017/019 회귀 기전: scaffold-주입 도구(get_reward_discrepancies)를 에이전트가
    unlock_discoverable_agent_tool로 오주소 → env 에러가 '없는 도구'라고만 말해
    직접-호출 재시도 없이 포기. 이름이 A2 scaffold_get_tools 명단에 있으면
    '직접 호출하라', 아니면 'KB에서 정확한(서픽스 포함) 이름을 재확인하라'.
    엔진은 이름 대조·문자열 덧붙임뿐([[05]] 리터럴 0). None=미변경."""
    m = re.search(r"Unknown agent tool '([^']+)'", c or "")
    bad = m.group(1) if m else None
    sg = {str(t.get("name")) for t in (a2 or {}).get("scaffold_get_tools") or []
          if isinstance(t, dict)}
    if bad and bad in sg:
        return (c + " [GUIDANCE] '%s' is NOT a discoverable tool — it is already "
                "available in your regular tool list. Do NOT unlock or dispatch it: "
                "call '%s' directly with its documented parameters." % (bad, bad))
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
        hit = None                                   # (tc, nm, fam) 최초 트리거
        for tc in tool_calls:
            for fam in _effective_fams(tc):
                if fam in fams and fam not in denied and not _has_evidence(msgs, fam):
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
                for r in (out or []):
                    c = getattr(r, "content", "") or ""
                    if nfb < 2 and "Unknown discoverable tool" in c:
                        r.content = (c + " [GUIDANCE] That tool name does not exist — you "
                                     "invented it. Do NOT fall back to verbal instructions: "
                                     "search the knowledge base for the documented tool that "
                                     "serves this customer request (query by the capability, "
                                     "e.g. 'travel notification tool' / 'referral link tool'), "
                                     "then re-issue give/call with the EXACT documented name.")
                        self._t2_utool_fb = nfb + 1
                        _mark("utool feedback appended (n=%d)" % (nfb + 1))
                        break
                    # ★C182: scaffold-주입 도구를 unlock/dispatcher 채널로 오주소(017/019
                    #   회귀 기전 — dispatcher-규율이 주입 도구까지 discoverable로 오분류하게
                    #   유도·에러 후 직접 호출 미시도·포기). 트리거=env 에러 문구(도메인-일반)·
                    #   이름 판정=A2 scaffold_get_tools 명단 대조만(엔진 리터럴 0). cap 2/sim.
                    if afb < 2 and "Unknown agent tool" in c:
                        newc = _atool_guidance(c, a2)
                        if newc:
                            r.content = newc
                            self._t2_atool_fb = afb + 1
                            _mark("atool feedback appended (n=%d)" % (afb + 1))
                            break
            except Exception:
                pass
            return out
        tc0, nm0, fam0 = hit
        denied.add(fam0)                             # cap: fam당 1회
        _mark("deny fam=%s (no action-keyed KB evidence) — instructing search" % fam0)
        out = []
        for tc in tool_calls:
            if tc is tc0:
                out.append(ToolMessage(id=tc.id, role="tool", requestor="assistant",
                                       error=True, content=deny_text(nm0, fam0)))
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
    print("selftest OK · trigger_fams=%s" % sorted(fams))
