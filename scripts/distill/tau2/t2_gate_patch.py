#!/usr/bin/env python
"""tau2 게이트 hook: BaseOrchestrator._execute_tool_calls 몽키패치 — A2-구동(키스톤 후).

★엔진 = `gate_interpreter.GateInterpreter`(벤치-일반·도메인 분기 0). 이 패치는 wiring만:
  에이전트 툴콜을 실행 *전* GateInterpreter로 검사, deny면 실행 없이 게이트 메시지를
  ToolMessage(error)로 반환, allow면 원본 실행 후 결과로 게이트 상태 갱신.
도메인 활성화·도구셋·autofetch producer·식별arg-types·placeholder = 전부 `a2/<domain>.gate.json`서
로드(env.domain_name로 선택). 코드 하드코딩(retail 도구명·GATE_DOMAINS) 폐기(2026-06-21 키스톤).

활성화: `import t2_gate_patch; t2_gate_patch.apply()`. 게이트는 orchestrator 인스턴스당 1개.
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate_interpreter import (  # noqa: E402
    observe_tools,
    GateInterpreter, auth_satisfier_tools, load_domain_a2, resolvers_from_env,
    candidate_summary, nested_candidate_summary, compute_facts)

# ── ★P7(C208⑥·DAY5_PRESCRIPTIONS §P7): 레버 무음실패 금지 — try/except로 삼켜진 레버 예외가
#    "정상"으로 위장하는 것을 런 종료 요약이 자동 고발한다(day5 unavail 0/223 전량 NameError 실측).
#    카운터만(판단 0·도메인 무관). 종료 시 [T2_LEVER_HEALTH] 1줄.
_LEVER_HEALTH = {}


def _lever_health(lever, kind):
    _LEVER_HEALTH.setdefault(lever, {}).setdefault(kind, 0)
    _LEVER_HEALTH[lever][kind] += 1


def _lever_health_report():
    for lv, c in sorted(_LEVER_HEALTH.items()):
        line = " ".join("%s=%d" % (k, v) for k, v in sorted(c.items()))
        flag = "  ⚠ALL-SKIPPED" if (c.get("skipped") and not c.get("ok")) else ""
        print("[T2_LEVER_HEALTH] %s: %s%s" % (lv, line, flag), file=sys.stderr, flush=True)


import atexit  # noqa: E402
atexit.register(_lever_health_report)


def _dyn_mt_target(err_str, margin=64, floor=256):
    """★P1(C208①) 순수함수: vLLM CWE 에러 원문에서 (model_max, input_tokens)를 파싱해
    새 max_tokens를 계산. 파싱 실패 또는 플로어 미만(진짜 창 소진) = None → graceful-stop.
    추정 0(에러가 정확한 수를 준다)·도메인 무관."""
    m = re.search(r"maximum context length is (\d+) tokens and your "
                  r"request has (\d+) input tokens", str(err_str or ""))
    if not m:
        return None
    new_mt = int(m.group(1)) - int(m.group(2)) - int(margin)
    return new_mt if new_mt >= int(floor) else None

# ── 도메인-일반 기본값 (A2가 override·enrich; retail/도메인 하드코딩 아님) ──
# ★[[05]] 감사(2026-07-13): 도메인-일반 식별 토큰만 엔진에. 도메인-특화 어휘
#   (order_id·item=retail·reservation=airline)는 A2 identifying_arg_types로 이관 —
#   엔진 DEFAULT가 도메인-union이면 새 도메인(banking)에 retail/airline 어휘가 누수됨.
#   'id'가 *_id 계열(order_id·item_ids·reservation_id) 전부 포괄하므로 이관은 behavior-preserving.
#   payment·address = 커머스-교차 일반(retail∩airline)이라 엔진 잔류. 도메인 어휘는 A2가 union.
DEFAULT_ARG_HINTS = ("email", "name", "zip", "user_id", "username", "id",
                     "payment", "address", "phone")
DEFAULT_PLACEHOLDERS = {
    # 도메인-일반 placeholder만(엔진). 도메인-특화 포맷(#W0000000=retail 주문-id)은
    # A2 placeholders로 이관(2026-07-13 [[05]] 감사). 아래는 전 도메인 공통 test-값.
    "something@example.com", "jane_doe@example.com",
    "john.doe@example.com", "johndoe@example.com", "john@example.com",
    "jane@example.com", "user@example.com", "test@example.com",
    "example@example.com", "123 Main St", "123 Main Street",
    "ABC123", "XYZ789",  # 도메인-일반 영숫자 placeholder 패턴 (구 airline A2서 이관·minimize-A2)
}
PROV_ARG_HINT = DEFAULT_ARG_HINTS          # 호환 alias
COMMON_PLACEHOLDERS = DEFAULT_PLACEHOLDERS  # 호환 alias

# ★tool_choice='required' 강제 생성의 max_tokens 하한 (2026-07-23·vLLM #19051/#36794 근본원인):
#   강제 tool-call JSON이 절단되면 hermes 파서 EOF→오도성 400. 라이브(미설정) 무영향·소형설정 방어.
_FORCE_MIN_TOKENS = int(os.environ.get("T2_FORCE_MIN_TOKENS", "1024"))

_A2_CACHE = {}


def obs_tools_g(gate):
    """live GateInterpreter -> observe 대상 도구 집합 (A2-구동)."""
    try:
        return observe_tools(getattr(gate, 'gates', []) or [])
    except Exception:
        return set()


def _domain_a2(domain):
    """env.domain_name → augmented A2 dict (없으면 None=게이트 비활성). 캐시."""
    if domain in _A2_CACHE:
        return _A2_CACHE[domain]
    a2 = load_domain_a2(domain) if domain else None
    if a2 is not None:
        a2 = dict(a2)
        a2["_auth_tools"] = auth_satisfier_tools(a2["gates"])
        a2["_observe_tools"] = observe_tools(a2["gates"])
        a2["_hints"] = tuple(set(DEFAULT_ARG_HINTS) | set(a2.get("identifying_arg_types") or ()))
        a2["_placeholders"] = set(a2.get("placeholders") or ()) | DEFAULT_PLACEHOLDERS
        a2["_producer"] = (a2.get("producers") or {}).get("authenticated_user_record")
        # ⚠ deprecated(NOTICE-PERGATE 2026-07-11): first-notice 스칼라 — 진단 스크립트
        #   호환용으로만 보존. ★신규 소비 금지 — 게이트 판정은 per-gate 커링(check callable)로.
        a2["_notice_text"] = next(
            (g.get("notice_text") for g in a2["gates"] if g.get("kind") == "notice"), "")
    _A2_CACHE[domain] = a2
    return a2


def _flatten(v):
    """인자값의 leaf 스칼라들. ★JSON-문자열도 풀어서 leaf까지 간다(2026-07-16 버그픽스):
    구조화 인자(예: {"date_of_birth": ..., "phone_number": ...})가 **문자열**로 오면 예전엔
    JSON 덩어리 전체를 문맥서 찾아 **항상 실패 → 전부 '날조' 오판**했다(라이브 실측: 우리
    verify_identity의 정당한 호출이 매번 반려됨). leaf(11/03/1990·312-555-0481)는 문맥에 실재한다."""
    if isinstance(v, str):
        s = v.strip()
        if s[:1] in "[{" and s[-1:] in "]}":
            try:
                yield from _flatten(json.loads(s))
                return
            except Exception:
                pass
        yield v
    elif isinstance(v, (list, tuple)):
        for x in v:
            yield from _flatten(x)
    elif isinstance(v, dict):
        for x in v.values():
            yield from _flatten(x)
    else:
        yield v


def _hint_hit(k, hints):
    """인자명이 식별자류인가. ★부분문자열 금지(2026-07-16 버그픽스): `"id" in "provided"` = True라
    `provided`가 식별자로 오판됐다. 토큰 분해 후 **토큰이 힌트로 시작**할 때만(→ `address1`·`item_ids`
    같은 접미변형은 유지, `provided`·`valid_until`류 오탐은 제거)."""
    toks = [t for t in re.split(r"[^a-z0-9]+", str(k).lower()) if t]
    return any(t.startswith(h) for t in toks for h in hints)


def _args_dict(tc):
    """ToolCall.arguments 를 dict로 (string JSON도 robust 파싱)."""
    a = getattr(tc, "arguments", None)
    if isinstance(a, dict):
        return a
    if isinstance(a, str):
        try:
            return json.loads(a)
        except Exception:
            return {}
    return {}


# ─── ★L1 bad_words 블랙리스트 (스키마-example + placeholder; 동적=세션-flagged) ───
_SUCH_AS_RE = re.compile(r"such as ['\"]([^'\"]+)['\"]|e\.g\.,?\s*['\"]?([^'\".)]+)", re.I)


def _blacklist_worthy(v):
    """generic 단어(John·CA·12345) 차단 회피: len>=6 & (이메일/숫자 포함) = ID/placeholder형만."""
    v = (v or "").strip()
    return len(v) >= 6 and (("@" in v) or any(c.isdigit() for c in v))


def _static_blacklist(tools, placeholders=None):
    bl = set(placeholders or DEFAULT_PLACEHOLDERS)
    for t in (tools or []):
        try:
            txt = json.dumps(t.openai_schema)
        except Exception:
            txt = str(getattr(t, "description", "") or "")
        for m in _SUCH_AS_RE.finditer(txt):
            v = next((g for g in m.groups() if g), "").strip()
            if _blacklist_worthy(v):
                bl.add(v)
    return bl


def _context_text(orch):
    """provenance 출처 = 모든 user 발화 + 도구 출력(assistant 제외)."""
    parts = []
    try:
        for m in orch.get_messages():
            r = getattr(m, "role", None)
            c = getattr(m, "content", None)
            if r in ("user", "tool") and c is not None:
                parts.append(c if isinstance(c, str) else str(c))
    except Exception:
        pass
    return " ".join(parts).lower()


def _ctx_with_toolnames(agent, ctx):
    """provenance ctx에 **에이전트에게 제시된 도구 이름**을 추가 (§15 오탐 수정·2026-07-16).
    도구 이름의 출처는 스키마(모델에게 제시됨)이지 대화가 아니다 — 라이브 실측: 에이전트가
    `unlock_discoverable_agent_tool(agent_tool_name="get_reward_discrepancies")`로 우리 도구를
    호출하려 하자 PROV가 "문맥에 없음=invented"로 11회 반려(ctl_20260716_2230). ★이름만 추가한다 —
    스키마 전체(설명·예시값)를 넣으면 C47(예시값=복사 원천 47%)의 날조 재료까지 정당화된다."""
    try:
        names = " ".join(sorted({str(getattr(t, "name", "") or "").lower()
                                 for t in (getattr(agent, "tools", None) or [])}))
        return ctx + " " + names if names else ctx
    except Exception:
        return ctx


def _ctx_has(s, ctx):
    """값 s의 ctx-매칭 (PROV-RESCUE-PERARG ②: id '#'-접두 정규화).
    '#W8665881' vs ctx의 'w8665881'(사용자 발화) = 접두 불일치 거짓양성 fab(t17 1차 방아쇠) →
    '#' 접두만 벗겨 재매칭. 정규화는 '#' 하나에 한정(과잉 정규화 금지)·strip 후에도 4자 이상일 때만."""
    if s.lower() in ctx:
        return True
    t = s.lstrip("#")
    return t != s and len(t) >= 4 and t.lower() in ctx


def _provenance_deny(tc, ctx, hints=DEFAULT_ARG_HINTS):
    """identifying 인자값이 컨텍스트에 없으면 fabricated → (gate, reason) 반환, 아니면 None."""
    args = _args_dict(tc)
    if not args:
        return None
    for k, v in args.items():
        if not _hint_hit(k, hints):
            continue
        for val in _flatten(v):
            s = str(val).strip()
            if len(s) < 4:
                continue
            if not _ctx_has(s, ctx):
                return ("PROVENANCE_R1B",
                        f"argument '{k}'='{s}' was not provided by the user nor returned by any tool — it looks invented "
                        "(possibly copied from a schema example value). Do NOT call any tool with a guessed/placeholder value. "
                        "Instead OBTAIN the real value first: if a lookup/getter tool can produce it "
                        "(e.g. call a getter to retrieve the user's records, payment methods, or addresses), call that and read the value from its output; "
                        "otherwise ASK the user for it.")
    return None


def _autofetch_text(self, orig, gate, producer):
    """T2_AUTOFETCH: provenance-deny 시, 인증됐으면 A2 producer(getter)를 결정론 호출해
    그 출력을 텍스트로 반환(모델에 *실값* 제공). = '날조-FIRST' default를 엔진이 결정론으로 메움.
    producer = A2 producers.authenticated_user_record {tool, args_from:{argname:@auth_user}}.
    decidable·도메인-일반(도구·인자 = A2서 도출·airline swap = 동일 메커니즘)·side-effect 0(getter)."""
    if producer is None or getattr(gate, "auth_user", None) is None:
        return ""
    try:
        from tau2.data_model.message import ToolCall
        pargs = {k: (gate.auth_user if v == "@auth_user" else v)
                 for k, v in (producer.get("args_from") or {}).items()}
        ptc = ToolCall(id="autofetch", name=producer["tool"],
                       arguments=pargs, requestor="assistant")
        out = orig(self, [ptc])
        if out and not getattr(out[0], "error", False):
            return ("\nI have fetched the authenticated user's actual record for you — copy a REAL value "
                    "from it, never a placeholder: " + _content_str(out[0])[:1800])
    except Exception:
        pass
    return ""


def _call_key(tc):
    """(도구명, 정규화 인자) = 동일-호출 식별 (retry-loop 탐지·decidable)."""
    return (getattr(tc, "name", "") or "") + "::" + json.dumps(_args_dict(tc), sort_keys=True, ensure_ascii=False)


def apply():
    from tau2.orchestrator.orchestrator import BaseOrchestrator

    orig = BaseOrchestrator._execute_tool_calls

    def gated(self, tool_calls):
        env = self.environment
        a2 = _domain_a2(getattr(env, "domain_name", None))
        if a2 is None:
            return orig(self, tool_calls)
        # T2_GATE_KINDS = 측정-격리 플래그(콤마구분 kind 화이트리스트). 미지정=전체.
        # 예 "preconditions"=G5만(flow-discipline 격리·floor 대비 confound 배제). 도메인-일반.
        _kinds = os.environ.get("T2_GATE_KINDS")
        _gate_list = a2["gates"]
        if _kinds:
            _allow = {k.strip() for k in _kinds.split(",") if k.strip()}
            _gate_list = [g for g in a2["gates"] if g.get("kind") in _allow]
        gate = getattr(self, "_t2_gate", None)
        if gate is None:
            gate = self._t2_gate = GateInterpreter(_gate_list, resolvers=resolvers_from_env(env))
        auth_tools = a2["_auth_tools"]
        producer = a2["_producer"]
        hints = a2["_hints"]
        last_user = _last_user_text(self)
        # ★NOTICE-PERGATE: 문구-매개 클로저(커링) — check()가 게이트별 notice_text로 평가.
        tms = lambda text: _transfer_msg_sent(self, text)  # noqa: E731

        # T2_PRESENT_READS=1 = REPLAY-SAFE present: 후보-producer 읽기응답에 clean 요약 덧붙임(deny 아님·측정 arm).
        present_on = os.environ.get("T2_PRESENT_READS") == "1"
        g6 = next((g for g in a2["gates"] if g.get("kind") == "select_confirm"), None) if present_on else None
        # T2_PRESENT_NESTED=1 = operand-grounding present 확장(L2 item/L3 variant): read record의
        # nested list/dict를 명시 choice-set으로 (priority-2·replay-safe·A2 present_specs 구동).
        nested_specs = (a2.get("present_specs") or []) if os.environ.get("T2_PRESENT_NESTED") == "1" else []
        # T2_CALC=1 = calc_NL offload(측정 arm): read record서 결정론 집계(available count·order total) 계산·주입.
        # 보고는 모델 → report-conversion 측정([COMPUTED FACTS] 블록=census 마커). A2 calc_specs 구동·엔진 general op.
        calc_specs = (a2.get("calc_specs") or []) if os.environ.get("T2_CALC") == "1" else []
        # T2_PROVENANCE=1 = orchestrator-레벨 게이트(날조 호출을 *실행 전* deny→error로 surface).
        prov_on = os.environ.get("T2_PROVENANCE") == "1"
        ctx = _context_text(self) if prov_on else None
        # T2_RETRY_CONTROLLER=1 = C8 recovery: 반복-동일-실패호출 차단+다양화 지시 (decidable·offload·무학습).
        retry_on = os.environ.get("T2_RETRY_CONTROLLER") == "1"
        retry_k = int(os.environ.get("T2_RETRY_K", "3"))  # rule②: 연속 실패 K회 가드
        failed = getattr(self, "_t2_failed", None)
        if failed is None:
            failed = self._t2_failed = {}
        DIVERSIFY = ("Take a DIFFERENT action: (a) FETCH a missing value from the tool that produces it, "
                     "(b) ASK the user for the correct value, or (c) if you genuinely cannot proceed, "
                     "transfer to a human agent.")

        def _mark_fail(k, why):  # rule①·② 공통: 실패 기록 + 연속카운트++
            if retry_on:
                if k is not None:
                    failed[k] = why
                self._t2_consec = getattr(self, "_t2_consec", 0) + 1

        # ★T2_WRITE_CAP (S1a-1·F5): 성공-write 반복 루프 차단. 기존 retry-controller(rule①)는
        #   *실패* 호출만 잡는다(failed dict) → t102형 19× *성공*-write 재발emit은 통과. 도메인-일반
        #   (write 도구=_confirm_write_tools·A2 confirm 게이트서 도출·리터럴 0)·3중키=_call_key(name+args).
        #   성공 K회(기본2) 초과 동일-write → deny("이미 완료"). 다른 args=다른 키=미차단. Δ계측: 정당 재시도 오차단 0.
        wcap_on = os.environ.get("T2_WRITE_CAP") == "1"
        wcap_k = int(os.environ.get("T2_WRITE_CAP_K", "2"))
        wcap_tools = _confirm_write_tools(a2) if wcap_on else set()
        wdone = self._t2_wdone = getattr(self, "_t2_wdone", {})
        # ★T2_WRITE_EVIDENCE=1: A2 write_evidence_specs 구동(기본 OFF·§_write_evidence_deny).
        wev_specs = (a2.get("write_evidence_specs") or []) \
            if os.environ.get("T2_WRITE_EVIDENCE") == "1" else []

        results = []
        for tc in tool_calls:
            if getattr(tc, "requestor", "assistant") != "assistant":
                results.extend(orig(self, [tc]))  # user-side 툴콜(타 도메인)은 비대상
                continue
            key = _call_key(tc) if retry_on else None
            # rule① 정확-반복 차단 (순환 loop·census ~70%)
            if retry_on and key in failed:
                self.num_errors += 1
                _mark_fail(key, failed[key])
                results.append(_deny_msg(tc, "RETRY_LOOP",
                    f"You ALREADY called {tc.name} with these exact arguments and it FAILED: {failed[key][:160]}. "
                    "Do NOT repeat the identical call. " + DIVERSIFY))
                continue
            # rule② 연속-실패 K 가드 (다양-실패 loop·census ~17%)
            if retry_on and getattr(self, "_t2_consec", 0) >= retry_k:
                self.num_errors += 1
                self._t2_consec = 0  # 가드 발동 후 리셋(매콜 발동 방지)
                results.append(_deny_msg(tc, "RETRY_ESCALATE",
                    f"You have failed {retry_k}+ tool calls in a row. STOP this approach. " + DIVERSIFY))
                continue
            if prov_on:  # L2 provenance: 날조 인자값 차단 (R1B)
                pd = _provenance_deny(tc, ctx, hints)
                if pd:
                    self.num_errors += 1
                    extra = _autofetch_text(self, orig, gate, producer) if os.environ.get("T2_AUTOFETCH") == "1" else ""
                    _mark_fail(key, pd[1])
                    results.append(_deny_msg(tc, pd[0], pd[1] + extra))
                    continue
            ok, g, why = gate.check(tc.name, tc.arguments or {}, last_user_msg=last_user,
                                    transfer_msg_sent=tms)
            if not ok:
                self.num_errors += 1
                _mark_fail(key, why)
                results.append(_deny_msg(tc, g, why))
                continue
            # ★T2_WRITE_EVIDENCE: 원장(도구 출력) 증거 없는 선언-write 반려 (§_write_evidence_deny)
            if wev_specs:
                wd = _write_evidence_deny(self, tc, wev_specs)
                if wd:
                    self.num_errors += 1
                    _mark_fail(key, wd)
                    print("[T2_WRITE_EVIDENCE] deny tool=%s" % tc.name, file=sys.stderr, flush=True)
                    results.append(_deny_msg(tc, "WRITE_EVIDENCE", wd))
                    continue
            # ★T2_WRITE_CAP (S1a-1·F5): 이미 K회 성공한 동일-write → 재실행 안 함·deny("완료")
            if wcap_on and tc.name in wcap_tools:
                _wk = _call_key(tc)
                if wdone.get(_wk, 0) >= wcap_k:
                    self.num_errors += 1
                    print("[T2_WRITE_CAP] capped tool=%s (already succeeded %dx)"
                          % (tc.name, wdone[_wk]), file=sys.stderr, flush=True)
                    results.append(_deny_msg(tc, "WRITE_CAP",
                        "You ALREADY successfully performed this exact action %dx; it is DONE. "
                        "Do NOT repeat the identical call. Move to the next task or confirm completion to the user."
                        % wdone[_wk]))
                    continue
            out = orig(self, [tc])
            results.extend(out)
            if out and getattr(out[0], "error", False):
                _mark_fail(key, _content_str(out[0]))
            elif retry_on:
                self._t2_consec = 0  # 성공 → 연속카운트 리셋
            # ★T2_WRITE_CAP 기록: 성공한 write만 카운트(err=False)
            if wcap_on and tc.name in wcap_tools and out and not getattr(out[0], "error", False):
                _wk2 = _call_key(tc)
                wdone[_wk2] = wdone.get(_wk2, 0) + 1
            if tc.name in obs_tools_g(gate) and out and not out[0].error:
                gate.observe(tc.name, tc.arguments, _content_str(out[0]))
            # ★REPLAY-SAFE present (T2_PRESENT_READS=1): 후보-producer 읽기 응답에 clean 요약 덧붙임.
            # 읽기는 evaluation replay서 skip → content 증강 안전(write-deny=replay 깨짐과 대조).
            if (present_on and out and not getattr(out[0], "error", False)
                    and g6 is not None and tc.name == g6.get("user_producer")):
                uid = (tc.arguments or {}).get(g6.get("user_id_arg", "user_id"))
                summ = candidate_summary(gate.resolvers, g6, uid)
                if summ:
                    try:
                        out[0].content = _content_str(out[0]) + summ
                    except Exception:
                        pass
            # ★read-augment(nested present + calc): RAW 응답을 augment *전* 1회 파싱·공유.
            # (버그수정 2026-06-26: nested가 out.content에 텍스트 append하면 calc의 _parse_json이
            #  오염 content서 실패→calc 미발화 31/342. raw 1회 파싱으로 두 증강 모두 정상.)
            _rec = None
            if (nested_specs or calc_specs) and out and not getattr(out[0], "error", False):
                _rec = _parse_json(_content_str(out[0]))
            # operand-grounding present (T2_PRESENT_NESTED=1): nested operand 명시 choice-set(L2/L3).
            if nested_specs and _rec is not None:
                spec = next((s for s in nested_specs if s.get("trigger_tool") == tc.name), None)
                if spec is not None:
                    summ = nested_candidate_summary(_rec, spec)
                    if summ:
                        try:
                            out[0].content = _content_str(out[0]) + summ
                        except Exception:
                            pass
            # calc_NL offload (T2_CALC=1): 결정론 집계 계산·주입(보고는 모델). [COMPUTED FACTS]=census 마커.
            if calc_specs and _rec is not None:
                cs = [s for s in calc_specs if s.get("trigger_tool") == tc.name]
                if cs:
                    facts = compute_facts(_rec, cs)
                    if facts:
                        try:
                            out[0].content = _content_str(out[0]) + facts
                        except Exception:
                            pass
        return results

    BaseOrchestrator._execute_tool_calls = gated
    return orig


def _last_user_text(orch):
    try:
        for m in reversed(orch.get_messages()):
            if getattr(m, "role", None) == "user" and getattr(m, "content", None):
                return m.content if isinstance(m.content, str) else str(m.content)
    except Exception:
        pass
    return None


def _transfer_msg_sent(orch, notice_text):
    """notice: 고정 transfer 문구가 어시스턴트 발화로 이미 송신됐는가 (불가 판단 시 None)."""
    if not notice_text:
        return None
    try:
        for m in orch.get_messages():
            if getattr(m, "role", None) == "assistant":
                c = getattr(m, "content", None)
                if isinstance(c, str) and notice_text in c:
                    return True
        return False
    except Exception:
        return None


def _parse_json(s):
    """tool content 문자열 → dict/list (실패 시 None). nested present용."""
    if isinstance(s, (dict, list)):
        return s
    if not isinstance(s, str):
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


def _content_str(tool_msg):
    c = tool_msg.content
    if isinstance(c, str):
        try:
            v = json.loads(c)
            return v if isinstance(v, str) else c
        except (ValueError, TypeError):
            return c
    return str(c)


def _deny_msg(tc, gate_name, reason):
    from tau2.data_model.message import ToolMessage
    return ToolMessage(
        id=tc.id, role="tool", requestor="assistant", error=True,
        content=f"Error: [POLICY GATE {gate_name}] {reason}",
    )


# ─── ★권장 설계: agent 생성-레벨 내부 재생성 (T2_PROV_REGEN=1) ───
# 검증기가 날조 인자 감지 → state.messages(공식 대화) 오염 없이 *작업본*에 거부 피드백 추가
# → generate() 재호출(최대 K) → 유효 호출만 반환. 측정 무변경(가드된 시스템을 정직 측정).
REGEN_FEEDBACK = (
    "Error: [PROVENANCE] argument '{k}'='{s}' was not provided by the user nor returned by any tool "
    "— it looks invented (e.g. a schema example value). Do NOT use placeholder/example values and do NOT "
    "ask the user. Instead call a lookup/getter tool that produces this value (e.g. a getter to "
    "retrieve the user's records, payment methods, or addresses) and read the real value from its output. "
    "Now emit a corrected tool call."
)

# T5-C 스펙#2 (prov_mode=rescue 전용): 예시 나열 제거 — 나열 자체가 프라이밍([[42]] 동형·t61형 오도 [P])
REGEN_FEEDBACK_NEUTRAL = (
    # ★V2.5 t17 교정(2026-07-11): 중립화가 priming(필드 예시)뿐 아니라 *getter-호출 지시*까지
    #   약화 → 에이전트가 조회 대신 사용자에게 되물어 no-write(회귀). 필드 예시만 제거하고
    #   "getter 도구를 *호출*해 그 출력에서 읽어라"는 행동 지시는 강하게 유지(사용자-되묻기 탈출구 삭제).
    "Error: [PROVENANCE] argument '{k}'='{s}' was not provided by the user nor returned by any tool "
    "— it looks invented. Do NOT use placeholder/example values. Call the lookup/getter tool that "
    "produces this value and read the real value from its output, then emit a corrected tool call."
)

# ★B-max① t17 지시형(directive) 피드백: fab 인자에 A2 resolver_path가 있으면 generic("getter를 불러라")
#   대신 *어느* producer를 *어떤 알려진 입력*으로 호출해 *어느 필드*를 읽는지 지목 (사용자-되묻기 탈출구 차단).
#   근거 NEWSTACK_GAIN_SIDEEFFECT §G t17 + OVERNIGHT §결과2 (지시형 4지선다가 base ASK 21→loop 0).
REGEN_FEEDBACK_DIRECTIVE = (
    "Error: [PROVENANCE] argument '{k}'='{s}' was not provided by the user nor returned by any tool "
    "— it looks invented. Do NOT ask the user; instead call `{producer}`({in_arg}={in_val}) and read "
    "'{field}' from its output, then emit a corrected tool call with the real value from that output."
)


def _write_evidence_deny(orch, tc, specs):
    """구 apply() 경로 어댑터 — 코어는 _wev_deny_msgs (unified와 공유)."""
    try:
        msgs = orch.get_messages()
    except Exception:
        return None
    return _wev_deny_msgs(msgs, tc, specs)


def _param_cap_deny(messages, tc, specs):
    """★정책-캡 게이트 (T2_PARAM_CAP=1·2026-07-22 §2br·rall9 054 실측). A2 `param_cap_check`:
    선언된 write의 수치 param이 [레코드 필드 × 티어별 비율(A2 맵)] 캡을 초과하면 deny+안내
    (consent 보존 — silent-repair 금지·정책 원문이 '최대치 안내 후 진행 여부 문의').
    엔진=파싱·산술·비교만(정책 수치·필드·맵 전부 A2·[[03b]]). 맵 미기재 카드=무개입."""
    import re as _re
    import t2_resolve as _rz
    name = getattr(tc, "name", None)
    args = _args_dict(tc)
    for sp in (specs or []):
        if name != sp.get("applies_to"):
            continue
        aw = sp.get("applies_when") or {}
        v = str(args.get(aw.get("arg")) or "")
        if aw.get("prefix") and not v.startswith(aw["prefix"]):
            continue
        pn = sp.get("param")
        nested = args.get("arguments")
        if isinstance(nested, str):
            try:
                nested = json.loads(nested)
            except Exception:
                nested = {}
        d = nested if isinstance(nested, dict) else args
        try:
            val = float(str(d.get(pn)).replace(",", "").replace("$", ""))
        except Exception:
            continue
        recs = []
        for m in messages:
            if getattr(m, "role", None) == "tool" and not getattr(m, "error", False):
                c = getattr(m, "content", None)
                if isinstance(c, str):
                    recs += _rz.parse_records(c, sp.get("record_key_field", "account_id"),
                                              tuple(sp.get("record_require") or ()))
        if not recs:
            continue
        rec = recs[-1]
        pb = sp.get("pct_by") or {}
        pct = (pb.get("map") or {}).get(str(rec.get(pb.get("field", "card_type"), "")).strip())
        if pct is None:
            continue
        try:
            lim = float(_re.sub(r"[^0-9.]", "", str(rec.get(sp.get("limit_field", "credit_limit")))))
        except Exception:
            continue
        cap = pct * lim
        if val > cap + 1e-6:
            fb = (sp.get("feedback") or "Error: [POLICY-CAP] {value} exceeds {cap}.")
            for k, vv in (("{value}", val), ("{cap}", cap), ("{pct}", "%d%%" % round(pct * 100)),
                          ("{limit}", lim)):
                fb = fb.replace(k, ("%g" % vv) if not isinstance(vv, str) else vv)
            return fb
    return None


def _wev_deny_msgs(messages, tc, specs):
    """★T2_WRITE_EVIDENCE (2026-07-19·task_029 포렌식): A2 `write_evidence_specs` — 선언된 write 전,
    요구 토큰이 대상 id와 **같은 도구 출력**(role=tool·env 생성물·user *발화*는 제외)에 공존해야 실행.
    029 실측: 사용자 거짓말("해결됐다")만 믿고 update 6건→db 오염. 도메인-일반: 도구명/조건/토큰/문구
    전부 A2·엔진은 substring 공존 실재확인만([[03b]] provenance 계열·값 추출/생성 0). id는 호출 인자
    (중첩 JSON-문자열 포함=_args_dict 계열·모델 자신의 출력 파싱). id 못 읽으면 skip(false-block 회피)."""
    name = getattr(tc, "name", None)
    args = _args_dict(tc)
    for sp in specs:
        if name != sp.get("applies_to"):
            continue
        aw = sp.get("applies_when") or {}
        if aw.get("arg"):
            v = str(args.get(aw["arg"]) or "")
            pref = aw.get("prefix")
            if pref and not v.startswith(pref):
                continue
        idk = sp.get("id_key")
        present = idk in args
        idv = args.get(idk)
        if idv is None and idk:
            for vv in args.values():                   # 중첩 JSON-문자열 인자(디스패처형 도구)
                if isinstance(vv, str) and idk in vv:
                    present = True
                    try:
                        idv = (json.loads(vv) or {}).get(idk)
                    except Exception:
                        pass
                if idv:
                    break
        # ★키 부재 vs 빈 값 구분 (2026-07-21 §2bc·054 t0 실측): 구판 `if not idv: skip`은
        #   `card_last_4_digits: ""` **빈-값 write를 무검사 통과**시켰다(false-block 회피 분기의
        #   구멍·gold와 유일한 diff가 빈 last4). 키 자체가 없는 변형 = skip 유지(변형 오차단 회피)·
        #   키가 실재하는데 값이 비면 = 불완전 write → deny(증거 요구 문구 그대로).
        if not (str(idv).strip() if idv is not None else ""):
            if not present:
                continue
            fb = sp.get("feedback") or "Error: [WRITE-EVIDENCE] required evidence not found for {id}."
            return _wev_fill(fb, "(missing — the argument was left empty)", messages,
                             sp.get("require_tokens_any") or [])
        # ★require_tokens_any (2026-07-23 C121·rall19 031.1 실측): 구판 AND-substring은 KB doc의
        #   무관 예시 숫자("(e.g., 1234, 4321)")와 도구명 substring이 한 출력에 우연 공존하면 날조값도
        #   통과(evidence-collision). any-토큰은 "{id}"를 값으로 치환한 라벨-값 인접 문자열 중 하나가
        #   도구 출력에 실재해야 통과 — 라벨·문구는 전부 A2·엔진은 치환+substring 대조만([[03b]]).
        tokens = sp.get("require_tokens") or []
        any_tokens = sp.get("require_tokens_any") or []
        found = False
        found_any = not any_tokens
        for m in messages:
            if getattr(m, "role", None) != "tool":
                continue
            c = getattr(m, "content", None)
            c = c if isinstance(c, str) else str(c or "")
            if not found and str(idv) in c and all(t in c for t in tokens):
                found = True
            if not found_any and any(t.replace("{id}", str(idv)) in c
                                     for t in any_tokens):
                found_any = True
            if found and found_any:
                break
        if not (found and found_any):
            fb = sp.get("feedback") or "Error: [WRITE-EVIDENCE] required evidence not found for {id}."
            return _wev_fill(fb, str(idv), messages, any_tokens)
    return None


def _wev_fill(fb, idtxt, messages, any_tokens):
    """WEV 피드백 치환: {id} + {evidence}. {evidence}=A2 any-토큰의 라벨 프리픽스({id} 제거분)가
    실재하는 도구-출력 **라인의 축자 인용**(2026-07-23 C122·rall19 031.0 실측: 정답이 도구출력에
    실재하는데 deny 피드백이 교정으로 안 이어져 6회 공전 — C116 "처방적 구체성만 유효 변수" 적용).
    엔진=라인 인용만(값 추출·생성·기입 0·문구는 A2 feedback 소관·[[03b]] present 계열)."""
    if "{evidence}" in fb:
        labels = [t.replace("{id}", "").strip() for t in (any_tokens or [])]
        labels = [lb for lb in labels if lb]
        ev = []
        if labels:
            for m in messages:
                if getattr(m, "role", None) != "tool":
                    continue
                c = getattr(m, "content", None)
                c = c if isinstance(c, str) else str(c or "")
                for ln in c.splitlines():
                    if any(lb in ln for lb in labels):
                        ln = ln.strip()
                        if ln and ln not in ev:
                            ev.append(ln)
        fb = fb.replace("{evidence}", " | ".join(ev)[:300])
    return fb.replace("{id}", idtxt)


def _ref_verify_deny(messages, tc, specs):
    """★T2_REF_VERIFY (2026-07-24 C128/C129·결정론 참조-검증기): 선언된 write가 가리키는 레코드의
    **판별 속성**(예: merchant_name)이 손님 발화에 없으면 deny+처방(손님이 실제 말한 속성값 나열).
    근거: rall19-22 wrong-pick 8/8이 전부 손님 미언급 상점(cross-merchant 인접-행 전사 슬립)·검증기
    8/8 검출·false-block 0(C128). LLM 재선택(REF_ISO)은 gold→wrong 해로운 switch(C129)라 이 결정론
    검증기가 robust. 엔진=substring 대조만(LLM 0·값 추출/생성 0·[[03b]]·[[10]] 검증기=결정론). 도구명·
    필드·문구 전부 A2. id의 레코드 못 찾으면 skip(false-block 회피). 속성값이 손님 발화에 있으면 통과.
    ★한계(A2 note): merchant-absence는 cross-merchant만 검출·동일상점 내 오선택은 amount 필요(별 스펙)."""
    name = getattr(tc, "name", None)
    args = _args_dict(tc)
    for sp in (specs or []):
        if name != sp.get("applies_to"):
            continue
        aw = sp.get("applies_when") or {}
        if aw.get("arg"):
            v = str(args.get(aw["arg"]) or "")
            if aw.get("prefix") and not v.startswith(aw["prefix"]):
                continue
        idk = sp.get("id_key")
        idv = args.get(idk)
        if idv is None and idk:                             # 중첩 JSON-문자열(디스패처형)
            for vv in args.values():
                if isinstance(vv, str) and idk in vv:
                    try:
                        idv = (json.loads(vv) or {}).get(idk)
                    except Exception:
                        pass
                if idv:
                    break
        idv = str(idv or "").strip()
        if not idv:
            continue
        field = sp.get("record_field", "merchant_name")
        # 1) 도구 출력(producer)서 idv 레코드 블록의 field 값 추출
        rec_val, all_vals = None, set()
        for m in messages:
            if getattr(m, "role", None) != "tool" or getattr(m, "error", False):
                continue
            c = getattr(m, "content", None)
            c = c if isinstance(c, str) else str(c or "")
            for mm in re.finditer(re.escape(field) + r":\s*([^\n]+)", c):
                all_vals.add(mm.group(1).strip())
            k = c.find(idv)
            if k >= 0 and rec_val is None:
                win = c[k:k + 400]
                fm = re.search(re.escape(field) + r":\s*([^\n]+)", win)
                if fm:
                    rec_val = fm.group(1).strip()
        if not rec_val:                                     # id의 레코드/필드 못 읽음 → skip
            continue
        # 2) 손님 발화(user)에 그 field 값이 있나. ★정확-문자열은 취약(레코드 "Marriott Hotels" vs
        #    손님 "Marriott hotel"=단복수/접미어 차이로 gold 오차단·rall22 031 실측). 유의미 토큰
        #    (길이≥min_tok·언어일반·도메인 리터럴 0) 하나라도 일치하면 언급으로 간주(false-block 회피
        #    우선=miss가 false-block보다 안전·다른 층이 슬립 재검). min_tok=A2(기본 5·"home"류 generic 제외).
        min_tok = int(sp.get("match_min_token", 5))
        utext = "\n".join(str(getattr(m, "content", "") or "") for m in messages
                          if getattr(m, "role", None) == "user").lower()

        def _mentioned(val):
            if not val:
                return False
            if val.lower() in utext:
                return True
            for tok in re.findall(r"[A-Za-z0-9]+", val):
                if len(tok) >= min_tok and tok.lower() in utext:
                    return True
            return False

        if _mentioned(rec_val):                             # 손님이 언급 → 통과
            continue
        # 3) 손님이 실제 언급한 (목록 내) 값들 = 처방 피드백용
        mentioned = sorted({v for v in all_vals if _mentioned(v)})
        fb = sp.get("feedback") or (
            "Error: [REF-VERIFY] the record you are filing ({id}) is a '{value}' entry, but the "
            "customer never mentioned '{value}'. The customer referred to: {mentioned}. Re-check "
            "which record they meant — read the listing and match on what the customer actually "
            "described — and file that one instead.")
        return (fb.replace("{id}", idv).replace("{value}", rec_val)
                .replace("{mentioned}", ", ".join(mentioned) if mentioned else "(none found)"))
    return None


def _write_arg_ground_deny(messages, tc, specs):
    """★T2_WRITE_ARG_GROUND (2026-07-22 §2bs·rall10 031 실측): A2 `write_arg_grounding` —
    선언된 write의 선언된 인자 **값**이 대화의 실측 근거(role=tool 출력 ∪ user 발화)에
    부분문자열로 실재해야 실행. 031: WRITE_EVIDENCE(선행-read 강제)는 통과했으나 뷰에 실재한
    5320 대신 '1234'를 기입해 dispute 제출 — read-강제와 값-전사는 별개 구멍. 도메인-일반:
    도구/인자/문구 전부 A2·엔진=값 실재확인만(substring·값 추출/생성 0·[[03b]] provenance 계열
    — SG_GROUND의 discoverable-write 확장). user 발화 포함=고객이 직접 준 값은 정당 근거.
    값 없음/키 없음=skip(false-block 회피·빈-값은 WEV 담당)."""
    name = getattr(tc, "name", None)
    args = _args_dict(tc)
    for sp in specs:
        if name != sp.get("applies_to"):
            continue
        aw = sp.get("applies_when") or {}
        if aw.get("arg"):
            v = str(args.get(aw["arg"]) or "")
            pref = aw.get("prefix")
            if pref and not v.startswith(pref):
                continue
        inner = {}
        for vv in args.values():                 # 중첩 JSON-문자열 인자(디스패처형 도구)
            if isinstance(vv, str) and vv.strip().startswith("{"):
                try:
                    j = json.loads(vv)
                    if isinstance(j, dict):
                        inner.update(j)
                except Exception:
                    pass
        for k2, v2 in args.items():
            if not isinstance(v2, (dict, list)):
                inner.setdefault(k2, v2)
        _markers = sp.get("arg_corpus_marker") or {}
        for ga in (sp.get("grounded_args") or []):
            gv = inner.get(ga)
            gs = str(gv).strip() if gv is not None else ""
            if not gs:
                continue
            # ★arg_corpus_marker (2026-07-22 §2bv·054 time_verified): 특정 arg는 marker를 포함하는
            #   출력에서만 grounding 검색(오매칭 방지). 054: time_verified는 "current time is"(get_current_time
            #   출력)에만 매칭 — 다른 레코드의 2023 날짜에 우연-통과 차단·병렬호출(결과 전)이면 미실재→deny.
            _mk = _markers.get(ga)
            found = False
            for m in messages:
                if getattr(m, "role", None) not in ("tool", "user"):
                    continue
                c = getattr(m, "content", None)
                c = c if isinstance(c, str) else str(c or "")
                if _mk and _mk not in c:
                    continue
                if gs in c:
                    found = True
                    break
            if not found:
                fb = sp.get("feedback") or ("Error: [WRITE-GROUNDING] value '{val}' for {arg} "
                                            "does not appear anywhere in this conversation.")
                return fb.replace("{arg}", str(ga)).replace("{val}", gs)
    return None


# ─── ★T2_HAVE_VALUE (2026-07-23 C115·"have-value → act" 일반레버) ───
# 통합 통찰(시각054·CLI052·last-4 039·flail050 = 한 가족): 에이전트가 write W의 필수 인자 A를
#   *이미 대화에 갖고 있는데도* 재요청·재확인을 반복하고 W를 재시도 안 함(temp0 고착). 개별 fix
#   (met_template·verdict-gate·dedup) 대신 provenance 채널서 단일 엔진으로 수렴.
# 정의: ①A가 이전엔 미충족(에이전트 재요청 이력) ②지금 대화에 실재(producer 성공 출력) ③에이전트가
#   A를 또 재요청(W 미시도) → None-anchor 리마인더("A는 이미 있다·다시 묻지 말고 W를 그 값으로 지금
#   호출"). W는 에이전트가 emit(강제 0).
# [[05]] 3질문: (1)도메인-특화 순증? No — 도구/인자/신호/문구 전부 A2 `have_value_reask`(리터럴0=WAG의
#   grounded_args 재사용 계열). 엔진=이력/실재/재요청 판정만. (2)유동판단 동결? No — 어느 값을 쓸지는
#   에이전트 판단·엔진은 "이미 있으니 재요청 말고 써라"만(값 선택 안 함·유일할 때만 인용=자기 fetch 회상).
#   (3)스캐폴드가 write 수행? No — 넛지만·W는 에이전트가 emit(autofetch/ToolCall 아님·§1.5 준수).

def _producer_outputs(messages, marker):
    """producer 성공-실행을 표시하는 marker(A2 문자열)를 담은 role=tool 출력 content 목록(시간순).
    ★user-측 실행(call_discoverable_user_tool)도 결과는 role=tool 메시지 → assistant-툴콜 페어링에
    의존 않고 marker 매칭으로 포착(039 실측: last-4는 손님이 실행). marker='Executed: <tool>'류가
    성공 출력에만 있어 에러출력(§'has not been given')·재요청과 구분(도메인 리터럴은 A2·엔진=substring)."""
    if not marker:
        return []
    ml = str(marker).lower()
    outs = []
    for m in messages:
        if getattr(m, "role", None) != "tool" or getattr(m, "error", False):
            continue
        c = _content_str(m)
        if ml in str(c).lower():
            outs.append(c)
    return outs


def _pattern_values(outs, pattern):
    """producer 출력서 value_pattern(A2 regex·그룹1) 매칭값(중복제거·순서보존). 없으면 []."""
    if not pattern:
        return []
    try:
        rx = re.compile(pattern, re.I)
    except Exception:
        return []
    seen, res = set(), []
    for c in outs:
        for mm in rx.finditer(str(c)):
            v = (mm.group(1) if mm.groups() else mm.group(0)).strip()
            if v and v not in seen:
                seen.add(v)
                res.append(v)
    return res


HAVE_VALUE_FEEDBACK_DEFAULT = (
    "[HAVE-VALUE] You ALREADY have '{arg}' in this conversation ({producer} has been run and its "
    "result is in the tool output above, and the customer has stated it){valclause}. Do NOT ask "
    "for it again and do NOT re-run any lookup for it. Call {write} NOW using that value, then "
    "continue to the next item.")


def _have_value_reask_fb(am, messages, specs):
    """★have-value→act (C115): (spec 순회) W 미시도 ∧ producer 값 실재(marker) ∧ (이전+지금) 재요청
    → None-anchor 리마인더 문구(또는 None). 값 선택 안 함(value_pattern이 유일값 낼 때만 자기-fetch 인용)."""
    if not specs:
        return None
    cur_calls = {_eff_tool_name(tc) for tc in (getattr(am, "tool_calls", None) or [])}
    cur_text = str(getattr(am, "content", "") or "").lower()
    for sp in specs:
        W = sp.get("write")
        producer = sp.get("producer")
        A = sp.get("arg")
        marker = sp.get("producer_marker") or producer
        signals = [str(s).lower() for s in (sp.get("reask_signals") or [])]
        if not (W and A and signals):
            continue
        if W in cur_calls:                                   # 이미 W 호출 중 → 넛지 불요
            continue
        outs = _producer_outputs(messages, marker)           # ② 값 실재(producer 성공 출력·user측 포함)
        if not outs:
            continue
        # ① 이전 이력: committed 이전 assistant 발화에 재요청 신호(정당한 첫-질문서 오발 방지)
        prior = any(getattr(m, "role", None) == "assistant"
                    and any(s in str(getattr(m, "content", "") or "").lower() for s in signals)
                    for m in messages)
        if not prior:
            continue
        # ③ 지금 턴: A 재요청(prose 신호) 또는 producer 재호출 (그리고 위에서 W 미시도 확인됨)
        if not (any(s in cur_text for s in signals) or (producer in cur_calls)):
            continue
        vals = _pattern_values(outs, sp.get("value_pattern"))  # 인용은 옵션(기본=value-free)
        valclause = (" — its value is %s" % vals[0]) if len(vals) == 1 else ""
        fb = sp.get("feedback") or HAVE_VALUE_FEEDBACK_DEFAULT
        # ★값 미유일인데 문구가 {value} 인라인 참조 → 빈 치환(깨진 문구·''날조) 방지: 일반 폴백
        if "{value}" in fb and len(vals) != 1:
            fb = HAVE_VALUE_FEEDBACK_DEFAULT
        return (fb.replace("{arg}", str(A)).replace("{producer}", str(producer or marker))
                  .replace("{write}", str(W)).replace("{valclause}", valclause)
                  .replace("{value}", vals[0] if len(vals) == 1 else ""))
    return None


# ─── ★T2_VALUE_ACQUIRE (C119·2026-07-23·8-task per-step 포렌식): "값 획득 경로 표면화" ───
# have-value의 *앞단계*: agent가 write W의 인자 A를 재요청하는데 ①값이 아직 대화에 없고(producer
#   성공출력0) ②손님이 직접 제공 못 함(카드 없음) ③획득 도구 P(get_card_last_4_digits)를 손님에게
#   *give 안 함*(give_discoverable_user_tool(P) 미호출) → 손님이 값을 못 얻어 재요청 무한(031hv·039 실측).
# 넛지: "이 값은 손님이 P로 직접 조회해야 함 — give_tool로 P를 손님에게 주고 실행을 안내하라."
# [[05]]: 도구/문구=A2(value_acquisition_specs)·엔진=재요청∧값미실재∧give미실행 판정만·리터럴0·넛지만.

def _tool_given(messages, give_tool, acquire_tool):
    """committed서 give_tool(예: give_discoverable_user_tool)로 acquire_tool을 손님에게 준 적 있나."""
    for m in messages:
        for tc in (getattr(m, "tool_calls", None) or []):
            if getattr(tc, "name", "") == give_tool or _eff_tool_name(tc) == give_tool:
                a = _args_dict(tc)
                vals = " ".join(str(v) for v in a.values())
                if acquire_tool in vals or a.get("discoverable_tool_name") == acquire_tool \
                        or a.get("agent_tool_name") == acquire_tool:
                    return True
    return False


VALUE_ACQUIRE_FEEDBACK_DEFAULT = (
    "[VALUE-ACQUIRE] The customer cannot provide '{arg}' directly, and it is NOT in the account "
    "records. It must be retrieved by the CUSTOMER running {acquire_tool}. Stop re-asking — use "
    "{give_tool} to give {acquire_tool} to the customer now, then have them run it and read the "
    "value from its output, then proceed with {write}.")


def _value_acquire_fb(am, messages, specs):
    """★give 표면화: (spec) W 재요청 ∧ 값 미실재(producer 출력 0) ∧ acquire_tool을 give 안 함 → 넛지.
    반환=문구 or None. have-value(값 실재)와 상보(값 미실재+획득경로 미실행)·중복 회피(값 있으면 skip)."""
    if not specs:
        return None
    cur_text = str(getattr(am, "content", "") or "").lower()
    cur_calls = {_eff_tool_name(tc) for tc in (getattr(am, "tool_calls", None) or [])}
    for sp in specs:
        W = sp.get("write")
        arg = sp.get("arg")
        acq = sp.get("acquire_tool")
        give = sp.get("give_tool")
        signals = [str(s).lower() for s in (sp.get("reask_signals") or [])]
        if not (W and arg and acq and give and signals):
            continue
        # ① 값이 이미 있으면(producer 성공출력) → have-value 관할·skip
        if _producer_outputs(messages, sp.get("producer_marker") or ("Executed: " + acq)):
            continue
        # ② acquire_tool을 이미 give했으면(031 base) → skip(경로 이미 열림)
        if _tool_given(messages, give, acq):
            continue
        # ③ 재요청: 이전 or 지금 assistant가 arg 재요청(prose 신호), 그리고 W 미완(값 없으니 당연)
        prior = any(getattr(m, "role", None) == "assistant"
                    and any(s in str(getattr(m, "content", "") or "").lower() for s in signals)
                    for m in messages)
        cur_reask = any(s in cur_text for s in signals)
        if not (prior or cur_reask):
            continue
        fb = sp.get("feedback") or VALUE_ACQUIRE_FEEDBACK_DEFAULT
        return (fb.replace("{arg}", str(arg)).replace("{acquire_tool}", str(acq))
                  .replace("{give_tool}", str(give)).replace("{write}", str(W)))
    return None


def _stale_call_ids(am, messages, wtools):
    """★T2_STALE_STRIP (over-action 억제): am.tool_calls 중 strip 대상 id 집합.
    ①같은 am 내 완전중복(동일 eff+args 2회+·read/write 공통) ②committed서 이미 성공한 *write* 재호출.
    read의 committed-재조회는 미포함(상태변화 존중·over-fire 방지). 도메인 리터럴 0(eff+args 대조)."""
    ok_ids = {getattr(m, "id", None) for m in messages
              if getattr(m, "role", None) == "tool" and not getattr(m, "error", False)}
    done_w = set()
    for m in messages:
        if getattr(m, "role", None) == "assistant":
            for tc in (getattr(m, "tool_calls", None) or []):
                if getattr(tc, "id", None) in ok_ids:
                    eff = _eff_tool_name(tc)
                    if eff in wtools or getattr(tc, "name", "") in wtools:
                        done_w.add((eff, _call_key(tc)))
    seen, stale = set(), set()
    for tc in (getattr(am, "tool_calls", None) or []):
        eff = _eff_tool_name(tc)
        key = (eff, _call_key(tc))
        is_w = eff in wtools or getattr(tc, "name", "") in wtools
        if key in seen:
            stale.add(id(tc))
        elif is_w and key in done_w:
            stale.add(id(tc))
        seen.add(key)
    return stale


def _resolver_directive(a2, tc, k, s):
    """★B-max① (2026-07-11): fab 인자 k에 A2 resolver_path가 있으면 지시형 피드백 문구, 없으면 None.
    resolver_path=[in_arg, producer, field] — 호출 인자에 in_arg 값이 실재(알려진 입력)할 때만 그 producer/
    입력/필드를 지목. 없으면 None → 호출측이 기존 중립 문구로 폴백. 엔진=도메인-일반(A2 소비만·값은 안 읽음=
    autofetch 아님·문구만). default_specs(원리-디폴트 silent 치환)와 regen_resolver_specs(directive-only) 모두 조회."""
    if not a2:
        return None
    specs = list(a2.get("default_specs") or []) + list(a2.get("regen_resolver_specs") or [])
    if not specs:
        return None
    nm = getattr(tc, "name", None)
    d = _args_dict(tc)
    for sp in specs:
        if sp.get("arg") != k:
            continue
        applies = sp.get("applies_to") or []
        if applies and nm not in applies:
            continue
        path = sp.get("resolver_path")
        if not path or len(path) < 3:
            continue
        in_arg, producer, field = path[0], path[1], path[2]
        in_val = d.get(in_arg)
        if not in_val:            # 알려진 입력이 없으면 producer/입력 지목 불가 → 중립 문구로
            return None
        return REGEN_FEEDBACK_DIRECTIVE.format(
            k=k, s=s, producer=producer, in_arg=in_arg, in_val=in_val, field=field)
    return None


def _ctx_from_messages(msgs):
    parts = []
    for m in msgs:
        r = getattr(m, "role", None)
        c = getattr(m, "content", None)
        if r in ("user", "tool") and c is not None:
            parts.append(c if isinstance(c, str) else str(c))
    return " ".join(parts).lower()


def _first_fab_call(am, ctx, hints=DEFAULT_ARG_HINTS, exclude=frozenset()):
    """am.tool_calls 중 첫 날조 (tc, k, s) 또는 None.
    exclude (PROV-RESCUE-PERARG ①): rescue-스킵된 (id(tc), k, s) 집합 — 해당 인자만 건너뛰고
    같은 호출의 다음 fab 인자·다음 호출을 계속 스캔 (구현: per-call 첫 인자 반환+break의 입도 구멍 봉합)."""
    for tc in (getattr(am, "tool_calls", None) or []):
        for k, v in _args_dict(tc).items():
            if not _hint_hit(k, hints):
                continue
            for val in _flatten(v):
                s = str(val).strip()
                if len(s) < 4 or _ctx_has(s, ctx):
                    continue
                if (id(tc), k, s) in exclude:
                    continue
                return (tc, k, s)
    return None


# ─── ★L3 origin-prov (T2_PROV_ORIGIN=1·v3.2·t97/t96 first-mention 세탁 차단) ───
# 원리(보편·도메인 리터럴 0): 에이전트가 *스스로 제안*한 식별값을 user가 yes로 복창하면
# 값∈ctx가 되어 기존 prov를 통과(확인-세탁·A1_V3_PROBE_FORENSIC §2-1). 차단 조건 =
#   값의 최초 등장 role == assistant  ∧  어떤 tool 출력에도 부재(tool-never).
# tool-never가 리뷰 caveat (a)(getter가 늦게 확인한 값=정당)를 처리하고,
# first-role이 caveat (b)(user-first 명시값 t43=정당)를 처리한다.

def _origin_role(s, msgs):
    """값 s의 최초 등장 (role, tool_ever). role∈{user,assistant,tool,None}."""
    first = None
    tool_ever = False
    sl = str(s).strip()
    for m in msgs:
        c = getattr(m, "content", None)
        if not isinstance(c, str):
            continue
        if _ctx_has(sl, c.lower()):
            r = getattr(m, "role", None)
            if r == "tool" and not getattr(m, "error", False):
                tool_ever = True
            if first is None and r in ("user", "assistant", "tool"):
                first = r
    return first, tool_ever


def _first_origin_fab(am, msgs, hints=DEFAULT_ARG_HINTS, exclude=frozenset()):
    """write 인자 중 origin-fab (tc, k, s) 또는 None — 값∈ctx인데 assistant-first ∧ tool-never.
    스코프: 주소류 free-text 인자만(_is_addr_arg·id류는 기존 rescue/fab 관할·Δspurious 보수)."""
    for tc in (getattr(am, "tool_calls", None) or []):
        for k, v in _args_dict(tc).items():
            if not _hint_hit(k, hints):
                continue
            if not _is_addr_arg(k):
                continue
            for val in _flatten(v):
                s = str(val).strip()
                if len(s) < 4 or (id(tc), k, s) in exclude:
                    continue
                first, tool_ever = _origin_role(s, msgs)
                if first == "assistant" and not tool_ever:
                    return (tc, k, s)
    return None


ORIGIN_FEEDBACK = (
    "Error: [PROVENANCE-ORIGIN] argument '{k}'='{s}' was first introduced by YOU (the assistant), "
    "not by the user or any tool output — a user's yes to your proposal does not make it real. "
    "Fetch the actual value from the relevant records (call the getter that produces it), or ask "
    "the user an OPEN question to provide the value themselves. Then re-emit the call."
)


# ─── ★v3.2 CONSISTENCY 가드 (T2_CONSISTENCY=1·L10 멤버십 t35형 + G-noop t71형) ───
# 검사 재료 = 에이전트 자신이 fetch한 tool 출력(grounded·규칙0)·키/도구명 = 전부 A2(eplan spec+
# confirm gates)·엔진 도메인 리터럴 0. 성격=예방(정직·PROBE_FORENSIC: t35/t71 실궤적은 문제
# write 미발행 — 실측 회복은 부분). Δspurious 보수: 컨테이너 상세 미조회면 침묵(read 강제=L2 몫).

def _record_for(msgs, id_key, id_val, lenient=True):
    """out[id_key]==id_val인 최신 tool 출력 record dict (없으면 None)."""
    if not id_key or not id_val:
        return None
    tgt = str(id_val).strip().lower()
    for out in _parse_tool_outputs(msgs, lenient=lenient):  # 최근 우선
        if isinstance(out, dict) and str(out.get(id_key, "")).strip().lower() == tgt:
            return out
    return None


def _ids_at_path(record, path):
    """record의 path=[container_key, id_field] 위치서 멤버 id 집합(lower). list/dict 컨테이너 지원."""
    if not (isinstance(record, dict) and path and len(path) >= 2):
        return set()
    seq = record.get(path[0])
    if isinstance(seq, dict):
        seq = list(seq.values())
    ids = set()
    for it in (seq or []):
        if isinstance(it, dict) and it.get(path[1]) is not None:
            ids.add(str(it.get(path[1])).strip().lower())
    return ids


def membership_violation(d, spec, msgs):
    """L10: d[items_key] 각 id ∈ d[entity_key] record 멤버 집합인지.
    위반 시 (bad_ids, oid, hint_oid|None) — hint = bad를 실제 담은 다른 grounded record."""
    ent_key = (spec or {}).get("entity_key")
    mem_key = (spec or {}).get("items_key")
    path = (spec or {}).get("items_id_path")
    if not (ent_key and mem_key and path):
        return None
    oid = d.get(ent_key)
    mems = d.get(mem_key)
    if not oid or not isinstance(mems, list) or not mems:
        return None
    rec = _record_for(msgs, ent_key, oid)
    if rec is None:
        return None
    ids = _ids_at_path(rec, path)
    if not ids:
        return None
    bad = [str(m) for m in mems if str(m).strip().lower() not in ids]
    if not bad:
        return None
    hint = None
    for out in _parse_tool_outputs(msgs):
        if (isinstance(out, dict) and out.get(ent_key)
                and str(out.get(ent_key)).lower() != str(oid).lower()
                and any(str(b).strip().lower() in _ids_at_path(out, path) for b in bad)):
            hint = str(out.get(ent_key))
            break
    return (bad, str(oid), hint)


def _leaf_scalar_map(obj, out=None):
    """중첩 record의 leaf 스칼라 {말단키(lower): str값} — first-win."""
    if out is None:
        out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                _leaf_scalar_map(v, out)
            elif isinstance(v, (str, int, float)) and not isinstance(v, bool):
                out.setdefault(str(k).lower(), str(v).strip())
    elif isinstance(obj, list):
        for v in obj:
            _leaf_scalar_map(v, out)
    return out


def noop_write(d, spec, msgs, min_match=3):
    """G-noop(t71형): write의 스칼라 인자 전부가 대상 record 현재값과 동일(매칭 ≥min_match)
    = 아무것도 안 바꾸는 write → 참조 오바인딩 신호. 하나라도 다르면 정상(False)."""
    ent_key = (spec or {}).get("entity_key")
    if not ent_key:
        return False
    oid = d.get(ent_key)
    if not oid:
        return False
    rec = _record_for(msgs, ent_key, oid)
    if rec is None:
        return False
    leaves = _leaf_scalar_map(rec)
    matched = 0
    for k, v in d.items():
        if k == ent_key or v is None or isinstance(v, (list, dict)):
            continue
        cur = leaves.get(str(k).lower())
        if cur is None:
            continue
        if str(v).strip().lower() == cur.lower():
            matched += 1
        else:
            return False
    return matched >= min_match


CONS_MEMBER_FEEDBACK = (
    "[CONSISTENCY] item(s) {bad} do not belong to {ent}='{oid}' according to its latest fetched "
    "details.{hint} Re-check which record actually contains the item(s) the user means, then "
    "re-emit a corrected call."
)

CONS_NOOP_FEEDBACK = (
    "[CONSISTENCY] this call would change nothing — every requested value equals the record's "
    "current value. This usually means the WRONG record was selected. Re-check by CONTENT which "
    "record the user means (compare the records you have listed/read; do not rely on list order), "
    "or ask the user to identify it by content. Then re-emit."
)


# ─── ★v5 L2R read-all 강제 (T2_READALL=1·A1_V2_NT2_FORENSIC §4-1) ───
# 근거 [M]: nt2 flip 분기 분석서 F→P의 승리 패턴이 "write 전 후보 record 전수 열거"로 단일
# 수렴(coverage+BIND+PROV 동시 폐쇄·t2/74/81/107/22)·P→F 구제 최대 레버(t5/82/83/91/111).
# 기전: 첫 write 시도 시 listed−examined ≠ ∅ 이면 read-지시 deny(read-only·[[05]] write 강제 0).
# Δspurious 보수: 후보 과다(>max) 침묵·sim-cap·재료=ledger(에이전트 자신의 열람 기록·규칙0).

def readall_unread(listed, examined, max_candidates=8):
    """read-all 순수 술어: 미열람 후보 목록(정렬) — 없거나 후보 과다면 []."""
    listed = {str(x).strip() for x in (listed or ()) if str(x).strip()}
    examined = {str(x).strip() for x in (examined or ())}
    if not listed or len(listed) > max_candidates:
        return []
    return sorted(listed - examined)


READALL_FEEDBACK = (
    "[READ-ALL] Before modifying anything, read the details of every record you have listed — "
    "call {reader} for: {ids}. The user's request may cover records you have not read yet. "
    "After reading, re-emit your call(s), acting on every record the request covers and only those."
)


# ─── ★COV FIND-subset 백스톱 (T2_COV=1·COVERAGE_LOOP_DESIGN §3·in-flight) ───
# READALL이 못 닫는 잔여 = "다 읽고도 일부에만 행동"(v3-probe t81: 4주문 read 후 1개만 취소).
# 기전: ≥1 write 실행 후, M(요청이 커버하는 record 집합·LLM formalize 1회 캐시) ∖ acted ≠ ∅ 이면
# 생성-레벨 리마인더 1회(기존 eplan 버퍼 채널·비커밋·in-flight — walk의 stop-time 사인 해소).
# 분담([[10]]): M 산출=LLM(formalize·방식1 content-match v1·방식2 predicate는 후속),
# diff/캐시/cap=결정론. 재료=에이전트 가시 대화+열람 record만(규칙0·DB 주입 0). write 강제 0.

def _cov_parse_ids(text, known_ids):
    """LLM 응답서 record id 목록 추출(순수) — grounded 교집합만(발명 id 차단)."""
    import re as _re
    if not text:
        return []
    m = _re.search(r'\{[^{}]*"ids"[^{}]*\}', text, _re.S)
    raw = []
    if m:
        try:
            raw = [str(x).strip() for x in (json.loads(m.group(0)).get("ids") or [])]
        except Exception:
            raw = []
    if not raw:  # 폴백: known id의 문자 그대로 등장
        raw = [k for k in known_ids if k in text]
    known = {str(k).strip() for k in known_ids}
    seen, out = set(), []
    for r in raw:
        if r in known and r not in seen:
            seen.add(r)
            out.append(r)
    return out


def _cov_formalize_M(agent, la, UserMessage, msgs, ep_spec, a2):
    """M 산출 서브콜(격리·1회): 유저 발화 + 열람 record 요약 → 요청이 커버하는 record ids.
    실패/미확신 = [] (침묵·안전측)."""
    ent_key = (ep_spec or {}).get("entity_key")
    if not ent_key:
        return []
    recs = []
    seen = set()
    path = (ep_spec or {}).get("items_id_path") or ()   # [container_key, id_field] = A2([[05]])
    for out in _parse_tool_outputs(msgs):
        if isinstance(out, dict) and out.get(ent_key):
            oid = str(out.get(ent_key)).strip()
            if oid in seen:
                continue
            seen.add(oid)
            names = []
            if len(path) >= 2:
                seq = out.get(path[0])
                if isinstance(seq, dict):
                    seq = list(seq.values())
                for it in (seq or []):
                    if isinstance(it, dict):
                        names.append(str(it.get("name") or it.get(path[1]) or ""))
            recs.append("%s status=%s items=%s"
                        % (oid, out.get("status"), ", ".join(n for n in names[:8] if n)))
    if len(recs) < 2:
        return []
    users = [str(getattr(m, "content", "") or "") for m in msgs
             if getattr(m, "role", None) == "user"][:6]
    prompt = (
        "You are auditing a customer-service conversation. Based ONLY on what the user asked, "
        "decide which of these records the user's request requires MODIFYING (write actions).\n"
        "User said:\n- " + "\n- ".join(u[:300] for u in users) +
        "\nRecords:\n- " + "\n- ".join(recs[:8]) +
        '\nReply with JSON only: {"ids": ["..."]} — include a record ONLY if the request clearly '
        "covers it; if the request targets a single record, list just that one."
    )
    try:
        try:
            um = UserMessage(role="user", content=prompt)
        except TypeError:
            um = UserMessage(content=prompt)
        kw = {k: v for k, v in dict(getattr(agent, "llm_args", None) or {}).items()
              if "tool" not in k}
        sub = la.generate(model=agent.llm, tools=None, messages=[um],
                          call_name="cov_formalize", **kw)
        return _cov_parse_ids(getattr(sub, "content", None) or "", seen)
    except Exception:
        return []


COV_REMINDER = (
    "[COVERAGE] Based on the conversation so far, record(s) {ids} may ALSO be covered by what "
    "the user asked, but you have not acted on them. If they are covered, handle them too now; "
    "if they are not, briefly confirm that with the user before finishing."
)

# ★T2_COV_MIDDRIVE (C118·EPLAN_MIDDRIVE_DESIGN §2.1): "종료시 1회" → "갭 열린 동안 매 드리프트 견인".
COV_REMINDER_DRIVE = (
    "[COVERAGE-DRIVE] The customer's request covers {n} item(s); you have completed {done} but "
    "{ids} are NOT yet done. Do the NEXT one NOW — call the write/action tool for it directly. "
    "Do not end, transfer, or move to other topics until every covered item is handled."
)


def _last_assistant_did_write(messages, write_tools):
    """가장 최근 assistant 메시지의 tool_calls에 write(eff∈write_tools)가 있나 (드리프트 판정 반대).
    write가 있으면 진행 중(드리프트 아님)·없으면(read/prose/타도구) 드리프트. write_tools=A2 도출."""
    for m in reversed(messages):
        if getattr(m, "role", None) != "assistant":
            continue
        for tc in (getattr(m, "tool_calls", None) or []):
            eff = _eff_tool_name(tc)
            if eff in write_tools or getattr(tc, "name", "") in write_tools:
                return True
        return False   # 최근 assistant 턴에 write 없음 = 드리프트
    return False


# ─── ★TOOLERR 도구-에러 라우팅 (T2_TOOLERR=1·사용자 지시 2026-07-13) ───
# 일반 로직(엔진): 방금 committed된 tool-error를 A2 `tool_error_specs`로 분류·라우팅.
#   class=recover → 인자 고쳐 재시도 강제(같은-실패-인자 재발행/조기 transfer deny).
#   class=abstain → 날조 금지·ASK/transfer 지시(비강제 directive).
# 도메인 정보(에러패턴·class·hint)는 전부 A2([[05]]). 재료=committed tool-error(규칙0).

def _trailing_tool_errors(msgs):
    """대화 말미의 tool-result 블록(에이전트가 지금 응답하려는) 중 error 메시지들(최근순).
    앞선 assistant-text/user를 만나면 중단 = '방금 난 에러'만."""
    errs, callmap = [], {}
    for m in msgs:
        if getattr(m, "role", None) == "assistant":
            for tc in (getattr(m, "tool_calls", None) or []):
                callmap[getattr(tc, "id", None)] = tc
    for m in reversed(msgs):
        r = getattr(m, "role", None)
        if r == "tool":
            if getattr(m, "error", False):
                errs.append(m)
        elif r == "assistant":
            break   # tool-call을 낸 assistant 턴 = 블록 경계
        else:
            break   # user 발화가 더 최근이면 이미 넘어감
    return errs, callmap


def classify_tool_error(msgs, a2):
    """방금 난 tool-error를 A2 tool_error_specs로 분류.
    반환 (spec, tool_name, failed_args) | None. spec={match,class,hint,applies_to?}."""
    specs = (a2 or {}).get("tool_error_specs") or []
    if not specs:
        return None
    errs, callmap = _trailing_tool_errors(msgs)
    if not errs:
        return None
    import re as _re
    for em in errs:                       # 최근 에러 우선
        content = str(getattr(em, "content", None) or "")
        tc = callmap.get(getattr(em, "id", None))
        tool = getattr(tc, "name", None) if tc else None
        for sp in specs:
            ap = sp.get("applies_to")
            if ap and tool not in ap:
                continue
            pat = sp.get("match")
            if pat and not _re.search(pat, content, _re.I):
                continue
            return (sp, tool, (_args_dict(tc) if tc else {}))
    return None


def _transfer_tools(a2):
    """포기/이관 도구 집합 = A2서 도출(엔진 리터럴 0·[[05]]). 우선순위:
    (1) a2["transfer_tools"] 명시 (2) notice-kind 게이트의 applies_to(이관 전 고지=이관도구)."""
    if not a2:
        return set()
    tt = set(a2.get("transfer_tools") or [])
    if tt:
        return tt
    for g in (a2.get("gates") or []):
        if g.get("kind") == "notice":
            tt |= set(g.get("applies_to") or [])
    return tt


TOOLERR_RECOVER = (
    "[TOOL-ERROR:RECOVER] Your last call to {tool} FAILED: {hint} This is a fixable error — "
    "do NOT give up, transfer, or invent a substitute. Re-derive the correct argument value from "
    "prior tool outputs and re-emit the call with the CORRECTED argument."
)
TOOLERR_ABSTAIN = (
    "[TOOL-ERROR:ABSTAIN] Your last call to {tool} returned no usable result: {hint} Do NOT make "
    "up an answer to fill this gap. Tell the user you could not find the information, or transfer."
)


# ─── ★GROUND (T2_PROV_GROUND=1): config-도출 candidate-surfacing resolver ───
# 모델은 *의도/op* 명명만, 구체값은 결정론이 직전 tool 출력서 grounding (추출 = 도메인-일반).

def _sig(s):
    s = str(s).strip()
    if "@" in s and "." in s:
        return "email"
    if s.startswith("#"):
        return "hashid"
    if s.replace("-", "").isdigit() and len(s) >= 5:
        return "numid"
    return "other"


# 주소류 free-text 인자 토큰(도메인일반 식별-어휘·PROV-ADDR-FULL §6.3). id-형(numid/hashid) 아님.
_ADDR_ARG_TOKENS = ("address", "street", "city", "state", "zip", "postal", "country")


def _is_addr_arg(arg_key):
    """arg_key가 주소류 free-text 인자인가(도메인일반 어휘 매칭)."""
    kl = str(arg_key).lower()
    return any(t in kl for t in _ADDR_ARG_TOKENS)


def _key_tokens(arg_key):
    """arg_key → 의미 토큰 (id/ids 제거·단수화). 'order_id'→{'order'}, 'item_ids'→{'item'}."""
    toks = set()
    for t in str(arg_key).lower().split("_"):
        if t in ("id", "ids", "no", "num", "number", "code"):
            continue
        toks.add(t[:-1] if t.endswith("s") and len(t) > 3 else t)
    return toks or {str(arg_key).lower()}


def _iter_scalars(obj, key=None):
    """JSON에서 (enclosing_key, scalar) 재귀 수집 + dict-key 자체(ID형)."""
    if isinstance(obj, dict):
        for kk, vv in obj.items():
            if isinstance(vv, (dict, list)):
                if isinstance(kk, str) and any(c.isdigit() for c in kk) and len(kk) >= 5:
                    yield (key or kk, kk)
                yield from _iter_scalars(vv, kk)
            else:
                yield (kk, vv)
    elif isinstance(obj, list):
        for vv in obj:
            yield from _iter_scalars(vv, key)
    else:
        yield (key, obj)


def _parse_tool_outputs(msgs, lenient=False):
    """role==tool·비-error 메시지 content를 JSON 파싱(최근→과거 순).
    lenient=True(T5-C 신규 경로 전용): JSON 뒤에 augment 텍스트가 append된 content도
    leading-JSON으로 구제(N4). 기본 False = v1 동작 바이트-동일."""
    outs = []
    for m in reversed(msgs):
        if getattr(m, "role", None) != "tool":
            continue
        if getattr(m, "error", False):
            continue
        c = getattr(m, "content", None)
        if not isinstance(c, str):
            continue
        try:
            outs.append(json.loads(c))
        except Exception:
            if lenient:
                try:
                    obj, _ = json.JSONDecoder().raw_decode(c.strip())
                    outs.append(obj)
                    continue
                except Exception:
                    pass
            outs.append(c)
    return outs


def _grounded_candidates(arg_key, fab_value, msgs, limit=8, lenient=False):
    """arg_key 타입에 맞는 grounded 후보값을 tool 출력서 추출(최근 우선·dedup·순서보존)."""
    toks = _key_tokens(arg_key)
    want_sig = _sig(fab_value)
    seen = set()
    key_cands, sig_cands = [], []
    for out in _parse_tool_outputs(msgs, lenient=lenient):
        if not isinstance(out, (dict, list)):
            continue
        for kk, vv in _iter_scalars(out):
            s = str(vv).strip()
            if len(s) < 4 or s in seen:
                continue
            kl = str(kk or "").lower()
            if any(t in kl or kl in t for t in toks):
                seen.add(s)
                key_cands.append(s)
            elif want_sig != "other" and _sig(s) == want_sig:
                seen.add(s)
                sig_cands.append(s)
    return (key_cands or sig_cands)[:limit]


GROUND_FEEDBACK = (
    "Error: [PROVENANCE] argument '{k}'='{s}' is invented — never use placeholder/guessed IDs. "
    "The real, grounded {k} value(s) already available from prior tool results are: {cands}. "
    "Use ONLY one of these. If you must determine WHICH one matches what the user described "
    "(e.g. which order contains that item), call the appropriate getter on these candidates "
    "and read the answer from its output. Now emit a corrected tool call."
)

# T5-C 스펙#2 (prov_mode=rescue 전용): 예시 절 제거(프라이밍 원천 중립화)
GROUND_FEEDBACK_NEUTRAL = (
    "Error: [PROVENANCE] argument '{k}'='{s}' is invented — never use placeholder/guessed IDs. "
    "The real, grounded {k} value(s) already available from prior tool results are: {cands}. "
    "Use ONLY one of these, or determine the right one by reading prior tool outputs. "
    "Now emit a corrected tool call."
)

# ─── ★T2_DISAMB=1: |C|>=2 write-인자 재확인 (E-AMB T5 라우터·규칙0 준수) ───
# 값이 문맥에 실재하되 같은-형식 후보가 2+개면(⋈ 지점) 한 번만 재확인 피드백 → 재생성.
# 열거하는 후보 = 이미 조회된 도구출력에서만(_grounded_candidates) = DB 주입 0.
DISAMB_FEEDBACK = (
    "Error: [DISAMBIGUATE] argument '{k}'='{s}' is one of {n} same-type values already seen in prior "
    "tool outputs: {cands}. Multiple candidates exist, so re-check against what the user explicitly "
    "asked for (the exact attributes, which order, the required payment rule) before writing. "
    "If '{s}' is exactly right, re-emit the SAME tool call unchanged. Otherwise emit the corrected "
    "call, or — only if the conversation truly does not determine the answer — ask the user to pick. "
    "Never invent values."
)

# ★enumerate 모드(T2_DISAMB_MODE=enumerate): 재추측 금지·후보 전부 사용자에 제시하고 선택 강제.
# frontier robust 방식(후보 나열+사용자 내용선택) 재현. list-order 관례 불필요.
DISAMB_ENUM_FEEDBACK = (
    "Error: [DISAMBIGUATE-FILTER] argument '{k}'='{s}' is one of {n} candidates already seen: {cands}. "
    "Before writing, FILTER these candidates by EVERY attribute the user explicitly asked about — the FULL "
    "request (all the items to change, all colors/sizes/details), not just one attribute. "
    "If EXACTLY ONE candidate matches the full request, use that one (correct '{s}' to it if different). "
    "If MORE THAN ONE matches, send the USER a message listing those matching candidates with their "
    "identifying details (what each contains) and ask which one they mean; wait for the user's choice. "
    "If NONE matches, ask the user to clarify. Do NOT guess by recency or position. Never invent values."
)


def _confirm_write_tools(a2):
    """A2-도출 write 도구 집합 = kind=='confirm' 게이트의 applies_to (도메인 리터럴 0)."""
    tools = set()
    for g in (a2 or {}).get("gates", []) or []:
        if g.get("kind") == "confirm":
            tools |= set(g.get("applies_to") or [])
    return tools


# ─── ★NL-NUM-PROV (T2_NLNUM_PROV=1) — t47형 NL-산술 환각 검사 (2026-07-11) ───
# 어시스턴트 *텍스트 발화*의 통화-금액(통화기호+숫자·도메인 어휘 0)이 ①이전 문맥(user/tool
# 텍스트)에 원문-부재 ∧ ②calculate류 도구 출력에도 부재(②⊂①: calc 출력=tool 텍스트)면
# 생성-레벨 regen 1회: "암산 금지·calculate 도구로 검증 후 재진술". 상한 1/턴·무과금·
# 채택 전 게이트 재검사(안전측). calculate 도구명은 A2 키 `calc_tool`(ABox·[[05]]).
_MONEY_RE = re.compile(r"[$€£¥]\s?(\d[\d,]*\.\d{1,2})\b")


def _num_variants(num):
    """금액 숫자부의 대조 변형 집합: 콤마 제거·float 정규형('875.50'→'875.5')."""
    s = num.replace(",", "")
    v = {s}
    try:
        v.add(repr(float(s)))
    except ValueError:
        pass
    t = s.rstrip("0").rstrip(".") if "." in s else s
    if t:
        v.add(t)
    return v


def _unverified_amounts(text, ctx_numeric):
    """text(어시스턴트 발화)의 통화-금액 중 ctx(user/tool 텍스트·콤마 정규화)에 부재분."""
    out = []
    for m in _MONEY_RE.finditer(text or ""):
        if not any(v in ctx_numeric for v in _num_variants(m.group(1))):
            out.append(m.group(0).strip())
    return out


NLNUM_FEEDBACK = (
    "Error: [NL-NUM] your reply states the amount '{amt}', which does not appear anywhere in the "
    "conversation or in any tool output — it looks like mental arithmetic. Do NOT do mental "
    "arithmetic. Verify the amount by calling the {tool} tool with the exact expression, then "
    "restate your reply using the verified value."
)


# ─── ★assertion-provenance (2026-07-16·`ASSERTION_PROVENANCE_ARMS_DESIGN_2026_07_16`) ───
# 병목: 에이전트가 producer 도구를 *선택하지 않고* 사용자에게 판단을 주장한다(t019g 3/3·호출 0).
# C45 출처선언은 **write 인자**에만 걸려 있어 이 지점을 통째로 놓친다. 여기서 assertion으로 확장.
# ★불변량: **엔진은 어시스턴트 답변 텍스트를 파싱하지 않는다.**
#   - discovery-required: 엔진이 보는 것 = {호출된 도구 이름} 집합뿐 (구조 이벤트).
#   - self-declaration : 엔진이 보는 것 = LLM이 내놓은 **선언 JSON 필드**뿐 ([[10]] LLM=formalize·엔진=검증).
#   텍스트 정규식으로 '판단'을 탐지하면 = 엔진-formalize + 도메인 리터럴 = [[03b]]/[[05]] 위반 = 실험무효.
# 도메인 사실(어느 데이터원/operand에 어느 producer가 붙는가)은 **전부 A2**(엔진 리터럴 0).

# ─── ★실효 write 술어 (도메인일반·2026-07-18·`A2_DOMAIN_GENERALIZATION_DESIGN §2.2`) ───
# 정본 = `BANK_EPLAN_ALLACTION_IMPL_DESIGN §3.1`의 `_is_write`. **한 정의만 둔다**([[03b]] 술어 이중화 금지) —
# 여기 hoist하고 `T2_FAB_STRIP`(2398)이 쓰던 인라인 사본을 이걸로 대체한다.
# ⚠️`mutates_state`(env 속성)를 쓰면 **안 된다**: 실측 결과 `log_verification`·`give_discoverable_user_tool`·
#   `unlock_…`이 전부 True라 "상태변경 0"이 거의 모든 sim서 거짓 → 게이트가 영영 안 뜬다(2026-07-18 설계 교정).
_READ_PREFIX_RE = re.compile(r"^(get|search|list|lookup|find|retrieve|read|view|check)_", re.I)
_PROCEDURAL_RE = re.compile(
    r"(^log_|^verify_|_verification$|^kb_|^shell$|discoverable|transfer_to_human|^give_|^unlock_|get_current_time)", re.I)
# ★^verify_ 추가(2026-07-20): verify_identity(scaffold 판정 도구·read-성)가 실효-write로 오분류되던 실버그 —
#   CLAIM_PROV write축 거짓통과 + WRITEPROV 조기 break(완료-주장 게이트 약화) 교정. _verification$의 대칭·도메인 일반.


def _eff_tool_name(tc):
    """디스패처 unwrap: write-ness는 **내부 도구**에 종속 — 단 **`call_` 디스패처만** unwrap한다.
    ⚠️`unlock_discoverable_agent_tool(agent_tool_name=X)`는 X를 *실행*하지 않고 *잠금해제*만 하므로
      inner로 풀면 안 된다(unlock=procedural=무해). 겉이름이 `call_`일 때만 inner를 본다.
    `_NNNN` 접미사 제거 = env의 discoverable 명명 관행(도메인 리터럴 아님·패턴)."""
    nm = str(getattr(tc, "name", "") or "")
    if nm.startswith("call_"):
        ar = _args_dict(tc)
        inner = ar.get("agent_tool_name") or ar.get("user_tool_name") or ar.get("discoverable_tool_name") or ""
        if inner:
            return re.sub(r"_\d+$", "", str(inner))
    return re.sub(r"_\d+$", "", nm)


def _claim_unbacked(claims, emap, evs, messages):
    """★claim_prov 원장대조 코어 (2026-07-20 관문5 추출·순수함수=단위테스트 공유·[[03b]]).
    LLM이 formalize한 주장 목록({kind, what})을 A2 event_map으로 원장 이벤트 실재 대조.
    미등재 kind=skip(오탐 방지)·kind=__effective_write__는 실효 write 존재로 판정. 반환=미입증 목록."""
    out = []
    for c in (claims or []):
        k = str((c or {}).get("kind", "")).strip().lower()
        spec = emap.get(k)
        if spec is None:
            continue
        if spec == "__effective_write__":
            if not _any_effective_write(messages):
                out.append(c)
            continue
        pats = spec if isinstance(spec, list) else [spec]
        if not any(any(str(e).startswith(p) for e in evs) for p in pats):
            out.append(c)
    return out


def _known_tool_names(self_tools, env, msgs):
    """★C207/C2-a 대조 집합 (순수함수·리뷰 필수3): 에이전트가 **실제로 쓸 수 있는** 도구 이름 전체.
    `self.tools`만 보면 discoverable 도구(잠금 상태·목록 밖·`_NNNN` 접미사)를 **미보유로 오탐**한다
    — 022/019처럼 유저-측 dispute 도구를 정당히 안내하는 경로가 정확히 그 길을 밟는다.
    집합 = 도구목록 ∪ env user-side discoverable ∪ 이 대화서 unlock/give된 이름 ∪ 실제 호출된 이름.
    전부 접미사 strip 정규화(도메인 리터럴 0·구조 사실만)."""
    def _n(x):
        return re.sub(r"_\d+$", "", str(x or "").strip())
    out = {_n(getattr(t, "name", None)) for t in (self_tools or []) if getattr(t, "name", None)}
    out |= {_n(x) for x in _user_discoverable(env)}
    for m in (msgs or []):
        for tc in (getattr(m, "tool_calls", None) or []):
            out.add(_n(getattr(tc, "name", None)))
            out.add(_n(_eff_tool_name(tc)))
            ar = _args_dict(tc)
            for k in ("agent_tool_name", "user_tool_name", "discoverable_tool_name"):
                if isinstance(ar, dict) and ar.get(k):
                    out.add(_n(ar[k]))
    return {x for x in out if x}


def _unavailable_promises(pending, known):
    """약속(pending)에 실린 도구명이 `known`에 없으면 = **모델이 없는 기능을 약속**. 집합 대조만.
    `tool` 미선언 항목은 판정하지 않는다(구판 A2 하위호환·거동 보존)."""
    def _n(x):
        return re.sub(r"_\d+$", "", str(x or "").strip())
    out = []
    for p in (pending or []):
        t = (p or {}).get("tool")
        if t and _n(t) not in known:
            out.append(p)
    return out


def _fu_window(cap_used, cap, reserve_declared, reserve_used, genuine_resign):
    """★C207/B1 chain 예산 판정 (순수함수·`_cpv_window`와 **동형 반환형**: None|'normal'|'reserve').
    035 day4b 실측: chain 3회 정확 발화(전부 빈손)→cap 소진→**정작 종국 notice 턴에 레버 부재**.
    ⇒ A2 `reserve: true` 선언 체인에 한해 sim당 1회 예비. 단 예비는 **진성 사임-턴**(텍스트-턴)에서만
    소비한다 — readloop 변환 턴(도구는 부르되 requires와 무관)서 태우면 종국에 또 비게 된다(리뷰 필수2)."""
    if cap_used < cap:
        return "normal"
    if reserve_declared and not reserve_used and genuine_resign:
        return "reserve"
    return None


def _claim_has_kind(claims, kinds):
    """★C201/D3 보조(순수함수·단위테스트 공유): 주장 목록에 A2 선언 `reserve_kinds` 중 하나가 있나.
    엔진은 kind 문자열 대조만 — 어떤 kind가 '중요'한지는 A2가 정한다(도메인 리터럴 0)."""
    ks = {str(k).strip().lower() for k in (kinds or [])}
    return any(str((c or {}).get("kind", "")).strip().lower() in ks for c in (claims or []))


def _cpv_window(resign, transfer, cur, cap, tr_spent, rsv_spent, has_reserve):
    """★C201/D3 발화창 판정 (순수함수·2026-07-26·§7-0 실측: `unbacked>0인데 regen 무발생` A11·B5 =
    cap 소진이 실재 → **행동-kind 주장 전용 예비 1회**를 sim당 보장). 반환: None | 'resign' | 'transfer' | 'reserve'.
    - 'reserve'는 cap 소진 후에만 열리고, 실제 regen은 unbacked에 reserve_kind가 있을 때만 집행(호출부).
    - 예비는 sim당 1회·전역 T2_REGEN_BUDGET과 함께 상한(컨텍스트 팽창=게이트 자신의 비용·등대 §1)."""
    if transfer and not tr_spent:
        return "transfer"
    if resign and cur < cap:
        return "resign"
    if resign and has_reserve and not rsv_spent:
        return "reserve"
    return None


def _is_transfer_call(am, emap):
    """★관문5(038 transfer-escape): 이번 응답에 transfer-류 호출이 있나 — 패턴=A2 event_map['transfer']
    재사용(새 A2 필드 0·엔진 리터럴 0). raw명+effective명 둘 다 대조."""
    pats = (emap or {}).get("transfer")
    pats = pats if isinstance(pats, list) else ([pats] if pats else [])
    if not pats:
        return False
    for tc in (getattr(am, "tool_calls", None) or []):
        for n in (str(getattr(tc, "name", "") or ""), _eff_tool_name(tc)):
            if any(n.startswith(p) for p in pats):
                return True
    return False


def _regen_budget_ok(self):
    """★전역 per-sim regen 예산 (2026-07-20·023 컨텍스트 초과 진단·§2ah).
    개별 게이트 cap(FORCE=T2_ACTION_DENY_CAP·RESOLVE 3·writeprov/claimprov/discreq 1)은 있으나
    **전역 예산이 없어** struggling 태스크(023)서 게이트 스택 regen 누적+에이전트 조사가 vLLM
    max_model_len(44672)을 초과→ContextWindowExceededError→sim 무효. 등대 §1.3: 게이트 자신도 비용
    (over-action)을 낸다 — 그 비용(여기선 컨텍스트)을 **측정·상한**한다. 도메인-일반·리터럴 0.
    T2_REGEN_BUDGET=정수(총 regen 상한)·미설정=무제한(기존거동 불변). 소진 후 모든 regen skip→
    에이전트가 종단행동으로 수렴하거나 max_steps로 종료(FORCE→RESOLVE 무한루프 차단)."""
    _b = os.environ.get("T2_REGEN_BUDGET")
    if not _b:
        return True
    return getattr(self, "_t2_regen_total", 0) < int(_b)


def _regen_budget_spend(self):
    self._t2_regen_total = getattr(self, "_t2_regen_total", 0) + 1


def _chain_dispatch(fc, eff):
    """★관문2(2026-07-20·§2aa): follow_up_chain 1건의 발화 판정 (순수 함수·단위테스트 공유 —
    [[03b]] 별도구현 금지·라이브와 같은 코드를 잰다).
    - requires = 문자열 or **리스트(full required-set)** — 누락 있으면 feedback(`{missing}`=누락 전량 나열·
      050 follow-through+054 query-gap 동시 커버).
    - requires 전부 충족 + `decision_tools` 전부 미호출이면 decision_feedback(종단결정 nudge —
      approve 강제 아님·문구가 양방향(approve|decline) 명시·Δspurious 계측 대상).
    반환: (feedback_text, tag) or None. 엔진=집합 대조·치환만(도메인 리터럴 0).
    ★after = 문자열 or 리스트(2026-07-22 §2bv 강건화·rall12 052 실측): scaffold 판정도구(check_cli)가
    절차 anchor(submit)를 우회하는 경로서도 chain이 발화하도록 anchor 다중화 — anchor 중 하나라도
    호출되면 requires(submit 포함) 대조 → submit 미실행이 {missing}에 뜸(절차 대체 방지)."""
    _after = fc.get("after")
    _anchors = _after if isinstance(_after, list) else [_after]
    if not any(a in eff for a in _anchors):
        return None
    req = fc.get("requires") or []
    req = [req] if isinstance(req, str) else list(req)
    missing = [r for r in req if r not in eff]
    if missing and fc.get("feedback"):
        return fc["feedback"].replace("{missing}", ", ".join(missing)), "followup_chain"
    if (not missing and fc.get("decision_tools") and fc.get("decision_feedback")
            and not any(t in eff for t in fc["decision_tools"])):
        return fc["decision_feedback"], "followup_decision"
    return None


def _is_effective_write(name):
    return bool(name) and not _READ_PREFIX_RE.match(name) and not _PROCEDURAL_RE.search(name)


def _any_effective_write(msgs):
    """원장에 **실효 write 실행**이 하나라도 있나 (requestor 무관 — 사용자 실행도 세상을 바꾼다).
    ★`_called_tools`와 달리 user 호출을 **포함**한다: 완료-주장의 근거는 *누가 했든* 실행 이벤트다."""
    for m in msgs:
        for tc in (getattr(m, "tool_calls", None) or []):
            if _is_effective_write(_eff_tool_name(tc)):
                return True
    return False


def _user_discoverable(env):
    """env의 **user-side discoverable** 집합 (도메인일반·리터럴 0).
    banking 실측: `{deposit_check_3847, get_card_last_4_digits, get_referral_link, submit_cash_back_dispute_0589}`.
    구조적 근거(축자 주석): *"These tools represent actions users take in the real world. The agent gives them
    to the user via `give_discoverable_user_tool` … NOT included in the default tool list."*"""
    try:
        ut = getattr(env, "user_tools", None)
        return set(ut.get_discoverable_tools()) if ut is not None else set()
    except Exception:
        return set()


def _called_tools(msgs):
    """지금까지 **에이전트가** 실제로 호출한 도구 이름 집합 (구조 이벤트만·텍스트 무관).
    ★requestor 격리: 사용자 실행 도구(gold `call_discoverable_user_tool` 등)는 세지 않는다 —
      이 arm의 술어는 *에이전트가* producer를 불렀는가이고, user 호출을 섞으면 §7 버그와 동종의
      범주 오류가 된다."""
    out = set()
    for m in msgs:
        for tc in (getattr(m, "tool_calls", None) or []):
            n = getattr(tc, "name", None)
            if n and getattr(tc, "requestor", "assistant") == "assistant":
                out.add(n)
    return out


def _parse_declaration(text):
    """LLM 선언의 JSON 라인만 파싱. 엔진은 이 **구조체**만 읽는다(답변 본문 아님).
    파싱 실패 = 선언 없음 = 무개입 폴백(T5-C 불변량 iv: 실패 시 레버 ≥ floor)."""
    out = []
    for ln in (text or "").splitlines():
        ln = ln.strip().rstrip(",")
        if not (ln.startswith("{") and ln.endswith("}")):
            continue
        try:
            d = json.loads(ln)
        except Exception:
            continue
        if isinstance(d, dict) and "operand" in d:
            out.append(d)
    return out


# A2 `analysis_producers`: [{data_source, producer, subject}] — 읽은 데이터원에 붙는 분석 producer.
DISCREQ_FEEDBACK = (
    "Error: [DISCOVERY] you have read records from '{data_source}', but you have not called "
    "'{producer}' — the tool that determines {subject}. Do not judge {subject} yourself by reading "
    "the raw records; that is what '{producer}' is for. Call '{producer}', passing the values you "
    "read from the records, and base your reply on what it returns. If you are missing an argument "
    "it needs, ask the customer for it."
)

# A2 `assertion_operands`: {operand: producer} — 그 operand를 산출하는 도구.
SELFDECL_PROMPT = (
    "Before your reply is sent, declare its factual basis. For each item listed below, output exactly "
    "one JSON line:\n"
    '{{"operand": "<item>", "claimed": true|false, "source": "GET"|"FIND"|"INFER"|"ASK"}}\n'
    "claimed = true only if your reply states or implies a conclusion about that item.\n"
    "source: GET = a tool you called returned it · FIND = it appears verbatim in this conversation · "
    "INFER = you worked it out yourself · ASK = you still need the customer to tell you.\n"
    "Output only the JSON lines, nothing else.\nItems: {items}"
)
SELFDECL_FEEDBACK = (
    "Error: [DECLARATION] you declared that your reply's conclusion about '{operand}' is INFER — you "
    "worked it out yourself. But '{producer}' is one of your available tools and it returns exactly "
    "that. Do not INFER what a tool can return: call '{producer}', then restate your reply from its "
    "result. If you are missing an argument it needs, ask the customer for it."
)


# ═══════════════════════════════════════════════════════════════════════════
# ★T5-C silent repair (2026-07-11) — 정본 `T5C_SILENT_REPAIR_DESIGN_2026_07_11.md` §6
#   불변량: (i) 커밋될 턴 불파기 (ii) 대화에 새 텍스트 0 (iii) 실행=기록(replay-clean)
#           (iv) 실패 시 폴백=무개입(레버 ≥ floor pointwise)
# ═══════════════════════════════════════════════════════════════════════════

def _subst_arg_value(tc, k, old, new):
    """인자 k 안의 old 값만 제자리 치환(위치 보존). 성공 True·그 외 no-op False. (B1)
    스칼라=전체 일치 시 교체 / 리스트=일치 원소가 정확히 1개일 때 그 원소만(new 중복 시 no-op)
    / nested dict=no-op. str-JSON arguments도 재할당으로 보존(N2)."""
    try:
        d = _args_dict(tc)
        if k not in d:
            return False
        v = d[k]
        new_s = str(new).strip()
        if isinstance(v, dict):
            return False
        if isinstance(v, list):
            hits = [i for i, x in enumerate(v) if str(x).strip() == old]
            if len(hits) != 1 or any(str(x).strip() == new_s for x in v):
                return False
            v2 = list(v)
            v2[hits[0]] = new
            d[k] = v2
        else:
            if str(v).strip() != old:
                return False
            d[k] = new
        tc.arguments = d
        return True
    except Exception:
        return False


def _min_enclosing_record(obj, target):
    """target 스칼라를 담는 최소 dict 반환(깊은 쪽 우선·재귀). 없으면 None."""
    if isinstance(obj, dict):
        for vv in obj.values():
            if isinstance(vv, (dict, list)):
                r = _min_enclosing_record(vv, target)
                if r is not None:
                    return r
        for kk, vv in obj.items():
            if isinstance(vv, (dict, list)):
                if isinstance(vv, list) and any(
                        not isinstance(x, (dict, list)) and str(x).strip() == target for x in vv):
                    return obj
                continue
            if str(vv).strip() == target or str(kk).strip() == target:
                return obj
        return None
    if isinstance(obj, list):
        for vv in obj:
            if isinstance(vv, (dict, list)):
                r = _min_enclosing_record(vv, target)
                if r is not None:
                    return r
        return None
    return None


def _record_snippet(rec, limit=500):
    try:
        s = json.dumps(rec, ensure_ascii=False, default=str)
    except Exception:
        s = str(rec)
    if len(s) > limit and isinstance(rec, dict):
        flat = {kk: vv for kk, vv in rec.items() if not isinstance(vv, (dict, list))}
        try:
            s = json.dumps(flat, ensure_ascii=False, default=str)
        except Exception:
            s = str(flat)
    return s[:limit]


def _candidate_records(arg_key, orig_value, msgs, limit=6):
    """후보값 + 그 후보가 등장한 최소 enclosing 레코드 snippet. (E-ISO ③: id-only 열거 금지)
    원천 = 에이전트 자신이 조회한 tool 출력만(규칙0·리뷰 N5 검증)."""
    cands = _grounded_candidates(arg_key, orig_value, msgs, limit=limit, lenient=True)
    outs = _parse_tool_outputs(msgs, lenient=True)
    recs = []
    for c in cands:
        snip = ""
        for out in outs:
            if isinstance(out, (dict, list)):
                r = _min_enclosing_record(out, str(c).strip())
                if r is not None:
                    snip = _record_snippet(r)
                    break
        recs.append((c, snip))
    return recs


SUBCALL_SYS = (
    "You are resolving ONE ambiguous tool-call argument for a customer-service agent. "
    "Read the conversation transcript and the candidate values (each shown with the data record "
    "it came from), then decide which single candidate the user actually intends."
)


def _text_transcript(msgs, limit_chars=6000):
    """user/assistant 텍스트 턴만 전사(tool 원문 제외·_BLOCK_NOTE 등 개입 메타텍스트 절단=N5)."""
    parts = []
    for m in msgs:
        role = getattr(m, "role", None)
        c = getattr(m, "content", None)
        if role not in ("user", "assistant") or not isinstance(c, str) or not c.strip():
            continue
        t = c
        i = t.find(_BLOCK_NOTE)
        if i >= 0:
            t = t[:i]
        if t.strip():
            parts.append(("User: " if role == "user" else "Agent: ") + t.strip())
    out = "\n".join(parts)
    return out[-limit_chars:]


def _parse_subcall_answer(txt, cands):
    """서브콜 응답 파싱(N3): ① 응답 전체가 정확히 후보 1개(따옴표/공백/구두점 strip) → 수락
    ② 경계-인식 부분검색 유일 매치 → 수락 ③ 그 외 None(UNSURE)."""
    raw = (txt or "").strip()
    t = raw.strip().strip('"\'`.,;: \n\t')
    for c in cands:
        if t == str(c).strip():
            return c
    found = []
    for c in cands:
        cs = str(c).strip()
        idx = raw.find(cs)
        if idx >= 0:
            before = raw[idx - 1] if idx > 0 else " "
            after = raw[idx + len(cs)] if idx + len(cs) < len(raw) else " "
            if not (before.isalnum() or after.isalnum()):
                found.append(c)
    return found[0] if len(found) == 1 else None


def _apply_principle_default(am, a2, gate, ctx):
    """★T5-C P2 원리-디폴트(silent): write tool의 특정 인자가 A2 default_specs에 있으면
    그 인자의 *기본값*을 resolver_path로 조회(read-only)하고, 현재값이 기본값과 다르며
    사용자가 명시 override하지 않았으면(현재값이 user 발화 ctx에 미등장) 기본값으로 제자리 치환.
    C58 원리디폴트 .940(payment=주문 원결제·환불규칙). 턴 불파기·대화 불변·엔진 general.
    a2=augmented dict·gate=GateInterpreter(resolvers)·ctx=user-발화 lower 텍스트. 카운터 반환."""
    import sys as _s
    specs = (a2 or {}).get("default_specs") or []
    if not specs or gate is None:
        return 0
    rf = (getattr(gate, "resolvers", None) or {}).get("resolve_field")
    if not rf:
        return 0
    n = 0
    for tc in (getattr(am, "tool_calls", None) or []):
        nm = getattr(tc, "name", None)
        d = _args_dict(tc)
        for sp in specs:
            arg = sp.get("arg")
            if nm not in (sp.get("applies_to") or []) or arg not in d:
                continue
            path = sp.get("resolver_path")
            if not path or not d.get(path[0]):
                continue
            try:
                default_val = rf(path, d)          # 주문 원결제 (read-only)
            except Exception:
                default_val = None
            if not default_val:
                continue
            cur = str(d.get(arg)).strip()
            if cur == str(default_val).strip():
                continue                            # 이미 원결제 = 정답
            # 사용자가 현재값을 명시했나 (override) → 유지·유동성 보존
            if cur.lower() in (ctx or ""):
                continue
            # 원리-디폴트 위반 = 원결제로 제자리 치환 (str-JSON 재할당 N2 = _subst 사용)
            if _subst_arg_value(tc, arg, cur, str(default_val)):
                n += 1
                print("[T2_PRINCIPLE_DEFAULT] %s.%s %s -> %s" % (nm, arg, cur, default_val),
                      file=_s.stderr, flush=True)
    return n


def _env_verified_args(a2):
    """env가 lookup으로 검증하는 id-형 인자의 key-token 집합 (B3): A2 preconditions/ownership
    게이트의 resolver_path[0] 파생 — 기존 A2 사실만 사용·신규 도메인 리터럴 0."""
    toks = set()
    for g in (a2 or {}).get("gates", []) or []:
        if g.get("kind") not in ("preconditions", "ownership"):
            continue
        paths = [g.get("resolver_path")] + [ch.get("resolver_path")
                                            for ch in (g.get("checks") or [])]
        for rp in paths:
            if rp:
                toks |= _key_tokens(rp[0])
    return toks


def _in_error_loop(msgs, tool_name, npairs=6):
    """최근 npairs개 tool-result 중 같은 tool의 error=True 존재 여부 (N6: tool_calls.id↔ToolMessage.id join).
    unified/prov 경로선 deny가 히스토리에 안 남으므로 error=True ≡ env-오류."""
    id2name = {}
    for m in msgs:
        for tc in (getattr(m, "tool_calls", None) or []):
            id2name[getattr(tc, "id", None)] = getattr(tc, "name", None)
    pairs = 0
    for m in reversed(msgs):
        if getattr(m, "role", None) != "tool":
            continue
        pairs += 1
        if pairs > npairs:
            break
        if getattr(m, "error", False) and id2name.get(getattr(m, "id", None)) == tool_name:
            return True
    return False


def _ref_iso_repair(self, la, UserMessage, msgs, am, specs):
    """★T2_REF_ISO (2026-07-24 C124/C125): 선언-write의 참조-인자를 write 직전 **최소-문맥 격리
    재선택**(유저 발화 축자 + producer 목록만) 후 불일치 시 제자리 치환. 근거=C124 실측: 동일 정보라도
    전체-궤적 문맥은 인접-행/자기-정박 슬립을 9/9 재현·최소-문맥은 9/9 정답 — DISAMB subcall(전체
    transcript)과 달리 **자기-생성 매핑을 프롬프트에서 제거**하는 것이 본질. 재선택 주체=모델(격리
    서브콜·§1.4 F2 처방 "격리 sub-call+결정론 실행")·엔진=문맥 재구성+정규식 대조+치환만(P-B
    disamb-subcall 선례·대화/턴 불변·모든 예외 no-op). A2 `ref_iso`(applies_to/applies_when/param/
    producer_tools/id_pattern) — 엔진 도메인 리터럴 0. 답이 목록에 실재해야만 채택(날조 차단)."""
    import re as _re
    import sys as _s
    for tc in (getattr(am, "tool_calls", None) or []):
        args = getattr(tc, "arguments", None)
        if not isinstance(args, dict):
            continue
        for sp in (specs or []):
            if getattr(tc, "name", None) != sp.get("applies_to"):
                continue
            aw = sp.get("applies_when") or {}
            if aw.get("arg") and not str(args.get(aw["arg"]) or "").startswith(aw.get("prefix", "")):
                continue
            pn = sp.get("param")
            nested = args.get("arguments")
            nd = None
            if isinstance(nested, str):
                try:
                    nd = json.loads(nested)
                except Exception:
                    nd = None
            elif isinstance(nested, dict):
                nd = nested
            cur = str((nd or {}).get(pn) or args.get(pn) or "")
            if not cur:
                continue
            # ★C126 라이브 교정(rall21): 같은 (param,값) 재검이 cap을 소진(031서 keep×8=동일 값)
            #   → verdict 메모이즈. switch 결과도 기억(같은 오값 재등장 시 무비용 치환).
            _memo = self._t2_refiso_memo = getattr(self, "_t2_refiso_memo", {})
            _mk = (pn, cur)
            if _mk in _memo:
                _mv = _memo[_mk]
                if _mv not in (None, cur) and nd is not None:
                    nd[pn] = _mv
                    if isinstance(nested, str):
                        args["arguments"] = json.dumps(nd)
                    print("[T2_REF_ISO] memo-switch param=%s %s->%s" % (pn, cur, _mv),
                          file=_s.stderr, flush=True)
                continue
            _prod = set(sp.get("producer_tools") or [])
            _pids = {getattr(c2, "id", None)
                     for m in msgs for c2 in (getattr(m, "tool_calls", None) or [])
                     if getattr(c2, "name", None) in _prod}
            listing = ""
            for m in msgs:
                if (getattr(m, "role", None) == "tool" and getattr(m, "id", None) in _pids
                        and not getattr(m, "error", False)):
                    c3 = getattr(m, "content", None)
                    if isinstance(c3, str) and len(c3) > len(listing):
                        listing = c3
            if not listing:
                continue
            utext = "\n".join(str(getattr(m, "content", "") or "") for m in msgs
                              if getattr(m, "role", None) == "user")[:6000]
            others = {k2: v2 for k2, v2 in (nd or {}).items() if k2 != pn}
            prompt = ("You are a precise banking assistant.\n\n"
                      "=== CUSTOMER MESSAGES (verbatim) ===\n" + utext
                      + "\n\n=== RECORD LISTING (tool output) ===\n" + listing[:20000]
                      + "\n\n=== ACTION BEING FILED ===\n"
                      + json.dumps(others, default=str)[:800]
                      + "\n\nWhich single '" + str(pn) + "' value from the RECORD LISTING does this "
                        "action refer to, based on the customer's messages? If the customer listed "
                        "several items, first match EVERY listed item to its record"
                      + ((" (" + str(sp.get("match_hint")) + ")") if sp.get("match_hint") else "")
                      + ", then answer for THIS action only. Answer with EXACTLY "
                        "one value copied from the listing, or UNSURE.")
            try:
                um = UserMessage(role="user", content=prompt)
            except TypeError:
                um = UserMessage(content=prompt)
            kw = {kk: vv for kk, vv in dict(getattr(self, "llm_args", None) or {}).items()
                  if "tool" not in kk}
            sub = la.generate(model=self.llm, tools=None, messages=[um],
                              call_name="ref_iso_subcall", **kw)
            stxt = str(getattr(sub, "content", "") or "")
            pat = sp.get("id_pattern") or r"[A-Za-z0-9_]{6,}"
            hits = [h for h in _re.findall(pat, stxt) if h in listing]
            self._t2_refiso = getattr(self, "_t2_refiso", 0) + 1
            # ★C126: 서로 다른 listing-멤버가 2개+ 언급되면(추론 산문 오염) 첫-hit 오채택 위험 →
            #   보수적으로 unsure. 단일 고유 hit만 채택.
            if "UNSURE" in stxt or not hits or len(set(hits)) > 1:
                _memo[_mk] = None
                print("[T2_REF_ISO] unsure param=%s cur=%s nhits=%d"
                      % (pn, cur, len(set(hits))), file=_s.stderr, flush=True)
                continue
            ans = hits[0]
            if ans == cur:
                _memo[_mk] = cur
                print("[T2_REF_ISO] keep param=%s val=%s" % (pn, cur), file=_s.stderr, flush=True)
                continue
            try:
                if nd is not None:
                    nd[pn] = ans
                    if isinstance(nested, str):
                        args["arguments"] = json.dumps(nd)
                else:
                    args[pn] = ans
            except Exception:
                continue
            _memo[_mk] = ans
            print("[T2_REF_ISO] switched param=%s %s->%s" % (pn, cur, ans),
                  file=_s.stderr, flush=True)


def _t5c_disamb_subcall(self, la, UserMessage, state_msgs, tc, k, s, sub_args):
    """P-B: 격리 서브콜로 |C|>=2 선택 판정 → 서브콜 답이 원값과 다르고 화이트리스트(A2
    disamb_sub_args) 타입일 때만 인자 제자리 치환. 원턴·대화 완전 불변·모든 예외 = no-op.
    반환: 'switch'|'keep'|'unsure'|'error' (계측용)."""
    import sys
    try:
        records = _candidate_records(k, s, state_msgs)
        if len(records) < 2:
            return "unsure"
        # ★FORMALIZE-EXEC(레버3·T2_FEXEC=1·NEXT_LEVER_GEN §2): 기준-형식화 서브콜 →
        #   결정론 실행 → 결과를 이 DISAMB 서브콜 *프롬프트*에 비커밋 후보-주석으로 첨부
        #   (P-B 좌석 공유·별도 후크 0·대화/턴 불변). 실패/none = 주석 없음(기존 폴백·§2.4).
        fx_note = ""
        if os.environ.get("T2_FEXEC") == "1":
            try:
                import t2_formalize_exec as _fx
                fx_note = _fx.fexec_for_disamb(self, la, UserMessage, state_msgs, k, s) or ""
            except Exception as _fe:
                print("[T2_FEXEC] error (no-op): %r" % (_fe,), file=sys.stderr, flush=True)
                fx_note = ""
        prompt = (SUBCALL_SYS + "\n\n=== Conversation ===\n" + _text_transcript(state_msgs)
                  + "\n\n=== Candidates for '" + str(k) + "' ===\n"
                  + "\n".join("- %s%s" % (c, ("   | record: " + sn) if sn else "")
                              for c, sn in records)
                  + (("\n\n" + fx_note) if fx_note else "")
                  + "\n\nThe agent currently chose '" + s + "'. Which single candidate does "
                    "the user intend? Answer with EXACTLY one candidate value, or UNSURE.")
        try:
            um = UserMessage(role="user", content=prompt)
        except TypeError:
            um = UserMessage(content=prompt)
        kw = {kk: vv for kk, vv in dict(getattr(self, "llm_args", None) or {}).items()
              if "tool" not in kk}
        sub = la.generate(model=self.llm, tools=None, messages=[um],
                          call_name="disamb_subcall", **kw)
        self._t2_subcall_fired = getattr(self, "_t2_subcall_fired", 0) + 1
        ans = _parse_subcall_answer(getattr(sub, "content", None) or "",
                                    [c for c, _ in records])
        if ans is None:
            self._t2_subcall_unsure = getattr(self, "_t2_subcall_unsure", 0) + 1
            return "unsure"
        if str(ans).strip().lower() == s.lower():
            self._t2_subcall_keep = getattr(self, "_t2_subcall_keep", 0) + 1
            return "keep"
        if (_key_tokens(k) & set(sub_args or ())) and _subst_arg_value(tc, k, s, ans):
            self._t2_subcall_switch = getattr(self, "_t2_subcall_switch", 0) + 1
            print("[T2_SUBCALL] switched arg=%s from=%s to=%s" % (k, s, ans),
                  file=sys.stderr, flush=True)
            return "switch"
        self._t2_subcall_confirmonly = getattr(self, "_t2_subcall_confirmonly", 0) + 1
        return "keep"
    except Exception as e:
        try:
            print("[T2_SUBCALL] error (no-op): %r" % (e,), file=sys.stderr, flush=True)
        except Exception:
            pass
        return "error"


def apply_provenance_regen(max_retries=4, use_badwords=True, ground=False, domain=None, disamb=False,
                           disamb_mode="dialog", prov_mode="full"):
    """LLMAgent._generate_next_message 패치 — R1b 통합 (A2-구동 hints/placeholders).
      L1 = bad_words 디코드-마스크(정적 블랙리스트=A2 placeholders ∪ 스키마-example + 세션-flagged − context).
      L2 = provenance 검증기 + 내부 재생성.
      GROUND = config-도출 candidate-surfacing.
      DISAMB = |C|>=2 write-인자 1회 재확인. disamb_mode='subcall'(T5-C P-B)이면 재확인을
        in-dialogue 재생성 대신 격리 서브콜+제자리 치환으로 수행(원턴·대화 불변).
      prov_mode='rescue'(T5-C P-C): env-검증형 id 날조는 개입 생략(env가 거부)·free-text/에러-루프만 개입.
    domain 주면 A2서 hints/placeholders 도출(없으면 도메인-일반 기본)."""
    import sys
    from tau2.agent.llm_agent import LLMAgent
    import tau2.agent.llm_agent as la
    from tau2.data_model.message import ToolMessage, UserMessage, MultiToolMessage

    a2 = _domain_a2(domain) if domain else None
    hints = a2["_hints"] if a2 else DEFAULT_ARG_HINTS
    placeholders = a2["_placeholders"] if a2 else DEFAULT_PLACEHOLDERS
    disamb_tools = _confirm_write_tools(a2) if disamb else set()
    env_args = _env_verified_args(a2) if prov_mode == "rescue" else set()
    sub_args = set((a2 or {}).get("disamb_sub_args") or [])  # B2: 치환 화이트리스트는 A2 필드
    if os.environ.get("T2_DISAMB_ORDER") == "1":            # ★order operand도 disamb 대상(filter-then-ask·env opt-in)
        sub_args |= {"order", "order_id"}

    def _append(state, message):
        if isinstance(message, UserMessage) and getattr(message, "is_audio", False):
            raise ValueError("audio not supported")
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

    def _gen(self, work, bad_words, call_name, tool_choice=None):
        kw = dict(self.llm_args)
        if use_badwords and bad_words:
            eb = dict(kw.get("extra_body") or {})
            eb["bad_words"] = sorted(bad_words)
            kw["extra_body"] = eb
        if tool_choice:                          # ★레버 A(2026-07-18): tau2 `generate`의 일급 파라미터로 통과
            kw["tool_choice"] = tool_choice
        return la.generate(model=self.llm, tools=self.tools,
                           messages=self._system_messages + work, call_name=call_name, **kw)

    def patched(self, message, state):
        if not hasattr(self, "_t2_static_bl"):
            self._t2_static_bl = _static_blacklist(self.tools, placeholders)
            self._t2_session_bl = set()
        self._system_messages = state.system_messages
        _append(state, message)
        ctx = _ctx_with_toolnames(self, _ctx_from_messages(state.messages))

        def bw():  # 동적: 정적∪세션 − context (진짜 값은 안 막음)
            return [v for v in (self._t2_static_bl | self._t2_session_bl) if v.lower() not in ctx]

        work = list(state.messages)
        am = _gen(self, work, bw(), "agent_response")
        n = 0
        subs = 0
        rescue_skipped = set()  # PROV-RESCUE-PERARG ①: (id(tc), k, s) — rescue 개별 pass-through
        while n < max_retries:
            fab = _first_fab_call(am, ctx, hints, exclude=rescue_skipped)
            if fab is None:
                break  # 잔여 fab 없음 or 전부 rescue-스킵일 때만 탈출 (구 break 의미 보존)
            tc, k, s = fab

            if ground and subs < 8:
                cands = _grounded_candidates(k, s, state.messages, lenient=True)
                # B1: 원소-치환 헬퍼(리스트 인자 통짜-덮어쓰기 버그 수정) — 실패 시 regen 폴백
                if len(cands) == 1 and cands[0] != s and _subst_arg_value(tc, k, s, cands[0]):
                    self._t2_ground_sub = getattr(self, "_t2_ground_sub", 0) + 1
                    subs += 1
                    print("[T2_GROUND] substituted arg=%s val=%s -> %s" % (k, s, cands[0]),
                          file=sys.stderr, flush=True)
                    continue

            if prov_mode == "rescue" and (_key_tokens(k) & env_args) and \
                    _sig(s) in ("hashid", "numid") and \
                    not _in_error_loop(state.messages, getattr(tc, "name", None)):
                # P-C: env-검증형 id 날조는 개입 생략(환경이 거부=C61 H-D 100% 중복) — 에러-루프 시만 개입
                # ★PROV-RESCUE-PERARG ①(2026-07-11 t17): 구 break=regen 루프 전체 탈출 → 같은 호출의
                #   자유텍스트 fab(address1류)이 미검사. per-arg 스킵 후 continue로 다음 fab 계속 스캔.
                #   exclude가 재방문 차단 → 카운터·마커는 스킵당 1회. n 미증가(rescue 무과금·현행 보존).
                rescue_skipped.add((id(tc), k, s))
                self._t2_prov_skipped_envdup = getattr(self, "_t2_prov_skipped_envdup", 0) + 1
                print("[T2_PROV] rescue pass-through tool=%s arg=%s val=%s"
                      % (getattr(tc, "name", "?"), k, s), file=sys.stderr, flush=True)
                continue

            n += 1
            self._t2_session_bl.add(s)
            self._t2_regen = getattr(self, "_t2_regen", 0) + 1
            work = work + [am]
            cands = _grounded_candidates(k, s, state.messages,
                                         lenient=(prov_mode == "rescue")) if ground else []
            if ground and cands:
                tmpl = GROUND_FEEDBACK_NEUTRAL if prov_mode == "rescue" else GROUND_FEEDBACK
                main_reason = tmpl.format(k=k, s=s, cands=", ".join(repr(c) for c in cands))
            else:
                directive = _resolver_directive(a2, tc, k, s)  # ★B-max① t17: resolver_path 지목
                if directive is not None:
                    main_reason = directive
                else:
                    tmpl = REGEN_FEEDBACK_NEUTRAL if prov_mode == "rescue" else REGEN_FEEDBACK
                    main_reason = tmpl.format(k=k, s=s)
            for c in (am.tool_calls or []):
                reason = main_reason if c is tc else \
                    "Error: [PROVENANCE] resolve the invented value first; do not call this yet."
                work.append(ToolMessage(id=c.id, role="tool", requestor="assistant",
                                        error=True, content=reason))
            am = _gen(self, work, bw(), "agent_response_regen")

        # ─── ARG-SCHEMA 위생: 스키마 밖 인자 키 → regen (T2_ARG_SCHEMA=1·기본 OFF) ───
        # 2026-07-19 포렌식: give_discoverable_user_tool에 스키마 밖 'arguments' 키를 얹어
        # 026/027/028 gold give 3건 전부 evaluator exact-match 실패(예측 키집합으로 dict 비교).
        # 도메인일반: 근거=자기 도구 스키마(properties)뿐·값 판단 0·재발화는 모델([[05]]/[[07]] enforced).
        if os.environ.get("T2_ARG_SCHEMA") == "1" and getattr(am, "tool_calls", None):
            if not hasattr(self, "_t2_schema_props"):
                _props = {}
                for _t in (self.tools or []):
                    try:
                        _sc = _t.openai_schema
                        _fn = _sc.get("function") if isinstance(_sc.get("function"), dict) else _sc
                        _nm = _fn.get("name")
                        _pr = ((_fn.get("parameters") or {}).get("properties")) or {}
                        if _nm and _pr:
                            _props[_nm] = set(_pr.keys())
                    except Exception:
                        pass
                self._t2_schema_props = _props
            _tries = 0
            while _tries < 2 and getattr(am, "tool_calls", None):
                _bad = None
                for _tc in (am.tool_calls or []):
                    _allowed = self._t2_schema_props.get(getattr(_tc, "name", None))
                    if not _allowed:
                        continue
                    _extra = [k for k in _args_dict(_tc).keys() if k not in _allowed]
                    if _extra:
                        _bad = (_tc, _extra, _allowed)
                        break
                if _bad is None:
                    break
                _tc, _extra, _allowed = _bad
                _tries += 1
                self._t2_schema_regen = getattr(self, "_t2_schema_regen", 0) + 1
                print("[T2_ARGSCHEMA] regen tool=%s extra=%s" % (_tc.name, _extra),
                      file=sys.stderr, flush=True)
                work = work + [am]
                for _c in (am.tool_calls or []):
                    _reason = ("Error: [ARG-SCHEMA] '%s' does not accept argument(s): %s. Its schema "
                               "declares ONLY these argument(s): %s. Re-issue the call with ONLY declared "
                               "arguments — remove everything else."
                               % (_tc.name, ", ".join(repr(x) for x in _extra),
                                  ", ".join(sorted(_allowed)))) if _c is _tc else \
                        "Error: [ARG-SCHEMA] fix the flagged call first; do not call this yet."
                    work.append(ToolMessage(id=_c.id, role="tool", requestor="assistant",
                                            error=True, content=_reason))
                am = _gen(self, work, bw(), "agent_response_schema_regen")

        # ─── DISAMB: 문맥-실재값인데 같은-형식 후보 2+개 → 1회 재확인 (선택은 모델) ───
        if disamb_tools and getattr(am, "tool_calls", None):
            if not hasattr(self, "_t2_disamb_seen"):
                self._t2_disamb_seen = set()
            hit = None
            for tc in am.tool_calls:
                if getattr(tc, "name", None) not in disamb_tools:
                    continue
                for k, v in _args_dict(tc).items():
                    if not _hint_hit(k, hints):
                        continue
                    for val in _flatten(v):
                        s = str(val).strip()
                        if len(s) < 4 or s.lower() not in ctx:
                            continue
                        memo = (tc.name, k, s.lower())
                        if memo in self._t2_disamb_seen:
                            continue
                        cands = _grounded_candidates(k, s, state.messages)
                        if len(cands) >= 2 and any(s.lower() == str(c).lower() for c in cands):
                            hit = (tc, k, s, cands, memo)
                            break
                    if hit:
                        break
                if hit:
                    break
            if hit and disamb_mode == "subcall":
                # ★T5-C P-B: in-dialogue 재확인 폐지 — 격리 서브콜 판정 + 제자리 치환(원턴·대화 불변)
                tc, k, s, cands, memo = hit
                self._t2_disamb_seen.add(memo)
                self._t2_disamb = getattr(self, "_t2_disamb", 0) + 1
                print("[T2_DISAMB] fired tool=%s arg=%s val=%s ncand=%d mode=subcall"
                      % (tc.name, k, s, len(cands)), file=sys.stderr, flush=True)
                _t5c_disamb_subcall(self, la, UserMessage, state.messages, tc, k, s, sub_args)
                hit = None
            if hit:
                tc, k, s, cands, memo = hit
                self._t2_disamb_seen.add(memo)
                self._t2_disamb = getattr(self, "_t2_disamb", 0) + 1
                print("[T2_DISAMB] fired tool=%s arg=%s val=%s ncand=%d" % (tc.name, k, s, len(cands)),
                      file=sys.stderr, flush=True)
                dwork = list(work) + [am]
                fb = DISAMB_FEEDBACK.format(k=k, s=s, n=len(cands),
                                            cands=", ".join(repr(c) for c in cands[:8]))
                for c in (am.tool_calls or []):
                    reason = fb if c is tc else \
                        "Error: [DISAMBIGUATE] re-check pending; re-emit this call after resolving."
                    dwork.append(ToolMessage(id=c.id, role="tool", requestor="assistant",
                                             error=True, content=reason))
                am2 = _gen(self, dwork, bw(), "agent_response_disamb")
                # 재확인 응답이 날조를 새로 들이면 prov 루프로 정화(2회 한도)·실패 시 원 응답 유지
                n2 = 0
                while n2 < 2:
                    fab2 = _first_fab_call(am2, ctx, hints)
                    if fab2 is None:
                        break
                    tc2, k2, s2 = fab2
                    n2 += 1
                    self._t2_session_bl.add(s2)
                    dwork = dwork + [am2]
                    for c in (am2.tool_calls or []):
                        reason = REGEN_FEEDBACK.format(k=k2, s=s2) if c is tc2 else \
                            "Error: [PROVENANCE] resolve the invented value first; do not call this yet."
                        dwork.append(ToolMessage(id=c.id, role="tool", requestor="assistant",
                                                 error=True, content=reason))
                    am2 = _gen(self, dwork, bw(), "agent_response_regen")
                if _first_fab_call(am2, ctx, hints) is None:
                    # ★T5-C fix(C61 H-C): 재확인 응답이 tool_calls 없는 텍스트-only면 원 호출 유지.
                    #   무조건 수락이 write 유실(39건·−37시행)의 코드-확정 기전 — 원값은 문맥-실재라 유지 무해.
                    if getattr(am2, "tool_calls", None):
                        sw = any(str(vv).strip().lower() != s.lower()
                                 for c2 in am2.tool_calls if c2.name == tc.name
                                 for kk, vv0 in _args_dict(c2).items() if kk == k
                                 for vv in _flatten(vv0))
                        if sw:
                            print("[T2_DISAMB] switched arg=%s from=%s" % (k, s),
                                  file=sys.stderr, flush=True)
                        am = am2
                    else:
                        self._t2_disamb_nowrite_keep = getattr(self, "_t2_disamb_nowrite_keep", 0) + 1
                        print("[T2_DISAMB] rejected: re-check dropped tool_calls; keeping original",
                              file=sys.stderr, flush=True)
        return am

    LLMAgent._generate_next_message = patched
    return patched


# ═══════════════════════════════════════════════════════════════════════════
# ★REPLAY-SAFE 게이트 (generation-level regen) — REPLAY_SAFE_GATE_DESIGN_2026_07_06
#   문제: apply()는 deny 시 합성 ToolMessage를 히스토리에 커밋 → tau2 평가의 set_state
#   replay(mutating tool 재실행·environment.py:389 assertion)가 깨짐 → infrastructure_error.
#   해결: 게이트를 생성-레벨로 이동 — deny면 작업버퍼서 피드백+재생성(K=MAX_REGEN), 유효 호출만
#   커밋. R8 종단: K 소진 후에도 deny면 차단 mutating 호출을 커밋 前 제거 → 히스토리 replay-clean.
#   R1 예산: deny turn마다 orchestrator.num_errors++ (best-of-K 아님·too_many_errors 동일 압박).
# ═══════════════════════════════════════════════════════════════════════════

_BLOCK_NOTE = ("\n\n[Note: the tool call(s) above were blocked by a policy gate and were NOT "
               "executed. Satisfy the gate requirement (authenticate / get explicit user confirmation / "
               "check the record's status / fix the operation) before attempting the action again.]")


def _iter_tc_result_pairs(messages):
    """committed 히스토리서 (assistant ToolCall, matching tool ToolMessage) 쌍 yield."""
    by_id = {}
    for m in messages:
        if getattr(m, "role", None) == "tool" and getattr(m, "id", None) is not None:
            by_id[m.id] = m
    for m in messages:
        if getattr(m, "role", None) == "assistant":
            for tc in (getattr(m, "tool_calls", None) or []):
                yield tc, by_id.get(getattr(tc, "id", None))


def _rebuild_gate_state(gate, a2, messages):
    """committed clean 히스토리서 auth 상태 재구성(denied 호출 부재 = 정확)."""
    gate.state.auth_user = None
    auth_tools = a2["_auth_tools"]
    for tc, tm in _iter_tc_result_pairs(messages):
        name = getattr(tc, "name", None)
        if name in obs_tools_g(gate) and tm is not None and not getattr(tm, "error", False):
            gate.observe(name, _args_dict(tc), _content_str(tm))


def _regen_last_user(messages):
    for m in reversed(messages):
        if getattr(m, "role", None) == "user" and getattr(m, "content", None):
            c = m.content
            return c if isinstance(c, str) else str(c)
    return None


def _regen_transfer_sent(messages, notice_text):
    if not notice_text:
        return None
    for m in messages:
        if getattr(m, "role", None) == "assistant":
            c = getattr(m, "content", None)
            if isinstance(c, str) and notice_text in c:
                return True
    return False


def _denied_calls(am, gate, last_user, transfer_sent):
    """am의 assistant tool_calls 중 gate-deny 되는 것 = [(tc, gid, why)]."""
    out = []
    for tc in (getattr(am, "tool_calls", None) or []):
        if getattr(tc, "requestor", "assistant") != "assistant":
            continue
        ok, gid, why = gate.check(getattr(tc, "name", "") or "", _args_dict(tc),
                                  last_user_msg=last_user, transfer_msg_sent=transfer_sent)
        if not ok:
            out.append((tc, gid, why))
    return out


def _dedup_cache_safe(orch, name):
    """★READ_DEDUP 캐시 가부 (2026-07-20 e2e10 038 크래시·§2at·순수함수=테스트 공유).
    근본: 우리 write-술어(_is_effective_write)와 **tau2 replay의 mutating-술어**가 불일치 —
    unlock은 우리에겐 procedural(non-write)이라 캐시됐는데 replay는 mutating으로 **재실행** →
    "Tool unlocked..." ≠ "[DUPLICATE-READ]" stub → sim 무효(실측 038 t0). 캐시 가부의 정본을
    tau2 술어로: env가 mutating으로 보는 도구는 **캐시 금지**(stub이 히스토리에 남으면 replay 불일치).
    판정 불가(env 부재/예외)=False(캐시 안 함·안전측). 도메인 리터럴 0."""
    env = getattr(orch, "environment", None)
    try:
        return env is not None and not env._is_mutating_tool(name)
    except Exception:
        return False


def _budget_tick(agent):
    """R1: 차단 turn마다 orchestrator.num_errors++ → too_many_errors 예산 동일(best-of-K 방지)."""
    orch = getattr(agent, "_t2_orch", None)
    if orch is not None:
        try:
            orch.num_errors = getattr(orch, "num_errors", 0) + 1
        except Exception:
            pass


def _install_overflow_guard():
    """★컨텍스트 초과 우아한 종료 (2026-07-20·023 진단·§2ah). 하네스-일반(도메인 무관·[[05]] 3질문 NO).
    문제: full_duplex `step()`이 `ContextWindowExceededError`를 안 잡아 예외가 러너까지 전파→sim 전체가
      `infrastructure_error`(무효·unscored·0 msg)로 **소실**. 023이 게이트스택 regen+에이전트 루프 누적으로
      46089>vLLM 44672 초과서 실측(§2ah). tau2엔 `CONTEXT_WINDOW_EXCEEDED` 종료사유가 **정의만 되고 미배선** —
      그 의도된 처리를 구현.
    처방: `step()`을 래핑해 overflow를 잡고 done=True + reason=CONTEXT_WINDOW_EXCEEDED로 정상 종료 →
      run 루프가 done 감지 → finalize → **이미 기록된 부분 tick으로 reward 계산(scored)**. 비수렴 궤적은
      태스크 미완이므로 reward≈0 — **소실(제외) 대신 정직한 실패 계상**(평균 인플레 방지). crash 픽스(_reassemble)로
      부분 tick의 call↔result 쌍이 이미 유효 → replay 안전."""
    try:
        from tau2.orchestrator.orchestrator import BaseOrchestrator as _BO
        from tau2.orchestrator.orchestrator import Orchestrator as _TO
        from tau2.orchestrator.full_duplex_orchestrator import FullDuplexOrchestrator
        from tau2.data_model.simulation import TerminationReason
        from litellm import ContextWindowExceededError
    except Exception as _e:
        print("[T2_OVERFLOW_GUARD] not installed (import): %r" % (_e,), file=sys.stderr, flush=True)
        return

    def _wrap_step(cls):
        """cls.step을 CWE-가드로 래핑. ★e2e9 097 정정(§2ao): 구판은 FullDuplex만 래핑 — text-모드
        (banking 런 실사용)는 BaseOrchestrator.step이라 CWE가 그대로 새어 sim 무효. **양쪽** 래핑
        (서브클래스 override별 개별·이중래핑 방지 마커)."""
        if getattr(cls, "_t2_overflow_wrapped", False) or "step" not in cls.__dict__:
            return
        _orig_step = cls.step

        def _guarded_step(self, *a, **kw):
            try:
                return _orig_step(self, *a, **kw)
            except ContextWindowExceededError as _ce:
                # 예외 전 기록 히스토리는 유효(부분 궤적). done+reason → run 루프 종료 → finalize 채점.
                self.done = True
                self.termination_reason = TerminationReason.CONTEXT_WINDOW_EXCEEDED
                print("[T2_OVERFLOW_GUARD] context window exceeded -> terminate sim as scored failure. %s"
                      % (str(_ce)[:140],), file=sys.stderr, flush=True)
                return None

        cls.step = _guarded_step
        cls._t2_overflow_wrapped = True

    _wrap_step(_BO)                       # BaseOrchestrator.step (기본 구현)
    _wrap_step(_TO)                       # ★Orchestrator.step **override**(text-모드 실사용·e2e10 038t2/097t1
    #                                       CWE 크래시 실측: base만 래핑해선 서브클래스 override가 우회 — 3번째 우회)
    _wrap_step(FullDuplexOrchestrator)    # speech 모드(자체 step override)
    print("[T2_OVERFLOW_GUARD] ON (base+text+full_duplex)", file=sys.stderr, flush=True)

    # ★(§2bd) 평가-시점 id-mismatch post-mortem 덤프 (로그 전용·예외는 그대로 재전파):
    #   rall4 050t2 재현 2회인데 T2_PAIRCHECK(에이전트-턴 검사)는 무발화 = 부패가 마지막
    #   에이전트 턴 이후(유저-측 종반 or 평가 입력 조립)에서 발생. set_state 실패 시
    #   message_history의 (idx·role·requestor·tool·id) 압축 시퀀스를 덤프해 지점 특정.
    try:
        from tau2.environment.environment import Environment as _PmEnv
        if not getattr(_PmEnv, "_t2_pairdump_wrapped", False):
            _orig_ss = _PmEnv.set_state

            def _ss2(self, initialization_data, initialization_actions, message_history):
                # ★T2_PAIRFIX @평가-입력 (§2bi 정정): rall6 실측 — 라이브 PAIRCHECK 침묵 + 평가서만
                #   mismatch = 스왑은 tick→message 변환/직렬화 층에서 발생. 여기(원 검사 직전)서
                #   같은 id 집합·순서-스왑 블록을 호출 순서로 교정(내용 불변·의미론 no-op).
                if os.environ.get("T2_PAIRFIX") == "1" and message_history:
                    try:
                        _nfx = _pairfix(message_history)
                        if _nfx:
                            print("[T2_PAIRFIX] eval-input: reordered %d swapped block(s)" % _nfx,
                                  file=sys.stderr, flush=True)
                    except Exception:
                        pass
                try:
                    return _orig_ss(self, initialization_data, initialization_actions,
                                    message_history)
                except ValueError as _ve:
                    if "id mismatch" in str(_ve) or "Tool message" in str(_ve):
                        print("[T2_PAIRDUMP] set_state failed: %s" % str(_ve)[:160],
                              file=sys.stderr, flush=True)
                        for _i, _m in enumerate(message_history or []):
                            _r = getattr(_m, "role", "?")
                            _rq = getattr(_m, "requestor", "")
                            _tcs = getattr(_m, "tool_calls", None) or []
                            if _tcs:
                                _d = ",".join("%s#%s" % (getattr(t, "name", "?"),
                                                         str(getattr(t, "id", ""))[-8:])
                                              for t in _tcs)
                                print("[T2_PAIRDUMP] %3d %s(%s) CALLS %s" % (_i, _r, _rq, _d),
                                      file=sys.stderr, flush=True)
                            elif _r == "tool":
                                print("[T2_PAIRDUMP] %3d tool(%s) id..%s err=%s"
                                      % (_i, _rq, str(getattr(_m, "id", ""))[-8:],
                                         getattr(_m, "error", False)),
                                      file=sys.stderr, flush=True)
                    raise

            _PmEnv.set_state = _ss2
            _PmEnv._t2_pairdump_wrapped = True
    except Exception as _e2:
        print("[T2_PAIRDUMP] not installed: %r" % (_e2,), file=sys.stderr, flush=True)


def _install_regen_exec():
    """slim _execute_tool_calls: 실행 + auth observe + read-augment(present/nested/calc). deny 없음
    (denied 호출은 생성-레벨서 이미 strip). augment=reads라 replay-safe(기존 apply와 동일 로직)."""
    from tau2.orchestrator.orchestrator import BaseOrchestrator
    orig_exec = getattr(BaseOrchestrator, "_t2_orig_exec", None) or BaseOrchestrator._execute_tool_calls
    BaseOrchestrator._t2_orig_exec = orig_exec

    def exec_augment(self, tool_calls):
        # ★T2_READ_DEDUP (2026-07-19·022 CWE 포렌식): 동일 (name,args) read 재호출 = 실행 생략·
        #   스텁 반환(전체 덤프 재적재 방지). 029 실측: byte-identical KB 20K×2·shell 4.8K×2 = 컨텍스트 22% 낭비.
        #   신선도 보장: **실효 write**(_eff_tool_name+_is_effective_write·정본 술어·user 호출 포함) 실행 시
        #   캐시 전체 무효화 → 상태-의존 read(dispute status 등)는 write 후 항상 재실행. 정보손실 0(원 출력이
        #   위에 실재)·대형 출력만 캐시(T2_READ_DEDUP_MIN 기본 2000자)라 시각/소형 read는 대상 밖.
        stub_ids = set()
        dedup_on = (os.environ.get("T2_READ_DEDUP") == "1"
                    and not getattr(self, "_t2_dedup_bypass", False))
        # ★_t2_dedup_bypass (2026-07-20·smoke023c 포렌식·§2al): 격리 서브의 env 호출은 dedup 제외 —
        #   main이 이미 읽은 (name,args)에 stub("위 출력 참조")을 주면 **서브 문맥엔 '위'가 없어**
        #   빈손 날조를 유발(실측: fetch-iso가 60건 대신 3건 날조→오판정). 서브는 신선 실행·캐시 불변.
        if dedup_on:
            from tau2.data_model.message import ToolMessage as _TM
            cache = self._t2_read_cache = getattr(self, "_t2_read_cache", {})
            # ★loop-break (2026-07-23 C123·rall19 043 실측): 캐시는 대형 출력(min_len)만 저장하므로
            #   한 줄짜리 결과의 동일-read 반복은 스텁이 영원히 안 걸림 — 043.1서 shell grep
            #   'checking_account_id' **15회+ 동일 반복**(값은 DB getter에만 실재) 방치. 동일
            #   (name,args) read가 K회(기본 3) *실행*되면 크기 무관 스텁+redirect. write는 대상 밖·
            #   실효 write 시 카운터도 리셋(신선도 동일 규율).
            seen = self._t2_read_seen = getattr(self, "_t2_read_seen", {})
            loop_k = int(os.environ.get("T2_READ_DEDUP_LOOP_K", "3"))
            to_run, stubs = [], {}
            _dgset = getattr(getattr(self, "agent", None), "_t2_view_digested", None) or set()
            for tc in tool_calls:
                k = _call_key(tc)
                # ★§2bi: 뷰-압축으로 다이제스트된 출력의 재열람은 stub 금지(재실행 허용) —
                #   안 그러면 "위 출력 참조" stub이 다이제스트를 가리켜 재열람 탈출구가 막힘.
                if k in cache and cache.get(k) in _dgset:
                    cache.pop(k, None)
                if ((k in cache or seen.get(k, 0) >= loop_k)
                        and not _is_effective_write(_eff_tool_name(tc))):
                    # ★2026-07-23 (050 flail 근본원인 확정·KB probe): 반복된 read가 KB/검색 도구면
                    #   redirect 힌트 추가 — 도메인-일반. 원인=에이전트가 discoverable 도구를 *함수명*으로
                    #   BM25 검색(점수 0.0·문서 산문엔 함수명 없음)→무한반복. plain-words로 돌림(closure 피드백이
                    #   이미 가진 사실을 flail 지점에 표면화). 배제=[[10]] 생성 무해·행동 게이트 아님.
                    #   ★C123: 키워드 검사를 이름+인자로 확장(043 실측: 도구명 'shell'·grep은 인자에).
                    _dn = ((getattr(tc, "name", "") or "")
                           + " " + str(getattr(tc, "arguments", "") or "")).lower()
                    _redir = ""
                    if "search" in _dn or "bm25" in _dn or "kb_" in _dn or "grep" in _dn:
                        _redir = (" Do NOT repeat this exact search. If you are looking up a discoverable "
                                  "tool, note that a bare function-name query matches no document text — "
                                  "search PLAIN WORDS describing the action/step (the everyday words a policy "
                                  "document would use), not the tool's function name. If you already have the "
                                  "information you need, proceed to the next step instead of searching again.")
                    # ★C194(2026-07-26·16건 정독 실측): 동일-문구 stub은 루프를 못 끊는다 — 041은
                    #   동일 KB 쿼리 15회+(stub 매번 발화·행동 불변)·020/027 5~6회·035/012 동형.
                    #   동일 입력→동일 출력 어트랙터라 같은 텍스트 반복 제시는 같은 선택을 재생산.
                    #   교정: ①반복 횟수를 문구에 넣어 매번 다른 텍스트 ②3회+ 시 행동-전환 지시
                    #   +error 채널 승격(피드백 채널 자체 변경). read 한정·도메인 리터럴 0.
                    _rep = self._t2_dup_rep = getattr(self, "_t2_dup_rep", {})
                    _rep[k] = _rep.get(k, 0) + 1
                    _n_rep = _rep[k]
                    _esc = ""
                    if _n_rep >= 3:
                        _esc = (" You have now issued this IDENTICAL call %d times and the result "
                                "has not changed once — repeating it again cannot produce new "
                                "information. Change what you do: use DIFFERENT search words, or "
                                "act on the information you already have, or ask the customer. Do "
                                "not issue this same call again." % _n_rep)
                    stubs[getattr(tc, "id", None)] = _TM(
                        id=tc.id, role="tool",
                        requestor=getattr(tc, "requestor", "assistant"), error=(_n_rep >= 3),
                        content="[DUPLICATE-READ] This exact call (same tool, same arguments) was "
                                "already executed earlier in this conversation; its full output is "
                                "shown above and has not changed. Refer to that output instead of "
                                "re-reading." + _redir + _esc)
                    stub_ids.add(getattr(tc, "id", None))
                    self._t2_read_dedup = getattr(self, "_t2_read_dedup", 0) + 1
                    print("[T2_READ_DEDUP] stub tool=%s" % getattr(tc, "name", None),
                          file=sys.stderr, flush=True)
                else:
                    # ★P5-3(C208①·DAY5_PRESCRIPTIONS §P5-3·T2_READ_NEARDUP=1·기본 OFF): 근사-중복
                    #   질의 안내 — 018/028/029 [S]: "get credit card transactions"↔"tool to get …"↔
                    #   "…by user" 재표현 5연발(각 15~23k자)이 exact-dedup을 전부 통과해 창 60% 소진.
                    #   판정=정규화 토큰집합 Jaccard(도메인 무관·검색류 read만)·안내 스텁(원 출력은
                    #   위에 실재). 오탐(정당 질의-정련) 리스크 → 기본 OFF·격리 arm 계측 후 승격.
                    _nd_hit = None
                    if (os.environ.get("T2_READ_NEARDUP") == "1"
                            and not _is_effective_write(_eff_tool_name(tc))):
                        _dn2 = ((getattr(tc, "name", "") or "")
                                + " " + str(getattr(tc, "arguments", "") or "")).lower()
                        if "search" in _dn2 or "bm25" in _dn2 or "kb_" in _dn2:
                            _stop = {"the", "a", "an", "for", "to", "of", "in", "on", "how",
                                     "tool", "get", "and", "or", "with", "by", "is", "do"}
                            _tk = {w for w in re.findall(r"[a-z0-9_]+", _dn2) if w not in _stop}
                            _hist = self._t2_nd_hist = getattr(self, "_t2_nd_hist", [])
                            for _pk, _ptk in _hist:
                                if _pk != getattr(tc, "name", None) or not (_tk | _ptk):
                                    continue
                                _j = len(_tk & _ptk) / float(len(_tk | _ptk))
                                if _j >= float(os.environ.get("T2_READ_NEARDUP_J", "0.8")):
                                    _nd_hit = sorted((_tk ^ _ptk))[:6]
                                    break
                            if _nd_hit is None:
                                _hist.append((getattr(tc, "name", None), _tk))
                    if _nd_hit is not None:
                        stubs[getattr(tc, "id", None)] = _TM(
                            id=tc.id, role="tool",
                            requestor=getattr(tc, "requestor", "assistant"), error=False,
                            content=("[NEAR-DUPLICATE-READ] This query is nearly identical to an "
                                     "earlier one in this conversation (it differs only in: %s); "
                                     "the earlier output is shown above and this rephrasing will "
                                     "return largely the same documents. Refine with genuinely NEW "
                                     "terms, or proceed with the information you already have."
                                     % (", ".join(_nd_hit) or "(word order)")))
                        stub_ids.add(getattr(tc, "id", None))
                        print("[T2_READ_NEARDUP] stub tool=%s diff=%s"
                              % (getattr(tc, "name", None), _nd_hit),
                              file=sys.stderr, flush=True)
                        continue
                    if not _is_effective_write(_eff_tool_name(tc)):
                        seen[k] = seen.get(k, 0) + 1     # C123: 실행되는 read만 계수
                    to_run.append(tc)
            ran = orig_exec(self, to_run) if to_run else []
            _rby = {getattr(r, "id", None): r for r in (ran or [])}
            # ★크래시 픽스(2026-07-20·023/031 infrastructure_error): 결과는 tool_calls와 **1:1·순서**
            #   보장. 안 그러면 full-duplex tick의 agent_tool_calls↔agent_tool_results 쌍이 깨져
            #   eval replay(`environment.get_actions_from_messages`)가 "Tool call id mismatch"로 크래시.
            #   구판은 id 매칭 실패 시 None을 **드롭**해 results가 짧아졌다(비결정론=하위 레이어 id/순서 의존).
            #   id 매칭 우선·id없는 결과만 위치폴백·그래도 없으면 에러 ToolMessage로 채운다(드롭 금지).
            _idless = iter([r for r in (ran or []) if getattr(r, "id", None) is None])
            results = []
            for tc in tool_calls:
                _tid = getattr(tc, "id", None)
                r = stubs.get(_tid) if _tid in stub_ids else _rby.get(_tid)
                if r is None:
                    r = next(_idless, None)
                if r is None:
                    r = _TM(id=_tid, role="tool", requestor=getattr(tc, "requestor", "assistant"),
                            error=True, content="(no result returned for this tool call)")
                results.append(r)
            min_len = int(os.environ.get("T2_READ_DEDUP_MIN", "2000"))
            for tc in to_run:
                out = _rby.get(getattr(tc, "id", None))
                if out is None:
                    continue
                if _is_effective_write(_eff_tool_name(tc)):
                    if not getattr(out, "error", False):
                        cache.clear()  # 세상이 바뀜 → 이전 read 신선도 보장 불가
                        seen.clear()   # C123: loop-break 계수도 동일 규율로 리셋
                elif (not getattr(out, "error", False) and len(_content_str(out) or "") >= min_len
                      and _dedup_cache_safe(self, getattr(tc, "name", "") or "")):
                    # ★§2at: env-mutating(unlock 등)은 캐시 금지 — stub이 히스토리에 남으면 replay 불일치
                    # ★§2bi: 값=결과 메시지 id(뷰-압축 다이제스트 면제 판정용·구판 True와 truthy 동일)
                    cache[_call_key(tc)] = getattr(out, "id", True)
        else:
            results = orig_exec(self, tool_calls)
        env = getattr(self, "environment", None)
        a2 = _domain_a2(getattr(env, "domain_name", None)) if env is not None else None
        if a2 is None:
            return results
        gate = getattr(getattr(self, "agent", None), "_t2_gate", None)
        by_id = {getattr(r, "id", None): r for r in (results or [])
                 if getattr(r, "id", None) not in stub_ids}
        auth_tools = a2["_auth_tools"]
        present_on = os.environ.get("T2_PRESENT_READS") == "1"
        g6 = next((g for g in a2["gates"] if g.get("kind") == "select_confirm"), None) if present_on else None
        nested_specs = (a2.get("present_specs") or []) if os.environ.get("T2_PRESENT_NESTED") == "1" else []
        calc_specs = (a2.get("calc_specs") or []) if os.environ.get("T2_CALC") == "1" else []
        for tc in tool_calls:
            out = by_id.get(getattr(tc, "id", None))
            if out is None or getattr(out, "error", False):
                continue
            name = getattr(tc, "name", None)
            if gate is not None and name in obs_tools_g(gate):
                gate.observe(name, _args_dict(tc), _content_str(out))
            if present_on and g6 is not None and gate is not None and name == g6.get("user_producer"):
                uid = _args_dict(tc).get(g6.get("user_id_arg", "user_id"))
                summ = candidate_summary(gate.resolvers, g6, uid)
                if summ:
                    try:
                        out.content = _content_str(out) + summ
                    except Exception:
                        pass
            _rec = _parse_json(_content_str(out)) if (nested_specs or calc_specs) else None
            if nested_specs and _rec is not None:
                spec = next((s for s in nested_specs if s.get("trigger_tool") == name), None)
                if spec is not None:
                    summ = nested_candidate_summary(_rec, spec)
                    if summ:
                        try:
                            out.content = _content_str(out) + summ
                        except Exception:
                            pass
            if calc_specs and _rec is not None:
                cs = [s for s in calc_specs if s.get("trigger_tool") == name]
                if cs:
                    facts = compute_facts(_rec, cs)
                    if facts:
                        try:
                            out.content = _content_str(out) + facts
                        except Exception:
                            pass
        return results

    BaseOrchestrator._execute_tool_calls = exec_augment


def apply_gate_regen(max_regen=1):
    """replay-safe 게이트 (apply() 대체·A/B 위해 apply()는 보존). T2_GATE_KINDS 필터 동일 지원."""
    import tau2.agent.llm_agent as la
    from tau2.agent.llm_agent import LLMAgent
    from tau2.orchestrator.orchestrator import BaseOrchestrator
    from tau2.data_model.message import ToolMessage, MultiToolMessage

    # (1) orchestrator.__init__ → env-bound gate를 agent에 주입 (per-sim)
    orig_init = BaseOrchestrator.__init__

    def init_inject(self, *a, **kw):
        orig_init(self, *a, **kw)
        env = getattr(self, "environment", None)
        ag = getattr(self, "agent", None)
        a2 = _domain_a2(getattr(env, "domain_name", None)) if env is not None else None
        if a2 is not None and ag is not None and hasattr(ag, "_generate_next_message"):
            _kinds = os.environ.get("T2_GATE_KINDS")
            gl = a2["gates"]
            if _kinds:
                allow = {k.strip() for k in _kinds.split(",") if k.strip()}
                gl = [g for g in a2["gates"] if g.get("kind") in allow]
            ag._t2_gate = GateInterpreter(gl, resolvers=resolvers_from_env(env))
            ag._t2_a2 = a2
            ag._t2_orch = self
            ag._t2_max_regen = max_regen

    BaseOrchestrator.__init__ = init_inject

    # (2) LLMAgent._generate_next_message → gate deny + regen + R8 종단
    orig_gen = LLMAgent._generate_next_message

    def gen_gated(self, message, state):
        gate = getattr(self, "_t2_gate", None)
        if gate is None:
            return orig_gen(self, message, state)
        a2 = self._t2_a2
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)
        _rebuild_gate_state(gate, a2, state.messages)
        last_user = _regen_last_user(state.messages)
        # ★NOTICE-PERGATE 커링 (apply()와 동형)
        transfer_sent = lambda text: _regen_transfer_sent(state.messages, text)  # noqa: E731
        base = state.system_messages + state.messages
        am = la.generate(model=self.llm, tools=self.tools, messages=base,
                         call_name="agent_response", **self.llm_args)
        n, max_regen_ = 0, getattr(self, "_t2_max_regen", 1)
        while n < max_regen_:
            denied = _denied_calls(am, gate, last_user, transfer_sent)
            if not denied:
                break
            _budget_tick(self)
            dids = {id(tc): (gid, why) for tc, gid, why in denied}
            fb = [am]
            for c in (am.tool_calls or []):
                if id(c) in dids:
                    gid, why = dids[id(c)]
                    content = f"Error: [POLICY GATE {gid}] {why}"
                else:
                    content = "Error: [POLICY GATE] resolve the blocked action first; do not call this tool yet."
                fb.append(ToolMessage(id=c.id, role="tool", requestor="assistant",
                                      error=True, content=content))
            am = la.generate(model=self.llm, tools=self.tools, messages=base + fb,
                             call_name="agent_response_gateregen", **self.llm_args)
            n += 1
        # R8 종단: K 소진 후에도 deny → 차단 mutating 호출 제거(히스토리 replay-clean 보장)
        # ★예산(R1·리뷰⚠️2): exhaustion turn은 위 루프서 이미 1 tick 소비 → 여기서 재-tick 안 함
        #   = "차단 turn당 1 error"로 현 게이트와 동일 예산압박(best-of-K도 double-charge도 아님).
        denied = _denied_calls(am, gate, last_user, transfer_sent)
        if denied:
            denied_ids = {tc.id for tc, _, _ in denied}
            kept = [tc for tc in (am.tool_calls or []) if tc.id not in denied_ids]
            am.tool_calls = kept or None
            note = "; ".join(f"[{gid}] {(why or '')[:70]}" for _, gid, why in denied)
            am.content = (am.content or "") + _BLOCK_NOTE + " (" + note + ")"
        return am

    LLMAgent._generate_next_message = gen_gated

    # (3) slim exec: 실행 + auth observe + read-augment (deny 없음)
    _install_regen_exec()
    return orig_gen


# ═══════════════════════════════════════════════════════════════════════════
# ★E-COMP 통합 생성-레벨 검증 체인 (RETAIL_PASS_COMPOSITION_DESIGN_2026_07_10 §2·리뷰 반영)
#   = 게이트(replay-safe regen) + provenance 검증기 + (선택) DISAMB 를 한 패치로.
#   예산 semantics는 두 GO arm을 *그대로 승계* (리뷰 블로킹1 — 귀속 오염 방지):
#     게이트: deny 피드백 라운드 최대 1회·그 라운드만 num_errors++ (apply_gate_regen K=1 동일)
#             잔존 deny = R8 strip(재과금 없음·replay-safe)
#     prov  : 무과금·재발화 최대 max_prov_retries=4 (apply_provenance_regen=C53 동일)
#     DISAMB: 1회 재확인·★채택 전 게이트 재검사 — deny면 원 am 유지 (리뷰 블로킹2)
#   present/autofetch/GROUND 미지원(C34 폐기·scope 밖). 도메인-일반: 게이트·힌트 전부 A2-구동.
# ═══════════════════════════════════════════════════════════════════════════


def _compact_view(messages, keep_recent=6, min_len=800, min_total=60000, msg_cap=0):
    """★뷰-압축 (T2_VIEW_COMPACT=1·2026-07-21 §2bi·097 컨텍스트 레버·사용자 승인 기본안).
    원리: **커밋 히스토리는 불변**(replay-불변식 자동 충족·게이트/관문은 원문 대조 유지) — LLM
    생성-시점 프롬프트 뷰에서만 오래된 벌크 tool 출력을 기계적 다이제스트(head+tail 절단)로 대체.
    read 액션의 주체는 모델로 유지(서브-이관 변형은 [[05]]③ autofetch-류로 기각·§2bi 문답).
    - 대상: role=tool·비에러·min_len 초과·최근 keep_recent개 제외. 전체 뷰가 min_total 미만이면 무개입.
    - 다이제스트=순수 절단(head 300+tail 150)+안내문 — 엔진의 내용 추출/합성 0([[03b]]).
    - 반환: (뷰 리스트, 다이제스트된 ToolMessage id 집합) — id는 READ_DEDUP 면제(재열람 탈출구)용.
    ★P5(C208①⑤·DAY5_PRESCRIPTIONS §P5·2026-07-28):
    - min_total 기본 120,000→60,000자 — 구 문턱은 사망선(≈40k tok) 위라 32sim 중 6회만 발동.
    - msg_cap(신설·>0일 때): **최신 배치**(마지막 assistant 이후의 전 tool 출력 — 리뷰 필수1:
      멀티-콜 턴은 출력들이 함께 커밋되고 다음 생성이 첫 노출이라 "최신 1개"면 미열람 절단)를
      제외한 비에러 tool 출력이 cap 초과면 **총량과 무관하게** 다이제스트. 모델은 도착 턴에
      전문을 봤고 이후 턴부터 다이제스트(read 주체=모델 유지)."""
    msgs = list(messages)
    last_a = max([i for i, m in enumerate(msgs)
                  if getattr(m, "role", None) == "assistant"] or [-1])
    batch = {i for i in range(last_a + 1, len(msgs))
             if getattr(msgs[i], "role", None) == "tool"}
    total = sum(len(str(getattr(m, "content", "") or "")) for m in msgs)
    if total < int(min_total) and not msg_cap:
        return msgs, set()
    tool_idx = [i for i, m in enumerate(msgs) if getattr(m, "role", None) == "tool"]
    keep = (set(tool_idx[-int(keep_recent):]) if keep_recent else set()) | batch
    out, digested = [], set()
    for i, m in enumerate(msgs):
        c = getattr(m, "content", None)
        _is_tool = (getattr(m, "role", None) == "tool" and isinstance(c, str)
                    and not getattr(m, "error", False))
        _hit = False
        if _is_tool and i not in batch:
            if msg_cap and len(c) > int(msg_cap):
                _hit = True                                   # P5-2 per-메시지 캡
            elif total >= int(min_total) and i not in keep and len(c) > int(min_len):
                _hit = True                                   # 기존 총량-문턱 경로
        if _hit:
            d = (c[:300] + "\n...[view digest: %d chars total. The FULL output was recorded "
                 "earlier in this conversation; re-call the same tool if you need the "
                 "details again.]...\n" % len(c) + c[-150:])
            try:
                m2 = m.model_copy(update={"content": d})
            except Exception:
                import copy as _cp
                m2 = _cp.copy(m)
                try:
                    m2.content = d
                except Exception:
                    m2 = m
            if getattr(m2, "content", None) == d:
                digested.add(getattr(m, "id", None))
                out.append(m2)
            else:
                out.append(m)
        else:
            out.append(m)
    return out, digested


def _annotate_view(messages, specs):
    """★뷰 필드-주석 (T2_VIEW_ANNOTATE=1·2026-07-22 §2bs·rall10 054 실측).
    원리: 커밋 히스토리는 불변(_compact_view와 동일·replay-safe) — 생성-시점 뷰의 role=tool
    비에러 출력에 A2 선언 부분문자열 집합(contains 전부 공존)이 보이면 A2 주석(note)을 append.
    054: 교체주문이 기록한 status: CLOSED(舊카드)·account_status: ACTIVE 병존 뷰를 "계좌 폐쇄"로
    오독→CLI 전면 거부. KB(replacements_003)=카드≠계좌 명시 — 주석 텍스트 전부 A2(KB 인용).
    엔진=부분문자열 공존 검사+append만(내용 추출/합성 0·[[03b]]·도메인 리터럴 0).
    반환: (뷰 리스트, 주석된 메시지 수)."""
    out, n = [], 0
    for m in messages:
        c = getattr(m, "content", None)
        if (getattr(m, "role", None) == "tool" and isinstance(c, str)
                and not getattr(m, "error", False)):
            adds = [str(sp.get("note")) for sp in specs
                    if sp.get("note") and (sp.get("contains") or [])
                    and all(s in c for s in (sp.get("contains") or []))
                    and str(sp.get("note")) not in c]
            if adds:
                d = c + "\n" + "\n".join(adds)
                try:
                    m2 = m.model_copy(update={"content": d})
                except Exception:
                    import copy as _cp
                    m2 = _cp.copy(m)
                    try:
                        m2.content = d
                    except Exception:
                        m2 = m
                if getattr(m2, "content", None) == d:
                    n += 1
                    out.append(m2)
                    continue
        out.append(m)
    return out, n


def _pairfix(messages):
    """★커밋-시점 pairing 교정 (T2_PAIRFIX=1·2026-07-21 §2bi·rall6 054t2 PAIRDUMP 실측).
    실측 부패 시그니처: 다중 동일-도구 read 턴에서 결과 **집합은 정확·순서만 스왑**(호출 [a,b,c] vs
    결과 [a,c,b]) → 평가 replay "Tool call id mismatch"로 sim 무효. 부패 주입층은 미상(우리 두 래퍼는
    id-재조립 정합·§2ah/_reassemble)이나, **같은 id 집합·순서 불일치** 블록을 호출 순서로 재정렬하면
    내용 불변의 의미론 no-op이고 pairing이 복원된다(read는 replay 재실행-비교 제외·env-일치).
    반환: 교정 블록 수. messages는 in-place 재배열."""
    fixed, i, n = 0, 0, len(messages)
    while i < n:
        m = messages[i]
        tcs = getattr(m, "tool_calls", None) or []
        if getattr(m, "role", None) in ("assistant", "user") and tcs:
            j, k = i + 1, len(tcs)
            block = messages[j:j + k]
            if (len(block) == k and all(getattr(b, "role", None) == "tool" for b in block)):
                want = [getattr(t, "id", None) for t in tcs]
                have = [getattr(b, "id", None) for b in block]
                if want != have and set(want) == set(have) and len(set(have)) == k:
                    by = {getattr(b, "id", None): b for b in block}
                    messages[j:j + k] = [by[w] for w in want]
                    fixed += 1
                i = j + k
            else:
                i += 1
        else:
            i += 1
    return fixed


def _paircheck(messages):
    """커밋 히스토리의 call↔result 쌍 불변식 검사(로그 전용·2026-07-21 §2bd).
    evaluator `get_actions_from_messages`(environment.py:334)와 동일 보행 — rall4 054t1/050t2가
    평가-시점 "Tool call id mismatch"로 sim 무효(신규 계열·전 런 0회)인데 메시지 덤프가 없어
    부패 지점 특정 불가 → 라이브서 매 턴 검사·첫 위반의 도구명/id를 로그. 행동 무변경."""
    i, n = 0, len(messages)
    while i < n:
        m = messages[i]
        tcs = getattr(m, "tool_calls", None) or []
        if getattr(m, "role", None) in ("assistant", "user") and tcs:
            j = i + 1
            for tc in tcs:
                if j >= n:
                    return None          # 말미 미실행 호출 = 아직 결과 대기 중(정상·판정 불가)
                tm = messages[j]
                if getattr(tm, "role", None) != "tool":
                    return ("idx %d: %s(id=%s) 다음이 tool이 아님 role=%s"
                            % (j, getattr(tc, "name", None), getattr(tc, "id", None),
                               getattr(tm, "role", None)))
                if getattr(tm, "id", None) != getattr(tc, "id", None):
                    return ("idx %d: call %s id=%s vs result id=%s"
                            % (j, getattr(tc, "name", None), getattr(tc, "id", None),
                               getattr(tm, "id", None)))
                j += 1
            i = j
        else:
            i += 1
    return None


def apply_unified_regen(max_prov_retries=4, domain=None, disamb=False, use_badwords=False,
                        ground=False, disamb_mode="dialog", prov_mode="full"):
    import sys as _sys
    from tau2.agent.llm_agent import LLMAgent
    import tau2.agent.llm_agent as la
    from tau2.data_model.message import ToolMessage, UserMessage, MultiToolMessage
    from tau2.orchestrator.orchestrator import BaseOrchestrator

    a2d = _domain_a2(domain) if domain else None
    hints = a2d["_hints"] if a2d else DEFAULT_ARG_HINTS
    placeholders = a2d["_placeholders"] if a2d else DEFAULT_PLACEHOLDERS
    disamb_tools = _confirm_write_tools(a2d) if disamb else set()
    env_args = _env_verified_args(a2d) if prov_mode == "rescue" else set()   # T5-C P-C (B3)
    sub_args = set((a2d or {}).get("disamb_sub_args") or [])                 # T5-C P-B (B2)
    if os.environ.get("T2_DISAMB_ORDER") == "1":            # ★order operand도 disamb 대상(filter-then-ask·env opt-in)
        sub_args |= {"order", "order_id"}

    # (1) per-sim 게이트 주입 (apply_gate_regen과 동일 패턴·T2_GATE_KINDS 지원)
    orig_init = BaseOrchestrator.__init__

    def init_inject(self, *a, **kw):
        orig_init(self, *a, **kw)
        env = getattr(self, "environment", None)
        ag = getattr(self, "agent", None)
        a2 = _domain_a2(getattr(env, "domain_name", None)) if env is not None else None
        if a2 is not None and ag is not None and hasattr(ag, "_generate_next_message"):
            _kinds = os.environ.get("T2_GATE_KINDS")
            gl = a2["gates"]
            if _kinds:
                allow = {k.strip() for k in _kinds.split(",") if k.strip()}
                gl = [g for g in a2["gates"] if g.get("kind") in allow]
            ag._t2_gate = GateInterpreter(gl, resolvers=resolvers_from_env(env))
            ag._t2_a2 = a2
            ag._t2_orch = self

    BaseOrchestrator.__init__ = init_inject

    def _append(state, message):
        if isinstance(message, UserMessage) and getattr(message, "is_audio", False):
            raise ValueError("audio not supported")
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

    def _gen(self, work, bad_words, call_name, tool_choice=None):
        kw = dict(self.llm_args)
        if use_badwords and bad_words:
            eb = dict(kw.get("extra_body") or {})
            eb["bad_words"] = sorted(bad_words)
            kw["extra_body"] = eb
        if tool_choice:                          # ★레버 A(2026-07-18): tau2 `generate`의 일급 파라미터로 통과
            kw["tool_choice"] = tool_choice
            # ★max_tokens 하한 (2026-07-23 근본원인 규명·vLLM #19051/#36794): tool_choice='required'는
            #   강제 tool-call JSON을 *완성*해야 하는데 max_tokens가 작으면 중간 절단 → hermes 파서 EOF →
            #   vLLM이 오도성 `__log_extra_fields__` 400 보고(실측: mt=20 실패·mt≥100 성공·전 도구수). 라이브
            #   llm_args엔 max_tokens 미설정(=vLLM 기본=문맥잔량=대형)이라 무영향 — 소형 설정 시만 상향(방어).
            _mt = kw.get("max_tokens")
            if _mt is not None and _mt < _FORCE_MIN_TOKENS:
                kw["max_tokens"] = _FORCE_MIN_TOKENS
        try:
            return la.generate(model=self.llm, tools=self.tools,
                               messages=self._system_messages + work, call_name=call_name, **kw)
        except Exception as _ce:
            # ★force_required 안전판 (2026-07-23): 하한 보장 뒤에도 남는 400 = 병리적 케이스(퇴행 루프서
            #   강제 시 닫히지 않는 runaway JSON·mt를 아무리 키워도 미완=039 실측)·기타 transient. → 강제 없이
            #   1회 재시도(넛지·work 유지=효과 보존·프로즈 봉쇄만 포기). 크래시(sim 무효) 대신 우아한 강등.
            #   un_fb 등 모든 force 경로 공용.
            if tool_choice and ("BadRequest" in type(_ce).__name__ or " 400" in (" " + str(_ce))):
                print("[T2_FORCE] tool_choice=%s rejected (400) -> retry without force" % tool_choice,
                      file=_sys.stderr, flush=True)
                kw.pop("tool_choice", None)
                try:
                    return la.generate(model=self.llm, tools=self.tools,
                                       messages=self._system_messages + work, call_name=call_name, **kw)
                except Exception as _ce2:
                    _ce = _ce2
            # ★CWE graceful-stop @_gen (§2bf·rall5 실측): step-래핑 가드가 4번째 경로로 우회 —
            #   LLM_DIAG가 특정한 두 누출(call_name=agent_response·followup_decision) 모두 _gen 경유.
            #   여기서 잡아 orch.done+CONTEXT_WINDOW_EXCEEDED(§2ah 의도된 종료사유)로 우아한 종료 →
            #   sim 무효(infra) 대신 부분 궤적 채점(정직한 실패 계상). step-가드는 백스톱 존치.
            if "ContextWindow" not in type(_ce).__name__:
                raise
            # ★P1(C208①·DAY5_PRESCRIPTIONS §P1): 동적 max_tokens — day5 ctxover 7건의 직접 사인은
            #   고정 예약(8192)이 깎은 천장(48,640−8,192=40,448·7건 전부 36.5~40.2k서 사망·모델 창
            #   초과 0건). vLLM 에러 원문이 정확한 수를 주므로 **추정 없이** 파싱→축소→1회 재시도.
            #   플로어 미만 = 진짜 창 소진 → 기존 graceful-stop 그대로. 도메인 무관·판단 0.
            if os.environ.get("T2_DYN_MT") == "1":
                _newmt = _dyn_mt_target(str(_ce),
                                        margin=int(os.environ.get("T2_DYN_MT_MARGIN", "64")),
                                        floor=int(os.environ.get("T2_MT_FLOOR", "256")))
                if _newmt is not None:
                    print("[T2_DYN_MT] shrink %s->%d (at %s)"
                          % (kw.get("max_tokens"), _newmt, call_name),
                          file=_sys.stderr, flush=True)
                    self._t2_dyn_shrunk = True         # P8이 참조(천장 근접 시 재제시 생략)
                    kw2 = dict(kw)
                    kw2["max_tokens"] = _newmt
                    try:
                        return la.generate(model=self.llm, tools=self.tools,
                                           messages=self._system_messages + work,
                                           call_name=call_name, **kw2)
                    except Exception as _ce3:
                        if "ContextWindow" not in type(_ce3).__name__:
                            raise
                        _ce = _ce3                   # 재시도도 CWE → graceful-stop으로
            _orch = getattr(self, "_t2_orch", None)
            try:
                from tau2.data_model.simulation import TerminationReason as _TR2
                if _orch is not None:
                    _orch.done = True
                    _orch.termination_reason = _TR2.CONTEXT_WINDOW_EXCEEDED
            except Exception:
                pass
            print("[T2_OVERFLOW_GUARD] CWE at %s -> graceful stop (scored partial)" % call_name,
                  file=_sys.stderr, flush=True)
            _txt = "(context limit reached - conversation ending)"
            try:
                from tau2.data_model.message import AssistantMessage as _AM2
                return _AM2(role="assistant", content=_txt)
            except Exception:
                import types as _types
                return _types.SimpleNamespace(role="assistant", content=_txt, tool_calls=None)

    def unified(self, message, state):
        if not hasattr(self, "_t2_static_bl"):
            self._t2_static_bl = _static_blacklist(self.tools, placeholders)
            self._t2_session_bl = set()
        self._system_messages = state.system_messages
        _append(state, message)
        # ★T2_PAIRFIX (§2bi): 같은 id 집합·순서-스왑 블록을 호출 순서로 교정(내용 불변·크래시 계열 종결).
        if os.environ.get("T2_PAIRFIX") == "1":
            try:
                _nfx = _pairfix(state.messages)
            except Exception:
                _nfx = 0
            if _nfx:
                print("[T2_PAIRFIX] reordered %d swapped result block(s)" % _nfx,
                      file=_sys.stderr, flush=True)
        # ★T2_PAIRCHECK (§2bd·로그 전용): 커밋 히스토리 call↔result 쌍 불변식 라이브 검사 —
        #   rall4 id-mismatch(평가-시점 크래시·덤프 부재)의 부패 지점을 다음 발생 시 특정.
        if os.environ.get("T2_PAIRCHECK") == "1" and not getattr(self, "_t2_paircheck_hit", False):
            try:
                _pc = _paircheck(state.messages)
            except Exception:
                _pc = None
            if _pc:
                self._t2_paircheck_hit = True
                print("[T2_PAIRCHECK] pairing broken: %s" % _pc, file=_sys.stderr, flush=True)
        gate = getattr(self, "_t2_gate", None)
        a2 = getattr(self, "_t2_a2", None)
        last_user = transfer_sent = None
        if gate is not None:
            _rebuild_gate_state(gate, a2, state.messages)
            last_user = _regen_last_user(state.messages)
            # ★NOTICE-PERGATE 커링 (apply()와 동형)
            transfer_sent = lambda text: _regen_transfer_sent(state.messages, text)  # noqa: E731
        ctx = _ctx_with_toolnames(self, _ctx_from_messages(state.messages))

        # ★E-PLAN v1.3 (T2_EPLAN=1): committed 히스토리서 결정론 ledger 재구성(관측만·[[10]])
        #   discovery L1/L2 = read-강제 deny(§1.5 허용축)·CP5 리마인더 소비 = 생성-레벨(비커밋)
        ep_led = ep_spec = _epmod = None
        ep_writes = set()
        if os.environ.get("T2_EPLAN") == "1" and a2 is not None and a2.get("eplan"):
            try:
                import t2_eplan_patch as _epmod
                ep_spec = a2.get("eplan")
                # ep_writes = confirm-gate write 도구 ∪ eplan spec write_tools(C101 (c)·디스패처 nested용)
                ep_writes = _confirm_write_tools(a2) | set(ep_spec.get("write_tools") or ())
                ep_led = _epmod.build_ledger_from_messages(state.messages, ep_spec, ep_writes)
            except Exception as _e:
                print("[T2_EPLAN] ledger build failed: %r" % (_e,), file=_sys.stderr, flush=True)
                ep_led = None

        # ★T2_WRITE_EVIDENCE unified 배선(2026-07-19 028 포렌식): 구 apply()에만 있던 WEV가
        #   unified 런(T2_GATE_REGEN∧T2_GROUND)서 死코드 → deny 0회/증거없는 update 6건 통과.
        #   생성-레벨 deny(ep/cons/ra/te와 동렬·무과금·sim당 cap)로 이설. 검사 코어=_wev_deny_msgs 공유.
        wev_specs = (a2.get("write_evidence_specs") or []) \
            if (a2 is not None and os.environ.get("T2_WRITE_EVIDENCE") == "1") else []
        # ★T2_WRITE_ARG_GROUND (§2bs): WEV 블록에 합류(동일 라운드·cap·적용 배관 공유 — 문서화됨)
        wag_specs = (a2.get("write_arg_grounding") or []) \
            if (a2 is not None and os.environ.get("T2_WRITE_ARG_GROUND") == "1") else []
        # ★T2_REF_VERIFY (C128/C129): 결정론 참조-검증기 spec(도구/필드/문구=A2)·WEV 블록 합류
        rv_specs = (a2.get("ref_verify") or []) \
            if (a2 is not None and os.environ.get("T2_REF_VERIFY") == "1") else []
        _wev_cap = int(os.environ.get("T2_WEV_CAP", "8"))
        # ★T2_HAVE_VALUE (C115·2026-07-23): have-value→act 일반레버 spec(도구/인자/신호/문구=A2)
        hv_specs = (a2.get("have_value_reask") or []) \
            if (a2 is not None and os.environ.get("T2_HAVE_VALUE") == "1") else []
        # ★T2_VALUE_ACQUIRE (C119): give 표면화 spec(도구/문구=A2)
        va_specs = (a2.get("value_acquisition") or []) \
            if (a2 is not None and os.environ.get("T2_VALUE_ACQUIRE") == "1") else []

        def bw():
            return [v for v in (self._t2_static_bl | self._t2_session_bl) if v.lower() not in ctx]

        work = list(state.messages)
        # ★T2_VIEW_COMPACT (§2bi): 생성-뷰만 압축(커밋 히스토리·게이트 ctx는 원문 유지=replay-safe).
        if os.environ.get("T2_VIEW_COMPACT") == "1":
            work, _dg = _compact_view(
                state.messages,
                keep_recent=int(os.environ.get("T2_VIEW_COMPACT_KEEP", "6")),
                min_len=int(os.environ.get("T2_VIEW_COMPACT_MINLEN", "800")),
                # ★P5: 기본 60,000자(구 120,000=사망선 위·day5 6/32만 발동)·per-메시지 캡 8,000자.
                min_total=int(os.environ.get("T2_VIEW_COMPACT_MINTOTAL", "60000")),
                msg_cap=int(os.environ.get("T2_VIEW_MSG_CAP", "8000")))
            self._t2_view_digested = _dg
            if _dg and not getattr(self, "_t2_vc_logged", False):
                self._t2_vc_logged = True
                print("[T2_VIEW_COMPACT] active: %d tool output(s) digested in view"
                      % len(_dg), file=_sys.stderr, flush=True)
        # ★T2_VIEW_ANNOTATE (2026-07-22 §2bs): A2-선언 필드-주석을 생성-뷰에만 부가(비커밋).
        #   COMPACT 뒤에 적용 — 실제로 보이는 뷰 기준으로만 주석. 기본 OFF(거동보존).
        if os.environ.get("T2_VIEW_ANNOTATE") == "1" and a2 is not None:
            _vas = a2.get("view_field_annotations") or []
            if _vas:
                work, _nva = _annotate_view(work, _vas)
                if _nva and not getattr(self, "_t2_va_logged", False):
                    self._t2_va_logged = True
                    print("[T2_VIEW_ANNOTATE] active: %d tool output(s) annotated in view"
                          % _nva, file=_sys.stderr, flush=True)
        _rem = getattr(self, "_t2_eplan_reminder", None)
        if _rem:  # CP5 walk 리마인더(작업버퍼만·히스토리 비커밋 = 채널 절대규칙)
            self._t2_eplan_reminder = None
            try:
                work = work + [UserMessage(role="user", content=_rem)]
            except TypeError:
                work = work + [UserMessage(content=_rem)]
        # ★P2(C208②·DAY5_PRESCRIPTIONS §P2): replay-비교 대상 도구의 피드백 뷰-채널 소비 —
        #   prekb가 큐잉(`_t2_view_fb`)·여기서 작업버퍼에만 주입(히스토리 비커밋=위 채널 절대규칙 동일).
        _vfb = getattr(self, "_t2_view_fb", None)
        if _vfb:
            # ★F5(C210): 항목=[텍스트, 잔여횟수] — 이번 뷰에 주입 후 잔여>0이면 다음 생성에도
            #   재노출(day6 033 [S]: 1회 노출은 무시됨). 구형 str 항목=1회(하위호환).
            _texts, _keep = [], []
            for _it in _vfb:
                if isinstance(_it, (list, tuple)) and len(_it) == 2:
                    _texts.append(str(_it[0]))
                    if int(_it[1]) - 1 > 0:
                        _keep.append([str(_it[0]), int(_it[1]) - 1])
                else:
                    _texts.append(str(_it))
            self._t2_view_fb = _keep or None
            _vtxt = "\n".join(_texts)
            print("[T2_FB_VIEW] %d queued feedback item(s) injected in view" % len(_texts),
                  file=_sys.stderr, flush=True)
            try:
                work = work + [UserMessage(role="user", content=_vtxt)]
            except TypeError:
                work = work + [UserMessage(content=_vtxt)]
        # ★COV FIND-subset 백스톱 (T2_COV=1): ≥1 write 후 M∖acted ≠ ∅ → in-flight 리마인더 1회
        if (os.environ.get("T2_COV") == "1" and ep_led is not None
                and not getattr(self, "_t2_cov_reminded", False)):
            try:
                execd = {str(e.get("entity") or "").strip()
                         for e in getattr(ep_led, "executed", [])}
                execd.discard("")
                if execd:
                    M = getattr(self, "_t2_cov_M", None)
                    if M is None:
                        M = _cov_formalize_M(self, la, UserMessage, state.messages,
                                             a2.get("eplan") if a2 else None, a2)
                        self._t2_cov_M = M  # []도 캐시(서브콜 1회)
                    remaining = [m for m in (M or []) if m not in execd]
                    if remaining and len(M) >= 2:
                        self._t2_cov_reminded = True
                        print("[T2_COV] reminder M=%s acted=%s remaining=%s"
                              % (",".join(M), ",".join(sorted(execd)), ",".join(remaining)),
                              file=_sys.stderr, flush=True)
                        _cr = COV_REMINDER.format(ids=", ".join(remaining))
                        try:
                            work = work + [UserMessage(role="user", content=_cr)]
                        except TypeError:
                            work = work + [UserMessage(content=_cr)]
            except Exception as _cve:
                print("[T2_COV] error (no-op): %r" % (_cve,), file=_sys.stderr, flush=True)
        # ★T2_COV_MIDDRIVE (C118·EPLAN_MIDDRIVE_DESIGN §2.1): "종료시 1회"(T2_COV) → "갭 열린 동안
        #   매 드리프트 견인". 트리거 3AND: ①M(coverage 대상·formalize)≥2 ②remaining=M∖executed≠∅
        #   ③직전 assistant 턴이 write 아님(드리프트). → 리마인더(남은 것 지금 처리) 매턴. 진행(executed↑)
        #   시 stall 리셋·K턴 연속 미진행 중단(무한넛지 방지·§2.1). M=LLM formalize(1회캐시·[[10]])·나머지
        #   결정론(drift/diff/K). write 강제 0(soft·§1.5). 052/043(CLI/close 사슬)엔 미적용(dispute-coverage).
        if (os.environ.get("T2_COV_MIDDRIVE") == "1" and ep_led is not None):
            try:
                execd = {str(e.get("entity") or "").strip()
                         for e in getattr(ep_led, "executed", [])}
                execd.discard("")
                M = getattr(self, "_t2_cov_M", None)
                if M is None:
                    M = _cov_formalize_M(self, la, UserMessage, state.messages,
                                         a2.get("eplan") if a2 else None, a2)
                    self._t2_cov_M = M
                remaining = [m for m in (M or []) if m not in execd]
                if len(execd) > getattr(self, "_t2_cov_execn", 0):   # 진행 감지 → stall 리셋
                    self._t2_cov_stall = 0
                self._t2_cov_execn = len(execd)
                _drift = not _last_assistant_did_write(state.messages, ep_writes)
                _K = int(os.environ.get("T2_COV_MIDDRIVE_K", "4"))
                if (remaining and len(M) >= 2 and _drift
                        and getattr(self, "_t2_cov_stall", 0) < _K):
                    self._t2_cov_stall = getattr(self, "_t2_cov_stall", 0) + 1
                    print("[T2_COV] mid-drive M=%s acted=%s remaining=%s stall=%d/%d"
                          % (",".join(M), ",".join(sorted(execd)), ",".join(remaining),
                             self._t2_cov_stall, _K), file=_sys.stderr, flush=True)
                    _cd = COV_REMINDER_DRIVE.format(n=len(M), done=(", ".join(sorted(execd)) or "none"),
                                                    ids=", ".join(remaining))
                    try:
                        work = work + [UserMessage(role="user", content=_cd)]
                    except TypeError:
                        work = work + [UserMessage(content=_cd)]
            except Exception as _cde:
                print("[T2_COV] mid-drive error (no-op): %r" % (_cde,), file=_sys.stderr, flush=True)
        # ★TOOLERR (T2_TOOLERR=1): 방금 난 tool-error를 A2로 분류·directive 주입(in-flight)
        terr = None
        if os.environ.get("T2_TOOLERR") == "1" and a2 is not None:
            try:
                terr = classify_tool_error(state.messages, a2)
                if terr is not None:
                    _sp, _tool, _fargs = terr
                    _cls = _sp.get("class")
                    _tmpl = TOOLERR_RECOVER if _cls == "recover" else TOOLERR_ABSTAIN
                    _td = _tmpl.format(tool=_tool, hint=_sp.get("hint", ""))
                    print("[T2_TOOLERR] %s tool=%s inject" % (_cls, _tool),
                          file=_sys.stderr, flush=True)
                    try:
                        work = work + [UserMessage(role="user", content=_td)]
                    except TypeError:
                        work = work + [UserMessage(content=_td)]
            except Exception as _te:
                terr = None
                print("[T2_TOOLERR] error (no-op): %r" % (_te,), file=_sys.stderr, flush=True)
        # ★P3(C208③·DAY5_PRESCRIPTIONS §P3): 터미널-턴 유예의 그 1턴만 tool_choice=required —
        #   재-notice 산문 봉쇄(기존 FORCE_ACTION 기제 재사용·도구/인자 미지정=write 강제 아님).
        if getattr(self, "_t2_term_force", False):
            self._t2_term_force = False
            am = _gen(self, work, bw(), "agent_response", tool_choice="required")
        else:
            am = _gen(self, work, bw(), "agent_response")
        gate_rounds = prov_rounds = eplan_rounds = cons_rounds = ra_rounds = te_rounds = wev_rounds = 0
        tl_rounds = 0
        subs = 0
        rescue_skipped = set()
        rescue_excl = set()   # ★PERARG(C65): (id(tc),k,s) — rescue-스킵된 fab 제외하고 재스캔
        while True:
            force_required = False   # ★T2_FORCE_ACTION: say-don't-do → 다음 재생성서 tool_choice=required 강제
            fab = _first_fab_call(am, ctx, hints, exclude=rescue_excl)
            # ★T5-C P-A (N1: _denied_calls 前 — 게이트 check는 상태-변이라 버려질 반복서 소진 금지)
            if fab is not None and ground and subs < 8:
                gtc, gk, gs = fab
                gcands = _grounded_candidates(gk, gs, state.messages, lenient=True)
                if len(gcands) == 1 and gcands[0] != gs and _subst_arg_value(gtc, gk, gs, gcands[0]):
                    self._t2_ground_sub = getattr(self, "_t2_ground_sub", 0) + 1
                    subs += 1
                    print("[T2_GROUND] substituted arg=%s val=%s -> %s" % (gk, gs, gcands[0]),
                          file=_sys.stderr, flush=True)
                    continue  # 치환값은 문맥-실재 → 다음 반복서 fab 해소·게이트가 최종 인자를 검사
            # ★T5-C P-C + PERARG(C65): env-검증형 id 날조 = *개별* 스킵 후 다음 fab 재스캔
            #   (구: fab=None → 같은 호출의 둘째 자유텍스트 fab[t17 address1]이 영영 미검사)
            while fab is not None and prov_mode == "rescue":
                rtc, rk, rs = fab
                if (_key_tokens(rk) & env_args) and _sig(rs) in ("hashid", "numid") \
                        and not _in_error_loop(state.messages, getattr(rtc, "name", None)):
                    rkey = (getattr(rtc, "name", None), rk, rs)
                    if rkey not in rescue_skipped:
                        rescue_skipped.add(rkey)
                        self._t2_prov_skipped_envdup = getattr(self, "_t2_prov_skipped_envdup", 0) + 1
                        print("[T2_PROV] rescue pass-through tool=%s arg=%s val=%s" % rkey,
                              file=_sys.stderr, flush=True)
                    rescue_excl.add((id(rtc), rk, rs))
                    fab = _first_fab_call(am, ctx, hints, exclude=rescue_excl)
                else:
                    break
            # ★L3 origin-prov (T2_PROV_ORIGIN=1·v3.2): fab 무해(값∈ctx) 통과분 중 확인-세탁 검사
            ofab = None
            if fab is None and os.environ.get("T2_PROV_ORIGIN") == "1":
                ofab = _first_origin_fab(am, state.messages, hints, exclude=rescue_excl)
            denied = _denied_calls(am, gate, last_user, transfer_sent) if gate is not None else []
            denied_by_objid = {id(tc): (gid, why) for tc, gid, why in denied}
            do_gate = bool(denied) and gate_rounds < 1
            _pcall = fab if fab is not None else ofab
            fab_covered = _pcall is not None and do_gate and id(_pcall[0]) in denied_by_objid
            do_prov = (_pcall is not None) and prov_rounds < max_prov_retries and not fab_covered
            # ★E-PLAN discovery deny (L1/L2·read-강제만·무과금·상한 = 턴당 2 + ★sim당 T2_EPLAN_DENY_CAP)
            # ★sim-cap 근거(A′ t5c_aprime1 t103/t27 포렌식·2026-07-11): eplan_rounds는 턴-로컬이라
            #   모델이 deny 피드백에 불응(텍스트-사과 커밋)하면 매 턴 ledger가 동일 재구성 → 동일 L2
            #   deny 무한 반복(무과금=too_many_errors 탈출로도 없음) → t103 max_steps(200)·t27 유저 포기.
            #   read-강제는 sim당 유한 예산으로 — 소진 후엔 write 통과(env/gold 판정에 맡김·개입 최소).
            ep_fb = None
            _ep_cap = int(os.environ.get("T2_EPLAN_DENY_CAP", "4"))
            if (ep_led is not None and eplan_rounds < 2 and not do_gate and not do_prov
                    and getattr(self, "_t2_eplan_deny", 0) < _ep_cap):
                for c in (am.tool_calls or []):
                    nm = getattr(c, "name", None)
                    _cargs = _args_dict(c)
                    # ★디스패처 unwrap(C101 (c)·retail 무영향=dispatch_tool 없으면 skip):
                    #   banking dispute write = call_discoverable_agent_tool(nested {name,args}).
                    _dt = ep_spec.get("dispatch_tool")
                    if _dt and nm == _dt:
                        nm = re.sub(r"_\d+$", "", str(_cargs.get(
                            ep_spec.get("dispatch_name_key", "agent_tool_name"), "")))
                        _inner = _cargs.get(ep_spec.get("dispatch_args_key", "arguments"))
                        if isinstance(_inner, str):
                            try:
                                _inner = json.loads(_inner)
                            except Exception:
                                _inner = {}
                        _cargs = _inner if isinstance(_inner, dict) else {}
                    if nm in ep_writes and id(c) not in denied_by_objid:
                        try:
                            # ★qty-conflation 가드(t27): 시도 call의 품목 id를 술어에 전달
                            #   (키=A2 "items_key"·ABox) — N이 품목-급으로 충족되면 deny 안 함.
                            _ep_items = _cargs.get(
                                ep_spec.get("items_key", "item_ids")) or ()
                            _ep_ent = _cargs.get(ep_spec.get("entity_key"))
                            fb = _epmod.discovery_precondition(
                                ep_led, ep_spec, nm, attempt_items=_ep_items,
                                attempt_entity=_ep_ent)
                        except Exception:
                            fb = None
                        if fb:
                            ep_fb = (c, fb)
                            break
            # ★BRANCH-REGROUND pre-close 트리거 (C144/C146·T2_BRANCH_REGROUND=1): finalize
            #   write(close류·A2 finalize_writes) 시도 시 정책조건 선행단계(apply_flag 등) 미완이면
            #   deny + 실제 정책문서 재부각 → apply_flag이 close 선행(현 user_stop 경로는 close 後라 늦음).
            #   ep_fb 채널·cap 재사용(무과금·regen 공유·discovery 미발화 시만). finalize 제외=for_finalize.
            # ★C173-corr(2026-07-25): pre-close 예산을 공유 _ep_cap에서 **분리** — 044 실측:
            #   초반 E-PLAN/discovery deny들이 공유 4회를 소진해 msg66의 2번째 close 시도가
            #   무방비 통과(선행 read/write 미완인 채 CLOSED·상태오염 3). claimprov transfer-창
            #   별도예산(§2ao) 선례와 동일 패턴: finalize 방어는 독립 예비(기본 2/sim)로 보장.
            #   (구 C173의 T2_GATE_REGEN_K=2는 unified 경로서 미사용=no-op이라 철회.)
            if (ep_fb is None and ep_led is not None and eplan_rounds < 2
                    and os.environ.get("T2_BRANCH_REGROUND") == "1"
                    and not do_gate and not do_prov
                    and getattr(self, "_t2_preclose_deny", 0)
                        < int(os.environ.get("T2_PRECLOSE_CAP", "2"))):
                _finals = {re.sub(r"_\d+$", "", f)
                           for f in (ep_spec.get("finalize_writes") or ())}
                if _finals:
                    for c in (am.tool_calls or []):
                        nm = getattr(c, "name", None)
                        _cargs = _args_dict(c)
                        _dt = ep_spec.get("dispatch_tool")
                        if _dt and nm == _dt:
                            nm = re.sub(r"_\d+$", "", str(_cargs.get(
                                ep_spec.get("dispatch_name_key", "agent_tool_name"), "")))
                        else:
                            # ★C148 버그수정: close는 직접호출(name=close_..._7834·디스패처 아님)이
                            #   가능 → suffix fam-strip 안 하면 _finals 미매칭·게이트 우회(043 실측).
                            nm = re.sub(r"_\d+$", "", str(nm or ""))
                        if nm in _finals and id(c) not in denied_by_objid:
                            try:
                                _bchain = _epmod.chain_gap(state.messages, ep_spec)
                            except Exception:
                                _bchain = None
                            if _bchain is not None:
                                _prereq_w = [w for w in _bchain["missing_writes"]
                                             if re.sub(r"_\d+$", "", w) not in _finals]
                                if _bchain["missing_reads"] or _prereq_w:
                                    try:
                                        _brem = _epmod.branch_reground_reminder(
                                            _bchain, state.messages, ep_spec, for_finalize=True)
                                    except Exception:
                                        _brem = None
                                    if _brem:
                                        ep_fb = (c, _brem)
                                        # C173-corr: 전용 예산 소모(독립 카운터·기본 2/sim)
                                        # C174: kind 플래그 — 공유 증가 지점서 discovery
                                        #   카운터를 건드리지 않게 표시(예산 전면 독립).
                                        self._t2_ep_fb_preclose = True
                                        self._t2_preclose_deny = getattr(
                                            self, "_t2_preclose_deny", 0) + 1
                                        print("[T2_BRANCH_REGROUND] pre-close deny: finalize=%s "
                                              "prereq_reads=%d prereq_writes=%d (pc_deny=%d)"
                                              % (nm, len(_bchain["missing_reads"]), len(_prereq_w),
                                                 self._t2_preclose_deny),
                                              file=_sys.stderr, flush=True)
                                        break
            # ★DISCOVERY-DISPATCH deny (C151·T2_DISCOVERY_DISPATCH=1): discoverable 도구를 *직접
            #   호출*(suffixed name·dispatcher 미경유)하면 평가 리플레이가 등록 안 함(registration은
            #   call_discoverable_agent_tool 내부에서만). compliance 게이트로 deny+프로토콜 지시 →
            #   에이전트가 unlock→call_discoverable로 재발행(reroute 아님=스캐폴드 조작 회피·[[05]]/[[10]]).
            #   도메인일반: dispatch_tool 선언 도메인만·이름 suffix 휴리스틱(discoverable=랜덤 suffix).
            dd_fb = None
            if (os.environ.get("T2_DISCOVERY_DISPATCH") == "1" and ep_spec is not None
                    and ep_spec.get("dispatch_tool") and ep_fb is None
                    and not do_gate and not do_prov
                    and getattr(self, "_t2_dd_deny", 0) < int(os.environ.get("T2_DD_CAP", "8"))):
                _disp = ep_spec.get("dispatch_tool")
                _unlock = ep_spec.get("unlock_tool", "unlock_discoverable_agent_tool")
                _nk = ep_spec.get("dispatch_name_key", "agent_tool_name")
                _safe = {_disp, _unlock, ep_spec.get("list_tool", "list_discoverable_agent_tools")}
                for c in (am.tool_calls or []):
                    nm = getattr(c, "name", "") or ""
                    if nm not in _safe and re.search(r"_\d{3,4}$", nm) \
                            and id(c) not in denied_by_objid:
                        # ★C153: 처방적 구체성(C116) — 에이전트가 방금 쓴 실제 인자를 그대로 echo한
                        #   복사-가능 2단계 예시. 막연한 '{...}'은 포기 유발(C152 abandon)·구체 예시=재발행 유도.
                        _uarg = json.dumps(_args_dict(c)).replace('"', '\\"')
                        dd_fb = (c, "You called '%s' directly, but it is a DISCOVERABLE tool — a direct "
                                    "call is NOT registered and does NOT complete the step. Do NOT abandon "
                                    "this step. Redo it in TWO steps with the SAME arguments you just used:\n"
                                    "  1) %s(%s=\"%s\")\n"
                                    "  2) %s(%s=\"%s\", arguments=\"%s\")"
                                 % (nm, _unlock, _nk, nm, _disp, _nk, nm, _uarg))
                        break
            # ★v3.2 CONSISTENCY (T2_CONSISTENCY=1): L10 멤버십(t35형)+G-noop(t71형)·cap 2/sim
            cons_fb = None
            if (os.environ.get("T2_CONSISTENCY") == "1" and a2 is not None
                    and not do_gate and not do_prov and ep_fb is None
                    and cons_rounds < 1 and getattr(self, "_t2_cons_deny", 0) < 2):
                try:
                    _cspec = a2.get("eplan") or {}
                    _cwrites = _confirm_write_tools(a2)
                    for c in (am.tool_calls or []):
                        if getattr(c, "name", None) not in _cwrites:
                            continue
                        dargs = _args_dict(c)
                        mv = membership_violation(dargs, _cspec, state.messages)
                        if mv:
                            _bad, _oid, _hint = mv
                            cons_fb = (c, CONS_MEMBER_FEEDBACK.format(
                                bad=", ".join(_bad), ent=_cspec.get("entity_key"), oid=_oid,
                                hint=(" They appear in %s='%s'." % (_cspec.get("entity_key"), _hint))
                                     if _hint else ""))
                            print("[T2_CONS] membership deny tool=%s bad=%s oid=%s hint=%s"
                                  % (c.name, ",".join(_bad), _oid, _hint),
                                  file=_sys.stderr, flush=True)
                            break
                        # ★v6 선별(T2_CONS_NOOP=0으로 분리 가능): G-noop은 v4 nt2서 효과 0(발화1·결과불변)
                        if os.environ.get("T2_CONS_NOOP", "1") != "0" \
                                and noop_write(dargs, _cspec, state.messages):
                            cons_fb = (c, CONS_NOOP_FEEDBACK)
                            print("[T2_CONS] noop deny tool=%s oid=%s"
                                  % (c.name, dargs.get(_cspec.get("entity_key"))),
                                  file=_sys.stderr, flush=True)
                            break
                except Exception as _ce:
                    cons_fb = None
                    print("[T2_CONS] error (no-op): %r" % (_ce,), file=_sys.stderr, flush=True)
            # ★v5 L2R read-all (T2_READALL=1·NT2 §4-1): 첫 write 전 미열람 후보 read 강제
            ra_fb = None
            if (os.environ.get("T2_READALL") == "1" and ep_led is not None
                    and not do_gate and not do_prov and ep_fb is None and cons_fb is None
                    and ra_rounds < 1
                    and getattr(self, "_t2_readall_deny", 0) < int(os.environ.get("T2_READALL_CAP", "2"))):
                try:
                    unread = readall_unread(getattr(ep_led, "listed", ()),
                                            getattr(ep_led, "examined", ()))
                    if unread:
                        _raw = _confirm_write_tools(a2)
                        for c in (am.tool_calls or []):
                            if getattr(c, "name", None) in _raw and id(c) not in denied_by_objid:
                                ra_fb = (c, READALL_FEEDBACK.format(
                                    reader=(a2.get("eplan") or {}).get("detail_reader"),
                                    ids=", ".join(unread)))
                                print("[T2_READALL] deny tool=%s unread=%s"
                                      % (c.name, ",".join(unread)), file=_sys.stderr, flush=True)
                                break
                except Exception as _re2:
                    ra_fb = None
                    print("[T2_READALL] error (no-op): %r" % (_re2,), file=_sys.stderr, flush=True)
            # ★TOOLERR teeth: (a) 같은-실패-인자 재발행 차단(class 무관·항상 안전) (b) recover면 조기 transfer 차단
            te_fb = None
            if (terr is not None and not do_gate and not do_prov
                    and ep_fb is None and cons_fb is None and ra_fb is None
                    and te_rounds < 1 and getattr(self, "_t2_toolerr_deny", 0) < 3):
                _sp, _tool, _fargs = terr
                _cls = _sp.get("class")
                _xfer = _transfer_tools(a2)
                for c in (am.tool_calls or []):
                    if id(c) in denied_by_objid:
                        continue
                    nm = getattr(c, "name", None)
                    dargs = _args_dict(c)
                    # (a) 동일 도구·동일 실패-인자 재발행 = 결정론적으로 또 실패 → deny(항상)
                    if nm == _tool and _fargs and dargs == _fargs:
                        te_fb = (c, TOOLERR_RECOVER.format(tool=_tool, hint=_sp.get("hint", ""))
                                 + " (You re-sent the SAME failing argument — you must change it.)")
                        print("[T2_TOOLERR] deny same-failed-args tool=%s" % nm,
                              file=_sys.stderr, flush=True)
                        break
                    # (b) recover-class인데 조기 transfer = 포기 차단(A2-판단부·Δspurious 계측대상)
                    if _cls == "recover" and nm in _xfer:
                        te_fb = (c, TOOLERR_RECOVER.format(tool=_tool, hint=_sp.get("hint", ""))
                                 + " (Do not transfer for a recoverable error — retry with a corrected argument.)")
                        print("[T2_TOOLERR] deny early-transfer(recover) tool=%s" % nm,
                              file=_sys.stderr, flush=True)
                        break
            # ★reference-filter (keystone·참조축·C77) — do_gate/do_prov와 *독립*.
            #   결정론 in-place silent-repair(수집 record ⋈)이므로 게이트/prov 피드백과 경합하지 않는다.
            #   게이트가 *교정된* nested id를 확인하도록 do_gate 판정을 소비하는 fb-빌드(아래) 前에 실행.
            #   (배선버그 교정 2026-07-14: 前엔 T2_RESOLVE 가드의 not do_gate 안에 있어 dispute류
            #    confirm-게이트가 do_gate=True면 영영 미발화 → 사용자가 *틀린* 거래를 확인.)
            if (os.environ.get("T2_RESOLVE") == "1" and a2 is not None
                    and (a2 or {}).get("reference_filter")
                    and getattr(self, "_t2_reffilter", 0) < 3):
                try:
                    import t2_resolve as _rz_rf
                    _rf = _rz_rf.resolve_reference_filter(am, state.messages, a2, self, la, UserMessage)
                    if _rf.get("status") == "deny":
                        _rtc = _rf.get("call"); _nested = _rf.get("nested") or {}
                        _nested[_rf["param"]] = _rf["correct"]      # 제자리 치환
                        try:                                        # nested 재직렬화(문자열/딕트)
                            if isinstance(getattr(_rtc, "arguments", {}).get("arguments"), str):
                                _rtc.arguments["arguments"] = json.dumps(_nested)
                            elif isinstance(_rtc.arguments.get("arguments"), dict):
                                _rtc.arguments["arguments"] = _nested
                            else:
                                _rtc.arguments[_rf["param"]] = _rf["correct"]
                        except Exception:
                            pass
                        self._t2_reffilter = getattr(self, "_t2_reffilter", 0) + 1
                        print("[T2_RESOLVE] reference-filter silent-repair %s->%s"
                              % (_rf["param"], _rf["correct"]), file=_sys.stderr, flush=True)
                except Exception as _rfe:
                    print("[T2_RESOLVE] reffilter error (no-op): %r" % (_rfe,),
                          file=_sys.stderr, flush=True)
            # ★compute 키스톤(§8·C81) — do_gate/prov와 독립·결정론 in-place silent-repair(정책-계산 param).
            #   §8-3: liability만(순+348)·provisional 드롭(net−4). 에이전트 제공값만·미확정=미개입.
            if (os.environ.get("T2_COMPUTE") == "1" and a2 is not None
                    and (a2 or {}).get("compute_ops")
                    and getattr(self, "_t2_compute", 0) < 8):
                try:
                    import t2_resolve as _rz_cp
                    for _cp in _rz_cp.resolve_compute_params(am, state.messages, a2):
                        _nz = _cp.get("nested") or {}
                        _nz[_cp["param"]] = _cp["computed"]
                        _rtc = _cp.get("call")
                        try:
                            if isinstance(getattr(_rtc, "arguments", {}).get("arguments"), str):
                                _rtc.arguments["arguments"] = json.dumps(_nz)
                            elif isinstance(_rtc.arguments.get("arguments"), dict):
                                _rtc.arguments["arguments"] = _nz
                        except Exception:
                            pass
                        self._t2_compute = getattr(self, "_t2_compute", 0) + 1
                        print("[T2_RESOLVE] compute silent-repair %s %s->%s"
                              % (_cp["param"], _cp["old"], _cp["computed"]), file=_sys.stderr, flush=True)
                except Exception as _cpe:
                    print("[T2_RESOLVE] compute error (no-op): %r" % (_cpe,), file=_sys.stderr, flush=True)
            # ★T2_REF_ISO (2026-07-24 C124/C125): 참조-슬립 최소-문맥 격리 재선택 → 제자리 치환.
            #   WEV *앞* 배치(치환된 id 기준으로 증거 검사). cap=T2_REF_ISO_CAP(기본 8)·예외 no-op.
            if (os.environ.get("T2_REF_ISO") == "1" and a2 is not None
                    and (a2 or {}).get("ref_iso")
                    and getattr(self, "_t2_refiso", 0) < int(os.environ.get("T2_REF_ISO_CAP", "8"))):
                try:
                    _ref_iso_repair(self, la, UserMessage, state.messages, am, (a2 or {}).get("ref_iso"))
                except Exception as _rie:
                    print("[T2_REF_ISO] error (no-op): %r" % (_rie,), file=_sys.stderr, flush=True)
            # ★T2_WRITE_EVIDENCE (unified 배선·2026-07-19 028 포렌식): 증거(도구출력 token+id 공존)
            #   없는 선언-write deny. silent-repair(reffilter/compute) *뒤* 배치 = 교정된 최종 인자를 검사.
            #   무과금·turn당 1회·sim당 T2_WEV_CAP(기본 8) — E-PLAN cap 선례(불응 무한루프 방지·소진 후 통과).
            #   ★T2_WEV_ROUNDS (2026-07-24 C125·rall20 031.0 실측): turn당 1회 규칙의 구멍 — deny 후
            #   regen된 호출은 같은 턴서 무검사 커밋(1234 날조가 deny #2 직후 regen으로 통과·오프라인
            #   재실행은 deny 확인=술어 정상·배관 우회). 재검사 횟수 env화(기본 1=현행 불변·2=regen 1회 재검).
            wev_fb = None
            if ((wev_specs or wag_specs or rv_specs) and not do_gate and not do_prov and ep_fb is None
                    and cons_fb is None and ra_fb is None and te_fb is None
                    and wev_rounds < int(os.environ.get("T2_WEV_ROUNDS", "1"))
                    and getattr(self, "_t2_wev_deny", 0) < _wev_cap):
                try:
                    for c in (am.tool_calls or []):
                        if id(c) in denied_by_objid:
                            continue
                        wd = _wev_deny_msgs(state.messages, c, wev_specs)
                        _wtag = "T2_WRITE_EVIDENCE"
                        if not wd and wag_specs:
                            # ★값-grounding(§2bs·031): WEV(선행-read)와 별개 구멍=값-전사.
                            wd = _write_arg_ground_deny(state.messages, c, wag_specs)
                            _wtag = "T2_WRITE_ARG_GROUND"
                        if not wd and rv_specs:
                            # ★결정론 참조-검증기(C128/C129): 레코드 판별속성(merchant)이 손님
                            #   발화에 없으면 deny — 전사-슬립 8/8 검출·false-block 0·LLM 0.
                            wd = _ref_verify_deny(state.messages, c, rv_specs)
                            _wtag = "T2_REF_VERIFY"
                        if wd:
                            wev_fb = (c, wd)
                            # ★내부 도구명 로깅(§2ba 오귀속 교훈: per-도구 로그에 이름 필수)
                            _inner = _args_dict(c).get("agent_tool_name") or ""
                            print("[%s] deny tool=%s inner=%s"
                                  % (_wtag, getattr(c, "name", None), _inner),
                                  file=_sys.stderr, flush=True)
                            break
                except Exception as _wve:
                    wev_fb = None
                    print("[T2_WRITE_EVIDENCE] error (no-op): %r" % (_wve,),
                          file=_sys.stderr, flush=True)
            # ★T2_HAVE_VALUE (C115): 값-실재인데 재요청 반복 → None-anchor 리마인더(W를 지금 호출).
            #   WAG(fab 값 차단)의 *반대* 케이스이므로 그 뒤에 배치(상호배타)·무과금·turn당1·sim당 cap.
            hv_fb = None
            if (hv_specs and not do_gate and not do_prov and ep_fb is None
                    and cons_fb is None and ra_fb is None and te_fb is None and wev_fb is None
                    and getattr(self, "_t2_havevalue_deny", 0)
                    < int(os.environ.get("T2_HAVE_VALUE_CAP", "3"))):
                try:
                    hv_fb = _have_value_reask_fb(am, state.messages, hv_specs)
                    if hv_fb:
                        print("[T2_HAVE_VALUE] reask-with-value → nudge (regen)",
                              file=_sys.stderr, flush=True)
                except Exception as _hve:
                    hv_fb = None
                    print("[T2_HAVE_VALUE] error (no-op): %r" % (_hve,),
                          file=_sys.stderr, flush=True)
            # ★T2_VALUE_ACQUIRE (C119): 값 미실재 + give 미실행 → give 표면화 넛지(have-value 앞단계).
            if (va_specs and hv_fb is None and not do_gate and not do_prov and ep_fb is None
                    and cons_fb is None and ra_fb is None and te_fb is None and wev_fb is None
                    and getattr(self, "_t2_valacq_deny", 0)
                    < int(os.environ.get("T2_VALUE_ACQUIRE_CAP", "3"))):
                try:
                    _va = _value_acquire_fb(am, state.messages, va_specs)
                    if _va:
                        hv_fb = _va   # hv_fb 채널 재사용(None-anchor 리마인더·상호배타)
                        self._t2_valacq_fired = True
                        print("[T2_VALUE_ACQUIRE] give-surfacing → nudge (regen)",
                              file=_sys.stderr, flush=True)
                except Exception as _vae:
                    print("[T2_VALUE_ACQUIRE] error (no-op): %r" % (_vae,),
                          file=_sys.stderr, flush=True)
            # ★T2_PARAM_CAP (§2br): 정책-캡 deny (A2 param_cap_check·054 실측)
            pc_fb = None
            _pcs = (a2 or {}).get("param_cap_check") or []
            if (os.environ.get("T2_PARAM_CAP") == "1" and _pcs
                    and not do_gate and not do_prov and ep_fb is None and cons_fb is None
                    and ra_fb is None and te_fb is None and wev_fb is None
                    and getattr(self, "_t2_paramcap_deny", 0)
                    < int(os.environ.get("T2_PARAM_CAP_CAP", "4"))):
                try:
                    for c in (am.tool_calls or []):
                        _pd = _param_cap_deny(state.messages, c, _pcs)
                        if _pd:
                            pc_fb = (c, _pd)
                            self._t2_paramcap_deny = getattr(self, "_t2_paramcap_deny", 0) + 1
                            print("[T2_PARAM_CAP] deny param over policy cap", file=_sys.stderr,
                                  flush=True)
                            break
                except Exception as _pce:
                    pc_fb = None
                    print("[T2_PARAM_CAP] error (no-op): %r" % (_pce,), file=_sys.stderr, flush=True)
            # ★T2_RESOLVE (통일 인터프리터·UNIFIED_OPERAND_A2 §7-3): per-operand 해소 디스패처.
            #   deny-kind(operator/membership/provenance) 통합 = L10+L3+operator 한 경로.
            #   개별 플래그(T2_CONSISTENCY/T2_PROV_ORIGIN) 대체용(driver가 상호배타 설정).
            rw_fb = None
            if (os.environ.get("T2_RESOLVE") == "1" and a2 is not None
                    and not do_gate and not do_prov and ep_fb is None and cons_fb is None
                    and ra_fb is None and te_fb is None and wev_fb is None
                    and getattr(self, "_t2_resolve_deny", 0) < 3):
                try:
                    import t2_resolve as _rz
                    for c in (am.tool_calls or []):
                        if id(c) in denied_by_objid:
                            continue
                        rr = _rz.resolve_write(getattr(c, "name", None), _args_dict(c),
                                               state.messages, a2, self, la, UserMessage)
                        if rr.get("status") == "deny":
                            rw_fb = (c, rr["feedback"])
                            print("[T2_RESOLVE] deny tool=%s arg=%s reason=%s"
                                  % (getattr(c, "name", None), rr.get("arg"), rr.get("reason")),
                                  file=_sys.stderr, flush=True)
                            break
                    # ★Lever 4 pre-recommendation 검증(오추천 방지·user-실행 operand): 에이전트가
                    #   offer_tool로 action 제안 시 operand를 요구→formalize 검증·틀리면 교정. cap 2/sim.
                    if (rw_fb is None and getattr(self, "_t2_recommend_deny", 0) < 2):
                        _rvr = _rz.resolve_recommendation(am, state.messages, a2, self, la, UserMessage,
                                                          transfer_tools=_transfer_tools(a2))
                        if _rvr.get("status") == "deny":
                            rw_fb = (_rvr.get("call") or (am.tool_calls or [None])[0], _rvr["feedback"])
                            self._t2_recommend_deny = getattr(self, "_t2_recommend_deny", 0) + 1
                            print("[T2_RESOLVE] %s deny" % _rvr.get("reason", "recommendation-verify"),
                                  file=_sys.stderr, flush=True)
                    # ★action-required (turn-level): am이 회피(action_tool 미호출)면 operator 해소
                    #   GET→FIND(intent→도구)→execute|ASK. 조언/포기로 종결 금지. cap 1/sim.
                    if (rw_fb is None and getattr(self, "_t2_action_deny", 0)
                            < int(os.environ.get("T2_ACTION_DENY_CAP", "1"))):
                        # ★Lever 0(BANK_ACTIONREQ_PROBE_FORENSIC §3): action-required는 agent-실행
                        #   도구만 대상 — user-실행(apply/submit 등)은 에이전트가 못 부르므로 스퓨리어스.
                        #   A2 action_tool_executor 맵(도메인 데이터)로 필터·미기재=assistant 폴백(retail 하위호환).
                        _exec_map = (a2 or {}).get("action_tool_executor") or {}
                        _acts = {t for t in ((a2 or {}).get("action_tools") or [])
                                 if _exec_map.get(t, "assistant") == "assistant"}
                        # ★user-실행 분기 (2026-07-22 §2bo·rall9 023 실측): apply류(user-실행)는
                        #   기존에 스퓨리어스 방지로 필터만 되고 **아무 nudge도 없어**, 모델이 "에이전트-측
                        #   절차가 KB에 있을 것"이라는 거짓 전제로 8턴 검색-루프→transfer(C108 변형).
                        #   executor 맵(A2·기왕 선언)을 모델에 *전달*: intent가 user-실행 도구로 formalize
                        #   되고 그 도구가 아직 미실행이면 "고객이 실행하는 도구다·검색 중단·안내하라"
                        #   피드백. tool_choice 강제 없음(정답 행동=안내 텍스트)·cap=action_deny 공유.
                        _uacts = {t for t in ((a2 or {}).get("action_tools") or [])
                                  if _exec_map.get(t) == "user"}
                        _called = {getattr(c, "name", None) for c in (am.tool_calls or [])}
                        _tgt_pre = None      # ★공유 formalize(합집합 1회) — 아래 원 블록이 재사용(이중 서브콜 방지)
                        if ((_uacts or _acts) and _rz._agent_ending(am, _transfer_tools(a2))):
                            _effall = {_eff_tool_name(tc) for m2 in state.messages
                                       for tc in (getattr(m2, "tool_calls", None) or [])}
                            _upending = sorted(_uacts - _effall)
                            if _upending or (_acts and not (_called & _acts)):
                                _tgt_pre = _rz.formalize_intent_tool(self, la, UserMessage,
                                                                    state.messages,
                                                                    set(_upending) | _acts)
                                _utgt = _tgt_pre
                                if _utgt in _upending:
                                    _ufb = str((a2 or {}).get("user_action_feedback")
                                               or ("Error: [ACTION] '{tool}' is run by the CUSTOMER, "
                                                   "not by you, and needs no agent-side KB procedure - "
                                                   "STOP searching. In your next message tell the "
                                                   "customer to run {tool} now with their details, "
                                                   "then confirm the result. Do not transfer for this.")
                                               ).replace("{tool}", _utgt)
                                    rw_fb = ((am.tool_calls or [None])[0], _ufb)
                                    self._t2_action_deny = getattr(self, "_t2_action_deny", 0) + 1
                                    print("[T2_RESOLVE] user-action instruct target=%s" % _utgt,
                                          file=_sys.stderr, flush=True)
                        if rw_fb is None and _acts and not (_called & _acts) and _rz._agent_ending(am, _transfer_tools(a2)):
                            _tgt = _tgt_pre if _tgt_pre in _acts else None
                            # ★고정밀(Δspurious): formalize가 구체 agent-실행 target을 낼 때만 발화.
                            #   target=None(=action-ask)은 미발화 — discovery/user-실행 의도서 스퓨리어스
                            #   (banking 잔여=⋈/reach이지 deflect-vs-ask 아님·BANK_ACTIONREQ_PROBE_FORENSIC).
                            _ar = (_rz.resolve_action_operator(
                                {"action_tools": list(_acts)}, am, state.messages, a2,
                                target_tool=_tgt, transfer_tools=_transfer_tools(a2))
                                if _tgt else {"status": "ok"})
                            if _ar.get("status") == "deny":
                                rw_fb = ((am.tool_calls or [None])[0], _ar["feedback"])
                                self._t2_action_deny = getattr(self, "_t2_action_deny", 0) + 1
                                # ★진행-감응 환급용 target 스냅샷 (2026-07-22 §2bt·rall10 097 실측:
                                #   초반 flail이 cap3 소진→종반 say-loop 8연속 무방비 = FOLLOWUP
                                #   cap-소진과 동형). 발화 시점에 미실행인 target만 기록 —
                                #   이후 시도-수준 착수가 보이면 cap 1회 환급(1b와 동일 원리).
                                _effa2 = {_eff_tool_name(tc) for m2 in state.messages
                                          for tc in (getattr(m2, "tool_calls", None) or [])}
                                self._t2_action_target = ({str(_tgt)}
                                                          if str(_tgt) not in _effa2 else None)
                                # ★T2_FORCE_ACTION (2026-07-20·사용자 통찰 "say한 tool do하면 됨"): say-don't-do
                                #   (모델이 실행 의도를 텍스트로만 말하고 호출 0) = 재생성서 tool_choice=required 강제.
                                #   의도는 이미 모델 안에 있으니 산문→호출로 뒤집힘. required=vLLM 구조화디코딩(봉투드롭 불가).
                                #   ★[[10]] 정합: 어느 도구·인자는 모델 몫(강제 안 함)·디코딩 제약만. cap=action_deny 승계·
                                #   provenance/게이트가 다음 라운드서 backstop(날조 인자 차단). 기본 OFF.
                                if os.environ.get("T2_FORCE_ACTION") == "1" and not (am.tool_calls or []):
                                    force_required = True
                                    print("[T2_FORCE_ACTION] say-don't-do → tool_choice=required 재생성",
                                          file=_sys.stderr, flush=True)
                                print("[T2_RESOLVE] action-required reason=%s target=%s"
                                      % (_ar.get("reason"), _tgt), file=_sys.stderr, flush=True)
                    # ★Lever 3 verify-persistence (task_023형·신원수집+검증미완+포기): action-required가
                    #   안 걸렸을 때(예: apply=user-실행 intent) 검증 완결 강제. cap 1/sim.
                    if (rw_fb is None and getattr(self, "_t2_verify_deny", 0)
                            < int(os.environ.get("T2_VERIFY_DENY_CAP", "1"))):
                        _vr = _rz.resolve_verify_persistence(am, state.messages, a2,
                                                             transfer_tools=_transfer_tools(a2))
                        if _vr.get("status") == "deny":
                            rw_fb = ((am.tool_calls or [None])[0], _vr["feedback"])
                            self._t2_verify_deny = getattr(self, "_t2_verify_deny", 0) + 1
                            print("[T2_RESOLVE] verify-persistence deny", file=_sys.stderr, flush=True)
                except Exception as _rze:
                    rw_fb = None
                    print("[T2_RESOLVE] error (no-op): %r" % (_rze,), file=_sys.stderr, flush=True)
            # ★action-required (turn-level·회피=순수조언은 tool_call 0 → 앵커할 ToolMessage 없음).
            #   ✅라이브 배선(2026-07-13): rw_fb[0] is None 케이스는 아래 fb-빌드서 UserMessage 리마인더로
            #   재생성(regenerate-with-directive·eplan-reminder류·작업버퍼만·test_action_reminder 14/14).
            #   transfer-only 회피는 tool_call 앵커로 이미 처리·per-operand rw_fb(deny-kind)도 라이브.
            # ★T2_TOOLLIST (2026-07-21 §2bb·r095g g-t0 실측): 도구목록 **밖** 이름 호출(발명명+
            #   discoverable 접미사-직호출 공히) = 생성-레벨 deny+재생성. 근거: ①TOOLGATE env-실재
            #   통과(§2ao 픽스)가 접미사 직호출을 허용 → unlock+디스패처 쌍(gold 액션형식) 건너뜀
            #   (g-t0 액션 1/9) ②gold census(로컬 results 전수): gold는 전 태스크서 직호출 0 =
            #   over-block 0 ③실행-레벨 deny는 mutating replay 불일치(§2ao·052) — 생성-레벨은
            #   작업버퍼만(비커밋)=replay-clean. 술어=자기 도구목록 대조뿐(도메인 리터럴 0)·도메인
            #   안내문=A2 `nonlisted_tool_feedback`. cap 소진 후엔 통과(liveness·env-실재=replay 정합).
            tl_fb = None
            if (os.environ.get("T2_TOOLLIST") == "1"
                    and not do_gate and not do_prov and ep_fb is None and cons_fb is None
                    and ra_fb is None and te_fb is None and wev_fb is None and rw_fb is None
                    and tl_rounds < 1
                    and getattr(self, "_t2_toollist_deny", 0)
                    < int(os.environ.get("T2_TOOLLIST_CAP", "6"))):
                _vis = {getattr(t, "name", None) for t in (self.tools or [])}
                for c in (am.tool_calls or []):
                    nm = getattr(c, "name", None)
                    if nm and _vis and nm not in _vis:
                        _extra = str((a2 or {}).get("nonlisted_tool_feedback") or "").strip()
                        tl_fb = (c, ("'%s' is not one of your provided tools, so it was not "
                                     "called. Only call tools that appear in your tool list.%s"
                                     % (nm, (" " + _extra) if _extra else "")))
                        print("[T2_TOOLLIST] deny nonlisted tool=%s" % nm,
                              file=_sys.stderr, flush=True)
                        break
            # ★T2_UNLOCK_NAME (2026-07-21 §2bh·rall5 실측): A2 `discoverable_name_check` — 선언된
            #   (도구→인자) 값이 접미사 패턴에 불일치하면 생성-레벨 deny+required regen(비커밋=replay-clean).
            #   근거: FOLLOWUP 강제는 행동을 움직였으나(체크 unlock 3회씩 시도) 전부 bare-name→env
            #   "Unknown tool"→포기 반복(6 sim×2도구×3회) = 마지막 고리가 이름-형식. 엔진=패턴 대조만
            #   (이름 리터럴 0·KB 검색·이름 선택=모델·[[05]](3) autocomplete-변형 기각). cap 소진=통과(현행).
            # ★T2_DISPATCH_ROLE (2026-07-22 §2bl·rall8 031 실측): 디스패처 역할 혼동 deny —
            #   dispute를 user-디스패처로 제출·user-도구를 에이전트가 직접 호출(4회). 술어=대화
            #   자기-이력(이 대화서 unlock된 이름=agent-도구·give된 이름=user-도구)·엔진 이름-리터럴 0.
            #   +tool_arg_allowlist: 선언 외 인자 키 strip(give 여분 'arguments' 실측·커밋 전=replay-safe).
            # ★T2_PRESCRIPTION (2026-07-22 §2bu·rall11 038 실측·격리 L2 8/8=활성화 실패): 처방-오선택 deny.
            #   038: 사기 dispute 요청(unauthorized/fraudulent)인데 apply_statement_credit 오선택(dispute 미착수).
            #   L2 프로브=명시질의 시 file_dispute 8/8 앎 → 자유생성 미발현(활성화). 게이트가 상기:
            #   대화에 dispute-신호 ∧ file_dispute 미호출인데 statement_credit류 write면 deny+dispute 안내.
            #   KB 근거=doc_014(dispute)·doc_017(statement_credit=선의/프로모션/수수료만). 신호·문구=A2·엔진=
            #   문자열 대조+집합(리터럴 0). Δspurious 계측(dispute 후 정당 credit=absent_tool 조건으로 통과).
            pr_fb = None
            _prs = (a2 or {}).get("prescription_redirect") or []
            if (os.environ.get("T2_PRESCRIPTION") == "1" and _prs
                    and not do_gate and not do_prov and ep_fb is None and cons_fb is None
                    and ra_fb is None and te_fb is None and wev_fb is None and rw_fb is None
                    and pc_fb is None
                    and getattr(self, "_t2_prescription_deny", 0)
                    < int(os.environ.get("T2_PRESCRIPTION_CAP", "3"))):
                _conv = " ".join(str(getattr(m2, "content", "") or "") for m2 in state.messages
                                 if getattr(m2, "role", None) in ("user", "tool")).lower()
                _effp = {_eff_tool_name(tc) for m2 in state.messages
                         for tc in (getattr(m2, "tool_calls", None) or [])}
                for c in (am.tool_calls or []):
                    for sp in _prs:
                        _pn = str(_args_dict(c).get(sp.get("arg", "")) or getattr(c, "name", "") or "")
                        if not _pn.startswith(sp.get("prefix", "\0")):
                            continue
                        _abt = sp.get("requires_absent_tool")
                        if _abt and _abt in _effp:
                            continue                     # 이미 정답 도구 호출됨(예: dispute 접수) → 정당 통과
                        if any(sig in _conv for sig in (sp.get("signals") or [])):
                            pr_fb = (c, str(sp.get("feedback") or "Error: [PRESCRIPTION] wrong tool."))
                            self._t2_prescription_deny = getattr(self, "_t2_prescription_deny", 0) + 1
                            print("[T2_PRESCRIPTION] deny tool=%s prefix=%s"
                                  % (_pn, sp.get("prefix")), file=_sys.stderr, flush=True)
                            break
                    if pr_fb:
                        break
            dr_fb = None
            _drs = (a2 or {}).get("dispatcher_role_check") or {}
            if (pr_fb is None and os.environ.get("T2_DISPATCH_ROLE") == "1" and _drs
                    and not do_gate and not do_prov and ep_fb is None and cons_fb is None
                    and ra_fb is None and te_fb is None and wev_fb is None and rw_fb is None
                    and getattr(self, "_t2_dispatchrole_deny", 0)
                    < int(os.environ.get("T2_DISPATCH_ROLE_CAP", "6"))):
                _na = _drs.get("name_args") or {}
                def _iname(tc):
                    _ar = _args_dict(tc)
                    return str(_ar.get(_na.get(getattr(tc, "name", None), ""), "") or "")
                _unl = {_iname(tc) for m2 in state.messages
                        for tc in (getattr(m2, "tool_calls", None) or [])
                        if getattr(tc, "name", None) == _drs.get("unlock_tool")}
                _gvn = {_iname(tc) for m2 in state.messages
                        for tc in (getattr(m2, "tool_calls", None) or [])
                        if getattr(tc, "name", None) == _drs.get("give_tool")}
                for c in (am.tool_calls or []):
                    nm = getattr(c, "name", None); iv = _iname(c)
                    if not iv:
                        continue
                    fbt = None
                    if nm == _drs.get("user_call") and iv in _unl:
                        fbt = _drs.get("agent_via_user_feedback")
                    elif nm == _drs.get("agent_call") and iv in _gvn:
                        fbt = _drs.get("user_via_agent_feedback")
                    elif nm == _drs.get("user_call") and iv in _gvn:
                        fbt = _drs.get("agent_runs_user_feedback")
                    # ★give 대상=agent-도구 deny (2026-07-22 §2bs·rall10 031 실측): 에이전트가
                    #   자기 도구(get_credit_card_accounts_by_user)를 give → env "Unknown discoverable
                    #   tool" 2회에도 같은 오선택 고수. 술어=자기 도구 목록 소속(인터페이스-구조·
                    #   카탈로그 census 불요·이름 리터럴 0): 에이전트가 스스로 부를 수 있는 도구는
                    #   정의상 user-측 discoverable이 아님. 문구=A2(user-지명 이름 재사용 힌트 포함).
                    elif (nm == _drs.get("give_tool")
                          and iv in {getattr(t, "name", None)
                                     for t in (getattr(self, "tools", None) or [])}):
                        fbt = _drs.get("give_agent_tool_feedback")
                    if fbt:
                        dr_fb = (c, str(fbt).replace("{name}", iv))
                        print("[T2_DISPATCH_ROLE] deny tool=%s name=%s" % (nm, iv),
                              file=_sys.stderr, flush=True)
                        break
            _awl = (a2 or {}).get("tool_arg_allowlist") or {}
            if os.environ.get("T2_DISPATCH_ROLE") == "1" and _awl:
                for c in (am.tool_calls or []):
                    _al = _awl.get(getattr(c, "name", None))
                    if not isinstance(_al, list):
                        continue
                    _ar = getattr(c, "arguments", None)
                    if isinstance(_ar, dict):
                        _extra = [k for k in list(_ar.keys()) if k not in _al]
                        for k in _extra:
                            _ar.pop(k, None)
                        if _extra:
                            print("[T2_DISPATCH_ROLE] stripped extra args %s from %s"
                                  % (_extra, getattr(c, "name", None)), file=_sys.stderr, flush=True)
            un_fb = None
            _unspec = (a2 or {}).get("discoverable_name_check") or {}
            if (os.environ.get("T2_UNLOCK_NAME") == "1" and _unspec
                    and not do_gate and not do_prov and ep_fb is None and cons_fb is None
                    and ra_fb is None and te_fb is None and wev_fb is None and rw_fb is None
                    and tl_fb is None
                    and getattr(self, "_t2_unlockname_deny", 0)
                    < int(os.environ.get("T2_UNLOCK_NAME_CAP", "6"))):
                _upat = _unspec.get("pattern") or "_[0-9]+$"
                for c in (am.tool_calls or []):
                    _uarg = (_unspec.get("tools") or {}).get(getattr(c, "name", None))
                    if not _uarg:
                        continue
                    _uval = str(_args_dict(c).get(_uarg) or "")
                    if _uval and not re.search(_upat, _uval):
                        # ★{name_words} (2026-07-22 §2bs·rall10 050/052 실측): bare-name을 자연어
                        #   질의로 파생(suffix 제거+언더스코어→공백) — 도구명 질의=BM25 0.0 마찰의 해소.
                        #   엔진=순수 문자열 연산(리터럴 0)·문구는 A2.
                        un_fb = (c, str(_unspec.get("feedback") or
                                        "Error: '{name}' is missing its required suffix.")
                                 .replace("{name}", _uval)
                                 .replace("{name_words}",
                                          re.sub(_upat, "", _uval).replace("_", " ").strip()))
                        force_required = True     # ★사용자 제안: 재생성은 반드시 도구 호출(KB 검색 유도)
                        print("[T2_UNLOCK_NAME] deny bare name tool=%s val=%s"
                              % (getattr(c, "name", None), _uval), file=_sys.stderr, flush=True)
                        break
            # ★T2_UNKNOWN_NAME_BL (2026-07-22 §2bt·rall11 050 실측): env가 "Unknown ... tool"로
            #   이미 거부한 이름의 재시도 차단. 050: 환각 접미사 _8374를 env 에러 후에도 3연발 —
            #   에러 에코로 이름이 ctx에 들어가 PROV의 ctx-실재 검사가 무력화되는 구멍.
            #   엔진=env 에러 문자열의 인용명 수집(인터페이스 사실)+집합 소속 deny(리터럴 0).
            if (un_fb is None and os.environ.get("T2_UNKNOWN_NAME_BL") == "1"
                    and not do_gate and not do_prov and ep_fb is None and cons_fb is None
                    and ra_fb is None and te_fb is None and wev_fb is None and rw_fb is None
                    and tl_fb is None
                    and getattr(self, "_t2_unknownbl_deny", 0)
                    < int(os.environ.get("T2_UNKNOWN_NAME_BL_CAP", "6"))):
                _ubl = getattr(self, "_t2_unknown_bl", None)
                if _ubl is None:
                    _ubl = self._t2_unknown_bl = set()
                for m in state.messages[-14:]:
                    if getattr(m, "role", None) == "tool":
                        for _mt in re.finditer(r"Unknown (?:agent|discoverable) tool '([^']+)'",
                                               str(getattr(m, "content", "") or "")):
                            _ubl.add(_mt.group(1))
                if _ubl:
                    _na3 = dict(((a2 or {}).get("dispatcher_role_check") or {}).get("name_args") or {})
                    for _k3, _v3 in ((_unspec.get("tools") or {}) if _unspec else {}).items():
                        _na3.setdefault(_k3, _v3)
                    for c in (am.tool_calls or []):
                        _ua3 = _na3.get(getattr(c, "name", None))
                        _uv3 = str(_args_dict(c).get(_ua3) or "") if _ua3 else ""
                        if _uv3 and _uv3 in _ubl:
                            _upat3 = (_unspec.get("pattern") if _unspec else None) or "_[0-9]+$"
                            un_fb = (c, ("Error: '{name}' was already rejected by the environment "
                                         "as an unknown tool earlier in this conversation - that "
                                         "exact name does not exist and retrying it will fail again. "
                                         "Do NOT guess or reuse it. Search the knowledge base with "
                                         "plain words describing the step (for '{name}' e.g. "
                                         "\"{name_words}\") to find the correct full suffixed name, "
                                         "then retry with that name.")
                                     .replace("{name}", _uv3)
                                     .replace("{name_words}",
                                              re.sub(_upat3, "", _uv3).replace("_", " ").strip()))
                            self._t2_unknownbl_deny = getattr(self, "_t2_unknownbl_deny", 0) + 1
                            force_required = True
                            print("[T2_UNKNOWN_NAME_BL] deny env-rejected name tool=%s val=%s"
                                  % (getattr(c, "name", None), _uv3), file=_sys.stderr, flush=True)
                            break
            if (not do_gate and not do_prov and ep_fb is None and cons_fb is None
                    and ra_fb is None and te_fb is None and wev_fb is None and rw_fb is None
                    and tl_fb is None and un_fb is None and dr_fb is None and pc_fb is None
                    and pr_fb is None and hv_fb is None and dd_fb is None):
                break
            main_prov = None
            if do_prov and fab is None:
                # ★L3 origin 케이스: 값∈ctx이나 assistant-first ∧ tool-never = 확인-세탁(t97)
                prov_rounds += 1
                ptc, k, s = ofab
                self._t2_session_bl.add(s)
                self._t2_prov_origin = getattr(self, "_t2_prov_origin", 0) + 1
                print("[T2_PROV] origin regen fired tool=%s arg=%s val=%s"
                      % (getattr(ptc, "name", "?"), k, s), file=_sys.stderr, flush=True)
                _directive = _resolver_directive(a2, ptc, k, s)
                main_prov = (ptc, _directive if _directive is not None
                             else ORIGIN_FEEDBACK.format(k=k, s=s))
            elif do_prov:
                prov_rounds += 1
                ptc, k, s = fab
                self._t2_session_bl.add(s)
                self._t2_regen = getattr(self, "_t2_regen", 0) + 1
                # 관측성(행동 무변경): p4-비용 귀속용 발화 로그 (C53 p4 −5.3pp·§3c)
                print("[T2_PROV] regen fired tool=%s arg=%s val=%s" % (getattr(ptc, "name", "?"), k, s),
                      file=_sys.stderr, flush=True)
                _directive = _resolver_directive(a2, ptc, k, s)  # ★B-max① t17: resolver_path 지목
                # ★PROV-ADDR-FULL (T2_PROV_ADDR_FULL=1·2026-07-12 HANDOFF §6.3·A1 부작용 교정):
                #   주소류 free-text 인자는 rescue 중립문(getter-일반)이 약해 날조가 통과(t43/96) →
                #   full 문구(REGEN_FEEDBACK=주소-getter 명시 예시)로 강제 조회 유도. rescue 스킵은
                #   env-검증 id(hashid/numid)만 — 주소=free-text는 애초 스킵 대상 아님(§6.3 "rescue=env-id만").
                _addr_full = (os.environ.get("T2_PROV_ADDR_FULL") == "1"
                              and prov_mode == "rescue" and _is_addr_arg(k))
                if _addr_full:
                    self._t2_prov_addr_full = getattr(self, "_t2_prov_addr_full", 0) + 1
                _tmpl = REGEN_FEEDBACK if (prov_mode != "rescue" or _addr_full) \
                    else REGEN_FEEDBACK_NEUTRAL
                main_prov = (ptc, _directive if _directive is not None
                             else _tmpl.format(k=k, s=s))
            if do_gate:
                gate_rounds += 1
                self._t2_gate_rounds = getattr(self, "_t2_gate_rounds", 0) + 1
                _budget_tick(self)  # ★게이트 라운드만 과금 (prov=무과금=C53 semantics)
            if ep_fb is not None:
                eplan_rounds += 1
                # ★C174(사용자 지시 2026-07-25): deny 예산 **전면 독립** — 공용 예산 폐지.
                #   같은 종류의 deny는 자기 cap 안에서 여러 번 발화 가능하되, 무관한 deny끼리는
                #   서로의 예산을 건드리지 않는다. pre-close 발화는 발화 지점서 자기 카운터
                #   (_t2_preclose_deny)만 소모하므로 여기선 discovery 카운터를 증가시키지 않음.
                if getattr(self, "_t2_ep_fb_preclose", False):
                    self._t2_ep_fb_preclose = False
                else:
                    self._t2_eplan_deny = getattr(self, "_t2_eplan_deny", 0) + 1
                    if self._t2_eplan_deny == _ep_cap:  # 관측 마커: 이후 discovery deny 중단
                        print("[T2_EPLAN] deny cap %d reached — no further discovery denies this sim"
                              % _ep_cap, file=_sys.stderr, flush=True)
            if cons_fb is not None:
                cons_rounds += 1
                self._t2_cons_deny = getattr(self, "_t2_cons_deny", 0) + 1
            if dd_fb is not None:
                self._t2_dd_deny = getattr(self, "_t2_dd_deny", 0) + 1
                print("[T2_DISCOVERY_DISPATCH] deny direct call=%s → force dispatcher"
                      % getattr(dd_fb[0], "name", "?"), file=_sys.stderr, flush=True)
            if ra_fb is not None:
                ra_rounds += 1
                self._t2_readall_deny = getattr(self, "_t2_readall_deny", 0) + 1
            if te_fb is not None:
                te_rounds += 1
                self._t2_toolerr_deny = getattr(self, "_t2_toolerr_deny", 0) + 1
            if wev_fb is not None:
                wev_rounds += 1
                self._t2_wev_deny = getattr(self, "_t2_wev_deny", 0) + 1
                if self._t2_wev_deny == _wev_cap:  # 관측 마커(sim당 1회): 이후 WEV deny 중단
                    print("[T2_WRITE_EVIDENCE] deny cap %d reached — no further WEV denies this sim"
                          % _wev_cap, file=_sys.stderr, flush=True)
            if rw_fb is not None:
                self._t2_resolve_deny = getattr(self, "_t2_resolve_deny", 0) + 1
            if tl_fb is not None:
                tl_rounds += 1
                self._t2_toollist_deny = getattr(self, "_t2_toollist_deny", 0) + 1
                if self._t2_toollist_deny == int(os.environ.get("T2_TOOLLIST_CAP", "6")):
                    print("[T2_TOOLLIST] deny cap reached — nonlisted calls pass through hereafter",
                          file=_sys.stderr, flush=True)
            if un_fb is not None:
                self._t2_unlockname_deny = getattr(self, "_t2_unlockname_deny", 0) + 1
            if dr_fb is not None:
                self._t2_dispatchrole_deny = getattr(self, "_t2_dispatchrole_deny", 0) + 1
            if hv_fb is not None:
                if getattr(self, "_t2_valacq_fired", False):   # ★C119: va가 hv_fb 채널 재사용 → 별도 카운터
                    self._t2_valacq_deny = getattr(self, "_t2_valacq_deny", 0) + 1
                    self._t2_valacq_fired = False
                else:
                    self._t2_havevalue_deny = getattr(self, "_t2_havevalue_deny", 0) + 1
                # ★force_required (T2_UNLOCK_NAME 선례·[[10]]): 실패모드=프로즈 재요청(say-don't-do).
                #   넛지 후 재생성을 tool_choice=required로 봉쇄 → 산문 대신 반드시 도구 호출(어느 도구=모델).
                #   기본 ON. ★2026-07-23 근본원인 규명: required는 라이브 경로서 정상 동작(라이브 llm_args가
                #   max_tokens 미설정=vLLM 기본 대형 → 강제 tool-call 완성). 앞선 프로브 400은 스크립트가
                #   max_tokens=450/20을 하드코딩해 강제 JSON이 절단된 아티팩트(vLLM #19051/#36794)였음 —
                #   _gen의 max_tokens 하한이 교정. 병리적 runaway(039 퇴행루프)만 _gen 폴백으로 강등.
                if os.environ.get("T2_HAVE_VALUE_FORCE", "1") == "1":
                    force_required = True
            fb = [am]
            for c in (am.tool_calls or []):
                if do_gate and id(c) in denied_by_objid:
                    gid, why = denied_by_objid[id(c)]
                    content = f"Error: [POLICY GATE {gid}] {why}"
                elif main_prov is not None and c is main_prov[0]:
                    content = main_prov[1]
                elif ep_fb is not None and c is ep_fb[0]:
                    content = "Error: " + ep_fb[1]
                elif dd_fb is not None and c is dd_fb[0]:
                    content = "Error: [DISCOVERY] " + dd_fb[1]
                elif cons_fb is not None and c is cons_fb[0]:
                    content = "Error: " + cons_fb[1]
                elif ra_fb is not None and c is ra_fb[0]:
                    content = "Error: " + ra_fb[1]
                elif te_fb is not None and c is te_fb[0]:
                    content = "Error: " + te_fb[1]
                elif wev_fb is not None and c is wev_fb[0]:
                    # A2 feedback이 "Error:"로 시작하면 그대로(이중 접두 방지)
                    content = wev_fb[1] if str(wev_fb[1]).lstrip().startswith("Error:") \
                        else "Error: " + wev_fb[1]
                elif rw_fb is not None and c is rw_fb[0]:
                    content = "Error: " + rw_fb[1]
                elif tl_fb is not None and c is tl_fb[0]:
                    content = "Error: " + tl_fb[1]
                elif un_fb is not None and c is un_fb[0]:
                    content = un_fb[1] if str(un_fb[1]).lstrip().startswith("Error:") \
                        else "Error: " + un_fb[1]
                elif pc_fb is not None and c is pc_fb[0]:
                    content = pc_fb[1] if str(pc_fb[1]).lstrip().startswith("Error:") else "Error: " + pc_fb[1]
                elif dr_fb is not None and c is dr_fb[0]:
                    content = dr_fb[1] if str(dr_fb[1]).lstrip().startswith("Error:") \
                        else "Error: " + dr_fb[1]
                elif pr_fb is not None and c is pr_fb[0]:
                    content = pr_fb[1] if str(pr_fb[1]).lstrip().startswith("Error:") \
                        else "Error: " + pr_fb[1]
                else:
                    content = "Error: resolve the flagged call(s) first; do not call this tool yet."
                fb.append(ToolMessage(id=c.id, role="tool", requestor="assistant",
                                      error=True, content=content))
            # ★action-required 리마인더 채널 (순수-조언 회피=tool_call 0 → 앵커할 ToolMessage 없음).
            #   rw_fb[0] is None = 순수-조언 action-required(2085행). UserMessage 리마인더로 재생성.
            #   작업버퍼(work)만·state.messages 비커밋 = 채널 절대규칙(1849·replay-clean).
            if rw_fb is not None and rw_fb[0] is None and not (am.tool_calls or []):
                try:
                    fb.append(UserMessage(role="user", content=rw_fb[1]))
                except TypeError:
                    fb.append(UserMessage(content=rw_fb[1]))
            # ★T2_HAVE_VALUE 리마인더 (None-anchor·산문 회피 또는 producer 재호출 커버·비커밋=replay-clean)
            if hv_fb is not None:
                try:
                    fb.append(UserMessage(role="user", content=hv_fb))
                except TypeError:
                    fb.append(UserMessage(content=hv_fb))
            work = work + fb
            am = _gen(self, work, bw(), "agent_response_unified_regen",
                      tool_choice="required" if force_required else None)

        # R8 종단: 잔존 게이트-deny 호출 strip (재과금 없음·히스토리 replay-clean)
        if gate is not None:
            denied = _denied_calls(am, gate, last_user, transfer_sent)
            if denied:
                d_ids = {tc.id for tc, _, _ in denied}
                kept = [tc for tc in (am.tool_calls or []) if tc.id not in d_ids]
                am.tool_calls = kept or None
                note = "; ".join(f"[{gid}] {(why or '')[:70]}" for _, gid, why in denied)
                am.content = (am.content or "") + _BLOCK_NOTE + " (" + note + ")"
                self._t2_gate_strips = getattr(self, "_t2_gate_strips", 0) + 1
                print("[T2_UNIFIED] R8 strip: %s" % note[:140], file=_sys.stderr, flush=True)
        # ★EXHAUSTION→FAIL (T2_FAB_STRIP=1·BANK_IMPL_REDESIGN §2·2026-07-16):
        #   regen 소진 후에도 근거 없는(id-operand ∉ctx) WRITE 호출 = pass-through 금지 → strip + abstain.
        #   (C12 "id 날조는 env가 거부" 가정이 banking 디스패처 dispute엔 불성립=날조 txn이 reward0로 통과.)
        #   read/procedural=무해(strip 안함)·over-block 방지=id-operand가 ctx에 없는 write만·디스패처 nested unwrap.
        if os.environ.get("T2_FAB_STRIP") == "1" and getattr(am, "tool_calls", None):
            _RDP, _PRC = _READ_PREFIX_RE, _PROCEDURAL_RE   # ★hoist 정본 재사용([[03b]] 술어 이중화 제거·동일 정규식)
            def _fab_write_ungrounded(tc):
                nm = getattr(tc, "name", "") or ""
                ar = _args_dict(tc)
                # ★F6b(C211·리뷰 가드레일: **국소 수정** — 공유 _PROCEDURAL_RE/_eff_tool_name 불변):
                #   ⓐ inner-key에 discoverable_tool_name 추가(give의 실효 이름은 이 키에 실림 —
                #   _eff_tool_name(1667)엔 있는데 여기 없던 unwrap 불일치가 day6 발명-id give 통과 구멍)
                #   ⓑ give/call 디스패처는 inner 이름으로만 면제 판정(외피 'give_/discoverable' 매칭 금지·
                #   unlock은 구판 외피-판정 유지=안전측). 회귀: _is_effective_write("give_…")=False 유지.
                inner = (ar.get("agent_tool_name") or ar.get("user_tool_name")
                         or ar.get("discoverable_tool_name") or "")
                eff = re.sub(r"_\d+$", "", str(inner or nm))
                if inner and re.match(r"^(give|call)_", str(nm), re.I):
                    if _RDP.match(eff) or _PRC.search(eff):
                        return False  # inner가 read/procedural일 때만 무해
                elif not eff or _RDP.match(eff) or _PRC.search(eff):
                    return False  # read/procedural = 무해
                sub = ar.get("arguments")
                if isinstance(sub, str):
                    try:
                        sub = json.loads(sub)
                    except Exception:
                        sub = {}
                d = sub if isinstance(sub, dict) else ar
                for k, v in (d or {}).items():
                    if not _hint_hit(k, hints):
                        continue  # id-like 인자만
                    for val in _flatten(v):
                        s = str(val).strip()
                        if len(s) >= 4 and s.lower() not in ctx:
                            return True  # 근거없는 id-operand 있는 write
                return False
            _fab_ids = {id(tc) for tc in (am.tool_calls or []) if _fab_write_ungrounded(tc)}
            if _fab_ids:
                _kept = [tc for tc in (am.tool_calls or []) if id(tc) not in _fab_ids]
                am.tool_calls = _kept or None
                # C125: 유저-대면 문자열은 영어(한글 산문 노출이 rall20 043.0 [24] 유저 혼란 실측)
                am.content = ((am.content or "")
                              + " [Note: items whose supporting records could not be verified were not processed.]")
                self._t2_fab_strips = getattr(self, "_t2_fab_strips", 0) + len(_fab_ids)
                print("[T2_FAB_STRIP] dropped %d ungrounded write call(s) (exhaustion->abstain)"
                      % len(_fab_ids), file=_sys.stderr, flush=True)
        # prov-fab 잔존 = 통과 (기존 prov semantics·id 날조는 env가 거부=C12)

        # ★T2_STALE_STRIP (2026-07-23·8-task per-step 포렌식): over-action 억제 — 한 턴에 12개 도구
        #   대량 병렬 중복 재호출(043[24]·054[55]·038[36] 실측: 이미 성공한 조회/write를 재호출=낭비/DB오염).
        #   ①같은 am 내 완전중복(동일 eff+args 2회+·read/write 공통·명백 무의미) ②committed서 이미 성공한
        #   *write* 재호출(중복 write=DB오염·054 gold-diff). strip만(넛지 없음·R8/FAB_STRIP 동형). read의
        #   committed-재조회는 상태변화 가능성 존중해 미strip(over-fire 방지·같은-턴 반복만). 도메인 리터럴 0
        #   (eff+args 대조·write집합=A2 confirm/eplan 도출). [[05]]: 결정론 중복감지·모델판단 아님·strip만.
        if os.environ.get("T2_STALE_STRIP") == "1" and getattr(am, "tool_calls", None):
            _wtools = _confirm_write_tools(a2) | set(((a2 or {}).get("eplan") or {}).get("write_tools") or [])
            _stale = _stale_call_ids(am, state.messages, _wtools)
            if _stale:
                _kept = [tc for tc in (am.tool_calls or []) if id(tc) not in _stale]
                am.tool_calls = _kept or None
                am.content = (am.content or "") + " [중복 호출 제거: 이미 완료한 조회/작업은 반복하지 않았습니다.]"
                self._t2_stale_strips = getattr(self, "_t2_stale_strips", 0) + len(_stale)
                print("[T2_STALE_STRIP] dropped %d stale/dup call(s)" % len(_stale),
                      file=_sys.stderr, flush=True)

        # ── DISAMB: 문맥-실재값·같은-형식 후보 2+ → 1회 재확인 (기존 로직 이식) ──
        # ★#2 operand 스코프(2026-07-13 A1-v2 실패분석): variant operand(new_item_ids)는 L4 전담 —
        #   order-filter가 여기서 new_item에 오발화해 틀린 변형 치환(3234800602 반복)했음. 제외.
        _v_ops = set((a2 or {}).get("variant_operand") or [])
        if disamb_tools and getattr(am, "tool_calls", None):
            if not hasattr(self, "_t2_disamb_seen"):
                self._t2_disamb_seen = set()
            hit = None
            for tc in am.tool_calls:
                if getattr(tc, "name", None) not in disamb_tools:
                    continue
                for k, v in _args_dict(tc).items():
                    if k in _v_ops:                     # ★L4 전담 operand는 disamb-filter 제외
                        continue
                    if not _hint_hit(k, hints):
                        continue
                    for val in _flatten(v):
                        s = str(val).strip()
                        if len(s) < 4 or s.lower() not in ctx:
                            continue
                        memo = (tc.name, k, s.lower())
                        if memo in self._t2_disamb_seen:
                            continue
                        cands = _grounded_candidates(k, s, state.messages)
                        if len(cands) >= 2 and any(s.lower() == str(c).lower() for c in cands):
                            hit = (tc, k, s, cands, memo)
                            break
                    if hit:
                        break
                if hit:
                    break
            if hit and disamb_mode == "subcall":
                # ★T5-C P-B: 격리 서브콜 판정 + 제자리 치환(원턴·대화 불변). switch가 게이트-deny를
                #   ★*새로* 유발할 때만 원값 복원(리뷰 블로킹2 정신). 스위치는 whitelist operand(item)만
                #   바꾸므로 통상 게이트-무관 — *이미* deny되던 호출(다른 사유)은 R8이 어차피 strip하니
                #   되돌리면 좋은 switch만 손실. 그래서 pre/post deny 비교로 switch-유발分만 복원.
                tc, k, s, cands, memo = hit
                self._t2_disamb_seen.add(memo)
                self._t2_disamb = getattr(self, "_t2_disamb", 0) + 1
                print("[T2_DISAMB] fired tool=%s arg=%s val=%s ncand=%d mode=subcall"
                      % (tc.name, k, s, len(cands)), file=_sys.stderr, flush=True)
                import copy as _copy
                _snap = _copy.deepcopy(getattr(tc, "arguments", None))
                _pre_denied = ({d[0].id for d in _denied_calls(am, gate, last_user, transfer_sent)}
                               if gate is not None else set())
                res = _t5c_disamb_subcall(self, la, UserMessage, state.messages, tc, k, s, sub_args)
                if res == "switch" and gate is not None:
                    _post = {d[0].id for d in _denied_calls(am, gate, last_user, transfer_sent)}
                    if tc.id in _post and tc.id not in _pre_denied:  # switch가 *새로* 유발한 deny만
                        tc.arguments = _snap
                        self._t2_disamb_gate_reject = getattr(self, "_t2_disamb_gate_reject", 0) + 1
                        print("[T2_UNIFIED] SUBCALL switch reverted: switch-caused gate-deny",
                              file=_sys.stderr, flush=True)
                hit = None
            if hit and disamb_mode == "enumerate":
                # ★결정론 filter-substitute (2026-07-12 HANDOFF §6.1·LOCK §4d FIND=fexec):
                #   t71 실증 = filter-then-ask *지시*는 32B 미준수([[42]]) → 지시 대신 엔진이 직접
                #   LLM-formalize→결정론 필터 실행. 1 통과=제자리 치환(subcall switch 동형:
                #   whitelist(B2)+게이트 재검사·switch-유발 deny만 복원) / ≥2=통과분으로 축소해
                #   열거-ASK / 판정불가·empty소진=기존 열거 피드백 폴백(아래 if hit: 블록).
                tc, k, s, cands, memo = hit
                fsub = None
                if _key_tokens(k) & sub_args:
                    try:
                        import t2_formalize_exec as _fx
                        fsub = _fx.fexec_filter_decide(self, la, UserMessage, state.messages, k, s)
                    except Exception as _fe:
                        print("[T2_FSUB] error (fallback): %r" % (_fe,), file=_sys.stderr, flush=True)
                        fsub = None
                if fsub and fsub["status"] == "one":
                    self._t2_disamb_seen.add(memo)
                    self._t2_disamb = getattr(self, "_t2_disamb", 0) + 1
                    print("[T2_DISAMB] fired tool=%s arg=%s val=%s ncand=%d mode=fsub"
                          % (tc.name, k, s, len(cands)), file=_sys.stderr, flush=True)
                    win = str(fsub["ids"][0]).strip()
                    if win.lower() == s.lower():
                        self._t2_fsub_keep = getattr(self, "_t2_fsub_keep", 0) + 1
                        print("[T2_FSUB] confirmed arg=%s val=%s" % (k, s),
                              file=_sys.stderr, flush=True)
                    else:
                        import copy as _copy
                        _snap = _copy.deepcopy(getattr(tc, "arguments", None))
                        _pre = ({d[0].id for d in _denied_calls(am, gate, last_user, transfer_sent)}
                                if gate is not None else set())
                        if _subst_arg_value(tc, k, s, win):
                            self._t2_fsub_switch = getattr(self, "_t2_fsub_switch", 0) + 1
                            print("[T2_FSUB] substituted arg=%s from=%s to=%s" % (k, s, win),
                                  file=_sys.stderr, flush=True)
                            if gate is not None:
                                _post = {d[0].id for d in
                                         _denied_calls(am, gate, last_user, transfer_sent)}
                                if tc.id in _post and tc.id not in _pre:  # switch가 *새로* 유발한 deny만
                                    tc.arguments = _snap
                                    self._t2_disamb_gate_reject = \
                                        getattr(self, "_t2_disamb_gate_reject", 0) + 1
                                    print("[T2_UNIFIED] FSUB switch reverted: switch-caused gate-deny",
                                          file=_sys.stderr, flush=True)
                        else:
                            self._t2_fsub_nosub = getattr(self, "_t2_fsub_nosub", 0) + 1
                            print("[T2_FSUB] substitution no-op arg=%s (value shape)" % k,
                                  file=_sys.stderr, flush=True)
                    hit = None
                elif fsub and fsub["status"] == "many":
                    # 통과 후보로 축소 → 아래 기존 열거-ASK 피드백이 축소본을 소비
                    self._t2_fsub_narrowed = getattr(self, "_t2_fsub_narrowed", 0) + 1
                    print("[T2_FSUB] narrowed arg=%s ncand %d->%d" % (k, len(cands), len(fsub["ids"])),
                          file=_sys.stderr, flush=True)
                    hit = (tc, k, s, [str(i) for i in fsub["ids"]], memo)
            if hit:
                tc, k, s, cands, memo = hit
                self._t2_disamb_seen.add(memo)
                self._t2_disamb = getattr(self, "_t2_disamb", 0) + 1
                print("[T2_DISAMB] fired tool=%s arg=%s val=%s ncand=%d" % (tc.name, k, s, len(cands)),
                      file=_sys.stderr, flush=True)
                dwork = list(work) + [am]
                _fbtmpl = DISAMB_ENUM_FEEDBACK if disamb_mode == "enumerate" else DISAMB_FEEDBACK
                fbtxt = _fbtmpl.format(k=k, s=s, n=len(cands),
                                       cands=", ".join(repr(c) for c in cands[:8]))
                for c in (am.tool_calls or []):
                    reason = fbtxt if c is tc else \
                        "Error: [DISAMBIGUATE] re-check pending; re-emit this call after resolving."
                    dwork.append(ToolMessage(id=c.id, role="tool", requestor="assistant",
                                             error=True, content=reason))
                am2 = _gen(self, dwork, bw(), "agent_response_disamb")
                n2 = 0
                while n2 < 2:  # 재확인 응답의 신규 날조는 prov 루프로 정화(2회 한도·무과금)
                    fab2 = _first_fab_call(am2, ctx, hints)
                    if fab2 is None:
                        break
                    tc2, k2, s2 = fab2
                    n2 += 1
                    self._t2_session_bl.add(s2)
                    dwork = dwork + [am2]
                    for c in (am2.tool_calls or []):
                        reason = REGEN_FEEDBACK.format(k=k2, s=s2) if c is tc2 else \
                            "Error: [PROVENANCE] resolve the invented value first; do not call this yet."
                        dwork.append(ToolMessage(id=c.id, role="tool", requestor="assistant",
                                                 error=True, content=reason))
                    am2 = _gen(self, dwork, bw(), "agent_response_regen")
                if _first_fab_call(am2, ctx, hints) is None:
                    # ★리뷰 블로킹2: 재확인 switch가 게이트-deny 호출을 들여오면 원 am(게이트-클린) 유지
                    if gate is not None and _denied_calls(am2, gate, last_user, transfer_sent):
                        self._t2_disamb_gate_reject = getattr(self, "_t2_disamb_gate_reject", 0) + 1
                        print("[T2_UNIFIED] DISAMB rejected: switch is gate-denied; keeping original",
                              file=_sys.stderr, flush=True)
                    elif getattr(am2, "tool_calls", None):
                        sw = any(str(vv).strip().lower() != s.lower()
                                 for c2 in am2.tool_calls if c2.name == tc.name
                                 for kk, vv0 in _args_dict(c2).items() if kk == k
                                 for vv in _flatten(vv0))
                        if sw:
                            print("[T2_DISAMB] switched arg=%s from=%s" % (k, s),
                                  file=_sys.stderr, flush=True)
                        am = am2
                    else:
                        # ★T5-C fix(C61 H-C): 텍스트-only 재확인은 원 호출 유지(write 유실 차단)
                        self._t2_disamb_nowrite_keep = getattr(self, "_t2_disamb_nowrite_keep", 0) + 1
                        print("[T2_DISAMB] rejected: re-check dropped tool_calls; keeping original",
                              file=_sys.stderr, flush=True)
        # ★L4 fexec-variants (opt-in T2_L4=1·A1_V3): write 도구의 variant operand(A2 variant_operand,
        #   예 new_item_ids)를 극값/속성 결정론 선택으로 치환. I7 floor-guard: one=치환·그 외 no-op.
        #   [[05]]: variant_operand/spec = A2 데이터(엔진 리터럴 0)·fexec_variant_decide = 도메인일반.
        #   ★v3.2 (A1_V3_PROBE_FORENSIC §3): 치환 성적 2/2 오답(t58 정답파괴=교차-품목 기준누출 F1·
        #   t20 제약절단 F2+승인-불일치 F4) → T2_L4_MODE 기본 "keep"(관측·audit only·치환 없음).
        #   "substitute"는 재설계 요건(F1 per-slot attested·F2 constrained/no-op·F4 G-approve) 충족 후.
        if os.environ.get("T2_L4") == "1" and a2 is not None and getattr(am, "tool_calls", None):
            v_ops = set((a2 or {}).get("variant_operand") or [])
            v_spec = (a2 or {}).get("variant_spec")
            if v_ops and v_spec:
                req_text = " ".join(str(getattr(m, "content", "") or "")
                                    for m in state.messages if getattr(m, "role", None) == "user")
                try:
                    import t2_formalize_exec as _fx
                    import copy as _copy
                    anchor_op = (v_spec or {}).get("anchor_operand")
                    for tc in (am.tool_calls or []):
                        d = _args_dict(tc)
                        for k in list(d.keys()):
                            if k not in v_ops:
                                continue
                            newv = d.get(k)
                            newl = newv if isinstance(newv, list) else [newv]
                            anchl = d.get(anchor_op) if anchor_op else None
                            anchl = anchl if isinstance(anchl, list) else ([anchl] if anchl else [])
                            for i, cur in enumerate(newl):    # ★인덱스 짝: new_item[i]↔item[i]=anchor
                                cur = str(cur).strip()
                                if len(cur) < 3:
                                    continue
                                anchor = anchl[i] if i < len(anchl) else (anchl[0] if len(anchl) == 1 else None)
                                vr = _fx.fexec_variant_decide(self, la, UserMessage, state.messages,
                                                              k, cur, v_spec, req_text, anchor_id=anchor)
                                if vr.get("status") == "one":
                                    win = str(vr["ids"][0]).strip()
                                    self._t2_l4 = getattr(self, "_t2_l4", 0) + 1
                                    if win.lower() != cur.lower():
                                        # ★v3.2 keep-모드(기본): 치환 없이 관측만(audit 라인 유지)
                                        if os.environ.get("T2_L4_MODE", "keep") != "substitute":
                                            print("[T2_L4] keep-mode: would-substitute arg=%s from=%s to=%s"
                                                  % (k, cur, win), file=_sys.stderr, flush=True)
                                            continue
                                        # ★G-approve (F4·t20): cur가 대화(비-tool 발화)에 verbatim 등장
                                        #   = 제시·승인된 값 → 몰래 치환 금지 (도메인 리터럴 0·문자열 대조)
                                        _dlg = " ".join(str(getattr(m, "content", "") or "")
                                                        for m in state.messages
                                                        if getattr(m, "role", None) in ("user", "assistant"))
                                        if cur and cur in _dlg:
                                            print("[T2_L4] G-approve: arg=%s val=%s surfaced in dialog — no substitute"
                                                  % (k, cur), file=_sys.stderr, flush=True)
                                            continue
                                        _snap = _copy.deepcopy(getattr(tc, "arguments", None))
                                        _pre = ({dd[0].id for dd in _denied_calls(am, gate, last_user, transfer_sent)}
                                                if gate is not None else set())
                                        if _subst_arg_value(tc, k, cur, win):
                                            self._t2_l4_sub = getattr(self, "_t2_l4_sub", 0) + 1
                                            print("[T2_L4] substituted arg=%s from=%s to=%s"
                                                  % (k, cur, win), file=_sys.stderr, flush=True)
                                            if gate is not None:
                                                _post = {dd[0].id for dd in
                                                         _denied_calls(am, gate, last_user, transfer_sent)}
                                                if tc.id in _post and tc.id not in _pre:  # I7: switch-유발 deny 복원
                                                    tc.arguments = _snap
                                                    print("[T2_L4] reverted: switch-caused gate-deny",
                                                          file=_sys.stderr, flush=True)
                                    else:
                                        print("[T2_L4] confirmed arg=%s val=%s" % (k, cur),
                                              file=_sys.stderr, flush=True)
                except Exception as _le:
                    print("[T2_L4] error (no-op): %r" % (_le,), file=_sys.stderr, flush=True)
        # ★T5-C P2 원리-디폴트(opt-in T2_PRINCIPLE_DEFAULT=1): write operand 기본값(원결제 등)
        #   위반 시 제자리 치환. user-발화만 override 근거(tool출력의 계정값 아님).
        if os.environ.get("T2_PRINCIPLE_DEFAULT") == "1" and gate is not None:
            uctx = " ".join(str(getattr(m, "content", "") or "").lower()
                            for m in state.messages if getattr(m, "role", None) == "user")
            nsub = _apply_principle_default(am, a2, gate, uctx)
            if nsub:
                self._t2_principle_default = getattr(self, "_t2_principle_default", 0) + nsub
        # ★NL-NUM-PROV (opt-in T2_NLNUM_PROV=1·t47형): 최종 반환 직전 텍스트 발화의
        #   통화-금액 provenance 검사 → 생성-레벨 regen 1회(무과금·비커밋). 상한 1/턴
        #   (이 블록은 턴당 1회 실행·루프 없음). 채택 전 게이트 재검사 — regen이
        #   게이트-deny 호출을 새로 들이면 원 am 유지(안전측·부작용 0).
        if (os.environ.get("T2_NLNUM_PROV") == "1" and (a2 or {}).get("calc_tool")
                and isinstance(getattr(am, "content", None), str)):
            ctx_num = _ctx_from_messages(state.messages).replace(",", "")
            bad = _unverified_amounts(am.content, ctx_num)
            if bad:
                self._t2_nlnum = getattr(self, "_t2_nlnum", 0) + 1
                print("[T2_NLNUM] fired amount=%s" % bad[0], file=_sys.stderr, flush=True)
                fbtxt = NLNUM_FEEDBACK.format(amt=bad[0], tool=a2["calc_tool"])
                try:
                    nfb = UserMessage(role="user", content=fbtxt)
                except TypeError:
                    nfb = UserMessage(content=fbtxt)
                am2 = _gen(self, work + [am, nfb], bw(), "agent_response_nlnum")
                if gate is not None and _denied_calls(am2, gate, last_user, transfer_sent):
                    self._t2_nlnum_gate_reject = getattr(self, "_t2_nlnum_gate_reject", 0) + 1
                    print("[T2_NLNUM] rejected: regen introduced gate-denied call; keeping original",
                          file=_sys.stderr, flush=True)
                else:
                    am = am2
        # ★assertion-provenance 2-arm (opt-in·기본 OFF·floor 불변).
        #   발화 조건 = **사임**(tool_calls 없는 텍스트 발화 = 턴을 사용자에게 넘김) — 구조 이벤트.
        #   상한 1/sim(에이전트 인스턴스=sim). regen 채택 전 게이트 재검사(NLNUM과 동형·안전측).
        _resign = (not getattr(am, "tool_calls", None)
                   and isinstance(getattr(am, "content", None), str) and am.content.strip())
        # ★T2_FOLLOWUP_READLOOP (2026-07-22 §2bk·rall7 050 실측): post-submit에 모델이 KB-검색류
        #   read만 계속 돌리면 사임(텍스트-턴) 이벤트가 영영 없어 chain nudge 0회로 종료(리서치-루프
        #   회피). 확장 술어(도메인일반·A2 chain 선언 대조): after-도구 실행됨 ∧ requires 누락 ∧ 이번
        #   턴 호출 전부 비-write ∧ 그 호출들이 누락 requires와 무관 → 사임-등가로 카운트.
        if (os.environ.get("T2_FOLLOWUP_READLOOP") == "1" and not _resign
                and getattr(am, "tool_calls", None) and a2 is not None):
            try:
                _effh = {_eff_tool_name(tc) for m2 in state.messages
                         for tc in (getattr(m2, "tool_calls", None) or [])}
                _hitc = next(((_fc, _chain_dispatch(_fc, _effh))
                              for _fc in (a2.get("follow_up_chains") or [])
                              if _chain_dispatch(_fc, _effh) is not None), None)
                if _hitc is not None:
                    _ame = [_eff_tool_name(tc) for tc in (am.tool_calls or [])]
                    _reqs = _hitc[0].get("requires")
                    _reqs = set(_reqs if isinstance(_reqs, list) else [_reqs])
                    if (all(not _is_effective_write(e) for e in _ame)
                            and not (set(_ame) & _reqs)):
                        _resign = True
                        # ★C207/B1(리뷰 필수2): readloop 변환 턴을 **표시**한다 — chain 예비-예산은
                        #   진성 사임-턴에서만 소비해야 한다(035 day4b 실측: cap 소진 국면이 전부
                        #   KB-루프 턴이었고 발화 3회 전부 빈손 → 예비도 같은 곳에 버려질 위험).
                        self._t2_fu_readloop_turn = True
                        print("[T2_FOLLOWUP] readloop-turn counted as resignation",
                              file=_sys.stderr, flush=True)
            except Exception:
                pass

        def _ap_regen(fbtxt, tag, tool_choice=None, am_override=None):
            """피드백 1회 → regen. 게이트-deny 유입 시 원본 유지(부작용 0). 성공 시 새 am.
            ★tool_choice(레버 A·2026-07-18·`HANDOFF_LEVER_DESIGN §2`): regen 응답의 **채널만** 강제
            (어느 도구를 부를지는 모델이 고름). 실측 근거 = forced 프로브: 강제 하 24/24 정답 선택 ·
            같은 지시를 **말로** 하면 56%로 악화(단일변수·`forced_probe_20260718`).
            ★전역 regen 예산(§2ah): 소진 시 발화 skip(원본 유지)=컨텍스트 누적 상한(023 overflow 차단)."""
            if not _regen_budget_ok(self):
                print("[%s] skipped: global regen budget exhausted" % tag.upper(),
                      file=_sys.stderr, flush=True)
                return None
            try:
                _fb = UserMessage(role="user", content=fbtxt)
            except TypeError:
                _fb = UserMessage(content=fbtxt)
            # ★C207(리뷰 필수1): regen 프롬프트에 실리는 직전 응답을 **호출부가 대체**할 수 있게 한다.
            #   근거: 폭주 응답(33k자·8k토큰)을 그대로 실으면 regen 호출 자체가 창 초과로 죽고, 그 예외는
            #   여기서 전파돼 **ctxover로 끝날 sim을 더 이른 크래시로** 바꾼다. blob은 어차피 비커밋이라
            #   절단본 대체는 역사 훼손이 아니다(프레임워크 층·도메인 리터럴 0).
            _am_for_prompt = am if am_override is None else am_override
            try:
                _am2 = _gen(self, work + [_am_for_prompt, _fb], bw(),
                            "agent_response_" + tag, tool_choice=tool_choice)
            except Exception as _ge:
                # ★C207: regen 실패(창 초과 등)는 **원본 유지로 흡수**(현행 거동) — 크래시 승격 금지.
                print("[%s] regen failed (keeping original): %r" % (tag.upper(), _ge),
                      file=_sys.stderr, flush=True)
                return None
            if gate is not None:
                _den = _denied_calls(_am2, gate, last_user, transfer_sent)
                if _den:
                    # ★C170 부분-수용(W6 조정·2026-07-25): all-or-nothing 기각이 좋은 호출까지 버렸다
                    #   (cand4 035 실측: regen=unlock+call(emergency)+transfer 묶음서 transfer만 GB2-deny
                    #   인데 전체 기각→emergency 호출 소실→유저 토큰 종료로 실행 기회 상실). 조정:
                    #   denied 호출만 제거·허용 호출 유지. 전부 denied면 원본 유지(구 거동). 게이트
                    #   판정 자체는 불변(denied는 어차피 실행 단계서도 deny)·부작용-0 원칙 유지·도메인 리터럴 0.
                    _denset = {id(x[0]) for x in _den}
                    _keep = [tc for tc in (getattr(_am2, "tool_calls", None) or [])
                             if id(tc) not in _denset]
                    if not _keep:
                        print("[%s] rejected: regen introduced gate-denied call; keeping original"
                              % tag.upper(), file=_sys.stderr, flush=True)
                        return None
                    _am2.tool_calls = _keep
                    print("[%s] partial-accept: dropped %d gate-denied, kept %d call(s)"
                          % (tag.upper(), len(_den), len(_keep)), file=_sys.stderr, flush=True)
            # ★§2bi (rall6 실측·UNLOCK_NAME 0발화 원인): bare-name unlock이 태어나는 곳이 바로 이
            #   resign-경로 regen인데, 반환 am은 while-루프의 un_fb 검사를 **우회**해 그대로 커밋됐다
            #   (chain 18발화·un_fb 0·bare 3회 커밋 = rall6 정합). 여기서 name-check 교정 1회 수행.
            _ns = (a2 or {}).get("discoverable_name_check") or {}
            if _ns and os.environ.get("T2_UNLOCK_NAME") == "1":
                for _c2 in (getattr(_am2, "tool_calls", None) or []):
                    _ua = (_ns.get("tools") or {}).get(getattr(_c2, "name", None))
                    _uv = str(_args_dict(_c2).get(_ua) or "") if _ua else ""
                    _upat2 = _ns.get("pattern") or "_[0-9]+$"
                    _fb2 = None
                    if _uv and not re.search(_upat2, _uv):
                        self._t2_unlockname_deny = getattr(self, "_t2_unlockname_deny", 0) + 1
                        print("[T2_UNLOCK_NAME] deny bare name (followup-regen) tool=%s val=%s"
                              % (getattr(_c2, "name", None), _uv), file=_sys.stderr, flush=True)
                        _fb2 = str(_ns.get("feedback") or "Error: '{name}' needs its suffix.")
                    # ★T2_UNLOCK_PROV (2026-07-22 §2bt·rall11 050 실측): regen-경로 접미사-환각 차단.
                    #   §2bi가 bare-name만 이식해, regen이 낸 fabricated 접미사(_8374)는 무검사 커밋
                    #   → env "Unknown tool" 3연발. 메인 PROV와 동형 술어: suffixed 값이 대화 실측
                    #   근거(role=tool∪user)에 부재 or env-거부 이력(_t2_unknown_bl)이면 deny+재생성.
                    #   KB서 발견한 진짜 이름은 검색결과에 실재→통과. 엔진=부분문자열 실재확인(리터럴 0).
                    elif (_uv and os.environ.get("T2_UNLOCK_PROV") == "1"):
                        _ctx2 = " ".join(
                            str(getattr(_m3, "content", "") or "") for _m3 in state.messages
                            if getattr(_m3, "role", None) in ("tool", "user")).lower()
                        if (_uv in getattr(self, "_t2_unknown_bl", set())
                                or _uv.lower() not in _ctx2):
                            print("[T2_UNLOCK_PROV] deny unprovenanced name (followup-regen) "
                                  "tool=%s val=%s" % (getattr(_c2, "name", None), _uv),
                                  file=_sys.stderr, flush=True)
                            _fb2 = ("Error: '{name}' does not appear in any knowledge-base result "
                                    "or tool output in this conversation - you may be inventing "
                                    "the numeric suffix, and a guessed name will be rejected. Do "
                                    "NOT guess suffixes. Search the knowledge base with plain "
                                    "words describing the step (for '{name}' e.g. \"{name_words}\") "
                                    "to find the real full suffixed name, then retry with it.")
                    if _fb2:
                        _fb2 = _fb2.replace("{name}", _uv)\
                            .replace("{name_words}",
                                     re.sub(_upat2, "", _uv).replace("_", " ").strip())
                        _fb2m = ToolMessage(id=_c2.id, role="tool", requestor="assistant",
                                            error=True, content=_fb2)
                        _am3 = _gen(self, work + [am, _fb, _am2, _fb2m], bw(),
                                    "agent_response_" + tag + "_namefix", tool_choice="required")
                        if not (gate is not None and _denied_calls(_am3, gate, last_user,
                                                                   transfer_sent)):
                            _am2 = _am3
                        break
            _regen_budget_spend(self)
            return _am2

        # (a0) follow-up required — **완료 날조(fabricated completion) 차단** (2026-07-16 §14.3).
        #   실측: 실패 4/10 sim 전부 = 에이전트가 "제출됐다" 주장(가짜 케이스번호 발급)하고 후속 도구를
        #   안 부름 → 사용자 제출 0/4. 그중 3/4은 follow_up 도구 호출 0 = **구조 이벤트만으로 탐지**.
        #   엔진이 보는 것: {호출된 도구 이름} 집합뿐(텍스트 파싱 0·[[03b]]). 어느 도구에 어느 후속이
        #   붙는지·피드백 문구 = A2(`scaffold_get_tools[].follow_up`). 상한 1/sim·regen 채택 전 게이트 재검사.
        #   ★임계=사임 2회째(기본·env 조정가능): 오프라인 replay 실측 — 1회째는 pass 6/6 전부에 발화
        #   (pass 궤적도 결과 제시→확인 사임이 한 번 있음)·2회째는 실패 4/4 커버 + pass 2/6만 접촉.
        # ★T2_FOLLOWUP_CAP (2026-07-20 §2au·e2e10 050/052 실측): 고정 1/sim은 chain 1회 발화 후 소진 —
        #   4체크 중 2개만 진행돼도 레버 끝(CLAIMPROV cap 전소와 동일 패턴). 기본 1=거동보존·재런 3.
        # ★진행-감응 cap 환급 T2_FOLLOWUP_PROGRESS_REFUND (2026-07-22 §2bs·rall10 043/050/052 실측):
        #   chain cap3 < 사슬 6단계 — 발화가 견인한 턴은 매 발화 ~1쌍 전진했는데 "일하는 발화"까지
        #   cap을 소모해 소진 후 잔여 사슬 무방비(043 TRANSFER-포기·050/052 wrap-up). 직전 chain
        #   발화의 {missing} 중 하나라도 이후 **시도-수준** 호출(§2bh: 성공 불문)이 생기면 그 발화의
        #   cap 소모를 1회 환급 — 무진행 발화만 예산 소모(over-fire 억제는 기존과 동일).
        #   엔진=집합 교집합만(리터럴 0·[[05]]). 기본 OFF(거동보존).
        if os.environ.get("T2_FOLLOWUP_PROGRESS_REFUND") == "1":
            _cmv = getattr(self, "_t2_chain_missing", None)
            if _cmv:
                _effn = {_eff_tool_name(tc) for m in state.messages
                         for tc in (getattr(m, "tool_calls", None) or [])}
                _effn |= {_eff_tool_name(tc)
                          for tc in (getattr(am, "tool_calls", None) or [])}
                _hitp = _cmv & _effn
                if _hitp:
                    self._t2_followup = max(0, getattr(self, "_t2_followup", 0) - 1)
                    print("[T2_FOLLOWUP] chain progress refund hit=%s"
                          % (sorted(_hitp),), file=_sys.stderr, flush=True)
                self._t2_chain_missing = None
        # ★T2_ACTION_PROGRESS_REFUND (2026-07-22 §2bt·rall10 097 실측): action-required deny의
        #   진행-감응 환급 — 1b(FOLLOWUP)와 동일 원리. 097: cap3이 초반 compute-flail에 소진돼
        #   종반 say-loop([122-134] 8연속 무-도구 산문·write 0)가 무방비. 발화 시 기록한 target에
        #   시도-수준 착수(§2bh)가 보이면 cap 1회 환급 — 무진행 발화만 예산 소모.
        #   엔진=집합 교집합만(리터럴 0·[[05]]). 기본 OFF(거동보존).
        if os.environ.get("T2_ACTION_PROGRESS_REFUND") == "1":
            _atv = getattr(self, "_t2_action_target", None)
            if _atv:
                _effn2 = {_eff_tool_name(tc) for m in state.messages
                          for tc in (getattr(m, "tool_calls", None) or [])}
                _effn2 |= {_eff_tool_name(tc)
                           for tc in (getattr(am, "tool_calls", None) or [])}
                _hita = set(_atv) & _effn2
                if _hita:
                    self._t2_action_deny = max(0, getattr(self, "_t2_action_deny", 0) - 1)
                    print("[T2_RESOLVE] action progress refund hit=%s"
                          % (sorted(_hita),), file=_sys.stderr, flush=True)
                self._t2_action_target = None
        # ★★C207/A2·A4 (2026-07-27·`RUNAWAY_CONVERSION_DESIGN` §8-1/§8-2·day4b 실측):
        #   **모든 의미-게이트보다 먼저** 돈다 — 폭주로 오염된 응답을 뒤 게이트가 판정하면 전부 오판이고
        #   (사임-창으로 분류돼 CLAIMPROV formalize 서브콜까지 낭비), 그 blob이 커밋되면 3~5턴 만에 창이 죽는다.
        #   근거(006 m4 재파싱): 닫힌 tool_call 블록 7/7 **JSON 유효**·깨진 곳은 미종결 8번째뿐 ⇒ 형식 위반이
        #   아니라 **정지 실패**. vLLM hermes 파서가 all-or-nothing이라 유효 7개가 통째로 폐기되고 33k가 content로.
        #   엔진은 봉투 태그(서빙 포맷 상수·env) 존재와 finish_reason만 본다 — 도메인 리터럴 0([[05]]).
        _envtag = os.environ.get("T2_ENVELOPE_TAG", "<tool_call>")

        def _trunc_for_prompt(_m):
            """폭주 blob을 regen 프롬프트에 싣지 않기 위한 절단본(비커밋·리뷰 필수1)."""
            _c = str(getattr(_m, "content", None) or "")
            _lim = int(os.environ.get("T2_ENVELOPE_TRUNC", "1200") or 1200)
            if len(_c) <= _lim:
                return _m
            try:
                _cp = _m.model_copy(deep=True)
            except Exception:
                try:
                    _cp = _m.copy(deep=True)
                except Exception:
                    return _m
            _cp.content = _c[:_lim] + "\n…[truncated: the reply degenerated into repeated output]"
            return _cp

        if (os.environ.get("T2_ENVELOPE_GUARD") == "1"
                and not getattr(am, "tool_calls", None)
                and _envtag in str(getattr(am, "content", None) or "")
                and getattr(self, "_t2_envguard", 0)
                < int(os.environ.get("T2_ENVELOPE_CAP", "2") or 2)):
            self._t2_envguard = getattr(self, "_t2_envguard", 0) + 1
            print("[T2_ENVGUARD] tool-call envelope unparsed (len=%d) — required-channel regen"
                  % len(str(getattr(am, "content", None) or "")), file=_sys.stderr, flush=True)
            _newE = _ap_regen(
                "Error: [TOOL-CALL ENVELOPE] your previous reply contained tool-call markup that "
                "could not be parsed, so NO tool was executed and the customer saw nothing useful. "
                "Do not repeat the same call over and over. Issue ONE tool call now, as a real tool "
                "call, and stop.", "envguard", tool_choice="required",
                am_override=_trunc_for_prompt(am))
            if _newE is not None:
                am = _newE
                print("[T2_ENVGUARD] regen tool_calls=%s"
                      % ([getattr(t, "name", None) for t in (getattr(am, "tool_calls", None) or [])],),
                      file=_sys.stderr, flush=True)
        # A4: 길이 상한 절단 응답은 **커밋하지 않는다**(cap 1 — 재생성도 잘리면 통과·무한 regen 금지).
        if (os.environ.get("T2_TRUNC_GUARD") == "1"
                and not getattr(self, "_t2_truncguard", 0)):
            _fr = None
            try:
                _fr = ((getattr(am, "raw_data", None) or {}).get("choices") or [{}])[0].get("finish_reason")
            except Exception:
                _fr = None
            if _fr == "length":
                self._t2_truncguard = 1
                print("[T2_TRUNCGUARD] finish_reason=length — regen (cap 1)",
                      file=_sys.stderr, flush=True)
                _newT = _ap_regen(
                    "Error: [TRUNCATED] your previous reply hit the length limit and was cut off — "
                    "it was almost certainly repeating itself. Answer again BRIEFLY, without "
                    "repetition; if an action is needed, make the tool call instead of describing it.",
                    "truncguard", am_override=_trunc_for_prompt(am))
                if _newT is not None:
                    am = _newT
        # ★A2/A4가 am을 교체했을 수 있다 — 뒤 게이트의 사임-판정을 **재계산**(오판 방지).
        if os.environ.get("T2_ENVELOPE_GUARD") == "1" or os.environ.get("T2_TRUNC_GUARD") == "1":
            _resign = (not getattr(am, "tool_calls", None)
                       and isinstance(getattr(am, "content", None), str) and am.content.strip())
        _fu_cap = int(os.environ.get("T2_FOLLOWUP_CAP", "1") or 1)
        if (os.environ.get("T2_FOLLOWUP_REQUIRED") == "1" and _resign
                and getattr(self, "_t2_followup", 0) < _fu_cap):
            _called0 = _called_tools(state.messages)
            for _d0 in ((a2 or {}).get("scaffold_get_tools") or []):
                _fu = _d0.get("follow_up") or {}
                _ft = _fu.get("tool")
                if (_ft and _d0.get("name") in _called0 and _ft not in _called0
                        and _fu.get("feedback")):
                    _th = int(os.environ.get("T2_FOLLOWUP_RESIGN_TH", "2") or 2)
                    self._t2_fu_resigns = getattr(self, "_t2_fu_resigns", 0) + 1
                    if self._t2_fu_resigns < _th:
                        break
                    self._t2_followup = getattr(self, "_t2_followup", 0) + 1
                    print("[T2_FOLLOWUP] fired tool=%s missing_follow_up=%s"
                          % (_d0.get("name"), _ft), file=_sys.stderr, flush=True)
                    # ★레버 A(T2_FOLLOWUP_FORCE=1·기본 OFF): FOLLOWUP regen의 **빈손 43~50%**(regen이 도구
                    #   대신 산문을 냄·nt=20 실측 dreq2 6/14·ctl2 7/14)를 채널 강제로 닫는다. 이 순간은
                    #   ASK가 정당한 출구가 아니다 — 데이터는 이미 손에 있고 남은 일이 인계뿐.
                    _new0 = _ap_regen(_fu["feedback"], "followup",
                                      tool_choice=("required"
                                                   if os.environ.get("T2_FOLLOWUP_FORCE") == "1" else None))
                    if _new0 is not None:
                        am = _new0
                        print("[T2_FOLLOWUP] regen tool_calls=%s"
                              % ([getattr(t, "name", None) for t in (getattr(am, "tool_calls", None) or [])],),
                              file=_sys.stderr, flush=True)
                    break
            # ★follow_up_chains (2026-07-20 Q1 coverage·050형 "submit 후 절차 미완 만족종료"):
            #   scaffold follow_up의 **디스패처 확장** — after/requires를 effective 도구명(_eff_tool_name·
            #   call_ unwrap·suffix strip)으로 대조. A2 선언(도구쌍·문구)·엔진=집합 대조만(리터럴 0).
            #   같은 사임-임계·1/sim cap 공유.
            # ★관문2 확장(2026-07-20·§2aa): requires = 문자열 or **리스트(full required-set)**.
            #   050 실증 = 단일 requires(history)는 이미 호출됐고 **pending을 건너뜀** → 단일 대조는 못 잡음.
            #   전량 대조 + 누락 도구 **전량 나열**(`{missing}` 치환·050 follow-through+054 query-gap 동시 커버).
            #   ＋종단결정 nudge: requires 전부 충족·사임·`decision_tools` 미호출이면 `decision_feedback` 1회
            #   (approve **강제 아님** — decline-정답 케이스(052)가 있어 문구가 양방향 명시·Δspurious 계측 대상).
            # ★C207/B1: cap 소진 후에도 A2 `reserve` 선언 체인은 **진성 사임-턴** 1회 보장(_fu_window).
            _fu_res_declared = any(_fc.get("reserve")
                                   for _fc in ((a2 or {}).get("follow_up_chains") or []))
            _fu_genuine = not getattr(self, "_t2_fu_readloop_turn", False)
            _fu_mode = _fu_window(getattr(self, "_t2_followup", 0), _fu_cap,
                                  _fu_res_declared, getattr(self, "_t2_fu_reserve", 0),
                                  _fu_genuine)
            if _fu_mode is not None:
                _eff0 = {_eff_tool_name(tc) for m in state.messages
                         for tc in (getattr(m, "tool_calls", None) or [])}
                for _fc in ((a2 or {}).get("follow_up_chains") or []):
                    if _fu_mode == "reserve" and not _fc.get("reserve"):
                        continue              # 예비-창은 선언 체인에만
                    _hit1 = _chain_dispatch(_fc, _eff0)     # (feedback, tag) or None — 순수함수(단위테스트 공유)
                    if _hit1 is None:
                        continue
                    _fb1, _tag1 = _hit1
                    # ★C201/D2(2026-07-26·리뷰 결함2 실측): 전역 임계(기본 2)는 **1회 사임 뒤 종료되는
                    #   궤적**(035: 에스컬 호출→notice 1턴→유저 terminal)에서 구조적 미발화. 체인별
                    #   `resign_th`로 override(미선언=env 기본=거동 보존). 전역을 낮추면 기존 체인까지
                    #   조기 발화해 Δspurious가 스택 전체로 번지므로 per-chain으로 국소화.
                    _th = int(os.environ.get("T2_FOLLOWUP_RESIGN_TH", "2") or 2)
                    if _fc.get("resign_th") is not None:
                        _th = int(_fc["resign_th"])
                    self._t2_fu_resigns = getattr(self, "_t2_fu_resigns", 0) + 1
                    if self._t2_fu_resigns < _th:
                        break
                    self._t2_followup = getattr(self, "_t2_followup", 0) + 1
                    print("[T2_FOLLOWUP] chain fired(%s) after=%s"
                          % (_tag1, _fc.get("after")), file=_sys.stderr, flush=True)
                    # ★진행-감응 환급용 스냅샷(2026-07-22 §2bs): 발화 시점의 미이행 집합 기록.
                    #   다음 평가에서 이 중 하나라도 시도-수준 호출이 보이면 cap 소모 1회 환급.
                    if _tag1 == "followup_decision":
                        self._t2_chain_missing = {d for d in (_fc.get("decision_tools") or [])
                                                  if d not in _eff0}
                    else:
                        _rq1 = _fc.get("requires")
                        _rq1 = _rq1 if isinstance(_rq1, list) else [_rq1]
                        self._t2_chain_missing = {r for r in _rq1 if r not in _eff0}
                    # ★§2bh: 구판은 followup_chain만 required — decision regen(종단결정 nudge)은
                    #   강제 없이 빈손 가능(rall5 실측: followup_decision 3회 발화·미이행). 문구가
                    #   양방향(approve/deny)이라 방향-중립·도구-호출 강제는 안전.
                    _new1 = _ap_regen(_fb1, _tag1,
                                      tool_choice=("required"
                                                   if os.environ.get("T2_FOLLOWUP_FORCE") == "1"
                                                   and _tag1 in ("followup_chain", "followup_decision")
                                                   else None))
                    if _new1 is not None:
                        am = _new1
                    break
        # (a1) write-provenance — **완료-주장 게이트**(③형·2026-07-17 사용자 제안: "출력도 출처를 밝혀라").
        #   C45(입력 출처선언)의 출력측 쌍대: 완료를 주장하려면 근거 이벤트가 원장에 있어야 한다.
        #   ③형의 급소 = pass("당신이 실행하라")와 fail("내가 제출했다")이 **구조 동일·말만 다름** →
        #   LLM이 자기 답변의 완료-주장을 **이진 선언**(formalize·[[10]])하고, 엔진은 선언 구조체 +
        #   결정론 원장만 검증. 답변 텍스트 파싱 0([[03b]]).
        # ★2026-07-18 일반화(`A2_DOMAIN_GENERALIZATION_DESIGN §2.2`): 트리거를 **task-불변**으로.
        #   ~~구판: follow_up 도구가 호출됨~~ = task_019 리터럴(give-missing 5건은 give를 *안* 불러서 못 잡음).
        #   신판: **완료 주장 ∧ 원장에 실효 write 실행 0 ∧ 사임**. producer/follow_up 개념 불요.
        #   `_any_effective_write`가 user·agent 실행 양쪽을 본다 → agent-side 태스크(69/97)서 에이전트가
        #   WRITE 실행했으면 발화 안 함(오탐 방지·F2). give 태스크(21/97)서 사용자 실행 0이면 주장은 거짓.
        #   completion_guard 문구 = A2 도메인-수준(`a2['completion_guard']`) 우선, 없으면 follow_up 사본(하위호환).
        _cgd = (a2 or {}).get("completion_guard") or {}
        if not _cgd:
            for _d1 in ((a2 or {}).get("scaffold_get_tools") or []):
                _c0 = (_d1.get("follow_up") or {}).get("completion_guard") or {}
                if _c0:
                    _cgd = _c0
                    break
        if (os.environ.get("T2_WRITE_PROV") == "1" and _resign
                and not getattr(self, "_t2_writeprov", 0)
                and _cgd.get("claim_question") and _cgd.get("feedback")):
            for _once in (True,):                 # 구조 유지용 단일 루프(break 재사용)
                _cg = _cgd
                if _any_effective_write(state.messages):
                    break                          # 세상을 바꾼 실행이 원장에 있음 → 완료 주장 정당(오탐 방지)
                _claims = None
                try:
                    try:
                        _dm1 = _gen(self, work + [am, UserMessage(role="user", content=_cg["claim_question"])],
                                    bw(), "agent_writeprov")
                    except TypeError:
                        _dm1 = _gen(self, work + [am, UserMessage(content=_cg["claim_question"])],
                                    bw(), "agent_writeprov")
                    for _ln in (getattr(_dm1, "content", None) or "").splitlines():
                        _ln = _ln.strip().rstrip(",")
                        if _ln.startswith("{") and _ln.endswith("}"):
                            try:
                                _j = json.loads(_ln)
                                if isinstance(_j, dict) and "claims_completion" in _j:
                                    _claims = bool(_j["claims_completion"])
                            except Exception:
                                pass
                except Exception as _we:
                    print("[T2_WRITEPROV] declaration failed (no-op): %r" % (_we,),
                          file=_sys.stderr, flush=True)
                print("[T2_WRITEPROV] window hit (no effective write in ledger) declared_completion=%s"
                      % (_claims,), file=_sys.stderr, flush=True)
                if _claims:
                    self._t2_writeprov = getattr(self, "_t2_writeprov", 0) + 1
                    _new1 = _ap_regen(_cg["feedback"], "writeprov")
                    if _new1 is not None:
                        am = _new1
                        print("[T2_WRITEPROV] regen tool_calls=%s"
                              % ([getattr(t, "name", None) for t in (getattr(am, "tool_calls", None) or [])],),
                              file=_sys.stderr, flush=True)
                break
        # ★C193 notice-재발화 억제 (2026-07-26·야간 97-런 회귀 3/3 실측: 005·032·035 — 어제 pass가
        #   전부 이 기전으로 fail. +004·016 동형). 도메인 정책(notice 게이트)은 [notice 송신 →
        #   손님 동의 → transfer 도구 호출] 순서를 요구하는데, 에이전트가 **동의 확보 후에도
        #   notice를 재발화**하며 턴을 넘기면 user-sim이 터미널 신호로 응답해 sim이 그 자리에서
        #   종료 → gold transfer 도구 영영 미호출(레이스). 실측: 005 msg19 손님 "Yes, please" 평문
        #   동의 → msg20 재발화 → 터미널 / 035는 도구 출력이 즉시-transfer를 지시했는데도 재발화.
        #   엔진이 보는 것 = A2 notice_text 문자열의 {현재 발화 포함 ∧ 과거 송신 실재 ∧ transfer
        #   호출 부재} 결정론 대조뿐. **도구명도 A2에서**: notice 게이트 tool 키 ∨
        #   claim_prov.event_map['transfer'] 패턴([[05]] 엔진 리터럴 0). 교정=재발화 대신 도구
        #   호출을 지시하는 regen 1회(cap 1/sim·호출·인자·계속여부=모델·[[10]]).
        if (os.environ.get("T2_NOTICE_REPEAT", "1") == "1"
                and not getattr(self, "_t2_noticerep", 0)):
            _ngate = next((g for g in ((a2 or {}).get("gates") or [])
                           if g.get("kind") == "notice" and g.get("notice_text")), None)
            _ntxt = (_ngate or {}).get("notice_text") or ""
            _emap4 = ((a2 or {}).get("claim_prov") or {}).get("event_map") or {}
            _trname = ((_ngate or {}).get("tool")
                       or (_emap4.get("transfer") if isinstance(_emap4.get("transfer"), str) else None)
                       or "the transfer tool named in your policy")
            _amc4 = str(getattr(am, "content", None) or "")
            _nkey = _ntxt[:40] if _ntxt else ""
            if _nkey and _nkey in _amc4:
                _sent_before = any(
                    _nkey in str(getattr(_m4, "content", None) or "")
                    for _m4 in state.messages
                    if getattr(_m4, "role", None) == "assistant")
                if _sent_before and not _is_transfer_call(am, _emap4):
                    self._t2_noticerep = 1
                    print("[T2_NOTICEREP] repeated notice without transfer call — regen",
                          file=_sys.stderr, flush=True)
                    _new4 = _ap_regen(
                        "You already sent that exact transfer notice earlier in this conversation, "
                        "and the customer has already responded to it. Do NOT send the notice again "
                        "— repeating the question only stalls the request. If the transfer should "
                        "proceed, CALL %s NOW (as a tool call, with an appropriate summary); "
                        "otherwise continue helping the customer without re-sending the notice."
                        % _trname, "noticerep")
                    if _new4 is not None:
                        am = _new4
                        print("[T2_NOTICEREP] regen tool_calls=%s"
                              % ([getattr(t, "name", None) for t in (getattr(am, "tool_calls", None) or [])],),
                              file=_sys.stderr, flush=True)
        # (a1b) ★T2_CLAIM_PROV (2026-07-20·사용자: "모든 '했다' 주장을 원장대조") — WRITEPROV의 일반형.
        #   완료-주장(write축)만 묻던 이진 선언을 **모든 과거-행동 주장 목록 formalize**로 확장:
        #   LLM이 자기 답변의 "이미 했다" 주장들을 {kind, what}로 선언([[10]] formalize) → 엔진은
        #   A2 `claim_prov.event_map`(kind→도구 접두 패턴 or "__effective_write__")으로 원장 이벤트 실재만
        #   대조(집합 교차·텍스트 파싱 0·[[03b]]). 미등재 kind=skip(오탐 방지). 043 확인-날조(KB 0회·
        #   "checked" 주장) 직접 표적·완료-주장은 kind=write로 흡수(WRITEPROV 상위호환·병행 시 중복 주의).
        #   기본 OFF(T2_CLAIM_PROV=1)·사임-윈도우·1/sim.
        _cpv = (a2 or {}).get("claim_prov") or {}
        # ★관문5(2026-07-20·038 transfer-escape·§2ad): 발화창 = 사임 ∨ **transfer-류 호출**.
        #   038 실측: "I will file 3 disputes..."(SAY)→TRANSFER NOTICE로 탈출(정당 도구호출이라
        #   FORCE_ACTION 사각·미래형이라 구판 CLAIM 사각). transfer 패턴=A2 event_map['transfer'] 재사용.
        _cpv_transfer = _is_transfer_call(am, _cpv.get("event_map") or {})
        # ★cap env화(2026-07-20·smoke023d 포렌식·§2am): 1/sim 고정은 빈손 regen(tool_calls=[]·기지의
        #   43~50%) 1회에 레버 전소 → 이후 완료날조·transfer-escape 무방비(실측: 첫 발화 빈손→msg21
        #   "이미 logged" 날조·msg37 transfer 탈출 모두 무검사). 기본 1=거동보존·스모크 3.
        _cpv_cap = int(os.environ.get("T2_CLAIMPROV_CAP", "1") or 1)
        # ★transfer-창 별도 예산(2026-07-20·e2e9 038 포렌식·§2ao): resign-창 발화가 cap을 소진하면
        #   **탈출 직전**(transfer 호출) 최후 감사가 무산(038 실측: 16 hit 전부 resign·transfer-창 0).
        #   transfer-창은 cap과 독립적으로 sim당 1회 보장(사임-창과 상호배타: transfer=tool_call 있음).
        # ★C201/D3(2026-07-26·§7-0): cap 소진 뒤에도 **행동-kind 주장 전용 예비 1회**(A2 reserve_kinds).
        #   근거 실측: unbacked>0인데 regen 무발생 A11·B5 — 초반 저가치 발화가 예산을 태우고 종단
        #   완료-날조(032 transfer·026/028 record_update)가 무검사 통과. 판정=순수함수 _cpv_window.
        _cpv_rsv_kinds = _cpv.get("reserve_kinds") or []
        _cpv_mode = _cpv_window(bool(_resign), bool(_cpv_transfer),
                                getattr(self, "_t2_claimprov", 0), _cpv_cap,
                                getattr(self, "_t2_claimprov_tr", 0),
                                getattr(self, "_t2_claimprov_rsv", 0), bool(_cpv_rsv_kinds))
        _cpv_win_ok = _cpv_mode is not None
        if (os.environ.get("T2_CLAIM_PROV") == "1" and (_resign or _cpv_transfer)
                and _cpv_win_ok
                and _cpv.get("question") and _cpv.get("feedback") and _cpv.get("event_map")):
            for _once in (True,):
                _cl, _pd = None, None
                try:
                    try:
                        _dm2 = _gen(self, work + [am, UserMessage(role="user", content=_cpv["question"])],
                                    bw(), "agent_claimprov")
                    except TypeError:
                        _dm2 = _gen(self, work + [am, UserMessage(content=_cpv["question"])],
                                    bw(), "agent_claimprov")
                    _txt2 = getattr(_dm2, "content", None) or ""
                    _mj = re.search(r"\{.*\}", _txt2, re.S)
                    if _mj:
                        _j2 = json.loads(_mj.group(0))
                        if isinstance(_j2, dict) and isinstance(_j2.get("claims"), list):
                            _cl = _j2["claims"]
                        # ★관문5 미래형: pending = 대화 전체서 "하겠다" 약속·미이행 목록(A2 question v2가 요구).
                        if isinstance(_j2, dict) and isinstance(_j2.get("pending"), list):
                            _pd = _j2["pending"]
                except Exception as _ce2:
                    print("[T2_CLAIMPROV] declaration failed (no-op): %r" % (_ce2,),
                          file=_sys.stderr, flush=True)
                if not _cl and not _pd:
                    print("[T2_CLAIMPROV] window hit claims=%s pending=%s" % (_cl, _pd),
                          file=_sys.stderr, flush=True)
                    break
                # 원장 이벤트 집합: 원명 + effective명(디스패처 unwrap·suffix strip)
                _evs = set()
                for _m3 in state.messages:
                    for _tc3 in (getattr(_m3, "tool_calls", None) or []):
                        _evs.add(str(getattr(_tc3, "name", "") or ""))
                        _evs.add(_eff_tool_name(_tc3))
                _emap = _cpv["event_map"]
                _unbacked = _claim_unbacked(_cl, _emap, _evs, state.messages)
                # 미래-약속: 같은 원장대조 — 이 창(사임/transfer)에서 미이행 약속 = 영영 미이행(탈출티켓).
                #   feedback_pending 미선언(구판 A2)이면 발화 0(거동보존).
                _unb_p = (_claim_unbacked(_pd, _emap, _evs, state.messages)
                          if _cpv.get("feedback_pending") else [])
                # ★C207/C2-a(2026-07-27): **미보유 기능 약속** — 약속에 실린 도구가 쓸 수 있는 도구
                #   집합에 아예 없으면(=OTP/SMS 발송처럼 존재하지 않는 기능) 원장 대조 이전에 불가능한
                #   약속이다(004·035 실측: 없는 OTP를 여러 턴 반복 약속하며 창 소진). 엔진=집합 대조만·
                #   discoverable(잠금·접미사)까지 포함해 오탐 0(리뷰 필수3).
                _unavail = []
                if os.environ.get("T2_UNAVAIL_PROMISE") == "1" and _cpv.get("feedback_unavailable"):
                    try:
                        # ★P7(C208⑥·DAY5_PRESCRIPTIONS §P7): 구판 `getattr(orch, ...)`는 이 스코프에
                        #   `orch`가 없어 **전량 NameError**(day5 0/223 무음 스킵). env는 init_inject가
                        #   심어둔 `_t2_orch`(orchestrator)에서 해석한다.
                        _known = _known_tool_names(getattr(self, "tools", None),
                                                   getattr(getattr(self, "_t2_orch", None),
                                                           "environment", None),
                                                   state.messages)
                        _unavail = _unavailable_promises(_pd, _known)
                        _lever_health("unavail", "ok")
                        if _unavail:
                            _lever_health("unavail", "fired")
                            print("[T2_UNAVAIL] promised tools not available: %s"
                                  % [p.get("tool") for p in _unavail][:3],
                                  file=_sys.stderr, flush=True)
                    except Exception as _ue:
                        _lever_health("unavail", "skipped")
                        print("[T2_UNAVAIL] skipped (no-op): %r" % (_ue,),
                              file=_sys.stderr, flush=True)
                print("[T2_CLAIMPROV] window hit(%s) claims=%d unbacked=%d pending=%d unb_p=%d %s"
                      % ("transfer" if _cpv_transfer and not _resign else "resign",
                         len(_cl or []), len(_unbacked), len(_pd or []), len(_unb_p),
                         [c.get("kind") for c in (_unbacked + _unb_p)][:4]), file=_sys.stderr, flush=True)
                # ★C201/D3: 예비-창은 **행동-kind 주장**에만 쓴다(저가치 주장으로 예비 소진 방지).
                if _cpv_mode == "reserve" and not (_claim_has_kind(_unbacked, _cpv_rsv_kinds)
                                                   or _claim_has_kind(_unb_p, _cpv_rsv_kinds)):
                    print("[T2_CLAIMPROV] reserve window: no action-kind claim — skip",
                          file=_sys.stderr, flush=True)
                    break
                if _unbacked or _unb_p:
                    self._t2_claimprov = getattr(self, "_t2_claimprov", 0) + 1
                    if _cpv_mode == "reserve":
                        self._t2_claimprov_rsv = 1       # 예비-창(1/sim) 소진 마킹
                    if _cpv_transfer and not _resign:
                        self._t2_claimprov_tr = 1        # transfer-창 예산(1/sim) 소진 마킹

                    def _desc3(cc):
                        return "; ".join("%s: %s" % (c.get("kind"), str(c.get("what"))[:60])
                                         for c in cc[:3])
                    _parts = []
                    if _unbacked:
                        _parts.append(_cpv["feedback"].replace("{claims}", _desc3(_unbacked)))
                    if _unb_p:
                        _parts.append(_cpv["feedback_pending"].replace("{claims}", _desc3(_unb_p)))
                    if _unavail:                       # C207/C2-a
                        _parts.append(_cpv["feedback_unavailable"].replace(
                            "{claims}", "; ".join("%s (tool: %s)" % (str(p.get("what"))[:50], p.get("tool"))
                                                  for p in _unavail[:3])))
                    _new2 = _ap_regen("\n".join(_parts), "claimprov")
                    if _new2 is not None:
                        am = _new2
                        print("[T2_CLAIMPROV] regen tool_calls=%s"
                              % ([getattr(t, "name", None) for t in (getattr(am, "tool_calls", None) or [])],),
                              file=_sys.stderr, flush=True)
                break
        # (a) discovery-required — 엔진이 보는 것: {호출된 도구 이름}뿐.
        if (os.environ.get("T2_DISCOVERY_REQUIRED") == "1" and (a2 or {}).get("analysis_producers")
                and _resign and not getattr(self, "_t2_discreq", 0)):
            _called = _called_tools(state.messages)
            for _sp in (a2.get("analysis_producers") or []):
                _ds, _pr = _sp.get("data_source"), _sp.get("producer")
                if _ds in _called and _pr and _pr not in _called:
                    self._t2_discreq = getattr(self, "_t2_discreq", 0) + 1
                    print("[T2_DISCREQ] fired data_source=%s producer=%s" % (_ds, _pr),
                          file=_sys.stderr, flush=True)
                    _new = _ap_regen(DISCREQ_FEEDBACK.format(
                        data_source=_ds, producer=_pr, subject=_sp.get("subject") or "this"), "discreq")
                    if _new is not None:
                        am = _new
                        print("[T2_DISCREQ] regen tool_calls=%s"
                              % ([getattr(t, "name", None) for t in (getattr(am, "tool_calls", None) or [])],),
                              file=_sys.stderr, flush=True)
                    break
        # (b) self-declaration — 엔진이 보는 것: LLM 선언 JSON 필드뿐([[10]]). 선언 sub-call은 비커밋.
        elif (os.environ.get("T2_SELF_DECLARATION") == "1" and (a2 or {}).get("assertion_operands")
                and _resign and not getattr(self, "_t2_selfdecl", 0)):
            _ao = a2.get("assertion_operands") or {}
            try:
                _dp = SELFDECL_PROMPT.format(items=", ".join(sorted(_ao)))
                try:
                    _dm = _gen(self, work + [am, UserMessage(role="user", content=_dp)], bw(),
                               "agent_selfdecl")
                except TypeError:
                    _dm = _gen(self, work + [am, UserMessage(content=_dp)], bw(), "agent_selfdecl")
                _decls = _parse_declaration(getattr(_dm, "content", None))
            except Exception as _de:
                print("[T2_SELFDECL] declaration failed (no-op): %r" % (_de,), file=_sys.stderr, flush=True)
                _decls = []
            print("[T2_SELFDECL] declared=%s" % (_decls or "(none — no-op)",), file=_sys.stderr, flush=True)
            _called = _called_tools(state.messages)
            for _d in _decls:
                _op = str(_d.get("operand", "")).strip()
                if not _d.get("claimed") or str(_d.get("source", "")).upper() != "INFER":
                    continue
                _pr = _ao.get(_op)
                if _pr and _pr not in _called:
                    self._t2_selfdecl = getattr(self, "_t2_selfdecl", 0) + 1
                    print("[T2_SELFDECL] fired operand=%s producer=%s" % (_op, _pr),
                          file=_sys.stderr, flush=True)
                    _new = _ap_regen(SELFDECL_FEEDBACK.format(operand=_op, producer=_pr), "selfdecl")
                    if _new is not None:
                        am = _new
                        print("[T2_SELFDECL] regen tool_calls=%s"
                              % ([getattr(t, "name", None) for t in (getattr(am, "tool_calls", None) or [])],),
                              file=_sys.stderr, flush=True)
                    break
        return am

    LLMAgent._generate_next_message = unified
    # (3) exec-side: auth observe + nested/calc 읽기증강 (게이트 경로와 동일·직교)
    _install_regen_exec()
    # (4) 컨텍스트 초과 우아한 종료 (023 진단·§2ah): overflow를 sim-무효 대신 scored 실패로.
    _install_overflow_guard()
    # (5) ★LLM CWE 발생원 로거 (§2bd·로그 전용): rall4 095t0 CWE가 overflow 가드(step-래핑)를
    #   우회해 러너까지 전파(4번째 우회 경로) — 어느 generate(call_name)서 새는지 특정.
    #   래핑=la.generate 모듈 함수·예외는 그대로 re-raise(행동 무변경).
    if not getattr(la, "_t2_llmdiag", False):
        _og_gen = la.generate

        def _gen_diag(*_a, **_kw):
            try:
                return _og_gen(*_a, **_kw)
            except Exception as _e:
                if "ContextWindow" in type(_e).__name__:
                    print("[T2_LLM_DIAG] CWE escaped at call_name=%s"
                          % _kw.get("call_name"), file=_sys.stderr, flush=True)
                raise

        la.generate = _gen_diag
        la._t2_llmdiag = True
    return unified


if __name__ == "__main__":
    apply()
    print("[t2_gate_patch] applied")
