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

# ── 도메인-일반 기본값 (A2가 override·enrich; retail/도메인 하드코딩 아님) ──
DEFAULT_ARG_HINTS = ("email", "name", "zip", "user_id", "order_id", "username", "id",
                     "payment", "address", "phone", "item", "reservation")
DEFAULT_PLACEHOLDERS = {
    "#W0000000", "something@example.com", "jane_doe@example.com",
    "john.doe@example.com", "johndoe@example.com", "john@example.com",
    "jane@example.com", "user@example.com", "test@example.com",
    "example@example.com", "123 Main St", "123 Main Street",
    "ABC123", "XYZ789",  # 도메인-일반 영숫자 placeholder 패턴 (구 airline A2서 이관·minimize-A2)
}
PROV_ARG_HINT = DEFAULT_ARG_HINTS          # 호환 alias
COMMON_PLACEHOLDERS = DEFAULT_PLACEHOLDERS  # 호환 alias

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
        a2["_notice_text"] = next(
            (g.get("notice_text") for g in a2["gates"] if g.get("kind") == "notice"), "")
    _A2_CACHE[domain] = a2
    return a2


def _flatten(v):
    if isinstance(v, (list, tuple)):
        for x in v:
            yield from _flatten(x)
    elif isinstance(v, dict):
        for x in v.values():
            yield from _flatten(x)
    else:
        yield v


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
        if not any(h in k.lower() for h in hints):
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
        tms = _transfer_msg_sent(self, a2["_notice_text"])

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
            out = orig(self, [tc])
            results.extend(out)
            if out and getattr(out[0], "error", False):
                _mark_fail(key, _content_str(out[0]))
            elif retry_on:
                self._t2_consec = 0  # 성공 → 연속카운트 리셋
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
            if not any(h in k.lower() for h in hints):
                continue
            for val in _flatten(v):
                s = str(val).strip()
                if len(s) < 4 or _ctx_has(s, ctx):
                    continue
                if (id(tc), k, s) in exclude:
                    continue
                return (tc, k, s)
    return None


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


def _confirm_write_tools(a2):
    """A2-도출 write 도구 집합 = kind=='confirm' 게이트의 applies_to (도메인 리터럴 0)."""
    tools = set()
    for g in (a2 or {}).get("gates", []) or []:
        if g.get("kind") == "confirm":
            tools |= set(g.get("applies_to") or [])
    return tools


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


def _t5c_disamb_subcall(self, la, UserMessage, state_msgs, tc, k, s, sub_args):
    """P-B: 격리 서브콜로 |C|>=2 선택 판정 → 서브콜 답이 원값과 다르고 화이트리스트(A2
    disamb_sub_args) 타입일 때만 인자 제자리 치환. 원턴·대화 완전 불변·모든 예외 = no-op.
    반환: 'switch'|'keep'|'unsure'|'error' (계측용)."""
    import sys
    try:
        records = _candidate_records(k, s, state_msgs)
        if len(records) < 2:
            return "unsure"
        prompt = (SUBCALL_SYS + "\n\n=== Conversation ===\n" + _text_transcript(state_msgs)
                  + "\n\n=== Candidates for '" + str(k) + "' ===\n"
                  + "\n".join("- %s%s" % (c, ("   | record: " + sn) if sn else "")
                              for c, sn in records)
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

    def _append(state, message):
        if isinstance(message, UserMessage) and getattr(message, "is_audio", False):
            raise ValueError("audio not supported")
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

    def _gen(self, work, bad_words, call_name):
        kw = dict(self.llm_args)
        if use_badwords and bad_words:
            eb = dict(kw.get("extra_body") or {})
            eb["bad_words"] = sorted(bad_words)
            kw["extra_body"] = eb
        return la.generate(model=self.llm, tools=self.tools,
                           messages=self._system_messages + work, call_name=call_name, **kw)

    def patched(self, message, state):
        if not hasattr(self, "_t2_static_bl"):
            self._t2_static_bl = _static_blacklist(self.tools, placeholders)
            self._t2_session_bl = set()
        self._system_messages = state.system_messages
        _append(state, message)
        ctx = _ctx_from_messages(state.messages)

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
                tmpl = REGEN_FEEDBACK_NEUTRAL if prov_mode == "rescue" else REGEN_FEEDBACK
                main_reason = tmpl.format(k=k, s=s)
            for c in (am.tool_calls or []):
                reason = main_reason if c is tc else \
                    "Error: [PROVENANCE] resolve the invented value first; do not call this yet."
                work.append(ToolMessage(id=c.id, role="tool", requestor="assistant",
                                        error=True, content=reason))
            am = _gen(self, work, bw(), "agent_response_regen")

        # ─── DISAMB: 문맥-실재값인데 같은-형식 후보 2+개 → 1회 재확인 (선택은 모델) ───
        if disamb_tools and getattr(am, "tool_calls", None):
            if not hasattr(self, "_t2_disamb_seen"):
                self._t2_disamb_seen = set()
            hit = None
            for tc in am.tool_calls:
                if getattr(tc, "name", None) not in disamb_tools:
                    continue
                for k, v in _args_dict(tc).items():
                    if not any(h in k.lower() for h in hints):
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


def _budget_tick(agent):
    """R1: 차단 turn마다 orchestrator.num_errors++ → too_many_errors 예산 동일(best-of-K 방지)."""
    orch = getattr(agent, "_t2_orch", None)
    if orch is not None:
        try:
            orch.num_errors = getattr(orch, "num_errors", 0) + 1
        except Exception:
            pass


def _install_regen_exec():
    """slim _execute_tool_calls: 실행 + auth observe + read-augment(present/nested/calc). deny 없음
    (denied 호출은 생성-레벨서 이미 strip). augment=reads라 replay-safe(기존 apply와 동일 로직)."""
    from tau2.orchestrator.orchestrator import BaseOrchestrator
    orig_exec = getattr(BaseOrchestrator, "_t2_orig_exec", None) or BaseOrchestrator._execute_tool_calls
    BaseOrchestrator._t2_orig_exec = orig_exec

    def exec_augment(self, tool_calls):
        results = orig_exec(self, tool_calls)
        env = getattr(self, "environment", None)
        a2 = _domain_a2(getattr(env, "domain_name", None)) if env is not None else None
        if a2 is None:
            return results
        gate = getattr(getattr(self, "agent", None), "_t2_gate", None)
        by_id = {getattr(r, "id", None): r for r in (results or [])}
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
        transfer_sent = _regen_transfer_sent(state.messages, a2["_notice_text"])
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

    def _gen(self, work, bad_words, call_name):
        kw = dict(self.llm_args)
        if use_badwords and bad_words:
            eb = dict(kw.get("extra_body") or {})
            eb["bad_words"] = sorted(bad_words)
            kw["extra_body"] = eb
        return la.generate(model=self.llm, tools=self.tools,
                           messages=self._system_messages + work, call_name=call_name, **kw)

    def unified(self, message, state):
        if not hasattr(self, "_t2_static_bl"):
            self._t2_static_bl = _static_blacklist(self.tools, placeholders)
            self._t2_session_bl = set()
        self._system_messages = state.system_messages
        _append(state, message)
        gate = getattr(self, "_t2_gate", None)
        a2 = getattr(self, "_t2_a2", None)
        last_user = transfer_sent = None
        if gate is not None:
            _rebuild_gate_state(gate, a2, state.messages)
            last_user = _regen_last_user(state.messages)
            transfer_sent = _regen_transfer_sent(state.messages, a2["_notice_text"])
        ctx = _ctx_from_messages(state.messages)

        # ★E-PLAN v1.3 (T2_EPLAN=1): committed 히스토리서 결정론 ledger 재구성(관측만·[[10]])
        #   discovery L1/L2 = read-강제 deny(§1.5 허용축)·CP5 리마인더 소비 = 생성-레벨(비커밋)
        ep_led = ep_spec = _epmod = None
        ep_writes = set()
        if os.environ.get("T2_EPLAN") == "1" and a2 is not None and a2.get("eplan"):
            try:
                import t2_eplan_patch as _epmod
                ep_spec = a2.get("eplan")
                ep_writes = _confirm_write_tools(a2)
                ep_led = _epmod.build_ledger_from_messages(state.messages, ep_spec, ep_writes)
            except Exception as _e:
                print("[T2_EPLAN] ledger build failed: %r" % (_e,), file=_sys.stderr, flush=True)
                ep_led = None

        def bw():
            return [v for v in (self._t2_static_bl | self._t2_session_bl) if v.lower() not in ctx]

        work = list(state.messages)
        _rem = getattr(self, "_t2_eplan_reminder", None)
        if _rem:  # CP5 walk 리마인더(작업버퍼만·히스토리 비커밋 = 채널 절대규칙)
            self._t2_eplan_reminder = None
            try:
                work = work + [UserMessage(role="user", content=_rem)]
            except TypeError:
                work = work + [UserMessage(content=_rem)]
        am = _gen(self, work, bw(), "agent_response")
        gate_rounds = prov_rounds = eplan_rounds = 0
        subs = 0
        rescue_skipped = set()
        rescue_excl = set()   # ★PERARG(C65): (id(tc),k,s) — rescue-스킵된 fab 제외하고 재스캔
        while True:
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
            denied = _denied_calls(am, gate, last_user, transfer_sent) if gate is not None else []
            denied_by_objid = {id(tc): (gid, why) for tc, gid, why in denied}
            do_gate = bool(denied) and gate_rounds < 1
            fab_covered = fab is not None and do_gate and id(fab[0]) in denied_by_objid
            do_prov = (fab is not None) and prov_rounds < max_prov_retries and not fab_covered
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
                    if nm in ep_writes and id(c) not in denied_by_objid:
                        try:
                            fb = _epmod.discovery_precondition(ep_led, ep_spec, nm)
                        except Exception:
                            fb = None
                        if fb:
                            ep_fb = (c, fb)
                            break
            if not do_gate and not do_prov and ep_fb is None:
                break
            main_prov = None
            if do_prov:
                prov_rounds += 1
                ptc, k, s = fab
                self._t2_session_bl.add(s)
                self._t2_regen = getattr(self, "_t2_regen", 0) + 1
                # 관측성(행동 무변경): p4-비용 귀속용 발화 로그 (C53 p4 −5.3pp·§3c)
                print("[T2_PROV] regen fired tool=%s arg=%s val=%s" % (getattr(ptc, "name", "?"), k, s),
                      file=_sys.stderr, flush=True)
                main_prov = (ptc, (REGEN_FEEDBACK_NEUTRAL if prov_mode == "rescue"
                                   else REGEN_FEEDBACK).format(k=k, s=s))
            if do_gate:
                gate_rounds += 1
                self._t2_gate_rounds = getattr(self, "_t2_gate_rounds", 0) + 1
                _budget_tick(self)  # ★게이트 라운드만 과금 (prov=무과금=C53 semantics)
            if ep_fb is not None:
                eplan_rounds += 1
                self._t2_eplan_deny = getattr(self, "_t2_eplan_deny", 0) + 1
                if self._t2_eplan_deny == _ep_cap:  # 관측 마커(sim당 1회): 이후 discovery deny 중단
                    print("[T2_EPLAN] deny cap %d reached — no further discovery denies this sim"
                          % _ep_cap, file=_sys.stderr, flush=True)
            fb = [am]
            for c in (am.tool_calls or []):
                if do_gate and id(c) in denied_by_objid:
                    gid, why = denied_by_objid[id(c)]
                    content = f"Error: [POLICY GATE {gid}] {why}"
                elif main_prov is not None and c is main_prov[0]:
                    content = main_prov[1]
                elif ep_fb is not None and c is ep_fb[0]:
                    content = "Error: " + ep_fb[1]
                else:
                    content = "Error: resolve the flagged call(s) first; do not call this tool yet."
                fb.append(ToolMessage(id=c.id, role="tool", requestor="assistant",
                                      error=True, content=content))
            work = work + fb
            am = _gen(self, work, bw(), "agent_response_unified_regen")

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
        # prov-fab 잔존 = 통과 (기존 prov semantics·id 날조는 env가 거부=C12)

        # ── DISAMB: 문맥-실재값·같은-형식 후보 2+ → 1회 재확인 (기존 로직 이식) ──
        if disamb_tools and getattr(am, "tool_calls", None):
            if not hasattr(self, "_t2_disamb_seen"):
                self._t2_disamb_seen = set()
            hit = None
            for tc in am.tool_calls:
                if getattr(tc, "name", None) not in disamb_tools:
                    continue
                for k, v in _args_dict(tc).items():
                    if not any(h in k.lower() for h in hints):
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
            if hit:
                tc, k, s, cands, memo = hit
                self._t2_disamb_seen.add(memo)
                self._t2_disamb = getattr(self, "_t2_disamb", 0) + 1
                print("[T2_DISAMB] fired tool=%s arg=%s val=%s ncand=%d" % (tc.name, k, s, len(cands)),
                      file=_sys.stderr, flush=True)
                dwork = list(work) + [am]
                fbtxt = DISAMB_FEEDBACK.format(k=k, s=s, n=len(cands),
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
        # ★T5-C P2 원리-디폴트(opt-in T2_PRINCIPLE_DEFAULT=1): write operand 기본값(원결제 등)
        #   위반 시 제자리 치환. user-발화만 override 근거(tool출력의 계정값 아님).
        if os.environ.get("T2_PRINCIPLE_DEFAULT") == "1" and gate is not None:
            uctx = " ".join(str(getattr(m, "content", "") or "").lower()
                            for m in state.messages if getattr(m, "role", None) == "user")
            nsub = _apply_principle_default(am, a2, gate, uctx)
            if nsub:
                self._t2_principle_default = getattr(self, "_t2_principle_default", 0) + nsub
        return am

    LLMAgent._generate_next_message = unified
    # (3) exec-side: auth observe + nested/calc 읽기증강 (게이트 경로와 동일·직교)
    _install_regen_exec()
    return unified


if __name__ == "__main__":
    apply()
    print("[t2_gate_patch] applied")
