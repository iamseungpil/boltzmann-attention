#!/usr/bin/env python
"""tau2 게이트 hook: BaseOrchestrator._execute_tool_calls 몽키패치 (BENCH_PORTFOLIO §3.6 ③).

SOPBench two_stage_client 패턴 동형 — 에이전트 툴콜을 실행 *전* RetailGate로 검사,
deny면 실행 없이 게이트 메시지를 ToolMessage(error)로 반환(모델이 보고 교정),
allow면 원본 실행 후 결과로 게이트 상태 갱신(find 성공 -> 인증 확립).

활성화: 시뮬 실행 파이썬에서 `import t2_gate_patch; t2_gate_patch.apply()`
또는 환경변수 T2_GATE=1 로 sitecustomize 류에서 호출. 게이트는 orchestrator
인스턴스당 1개(_t2_gate) — 대화별 인증 상태 분리.

G2(쓰기-전-확인)는 직전 user 발화를 메시지 이력에서 추출. user 턴이 아직 없으면
(이론상 WRITE가 첫 액션일 수 없음) deny 쪽으로 떨어짐 — 정책 부합.
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t2_gate import AUTH_TOOLS, TRANSFER_MSG, RetailGate  # noqa: E402

GATE_DOMAINS = {"retail"}  # airline 등은 게이트 컴파일 후 추가

# ─── L2 provenance 게이트 (R1B, env T2_PROVENANCE=1; D1 직교·G1-G4와 합성) ───
PROV_ARG_HINT = ("email", "name", "zip", "user_id", "order_id", "username", "id",
                 "payment", "address", "phone", "item")


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


# ─── ★L1 bad_words 블랙리스트 (스키마-example + 흔한 placeholder; 동적=세션-flagged) ───
_SUCH_AS_RE = re.compile(r"such as ['\"]([^'\"]+)['\"]|e\.g\.,?\s*['\"]?([^'\".)]+)", re.I)
COMMON_PLACEHOLDERS = {
    "#W0000000", "something@example.com", "jane_doe@example.com",
    "john.doe@example.com", "johndoe@example.com", "john@example.com",
    "jane@example.com", "user@example.com", "test@example.com",
    "example@example.com", "123 Main St", "123 Main Street",
}


def _blacklist_worthy(v):
    """generic 단어(John·CA·12345) 차단 회피: len>=6 & (이메일/숫자 포함) = ID/placeholder형만."""
    v = (v or "").strip()
    return len(v) >= 6 and (("@" in v) or any(c.isdigit() for c in v))


def _static_blacklist(tools):
    bl = set(COMMON_PLACEHOLDERS)
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


def _provenance_deny(tc, ctx):
    """identifying 인자값이 컨텍스트에 없으면 fabricated → (gate, reason) 반환, 아니면 None."""
    args = _args_dict(tc)
    if not args:
        return None
    for k, v in args.items():
        if not any(h in k.lower() for h in PROV_ARG_HINT):
            continue
        for val in _flatten(v):
            s = str(val).strip()
            if len(s) < 4:
                continue
            if s.lower() not in ctx:
                return ("PROVENANCE_R1B",
                        f"argument '{k}'='{s}' was not provided by the user nor returned by any tool — it looks invented "
                        "(possibly copied from a schema example value). Do NOT call any tool with a guessed/placeholder value. "
                        "Instead OBTAIN the real value first: if a lookup/getter tool can produce it "
                        "(e.g. call get_user_details to retrieve the user's orders, payment methods, or addresses), call that and read the value from its output; "
                        "otherwise ASK the user for it.")
    return None


def _autofetch_text(self, orig, gate):
    """T2_AUTOFETCH: provenance-deny 시, 인증됐으면 producer(get_user_details)를 결정론 호출해
    그 출력을 텍스트로 반환(모델에 *실값* 제공). = '날조-FIRST' default를 엔진이 결정론으로 메움.
    decidable: producer 도출=A2(여기 retail=get_user_details, airline swap)·인자=auth_user(gate 추적).
    side-effect 0(getter)·error budget 무소비(성공). 실패시 ''."""
    if getattr(gate, "auth_user", None) is None:
        return ""
    try:
        from tau2.data_model.message import ToolCall
        ptc = ToolCall(id="autofetch", name="get_user_details",
                       arguments={"user_id": gate.auth_user}, requestor="assistant")
        out = orig(self, [ptc])
        if out and not getattr(out[0], "error", False):
            return ("\nI have fetched the authenticated user's actual record for you — copy a REAL value "
                    "from it, never a placeholder: " + _content_str(out[0])[:1800])
    except Exception:
        pass
    return ""


def apply():
    from tau2.orchestrator.orchestrator import BaseOrchestrator

    orig = BaseOrchestrator._execute_tool_calls

    def gated(self, tool_calls):
        env = self.environment
        if getattr(env, "domain_name", None) not in GATE_DOMAINS:
            return orig(self, tool_calls)
        gate = getattr(self, "_t2_gate", None)
        if gate is None:
            gate = self._t2_gate = RetailGate(db=env.tools.db)
        last_user = _last_user_text(self)
        tms = _transfer_msg_sent(self)

        # T2_PROVENANCE=1 = orchestrator-레벨 게이트(날조 호출을 *실행 전* deny→error로 surface).
        #   ⚠️ 이건 모델이 user에게 묻게 만들고 error budget 소모 → 차선. 권장 = apply_provenance_regen
        #   (agent 생성-레벨 내부 재생성; 벤치 측정 무변경). T2_PROV_SOFT(budget 안 셈)=metric gaming이라 폐기.
        prov_on = os.environ.get("T2_PROVENANCE") == "1"
        ctx = _context_text(self) if prov_on else None

        results = []
        for tc in tool_calls:
            if getattr(tc, "requestor", "assistant") != "assistant":
                results.extend(orig(self, [tc]))  # user-side 툴콜(타 도메인)은 비대상
                continue
            if prov_on:  # L2 provenance: 날조 인자값 차단 (R1B)
                pd = _provenance_deny(tc, ctx)
                if pd:
                    self.num_errors += 1
                    extra = _autofetch_text(self, orig, gate) if os.environ.get("T2_AUTOFETCH") == "1" else ""
                    results.append(_deny_msg(tc, pd[0], pd[1] + extra))
                    continue
            ok, g, why = gate.check(tc.name, tc.arguments or {}, last_user_msg=last_user,
                                    transfer_msg_sent=tms)
            if not ok:
                self.num_errors += 1
                results.append(_deny_msg(tc, g, why))
                continue
            out = orig(self, [tc])
            results.extend(out)
            if tc.name in AUTH_TOOLS and out and not out[0].error:
                gate.observe(tc.name, tc.arguments, _content_str(out[0]))
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


def _transfer_msg_sent(orch):
    """G4: 고정 transfer 문구가 어시스턴트 발화로 이미 송신됐는가 (불가 판단 시 None)."""
    try:
        for m in orch.get_messages():
            if getattr(m, "role", None) == "assistant":
                c = getattr(m, "content", None)
                if isinstance(c, str) and TRANSFER_MSG in c:
                    return True
        return False
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
# → generate() 재호출(최대 K) → 유효 호출만 반환. 거부 시도는 벤치(턴·error budget·user-sim)
# 에 *안 보임* = constrained-decoding의 call-레벨 resample. 측정 무변경(가드된 시스템을 정직 측정).
# 피드백 = "lookup으로 obtain·user에 묻지 마·placeholder 금지"(gather 유도).
REGEN_FEEDBACK = (
    "Error: [PROVENANCE] argument '{k}'='{s}' was not provided by the user nor returned by any tool "
    "— it looks invented (e.g. a schema example value). Do NOT use placeholder/example values and do NOT "
    "ask the user. Instead call a lookup/getter tool that produces this value (e.g. get_user_details to "
    "retrieve the user's orders, payment methods, or addresses) and read the real value from its output. "
    "Now emit a corrected tool call."
)


def _ctx_from_messages(msgs):
    parts = []
    for m in msgs:
        r = getattr(m, "role", None)
        c = getattr(m, "content", None)
        if r in ("user", "tool") and c is not None:
            parts.append(c if isinstance(c, str) else str(c))
    return " ".join(parts).lower()


def _first_fab_call(am, ctx):
    """am.tool_calls 중 첫 날조 호출 (tc, k, s) 또는 None."""
    for tc in (getattr(am, "tool_calls", None) or []):
        if _provenance_deny(tc, ctx):
            for k, v in _args_dict(tc).items():
                for val in _flatten(v):
                    s = str(val).strip()
                    if any(h in k.lower() for h in PROV_ARG_HINT) and len(s) >= 4 and s.lower() not in ctx:
                        return (tc, k, s)
            return (tc, "?", "?")
    return None


# ─── ★GROUND (T2_PROV_GROUND=1): config-도출 candidate-surfacing resolver (read-id grounding leg) ───
# §25 처방: 모델은 *의도/op* 명명만, 구체값(order_id/item_id/payment)은 결정론이 직전 tool 출력서 grounding.
# deny-only(L2)는 "obtain하라"만 해 모델이 getter 출력서 *추출*을 못해 실패(v7 0.05→0.0). GROUND는
# (a) 같은 타입 후보를 tool 출력서 *추출해 제시*(retry-loop 차단) + (b) 단일후보면 *자동 치환*(true offload).
# 추출 = arg-key 토큰 매치 ∪ 값-시그니처 매치 = 도메인-일반(ABox-swap = airline 출력에 동일 적용). bespoke 아님.

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
    """JSON에서 (enclosing_key, scalar) 재귀 수집 + dict-key 자체(예 payment_method id 'paypal_..')."""
    if isinstance(obj, dict):
        for kk, vv in obj.items():
            if isinstance(vv, (dict, list)):
                # dict-key가 ID형이면 후보(payment_methods의 'paypal_7644869')
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


def _parse_tool_outputs(msgs):
    """role==tool·비-error 메시지 content를 JSON 파싱(최근→과거 순)."""
    outs = []
    for m in reversed(msgs):  # 최근 우선
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
            outs.append(c)  # 비-JSON 문자열도 보관(부분 매치용)
    return outs


def _grounded_candidates(arg_key, fab_value, msgs, limit=8):
    """arg_key 타입에 맞는 grounded 후보값을 tool 출력서 추출(최근 우선·dedup·순서보존)."""
    toks = _key_tokens(arg_key)
    want_sig = _sig(fab_value)
    seen = set()
    key_cands, sig_cands = [], []
    for out in _parse_tool_outputs(msgs):
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
    # arg-key 매치가 있으면 그것만(정밀) — 없을 때만 값-시그니처 후보로 폴백
    return (key_cands or sig_cands)[:limit]


GROUND_FEEDBACK = (
    "Error: [PROVENANCE] argument '{k}'='{s}' is invented — never use placeholder/guessed IDs. "
    "The real, grounded {k} value(s) already available from prior tool results are: {cands}. "
    "Use ONLY one of these. If you must determine WHICH one matches what the user described "
    "(e.g. which order contains that item), call the appropriate getter (e.g. get_order_details) "
    "on these candidates and read the answer from its output. Now emit a corrected tool call."
)


def apply_provenance_regen(max_retries=4, use_badwords=True, ground=False):
    """LLMAgent._generate_next_message 패치 — R1b 통합:
      L1 = bad_words 디코드-마스크(정적 블랙리스트 + 세션-flagged − 현재 context).
      L2 = provenance 검증기 + 내부 재생성(verifier가 날조 잡으면 작업본서 regen·세션 블랙리스트 추가).
      GROUND(ground=True) = config-도출 candidate-surfacing: 단일후보 자동치환(true offload)·
        다중후보는 grounded 집합을 피드백에 제시(retry-loop 차단·모델은 enumerate/select만).
    use_badwords=False면 L2만. max_retries=0이면 L1만."""
    from tau2.agent.llm_agent import LLMAgent
    import tau2.agent.llm_agent as la
    from tau2.data_model.message import ToolMessage, UserMessage, MultiToolMessage

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
            self._t2_static_bl = _static_blacklist(self.tools)
            self._t2_session_bl = set()
        self._system_messages = state.system_messages
        _append(state, message)
        ctx = _ctx_from_messages(state.messages)

        def bw():  # 동적: 정적∪세션 − context (진짜 값은 안 막음)
            return [v for v in (self._t2_static_bl | self._t2_session_bl) if v.lower() not in ctx]

        work = list(state.messages)
        am = _gen(self, work, bw(), "agent_response")
        n = 0
        subs = 0  # 결정론 자동치환 횟수 (regen budget n과 분리)
        while n < max_retries:
            fab = _first_fab_call(am, ctx)
            if fab is None:
                break
            tc, k, s = fab

            # ── GROUND: 단일 grounded 후보 → 결정론 자동치환 (regen 소모 없음·true offload) ──
            if ground and subs < 8:
                cands = _grounded_candidates(k, s, state.messages)
                if len(cands) == 1 and cands[0] != s:
                    d = _args_dict(tc)
                    d[k] = cands[0]
                    tc.arguments = d
                    self._t2_ground_sub = getattr(self, "_t2_ground_sub", 0) + 1
                    subs += 1
                    continue  # 같은 am 재검사(다른 fab arg 있을 수 있음)

            n += 1
            self._t2_session_bl.add(s)  # 동적 블랙리스트: 날조값 → 다음 gen bad_words 차단(루프 방지)
            self._t2_regen = getattr(self, "_t2_regen", 0) + 1
            work = work + [am]  # 거부된 assistant 턴 (작업본만·state 무오염)
            cands = _grounded_candidates(k, s, state.messages) if ground else []
            if ground and cands:  # 다중후보 → grounded 집합 제시
                main_reason = GROUND_FEEDBACK.format(k=k, s=s, cands=", ".join(repr(c) for c in cands))
            else:
                main_reason = REGEN_FEEDBACK.format(k=k, s=s)
            for c in (am.tool_calls or []):
                reason = main_reason if c is tc else \
                    "Error: [PROVENANCE] resolve the invented value first; do not call this yet."
                work.append(ToolMessage(id=c.id, role="tool", requestor="assistant",
                                        error=True, content=reason))
            am = _gen(self, work, bw(), "agent_response_regen")
        return am

    LLMAgent._generate_next_message = patched
    return patched


if __name__ == "__main__":
    apply()
    print("[t2_gate_patch] applied")
