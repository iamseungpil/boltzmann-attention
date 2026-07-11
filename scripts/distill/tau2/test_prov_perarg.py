# -*- coding: utf-8 -*-
"""Unit tests for PROV-RESCUE-PERARG (CENSUS_LEVERS_DESIGN_2026_07_11 §1, 2026-07-11).

오프라인·tau2 임포트 0 (stub 주입 = test_t5c_silent.py 동일 패턴).
Covers:
  ② id '#'-접두 정규화: _ctx_has / _provenance_deny / _first_fab_call — '#W123' vs ctx 'w123'
     = fab 아님. '#' 하나에 한정(다른 기호 미정규화)·strip 후 4자 미만은 비적용.
  ① rescue per-arg: _first_fab_call(exclude=...) 스킵 동작, regen 루프에서 env-검증형 id fab
     스킵 후 같은 호출의 자유텍스트 fab이 regen 발화되는지, 전부-스킵 시 종료 경로(구 break 의미),
     마커/카운터 스킵당 1회, rescue 무과금(n 미증가).
  t17 재현 형상: {order_id:'#W8665881'(ctx 'w8665881'), address1:'123 Elm St'(ctx 부재)}
     → ② 적용 후 첫 fab = address1.
"""
import sys, os, types, json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ["T2_GATE_KINDS"] = "auth,confirm,ownership,notice,preconditions,constraints"

# ---------- tau2 stubs (tau2 실임포트 0) ----------
def mkmod(name):
    m = types.ModuleType(name)
    sys.modules[name] = m
    return m

mkmod("tau2"); mkmod("tau2.agent"); la = mkmod("tau2.agent.llm_agent")
mkmod("tau2.data_model"); msgmod = mkmod("tau2.data_model.message")
mkmod("tau2.orchestrator"); oo = mkmod("tau2.orchestrator.orchestrator")

class ToolMessage:
    def __init__(self, id=None, role="tool", requestor="assistant", error=False, content=""):
        self.id, self.role, self.requestor, self.error, self.content = id, role, requestor, error, content
class UserMessage:
    def __init__(self, content="", role="user"):
        self.role, self.content, self.is_audio = role, content, False
class MultiToolMessage:
    def __init__(self, tool_messages=None):
        self.tool_messages = tool_messages or []
msgmod.ToolMessage, msgmod.UserMessage, msgmod.MultiToolMessage = ToolMessage, UserMessage, MultiToolMessage
msgmod.ToolCall = None

class ToolCall:  # duck-typed 가짜 tool_call
    _n = 0
    def __init__(self, name, arguments, id=None):
        ToolCall._n += 1
        self.id = id or ("tc%d" % ToolCall._n)
        self.name, self.arguments, self.requestor = name, arguments, "assistant"
class AM:  # duck-typed 가짜 assistant message
    def __init__(self, tool_calls=None, content=None):
        self.role, self.tool_calls, self.content = "assistant", tool_calls, content

SCRIPT, GENCALLS, GENKW = [], [], []
def generate(model=None, tools=None, messages=None, call_name=None, **kw):
    GENCALLS.append(call_name)
    GENKW.append({"tools": tools, "messages": messages, "kw": kw})
    if not SCRIPT:
        raise AssertionError("SCRIPT exhausted (unexpected generate: %s)" % call_name)
    return SCRIPT.pop(0)
la.generate = generate
class LLMAgent:
    pass
la.LLMAgent = LLMAgent

class BaseOrchestrator:
    def __init__(self, environment=None, agent=None):
        self.environment, self.agent, self.num_errors = environment, agent, 0
    def _execute_tool_calls(self, tool_calls):
        return []
oo.BaseOrchestrator = BaseOrchestrator

import t2_gate_patch as G  # noqa: E402

class State:
    def __init__(self, messages):
        self.system_messages, self.messages = [], list(messages)

FAILS = []
NCHECKS = [0]
def check(name, cond, detail=""):
    NCHECKS[0] += 1
    print(("PASS " if cond else "FAIL ") + name + ((" | " + str(detail)) if (detail and not cond) else ""))
    if not cond:
        FAILS.append(name)

def reset(*ams):
    SCRIPT[:] = list(ams); GENCALLS[:] = []; GENKW[:] = []

def agent():
    ag = LLMAgent()
    ag.llm, ag.llm_args, ag.tools = "m", {"temperature": 0.0, "tool_choice": "auto"}, []
    return ag

# ══════════════ ② '#'-접두 정규화 (함수 단위) ══════════════

CTX = "please modify order w8665881 for me"  # 사용자가 '#' 없이 발화 (t17 msg[7] 형상)

check("N1_hash_norm_true", G._ctx_has("#W8665881", CTX))                # '#W...' vs ctx 'w...' = 실재
check("N2_exact_still_true", G._ctx_has("w8665881", CTX))               # 기존 경로 불변
check("N3_absent_false", not G._ctx_has("#W9999999", CTX))              # 진짜 fab은 여전히 fab
check("N4_other_symbol_not_normalized", not G._ctx_has("*W8665881", CTX))  # '#' 하나에 한정
check("N5_short_after_strip_false", not G._ctx_has("#abc", "the abc value"))  # strip 후 4자 미만 비적용
check("N6_multi_hash_ok", G._ctx_has("##W8665881", CTX))                # lstrip('#') = '#'만 제거

# _provenance_deny: '#'-접두 거짓양성 제거 (t17 1차 방아쇠)
tc = ToolCall("modify_pending_order_address", {"order_id": "#W8665881"})
check("N7_deny_none_hash", G._provenance_deny(tc, CTX) is None)
tc = ToolCall("modify_pending_order_address", {"order_id": "#W9999999"})
check("N8_deny_real_fab", G._provenance_deny(tc, CTX) is not None)

# ══════════════ t17 재현 형상: ② 후 첫 fab = address1 ══════════════

t17 = ToolCall("modify_pending_order_address",
               {"order_id": "#W8665881", "address1": "123 Elm St"})
fab = G._first_fab_call(AM(tool_calls=[t17]), CTX)
check("T1_first_fab_is_address1", fab is not None and fab[1] == "address1" and fab[2] == "123 Elm St",
      fab)

# ══════════════ ① _first_fab_call exclude 동작 (함수 단위) ══════════════

CTX2 = "hello, i want to change my order"  # 두 값 모두 부재
dual = ToolCall("modify_pending_order_address",
                {"order_id": "#W5555555", "address1": "9999 Nowhere Rd"})
am = AM(tool_calls=[dual])
f0 = G._first_fab_call(am, CTX2)
check("X1_no_exclude_first_arg", f0 == (dual, "order_id", "#W5555555"), f0)
f1 = G._first_fab_call(am, CTX2, exclude={(id(dual), "order_id", "#W5555555")})
check("X2_exclude_skips_to_freetext", f1 == (dual, "address1", "9999 Nowhere Rd"), f1)
f2 = G._first_fab_call(am, CTX2, exclude={(id(dual), "order_id", "#W5555555"),
                                          (id(dual), "address1", "9999 Nowhere Rd")})
check("X3_all_excluded_none", f2 is None, f2)
# exclude는 다음 *호출*로도 계속 스캔
dual2 = ToolCall("get_order_details", {"order_id": "#W7777777"})
am2 = AM(tool_calls=[dual, dual2])
f3 = G._first_fab_call(am2, CTX2, exclude={(id(dual), "order_id", "#W5555555"),
                                           (id(dual), "address1", "9999 Nowhere Rd")})
check("X4_exclude_scans_next_call", f3 == (dual2, "order_id", "#W7777777"), f3)

# ══════════════ ① regen 루프 (apply_provenance_regen·prov_mode=rescue) ══════════════

def hist(user_text):
    return [UserMessage(user_text)]

G.apply_provenance_regen(max_retries=4, use_badwords=False, ground=False, domain="retail",
                         disamb=False, prov_mode="rescue")

# L1: fab 2개(env-검증형 id + 자유텍스트) — id는 스킵되고 자유텍스트로 regen 발화
ag, st = agent(), State(hist("please change the address on my pending order"))
w = ToolCall("modify_pending_order_address",
             {"order_id": "#W5555555", "address1": "9999 Nowhere Rd"})
reset(AM(tool_calls=[w]), AM(content="let me look up your real address first"))
out = ag._generate_next_message(UserMessage("go ahead"), st)
check("L1_regen_fired_on_freetext", GENCALLS == ["agent_response", "agent_response_regen"], GENCALLS)
check("L2_skip_counter_once", getattr(ag, "_t2_prov_skipped_envdup", 0) == 1)
fb = ""
for gk in GENKW:
    for m in (gk["messages"] or []):
        c = getattr(m, "content", "") or ""
        if isinstance(c, str) and "[PROVENANCE]" in c and "address1" in c:
            fb = c
check("L3_feedback_targets_address1", "address1" in fb and "9999 Nowhere Rd" in fb, fb[:120])
check("L4_regen_budget_charged_once", getattr(ag, "_t2_regen", 0) == 1)  # rescue 스킵=무과금

# L5: 전부-스킵(잔여 fab = env-검증형 id 하나뿐) → 종료 경로 = pass-through(구 break 의미 보존)
ag, st = agent(), State(hist("modify my pending order please"))
w = ToolCall("modify_pending_order_items", {"order_id": "#W5555555"})
reset(AM(tool_calls=[w]))
out = ag._generate_next_message(UserMessage("modify it"), st)
check("L5_allskip_terminates_passthrough",
      out.tool_calls and w.arguments["order_id"] == "#W5555555")
check("L6_allskip_no_regen", GENCALLS == ["agent_response"], GENCALLS)
check("L7_marker_counter_once", getattr(ag, "_t2_prov_skipped_envdup", 0) == 1)
check("L8_no_regen_budget", getattr(ag, "_t2_regen", 0) == 0)

# L9: t17 재현 (루프 레벨) — order_id는 ②로 fab 아님(rescue 스킵조차 불필요)·address1이 regen
ag, st = agent(), State(hist("send order W8665881 to my other address"))
w = ToolCall("modify_pending_order_address",
             {"order_id": "#W8665881", "address1": "123 Elm St"})
reset(AM(tool_calls=[w]), AM(content="let me fetch your address"))
out = ag._generate_next_message(UserMessage("yes"), st)
check("L9_t17_regen_on_address1", GENCALLS == ["agent_response", "agent_response_regen"], GENCALLS)
check("L10_t17_no_rescue_skip", getattr(ag, "_t2_prov_skipped_envdup", 0) == 0)
fb = ""
for gk in GENKW:
    for m in (gk["messages"] or []):
        c = getattr(m, "content", "") or ""
        if isinstance(c, str) and "[PROVENANCE]" in c and "123 Elm St" in c:
            fb = c
check("L11_t17_feedback_elm", "address1" in fb, fb[:120])

print()
if FAILS:
    print("FAILED: %d -> %s" % (len(FAILS), FAILS))
    sys.exit(1)
print("ALL PASS (%d checks)" % NCHECKS[0])
