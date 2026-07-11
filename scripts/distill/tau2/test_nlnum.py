# -*- coding: utf-8 -*-
"""NL-NUM-PROV 단위 (레버1·NEXT_LEVER_GEN 후속 t47 처방·2026-07-11).

unified() 최종 반환 직전의 통화-금액 provenance 검사:
  발화의 `$금액`(통화기호+숫자·도메인 어휘 0)이 이전 문맥(user/tool 텍스트)에
  원문-부재 → 생성-레벨 regen 1회(무과금·비커밋·[T2_NLNUM] 마커·게이트 재검사).
tau2 stub 주입 = test_unified_regen.py 동형. toggle T2_NLNUM_PROV=1.
"""
import sys, os, types, json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ["T2_GATE_KINDS"] = "auth,confirm,ownership,notice,preconditions,constraints"
os.environ["T2_NLNUM_PROV"] = "1"

# ---------- tau2 stubs ----------
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
    def __init__(self, content=""):
        self.role, self.content, self.is_audio = "user", content, False
class MultiToolMessage:
    def __init__(self, tool_messages=None):
        self.tool_messages = tool_messages or []
msgmod.ToolMessage, msgmod.UserMessage, msgmod.MultiToolMessage = ToolMessage, UserMessage, MultiToolMessage
msgmod.ToolCall = None

class ToolCall:
    _n = 0
    def __init__(self, name, arguments, id=None):
        ToolCall._n += 1
        self.id = id or ("tc%d" % ToolCall._n)
        self.name, self.arguments, self.requestor = name, arguments, "assistant"
class AM:
    def __init__(self, tool_calls=None, content=None):
        self.role, self.tool_calls, self.content = "assistant", tool_calls, content

SCRIPT, GENCALLS = [], []
def generate(model=None, tools=None, messages=None, call_name=None, **kw):
    GENCALLS.append((call_name, messages))
    if not SCRIPT:
        raise AssertionError("SCRIPT exhausted (unexpected extra generate call)")
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

class Env:
    def __init__(self):
        self.domain_name, self.tools = "retail", None
class State:
    def __init__(self, messages):
        self.system_messages, self.messages = [], list(messages)

G.apply_unified_regen(max_prov_retries=4, domain="retail", disamb=False)

def setup(history):
    ag = LLMAgent()
    ag.llm, ag.llm_args, ag.tools = "m", {}, []
    orch = BaseOrchestrator(environment=Env(), agent=ag)
    return ag, orch, State(history)

def auth_pair():
    tc = ToolCall("find_user_id_by_email", {"email": "real@user.com"}, id="auth1")
    return [AM(tool_calls=[tc]), ToolMessage(id="auth1", content="usr_123")]

FAILS, NCHECK = [], [0]
def check(name, cond, detail=""):
    NCHECK[0] += 1
    print(("PASS " if cond else "FAIL ") + name + (" | " + str(detail) if detail and not cond else ""))
    if not cond:
        FAILS.append(name)

# ── U0: 순수 술어 (_unverified_amounts / _num_variants) ─────────────────────
ctx = "the price is 622.12 and 473.43 total 1095.55 order #W1"
check("U0a_absent_fires", G._unverified_amounts("you keep $928.13 worth", ctx) == ["$928.13"])
check("U0b_present_silent", G._unverified_amounts("refund of $1095.55", ctx) == [])
check("U0c_comma_normalized", G._unverified_amounts("total $1,095.55", ctx) == [])
check("U0d_trailing_zero", G._unverified_amounts("costs $622.10", "value 622.1 here") == [])
check("U0e_no_currency_silent", G._unverified_amounts("code 928.13 is set", ctx) == [])
check("U0f_integer_amount_silent", G._unverified_amounts("flat $50 fee", ctx) == [],
      "정수 금액(센트 없음)=보수적 미발화")
check("U0g_multi_first", G._unverified_amounts("$11.11 then $22.22", "nothing")
      == ["$11.11", "$22.22"])

# ── U1: 발화 → regen 1회·마커·무과금·채택 ──────────────────────────────────
hist = auth_pair() + [UserMessage("what did I pay for the rest?"),
                      ToolMessage(id="r1", content=json.dumps(
                          {"items": [{"price": 329.85}, {"price": 545.68}]}))]
ag, orch, st = setup(hist)
SCRIPT[:] = [AM(content="You paid $928.13 for the remaining items."),
             AM(tool_calls=[ToolCall("calculate", {"expression": "329.85+545.68"})])]
am = ag._generate_next_message(UserMessage("yes tell me"), st)
check("U1_fired_once", getattr(ag, "_t2_nlnum", 0) == 1)
check("U1_regen_adopted", am.tool_calls and am.tool_calls[0].name == "calculate")
check("U1_no_tick", orch.num_errors == 0, orch.num_errors)
check("U1_gen_call_name", GENCALLS[-1][0] == "agent_response_nlnum")
fb = GENCALLS[-1][1][-1]
check("U1_feedback_uncommitted", fb.content.startswith("Error: [NL-NUM]")
      and all(getattr(m, "content", "") != fb.content for m in st.messages),
      "피드백=작업버퍼만·히스토리 비커밋")
check("U1_feedback_names_calc_tool", "calculate" in fb.content)

# ── U2: 금액이 tool 출력에 실재 → 미발화 ───────────────────────────────────
ag, orch, st = setup(auth_pair() + [ToolMessage(id="r2", content='{"total": 875.53}')])
SCRIPT[:] = [AM(content="The total is $875.53.")]
am = ag._generate_next_message(UserMessage("total?"), st)
check("U2_silent_in_tool", getattr(ag, "_t2_nlnum", 0) == 0 and am.content == "The total is $875.53.")

# ── U3: 금액이 user 발화에 실재 → 미발화 ───────────────────────────────────
ag, orch, st = setup(auth_pair())
SCRIPT[:] = [AM(content="Confirming your budget of $500.25.")]
am = ag._generate_next_message(UserMessage("my budget is $500.25"), st)
check("U3_silent_in_user", getattr(ag, "_t2_nlnum", 0) == 0)

# ── U4: 상한 1/턴 — regen 산출에 여전히 미검증 금액이어도 재발화 없음 ───────
ag, orch, st = setup(auth_pair())
SCRIPT[:] = [AM(content="You owe $111.11."), AM(content="Actually $222.22.")]
am = ag._generate_next_message(UserMessage("how much?"), st)
check("U4_cap_1_per_turn", getattr(ag, "_t2_nlnum", 0) == 1 and am.content == "Actually $222.22.")
check("U4_two_gens_only", len(GENCALLS) >= 2 and GENCALLS[-2][0] == "agent_response"
      and GENCALLS[-1][0] == "agent_response_nlnum")

# ── U5: regen이 게이트-deny 호출을 새로 들이면 원 am 유지 ───────────────────
ag, orch, st = setup([UserMessage("hello")])  # 미인증
SCRIPT[:] = [AM(content="Refund of $333.33 confirmed."),
             AM(tool_calls=[ToolCall("cancel_pending_order", {"order_id": "#W1112223"})])]
am = ag._generate_next_message(UserMessage("ok?"), st)
check("U5_gate_reject_keeps_original", am.content == "Refund of $333.33 confirmed."
      and getattr(ag, "_t2_nlnum_gate_reject", 0) == 1)
check("U5_no_tick", orch.num_errors == 0, orch.num_errors)

# ── U6: toggle off → 완전 무개입 ───────────────────────────────────────────
os.environ["T2_NLNUM_PROV"] = "0"
ag, orch, st = setup(auth_pair())
SCRIPT[:] = [AM(content="You owe $999.99.")]
am = ag._generate_next_message(UserMessage("how much?"), st)
check("U6_toggle_off_silent", getattr(ag, "_t2_nlnum", 0) == 0 and am.content == "You owe $999.99.")
os.environ["T2_NLNUM_PROV"] = "1"

# ── U7: A2에 calc_tool 없으면 비활성 (키 부재=안전측) ───────────────────────
_saved = G._A2_CACHE.get("retail")
_noca = dict(_saved); _noca.pop("calc_tool", None)
G._A2_CACHE["retail"] = _noca
ag, orch, st = setup(auth_pair())
SCRIPT[:] = [AM(content="You owe $999.99.")]
am = ag._generate_next_message(UserMessage("how much?"), st)
check("U7_no_calc_tool_silent", getattr(ag, "_t2_nlnum", 0) == 0)
G._A2_CACHE["retail"] = _saved

print("\n%d FAIL" % len(FAILS) if FAILS else "\nALL PASS (%d checks)" % NCHECK[0])
sys.exit(1 if FAILS else 0)
