# -*- coding: utf-8 -*-
"""T2_DISPATCH_ROLE 오프라인 배선 테스트 (§2bl 기존 3술어 + 2026-07-22 §2bs give-agent-tool 술어).

rall10 031 실측: 에이전트가 자기 도구(get_credit_card_accounts_by_user)를 give →
env "Unknown discoverable tool" 2회에도 오선택 고수 → dispute args fabrication으로 직행.
신규 술어: give 대상 ∈ 자기 도구 목록 → 생성-레벨 deny(인터페이스-구조·이름 리터럴 0).
검정: ①give(자기 agent-도구) → deny·A2 문구·교정 재생성 ②give(목록 밖 이름=정상 user-도구
후보) → 무간섭 ③기존 술어 회귀(unlock된 이름을 user_call로 → deny) ④OFF → 통과.
⚠️단위통과≠라이브발화([[30]])."""
import sys, os, types
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8"); sys.stderr.reconfigure(encoding="utf-8")
except Exception: pass
def mkmod(n):
    m = types.ModuleType(n); sys.modules[n] = m; return m
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
SCRIPT, FEEDBACKS = [], []
def generate(model=None, tools=None, messages=None, call_name=None, **kw):
    for m in reversed(messages or []):
        if getattr(m, "role", "") == "tool" and getattr(m, "error", False):
            FEEDBACKS.append(m.content); break
    if not SCRIPT: raise AssertionError("SCRIPT exhausted")
    return SCRIPT.pop(0)
la.generate = generate
class LLMAgent: pass
la.LLMAgent = LLMAgent
class BaseOrchestrator:
    def __init__(self, environment=None, agent=None):
        self.environment, self.agent, self.num_errors = environment, agent, 0
    def _execute_tool_calls(self, tool_calls): return []
oo.BaseOrchestrator = BaseOrchestrator
import t2_gate_patch as G  # noqa: E402
class Env:
    def __init__(self): self.domain_name, self.tools = "banking_knowledge", None
class State:
    def __init__(self, messages): self.system_messages, self.messages = [], list(messages)
G.apply_unified_regen(max_prov_retries=4, domain="banking_knowledge")
AGENT_TOOLS = ("get_credit_card_accounts_by_user", "give_discoverable_user_tool",
               "call_discoverable_user_tool", "unlock_discoverable_agent_tool",
               "call_discoverable_agent_tool", "KB_search_bm25")
def setup(hist):
    ag = LLMAgent(); ag.llm, ag.llm_args = "m", {}
    ag.tools = [types.SimpleNamespace(name=n) for n in AGENT_TOOLS]
    orch = BaseOrchestrator(environment=Env(), agent=ag)
    return ag, orch, State(hist)
def auth():
    tc = ToolCall("log_verification", {"user_id": "u1"}, id="a1")
    return [AM(tool_calls=[tc]), ToolMessage(id="a1", content="Verification logged.")]
FAILS = []
def check(n, c, d=""):
    print(("PASS " if c else "FAIL ") + n + ((" | " + str(d)) if d and not c else ""))
    if not c: FAILS.append(n)

# user가 도구명을 지명(031 [33] 재현) — PROV 근거도 겸함
HIST = auth() + [UserMessage("please give me access to the get_card_last_4_digits tool")]
GIVE_AGENT = lambda: ToolCall("give_discoverable_user_tool",
                              {"discoverable_tool_name": "get_credit_card_accounts_by_user"})
GIVE_USER = lambda: ToolCall("give_discoverable_user_tool",
                             {"discoverable_tool_name": "get_card_last_4_digits"})

os.environ["T2_DISPATCH_ROLE"] = "1"
# ① give(자기 agent-도구) → deny + A2 문구 + 재생성이 user-지명 도구로 교정
ag, orch, st = setup(HIST); SCRIPT[:] = [AM([GIVE_AGENT()]), AM([GIVE_USER()])]; FEEDBACKS[:] = []
am = ag._generate_next_message(UserMessage("yes"), st)
check("D1_corrected", am.tool_calls and
      am.tool_calls[0].arguments["discoverable_tool_name"] == "get_card_last_4_digits")
check("D1_fb_own_tool", any("YOUR OWN agent tools" in str(f) for f in FEEDBACKS))
check("D1_fb_named_hint", any("customer already told you" in str(f) for f in FEEDBACKS))
check("D1_cnt", getattr(ag, "_t2_dispatchrole_deny", 0) == 1)

# ② give(목록 밖 이름=정상 user-도구 후보) → 무간섭
ag, orch, st = setup(HIST); SCRIPT[:] = [AM([GIVE_USER()])]
am = ag._generate_next_message(UserMessage("yes"), st)
check("D2_user_tool_untouched", am.tool_calls and
      am.tool_calls[0].arguments["discoverable_tool_name"] == "get_card_last_4_digits")

# ③ 기존 술어 회귀: 이 대화서 unlock된 이름을 call_discoverable_user_tool로 → deny
H3 = auth() + [AM([ToolCall("unlock_discoverable_agent_tool",
                            {"agent_tool_name": "get_user_dispute_history_7291"}, id="u9")]),
               ToolMessage(id="u9", content="Tool unlocked: get_user_dispute_history_7291"),
               UserMessage("ok")]
BADCALL = lambda: ToolCall("call_discoverable_user_tool",
                           {"discoverable_tool_name": "get_user_dispute_history_7291"})
GOODCALL = lambda: ToolCall("call_discoverable_agent_tool",
                            {"agent_tool_name": "get_user_dispute_history_7291",
                             "arguments": "{}"})
ag, orch, st = setup(H3); SCRIPT[:] = [AM([BADCALL()]), AM([GOODCALL()])]; FEEDBACKS[:] = []
am = ag._generate_next_message(UserMessage("yes"), st)
check("D3_regression_agent_via_user", any("AGENT tool" in str(f) for f in FEEDBACKS))

# ④ OFF → 통과(거동보존)
os.environ.pop("T2_DISPATCH_ROLE", None)
ag, orch, st = setup(HIST); SCRIPT[:] = [AM([GIVE_AGENT()])]
am = ag._generate_next_message(UserMessage("yes"), st)
check("D4_off_pass", am.tool_calls and
      am.tool_calls[0].arguments["discoverable_tool_name"] == "get_credit_card_accounts_by_user")

print("\n%s" % ("ALL PASS" if not FAILS else "FAILS: %s" % FAILS))
sys.exit(1 if FAILS else 0)
