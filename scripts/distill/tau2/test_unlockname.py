# -*- coding: utf-8 -*-
"""T2_UNLOCK_NAME(생성-레벨 bare-name deny·2026-07-21 §2bh) 오프라인 배선 테스트.
검정: ①bare-name unlock → deny+피드백(KB 검색 지시)+required 재생성 → 접미사명으로 교정
②패턴 일치(정상 접미사) → 무간섭 ③OFF → 통과 ④cap 소진 → 통과.
⚠️단위통과≠라이브발화([[30]])."""
import sys, os, types, json
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
SCRIPT, FEEDBACKS, CHOICES = [], [], []
def generate(model=None, tools=None, messages=None, call_name=None, **kw):
    CHOICES.append(kw.get("tool_choice"))
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
def setup(hist):
    ag = LLMAgent(); ag.llm, ag.llm_args = "m", {}
    ag.tools = [types.SimpleNamespace(name=n) for n in
                ("unlock_discoverable_agent_tool", "call_discoverable_agent_tool", "KB_search_bm25")]
    orch = BaseOrchestrator(environment=Env(), agent=ag)
    return ag, orch, State(hist)
def auth():
    tc = ToolCall("log_verification", {"user_id": "u1"}, id="a1")
    return [AM(tool_calls=[tc]), ToolMessage(id="a1", content="Verification logged.")]
FAILS = []
def check(n, c, d=""):
    print(("PASS " if c else "FAIL ") + n + ((" | " + str(d)) if d and not c else ""))
    if not c: FAILS.append(n)
HIST = auth() + [UserMessage("please check my dispute history get_user_dispute_history_7291")]
BARE = lambda: ToolCall("unlock_discoverable_agent_tool", {"agent_tool_name": "get_user_dispute_history"})
FULL = lambda: ToolCall("unlock_discoverable_agent_tool", {"agent_tool_name": "get_user_dispute_history_7291"})
os.environ["T2_UNLOCK_NAME"] = "1"
ag, orch, st = setup(HIST); SCRIPT[:] = [AM([BARE()]), AM([FULL()])]; FEEDBACKS[:] = []; CHOICES[:] = []
am = ag._generate_next_message(UserMessage("yes"), st)
check("U1_corrected", am.tool_calls and am.tool_calls[0].arguments["agent_tool_name"].endswith("_7291"))
check("U1_fb_kb", any("search the knowledge base" in str(f) for f in FEEDBACKS))
check("U1_required", "required" in [c for c in CHOICES if c])
check("U1_cnt", getattr(ag, "_t2_unlockname_deny", 0) == 1)
ag, orch, st = setup(HIST); SCRIPT[:] = [AM([FULL()])]
am = ag._generate_next_message(UserMessage("yes"), st)
check("U2_full_untouched", am.tool_calls and am.tool_calls[0].arguments["agent_tool_name"].endswith("_7291"))
os.environ.pop("T2_UNLOCK_NAME", None)
ag, orch, st = setup(HIST); SCRIPT[:] = [AM([BARE()])]
am = ag._generate_next_message(UserMessage("yes"), st)
check("U3_off_pass", am.tool_calls and am.tool_calls[0].arguments["agent_tool_name"] == "get_user_dispute_history")
os.environ["T2_UNLOCK_NAME"] = "1"; os.environ["T2_UNLOCK_NAME_CAP"] = "1"
ag, orch, st = setup(HIST); SCRIPT[:] = [AM([BARE()]), AM([BARE()])]
am = ag._generate_next_message(UserMessage("yes"), st)
check("U4_cap1_then_pass", am.tool_calls and am.tool_calls[0].arguments["agent_tool_name"] == "get_user_dispute_history")
check("U4_cnt1", getattr(ag, "_t2_unlockname_deny", 0) == 1)
os.environ.pop("T2_UNLOCK_NAME_CAP", None); os.environ.pop("T2_UNLOCK_NAME", None)
print("\n%s" % ("ALL PASS" if not FAILS else "FAILS: %s" % FAILS))
sys.exit(1 if FAILS else 0)
