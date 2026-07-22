# -*- coding: utf-8 -*-
"""T2_UNKNOWN_NAME_BL + T2_UNLOCK_PROV (2026-07-22 §2bt·rall11 050 실측) 오프라인 배선 테스트.

050: regen이 환각 접미사 get_pending_replacement_orders_8374를 커밋(§2bi는 bare-name만 검사)
→ env "Unknown agent tool" 후에도 같은 이름 재시도(에러 에코가 ctx에 이름을 넣어 PROV 무력화).
검정: ①env-거부명 재시도 → deny(un_fb 경로·교정 재생성) ②거부 이력 없는 정상 이름 → 통과
③OFF → 통과 ④regen-경로: 환각 접미사(ctx 부재) → T2_UNLOCK_PROV deny(간접: 로그 카운트).
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
def check(n, c):
    print(("PASS " if c else "FAIL ") + n)
    if not c: FAILS.append(n)

BAD = "get_pending_replacement_orders_8374"
GOOD = "get_pending_replacement_orders_5765"
# 히스토리: env 에러가 BAD를 에코(=ctx에 실재→PROV 무력 재현) + KB 결과에 GOOD 실재
HIST = auth() + [
    AM(tool_calls=[ToolCall("unlock_discoverable_agent_tool", {"agent_tool_name": BAD}, id="e1")]),
    ToolMessage(id="e1", content="Error: Unknown agent tool 'get_pending_replacement_orders_8374'. This tool is not available."),
    ToolMessage(id="k1", content="1. Checking Pending Replacement Card Orders (Internal) ... use the get_pending_replacement_orders_5765 tool"),
    UserMessage("please run the checks")]
RETRY_BAD = lambda: ToolCall("unlock_discoverable_agent_tool", {"agent_tool_name": BAD})
TRY_GOOD = lambda: ToolCall("unlock_discoverable_agent_tool", {"agent_tool_name": GOOD})

# ① env-거부명 재시도 → deny + 교정 재생성
os.environ["T2_UNKNOWN_NAME_BL"] = "1"
ag, orch, st = setup(HIST); SCRIPT[:] = [AM([RETRY_BAD()]), AM([TRY_GOOD()])]; FEEDBACKS[:] = []
am = ag._generate_next_message(UserMessage("go"), st)
check("B1_corrected", am.tool_calls and am.tool_calls[0].arguments["agent_tool_name"] == GOOD)
check("B1_fb_rejected", any("already rejected by the environment" in str(f) for f in FEEDBACKS))
check("B1_fb_name_words", any("get pending replacement orders" in str(f) for f in FEEDBACKS))
check("B1_cnt", getattr(ag, "_t2_unknownbl_deny", 0) == 1)
# ② 정상 이름 → 통과
ag, orch, st = setup(HIST); SCRIPT[:] = [AM([TRY_GOOD()])]
am = ag._generate_next_message(UserMessage("go"), st)
check("B2_good_pass", am.tool_calls and am.tool_calls[0].arguments["agent_tool_name"] == GOOD)
# ③ OFF → 통과(거동보존)
os.environ.pop("T2_UNKNOWN_NAME_BL", None)
ag, orch, st = setup(HIST); SCRIPT[:] = [AM([RETRY_BAD()])]
am = ag._generate_next_message(UserMessage("go"), st)
check("B3_off_pass", am.tool_calls and am.tool_calls[0].arguments["agent_tool_name"] == BAD)

print("\n%s" % ("ALL PASS" if not FAILS else "FAILS: %s" % FAILS))
sys.exit(1 if FAILS else 0)
