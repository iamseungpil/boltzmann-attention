# -*- coding: utf-8 -*-
"""T2_FOLLOWUP_PROGRESS_REFUND(진행-감응 cap 환급·2026-07-22 §2bs) 오프라인 배선 테스트.

rall10 실측(043/050/052): chain cap3 < 사슬 6단계 — 발화가 견인한 턴까지 cap을 소모해
소진 후 잔여 사슬 무방비. 신규 스위치: 직전 chain 발화의 {missing} 중 하나라도 이후
시도-수준 호출(§2bh: 성공 불문)이 보이면 cap 소모 1회 환급.
검정: ①진행(call_discoverable 언랩·suffix strip 대조) → 환급 ②무진행 → 미환급·스냅샷 소거
③OFF → 미환급·스냅샷 보존 ④바닥 0 클램프.
⚠️단위통과≠라이브발화([[30]]) — 라이브 검정은 rall11 로그의 'chain progress refund' 태그.
"""
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
SCRIPT = []
def generate(model=None, tools=None, messages=None, call_name=None, **kw):
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

# suffixed 이름을 히스토리에 둬 PROV(출처) 게이트 충족 — 환급 로직만 고립 검정
HIST = auth() + [UserMessage("docs mention get_user_dispute_history_7291. thanks, that's all")]
PROG = lambda: ToolCall("call_discoverable_agent_tool",
                        {"agent_tool_name": "get_user_dispute_history_7291", "arguments": "{}"})
NOPROG = lambda: ToolCall("KB_search_bm25", {"query": "dispute history"})

# ① 진행 → 환급 (call_discoverable 언랩+suffix strip으로 missing과 대조)
os.environ["T2_FOLLOWUP_PROGRESS_REFUND"] = "1"
ag, orch, st = setup(HIST); SCRIPT[:] = [AM([PROG()])]
ag._t2_followup, ag._t2_chain_missing = 2, {"get_user_dispute_history"}
ag._generate_next_message(UserMessage("ok"), st)
check("R1_refund", getattr(ag, "_t2_followup", None) == 1)
check("R1_snapshot_cleared", getattr(ag, "_t2_chain_missing", "X") is None)

# ② 무진행 → 미환급·스냅샷 소거(발화당 1회 판정)
ag, orch, st = setup(HIST); SCRIPT[:] = [AM([NOPROG()])]
ag._t2_followup, ag._t2_chain_missing = 2, {"get_user_dispute_history"}
ag._generate_next_message(UserMessage("ok"), st)
check("R2_no_refund", getattr(ag, "_t2_followup", None) == 2)
check("R2_snapshot_cleared", getattr(ag, "_t2_chain_missing", "X") is None)

# ③ OFF → 미환급·스냅샷 보존(거동보존)
os.environ.pop("T2_FOLLOWUP_PROGRESS_REFUND", None)
ag, orch, st = setup(HIST); SCRIPT[:] = [AM([PROG()])]
ag._t2_followup, ag._t2_chain_missing = 2, {"get_user_dispute_history"}
ag._generate_next_message(UserMessage("ok"), st)
check("R3_off_no_refund", getattr(ag, "_t2_followup", None) == 2)
check("R3_off_snapshot_kept", getattr(ag, "_t2_chain_missing", None) == {"get_user_dispute_history"})

# ④ 바닥 0 클램프
os.environ["T2_FOLLOWUP_PROGRESS_REFUND"] = "1"
ag, orch, st = setup(HIST); SCRIPT[:] = [AM([PROG()])]
ag._t2_followup, ag._t2_chain_missing = 0, {"get_user_dispute_history"}
ag._generate_next_message(UserMessage("ok"), st)
check("R4_floor_zero", getattr(ag, "_t2_followup", None) == 0)
os.environ.pop("T2_FOLLOWUP_PROGRESS_REFUND", None)

print("\n%s" % ("ALL PASS" if not FAILS else "FAILS: %s" % FAILS))
sys.exit(1 if FAILS else 0)
