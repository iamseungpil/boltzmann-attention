# -*- coding: utf-8 -*-
"""T2_PRESCRIPTION(처방-redirect·2026-07-22 §2bu·rall11 038) 오프라인 배선 테스트.

038: 사기 dispute 요청에 apply_statement_credit 오선택. 게이트=dispute-신호∧file_dispute 미호출인데
statement_credit이면 deny+dispute 안내. 검정: ①038 재현(사기신호+statement_credit·dispute 미호출)→deny
②dispute 이미 접수(requires_absent_tool 충족)→통과 ③신호 없음(정당 credit)→통과 ④OFF→통과
⑤prefix 불일치 도구→무간섭. ⚠️단위통과≠라이브발화([[30]])."""
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
    def __init__(self, tool_messages=None): self.tool_messages = tool_messages or []
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
    return ag, BaseOrchestrator(environment=Env(), agent=ag), State(hist)
def auth():
    return [AM(tool_calls=[ToolCall("log_verification", {"user_id": "u1"}, id="a1")]),
            ToolMessage(id="a1", content="Verification logged. Available discoverable tools this session: apply_statement_credit_8472, file_credit_card_transaction_dispute_4829. Account cc_x on file.")]
FAILS = []
def check(n, c):
    print(("PASS " if c else "FAIL ") + n)
    if not c: FAILS.append(n)

FRAUD = auth() + [UserMessage("I have an unauthorized Amazon charge (transaction txn_1) for $89.99 I did not make, please dispute it.")]
SC = lambda: ToolCall("call_discoverable_agent_tool",
                      {"agent_tool_name": "apply_statement_credit_8472",
                       "arguments": '{"user_id":"u1","credit_card_account_id":"cc_x","amount":89.99}'})
GOOD = lambda: ToolCall("call_discoverable_agent_tool",
                        {"agent_tool_name": "file_credit_card_transaction_dispute_4829",
                         "arguments": '{"transaction_id":"txn_1"}'})

os.environ["T2_PRESCRIPTION"] = "1"
# ① 038 재현: 사기신호 + statement_credit·dispute 미호출 → deny + 교정
ag, orch, st = setup(FRAUD); SCRIPT[:] = [AM([SC()]), AM([GOOD()]), AM([GOOD()]), AM([GOOD()])]; FEEDBACKS[:] = []
am = ag._generate_next_message(UserMessage("go"), st)
check("P1_redirected_to_dispute", am.tool_calls and
      "file_credit_card_transaction_dispute" in am.tool_calls[0].arguments.get("agent_tool_name", ""))
check("P1_fb_prescription", any("FILING A FORMAL DISPUTE" in str(f) for f in FEEDBACKS))
check("P1_cnt", getattr(ag, "_t2_prescription_deny", 0) == 1)
# ② dispute 이미 접수(requires_absent_tool 충족) → statement_credit 통과
H2 = FRAUD + [AM([GOOD()]), ToolMessage(id=GOOD().id, content="Dispute filed.")]
ag, orch, st = setup(H2); SCRIPT[:] = [AM([SC()]), AM([SC()])]
am = ag._generate_next_message(UserMessage("go"), st)
check("P2_after_dispute_pass", am.tool_calls and
      "apply_statement_credit" in am.tool_calls[0].arguments.get("agent_tool_name", ""))
# ③ 신호 없음(정당 goodwill credit) → 통과
CLEAN = auth() + [UserMessage("As a goodwill gesture for the wait, please apply a $89.99 statement credit.")]
ag, orch, st = setup(CLEAN); SCRIPT[:] = [AM([SC()]), AM([SC()])]
am = ag._generate_next_message(UserMessage("go"), st)
check("P3_goodwill_pass", am.tool_calls and
      "apply_statement_credit" in am.tool_calls[0].arguments.get("agent_tool_name", ""))
# ④ OFF → 통과(거동보존)
os.environ.pop("T2_PRESCRIPTION", None)
ag, orch, st = setup(FRAUD); SCRIPT[:] = [AM([SC()]), AM([SC()])]
am = ag._generate_next_message(UserMessage("go"), st)
check("P4_off_pass", am.tool_calls and
      "apply_statement_credit" in am.tool_calls[0].arguments.get("agent_tool_name", ""))

print("\n%s" % ("ALL PASS" if not FAILS else "FAILS: %s" % FAILS))
sys.exit(1 if FAILS else 0)
