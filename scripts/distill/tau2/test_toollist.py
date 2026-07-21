# -*- coding: utf-8 -*-
"""T2_TOOLLIST(생성-레벨 도구목록-밖 deny·2026-07-21 §2bb) 오프라인 배선 테스트.
검정: ①nonlisted 호출 → deny 피드백(A2 nonlisted_tool_feedback 포함)·regen서 listed로 교정
②OFF=통과(거동보존) ③sim-cap 소진 후 통과(liveness) ④턴당 1라운드 후 통과 ⑤무과금.
⚠️단위통과≠라이브발화([[30]]) — 배선만 본다.
"""
import sys, os, types, json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

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

SCRIPT, GENCALLS, FEEDBACKS = [], [], []
def generate(model=None, tools=None, messages=None, call_name=None, **kw):
    GENCALLS.append(call_name)
    # regen이면 마지막 피드백(ToolMessage) 내용 채집
    for m in reversed(messages or []):
        if getattr(m, "role", "") == "tool" and getattr(m, "error", False):
            FEEDBACKS.append(m.content)
            break
    if not SCRIPT:
        raise AssertionError("SCRIPT exhausted")
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
        self.domain_name, self.tools = "banking_knowledge", None
class State:
    def __init__(self, messages):
        self.system_messages, self.messages = [], list(messages)

G.apply_unified_regen(max_prov_retries=4, domain="banking_knowledge")

A2 = json.load(open(os.path.join(HERE, "a2", "banking_knowledge.gate.json"), encoding="utf-8"))
LISTED = ["unlock_discoverable_agent_tool", "call_discoverable_agent_tool",
          "KB_search_bm25", "log_verification"]

def setup(history):
    ag = LLMAgent()
    ag.llm, ag.llm_args = "m", {}
    ag.tools = [types.SimpleNamespace(name=n) for n in LISTED]
    orch = BaseOrchestrator(environment=Env(), agent=ag)
    return ag, orch, State(history)

FAILS = []
def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (" | " + str(detail) if detail and not cond else ""))
    if not cond:
        FAILS.append(name)

def auth_pair():
    tc = ToolCall("log_verification", {"user_id": "lm83", "verified": True}, id="auth1")
    return [AM(tool_calls=[tc]), ToolMessage(id="auth1", content="Verification logged for user lm83.")]

HIST = auth_pair() + [UserMessage(
    "please check my savings account sav_lm83 transactions, id get_bank_account_transactions_9173")]
NONLISTED = lambda: ToolCall("get_bank_account_transactions_9173", {"account_id": "sav_lm83"})
DISPATCH = lambda: ToolCall("call_discoverable_agent_tool",
                            {"agent_tool_name": "get_bank_account_transactions_9173",
                             "arguments": json.dumps({"account_id": "sav_lm83"})})

# U1: ON — nonlisted deny -> regen이 디스패처 형식으로 교정
os.environ["T2_TOOLLIST"] = "1"
os.environ.pop("T2_TOOLLIST_CAP", None)
ag, orch, st = setup(HIST)
SCRIPT[:] = [AM([NONLISTED()]), AM([DISPATCH()])]
FEEDBACKS[:] = []
am = ag._generate_next_message(UserMessage("yes"), st)
check("U1_final_dispatch", am.tool_calls and am.tool_calls[0].name == "call_discoverable_agent_tool")
check("U1_deny_1", getattr(ag, "_t2_toollist_deny", 0) == 1, getattr(ag, "_t2_toollist_deny", 0))
check("U1_fb_text", any("not one of your provided tools" in str(f) for f in FEEDBACKS))
check("U1_fb_a2", any("unlock_discoverable_agent_tool" in str(f) for f in FEEDBACKS))
check("U1_no_tick", orch.num_errors == 0, orch.num_errors)

# U2: OFF — 통과(거동보존)
os.environ.pop("T2_TOOLLIST", None)
ag, orch, st = setup(HIST)
SCRIPT[:] = [AM([NONLISTED()])]
am = ag._generate_next_message(UserMessage("yes"), st)
check("U2_off_passthrough", am.tool_calls and am.tool_calls[0].name == "get_bank_account_transactions_9173")
check("U2_off_no_counter", getattr(ag, "_t2_toollist_deny", 0) == 0)

# U3: cap 소진 후 통과(liveness)
os.environ["T2_TOOLLIST"] = "1"
os.environ["T2_TOOLLIST_CAP"] = "1"
ag, orch, st = setup(HIST)
SCRIPT[:] = [AM([NONLISTED()]), AM([NONLISTED()])]   # deny 1회 후 재생성도 nonlisted -> 턴당 1라운드 소진 -> 통과
am = ag._generate_next_message(UserMessage("yes"), st)
check("U3_turn1_passthrough_after_1round", am.tool_calls
      and am.tool_calls[0].name == "get_bank_account_transactions_9173")
check("U3_deny_1", getattr(ag, "_t2_toollist_deny", 0) == 1)
SCRIPT[:] = [AM([NONLISTED()])]                       # 새 턴: cap(1) 소진 -> deny 없이 즉시 통과
am = ag._generate_next_message(UserMessage("go on"), st)
check("U3_cap_passthrough", am.tool_calls
      and am.tool_calls[0].name == "get_bank_account_transactions_9173")
check("U3_deny_still_1", getattr(ag, "_t2_toollist_deny", 0) == 1)
os.environ.pop("T2_TOOLLIST_CAP", None)
os.environ.pop("T2_TOOLLIST", None)

print("\n%s" % ("ALL PASS" if not FAILS else "FAILS: %s" % FAILS))
sys.exit(1 if FAILS else 0)
