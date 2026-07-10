# -*- coding: utf-8 -*-
"""Unit tests for apply_unified_regen (E-COMP, review blockers 1+2).

tau2 stubs injected into sys.modules -> drives the real generation loop locally.
Covers: per-lever budgets (gate 1 ticked round / prov 4 unticked), R8 strip,
same-call double-feedback suppression, combined round, DISAMB gate-recheck
(blocker 2), DISAMB accept path, retail A2 regression via real gate.json.
"""
import sys, os, types, json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ["T2_GATE_KINDS"] = "auth,confirm,ownership,notice,preconditions,constraints"

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
msgmod.ToolCall = None  # not used by unified

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
    GENCALLS.append(call_name)
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

import t2_gate_patch as G  # noqa: E402  (imports tau2 lazily inside functions)

class Env:
    def __init__(self):
        self.domain_name, self.tools = "retail", None
class State:
    def __init__(self, messages):
        self.system_messages, self.messages = [], list(messages)

G.apply_unified_regen(max_prov_retries=4, domain="retail", disamb=True)

def setup(history):
    ag = LLMAgent()
    ag.llm, ag.llm_args, ag.tools = "m", {}, []
    orch = BaseOrchestrator(environment=Env(), agent=ag)   # init_inject wires gate
    return ag, orch, State(history)

def auth_pair():
    tc = ToolCall("find_user_id_by_email", {"email": "real@user.com"}, id="auth1")
    return [AM(tool_calls=[tc]), ToolMessage(id="auth1", content="usr_123")]

FAILS = []
def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (" | " + str(detail) if detail and not cond else ""))
    if not cond:
        FAILS.append(name)

# T1: gate deny -> regen fixes -> 1 tick, no strip
hist = [UserMessage("my order is #W1112223, email real@user.com, x@y.com")]
ag, orch, st = setup(hist)
SCRIPT[:] = [AM([ToolCall("cancel_pending_order", {"order_id": "#W1112223"})]),
             AM([ToolCall("find_user_id_by_email", {"email": "x@y.com"})])]
am = ag._generate_next_message(UserMessage("yes, please do it"), st)
check("T1_final_is_fixed", am.tool_calls and am.tool_calls[0].name == "find_user_id_by_email")
check("T1_tick_1", orch.num_errors == 1, orch.num_errors)
check("T1_gate_rounds_1", getattr(ag, "_t2_gate_rounds", 0) == 1)
check("T1_no_strip", getattr(ag, "_t2_gate_strips", 0) == 0)

# T2: persistent gate deny -> 1 tick only, R8 strip + note
ag, orch, st = setup([UserMessage("my order is #W1112223")])
SCRIPT[:] = [AM([ToolCall("cancel_pending_order", {"order_id": "#W1112223"})]),
             AM([ToolCall("cancel_pending_order", {"order_id": "#W1112223"})])]
am = ag._generate_next_message(UserMessage("yes, please do it"), st)
check("T2_stripped", not am.tool_calls)
check("T2_note", isinstance(am.content, str) and "blocked by a policy gate" in am.content)
check("T2_tick_still_1", orch.num_errors == 1, orch.num_errors)
check("T2_strips_1", getattr(ag, "_t2_gate_strips", 0) == 1)

# T3: prov fab -> regen (no tick), fixed
hist = auth_pair() + [UserMessage("cancel my order #W9990001 please")]
ag, orch, st = setup(hist)
SCRIPT[:] = [AM([ToolCall("cancel_pending_order", {"order_id": "#W0000000"})]),
             AM([ToolCall("cancel_pending_order", {"order_id": "#W9990001"})])]
am = ag._generate_next_message(UserMessage("yes, go ahead"), st)
check("T3_fixed", am.tool_calls and am.tool_calls[0].arguments["order_id"] == "#W9990001")
check("T3_no_tick", orch.num_errors == 0, orch.num_errors)
check("T3_prov_1", getattr(ag, "_t2_regen", 0) == 1)

# T4: prov fab persists -> 4 unticked rounds, pass-through
ag, orch, st = setup(auth_pair() + [UserMessage("cancel my order #W9990001")])
SCRIPT[:] = [AM([ToolCall("cancel_pending_order", {"order_id": "#W0000009"})]) for _ in range(5)]
am = ag._generate_next_message(UserMessage("yes, go ahead"), st)
check("T4_passthrough", am.tool_calls and am.tool_calls[0].arguments["order_id"] == "#W0000009")
check("T4_no_tick", orch.num_errors == 0, orch.num_errors)
check("T4_prov_4", getattr(ag, "_t2_regen", 0) == 4, getattr(ag, "_t2_regen", 0))

# T5: combined round — gate-denied call + fab on separate ungated call -> both fed back in ONE round
ag, orch, st = setup([UserMessage("my order is #W1112223, my email is real@user.com")])
SCRIPT[:] = [AM([ToolCall("cancel_pending_order", {"order_id": "#W1112223"}),
                 ToolCall("find_user_id_by_email", {"email": "fab@nowhere.com"})]),
             AM([ToolCall("find_user_id_by_email", {"email": "real@user.com"})])]
am = ag._generate_next_message(UserMessage("yes, please"), st)
check("T5_final_clean", am.tool_calls and am.tool_calls[0].arguments["email"] == "real@user.com")
check("T5_tick_1", orch.num_errors == 1, orch.num_errors)
check("T5_prov_1", getattr(ag, "_t2_regen", 0) == 1)
check("T5_one_regen_round", GENCALLS.count("agent_response_unified_regen") >= 1)

# T5b: fab call itself gate-denied -> round1 gate feedback covers it (prov NOT burned),
#      round2 prov fires on new ungated fab, round3 clean.
ag, orch, st = setup([UserMessage("hello, my email is real@user.com")])
SCRIPT[:] = [AM([ToolCall("cancel_pending_order", {"order_id": "#W5550001"})]),        # fab AND denied
             AM([ToolCall("find_user_id_by_email", {"email": "fab2@nowhere.com"})]),   # fab, ungated
             AM([ToolCall("find_user_id_by_email", {"email": "real@user.com"})])]      # clean
am = ag._generate_next_message(UserMessage("yes"), st)
check("T5b_final_clean", am.tool_calls and am.tool_calls[0].arguments["email"] == "real@user.com")
check("T5b_gate_covered_first", getattr(ag, "_t2_gate_rounds", 0) == 1)
check("T5b_prov_used_once", getattr(ag, "_t2_regen", 0) == 1, getattr(ag, "_t2_regen", 0))
check("T5b_tick_1", orch.num_errors == 1, orch.num_errors)

# T6: DISAMB fires; switch introduces gate-denied call -> rejected, original kept (blocker 2)
hist = auth_pair() + [
    UserMessage("my order is #W1112223, exchange to what I asked"),
    ToolMessage(id="r1", content=json.dumps({"payment_method_id": "credit_card_1111111"})),
    ToolMessage(id="r2", content=json.dumps({"payment_method_id": "gift_card_2222222"})),
    ToolMessage(id="r3", content=json.dumps({"items": [{"item_id": "1234567890"}]})),
]
ag, orch, st = setup(hist)
orig = AM([ToolCall("modify_pending_order_payment",
                    {"order_id": "#W1112223", "payment_method_id": "credit_card_1111111"})])
bad_switch = AM([ToolCall("modify_pending_order_items",
                          {"order_id": "#W1112223", "item_ids": ["1234567890"],
                           "new_item_ids": ["1234567890"]})])  # constraints disjoint violation
SCRIPT[:] = [orig, bad_switch]
am = ag._generate_next_message(UserMessage("yes, confirm"), st)
check("T6_disamb_fired", getattr(ag, "_t2_disamb", 0) == 1)
check("T6_rejected", getattr(ag, "_t2_disamb_gate_reject", 0) == 1)
check("T6_original_kept", am.tool_calls and am.tool_calls[0].name == "modify_pending_order_payment"
      and am.tool_calls[0].arguments["payment_method_id"] == "credit_card_1111111")
check("T6_no_tick", orch.num_errors == 0, orch.num_errors)

# T7: DISAMB accept — clean switch adopted
ag, orch, st = setup([m for m in hist])
good_switch = AM([ToolCall("modify_pending_order_payment",
                           {"order_id": "#W1112223", "payment_method_id": "gift_card_2222222"})])
SCRIPT[:] = [AM([ToolCall("modify_pending_order_payment",
                          {"order_id": "#W1112223", "payment_method_id": "credit_card_1111111"})]),
             good_switch]
am = ag._generate_next_message(UserMessage("yes, confirm"), st)
check("T7_switch_adopted", am.tool_calls and am.tool_calls[0].arguments["payment_method_id"] == "gift_card_2222222")
check("T7_no_tick", orch.num_errors == 0, orch.num_errors)

print("\n%d FAIL" % len(FAILS) if FAILS else "\nALL PASS")
sys.exit(1 if FAILS else 0)
