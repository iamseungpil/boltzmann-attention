# -*- coding: utf-8 -*-
"""action-required 리마인더 채널 라이브-배선 offline 검증 (2026-07-13 LATE 핸드오프 #1).

순수-조언 회피(tool_call 0개)는 앵커할 ToolMessage가 없다 → UserMessage 리마인더 채널로
재생성돼야 한다. tau2 stub으로 실제 apply_unified_regen 루프를 오프라인·무료 구동하고,
재생성(agent_response_unified_regen) 호출의 work 버퍼에 ACTION 피드백이 실제 전달됐는지 직접 확인.
Δspurious: 조회 등 다른 도구 호출 중(회피 아님)엔 발화하지 않아야 한다.
"""
import sys, os, types, json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ["T2_GATE_KINDS"] = "auth,confirm"
os.environ["T2_RESOLVE"] = "1"     # ★통일 인터프리터 경로(action-required 포함) 활성

# ---------- tau2 stubs ----------
def mkmod(name):
    m = types.ModuleType(name); sys.modules[name] = m; return m

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

SCRIPT, GENCALLS = [], []          # GENCALLS: (call_name, messages) — work 버퍼 감사용
def generate(model=None, tools=None, messages=None, call_name=None, **kw):
    GENCALLS.append((call_name, list(messages or [])))
    if not SCRIPT:
        raise AssertionError("SCRIPT exhausted (unexpected extra generate call)")
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

# banking_knowledge = 자연스러운 action-required 케이스(apply 회피·task_001/003/007형·핸드오프 §3)
G.apply_unified_regen(max_prov_retries=4, domain="banking_knowledge", disamb=False)

def setup(history):
    ag = LLMAgent(); ag.llm, ag.llm_args, ag.tools = "m", {}, []
    orch = BaseOrchestrator(environment=Env(), agent=ag)
    return ag, orch, State(history)

FAILS = []
def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (" | " + str(detail) if detail and not cond else ""))
    if not cond: FAILS.append(name)

def regen_user_msgs():
    """재생성 호출들의 work 버퍼에서 UserMessage content 목록 (리마인더 전달 감사)."""
    out = []
    for cn, msgs in GENCALLS:
        if cn == "agent_response_unified_regen":
            out += [m.content for m in msgs if getattr(m, "role", None) == "user"]
    return out

# ── T1: 순수-조언 회피 + 의도 해소됨(FIND) → ACTION-REQUIRED 리마인더 재생성 ──
GENCALLS[:] = []
ag, orch, st = setup([UserMessage("I want to apply for a new credit card.")])
SCRIPT[:] = [
    AM(content="To apply, please visit our website and go to the credit cards page."),  # 순수-조언 회피
    AM(content='{"tool": "apply_for_credit_card"}'),                                     # intent formalize 서브콜
    AM(tool_calls=[ToolCall("apply_for_credit_card", {"card_type": "visa"})]),           # 재생성=실행
]
am = ag._generate_next_message(UserMessage("please do it"), st)
check("T1_action_fired", getattr(ag, "_t2_action_deny", 0) == 1, getattr(ag, "_t2_action_deny", 0))
check("T1_regen_happened", any(cn == "agent_response_unified_regen" for cn, _ in GENCALLS))
_rum = regen_user_msgs()
check("T1_reminder_delivered", any("[ACTION-REQUIRED]" in (u or "") for u in _rum), _rum)
check("T1_target_named", any("apply_for_credit_card" in (u or "") for u in _rum), _rum)
check("T1_final_calls_action", am.tool_calls and am.tool_calls[0].name == "apply_for_credit_card")
check("T1_no_tick", orch.num_errors == 0, orch.num_errors)   # prov류=무과금·리마인더도 무과금

# ── T2: 순수-조언 회피 + 의도 미해소(target None) → ACTION-ASK 리마인더 ──
GENCALLS[:] = []
ag, orch, st = setup([UserMessage("can you help me with something unusual?")])
SCRIPT[:] = [
    AM(content="I'm not sure, maybe check the website."),   # 순수-조언 회피
    AM(content='{"tool": "none"}'),                          # intent=none → target None → ACTION-ASK
    AM(content="What specific detail do you need me to act on?"),  # 재생성(개방질문)
]
am = ag._generate_next_message(UserMessage("go on"), st)
check("T2_action_fired", getattr(ag, "_t2_action_deny", 0) == 1)
check("T2_ask_reminder", any("[ACTION-ASK]" in (u or "") for u in regen_user_msgs()), regen_user_msgs())

# ── T3 (Δspurious): 다른 도구 호출 중(회피 아님) = 발화 0·재생성 0 ──
#   log_verification=게이트 비대상·action tool 아님·인자 문맥 접지 → 깨끗이 통과(action 침묵 격리)
GENCALLS[:] = []
ag, orch, st = setup([UserMessage("I'm Jane, user id u1, verifying my identity for support")])
SCRIPT[:] = [AM(tool_calls=[ToolCall("log_verification", {"name": "Jane", "user_id": "u1"})])]
am = ag._generate_next_message(UserMessage("here are my details"), st)
check("T3_no_fire", getattr(ag, "_t2_action_deny", 0) == 0, getattr(ag, "_t2_action_deny", 0))
check("T3_no_regen", not any(cn == "agent_response_unified_regen" for cn, _ in GENCALLS))
check("T3_call_kept", am.tool_calls and am.tool_calls[0].name == "log_verification")

# ── T4 (cap): 재생성돼도 여전히 조언이면 cap=1 → 무한루프 없이 종결 ──
GENCALLS[:] = []
ag, orch, st = setup([UserMessage("apply for a credit card")])
SCRIPT[:] = [
    AM(content="Please visit the website."),          # 조언
    AM(content='{"tool": "apply_for_credit_card"}'),   # formalize
    AM(content="You can do it online yourself."),      # 재생성=여전히 조언 → cap 도달, 재발화 X
]
am = ag._generate_next_message(UserMessage("do it"), st)
check("T4_cap_1", getattr(ag, "_t2_action_deny", 0) == 1)
check("T4_terminates", am.content == "You can do it online yourself.")
check("T4_one_regen", sum(1 for cn, _ in GENCALLS if cn == "agent_response_unified_regen") == 1)

print("\n%d FAIL" % len(FAILS) if FAILS else "\nALL PASS (action-required reminder channel)")
sys.exit(1 if FAILS else 0)
