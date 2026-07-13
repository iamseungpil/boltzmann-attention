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

# ── T1: 순수-조언 회피 + 의도 해소됨(FIND, agent-실행 도구) → ACTION-REQUIRED 재생성 ──
#   ★Lever 0: target=change_user_email(agent-실행)·apply(user-실행)는 필터돼 대상 아님
GENCALLS[:] = []
#   (툴명 reset_pin_tool_1234을 문맥에 접지 — prov 날조검출과 격리해 리마인더 채널만 검정)
ag, orch, st = setup([
    UserMessage("Please unlock the reset_pin_tool_1234 procedure for me."),
    AM(content="Let me check."), ToolMessage(id="k1", content="discovered: reset_pin_tool_1234"),
])
SCRIPT[:] = [
    AM(content="I can point you to the right procedure on our website for that."),  # 조언 회피
    AM(content='{"tool": "unlock_discoverable_agent_tool"}'),                        # intent formalize 서브콜
    AM(tool_calls=[ToolCall("unlock_discoverable_agent_tool", {"agent_tool_name": "reset_pin_tool_1234"})]),  # 재생성=실행
]
am = ag._generate_next_message(UserMessage("please do it"), st)
check("T1_action_fired", getattr(ag, "_t2_action_deny", 0) == 1, getattr(ag, "_t2_action_deny", 0))
check("T1_regen_happened", any(cn == "agent_response_unified_regen" for cn, _ in GENCALLS))
_rum = regen_user_msgs()
# unlock=discoverable dispatcher → Lever 2 discovery-required 피드백(발견체인 안내)
check("T1_reminder_delivered", any("[DISCOVERY-REQUIRED]" in (u or "") for u in _rum), _rum)
check("T1_target_named", any("unlock_discoverable_agent_tool" in (u or "") for u in _rum), _rum)
check("T1_final_calls_action", am.tool_calls and am.tool_calls[0].name == "unlock_discoverable_agent_tool")
check("T1_no_tick", orch.num_errors == 0, orch.num_errors)   # prov류=무과금·리마인더도 무과금

# ── T2 (고정밀): 회피이나 의도 미해소(target None) → 미발화(action-ask 스퓨리어스 방지) ──
GENCALLS[:] = []
ag, orch, st = setup([UserMessage("can you help me with something unusual?")])
SCRIPT[:] = [
    AM(content="I'm not sure, maybe check the website."),   # 순수-조언 회피
    AM(content='{"tool": "none"}'),                          # intent=none → target None → 미발화
]
am = ag._generate_next_message(UserMessage("go on"), st)
check("T2_no_fire_on_none", getattr(ag, "_t2_action_deny", 0) == 0, getattr(ag, "_t2_action_deny", 0))
check("T2_no_regen", not any(cn == "agent_response_unified_regen" for cn, _ in GENCALLS))

# ── T2b (Lever 0): user-실행 의도(apply)만 회피 → agent-실행 target 없음 → 무발화(스퓨리어스 방지) ──
GENCALLS[:] = []
ag, orch, st = setup([UserMessage("I want to apply for a credit card.")])
SCRIPT[:] = [
    AM(content="To apply, please visit our website credit-cards page."),  # 조언
    AM(content='{"tool": "none"}'),   # formalize(agent-실행 도구 중 apply-의도 매칭 없음)=none → 미발화
]
am = ag._generate_next_message(UserMessage("please"), st)
check("T2b_no_fire_on_user_exec", getattr(ag, "_t2_action_deny", 0) == 0, getattr(ag, "_t2_action_deny", 0))
check("T2b_no_regen", not any(cn == "agent_response_unified_regen" for cn, _ in GENCALLS))

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
ag, orch, st = setup([UserMessage("please update my email")])
SCRIPT[:] = [
    AM(content="Please visit the website."),          # 조언
    AM(content='{"tool": "unlock_discoverable_agent_tool"}'),  # formalize (agent-실행)
    AM(content="You can do it online yourself."),      # 재생성=여전히 조언 → cap 도달, 재발화 X
]
am = ag._generate_next_message(UserMessage("do it"), st)
check("T4_cap_1", getattr(ag, "_t2_action_deny", 0) == 1)
check("T4_terminates", am.content == "You can do it online yourself.")
check("T4_one_regen", sum(1 for cn, _ in GENCALLS if cn == "agent_response_unified_regen") == 1)

# ── T5 (Lever 3 live): 신원수집(get_user_information_by_*)+검증미완+조언포기 → verify-persistence 재생성 ──
GENCALLS[:] = []
ag, orch, st = setup([
    UserMessage("I want to apply for a credit card. I'm Kenji, dob 1990-01-01, email k@x.com."),
    AM(tool_calls=[ToolCall("get_user_information_by_date_of_birth", {"date_of_birth": "1990-01-01"})]),
    ToolMessage(id="g1", content="user record: Kenji"),
])
SCRIPT[:] = [
    AM(content="To apply, please visit our website — I'll transfer you if needed."),  # 검증미완 조언포기
    AM(content='{"tool": "none"}'),         # action-required formalize=none(apply=user-실행) → 미발화
    AM(tool_calls=[ToolCall("log_verification", {"name": "Kenji", "date_of_birth": "1990-01-01"})]),  # 재생성=검증
]
am = ag._generate_next_message(UserMessage("please proceed"), st)
check("T5_verify_fired", getattr(ag, "_t2_verify_deny", 0) == 1, getattr(ag, "_t2_verify_deny", 0))
check("T5_reminder_delivered",
      any("[VERIFY-PERSISTENCE]" in (u or "") for u in regen_user_msgs()), regen_user_msgs())
check("T5_action_not_fired", getattr(ag, "_t2_action_deny", 0) == 0)  # apply=user-실행이라 action-req 무발화

# ── T6 (Lever 4 live·오추천): KB_search 후 텍스트로 틀린 카드 추천+종결 → recommendation-offer 재생성 ──
GENCALLS[:] = []
ag, orch, st = setup([
    UserMessage("I want a credit card with no foreign transaction fees and purchase protection."),
    AM(tool_calls=[ToolCall("KB_search", {"query": "cards no foreign fee"})]),
    ToolMessage(id="k1", content="Silver Rewards Card: no foreign fee, purchase protection. "
                                 "Platinum Rewards Card: has foreign fee."),
])
SCRIPT[:] = [
    AM(content="I recommend the Platinum Rewards Card for you."),        # 텍스트 오추천+종결
    AM(content='{"applies": true, "card_type": "Silver Rewards Card"}'),  # recommend formalize
    AM(content="Actually, the Silver Rewards Card fits — let me offer it."),  # 재생성
]
am = ag._generate_next_message(UserMessage("which card?"), st)
check("T6_recommend_fired", getattr(ag, "_t2_recommend_deny", 0) == 1, getattr(ag, "_t2_recommend_deny", 0))
check("T6_offer_reminder",
      any("[RECOMMEND-OFFER]" in (u or "") and "Silver Rewards Card" in (u or "")
          for u in regen_user_msgs()), regen_user_msgs())

# ── T7 (reference-filter live·silent-repair): 틀린 transaction_id 지목 → formalize→filter→제자리 치환 ──
GENCALLS[:] = []
RECTXT = ("Transactions:\n1. Record ID: btxn_atm\n   transaction_id: btxn_atm\n   date: 11/05/2025\n"
          "   description: RHO-BANK ATM #4827 WITHDRAWAL\n   amount: -300.0\n   type: atm_withdrawal\n"
          "2. Record ID: btxn_gym1\n   transaction_id: btxn_gym1\n   date: 11/06/2025\n"
          "   description: CITYFIT GYM\n   amount: -89.99\n   type: debit_card_purchase")
ag, orch, st = setup([
    UserMessage("Dispute my ATM withdrawal on November 5th, I'm short $100."),
    AM(tool_calls=[ToolCall("KB_search", {})]),
    ToolMessage(id="k0", content="Use file_debit_card_transaction_dispute_6281 to file disputes."),
    AM(tool_calls=[ToolCall("get_bank_account_transactions", {})]),
    ToolMessage(id="r1", content=RECTXT),
])
_wrongdisp = AM(tool_calls=[ToolCall("call_discoverable_agent_tool",
                {"agent_tool_name": "file_debit_card_transaction_dispute_6281",
                 "arguments": json.dumps({"transaction_id": "btxn_gym1",  # 틀린 id
                                          "dispute_category": "atm_cash_discrepancy"})})])
SCRIPT[:] = [
    _wrongdisp,
    AM(content='{"date": "11/05/2025", "merchant": "ATM", "transaction_type": "atm_withdrawal"}'),  # formalize
]
am = ag._generate_next_message(UserMessage("please file it"), st)
# silent-repair: nested transaction_id가 btxn_atm으로 치환됐나
_nested = {}
for tc in (am.tool_calls or []):
    if tc.name == "call_discoverable_agent_tool":
        try: _nested = json.loads(tc.arguments["arguments"])
        except Exception: _nested = {}
check("T7_reffilter_fired", getattr(ag, "_t2_reffilter", 0) == 1, getattr(ag, "_t2_reffilter", 0))
check("T7_silent_repair", _nested.get("transaction_id") == "btxn_atm", _nested)

print("\n%d FAIL" % len(FAILS) if FAILS else "\nALL PASS (action + verify + recommend + reference-filter live)")
sys.exit(1 if FAILS else 0)
