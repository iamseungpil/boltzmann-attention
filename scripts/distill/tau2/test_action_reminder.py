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

# ★스텁 충실도 수리 (2026-08-19). 구판은 `ag.tools = []` 였다 — 이 테스트를 쓴 2026-07-13
#   당시엔 실행 주체가 A2 `action_tool_executor` 맵에서 왔으므로 빈 목록이어도 무해했다.
#   **2026-07-31 `c801a3e2`/`23c6063a`** 가 그 A2 키를 지우고 판정을 *인터페이스 구조*로 옮기면서
#   (`t2_gate_patch.py:7340` `_agent_names = {t.name for t in self.tools}` ·
#    `:7352 _exec_side()` · `t2_role.py:50`), 빈 목록 = **모든 action_tool 이 user-실행** 이 됐다.
#   결과: `t2_gate_patch.py:8016` 의 `_acts` 가 공집합이라 `resolve_action_operator`(=Lever 2
#   [DISCOVERY-REQUIRED] 의 유일한 생산자·`t2_resolve.py:504`)가 **호출조차 되지 않고**,
#   대신 :7453 손님-실행 분기의 `[ACTION] ... run by the CUSTOMER` 가 배달됐다.
#   엔진 계약은 멀쩡하다 — 2026-08-19 라이브 로그에 `action-required reason=discovery-required`
#   **460건**. 틀린 것은 스텁뿐이었다. ⇒ 실환경 목록을 축자로 박는다.
# 출처(축자): `registry.get_env_constructor("banking_knowledge")().get_tools()` — 2026-08-19 실측.
#   여기 **없는** 이름(apply_for_credit_card·submit_referral·submit_transaction·
#   call_discoverable_user_tool)이 곧 user-실행이고, 그것이 T2b 가 재는 Lever 0 이다.
AGENT_TOOLS = [
    "KB_search", "call_discoverable_agent_tool", "change_user_email",
    "get_credit_card_accounts_by_user", "get_credit_card_transactions_by_user",
    "get_current_time", "get_referrals_by_user", "get_user_information_by_email",
    "get_user_information_by_id", "get_user_information_by_name",
    "give_discoverable_user_tool", "list_discoverable_agent_tools", "log_verification",
    "transfer_to_human_agents", "unlock_discoverable_agent_tool",
]


class Tool:            # `_exec_side`/`t2_role` 가 읽는 것은 `.name` 뿐이다
    def __init__(self, name): self.name = name


def setup(history, agent_tools=None):
    ag = LLMAgent(); ag.llm, ag.llm_args = "m", {}
    ag.tools = [Tool(n) for n in (AGENT_TOOLS if agent_tools is None else agent_tools)]
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

# ── T5 (Lever 3 — **死배선**·2026-08-19 확인): 신원수집+검증미완+조언포기 = verify-persistence 의
#     트리거 모양 그대로이나, 선언이 없어 발화할 수 없다. 사유·증거는 아래 `_generate_next_message` 뒤. ──
GENCALLS[:] = []
ag, orch, st = setup([
    UserMessage("I want to apply for a credit card. I'm Kenji, dob 1990-01-01, email k@x.com."),
    AM(tool_calls=[ToolCall("get_user_information_by_date_of_birth", {"date_of_birth": "1990-01-01"})]),
    ToolMessage(id="g1", content="user record: Kenji"),
])
SCRIPT[:] = [
    AM(content="To apply, please visit our website — I'll transfer you if needed."),  # 검증미완 조언포기
    AM(content='{"tool": "none"}'),         # action-required formalize=none(apply=user-실행) → 미발화
    # ↓레버가 무장돼 있었다면 재생성이 여기서 검증을 불렀을 자리. 지금은 소비되지 않는다(미소진 OK).
    AM(tool_calls=[ToolCall("log_verification", {"name": "Kenji", "date_of_birth": "1990-01-01"})]),
]
am = ag._generate_next_message(UserMessage("please proceed"), st)
# ★계약 변경 2026-08-06(`5fad930c`) — **Lever 3 은 지금 배선이 끊겨 있다**. 위 시나리오는
#   여전히 이 레버의 정확한 트리거 모양이지만, 발화할 수 없다:
#     ⓐ `t2_resolve.py:668` 이 `g.get("verify_gather_prefix")` 를 읽고
#     ⓑ `:670` 에서 `if not prefix or not sats: continue` 로 건너뛴다
#     ⓒ 그런데 그 키는 **어느 A2 에도 없다** — banking/airline/retail 의 auth 게이트 3개가
#        전부 `verify_gather_prefix=None`(2026-08-19 전수). ⇒ `:681` 무조건 `{"status":"ok"}`.
#   `5fad930c` 는 그 키를 A2 에서 지우면서 커밋 문면에 *"Nothing consumed it"* 이라고 적었는데,
#   그것이 사실이 아니었다. 그 진술은 `t2_phase.py:67`(문자열을 `set()` 으로 감싸 **문자 15개의
#   집합**을 만들던 진짜 死코드) 에 대해서만 참이고, `resolve_verify_persistence` 는 같은 키를
#   **정상적으로** 소비하고 있었다. 한 소비자의 버그를 고치면서 다른 소비자의 선언을 같이 지운 것이다.
#   라이브 증거: `verify-persistence deny` 는 `bank_smk_gpu0_20260806c.log`(08-06 03:26)를
#   마지막으로 끊겼고 — 커밋 시각 03:40 의 14분 전이다 — 이후 2026-08-19 런까지 **0건**.
#   ⚠엔진을 고쳐 통과시키지 않는다(사용자 지시). A2 에 키를 되살리는 것 = 레버 재무장이고,
#     그것은 별도 측정이 선행돼야 한다([[55]] 마크≠전달 — 재무장하면 배달까지 다시 재야 한다).
#   ⇒ 여기서는 **현재 계약을 그대로 적는다**. ⓐ가 뒤집히는 순간(=누가 키를 되살리면) 아래
#     첫 검정이 실패해 이 자리로 돌아오게 되어 있다 — 死배선을 계약으로 굳히지 않는다.
import gate_interpreter as _GI                                             # noqa: E402
_a2v = _GI.load_domain_a2("banking_knowledge") or {}
_vgp = [g.get("verify_gather_prefix") for g in (_a2v.get("gates") or []) if g.get("kind") == "auth"]
check("T5_lever_unarmed(dead-wiring·5fad930c 2026-08-06)", not any(_vgp),
      "verify_gather_prefix=%s — 되살아났다면 t2_resolve.py:668 소비자와 배달을 다시 재라" % (_vgp,))
check("T5_verify_silent_while_unarmed", getattr(ag, "_t2_verify_deny", 0) == 0,
      getattr(ag, "_t2_verify_deny", 0))
check("T5_no_reminder_while_unarmed",
      not any("[VERIFY-PERSISTENCE]" in (u or "") for u in regen_user_msgs()), regen_user_msgs())
check("T5_action_not_fired", getattr(ag, "_t2_action_deny", 0) == 0)  # apply=user-실행이라 action-req 무발화

# ── T6 (Lever 4 live·오추천): KB_search_bm25(=spec research_tool·실환경명) 후 텍스트로 틀린 카드
#     추천+종결 → recommendation-offer 재생성 ──
GENCALLS[:] = []
ag, orch, st = setup([
    UserMessage("I want a credit card with no foreign transaction fees and purchase protection."),
    AM(tool_calls=[ToolCall("KB_search_bm25", {"query": "cards no foreign fee"})]),
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

# ── T7 (reference-filter live): 틀린 transaction_id 지목 → formalize→filter→**거부·표면화**(치환 없음) ──
#   ★검증완료 고객(log_verification 이력)이라 auth 게이트(GB1) 만족 → 게이트 재생성 없이 단일 패스로
#     reference-filter만 격리 검정. reference-filter는 do_gate/do_prov와 독립이나(배선 교정 2026-07-14),
#     이 케이스는 게이트-무관 단일 패스로 silent-repair 결과를 직접 확인.
GENCALLS[:] = []
RECTXT = ("Transactions:\n1. Record ID: btxn_atm\n   transaction_id: btxn_atm\n   date: 11/05/2025\n"
          "   description: RHO-BANK ATM #4827 WITHDRAWAL\n   amount: -300.0\n   type: atm_withdrawal\n"
          "2. Record ID: btxn_gym1\n   transaction_id: btxn_gym1\n   date: 11/06/2025\n"
          "   description: CITYFIT GYM\n   amount: -89.99\n   type: debit_card_purchase")
ag, orch, st = setup([
    UserMessage("Dispute my ATM withdrawal on November 5th, I'm short $100."),
    AM(tool_calls=[ToolCall("get_user_information_by_date_of_birth", {"date_of_birth": "1988-03-02"})]),
    ToolMessage(id="v0", content="user record: Alex, dob 1988-03-02, email a@x.com"),
    AM(tool_calls=[ToolCall("log_verification", {"name": "Alex", "date_of_birth": "1988-03-02"}, id="v1")]),
    ToolMessage(id="v1", content="verification logged for Alex"),   # ← id 매칭(페어링) → auth 게이트(GB1) 만족
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
    # ★2026-08-19: 치환이 없어졌으므로 거부 → **재생성**이 한 번 더 돈다. 재생성에서도 모델이
    #   같은(틀린) id 를 고집하게 두어, 엔진이 몰래 고치지 않았음을 최종 인자로 확인한다.
    _wrongdisp,
]
am = ag._generate_next_message(UserMessage("please file it"), st)
# ★계약 변경 2026-08-19(사용자 결정·A안): reference-filter 는 **제자리 치환을 하지 않는다**.
#   엔진이 옳은 엔티티를 골라 써 넣으면 측정 대상이 사라진다([[62]]). 남는 것은 닫힌 술어 하나
#   (지목한 id 가 읽은 레코드에 있는가)와 **옳은 값을 말하지 않는 거부 문면**뿐이다([[64]]).
_nested = {}
for tc in (am.tool_calls or []):
    if tc.name == "call_discoverable_agent_tool":
        try: _nested = json.loads(tc.arguments["arguments"])
        except Exception: _nested = {}
check("T7_reffilter_fired", getattr(ag, "_t2_reffilter", 0) == 1, getattr(ag, "_t2_reffilter", 0))
check("T7_no_silent_repair", _nested.get("transaction_id") != "btxn_atm", _nested)
# ★배달 채널 확인 2026-08-19: reference 거부는 **도구-결과(Error:) 채널**로 간다 —
#   user 리마인더 채널이 아니다. 재생성 호출에 실린 **모든** 메시지 본문을 훑는다([[55]] 마크≠전달).
_body = " ".join(str(getattr(m, "content", "") or "")
                 for cn, msgs in GENCALLS if cn == "agent_response_unified_regen" for m in msgs)
_ref_msgs = [str(getattr(m, "content", "") or "")
             for cn, msgs in GENCALLS if cn == "agent_response_unified_regen"
             for m in msgs if "[REFERENCE]" in str(getattr(m, "content", "") or "")]
check("T7_surfaced_reason", bool(_ref_msgs), _body[:160])
# ⚠누출 검사는 **우리가 넣은 문면만** 본다 — 대화 본문에는 레코드가 있으니 전체를 훑으면
#   우리 잘못이 아닌 것을 잡는다(2026-08-19 자기 계기 오류 교정).
check("T7_answer_not_leaked", all("btxn_atm" not in m for m in _ref_msgs), _ref_msgs[:1])

print("\n%d FAIL" % len(FAILS) if FAILS
      else "\nALL PASS (action + recommend + reference-filter live · verify=死배선 확인)")
sys.exit(1 if FAILS else 0)
