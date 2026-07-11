# -*- coding: utf-8 -*-
"""V1 unit tests for T5-C silent repair (T5C_SILENT_REPAIR_DESIGN_2026_07_11 §7 V1).

tau2 stubs injected into sys.modules -> drives the real generation paths locally.
Covers: (B1) _subst_arg_value element-wise semantics incl. str-JSON args, (P-A) GROUND
in-place substitution in both branches, (P-B) subcall fire/switch/unsure/whitelist +
turn-preservation invariant, (P-C) rescue pass-through / error-loop / free-text regen
with neutral feedback, (N3) subcall answer parsing, (B3) env-verified args from real A2,
regression: dialog-mode DISAMB text-only keep (fix 07337a3) and default flags = v1.
"""
import sys, os, types, json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ["T2_GATE_KINDS"] = "auth,confirm,ownership,notice,preconditions,constraints"

# ---------- tau2 stubs (test_unified_regen.py와 동일 패턴) ----------
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
    def __init__(self, content="", role="user"):
        self.role, self.content, self.is_audio = role, content, False
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

SCRIPT, GENCALLS, GENKW = [], [], []
def generate(model=None, tools=None, messages=None, call_name=None, **kw):
    GENCALLS.append(call_name)
    GENKW.append({"tools": tools, "messages": messages, "kw": kw})
    if not SCRIPT:
        raise AssertionError("SCRIPT exhausted (unexpected generate: %s)" % call_name)
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
        self.domain_name, self.tools = "retail", None
class State:
    def __init__(self, messages):
        self.system_messages, self.messages = [], list(messages)

FAILS = []
def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + ((" | " + str(detail)) if (detail and not cond) else ""))
    if not cond:
        FAILS.append(name)

def reset(*ams):
    SCRIPT[:] = list(ams); GENCALLS[:] = []; GENKW[:] = []

def agent():
    ag = LLMAgent()
    ag.llm, ag.llm_args, ag.tools = "m", {"temperature": 0.0, "tool_choice": "auto"}, []
    return ag

# ══════════════ 헬퍼 단위 (B1·N3·B3·N6·N4) ══════════════

# _subst_arg_value
tc = ToolCall("t", {"order_id": "#W1"})
check("H1_scalar", G._subst_arg_value(tc, "order_id", "#W1", "#W2") and tc.arguments["order_id"] == "#W2")
tc = ToolCall("t", {"item_ids": ["a111", "b222", "c333"]})
check("H2_list_positional", G._subst_arg_value(tc, "item_ids", "b222", "x999")
      and tc.arguments["item_ids"] == ["a111", "x999", "c333"])
tc = ToolCall("t", {"item_ids": ["a111", "b222"]})
check("H3_list_dup_noop", not G._subst_arg_value(tc, "item_ids", "b222", "a111")
      and tc.arguments["item_ids"] == ["a111", "b222"])
tc = ToolCall("t", {"item_ids": ["a111", "a111"]})
check("H4_list_multihit_noop", not G._subst_arg_value(tc, "item_ids", "a111", "x999"))
tc = ToolCall("t", {"nested": {"a": "b"}})
check("H5_dict_noop", not G._subst_arg_value(tc, "nested", "b", "c"))
tc = ToolCall("t", json.dumps({"order_id": "#W1"}))
ok = G._subst_arg_value(tc, "order_id", "#W1", "#W2")
check("H6_strjson_reassign", ok and isinstance(tc.arguments, dict) and tc.arguments["order_id"] == "#W2")
tc = ToolCall("t", {"order_id": "#W1"})
check("H7_wrong_old_noop", not G._subst_arg_value(tc, "order_id", "#WX", "#W2"))

# _parse_subcall_answer
CANDS = ["1111111111", "2222222222"]
check("H8_whole_exact", G._parse_subcall_answer("'2222222222'", CANDS) == "2222222222")
check("H9_boundary_unique", G._parse_subcall_answer("The user means 2222222222 here.", CANDS) == "2222222222")
check("H10_multi_none", G._parse_subcall_answer("either 1111111111 or 2222222222", CANDS) is None)
check("H11_empty_none", G._parse_subcall_answer("", CANDS) is None)
check("H12_unsure_none", G._parse_subcall_answer("UNSURE", CANDS) is None)
check("H13_substring_guard", G._parse_subcall_answer("pick 12222222222!", CANDS) is None)  # 경계 위반=매치 금지

# _env_verified_args from real retail A2 (B3)
a2r = G._domain_a2("retail")
ev = G._env_verified_args(a2r)
check("H14_env_args_order", "order" in ev, ev)

# _in_error_loop (N6 id-join)
h = [AM(tool_calls=[ToolCall("modify_pending_order_items", {}, id="x1")]),
     ToolMessage(id="x1", error=True, content="Error: not found")]
check("H15_errloop_true", G._in_error_loop(h, "modify_pending_order_items"))
check("H16_errloop_other_tool", not G._in_error_loop(h, "cancel_pending_order"))

# _parse_tool_outputs lenient (N4) + _candidate_records
aug = json.dumps({"items": [{"item_id": "1111111111"}, {"item_id": "2222222222"}]}) + "\n\n[augment note]"
h2 = [ToolMessage(id="y1", content=aug)]
check("H17_strict_misses", not any(isinstance(o, dict) for o in G._parse_tool_outputs(h2)))
check("H18_lenient_parses", any(isinstance(o, dict) for o in G._parse_tool_outputs(h2, lenient=True)))
recs = G._candidate_records("item_ids", "1111111111", h2)
check("H19_records_2_with_snippet", len(recs) == 2 and all(sn for _, sn in recs), recs)

# ══════════════ prov 분기 (apply_provenance_regen) ══════════════

TOOLOUT = json.dumps({"order_id": "#W1234567", "items": [
    {"item_id": "1111111111", "name": "Lamp A"},
    {"item_id": "2222222222", "name": "Lamp B"},
    {"item_id": "3333333333", "name": "Lamp C"}]})
def hist():
    return [UserMessage("exchange item in order #W1234567, pay with pm_9012345"),
            AM(tool_calls=[ToolCall("get_order_details", {"order_id": "#W1234567"}, id="g1")]),
            ToolMessage(id="g1", content=TOOLOUT)]

# P-B subcall: fire -> switch (item = whitelist)
G.apply_provenance_regen(max_retries=4, use_badwords=False, ground=False, domain="retail",
                         disamb=True, disamb_mode="subcall")
ag, st = agent(), State(hist())
w = ToolCall("exchange_delivered_order_items",
             {"order_id": "#W1234567", "item_ids": ["1111111111"],
              "new_item_ids": ["2222222222"], "payment_method_id": "pm_9012345"})
orig_am = AM(tool_calls=[w])
reset(orig_am, AM(content="2222222222"))
am = ag._generate_next_message(UserMessage("yes please"), st)
check("P1_turn_preserved", am is orig_am and am.tool_calls and am.tool_calls[0] is w)
check("P2_switched", w.arguments["item_ids"] == ["2222222222"], w.arguments)
check("P3_gencalls", GENCALLS == ["agent_response", "disamb_subcall"], GENCALLS)
check("P4_subcall_tools_none", GENKW[1]["tools"] is None)
check("P5_subcall_no_toolkw", "tool_choice" not in GENKW[1]["kw"], GENKW[1]["kw"])
check("P6_counter_switch", getattr(ag, "_t2_subcall_switch", 0) == 1)
sub_prompt = GENKW[1]["messages"][0].content
check("P7_records_in_prompt", "Lamp A" in sub_prompt and "Lamp B" in sub_prompt)
check("P8_no_toolraw_in_prompt", TOOLOUT not in sub_prompt)  # 전사=텍스트 턴만

# P-B: UNSURE -> no-op
ag, st = agent(), State(hist())
w = ToolCall("exchange_delivered_order_items", {"item_ids": ["1111111111"]})
reset(AM(tool_calls=[w]), AM(content="I am not sure which."))
am = ag._generate_next_message(UserMessage("yes"), st)
check("P9_unsure_noop", w.arguments["item_ids"] == ["1111111111"] and getattr(ag, "_t2_subcall_unsure", 0) == 1)

# P-B: 화이트리스트 밖(payment) -> confirm-only(치환 금지)
PAYOUT = json.dumps({"payment_methods": [{"payment_method_id": "pm_9012345"},
                                         {"payment_method_id": "pm_8887777"}]})
ag = agent()
st = State([UserMessage("pay for order #W1234567 with pm_9012345"),
            AM(tool_calls=[ToolCall("get_user_details", {}, id="u1")]),
            ToolMessage(id="u1", content=PAYOUT)])
w = ToolCall("exchange_delivered_order_items", {"payment_method_id": "pm_9012345"})
reset(AM(tool_calls=[w]), AM(content="pm_8887777"))
am = ag._generate_next_message(UserMessage("yes"), st)
check("P10_whitelist_confirmonly", w.arguments["payment_method_id"] == "pm_9012345"
      and getattr(ag, "_t2_subcall_confirmonly", 0) == 1, w.arguments)

# P-B: 서브콜 예외 -> no-op (SCRIPT 고갈로 generate가 raise)
ag, st = agent(), State(hist())
w = ToolCall("exchange_delivered_order_items", {"item_ids": ["1111111111"]})
reset(AM(tool_calls=[w]))  # 서브콜용 SCRIPT 없음 -> generate raise -> no-op이어야
am = ag._generate_next_message(UserMessage("yes"), st)
check("P11_exception_noop", am.tool_calls and w.arguments["item_ids"] == ["1111111111"])

# P-A GROUND: |C|=1 제자리 치환·regen 0
G.apply_provenance_regen(max_retries=4, use_badwords=False, ground=True, domain="retail", disamb=False)
ag, st = agent(), State(hist())
w = ToolCall("get_order_details", {"order_id": "#W9999999"})  # fab: ctx에 없음·후보=#W1234567 유일
reset(AM(tool_calls=[w]))
am = ag._generate_next_message(UserMessage("check my order"), st)
check("A1_ground_sub", w.arguments["order_id"] == "#W1234567", w.arguments)
check("A2_no_regen", GENCALLS == ["agent_response"], GENCALLS)
check("A3_counter", getattr(ag, "_t2_ground_sub", 0) == 1)

# P-C rescue: env-검증형 id 날조 -> pass-through(개입 0)
G.apply_provenance_regen(max_retries=4, use_badwords=False, ground=False, domain="retail",
                         disamb=False, prov_mode="rescue")
ag, st = agent(), State(hist())
w = ToolCall("modify_pending_order_items", {"order_id": "#W5555555"})  # fab·order=env-검증형·hashid
reset(AM(tool_calls=[w]))
am = ag._generate_next_message(UserMessage("modify it"), st)
check("C1_passthrough", am.tool_calls and w.arguments["order_id"] == "#W5555555")
check("C2_no_regen", GENCALLS == ["agent_response"], GENCALLS)
check("C3_counter", getattr(ag, "_t2_prov_skipped_envdup", 0) == 1)

# P-C rescue: 에러-루프 중이면 개입 유지(regen)
ag = agent()
st = State(hist() + [AM(tool_calls=[ToolCall("modify_pending_order_items", {}, id="e1")]),
                     ToolMessage(id="e1", error=True, content="Error: order not found")])
w = ToolCall("modify_pending_order_items", {"order_id": "#W5555555"})
reset(AM(tool_calls=[w]), AM(tool_calls=[ToolCall("modify_pending_order_items", {"order_id": "#W1234567"})]))
am = ag._generate_next_message(UserMessage("try again"), st)
check("C4_errloop_regen", "agent_response_regen" in GENCALLS, GENCALLS)
check("C5_regen_result", am.tool_calls[0].arguments["order_id"] == "#W1234567")

# P-C rescue: free-text(비 env-검증형) -> 개입 유지 + 중립 피드백(예시 나열 없음)
ag, st = agent(), State(hist())
w = ToolCall("modify_user_address", {"address1": "9999 Nowhere Rd"})
reset(AM(tool_calls=[w]), AM(content="let me look that up"))
am = ag._generate_next_message(UserMessage("update my address"), st)
check("C6_freetext_regen", "agent_response_regen" in GENCALLS, GENCALLS)
fb = ""
for g in GENKW:
    if g["messages"]:
        for m in g["messages"]:
            c = getattr(m, "content", "") or ""
            if isinstance(c, str) and "[PROVENANCE]" in c:
                fb = c
check("C7_neutral_feedback", fb and "payment methods" not in fb, fb[:120])

# 회귀: dialog 모드 §3 fix — 텍스트-only 재확인 = 원 호출 유지
G.apply_provenance_regen(max_retries=4, use_badwords=False, ground=False, domain="retail",
                         disamb=True, disamb_mode="dialog")
ag, st = agent(), State(hist())
w = ToolCall("exchange_delivered_order_items", {"item_ids": ["1111111111"]})
orig_am = AM(tool_calls=[w])
reset(orig_am, AM(content="Let me double-check with you first."))  # am2 = 텍스트-only
am = ag._generate_next_message(UserMessage("yes"), st)
check("R1_dialog_nowrite_keep", am is orig_am and am.tool_calls, am)
check("R2_counter", getattr(ag, "_t2_disamb_nowrite_keep", 0) == 1)

# 회귀: dialog 모드 정상 수락(tool_calls 있는 am2)
ag, st = agent(), State(hist())
w = ToolCall("exchange_delivered_order_items", {"item_ids": ["1111111111"]})
w2 = ToolCall("exchange_delivered_order_items", {"item_ids": ["2222222222"]})
reset(AM(tool_calls=[w]), AM(tool_calls=[w2]))
am = ag._generate_next_message(UserMessage("yes"), st)
check("R3_dialog_accept", am.tool_calls and am.tool_calls[0] is w2)

# ══════════════ unified 분기 ══════════════

G.apply_unified_regen(max_prov_retries=4, domain="retail", disamb=True, use_badwords=False,
                      ground=True, disamb_mode="subcall", prov_mode="full")

def usetup(history):
    ag = agent()
    orch = BaseOrchestrator(environment=Env(), agent=ag)  # init_inject가 게이트 배선
    return ag, orch, State(history)

def auth_hist():
    return [UserMessage("my email is real@user.com; exchange item in order #W1234567"),
            AM(tool_calls=[ToolCall("find_user_id_by_email", {"email": "real@user.com"}, id="a1")]),
            ToolMessage(id="a1", content="usr_123"),
            AM(tool_calls=[ToolCall("get_order_details", {"order_id": "#W1234567"}, id="g1")]),
            ToolMessage(id="g1", content=TOOLOUT)]

# U1: unified P-A ground (read call·게이트 무관)
ag, orch, st = usetup(auth_hist())
w = ToolCall("get_order_details", {"order_id": "#W9999999"})
reset(AM(tool_calls=[w]))
am = ag._generate_next_message(UserMessage("look it up again"), st)
check("U1_ground_sub", w.arguments["order_id"] == "#W1234567", w.arguments)
check("U2_no_regen_no_tick", GENCALLS == ["agent_response"] and orch.num_errors == 0,
      (GENCALLS, orch.num_errors))

# U3: unified P-B — 스위치 값이 게이트-deny(사용자 미언급)면 원값 복원 (revert 가드)
ag, orch, st = usetup(auth_hist())
w = ToolCall("exchange_delivered_order_items",
             {"order_id": "#W1234567", "item_ids": ["1111111111"],
              "new_item_ids": ["2222222222"], "payment_method_id": "pm_9012345"})
st.messages.append(UserMessage("pay with pm_9012345"))
orig_am = AM(tool_calls=[w])
reset(orig_am, AM(content="2222222222"))
am = ag._generate_next_message(UserMessage("yes, please exchange item 1111111111"), st)
check("U3_gate_denied_reverted", am is orig_am and w.arguments["item_ids"] == ["1111111111"]
      and getattr(ag, "_t2_disamb_gate_reject", 0) >= 1, w.arguments)
check("U4_gencalls", GENCALLS == ["agent_response", "disamb_subcall"], GENCALLS)

# U5: unified P-B — 스위치가 게이트-클린이면 치환 유지 (well-posed: new_item_ids≠후보라 switch가
#   disjoint 위반 안 만듦. 구 U5는 new_item_ids=2222222222라 switch→자기교환=constraints deny=축퇴).
ag, orch, st = usetup(auth_hist())
w = ToolCall("exchange_delivered_order_items",
             {"order_id": "#W1234567", "item_ids": ["1111111111"],
              "new_item_ids": ["3333333333"], "payment_method_id": "pm_9012345"})
st.messages.append(UserMessage("pay with pm_9012345"))
orig_am = AM(tool_calls=[w])
reset(orig_am, AM(content="2222222222"))
am = ag._generate_next_message(
    UserMessage("yes, exchange item 1111111111 — actually the one I mean is 2222222222"), st)
check("U5_turn_preserved_switch", am is orig_am and w.arguments["item_ids"] == ["2222222222"]
      and getattr(ag, "_t2_disamb_gate_reject", 0) == 0, (am is orig_am, w.arguments))

# U5: unified 기본값(=v1) 회귀 — 기존 test_unified_regen.py 전체가 별도 프로세스서 검증됨(러너에서 실행)

# ===== P2 원리-디폴트 (_apply_principle_default) 단위 =====
class FakeGate:
    def __init__(self, orig_pay):
        # resolve_field(path, args) → 주문 원결제
        self.resolvers = {"resolve_field": lambda path, args: orig_pay}
A2P = {"default_specs": [{"arg": "payment_method_id",
        "resolver_path": ["order_id", "get_order_details", "payment_method_id"],
        "applies_to": ["modify_pending_order_items", "modify_pending_order_payment"]}]}

# P2-1: 계정 gift_card(원결제 아님·user 미언급) → 원결제 paypal로 치환
w = ToolCall("modify_pending_order_items",
             {"order_id": "#W5061109", "item_ids": ["a"], "payment_method_id": "gift_card_3406421"})
am2 = AM(tool_calls=[w])
n = G._apply_principle_default(am2, A2P, FakeGate("paypal_3742148"), "i want blue earbuds price same")
check("P2_1_substituted", n == 1 and w.arguments["payment_method_id"] == "paypal_3742148", w.arguments)

# P2-2: 이미 원결제 → no-op
w = ToolCall("modify_pending_order_items", {"order_id": "#W1", "payment_method_id": "paypal_3742148"})
am2 = AM(tool_calls=[w])
check("P2_2_already_default_noop",
      G._apply_principle_default(am2, A2P, FakeGate("paypal_3742148"), "") == 0
      and w.arguments["payment_method_id"] == "paypal_3742148")

# P2-3: user가 gift_card 명시(override) → 유지(유동성 보존)
w = ToolCall("modify_pending_order_items", {"order_id": "#W1", "payment_method_id": "gift_card_3406421"})
am2 = AM(tool_calls=[w])
check("P2_3_user_override_keep",
      G._apply_principle_default(am2, A2P, FakeGate("paypal_3742148"),
                                 "please use my gift_card_3406421 instead") == 0
      and w.arguments["payment_method_id"] == "gift_card_3406421")

# P2-4: applies_to 밖 도구 → no-op
w = ToolCall("cancel_pending_order", {"order_id": "#W1", "payment_method_id": "gift_card_3406421"})
am2 = AM(tool_calls=[w])
check("P2_4_tool_scope_noop",
      G._apply_principle_default(am2, A2P, FakeGate("paypal_3742148"), "") == 0)

# P2-5: resolver None(조회 실패) → no-op(안전측)
w = ToolCall("modify_pending_order_items", {"order_id": "#W1", "payment_method_id": "gift_card_x"})
am2 = AM(tool_calls=[w])
check("P2_5_resolver_none_noop",
      G._apply_principle_default(am2, A2P, FakeGate(None), "") == 0
      and w.arguments["payment_method_id"] == "gift_card_x")

print()
if FAILS:
    print("FAILED: %d -> %s" % (len(FAILS), FAILS))
    sys.exit(1)
print("ALL PASS (%s checks)" % (sum(1 for _ in open(__file__, encoding='utf-8') if 'check(' in _) - 2))
