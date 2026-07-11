# -*- coding: utf-8 -*-
"""FORMALIZE-EXEC 단위 (레버3·NEXT_LEVER_GEN §2·2026-07-11).

[E] 실행기 op×constraints 전수: argmax/argmin/filter·동률·필드부재(=unresolvable)·
    빈후보·비교자(le)·불리언/숫자/문자 값 비교.
[P] parse_formalize: 유효/무효 op·constraints 형식·산문 속 JSON·파싱실패=UNSURE(None).
[W] 배선(tau2 stub): T2_FEXEC=1이면 DISAMB 서브콜 프롬프트에 [FORMALIZED CRITERION …]
    주석 첨부(비커밋)·off면 무개입·형식화 실패(garbage)=주석 없이 DISAMB 그대로(no-op).
"""
import sys, os, types, json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ["T2_GATE_KINDS"] = "auth,confirm,ownership,notice,preconditions,constraints"

import t2_formalize_exec as FX  # noqa: E402  (엔진 순수 로직 — tau2 불요)

FAILS, N = [], [0]
def check(name, cond, detail=""):
    N[0] += 1
    print(("PASS " if cond else "FAIL ") + name + (" | " + str(detail) if detail and not cond else ""))
    if not cond:
        FAILS.append(name)

# ── [E] 실행기 ──────────────────────────────────────────────────────────────
RECS = [
    ("1111111111", {"item_id": "1111111111", "price": 158.6, "available": True,
                    "options": {"size": "9", "color": "black"}}),
    ("2222222222", {"item_id": "2222222222", "price": 172.0, "available": True,
                    "options": {"size": "9", "color": "white"}}),
    ("3333333333", {"item_id": "3333333333", "price": 189.9, "available": False,
                    "options": {"size": "10", "color": "black"}}),
]
sp = {"op": "argmax", "field": "price",
      "constraints": [{"field": "size", "op": "eq", "value": "9"},
                      {"field": "available", "op": "eq", "value": True}]}
r = FX.execute_formalized(sp, RECS)
check("E1_argmax_constrained_t20형", r["status"] == "ok" and r["ids"] == ["2222222222"], r)
r = FX.execute_formalized({"op": "argmin", "field": "price", "constraints": []}, RECS)
check("E2_argmin_unconstrained", r["status"] == "ok" and r["ids"] == ["1111111111"], r)
TIE = RECS + [("4444444444", {"item_id": "4444444444", "price": 172.0, "available": True,
                              "options": {"size": "9"}})]
r = FX.execute_formalized({"op": "argmax", "field": "price",
                           "constraints": [{"field": "size", "op": "eq", "value": "9"}]}, TIE)
check("E3_tie_all_returned", r["status"] == "ok" and sorted(r["ids"]) == ["2222222222", "4444444444"], r)
r = FX.execute_formalized({"op": "argmax", "field": "date", "constraints": []}, RECS)
check("E4_rankfield_absent_unresolvable_t71형", r["status"] == "unresolvable", r)
r = FX.execute_formalized({"op": "filter", "field": None,
                           "constraints": [{"field": "purchase_date", "op": "eq", "value": "2024"}]}, RECS)
check("E5_consfield_absent_unresolvable", r["status"] == "unresolvable", r)
r = FX.execute_formalized({"op": "filter", "field": None,
                           "constraints": [{"field": "price", "op": "le", "value": 175}]}, RECS)
check("E6_filter_le_t37형", r["status"] == "ok" and sorted(r["ids"]) == ["1111111111", "2222222222"], r)
r = FX.execute_formalized({"op": "filter", "field": None,
                           "constraints": [{"field": "color", "op": "eq", "value": "Black"}]}, RECS)
check("E7_filter_str_ci_t79형", r["status"] == "ok" and sorted(r["ids"]) == ["1111111111", "3333333333"], r)
r = FX.execute_formalized({"op": "filter", "field": None,
                           "constraints": [{"field": "size", "op": "eq", "value": "12"}]}, RECS)
check("E8_empty_after_filter", r["status"] == "empty" and r["ids"] == [], r)
r = FX.execute_formalized({"op": "argmax", "field": "price", "constraints": []}, [("x", None)])
check("E9_no_records_unresolvable", r["status"] == "unresolvable", r)
r = FX.execute_formalized({"op": "none", "field": None, "constraints": []}, RECS)
check("E10_none_passthrough", r["status"] == "none")
r = FX.execute_formalized({"op": "filter", "field": None,
                           "constraints": [{"field": "size", "op": "eq", "value": 9}]}, RECS)
check("E11_numeric_str_eq", r["status"] == "ok" and sorted(r["ids"]) == ["1111111111", "2222222222"], r)

# ── [P] parse_formalize ─────────────────────────────────────────────────────
ok_json = '{"op": "argmax", "field": "price", "constraints": [{"field": "size", "value": "9"}]}'
p = FX.parse_formalize("Sure, here it is:\n" + ok_json + "\nDone.")
check("P1_json_in_prose", p is not None and p["op"] == "argmax"
      and p["constraints"][0]["op"] == "eq", p)
check("P2_invalid_op_none", FX.parse_formalize('{"op": "sort", "field": "price"}') is None)
check("P3_malformed_none", FX.parse_formalize("no json here") is None)
check("P4_bad_constraints_none",
      FX.parse_formalize('{"op": "filter", "constraints": [{"value": 1}]}') is None)
check("P5_argmax_needs_field", FX.parse_formalize('{"op": "argmax"}') is None)
check("P6_bad_cons_op_none",
      FX.parse_formalize('{"op": "filter", "constraints": [{"field": "p", "op": "regex", "value": 1}]}') is None)
check("P7_unresolvable_ok", (FX.parse_formalize('{"op": "unresolvable"}') or {}).get("op") == "unresolvable")

# 주석 텍스트
note = FX.fexec_annotation({"op": "argmax", "field": "price", "constraints": []},
                           {"status": "ok", "ids": ["2222222222"], "why": "argmax(price)=172.0"}, "item_ids")
check("P8_annotation_ok", note and note.startswith("[FORMALIZED CRITERION") and "2222222222" in note)
note = FX.fexec_annotation({"op": "argmax", "field": "date", "constraints": []},
                           {"status": "unresolvable", "ids": [], "why": "rank field 'date' absent"}, "item_ids")
check("P9_annotation_unresolvable_ask", note and "CANNOT be evaluated" in note and "ask the user" in note)
check("P10_annotation_none_null",
      FX.fexec_annotation({"op": "none", "field": None, "constraints": []},
                          {"status": "none", "ids": [], "why": ""}, "k") is None)

# ── [W] 배선 (tau2 stub — test_t5c_silent 동형) ─────────────────────────────
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

SCRIPT, GENCALLS, GENMSGS = [], [], []
def generate(model=None, tools=None, messages=None, call_name=None, **kw):
    GENCALLS.append(call_name)
    GENMSGS.append(messages)
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
        self.domain_name, self.tools = "retail", None
class State:
    def __init__(self, messages):
        self.system_messages, self.messages = [], list(messages)

TOOLOUT = json.dumps({"order_id": "#W1234567", "items": [
    {"item_id": "1111111111", "name": "Lamp A", "price": 10.0},
    {"item_id": "2222222222", "name": "Lamp B", "price": 20.0},
    {"item_id": "3333333333", "name": "Lamp C", "price": 30.0}]})
def hist():
    return [UserMessage("exchange item in order #W1234567 for the most expensive one, pay with pm_9012345"),
            AM(tool_calls=[ToolCall("get_order_details", {"order_id": "#W1234567"}, id="g1")]),
            ToolMessage(id="g1", content=TOOLOUT)]

G.apply_provenance_regen(max_retries=4, use_badwords=False, ground=False, domain="retail",
                         disamb=True, disamb_mode="subcall")
def agent():
    ag = LLMAgent()
    ag.llm, ag.llm_args, ag.tools = "m", {}, []
    ag._t2_a2 = G._domain_a2("retail")
    return ag

def run_case(fexec_env, scripts):
    os.environ["T2_FEXEC"] = fexec_env
    ag, st = agent(), State(hist())
    w = ToolCall("exchange_delivered_order_items",
                 {"order_id": "#W1234567", "item_ids": ["1111111111"],
                  "new_item_ids": ["2222222222"], "payment_method_id": "pm_9012345"})
    GENCALLS[:], GENMSGS[:], SCRIPT[:] = [], [], [AM(tool_calls=[w])] + scripts
    am = ag._generate_next_message(UserMessage("yes please"), st)
    return ag, w, am

# W1: T2_FEXEC=1 — 형식화 subcall → 실행 → DISAMB 프롬프트에 주석
ag, w, am = run_case("1", [AM(content='{"op": "argmax", "field": "price", "constraints": []}'),
                           AM(content="3333333333")])
check("W1_gencalls", GENCALLS == ["agent_response", "formalize_subcall", "disamb_subcall"], GENCALLS)
disamb_prompt = GENMSGS[2][0].content
check("W1_annotation_in_disamb_prompt", "[FORMALIZED CRITERION" in disamb_prompt
      and "3333333333" in disamb_prompt.split("[FORMALIZED CRITERION")[1][:400])
check("W1_counters", getattr(ag, "_t2_fexec_fired", 0) == 1
      and getattr(ag, "_t2_fexec_annotated", 0) == 1)
check("W1_switch_applied", w.arguments["item_ids"] == ["3333333333"], w.arguments)
check("W1_history_uncommitted", all("[FORMALIZED CRITERION" not in str(getattr(m, "content", ""))
                                    for m in GENMSGS[0]), "주석=서브콜 프롬프트만·본대화 불변")

# W2: toggle off — 형식화 subcall 없음(기존 DISAMB 그대로)
ag, w, am = run_case("0", [AM(content="2222222222")])
check("W2_off_no_fexec", GENCALLS == ["agent_response", "disamb_subcall"], GENCALLS)
check("W2_no_counter", getattr(ag, "_t2_fexec_fired", 0) == 0)

# W3: 형식화 응답 garbage → UNSURE no-op → 주석 없는 DISAMB 폴백(§2.4 ②)
ag, w, am = run_case("1", [AM(content="I think the user wants the lamp."),
                           AM(content="2222222222")])
check("W3_gencalls", GENCALLS == ["agent_response", "formalize_subcall", "disamb_subcall"], GENCALLS)
check("W3_no_annotation", "[FORMALIZED CRITERION" not in GENMSGS[2][0].content)
check("W3_unsure_counter", getattr(ag, "_t2_fexec_unsure", 0) == 1)

# W4: op=none → 폴백(주석 없음·counter)
ag, w, am = run_case("1", [AM(content='{"op": "none"}'), AM(content="UNSURE")])
check("W4_none_fallback", "[FORMALIZED CRITERION" not in GENMSGS[2][0].content
      and getattr(ag, "_t2_fexec_none", 0) == 1)

# W5: 형식화 subcall 예외(SCRIPT 고갈) → no-op·DISAMB도 예외 → 원 호출 유지
ag, st = agent(), State(hist())
w = ToolCall("exchange_delivered_order_items", {"item_ids": ["1111111111"]})
os.environ["T2_FEXEC"] = "1"
GENCALLS[:], GENMSGS[:], SCRIPT[:] = [], [], [AM(tool_calls=[w])]
am = ag._generate_next_message(UserMessage("yes"), st)
check("W5_exception_noop", am.tool_calls and w.arguments["item_ids"] == ["1111111111"])
os.environ["T2_FEXEC"] = "0"

print("\n%d FAIL" % len(FAILS) if FAILS else "\nALL PASS (%d checks)" % N[0])
sys.exit(1 if FAILS else 0)
