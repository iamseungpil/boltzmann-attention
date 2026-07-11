#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""test_eplan.py — E-PLAN 결정론 로직 단위테스트 (설계 §5(c)·tau2 임포트 0·순수 오프라인).

실행: py -3 scripts/distill/tau2/test_eplan.py  →  "ALL PASS (n checks)".
도구명은 전부 가짜 spec(도메인일반 검증) — retail A2는 로딩 검사에서만 실파일 사용.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t2_eplan_patch import (  # noqa: E402
    PlanLedger, expand_scope, discovery_L1, discovery_L2, coverage_gap,
    is_scope_token, load_eplan_spec, _covers, discovery_precondition,
    _enum_items, walk_required_n, _intent_match, ledger_summary,
    structured_replan_prompt, parse_obligations, transcript_text,
    qty_item_covered, _parse_qty)

# 가짜 A2 eplan spec — 엔진이 도구명 하드코딩 0([[05]])임을 도구명 자체로 검증
SPEC = {"list_enumerator": "list_tool", "detail_reader": "detail_tool", "entity_key": "eid"}

N = 0


def ok(cond, msg):
    global N
    N += 1
    if not cond:
        print("FAIL #%d: %s" % (N, msg))
        sys.exit(1)


# ── 1. qty 누적 ──────────────────────────────────────────────────────────────
led = PlanLedger(SPEC)
ok(led.required_qty() == 1, "초기 qty=1")
led.accumulate_qty("I want to exchange two laptops for a bigger screen")
ok(led.required_qty() == 2, "'two laptops' → 2")
led.accumulate_qty("actually just a laptop")           # 수량 신호 없음 → 하향 금지
ok(led.required_qty() == 2, "'a laptop'은 최대값 2 유지")
led.accumulate_qty("make that 3 of them")
ok(led.required_qty() == 3, "숫자 3 → 최대값 갱신")
led2 = PlanLedger(SPEC)
led2.accumulate_qty("a laptop please")
ok(led2.required_qty() == 1, "'a laptop' 단독 → 1 유지")
led3 = PlanLedger(SPEC)
led3.accumulate_qty("return both of them")
ok(led3.required_qty() == 2, "'both' → 2 (소사전)")
led3.accumulate_qty("order #W2905754 costs 500 dollars")  # id 조각·금액 오염 방지
ok(led3.required_qty() == 2, "긴 숫자(id·금액)는 수량 아님")

# ── 2. expand_scope ──────────────────────────────────────────────────────────
w_tok = [{"intent_class": "item_change", "entity": "ALL_PENDING",
          "items": frozenset(), "qty": 2}]
exp = expand_scope(w_tok, {"W1", "W2"})
ok(len(exp) == 2 and {e["entity"] for e in exp} == {"W1", "W2"},
   "SCOPE_TOKEN × listed={W1,W2} → 2건 구체화")
ok(all(e["intent_class"] == "item_change" for e in exp), "확장이 intent_class 보존")
exp0 = expand_scope(w_tok, set())
ok(len(exp0) == 1 and exp0[0]["entity"] == "ALL_PENDING",
   "discovery 전(listed 빈) → 토큰 그대로 보존(L1 신호)")
ok(is_scope_token("ALL_PENDING") and is_scope_token("EACH") and not is_scope_token("#W1"),
   "SCOPE_TOKEN 어휘 판별")

# ── 3. discovery L1 ──────────────────────────────────────────────────────────
lg = PlanLedger(SPEC)
lg.seed([{"intent_class": "item_change", "entity": "ALL_PENDING", "items": (), "qty": 1}])
ok(discovery_L1(lg) is True, "L1: 토큰 있고 listed 비면 True")
lg.listed = {"W1"}
ok(discovery_L1(lg) is False, "L1: listed 차면 False")
lg_q = PlanLedger(SPEC)
lg_q.accumulate_qty("exchange two laptops")            # 토큰 없어도 수량>=2면
ok(discovery_L1(lg_q) is True, "L1: qty>=2·listed 비면 True")

# ── 4. discovery L2 — t95 두 형상 (설계 §5(d)④ 술어 특이도) ──────────────────
# ⓐ tr1/tr3형: qty=2·executed 1주문·listed 3·examined 1 → 미검토 [W1,W3] 발화
la = PlanLedger(SPEC)
la.accumulate_qty("I want to exchange two laptops")
la.note_write("item_change", "W2", {"i1"})
la.listed = {"W1", "W2", "W3"}
la.examined = {"W2"}
ok(discovery_L2(la, "item_change") == ["W1", "W3"], "L2 ⓐ: 미검토 sibling [W1,W3] 발화")
fb = discovery_precondition(la, SPEC, "item_change")
ok(fb is not None and "W1" in fb and "W3" in fb and "detail_tool" in fb,
   "precondition: L2 피드백에 미검토 id + A2 detail 도구명")
# ⓑ tr0/tr2형: 전부 examined(binding-gap) → 침묵 (ⓑ는 CP5 재-plan walk 관할)
lb = PlanLedger(SPEC)
lb.accumulate_qty("exchange two laptops")
lb.note_write("item_change", "W2", {"i1"})
lb.listed = {"W1", "W2", "W3"}
lb.examined = {"W1", "W2", "W3"}
ok(discovery_L2(lb, "item_change") == [], "L2 ⓑ: 전부 examined → 침묵(특이도)")
ok(discovery_precondition(lb, SPEC, "item_change") is None,
   "precondition: ⓑ형은 통과(None)")
# intent-class 불일치 executed는 M에 미산입
lc = PlanLedger(SPEC)
lc.accumulate_qty("two")
lc.note_write("cancel", "W2", ())
lc.listed = {"W1", "W2"}
lc.examined = {"W2"}
ok(discovery_L2(lc, "item_change") == ["W1"], "L2: M은 intent-class 매칭만 계수")

# ── 5. coverage_gap — replan 기준 diff·items 관대 커버 ───────────────────────
cg = PlanLedger(SPEC)
cg.listed = {"W1", "W2"}
cg.note_write("item_change", "W1", {"i1", "i2", "i9"})
cg.set_replan([
    {"intent_class": "item_change", "entity": "W1", "items": {"i1", "i2"}, "qty": 1},
    {"intent_class": "item_change", "entity": "W2", "items": {"i3"}, "qty": 1},
])
gaps = coverage_gap(cg)
ok(len(gaps) == 1 and gaps[0]["entity"] == "W2",
   "coverage_gap: replan 2 vs executed 1 → gap 1건(W2)")
ok(_covers({"intent_class": "c", "entity": "W1", "items": frozenset({"a", "b"})},
           {"intent_class": "c", "entity": "W1", "items": frozenset({"a"})}),
   "_covers: plan items ⊆ executed items → 커버")
ok(_covers({"intent_class": "c", "entity": "W1", "items": frozenset({"a"})},
           {"intent_class": "c", "entity": "W1", "items": frozenset({"a", "b"})}),
   "_covers: 부분집합 관계(역방향)도 관대 커버")
ok(not _covers({"intent_class": "c", "entity": "W1", "items": frozenset({"a"})},
               {"intent_class": "c", "entity": "W2", "items": frozenset({"a"})}),
   "_covers: entity 불일치 → 미커버")
# SCOPE_TOKEN replan이 listed로 확장돼 diff
ct = PlanLedger(SPEC)
ct.listed = {"W1", "W2"}
ct.note_write("item_change", "W1", {"i1"})
ct.set_replan([{"intent_class": "item_change", "entity": "ALL_PENDING",
                "items": (), "qty": 2}])
gt = coverage_gap(ct)
ok(len(gt) == 1 and gt[0]["entity"] == "W2", "coverage_gap: 토큰 replan 확장 후 W2만 gap")

# ── 6. note_read — A2 도구명 분기 ────────────────────────────────────────────
nr = PlanLedger(SPEC)
nr.note_read("list_tool", {}, json.dumps(
    {"user_id": "u_1", "orders": ["#W1", "#W2"], "name": {"first": "a"}}))
ok(nr.listed == {"#W1", "#W2"} and nr.examined == set(),
   "note_read: enumerator 출력의 문자열-리스트 → listed")
nr.note_read("detail_tool", {"eid": "#W1"}, "{}")
ok(nr.examined == {"#W1"}, "note_read: detail 호출 인자(entity_key) → examined")
nr.note_read("other_tool", {"eid": "#W9"}, "{}")
ok("#W9" not in nr.examined and "#W9" not in nr.listed,
   "note_read: 무관 도구는 무갱신")
nr2 = PlanLedger(SPEC)
nr2.note_read("list_tool", {}, json.dumps([{"eid": "#W7", "status": "pending"}]))
ok(nr2.listed == {"#W7"}, "note_read: entity_key 중첩 dict 형태도 수집")
nr3 = PlanLedger(SPEC)
nr3.note_read("list_tool", {}, "not-json output")
ok(nr3.listed == set(), "note_read: 비JSON 출력 → 안전측(무갱신)")

# ── 6b. 품목-나열 힌트 (A′ t81 교정·도메인일반 구문 신호) ────────────────────
T81 = ("But I can tell you which items I don't need anymore: the hiking boots, "
       "watch, keyboard, charger, jacket, and running shoes.")
ok(_enum_items(T81) == 6, "나열: t81 실발화 6-품목 나열 → 6")
ok(_enum_items("I need to cancel and return a bookshelf and a jigsaw I got") == 0,
   "나열: 2-품목(and만·쉼표 0) → 미발화(보수)")
ok(_enum_items("Yes, that's the right address—921 Park Avenue, Suite 892, "
               "Chicago, IL 60612.") == 0,
   "나열: 주소(digit 세그먼트가 run 절단) → 미발화")
ok(_enum_items("please make sure the refund is done within 3 days, not 5 to 7") == 0,
   "나열: 날짜/수치 문구 → 미발화")
ok(_enum_items("") == 0 and _enum_items(None) == 0, "나열: 빈 입력 → 0")
ok(_enum_items("I want the hose, backpack, and hiking boots") == 3,
   "나열: 3-품목(쉼표 2 + and) → 3 (하한 경계)")
en = PlanLedger(SPEC)
en.accumulate_qty(T81)
ok(en.qty_mentioned == 1, "나열 힌트는 qty-소사전 불변(L2 conflation 방지)")
ok(en.multi_entity_hint is True and en.enum_items == 6, "ledger가 나열 힌트 누적")
ok(discovery_L1(en) is True, "L1: 수량어 0이어도 나열>=3 ∧ listed 비면 True")
en.listed = {"W1"}
ok(discovery_L1(en) is False, "L1: listed 차면 나열 힌트 있어도 False")
ok(discovery_L2(en, "item_change") == [], "L2: 나열 힌트에 불변(qty=1 → 침묵 유지)")
ok(walk_required_n(en, ["W2"]) == 6,
   "walk N: 나열 + 미검토 sibling 존재 → 6 (t81 stop-time walk 발화)")
ok(walk_required_n(en, []) == 1,
   "walk N: 미검토 없으면 나열 힌트 무기여(passing-sim 과발화 0·census 근거)")
en2 = PlanLedger(SPEC)
en2.accumulate_qty("just a laptop")
ok(walk_required_n(en2, ["W2"]) == 1, "walk N: 힌트 없으면 기존 required_qty 그대로")
ok(_enum_items("Oh, um, yes, please. I'd like to change it.") == 0,
   "나열: 간투사 체인(filler) → 미발화")
ok(_enum_items("Oh, um, the hiking boots, watch, and keyboard please") == 3,
   "나열: 간투사 섞인 실나열 → filler skip 후 3")

# ── 7. 경계 ─────────────────────────────────────────────────────────────────
eb = PlanLedger(SPEC)                                   # qty=1 → L2 침묵
eb.listed = {"W1", "W2"}
eb.examined = set()
ok(discovery_L2(eb, "item_change") == [], "경계: qty=1이면 L2 침묵(over-read 방지)")
er = PlanLedger(SPEC)                                   # replan 비면 gap 없음
er.note_write("item_change", "W1", {"i1"})
ok(coverage_gap(er) == [], "경계: replan 비면 gap 없음(즉시 종결·R1b)")

# ── 8. A2 로딩 — retail.gate.json "eplan" 키 ────────────────────────────────
rs = load_eplan_spec("retail")
ok(rs is not None, "retail A2에 eplan 키 존재")
ok(rs.get("list_enumerator") == "get_user_details"
   and rs.get("detail_reader") == "get_order_details"
   and rs.get("entity_key") == "order_id", "retail eplan spec 3키 정확")
ok(load_eplan_spec("no_such_domain") is None, "미존재 도메인 → None(비활성)")
# 전체 파일 json.load 무결 확인
_p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "a2", "retail.gate.json")
with open(_p, encoding="utf-8") as f:
    _full = json.load(f)
ok("gates" in _full and "eplan" in _full, "retail.gate.json 전체 json.load 무결")

# ── 8b. ★qty-conflation 수리 (T27_T103_PERSTEP §t27 부결함②·2026-07-11) ─────
# (i) t47형: 시간/기간-인접 수사 + bare 범위 = 수량 아님 (닫힌 기능어 집합·도메인 0)
ok(_parse_qty("please make sure the refund is done within 3 days") == 0,
   "t47형: '3 days' → 시간단위 인접 = 수량 아님")
ok(_parse_qty("completed within 3 days, not 5 to 7") == 0,
   "t47형: bare 범위 '5 to 7'(단위 생략) → 수량 아님")
ok(_parse_qty("it usually takes 5 to 7 business days") == 0,
   "t47형: '5 to 7 business days' → 수량 아님")
ok(_parse_qty("I'll wait a couple of weeks then") == 0,
   "t47형: 'couple of weeks' → 수량 아님")
ok(_parse_qty("exchange two laptops in 3 days") == 2,
   "t47형 특이도: 시간구만 절제·'two laptops'=2 보존")
t47l = PlanLedger(SPEC)
t47l.accumulate_qty("Please process the refund within 3 days, not 5 to 7.")
ok(t47l.required_qty() == 1 and discovery_L1(t47l) is False,
   "t47형: 기간 오염 제거 → qty=1·L1 침묵")
ok(discovery_L2(t47l, "return_tool") == [], "t47형: L2도 침묵")

# (ii) t27형: 단일 주문 다품목 — 발화 수량("a couple of things")이 품목 수.
#     시도 write의 distinct 품목 id가 N을 설명하면 deny 침묵 (qty_item_covered).
t27l = PlanLedger(SPEC)
t27l.accumulate_qty("I need to return a couple of things from my recent order")
ok(t27l.required_qty() == 2, "t27형: 'couple' → qty=2 (소사전 불변)")
t27l.listed = {"W1", "W2", "W3"}
t27l.examined = {"W1"}
ok(discovery_L2(t27l, "return_tool", attempt_items=("i1", "i2")) == [],
   "t27형: 시도 write 2품목 = qty 2 커버 → L2 침묵 (거짓 sibling-deny 제거)")
ok(discovery_precondition(t27l, SPEC, "return_tool", attempt_items=("i1", "i2")) is None,
   "t27형: precondition 통과 (ap2 t27 3중 deny 소멸 경로)")
ok(discovery_L2(t27l, "return_tool", attempt_items=("i1",)) == ["W2", "W3"],
   "t27형 한계(정직): 1품목 시도는 qty=2 미커버 → deny 유지 (ap1형 잔존·cap이 상한)")
# 실행-누적 커버: 이전 write 1품목 + 시도 1품목 = 2 → 침묵
t27c = PlanLedger(SPEC)
t27c.accumulate_qty("return a couple of things")
t27c.listed = {"W1", "W2"}
t27c.note_write("return_tool", "W1", {"i1"})
ok(discovery_L2(t27c, "exchange_tool", attempt_items=("i2",)) == [],
   "t27형: 실행품목+시도품목 누적 커버 → 침묵 (intent 불문 goods 계정)")
# walk 억제: 실행 2품목이 qty=2 설명 → 거짓 gap 아님 (ap2 t27 walk 962-963 대응)
t27w = PlanLedger(SPEC)
t27w.accumulate_qty("return a couple of things from my order")
t27w.listed = {"W1", "W2", "W3"}
t27w.note_write("return_tool", "W1", {"i1", "i2"})
ok(qty_item_covered(t27w, walk_required_n(t27w, ["W2", "W3"])) is True,
   "t27형: walk N=2가 실행 2품목으로 커버 → walk 억제 재료")

# (iii) t95형(회귀 필수): 진짜 2주문 — 시도가 1품목(tr1/tr3) 또는 동일-id 중복(ap2)
#      → 미커버 → L2 발화 유지.
t95l = PlanLedger(SPEC)
t95l.accumulate_qty("I just received a couple of laptops")
t95l.accumulate_qty("exchange two laptops for the i7 model")
t95l.listed = {"W1", "W2", "W3"}
t95l.examined = {"W1", "W2"}
ok(discovery_L2(t95l, "exchange_tool", attempt_items=("i1",)) == ["W3"],
   "t95형 유지: 1품목 시도 < qty 2 → 미검토 sibling 발화")
ok(discovery_L2(t95l, "exchange_tool", attempt_items=("i1", "i1")) == ["W3"],
   "t95형 유지: 동일-id 중복(ap2 이중지정)은 distinct 1 → 발화 유지")
ok("W3" in (discovery_precondition(t95l, SPEC, "exchange_tool",
                                   attempt_items=("i1", "i1")) or ""),
   "t95형 유지: precondition이 L2 피드백 반환")
# t95 walk 유지: 실행 0 또는 중복-id → coverage < 2 → walk 발화 재료 유지
ok(qty_item_covered(t95l, 2) is False, "t95형: 실행 0 → walk 억제 없음")
# t81형 불변: cancel(품목 인자 0) → coverage 0 → 나열-힌트 walk 불변
t81l = PlanLedger(SPEC)
t81l.accumulate_qty(T81)
t81l.listed = {"W1", "W2"}
t81l.note_write("cancel_tool", "W1", ())
ok(qty_item_covered(t81l, walk_required_n(t81l, ["W2"])) is False,
   "t81형 불변: 품목-인자 없는 write → coverage 0 → walk 유지")

# ── 9. ★레버4 formalize-obligation (T2_EPLAN_REPLAN·부록 Y·2026-07-11) ──────
# 9a. _intent_match: 축약형('exchange')이 executed 도구명을 커버 (양방향 포함)
ok(_intent_match("exchange", "exchange_delivered_items_tool"), "intent: 축약⊂도구명")
ok(_intent_match("exchange_delivered_items_tool", "exchange"), "intent: 역방향 포함")
ok(_intent_match("cancel_tool", "cancel_tool"), "intent: 정확 일치")
ok(not _intent_match("exchange", "cancel_tool"), "intent: 무관 도구 불일치")
ok(not _intent_match("", "cancel_tool") and not _intent_match(None, "x"),
   "intent: 빈/None 불일치(안전측)")

# 9b. _covers가 intent 포함관계로 커버 (레버4 replan 산출 정합)
e9 = {"intent_class": "exchange_delivered_items_tool", "entity": "W1",
      "items": frozenset({"i1"})}
p9 = {"intent_class": "exchange", "entity": "W1", "items": frozenset()}
ok(_covers(e9, p9), "covers: 축약 intent + items 관대")
ok(not _covers(e9, dict(p9, entity="W2")), "covers: entity 불일치는 여전히 gap")

# 9c. ledger_summary — qty·listed·examined·executed 열거(결정론·도메인일반 어휘)
l9 = PlanLedger(SPEC)
l9.accumulate_qty("exchange two things")
l9.listed = {"W1", "W2"}
l9.examined = {"W1"}
l9.note_write("exchange_tool", "W1", {"i1"})
s9 = ledger_summary(l9)
ok("max quantity the user mentioned: 2" in s9, "summary: qty 열거")
ok("W1, W2" in s9 and "EXAMINED" in s9, "summary: listed/examined 열거")
ok("exchange_tool on W1" in s9 and "items: i1" in s9, "summary: executed 열거")
l9e = PlanLedger(SPEC)
ok("(none)" in ledger_summary(l9e), "summary: 빈 ledger는 (none)")

# 9d. structured_replan_prompt — ledger + 의무-JSON 지시 + entity_key(A2) + 전사
p = structured_replan_prompt(l9, "[user] hello", "eid")
ok(p.startswith("[LEDGER"), "prompt: ledger 요약이 선두(구조화 문맥=⑤ raw-실패 교정)")
ok('"eid"' in p and '"action"' in p and "JSON array" in p, "prompt: 의무-JSON 스키마(entity_key=ABox)")
ok("--- TRANSCRIPT ---" in p and "[user] hello" in p, "prompt: 전사 포함")

# 9e. parse_obligations — 정상/산문 속/실패=None/빈 배열
arr = parse_obligations('Here: [{"action": "exchange_tool", "eid": "W2", "items": ["i2"]}]', "eid")
ok(arr is not None and len(arr) == 1 and arr[0]["intent_class"] == "exchange_tool"
   and arr[0]["entity"] == "W2" and arr[0]["items"] == ["i2"], "parse: 산문 속 JSON 배열")
ok(parse_obligations("no json", "eid") is None, "parse: JSON 없음 → None(결정론 폴백)")
ok(parse_obligations('{"action": "x"}', "eid") is None, "parse: 배열 아님 → None")
ok(parse_obligations("[]", "eid") == [], "parse: 빈 배열 = 유효(의무 0)")
ok(parse_obligations('[{"eid": "W1"}, "junk", {"action": "cancel_tool", "entity": "W3"}]',
                     "eid") == [{"intent_class": "cancel_tool", "entity": "W3",
                                 "items": (), "qty": 1}],
   "parse: action 없는 항목·비dict 스킵 + entity 폴백 키")

# 9f. replan 경로 e2e(순수): 의무 2건 중 1건만 executed → gap 1건 (coverage_gap 재사용)
l9.set_replan(parse_obligations(
    '[{"action": "exchange", "eid": "W1", "items": ["i1"]},'
    ' {"action": "exchange", "eid": "W2", "items": ["i2"]}]', "eid"))
g9 = coverage_gap(l9)
ok(len(g9) == 1 and g9[0]["entity"] == "W2",
   "replan diff: 축약 intent로도 executed(W1) 커버·W2만 gap")
l9.note_write("exchange_tool", "W2", {"i2"})
ok(coverage_gap(l9) == [], "replan diff: W2 수행 후 gap=∅(즉시 종결·R1b)")

# 9g. transcript_text — dict 메시지·툴콜 라인·절단(앞뒤 보존)
msgs9 = [{"role": "user", "content": "hi"},
         {"role": "assistant", "content": "ok",
          "tool_calls": [{"name": "detail_tool", "arguments": '{"eid": "W1"}'}]}]
t9 = transcript_text(msgs9)
ok("[user] hi" in t9 and "[assistant->tool] detail_tool" in t9 and '"eid": "W1"' in t9,
   "transcript: 텍스트+툴콜 라인(dict 겸용)")
tlong = transcript_text([{"role": "user", "content": "A" * 300}], max_chars=100)
ok("[middle truncated]" in tlong and tlong.find("A") >= 0, "transcript: 절단 시 앞뒤 보존")

print("ALL PASS (%d checks)" % N)
