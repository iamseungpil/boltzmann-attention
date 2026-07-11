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
    is_scope_token, load_eplan_spec, _covers, discovery_precondition)

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

print("ALL PASS (%d checks)" % N)
