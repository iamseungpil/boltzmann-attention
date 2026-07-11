#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""test_assembled_run.py — ⓑ 구현 게이트 (단위테스트로 [[05]] 드리프트 강제 차단).
사용자 리뷰 2026-06-25: 구현 중 "도와주려다" [[05]] 반복위반(autofetch/RetailGate) → 코드리뷰 아닌 *테스트 게이트*.
실행: python test_assembled_run.py  (GPU-free·전부 통과해야 full-run 자격)
게이트4(operand make-or-break task 71/74/101/102 스모크)=GPU 필요 → 드라이버 pre-run(reexp_assembled.sh)서.
"""
import os
import re
import sys
import inspect
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_interpreter as gi
from gate_interpreter import (GateInterpreter, compute_facts, candidate_summary,
                              resolvers_from_env, load_domain_a2)

RETAIL_FIELDS = ["variants", "available", "item_ids", "new_item_ids", "payment_method_id",
                 "options", "product_id", "orders_field"]
_fails = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        _fails.append(name)


# ─── 게이트1: join-resolver 금지 (#1·make-or-break rig 차단) ───
# present는 raw 엔티티 *열거*만·criterion pre-filter/join pre-resolve 금지.
print("게이트1 — join-resolver 금지 (enumerate-not-resolve):")
# (a) 행동: 2주문(서로 다른 주소) → present가 *둘 다* 열거·하나 pre-select 안 함
_orders = {"#A": {"order_id": "#A", "status": "delivered", "address": {"city": "NYC"}, "items": []},
           "#B": {"order_id": "#B", "status": "pending", "address": {"city": "LA"}, "items": []}}
_rv = {"resolve_field": lambda path, a: ["#A", "#B"] if path[-1] == "orders" else None,
       "fetch_record": lambda prod, idarg, idv: _orders.get(idv)}
g6 = {"kind": "select_confirm", "user_id_arg": "user_id", "user_producer": "get_user_details",
      "orders_field": "orders", "detail_producer": "get_order_details", "detail_id_arg": "order_id",
      "present_fields": ["status", "address"], "present_label": "order"}
summ = candidate_summary(_rv, g6, "u1")
check("present가 모든 주문 열거(둘 다)", summ and "#A" in summ and "#B" in summ and "NYC" in summ and "LA" in summ)
check("present가 단일 source pre-select 안 함(둘 다 노출)", summ and summ.count("city") == 2)
# (b) 구조: resolvers_from_env는 read-only 3개만 (신규 cross-entity join resolver 0)
class _Env:
    tools = None
_res = resolvers_from_env(_Env())
check("resolvers = {resolve_owner,resolve_field,fetch_record}만 (신규 join-resolver 0)",
      set(_res.keys()) == {"resolve_owner", "resolve_field", "fetch_record"}, str(set(_res.keys())))

# ─── 게이트2: 엔진 [[05]] — compute_facts·constraints에 retail 필드 리터럴 0 ───
print("게이트2 — 엔진 도메인-특화 0 (calc/constraints):")
src_calc = inspect.getsource(compute_facts)
hits_calc = [f for f in RETAIL_FIELDS if re.search(r'["\']' + re.escape(f) + r'["\']', src_calc)]
check("compute_facts 소스에 retail 필드 리터럴 0", not hits_calc, f"hits={hits_calc}")
# constraints 분기 소스 추출(check 메서드서 'elif kind == \"constraints\"' 블록)
src_check = inspect.getsource(GateInterpreter.check)
m = re.search(r'kind == "constraints":(.*?)elif kind ==', src_check, re.S)
src_con = m.group(1) if m else ""
hits_con = [f for f in RETAIL_FIELDS if re.search(r'["\']' + re.escape(f) + r'["\']', src_con)]
check("constraints 분기에 retail 필드 리터럴 0", not hits_con, f"hits={hits_con}")

# ─── 게이트3: report-conversion 계측 (#3·calc 주입이 파싱가능 마커) ───
print("게이트3 — report-conversion 계측:")
prod = {"variants": {"a": {"available": True}, "b": {"available": False}, "c": {"available": True}}}
facts = compute_facts(prod, [{"op": "count_where", "nested_field": "variants",
                              "cond_field": "available", "cond_value": True, "label": "available variants"}])
check("calc 주입에 [COMPUTED FACTS] 마커", facts and "[COMPUTED FACTS" in facts)
check("calc 주입이 파싱가능 'label: value'", facts and re.search(r"- available variants: 2", facts))

# ─── A2 정합 (calc_specs·G7 로드·disjoint 동작) ───
print("A2 정합:")
a2 = load_domain_a2("retail")
check("calc_specs 4개 로드(기존2 + CALC-EXT argmax/argmin 2)", len(a2.get("calc_specs") or []) == 4)
check("G7_OP_CONSTRAINTS(disjoint) 로드", any(g.get("id") == "G7_OP_CONSTRAINTS" for g in a2["gates"]))
gi_c = GateInterpreter([g for g in a2["gates"] if g.get("kind") == "constraints"])
ok1, _, _ = gi_c.check("exchange_delivered_order_items", {"item_ids": ["x"], "new_item_ids": ["x"]})
ok2, _, _ = gi_c.check("exchange_delivered_order_items", {"item_ids": ["x"], "new_item_ids": ["y"]})
check("disjoint: new==old → deny", not ok1)
check("disjoint: new!=old → allow", ok2)
ok3, _, _ = gi_c.check("modify_pending_order_items", {"item_ids": ["a", "b"], "new_item_ids": ["c"]})
ok4, _, _ = gi_c.check("modify_pending_order_items", {"item_ids": ["a"], "new_item_ids": ["c"]})
check("equal_len: count mismatch(2≠1) → deny (remove 차단)", not ok3)
check("equal_len: count match(1=1) → allow", ok4)

# ─── 게이트5: deny-present 부활 차단 (리뷰 2026-06-26·config 불변식) ───
# present는 read-증강(candidate_summary)만·select_confirm deny-경로는 replay 깸([[06]] 폐기).
# 드라이버 env가 select_confirm을 T2_GATE_KINDS서 제외 + T2_PRESENT_READS=1(read-aug 활성)을 강제하는지 test로 박음.
print("게이트5 — deny-present 부활 차단 (드라이버 config 불변식):")
_drv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reexp_assembled.sh")
if os.path.exists(_drv):
    _txt = open(_drv, encoding="utf-8").read()
    _gk = re.search(r"T2_GATE_KINDS=([^\s]+)", _txt)
    _kinds = (_gk.group(1) if _gk else "")
    check("드라이버 T2_GATE_KINDS에 select_confirm 없음(deny-present 제외)", "select_confirm" not in _kinds, _kinds)
    check("드라이버 T2_PRESENT_READS=1 (read-aug present 활성)", "T2_PRESENT_READS=1" in _txt)
    check("드라이버 T2_CALC=1 (calc arm 활성)", "T2_CALC=1" in _txt)
else:
    check("reexp_assembled.sh 존재", False, "드라이버 없음")

# ─── 게이트6: calc-주입 회귀 차단 (버그 2026-06-26·nested 오염→calc 미발화) ───
# nested가 out.content에 텍스트 append 후 calc가 *오염된* content를 재파싱하면 실패→calc 미발화(31/342).
# 수정 불변식: raw를 augment 전 1회 파싱(_rec) 후 nested·calc 둘 다 _rec 공유.
print("게이트6 — calc-주입 회귀 차단 (nested 오염 후에도 calc 발화):")
import t2_gate_patch as _tg
_src = inspect.getsource(_tg.apply)
check("calc 블록이 공유 _rec 사용(오염 content 재파싱 안 함)",
      "compute_facts(_rec" in _src and "compute_facts(_parse_json" not in _src and "compute_facts(rec" not in _src)
check("_rec를 augment 전 1회 파싱", "_rec = _parse_json(_content_str(out[0]))" in _src)
# 행동: nested가 먼저 append해도 calc가 raw _rec로 정상 계산
_prod = {"name": "T", "product_id": "p", "variants": {"a": {"item_id": "a", "available": True, "options": {}},
         "b": {"item_id": "b", "available": False, "options": {}}, "c": {"item_id": "c", "available": True, "options": {}}}}
_polluted = '{"x":1}' + "\n\n[OPERAND DISAMBIGUATION ...]\n- item_id=a: {}"  # nested가 오염시킨 형태
check("교훈: 오염 content는 JSON 파싱 실패(버그 재현)", _tg._parse_json(_polluted) is None)
check("교훈: raw record는 calc 정상", isinstance(compute_facts(_prod, [{"op": "count_where", "nested_field": "variants",
      "cond_field": "available", "cond_value": True, "label": "x"}]), str))

print()
if _fails:
    print(f"❌ {len(_fails)} GATE FAIL → full-run 자격 없음: {_fails}")
    sys.exit(1)
print("✅ 게이트1-3 + A2 전부 PASS — (게이트4=드라이버 task-smoke 후) full-run 자격")
