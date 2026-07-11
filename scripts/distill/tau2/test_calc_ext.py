#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""test_calc_ext.py — CALC-EXT(CENSUS_LEVERS_DESIGN_2026_07_11 §2a v1.1) 오프라인 단위테스트.
op 3종(argmax_where/argmin_where/most_recent) + 기존 op 동작-불변 회귀.
합성 레코드만 사용(GPU/네트워크 0). 실행: python test_calc_ext.py
경계: 동률·빈 후보·cond 불일치·non-dict 원소·rank 비숫자·dict-of-dict vs list.
(pairwise_diff_sum은 v1.1 리뷰서 notice 채널 이동 — 여기/compute_facts에 없음이 정상.)
"""
import os
import re
import sys
import inspect
if hasattr(sys.stdout, "reconfigure"):  # Windows cp949 콘솔 대비
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate_interpreter import compute_facts, load_domain_a2

_fails = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        _fails.append(name)


def facts_line(record, spec):
    """compute_facts 1-spec 실행 → '- label: ...' 라인(없으면 None)."""
    txt = compute_facts(record, [spec])
    if txt is None:
        return None
    for ln in txt.splitlines():
        if ln.startswith("- "):
            return ln
    return None


# ─── 합성 레코드 (retail 스키마 *형상*만 모사·테스트 데이터일 뿐 엔진과 무관) ───
# dict-of-dict (product.variants 형)
PROD = {"name": "X", "variants": {
    "i1": {"item_id": "i1", "available": True, "price": 10.0},
    "i2": {"item_id": "i2", "available": True, "price": 99.5},
    "i3": {"item_id": "i3", "available": False, "price": 200.0},   # cond 불일치(최고가지만 제외돼야)
    "i4": {"item_id": "i4", "available": True, "price": 5.25},
}}
ARGMAX = {"op": "argmax_where", "nested_field": "variants", "cond_field": "available",
          "cond_value": True, "rank_field": "price", "id_field": "item_id", "label": "max avail"}
ARGMIN = dict(ARGMAX, op="argmin_where", label="min avail")

print("1. argmax_where / argmin_where — 기본(dict-of-dict·cond 필터):")
ln = facts_line(PROD, ARGMAX)
check("argmax: cond=true 중 최고가 i2 (i3=200 unavailable 제외)",
      ln == "- max avail: item_id=i2 (price=99.5)", repr(ln))
ln = facts_line(PROD, ARGMIN)
check("argmin: cond=true 중 최저가 i4", ln == "- min avail: item_id=i4 (price=5.25)", repr(ln))

print("2. 동률 — 전부 나열:")
TIE = {"variants": {
    "a": {"item_id": "a", "available": True, "price": 7.0},
    "b": {"item_id": "b", "available": True, "price": 7.0},
    "c": {"item_id": "c", "available": True, "price": 3.0},
}}
ln = facts_line(TIE, ARGMAX)
check("argmax 동률(a·b=7.0) 둘 다 나열('; ' 연결)",
      ln is not None and "item_id=a (price=7.0)" in ln and "item_id=b (price=7.0)" in ln
      and "item_id=c" not in ln and "; " in ln, repr(ln))
ln = facts_line({"variants": {"c": {"item_id": "c", "available": True, "price": 3.0}}}, ARGMIN)
check("단일 후보=동률 아님·단독 나열", ln == "- min avail: item_id=c (price=3.0)", repr(ln))

print("3. 빈 후보 — 주입 자체가 없어야(None):")
check("cond 전부 불일치 → None",
      compute_facts({"variants": {"z": {"item_id": "z", "available": False, "price": 1.0}}}, [ARGMAX]) is None)
check("nested_field 부재 → None", compute_facts({"other": {}}, [ARGMAX]) is None)
check("빈 dict → None", compute_facts({"variants": {}}, [ARGMAX]) is None)
check("빈 list → None", compute_facts({"variants": []}, [ARGMAX]) is None)
check("record가 dict 아님 → None", compute_facts("not-a-dict", [ARGMAX]) is None)

print("4. non-dict 원소·rank 비숫자 — skip(크래시 0):")
MIXED = {"variants": [
    "junk-string", 42, None,
    {"item_id": "ok", "available": True, "price": 8.0},
    {"item_id": "bad", "available": True, "price": "N/A"},   # rank 비숫자 → skip
    {"item_id": "nul", "available": True},                    # rank 부재(None) → skip
]}
ln = facts_line(MIXED, ARGMAX)
check("list-형 + non-dict/비숫자 원소 skip·정상 원소만", ln == "- max avail: item_id=ok (price=8.0)", repr(ln))

print("5. cond_field 생략 — 무필터 argmax:")
NOCOND = {"op": "argmax_where", "nested_field": "variants", "rank_field": "price",
          "id_field": "item_id", "label": "max any"}
ln = facts_line(PROD, NOCOND)
check("cond 생략 시 전 원소 대상(i3=200 포함)", ln == "- max any: item_id=i3 (price=200.0)", repr(ln))

print("6. most_recent — 날짜 최대 원소(문자열 정렬·ISO형):")
ORDERS = {"orders": [
    {"order_id": "#A", "order_date": "2025-03-01"},
    {"order_id": "#B", "order_date": "2025-11-20"},
    {"order_id": "#C", "order_date": "2024-12-31"},
]}
MR = {"op": "most_recent", "nested_field": "orders", "date_field": "order_date",
      "id_field": "order_id", "label": "most recent order"}
ln = facts_line(ORDERS, MR)
check("최대 날짜 #B", ln == "- most recent order: order_id=#B (order_date=2025-11-20)", repr(ln))
TIE_D = {"orders": [{"order_id": "#A", "d": "2025-01-01"}, {"order_id": "#B", "d": "2025-01-01"},
                    {"order_id": "#C", "d": "2024-01-01"}, "junk", {"order_id": "#X"}]}
ln = facts_line(TIE_D, {"op": "most_recent", "nested_field": "orders", "date_field": "d",
                        "id_field": "order_id", "label": "mr"})
check("동률 전부 나열 + non-dict/date-부재 skip",
      ln is not None and "order_id=#A" in ln and "order_id=#B" in ln and "#C" not in ln and "#X" not in ln, repr(ln))
check("date_field 전부 부재 → None",
      compute_facts({"orders": [{"order_id": "#A"}]}, [MR]) is None)
check("빈 리스트 → None", compute_facts({"orders": []}, [MR]) is None)

print("7. 기존 op 동작 불변 (회귀):")
ln = facts_line(PROD, {"op": "count_where", "nested_field": "variants", "cond_field": "available",
                       "cond_value": True, "label": "n avail"})
check("count_where 불변", ln == "- n avail: 3", repr(ln))
ln = facts_line({"items": [{"price": 1.5}, {"price": 2.5}]},
                {"op": "sum", "nested_field": "items", "item_field": "price", "label": "total"})
check("sum 불변", ln == "- total: 4.0", repr(ln))
ln = facts_line({"items": [1, 2, 3]}, {"op": "count", "nested_field": "items", "label": "n"})
check("count 불변", ln == "- n: 3", repr(ln))
ln = facts_line({"status": "pending"}, {"op": "lookup", "field": "status", "label": "st"})
check("lookup 불변", ln == "- st: pending", repr(ln))

print("8. [[05]] — 엔진 소스에 retail 필드 리터럴 0 (신규 분기 포함):")
RETAIL_FIELDS = ["variants", "available", "item_id", "price", "orders", "order_id",
                 "order_date", "product_id", "options"]
src = inspect.getsource(compute_facts)
hits = [f for f in RETAIL_FIELDS if re.search(r'["\']' + re.escape(f) + r'["\']', src)]
check("compute_facts 소스에 retail 필드 리터럴 0", not hits, f"hits={hits}")
check("pairwise_diff_sum op 분기 미구현(§2a v1.1: notice 채널로 이동·docstring 언급만 허용)",
      '"pairwise_diff_sum"' not in src and "'pairwise_diff_sum'" not in src)

print("9. A2 정합 — retail calc_specs에 CALC-EXT 스펙:")
a2 = load_domain_a2("retail")
cs = a2.get("calc_specs") or []
ops = [s.get("op") for s in cs]
check("argmax_where 스펙 존재", "argmax_where" in ops)
check("argmin_where 스펙 존재", "argmin_where" in ops)
check("most_recent 스펙 *부재*(날짜 필드 없음 — 정직 판정)", "most_recent" not in ops)
check("pairwise_diff_sum 스펙 부재", "pairwise_diff_sum" not in ops)
check("기존 스펙 불변(count_where·sum 잔존)", "count_where" in ops and "sum" in ops)

print()
if _fails:
    print(f"❌ {len(_fails)} FAIL: {_fails}")
    sys.exit(1)
print("✅ CALC-EXT 단위 전부 PASS")
