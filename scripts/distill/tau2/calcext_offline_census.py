#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""calcext_offline_census.py — CALC-EXT(§2a v1.1) V0 무료 오프라인 census.

질문: t20 실패 trial들의 get_product_details 실출력에 새 argmax_where 스펙(A2 retail
calc_specs)을 오프라인 적용하면, 주입될 주석의 item_id가 t20 gold(new_item_ids)와
일치하는가? — 일치 product 수 = 이 레버가 t20에 줄 수 있는 재료의 상한(무료 판정).

gold 출처: RETAIL_FULL_FAIL_CENSUS_2026_07_11 라우팅의 t20 GOLD 4 ids
(4579334072·1151293680·4107812777·2882812427)를 상수/인자로 받음. 단 comp gz 자체의
tasks[].evaluation_criteria.actions(modify_pending_order_items의 new_item_ids)에도 gold가
실재하므로(지시서의 "gz에 gold 없음" 가정과 달리 tasks 블록엔 있음) 상수와 교차검증한다.

부수 판정(most_recent): 전 sim·전 tool 출력에서 날짜형 필드(date/time/created/…) 유무를
전수 스캔 → retail서 most_recent 트리거 가능 여부를 정직 판정(t71 커버 가능성).

실행: python calcext_offline_census.py [--gz PATH] [--task 20] [--gold id1,id2,...]
GPU/네트워크 0 (로컬 gz만).
"""
import argparse
import gzip
import io
import json
import os
import re
import sys

if hasattr(sys.stdout, "reconfigure"):  # Windows cp949 콘솔 대비
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate_interpreter import compute_facts, load_domain_a2

DEFAULT_GZ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
                          "reports", "facet_rft_2026", "sim_results", "comp_retail_t4.results.json.gz")
# gold 상수 출처: CALC-EXT 지시(라우팅 원문=RETAIL_FULL_FAIL_CENSUS_2026_07_11 t20).
# gz tasks[20].evaluation_criteria.actions '20_8' new_item_ids와 교차검증(아래 xcheck).
DEFAULT_GOLD = ["4579334072", "1151293680", "4107812777", "2882812427"]

DATE_KEY = re.compile(r"date|time|created|updated|timestamp", re.I)


def tool_outputs(sim):
    """(tool_name, content_str) — assistant tool_calls의 id→name을 role=tool 메시지 id와 매칭."""
    id2name = {}
    for m in sim.get("messages") or []:
        for tc in (m.get("tool_calls") or []):
            id2name[tc.get("id")] = tc.get("name")
        if m.get("role") == "tool" and isinstance(m.get("content"), str):
            yield id2name.get(m.get("id")), m["content"]


def parse_record(content):
    """tool content 문자열 → JSON record (기존 주입 블록 '\n\n[' 이후는 잘라냄)."""
    raw = content.split("\n\n[")[0]
    try:
        return json.loads(raw)
    except Exception:
        return None


def sim_failed(sim):
    ri = sim.get("reward_info") or {}
    r = ri.get("reward")
    return r is not None and float(r) < 1.0


def walk_date_fields(obj, path, hits):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if DATE_KEY.search(str(k)):
                hits.add(f"{path}.{k}")
            walk_date_fields(v, f"{path}.{k}", hits)
    elif isinstance(obj, list):
        for v in obj[:3]:
            walk_date_fields(v, path + "[]", hits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gz", default=DEFAULT_GZ)
    ap.add_argument("--task", default="20")
    ap.add_argument("--gold", default=",".join(DEFAULT_GOLD),
                    help="gold new_item_ids (콤마구분·출처는 파일 상단 주석)")
    args = ap.parse_args()
    gold = [g.strip() for g in args.gold.split(",") if g.strip()]

    d = json.load(gzip.open(args.gz, "rt", encoding="utf-8"))
    sims = d["simulations"]

    # ── 0. gold 교차검증: gz tasks 블록의 evaluation_criteria에서 new_item_ids 추출 ──
    gz_gold = None
    for t in d.get("tasks") or []:
        if str(t.get("id")) == str(args.task):
            for a in ((t.get("evaluation_criteria") or {}).get("actions") or []):
                nii = (a.get("arguments") or {}).get("new_item_ids")
                if nii:
                    gz_gold = [str(x) for x in nii]
    print(f"[xcheck] gold 상수 {gold}")
    print(f"[xcheck] gz tasks[{args.task}] eval new_item_ids = {gz_gold}"
          + ("  → 일치" if gz_gold == gold else "  → ★불일치(검토 필요)" if gz_gold else "  (gz에 없음)"))

    # ── 1. A2 스펙 로드 (argmax_where만 — 이 census의 질문) ──
    a2 = load_domain_a2("retail")
    specs = [s for s in (a2.get("calc_specs") or [])
             if s.get("op") == "argmax_where" and s.get("trigger_tool") == "get_product_details"]
    if not specs:
        print("FATAL: retail A2에 get_product_details argmax_where 스펙 없음")
        sys.exit(1)

    # ── 2. t20 실패 trial들의 get_product_details 출력 → compute_facts 재계산 ──
    tsims = [s for s in sims if str(s.get("task_id")) == str(args.task)]
    fails = [s for s in tsims if sim_failed(s)]
    print(f"\n[t{args.task}] sims={len(tsims)}  failed={len(fails)} "
          f"(rewards={[ (s.get('reward_info') or {}).get('reward') for s in tsims ]})")

    ID_RE = re.compile(r"item_id=(\S+) \(")
    per_product = {}   # product_id -> {"argmax": set(ids), "gold_in_variants": set(gold∩variants)}
    for sim in fails:
        for nm, content in tool_outputs(sim):
            if nm != "get_product_details":
                continue
            rec = parse_record(content)
            if not isinstance(rec, dict):
                continue
            pid = str(rec.get("product_id"))
            facts = compute_facts(rec, specs)
            annotated = set(ID_RE.findall(facts)) if facts else set()
            variants = rec.get("variants") or {}
            slot = per_product.setdefault(pid, {"name": rec.get("name"), "argmax": set(),
                                                "gold_in_variants": set(), "n_out": 0})
            slot["n_out"] += 1
            slot["argmax"] |= annotated
            slot["gold_in_variants"] |= {g for g in gold if g in variants}

    # ── 3. 대조: 주석 item_id ∈ gold ? ──
    print(f"\n=== census: 주석(argmax_where·most expensive available) vs gold ===")
    match_products = 0
    gold_products = 0
    for pid, s in sorted(per_product.items()):
        gv = s["gold_in_variants"]
        if not gv:
            print(f"  product {pid} ({s['name']}): gold 변형 없음(비표적 조회) — argmax={sorted(s['argmax'])}")
            continue
        gold_products += 1
        verdict = "MATCH" if s["argmax"] & gv else "MISS"
        if s["argmax"] & gv:
            match_products += 1
        print(f"  product {pid} ({s['name']}): argmax={sorted(s['argmax'])} · gold(이 product)={sorted(gv)}"
              f" · 조회 {s['n_out']}회 → {verdict}")
    print(f"\n요약: gold-표적 product {gold_products}개 중 주석=gold 일치 {match_products}개"
          f" (t{args.task} 실패 {len(fails)} trial의 실출력 기준)")

    # ── 4. most_recent 트리거 가능성: 전 sim·전 tool 출력 날짜형 필드 전수 스캔 ──
    hits = set()
    for sim in sims:
        for nm, content in tool_outputs(sim):
            rec = parse_record(content)
            if rec is not None:
                walk_date_fields(rec, nm or "?", hits)
    print(f"\n=== most_recent 판정: 전 {len(sims)} sim tool 출력의 날짜형 필드 ===")
    if hits:
        for h in sorted(hits):
            print("  ", h)
    else:
        print("  없음 → retail선 most_recent 트리거 불가(A2 스펙 미부착이 정직) — t71은 이 op로 못 닫음")


if __name__ == "__main__":
    main()
