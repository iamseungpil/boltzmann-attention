#!/usr/bin/env python
"""P-A2-0b: A2 컴파일 크기-하한 census (A2_FRONTEND_DISTILL §6).

동일 프롬프트(GATE_SPEC 형식 설명 + retail 1-shot + airline 정책 + A1 카탈로그)로
각 모델이 airline GATE_SPEC을 컴파일 → Fable-5 reference 대비 채점:
  ①게이트 발견률(6중 몇 개를 applies_to-매칭으로 찾았나) ②applies_to F1 ③핵심술어 recall.
v1 채점 = 구조·키워드 tier (생성 db_check prose의 replay 실행은 DSL 후).

Usage: --api_base http://localhost:8000/v1 --model <served> [--openrouter] --out <dir> [--k 4]
"""
import argparse, json, os, urllib.request

REF_KEYWORDS = {
    "G1": ["user id", "provide"],
    "G2": ["confirm", "yes"],
    "G3": ["belong", "own"],
    "G4": ["24", "business", "insurance", "cancel"],
    "G5": ["basic", "passenger", "baggage"],
    "G6": ["certificate", "credit", "gift", "profile"],
}


def build_prompt(tb_root):
    retail_spec = open(f"{tb_root}/retail_gate_spec.json").read() if os.path.exists(
        f"{tb_root}/retail_gate_spec.json") else "{}"
    airline_policy = open(f"{tb_root}/airline_policy.md").read()
    airline_cat = open(f"{tb_root}/airline_tool_catalog.json").read()
    return f"""You are a policy compiler. Convert a domain policy document into a GATE_SPEC JSON: a dict of deterministic conversation-level gates. Each gate has:
- "predicate": condition that must hold before the listed tools may run
- "satisfiers": dict tool_name -> [required user-provided inputs] (empty if satisfied conversationally)
- "ask": what to ask the user when not satisfied (optional)
- "terminal": message when the condition can never be satisfied (opt