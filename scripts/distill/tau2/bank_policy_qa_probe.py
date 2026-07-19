#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""policy_qa (W)wrap 기능서브 무료 프로브 — `FUNCTION_AGENT_ISOLATION_DESIGN_2026_07_19` §5-1 측정 게이트.

라이브 fidelity([[30]] 프로브≠라이브 함정 회피):
  - 서브 = **실제 엔진 `SG._sub_wrap`** (별도 재구현 0·[[03b]])
  - getter 실행 = **실제 tau2 env**(banking_knowledge)의 KB 도구 결정론 실행
  - 질의 = **계열 궤적(7b/7c results.json)의 실제 KB_search/shell 호출** 전수(중복 제거)
계측(판정은 [[08]] 정독과 함께):
  (1) 폴백률(None 반환)  (2) quote grounding 통과/드롭 수  (3) 압축률(원 덤프 vs wrap 반환)
  (4) answer 전문 출력 → 수동 정독으로 정보 충분성 판단(Δspurious 후보 식별)
게이트(§5): found-답변의 grounding 전멸 0 · 압축 실질(≥80%) · answer가 원 덤프의 결정-관련 정보 보존.

Run(리모트): /home/woori/venvs/seka_env/bin/python bank_policy_qa_probe.py \
  --base http://localhost:8140/v1 \
  --results '/home/woori/scratch/tau2-bench/data/simulations/bank_redesign7b_20260719/results.json,/home/woori/scratch/tau2-bench/data/simulations/bank_redesign7c_20260719/results.json'
"""
import argparse
import json
import os
import sys
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")
import t2_scaffold_get as SG  # noqa: E402

DOM = "banking_knowledge"


def extract_queries(paths):
    """계열 궤적서 에이전트 KB_search/shell 호출 전수 추출(중복 제거·태스크 태깅)."""
    seen, out = set(), []
    for p in paths:
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception as e:
            print("  [skip] %s: %r" % (p, e))
            continue
        for s in d.get("simulations", []):
            for m in s.get("messages", []):
                for tc in (m.get("tool_calls") or []):
                    if tc.get("requestor", "assistant") != "assistant":
                        continue
                    n = tc.get("name")
                    if n not in ("KB_search_bm25", "KB_search_dense", "shell"):
                        continue
                    k = (n, json.dumps(tc.get("arguments") or {}, sort_keys=True))
                    if k in seen:
                        continue
                    seen.add(k)
                    out.append({"task": s.get("task_id"), "name": n,
                                "args": tc.get("arguments") or {}})
    return out


def build_env():
    """실제 banking_knowledge env (KB 도구 결정론 실행용)."""
    from tau2.registry import registry
    for getter in ("get_env_constructor", "get_env"):
        fn = getattr(registry, getter, None)
        if fn is None:
            continue
        try:
            ctor = fn(DOM)
            env = ctor() if callable(ctor) else ctor
            return env
        except Exception as e:
            print("  [env] %s 실패: %r" % (getter, e))
    raise SystemExit("env 구축 실패")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8140/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    ap.add_argument("--results", required=True, help="콤마구분 results.json 경로들")
    ap.add_argument("--max", type=int, default=30)
    a = ap.parse_args()

    fa = json.load(open(os.path.join(HERE, "a2", "banking_knowledge.gate.json"),
                        encoding="utf-8"))["function_agents"][0]
    queries = extract_queries([p.strip() for p in a.results.split(",") if p.strip()])[: a.max]
    print("★policy_qa 프로브 — 질의 %d개 (계열 궤적 실호출·중복 제거)\n" % len(queries))

    env = build_env()
    tools = list(env.get_tools() or [])
    names = {getattr(t, "name", None) for t in tools}
    print("env tools: %s\n" % sorted(x for x in names if x))
    getters = set(fa.get("getter_tools") or [])
    if not (getters & names):
        raise SystemExit("getter %s가 env에 없음 — retrieval config 확인 필요" % sorted(getters))

    ag = SimpleNamespace(
        llm="openai/" + a.model,
        llm_args={"api_base": a.base, "api_key": "dummy", "temperature": 0.0},
        tools=[t for t in tools if getattr(t, "name", None) in getters])
    orch = SimpleNamespace(agent=ag, environment=env)
    if not getattr(env, "domain_name", None):
        try:
            env.domain_name = DOM
        except Exception:
            orch.environment = SimpleNamespace(domain_name=DOM)

    from tau2.data_model.message import ToolCall

    def run_env(tcs):
        return [env.get_response(t) for t in tcs]

    stats = {"n": 0, "fallback": 0, "ok": 0, "orig_chars": 0, "wrap_chars": 0, "dropped": 0}
    for i, q in enumerate(queries):
        tc = ToolCall(id="probe%d" % i, name=q["name"], arguments=q["args"],
                      requestor="assistant")
        try:
            orig = env.get_response(tc)
            olen = len(str(getattr(orig, "content", "") or ""))
        except Exception as e:
            olen = -1
            print("  [%d] 원 실행 실패 %r" % (i, e))
        txt = None
        try:
            txt = SG._sub_wrap(orch, fa, tc, run_env)
        except Exception as e:
            print("  [%d] _sub_wrap 예외 %r" % (i, str(e)[:200]))
        stats["n"] += 1
        print("=" * 100)
        print("[%d] %s %s %s" % (i, q["task"], q["name"],
                                 json.dumps(q["args"], ensure_ascii=False)[:160]))
        if txt is None:
            stats["fallback"] += 1
            print("  → 폴백(None) · 원 덤프 %d chars가 메인에 남았을 것" % olen)
        else:
            stats["ok"] += 1
            stats["orig_chars"] += max(olen, 0)
            stats["wrap_chars"] += len(txt)
            print("  → wrap %d chars (원 %d · 압축 %.0f%%)" % (
                len(txt), olen, 100.0 * (1 - len(txt) / olen) if olen > 0 else 0))
            print("  ---- 반환 전문 ----")
            print("  " + txt.replace("\n", "\n  "))
    print("\n" + "#" * 100)
    print("총 %d · wrap 성공 %d · 폴백 %d" % (stats["n"], stats["ok"], stats["fallback"]))
    if stats["orig_chars"]:
        print("압축: 원 %d chars → wrap %d chars (%.0f%% 절감)" % (
            stats["orig_chars"], stats["wrap_chars"],
            100.0 * (1 - stats["wrap_chars"] / stats["orig_chars"])))


if __name__ == "__main__":
    main()
