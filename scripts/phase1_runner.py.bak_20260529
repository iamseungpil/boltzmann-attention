"""Phase 1 Baseline runner: B0 / B1 / B2 on tau2-bench telecom.

B0 — Vanilla: default llm_agent (no scaffolding)
B1 — ReAct: AGENT_INSTRUCTION augmented with Thought/Action/Observation pattern
B2 — Text Serialization: domain_policy augmented with serialized ontology relations

All three use Qwen2.5-7B-Instruct served via local vLLM (OpenAI-compatible at :8000).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Patch module constants BEFORE importing run.* so any cached refs see the new value
from tau2.agent import llm_agent as _llm_agent_mod  # noqa: E402

ORIG_AGENT_INSTRUCTION = _llm_agent_mod.AGENT_INSTRUCTION

REACT_AGENT_INSTRUCTION = """
You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Reason explicitly before every action. Use this format:
  Thought: <one sentence explaining what you will do and why>
  Action: <the tool call or message you produce>
Then produce the actual tool call or message.

Always follow the policy. Always make sure you generate valid JSON only.
""".strip()


# Human-readable definitions of the 42 ontology relations (from EXPERIMENT_DESIGN.md §3.2.2).
RELATION_DEFINITIONS = {
    # Group A: 기초 5종
    "precedes": "A는 B보다 먼저 호출되어야 한다.",
    "requires": "A 호출 전 B가 성공해야 한다.",
    "enables": "A 호출 후 B가 가능해진다.",
    "mutex": "A와 B는 동시 호출 불가.",
    "workflow_role": "도구의 역할 (prerequisite/main/validation/cleanup).",
    # Group B: 서베이 7종
    "parameter_feeds": "A의 출력이 B의 입력이다.",
    "conditional_on": "조건 c가 충족될 때만 호출.",
    "validates": "A가 B의 결과를 검증한다.",
    "retry_after_fail": "A 실패 시 B로 재시도.",
    "compensates": "A가 B의 효과를 역전한다.",
    "parallel_safe": "A와 B는 순서 무관, 병렬 안전.",
    "precondition_state": "호출 전 상태 P 요구.",
    # Group C
    "causal_link": "A가 B의 결과를 인과적으로 야기.",
    "directly_follows": "프로세스 로그상 A 직후 B (강도 가중치).",
    # Group D
    "error_fallback": "오류 시 대체 도구로 폴백.",
    "tool_subsumes": "A가 B의 기능을 포함.",
    # Group E
    "and_join": "여러 갈래가 한 지점에 모두 합류.",
    "state_transition": "A 호출이 상태 S→S'를 야기.",
    "exclusive_choice": "분기 중 하나만 선택.",
    "effect_state": "도구가 종료 후 상태 변화.",
    # Group F
    "domain_category": "도구의 도메인 카테고리 (단항 속성).",
    "checkpoint": "복귀 가능한 중간 저장점.",
    "idempotent": "여러 번 호출해도 결과 동일.",
    "reversible": "호출을 되돌릴 수 있다.",
    "mandatory_in_flow": "흐름에서 반드시 호출되어야 함.",
    "optional_in_flow": "흐름에서 선택적 호출.",
    "loop_capable": "반복 호출 가능.",
    # Group G: GoT/ToT + Harness
    "fan_out": "한 도구가 여러 후보로 분기.",
    "pruned_by": "후보가 점수기로 가지치기됨.",
    "scored_preference": "맥락에서 A가 B보다 선호됨.",
    "backtrack_to": "막다른 길에서 상태 복원 지점으로 복귀.",
    "observation_triggers": "관찰이 특정 도구 호출을 유발.",
    "guardrail": "호출 금지 제약.",
    # Group H: HTN
    "decomposes_into": "목표가 도구 집합으로 분해.",
    "subtask_of": "도구가 목표에 포함 (카테고리).",
    "achieves_goal": "도구 호출이 목표를 달성.",
    "refines": "추상 행동이 구체 도구로 정제.",
    # Group I: GoalAct
    "plan_step_precedes": "플랜의 추상 단계 i가 j보다 먼저.",
    "plan_step_skill": "스텝이 어떤 skill 타입인지.",
    "plan_revised_to": "관찰 시 플랜 스텝이 새 스텝으로 교체.",
    "step_realizes_tool": "추상 계획 단계가 구체 도구로 실현.",
    "plan_committed_to_goal": "플랜 스텝이 목표에 커밋.",
}


def load_ontology_text(intervention_map_path: str) -> str:
    """Serialize PCLI intervention_map.json relations into a text block for B2."""
    with open(intervention_map_path) as f:
        data = json.load(f)
    rel_map = data["intervention_map"] if "intervention_map" in data else data
    lines = ["The following ontology relations describe tool-call planning constraints."]
    lines.append("Use them when reasoning about which tools to call and in what order.\n")
    for name, info in rel_map.items():
        if not isinstance(info, dict):
            continue
        if not info.get("go_nogo", True):
            continue  # skip relations that failed probing (refines, guardrail, domain_category)
        desc = RELATION_DEFINITIONS.get(name, "(no description)")
        geom = info.get("geometry", "?")
        lines.append(f"- {name} [{geom}]: {desc}")
    return "\n".join(lines)


def make_run_config(domain: str, task_set: str, task_split: str, agent_llm: str,
                    base_url: str, save_to: str, num_tasks: int | None,
                    num_trials: int, seed: int, max_concurrency: int,
                    max_steps: int,
                    user_llm: str | None = None,
                    user_base_url: str | None = None,
                    user_api_key: str | None = None,
                    timeout: float | None = None,
                    auto_resume: bool = False,
                    task_ids: list | None = None):
    from tau2.data_model.simulation import TextRunConfig

    # Agent LLM args (local vLLM by default)
    llm_args = {
        "api_base": base_url,
        "api_key": "sk-noauth",
        "temperature": 0.0,
    }

    # User simulator LLM args:
    # - if user_llm is None or matches agent_llm -> use same backend
    # - if user_llm specified -> use separate backend (e.g., OpenAI GPT-4o API)
    if user_llm is None or user_llm == agent_llm:
        effective_user_llm = agent_llm
        user_llm_args = dict(llm_args)
        user_llm_args["temperature"] = 0.5
    else:
        effective_user_llm = user_llm
        user_llm_args = {"temperature": 0.5}
        if user_base_url:
            user_llm_args["api_base"] = user_base_url
        if user_api_key:
            user_llm_args["api_key"] = user_api_key
        # else: rely on env var (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)

    cfg_kwargs = dict(
        domain=domain,
        agent="llm_agent",
        llm_agent=agent_llm,
        llm_args_agent=llm_args,
        user="user_simulator",
        llm_user=effective_user_llm,
        llm_args_user=user_llm_args,
        task_set_name=task_set,
        task_split_name=task_split,
        num_trials=num_trials,
        seed=seed,
        max_concurrency=max_concurrency,
        save_to=save_to,
        log_level="INFO",
        max_steps=max_steps,
    )
    if num_tasks is not None:
        cfg_kwargs["num_tasks"] = num_tasks
    if timeout is not None:
        cfg_kwargs["timeout"] = timeout
    if auto_resume:
        cfg_kwargs["auto_resume"] = True
    if task_ids:
        cfg_kwargs["task_ids"] = list(task_ids)
    return TextRunConfig(**cfg_kwargs)


def run_one(variant: str, args, ontology_text: str | None):
    if variant == "B0":
        _llm_agent_mod.AGENT_INSTRUCTION = ORIG_AGENT_INSTRUCTION
        domain_policy_suffix = ""
    elif variant == "B1":
        _llm_agent_mod.AGENT_INSTRUCTION = REACT_AGENT_INSTRUCTION
        domain_policy_suffix = ""
    elif variant == "B2":
        _llm_agent_mod.AGENT_INSTRUCTION = ORIG_AGENT_INSTRUCTION
        assert ontology_text is not None
        domain_policy_suffix = "\n\n<ontology>\n" + ontology_text + "\n</ontology>"
    else:
        raise ValueError(variant)

    # Patch SYSTEM_PROMPT to append ontology suffix if needed
    if domain_policy_suffix:
        base = """
<instructions>
{agent_instruction}
</instructions>
<policy>
{domain_policy}
</policy>
""".strip()
        _llm_agent_mod.SYSTEM_PROMPT = base + domain_policy_suffix
    else:
        _llm_agent_mod.SYSTEM_PROMPT = """
<instructions>
{agent_instruction}
</instructions>
<policy>
{domain_policy}
</policy>
""".strip()

    save_to = str(Path(args.out_dir) / f"{variant}_{args.task_set}_{args.task_split}.json")
    config = make_run_config(
        domain=args.domain,
        task_set=args.task_set,
        task_split=args.task_split,
        agent_llm=args.agent_llm,
        base_url=args.base_url,
        save_to=save_to,
        num_tasks=args.num_tasks,
        num_trials=args.num_trials,
        seed=args.seed,
        max_concurrency=args.max_concurrency,
        max_steps=args.max_steps,
        user_llm=args.user_llm,
        user_base_url=args.user_base_url,
        user_api_key=args.user_api_key,
        timeout=args.timeout,
        auto_resume=args.auto_resume,
        task_ids=getattr(args, "task_ids_list", None),
    )
    print(f"\n=== Running {variant} ===")
    print(f"  task_set={args.task_set} num_tasks={args.num_tasks} trials={args.num_trials}")
    print(f"  agent_llm={args.agent_llm} base_url={args.base_url}")
    print(f"  save_to={save_to}")

    from tau2.run import run_domain
    results = run_domain(config)
    print(f"  done. saved to {save_to}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["B0", "B1", "B2"],
                    choices=["B0", "B1", "B2"])
    ap.add_argument("--domain", default="telecom")
    ap.add_argument("--task-set", default="telecom",
                    help="telecom (default; uses splits) | telecom_full | telecom_small")
    ap.add_argument("--task-split", default="base",
                    help="small (N=20) | base (N=114) | train | test | full (N=2285)")
    ap.add_argument("--num-tasks", type=int, default=None)
    ap.add_argument("--num-trials", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-concurrency", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--timeout", type=float, default=None,
                    help="Per-sim wallclock timeout in seconds. None = no timeout.")
    ap.add_argument("--auto-resume", action="store_true",
                    help="Resume from existing save_to (skip completed sims, retry incomplete)")
    ap.add_argument("--agent-llm", default="openai/Qwen2.5-7B-Instruct")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--user-llm", default=None,
                    help="If specified, use a separate model for user simulator (e.g., 'openai/gpt-4o'). "
                         "Defaults to same as agent-llm.")
    ap.add_argument("--user-base-url", default=None,
                    help="API base URL for user simulator (default: official OpenAI/Anthropic endpoint).")
    ap.add_argument("--user-api-key", default=None,
                    help="API key for user simulator (default: read from env OPENAI_API_KEY etc).")
    ap.add_argument("--ontology-map",
                    default="/home/woori/workspace_common/boltzmann-attention-pi/reports/facet_rft_2026/phase0_probing/intervention_map.json")
    ap.add_argument("--out-dir",
                    default="/home/woori/workspace_common/boltzmann-attention-pi/reports/facet_rft_2026/phase1_baseline")
    ap.add_argument("--task-ids-file", default=None,
                    help="JSON file with list of task IDs (overrides --num-tasks)")
    args = ap.parse_args()

    # Load explicit task ids if provided.
    args.task_ids_list = None
    if args.task_ids_file:
        import json as _json
        with open(args.task_ids_file) as _f:
            args.task_ids_list = _json.load(_f)
        print(f"[phase1] using {len(args.task_ids_list)} explicit task ids from {args.task_ids_file}")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    ontology_text = None
    if "B2" in args.variants:
        ontology_text = load_ontology_text(args.ontology_map)
        print(f"[B2] ontology serialized to {len(ontology_text)} chars")

    for variant in args.variants:
        run_one(variant, args, ontology_text)


if __name__ == "__main__":
    main()
