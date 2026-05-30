#!/usr/bin/env python3
"""two_stage_agent.py — Layered hierarchical agent (design §15.11 / §15.12).

Wraps tau2's LLMAgent with a two-stage decision per turn:

  (1) PLANNER (LLM = the B abstract-SFT adapter, served via vLLM): emits an abstract
      PLAN_STEP as a "Plan: <step>" content prefix, plus — because it was trained on
      abstract SFT — a concrete tool call we keep as the LLM-fallback option.
  (2) EXECUTOR (DETERMINISTIC ontology resolver, ontology_resolver.OntologyResolver):
      turns (plan_step + accumulated read-state) into a concrete (tool, args) with NO
      LLM. On a miss it falls back to the planner's own (or a candidate-restricted) call.

Transfer = swap the per-domain ontology files; the planner / PLAN_STEP vocab stay fixed.

Ablation modes (--mode), matching design §15.11:
  base        : vanilla LLMAgent over the full tool set (no planner/resolver)        [i]
  resolver    : planner step + deterministic resolver; on miss KEEP the planner's
                concrete call (measures deterministic COVERAGE, no extra LLM)         [ii]
  fallback    : planner step + resolver + candidate-restricted LLM fallback on miss   [iii]
  monolithic  : the single abstract model emits Plan + concrete call directly, the
                resolver is bypassed (the B model used end-to-end)                     [iv]

Coverage: fraction of agent tool-call turns resolved DETERMINISTICALLY (resolver
decisive, all args filled) vs. fell back to the LLM. Printed + saved to
<out-dir>/coverage_<mode>_<domain>_<split>.json.

Implementation note: rather than touching tau2's agent registry, we monkeypatch
LLMAgent._generate_next_message (the same philosophy as phase1_runner.py, which patches
AGENT_INSTRUCTION/SYSTEM_PROMPT). Field names for the tau2 message objects
(ToolCall.id/name/arguments/requestor, tool message .id/.content, AssistantMessage
.content/.tool_calls) are the same ones the induce_* scripts consume from the serialized
trajectories, so the resolver's ObservedState ingests them unchanged.

Usage (NONE arm = abstract adapter trained system-mode none), telecom deterministic-coverage:
  python scripts/distill/two_stage_agent.py --mode fallback --domain telecom \
      --task-set telecom --task-split test \
      --agent-llm openai/qwen7b_telret_abstract --base-url http://127.0.0.1:9000/v1
LODO transfer (planner unchanged, swap ontology to airline):
  ... --domain airline --task-set airline --task-split test --ontology-domain airline
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ontology_resolver import OntologyResolver, ObservedState  # noqa: E402
from score_fix_coverage import is_read  # noqa: E402  (domain-general read/write classifier)

# tau2 imports (resolved from the seka_env install)
from tau2.agent import llm_agent as _llm_agent_mod  # noqa: E402
from tau2.data_model.message import (  # noqa: E402
    AssistantMessage,
    MultiToolMessage,
    ToolCall,
)
from tau2.utils.llm_utils import generate  # noqa: E402

PLAN_RE = re.compile(r"Plan:\s*([a-zA-Z_]+)")

# Module-level config + coverage accumulator (run_domain may run sims concurrently).
_CFG: dict = {"mode": "fallback", "resolver": None}
_COV_LOCK = threading.Lock()
_COV = {"tool_turns": 0, "deterministic": 0, "miss": 0, "no_step": 0,
        "by_tool_det": {}, "by_step_det": {}, "by_step_miss": {}}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_plan_step(content) -> str | None:
    if not content:
        return None
    m = PLAN_RE.search(content)
    return m.group(1) if m else None


def _msg_role(m):
    return getattr(m, "role", None)


def _observed_from_messages(messages) -> ObservedState:
    """Rebuild the accumulated read-state by replaying the conversation.

    Map each assistant tool_call id -> read-tool name, then feed every matching tool
    result into ObservedState (same keying the inducer used)."""
    obs = ObservedState()
    id2read: dict[str, str] = {}
    for m in messages:
        role = _msg_role(m)
        if role == "assistant":
            for tc in (getattr(m, "tool_calls", None) or []):
                if getattr(tc, "requestor", "assistant") != "assistant":
                    continue
                name = getattr(tc, "name", None)
                if name and is_read(name):
                    id2read[getattr(tc, "id", None)] = name
        elif role == "tool":
            read = id2read.get(getattr(m, "id", None))
            if read is not None:
                obs.update(read, getattr(m, "content", None))
    return obs


def _restricted_tools(agent, candidates):
    """Subset of the agent's tools restricted to the resolver's candidate set."""
    if not candidates:
        return agent.tools
    sub = [t for t in agent.tools if getattr(t, "name", None) in set(candidates)]
    return sub or agent.tools


def _record(deterministic=False, miss=False, no_step=False, tool=None, step=None):
    with _COV_LOCK:
        _COV["tool_turns"] += 1
        if no_step:
            _COV["no_step"] += 1
        if deterministic:
            _COV["deterministic"] += 1
            if tool:
                _COV["by_tool_det"][tool] = _COV["by_tool_det"].get(tool, 0) + 1
            if step:
                _COV["by_step_det"][step] = _COV["by_step_det"].get(step, 0) + 1
        elif miss:
            _COV["miss"] += 1
            if step:
                _COV["by_step_miss"][step] = _COV["by_step_miss"].get(step, 0) + 1


# ---------------------------------------------------------------------------
# The patched agent step
# ---------------------------------------------------------------------------

def _two_stage_generate(self, message, state):
    """Replacement for LLMAgent._generate_next_message implementing planner+executor."""
    # mirror the original message bookkeeping
    if isinstance(message, MultiToolMessage):
        state.messages.extend(message.tool_messages)
    else:
        state.messages.append(message)
    messages = state.system_messages + state.messages

    # Stage 1: planner LLM (abstract adapter emits "Plan: <step>" + a concrete call)
    am = generate(model=self.llm, tools=self.tools, messages=messages,
                  call_name="agent_response", **self.llm_args)

    mode = _CFG["mode"]
    resolver: OntologyResolver | None = _CFG["resolver"]

    # base / monolithic: no deterministic override
    if mode == "base":
        return am
    if not am.is_tool_call():
        return am  # the agent chose to message the user; nothing to resolve
    step = _parse_plan_step(am.content)
    if mode == "monolithic":
        _record(deterministic=False, tool=None, step=step)  # B model end-to-end
        return am
    if resolver is None or step is None:
        _record(no_step=(step is None), miss=(step is not None), step=step)
        return am

    # Stage 2: deterministic resolver
    res = resolver.resolve(step, _observed_from_messages(state.messages))
    if res is not None and res.decisive and not res.missing_args:
        tc0 = am.tool_calls[0]
        am.tool_calls = [ToolCall(id=getattr(tc0, "id", "call_0"),
                                  name=res.tool, arguments=res.args,
                                  requestor="assistant")]
        _record(deterministic=True, tool=res.tool, step=step)
        return am

    # miss -> fallback policy
    _record(miss=True, step=step)
    if mode == "fallback":
        cands = resolver.candidates_for(step)
        am2 = generate(model=self.llm, tools=_restricted_tools(self, cands),
                       messages=messages, call_name="agent_response_fallback",
                       **self.llm_args)
        return am2 if am2.is_tool_call() else am
    # mode == "resolver": keep the planner's own concrete call (no extra LLM)
    return am


# ---------------------------------------------------------------------------
# none-arm patches (abstract adapter trained with system-mode none) — mirrors phase1
# ---------------------------------------------------------------------------

def _apply_none_arm():
    _llm_agent_mod.AGENT_INSTRUCTION = ""
    _llm_agent_mod.SYSTEM_PROMPT = ""
    import tau2.utils.llm_utils as _llm_utils
    if getattr(_llm_utils, "_none_arm_patched", False):
        return
    _SystemMessage = _llm_utils.SystemMessage
    _orig = _llm_utils.validate_message

    def _patched(message):
        if isinstance(message, _SystemMessage):
            return
        return _orig(message)

    _llm_utils.validate_message = _patched
    _llm_utils._none_arm_patched = True


def _route_nl_judge_via_openrouter():
    judge = os.environ.get("TAU2_NL_JUDGE", "openrouter/openai/gpt-4.1")
    for modname in ("tau2.evaluator.evaluator_nl_assertions", "tau2.config"):
        try:
            mod = __import__(modname, fromlist=["DEFAULT_LLM_NL_ASSERTIONS"])
            if hasattr(mod, "DEFAULT_LLM_NL_ASSERTIONS"):
                mod.DEFAULT_LLM_NL_ASSERTIONS = judge
        except Exception as e:  # pragma: no cover
            print(f"[warn] nl-judge patch skip {modname}: {e}")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def _build_config(args, save_to):
    from tau2.data_model.simulation import TextRunConfig
    llm_args = {"api_base": args.base_url, "api_key": args.agent_api_key, "temperature": 0.0}
    if args.user_llm is None or args.user_llm == args.agent_llm:
        user_llm = args.agent_llm
        user_llm_args = dict(llm_args); user_llm_args["temperature"] = 0.5
    else:
        user_llm = args.user_llm
        user_llm_args = {"temperature": 0.5}
        if args.user_base_url:
            user_llm_args["api_base"] = args.user_base_url
        if args.user_api_key:
            user_llm_args["api_key"] = args.user_api_key
    kw = dict(domain=args.domain, agent="llm_agent", llm_agent=args.agent_llm,
              llm_args_agent=llm_args, user="user_simulator", llm_user=user_llm,
              llm_args_user=user_llm_args, task_set_name=args.task_set,
              task_split_name=args.task_split, num_trials=args.num_trials, seed=args.seed,
              max_concurrency=args.max_concurrency, save_to=save_to, log_level="INFO",
              max_steps=args.max_steps)
    if args.num_tasks is not None:
        kw["num_tasks"] = args.num_tasks
    if args.timeout is not None:
        kw["timeout"] = args.timeout
    if args.auto_resume:
        kw["auto_resume"] = True
    return TextRunConfig(**kw)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="fallback",
                    choices=["base", "resolver", "fallback", "monolithic"])
    ap.add_argument("--domain", default="telecom")
    ap.add_argument("--ontology-domain", default=None,
                    help="Domain whose ontology the resolver uses (default = --domain). "
                         "For an ABox-swap ablation set this != --domain.")
    ap.add_argument("--induced-dir",
                    default="reports/facet_rft_2026/phase4_distill/induced")
    ap.add_argument("--task-set", default="telecom")
    ap.add_argument("--task-split", default="test")
    ap.add_argument("--num-tasks", type=int, default=None)
    ap.add_argument("--num-trials", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-concurrency", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--min-score", type=float, default=1.0,
                    help="Resolver decisiveness threshold (sum precision*support).")
    ap.add_argument("--timeout", type=float, default=None)
    ap.add_argument("--auto-resume", action="store_true")
    ap.add_argument("--agent-llm", default="openai/qwen7b_telret_abstract")
    ap.add_argument("--base-url", default="http://127.0.0.1:9000/v1")
    ap.add_argument("--agent-api-key", default="sk-noauth")
    ap.add_argument("--user-llm", default=None)
    ap.add_argument("--user-base-url", default=None)
    ap.add_argument("--user-api-key", default=None)
    ap.add_argument("--system-mode", default="none", choices=["none", "full"])
    ap.add_argument("--out-dir",
                    default="/home/woori/workspace_common/boltzmann-attention-pi/reports/facet_rft_2026/phase4_distill/two_stage")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    _route_nl_judge_via_openrouter()
    if args.system_mode == "none":
        _apply_none_arm()

    # Configure the two-stage decision.
    _CFG["mode"] = args.mode
    if args.mode in ("resolver", "fallback"):
        ont_dom = args.ontology_domain or args.domain
        _CFG["resolver"] = OntologyResolver(ont_dom, induced_dir=args.induced_dir,
                                            min_score=args.min_score)
        print(f"[two_stage] resolver ontology = {ont_dom} "
              f"(steps: {dict((s, len(c)) for s, c in _CFG['resolver'].candidates_by_step.items())})")

    # Install the patched agent step (skipped for pure base, but harmless).
    _llm_agent_mod.LLMAgent._generate_next_message = _two_stage_generate

    save_to = str(Path(args.out_dir) /
                  f"{args.mode}_{args.task_set}_{args.task_split}.json")
    cfg = _build_config(args, save_to)
    print(f"=== two_stage mode={args.mode} domain={args.domain} "
          f"agent={args.agent_llm} base={args.base_url} ===")
    print(f"  save_to={save_to}")

    from tau2.run import run_domain
    run_domain(cfg)

    # Coverage report.
    tt = max(_COV["tool_turns"], 1)
    cov = {
        "mode": args.mode, "domain": args.domain,
        "ontology_domain": args.ontology_domain or args.domain,
        "tool_turns": _COV["tool_turns"],
        "deterministic": _COV["deterministic"],
        "miss": _COV["miss"], "no_step": _COV["no_step"],
        "deterministic_coverage": round(_COV["deterministic"] / tt, 4),
        "by_tool_det": _COV["by_tool_det"],
        "by_step_det": _COV["by_step_det"],
        "by_step_miss": _COV["by_step_miss"],
    }
    cov_path = Path(args.out_dir) / f"coverage_{args.mode}_{args.task_set}_{args.task_split}.json"
    json.dump(cov, open(cov_path, "w"), indent=2, ensure_ascii=False)
    print(f"\n[coverage] deterministic={cov['deterministic']}/{cov['tool_turns']} "
          f"= {cov['deterministic_coverage']:.1%}  miss={cov['miss']}  no_step={cov['no_step']}")
    print(f"  by_step_det={cov['by_step_det']}")
    print(f"  -> {cov_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
