#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run a tau2 domain with the LB engines installed. Same arguments the lane scripts already pass.

  cd $GO_TAU2 && PYTHONPATH=src:$REPO/scripts/distill/lb python $REPO/scripts/distill/lb/lb_run.py \
     --domain banking_knowledge --agent_model <id> --agent_base http://host:port/v1 \
     --user_llm openrouter/openai/gpt-5.2 --user_temp 0.0 --user_reasoning_effort low \
     --task_ids task_048 --num_trials 4 --max_concurrency 4 --max_steps 200 --save_to <TAG>

Flags: T2_LB1..T2_LB7 (default on). Paths: LB_SIDECAR, LB_DOCS_DIR. Nothing else.
"""

import argparse
import os

import lb_runtime


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="banking_knowledge")
    ap.add_argument("--retrieval_config", default="alltools")
    ap.add_argument("--agent_model", required=True)
    ap.add_argument("--agent_base", required=True)
    ap.add_argument("--user_llm", required=True)
    ap.add_argument("--user_temp", type=float, default=0.0)
    ap.add_argument("--user_reasoning_effort", default=None)
    ap.add_argument("--task_ids", default=None)
    ap.add_argument("--num_trials", type=int, default=1)
    ap.add_argument("--max_concurrency", type=int, default=4)
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--max_retries", type=int, default=None)
    ap.add_argument("--retry_delay", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max_tokens", type=int, default=int(os.environ.get("T2_AGENT_MAX_TOKENS", "8192")))
    ap.add_argument("--save_to", required=True)
    a = ap.parse_args()
    if any(t in a.user_llm.lower() for t in ("anthropic", "claude", "opus", "sonnet", "haiku")):
        raise SystemExit("[COST GUARD] frontier user-sim refused on the shared key")

    if a.domain == "banking_knowledge":
        import tau2.knowledge.sandbox_manager as sbm
        sbm._check_sandbox_dependencies = lambda *x, **k: None
    lb_runtime.install(a.domain)

    import tau2.evaluator.evaluator_nl_assertions as nle
    from tau2.data_model.simulation import TextRunConfig
    from tau2.run import run_domain
    user_args = {"temperature": a.user_temp}
    if a.user_reasoning_effort:
        user_args["reasoning_effort"] = a.user_reasoning_effort
    nle.DEFAULT_LLM_NL_ASSERTIONS = a.user_llm
    nle.DEFAULT_LLM_NL_ASSERTIONS_ARGS = {"temperature": 0.0, "response_format": {"type": "json_object"}}
    cfg = dict(domain=a.domain, agent="llm_agent", llm_agent="openai/" + a.agent_model,
               llm_args_agent={"api_base": a.agent_base, "api_key": "dummy", "temperature": 0.0, "max_tokens": a.max_tokens},
               llm_user=a.user_llm, llm_args_user=user_args, num_trials=a.num_trials,
               task_ids=a.task_ids.split(",") if a.task_ids else None, max_concurrency=a.max_concurrency,
               max_steps=a.max_steps, save_to=a.save_to)
    if a.retrieval_config:
        cfg["retrieval_config"] = a.retrieval_config
    for k in ("max_retries", "retry_delay", "seed"):
        if getattr(a, k) is not None:
            cfg[k] = getattr(a, k)
    run_domain(TextRunConfig(**cfg))


if __name__ == "__main__":
    main()
