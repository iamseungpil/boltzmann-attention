#!/usr/bin/env python
"""tau2 retail ±게이트 실행 러너 (BENCH_PORTFOLIO §3.6 ③ — 7B base census, R7).

--gate 1 이면 t2_gate_patch 적용 후 run_domain. 에이전트/유저-sim = 로컬 vllm
OpenAI-호환 엔드포인트 (litellm openai/<served-name> + api_base).

Run: cd /home/woori/scratch/tau2-bench && PYTHONPATH=src:$REPO/scripts/distill/tau2 \
  /home/woori/venvs/seka_env/bin/python $REPO/scripts/distill/tau2/t2_run_gated.py \
  --gate 1 --num_trials 4 --save_to retail_7b_gate
"""
import argparse
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", type=int, default=0)
    ap.add_argument("--resolve", type=int, default=0,
                    help="resolve_selection live wiring (t2_resolve_patch) on — 실 e2e operand offload")
    ap.add_argument("--domain", default="retail")
    ap.add_argument("--agent_model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--agent_base", default="http://localhost:8351/v1")
    ap.add_argument("--user_model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    ap.add_argument("--user_base", default="http://localhost:8352/v1")
    ap.add_argument("--user_llm", default=None,
                    help="full litellm model string override (e.g. openrouter/openai/gpt-4.1-mini; "
                         "key는 env OPENROUTER_API_KEY) — judge도 이 모델을 따름")
    ap.add_argument("--agent_llm", default=None,
                    help="full litellm AGENT override (예 openrouter/openai/gpt-4.1) — "
                         "frontier-arm F4b census용; 지정 시 로컬 vllm 불요")
    ap.add_argument("--user_temp", type=float, default=0.7,
                    help="user-sim temperature (ⓟ1 분산통제 arm = 0.0)")
    ap.add_argument("--agent_seed", type=int, default=None,
                    help="agent 요청 seed 고정 (결정론 실험; 로컬 vllm arm 전용)")
    ap.add_argument("--allow-frontier", action="store_true", dest="allow_frontier",
                    help="COST GUARD override: explicitly allow Claude/Anthropic (expensive) models "
                         "on the shared OpenRouter key. Default REFUSES — user-sim must be gpt-4.1.")
    ap.add_argument("--num_trials", type=int, default=1)
    ap.add_argument("--num_tasks", type=int, default=None)
    ap.add_argument("--max_concurrency", type=int, default=8)
    ap.add_argument("--save_to", required=True)
    a = ap.parse_args()

    # ---- COST GUARD (2026-06-16 incident: ~$600 OpenRouter drain via Claude on the shared key) ----
    # The "facet-rft-tau2-user-sim" OpenRouter key bills ALL calls incl. the AGENT. Claude is
    # ~15-30x gpt-4.1 and agentic runs are token-heavy, so a single mis-called --agent_llm/--user_llm
    # = anthropic/claude-* drains the budget. user-sim MUST be gpt-4.1. Refuse unless --allow-frontier.
    def _is_claude(m):
        return bool(m) and any(t in str(m).lower() for t in ("anthropic", "claude", "opus", "sonnet", "haiku"))
    _bad = [(n, v) for n, v in (("--user_llm", a.user_llm), ("--agent_llm", a.agent_llm)) if _is_claude(v)]
    if _bad and not a.allow_frontier:
        raise SystemExit(
            "[COST GUARD] REFUSING Claude/Anthropic model(s) on the shared OpenRouter key: "
            + ", ".join(f"{n}={v}" for n, v in _bad)
            + "\n  user-sim MUST be gpt-4.1 (openrouter/openai/gpt-4.1). Claude is ~15-30x the price"
              " and the AGENT calls bill to this key too — this is the 2026-06-16 $600 drain.\n"
            "  If you REALLY intend an expensive frontier arm: set an OpenRouter per-key spend cap"
            " FIRST, then re-run with --allow-frontier.")
    if _bad:  # allowed but loud
        print("[COST GUARD][WARN] frontier override ON — billing Claude to the OpenRouter key: "
              + ", ".join(f"{n}={v}" for n, v in _bad))

    if a.resolve:
        import t2_resolve_patch
        t2_resolve_patch.apply()
        print("[t2_run] resolve_selection wiring ON")

    if a.gate:
        import t2_gate_patch
        t2_gate_patch.apply()
        print("[t2_run] gate ON")
        regen_on = os.environ.get("T2_PROV_REGEN") == "1"
        badwords_on = os.environ.get("T2_PROV_BADWORDS", "0") == "1"
        ground_on = os.environ.get("T2_PROV_GROUND", "0") == "1"
        if regen_on or badwords_on or ground_on:
            # GROUND는 regen 인프라(생성-레벨 작업본) 위에서 동작 → regen 경로 활성 필요
            t2_gate_patch.apply_provenance_regen(
                max_retries=int(os.environ.get("T2_PROV_REGEN_K", "4")) if (regen_on or ground_on) else 0,
                use_badwords=badwords_on,
                ground=ground_on)
            print("[t2_run] provenance L1(badwords)=%s L2(regen)=%s GROUND=%s"
                  % (badwords_on, regen_on, ground_on))

    # user-sim·judge 모델 결정: --user_llm(원격 API, 예 openrouter/...) 우선, 아니면 로컬 vllm
    if a.user_llm:
        user_llm, user_args = a.user_llm, {"temperature": a.user_temp}
        judge_model, judge_args = a.user_llm, {"temperature": 0.0,
                                               "response_format": {"type": "json_object"}}
    else:
        user_llm = f"openai/{a.user_model}"
        user_args = {"api_base": a.user_base, "api_key": "dummy", "temperature": 0.7}
        judge_model = f"openai/{a.user_model}"
        judge_args = {"temperature": 0.0, "api_base": a.user_base, "api_key": "dummy",
                      "response_format": {"type": "json_object"}}

    # NL-assertion judge가 gpt-4.1 하드 기본값(config.py) → 재바인딩
    # (40/114 태스크가 nl_assertions 보유 — 미패치 시 키 부재로 영구 실패, 2026-06-12 사고.
    #  judge가 json.loads(content) 직접 호출 — response_format으로 코드펜스/서문 차단)
    import tau2.evaluator.evaluator_nl_assertions as _nle
    _nle.DEFAULT_LLM_NL_ASSERTIONS = judge_model
    _nle.DEFAULT_LLM_NL_ASSERTIONS_ARGS = judge_args
    print(f"[t2_run] user-sim={user_llm} judge={judge_model}")

    from tau2.data_model.simulation import TextRunConfig
    from tau2.run import run_domain

    if a.agent_llm:
        llm_agent, llm_args_agent = a.agent_llm, {"temperature": 0.0}
    else:
        llm_agent = f"openai/{a.agent_model}"
        llm_args_agent = {"api_base": a.agent_base, "api_key": "dummy", "temperature": 0.0}
    # 결정론 실험(p1 재개): agent 요청에 seed 고정 (vLLM은 seed param 지원 — 배칭
    # 비결정성은 serve-side enforce-eager/max-num-seqs=1로, seed는 샘플링-RNG 보조)
    if a.agent_seed is not None and not a.agent_llm:
        llm_args_agent["seed"] = a.agent_seed
    cfg = TextRunConfig(
        domain=a.domain,
        agent="llm_agent",
        llm_agent=llm_agent,
        llm_args_agent=llm_args_agent,
        llm_user=user_llm,
        llm_args_user=user_args,
        num_trials=a.num_trials,
        num_tasks=a.num_tasks,
        max_concurrency=a.max_concurrency,
        save_to=a.save_to,
    )
    results = run_domain(cfg)
    sims = getattr(results, "simulations", [])
    rewards = [getattr(s, "reward_info", None) and s.reward_info.reward for s in sims]
    rewards = [r for r in rewards if r is not None]
    if rewards:
        print(f"[t2_run RESULT] gate={a.gate} n={len(rewards)} "
              f"mean_reward={sum(rewards) / len(rewards):.4f} "
              f"pass1={sum(r >= 1 for r in rewards)}/{len(rewards)}")
    else:
        print(f"[t2_run RESULT] gate={a.gate} no rewards parsed — check save file")

    # eval-후크: compliant-pass(F4b) 자동 산출 — 저장된 results.json 위 replay,
    # compliance.json 사이드카. 실패해도 본 결과에 영향 없음.
    try:
        from t2_compliance import report_for_dir
        sim_dir = os.path.join("data", "simulations", a.save_to)
        if os.path.exists(os.path.join(sim_dir, "results.json")):
            report_for_dir(sim_dir, domain=a.domain)
        else:
            print(f"[t2_run] compliance hook: {sim_dir}/results.json 없음 — skip")
    except Exception as e:
        print(f"[t2_run] compliance hook failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
