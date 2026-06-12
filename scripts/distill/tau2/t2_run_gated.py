#!/usr/bin/env python
"""tau2 retail ±게이트 실행 러너 (BENCH_PORTFOLIO §3.6 ③ — 7B base census, R7).

--gate 1 이면 t2_gate_patch 적용 후 run_domain. 에이전트/유저-sim = 로컬 vllm
OpenAI-호환 엔드포인트 (litellm openai/<served-name> + api_base).

Run: cd /home/woori/scratch/tau2-bench && PYTHONPATH=src:$REPO/scripts/distill/tau2 \
  /home/woori/venvs/seka_env/bin/python $REPO/scripts/distill/tau2/t2_run_gated.py \
  --gate 1 --num_trials 4 --save_to retail_7b_gate
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", type=int, default=0)
    ap.add_argument("--domain", default="retail")
    ap.add_argument("--agent_model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--agent_base", default="http://localhost:8351/v1")
    ap.add_argument("--user_model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    ap.add_argument("--user_base", default="http://localhost:8352/v1")
    ap.add_argument("--num_trials", type=int, default=1)
    ap.add_argument("--num_tasks", type=int, default=None)
    ap.add_argument("--max_concurrency", type=int, default=8)
    ap.add_argument("--save_to", required=True)
    a = ap.parse_args()

    if a.gate:
        import t2_gate_patch
        t2_gate_patch.apply()
        print("[t2_run] gate ON")

    # NL-assertion judge가 gpt-4.1 하드 기본값(config.py) → 로컬 user-sim 모델로 재바인딩
    # (40/114 태스크가 nl_assertions 보유 — 미패치 시 키 부재로 영구 실패, 2026-06-12 사고)
    import tau2.evaluator.evaluator_nl_assertions as _nle
    _nle.DEFAULT_LLM_NL_ASSERTIONS = f"openai/{a.user_model}"
    _nle.DEFAULT_LLM_NL_ASSERTIONS_ARGS = {
        "temperature": 0.0, "api_base": a.user_base, "api_key": "dummy"}
    print(f"[t2_run] nl-assertion judge -> local {a.user_model}")

    from tau2.data_model.simulation import TextRunConfig
    from tau2.run import run_domain

    cfg = TextRunConfig(
        domain=a.domain,
        agent="llm_agent",
        llm_agent=f"openai/{a.agent_model}",
        llm_args_agent={"api_base": a.agent_base, "api_key": "dummy", "temperature": 0.0},
        llm_user=f"openai/{a.user_model}",
        llm_args_user={"api_base": a.user_base, "api_key": "dummy", "temperature": 0.7},
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


if __name__ == "__main__":
    main()
