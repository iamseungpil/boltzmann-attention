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
    ap.add_argument("--resolve_spec", default=None,
                    help="A2 grounding-spec 경로. 미지정 시 a2/<domain>.grounding.json (런처가 파일 선택·"
                         "코드 분기 아님). 도메인별 grounding은 *오직 이 spec 차이*.")
    ap.add_argument("--rules_prompt", default=None,
                    help="도메인-일반 rules-prompt 파일(닫힌 기저 명시) 주입 = prompt-vs-SFT arm B. "
                         "비지정 시 미주입(floor/SFT arm).")
    ap.add_argument("--domain", default="retail")
    ap.add_argument("--retrieval_config", default=None,
                    help="knowledge-domain retrieval variant (banking_knowledge). ★미지정 시 banking은 "
                         "'openai_embeddings'(dense KB·전 도구 작동·샌드박스 불요)로 고정 — RunConfig의 "
                         "자동 디폴트 'alltools'는 shell 도구를 노출하나 sandbox binaries(srt/rg/socat) "
                         "부재 시 고장난 도구가 스키마에 실림. (2026-07-10: 구 registry-override는 "
                         "RunConfig.retrieval_config가 env_kwargs로 partial 키워드를 덮어써 무효였음)")
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
    ap.add_argument("--task_ids", default=None,
                    help="comma-separated task ids; run ONLY these (for targeted re-runs)")
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

    if a.domain == "banking_knowledge":
        # ★변종은 RunConfig.retrieval_config(지원 경로)로만 지정한다.
        #   (구 방식 registry._domains partial-override는 죽은 코드였음: RunConfig가
        #    banking일 때 retrieval_config를 'alltools'로 자동 디폴트 → env_kwargs로
        #    retrieval_variant를 명시 전달 → partial 키워드를 덮어씀. 2026-07-10 발견 —
        #    "no_knowledge로 돌았다"는 종전 출력은 거짓이었고 실제는 alltools.)
        if a.retrieval_config is None:
            a.retrieval_config = "openai_embeddings"  # dense KB·bm25/shell 없음 = 전 도구 작동
        # sandbox 의존성 체크 stub: 변종 무관 build_tools가 체크 호출 — 실사용 안 함(shell 미노출 변종).
        import tau2.knowledge.sandbox_manager as _sbm
        _sbm._check_sandbox_dependencies = lambda *a, **k: None
        print(f"[t2_run] banking_knowledge retrieval_config={a.retrieval_config} + sandbox-check stubbed")

    if a.resolve:
        import t2_resolve_patch
        spec = a.resolve_spec or os.path.join(
            os.path.dirname(os.path.abspath(t2_resolve_patch.__file__)),
            "a2", f"{a.domain}.grounding.json")
        t2_resolve_patch.apply(spec)
        print(f"[t2_run] resolve_selection wiring ON · spec={spec}")

    if a.rules_prompt:
        import t2_agent_rules_patch
        t2_agent_rules_patch.apply(a.rules_prompt)
        print(f"[t2_run] rules-prompt 주입 ON · {a.rules_prompt}")

    if os.environ.get("T2_MAXPROMPT"):  # 최대-강도·위치반복 프롬프트(프롬프트 한계 실험)
        import t2_agent_maxprompt_patch
        t2_agent_maxprompt_patch.apply()

    if a.gate:
        import t2_gate_patch
        regen_on = os.environ.get("T2_PROV_REGEN") == "1"
        badwords_on = os.environ.get("T2_PROV_BADWORDS", "0") == "1"
        ground_on = os.environ.get("T2_PROV_GROUND", "0") == "1"
        disamb_on = os.environ.get("T2_DISAMB", "0") == "1"
        _unified = os.environ.get("T2_GATE_REGEN") == "1" and (
            regen_on or badwords_on or ground_on or disamb_on)
        if os.environ.get("T2_GATE_REGEN") == "1" and not _unified:
            # ★replay-safe 게이트: 생성-레벨 deny+regen+R8 종단 (apply() 대체·리더보드-동일 채점).
            t2_gate_patch.apply_gate_regen(max_regen=int(os.environ.get("T2_GATE_REGEN_K", "1")))
            print("[t2_run] gate ON (REPLAY-SAFE regen·K=%s)" % os.environ.get("T2_GATE_REGEN_K", "1"))
        elif not _unified:
            t2_gate_patch.apply()
            print("[t2_run] gate ON")
        # ★E-COMP unified (2026-07-10·리뷰 반영): T2_GATE_REGEN ∧ T2_PROV_REGEN/T2_DISAMB 동시
        #   활성 시 단일 통합 패치(apply_unified_regen)로 라우팅 — 구 이중패치 CONFLICT 해소.
        #   예산 semantics = 두 GO arm 승계(게이트 1라운드 tick·prov 무과금 K=4). GROUND는 scope 밖.
        if os.environ.get("T2_GATE_REGEN") == "1" and (regen_on or badwords_on or ground_on or disamb_on):
            if ground_on:
                raise SystemExit("[t2_run] T2_PROV_GROUND is not supported in unified mode (E-COMP scope).")
            t2_gate_patch.apply_unified_regen(
                max_prov_retries=int(os.environ.get("T2_PROV_REGEN_K", "4")),
                domain=a.domain,
                disamb=disamb_on,
                use_badwords=badwords_on)
            print("[t2_run] UNIFIED regen ON: gate(K=1·tick) + prov(K=%s·무과금)%s%s"
                  % (os.environ.get("T2_PROV_REGEN_K", "4"),
                     " + DISAMB" if disamb_on else "",
                     " + badwords" if badwords_on else ""))
        elif regen_on or badwords_on or ground_on or disamb_on:
            # GROUND/DISAMB는 regen 인프라(생성-레벨 작업본) 위에서 동작 → regen 경로 활성 필요
            t2_gate_patch.apply_provenance_regen(
                max_retries=int(os.environ.get("T2_PROV_REGEN_K", "4")) if (regen_on or ground_on) else 0,
                use_badwords=badwords_on,
                ground=ground_on,
                domain=a.domain,
                disamb=disamb_on)
            print("[t2_run] provenance L1(badwords)=%s L2(regen)=%s GROUND=%s DISAMB=%s"
                  % (badwords_on, regen_on, ground_on, disamb_on))

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
    _extra_cfg = {}
    if a.retrieval_config:
        _extra_cfg["retrieval_config"] = a.retrieval_config
    cfg = TextRunConfig(
        **_extra_cfg,
        domain=a.domain,
        agent="llm_agent",
        llm_agent=llm_agent,
        llm_args_agent=llm_args_agent,
        llm_user=user_llm,
        llm_args_user=user_args,
        num_trials=a.num_trials,
        num_tasks=a.num_tasks,
        task_ids=([t.strip() for t in a.task_ids.split(",")] if a.task_ids else None),
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
