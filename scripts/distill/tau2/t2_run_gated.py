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
    # ★중단·재개 + arm 정렬 (2026-07-18·사용자 지시·`ARM_ALIGNMENT_RESUME_DESIGN_2026_07_18`)
    ap.add_argument("--auto_resume", action="store_true",
                    help="기존 save_to서 재개(tau2 `try_resume`: 완료된 (trial,task_id,seed)는 건너뜀). "
                         "중간에 죽여도 재실행하면 이어서 — 죽인 만큼만 다시 돈다([[09]] 비용).")
    # ★재시도 금지 = 삭제 편향 제거 (2026-07-18 사용자 지시·`ARM_ALIGNMENT_RESUME_DESIGN §5`)
    #   기본 3회(총 4시도)는 예외 시 **sim을 통째로 재실행하고 실패 궤적을 버린다**(`run_with_retry`).
    #   컨텍스트 초과는 **대화가 길다=배회=실패할 궤적**서 나므로, 재시도는 **나쁜 표본만 골라 다시 뽑는다**
    #   = reward 상향 편향(실측: dreq2 20건·ctl2 14건 초과 / 재시도 11 vs 9).
    #   ⇒ **0 = 재시도 없음 → 초과가 `INFRASTRUCTURE_ERROR`로 *남는다*(삭제 대신 가시화).**
    #   ⚠️그 sim은 `messages=[]`·`reward_info` 없음 ⇒ **분석에서 reward 0(fail)로 세야** 지시대로 된다
    #   (`bank_paired_arms.py --infra-as-zero`). 안 그러면 편향이 '결측'으로 형태만 바뀐다.
    #   ⚠️궤적이 안 남아 **포렌식 불가**가 된다 — 점수 정직성과 맞바꾸는 값.
    ap.add_argument("--max_retries", type=int, default=None,
                    help="tau2 기본 3(총 4시도). **0 = 재시도 없음**(초과=fail·삭제편향 제거).")
    ap.add_argument("--max_steps", type=int, default=None,
                    help="tau2 기본 200(텍스트 모드의 실질 한계). 명시=기록용.")
    ap.add_argument("--seed", type=int, default=None,
                    help="배치 seed(tau2 기본 300). ★**양 arm에 같은 값**을 명시해야 "
                         "`done_runs` 키 (trial,task_id,seed)가 일치해 **페어 비교**가 성립한다. "
                         "기본(미지정)도 300으로 같지만, 명시 = 기록 + 드리프트 방지.")
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

    # ★GUIDED DECODING (T2_GUIDED=1) — 반드시 gate **이전** 적용(C166):
    #   gate의 regen이 apply 시점의 la.generate를 _og_gen으로 캡처해 직접 호출하므로,
    #   guided가 나중이면 regen 경로가 문법을 우회한다(032 관통 사고). 여기서 먼저 감싸
    #   gate가 guided-포함 체인을 캡처하게 한다. 문법=라이브 스키마 유래([[05]] 리터럴 0).
    if os.environ.get("T2_GUIDED") == "1":
        import t2_guided_patch
        t2_guided_patch.apply()
        print("[t2_run] GUIDED ON (tool-name grammar from live schema · auto 유지 · pre-gate)")

    if a.gate:
        import t2_gate_patch
        regen_on = os.environ.get("T2_PROV_REGEN") == "1"
        badwords_on = os.environ.get("T2_PROV_BADWORDS", "0") == "1"
        ground_on = os.environ.get("T2_PROV_GROUND", "0") == "1"
        disamb_on = os.environ.get("T2_DISAMB", "0") == "1"
        # ★T5-C silent repair (opt-in·기본값=v1 동작): T5C_SILENT_REPAIR_DESIGN_2026_07_11 §6.4
        ground2_on = os.environ.get("T2_GROUND", "0") == "1"          # P-A (양 분기)
        disamb_mode = os.environ.get("T2_DISAMB_MODE", "dialog")      # P-B: dialog|subcall
        prov_mode = os.environ.get("T2_PROV_MODE", "full")            # P-C: full|rescue
        _unified = os.environ.get("T2_GATE_REGEN") == "1" and (
            regen_on or badwords_on or ground_on or ground2_on or disamb_on)
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
        if _unified:
            if ground_on:
                raise SystemExit("[t2_run] T2_PROV_GROUND is not supported in unified mode (E-COMP scope). Use T2_GROUND=1.")
            t2_gate_patch.apply_unified_regen(
                max_prov_retries=int(os.environ.get("T2_PROV_REGEN_K", "4")),
                domain=a.domain,
                disamb=disamb_on,
                use_badwords=badwords_on,
                ground=ground2_on,
                disamb_mode=disamb_mode,
                prov_mode=prov_mode)
            print("[t2_run] UNIFIED regen ON: gate(K=1·tick) + prov(K=%s·무과금·mode=%s)%s%s%s"
                  % (os.environ.get("T2_PROV_REGEN_K", "4"), prov_mode,
                     " + DISAMB(%s)" % disamb_mode if disamb_on else "",
                     " + GROUND" if ground2_on else "",
                     " + badwords" if badwords_on else ""))
        elif regen_on or badwords_on or ground_on or ground2_on or disamb_on:
            # GROUND/DISAMB는 regen 인프라(생성-레벨 작업본) 위에서 동작 → regen 경로 활성 필요
            t2_gate_patch.apply_provenance_regen(
                max_retries=int(os.environ.get("T2_PROV_REGEN_K", "4")) if (regen_on or ground_on or ground2_on) else 0,
                use_badwords=badwords_on,
                ground=(ground_on or ground2_on),
                domain=a.domain,
                disamb=disamb_on,
                disamb_mode=disamb_mode,
                prov_mode=prov_mode)
            print("[t2_run] provenance L1(badwords)=%s L2(regen)=%s mode=%s GROUND=%s DISAMB=%s(%s)"
                  % (badwords_on, regen_on, prov_mode, (ground_on or ground2_on), disamb_on, disamb_mode))

    # ★E-PLAN (T2_EPLAN=1): gate와 독립 적용 — CP5 walk = orchestrator wrap(gate 불요),
    #   L1/L2 discovery deny = unified() 감지(--gate 1 필요). gate0+T2_EPLAN = 순수 CP5 격리 arm.
    if os.environ.get("T2_EPLAN") == "1":
        import t2_eplan_patch
        t2_eplan_patch.apply()
        print("[t2_run] E-PLAN ON (CP5 walk=%s · L1/L2 deny는 --gate 1 필요)"
              % ("ON" if os.environ.get("T2_EPLAN_WALK") == "1" else "off"))

    # ★SCAFFOLD-GET (T2_SCAFFOLD_GET=1): A2 scaffold_get_tools = 우리가 제공하는 GET 도구.
    #   LLM이 계산 직접 안 하고 이 도구 호출→scaffold가 t2_compute.apply_op로 결정론 계산·반환.
    #   gate/unified 뒤에 apply(체이닝·_execute_tool_calls 래핑).
    if os.environ.get("T2_SCAFFOLD_GET") == "1":
        import t2_scaffold_get
        t2_scaffold_get.apply()
        print("[t2_run] SCAFFOLD-GET ON (A2 scaffold_get_tools)")

    # (T2_GUIDED는 gate 이전 블록에서 이미 적용됨 — C166 순서 교정. 문법은 per-call tools
    #  인자에서 생성되므로 scaffold_get이 주입한 도구도 런타임에 자동 포함된다.)

    # ★PRE-ACTION-KB (T2_PREKB=1): 종결성 도구 실행 직전, '그 행동'으로 KB를 조회했는지 확인.
    #   미조회면 1회 deny + 행동-키 검색 지시(C165: 문제-기반 쿼리는 절차 문서를 못 찾고
    #   행동-기반 쿼리는 1~2위 — 032/033/035 기전). 문서를 찾게만 하고 답은 안 줌([[03b]]).
    #   scaffold_get/gate 뒤 적용(_execute_tool_calls 최외곽 체이닝).
    if os.environ.get("T2_PREKB") == "1":
        import t2_prekb_patch
        t2_prekb_patch.apply()
        print("[t2_run] PRE-ACTION-KB ON (action-keyed retrieval check · 1회/fam)")

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
    # ★C205(2026-07-27·[S] 서버로그 확정): max_tokens 미설정 = vLLM 기본 "EOS ∨ 컨텍스트 한계까지 생성"
    #   → 캡은 **정당 최장 응답(77-행 거래 JSON 에코 ~8k tok)이 안 잘리는 8192 권장** — 그래도 폭주 상한 8192/10.7≈13분 < timeout이라 재시도 폭풍 소멸. 폭주(반복-루프) 응답 1건이 10.7tok/s로 20분+ 단독 디코드(001 실측: prompt 0.0/gen 10.7/
    #   Running 1/Waiting 0 연속)→클라 타임아웃→전체 재시도. **생성 상한 = 단독-초과 클래스의 근본 캡**
    #   (정상 응답 수백 tok에는 무영향·폭주 응답은 어차피 쓰레기). 에이전트(vLLM)에만 적용 —
    #   user-sim(원격 추론 모델)은 reasoning 토큰이 있어 캡 금지. opt-in env·미설정=거동 불변.
    if os.environ.get("T2_AGENT_MAX_TOKENS"):
        llm_args_agent["max_tokens"] = int(os.environ["T2_AGENT_MAX_TOKENS"])
    # ★LLM 요청 timeout/재시도 (2026-07-20·097 stall 진단): completion()에 timeout이 없어 hang 요청이
    #   litellm 기본(~600s)×num_retries(config 3)=~40분 조용한 stall(097 실측·conc=1 블록). **opt-in env**로만
    #   주입(미설정=공유 드라이버 기본거동 불변). T2_LLM_TIMEOUT=초(요청당 상한)·T2_LLM_RETRIES=재시도수.
    #   agent(vLLM)·user-sim·judge 셋 다 적용(어느 호출이 hang하든 bound). generate()는 kwargs로 completion에 통과.
    _llm_to = os.environ.get("T2_LLM_TIMEOUT")
    _llm_rt = os.environ.get("T2_LLM_RETRIES")
    if _llm_to or _llm_rt:
        for _ar in (user_args, judge_args, llm_args_agent):
            if _llm_to:
                _ar["timeout"] = float(_llm_to)
            if _llm_rt is not None:
                _ar["num_retries"] = int(_llm_rt)
        print("[t2_run] LLM timeout=%s num_retries=%s (hang 방지·097 stall 진단)"
              % (_llm_to, _llm_rt))
    _extra_cfg = {}
    if a.retrieval_config:
        _extra_cfg["retrieval_config"] = a.retrieval_config
    if a.auto_resume:
        _extra_cfg["auto_resume"] = True
    if a.seed is not None:
        _extra_cfg["seed"] = a.seed
    if a.max_retries is not None:
        _extra_cfg["max_retries"] = a.max_retries
    if a.max_steps is not None:
        _extra_cfg["max_steps"] = a.max_steps
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
