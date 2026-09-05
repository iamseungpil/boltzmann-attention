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
import sys


def _install_failed_persist(env_cls):
    """★P10(C208②·DAY5_PRESCRIPTIONS §P10): 실패-sim 궤적 사이드카 영속(모듈-레벨=테스트 가능).
    tau2 러너는 4회 재시도 소진 시 메시지를 하나도 영속하지 않아(day5 024/010 messages 0) infra
    원인 규명 불가. replay-검증 ValueError의 발생지 `Environment.set_state`가 전체 궤적을 인자로
    받으므로 예외 시 best-effort 덤프 후 재-raise(러너 거동 무변·tau2 코어 비수정)."""
    import gzip as _gz
    import json as _json
    import time as _time
    _orig_set_state = env_cls.set_state
    if getattr(_orig_set_state, "_t2_fp_wrapped", False):
        return

    def _set_state_persist(self, *aa, **kk):
        try:
            return _orig_set_state(self, *aa, **kk)
        except Exception as _spe:
            try:
                _mh = kk.get("message_history")
                if _mh is None and len(aa) >= 3:
                    _mh = aa[2]
                _dir = os.environ.get("T2_FAILED_DIR", "failed_sims")
                os.makedirs(_dir, exist_ok=True)
                _fp = os.path.join(_dir,
                                   "failed_setstate_%d.json.gz" % int(_time.time() * 1000))
                _ser = []
                for _m in (_mh or []):
                    try:
                        _ser.append(_json.loads(_m.model_dump_json()))
                    except Exception:
                        _ser.append({"repr": repr(_m)[:2000]})
                with _gz.open(_fp, "wt", encoding="utf-8") as _f:
                    _json.dump({"error": str(_spe)[:4000], "n_messages": len(_ser),
                                "messages": _ser}, _f, ensure_ascii=False)
                print("[T2_FAILED_PERSIST] set_state failed -> trajectory persisted: %s"
                      % _fp, flush=True)
            except Exception as _ppe:
                print("[T2_FAILED_PERSIST] persist failed (no-op): %r" % (_ppe,), flush=True)
            raise

    _set_state_persist._t2_fp_wrapped = True
    env_cls.set_state = _set_state_persist


def main():
    # ★stderr 줄마다 sim 태그 (2026-08-08·C325). 러너는 sim을 스레드로 **동시에** 돌리므로
    #   로그가 인터리브되는데, 지금까지 sim을 다는 줄은 `[T2_LEVER]` 하나뿐이었고 그 태그마저
    #   전역 변수라 경합했다(실측: beat가 반대쪽 sim 이름을 달았고 그것을 근거로 원장까지 갔다).
    #   무기명 줄은 사후에 귀속을 복원할 방법이 **원리적으로 없다** ⇒ 드라이버가 기억해서 켜는
    #   방식이 아니라 **모든 라이브 런이 통과하는 이 한 자리**에 둔다(사이드카 선례·[[07]]).
    #   관측 전용(프리픽스만·거동 0)이고, 프리픽스라 행말 앵커 파서(`x134`)도 안 깨진다.
    try:
        import t2_lever_beat as _LB
        if _LB.install_stderr_tagger():
            print("[t2_run] stderr sim-tagger on (per-line attribution)")
    except Exception as _te:
        print("[t2_run] stderr sim-tagger skipped (no-op): %r" % (_te,))

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
    ap.add_argument("--user_reasoning_effort", default=None,
                    help="user-sim reasoning_effort (리더보드 정합: GPT-5.5 제출이 gpt-5.2 user-sim을 "
                         "reasoning_effort='low'로 돌렸다 — 비교 런은 반드시 맞출 것. "
                         "미지정=전달 안 함(기존 거동 보존).")
    ap.add_argument("--agent_seed", type=int, default=None,
                    help="agent 요청 seed 고정 (결정론 실험; 로컬 vllm arm 전용)")
    ap.add_argument("--user_seed", type=int, default=None,
                    help="user-sim 요청 seed 고정 (E-MFIX Y1a: 원격 provider가 seed를 "
                         "존중하는지 마이크로-확인용·agent_seed와 동일 계열 인프라 인자)")
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
    # 2026-09-03: 재시도 **간격**도 넘긴다. tau2 기본은 `retry_delay=1.0` 고정(지수 백오프 없음)
    #   이라 `max_retries=3` 과 합쳐 **총 4시도가 3초 안에** 끝난다. 실물: 09-03 14:22 DNS 순단에
    #   user-sim(openrouter)이 걸려 task_092/095 가 **메시지 0 · infrastructure_error** 로 죽었다.
    #   실험 레버가 아니다 — 재시도는 sim 을 처음부터 다시 돌리므로 모델 입력을 바꾸지 않는다.
    ap.add_argument("--retry_delay", type=float, default=None,
                    help="tau2 기본 1.0초(고정). 순단이 잦은 환경에서는 올린다.")
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

    # ★P10(C208②·DAY5_PRESCRIPTIONS §P10·T2_FAILED_PERSIST=1): 실패-sim 궤적 사이드카 영속.
    #   tau2 러너는 4회 재시도 소진 시 메시지를 하나도 영속하지 않아(day5 024/010 messages 0)
    #   infra 원인 규명이 불가능했다. replay-검증 ValueError의 발생지 `Environment.set_state`가
    #   전체 궤적을 인자로 받으므로, 거기서 예외 시 best-effort 덤프 후 재-raise(러너 거동 무변).
    if os.environ.get("T2_FAILED_PERSIST") == "1":
        from tau2.environment.environment import Environment as _T2Env
        _install_failed_persist(_T2Env)
        print("[t2_run] FAILED-PERSIST ON (set_state failure -> sidecar trajectory)")

    # user-sim·judge 모델 결정: --user_llm(원격 API, 예 openrouter/...) 우선, 아니면 로컬 vllm
    if a.user_llm:
        user_llm, user_args = a.user_llm, {"temperature": a.user_temp}
        # ★리더보드 정합(2026-08-02·[[54]]): GPT-5.5 제출 = user-sim gpt-5.2 `reasoning_effort: low`.
        #   맞추지 않으면 user-sim 난이도가 달라져 비교가 깨진다. 미지정 시 전달 안 함(기존 거동 보존).
        if a.user_reasoning_effort:
            user_args["reasoning_effort"] = a.user_reasoning_effort
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
    # E-MFIX Y1a: user-sim seed 고정(원격 provider의 seed 존중 여부 확인용·미설정=거동 불변)
    if a.user_seed is not None:
        user_args["seed"] = a.user_seed
        print(f"[t2_run] user_seed={a.user_seed} (E-MFIX Y1a)")
    # ★C205(2026-07-27·[S] 서버로그 확정): max_tokens 미설정 = vLLM 기본 "EOS ∨ 컨텍스트 한계까지 생성"
    #   → 캡은 **정당 최장 응답(77-행 거래 JSON 에코 ~8k tok)이 안 잘리는 8192 권장** — 그래도 폭주 상한 8192/10.7≈13분 < timeout이라 재시도 폭풍 소멸. 폭주(반복-루프) 응답 1건이 10.7tok/s로 20분+ 단독 디코드(001 실측: prompt 0.0/gen 10.7/
    #   Running 1/Waiting 0 연속)→클라 타임아웃→전체 재시도. **생성 상한 = 단독-초과 클래스의 근본 캡**
    #   (정상 응답 수백 tok에는 무영향·폭주 응답은 어차피 쓰레기). 에이전트(vLLM)에만 적용 —
    #   user-sim(원격 추론 모델)은 reasoning 토큰이 있어 캡 금지. opt-in env·미설정=거동 불변.
    if os.environ.get("T2_AGENT_MAX_TOKENS"):
        llm_args_agent["max_tokens"] = int(os.environ["T2_AGENT_MAX_TOKENS"])
    # ★T2_GEN_TRACE (2026-08-30·§L-5 수리) — 생성 깔때기의 **우회 불가 계기**.
    #   `T2_PROMPT_DUMP` 는 `t2_gate_patch._gen` 두 곳(6547·7990)에 달려 있는데 x659 에서 **레코드 0**
    #   이었다 — 라이브(`apply_unified_regen`->`unified`)가 그 경로를 안 탄다. 자국 없는 계기는
    #   계기가 아니다. 여기서 tau2 의 `generate` **모듈 전역**을 감싸면 어느 호출 형태든 지난다.
    #   재는 것: 요청의 max_tokens/tool_choice · 응답의 content 길이 · tool_calls 수.
    #   ⛔판단 0 · 거동 변경 0(원함수를 그대로 부르고 로그만 남긴다) · opt-in.
    if any(os.environ.get(_k) == "1" for _k in
           ("T2_GEN_TRACE", "T2_NO_FORCE_TOOLCHOICE", "T2_PROBE_TERSE", "T2_TC_SALVAGE", "T2_STOP_FIRST_TOOLCALL", "T2_P2_REGEN", "T2_TOOL_OBS"))             or os.environ.get("T2_FAILDUMP"):
        import sys as _sys_tr
        import tau2.agent.llm_agent as _la_tr
        _orig_gen_t2 = _la_tr.generate

        _t2_notc = os.environ.get("T2_NO_FORCE_TOOLCHOICE") == "1"
        # ★T2_PROBE_TERSE (2026-08-30·§L-11·사용자 지적 "1줄만 뱉게 해야 하지 않나")
        #   선언-프로브 서브콜(`agent_writeprov` 등)은 **한 줄 JSON** 한 개를 기대한다. 그런데
        #   Q3.8 은 reasoning 모델이라 8192 토큰(=32KB)을 꽉 채우고 그 안에 JSON 이 없다 →
        #   `_claims=None` → 게이트 무발화 → **5.8분을 태우고 아무것도 못 얻는다**.
        #   ⇒ ⑴ 그 호출에만 `enable_thinking=False`(vLLM chat_template_kwargs) ⑵ max_tokens 축소.
        #   질문 자체가 예/아니오이므로 사고를 끄는 것이 **정확도 손실이 아니다**.
        #   call_name 은 엔진 내부 이름이라 도메인 리터럴이 아니다([[05]] 안전).
        _t2_probe_terse = os.environ.get("T2_PROBE_TERSE") == "1"
        # ⒜ **사실확인 프로브** — 자기 발화를 읽고 예/아니오. 추론 불필요 → 사고 OFF·짧게.
        _t2_probe_calls = {c.strip() for c in (os.environ.get("T2_PROBE_CALLS") or
                           "agent_writeprov,agent_claimprov,agent_selfdecl").split(",") if c.strip()}
        # ★§S-6 (2026-09-01): 이 상한은 **모델에 매인 값**이라 코드가 아니라 프로필이 갖는다
        #   (`model_profiles/*.env` · 런처는 프로필 없으면 발사를 거부한다). 사고를 쓰는 모델에서
        #   256 은 사고 예산(`max(256, cap//2)`)과 같아 **답 자리가 0** 이었고, 밤샘런 TRUNC
        #   **85건 전량**이 `call=agent_claimprov max_tokens=256` 이었다. Q3.8 은 512(=예산 256의 2배).
        _t2_probe_mt = int(os.environ.get("T2_PROBE_MAX_TOKENS", "256") or 0)
        # ⒝ **판단 프로브** — 도구/값을 고른다. 사고를 끄면 답이 바뀐다(실측: 사고ON 'none' ↔
        #    사고OFF 'get_current_time'). → **사고는 두고 형식만** guided JSON 으로 보장한다.
        #    실측(8141): guided+사고ON 은 1,247 토큰에서 답이 나오고 파싱 OK. 상한 800 은 부족했고
        #    guided 없이 같은 사고를 하면 `tool_name='none'`(따옴표 형식 붕괴)로 **파싱 실패**한다.
        #    ⛔스키마는 **형태만** 잡는다 — 이름 유효성은 호출부가 이미 검사한다
        #      (`t2_resolve.py` `cand in action_tools`). 도메인 리터럴 0([[05]] 안전).
        # 실측(8141): guided+사고ON 이 **1,247 생성토큰**에 답을 냈다(content 38B).
        #   ⇒ 사고 여지를 남기되 **최소로** 조인다(사용자 지시 2026-08-31:
        #   "thinking 을 사용한 경우라도 guided JSON 으로 결과만 정확하게 최소 max 로 요청하라").
        #   1536 = 1,247 + 23% 여유. 절단은 계기의 TRUNC 표시로 감시한다.
        _t2_judge_mt = int(os.environ.get("T2_JUDGE_MAX_TOKENS", "4096") or 4096)
        #   ★2048 -> 4096 (2026-08-31): 라이브에서 `intent_operator_formalize` 가 2048 에서
        #   **TRUNC 2회**. 절단은 `finish=length · content 0B` = **답 전손**이라 상한은 넉넉해야
        #   한다. 절감은 상한이 아니라 **프롬프트의 "200자 이내" 지시**가 낸다(실측 33~66%).
        #   격리 최대 1,625 였으나 라이브 문맥이 더 길어 넘겼다 — 격리치를 상한으로 쓰지 마라.
        # ★§T-12: TERSE 경로 프로브의 출력 스키마 — 소비부가 실제로 읽는 모양 그대로다
        #   (`t2_gate_patch`: `_j2["claims"]` 리스트 · 항목의 `tool` · `_j2["pending"]` 리스트).
        _t2_terse_schemas = {
            "agent_claimprov": {
                "type": "object",
                "properties": {
                    # ★D8 수리 (2026-09-05) — 키 이름이 세 곳에서 어긋나 있었다.
                    #   A2 질문(`gate.json` claim_prov.question)은 {"kind","what","tool"} 을 요구하고
                    #   소비부(`t2_gate_patch` :14986 _desc3 · :15015 · :15026 · :15033 · :5218 · :5235 ·
                    #   :5357)는 전부 `c.get("what")` 을 읽는데, **이 스키마만** `claim` 으로 묶고 있었다.
                    #   guided decoding 은 스키마 밖 키의 생성 자체를 막으므로 `what` 은 영원히 None —
                    #   실측 전송 문면 **73/73 이 `None: None`**, `unb_p>=1` 158/158, 날짜 절벽의 유일
                    #   변경이 `f6224e26` 의 개명이었다. `pending` 은 `kind` 조차 없어 그쪽도 None 이었다.
                    #   ⚠`t2_source.py:289` 의 `c.get("claim")` 은 **다른 프로브**(source_claim_formalize)라
                    #   같은 낱말이어도 계약이 다르다 — 함께 바꾸지 않는다.
                    "claims": {"type": "array", "items": {
                        "type": "object",
                        "properties": {"what": {"type": "string"},
                                       "tool": {"type": "string"},
                                       "kind": {"type": "string"}},
                        "required": ["what"]}},
                    "pending": {"type": "array", "items": {
                        "type": "object",
                        "properties": {"what": {"type": "string"},
                                       "tool": {"type": "string"},
                                       "kind": {"type": "string"}},
                        "required": ["what"]}},
                },
                "required": ["claims", "pending"],
            },
        }
        _t2_judge_schemas = {
            "intent_operator_formalize": {
                "type": "object", "properties": {"tool": {"type": "string"}},
                "required": ["tool"], "additionalProperties": False},
            # operand 키가 동적이라 `applies` 만 보장하고 나머지는 열어 둔다.
            "recommend_formalize": {
                "type": "object", "properties": {"applies": {"type": "boolean"}},
                "required": ["applies"], "additionalProperties": True},
            # ★2026-08-31 추가: 인벤토리에서 **미등록**으로 발견돼 전역 8192 를 받고 있었다
            #   (`t2_scaffold_get.py:2839`). 계약은 {"<동적 arg>": ..., "quote": "..."} 이므로
            #   `quote` 만 보장하고 나머지는 연다.
            "sg_arg_docs": {
                "type": "object", "properties": {"quote": {"type": "string"}},
                "required": ["quote"], "additionalProperties": True},
        }
        for _drop in (os.environ.get("T2_JUDGE_DISABLE") or "").split(","):
            _t2_judge_schemas.pop(_drop.strip(), None)

        import re as _re_tr
        import json as _json_tr
        _t2_salvage = os.environ.get("T2_TC_SALVAGE") == "1"
        _t2_salvage_gj = os.environ.get("T2_TC_SALVAGE_GUIDED") == "1"
        _t2_faildump = os.environ.get("T2_FAILDUMP") or None
        # ★T2_TOOL_OBS (2026-08-31·사용자 지시 "진행중 도구 응답 보게 수리하라")
        #   왜: 도구 응답은 궤적(results.json)에만 남고 **sim 이 끝나야** 기록된다. 사이드카에는
        #   우리 주입만 있고, stderr 배너는 호출 여부를 말하지 않는다 ⇒ 진행 중 판정이 불가능했다.
        #   여기(생성 깔때기)로 오는 `messages` 에는 **직전 도구 응답이 이미 들어 있다.**
        #   새 것만 한 번씩 찍으면 실시간 기록이 된다. 판단 0 · 거동 0(읽기만).
        _t2_toolobs = os.environ.get("T2_TOOL_OBS") == "1"
        _t2_toolobs_cap = int(os.environ.get("T2_TOOL_OBS_MAX", "600") or 600)
        _t2_seen_tool = set()
        _t2_p2 = os.environ.get("T2_P2_REGEN") == "1"
        _t2_p2_cap = int(os.environ.get("T2_P2_CAP", "3") or 3)
        _t2_p2_used = [0]
        # ★T2_STOP_FIRST_TOOLCALL (2026-08-30·§L-14) — **첫 tool_call 에서 생성을 끊는다.**
        #   faildump 실측으로 사슬 확정: 모델이 같은 호출을 **142회 반복** 생성하다 max_tokens=8192 를
        #   소진하고 **마지막 블록이 절단**되며, 그 절단 블록 때문에 파서가 통째로 버려 content 로 떨어진다.
        #   프롬프트로는 못 막는다 — 같은 문맥을 재생하면 219토큰에 정상 종료(**비결정적 폭주**).
        #   정지어는 그 폭주를 **구조적으로 불가능**하게 만든다.
        #   실측(8141): 정지어 없음 gen108/tool_calls=2 · **정지어 gen92/tool_calls=1 · 파싱정상**.
        #   ⛔[[70]] 부호표: **병렬 발사를 판다**(2→1 · [[80]] 이 기록한 Q3.8 의 강점).
        #     사는 것 = 절단 0 · 파싱 100% · 한 턴 8192토큰(5.8분) -> 92토큰.
        #   ⛔프로브에는 안 건다(그쪽은 도구를 안 부른다).
        _t2_stopfirst = os.environ.get("T2_STOP_FIRST_TOOLCALL") == "1"
        try:
            from tau2.data_model.message import ToolCall as _TC_tr
        except Exception:
            _TC_tr = None
        # ★표면형 파싱은 **정본 한 곳**에서만 한다 (2026-08-31·[[67]] 사본 금지).
        #   `t2_salvage.extract_calls/strip_calls` 가 hermes(JSON)와 qwen3_xml(XML) 둘 다 읽는다.
        #   구판은 여기 hermes 정규식이 인라인으로 있었고, 서버 파서가 qwen3_coder 로 바뀐 뒤
        #   **눈이 먼 채로** 두 달을 돌았다([[84]]).
        import t2_salvage as _SALV


        def _think_budget(_declared, _cap, _who):
            """사고 예산은 **상한보다 작아야** 답 자리가 남는다 — 선언 실수를 엔진이 막는다.

            ★2026-08-31 실물: 프로필에 프로브 예산을 상한과 **같은 4096** 으로 적었더니
              보호가 그대로 꺼졌다. x706 축자:
                `call=intent_operator_formalize max_tokens=4096 -> gen=4096 **TRUNC**
                 reason=18,535B content=0B`
              같은 런의 다른 프로브는 gen 2,171·3,965 로 **상한 바로 밑**을 오간다(x707 도 3,965)
              ⇒ 예산이 상한과 같으면 답 전손은 시간 문제다.
            ★격리 x705: 예산 < 상한이면 전손 0/2(mt 512·예산 256 → content 1,046B).
            ⚠하한 256 은 종전 파생과 같다. 선행 실측 *"486토큰에서 답이 바뀐다"* 가 있으므로
              이 자리는 **자동으로 조이지 않는다** — 선언값이 상한 미만이면 그대로 존중한다.
            """
            try:
                _cap = int(_cap or 0)
            except Exception:
                _cap = 0
            try:
                _b = int(_declared) if _declared else 0
            except Exception:
                _b = 0
            if not _b:
                _b = max(256, _cap // 2) if _cap else 4096
            elif _cap and _b >= _cap:
                _fixed = max(256, _cap // 2)
                print("[t2_run] ⛔사고 예산(%s) %d ≥ 상한 %d — 답 자리가 없다. %d 로 조인다"
                      % (_who, _b, _cap, _fixed), file=_sys_tr.stderr, flush=True)
                _b = _fixed
            return _b

        def _t2_msg_empty(_m):
            """tau2 **자신의 유효성 법**을 그대로 읽는다 (`data_model/message.py:311-318` ·
            `utils/llm_utils.py:234`): 본문도 도구호출도 없으면 그 메시지는 **존재할 수 없다**.

            ★왜 (2026-09-01·§S-2): 재생성 사다리의 트리거가 `여는태그 > 닫는태그` 뿐이라
              `content==''` 이면 `0>0=False` — **가장 비싼 실패(전손)에 사다리가 한 번도 안 걸렸다**.
              밤샘런 실측: 전손 5건 = `Retry` 5건 = 태스크 전체 재시작 · 폐기 벽시계 16,746초.
              같은 파일의 faildump 술어는 이미 `or not _c0.strip()` 으로 넓혀져 있었다 —
              계기에만 이식하고 수리에는 안 한 것이다([[81]]).
            ⚠`.strip()` 필수 — task_092 실물이 `'

        '` 이라 길이 기준은 놓친다.
            """
            return not (str(getattr(_m, "content", None) or "").strip()
                        or (getattr(_m, "tool_calls", None) or []))


        def _reasoning_of(_r):
            """응답의 reasoning 원문 — 타입 표면에 없으면 `raw_data` 에서 꺼낸다(읽기만)."""
            for _k in ("reasoning", "reasoning_content"):
                _v = getattr(_r, _k, None)
                if _v:
                    return str(_v)
            try:
                _rd = getattr(_r, "raw_data", None) or {}
                _m = ((_rd.get("choices") or [{}])[0] or {}).get("message") or {}
                return str(_m.get("reasoning") or _m.get("reasoning_content") or "")
            except Exception:
                return ""

        def _t2_salvage_calls(_r, _kw):
            """★T2_TC_SALVAGE (2026-08-30·§L-12) — vLLM 파서가 **통째로 버린** 유효 블록 되살리기.

            근거(우리 코드 주석 `t2_gate_patch.py:13355`): *"닫힌 tool_call 블록 7/7 JSON 유효 ·
            깨진 곳은 미종결 8번째뿐 => hermes 파서가 all-or-nothing 이라 유효 7개가 통째로 폐기"*.
            ⇒ 추가 LLM 호출 0 · **판단 0**(고르지 않는다·있는 것을 형식만 복구) · 도메인 리터럴 0.
            """
            if _TC_tr is None:
                return None
            _c = str(getattr(_r, "content", None) or "")
            if "<tool_call>" not in _c:
                return None
            _made = []
            for _i, (_nm, _ar) in enumerate(_SALV.extract_calls(_c)):
                try:
                    _made.append(_TC_tr(id="salv_%d" % _i, name=str(_nm), arguments=_ar))
                except Exception:
                    continue
            # ★첫 블록만 (2026-08-31 수리 · 7월 설계서 §2-1 축자를 내가 안 읽고 어겼다):
            #   *"본문에서 **첫 번째 완결 블록**만 파싱해 그 호출로 진행한다. **복제분은 버린다**
            #     (중복 실행 금지). ... 93개 복제를 전부 실행하면 **over-action 재앙**이다."*
            #   실측 피해: x675 에서 `SALVAGED=34/35/68` — 한 턴에 도구 68개를 내보냈다.
            #   복제는 **정지 실패의 산물이지 의도가 아니다**([[10]] 선택은 모델이 이미 했다 —
            #   그 선택은 **첫 블록 하나**다).
            #   `T2_SALVAGE_ALL=1` 이면 종전대로 전부(대조팔 전용).
            if _made and os.environ.get("T2_SALVAGE_ALL") != "1":
                _dropped = len(_made) - 1
                _made = _made[:1]
                if _dropped:
                    print("[T2_SALVAGE] first-block only: dropped %d duplicate block(s)"
                          % _dropped, file=_sys_tr.stderr, flush=True)
            if not _made:
                return None
            try:
                _r.tool_calls = _made
                _r.content = _SALV.strip_calls(_c).strip() or None
            except Exception:
                return None
            return _made

        def _t2_traced_generate(*_a, **_kw):
            # ★T2_NO_FORCE_TOOLCHOICE (2026-08-30·사용자 지시 "tool choice required 는 off 하라")
            #   `tool_choice="required"` 는 코드 **6곳**에 흩어져 있다(8547·12557·13207·13300·13390 등).
            #   여기 깔때기 한 곳에서 벗기면 어느 경로로 오든 걸린다([[62]] 최소 개입·우회 불가).
            #   근거: x667 계기 실측 — `agent_response` 가 tool_calls=0 을 반복하자 우리가 매번
            #   `agent_response_unified_regen tool_choice=required` 로 강제했고, 그렇게 짜낸 도구가
            #   x659 에서 KB_search_bm25 같은 **엉뚱한 것**이었다(필요한 log_verification 은 0회).
            #   ⛔거동 변경이다 — [[70]] 부호표 대상. 무엇을 사고 무엇을 파는지 런으로 재야 한다.
            # ★2026-08-31 — **열거 대신 규칙**([[58]] 일반 규칙만).
            #   호출 이름을 하나씩 등록하다 `sg_arg_docs` → `sg_docs_class` → `sg_fetch_iso` 로
            #   **세 번 놓쳤다**. 코드의 `sub_generate` call_name 은 수십 개이고 계속 는다.
            #   ⇒ 닫힌 규칙: **`agent_response*` 만 본 응답**이고 나머지 서브콜은 **전부 프로브**다.
            #     본 응답만 전역 상한(폭주 통제 대상)을 받고, 프로브는 프로브 상한을 받는다.
            #   실측 근거: `sg_docs_class` 가 전역 3072 를 꽉 채우고 `content=0B`(답 없음)로 잘렸다.
            _cn = str(_kw.get("call_name") or "")
            _is_probe = bool(_cn) and not _cn.startswith("agent_response")
            # ★사고 예산은 **매 호출에 따로 준다** (2026-08-31·사용자 지시).
            #   구판은 예산을 **프로브 가지에만** 걸었다(:513). 본 응답에는 없었고, 그래서
            #   상한이 사고 도중에 걸리면 생성 전량이 reasoning 으로 분류돼 `content=None` 이
            #   되고 tau2 가 태스크를 통째로 버렸다(x693 에서 1,590초 폐기).
            #   격리 x705(같은 서버·같은 프롬프트·n=2·전부 결정론):
            #     예산 없음 mt=512  → 전손 2/2 (reason 2,250B · content 0B)
            #     예산 없음 mt=2048 → 전손 2/2 (reason 8,340B · content 0B)
            #     예산 256  mt=512  → 전손 **0/2** (reason 1,131B · content 1,046B)
            #     예산 1024 mt=2048 → 전손 **0/2** · finish=stop(절단 자체가 사라진다)
            #     예산 1024 + 도구  → tool_calls 1 정상
            #   ⇒ 상한을 키우는 것이 아니라 **사고에 예산을 걸어 답 자리를 남긴다**.
            #   선언(`T2_THINK_BUDGET`) 없으면 상한의 절반으로 파생하고, 그 사실을 로그에 남긴다.
            #   ⚠[[70]] 무엇을 파나: 사고가 예산에서 끊긴다. 예산이 너무 작으면 답이 바뀐다
            #     (선행 실측: 486토큰에서 답이 바뀐다) — 그래서 모델 프로필에 값을 선언한다.
            if not _is_probe:
                _capm = (_kw.get("max_tokens")
                         or os.environ.get("T2_AGENT_MAX_TOKENS") or 8192)
                _tbm = _think_budget(os.environ.get("T2_THINK_BUDGET"), _capm, "본 응답")
                if _tbm and not _kw.get("thinking_token_budget"):
                    _kw = dict(_kw)
                    _kw["thinking_token_budget"] = _tbm
                    if not globals().get("_T2_TB_SAID"):
                        globals()["_T2_TB_SAID"] = True
                        print("[t2_run] 사고 예산: 본 응답 %d (상한 %s · %s)"
                              % (_tbm, _capm,
                                 "선언" if os.environ.get("T2_THINK_BUDGET") else "상한의 절반 파생"),
                              file=_sys_tr.stderr, flush=True)
            if _t2_probe_terse and _is_probe and _cn not in _t2_probe_calls:
                _kw = dict(_kw)
                _jmt2 = _t2_judge_mt
                _cur2 = _kw.get("max_tokens")
                # ★수리③ 동형(위 주석) — 프로브 상한은 자기 값이 정한다(전역 무관).
                _kw["max_tokens"] = _jmt2 if _jmt2 else _cur2
                # ★2026-08-31 — **사고 예산만 제한**(사용자 지시 "2로 가라").
                #   왜: 생성 순서가 [사고 …] → [답] 이라, 상한이 사고 도중에 걸리면 **답이 통째로
                #   사라진다**. 실측: TRUNC 89건 중 31건(35%)이 `content=0B · tool_calls=0` 이고,
                #   판단 프로브는 **전부** 그 유형이었다(`intent_operator_formalize gen=4096 content=0B`).
                #   ⇒ 상한을 올리는 대신 **사고에만 예산**을 걸어 답 자리를 반드시 남긴다.
                #   `thinking_token_budget` 은 vLLM 이 Qwen3 계열에 지원한다(실측: 최상위 인자로 유효).
                #   기본 = 전체 상한의 절반 → 답 자리 절반 확보.
                _tb = _think_budget(os.environ.get("T2_PROBE_THINK_BUDGET"),
                                    _kw.get("max_tokens") or _jmt2 or 4096, "프로브")
                if _tb:
                    _kw["thinking_token_budget"] = _tb
                _sch2 = _t2_judge_schemas.get(_cn)
                if _sch2:
                    _kw["response_format"] = {"type": "json_schema",
                                              "json_schema": {"name": "t2probe", "schema": _sch2}}
                _kw["_t2_terse"] = "PROBE"
            elif _t2_probe_terse and _kw.get("call_name") in _t2_judge_schemas:
                # ★답만 받는다 (2026-08-31·사용자 지시: "쓸데없는 중간과정은 모두 제외하고 답만
                #   받아라. 그래야 JSON 폭주를 멈출 수 있다. 답 요구 방식을 바꾸라.")
                #   기전: **응답 형태가 산문을 허용하면 모델이 그걸 채운다.** 스키마로 출력면을
                #   답 하나로 좁히고 사고를 끄면 **채울 여지 자체가 사라진다.**
                #   실측(8141·같은 프로브): guided+사고ON 1,247토큰/52.7s ↔
                #   **guided+사고OFF 22토큰/1.0s** — 둘 다 파싱 OK. 상한도 사실확인과 같은 값으로 둔다.
                #   ⛔[[70]] 부호표: **사고를 판다** — 도구 선택이 갈릴 수 있다
                #     (실측: 사고ON "none" ↔ 사고OFF "get_current_time").
                #     `T2_PROBE_KEEP_THINK=1` 이 그 대조팔이다(사고 유지 + 상한 _t2_judge_mt).
                _kw = dict(_kw)
                # ★2026-08-31 되돌림 (사용자 권고: "사고를 유지하되, 답 형식만 제한하는 걸 추천한다")
                #   실측이 그 권고를 지지한다 — **사고를 끄면 답이 틀린다**:
                #     사고ON  -> {"tool": "verify_identity"}   (무제약·budget·간결 세 팔 모두 일관)
                #     사고OFF -> {"tool": "none"}              **오답**
                #   그리고 사고는 **끊을 수도 없다**: `thinking_budget=64` 는 무시되고(1,579 vs 1,625토큰),
                #   `max_tokens=200` 으로 조이면 finish=length·content 0B 로 **답이 아예 안 나온다**.
                #   ⛔`reasoning_content` 는 전 팔 **0 B** — `--reasoning-parser qwen3` 가 붙어 있어도
                #     분리되지 않아 **파서로도 통제 불가**다.
                #   ⇒ 남는 통제면은 **출력 형식(guided JSON)** 하나뿐이고, 그것으로 충분하다.
                #   비용: 판단 프로브당 1,096~1,625 생성토큰(90~132s). 프롬프트에 "간결히"를 넣으면
                #     33% 준다(별건 · `t2_resolve.py` 프롬프트 수정 필요).
                _keep = os.environ.get("T2_PROBE_NOTHINK") != "1"   # 기본 = 사고 유지
                _jmt = _t2_judge_mt if _keep else _t2_probe_mt
                _cur = _kw.get("max_tokens")
                # ★2026-08-31 수리②: 종전 `min` 이라 **전역 상한이 판단 프로브를 깎았다**
                #   (전역 3072 vs 판단 4096 -> min=3072). 실측 x687: `intent_operator_formalize`
                #   13회에 **20,768토큰**(평균 1,597·최대 3072=상한 도달) — 상한에 닿으면
                #   `finish=length·content 0B` 로 **답 전손**이다.
                #   판단 프로브는 사고 여지가 **더** 필요하다. 전역은 `agent_response` 폭주용이고
                #   판단 프로브는 스키마로 출력이 묶여 있어 폭주 위험이 없다 ⇒ `max` 로 바꾼다.
                # ★2026-08-31 수리③ — `max` 는 **전역이 커지면 프로브가 부푼다**.
                #   ②의 병(전역 3072 이 프로브 4096 을 깎음)은 `max` 로 고쳤지만, 그날 저녁
                #   전손 수리로 전역이 8192 로 복원되자 이번엔 프로브 상한이 **8192** 가 되고
                #   사고 예산(상한의 절반)이 2048 → **4096** 으로 배가됐다. 라이브 실측:
                #     x697 `intent_operator_formalize` 6회 = **24,560토큰 = 런 생성의 75%**
                #     한 호출 지문: prompt 239 · gen 4,108 · reason **18,220B** · content **40B**
                #   ⇒ 프로브 상한은 **자기 상한(judge cap)** 이 정한다. 전역과 무관해야 양쪽
                #     사고(깎임·부풂)가 다 닫힌다. 전역이 더 작아도 프로브는 자기 값을 쓴다.
                _kw["max_tokens"] = _jmt if _jmt else _cur
                _kw["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "t2probe",
                                    "schema": _t2_judge_schemas[_kw["call_name"]]}}
                if not _keep:
                    _eb3 = dict(_kw.get("extra_body") or {})
                    _ctk3 = dict(_eb3.get("chat_template_kwargs") or {})
                    _ctk3["enable_thinking"] = False
                    _eb3["chat_template_kwargs"] = _ctk3
                    _kw["extra_body"] = _eb3
                _kw["_t2_terse"] = "JUDGE"
            elif _t2_probe_terse and _kw.get("call_name") in _t2_probe_calls:
                _kw = dict(_kw)
                # 사고를 유지하는 팔은 사고 토큰이 들어갈 자리가 있어야 하므로 상한을 넉넉히 둔다.
                _mt_cap = (int(os.environ.get("T2_PROBE_KEEP_THINK_MAX", "2048") or 2048)
                           if os.environ.get("T2_PROBE_KEEP_THINK") == "1" else _t2_probe_mt)
                if _mt_cap:
                    _cur = _kw.get("max_tokens")
                    _kw["max_tokens"] = min(int(_cur), _mt_cap) if _cur else _mt_cap
                # ★사고를 끌지 말지 (2026-08-30·사용자 지적 "thinking 은 하되 표시만 안 하게").
                #   ⚠"표시"는 이미 분리돼 있다 — 엔진에 `--reasoning-parser qwen3` 가 붙어
                #     reasoning 은 `reasoning_content` 로 온다. **비용은 표시가 아니라 생성**이다.
                #   실측(8141·같은 프로브): 사고ON 178토큰/10.2s · 사고OFF **7토큰/0.4s** ·
                #     사고ON+guided_json 177토큰/10.1s — **세 팔 답이 동일**(false).
                #   기본은 사고 OFF(이 프로브는 *자기 발화를 읽는 사실 확인*이라 추론이 필요 없다).
                #   `T2_PROBE_KEEP_THINK=1` 이면 사고를 유지하고 **길이만** 묶는다([[70]] 부호표용 대조팔).
                if os.environ.get("T2_PROBE_KEEP_THINK") != "1":
                    _eb = dict(_kw.get("extra_body") or {})
                    _ctk = dict(_eb.get("chat_template_kwargs") or {})
                    _ctk["enable_thinking"] = False
                    _eb["chat_template_kwargs"] = _ctk
                    _kw["extra_body"] = _eb
                # ★§T-12 (2026-09-01): TERSE 프로브에도 **출력 스키마**를 건다(경로는 그대로).
                #   왜: `agent_claimprov` 는 스키마가 없어 **산문 1,825B** 를 뱉고 상한(512)에 정확히
                #   닿아 잘렸다. 잘린 JSON 은 소비부(`t2_gate_patch` `re.search(r"{.*}")` → `json.loads`)
                #   에서 파스 실패 → `except` → `if not _cl and not _pd: break` ⇒ **날조-완료 차단
                #   게이트가 그 턴에 조용히 꺼진다**. 라이브 실측 1:1(TRUNC 1 ↔ no-op 1, 양 팔).
                #   ⚠JUDGE 로 옮기지 않는다 — 그러면 상한 8192·사고 4096 이 되어 콜 ~100회에
                #     비용이 폭증한다. 여기서는 **형식만** 묶는다(사고는 이 경로 기본대로 OFF).
                _tsch = _t2_terse_schemas.get(_kw.get("call_name"))
                if _tsch:
                    _kw["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {"name": "t2terse", "schema": _tsch}}
                _kw["_t2_terse"] = "TERSE"
            if (_t2_stopfirst and _kw.get("call_name") not in _t2_probe_calls
                    and _kw.get("call_name") not in _t2_judge_schemas):
                _kw = dict(_kw)
                _st = list(_kw.get("stop") or [])
                if "</tool_call>" not in _st:
                    _st.append("</tool_call>")
                _kw["stop"] = _st
                _eb2 = dict(_kw.get("extra_body") or {})
                _eb2["include_stop_str_in_output"] = True
                _kw["extra_body"] = _eb2
            _terse = _kw.pop("_t2_terse", False)
            if _t2_notc and _kw.get("tool_choice") == "required":
                _kw = dict(_kw)
                _kw.pop("tool_choice", None)
                _kw["_t2_stripped_tc"] = True
            _stripped = bool(_kw.pop("_t2_stripped_tc", False))
            if _t2_toolobs:
                try:
                    _ms0 = _a[1] if len(_a) > 1 else _kw.get("messages")
                    for _m0 in reversed(list(_ms0 or [])):
                        if str(getattr(_m0, "role", "")) != "tool":
                            continue
                        _k0 = str(getattr(_m0, "id", None) or id(_m0))
                        if _k0 in _t2_seen_tool:
                            break
                        _t2_seen_tool.add(_k0)
                        _c0t = " ".join(str(getattr(_m0, "content", "") or "").split())
                        print("[T2_TOOL_OBS] id=%s err=%s -> %s"
                              % (_k0[:14], bool(getattr(_m0, "error", False)),
                                 _c0t[:_t2_toolobs_cap]),
                              file=_sys_tr.stderr, flush=True)
                        break
                except Exception as _oe:
                    print("[T2_TOOL_OBS] skipped: %r" % (_oe,), file=_sys_tr.stderr, flush=True)
            _r = _orig_gen_t2(*_a, **_kw)
            # ★P2 탐지 후 재생성 (2026-08-31·7월 설계서 §2-2 그대로 구현)
            #   사용자 지시: *"시간은 줄일 필요가 없다. **JSON 폭주하는 것만 막으면 된다.**"*
            #   트리거(닫힌 술어): tool_calls 가 비었고 ∧ 본문에 여는 태그가 닫는 태그보다 많다
            #     (= 정지 실패로 마지막 블록이 절단됐다).
            #   동작: 같은 문맥으로 **`stop=["</tool_call>"]` 켜서 1회만** 재생성.
            #   ⇒ **전역 stop 이 아니라 병리 턴에만** 걸므로 정상 턴의 다중 호출은 보존된다
            #     (7월 §1-4 실측: 다중호출 2.3%턴 · **19% sim** — 전역 stop 은 이걸 판다).
            #   cap: sim 당 `T2_P2_CAP`(기본 3) — 무한 재생성 금지.
            _p2 = None
            if (_t2_p2 and not (getattr(_r, "tool_calls", None) or None)
                    and _t2_p2_used[0] < _t2_p2_cap):
                _c2 = str(getattr(_r, "content", None) or "")
                _o2 = _c2.count("<tool_call>"); _cl2 = _c2.count("</tool_call>")
                # ★전손(content 0 ∧ tool_calls 0)도 같은 사다리로 보낸다 (§S-2 1층·기본 ON).
                #   이 갈래는 **1차 재샘플까지만** — 아래 `tool_choice="required"` 계단은 건너뛴다
                #   (산문 턴에 도구를 강제로 사는 것을 막는다·[[70]]).
                _empty2 = (os.environ.get("T2_P2_EMPTY", "1") == "1" and _t2_msg_empty(_r))
                if _o2 > _cl2 or _empty2:
                    _t2_p2_used[0] += 1
                    # ★2026-08-31 오프라인 재생으로 `stop` 을 뺐다 (faildump 29건 · 6건 재생):
                    #     회복률 **stop OFF 5/6 == stop ON 5/6** — stop 은 기여 0.
                    #     그리고 한 건에서 **tc=4 -> tc=1** 로 **다중호출을 3개 팔았다**.
                    #   ⇒ 끊김은 **비결정적**이고, 필요한 것은 **재샘플링 하나**다.
                    #     `T2_P2_STOP=1` 이면 종전대로 stop 을 켠다(대조팔 전용).
                    _kw2 = dict(_kw)
                    if os.environ.get("T2_P2_STOP") == "1":
                        _st2 = list(_kw2.get("stop") or [])
                        if "</tool_call>" not in _st2:
                            _st2.append("</tool_call>")
                        _kw2["stop"] = _st2
                        _eb4 = dict(_kw2.get("extra_body") or {})
                        _eb4["include_stop_str_in_output"] = True
                        _kw2["extra_body"] = _eb4
                    print("[T2_P2] truncated tool_call detected (opens=%d closes=%d) -> resample (%d/%d)"
                          % (_o2, _cl2, _t2_p2_used[0], _t2_p2_cap),
                          file=_sys_tr.stderr, flush=True)
                    try:
                        _r2 = _orig_gen_t2(*_a, **_kw2)
                        if getattr(_r2, "tool_calls", None):
                            print("[T2_P2] regen recovered tool_calls=%d"
                                  % len(_r2.tool_calls), file=_sys_tr.stderr, flush=True)
                            _r = _r2; _p2 = True
                        else:
                            # ★2차 계단 (2026-08-31·사용자 제안 "tool_call 실패한 경우에만
                            #   tool_call 강제하거나 요청할 방법은 없나"):
                            #   전역 `required` 는 엉뚱한 도구를 낸다(x659: KB_search_bm25 로 때움).
                            #   그러나 **실패 경로에서만** 쓰면 그 부작용이 그 턴에 갇히고,
                            #   대안은 "행동 0 = reward 0" 이므로 밑질 것이 없다.
                            #   ⛔`T2_P2_NOFORCE=1` 이면 이 계단을 끈다(대조팔).
                            if os.environ.get("T2_P2_NOFORCE") != "1":
                                _kw3 = dict(_kw2)
                                _kw3["tool_choice"] = "required"
                                try:
                                    _r3 = _orig_gen_t2(*_a, **_kw3)
                                    if getattr(_r3, "tool_calls", None):
                                        print("[T2_P2] tier2 required recovered tool_calls=%d"
                                              % len(_r3.tool_calls),
                                              file=_sys_tr.stderr, flush=True)
                                        _r = _r3; _p2 = "tier2"
                                    else:
                                        print("[T2_P2] tier2 required still empty -> salvage",
                                              file=_sys_tr.stderr, flush=True)
                                except Exception as _pe3:
                                    print("[T2_P2] tier2 failed (no-op): %r" % (_pe3,),
                                          file=_sys_tr.stderr, flush=True)
                            else:
                                print("[T2_P2] regen still no tool_calls - falling through to salvage",
                                      file=_sys_tr.stderr, flush=True)
                    except Exception as _pe2:
                        print("[T2_P2] regen failed (no-op): %r" % (_pe2,),
                              file=_sys_tr.stderr, flush=True)
            # ★계기 수리 (2026-08-31): reasoning 은 tau2 `AssistantMessage` 에 **필드가 없다**
            #   (`data_model/message.py` 에 reasoning* 0회) — 그래서 `getattr(_r,"reasoning")` 은
            #   언제나 None 이고 `reason=` 칸이 **2336/2336 상수 0** 이었다. 원문은 응답 그대로가
            #   실린 `raw_data` 에만 남는다(`utils/llm_utils.py` 가 `raw_data=response.to_dict()`).
            #   ⇒ 그 자리를 읽는다. 절단 시 생성물이 전부 여기로 가므로, 이 칸이 0인지 아닌지가
            #     *"전손"* 과 *"진짜 빈 응답"* 을 가르는 유일한 계기다.
            _rsn = _reasoning_of(_r)
            _salv = None
            if _t2_salvage and not (getattr(_r, "tool_calls", None) or None):
                _salv = _t2_salvage_calls(_r, _kw)
            # ★T2_FAILDUMP (2026-08-30·§L-13) — **실패한 그 호출의 요청을 통째로** 떨군다.
            #   왜: 궤적(results.json)에 기록된 문맥으로 재생하면 **정상 파싱된다**(실측).
            #   ⇒ 런이 모델에 보낸 것은 궤적과 다르다 — 우리 스택이 `work` 에 얹는 주입은
            #     궤적에 안 남는다. 그 차이가 유일한 미지수다.
            #   `T2_PROMPT_DUMP` 는 `_gen` 두 곳에 달려 라이브 경로를 못 잡았다(§L-10).
            #   여기는 **모든 생성이 지나는 자리**이고, **실패 시에만** 쓰므로 비용이 0에 가깝다.
            if _t2_faildump and (_salv or not (getattr(_r, "tool_calls", None) or None)):
                _c0 = str(getattr(_r, "content", None) or "")
                # ★2026-08-31: 구판 술어는 본문에 `<tool_call>` 이 **있을 때만** 떴다. 그런데
                #   제일 비싼 실패(전손 = content 0B ∧ tool_calls 0)에는 그 문자열이 없다 —
                #   그래서 원문 회수가 구조적으로 0이었다. 전손도 뜬다.
                # ★2026-09-03: **절단도 뜬다**. 구판은 `<tool_call>` 이 있거나 전손일 때만 떠서
                #   **JSON 프로브의 절단**(claimprov·selfdecl·writeprov)은 회수가 구조적으로 0이었다.
                #   실물: task_071 `gen=8192 TRUNC reason=0B content=39530B` — 그 8k 가 반복 폭주인지
                #   정상 장문인지 **판정할 원문이 없었다**([[30]] 계기는 회수돼야 존재한다).
                _fr0 = (getattr(_r, "finish_reason", None)
                        or (((getattr(_r, "raw_data", None) or {}).get("choices")
                             or [{}])[0] or {}).get("finish_reason"))
                if "<tool_call>" in _c0 or not _c0.strip() or str(_fr0) == "length":
                    try:
                        _msgs = _a[1] if len(_a) > 1 else _kw.get("messages")
                        _rec = {"call_name": _kw.get("call_name"),
                                "max_tokens": _kw.get("max_tokens"),
                                # ★2026-09-01 §S-0′: **이 키가 없었다**(주석은 요구하는데 코드가
                                #   안 넣었다) — 그래서 전손 5건에서 `stop`/`length` 를 못 갈랐다.
                                "finish_reason": (
                                    getattr(_r, "finish_reason", None)
                                    or (((getattr(_r, "raw_data", None) or {}).get("choices")
                                         or [{}])[0] or {}).get("finish_reason")),
                                "thinking_budget": ((_kw.get("extra_body") or {})
                                                    .get("thinking_token_budget")
                                                    or _kw.get("thinking_token_budget")),
                                # ★2026-08-31: 끊김의 원인을 가르려면 이 둘이 있어야 한다 —
                                #   finish=length 면 예산 소진, stop 이면 모델이 스스로 멈춘 것.
                                #   reasoning 이 길면 "사고가 예산을 먹어 tool_call 이 잘렸다"가 확정된다.
                                "usage": (getattr(_r, "usage", None) or {}),
                                "reasoning_len": len(_rsn),
                                "tool_choice": _kw.get("tool_choice"),
                                "n_tools": len(_kw.get("tools") or _a[2] if len(_a) > 2 else
                                               (_kw.get("tools") or [])),
                                # ★반복 폭주 판정용: 앞/뒤 토막 + 압축비. 반복이면 비가 급락한다
                                #   (엔진이 내용을 해석하지 않는다 — 바이트만 센다).
                                "content_len": len(_c0),
                                "content_gzip_ratio": (
                                    round(len(__import__("zlib").compress(
                                        _c0.encode("utf-8", "replace"), 6)) / max(1, len(_c0)), 4)
                                    if _c0 else None),
                                "content_head": _c0[:4000],
                                "content_tail": _c0[-2000:] if len(_c0) > 6000 else "",
                                "messages": [{"role": str(getattr(_m, "role", "?")),
                                              "content": str(getattr(_m, "content", "") or "")[:2500],
                                              "tool_calls": [str(getattr(_t, "name", "?"))
                                                             for _t in (getattr(_m, "tool_calls", None) or [])]}
                                             for _m in (_msgs or [])]}
                        with open(_t2_faildump, "a", encoding="utf-8") as _fh:
                            _fh.write(_json_tr.dumps(_rec, ensure_ascii=False) + chr(10))
                        print("[T2_FAILDUMP] wrote (call=%s msgs=%d)"
                              % (_kw.get("call_name"), len(_msgs or [])),
                              file=_sys_tr.stderr, flush=True)
                    except Exception as _fe:
                        print("[T2_FAILDUMP] skipped: %r" % (_fe,), file=_sys_tr.stderr, flush=True)
            try:
                _c = str(getattr(_r, "content", None) or "")
                _tcs = getattr(_r, "tool_calls", None) or []
                # ★생성 토큰 (2026-08-31·사용자 지시 "계기 넣고") — **content 길이 != 생성 토큰**이다.
                #   사고 ON 호출은 reasoning 이 토큰을 먹는데 content 에는 안 잡힌다(실측: content 38B ·
                #   생성 1,247토큰). 형식별 상한을 정하려면 이 수가 있어야 한다.
                #   `tau2` 가 `usage`(completion_tokens/prompt_tokens)를 메시지에 실어 준다
                #   (`llm_utils.py:134 get_response_usage` · `message.py:426 usage`).
                # ★`reasoning` 이 정식 필드다 (2026-08-31 정정) — 내가 `reasoning_content` 를 읽어
                #   계속 0B 로 보였고 그 근거로 "파서가 분리를 안 한다"고 오진했다. 실제 응답 키는
                #   ['annotations','audio','content','function_call','**reasoning**','refusal','role']
                #   이고 reasoning 1,761B / content 27B 로 **정상 분리**되고 있었다.
                #   ⚠vLLM #35221: 절단되면(끝 토큰 부재) 파서가 경계를 못 찾아 **전부를 한쪽으로 쏟는다**.
                #     그때 content 에 담긴 <tool_call> 은 **사고 중 검토물**일 수 있다 -> salvage 위험.
                _rsn = _reasoning_of(_r)
                _u = getattr(_r, "usage", None) or {}
                _ct = _u.get("completion_tokens") if isinstance(_u, dict) else None
                _pt = _u.get("prompt_tokens") if isinstance(_u, dict) else None
                _mtq = _kw.get("max_tokens")
                _trunc = bool(_ct and _mtq and int(_ct) >= int(_mtq))
                # ★§S-0′ (2026-09-01): 사고 예산을 한 칸 남긴다 — 없으면 다음번에도
                #   "예산이 상한을 먹었나"를 못 가른다(전손 5건에서 실제로 못 갈랐다).
                _tb_tr = ((_kw.get("extra_body") or {}).get("thinking_token_budget")
                          or _kw.get("thinking_token_budget"))
                print("[T2_GEN_TRACE] call=%s max_tokens=%s tb=%s tool_choice=%s%s -> gen=%s prompt=%s%s reason=%dB content=%dB tool_calls=%d"
                      % (_kw.get("call_name"), _kw.get("max_tokens"), _tb_tr, _kw.get("tool_choice"),
                         ("%s%s" % (" (required STRIPPED)" if _stripped else "",
                                    (" [%s]" % _terse) if _terse else "")
                                   + (" SALVAGED=%d" % len(_salv) if _salv else "")),
                         _ct, _pt, " **TRUNC**" if _trunc else "",
                         len(_rsn), len(_c), len(_tcs)), file=_sys_tr.stderr, flush=True)
            except Exception:
                pass
            return _r

        _la_tr.generate = _t2_traced_generate
        print("[t2_run] GEN_TRACE ON (생성 깔때기 계기·§L-5)%s"
              % ((" · NO_FORCE_TOOLCHOICE" if _t2_notc else "")
                 + (" · PROBE_TERSE(no-think·mt=%d)" % _t2_probe_mt if _t2_probe_terse else "")),
              file=_sys_tr.stderr, flush=True)
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
    if a.retry_delay is not None:
        _extra_cfg["retry_delay"] = a.retry_delay
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
    # ★모델에 매인 상수는 **서버에서 읽는다** (2026-08-31 · 사용자 지시 "config 파일과 옵션으로").
    #   결손: `_ctx_fits` 의 캡이 `44672`(Qwen2.5-32B 의 max_model_len)로 **박혀 있었다**.
    #   Q3.8 은 131,072 라 그 캡은 배달 게이트를 필요보다 훨씬 일찍 닫는다 — 모델을 바꿔도
    #   따라오지 않는 상수는 [[84]] 가 기록한 사고(표면형)와 같은 종류다.
    #   여기서 한 번 읽어 `T2_MAX_MODEL_LEN` 으로 깔면, 소비자는 그 하나만 본다([[67]] 사본 금지).
    #   ⚠프로필/런처가 이미 선언했으면 **덮지 않는다**(선언 > 자동탐지).
    if not os.environ.get("T2_MAX_MODEL_LEN"):
        try:
            import json as _js0, urllib.request as _ur0
            with _ur0.urlopen((a.agent_base or "").rstrip("/") + "/models", timeout=5) as _r0:
                _d00 = ((_js0.load(_r0).get("data") or [{}])[0]) or {}
            _mml = _d00.get("max_model_len")
            if _mml:
                os.environ["T2_MAX_MODEL_LEN"] = str(int(_mml))
                print("[t2_run] served max_model_len=%s (id=%s) -> T2_MAX_MODEL_LEN"
                      % (_mml, _d00.get("id")), file=sys.stderr, flush=True)
        except Exception as _e0:
            print("[t2_run] max_model_len 탐지 실패(기본값 사용): %r" % (_e0,),
                  file=sys.stderr, flush=True)

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

    # ★provenance 사이드카 (2026-08-29 · 계기 결함 수리).
    #
    # 왜: `results.json` 의 `info.git_commit` 은 **cwd 의 sha** 이고, go_stack 이 `cd $GO_TAU2`
    #     한 뒤 실행하므로 그것은 **벤치마크 sha 이지 우리 엔진 sha 가 아니다**. 그래서 과거 런을
    #     엔진 버전에 묶을 수 없었다 — 2026-08-29 retail 회귀를 [[70]] 의 같은-sha A/B 로 귀속하려다
    #     7월 런의 엔진 버전을 복원할 수 없어 막혔다(로그는 로테이션으로 소실).
    # 무엇을: 엔진 sha + **런을 재현하는 데 필요한 조건 전부**를 결과 옆에 남긴다. 조건이 결과와
    #     같은 자리에 없으면 다음 사람이 다시 추측한다(오늘 `max_model_len` 44,672 가 그랬다 —
    #     Qwen2.5 의 YaRN 잔재였는데 아무 기록이 없어 Qwen3.8 런을 9회 ContextWindowExceeded 로
    #     태웠다).
    # 실패해도 본 결과에 영향 없음.
    try:
        import io as _io2
        import json as _js
        import subprocess as _sp
        import urllib.request as _ur

        def _sha(path):
            try:
                return _sp.check_output(["git", "-C", path, "rev-parse", "--short", "HEAD"],
                                        stderr=_sp.DEVNULL).decode().strip()
            except Exception:
                return None

        def _dirty(path):
            try:
                return bool(_sp.check_output(["git", "-C", path, "status", "--porcelain"],
                                             stderr=_sp.DEVNULL).decode().strip())
            except Exception:
                return None

        _eng_dir = os.path.dirname(os.path.abspath(__file__))
        _prov = {
            "engine_sha": _sha(_eng_dir),
            "engine_dirty": _dirty(_eng_dir),
            "bench_sha_cwd": _sha(os.getcwd()),
            "cwd": os.getcwd(),
            "domain": a.domain,
            "gate": a.gate,
            "agent_model": a.agent_model,
            "agent_base": a.agent_base,
            "user_llm": a.user_llm,
            "user_reasoning_effort": a.user_reasoning_effort,
            "retrieval_config": a.retrieval_config,
            "num_trials": a.num_trials,
            "max_concurrency": a.max_concurrency,
            "max_steps": a.max_steps,
            "task_ids": a.task_ids,
            "llm_timeout": os.environ.get("T2_LLM_TIMEOUT"),
            "llm_retries": os.environ.get("T2_LLM_RETRIES"),
            "levers_on": sorted(k for k, v in os.environ.items()
                                if k.startswith("T2_") and v not in ("", "0")),
        }
        # 에이전트 서버가 실제로 무엇을 서빙하는지 — 컨텍스트 한계가 조건이다.
        try:
            _base = (a.agent_base or "").rstrip("/")
            with _ur.urlopen(_base + "/models", timeout=5) as _r:
                _m = _js.load(_r)
            _d0 = (_m.get("data") or [{}])[0]
            _prov["served_model"] = _d0.get("id")
            _prov["served_max_model_len"] = _d0.get("max_model_len")
        except Exception as _e:
            _prov["served_probe_error"] = repr(_e)[:120]

        _sim_dir = os.path.join("data", "simulations", a.save_to)
        _out = (os.path.join(_sim_dir, "provenance.json")
                if os.path.isdir(_sim_dir) else (a.save_to + ".provenance.json"))
        with _io2.open(_out, "w", encoding="utf-8", newline="\n") as _f:
            _f.write(_js.dumps(_prov, ensure_ascii=False, indent=1))
        print("[t2_run] provenance -> %s (engine_sha=%s dirty=%s ctx=%s)"
              % (_out, _prov["engine_sha"], _prov["engine_dirty"],
                 _prov.get("served_max_model_len")))
    except Exception as e:
        print("[t2_run] provenance sidecar failed: %s: %s" % (type(e).__name__, e))

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
