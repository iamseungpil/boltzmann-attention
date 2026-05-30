# Coworker 실험 계획서 — Layered Ontology Agent (TBox planner + 결정적 executor)

> 대상: 4× A100 80GB coworker. 공유 채널 = GitHub `iamseungpil/boltzmann-attention` branch **`facet-rft-2026`**.
> 본 계획은 `reports/EXPERIMENT_DESIGN_v1_7_facet_rft.md` **§13(방향)·§15.9~15.12(layered ontology agent)**를 구현한다. **먼저 §15.11/§15.12를 읽을 것.**

---

## 0. ★ 방향 전환 (2026-05-31) — 단순 distillation → 다층 온톨로지 에이전트

이전 계획(B1–B4: 32B plain/facet × full/none 4조합 SFT 매트릭스)은 **단순 goal→tool distillation** 검증용이었다. 그 thesis는 **7B에서 이미 검증 완료**(아래 §1) → coworker는 그 중간단계를 **재현하지 않는다**. 대신 그 위에 올린 **layered hierarchical agent**(planner=TBox + 결정적 온톨로지 executor + LLM fallback)를 **A100×4 우위로 대규모 검증**한다.

**핵심 구조** (§15.11):
```
  PLANNER (LLM, 학습)  : 추상 PLAN_STEP만 emit ("Plan: apply_targeted_fix") — 구체 도구 안 봄 = 순수 TBox
        │
        ▼
  EXECUTOR (결정적, 무학습) : step + 관찰상태 → 구체 (tool, args)
        = step_realizes_tool⁻¹(후보) × observation_triggers(상태매칭) × arg_source(인자 provenance)
        │  miss(후보 모호)
        ▼
  LLM FALLBACK : 후보-제한 LLM이 도구 선택
```
- **전이 = 온톨로지 파일만 swap** (planner·PLAN_STEP vocab 불변 = TBox / `obs_triggers_<dom>.json`·`step_realization_<dom>.json` = ABox).
- **헤드라인 지표 = 결정적 coverage%** (온톨로지만으로 해결된 도구선택 비율). 나머지는 LLM fallback.
- **이미 구현 완료**(Track A, repo): `scripts/distill/ontology_resolver.py`(결정적 executor) + `scripts/distill/two_stage_agent.py`(planner+resolver+fallback wrap, 4 ablation mode). 필드·API 검증 완료.

---

## 0.1 ★★ 7B에서 검증 완료 → coworker가 SKIP 할 것 (재현 불필요)

| 검증된 것 (7B) | 결과 | → coworker 스킵 |
|---|---|---|
| **efficiency thesis** (NONE 정책제거 ≥ FULL 정책유지) | 3도메인 모두 NONE≥FULL (telecom .35/.30 · retail .77/.64 · airline .40/.30) | **32B full-arm 전부 스킵.** none(내부화)만 학습 |
| **goal→tool distillation 작동** (base≈0 → SFT 상승) | F1/seq_F1 AUC 0.902, base recall~0.04→student↑ | **단순 SFT-lift 재현(구 G1) 스킵.** base vs plain-SFT 매트릭스 불필요 |
| **plain vs facet** | facet 별이득 불명, 7B는 plain만으로 thesis 입증 | **facet-SFT arm(구 G3) 폐기.** plain/abstract만 |
| **F1/seq_F1·arg_bind 지표, GRPO dense reward 설계** | scorecard·grpo_reward.py 검증 | **지표/reward 재설계 불필요.** 그대로 사용 |
| **scorecard 3도메인 일반화** | read-제외·requestor 정정 후 +0.2~0.6 | **scorer 재검증 스킵** |

= 4조합(plain/facet × full/none) × 3도메인 = 12셀 SFT 매트릭스 → **abstract-none 32B 1개 + 큰모델 fallback probe로 축소.** "큰 모델로 같은 thesis 재확인"이 아니라 **큰 모델이 아니면 못 보는 것**(아래 §3)만 한다.

---

## 1. 배경 (7B 기확립, 요약)
- tau2-bench 도구 에이전트. 큰/작은 격차 = **94% 절차(distillable)**, capability 벽 1%. goal→tool이 success/failure 최강 변별.
- NONE 실패 63% = **recall-miss / anti-loop**(정책 없이 진단만 반복, fix commit 안 함). = 결정적 executor + harness 재시도가 직접 겨냥하는 실패모드.
- **★결정적 executor의 경계 (7B inducer 분석, §15.11 v3)**:
  - **telecom류 상태머신 = 결정적 작동.** `enable_roaming ⇐ roaming_enabled=False`(prec 0.81) · `resume_line ⇐ status=Paid`(0.98) · `send_payment_request ⇐ bill.status=Draft`(0.85). 관찰상태가 도구를 일의적으로 결정.
  - **retail/airline 카탈로그-선택형 = 과적합 → LLM fallback 필요.** variant.size=S·destination=PHX 등 인스턴스-특수 trigger(소표본). 결정적 안 됨.
  - → **coverage%가 곧 "온톨로지가 어디까지 일하나"의 정량 경계.** 이게 이 트랙의 핵심 결과물.

---

## 2. 가설 & 지표

**Thesis (layered)**: (H1) goal→tool 절차를 **추상 PLAN_STEP planner(TBox)** + **결정적 온톨로지 executor(ABox)**로 분해하면, (H2) executor가 상태머신 도메인에서 **높은 결정적 coverage**로 도구를 LLM 없이 선택하고, (H3) **온톨로지 파일 swap만으로 held-out 도메인 전이**(planner 재학습 0)하며, (H4) 큰 모델은 **결정적이 못 푸는 catalogue-선택 잔차의 LLM-fallback 품질**에서 우위를 보인다.

**지표**
- **★결정적 coverage%** (headline): `deterministic / tool_call_turns`. `two_stage_agent.py`가 `coverage_<mode>_<set>_<split>.json`에 자동 저장(by_step 분해 포함).
- **pass^1** (tau2 test split) — 모드별·도메인별 최종 성능.
- **F1 / seq_F1 / arg_bind** (scorecard, 기존) — goal→tool 품질.
- **transfer Δ**: in-distribution vs ontology-swap LODO의 pass^1·coverage 차이.
- **efficiency**: none-arm 토큰/KV 절감 (7B에서 입증, 32B는 확인만).

**Ablation 6모드** (`two_stage_agent.py --mode`, §15.11 + §15.13):
| mode | 구성 | 측정 의미 | 상태 |
|---|---|---|---|
| `base` | 전체도구 LLM (planner/resolver 無) | 하한 baseline | done |
| `resolver` | planner step + 결정적 rule resolver(ABox=dict), miss시 planner 자기콜 | **순수 결정적 coverage** (LLM 추가 0) | done(§15.11) |
| `ontollm` | ABox를 프롬프트로 직렬화, LLM in-context 선택 | 프롬프트 천장 (토큰 비쌈) | 신규(b) |
| `xattn` | **ABox=cross-attn 메모리, TBox=학습 weights** | **★본 트랙 novelty**(토큰0+학습 유연성) | 신규(c)·B5* |
| `fallback` | rule resolver + miss시 후보제한 LLM | 결정적+fallback 조합 (실사용) | done(§15.11) |
| `monolithic` | abstract 모델이 Plan+구체콜 end-to-end (resolver bypass) | planner 단독 상한 | done(§15.11) |

→ 사다리 판독: `base→ontollm`=온톨로지 프롬프트-side 기여 / `resolver→xattn`=rule이 못 푼 걸 학습 attention이 메운 양(catalogue 도메인서 격차 클 것) / `ontollm vs xattn`=프롬프트 vs weights(토큰·전이·정확도) / `xattn vs monolithic`=ABox-conditioning 이득.

---

## 3. Coworker 태스크 (A100×4, layered 중심)

### B1*. 32B abstract-planner SFT  ★우선 (none-only, 1 arm)
- **목적**: 큰 planner가 추상 PLAN_STEP을 7B보다 정확히 emit → 결정적 executor의 입력 품질↑.
- **데이터**: `reports/facet_rft_2026/phase4_distill/sft_data/sft_abstract_train_all.jsonl` (Plan-step prefix 주입본, repo). **plain/facet/full 변형 없음** — abstract-none 단일.
- **trainer**: `scripts/distill/lora_train_chat_toolcall.py --system-mode none --max-seq-len 8192` (7B 레시피 그대로, 32B만 교체). base=`Qwen/Qwen2.5-32B-Instruct`.
- **GPU**: 1× 80GB (LoRA r16 + grad-ckpt) 또는 2× FSDP.
- **출력**: adapter `lora_adapters/qwen32b_abstract_none/` → HF private. (구 4조합 매트릭스 폐기.)
- **성공기준**: 수렴 + monolithic 모드 pass^1 ≥ 7B abstract.

### B2*. two_stage_agent ablation × LODO 매트릭스  ★★대규모 병렬 (coworker 핵심 기여)
- **목적**: layered agent의 결정적 coverage + 전이를 **모드·도메인·온톨로지 전부** 채운다. A100×4 fan-out 최적.
- **러너**: `scripts/distill/two_stage_agent.py` (repo). 모드별 호출:
  ```
  python scripts/distill/two_stage_agent.py --mode {base|resolver|ontollm|xattn|fallback|monolithic} \
    --domain <dom> --task-set <dom> --task-split test \
    --agent-llm openai/<served-lora> --base-url http://127.0.0.1:<port>/v1 --agent-api-key sk-noauth \
    --user-llm openai/openai/gpt-4.1 --user-base-url https://openrouter.ai/api/v1 --user-api-key $OPENROUTER_API_KEY
  ```
- **매트릭스** (각 셀 = pass^1 + 결정적 coverage%):
  - **{7B-abstract, 32B-abstract}** × **6 modes**(base/resolver/ontollm/xattn/fallback/monolithic) × **{telecom, retail, airline} in-distribution test**. (xattn 모드는 B5* 학습 완료 후 추가.)
  - **★LODO ontology-swap**: planner 불변, `--ontology-domain <other>`로 ABox만 교체. 핵심 셀 = **telecom planner + airline ontology**(전이) vs **airline planner + airline ontology**(in-dist) — coverage·pass^1 격차로 "온톨로지만 swap해도 전이되나"(H3) 판정.
  - **ABox-swap sanity**: `--ontology-domain`을 틀린 도메인으로 주면 coverage가 무너져야 함(온톨로지가 실제로 일한다는 음성대조).
- **예상 결과**(7B 기준 외삽): telecom = `resolver` 높은 coverage(상태머신), retail/airline = `resolver` 낮음 → `fallback` 비중↑. **coverage%가 도메인별로 갈리는 곡선이 메인 figure.**
- **출력**: `reports/facet_rft_2026/phase4_distill/coworker_a100/two_stage/<run>/` — results.json + coverage_*.json manifest.

### B3*. Capability-ceiling = fallback 품질 probe  (B4 재정의)
- **목적**: 결정적 executor가 **못 푸는 잔차**(retail/airline catalogue-선택, coverage의 1−x)를 **큰 모델 LLM-fallback이 메우는가**. = capability가 도움되는 지점을 layered 구조 안에서 정확히 격리.
- **방법**: `--mode fallback` 고정, fallback LLM만 {7B, 32B, 70B} 교체 → miss-turn에서의 도구선택 정확도·pass^1 비교. (planner·resolver 동일.)
- **GPU**: 70B = 2× A100(AWQ) / 4× bf16. (Qwen2.5-72B / Llama-3.3-70B, Track A 다운로드본 HF 공유.)
- **출력**: fallback-모델 크기 함수의 catalogue-도메인 pass^1 → "결정적이 못 푸는 부분 = capability냐 절차냐" 결론.

### B4*. (조건부) On-policy GRPO with Group J reward
- **선행조건**: B2* 결과 layered가 base 대비 양성(coverage 의미 + pass^1 ≥ baseline)일 때만.
- **reward**: `scripts/distill/grpo_reward.py`(검증) + **Group J 항**(repairs_state recall, distractor penalty, step penalty=anti-loop). 정책 init=B1* abstract-none 어댑터. planner의 step-emit에 dense reward.
- **trl**: seka_env(transformers 4.51.3) 충돌 → coworker는 **trl 호환 별도 venv** 권장(transformers 버전 맞춤). 안 되면 수동 GRPO 루프(Track A 방식).
- **출력**: GRPO adapter + reward curve + B2* 매트릭스 갱신.

### B5*. ★Neural ABox-conditioned resolver (cross-attn) — 본 트랙 novelty (design §15.13, v1.29)
- **목적**: §15.11의 결정적 rule resolver(코드+dict)를 **학습된 neural resolver**로 일반화. **TBox=ABox를 읽어 도구·인자를 고르는 절차(cross-attn weights, 도메인무관 고정) / ABox=온톨로지 관계 메모리(도메인별 swap)**. rule이 못 푸는 catalogue-선택형(retail/airline)까지 coverage 천장을 올리고, **온톨로지 메모리 swap만으로 전이**.
- **왜 coworker(A100)**: 아키텍처 수술(base에 cross-attn block 삽입)+학습이 무거움. 7B는 (a)rule+(b)프롬프트 baseline·gap 정량화(Track A), **(c)xattn 학습·매트릭스는 B 트랙**.
- **아키텍처(우선 C-1)**: ABox 관계를 자연어 직렬화→frozen 텍스트 인코더→메모리 M={e_1..e_N}(도메인별). executor hidden h_t(관찰+abstract step)=Query → `cross_attn(Q=h_t,K=V=M)` → head가 (tool,args) emit. **학습=cross-attn W_Q/K/V+readout=TBox / swap=M=ABox**. 토큰0(프롬프트 아님). 대안: C-2 hypernet→ABox-LoRA(공유 TBox-LoRA + per-domain ABox-LoRA), C-3 graph encoder.
- **학습**: teacher SUCCESS 궤적(telret 등). 입력=관찰상태+planner의 abstract step, 타깃=GT(tool,args). planner(B1* abstract-none 어댑터)는 freeze 권장. **ABox 인코더는 도메인무관 텍스트 인코더**(관계→자연어→frozen embed) — swap 도메인 M이 학습분포와 같은 의미슬롯이어야 전이(★최난점).
- **eval/ablation**:
  - **`two_stage_agent --mode xattn`** 신규 → B2* 매트릭스에 모드 1개 추가({7B,32B}×{base,resolver,ontollm,**xattn**,fallback,monolithic}×3도메인×{in-dist,swap}).
  - **ABox-memory swap LODO**: TBox(cross-attn weights) 불변, M_telecom→M_airline 교체만으로 held-out airline 작동?
  - **★ABox-ablation(검증가능성)**: 빈 M / 틀린 도메인 M 주입 시 성능 **붕괴**해야 "온톨로지가 실제로 일한다" 입증(attention이 ABox 무시·암기 아님). attention map으로 "어느 관계 읽었나" 해석.
- **선행조건**: Track A가 (a)resolver+(b)ontollm baseline으로 retail/airline rule-coverage gap을 정량화(=xattn이 메울 표적). gap이 크면 B5* 메인 기여로 진입.
- **출력**: `ontology_encoder.py`(관계→메모리) + cross-attn executor block + 학습된 TBox weights(HF) + 도메인별 ABox 메모리 + coverage/swap/ablation manifest (`coworker_a100/xattn/`).
- **리스크**: telret~1300 데이터로 cross-attn 신규 파라미터 학습 충분한지(→증강/32B), ABox 인코딩 분포정합(전이 핵심·최난점), 구현 복잡도((a)(b)보다 훨씬 무거움).

### B6*. ★Routine-derived layers — scenario/branch/placeholder 자동 induce (design §15.14, v1.30)
Routine(2507.14447) 4 메커니즘을 **자동 induce + 다층 executor**로 일반화. 전부 기존 induced 맵에서 추출(새 데이터 0). **우선순위 R4 > R3 > R1/R2.**
- **R4 scenario(★최대 레버, 3도메인)**: `induce_scenario_workflow.py` — fault-유형 클러스터=scenario(`fault_fix_map` 키), scenario별 workflow DAG, 초기 read→fault 시그니처 결정적 매칭. **planner 2단계(task→scenario→step)** + multi-fault 합집합 활성화(NONE 누락 직격). ABox swap·xattn 메모리를 scenario 슬롯으로.
- **R3 branch(telecom 결정적)**: `induce_branch_dag.py` — 같은 step 후 분기점+직전 read 대조 → `exclusive_choice(step,[(cond,tool)])`. 재료=observation_triggers∪distractor_for∪escalate_when(완료). mutual-exclusion + else→escalate(anti-loop 차단).
- **R1/R2 placeholder(arg_bind 계약)**: `induce_variable_slots.py` — step input/output 슬롯을 `ObservedState.by_source`(런타임 variable memory) key로 강제 채움. 빈 슬롯→miss→fallback. **인자 할루시네이션 구조적 불가**(arg_bind 0.32→계약).
- **eval**: 각 층 결정적 coverage% + marginal pass^1 + **multi-fault 누락 감소(R4)** + arg_bind 향상(R1/R2) + anti-loop 감소(R3). telecom 결정적 / retail·airline neural(→B5* xattn) 경계.
- **담당**: induce·결정적 검증=Track A(7B). neural scenario/branch(xattn 메모리에 scenario·분기 슬롯)=coworker. scenario-conditioned planner SFT는 B1* 데이터에 scenario 라벨 추가로 흡수 가능.
- **출력**: 3 inducer + scenario/branch/variable 맵(`induced/{scenario_workflow,branch_dag,variable_slots}_<dom>.json`) + two_stage_agent scenario-2단계 통합.

---

## 4. 환경 셋업 (coworker box)

```bash
# 1) 코드 + 학습데이터 (전부 git)
git clone -b facet-rft-2026 https://github.com/iamseungpil/boltzmann-attention.git bap-pi
#  → scripts/distill/{two_stage_agent,ontology_resolver,lora_train_chat_toolcall,...}.py
#  → reports/.../sft_data/sft_abstract_train_all.jsonl + induced/{obs_triggers,step_realization}_<dom>.json

# 2) eval용 tau2-bench (public)
git clone https://github.com/sierra-research/tau2-bench.git && cd tau2-bench && pip install -e .

# 3) python env
#   학습: torch + transformers>=4.51 + peft + accelerate (검증: 4.51.3 / torch 2.7.0+cu126; flash-attn 옵션)
#   서빙/eval: vllm==0.11.0 (--enable-lora --max-lora-rank 16 --lora-modules <name>=<adapter>, hermes parser)
#   (B4* GRPO) trl 호환 별도 venv

# 4) 모델 (HF): Qwen/Qwen2.5-32B-Instruct, Qwen/Qwen2.5-7B-Instruct, (B3*) Qwen2.5-72B / Llama-3.3-70B

# 5) OpenRouter (user_sim + airline/retail judge) — 키 공유받기
export OPENROUTER_API_KEY=...
#   user_sim: --user-llm openai/openai/gpt-4.1 --user-base-url https://openrouter.ai/api/v1 --user-api-key $OPENROUTER_API_KEY
#   ⚠️ openai/openai/gpt-4.1 (double 접두사) 필수. judge는 phase1_runner/two_stage_agent가 자동 openrouter 라우팅(pull만).
```

---

## 5. 데이터/아티팩트 핸드오프

| 아티팩트 | 채널 | 비고 |
|---|---|---|
| abstract SFT jsonl + induced 온톨로지 맵 | **GitHub repo** | `sft_abstract_train_all.jsonl`, `induced/{obs_triggers,step_realization}_<dom>.json` |
| two_stage_agent.py / ontology_resolver.py / trainer / scorecard / grpo_reward | **GitHub repo** | `scripts/distill/` |
| eval 데이터 (domains/split/env) | **public tau2-bench** | clone+pip |
| 학습된 adapter (32B abstract, 70B) | **HF private** | repo push 금지(GB급) |
| 결과 results/coverage manifest | **GitHub** (`coworker_a100/` 하위) | 100MB↑면 manifest만 |

---

## 6. 협업 규약
- **branch**: 공유 `facet-rft-2026`. commit 전 `git pull --rebase origin facet-rft-2026`.
- **출력 서브트리**: coworker = `reports/facet_rft_2026/phase4_distill/coworker_a100/` 아래만 (Track A는 그 밖). 충돌 회피.
- **대용량**: results.json 100MB↑ commit 금지 → manifest만, 원본 HF/디스크. adapter는 HF.
- **git user**: `iamseungpil <iamseungpil@users.noreply.github.com>`. 파일 수정/추가 시 자동 commit+push.

---

## 7. 일정 (4× A100, 3주 — xattn 트랙 추가 반영)

| 주 | Track B (coworker) | Track A (우리) |
|---|---|---|
| **W1** | 셋업 + **B1* 32B abstract-none SFT** + **B2* ablation×LODO 매트릭스 착수**(base/resolver/fallback/monolithic) | B LODO(telret) 학습완료 → 7B two_stage telecom coverage + airline swap LODO |
| **W2** | **B2* 완성**(rule계열 전모드×3도메인×LODO) + **ontollm 모드** + **B3* fallback 70B probe** + (조건부)B4* GRPO | (a)resolver+(b)ontollm baseline로 **retail/airline rule-coverage gap 정량화** → B5* 표적 확정 |
| **W3** | **★B5* xattn neural resolver**(C-1 cross-attn 학습 → xattn 모드 매트릭스 + ABox-swap LODO + ABox-ablation) | 결과 종합, coverage 곡선·xattn vs rule/프롬프트 figure, 논문 표 |

---

## 8. Go/No-Go 게이트 (layered 재정의)

| 게이트 | 기준 | 판단 |
|---|---|---|
| **G1* (planner SFT)** | 32B abstract monolithic pass^1 ≥ 7B abstract, val 수렴 | 진행 / 데이터·trainer 점검 |
| **G2* (결정적 coverage)** | telecom `resolver` 모드 **coverage ≥ 60%** (상태머신서 온톨로지가 실제로 도구선택) | layered 핵심 양성 / inducer precision 재정제 |
| **G3* (ontology-swap 전이)** | telecom-planner + airline-ontology 의 pass^1·coverage가 **base 대비 +, in-dist의 ≥70% 회수** | H3 전이 입증 / ABox 재설계 |
| **G4* (fallback capability)** | catalogue 도메인(retail/airline)서 70B fallback이 7B fallback 대비 **miss-turn 정확도 +≥10%p** | capability가 잔차 메움 확인 / 결정적 확장 필요 |
| **G5* (GRPO, 조건부)** | anti-loop(step penalty)로 NONE max-step 실패 직접 감소 + pass^1 +≥5%p | 진입 / SFT로 충분 보고 |
| **G6* (xattn neural resolver, B5*)** | catalogue 도메인(retail/airline)서 `xattn` coverage·pass^1이 `resolver`(rule) 및 `ontollm`(프롬프트) **둘 다 상회** + **ABox-ablation으로 붕괴**(빈/틀린 M) + **swap LODO가 in-dist의 ≥70% 회수** | 본 트랙 novelty 입증 / 인코딩 분포정합·데이터 재설계 |
| **G7* (R4 scenario, B6*)** | scenario-2단계 planner가 평면 대비 **multi-fault task pass^1 +≥5%p**(누락 감소) + scenario 매칭 정확도 ≥80% (3도메인) | 계획층 가치 입증 / fault 클러스터 재정의 |
| **G8* (R3 branch / R1·R2 placeholder, B6*)** | branch=telecom anti-loop(max-step 실패) 감소 + placeholder=arg_bind **0.32→≥0.7**(인자 계약) | 실행층 가치 입증 / 슬롯 induce 정제 |

---

## 9. 핵심 리스크 / 주의
- **결정적 executor 경계**: retail/airline은 인스턴스-특수 trigger 과적합 → `resolver` 단독 coverage 낮을 것(예상·정상). **fallback 비중 자체가 결과** — 낮은 coverage를 실패로 보지 말 것.
- **predicate 정합성**: `ontology_resolver`의 `ObservedState` flatten/_scalar는 `induce_observation_triggers.py`와 **동일 키잉** 필수(런타임 pred가 induced pred와 글자단위 일치). 도메인 추가 시 inducer 먼저 재실행.
- **planner step 파싱**: abstract 모델이 `Plan: <step>` 포맷을 안정적으로 emit해야 resolver 동작. monolithic 모드로 emit률 먼저 확인.
- **cross-domain 분포이동**: 도메인특수성 내부화 시 전이 취약(Transmuting 100→42.7). **ABox-swap sanity(틀린 온톨로지 → coverage 붕괴)** 로 "불변 절차만 weights" 확인.
- **user_sim 비용**: OpenRouter gpt-4.1 과금. 매트릭스 셀 多(2모델×4모드×3도메인×{in-dist,LODO}) → test N≈20–40/셀로 관리, 우선순위=telecom resolver/fallback + airline swap.
- **judge**: airline/retail reward_basis=nl_assertions → LLM judge 필수. two_stage_agent도 `_route_nl_judge_via_openrouter()` 포함(pull만).
- **trl**: seka_env 충돌 → 별도 venv. vLLM 0.11.0 / tau2 버전 Track A와 일치.
