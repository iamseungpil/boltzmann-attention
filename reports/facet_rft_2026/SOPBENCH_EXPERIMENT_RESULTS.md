# SOPBench Experiment Results (2026-05-31~)

> **이 문서가 SOPBench 실험의 유일한 결과 기록 문서다.** 가설·결과·해석·다음 스텝을 누적 기록.
> coworker와 공유: branch `facet-rft-2026`, `reports/facet_rft_2026/SOPBENCH_EXPERIMENT_RESULTS.md`.
> 설계 권위본 = `scripts/distill/WORKFLOW_ONTOLOGY_DESIGN.md §9`.
> 코워커 실험 계획 = `reports/facet_rft_2026/COWORKER_EXPERIMENT_PLAN.md`.

---

## 실험 설정 (표준, 이하 모든 실험에 적용)

| 항목 | 표준값 | 근거 |
|---|---|---|
| **tool_list** | **`full`** (leaderboard 정합) | 저자 `scripts/simulation/*.sh` + 8모델 bank 전수 대조로 확정 |
| **tool_list (진단)** | `oracle` 병행 | oracle−full 격차 = distractor 도구선택 난이도 = 우리 planner 표적 |
| **user_model** | **없음(dummy user)** | `user_known` 첫 메시지로 선제공, agent 단독. 저자 표준과 동일 (multi-turn/adv는 별도) |
| **평가** | `env/evaluator.py` rule oracle | pass@1=5체크 AND (no_tool_call_error ∧ constraint_not_violated ∧ database_match ∧ action_called_correctly ∧ dirgraph_satisfied). **LLM judge 0** |
| **도메인** | 7 (bank·dmv·healthcare·hotel·library·online_market·university) | SOPBench(Zekun Li, 2503.08669) 전체 |
| **모델** | Qwen2.5-7B-Instruct (baseline/arm-1/2) | 약한 모델 regime = 구조 향상 여지 최대 |
| **서빙** | vLLM 0.11.0, GPU0:9100(fc) · GPU1:9000(react) | tau2_vllm_env; hermes tool-call parser |
| **셀 표기** | `(mode)/(tool_list)` | 예: fc/full, react/oracle |

---

## Arm 정의 (실험 비교군)

| arm | 구성 | 목적 |
|---|---|---|
| **arm-1: LLM-alone (baseline)** | 단일 LLM, full tool list, 우리 harness | leaderboard 재현·출발점 |
| **arm-0: oracle ceiling** | GT `directed_action_graph` walk = 결정론 executor | 상한; 기여 아님 |
| **arm-2: L0 symbolic** | forward-chain over operator pre/effect, 무학습 | "operator만으로 얼마나" 바닥 |
| **arm-3: L1 LLM+structure** | planner(LLM, 추상 operator in-context) + resolver(b), 무학습 | 구조 기여 격리, 무학습 |
| **arm-4: L2 학습 planner (★헤드라인)** | 학습 ABox-conditioned planner + resolver(c, xattn) | 전이 주장; N-1→held-out LODO |

---

## Exp-1 — arm-1 전체 baseline (2026-05-31, 완료)

### 가설
- H0: leaderboard pass@1 = full tool list 기준 → 우리 harness로 재현 가능.
- H1: oracle과 full의 격차가 크다 → tool-selection@scale이 7B의 주 병목.
- H2: FC와 ReAct는 7B full에서 큰 차이 없다.

### 실행 조건
- Qwen2.5-7B-Instruct, 7도메인 × {fc, react} × {oracle, full}, dummy user, max_turns=20, N=task_all(도메인별 전체).
- 실행: `/tmp/overnight.sh` 자동 스크립트, GPU0/1 2레인 병렬, 결과 `/tmp/overnight_results.tsv` + `output_overnight/<domain>/`.

### 결과 — pass@1 (2026-05-31)

| domain | fc/oracle | fc/full | react/oracle | **react/full** | README(react/full) |
|---|--:|--:|--:|--:|--:|
| bank | 0.552 | 0.037 | 0.590 | **0.052** | 0.0522 ✓ |
| dmv | 0.454 | 0.113 | 0.629 | **0.217** | 0.2062 ✓ |
| healthcare | 0.121 | 0.081 | 0.347 | **0.161** | 0.1694 ✓ |
| hotel | 0.077 | 0.005 | 0.169 | **0.000** | 0.0051 ✓ |
| library | 0.379 | 0.136 | 0.470 | **0.136** | 0.1515 ✓ |
| online_market | 0.198 | 0.093 | 0.430 | **0.076** | 0.0930 ✓ |
| university | 0.191 | 0.048 | 0.381 | **0.024** | 0.0000 ✓ |
| **평균** | **0.282** | **0.073** | **0.431** | **0.095** | 0.097 ✓ |

react/full 평균 9.5% ≈ 공식 leaderboard Qwen-7B 9.7% → **7도메인 전체 재현 성공.** (university 차이: 공식 0.00 vs 우리 2.4% — 소표본 노이즈.)

### 해석

**1. H0 채택: harness가 공식 수치를 재현한다.**
7도메인 react/full 전부 leaderboard와 일치 → 실험 신뢰성 확립. 이 harness 위에 모든 후속 arm을 얹으면 공정 비교 가능.

**2. H1 채택: tool_list가 압도적 변수 (oracle ≫ full).**

| | fc | react |
|---|--:|--:|
| oracle 평균 | 28.2% | 43.1% |
| full 평균 | **7.3%** | **9.5%** |
| 격차 (oracle−full) | 20.9%p | 33.6%p |

oracle에서 full로 가면 성능이 **3~7배 하락**. distractor 도구 중에서 올바른 도구를 고르는 것이 7B의 주 실패 원인. → 우리 planner(ABox operator 정보를 구조적으로 주입)가 이 격차를 줄이는 것이 **핵심 표적**.

**3. H2 채택: FC와 ReAct는 full에서 비슷 (7.3% vs 9.5%), oracle에서 ReAct 우세 (28% vs 43%).**
FC가 7B에 "유리"하다는 초기 가설 틀림. ReAct가 oracle에서 다소 우위지만 full에서는 모드 자체보다 tool_list가 지배적 → **우리 leaderboard 정합 표준 = react/full** 확정.

**4. 도메인별 난이도 변이 (react/full 기준):**
쉬운 편: dmv(21.7%) · library(13.6%) · healthcare(16.1%). 어려운 편: hotel(0.0%) · university(2.4%) · bank(5.2%). hotel/university는 극히 어려움 → transfer 목표로 삼으면 통계적 검증력 낮을 수 있음. **주 전이 대상 = dmv·library·online_market (중간 난이도, 변동 여지).**

**5. 상한(oracle) 도메인별 분포 메모:**
react/oracle: dmv 62.9% · library 47.0% · online_market 43.0% → 이 도메인들은 oracle에서 의미있는 성능, full에서 크게 떨어짐 = distractor 제거가 큰 효과를 보이는 도메인 = 우리 방법의 기여가 가장 선명히 드러날 곳.

### bank 추가 분석 (should_succeed 분해)

| 설정 | overall | should_succeed=True (n=48) | should_succeed=False (n=86) |
|---|--:|--:|--:|
| 저자 ReAct/oracle | 58.2% | 31.2% | 73.3% |
| 우리 FC/oracle | 55.2% | 45.8% | 60.5% |

- should_succeed=False(거부 케이스) 쪽이 높음 = 모델이 소극적으로 행동하면 거부 케이스를 "우연히" 통과 가능. 그러나 True/False 양쪽 다 실질 성능(0이 아님) → trivial-refusal 게이밍 아님.
- True(실제 실행 필요) 케이스의 성능이 낮음 = 실제 task 수행이 어려운 주 원인. arm-3/4의 향상이 이 쪽에서 나와야 의미 있음.

---

## Exp-0 — arm-0 oracle ceiling (예정, 2026-06~)

### 가설
- HC1: 결정론 executor(GT call-graph walk)는 7도메인 oracle tool list에서 ~97% pass@1(12-domain ABox 자산에서 검증).
- HC2: full tool list에서도 높음 (distractor에서 올바른 도구만 부르면 됨 → executor는 GT action graph 가짐, 사실상 100% 근접).
- HC3: ⚠️ **should_succeed=False 처리가 비자명**: GT call-graph 성공경로가 있어도 제약 위반 케이스엔 "실행하지 않음"이 정답 → 결정론 executor가 이를 어떻게 처리하는지 신중 구현 필요.

### 실행 계획
- `workflow_executor.py` + `abox/bank` → SOPBench custom agent 인터페이스 연결.
- **full tool list** 표준 + oracle 병행. should_succeed=False 처리 명시적 구현 후 진행.
- 목적: **arm-1과의 gap 고정 (상한-baseline)** = 구조로 메울 수 있는 최대 공간 정량화.

---

## Exp-2/3 — arm-2(L0) + arm-3(L1) 2-stage agent (예정, 2026-06~)

### 가설
- HL1: arm-3(LLM + operator structure + global-plan re-plan)이 arm-1(LLM-alone) 대비 **full tool list에서 pass@1 향상**. 특히 oracle−full 격차가 큰 도메인(dmv·library·online_market).
- HL2: arm-2(symbolic L0, operator만)는 arm-3보다 낮지만 arm-1보다 높음 → "operator affordance 자체가 일부 정보".
- HL3: full tool list에서의 향상 > oracle에서의 향상 → 구조가 tool-selection 병목을 직접 완화.

### 실행 계획 (2-stage agent 빌드 — Track A)
- **plug-in**: `Agent.client`를 planner→resolver로 교체(검증된 `SOPBENCH_VLLM_BASE_URL` 패치 위).
- **planner(L1)**: goal + abstract operator affordance(name+precondition type, 구체 schema 안 봄) + global plan + history → 추상 step. GoalAct식 매턴 re-plan.
- **resolver(b)**: 추상 step + 도메인 ABox(구체 schema, arg source) → concrete tool call.
- bank 파일럿 first: arm-1(react/full 5.2%) vs arm-3(L1, full) → 첫 신호.
- 성공기준: **arm-3 full > arm-1 full by ≥ 5%p on bank** → scale to 7 domains.
- ⚠️ 주의: planner가 concrete tool schema 보면 안 됨(전이 오염).

### ✅ Exp-3 결과 — arm-3 (L1) bank N=134, Qwen-7B, fc/full (2026-06-01, Track A)

> **파이프라인 BLOCKING 수정 완료·검증.** 구 `run_two_stage.py`의 인라인 eval(tuple/포맷 불일치)을
> 폐기하고, 저자 `run_simulation.py`에 `--two_stage` 플래그(assistant client만 `TwoStageClient`로 교체)
> 추가 → 표준 `run_evaluation.py` **무수정** 재사용. 배포 = `apply_two_stage_patch.py <clone>`.
> smoke 5 + bank 134 전부 저자 evaluator로 채점됨(포맷·합성객체 호환 검증). deterministic shortcut은
> **opt-in 기본 off**(`--two_stage_det`) → 본 수치는 **순수 L1**(planner + LLM resolver).

| arm | mode/tool | bank pass@1 | vs arm-1 |
|---|---|--:|--:|
| arm-1 (LLM-alone) | fc/full | 3.7% (5/134) | — |
| arm-1 (LLM-alone) | react/full | 5.2% (7/134) | — |
| **arm-3 (L1, naive planner)** | **fc/full** | **0.0% (0/133)** | **−3.7%p (HURTS)** |

**오류 분해 (arm-3, 133 실패 중):** dirgraph_violations **122**(92%) · constraint_violations **116**(87%) ·
database_mismatches 103 · incorrect_action_calls 72 · tool_call_errors 14.

**해석 (HL1 기각, 현 구현 기준):**
1. **지배적 실패 = 제약 순서 위반**(constraint/dirgraph ~90%), tool-selection 아님. agent가 타깃 액션
   (예: `apply_credit_card`)을 **SOP 선행제약 검증(`internal_check_*`/`internal_get_database`) 없이 즉시 호출** →
   `directed_action_graph` 위반. (궤적 검수로 확인: 타깃 호출 후 read-loop 진입.)
2. **순진한 L1 planner는 이 병목을 못 건드린다.** planner 프롬프트가 operator **이름+desc[:120]**와
   goal 첫 400자만 받음 → **의존성/제약 그래프(어느 operator가 무엇을 선행해야 하는지)를 전혀 모름.**
   arm-1은 같은 SOP를 system prompt로 다 보고도 5%이고, arm-3는 그 구조를 **버려서** 0%.
3. **매턴 강제 도구호출의 부작용**: planner가 항상 도구를 고르고 resolver가 `tool_choice`로 강제 →
   모델이 (a)우아하게 멈추거나 (b)명확화 질문을 할 수 없음 → **read-loop + 인자 환각**(resolver가
   `username`에 문장 전체 주입 → max_tokens 절단 → JSON 깨짐, 6태스크 retry 소진). = telecom read-loop
   메모리와 동일 패턴.
4. **⇒ 결론: arm-3-naive는 음성. 그러나 이것이 구조적 planner의 필요를 깨끗이 증명한다.** 다음 iteration =
   planner에 **call-graph/의존성 관계(§2 8-relation 온톨로지: precondition·step_realizes·observation_triggers)**
   를 주입 + **종료/비호출 허용**. arm-3-naive는 그 구조판의 **clean baseline**.

**다음(arm-3 v2 설계 — 사용자 설계리뷰 대상):**
- (a) planner 입력에 **SOP 의존성 그래프** 추가(어느 operator가 어느 선행검증을 요구하는지; concrete param
  schema는 여전히 숨김 = 전이가드 유지). (b) planner가 **선행제약 미충족 시 타깃 액션 금지**(gate). (c)
  planner가 **exit/no-op 선택 허용**(read-loop·강제호출 차단). (d) resolver 인자 환각 가드(슬롯-우선·길이제한).
- 코워커(32B/72B)는 우선 **arm-3-naive를 그대로** 돌려 **모델크기 × 구조 상호작용** 측정(강한 모델이 강제호출
  패널티를 흡수하는지) → 구조판 v2의 비교 baseline 확보.

---

## Exp-4 — arm-4 L2 학습 planner + 전이 (예정, 2026-06~, coworker 주도)

### 가설 (Phase 1b, 헤드라인)
- HT1: N-1 도메인 학습 후 held-out 도메인에 ABox만 swap → pass@1이 in-domain의 ≥70%.
- HT2: ABox-ablation(빈/틀린 operator) → 성능 붕괴. "플래너가 메모리를 실제로 읽음" 증명.
- HT3: L0 < L1 < L2 ladder.

### 실행 계획 (coworker B1*/B2*/B5*)
- 학습데이터: success 궤적 → (goal, operators, state) → next-operator 라벨.
- 7도메인 LODO: 6개 학습 → held-out ABox swap → rule pass-rate. ABox-ablation 포함.
- oracle 진단(coverage%) 병행: resolver가 LLM 없이 푼 비율.
- **주 전이 도메인**: dmv·library·online_market (중간 난이도, 통계적 검증력 충분).

---

## 결과 요약 대시보드 (갱신 예정)

| Exp | arm | 모드/tool | bank | 7D avg | 상태 |
|---|---|---|---|---|---|
| Exp-1 | arm-1(LLM-alone) | react/full | 5.2% | 9.5% | ✅ 완료 |
| Exp-1 | arm-1 | fc/full | 3.7% | 7.3% | ✅ 완료 |
| Exp-1 | arm-1 | react/oracle | 59.0% | 43.1% | ✅ 완료 (진단) |
| Exp-1 | arm-1 | fc/oracle | 55.2% | 28.2% | ✅ 완료 (진단) |
| Exp-0 | arm-0(ceiling) | oracle+full | — | — | 예정 |
| Exp-2 | arm-2(L0 symbolic) | react/full | — | — | 예정 |
| Exp-3 | arm-3(L1 naive planner) | fc/full | **0.0%** | — | ✅ 완료 (음성: arm-1 3.7%↓, 제약위반 지배 → 구조판 v2 동기화) |
| Exp-4 | arm-4(L2 LODO) | react/full | — | — | 예정 (coworker) |

---

## 인프라 메모

- SOPBench clone: `/home/woori/scratch/SOPBench`
- 저자 공식 결과 파일: `output/<domain>/` — 7도메인 다수 모델 존재, 직접 비교/재현 가능.
- 우리 baseline 결과: `output_overnight/<domain>/`
- 패치 2종(*.py.bak 백업): (a) `llm_handler._init_vllm` → `SOPBENCH_VLLM_BASE_URL` 환경변수로 사전서빙 endpoint 사용, (b) `constants.FUNCTION_CALLING_MODELS["vllm"]`에 qwen/llama 등록.
- 실행 env: `seka_env`(py3.12), colorama·termcolor·anthropic 추가 설치.
- vLLM 서버: GPU0:9100 / GPU1:9000 (Qwen2.5-7B-Instruct, hermes parser, nohup).
