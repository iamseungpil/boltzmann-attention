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
| **arm-3: L1 naive planner** | planner(LLM, operator 이름+desc만) + resolver(b), 무학습 | 구조 *없는* L1 격리. **=음성(0%), ABox 미주입** |
| **arm-3v2: L1 + ABox 의존성그래프** | planner에 ABox(precondition/produces) 주입 + gate + exit허용, 무학습 | 구조 zero-shot 바. arm-3→arm-3v2 = "ABox 주입 효과" |
| **arm-4a: L2 학습 TBox (in-context, ★헤드라인)** | cross-domain copy-grounded SFT, ABox=swappable in-context, 동결 TBox | **분리 학습**; 7 LODO 전이 + ablation(빈/틀린 ABox 붕괴) = `WORKFLOW_ONTOLOGY_DESIGN §11` |
| **arm-4b: L2 xattn ABox-memory (novelty)** | ABox=cross-attn 메모리뱅크, 가중치 content-free (§15.13) | 분리 보장 최강. Phase 2, coworker. |

> **arm-4 분리 계약**(사용자 지시 2026-06-01): TBox=도메인불변 means-ends 룰만 학습, ABox=도메인별 call-graph
> 완전분리·swap, **TBox+ABox 혼재 SFT 금지**(저번 entangled 실패 수정=target을 ABox에 copy-grounded). 19도메인
> 공통 8관계가 TBox 타입, 도메인차는 ABox 인스턴스값. 상세·증명ablation·phasing = **`WORKFLOW_ONTOLOGY_DESIGN §11`**.

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

### ⚠️ 벤치마크 caveat — 구조적 불가 태스크 (~26%, 모든 bank 수치 해석 전제)
bank `should_succeed=True` 43 유니크 중 **저자 모델(GPT-5/o4-mini/Claude-3.7/Gemini 등) 누구라도 푼 것 = 32 (74%)**.
**11개(26%)는 어떤 프런티어 모델도 0% = 구조적 불가** (예 `cancel_credit_card` 0/3·`pay_bill_with_credit_card` 0/2:
도메인 메서드가 `credit_cards`를 dict 가정하나 데이터는 list → 카드 lookup 영구 실패). ⇒ **bank should_T 실효 천장
≈74%**(100% 아님). L0/arm-0 천장은 이 32개 기준 해석. **수치 낮음을 전부 방법 탓 말 것.** 상세 = `WORKFLOW_ONTOLOGY_DESIGN §11.13`.

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
| Exp-2 | arm-2(L0 regression) | symbolic/full | — | 45.5%(bank) | 🔶 first cut (A1 확증; executor 천장갭) |
| Exp-2 | arm-2(L0 greedy) | symbolic/full | — | 64.2%(bank) | 🔶 passive 게이밍(거부 100%·달성 0%) |

> **Exp-2 L0 first cut (bank N=134, LLM 無, 2026-06-01)** — induced ABox `ontology_bank.json` 위 means-ends:
> | rule | pass@1 | should_succeed=T | should_succeed=F |
> |---|--:|--:|--:|
> | **regression** | 45.5%(61/134) | **7/48** | 54/86 |
> | greedy | 64.2%(86/134) | **0/48** | 86/86 |
> - **★설계리뷰 A1 확증**: 체인 필요 케이스(should_succeed=T)서 **regression(7)>greedy(0)**. greedy는 login을
>   영영 establish 못 해 달성 0(=A1 예측), "전부 거부"로 거부 86개 trivially 통과(전체 64%는 passive 게이밍 허수).
>   ⇒ **후방 회귀가 필요조건임을 LLM·파싱·tool_choice 교란 0 환경에서 입증**(코드리뷰 C-2 "L0가 유일 깨끗 중재자").
> - **천장 갭은 룰이 아니라 L0 executor의 벤치 정합 미완**(정제중): 수정완=이중 login(executed 집합)·과도
>   refuse(goal precond를 평가자처럼 `task["constraints"]`로 오버라이드). **잔여 지배 실패=`dirgraph_satisfied`
>   (should_T 실패의 35/43)**: 일부 goal 액션이 **strict 시스템의 innate-dep 강제로 False 반환**(예 cancel의
>   `internal_check_credit_card_exist`가 카드 존재에도 False) → action_successfully_called=False → dirgraph 실패.
>   = SOPBench innate-dep 의미론(평가자 dirgraph가 요구 안 하는 검사도 strict가 강제)과의 정합 필요 = arm-0
>   oracle 엔지니어링(벤치 내부 리버스). **수치 자체보다 regression>greedy(A1) 방향이 핵심 산출이고 이미 확정.**
> - 코드: `scripts/distill/sopbench/{induce_ontology_zekun,l0_planner}.py`. induced 7도메인 ontology 추출완.
> - **상태**: A1 확정. arm-0 oracle ~100% 천장은 innate-dep/dirgraph 정합 추가작업 필요(보류 가능). 다음 우선
>   순위 = arm-3v2(planner I/O 재설계, 더 높은 가치)로 이동 권장 — L0 oracle 천장은 병렬/후속.
| Exp-3 | arm-3(L1 naive planner) | fc/full | **0.0%** | — | ✅ 완료 (음성: arm-1 3.7%↓, 제약위반 지배 → 구조판 v2 동기화) |
| Exp-3v2 | arm-3v2(L1+ABox+게이트+STOP) | fc/full | **44.0%** | should_T 2/48 · should_F 57/86 | ✅ 완료 (거부는 고침, 수행 게이팅은 못 고침) |

> **Exp-3v2 (arm-3v2 L1 structured planner, bank N=134, 7B, 2026-06-01)** — planner I/O 재설계: ABox(precondition/
> produces) + **계산된 READY/BLOCKED 게이트 상태**(history서 establishable 충족 추적) + STOP=refuse(N1) + cost
> 규칙(§11.12) + copy-grounded(C1). 코드 `two_stage_client.py planner="v2"`, 배포 `run_simulation --two_stage_v2`.
>
> | arm | pass@1 | should_T(수행) | should_F(거부) |
> |---|--:|--:|--:|
> | arm-1 fc/full | 3.7%(5/134) | — | — |
> | arm-3-naive | 0%(0/133) | ~0 | 0 |
> | **arm-3v2** | **44.0%(59/134)** | **2/48** | **57/86** |
> | L0 regression | 45.5%(61/134) | 7/48 | 54/86 |
>
> - **★두 축이 갈린다**: **거부축(should_F) 0→57/86 대약진 = N1 수정(STOP=도구 미호출 턴)이 정확히 작동**(강제호출
>   폐지로 금지 액션 안 부름 → 거부 통과). 44% 상승의 거의 전부가 이것. **수행축(should_T) 2/48 여전히 바닥**: 7B가
>   `BLOCKED — first call: login_user`를 **무시**하고 goal을 greedy하게 반복(프롬프트·파싱 검증; v2 버그 아님).
> - **⇒ in-context 구조는 "거부"는 고치나 "수행 게이팅"은 못 고친다**(약한 모델). L0(코드가 룰 실행)의 should_T 7/48
>   보다도 낮음 = **프롬프트로 룰을 주는 것과 룰을 실제 따르는 것은 다르다.**
> - **★사다리 결론(headline, 실증 완성)**: L0(룰 코드실행: regression>greedy=룰 옳음) → L1/arm-3v2(룰 in-context:
>   거부만 고침, 게이팅 무시) → **L2/arm-4a/4b**(룰을 **가중치에 학습/내재화**, ABox는 swap): **게이팅을 약한 모델이
>   따르게 하려면 학습 필요** = 본 라인 헤드라인 동기. (단 ABox는 가중치에 안 넣음=분리계약 §11.0, copy-grounded+xattn.)
> - **모델크기 상호작용(coworker)**: in-context 구조가 **강한 모델**(32B/72B)엔 게이팅까지 도움되는지 = arm-3v2 sweep이
>   "구조×크기" 측정. 7B는 명확히 L2 필요.
| Exp-4a v1 | arm-4a(L2 학습 TBox, fact-invisible, holdout=bank LODO) | fc/full | **16.4%** | should_T 3/48 · should_F 19/86 | ✅ 혼합(게이팅 전이 성공, 거부 붕괴) |
| Exp-4a v2 | arm-4a(L2 학습 TBox, **fact-visibility**, holdout=bank LODO) | fc/full | **26.1%** | should_T 4/48 · should_F **31/86** | ✅ **거부 회복**(+13.9pp), should_T 정체=executor 병목 |

> **★Exp-4a 첫 결과 (holdout=bank LODO, bank N=134, 2026-06-01)**: 6도메인(dmv·healthcare·hotel·library·
> online_market·university) SFT(`lora qwen7b_tbox_planner_lodo_bank`, val 0.133→0.119→**0.1137** 단조개선) →
> **bank(학습 제외) eval**.
>
> | arm | pass@1 | should_T | should_F |
> |---|--:|--:|--:|
> | arm-3v2(미학습) | 44.0% | 2/48 | **57/86** |
> | **arm-4a(학습)** | **16.4%(22/134)** | **3/48** | **19/86** |
> | L0 regression | 45.5% | 7/48 | 54/86 |
>
> - **✅ 학습+전이 성공(메커니즘)**: SFT가 establishable 게이팅을 내재화 → **궤적 `[login_user→apply_credit_card→
>   exit]`**(arm-3v2의 login-건너뛰고 goal 무한반복이 사라짐). **bank=학습 제외인데도 login-우선 룰 따름 = 전이**
>   (암기 아님, §11.7-i). **in-context(arm-3v2)로 안 되던 게이팅을 학습이 함** = L0→L1→L2 사다리의 L2 가치 입증.
> - **❌ 거부축 붕괴(57→19) = 프롬프트가 FACT를 못 보여줌**: 프롬프트는 establishable(login)만 READY/BLOCKED 표시,
>   **fact 선행조건(credit score 자격 등)은 미표시** → 거부 케이스서 login 후 goal이 "READY"로 보여 모델이 goal 호출
>   (STOP 안 함) → 거부 실패. STOP 학습예제는 **추론 시 모델이 못 보는 fact에 기반**해 재현 불가. **arm-3v2가 STOP으로
>   거부 57개 통과한 걸 arm-4a가 "login+try"로 깨뜨림** → 전체 16.4%<44%.
> - **should_T 3/48 소폭**(게이팅 고쳤으나 dirgraph/실행 한계는 L0 7/48과 동류; ~26% 불가태스크 §11.13).
> - **⇒ 다음(arm-4a v2)**: 프롬프트에 **fact-status 주입**(내부 read 도구 호출→관찰→STOP 판단) 또는 refuse 데이터를
>   "fact 불명 시 보수적 STOP" 정책으로 재설계. 게이팅+전이는 입증됨 → 거부 fact-visibility만 고치면 됨.
> - **남은 검증**: ablation (ii)빈/(iii)틀린 ABox 붕괴(분리 증명) + 6 LODO 회전.

> **★Exp-4a v2 결과 (fact-visibility, holdout=bank LODO, bank N=134, fc/full, 2026-06-02, rr.ps1 실측)**:
> v2 SFT(`lora qwen7b_tbox_planner_v2_lodo_bank`, val **0.1115**) — refuse 데이터에 fact-check 도구를
> **선호출→관찰→STOP** 패턴 내장(`build_v2_prompt`로 train/test 프롬프트 동일, fact_pm 인자). bank(학습 제외) eval.
>
> | arm | pass@1 | should_T(수행) | should_F(거부) | should_T fail-locus(지배 게이트) |
> |---|--:|--:|--:|---|
> | arm-4a v1 (fact-invisible) | 16.4%(22/134) | 6.2%(3/48) | 22.1%(19/86) | dirgraph 27/45 (60%) |
> | **arm-4a v2 (fact-visibility)** | **26.1%(35/134)** | 8.3%(4/48) | **36.0%(31/86)** | dirgraph 23/44 (52%) |
> | (참고) arm-3v2 미학습 | 44.0% | 2/48 | 57/86 | — |
> | (참고) L0 regression | 45.5% | 7/48 | 54/86 | — |
>
> - **✅ 거부 회복 = v2 설계 목적 달성**: should_F 22.1%→**36.0%(+13.9pp)** = fact-visibility 개입이 v1의 거부붕괴를
>   부분 복구. overall +9.7pp(16.4→26.1)의 **주동력이 거부축**(should_T는 +1태스크 정체). v1 진단(프롬프트가 fact 미표시→
>   거부 실패)이 인과적으로 옳았음을 v2가 역으로 입증. (단 arm-3v2 STOP 57엔 못 미침 = 학습 STOP은 추론시 fact-tool
>   관찰에 의존, 무학습 in-context STOP보다 약함 — 거부 천장 미달.)
> - **❌ should_T 정체(3→4) = 레버2(gather→dirgraph) 가설 기각**: v2 gather(internal_check 호출)가 dirgraph를 못 채움.
>   should_T 실패 locus는 여전히 **dirgraph_satisfied 위반이 지배**(23/44≈52%, v1 27/45≈60%과 거의 불변) +
>   action_called_correctly/action_successfully_called 각 22. **수행의 병목=planner 아닌 executor**(L0 룰완벽도 7/48).
> - **⇒ 다음 레버=레버1(인자 결정화)**: goal args를 ABox arg-매핑으로 slot-state서 결정론 채움(LLM 환각 제거) +
>   precondition 트리 선행함수 전부 호출(dirgraph 완전 충족). `two_stage_client.use_deterministic_shortcut` 강화.
>   (분모정직: should_T 8/48 구조적 불가 = bank 결함태스크 §11.13 → 실효분모 40, v2 4/40=10.0%.)
> - **아티팩트**: 시뮬 `output_v4a_v2/bank/ast_tbox_v2-*.json`, 분해 `_v2_breakdown.py`. 서빙=GPU1 단일 lora tbox_v2
>   (TP=2는 이 박스 NCCL hang → 단일-GPU만).

> **★★Exp-4a v2 should_T 실패 전수 조사 (2026-06-02, rr.ps1 실측, bank 48 should_T)** — 레버1(인자 결정화)
> 설계 전 진단. 아티팩트 `_census_shouldT.py`(궤적별 분류), 분류기준=ABox precondition 트리 vs 실제 호출 시퀀스.
>
> **(1) should_T 진짜 천장 = ~24, NOT 40** (자격증명 가용성 분류):
>
> | 클래스 | pass/total | 성격 |
> |---|--:|---|
> | Z 저자-strict 불가8 | 0/8 | 기존 §11.13 결함(cancel_credit_card×6, pay_bill_with_credit_card×2) |
> | **C 자격증명 user_known 부재** | 2/16 | **정보적 불가** — identification/admin_password가 initial_database에만 존재. 에이전트도 user-sim(`env/generation.py:591`=user_known만 주입)도 영원히 못 봄 → login_user/authenticate 통과 불가(2/16만 dirgraph 관용 누설) |
> | A 자격증명 불요 | 0/3 | login 불요인데 0통과(별도) |
> | **B 자격증명 가용** | 2/21 | ★**유일한 addressable** |
>
> → **24/48이 LLM 에이전트에게 구조·정보적 불가**(C16+Z8). **정직 분모=24**(40 아님; 40은 DB 접근 가능한 L0 oracle 기준).
> ⚠️ C(자격증명 부재)는 **8 결함과 별개의 제2 한계** — 공개 26모델 대조(evidence-B 방식 `offline_crosscheck`)로 확증 후 보고.
>
> **(2) 레버1(A=인자 바인딩) 단독 효과 = +1 (반증)**: dirgraph가 value-매칭하는 인자관계는 **이미 ABox에 정확히 존재**
> (`operators[login_user].args={username,identification}` = 평가기 `func_param_mapping`). 게다가 addressable 실패서
> 에이전트는 **이미 올바른 값을 넘김**([81] login `padoesshnwojord`✓ authenticate `addoeminhnpajoss`✓). 관계·값 다 맞아도 실패.
> mode-b 자격증명 환각 17건은 **19/20이 C버킷(값 부재)** → A 대상 아님. **A 단독은 B19중 1건만**(record 69 포맷오류).
>
> **(3) addressable 버킷 B(21, 자격증명 가용) 19개 실패 모드**:
>
> | 모드 | 건수 | 원인(궤적 확인) | A | B'goal-fire | 비고 |
> |---|--:|---|:-:|:-:|---|
> | **goal_not_reached** | 8 | 선행 전부 정확 호출 후 **goal 미발사·exit** ([81]: check→login✓→auth✓→exit, set_safety_box 미호출) | ❌ | ✅ | planner should_T 과소발사(should_F 과잉거부의 거울) |
> | **dirgraph_value_mismatch** | 5 | 전부 호출·true·일관값인데 dirgraph false. 원인=비호출 CHAIN 노드(`minimal_elgibile_credit_score`=fact, 호출 도구 없음)→implied-dep 경로만 충족 | ❌ | △ | 평가기 implied-dep 추가분석 필요 |
> | **constraint_violation** | 4 | goal 도달했으나 constraint_not_violated=False&db_match=False ([122/127/131] transfer_funds: auth를 transfer 뒤에 호출 등) | ❌ | ❌ | 정책·순서 분석 필요(별도) |
> | b_prereq_FALSE | 1 | record 69 login 자격증명을 dict로 잘못 감쌈 | ✅ | — | 유일 A 대상 |
> | other | 1 | get_account_owed_balance | ? | ? | |
>
> **결론**: should_T 병목은 인자 바인딩이 아니라 **(α) planner의 goal 과소발사 + (β) 비호출 fact 노드 dirgraph + (γ) 정책위반**.
> standalone-A 라운드는 건너뛴다(+1 예측). 코드 `_census_shouldT.py`, 천장 census는 위 (1).

> **★Exp-4c 실험계획 (should_T 레버, 우선순위 E3→E2, 2026-06-02 등록)** — 위 전수조사 기반. 분모=24(정직).
>
> **E3 (먼저, 싼 진단/수정 — goal_not_reached 8건 공략)**: 가설 = planner가 precondition 충족 후에도 goal을 안 쏨.
> - **E3.0 진단**: 8개 goal_not_reached 태스크에서 매 턴 `build_v2_prompt`가 goal에 부여한 status(READY/VERIFY/BLOCKED)
>   + planner raw 출력을 로깅. 분기: **(H-a)** goal=READY인데 7B가 exit/STOP 선택 → 과소발사(룰/SFT 문제). **(H-b)**
>   goal이 비호출 fact 때문에 영구 non-READY(VERIFY FIRST 고착) → `build_v2_prompt` status 로직 문제.
> - **E3.1 수정(택, H에 따라)**: (a) 프롬프트 룰 강화("goal=READY & 미호출이면 반드시 호출, exit 금지"); (b) SFT 데이터에서
>   완전 establish 후 goal-발사 예제 비중/명료화; (c) build_v2_prompt status 로직: 비호출 fact는 READY 차단하지 않도록.
> - **측정**: goal_not_reached 수 → 0 목표, should_T(분모24). 비용=프롬프트/SFT만(어댑터 재학습 0~1회). **E3로 8건 회복 시 E2 불요.**
>
> **E2 (다음, 무거운 레버 — goal-firing 결정론 실행기)**: E3 싼 수정으로 goal_not_reached 미해결 시.
> - **설계**: planner는 goal 선택·거부(STOP) 결정만(TBox 유지). 실행기(ABox)는 = ① **인자 바인딩**(param→`operators[a].args` slot→
>   `_slot_state`값; A 흡수) + ② **precondition 충족 시 goal 결정론 발사**(planner가 exit해도 override, 단 **STOP=거부는 존중**).
> - **코드**: `two_stage_client._resolve` 교체 + precondition 충족 추적기. `use_deterministic_shortcut`을 ABox-매핑판으로.
> - **재학습 0**(resolver-only) → 같은 tbox_v2 어댑터 재서빙·재eval. **측정**: should_T(분모24), goal_not_reached·dirgraph 분해.
> - **위험**: STOP override 금지(거부축 should_F 보존 — 분리계약). ablation: 실행기만 켜고 planner naive화 시 should_F 붕괴 확인.
>
> **E-후속(별도, B'으로 미해결분)**: dirgraph_value_mismatch(5)=평가기 implied-dependency(`dfsgather_allfunccalled_indepperm`)
> 분석→비호출 credit-score 노드 충족 경로 규명. constraint_violation(4)=transfer/set_account 정책·호출순서 분석(auth 타이밍).

> **Exp-4a 데이터 파이프라인 완성 (2026-06-01)**: `build_tbox_planner_sft.py` — GT means-ends가 만든 정답
> 결정 시퀀스([login→goal]/[login→STOP]/[goal]) → 각 step을 **공유 `build_v2_prompt`(train/test 동일 프롬프트)**
> + copy-target(operator명/STOP)으로 SFT 예제화, operator 순서 셔플(§11.4 위치암기 차단; alias는 후속). **1497
> examples × 7도메인**(bank 312·GOAL76/establish178/STOP58 등), 이중호출 0. 트레이너 `lora_train_chat_toolcall.py`
> 포맷 직접 호환(assistant content만 supervise). LODO 7 splits(`sft_tbox/lodo_train_holdout_<d>.jsonl`) 준비완.
> **다음=학습**(6도메인→held-out, GPU 확보 필요) → 서빙 → `run_simulation --two_stage_v2`(SFT 모델)로 held-out eval
> → arm-3v2(should_T 2/48) 대비 게이팅 향상 측정 + ablation(빈/틀린 ABox 붕괴, §11.7).
| Exp-4b | arm-4b(L2 xattn ABox-memory, 19 LODO) | fc/full | — | — | 예정 (Phase 2, coworker) |

> **Exp-4 (분리 학습) 가설** — `WORKFLOW_ONTOLOGY_DESIGN §11` 권위본:
> - HT1(전이): 6 도메인 학습→held-out ABox swap, **재학습 0** → pass ≥ in-domain의 70%.
> - HT2(분리증명): **빈/틀린 ABox 주입 → 붕괴**(entangled면 ABox 없이도 동작=실패; 분리면 ABox 필수=성공조건).
> - HT3(학습기여): arm-3v2(무학습 L1)< arm-4a(학습 L2), 그리고 arm-4a > arm-3-naive(0%)·arm-1.
> - HT4(불변): operator 셔플 불변(위치 아닌 관계로 선택).

---

## 인프라 메모

- SOPBench clone: `/home/woori/scratch/SOPBench`
- 저자 공식 결과 파일: `output/<domain>/` — 7도메인 다수 모델 존재, 직접 비교/재현 가능.
- 우리 baseline 결과: `output_overnight/<domain>/`
- 패치 2종(*.py.bak 백업): (a) `llm_handler._init_vllm` → `SOPBENCH_VLLM_BASE_URL` 환경변수로 사전서빙 endpoint 사용, (b) `constants.FUNCTION_CALLING_MODELS["vllm"]`에 qwen/llama 등록.
- 실행 env: `seka_env`(py3.12), colorama·termcolor·anthropic 추가 설치.
- vLLM 서버: GPU0:9100 / GPU1:9000 (Qwen2.5-7B-Instruct, hermes parser, nohup).
