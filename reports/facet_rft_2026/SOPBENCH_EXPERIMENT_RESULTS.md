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
> **정밀 갱신(2026-06-02, 아래 "bank leaderboard 실측" 블록 참조)**: instance 기준(48)으로는 불가 = **14개**(8 PartA 코드결함 + 6 PartB cred-부재). 정직 천장 = **34/48** (full·oracle 공통). 위 "11개/≈74%"는 unique(43) 기준 근사·구버전 — 정밀치는 아래 블록·`BUGREPORT Part A/B`.

### bank 추가 분석 (should_succeed 분해)

| 설정 | overall | should_succeed=True (n=48) | should_succeed=False (n=86) |
|---|--:|--:|--:|
| 저자 ReAct/oracle | 58.2% | 31.2% | 73.3% |
| 우리 FC/oracle | 55.2% | 45.8% | 60.5% |

- should_succeed=False(거부 케이스) 쪽이 높음 = 모델이 소극적으로 행동하면 거부 케이스를 "우연히" 통과 가능. 그러나 True/False 양쪽 다 실질 성능(0이 아님) → trivial-refusal 게이밍 아님.
- True(실제 실행 필요) 케이스의 성능이 낮음 = 실제 task 수행이 어려운 주 원인. arm-3/4의 향상이 이 쪽에서 나와야 의미 있음.

### ★ bank leaderboard 실측 (저자 released 59 files, 2026-06-02, `_lbmax`/`_lboracle`/`_ceil`) — 천장 권위 기록
overall pass@1 = (should_T+should_F)/134. should_T 별도 표기. **모드별 최고 + should_T 실현 최대치:**

| 모드 | overall 최대 (모델) | should_T 최대 (별도 모델) | should_F 최대 |
|---|---|---|---|
| **full** | **o4-mini-high 76.9% (103/134)** [sT 25·sF 78] | gemini-2.0-flash-thinking(react) **27/48** | 78/86 |
| **oracle** | **gpt-5 79.9% (107/134)** [sT 22·sF **85**] | llama3.1-70b(react) **31/48** | gpt-5 85/86 |

full 상위: o4-mini-high 76.9 · gemini-thinking 73.1 · gpt-5 71.6 · gpt-4.1 69.4 · claude-3.5 67.2.
oracle 상위: gpt-5 79.9 · gpt-4o 76.9 · qwen2.5-32b(react) 76.9 · llama3.1-70b(react) 75.4 · claude-3.5 75.4.

**★천장 해석 (정정·확정):**
- **should_T 실현 최대 = full 27 / oracle 31** — 둘 다 **40에 한참 못 미침**. 즉 **"40"은 에이전트(full·oracle 포함) 천장이 아니라** `evidence_a_probe`(전제 강제충족=DB에서 자격증명 주입하는 프로버)의 **"코드결함 아닌 task 수"**(48−8 PartA).
- **정직한 should_T 천장 ≈ 34** = 48 − 8(PartA 코드결함) − 6(PartB cred-부재, login 필수인데 자격증명 미제공·미노출). **full·oracle 공통**(oracle 도구모드는 자격증명을 주지 않으므로 6 PartB 동일 차단).
- oracle>full(실현 31 vs 27)은 **천장이 높아서가 아니라 도구선택 부담↓로 천장에 더 근접**. 14개(8+6)는 전 모델·전 모드 0 → should_T를 48→34로 누름.
- 레버(검증 게더) 타깃 = full에서 27→최대 34 헤드룸. gpt-5 oracle은 거부축(85/86)에 치중해 should_T 22로 낮음(전체 1위지만 should_T는 중위).

**★134 환산 (leaderboard 비교용, identity-matched union `_union.py`):**
- should_T 통과(≥1 모델)=**34/48**(never 14=전부 should_T), should_F=**86/86**(never 0=should_F 구조결함 없음).
- ⇒ **구조적 천장(union) = 120/134 ≈ 89.6%** (= 134 − 14 불가; 34 sT + 86 sF). "34/48 should_T"는 134 기준 **120/134**에 대응.
- **⚠️ union ≠ 단일모델**: should_T↔should_F 트레이드오프로 한 모델이 120 못 찍음. 실현 단일 최대 = **oracle gpt-5 107/134(79.9%)**, **full o4-mini-high 103/134(76.9%)**. 우리 arm-4a v2 = **35/134(26.1%)**(sT4+sF31).
- 134 그림: 우리 35 → SOTA 단일 103~107 → union 천장 120. 헤드룸 sT(4→34)·sF(31→86) 양쪽, 레버는 sT 겨냥.

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
| Exp-4a v2 | arm-4a(L2 학습 TBox, **fact-visibility**, holdout=bank LODO) | fc/full | **26.1%** | should_T 4/48 · should_F **31/86** | ✅ **거부 회복**(+13.9pp); should_T 정체=task-default 과잉게이팅(전수조사 정정, 천장 40 not 24) |

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

> **★★Exp-4a v2 should_T 실패 전수 조사 (2026-06-02, rr.ps1 실측, bank 48 should_T)** — should_T 레버 설계 전 진단.
> 아티팩트 `census_shouldT.py`·`leaderboard_bankcheck.py`·`evidence_a_probe.py`. 설계 후속=`scripts/distill/TASK_CONSTRAINT_DESIGN.md`.
>
> ⚠️ **정정(2026-06-02): 본 조사의 1차 결론 "천장 24 / 자격증명-부재 16개 불가"는 오류였음.** census가 자격증명 요구를
> **task별 실제 제약이 아니라 domain-default precondition(`dep_full`)** 으로 판정해 과대분류했음. 아래는 정정본.
>
> **(1) should_T 진짜 천장 = 40/48** (strict-oracle 직접검사 `evidence_a_probe.py` 재실행, 2026-06-02):
> 오라클이 못 푸는 건 **8개뿐**(cancel_credit_card×6 `goal_return=False`; pay_bill_with_credit_card×2 `KeyError:'credit_limit'`;
> 원인 §11.13 credit_cards list/dict). **나머지 40은 오라클 통과** → 정직 분모=**40**. (이전 "24"는 폐기.)
>
> **(2) "자격증명 부재 16개 불가" = bucketing 오류.** 52개 공개 모델 task별 통과수(`leaderboard_bankcheck.py`)로 검증:
> - 그 16개 중 **8개는 19~30모델이 통과**(task 0/2/28/78/98/111/115/124) — 통과 가능. (예: task 111 transfer_funds는 실제
>   제약이 `internal_check_username_exist`만 → gpt-4o가 login 없이 통과; task 78은 user_known에 identification 있음.)
> - **6개**(39/44 get_loan·56 pay_bill·76/89 set_safety_box·120 transfer_funds)는 전원 미통과지만 **오라클은 통과**
>   = 극難(자격증명이 에이전트엔 부재, 오라클만 DB접근)·**결함 아님**(§11 evidence-B와 동일 판정). 2개(66/67) 경계.
> - ⇒ 자격증명-부재로 실제 못 푸는 건 ~6개(현실 상한 ~34), 결함은 8개. **16개 불가 주장은 철회.**
>
> **(3) leaderboard >50%의 이유(정합)**: 상위권은 거의 전부 **oracle 도구모드**(gpt-4o oracle 62% vs full 35%; llama70b-react-oracle 65%).
> 대부분 task는 login 불요(task별 제약이 username 체크만), login 필요 시 자격증명이 user_known에 있음, fact는 호출가능 도구
> (`internal_get_credit_score`·`get_account_balance`)로 검증. 우리(full·7B)가 낮은 건 §4 병목 때문.
>
> **(4) ★진짜 should_T 병목 (코드-증명, `TASK_CONSTRAINT_DESIGN.md §2`)**: 인자 바인딩 아님.
> `build_v2_prompt`가 goal precondition을 **domain-default(무거움)** 로 렌더 → login 불필요 task에도 "login 먼저"
> 지시 → 7B가 불필요 login/auth를 **환각 자격증명**으로 호출(`identification='password'`) → 스스로 dirgraph 위반.
> + **full-tool 모드**(도구 ~20개)의 선택 부담. 공통뿌리=**WHAT(적용 제약)을 task별 아닌 default로 다룸**.
> (이전 표의 goal_not_reached/dirgraph_value_mismatch 분류는 default-precond 기준이라 부분 distort; 정밀 재분류는 설계서 실험에서.)

> **★Exp-4c — 설계서 `TASK_CONSTRAINT_DESIGN.md`(리뷰 1라운드+zero-train 게이트 완료). 분모 /48 주·/40 보조.**
> 해법 가설=task-instance 제약을 per-task ABox로: (A) precond status를 task 제약으로 렌더(과잉게이팅 제거)+(B) 도구 프루닝.
>
> **★zero-train 게이트 결과 (2026-06-02, `_lighten_compare.py`, env SOPBENCH_LIGHTEN, 재학습0)**:
> | 지표 | baseline v2 | LIGHTEN | 게이트 |
> |---|--:|--:|:--:|
> | login/auth 호출(A_HELPS 14) | 17 | **7(-59%)** | (i)✅ |
> | should_T | 4/48 | **4/48** | (ii)❌ |
> | should_F | 31/86 | 31/86 (fragile 1 회귀) | (iii)✅ |
>
> - **mechanism A 라이브 작동 확인**(유닛테스트 task111 OFF=BLOCKED-login/ON=VERIFY-internal_check DIFFERENT; 행동 login 183→151). no-op 아님.
> - **게이트 미통과((ii) should_T 불변) → 재학습 보류.** login 과잉호출은 실재했으나 **should_T binding constraint 아님**(남은 실패=destination 체크 누락·constraint_violation·자격증명부재 극難=비-login). R2 비대칭(OOD 어댑터)이라 null은 비결론.
> - **다음 = 재학습 아님 → should_T binding-constraint 진단**(task 제약 기준 재census). should_F 회귀 모니터=14건(위험5+미정9, R3).
> - 코드: `two_stage_client.build_v2_prompt(goal_constraint=)`·`apply_two_stage_patch`(reset 배선)·`gates_p2p3.py`·`_lighten_compare.py`. 리뷰=`TASK_CONSTRAINT_DESIGN_REVIEW.md`.

> **★★★Exp-4c 확정 결론 (2026-06-02, harness sim `run_scripted.py` + leaderboard 재확인 `_leaderboard_bankcheck`) — 이 블록이 권위본. 아래는 재유도 금지(반복 방지).**
>
> **should_T binding = 도구 선택(어떤 검증/establish 도구를 호출하는가), NOT 정보부재·NOT 게이팅·NOT 모델용량·NOT login제거.**
> 결정론 scripted-gather(LLM無)를 실제 evaluator로 채점:
> | gather | should_T | 비고 |
> |---|--:|---|
> | 실제 7B baseline | 4/48 | 현재 |
> | ab (A args-aware체크 + B condition→getter) | 24/48 | login 없음 |
> | **abc (A+B + C establishable login/auth)** | **37/48 = scripted-oracle** | 완전 게더 |
> | scripted-oracle plan | 37/48 | =실질 오라클 40(아래 gap3) |
>
> - **레버 = goal의 induced precondition 완전 충족**: A(args-aware 콜러블 체크, 예 transfer dest 2회) + B(condition 술어→getter; `induce_ontology_zekun`가 `by:null`로 둔 결함 보강: minimal_credit_score→internal_get_credit_score, sufficient_account_balance→get_account_balance, …) + C(establishable login/auth, task-conditional). root cause=GT teacher `build_tbox_planner_sft.py:108` `goal_fact_checkable`가 name-dedup·establishable제외·동명callable한정으로 **검증 시퀀스 불완전 생성** → 모델이 "덜 검증" 학습 → dirgraph 위반.
> - **C(login/auth)는 task-conditional**: should_T 48 중 dirgraph상 login 필요 41/불요 7. C 출처=**induced precond establishable**(dep_innate은 null이라 못 잡음). evaluator dirgraph는 task제약 아닌 default deps를 따름.
> - **mechanism A(게이팅 경량화)·zero-train(LIGHTEN) 폐기**: login은 제거 대상이 아니라 (자격증명 가용 시)올바르게 수행 대상. zero-train null은 binding이 게이팅 아님을 보였을 뿐.
>
> **★16개 "cred-absent"(이전 crude bucket)의 정확한 분해 — leaderboard(저자 59 released model files) 재확인:**
> - **8개 = 도구선택(정보 가용, 레버로 극복)**: apply_credit_card(idx0:full-pass13,idx2:16)·deposit_funds(28:17)·set_safety_box(78:10,98:7)·transfer_funds(111:10,115:11,124:13). released 모델 다수가 **full 모드에서 통과**. 성공궤적=`internal_check→internal_get_credit_score→goal` / `login→authenticate→internal_get_credit_score→get_account_balance→set_safety_box`. ⇒ getter 호출이 열쇠.
> - **8개 = 진짜 자격증명-부재(정직 극복 불가)**: get_loan(39,44)·pay_bill(56)·pay_loan(66,67)·set_safety_box(76,89)·transfer_funds(120). identification(+admin_password)가 **user_known에 부재** → **0~1/42 모델만 통과(oracle 도구모드 포함)**. oracle 궤적=`check→login→exit`(포기) 또는 goal시도→dirgraph False. =메모리 극難6(39,44,56,76,89,120)+경계2(66,67).
>   - **★admin DB-read 가능성 검증(2026-06-02, `_admincheck.py`)**: creds는 **DB에 존재**하고 `internal_get_database()`가 반환함(identification·admin_password 전부). 8개 dirgraph에 internal_get_database 노드 **있음**. **그러나 internal_get_database는 에이전트 도구목록에 미노출(full·oracle 둘 다)** — 노출된 internal_*는 check/score용뿐, 자격증명-반환 도구 0개. ⇒ evidence_a_probe/abc-augmented의 통과는 `dss.internal_get_database()` 직접호출(에이전트 불가 시스템 메서드)=cheat 확정. **주어진 도구셋에선 정직 극복 불가.**
>   - **★★신규 벤치 결함 후보 = 6개(8 아님; Part B)**: `_defectchk`/`_intent8`/`_lbident` 실측. **task identity 매칭(index 아님 — released 파일들이 task 순서 불공유라 index 대조 무효)** 로 재검증. 6개(get_loan 39,44·pay_bill 56·set_safety_box 76,89·transfer 120) = **task 의도=username-only**(user_instruction "using your username to identify yourself", constraints_original에 login 0)인데 **directed_action_graph는 login 요구** → login 없는 제약-충실 궤적은 goal 성공(asc·db_match True)이나 **dirgraph_satisfied=False∧constraint_not_violated=False**→불가. creds 미제공·미노출. **43개 매칭 released run(full-static37+oracle-static16+usersim-full6) 전원 0 통과**(oracle+usersim 조합은 released에 없음). = 내부 불일치 결함. **⚠️pay_loan 66,67은 제외**(통과됨: 66=qwen-7b oracle 1/44, 67=gemini full+qwen oracle 2/43; pay_loan 제약에 no-login or-분기). **이전 "merely hard, not defect" 정정**: 그 근거(oracle통과)는 evidence_a_probe cred-injection(에이전트 불가)이었음. 보고초안=`BUGREPORT_..._impossible_tasks.md` Part B(제출 전 dup검색+저자에 login의도 확인 권장). 정직 천장 ≈**34/48**(48−8 PartA−6 PartB).
> - **⚠️ 이전 "realistic=21 / 16 전부 refusal" 철회**: 16을 lump한 오류. realistic 모드(account aug 끔)가 rigid scripted+과대strip으로 16 drop했으나, 실제는 8(도구선택)+8(cred-부재).
>
> **정직한 천장 = ~32/48** (48 − 8 결함(cancel_cc×6+pay_bill_cc×2) − 8 자격증명부재). 레버 타깃=이 32(baseline 4). gap3(scripted oracle 37 vs evidence_a_probe 40)=값-반환 goal(exchange_foreign_currency·get_account_owed_balance) **run_scripted 아티팩트, 결함 아님**(evidence_a_probe 통과; 결함목록 8개 유지).
> - **재학습 타깃 확정**: 완전 게더(A+B+C, C는 cred 가용 시 조건부 login) → `build_tbox_planner_sft`/`build_v2_prompt` 내재화. should_F 거부축 별도 보존(gross 14-scope). 8 자격증명부재는 거부/can't-do로(should_T 라벨이나 정직 에이전트 불가).
> - 코드: `run_scripted.py`(scripted-gather, --realistic/--analyze), `lever_decomp.py`, `binding_diag.py`, `_leaderboard_bankcheck.py`(59 released files, bucket C 분해).

> **Exp-4a 데이터 파이프라인 완성 (2026-06-01)**: `build_tbox_planner_sft.py` — GT means-ends가 만든 정답
> 결정 시퀀스([login→goal]/[login→STOP]/[goal]) → 각 step을 **공유 `build_v2_prompt`(train/test 동일 프롬프트)**
> + copy-target(operator명/STOP)으로 SFT 예제화, operator 순서 셔플(§11.4 위치암기 차단; alias는 후속). **1497
> examples × 7도메인**(bank 312·GOAL76/establish178/STOP58 등), 이중호출 0. 트레이너 `lora_train_chat_toolcall.py`
> 포맷 직접 호환(assistant content만 supervise). LODO 7 splits(`sft_tbox/lodo_train_holdout_<d>.jsonl`) 준비완.
> **다음=학습**(6도메인→held-out, GPU 확보 필요) → 서빙 → `run_simulation --two_stage_v2`(SFT 모델)로 held-out eval
> → arm-3v2(should_T 2/48) 대비 게이팅 향상 측정 + ablation(빈/틀린 ABox 붕괴, §11.7).
| Exp-4b | arm-4b(L2 xattn ABox-memory, 19 LODO) | fc/full | — | — | 예정 (Phase 2, coworker) |

> **★★★Exp-4c-eval — 3-regime SFT(alias 마스킹·source3) bank 실측 + should_T 전수 재진단 (2026-06-01, rr.ps1 실측)**
> 완전게더 teacher(§8.5)로 3 regime 재학습(7B LoRA r16, grad-ckpt, seqlen2048, ep3, LODO holdout=bank, 6 non-bank
> 2382예제) → bank eval(`run_simulation --two_stage_v2`→`run_evaluation`, fc/full) + **134 task 전수 궤적 분류**.
>
> | regime | should_T 성공 | **dirgraph(게더)** | **act_called(goal)** | constraint | should_F 성공 | 총/134 |
> |---|---|---|---|---|---|---|
> | **s1** (STATUS 렌더) | **0/48** | 45/48 | 3/48 | 48/48 | 81/86 | **0.605** |
> | **s3** (NL-only source3) | 5/48 | 16/48 | 35/48 | 38/48 | 24/86 | 0.218 |
> | **alias_s3** (alias+source3) | 5/48 | 22/48 | 27/48 | 37/48 | 28/86 | 0.246 |
> | arm-4a v2 (기준선) | 4/48 | — | — | — | 31/86 | 0.261 |
>
> **★s1 총 0.605는 함정**(=should_F 81 + should_T 0 = degenerate 과잉거부; 거부-all 0.642에 근접). **헤드라인 금지.**
> **★진단(사분면 전수, dirgraph×goal)**: should_T 성공=dirgraph∩goal 동시충족, 대각선=0~5뿐.
> - **s1**: 게더✓(45) goal✗(3) — 궤적 `[check, login, exit]`×45 = **게더 완벽 후 goal 직전 exit**(과잉STOP).
> - **s3**: goal✓(35) 게더✗(16) — `[…, login, goal, get_*, exit]` = **안게더 act**(과소거부).
> - **병목 이동**: 이전 완전게더 레버 **성공**(s1 dirgraph 45/48=under-verification 해소) → **새 병목=터미널 "게더후 act/STOP 게이트"** 노출. 비대칭(STOP=상수쉬움 vs goal=varying드묾) + should_F GOAL-오라벨 32/86 오염.
> - **alias 중립**: alias_s3≈s3(sT 5=5, 총 .246 vs .218), 7B G-alias 사다리 평평(0.254/0.215/0.205)과 정합 → 병목이 게이트라 alias 무관(채택은 유효, should_T 안 가름).
> - **★재설계 = Exp-4d**: 게이트-토큰(터미널 goal명→상수 "ACT"/"STOP", varying-name 비대칭 제거) + should_F→STOP 정정. 헤드라인=**dirgraph+∩goal+**(총점 폐기). 설계 권위본 `TASK_CONSTRAINT_DESIGN.md §8.6`.
> - 인프라 교훈: rr.ps1 `pkill -f`가 **자기 셸 self-match→SSH drop**(문자열 split로 회피); vLLM `kill-9` 잔여 `/dev/shm` 세그먼트가 **GPU wedge→engine-init 반복실패**(shm 정리+GPU별 PID kill로 복구). 병렬 eval=GPU별 격리.

| Exp-4d | gate-token SFT(ACT/STOP) **s1_gate** | fc/full | 0.527 | should_T 2/48 · should_F 67/86 | ⚠️ **부분 성공**: goal호출 3→13(4배)·과잉거부 81→67 완화(비대칭↓) **but BOTH(dirgraph∩goal)=0→2 거의 불변**(게더 덜 하고 act). s3_gate=OOM후 solo재학습·alias_s3_gate=serve실패 재eval |

> **★Exp-4d gate-token 1차 (2026-06-01, rr.ps1 실측)**: 터미널 타깃을 varying goal명→상수 ACT/STOP으로 교체.
> **s1_gate vs s1**: goal_called **3→13/48**(4배↑), should_F **81→67**(과잉거부 완화), dirgraph 45→36, **BOTH=dirgraph+∩goal+ 0→2**, should_T 0→2, 총 0.605→0.527.
> - ✅ act/STOP 비대칭 완화(가설 방향 맞음) / ❌ 근본(게더 AND act 공존) 미해결 — ACT 쉽게 하니 게더 덜 하고 act(s1↔s3 중간 이동).
> - **★다음 Exp-4e = ACT 전제조건 가드**(resolver: required 미게더면 ACT 무효→게더 강제 → BOTH 직접 보장) + BOTH-직접 RFT. 설계 `TASK_CONSTRAINT_DESIGN §8.6.9`.
> - 인프라: 2잡/GPU0(48GB)=OOM(s3_gate, 한 잡 26.9GB) → solo 재학습+expandable_segments. eval serve engine-init은 학습 직후 전이상태→GPU free 후 정상.

| Exp-4e | ~~gate-token + ACT 전제조건 가드 (resolver)~~ → **SUPERSEDED** | — | — | — | ⛔ 폐기: "가드"가 아니라 **"게더 미완이면 ACT 금지"를 *학습*** 으로 피벗(§3 학습 사다리). 아래 Exp-4f(Rung1)로 대체. |
| Exp-4f (Rung1 ① v1) | **readiness-gate SFT** (s1_scratch + alias_s3_scratch, LODO holdout=bank) | fc/full | s1 0.6418(degenerate) | **should_T 0/48 · dirgraph 48 · goal 0 · BOTH 0 · should_F 86/86** | ⛔ **INVALIDATED (teacher 라벨링 버그)** — `all_verified=true;STOP` 380/473 → av=true⟹STOP 학습 → ACT 붕괴(goal 0/48, login=True 29개서도 0). alias_s3=serve 실패(NO DATA). **버그 수정(`32036bc`) 후 데이터 재생성+Rung1 재학습 필요.** |

> **★★Exp-4f — Rung1 ① readiness-gate SFT (2026-06-01, 학습 중)** — `EXPERIMENT_DESIGN §3` / `TASK_CONSTRAINT_DESIGN §8.7` 권위본.
> **사다리 피벗**: Exp-4d gate-token이 act/STOP 비대칭은 완화했으나 **BOTH(게더∩act) 0→2 정체**(ACT 쉬워지니 게더 덜 함) → 근본 = **게더-act 경합(순서)**. 해법 = "순서를 *학습*"(가드 아님): per-step readiness 게이트 토큰 `ready=false; <tool>`(미완→ACT 구조적 금지) / `ready=true; all_verified=<T/F>; ACT|STOP`(완료 후 분기). 데이터 검증=ready=false 뒤 ACT 0건.
> - **학습**: 7B LoRA r16, grad-ckpt, seqlen2048, ep3, LODO holdout=bank, 6 non-bank **2382예제/regime**. 2 regime SOLO(GPU0 `s1_scratch`=source1+gate+scratch / GPU1 `alias_s3_scratch`=alias+source3+gate+scratch). 어댑터 `sft_runs/qwen7b_tbox_{s1,alias_s3}_scratch_lodo_bank`.
> - **헤드라인**: should_T **BOTH(dirgraph_satisfied ∩ action_called_correctly)** + should_F gross (총점 금지). **격파 기준선 = gate-token BOTH s1_gate 2 / alias_s3_gate 1.**
> - **★v1 결과 (2026-06-01, rr.ps1 실측) = INVALIDATED (teacher 라벨링 버그)**: 학습 정상완료(train_meta 2/2, loss 수렴) but eval degenerate. **s1_scratch: should_T 0/48 · dirgraph 48/48 · goal 0/48 · BOTH 0 · should_F 86/86**(Mean 0.6418=거부-all). alias_s3=vLLM engine-init 실패(학습직후 전이상태)→NO DATA.
>   - **근본원인(eval JSON+학습데이터 전수)**: 파서는 정상(`ready=true;...;ACT`→goal 검증). goal=0은 **login=True인 29개 태스크서도 0** → 모델이 ACT를 *전혀* emit 안 함. 학습데이터 전수: 터미널 `all_verified=true; ACT`=223 / `all_verified=false; STOP`=93 / **`all_verified=true; STOP`=380(버그)**. `build_tbox_planner_sft`가 av를 *관찰된 check의 AND*(target 독립; `_ro` 비면 default True)로 계산 → 정책/제약 거부(체크 통과) STOP이 av=true → "av=true⟹STOP"(380>223 ACT) 학습 → ACT 붕괴. (gate-token s1_gate가 goal 13이었던 건 av 토큰 부재.)
>   - **수정(`32036bc`)**: `av = (target == "ACT")` — all_verified를 게이트와 인과 일치(true⟺ACT). 모델은 in-context 정책+요청으로 "행동 허용" 예측.
>   - **다음 = Rung1 v2 재실행**: 데이터 재생성(수정된 teacher)→재학습(2 regime solo)→eval. alias_s3는 serve 전 `/dev/shm/vllm*` 정리+GPU free 확인(engine-init 전이실패 회피). **헤드라인=alias_s3 BOTH** vs 기준선 s1_gate 2/alias_s3_gate 1.
> - **다음 사다리**: v2 BOTH 다수면 → ② DPO(`build_dpo_pairs`+`dpo_train`, init=Rung1 어댑터; ⚠️ DPO 쌍도 같은 teacher서 파생→재생성 필요) → ③ RFT GRPO(`sopbench_reward`+`grpo_train_sopbench`). ⚠️③ `assemble` goal당 첫 task 매핑(`recs[0]`) 갭=Rung2 per-task isolation 필요.

> **Exp-4-precheck (getter-groundability census, 2026-06-02 PM, rr.ps1 실측)** — 결정적 pre-check: "합성 먼저냐 데이터-fix(Track A) 먼저냐"를 경험적으로 종결. 코드 `precheck_getter_groundability.py`, 7도메인 전 판별조건(condition-kind constraint leaf) → 존재 getter 매칭.
> - **결과(name-token 휴리스틱, first-cut)**: groundable **113/157 = 72%**(A_callable 32=직접호출 + B_getter 81=토큰오버랩) · ungroundable **44/157 = 28%**. 도메인별 U: bank **0**(손-map 완전)·dmv 2·healthcare 5·hotel 8·library 1·online_market 9·**university 19**(최다).
> - **auto-derive 검증(bank 손-map 9개 ground-truth)**: co-present 키 **6/8 = 75% 일치**. DIFF 2 = pay_loan_amount_restr(hand=get_account_balance / auto=get_bank_maximum_loan_amount), safety_box_eligible(hand=get_account_balance / auto=get_safety_box). → token-overlap은 *대략* 맞으나 precision 불완전(같은-이름 도구를 eligibility-결정 getter보다 우선).
> - **★해석(중요, 과대해석 금지)**: **28%는 진짜 ungroundable이 아니라 name-token 휴리스틱의 상한.** 44개 수동 triage: **시간창**(within_*_period·before_*_deadline·appointment_date_valid ~10)·**수량/카운트 한도**(within_*_limit·less_than_max_*·under_max_*·has_remaining_nights·credits_within_limit ~9)·**상태/존재**(has_items_in_cart·has_shipping_address·enough_stock·guest_already_checked_in·tuition_balance_zero·no_*_conflict ~12)·**값-임계**(income_proof_enough·gpa/credit/age/residency ~11)·기타 ~2. 앞 3범주(~31)는 **시간 getter+날짜필드 / 카운트 getter / 엔티티-상태 getter로 grounding 가능**(이름 토큰만 안 겹쳐 누락). **진짜 no-getter 잔여 = 주로 값-임계 부분집합**(income/gpa/age 등 value getter 부재 시) → **실제 ungroundable 천장은 28%보다 훨씬 작음.**
> - **결론(1차)**: Track A primary, auto-derive는 token-overlap보다 강해야, 진짜 ungroundable 수치화 필요. → **아래 env-소스 확정으로 종결.**

> **Exp-4-precheck-FINAL (env predicate-source 확정, 2026-06-02 PM, rr.ps1 실측) — ★ungroundable ≈ 0**: 44개 UNCERTAIN을 env predicate 본체(`env/domains/<d>/<d>.py`)로 정의적 분류. **각 condition predicate가 필요한 getter를 자기 본체에서 *직접 호출***(예: `above_minimum_age`→`internal_get_interaction_time`+`internal_get_user_birthday`; `within_attempt_limit`→`internal_get_test_details["attempts"]`; `meets_income_requirements`→`internal_get_student_info["income"]`; `gen_ed_requirements_met`→`student["completed_courses"]`; `valid_identification`/`income_proof_enough`→**arg-only**=getter 불요).
> - **★"28% ungroundable"은 전부 휴리스틱 결함**: (a) name-token 미스, (b) `is_getter`가 **`view_*` 누락**(online_market `view_cart`/`view_shipping_addresses` 실재→cart/shipping/review 전부 groundable), (c) multi-getter, (d) arg-only, (e) rich-record getter(`internal_get_student_info`=student 레코드 전체: academic_status·tuition_balance·residency·minors·income·completed_courses·gpa). **진짜 no-access-path 조건 = 사실상 0.**
> - **결론(확정)**: ① **Track A로 ~100% groundable** — 합성 §3.0b는 grounding 목적 고유 scope ≈ 없음(many-conditions 일반화 GO/NO-GO 용도만). in-dist 붕괴 = 순수 getter-map 결손+teacher policy조건 미게더, 절차-학습성 문제 아님. ② **★auto-derive v2 정답 방법 = predicate 소스 정적 파싱**: `self.domain_system.<getter>(...)` 호출 추출 = condition→getter-집합(구조적·전수·전이무결, 토큰추측 아님). 코드 `autoderive_getter_map.py`. ③ 다음 = 파서로 7도메인 맵 자동생성→bank 손-map 검증→`build_tbox_planner_sft` GETTER_BY_DOMAIN 대체→teacher 재생성.

> **Exp-4-mapwire (auto-map → teacher 배선 + census 검증, 2026-06-02 PM, rr.ps1 실측)**: `build_tbox_planner_sft`에 `getter_map.json`(multi-getter) 배선(011deb9) 후 SFT_CENSUS 7도메인 실행. 혼동행렬(should×modeled):
> - **✅판별축 복구**: `should0_model1(BAD=차단조건 못봐 허위permit)` ≈ **0 전도메인**(bank 1·나머지 0). 옛 permitted-collapse 원인(condition 미게더) **제거 확인**.
> - **⚠️`should1_model0` 큼**(bank 35/48·healthcare 44/45·online_market 59/60·library 24/24) **but 대부분 census 진단결함**: bank 35 중 **33이 reach=0인데 leaf는 전부 True**(`lt=[True,True,True]`) → census `_reach=de.process(goal)`가 **로그인 establishment 전 초기상태**서 goal precond 평가해 0(teacher는 step3서 login 실제수행→데이터 정상). 나머지 **2는 reach=1+False leaf = pay_loan OR/조건부**(flat-AND가 OR 오판=리뷰어 OR경고, 소수).
> - **결론**: map 배선 **건전**(판별 복구·hand-map recall 8/9). 진짜 잔여 2개: **(a) census `_reach` establishment 미시뮬(다른세션 census 과엄격 진단버그)** · **(b) teacher `next_decision` step2 flat-AND(`any False→STOP`)가 OR/조건부 과잉거부**(소수 태스크). 다음=teacher를 **constraint-TREE eval**(AND/OR/chain)로 + census `_reach` 수정 → teacher 재생성 → 재학습 → ACT-recall|게더 + STOP-recall 분리 재측정.

> **Exp-4-rung1-trained (corrected teacher → 7B LODO 재학습+bank eval, 2026-06-02 17:40 KST, rr.ps1 실측) — ★헤드라인 NULL**: getter-map+should_succeed-terminal로 고친 teacher(검증: 터미널 48/86 정확)로 2 regime SOLO 재학습(LODO holdout=bank, ep3, val_loss s1 0.053/alias_s3 0.087 수렴) 후 bank held-out eval(fresh, freshness-guard 통과).
> - **결과(should_T BOTH = dirgraph∩goal)**: **s1 BOTH=0**(dirgraph 9·goal 20) · **alias_s3 BOTH=2**(dirgraph 18·goal 17). 기준선 gate-token s1 2/alias_s3 1 대비 **개선 없음**(null). Mean Pass Rate s1 0.246/alias_s3 0.205(거부 부풀림, 헤드라인 아님).
> - **★corrected 지표(reviewer, eval JSON 산출)**: **ACT-recall|게더(dirgraph) = 0/9(s1)·2/18(alias_s3)** = 게더 충분해도 거의 ACT 안 함; **gather-then-STOP = 9/9·16/18**(게더후 거부); **premature-act(게더없이 goal) = 20/45·15/47**; **STOP-recall(should_F 정답) = 32/89=36%·26/87=30%**(과소거부).
> - **⚠️1차 진단 철회("gather-act XOR / SFT는 순서 못 배움")**: 궤적 전수조사로 반증됨. 끼워맞춤이었음.
> - **★★정밀 분류 (궤적 전수조사 2026-06-02, eval JSON 실제 tool_calls/results) — 애매어 폐기**: should_T 45 두 **분리·정의가능** 실패군:
>   - **(A) REFUSE_after_login_False = 19/45 (#1)**: 모델이 `login_user`를 호출 → **False** 반환 → 거부. 그러나 그 태스크의 GT 제약은 login 불요(예: apply_credit_card 제약=`{internal_check_username_exist}`만, username_exist=True 충족). 모델이 **"항상 login" prior**로 비필수 login을 부르고, creds 미제공이라 login=False, 그 False를 차단으로 오인해 거부. = **조건부 login 미학습**(login은 `logged_in_user∈required`일 때만; 비필수 도구 실패에 거부 금지). [메모리의 login↔creds 컨파운드와 동일]
>   - **(B) ACTED_but_dirgraph0/cnv0 = 16/45 (#2)**: goal 호출·**성공**(apply=True)했으나 **종료 안 함** — goal을 **평균 4.1회(s1)·7.2회(alias_s3, 13/18이 8회=max-turns) 반복호출** + 성공 후 늦은 잡 호출(credit_score=False 등). 반복/늦은 호출이 GT 응답열과 불일치 → `constraint_not_violated=0`, 의존순서 replay 깨짐 → `dirgraph_satisfied=0`. = **"성공 후 1회로 멈춤" 미학습**. 근본 = teacher가 ACT에서 루프 break(L279) → **"goal 성공→STOP/exit" 예제 0개**(post-success 종료를 시연한 적 없음).
>   - 잔여: premature-act(goal을 필수체크 전 호출, 4) · REFUSE_no_False(순수 오결정/도구에러, s1 2·alias_s3 9=alias 가중).
> - **★eval 플래그 정의(evaluator.py)**: `dirgraph_satisfied`=각 action 호출 시 의존 선행함수가 먼저 호출됐나(replay) · `constraint_not_violated`=각 호출 응답이 GT 응답과 일치하나 · `action_successfully_called`=goal 호출·성공 · `action_called_correctly`=(should_succeed==successfully_called) · success=5개 AND.
> - **★★before/after (동일 분류기, 수정 전 `eval_s1_scratch_re` 06-02 08:49 vs 현재)**: **수정 전 = 완전 refuse-all**(should_T goal-called **0/48**·dirgraph 48/48·BOTH 0 / should_F refuse **86/86=100%**). **현재 = goal-called 20/45**(행동 시작)·dirgraph 9·BOTH 0 / should_F refuse ~70%. → **이번 fix(getter-map+should_succeed terminal)는 타깃(permitted-collapse=never-act)을 해소**: 모델이 정책조건을 게더하고(궤적에 credit_score/balance 출현) **행동을 시작**(0→20), refuse-all 탈출. **단 헤드라인 BOTH는 불변(0)** — 실패가 "안 함"에서 **(A)login 과적용 + (B)성공후 비종료**로 *이동*했기 때문(둘 다 새 타깃, SFT 원리한계 아님).
> - **★gather-grounding(이번 실험의 핵심 타깃) = 달성**: "정책조건 getter를 실제 호출"이 **PRIOR(수정 전) 0/32 → CURRENT 14/30(완전 13/30≈43%)**. 즉 getter-map fix 전엔 정책조건을 *한 번도* 안 게더(permitted 콜드붕괴) → 후엔 게더함(apply→credit_score 등). **데이터 수준 100%(158/158 groundable·검증), 모델 수준 ~43%**(나머지 57%는 gather 실패가 아니라 아래 next-step 실패(login 과적용·조기exit)가 gather를 *중도절단*). dirgraph_sat은 게더품질 아님(goal 미호출시 vacuous True; prior 48/48=한번도 행동안함의 부산물).
> - **결론(정정·정밀)**: "SFT가 조건부 action 못 배움/gather-act XOR"는 **철회**. **gather-grounding 스텝 자체는 성공**(0→43%, refuse-all 탈출, goal 0→20). BOTH=0 불변의 원인은 gather가 아니라 **next-step 설계 누락 2개**(설계서 어디에도 없음): **(1) 성공후 종료**(teacher가 ACT에서 break→post-success STOP 예제 0개→모델이 goal 4-8회 반복→cnv/dirgraph 파괴, ACTED_dg0/cnv0 16건) · **(2) over-login(REFUSE_login_False 19건) = required_set 정합성** — ⚠️**원인 미확정(prior-override 단정 철회)**: 리뷰 코드확인상 eval/teacher가 goal을 *full operator precond*(login 포함)로 렌더(gconstr=None) → "BLOCKED→login" 명시표시 → 15/19는 잘못 렌더된 required_set을 *따른* 것일 수 있음(렌더 confound; "lighten→login −59%" 정황 일치). **L0 선행**(15건 goal-라인 덤프)로 confound vs 진짜 R-도출오류 판별 후에야 처방 결정. **★login 특별취급 금지(설계 결정)**: ~~"조건부 login"~~ 철회 — login은 도메인-특화라 login-게이트 학습=ABox 베이킹=전이파괴. **TBox는 "required_set 도구만 사용"(R1)만 학습**, login은 평범한 도구. 설계 L69 DPO 사전정당화("조기ACT")는 실패와 불일치(조기ACT 4건) → 기각. 진짜 재학습 = **(1) R3 성공후-종료(positive, 깨끗)** + (2) source=3 R-도출 정합(login-특별 아님, L0 후). 상세 처방·L0–L4 사다리 = `scripts/distill/EXPERIMENT_DESIGN.md` Rung1 실행계획 #1–#6.

> **Exp-4-rung1-T1T2 (login-uniform + post-success termination → 7B LODO 재학습+bank eval, 2026-06-03, rr.ps1 실측)**: T1(required_set 균일화=login 비특수; source = task["constraints"] leaves ∪ goal-default establishable[`dep_innate[goal]`∪`dep_full_raw[goal]`∪`ops[goal]["precondition"]`] — **gate=DIFFERENT 98/830** 반영, establish 우선·`is_est` 플래그) + T2(post-success `done=true; STOP` 종료) → 2 regime SOLO 재학습(val_loss s1 0.041/alias_s3 0.075, n_train 3980/regime, ep3) → bank held-out eval(fresh: eval 02:17 > adapter).
> - **헤드라인 BOTH(dirgraph∩goal)**: s1 **4**(dirgraph 35·goal 12) · alias_s3 **4**(dirgraph 33·goal 15). vs gate-baseline(s1 2·alias_s3 1)·직전 Exp-4-rung1-trained(s1 0·alias_s3 2) = 소폭↑, **G-SFT(≥15) 미달**.
> - **교정지표(잠긴 위계)**: ACT-recall|게더 0.11/0.12 · gathered_but_no_act(과잉거부) 31/29 · acted_but_dirgraph0 8/11 · should_F STOP-recall 65/86(76%)·48/86(56%) [직전 36%·30% → 거부회복↑].
> - **✅T2(종료) 완전 성공**: 도구반복≥3회 **0/48**(직전 4-8회 반복 소멸), acted_dg0 16→8/11. ✅**(A) login→False→refuse 소멸(0건)** = T1이 해소.
> - **★★근본원인 확정 — RLLOG raw planner 출력 전수(alias_s3 재평가, 526스텝)**: "게더 후 멈춤"의 정체 = **`permitted=false; STOP` 과잉거부**. 터미널 census: **permitted=false;STOP 81 · DONE(post-success) 49 · permitted=true;ACT 40 · permitted=true;STOP(모순)=0 · ACT-미발화/malformed=0**. → (i) **ACT 미발화 코드버그 기각**(permitted=true는 항상 ACT 발화→done 종료 정상), (ii) **scaffold 토큰 모순 기각**(permitted=true;STOP 0건), (iii) **확정 = permitted 게이트 콜드붕괴**(`preconds_verified=true` 게더성공인데 정책적용 추론이 false로 무너짐) = **v3 grounding-비대칭 재현**. 예: `ready=false;op_33 → op_20 → permitted=true;ACT → done;STOP`(정상) vs 동일 goal 다른 인스턴스 `... permitted=false;STOP`(거부).
> - **★게이트 정합성 분해(should_T 48, 헤드라인 eval) — "permitted=true;ACT면 성공 아닌가?" 검증(리뷰 지적)**: **과잉거부(permitted=false;STOP) s1 32·alias_s3 30** + **조기행동(permitted=true;ACT인데 dirgraph0) 8·11** + goal호출-실패 1·2 + OK_BOTH 4·4. → ACT한 것(s1 13/alias 17)은 goal은 거의 성공(goal-fail 1·2)하나 **다 게더 전 `preconds_verified=true` 거짓선언→조기 ACT→dirgraph 위반**(예 `close_account`가 admin-auth 없이·`get_loan`이 credit체크 없이 행동). → **`permitted` 거짓-음성(과잉거부 30-32)** + **조기행동(8-11)=required-set 과소추론**(작은 required-set에 AND→ACT, GT는 더 요구). ⚠️**preconds-AND는 정상**(RLLOG 결합분포서 preconds=T 항상·false 0; 아래 메커니즘 bullet 참조) — 조기행동은 preconds 오류가 아니라 required-set 추론 오류. 근본 = permitted이 게더값 *룩업* 아닌 *추측* → **permitted을 게더 truth의 AND로 grounding하면 교정**(DPO보다 근본).
> - **★★permitted 실패 메커니즘 + 이전실험 대조 (리뷰지적 "AND는 이전에 잘 됐다" → 팩트확정: 맞음)**: RLLOG 터미널 (preconds_verified, permitted) **결합분포 = `preconds=T;permitted=F;STOP` 81 · `preconds=T;permitted=T;ACT` 40 · `preconds=F` = 0**. → **AND(preconds_verified)는 터미널 121개 *전부* true(false 0건) = AND 자체는 작동**(Rung1 v2 `preconds=true;permitted=false 92/98`와 *동일*, 반전 아님). **단독 실패 = `permitted`**(게더값과 분리된 콜드 별도판단). ⚠️**정정**: 앞서 "conjunction 붕괴"라 한 건 오류 — 결합분포가 반증(preconds 항상 true). 조건수 상관(BOTH cond **2.75/3.75** < refuse **3.81/3.73** < premature **4.50/4.45**; set_safety_box cond7 10/10거부 vs close_account cond2 0거부)의 진의 = **AND 붕괴 아니라 "콜드 정책조건이 많을수록 permitted 추측이 false로 기움"**. **조기행동(dirgraph0)도 AND오류 아님 = required-set 과소추론**(모델이 *작은* required-set에 올바르게 AND→ACT했으나 GT는 더 요구). prior 편향(학습 permitted true240:false456=1.9× false)이 증폭. **through-line: permitted 콜드붕괴는 v2부터 미해결 동일범인, 주변(게더·login·반복) 수리로 *단독 노출*됨**. **★처방확정: 모델은 AND 가능(preconds 증명) → `permitted`을 콜드추측이 아니라 *게더한 정책조건 truth의 AND(기록 룩업)*로 강등**(= "메모장": 도구결과 기록→그 기록만으로 permitted 계산). 새 능력 불요. DPO는 편향 한방향만 눌러 부적합.
> - **결론**: T1·T2 의도대로 작동(반복·login-refuse 제거, 게더 9-18→33-35). **병목이 (A)login-refuse/(B)반복 → `permitted` 콜드붕괴(단독; AND/preconds는 정상)로 이동·노출**. (permitted을 게더 truth의 AND-룩업으로 grounding = 다음 처방.) BOTH 미달의 단일 지배원인 = should_T 게더 후 `permitted=false` 거부(~29-32/48). 코드/토큰 아님. **다음 레버 = permitted을 게더 사실에 grounding(wait-until-resolved/v3) 또는 ②DPO(should_T→`permitted=false;STOP` dispreferred)**. login over-call(45/48)은 거부 유발 안 함(부차). ⚠️주의: apply_two_stage_patch가 constants.py 재패치서 AssertionError였으나 이미-패치된 클론이라 eval 유효(FC 정상 작동·결과 산출 확인).

> **Exp-4-rung1-CAST (conditional steering probe, 2026-06-03, rr.ps1 실측)**: alias_s3 어댑터에서 ACT-vs-STOP behavior vector 추출(`cast_extract_actvec.py`, 같은 터미널 프롬프트 content-통제, layers 10-23, 300쌍) → gated 서버(`_steering_vllm_server_gated.py`, gate=orth, layers 14-20) alpha-sweep bank eval. **결과 = NULL**: α0(control) dirgraph32·acted16·BOTH**4**·should_F STOP46 / α8 32·16·**4**·46(control과 동일) / α16 31·17·**4**·46. → **always-on/orth steering이 BOTH·should_F 전혀 못 옮김**(Phase2a steering null 재현). **함의**: steering=bias-at-most이고 trie평가(grounding+serial compute)는 못 고침(설계 Rung3 재검토·AC0/TC0 논거 실증). 단 caveat: always-on/orth·좁은 α(decode-anchor gated 강버전 미구현). **→ 처방은 derivation/xattn(v3), steering 아님.** 인프라: gated 서버는 **tau2_vllm_env python**(seka엔 vllm 無)·vllm 0.11.0서 monkey-patch 정상.

> **Exp-4-rung1-v3-AB (grounded 트리평가 teacher A/B → 7B LODO 재학습+bank eval, 2026-06-03 19:08 KST, rr.ps1 실측) — ★헤드라인 NULL (v3 회귀)**: control(비-treeval, should_succeed 터미널 `alias_s3_nt`) vs v3(`--treeval` grounded per-leaf truth→AND/OR/chain derivation, `alias_s3_treeval`). 둘 다 key-fixed·alias_s3 헤드라인·LODO holdout=bank·ep3·n_train 11,940·SOLO(GPU0 control/GPU1 v3). val_loss control **0.0679** < treeval **0.1064**(treeval 토큰이 fit 더 어려움). eval fresh(18:51 시작 > adapter, freshness-guard 통과). 알려진 무해 AssertionError(constants.py 재패치, 이미-패치 클론 → eval 정상).
> - **분리지표 (should_T 48 / should_F 86)**:
>   - **control(nt)**: dirgraph **31** · acted 15 · goal 15 · **BOTH 5** | ACT-recall|게더 **0.16** · over-refuse(noact) 31 · premature 10 ‖ should_F STOP-recall **36 (42%)**.
>   - **v3(treeval)**: dirgraph **18** · acted 7 · goal 7 · **BOTH 2** | ACT-recall|게더 **0.11** · over-refuse(noact) 38 · premature 5 ‖ should_F STOP-recall **17 (20%)**.
>   - baseline(T1/T2 alias_s3): BOTH 4 · over-refuse~30 · STOP~46.
> - **판정 = v3 무효, 그것도 control 대비 명확한 회귀(분기 B 강버전)**: treeval가 **모든 축에서 control 미달** — BOTH 5→2(baseline 4보다도 낮음), dirgraph 31→18(계획 생성 급감), ACT-recall 0.16→0.11, over-refuse 31→38(거부 증가), STOP-recall 42%→20%(회귀). grounded emit derivation은 콜드붕괴를 해소하기는커녕 dirgraph 생성·STOP까지 악화. val_loss 상승이 선행 신호였음(grounded treeval 토큰 학습이 더 나쁜 정책으로 수렴).
> - ⚠️**위 "회귀" 판정 = 디코드 예산 아티팩트, 철회**(아래 전수조사·재시험으로 정정).
>
> **Exp-4-rung1-v3-AB ★전수조사 정정 (2026-06-03 19:43 KST, rr.ps1 실측) — 회귀는 truncation 아티팩트, v3 진짜 판정 = control과 동(무개선)**: 위 헤드라인 NULL을 분기 결정 전 **전 궤적 전수조사**(행동층=도구시퀀스 + 메커니즘층=SOPBENCH_RLLOG planner raw 출력)로 원인 규명.
>   - **행동층**: treeval should_T 48 중 **35개가 max_steps=10 캡까지 루프**(같은 게더 도구 반복, avg 8→20스텝), should_F도 57/86 루프. control은 4–6스텝서 깨끗이 종결. = 콜드붕괴(over-refuse)가 아니라 **전역 비수렴**.
>   - **메커니즘층(RLLOG)**: control terminal-attempt(ready=true) **27/27=100% 결정 도달**(`...; permitted=..; ACT`), treeval **0/29=0% 도달** — 전부 `ready=true; gate = AND(op_32=false, AND(op_39=true, op_25=true` 에서 **planner max_tokens=24 절단**(닫는 괄호·`= <val>; ACT/STOP` 종결 전). 즉 **모델은 grounded tree-eval emit을 정확히 학습**(중첩 AND 전개 중)했으나, 디코드 예산이 종결 전 잘라 결정 미파싱→재게더→루프.
>   - **무재학습 재시험(`SOPBENCH_PLAN_MAXTOK=1024`, 동일 어댑터 재서빙·재eval, `rung1_v3_maxtok_retest.sh`)**: treeval **loop 35→0, terminal 도달 0%→100%(212/213), BOTH 2→5**. control(maxtok불민감) BOTH 5 유지. → **"v3가 모든 축 회귀"는 100% truncation 아티팩트.**
>   - **★v3 진짜 판정 (maxtok=1024, 수렴 상태)**: treeval **BOTH 5 = control 5 = 무개선**. 단 실패 모드 이동: over-refuse 33→**29**(거부↓), acted 14→**15**(행동↑), dirgraph 32→**22**(완전게더↓), should_F STOP 42%→33%. = grounded 게이트가 콜드 보수성은 풀었으나(덜 거부·더 행동) 그 결정을 *완전 게더* 위에 못 얹음(acted 15 중 ~10이 dirgraph 미충족 premature). **단일-스텝 whole-expression 평가의 한계**(globality; Abbe/Feng/Kim&Suzuki가 예측 — 식 emit ≠ 정확 평가).
>   - **함의 (정정)**: ❌"SFT로 트리 불가" 아님(우리 데이터: AND(preconds) 0오류로 학습됨). ❌"v3 회귀"도 아님(아티팩트). ✅**진짜 결론 = 단일-스텝 grounded derivation은 콜드 permitted 대비 BOTH 무개선**. 처방: (i) **inductive multi-step derivation**(잎별 reduce + 중간집계 SFT loss; litreview §4 — 단일식→단계화, 여전히 SFT) 또는 (ii) **T1c**(treeval의 premature 10건이 게더-완전성/slot 결함이면 BOTH로 전환 가능 — treeval base가 control보다 T1c 수익 클 수 있음) 또는 (iii) **②DPO**(단 treeval는 이미 over-refuse↓라 DPO 레버 약함). 코드: `rung1_v3_rllog_census.sh`(메커니즘 census)·`rung1_v3_maxtok_retest.sh`·`two_stage_client.py SOPBENCH_PLAN_MAXTOK`.
>
> **Exp-4-rung1-v3ind (inductive multi-step reduction-chain teacher → 7B LODO 재학습+bank eval @maxtok=1024, 2026-06-04 00:59 KST, rr.ps1 실측) — ★NULL, 단일식보다 더 나쁨**: 위 처방(i) 검증. teacher가 단일식 대신 **bottom-up reduction 체인**(`t1=AND(a=t,b=t)=t; t2=AND(t1,c=f)=f; gate=f; ACT|STOP`, 중간 fold supervised) emit하도록 학습(`--treeval_inductive`, 설계서 `RUNG1_V3_INDUCTIVE_DESIGN.md`, chain_val 일관성 가드, 빌드검증 grounded 757/0불일치/90.4%). control=기존 single-step treeval 어댑터. 3-way 동일 maxtok=1024.
>   - **헤드라인**: treevalind **BOTH 3** · dirgraph 17 · acted 8 · over-refuse **39** · should_F STOP **21%** ‖ single-step treeval BOTH 4(n_T=45*) · acted 17 · over-refuse 25 · STOP 33% ‖ baseline(maxtok retest) nt/treeval BOTH 5 · nt STOP 42%. (*treeval n_T=45: ACT-call `tool_choice` 미스매치 버그로 should_T 3개 드롭=ACT 많이 한 arm만 영향, treeval 불리쪽 confound→결론 강건. nt arm은 run_evaluation 타이밍 실패로 0=무효, 베이스라인 retest nt=5 사용.)
>   - **판정 = inductive NULL이며 단일식보다 악화**(BOTH 3<4<5). 단계별 derivation supervise가 BOTH를 못 올리고 **over-refuse↑(25→39)·acted↓(17→8)·STOP↓(33→21%)** = 모델이 훨씬 보수적.
>   - **★전 궤적 전수조사 — 근본원인 = derivation 타깃이 (a) 구조 fabrication + (b) over-gather를 유발**: RLLOG 1226 planner호출 중 **gather 1159(94.5%)·terminal 61**(STOP 43/ACT 18) → **~73 태스크가 terminal 없이 step cap까지 게더**. should_T 게더깊이 median **10(=cap)** vs single-step 6; over-refuse 39개 전부 ~10스텝. terminal chain 35개: distinct-op median 6·**max 11**, **per-goal 동원조건이 실제보다 과대**(pay_bill 평균 **10**op/실제~3·set_account_info 9/2·deposit 6.5/3). **STOP chain 17/17 전부 `=false` leaf 포함** → 큰 fabricate 트리가 false leaf 노출→`gate=false;STOP`. = "rich derivation을 supervise하니 모델이 rich derivation을 생성, 그게 fabrication+over-gather로 역효과."
>   - **★조건수별 BOTH 분해 (serial-depth 병목 여부 최종판정)**: treeval `1c:0/3 2c:0/7 3c:1/9 4c:2/10 5c:1/9 6c:0/6` · treevalind `1c:0/3 2c:1/8 3c:1/10 4c:0/11 5c:1/9 6c:0/6`. **BOTH가 모든 조건수서 균일 바닥, 최단순 1-2조건도 0.** depth/fan-in 병목이면 저조건수서 높고 decay해야 하나 **1조건부터 0** → **serial-depth/조건수는 병목 아님**(트리복잡도 무관 균일 실패). **→ depth-recurrence/트리평가-깊이 방법(Universal/Looped TF, litreview §9)은 비-문제를 푸는 것.** 병목 = gather/결정 정책(거부편향·fabrication·premature).
>   - **함의 (방향 전환)**: 트리평가-*형식* 라인 종료(단일식 NULL·inductive 더 나쁨, 둘 다 source=3 보상). fabrication·over-gather는 **모델이 트리를 추론/emit하게 둬서** 발생 → **source=1(태스크별 구조 제공)이 fabrication 직접 차단**(pay_bill에 10op 환각 불가) + 트리-emit 타깃 불필요. **다음 = ablation 사다리 `s3 → s3+조건수budget+termination → s1`**(7B가 self-termination/faithfulness 중 무엇을 외부 보철해야 하는지 분리) + 게더-종료(T1c) + 거부편향(DPO). recurrence 아님. 코드: `build_tbox_planner_sft.py --treeval_inductive`(treeval_reduce)·`rung1_v3ind_train_eval.sh`. ⚠️수정요: eval ACT-call `tool_choice`/tools 미스매치 버그(ACT 태스크 일부 드롭)·헤드라인 python이 nt eval 완료 전 실행(레이스).
>
> **Exp-4-rung1-upperbound (Agent2 upper-bound = source-effect, 2026-06-04 06:01 KST, rr.ps1 실측) — ★1차 게이트: 구조 제공만으론 BOTH 무개선, 병목=결정**: 설계서 `RUNG1_SOURCE_LADDER_DESIGN.md` §11-12. 2 fresh arm, **둘 다 tree-emit OFF·버그수정 client**(`_resolve` no-400 적용 → **양쪽 n_T=48, 드롭 0·레이스 0** = 직전 두 버그 해소 확인): **A**(`--source 3 --scratchpad`, 구조 추론=T1T2 regime) vs **C**(`--source 1 --scratchpad`, 구조 제공=Agent2@oracle, 익명 dirgraph `op:needs[..]` 프롬프트 렌더). two-gate 타깃(permitted=should_succeed), LODO=bank, ep3, maxtok=1024.
>   - **헤드라인**: A BOTH **3** · dirgraph 29 · acted 8 · over-refuse 38 · premature 5 · STOP 40% ‖ C BOTH **3** · dirgraph **34** · acted **13** · over-refuse **33** · premature **10** · STOP **49%**. 조건수별 BOTH 양쪽 균일 바닥(A `1c:0/3 2c:0/8 3c:1/10 4c:1/11 5c:0/9 6c:1/6` · C `2c:1/8 3c:1/10 4c:1/11` 나머지 0).
>   - **★판정 = 구조 제공이 BOTH 못 올림(C=A=3, 노이즈 구분 불가)**. 단 구조는 **게더·거부정확도는 개선**: dirgraph 29→34(완전게더↑)·STOP-recall 40→**49%**(>baseline nt 42%, 최고치)·over-refuse 38→33. **그러나 should_T ACT 결정이 벽**: dirgraph 34 충족인데 BOTH 3뿐(~31이 게더후 거부) + acted 13 중 **10 premature**(dirgraph 미충족 상태 행동). = **게더한 상태에서 ACT/STOP *결정*이 깨짐.**
>   - **★함의 (리뷰 A2 정확히 적중)**: source=1은 *어떤 조건을 게더*할지(구조)는 주지만 *게더 truth로 ACT/STOP*은 다운스트림 *결정* → 구조만으론 안 풀림. 권위본 line 474 "permitted 콜드붕괴=v2부터 동일범인"과 일관. **→ 다음 = Agent1(NL→구조) 아님, *결정 레버*: (i) T1c=permitted을 *주어진 구조의 leaf-truth AND 룩업*으로 강등(콜드추측 제거)+게더완료 게이트(premature 차단) (ii) DPO=over-refuse·premature dispreferred.** 구조 제공(C)은 게더·STOP 기반으로 유지하되 그 위에 결정 레버. 코드: `two_stage_client._resolve`(no-400 fix)·`rung1_agent2_upperbound.sh`. **버그수정 검증완**(양쪽 48 평가, 레이스 무).
>   - **★★전 궤적 전수조사 (should_T 48 버킷, 2026-06-04)**: 지배 실패 = **gathered_then_REFUSE**(dirgraph 충족인데 goal 미호출·exit 거부) **C 29 / A 24**. premature(acted·¬dirgraph) C 10/A 5. BOTH 3. dg+goal호출-부정확 2. → **dirgraph-충족 34 중 29가 거부**(goal 호출 5뿐). RLLOG terminal: A `permitted=false;STOP 114`≫ACT 19 / C `permitted=false;STOP 53`·ACT 38·done;STOP 81 — **C가 콜드거부를 planner수준선 절반으로 줄였(114→53)으나, should_T 완전게더분은 여전히 29 거부**. **확정: 병목 = "완전 게더 → permitted 콜드붕괴 → 거부", source=1은 게더만 늘려(dirgraph 29→34) 거부 대상만 늘림(24→29).** ★**T1c 강한 동기**: permitted=게더 truth AND 룩업이면 dirgraph-충족 29개 = AND-true → ACT → **BOTH 3→~29-34 전환 잠재**(29가 "전환 대기"). + 게더완료 게이트가 premature 10 차단.
>
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
