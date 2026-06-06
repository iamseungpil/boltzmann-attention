# FIELD-GAP / LLM-VALUE 설계서 — "결정론 프로그램 대비 LLM의 효용을 어디서·어떻게 증명하나"

> **신설 2026-06-06.** 동기 = ICLR 리뷰어의 **"이건 그냥 결정론 프로그램 아니냐"** 공격에 대한 정면 방어 + 그 방어를 *정량 실험*으로 박는 설계.
> **상태**: DRAFT (리뷰 대기). **마스터 = `EXPERIMENT_DESIGN.md`**(목표·순서·지표 권위본; 이 문서는 그 §1/§2의 detail). 관련 = `CROSS_DOMAIN_TRANSFER_DESIGN.md`(Exp-5=A축 transfer) · `RUNG1_SOURCE_LADDER_DESIGN.md`(LOCK·2-agent §11) · `RUNG1_REDESIGN_2026_06_04.md`(decision-axis A/B/C §9). 결과 권위본 = `../../reports/facet_rft_2026/SOPBENCH_EXPERIMENT_RESULTS.md`.
> **메타규칙(승계)**: 강한 주장은 reliable test 후 박제 · GPU 전 zero-cost 진단 · 공식 success로만 보고 · scaffold 도메인-분기 금지 · **offline forced-ACT 헤드룸 측정 금지(§8 RETRACTED 교훈)**.

---

## §0. 한 줄
**SOPBench-기본 regime(구조화 제약을 떠먹임)에서는 결정론 프로그램이 LLM을 대체/추월한다 — 이건 벤치의 입력 제시 방식 탓이지 우리 방법 결함이 아니다. LLM의 대체불가 효용은 "지저분하고·변하고·자연어인 현장 입력 → 결정론 엔진이 먹을 구조"로 변환하는 데 있고, 그 효용을 입력-regime을 현장 쪽으로 돌린 실험(NL-only·변경·멀티턴·벤치횡단)으로 정량 증명한다.**

## §1. 위협 정식화 (the deterministic-program threat) — 정직하게 인정부터
리뷰어 steelman: *"Guard-2가 증명했듯 너희 게이트(DGGATE)는 `task["constraints_original"]`+도메인규칙에서 evaluator의 directed_action_graph를 결정론적으로 정확 재구성한다(OVER=0/UNDER=0). 즉 '어떤 precond·순서'(절차)는 학습이 아니라 결정론 알고리즘이 계산한다. 게다가 너희 ablation이 `adapter-only≈0 → stack 75–95%, scaffold가 전부`라 적었다. 그러면 학습 7B는 ~0 기여, 성능 전부가 도메인-일반 결정론 프로그램이다. scaffold+scripted-gather / scaffold+base-7B 베이스라인은?"*

**내부 증거가 이 공격을 떠받친다(인정)**:
1. **Guard-2 = 절차 결정론 재구성** = 마스터 §1이 금지한 "절차 offload(=답지=L0)"에 근접. `dfsgather_invfunccalldirgraph(constraints_original,...)`는 per-domain 분기 0의 도메인-일반 그래프 알고리즘 = **구조화 입력만 있으면 결정론이 plan을 푼다.**
2. **결정론이 실행축서 이미 우리를 이김**: `run_scripted`(결정론 직접제어) should_T **37/48** > 우리 stack honest **33/48**.
3. **Exp-5 cross-domain(77.3% 등)은 전부 decision-axis A(결정론 offload)** = 즉 현 헤드라인 transfer 수치는 *결정론 프로그램의 성능*에 가깝다. "scaffold가 전이된다"는 결정론 알고리즘의 도메인-일반성을 확인한 것 = 공격 강화.

⇒ **결정론과 "정확도 누가 높냐"로 싸우면 진다.** 방어는 다른 축에서만.

## §2. 진단 — 왜 SOPBench-기본이 LLM을 underplay하나
SOPBench(2503.08669)는 **"구조가 주어졌을 때 SOP를 따르는가"를 테스트하려고** 설계됐다(NL→구조 컴파일 테스트가 아님). 기본 regime이 어려운 부분을 미리 제거한다:
- **(D1) 구조화 제약 떠먹임**: task마다 `constraints_original`(파싱된 정책조건) + 의존규칙 + getter_map 제공 → 결정론 재구성 가능.
- **(D2) dummy-user**: `user_known`을 구조화 덤프로 선제공(leaderboard 표준) → NL 대화 이해 불요. ARGFIX가 인자를 user_known서 결정론 추출.
- **(D3) 고정 도구·고정 정책**: 변경 없음 → 하드코딩이 안 깨짐.

이 셋이 LLM의 대체불가 역할(NL→구조, 대화이해, 변경적응)을 전부 우회시킨다. **효용은 D1–D3를 현장 쪽으로 되돌릴 때만 드러난다.**

## §3. ★이전 Track A/B/C·LOCK과의 정합 (사용자 요구 — 면밀 평가, 죽은 길 재답습 금지)
> 이 §은 "모델이 NL→구조를 해야 한다"는 본 설계의 핵심 처방이 **이미 7B에서 NULL인 Track C(decision-emission)를 되살리는 게 아님**을 못박는다.

**Track C(=decision-axis C, self-emit) = 7B NULL — 무엇이 죽었나 (LOCK 정밀범위, `RUNG1_SOURCE_LADDER_DESIGN.md` §8-10)**:
- 죽은 것 = **단일 agent가 terminal에 결정(permitted AND-진리도출)을 execution과 얽어서 emit**하는 SFT 스캐폴드(treeval 단일식 → inductive → T1c grounded-permitted). 실패 = 구조 fabrication(실제보다 큰 트리 환각) + AND가 1 false-leaf로 붕괴 + permitted 콜드붕괴. over-refuse/over-call/early-act = MODEL 회귀 = SFT-positive로 불가.
- **★Track C는 *구조화-입력* regime에서 테스트됐다**: emit한 "구조"는 *이미 주어진 leaf*에 대한 tree-eval 결정도출이지 **NL→구조 컴파일이 아니었다.** 즉 NL-only regime(§6 E1)은 Track C의 시험대가 **아니었다.**

**LOCK이 명시적으로 살려둔 것 (over-prune 금지)**:
- **gather-grounding (0→43%, LODO 전이됨)** = 어떤 precond/검증도구를 establish할지 = *부분 구조 생산* — 이미 학습·전이.
- **2-agent Agent1 (구조분리)** = NL→dirgraph 전담 LoRA(GT constraints 검증), Agent2/결정론 실행과 분리 → fabrication 근본원인(얽힘) 구조적 제거. `RUNG1_SOURCE_LADDER_DESIGN.md` §11 = **유일 헤드라인·미-killed**.
- **DPO/RFT (decision-axis B)** = 모델 판단과 결정론 verifier의 *차이*를 음성신호로 교정 = LOCK escape(emission 재답습 아님).

**⇒ 본 설계의 "NL→구조"는 살아있는 두 경로로만 한다**: ① gather-style(alive) ② 2-agent Agent1(분리, 미-killed). **decision-emission terminal(C)은 금지.** 검증은 Guard-2식 독립검증(emit된 구조 vs GT, OVER/UNDER). SFT-positive가 구조생산에 저항하면 → **Track B(DPO/RFT)로 강등**(C 변종 금지).

**decision-axis 재좌표 (이 설계의 위치)**:
| axis | 정의 | 상태 | 이 설계서에서 |
|---|---|---|---|
| **A** 결정론 offload | 시스템이 기록된 raw 결과로 permitted 계산 | LIVE(=Exp-5 stack) | **결정론 baseline = 공격 대상**. LLM-Modulo 인용. 단독 novelty 약 |
| **B** verifier-DPO/RFT 내재화 | 모델 판단−verifier 차이를 음성신호로 weight 학습 | 미시험 | **adapter-only>0 만드는 길 = "학습이 결정론을 내재화·전이"**(§6 E5) |
| **C** self-emit | terminal에 진리도출 emit | **NULL(7B)** | **금지(재답습 안 함)** |
| **2-agent** | Agent1 NL→구조 + Agent2 실행 | 미시험(헤드라인) | **NL→구조 load-bearing의 정당 구현(§6 E1)** |

## §4. ★연구 동기 — LLM만의 대체불가 장점 (결정론이 원리상 못하는 것)
현장은 SOPBench-기본과 D1–D3에서 다르고, 각 차이마다 결정론은 *사람 노동으로만* 대응한다:
1. **NL→구조 컴파일**: 현장 정책은 NL 산문(규정·10k줄 SOP)으로 온다. 결정론은 사람이 절차를 코드로 작성해야. **LLM = NL→dirgraph 자동 컴파일.** (= Logic-LM/PAL의 semantic-parsing 역할.)
2. **지저분한 멀티턴 NL 요청**: 모호·불완전·턴 분산·동의어/패러프레이즈. 결정론 인자추출이 깨짐. **LLM = NL 이해 + 명료화 대화.**
3. **변경 robustness**: 정책 조건 add/remove, 도구 rename/add/remove가 상시. 결정론 = 변경마다 코드 경로 찾아 재작성. **LLM = 바뀐 NL/도구리스트만 교체(코드 0).**
4. **미지·롱테일 도메인**: 모든 도메인에 프로그램 미리 못 씀. **LLM = ABox-swap 재학습0 전이.**
5. **정책 규모**: 10k줄을 per-decision context에 못 넣음. **LLM = 압축 컴파일·내재화.**

**핵심 프레이밍**: LLM의 가치 = "지저분/변화/자연어 현실 → 결정론 엔진이 먹을 구조" 변환. **결정론과 정확도로 싸우지 말고, 노력·변경·자연어 축으로 싸운다** — 거기선 결정론이 원리상 못 따라온다.

## §5. ★기존 연구 동향 비교 + 우리 비판 극복법
| 계열 | 주장 | 우리와의 관계 / 극복 |
|---|---|---|
| **LLM-Modulo** (Kambhampati ICML'24, `2402.01817`) | LLM은 plan·self-verify 못함; 외부 건전 검증기가 soundness, LLM은 후보 생성기 | **A축 인용 권위**. 단 LLM-Modulo는 *LLM이 생성*. 우리가 검증기에 생성(그래프 재구성) 시키고 LLM이 ~0이면 *역전*=인용 안 삶 → LLM을 생성(NL→구조) 자리로 되돌려야 정당 인용 |
| **Logic-LM / PAL / SatLM / PoT / LINC** (`2305.12295`/`2211.10435`/SatLM NeurIPS'23/`2211.12588`/LINC EMNLP'23) | LLM=NL→형식스펙(파싱), 솔버=건전 실행; **양면 ablation**(솔버 단독·LLM 단독 모두 실패) | **★핵심 템플릿.** 우리 방어 = 정확히 이 양면 ablation. **우리 delta**: 이들은 프롬프트(GPT)+솔버, 우리는 NL→구조 스킬을 *작은모델 weight에 학습·도메인/벤치 전이* |
| **RoG** (Luo ICLR'24, `2310.01061`) | 7B가 relation-path *구조 생성* + GT 그래프 distill | 방법론 최인접. **차별**: RoG=기존 KG 스키마서 path-finding / 우리=NL서 절차 graph-construction; RoG=KG마다 재instruction-tune(약점) / 우리=재학습0 전이 |
| **OISA / CDP** (현장 시스템, `/workspace_common/CDP`) | 도메인-특화 파이프라인(코드 AST induce)으로 구조 제공, 도구변경 시 재구축 | **현장 결정론 baseline.** 차별 = Agent1이 구조를 NL서 *학습·전이*로 제공 → 도구/도메인 변경 시 **리스트·NL만 갱신(재학습0)** vs OISA per-domain 재구축. = §6 E2 비용 우위의 실세계 대조군 |
| **process-supervision / verifier-RFT** (PRM 등) | 중간단계 보상으로 추론 교정 | **B축 positioning**: 우리 B = verifier 차이를 음성신호 내재화. process-supervision은 통계우위 불명[Jia ICML'25] → 우리는 *전이*로 정당화 |
| **Self-Ask / compositionality gap** (`2210.03350`) + **LLM self-verify limits** (Kambhampati 2402.08115) | 서브Q는 답하나 *조합* 실패; LLM self-verify 불가 | 우리 Track C NULL의 *이론적 예언*. → 결정 자기-emit 포기·offload/2-agent 정당화 |

**극복 요지**: (a) 결정론-프로그램 공격 = Logic-LM 양면 ablation으로(§8); (b) "LLM 효용 없음" = §4 장점을 §6 regime서 측정; (c) LLM-Modulo 역전 비판 = 2-agent Agent1로 LLM을 생성 자리로 복귀; (d) "프롬프트면 충분(학습 불요)" = 작은모델 weight 내재화 + 전이 delta(RoG·Logic-LM 대비).

## §6. ★실험 — 현장↔SOPBench 차이를 regime으로 (상세)
각 실험: regime 정의 → 결정론 baseline → 지표 → 사전등록 성공기준 → Track/LOCK 매핑.

### E1. NL-only 입력 (D1 제거) — ★핵심 실험 (= 우리 thesis의 정직한 버전)
- **regime**: `constraints_original`(구조화 제약) **제거**, 입력 = **NL 정책 + 도구 API 설명만**. 구조는 누군가 생산해야.
- **arms**: ① 결정론(NL→구조를 *사람이 작성*) = per-domain authoring 필요 ② 우리(모델이 NL→구조 emit: gather-style 또는 2-agent Agent1, Guard-2식 독립검증 후 결정론 게이트(A)에 투입).
- **지표**: 공식 success + **도메인당 사람-작성 LOC**(결정론) vs **0**(우리, NL은 *발견*) + 모델이 구조화-입력 천장의 회복률(%).
- **성공기준(사전등록)**: 우리 NL-only success ≥ (구조화-입력 stack의 **X%**, X 사전등록 예: ≥60%) ∧ 결정론은 사람작성 0줄에서 ≈0%. = "LLM이 NL만으로 결정론이 사람 없이 못 만드는 구조를 회복."
- **Track/LOCK**: 살아있는 gather/Agent1 경로(§3). **decision-emission(C) 금지.** SFT-positive 구조생산 저항 시 → E5(B축)로.
- **⚠️정직 caveat**: NL→구조가 7B에 어려울 수 있음(gather는 학습됐으나 full NL→dirgraph는 더 큼). 그래서 *정확도 지배*가 아니라 **비용/효용 대조**(사람작성 vs 0)가 1급 주장. 천장 못 채워도 "결정론은 0(사람 없이)"이 결론을 만든다.

### E2. 변경 robustness (D3 제거) — ★가장 방어적
- **regime**: K개 perturbation 주입 — 정책 조건 add/remove, 도구 rename/add/remove(distractor 추가).
- **지표 (2D Pareto)**: x = 적응에 든 **사람 edit(Δ-LOC)**, y = 적응 후 success. 결정론 = (N edit, 성공) / 우리 = (0 edit = NL·리스트만 교체, 성공).
- **성공기준**: 우리 perturbation-후 success가 무편집으로 비-perturbation의 **≥Y%** 유지(Y 사전등록) ∧ 결정론은 편집 없이는 깨짐.
- **Track/LOCK**: A(offload)는 도구리스트만 갱신; gather/Agent1은 NL/desc 재매칭. = coworker plan "도구변경 robust 전이" 축(v1.36/v1.38) 정량화.

### E3. 멀티턴 messy user (D2 제거)
- **regime**: dummy-user → **user_sim**(`--user_model`) + paraphrase/모호/정보누락→명료화 요구. (벤치: τ²-bench가 기질적으로 이 축.)
- **지표**: success + 명료화-질문 정확도(필요시 cred 요청) + 패러프레이즈 강건성. 결정론 인자추출 vs LLM 대조.
- **성공기준**: 멀티턴 success가 결정론 인자추출 baseline 대비 유의 우위.
- **Track/LOCK**: NL 이해 = LLM 고유, 결정론 미해당. credential-binding(alive) 활용.

### E4. 정책 규모 / 미지 task (선택, stress)
- 대형 NL 정책(연결 또는 실 엔터프라이즈 SOP, 10k줄급) + 구조 스펙에 없는 unseen 조건(NL엔 있음). per-decision context 초과 → 컴파일 필수. 결정론은 코드경로 없음.

### E5. ★Track B — verifier-corrected DPO/RFT (adapter-only>0 만들기 = "결정론을 학습이 내재화")
- **regime**: 현 scaffold(A) 행동을 **음성신호(DPO: 결정론 게이트가 deny한 궤적 dispreferred / RFT: 게이트 차이 보상)** 로 weight 학습 → scaffold OFF에서도(adapter-only) 행동 보존되나?
- **지표**: **adapter-only success ↑**(현 ~0 → ?). adapter-only가 오르면 "학습 모델이 scaffold 행동을 내재화" = 결정론-프로그램 공격 직접 무력화 + novelty A↔B 전이.
- **성공기준(사전등록)**: adapter-only success가 base 대비 유의↑ ∧ 전이(held-out)서 유지. NULL이면 = "offload 필수"(LLM-Modulo 강화) — **양 결과 게재가능**.
- **Track/LOCK**: B축(음성신호=LOCK escape, emission 아님). C 아님.

## §7. ★Cross-benchmark 유용성 증명 (SOPBench + SOP-Bench + 제3벤치)
도메인-일반을 넘어 **벤치-일반**으로 = "학습된 NL→구조 스킬이 벤치 경계를 넘는다" + 각 벤치가 다른 현장 축을 커버.
| 벤치 | 성격 | 본 설계서 역할 |
|---|---|---|
| **SOPBench** (Zekun Li, `2503.08669`) | 7도메인, **구조화 제약** 제공, rule evaluator | 메커니즘 증명 substrate(학습원). E1에서 구조 제거해 NL-only로 변환 |
| **SOP-Bench** (Amazon, `2506.08119`) | 12도메인, **free-text NL SOP**(input/decision/outcome 메타·tool I/O·복잡도 7–8/10) | **NL-native 기질** = E1(NL→구조)의 자연 시험대 + **벤치횡단 전이 타깃**(SOPBench 학습→SOP-Bench ABox-swap 재학습0). 8관계 표현적합성 검증완(마스터 §1) |
| **τ²-bench** (Sierra, dual-control; cf. τ-bench `2406.12045`) | 멀티턴 NL 정책 대화, 실 고객서비스, user-sim | **E3(멀티턴 messy user) 기질.** 정책-준수 대화 = D2 제거 축 |
| (선택) **AppWorld** (ACL'24) | 복잡 agentic 도구사용·실앱 | E4 scale/복잡도 stress |

- **횡단 전이 매트릭스**: 학습=SOPBench(LODO/train-1) → 전이 평가=SOP-Bench·τ² ABox-swap **재학습0**. 헤드라인 = "한 벤치서 배운 스킬이 다른 벤치(다른 형식·다른 도메인)서 작동" = LODO보다 강한 주장.
- **각 벤치가 다른 축 증명**: SOPBench=메커니즘·구조화, SOP-Bench=NL→구조·규모, τ²=멀티턴 대화. 셋이 §4 장점 1·2·5를 분담 커버.
- ⚠️ τ2는 과거 "user_sim이 핵심실행"으로 피벗-아웃(마스터 §0.0)했으나, **본 설계서는 τ²를 E3 *전용*(멀티턴 NL 강건성 축)으로만** 쓴다(헤드라인 substrate 아님) — 역할 한정으로 그 mismatch 회피.

## §8. 베이스라인 (사전등록) — 양면 ablation
결정론-프로그램 공격은 **양면 ablation 표**로만 닫힌다(Logic-LM 템플릿):
| arm | 구성 | 기대(우리 가설) |
|---|---|---|
| **det-scripted** | scaffold(A) + **scripted-gather**(LLM 없음) | 구조화-입력서 강(천장); **NL-only(E1)·변경(E2)·멀티턴(E3)서 실패** |
| **det+base** | scaffold(A) + **base-7B**(SFT 없음) | SFT 기여 격리 |
| **adapter-only** | SFT 모델 + scaffold OFF | 모델 단독(현 ~0; E5로 올리는 게 목표) |
| **stack(ours)** | SFT + scaffold(A) | 구조화-입력 강(=Exp-5); **현장 regime서도 유지** |
| **human-author** | 결정론 프로그램 사람작성 | per-domain LOC/시간 = 비용 baseline(§9) |
- **닫힘 조건**: det-scripted ≫ ours (구조화-입력) 이지만 **det-scripted ≪ ours (NL-only/변경/멀티턴)**, 그리고 **adapter-only ≪ stack** 이되 E5가 adapter-only↑. = 두 컴포넌트(LLM·결정론) 각자 어떤 regime서 load-bearing인지 분리.

## §9. 지표 (3축)
1. **성능**: 공식 success(`evaluator.py:277`, tool_full, BOTH 금지, honest). 도메인·regime별.
2. **비용/노력**: per-domain 사람-작성 LOC + 변경 적응 Δ-LOC + amortization(우리 1회 고정비 scaffold+SFT vs 결정론 선형 도메인×LOC; crossover 도메인 수).
3. **robustness**: perturbation Pareto(edit vs success).
- ⚠️ **성능 < 결정론이면 "지배" 아님 = trade-off 프레이밍**(0 작성·무재학습·변경robust 대가로 정확도 X% 양보) — Pareto frontier로 정직 보고.

## §10. 정직 범위 / threats
1. **found vs authored vs benchmark-inherited 분류 필수**: 우리 per-domain 산출물(ontology=induced, getter_map=auto-derived, 정책=found)을 "작성 0"이라 쓰려면 각각이 NL/API서 자동도출인지 vs 벤치 구조 상속인지 명시(induce는 벤치 구조 추출이라 "0"이 일부 상속). 안 하면 "너희 0도 벤치 덕" 반박.
2. **offline 측정 함정**(§8 RETRACTED 교훈): forced-ACT full-success는 constraint/database 게이트서 artifact → **실제 rollout eval로만**. E1–E5 전부 live rollout.
3. **검정력**: should_T effective n 작음 → seed·도메인-mix 고정·사전등록.
4. **NL→구조 난이도 미지**: gather는 학습됐으나 full NL→dirgraph는 더 큼 → E1은 *비용 대조*가 1급(정확도 지배 아님), 저항 시 E5(B).
5. **벤치 induce 품질**: SOP-Bench 8관계 적합성(마스터 §1)·τ² 역할한정(§7).

## §11. 실행 순서 (zero-GPU 우선, 학습 큐·GPU 충돌 없이)
1. **(L0, 지금·무GPU) 비용 baseline 측정**: SOPBench 클론서 도메인당 결정론 작성 LOC 카운트(`<domain>_assistant.py`+constraint 정의+getter) + 우리 repo `if domain==` 분기 grep(B-1=0 확인) + 산출물 found/authored/inherited 분류. → §9 비용축 첫 수치 + §1 공격의 정량 반박.
2. **(L0) E1/E2 harness 설계**: constraints 제거 토글(NL-only) + perturbation 주입기 + scripted-gather baseline 구현. apply_two_stage_patch에 통합.
3. **(GPU, 학습 큐 후) E1 NL-only** on bank → 양성 시 SOP-Bench로 확장.
4. **E2 perturbation** (무재학습, A축 toggle) → Pareto.
5. **E3 멀티턴**(user_sim) on τ² + SOPBench.
6. **E5 Track B**(DPO/RFT) = adapter-only 올리기 (가장 비쌈, 마지막).
> 1–2는 GPU 불요 = 학습 큐와 병행 가능. 3–6은 큐 완료 후.

## §12. 리뷰 훅 (박제 전 확인)
- [ ] E1이 죽은 Track C(decision-emission)를 안 되살리나? (gather/Agent1·독립검증·decision-emit 금지 확인)
- [ ] 양면 ablation(§8)이 det-scripted·adapter-only 양쪽을 *실패*시켜 두 컴포넌트 load-bearing 분리하나?
- [ ] 비용축(§9)이 found/authored/inherited(§10-1)를 정직 분류했나?
- [ ] 성능<결정론 시 trade-off로 정직 보고하나(지배 주장 아님)?
- [ ] τ² 역할이 E3 전용으로 한정됐나(헤드라인 substrate 아님)?
- [ ] 벤치횡단 전이가 재학습0(ABox-swap)으로 정의됐나?
