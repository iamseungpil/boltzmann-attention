# 표현 다양성 → 절차-추상 전이: 비용 효율적 추상 교육의 정량 기준 (설계) — 2026-06-17

> 상위 = `C8_PROCEDURE_ROUTING_TRANSFER_DESIGN` · `feedback-expression-diversity-required-for-transfer` · `NL_PROCEDURE_OFFLOAD_THEORY §7e/§7f`. 동기 = 사용자(2026-06-17): K 추적 + 표현 orthogonality + 최적 학습데이터 + **비용 효율적 추상 교육 기준**.

## 0. 왜 (기여 진술)
**빅모델은 통제되지 않은 대량 데이터의 *우연한* 표현 다양성으로 깊은 추상을 학습한다 — brute-force·비효율.** 우리는 표현 다양성 **D를 측정·최적화**(직교 선택)해서, **깊이별 최소 임계 D\*(depth)로 목표 추상을 비용 효율적으로 가르치는 정량 기준**을 제공한다. = 작은 on-prem 모델이 무한 데이터 없이 깊은 절차 추상을 획득하는 길(주권·비용 thesis와 직결).

근거(실증): 단일-템플릿 SFT(D≈0)는 표면매핑 주입→τ² 역전이(`M_A_RESULTS §17b`). ⇒ 전이엔 다양성 필수. 남은 질문 = *얼마나·어떻게*.

## 1. 명제 (반증가능)
1. **전이 = f(D), not f(K)**: 전이는 표현 *개수* K가 아니라 표현 *다양성* D가 결정. **★증명은 곡선이 아니라 matched-D 디커플링 대조로**(리뷰 보강2): K와 D는 상관(고정 선택법서 K↑→D↑)이라 곡선만으론 분리 불가. **사전등록 결정대조 = orthogonal-K₁ vs random-K₂를 *D가 같아지는 지점*서 비교 → Y 같으면 "전이는 K 아닌 D" 증명**(D 다른 K는 곡선·D 같은 K는 대조).
2. **깊이별 무릎 D\*(depth)** *또는 부재*: 깊은 절차일수록 큰 D\* 필요. **★단 D\* 존재를 가정 말 것**(리뷰 보강5·정련): 곡선이 3-way 구분 — `diversity-curable`(D↑→Y→1·D\* 정의됨) / `capability-capped`(Y가 D 무관 <1 포화·D\* 미정의) / `vocab-gated`(gloss로는 1.0인데 gloss-free 다양성으론 미달). comparative는 §15-bis서 gloss로 0→1.0 회복 = capability 있음 → **capped 아닌 gated 공산**(내재화엔 다양성+생성원-정의 필요). 셋 중 무엇인지가 1급 결과.
3. **직교 효율 r**: 같은 K에서 orthogonal/axis-balanced가 random 대비 전이를 앞당김. **★단 "직교=高D"는 측정 말 것**(리뷰 보강3·순환): k-center가 임베딩 spread로 뽑고 D를 그 임베딩 eff-rank로 재면 D-achieved↑는 tautology(선택=측정). **정보값은 오직 Y**(高-D-by-construction가 전이를 올리나).

## 2. 표현 축 분해 (orthogonality의 구조)
"argmax를 말하는 법"은 단일 차원이 아님 — 직교 축들의 곱공간:
- **어휘축 L**: most / largest / highest / maximum / top / greatest / biggest
- **구문축 S**: "the one with the most X" / "the X-est" / "whichever maxes X" / "max by X" / "X is highest"
- **화용축 P**: 명령(select/pick) / 요청(I want/give me/I'd like) / 평서(the one that…) / 거래(exchange for/swap to)
- **우회축 R**: 직접("highest X") / 부정("none has more X") / 비교전체("beats all in X") / 역할("the X-leader")
표현 e = (L,S,P,R) 좌표. **진짜 다양성 = 축들을 *독립* 커버**(단일축 K↑는 한 차원만). depth는 별도 축(과제 구조)·D\*는 depth의 함수.

## 3. 측정 (★리뷰 보강1 — D-지표 타당성을 게이트로)
- **★Y(전이)가 유일한 *정보값***(리뷰 보강3·정련1): 완전 비순환 D-지표는 없다(임베딩-D=관찰·confound / 축-커버리지=축-가정 의존). ⇒ 모든 D-지표는 *설명변수 후보*로만·**"어떤 D-지표가 Y를 가장 잘 예측하나"를 사전 고정 아닌 *결과*로** 보고.
- **주 다양성 지표 = 축-커버리지**(L×S×P×R 격자 채운 cell 수): 설계로 *통제되는* causal 측정. 임베딩-D(eff-rank·pairwise)는 *보조*(관찰·confound). **복수 지표 곡선 일치 확인**(불일치면 "f(D)"는 지표-의존 = 그 자체 결과).
- **★게이트(보강1·4): 비싼 sweep 전 파일럿으로 "D가 Y와 상관이 있긴 한가" 먼저 확인** — 상관 없으면 지표부터 교체(sweep 무의미).
- **전이 Y**: (a) **τ² op-라우팅 정확도** + op-슬롯 붕괴율 (b) held-out **OOD 표현** op-recognition (大-n·아래 §5).

## 4. K-선택 방법 (최적 학습데이터)
고정 예산 K에서 D 최대화:
- **random-K**: 표현 풀서 무작위 K (baseline).
- **orthogonal-K**: 임베딩 공간 **k-center greedy**(max-min distance) 또는 **DPP**(determinantal·다양성 사전) — 서로 가장 동떨어진 K.
- **axis-balanced-K**: L×S×P×R 격자를 균등 커버(factorial·임베딩-무관) — 비순환 다양성 notion(보강3). **★단 이것도 우리 축 정의가 직교·완전하다는 가정 의존**(정련1) — 완전 독립 아님.
- **★순환성 명시**(보강3): kcenter는 임베딩 spread 최대화로 뽑으니 "kcenter=高 D-achieved"는 *tautology*(선택=측정). **D-achieved를 결과로 읽지 말 것**·**비교 정보값 = Y만**. axis-balanced(임베딩-무관 선택)의 임베딩-D vs Y 일치가 비순환 교차검증(보강1·3 합류).

## 5. 실험 = D-전이 곡선 (★보강4 — tier·seed·大-n)
- **★파일럿 tier 먼저**(보강4·[[feedback-zero-cost-diagnosis-strongest-case]]): 1 depth × 3 선택 × few K로 (i) **D가 Y와 상관 있나**(보강1 게이트) (ii) **matched-D 디커플링 대조**(보강2: orthogonal-K₁ vs random-K₂ at same D → Y 같나) 먼저 검증 → **신호 있을 때만 full grid**. 신호 없으면 지표/설계부터 교체.
- **full grid (신호 후만)**: K ∈ {1,2,4,8,16,32} × 선택 {random,axis,kcenter} × depth {filter,argmax,rank,comparative}. **★random은 seed band 필수**(×3·확률적). = ~250 run → tier 게이트로 정당화.
- **★곡선은 大-n OOD-synth 전이로**(보강4): held-out *표현*(학습에 없는 축조합)·n≥100. **τ²-29(±9pp)는 anchor**(무릎 탐색 아님·곡선은 합성). P4 보강B와 동일 분리.
- **산출**: Y vs D 곡선(복수 D-지표)·**무릎 D\*(depth) *또는 부재*(§1.2 3-way)**·**디커플링 대조 결과**(f(D) vs f(K))·r.

## 6. 기여 (3겹)
1. **전이는 K 아닌 D가 결정** (다양성의 질>양).
2. **깊이별 D\*(depth) 무릎** = 추상 깊이별 데이터 예산 처방.
3. **직교/축-균등 선택이 r배 효율** = 비용 효율적 추상 교육의 *방법*.
⇒ **빅모델 brute-force 다양성을 *측정된 최소 다양성*으로 대체** → 작은 on-prem 모델이 깊은 절차를 저비용 학습(주권 thesis 완성 축).

## 7. 도구 (구현 계획)
1. `synth_expr.py`: 축 분해 표현 풀(L×S×P×R) + K-선택(random/kcenter/axis) + render(K개 표현으로 NL 생성). depth별.
2. `expr_diversity.py`: 표현 집합 → D(effective rank·pairwise dist) + 축 커버리지. 임베딩 = 로컬 sentence-emb(인프라 확인) or vLLM embedding.
3. `kshot_sweep.sh`: K×방법×depth 배치(데이터→학습→τ²+OOD 전이→D 기록). preempt-safe per-cell json.
4. `expr_diversity_summary.py`: Y-vs-D 곡선·D\*(depth)·r 집계.
- 선행 = 현 DIV_ep3(단일 다양화·K-max 근사) 결과가 "다양화→τ² 전이 회복" 1차 확인. 그 위에 K-sweep로 *곡선* 산출.

## 8. 정직 경계
- 임베딩 D는 *대리지표*(임베더 의존) — Y가 유일 정보값·복수 D-지표 교차(§3·단일지표 단정 금지·[[feedback-no-fundamental-claims-from-convenience-data]]).
- **★anti-targeting 하드룰**(리뷰): 표현 풀 L×S×P×R은 **일반 축 분류에서 τ²-blind 설계**·**τ² 표면형 역설계 금지**([[feedback-thesis-tbox-transfer-direction]] 금지선). 풀 설계 후 *별도로* τ² 커버율 측정(풀이 τ²를 포함하는 정도)·풀에 τ²를 *맞추지* 않음.
- 합성 표현 풀이 τ² 표현 분포 다 못 덮음 — τ² 전이는 *외삽* 시험.
- depth-구조 다양성(과제 형태)도 별 축 — 1차 표현축 고정-depth·2차 구조축.

## 9. ★리뷰 반영 요약 (2026-06-17)
사용자 리뷰 5보강+2정련 반영: (1)D-지표 타당성 게이트화·**축-커버리지 주 지표·Y 유일 정보값**(§3) (2)**matched-D 디커플링 대조 사전등록**(§1.1·§5) (3)**순환성 명시**(kcenter D-achieved=tautology·§4) (4)**파일럿 tier·seed band·大-n OOD 곡선**(§5) (5)**D\* 부재 가능성**=3-way `curable/capped/gated` 사전등록(§1.2). 정련: 완전 비순환 D-지표 없음→Y 유일 정보값·"어떤 D가 Y 예측"을 결과로(§3); comparative는 §15-bis gloss 회복→capped 아닌 **gated** 공산(§1.2). anti-targeting: 풀 τ²-blind(§8).
