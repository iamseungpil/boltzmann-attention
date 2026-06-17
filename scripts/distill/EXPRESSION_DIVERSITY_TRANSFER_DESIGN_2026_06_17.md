# 표현 다양성 → 절차-추상 전이: 비용 효율적 추상 교육의 정량 기준 (설계) — 2026-06-17

> 상위 = `C8_PROCEDURE_ROUTING_TRANSFER_DESIGN` · `feedback-expression-diversity-required-for-transfer` · `NL_PROCEDURE_OFFLOAD_THEORY §7e/§7f`. 동기 = 사용자(2026-06-17): K 추적 + 표현 orthogonality + 최적 학습데이터 + **비용 효율적 추상 교육 기준**.

## 0. 왜 (기여 진술)
**빅모델은 통제되지 않은 대량 데이터의 *우연한* 표현 다양성으로 깊은 추상을 학습한다 — brute-force·비효율.** 우리는 표현 다양성 **D를 측정·최적화**(직교 선택)해서, **깊이별 최소 임계 D\*(depth)로 목표 추상을 비용 효율적으로 가르치는 정량 기준**을 제공한다. = 작은 on-prem 모델이 무한 데이터 없이 깊은 절차 추상을 획득하는 길(주권·비용 thesis와 직결).

근거(실증): 단일-템플릿 SFT(D≈0)는 표면매핑 주입→τ² 역전이(`M_A_RESULTS §17b`). ⇒ 전이엔 다양성 필수. 남은 질문 = *얼마나·어떻게*.

## 1. 명제 (반증가능)
1. **전이 = f(D), not f(K)**: 전이는 표현 *개수* K가 아니라 표현 *다양성* D(임베딩 span)가 결정. K개 유사표현 < 소수 직교표현.
2. **깊이별 무릎 D\*(depth)**: 깊은 절차(comparative·중첩)일수록 표면매핑 유혹↑ → 더 큰 D\* 필요. filter는 작은 D\*에 포화.
3. **직교 효율 r**: 같은 K에서 orthogonal/axis-balanced 선택이 random 대비 D를 r배 채워 전이를 앞당김. = 비용 효율의 정량.

## 2. 표현 축 분해 (orthogonality의 구조)
"argmax를 말하는 법"은 단일 차원이 아님 — 직교 축들의 곱공간:
- **어휘축 L**: most / largest / highest / maximum / top / greatest / biggest
- **구문축 S**: "the one with the most X" / "the X-est" / "whichever maxes X" / "max by X" / "X is highest"
- **화용축 P**: 명령(select/pick) / 요청(I want/give me/I'd like) / 평서(the one that…) / 거래(exchange for/swap to)
- **우회축 R**: 직접("highest X") / 부정("none has more X") / 비교전체("beats all in X") / 역할("the X-leader")
표현 e = (L,S,P,R) 좌표. **진짜 다양성 = 축들을 *독립* 커버**(단일축 K↑는 한 차원만). depth는 별도 축(과제 구조)·D\*는 depth의 함수.

## 3. 측정
- **다양성 D**: 표현 집합을 임베딩(sentence-emb) → **effective rank**(특이값 분포의 유효차원·`exp(entropy(σ̂))`) 또는 평균 pairwise cosine distance / span volume. K가 아니라 D를 x축으로 plot.
- **전이 Y**: (a) **τ² op-라우팅 정확도** + op-슬롯 붕괴율(`op∈{exchange,replace,…}` 비율) — 표면매핑 잔존 측정. (b) held-out **OOD 표현**(학습 축조합에 없는 표현) op-recognition.
- **축 커버리지**: 학습 표현이 L×S×P×R 격자의 몇 cell을 채웠나(독립 축 커버 직접 측정).

## 4. K-선택 방법 (최적 학습데이터)
고정 예산 K에서 D 최대화:
- **random-K**: 표현 풀서 무작위 K (baseline).
- **orthogonal-K**: 임베딩 공간 **k-center greedy**(max-min distance) 또는 **DPP**(determinantal·다양성 사전) — 서로 가장 동떨어진 K.
- **axis-balanced-K**: L×S×P×R 격자를 균등 커버(factorial) — 각 축 독립 최대.
- 비교 = 같은 K에서 D(achieved)와 전이 Y. **orthogonal/axis ≫ random in Y/K** 예상 = 직교 효율 r.

## 5. 실험 = D-전이 곡선
- **K ∈ {1,2,4,8,16,32,64}** × **선택 {random, orthogonal, axis}** × **op-depth {filter,argmax,rank,comparative}**.
- 각 셀: K개 표현으로 데이터 생성(어휘/스키마는 등방 유지·표현만 K-제어) → LoRA 학습 → 전이 Y 측정 + D 측정.
- **산출**: Y vs D 곡선(전 셀 한 plot)·**무릎 D\*(depth)** 표·**r = (random이 같은 Y 내는 K) / (orthogonal K)**.
- **논문 그림**: x=D, y=전이, 색=depth, 마커=선택방법. "D>D\*(depth)서 깊은 전이·직교가 곡선 좌측이동(저비용)."

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
- 임베딩 D는 *대리지표*(임베더 의존) — effective rank/축커버리지/pairwise를 *함께* 보고(단일지표 단정 금지·[[feedback-no-fundamental-claims-from-convenience-data]]).
- 합성 표현 풀이 실벤치(τ²) 표현 분포를 다 못 덮음 — τ² 전이는 *외삽* 시험(축 커버가 τ² 표현을 포함하는 정도 별도 측정).
- depth-구조 다양성(과제 형태)도 별 축 — 1차는 표현축 고정-depth, 2차는 구조축.
