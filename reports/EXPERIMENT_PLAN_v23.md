# 실험 계획 v23: Lloyd vs Uniform — PPL을 넘어서 (2026-04-09)

## 핵심 발견 요약

### 확정된 사실 (반박 불가)

| # | 사실 | 증거 |
|:-:|------|------|
| 1 | Pre-RoPE PCA는 Class C 내 MSE 최적 | 증명 + 624/624 |
| 2 | Per-head > Shared PCA | Fischer 증명 + 46.3% |
| 3 | 3-bit에서 PCA PPL 4/4 최선 | 49K test |
| 4 | MMLU: PCA +9.2%p | Qwen 2-bit |
| 5 | Lloyd L∞ > Uniform L∞ (624/624) | 같은 데이터에서 측정 |
| 6 | **수정 Lloyd PPL > Uniform PPL (3모델 × 3비트, 전부)** | 49K test, mean-centering 수정됨 |

### 새 발견: Lloyd는 올바르게 구현해도 Uniform보다 PPL이 나쁘다

| 모델 | 2-bit | 3-bit | 4-bit |
|------|:-----:|:-----:|:-----:|
| Qwen | 1.03× | 1.08× | 1.05× |
| Mistral | 2.46× | ? | ? |
| Llama | 4.25× | 2.87× | 1.58× |

이유: Lloyd는 MSE를 줄이지만 L∞ (최악 오차)를 증가시킨다. Softmax가 L∞에 민감하므로 PPL이 악화된다.

### 미해결 질문

**"Lloyd가 PPL에서 나쁘면, MMLU에서도 나쁜가?"**
→ PPL은 모든 토큰의 평균. MMLU는 핵심 토큰의 정확성.
→ Lloyd의 tail 오차가 MMLU에도 영향을 주는지 미확인.

## 실험 계획

### Phase 1: Mistral 나머지 비트 결과 수집 (진행 중)
- E8 GPU 0에서 실행 중
- 3-bit, 4-bit 결과 대기

### Phase 2: 핵심 미해결 실험

#### 실험 A: Lloyd vs Uniform의 MMLU 비교
```
의도: PPL에서 Lloyd가 나쁘다면, 실제 task에서도 나쁜가?
가설: Lloyd의 tail 오차가 MMLU에도 영향을 줌 → Lloyd MMLU < Uniform MMLU
검증: Qwen 2-bit에서 Lloyd MMLU vs Uniform MMLU (PCA 회전 동일)
metric: MMLU 5-shot accuracy
기대: 
  H1 (Lloyd MMLU도 나쁨): "MSE 최적화는 PPL도 MMLU도 해친다" → 강한 negative result
  H2 (Lloyd MMLU는 비슷): "PPL과 MMLU가 다른 것을 측정" → PPL 지표의 한계 발견
```

#### 실험 B: Clipped Lloyd (새 방법)
```
의도: Lloyd의 MSE 이점을 살리면서 L∞를 제한할 수 있나?
방법: 바깥 centroid를 data min/max에 고정, 안쪽만 Lloyd 최적화
가설: Clipped Lloyd PPL < Uniform PPL (MSE 이점이 살아남)
검증: 3모델 × 2-bit에서 Clipped Lloyd PPL 비교
metric: PPL + L∞ + MSE 동시 측정
```

#### 실험 C: Attention-Optimal Clipping (Method 4)
```
의도: 이론에서 직접 도출된 방법이 PPL을 개선하나?
방법: per-PCA-dim clip range를 eigenvalue × σ_q에 비례하게 설정
가설: 중요 차원의 tail 보존 → PPL 개선
검증: Mistral 2-bit에서 Method 4 PPL < Uniform PPL
metric: PPL
```

### Phase 3: 논문 작성

결과에 따른 논문 방향:

| Phase 2 결과 | 논문 |
|-------------|------|
| Clipped Lloyd > Uniform > Lloyd | "L∞-aware quantizer" 방법 논문 |
| Uniform > Clipped Lloyd > Lloyd | "왜 MSE 최적화가 실패하는가" 이해 논문 |
| Method 4 > 모두 | "이론 → 진단 → 방법" 완전한 논문 |

## 노드 상태

| 노드 | GPU | 현재 작업 |
|------|:---:|----------|
| **E8** | 0: Mistral 3,4-bit | **Lloyd vs Uniform** |
| **E8** | 1: Qwen **완료** | 유휴 |
| **E8** | 2: Llama **완료** | 유휴 |
| **E8** | 3: 유휴 | Phase 2 실험용 |
| TOPS | 4× busy | 다른 작업 |
