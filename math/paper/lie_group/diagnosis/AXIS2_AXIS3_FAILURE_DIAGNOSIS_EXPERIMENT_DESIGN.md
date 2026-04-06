# 축 2·축 3 실패 원인 진단 실험 설계서

**문서 버전**: v1
**작성일**: 2026-04-06
**대상**: NeurIPS 2026 제출 논문 Limitation 섹션 보강 및 Rebuttal 대비
**목표**: V3 검증 보고서에서 발견된 두 이상 현상(축 2 Lloyd-Max PPL 재앙, 축 3 WF floor=1 붕괴)의 근본 원인을 5개 가설 중 하나 이상으로 특정한다.

---

## 0. 배경 및 문제 정의

V3 검증 보고서(2026-04-04)에서 확인된 두 이상 현상:

| 현상 | 이론 예측 | 실측 결과 |
|---|---|---|
| **축 2 실패** | Lloyd-Max MSE 이득 3.5× → PPL 개선 | Llama 2-bit PPL 10.14 → **65.46** (6.5× 악화) |
| **축 3 부분 불일치** | Shannon WF가 uniform 대비 개선 | Qwen 2-bit WF(floor=1) **11.26** (uniform 7.98 대비 41% 악화), floor=2에서만 회복 |

두 현상 모두 "MSE 최적 = PPL 최적"이라는 암묵적 가정이 깨진 것으로, 본 실험의 목적은 **어떤 수학적 메커니즘이 이 불일치를 만드는지**를 특정하여 논문의 이론적 적용 경계를 정확히 기술하는 것이다.

---

## 1. 가설 요약

### 축 2 가설

| ID | 가설명 | 핵심 주장 | 예측 메커니즘 |
|---|---|---|---|
| **A1** | 쿼리 메트릭 미스매치 | 구현된 Gaussian Lloyd-Max는 Σ_q = I를 가정하지만 실제 Σ_q는 강하게 이방적 | Lloyd 셀 경계가 쿼리 이방성과 악성 정렬 |
| **A2** | 헤비테일·싱크 클리핑 | Lloyd는 가우시안 꼬리 바깥의 attention sink 토큰을 강제 클리핑 | 싱크 키 왜곡 → softmax 질량 이동 |
| **A3** | L∞ vs L² 증폭 | Lloyd는 평균 MSE를 줄이는 대신 최악-케이스 오차를 키움 | softmax Lipschitz가 최악-케이스에 지수 민감 |

### 축 3 가설

| ID | 가설명 | 핵심 주장 | 예측 메커니즘 |
|---|---|---|---|
| **B1** | 저속 WF 공식 오류 | Shannon WF는 연속 rate 점근이며 b=1에서 Gersho 공식이 깨짐 | 1-bit 잔차 분산 0.363σ² (공식 예측 0.25σ²보다 45% 나쁨) |
| **B2** | 정수 제약 WF | 정수 비트 할당의 rate-distortion 곡선이 convex hull 아래로 떨어짐 | 최적 b_j ∈ {0} ∪ {≥2}의 이분법적 구조 |
| **B3** | Softmax 임계 비트 | 1-bit 잔차가 softmax Lipschitz 임계를 넘김 | logit 오차 O(1) → softmax 고온 영역 진입 |

---

## 2. 실험 설계 원칙

1. **공통 베이스라인**: 모든 실험은 Pre-RoPE PCA + 기존 V3 파이프라인 위에서 **하나의 변수만** 바꾼다. Ablation의 인과성을 보장하기 위함.
2. **3모델 동시**: Qwen2.5-7B, Llama-3.1-8B, Mistral-7B-v0.3. 3모델 간 R_aniso 차이(4.27 / 7.97 / 131.62)가 가설 판별에 직접적으로 쓰인다.
3. **음성 결과도 기록**: 각 가설이 기각되는 경우에도 결과는 Limitation에 기여.
4. **재현성**: 모든 random seed 고정. 캘리브레이션 토큰 크기는 V3과 동일(2048 토큰, WikiText-2).
5. **1차 metric은 PPL**: MSE는 보조. 본 실험의 목적 자체가 "MSE-PPL gap의 원인"을 찾는 것이므로 PPL이 판정 기준이다.

---

## 3. 실험 E1: Mahalanobis Lloyd-Max (가설 A1 검증)

### 3.1 동기

V3에서 구현된 "Gaussian Lloyd-Max"는 사실상 논문 정리 6.19.8(MK)의 **Σ_q = I 특수 사례**이다. 실제 MK는 쿼리 공분산을 반영한 가중 MSE를 최소화한다:

```
Q_MK*(x) = argmin_{c ∈ C} (x - c)^T Σ_q (x - c)
```

Pre-RoPE PCA 기저에서 Σ_q를 대각화한 뒤 각 차원에 대해 **쿼리 가중치 w_j = (Σ_q)_jj 기반으로 스케일된 Lloyd-Max**를 적용한다. 만약 A1이 맞다면, MK는 Uniform PPL을 이겨야 한다.

### 3.2 구현

```python
# 의사코드
def mahalanobis_lloydmax(k, U_pca, Sigma_q_eff, bits):
    # 1. Pre-RoPE PCA 기저로 회전
    k_rot = U_pca.T @ k

    # 2. 쿼리 공분산을 같은 기저에서 측정
    Sigma_q_rot = U_pca.T @ Sigma_q_eff @ U_pca

    # 3. 대각 근사 (블록별로는 full Σ 유지 가능)
    w = diag(Sigma_q_rot)  # shape: [d_head]

    # 4. 차원별 가중 Lloyd-Max
    #    min_c E[w_j * (k_rot_j - c)^2]
    #    => 표준 Lloyd-Max (가중치는 스케일만 바꿈)
    for j in range(d_head):
        codebook_j = gaussian_lloydmax(sigma=sqrt(lambda_j), bits=bits)
        k_quant[j] = quantize(k_rot[j], codebook_j)

    # 5. 비트 할당: w_j · λ_j^{1/2} 비례 water-filling
    #    (쿼리 가중 왜곡 D_j = w_j · c(b_j) · λ_j · 2^{-2b_j})
    ...
```

**핵심 차이점**: 
- V3의 Gaussian Lloyd: 비트 할당이 λ_j만 고려
- MK: 비트 할당이 **w_j · λ_j** 고려 (쿼리가 자주 보는 방향에 비트 집중)

### 3.3 Σ_q 추정 프로토콜

- **GQA 대응**: Σ_q^eff = (1/G) Σ_{g=1}^G Σ_q^(g) (G개 쿼리 헤드 평균)
- **캘리브레이션**: 2048 토큰 × 전 레이어에서 쿼리 벡터 추출 → 헤드별 Σ_q 계산
- **레이어·헤드 독립**: 112 × n_layers 개 Σ_q 행렬을 개별 저장

### 3.4 실험 매트릭스

| 모델 | 비트 | 방법 | 측정 |
|---|---|---|---|
| Qwen / Llama / Mistral | 2, 3 | MK (Axis 1+2 완전체) | PPL, MSE, attention MSE (tr(Σ_q·ΔΣ_k)) |
| (대조군) | 2, 3 | Gaussian Lloyd (V3 기존) | 동일 |
| (대조군) | 2, 3 | Uniform (V3 기존) | 동일 |

총 3모델 × 2비트 × 3방법 = **18 실험**. 예상 소요: 4시간.

### 3.5 판정 기준

| 결과 | 판정 |
|---|---|
| MK PPL < Uniform PPL (3모델 모두 2-bit) | **A1 확정**: 축 2 실패는 쿼리 메트릭 미스매치. 논문 축 2 클레임 완전 복구. |
| MK PPL < Gaussian Lloyd PPL (개선 있으나 Uniform 미달) | A1 부분 지지. 다른 가설과 중첩 가능. |
| MK ≈ Gaussian Lloyd (차이 없음) | **A1 기각**. A2/A3로 넘어감. |

### 3.6 리스크

- Σ_q가 대각이 아니라 교차항이 지배적일 경우, 대각 근사만으로 부족. → 풀-매트릭스 MK 버전을 폴백으로 준비 (비용 증가).
- Σ_q가 Pre-RoPE PCA 기저에서 회전 기저와 동일하지 않아 혼동 가능. → 명시적으로 "두 기저가 분리된다"는 것을 코드에 주석화.

---

## 4. 실험 E2: 싱크 토큰 보존 (가설 A2 검증)

### 4.1 동기

Xiao et al. (2024)의 attention sink 현상은 초기 몇 개 토큰(보통 위치 0~3)에 어텐션 질량이 불균형하게 집중됨을 보인다. Gaussian Lloyd-Max는 가우시안 꼬리 바깥의 싱크 키를 셀 경계로 클리핑하여 큰 오차를 만들 수 있다.

### 4.2 구현

```python
def lloyd_with_sink_preservation(k, bits, n_sink=4):
    # 1. 처음 n_sink개 토큰은 fp16 유지
    k_sink = k[:n_sink]  # 양자화 안 함

    # 2. 나머지 토큰만 Lloyd-Max
    k_rest = gaussian_lloydmax(k[n_sink:], bits)

    return concat(k_sink, k_rest)
```

### 4.3 실험 매트릭스

| 모델 | 비트 | 방법 | 싱크 크기 |
|---|---|---|---|
| Qwen / Llama / Mistral | 2 | Lloyd + sink preserve | 0, 4, 16, 64 |

총 3모델 × 4설정 = **12 실험**. 예상 소요: 2시간.

### 4.4 판정 기준

| 결과 | 판정 |
|---|---|
| n_sink=4만으로 PPL catastrophe 소멸 (Llama 65→10 수준) | **A2 확정**: 헤비테일·싱크가 주원인. 논문 축 2에 "싱크 보존" 조건 추가. |
| n_sink=64까지 올려야 회복 | A2 부분 확정 + 일반 헤비테일 문제 존재. |
| 싱크 보존이 효과 없음 | **A2 기각**. |

### 4.5 분석

보조 분석으로 **레이어별 ||k|| 분포의 kurtosis**를 측정하여, kurtosis가 큰 모델(Llama?)에서 싱크 효과가 더 클 것이라는 가설을 교차검증한다.

---

## 5. 실험 E3: L∞ / Tail-conditional 양자화기 (가설 A3 검증)

### 5.1 동기

A3가 맞다면 "평균 MSE 최소화"가 아니라 "최악-케이스 오차 최소화"가 올바른 목적함수이다. 이를 직접 테스트하려면 다음 중 하나를 구현한다:

**방법 1: L∞ Lloyd (Chebyshev center)**
- 각 셀의 중심을 cell의 **최대 반경**을 최소화하도록 배치
- 구현: 반복적으로 cell boundary를 이동하되 최대 거리를 줄이는 방향으로

**방법 2: Tail-conditional MSE**
- 상위 5% 꼬리에서의 조건부 MSE를 최소화
- CVaR-style 목적함수

### 5.2 실험 매트릭스

| 모델 | 비트 | 방법 |
|---|---|---|
| Llama-3.1-8B (가장 재앙적) | 2 | L∞ Lloyd, Tail-MSE Lloyd, 기존 Gaussian Lloyd, Uniform |

단일 모델 × 2비트 × 4방법 = **8 실험**. 예상 소요: 3시간 (구현 포함).

### 5.3 판정 기준

| 결과 | 판정 |
|---|---|
| L∞ Lloyd PPL ≈ Uniform PPL (Uniform과 동급) | **A3 확정**: Uniform이 사실상 L∞ 양자화기에 가까우며, L² 최적화가 문제. |
| L∞ Lloyd > Uniform | A3 부분 지지 + 다른 요인 존재. |
| 차이 없음 | **A3 기각**. |

### 5.4 이론적 예측

Uniform 양자화기는 실제로 유한 지지 균등 분포에 대한 L∞-optimal 양자화기의 특수 사례이다. 따라서 "L² Lloyd가 L∞ Lloyd보다 나빠야 softmax 민감도로 진다"는 논리이므로, A3가 맞다면 L∞ Lloyd가 Uniform을 약간이라도 이겨야 한다. 만약 L∞도 Uniform과 같다면 A3가 아니라 다른 무언가(예: A2 싱크)가 지배적.

---

## 6. 실험 E4: 2의 배수 제약 WF (가설 B2 검증)

### 6.1 동기

정수 제약 WF 이론이 맞다면, 비트 할당을 **b_j ∈ {0, 2, 3, 4, ...}** (1은 금지)로 제한한 WF가 floor=2 WF와 동등하거나 더 나아야 한다.

### 6.2 구현

```python
def integer_wf_no_one(lambda_eigenvalues, total_bits, min_bits=2):
    # 연속 WF 먼저 실행
    b_continuous = shannon_wf(lambda_eigenvalues, total_bits)

    # 정수 반올림 + {0, ≥2} 제약
    b_int = round(b_continuous)
    b_int[b_int == 1] = 0  # 1-bit 차원은 0으로 강등
    # 예산 재분배
    freed_bits = total_bits - sum(b_int)
    # 가장 이득이 큰 차원에 추가 할당
    ...
```

### 6.3 실험 매트릭스

| 모델 | 비트 | 방법 |
|---|---|---|
| Qwen / Llama / Mistral | 2, 3 | WF(floor=1), WF(floor=2), WF(no-one), Uniform |

3모델 × 2비트 × 4방법 = **24 실험**. 예상 소요: 2시간.

### 6.4 판정 기준

| 결과 | 판정 |
|---|---|
| WF(no-one) ≈ WF(floor=2) in PPL | **B2 확정**: 1-bit 금지가 본질. floor=2는 그 특수 사례. |
| WF(no-one) < WF(floor=2) | B2 강하게 지지 + 추가 개선 여지. |
| WF(no-one) > WF(floor=2) | **B2 기각** (floor=2의 성공은 다른 이유). |

### 6.5 보조 분석

각 방법의 **실제 비트 할당 히스토그램**을 기록. floor=2 WF가 실제로 몇 개의 차원에 1-bit를 배정하고 있는지(답: 0개여야 함), WF(no-one)이 freed bits를 어디로 보내는지 시각화.

---

## 7. 실험 E5: Softmax 온도 스윕 (가설 B3 검증)

### 7.1 동기

B3가 맞다면 softmax 온도를 올리면(출력을 부드럽게) 1-bit 할당의 재앙이 완화되어야 한다. 반대로 온도를 내리면 floor=2조차 깨져야 한다.

### 7.2 구현

어텐션 계산을 다음과 같이 수정:

```python
attn = softmax(q @ k.T / (sqrt(d) * temperature))
```

단, **추론 시에만** 온도를 바꾸므로 학습된 모델은 건드리지 않는다. 온도 ≠ 1은 모델 성능에 영향을 주므로 baseline fp16 PPL도 같이 측정.

### 7.3 실험 매트릭스

| 모델 | 비트 | 방법 | 온도 |
|---|---|---|---|
| Qwen2.5-7B | 2 | WF(floor=1), WF(floor=2), Uniform | 0.5, 0.8, 1.0, 1.25, 1.5 |

1모델 × 3방법 × 5온도 = **15 실험** + fp16 baseline 5 = 20. 예상 소요: 2시간.

### 7.4 판정 기준

| 결과 | 판정 |
|---|---|
| 온도 ↑ 시 floor=1 PPL 재앙 완화 (단조) | **B3 확정**: 1-bit 실패는 softmax 증폭. |
| 온도 ↓ 시 floor=2도 깨짐 | B3 추가 지지 (임계 비트수 예측과 정합). |
| 온도 변화가 무관 | **B3 기각**. |

### 7.5 이론적 교차검증

가설 B3에서 제시한 임계 비트수 공식:

```
b_crit ≈ (1/2) log_2(λ_max(Σ_q) · c_Gersho) + O(1)
```

에 Qwen의 실측 λ_max(Σ_q)를 대입하여 b_crit를 계산. 만약 b_crit ≈ 1.5가 나오고 실측이 "1-bit 실패, 2-bit 성공"이라면 공식의 정합성이 확보된다.

---

## 8. 실험 간 상호작용과 우선순위

### 8.1 가설 간 비배타성

A1(메트릭), A2(싱크), A3(L∞)는 **동시 성립 가능**. 세 가지 모두가 부분적으로 기여할 수 있다. 따라서 각 실험이 독립적으로 부분 지지를 얻어도 의미 있다.

B1/B2/B3는 **강하게 상관**되어 있다:
- B1(WF 공식 점근 오류) → B2(정수 제약)의 이론적 근거
- B2(정수 제약) → 실무적 해결책
- B3(softmax 증폭) → B1/B2가 "왜 PPL에서 폭발하는가"를 설명하는 증폭 메커니즘

따라서 B2 실험(E4)이 가장 결정적이고, B3(E5)는 메커니즘 확인용이다.

### 8.2 우선순위

| 순위 | 실험 | 이유 | 예상 소요 |
|---|---|---|---|
| **1** | E1 (MK) | 축 2 클레임을 직접 구제하는 유일한 실험 | 4시간 |
| **2** | E4 (정수 WF) | 축 3 이론을 "이산 채널 제약"으로 격상 | 2시간 |
| **3** | E2 (싱크 보존) | 낮은 비용, 높은 정보량 | 2시간 |
| 4 | E5 (온도 스윕) | B3 메커니즘 확인 | 2시간 |
| 5 | E3 (L∞ Lloyd) | 구현 복잡, 판별력 낮음 | 3시간 |

**Critical path**: E1 → E4 → E2 (총 8시간, 1일 작업)

---

## 9. 결과 해석 시나리오

### 시나리오 A: E1 성공 (MK가 Uniform 승리)

**해석**: 축 2의 실패는 Σ_q=I 가정의 단순화 때문이었다. 논문의 정리 6.19.8은 옳다.

**논문 수정**:
- 축 2 PPL 결과를 MK로 교체 (V3의 Gaussian Lloyd는 Appendix로 강등)
- 본문: "MK가 Gaussian Lloyd의 PPL 실패를 구제한다"
- 핵심 클레임 "3축 동시 최적"이 **완전 복구**

### 시나리오 B: E1 실패, E2 성공 (싱크 보존이 해결)

**해석**: 축 2 실패는 분포 가정 위반(헤비테일 + 싱크).

**논문 수정**:
- 정리 6.19.8의 가정에 "경량-tail 또는 sink preservation" 조건 추가
- 축 2를 "이론적으로 최적이지만 실용상 outlier-aware 구현 필요"로 프레이밍
- 새로운 보조 정리: "Lloyd-Max + sink preservation의 PPL 상한"

### 시나리오 C: E4 성공 (정수 WF 확정)

**해석**: 축 3의 floor=2는 **임시방편이 아니라 이론적 최적**.

**논문 수정**:
- Section 6.19 또는 6.20에 "정수 제약 water-filling" 소정리 추가
- floor=2를 "실무 트릭"에서 "정리의 직접 결과"로 승격
- 관련 정리: "이산 rate 제약 하에서 최적 비트 할당은 {0} ∪ {≥b_min} 구조를 가진다"

### 시나리오 D: 모든 실험이 기각됨

**해석**: 더 깊은 원인 존재. 가장 가능성 높은 후보:
- 캘리브레이션 데이터 편향
- PCA 고유벡터의 레이어 간 불일치
- RoPE의 수반 표현과 Pre-RoPE PCA의 비가환성

**논문 수정**:
- Limitation 섹션에 "축 2·축 3의 실패는 5개 가설 중 어느 것으로도 설명되지 않으며, 향후 연구 대상"
- 이것도 **정직한 과학적 기여**로 가치 있음

---

## 10. 타임라인

| 일차 | 작업 | 산출물 |
|---|---|---|
| Day 1 오전 | E1 (MK) 구현 및 Qwen 실행 | MK 코드, Qwen PPL |
| Day 1 오후 | E1 Llama·Mistral 실행, E2 구현 | 3모델 MK 결과 |
| Day 2 오전 | E2 (싱크 보존) 3모델 실행 | sink ablation 테이블 |
| Day 2 오후 | E4 (정수 WF) 구현 및 실행 | WF ablation 테이블 |
| Day 3 오전 | E5 (온도 스윕) 실행 | 온도 민감도 곡선 |
| Day 3 오후 | E3 (L∞ Lloyd) 구현 및 실행 (조건부) | L∞ 결과 |
| Day 4 | 결과 통합, 논문 수정안 작성 | 수정된 Section 6.19, Limitation 섹션 |

총 **4일** (약 20 시간 실험 + 분석).

---

## 11. 재현성 체크리스트

- [ ] 모든 random seed 고정 (torch, numpy, transformers)
- [ ] 캘리브레이션 데이터: WikiText-2 train split, 2048 토큰, seed=42
- [ ] 평가: WikiText-2 test split, 전체
- [ ] 결과 JSON 스키마: `{model, method, bits, ppl, mse, attn_mse, config}`
- [ ] 코드: `experiments/axis2_axis3_diagnosis/`
- [ ] 모든 실험 로그 + 시각화 스크립트 포함

---

## 12. 논문 기여로의 연결

이 실험 결과는 논문의 다음 섹션에 직접 반영된다:

| 실험 결과 | 논문 섹션 | 기여 |
|---|---|---|
| E1 (MK 성공/실패) | Section 6.19 (축 2 정리) | 정리의 적용 조건 명확화 |
| E2 (싱크 효과) | Section 6.20 (축 3 토큰 관리, HEAT) | HEAT와 Lloyd의 상호작용 증거 |
| E4 (정수 WF) | 새 소정리 (6.19.x 또는 Appendix) | 이론 확장 |
| E5 (온도 민감도) | Limitation / Discussion | softmax-양자화 상호작용 논의 |
| 통합 | Limitation 섹션 전면 개편 | "음성 결과의 이론적 특성화" |

**최종 목표**: V3에서 "원인 불명의 음성 결과 2개"로 기록된 것을 **"이론적으로 특성화된 실패 경계"**로 승격하여, 리뷰어의 soundness 점수를 7.5 → 8.5로 올리는 것.

---

*문서 끝. 실험 착수 전 팀 내 검토 요청.*
