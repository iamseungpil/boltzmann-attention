# NeurIPS 2026 실험 계획 v14: Novelty-Driven Verification

**날짜**: 2026-04-05
**기반**: 선행연구 서베이 (17편) + 이론 논문 novelty 분석 + Review v1 피드백
**목표**: 논문의 5대 수학적 기여를 end-to-end PPL로 검증하여 NeurIPS 테이블 완성

---

## 0. Novelty 지도: 우리가 가진 것 vs 선행연구

### 선행연구 17편의 공통 한계

| 한계 | 해당 논문 |
|------|----------|
| Lie group 명시 사용 전무 | TurboQuant, KVTC, SpinQuant, PolarQuant, CommVQ (모두 암묵적) |
| PCA 양자화 최적성 증명 없음 | KVTC (PCA 사용하지만 왜 최적인지 미증명) |
| Pre-RoPE vs Post-RoPE 이론 비교 없음 | 전체 (경험적 관찰만) |
| Block size 선택 이론 없음 | RotorQuant(b=3), IsoQuant(b=4), PolarQuant(b=2) — 모두 ad-hoc |
| 방법 간 통일적 비교 없음 | 각 논문이 자기 baseline만 비교 |
| 양자화기 최적성 증명 없음 | TurboQuant만 Shannon bound 제시 (방법 간 비교 아님) |

### 우리의 5대 수학적 기여 (검증 필요 수준)

| # | 기여 | 정리 | MSE 검증 | PPL 검증 | 논문 테이블 |
|---|------|------|:--------:|:--------:|:----------:|
| C1 | 통일 Hamiltonian 프레임워크 | 5.2.1 | N/A (분류) | N/A | Table 구조 |
| C2 | Pre-RoPE PCA 최적성 | **6.16.3** | ✅ 3모델 | ❌ 미완 | **Table 1** |
| C3 | 3축 독립 분해 | 6.18.2 | ✅ 부분 | ❌ 미완 | Table 3 (ablation) |
| C4 | MK Lloyd-Max 우월성 | 6.19.8 | ✅ 3모델 | ❌ 미완 | **Table 1** |
| C5 | Master Formula G(b,Σ) 예측 | 6.1.2 | ✅ 부분 | ❌ R² 미정리 | Figure 1 |

**핵심 gap**: MSE 수준 검증은 완료되었으나, **PPL end-to-end 검증과 예측-실측 정량 비교가 전무**.

---

## 1. 실험 매트릭스: NeurIPS Table 1

### 의도 (Intent)
NeurIPS 논문의 핵심 테이블을 채울 PPL 비교 데이터를 생성한다.
3모델 × 4방법 × 3비트 = 36 PPL 측정으로 정리 6.16.3과 6.19.8을 end-to-end 검증.

### 가설 (Hypothesis)
```
PPL 순서 (낮을수록 좋음):
  PreRoPE PCA + MK(Lloyd) ≤ PreRoPE PCA + Uniform(WF) < TurboQuant < No rotation

구체적으로:
H1: PreRoPE PCA+Uni가 TurboQuant보다 모든 모델/비트에서 낮은 PPL
H2: PreRoPE PCA+Lloyd가 PreRoPE PCA+Uni보다 낮은 PPL (축 2 이득)
H3: PPL 순서가 MSE 순서와 일치 (Level 2 정합성)
H4: PPL 이득은 R_aniso에 비례하여 증가 (Mistral > Llama > Qwen)
```

### 검증 방법 (Verification)
```
모델: Qwen2.5-7B, Llama-3.1-8B, Mistral-7B-v0.3
방법:
  1. No rotation + Uniform (baseline)
  2. TurboQuant (random SO(d) + Uniform)
  3. PreRoPE PCA + Uniform + WF (축 1 최적)
  4. PreRoPE PCA + Lloyd-Max (축 1+2 최적)

비트: 2, 3, 4
데이터: WikiText-2 test set, sliding window, 50K+ tokens
프로토콜: k_proj hook → rotate → quantize → inverse rotate → PPL 측정

성공 기준:
  - H1: p < 0.05 paired test across chunks
  - H2: 2-bit에서 1% 이상 PPL 개선
  - H3: 36개 실험 중 MSE-PPL 순서 불일치 ≤ 2건
  
실패 대응:
  - H1 실패: 양자화기가 회전 이득을 상쇄 → "회전은 맞지만 스칼라 양자화가 병목" 프레이밍
  - H2 실패: Lloyd-Max의 가우시안 가정이 rotated key에 부적합 → 축 2를 "이론적 상한" 위치로 조정
```

---

## 2. KVTC 직접 비교 (Llama-3-8B)

### 의도 (Intent)
KVTC (ICLR 2026)가 사용하는 **공유 Pre-RoPE PCA**와 우리의 **헤드별 Pre-RoPE PCA**를 
동일 모델(Llama-3-8B)에서 직접 비교한다. 이것은 정리 6.16.3의 per-head 최적성이
공유 PCA 대비 실질적 이득을 주는지를 검증한다.

### 가설 (Hypothesis)
```
H1: 헤드별 PCA MSE < 공유 PCA MSE (Fischer inequality에 의해 보장)
H2: PPL 차이는 미미할 수 있음 (KVTC의 DP bit allocation이 보상)
H3: 헤드 간 공분산 이질성이 높은 레이어에서 per-head 이득이 극대화
```

### 검증 방법 (Verification)
```
구현:
  - 공유 PCA: 전체 KV head의 key를 합쳐서 하나의 PCA basis 계산
  - 헤드별 PCA: 각 KV head에서 독립적으로 PCA basis 계산
  
비교:
  - Llama-3.1-8B × {공유PCA+Uni, 헤드별PCA+Uni, 헤드별PCA+WF} × {2,3,4}bit PPL
  - Fischer ratio: ||Σ_shared - Σ_perhead|| 측정 (이론적 이득 예측)

성공 기준:
  - 헤드별 PCA MSE < 공유 PCA MSE (이론 보장, smoke test)
  - PPL 차이 > 0.1% (실질적 의미)

실패 대응:
  - PPL 차이가 무의미하면: "이론적으로 최적이나 실무적 차이 미미, KVTC의 공유 PCA는 
    near-optimal approximation" → 이것 자체가 유용한 결론
```

---

## 3. Level 1-2 예측 정합성 R² 정량화

### 의도 (Intent)
기여 C5 (Master Formula)의 예측력을 정량화한다. "Σ만으로 최적 방법을 추천"이라는
핵심 주장을 R² 값으로 뒷받침한다.

### 가설 (Hypothesis)
```
Level 1: G(method, Σ) 예측 MSE 순서 → 100% 정합 (3모델 전체)
Level 2: ln(PPL/fp16) = c × D_actual → R² > 0.85 (3모델 통합)
```

### 검증 방법 (Verification)
```
데이터:
  - 3모델 × 4방법 × 3비트 = 36 (MSE, PPL) 쌍 (실험 1에서 획득)
  
분석:
  1. Level 1: 각 모델 내에서 MSE 순서가 G(method) 순서와 일치하는 비율
  2. Level 2: 36개 점에서 ln(PPL/fp16) vs D_actual의 선형 회귀 R²
  3. 모델별 R²: 각 모델 내에서의 R² (calibration constant c가 모델마다 다를 수 있음)

산출물:
  - Figure 1: scatter plot (D_actual vs ln(PPL/fp16)), 색=모델, 점=방법
  - R² 값, slope c, intercept

성공 기준:
  - Level 1: 순서 100% (이론 보장)
  - Level 2: R² > 0.85 (전체), R² > 0.95 (모델 내)
```

---

## 4. 3축 Ablation (독립성 검증)

### 의도 (Intent)
명제 6.18.2 (3축 독립)를 검증한다. 축 1+2+3의 이득이 곱셈적으로 결합하는지 확인.

### 가설 (Hypothesis)
```
D(축1+축2) ≈ D(축1만) × gain(축2)
D(축1+축2+축3) ≈ D(축1+축2) × gain(축3)
```

### 검증 방법 (Verification)
```
Ablation 조합 (Qwen2.5-7B, 3-bit):
  a. No rotation + Uniform           (baseline)
  b. PreRoPE PCA + Uniform           (축 1만)
  c. No rotation + Lloyd-Max         (축 2만)
  d. PreRoPE PCA + Lloyd-Max         (축 1+2)
  e. PreRoPE PCA + Uniform + WF      (축 1 + WF)
  f. PreRoPE PCA + Lloyd-Max + WF    (축 1+2 + WF)

MSE 확인:
  gain(축1) = MSE(a) / MSE(b)
  gain(축2) = MSE(a) / MSE(c)
  independence check: MSE(d) ≈ MSE(a) / (gain(축1) × gain(축2))

PPL 확인: 같은 조합으로 PPL 측정
```

---

## 5. 에너지/Hamiltonian 해석 검증

### 의도 (Intent)
논문의 독특한 기여인 Hamiltonian 해석(정의 5.1.1)의 실증적 의미를 보인다.
각 방법의 "에너지 구조"가 실제로 양자화 성능을 결정하는지 확인한다.

### 가설 (Hypothesis)
```
H1: 방법의 block size b는 Lie algebra 차원과 비례 관계
    PolarQuant(b=2) → dim=1, RotorQuant(b=3) → dim=3, IsoQuant(b=4) → dim=6
    파라미터 수 ∝ Σ dim(so(b_k))
    
H2: Master Formula의 G(b) 계층 G(d) ≤ G(4) ≤ G(3) ≤ G(2) ≤ G(1)이
    실측 MSE에서 성립 (블록 크기별 MSE 비교)
    
H3: R_aniso가 큰 모델일수록 rotation 이득이 크다 (에너지 분산이 불균일할수록)
```

### 검증 방법 (Verification)
```
실험:
  1. 3모델에서 block size b ∈ {2, 4, d} 별 G(b,Σ) 이론값 계산
  2. 같은 block size별 실제 block-diagonal rotation MSE 측정
  3. G 계층 성립 여부 확인

이것은 이미 Level 1 예측의 일부이므로, 기존 Σ 데이터로 계산 가능.
추가 실험 없이 이론적 분석으로 충족.
```

---

## 실행 우선순위

```
즉시 (Day 1-2):
  ┌─ Exp 1: PPL Table (3모델 × 4방법 × 3비트) ── NeurIPS Table 1의 핵심
  └─ Exp 3: Level 1-2 R² (기존 MSE 데이터 + Exp 1 PPL)

이어서 (Day 3):
  ┌─ Exp 2: KVTC 직접 비교 (Llama, 공유 vs 헤드별 PCA)
  └─ Exp 4: 3축 Ablation (Qwen, 6개 조합)

분석 (Day 4):
  └─ Exp 5: 에너지 해석 (이론 계산, 추가 실험 불필요)
```

---

## 구현 요구사항

### PPL 측정 스크립트 (verify_ppl_table.py)
```
입력: --model, --methods [norot, turbo, pre_pca_uni, pre_pca_lloyd], --bits [2,3,4]
출력: JSON with {method, bits, ppl, total_tokens, nll}

핵심 구현:
  1. Calibration pass: Pre-RoPE key 공분산 추출 → PCA basis
  2. k_proj hook: 
     - rotate (PCA or random)
     - quantize (uniform or lloyd-max)
     - inverse rotate
  3. Non-overlapping chunk PPL 측정
  4. WF: PCA 후 분산 기반 비트 할당 (water-filling)
```

### KVTC 비교 구현
```
공유 PCA: 모든 KV head의 key를 concatenate → 하나의 PCA
헤드별 PCA: 각 KV head 독립 PCA
나머지는 동일 (uniform quantization, WF bit allocation)
```
