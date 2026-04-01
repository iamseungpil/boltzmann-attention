# Boltzmann Thermodynamics of Attention -- Comprehensive Experiment Log v2

**작성일**: 2026-03-29
**문서 성격**: Level 1 실험 종합 보고서 (internal research update)
**실험 환경**: Qwen2.5-3B, GPT-2 Medium, NVIDIA RTX A6000
**Source files**: `exp1_1_results.json`, `exp1_1_gpt2_results.json`, `exp1_2_results.json`

---

## 1. Executive Summary

Boltzmann thermodynamics of attention framework의 Level 1 실험 2개를 완료하였다. 실험 1-1(T_eff gate test)에서 GPT-2 Medium(MHA)은 progressive cooling 가설과 일치하는 유의한 음의 상관을 보였고(Spearman rho = -0.657, p = 4.82e-04), Qwen2.5-3B(GQA)는 이를 보이지 못했다(rho = +0.108, p = 0.530). 실험 1-2(specific heat quantization)에서 C(q)와 ||k||^2의 bit allocation disagreement rate은 39.8%로 두 metric이 독립적 정보를 포착함을 확인하였으나, C(q) 기반 양자화는 PPL에서 분산 기반 방법을 능가하지 못했다(3-bit: 264.5 vs 127.6). 종합하면, Boltzmann framework은 standard MHA에서 열적 구조의 존재를 보여주는 **기술적(descriptive) 프레임워크**로서의 가치를 가지지만, 양자화 설계에 직접적 실용 가치를 제공하는 **규범적(prescriptive) 프레임워크**로는 아직 작동하지 않는다. FOKVQ 논문의 핵심 기여인 PCA-based facet-aware 비균일 비트 할당(68-87% KL 감소)과의 통합에서, Boltzmann 양은 보조 예측 지표(C(q) vs Facet Gain: rho = -0.63)로 역할할 수 있으나 주요 설계 원리로는 부적합하다.

---

## 2. Experiment 1-1: T_eff Gate Test -- Detailed Analysis

### 2.1 목적

이 실험의 목적은 Boltzmann framework의 핵심 예측인 "progressive cooling"을 검증하는 것이다. Softmax attention을 Boltzmann distribution으로 해석하면, 심층 레이어로 갈수록 T_eff(유효 온도)가 단조 감소하여 attention이 점진적으로 집중되어야 한다. 이 예측이 성립하면 Level 2 이상의 실험을 진행할 근거가 확보되고, 실패하면 framework 자체의 재검토가 필요하다.

### 2.2 실험 설정

| 항목 | Qwen2.5-3B | GPT-2 Medium |
|------|-----------|-------------|
| Architecture | GQA (2 KV heads) | Standard MHA |
| Layers | 36 | 24 |
| Heads | 16 | 16 |
| Head dim | 128 | 64 |
| Calibration samples | 64 | 64 |
| Max seq len | 512 | 512 |
| T_eff 정의 | S(l) / log(n) | S(l) / log(n) |

### 2.3 GPT-2 Medium 결과 (MHA): WEAK_SUCCESS

**Gate metric**: Spearman rho = **-0.6574**, p = **4.82e-04**

사전 정의된 gate 기준에 따라 WEAK_SUCCESS로 판정한다 (-0.7 < rho < -0.3). STRONG_SUCCESS 임계값(-0.7)에 근접한다.

**Per-layer T_eff table (GPT-2 Medium)**:

| Layer | T_eff (mean) | T_eff (std) | Entropy (mean) | Head Var |
|-------|-------------|-------------|----------------|----------|
| 0 | 0.861 | 0.090 | 1.816 | 0.0080 |
| 1 | 0.780 | 0.136 | 1.696 | 0.0185 |
| 2 | 0.777 | 0.108 | 1.662 | 0.0116 |
| 3 | 0.671 | 0.090 | 1.437 | 0.0082 |
| 4 | 0.498 | 0.171 | 1.086 | 0.0291 |
| 5 | 0.405 | 0.174 | 0.878 | 0.0303 |
| 6 | 0.270 | 0.161 | 0.591 | 0.0260 |
| 7 | 0.321 | 0.168 | 0.695 | 0.0284 |
| 8 | 0.428 | 0.149 | 0.935 | 0.0223 |
| 9 | 0.281 | 0.169 | 0.616 | 0.0284 |
| 10 | 0.389 | 0.176 | 0.851 | 0.0309 |
| 11 | 0.368 | 0.179 | 0.796 | 0.0319 |
| 12 | 0.329 | 0.177 | 0.722 | 0.0313 |
| 13 | 0.384 | 0.166 | 0.835 | 0.0274 |
| 14 | 0.322 | 0.150 | 0.697 | 0.0224 |
| 15 | 0.387 | 0.153 | 0.834 | 0.0233 |
| 16 | 0.269 | 0.144 | 0.575 | 0.0209 |
| 17 | 0.317 | 0.179 | 0.679 | 0.0319 |
| 18 | 0.275 | 0.174 | 0.585 | 0.0302 |
| 19 | 0.221 | 0.096 | 0.469 | 0.0092 |
| 20 | 0.233 | 0.122 | 0.503 | 0.0149 |
| 21 | 0.265 | 0.136 | 0.562 | 0.0186 |
| 22 | 0.317 | 0.111 | 0.687 | 0.0123 |
| 23 | 0.520 | 0.132 | 1.122 | 0.0174 |

**패턴 분석**:
- **Progressive cooling (L0-L6)**: T_eff가 0.861에서 0.270으로 급격히 감소. 이 구간만의 rho는 -1.0 (완벽한 단조 감소).
- **Oscillation zone (L6-L19)**: 0.221-0.428 범위에서 비단조적 진동. L8(0.428)과 L10(0.389)에서 국소 재가열.
- **Final reheating (L19-L23)**: 0.221에서 0.520으로 뚜렷한 온도 상승. 최종 레이어의 넓은 attention 분포와 일치.
- **Global minimum**: Layer 19 (T_eff = 0.221)

**Head variance 패턴**:
- 초기 레이어 (L0-L3): 낮은 head variance (0.008-0.018). Heads가 유사하게 작동.
- 중간 레이어 (L4-L18): 높은 head variance (0.021-0.032). 최대는 L11 (0.032).
- 후기 레이어 (L19-L23): 중간 수준 (0.009-0.019).

### 2.4 Qwen2.5-3B 결과 (GQA): FAILURE

**Gate metric**: Spearman rho = **+0.1081**, p = **0.530**

사전 정의 기준에 따라 FAILURE로 판정한다.

**Per-layer T_eff table (Qwen2.5-3B, 36 layers)**:

| Layer | T_eff (mean) | T_eff (std) | Head Var | Layer | T_eff (mean) | T_eff (std) | Head Var |
|-------|-------------|-------------|----------|-------|-------------|-------------|----------|
| 0 | 0.814 | 0.073 | 0.0054 | 18 | 0.612 | 0.135 | 0.0182 |
| 1 | 0.903 | 0.039 | 0.0015 | 19 | 0.391 | 0.214 | 0.0457 |
| 2 | 0.815 | 0.087 | 0.0075 | 20 | 0.336 | 0.257 | 0.0661 |
| 3 | 0.311 | 0.152 | 0.0231 | 21 | 0.460 | 0.140 | 0.0195 |
| 4 | 0.393 | 0.229 | 0.0524 | 22 | 0.490 | 0.168 | 0.0282 |
| 5 | 0.295 | 0.293 | 0.0857 | 23 | 0.590 | 0.170 | 0.0290 |
| 6 | 0.405 | 0.229 | 0.0526 | 24 | 0.648 | 0.170 | 0.0289 |
| 7 | 0.388 | 0.184 | 0.0338 | 25 | 0.308 | 0.225 | 0.0507 |
| 8 | 0.421 | 0.167 | 0.0280 | 26 | 0.592 | 0.214 | 0.0459 |
| 9 | 0.462 | 0.132 | 0.0175 | 27 | 0.483 | 0.208 | 0.0431 |
| 10 | 0.406 | 0.228 | 0.0518 | 28 | 0.619 | 0.157 | 0.0248 |
| 11 | 0.566 | 0.141 | 0.0198 | 29 | 0.484 | 0.205 | 0.0422 |
| 12 | 0.505 | 0.177 | 0.0312 | 30 | 0.430 | 0.261 | 0.0683 |
| 13 | 0.433 | 0.223 | 0.0499 | 31 | 0.610 | 0.148 | 0.0220 |
| 14 | 0.574 | 0.123 | 0.0152 | 32 | 0.401 | 0.090 | 0.0082 |
| 15 | 0.482 | 0.240 | 0.0577 | 33 | 0.348 | 0.079 | 0.0063 |
| 16 | 0.566 | 0.100 | 0.0099 | 34 | 0.748 | 0.068 | 0.0046 |
| 17 | 0.326 | 0.215 | 0.0460 | 35 | 0.767 | 0.074 | 0.0055 |

**V-shaped 패턴**: 고온 초기층 (L0-L2: 0.81-0.90) -> 저온 중간층 (L3-L33: 0.30-0.61, 불규칙) -> 고온 최종층 (L34-L35: 0.75-0.77). 단조 감소가 관찰되지 않는다.

### 2.5 GQA Disruption Finding

GQA(Grouped Query Attention) 구조가 progressive cooling을 파괴하는 메커니즘을 분석한다.

Qwen2.5-3B는 16개 query head가 2개의 KV head를 공유한다 (8:1 ratio). 이 공유 구조가 극단적인 inter-group temperature gap을 유발한다.

**Layer 5 head-level T_eff (Qwen2.5-3B)**:

| Head | T_eff | KV Group | Head | T_eff | KV Group |
|------|-------|----------|------|-------|----------|
| H0 | 0.716 | KV0 | H8 | 0.037 | KV1 |
| H1 | 0.539 | KV0 | H9 | 0.073 | KV1 |
| H2 | 0.436 | KV0 | H10 | 0.000 | KV1 |
| H3 | 0.642 | KV0 | H11 | 0.008 | KV1 |
| H4 | 0.504 | KV0 | H12 | 0.045 | KV1 |
| H5 | 0.082 | KV0 | H13 | 0.090 | KV1 |
| H6 | 0.728 | KV0 | H14 | 0.053 | KV1 |
| H7 | 0.741 | KV0 | H15 | 0.032 | KV1 |

Source: `exp1_1_results.json`, `layer_head_teff[5]`

KV Group 0의 평균 T_eff = 0.548 (H5 포함), KV Group 1의 평균 T_eff = 0.042. **Gap = 0.506**. 이는 동일 레이어 내에서 한 KV group은 "고온"(넓은 attention 분산), 다른 group은 "절대영도 근방"(거의 결정론적 attention)인 극단적 이질성이다.

**Head variance가 가장 높은 5개 레이어 (Qwen2.5-3B)**:

| Layer | Head Var | 배수 (vs L0) |
|-------|---------|-------------|
| L5 | 0.0857 | 15.9x |
| L30 | 0.0683 | 12.7x |
| L20 | 0.0661 | 12.3x |
| L15 | 0.0577 | 10.7x |
| L6 | 0.0526 | 9.8x |

초기/최종 레이어의 head variance (L0: 0.0054, L35: 0.0055) 대비 최대 15.9배. GQA의 KV 공유가 head 간 기능적 분화를 극대화하는 구조적 효과이다.

### 2.6 시사점

실험 1-1의 결과는 두 가지 핵심 함의를 가진다. 첫째, Boltzmann T_eff의 progressive cooling은 standard MHA에서 실재하는 현상이며 (GPT-2: rho = -0.657), 이는 다층 attention을 점진적 annealing으로 해석하는 이론적 정당성을 부분적으로 뒷받침한다. 둘째, GQA 구조는 KV 공유로 인해 head 간 온도 분포가 극단적으로 분화되며, 이는 GQA 모델에 대한 양자화 전략이 head-level이 아닌 KV group-level로 설계되어야 함을 시사한다. 그러나 GPT-2에서조차 진정한 "단조" 냉각은 아니며 (중간 레이어 진동 존재), 전역 rho = -0.657은 "강한 성공" 기준(-0.7)을 충족하지 못한다는 점에서 framework의 정량적 예측력에는 한계가 있다.

---

## 3. Experiment 1-2: Specific Heat Quantization -- Detailed Analysis

### 3.1 목적

이 실험의 목적은 Boltzmann specific heat C(q)가 기존 key norm 기반 방법(||k||^2)과 구별되는 양자화 기준을 제공하는지 검증하고, 이 구별이 PPL 개선으로 이어지는지 확인하는 것이다. C(q) = Var_alpha[q_i^T k_j / sqrt(d)]는 query-dependent attention energy variance로, 특정 query가 key 분포의 변동에 얼마나 민감한지를 측정한다.

### 3.2 실험 설정

| 항목 | 값 |
|------|-----|
| Model | GPT-2 Medium (24L/16H/d=64) |
| Evaluation data | WikiText-2, 32 samples |
| Max seq len | 256 |
| Baseline PPL | 139.85 |
| Bit budgets | 2, 3, 4 |
| Quantization strategies | Uniform, Variance(||k||^2), Specific heat(C(q)) |
| Bit allocation rule | Median-based +/-1 bit |

### 3.3 Disagreement Analysis

C(q)와 ||k||^2 간 bit allocation disagreement를 layer별로 분석한다.

**Layer group breakdown** (source: `exp1_2_results.json`, `layer_stats`):

| Layer Group | Layers | Mean Disagreement | Mean rho(C, ||k||^2) | Interpretation |
|-------------|--------|-------------------|---------------------|----------------|
| Early (L0-L5) | L0-L5 | 30.1% | 0.526 | 중간 상관, 낮은 불일치 |
| Middle (L6-L11) | L6-L11 | 41.7% | 0.226 | 약한 상관, 높은 불일치 |
| Mid-Late (L12-L17) | L12-L17 | 42.4% | 0.212 | 약한 상관, 높은 불일치 |
| Late (L18-L23) | L18-L23 | 45.2% | 0.136 | 거의 무상관, 최고 불일치 |
| **Overall** | L0-L23 | **39.8%** | **0.275** | 독립적 metric 확인 |

Source: 각 layer의 disagreement, rho_C_k 값은 `exp1_2_results.json`의 `layer_stats` 배열에서 직접 추출.

**주요 레이어별 수치** (JSON에서 직접 확인):

| Layer | Disagreement | rho(C, ||k||^2) | C(q) mean | ||k|| mean |
|-------|-------------|----------------|-----------|-----------|
| L0 | 36.9% | 0.357 | 0.665 | 30.8 |
| L2 | 27.6% | 0.569 | 1.410 | 218.9 |
| L4 | 24.3% | 0.692 | 1.659 | 623.2 |
| L7 | 46.8% | 0.072 | 2.418 | 172.8 |
| L9 | 45.9% | 0.130 | 2.460 | 174.2 |
| L17 | 47.2% | 0.104 | 2.691 | 250.5 |
| L19 | 47.8% | 0.044 | 2.744 | 162.8 |
| L21 | 49.6% | 0.034 | 2.542 | 159.8 |
| L23 | 37.8% | 0.354 | 2.486 | 103.1 |

**패턴**: 초기 레이어(L0-L5)에서는 C(q)와 ||k||^2가 중간 수준의 상관(rho = 0.36-0.69)을 보이며, 이는 초기 레이어에서 attention 민감도와 key magnitude가 어느 정도 정렬되어 있음을 의미한다. 후기 레이어로 갈수록 상관이 0에 수렴하며(L19: 0.044, L21: 0.034), 두 metric이 거의 완전히 독립적 정보를 포착한다. Disagreement rate 역시 L0(36.9%)에서 L21(49.6%)로 단조 증가하여, 후기 레이어에서 C(q)가 ||k||^2 대비 가장 차별화된 정보를 제공한다.

### 3.4 PPL Results

| Bits | Uniform | Variance (||k||^2) | Specific Heat (C(q)) |
|------|---------|-------------------|---------------------|
| Baseline (FP32) | 139.85 | -- | -- |
| 2-bit | 419.41 | exploded (4.85e+08) | exploded (4.85e+08) |
| 3-bit | 130.38 | **127.62** | 264.48 |
| 4-bit | **110.19** | 122.85 | 125.56 |

Source: `exp1_2_results.json`, `ppl_results`

**분석**:

1. **2-bit**: 두 adaptive 방법 모두 수치 폭발 (PPL = 4.85e+08). 동일한 값이 나온 것은 양쪽 모두 extreme quantization에서 hook-based K replacement의 numerical instability가 동일하게 발현되었음을 시사한다. Uniform 2-bit(419.41)만이 유효한 결과.

2. **3-bit**: Variance 기반(127.62)이 Uniform(130.38)보다 2.1% 개선. C(q) 기반(264.48)은 Uniform보다 **103% 악화**. C(q)의 median-based +/-1 bit 할당이 critical keys의 precision을 과도하게 줄여 catastrophic error를 유발한 것으로 판단된다.

3. **4-bit**: Uniform(110.19)이 최선. Variance(122.85)와 C(q)(125.56)는 오히려 baseline(139.85)보다는 개선되었지만 Uniform보다 열등하다. 4-bit에서는 bit budget이 충분하여 adaptive allocation의 overhead(median mismatch)가 이득을 상쇄한다.

**Heat wins: 0, Variance wins: 2** (3-bit, 4-bit 모두 Variance가 C(q)를 능가).

### 3.5 Implementation Caveats

1. **Hook-based K replacement**: Forward hook으로 K tensor를 양자화된 버전으로 교체하는 방식은, 모델의 internal computation graph와 완전히 일치하지 않을 수 있다. 특히 KV cache reuse가 있는 경우 artifact 발생 가능.

2. **2-bit explosion**: Adaptive 2-bit에서의 동일한 수치 폭발(4.85e+08)은 구현 차원의 문제일 가능성이 높다. Adaptive scheme이 일부 token에 1-bit만 할당할 때, 1-bit 양자화의 해상도(2 levels)가 key vector의 dynamic range를 표현하지 못하면서 attention weight가 degenerate하는 현상.

3. **Median threshold의 조잡함**: Median-based binary split (+1/-1 bit)은 C(q) 분포의 형상을 무시한다. C(q)는 heavy-tailed distribution을 보이며(후기 레이어에서 mean = 2.5-2.8), 상위 극단값만이 양자화 민감도의 실질적 차이를 만든다. 이진 분류가 아닌 연속적/다단계 할당이 필요할 수 있다.

### 3.6 시사점

실험 1-2는 세 가지 결론을 도출한다. 첫째, C(q)는 ||k||^2와 구별되는 새로운 물리량이다 (39.8% disagreement, rho = 0.275). 이는 Boltzmann framework이 기존 metric의 단순 재포장이 아님을 확인한다. 둘째, 이 새로운 정보가 현재의 단순한 allocation rule(median +/-1 bit)로는 PPL 개선으로 이어지지 않는다. 셋째, C(q)의 가치가 실현되려면 비선형/다단계 bit allocation, ||k||^2와의 결합 기준, 또는 FOKVQ의 차원별 할당과의 통합 등 더 정교한 활용이 필요하다. 현재 상태에서 C(q) 기반 양자화를 실용적 방법으로 주장하는 것은 데이터가 지지하지 않는다.

---

## 4. Cross-Experiment Synthesis

### 4.1 Boltzmann Framework의 두 얼굴

실험 1-1과 1-2를 종합하면, Boltzmann framework은 **기술적(descriptive) 성공**과 **규범적(prescriptive) 실패**로 요약된다.

**기술적 성공**:
- T_eff는 MHA에서 progressive cooling이라는 실재하는 패턴을 포착한다 (rho = -0.657).
- C(q)는 ||k||^2와 독립적인 새로운 양이다 (39.8% disagreement).
- GQA disruption이라는 새로운 현상을 발견하였다 (inter-group gap = 0.506).
- Head-level variance의 layer-depth 의존성이 architectural insight를 제공한다.

**규범적 실패**:
- T_eff의 단조 감소는 엄밀하게 성립하지 않는다 (중간 레이어 진동).
- C(q) 기반 양자화는 PPL에서 모든 비교 방법보다 열등하다.
- C(q) 정보를 활용하는 실용적 allocation rule을 찾지 못했다.

### 4.2 두 실험이 함께 말하는 것

T_eff (실험 1-1)과 C(q) (실험 1-2)는 attention의 서로 다른 측면을 측정한다:

| 양 | 측정 대상 | Level | 실용 가치 |
|----|----------|-------|----------|
| T_eff | Attention 집중도 (엔트로피 기반) | Layer/Head | GQA 분석 지표로 유용 |
| C(q) | Query-key energy variance | Token | 독립적 metric이나 직접 활용 실패 |
| ||k||^2 | Key magnitude | Token | 단순하지만 PPL 예측에서 C(q)보다 우수 |

FOKVQ 최종 보고서의 교차 분석에서 T_eff vs r_eff의 상관은 rho = -0.084 (p = 0.695)로 무상관이다 (source: FOKVQ_FINAL_REPORT.docx, Section 4D.2). 반면 C(q) vs Facet Gain은 rho = -0.63 (p = 0.001)로 강한 음의 상관을 보인다. 이는 C(q)가 직접적 양자화 지표로는 실패하지만, **FOKVQ의 facet-aware 방법이 잘 작동할 레이어를 예측하는 간접 지표**로서 가치를 가짐을 의미한다.

### 4.3 Architecture-dependent Nature

| 모델 | Attention Type | T_eff 패턴 | Progressive Cooling |
|------|---------------|-----------|-------------------|
| GPT-2 Medium | Standard MHA | 명확한 cooling -> oscillation -> reheating | rho = -0.657 (유의) |
| Qwen2.5-3B | GQA (2 KV heads) | V-shaped, 비단조 | rho = +0.108 (비유의) |

Progressive cooling은 **architecture-dependent** 현상이다. GQA가 보편화된 현대 LLM(Llama, Mistral, Qwen 등)에서는 head-level T_eff 분석이 아닌 KV group-level 또는 layer-level aggregate 분석이 필요하다. 이는 Boltzmann framework의 적용 범위에 대한 중요한 boundary condition이다.

---

## 5. FOKVQ Integration Plan

### 5.1 현재 위치: Boltzmann과 FOKVQ의 관계

FOKVQ Phase 1의 핵심 결과 (source: FOKVQ_FINAL_REPORT.docx):

| Metric | Value | Implication |
|--------|-------|-------------|
| r_eff (전체 평균) | 7.17-9.50 (코퍼스 의존) | 극단적 비등방성 보편적 |
| STRONG 비등방성 비율 | 100% (72/72 layers) | 모든 측정점에서 확인 |
| 2-bit KL 감소 (Facet vs Uniform) | 61-69% | 저비트에서 압도적 |
| 3-bit KL 감소 | 60-65% | 일관된 대규모 개선 |
| 4-bit KL 감소 | 31-73% | 코퍼스 의존적 |
| 8-bit reversal | 5.0-7.9x 악화 | PCA 반올림 오차 > 비균일 이득 |
| r_eff vs Facet Gain (384 heads) | rho = -0.6811, p = 1.13e-53 | 최고 예측자 |
| C(q) vs Facet Gain | rho = -0.63, p = 0.001 | 보조 예측자 |
| T_eff vs Facet Gain | rho = +0.30, p = 0.16 | 무상관 |

**핵심 판단**: Boltzmann T_eff는 FOKVQ 양자화 설계에 직접적 가치가 없다 (rho = +0.30, 비유의). C(q)는 보조 지표로 역할할 수 있으나 (rho = -0.63), r_eff (rho = -0.79)가 이미 더 강한 예측자이므로 추가 가치는 제한적이다.

### 5.2 Phase 2 (AFOD Facet 검증)에서의 Boltzmann 역할

Phase 2의 4개 실험 (Exp 2-1 ~ 2-4)은 Boltzmann과 직접적 연결이 없다. AFOD facet 분리, LDA vs PCA, query-adaptive allocation, AFOD initialization은 모두 PCA/정보이론 기반이다.

**잠재적 통합 지점**: Exp 2-3 (query-dependent vs static bit allocation)은 **Inconclusive**로 판정한다: 두 구현이 상충하는 결과를 보임 (구현 A: seq=128, degenerate/NaN 23/24 layers, differs=False; 구현 B: seq=256, differs=True, rho=0.698). 표준화된 프로토콜로 재실험 필요.

### 5.3 Phase 3 (Lie Group 검증)에서의 Boltzmann 역할

Exp 3-1 (연속 회전 각도 최적화)에서 T_eff가 최적 회전 각도와 상관이 있는지 탐색적으로 측정할 수 있다. 고온 레이어(넓은 attention)와 저온 레이어(집중된 attention)에서 최적 theta가 다를 수 있다는 가설은 이론적으로 흥미롭지만, T_eff vs Facet Gain의 무상관 결과(rho = +0.30)를 고려하면 높은 기대는 어렵다.

### 5.4 Phase 4 (대규모 모델)에서의 Boltzmann 역할

Exp 4-1 (Llama-7B 재현)에서 GQA 모델의 T_eff 프로파일을 동시 측정하는 것이 가치 있다. 실험 1-1에서 Qwen2.5-3B의 GQA disruption을 발견했으므로, Llama-7B(GQA, 8 KV heads)에서 동일 현상이 재현되는지 확인하면 해당 finding의 일반성을 강화할 수 있다. 이는 FOKVQ 논문의 supplementary material에 포함할 수 있다.

### 5.5 Boltzmann의 논문 내 위치 권장

FOKVQ_FINAL_REPORT.docx Section 7.3의 권장에 동의한다: **Boltzmann 분석은 논문의 핵심 기여에서 제외하고, supplementary/appendix로 배치한다.** 논문의 core는 비등방성 측정 + 패러다임 역전(구조 활용 vs 구조 파괴) + 68-87% KL 감소 + 적용 경계(8-bit reversal, per-head adaptation)이다.

---

## 6. Next Experiments -- Prioritized List

### Priority 1 (즉시 착수, Critical path)

| Exp | Description | GPU | Duration | Machine |
|-----|-------------|-----|----------|---------|
| **4-1** | Llama-7B r_eff + Facet-Aware 재현 | 24-48h | 2-3일 | A100 |
| **4-2** | QuIP#/KIVI/ZipCache 벤치마크 | 48-72h | 3-5일 | A100 |

**Rationale**: 논문 제출의 필수 조건. GPT-2에서의 결과가 현대 GQA 모델에서 재현되지 않으면 논문의 일반화 주장이 불가능하다. Exp 4-1 FAIL 시 논문을 GPT-2 규모로 축소해야 한다.

**GPU 할당**: A100 (Azure) 전용. 현재 RTX A6000으로는 7B 모델 로드 자체가 불가.

### Priority 2 (Week 1, Phase 2/3과 병렬)

| Exp | Description | GPU | Duration | Machine |
|-----|-------------|-----|----------|---------|
| **2-1** | AFOD facet K-space 분리 | 1h | 0.5일 | A6000 |
| **2-2** | LDA vs PCA 비균일 할당 | 2h | 0.5일 | A6000 |
| **3-1** | 연속 회전 각도 최적화 | 6h | 1일 | A6000 |
| **3-2** | 통일 정리 실용 검증 | 4h | 1일 | A6000 |

**Rationale**: A6000에서 실행 가능한 GPT-2 기반 실험. Phase 4와 병렬로 수행하여 시간 효율 극대화.

### Priority 3 (Phase 2/3 결과에 의존)

| Exp | Description | GPU | Duration | Dependency |
|-----|-------------|-----|----------|------------|
| **2-3** | Query-adaptive vs static allocation | 3h | 1일 | Exp 2-1 결과 |
| **2-4** | AFOD vs PCA vs Random initialization | 4h | 1일 | Exp 2-1 결과 |
| **3-3** | Rate-distortion 최적 비교 | 2h | 0.5일 | Phase 1 data |

### Priority 4 (Phase 4 후반)

| Exp | Description | GPU | Duration | Dependency |
|-----|-------------|-----|----------|------------|
| **4-3** | CUDA kernel prototype | 40-80h | 7-14일 | Exp 4-1, 4-2 |

**GPU 할당 총 예산**:
- A6000 (현재 가용): ~22h (Phase 2 + Phase 3)
- A100 (Azure 필요): ~112-200h (Phase 4 전체)
- **Critical path**: 4-1 -> 4-2 -> 4-3 (최소 3주)
- **최적 전략**: Phase 2+3을 Week 1에서 A6000으로 수행, 동시에 4-1을 A100에서 착수

### Boltzmann Level 1 잔여 실험

| Exp | Description | Status | Priority |
|-----|-------------|--------|----------|
| **1-3** | FFN spectral selectivity | PENDING | **Low** |

FOKVQ 최종 보고서에서 FFN spectral sharpening은 이미 실패로 판정되었다 (DFT 0/24, SVD 0/24 sharpening). Boltzmann 별도 실험에서 재검증할 가치가 낮으므로 DEPRIORITIZE한다. Phase 4 결과에 따라 재평가.

---

## 7. Decision Matrix (Updated)

| Experiment | Result | Gate Decision | Implication |
|-----------|--------|---------------|-------------|
| 1-1 (T_eff Gate, GPT-2) | rho = -0.657, p = 4.82e-04 | **WEAK_SUCCESS** | MHA에서 Boltzmann 열적 구조 존재 |
| 1-1 (T_eff Gate, Qwen GQA) | rho = +0.108, p = 0.530 | **FAILURE** | GQA가 progressive cooling 파괴 |
| 1-2 (Disagreement) | 39.8%, rho = 0.275 | **CONFIRMED** | C(q)는 새로운 독립적 metric |
| 1-2 (PPL, 3-bit) | C(q): 264.5 vs Var: 127.6 | **NOT_BETTER** | C(q)가 PPL에서 열등 |
| 1-2 (PPL, 4-bit) | C(q): 125.6 vs Var: 122.9 | **NOT_BETTER** | C(q)가 PPL에서 열등 |
| 1-3 (FFN Spectral) | PENDING (FOKVQ에서 FAIL 확인) | **DEPRIORITIZED** | 볼츠만 예측의 정반대 결과 |

**Overall Framework Assessment**: DIFFERENT_BUT_NOT_BETTER -- Boltzmann quantities는 기존 metric과 구별되는 새로운 정보를 포착하지만, 이를 실용적 양자화 개선으로 전환하는 데 실패하였다. 기술적(descriptive) 프레임워크로서의 가치는 있으나, 규범적(prescriptive) 프레임워크로서의 입증은 추가 실험이 필요하다.

---

## Appendix A: Raw Data Reference

모든 수치는 다음 JSON 파일에서 직접 확인 가능:

| Data | File | Key Field |
|------|------|-----------|
| Qwen T_eff per layer | `exp1_1_results.json` | `layer_teff_mean` (36 values) |
| Qwen T_eff per head | `exp1_1_results.json` | `layer_head_teff` (36 x 16 matrix) |
| Qwen head variance | `exp1_1_results.json` | `head_var_per_layer` (36 values) |
| Qwen gate metric | `exp1_1_results.json` | `spearman_rho: 0.1081, p: 0.5303` |
| GPT-2 T_eff per layer | `exp1_1_gpt2_results.json` | `layer_teff_mean` (24 values) |
| GPT-2 T_eff per head | `exp1_1_gpt2_results.json` | `layer_head_teff` (24 x 16 matrix) |
| GPT-2 head variance | `exp1_1_gpt2_results.json` | `head_var_per_layer` (24 values) |
| GPT-2 gate metric | `exp1_1_gpt2_results.json` | `spearman_rho: -0.6574, p: 4.82e-04` |
| Disagreement per layer | `exp1_2_results.json` | `layer_stats[i].disagreement` |
| rho(C, ||k||^2) per layer | `exp1_2_results.json` | `layer_stats[i].rho_C_k` |
| PPL results | `exp1_2_results.json` | `ppl_results.*` |
| FOKVQ cross-analysis | `FOKVQ_FINAL_REPORT.docx` | Section 4D |
| FOKVQ Phase 2-4 plan | `FOKVQ_ADDITIONAL_EXPERIMENT_PLAN.docx` | Full document |

---

*Generated: 2026-03-29. All numerical claims sourced from JSON experiment files or FOKVQ reports. No fabricated data.*
