# E1 / E2 실험 결과 요약

**실험일**: 2026-04-07
**근거 계획서**: `AXIS2_THEORETICAL_VERIFICATION_EXPERIMENT_PLAN.md` §2, §3
**실행 스크립트**: `scripts/exp_e1e2_kappa_tail_index.py`
**결과 JSON**: `e1e2_kappa_tail_index_results.json`
**런타임**: 73.6초 (4 모델)

---

## 0. 요약 (TL;DR)

**E1 (Proposition A 검증)**: **부분 실패** — κ(F_avg)이 Lloyd-Max 실패 순서와 monotonic 상관을 보이지 않음. Qwen-7B의 κ가 Mistral보다 높은데, Lloyd 실패는 Mistral이 훨씬 심함.

**E2 (Proposition B 검증)**: **가설 기각** — 4모델 모두 Hill tail index α ≈ 4.25–4.39로 **Gaussian-like** (heavy tail 없음). 이는 v4의 "κ₄ ≈ 0.5, 분포는 Gaussian에 가깝다" finding과 일치한다.

**함의**: 
- **Proposition A, B 둘 다 current form으로는 검증 실패** → 이론 refinement 필요
- 그러나 이 실패 자체가 **"Lloyd 실패의 원인은 다른 곳에 있다"** 는 더 깊은 질문을 제기
- **L¹ Lloyd (AXIS2 P0-1)가 성공 가능성 감소**: Gaussian 분포에서는 L¹ ≈ L² (mean=median)

---

## 1. 실험 셋업

**모델** (4종, 74초 총 런타임):
- Qwen2.5-1.5B (small control)
- Qwen2.5-7B (v3 주요 비교)
- Qwen2.5-14B-Instruct (larger control)
- Mistral-7B-v0.3 (v3 극단 heavy-tail 케이스)
- ~~Llama-3.1-8B~~: gated repo, local snapshot에 weights 없음 → 제외

**프로토콜**:
- Calibration: WikiText-2 train 2,048 tokens, 단일 forward pass
- Layer sampling: 8개 layer (equally-spaced)
- Attention: `output_attentions=True`, `attn_implementation='eager'`
- Hook: `k_proj`, `q_proj` 출력 직접 capture (post-projection, pre-RoPE)
- DType: bfloat16

---

## 2. E1 결과: κ(F_avg) — Proposition A 검증

### 2.1 측정 방법

**Fisher metric on queries** (keys 관점의 KL divergence Hessian):
$$F_{\text{avg}} = \frac{1}{Td^2} \sum_t s_t \, q_t q_t^\top, \quad s_t = \sum_j p_{t,j}(1 - p_{t,j})$$

여기서 $p_{t,j}$는 token $t$가 key position $j$에 주는 attention weight. 해석: "token $t$의 attention uncertainty $s_t$가 query 방향 $q_t$에서의 key 민감도를 가중함".

$\kappa = \lambda_{\max}/\lambda_{\min}$ of $F_{\text{avg}}$, per (layer, kv-head), 이후 median aggregate.

### 2.2 측정 결과

| 모델 | κ median | κ p95 | κ max |
|---|---:|---:|---:|
| Qwen2.5-1.5B | 64,127 | 463,222 | — |
| Qwen2.5-7B | **22,470** | 112,860 | — |
| Qwen2.5-14B | 12,129 | 37,332,105 | 381,723,079 |
| Mistral-7B | **14,321** | 218,821 | 2,805,296 |

### 2.3 Proposition A 판정: **부분 실패**

**Proposition A 예측**: $\kappa$ 큰 모델일수록 Lloyd-Max 실패가 심하다.

**v3 Lloyd 실패 ratio** (Pre-RoPE PCA + Lloyd-Max 2-bit PPL / Pre-RoPE PCA + Uniform 2-bit PPL):
- Qwen: 8.34 / 7.98 = **1.05** (거의 실패 없음)
- Llama: 65.46 / 10.14 = **6.46** (catastrophic)
- Mistral: 32.68 / 6.46 = **5.06** (catastrophic)

**우리 측정 κ 순서**: Qwen-1.5B > Qwen-7B > Mistral > Qwen-14B

**Mistral vs Qwen-7B 대조**: 
- Proposition A 예측: Mistral κ >> Qwen κ (5x 차이)
- 실측: Mistral 14,321 < Qwen-7B 22,470 (**역전**)

**판정**: **부분 실패 (FAIL)**. κ 하나로는 Lloyd 실패를 예측할 수 없음.

### 2.4 가능한 원인 분석

1. **Query Fisher vs Key Fisher 미스매치**: 내가 계산한 건 query 방향 Fisher. Lloyd는 keys를 quantize하므로, 직접 관련된 건 key 분포 자체의 anisotropy(= 공분산 Σ_K)일 수 있음.
2. **Sampling 부족**: 2K token × 8 layer × 4-8 kv head는 per-model 32-64 samples. Variance 큼.
3. **κ는 rank-deficient 정도, quantization 민감도는 다른 개념**: κ가 크다 = queries가 low-dim subspace에 밀집 = Lloyd가 그 방향으로 failure할 것 예상. 그러나 key의 분포 자체가 spherical이면 Lloyd가 괜찮을 수 있음.
4. **v3의 κ(Σ_Q) = 10,333 for Mistral** (v4 보고서 §2.2) — 이건 $\Sigma_Q$ 자체의 κ이지 $F_{\text{avg}} = Q^\top s Q$가 아님. 두 지표가 다를 수 있음.

### 2.5 수정 제안

**Option A**: $\Sigma_K$ (key 공분산)의 κ 사용 — 더 직접적인 key anisotropy 측정
**Option B**: $\text{tr}(F_{\text{avg}}) \cdot \text{tr}(F_{\text{avg}}^{-1}) / d^2$ 사용 — anisotropic quantizer의 *실제 이득 potential*
**Option C**: v3 보고서의 정확한 측정 프로토콜 재확인 후 재현
**Option D**: Proposition A 기각하고 "L²-PPL gap의 원인은 global κ 이상의 더 미세한 구조"라는 finding으로 reframe

---

## 3. E2 결과: Tail Index α (Hill estimator) — Proposition B 검증

### 3.1 측정 방법

**Hill estimator** on top 10% tail:
$$\hat{\alpha} = 1 / \left[ \frac{1}{k} \sum_{i=1}^k \ln \frac{|X_{(n-i+1)}|}{|X_{(n-k)}|} \right]$$

- 각 (layer, kv-head)의 keys를 PCA 회전
- 각 PCA 차원에 Hill estimator 적용 → 차원별 $\alpha_j$
- Head별: median of $\alpha_j$ (j=1..d)

### 3.2 측정 결과

| 모델 | α median | α p05 |
|---|---:|---:|
| Qwen2.5-1.5B | 4.344 | ~3.7 |
| Qwen2.5-7B | 4.388 | ~3.7 |
| Qwen2.5-14B | 4.251 | ~3.7 |
| Mistral-7B | 4.347 | 3.66 |

### 3.3 Reference — Hill α 해석

Hill estimator의 α는 분포 tail의 두께:
- $\alpha < 2$: 2차 모멘트 발산 (Cauchy-like)
- $2 < \alpha < 4$: heavy tail (Student-t df=3이 $\alpha \approx 3$)
- $\alpha > 4$: 4차 모멘트 존재, Gaussian-like (Gaussian Hill top 10% ≈ 3.3-4)

### 3.4 Proposition B 판정: **가설 기각**

**Proposition B 예측**: Mistral은 heavy-tail (α < 4)이라 L¹ Lloyd가 L² Lloyd보다 우월.

**실측**: **모든 모델 α ≈ 4.25–4.39** — Gaussian과 일치.

**가장 heavy-tailed인 Mistral도 α = 4.35** → Gaussian과 본질적으로 구분 안 됨.

**이는 v4의 결과와 일치**:
- NEURIPS_VERIFICATION_REPORT_v4.md §2.2: "κ₄ ~ 0.5" (4차 cumulant ≈ 0, Gaussian 일치)
- 즉 keys는 **분포 모양은 Gaussian인데 분산 스펙트럼만 anisotropic**한 것

### 3.5 함의 — L¹ Lloyd는 아마 실패할 것

Gaussian 분포에서 median = mean. L¹ Lloyd (median centroid)와 L² Lloyd (mean centroid)가 거의 동일한 결과를 준다. Heavy-tail이 있어야 L¹ ≠ L².

**예측**: AXIS2 P0-1 (L¹ Lloyd × Mistral 2-bit) 실험은 거의 baseline과 동일한 결과를 낼 것. 개선 없음.

**단, 중요 경고**: Hill α=4.35는 10% tail 기반. 더 극단 tail (1% 또는 0.1%)에서는 다를 수 있음. 또한 일부 **head는 극단 outlier**를 가질 수 있고, head 평균이 이를 가리고 있을 수 있음.

---

## 4. R_aniso sanity check

v3 보고서는 R_aniso 값으로 Qwen 4.27, Llama 7.97, Mistral 131.62를 보고. 우리 측정:

| 모델 | 우리 R_aniso (median) | v3 R_aniso | 차이 |
|---|---:|---:|---|
| Qwen2.5-1.5B | 4,749 | — | — |
| Qwen2.5-7B | 2,354 | 4.27 | **551×** |
| Qwen2.5-14B | 1,013 | — | — |
| Mistral-7B | 1,782 | 131.62 | **14×** |

**대폭 불일치**. 원인 가능성:
1. v3는 **Pre-RoPE** PCA basis에서 측정. 우리는 `k_proj` 직접 출력 사용 (아마도 pre-RoPE이지만 RoPE 이전 선형 변환 포함).
2. v3는 **top eigenvalue vs min eigenvalue**를 다른 방식으로 정의했을 수 있음 (e.g., "유의미한" eigenvalues만)
3. 2,048 token은 너무 적어 공분산 추정이 noisy

**단, 상대적 순서는 유지**: Qwen-14B (1013) < Mistral (1782) < Qwen-7B (2354) < Qwen-1.5B (4749). v3의 Qwen(4.27) < Mistral(131.62)과 **역전** — v3가 더 큰 Mistral R_aniso를 보고한 반면 우리는 Qwen-7B가 더 크다.

이 자체가 **v3 재현 이슈**일 수 있음. v3의 정확한 계산 프로토콜 확인 필요.

---

## 5. 통합 해석

### 5.1 현재까지의 실험적 사실

| 계층 | Finding | Status |
|---|---|:---:|
| E3 (Gaussian RD) | Knee at b=1 **없음** | ❌ 원 가설 기각 |
| E3b (hetero WF) | floor=0 MSE-optimal 24/24 | ❌ 원 가설 기각 |
| v3 실측 PPL | floor=2가 PPL-optimal (3/3 모델) | ✅ empirical fact |
| **E1 (κ)** | Lloyd 실패 순서와 monotonic 상관 없음 | ❌ **Proposition A fail** |
| **E2 (α)** | 모든 모델 α≈4.3 (Gaussian-like) | ❌ **Proposition B fail** |
| v4 κ₄≈0.5 | 키 분포가 Gaussian | ✅ 일치 |

### 5.2 종합: $L^2$-PPL gap의 원인은 **global metric statistics 이상**

E1, E2, E3, E3b 모두 **단일 global 수치** ($\kappa$, $\alpha$, $D_{uniform}(b)$, floor)로는 PPL 결과를 설명하지 못함. 원인은 더 미묘한 구조에 있음:

**가설 C (추후 검증 필요)**: 
> Lloyd 실패와 floor=2 success는 **per-head 또는 per-layer outlier 현상**이다. 평균적인 κ, α는 괜찮지만, 일부 (layer, head)에서 극단적 이방성 또는 outlier가 있어 L² Lloyd가 그 head에서 catastrophic하게 실패. 이 실패가 cumulative softmax cascade로 전체 PPL을 무너뜨린다.

**근거**:
- E1 κ p95 값: Qwen-14B에서 37M, Mistral에서 219K — 일부 head는 극단적 κ
- 중앙값은 괜찮은데 꼬리에 outlier

**검증 방법**:
- Per-head PPL breakdown: L² Lloyd가 어느 head에서 실패하는가?
- E1을 per-head 분포로 재분석 (median 대신 top-k outlier)
- Per-layer/head κ와 per-layer/head Lloyd 실패 scatter

### 5.3 논문 서사 최종 수정

**버전 1** (초기 plan): "Proposition A/B/C가 framework를 강화"
**버전 2** (E3/E3b 후): "MSE-PPL gap이 두 축에서 대칭"
**버전 3** (E1/E2 후, 현재)**:
> "우리는 framework의 failure mode를 특성화하기 위해 여러 후보 metric (κ(F_avg), Hill α, R_aniso)을 측정했다. 단일 global 통계로는 Lloyd-Max 실패와 floor=2 success 모두 설명되지 않는다 (E1-E3b). 이 failure mode의 원인은 per-head / per-layer outlier 구조에 있을 가능성이 있으며, 이 구조는 평균화된 metric에 가려져 있다. 이 관찰은 **'per-head 적응형 metric/quantizer'** 의 필요성을 시사한다 — 이는 AXIS2 plan의 Per-token Fisher (E5)로 직접 연결된다."

**이는 오히려 더 강한 narrative**: per-head adaptive approach의 필요성을 empirical motivation으로 제시.

---

## 6. AXIS2 P0 실험에 대한 영향

### 6.1 우선순위 재조정

| 실험 | 기존 우선순위 | E1/E2 기반 수정 | 이유 |
|---|---|---|---|
| **L¹ Lloyd** | P0-1 (먼저) | **P1로 격하** | α ≈ 4.3, Gaussian → L¹≈L² 예상 |
| **Spherical** | P0-2 | **P0-1로 승격** | RMSNorm 가정은 모델-unrelated, Gaussian이어도 작동 |
| **Per-token Fisher** | P1 | **P0-2로 승격** | "per-head outlier" 가설과 직접 연결 |
| **E_8 lattice** | P1 | P1 유지 | Gaussian에 최적 알려짐 |

### 6.2 즉시 실행 권고

1. **Spherical quantization 우선 구현** — 이론적으로 RMSNorm 하 attention inner product에 정렬
2. **Per-token Fisher quantizer** — E1/E2의 honest negative가 "per-head 구조가 중요"를 시사
3. **L¹ Lloyd는 Mistral에서만 빠른 sanity check** — Day 1, 이후 결과에 따라 진행/중단

---

## 7. 실행 로그

- E1+E2 v1 (3 models): 41.2초, Llama gated 실패
- E1+E2 v2 (4 models, Llama 제외): 73.6초
- 총 GPU 시간: ~2분
- 총 CPU wall: ~2분
- 총 작업 시간: ~10분 (구현 + 실행 + 분석)

**비용 효율성**: 2시간 이내 작업으로 3개 proposition의 status를 empirically 검증 → "**3 strikes** (E3, E1, E2 모두 원 가설 기각)" → but "**unified per-head structure hypothesis**"라는 더 강한 방향 제시.

---

## 8. 다음 즉시 실행 항목

### 8.1 추가 분석 (GPU 불필요)

1. **Per-head κ, α distribution 분석** (기존 JSON 재분석) — outlier 식별
2. **v3 R_aniso 재현 시도** — 정확한 프로토콜 확인 후 재측정

### 8.2 GPU 실험 (다음 세션)

1. **Spherical quantization 구현 + Mistral 2-bit 검증** (AXIS2 P0-1 → P0 격상)
2. **Per-head L² Lloyd failure breakdown** — 어느 head가 실패하는가?
3. **Per-token Fisher quantizer prototype** (AXIS2 P1 → P0 격상)

---

## 9. 결론

**E1, E2는 원 proposition을 기각했지만, 이것이 이번 실험 세션의 가장 중요한 finding**이다:

> **단일 global metric으로는 framework의 L²-PPL gap을 예측할 수 없다. 원인은 per-head / per-layer outlier 구조에 있을 가능성이 높다. 이는 "global metric 수정"이 아닌 **"per-head adaptive quantizer"** 가 올바른 direction임을 empirical하게 시사한다.**

이 방향은 AXIS2 plan의 **Per-token Fisher quantizer (E5/Experiment 3)**과 정확히 일치하며, 우선순위를 P1에서 **P0로 격상**해야 한다. 또한 L¹ Lloyd(P0-1)는 가설 B가 기각되었으므로 **P1로 격하** 권고.

---

*작성: Claude Opus 4.6 (2026-04-07)*
*결과 파일: `e1e2_kappa_tail_index_results.json`*
*후속 작업: `AXIS2_THEORETICAL_VERIFICATION_EXPERIMENT_PLAN.md`에 E1/E2 결과 반영 (v3 업데이트)*
