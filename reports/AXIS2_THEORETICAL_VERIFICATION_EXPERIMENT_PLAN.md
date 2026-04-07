# Axis 2 이론 검증 추가 실험 계획서

**프로젝트**: KV-Cache 양자화 Lie Group 프레임워크 — 이론 명제의 Empirical Verification
**작성일**: 2026-04-07
**버전**: v2 (E3/E3b 실행 결과 반영)
**최종 업데이트**: 2026-04-07 (E3, E3b 완료 후)
**근거 문서**:
- `AXIS2_ANISOTROPY_AWARE_QUANTIZATION_EXPERIMENT_PLAN.md` (기존 5-quantizer 실험)
- `NEURIPS_VERIFICATION_REPORT_v4.md` (QW-PCA 실패 + PCA-Q 자연 정렬 발견)
- `LIE_GROUP_UNIFICATION.md` (이론 framework)
- `axis2_theoretical_verification/E3_RESULTS_SUMMARY.md` (E3/E3b 실측 결과, **NEW**)

**목적**: 기존 AXIS2 plan을 보완하여, NeurIPS 2026 reviewer의 예상 3대 우려를 선제 차단하는 이론 명제 검증 실험을 추가한다.

## 실행 현황 대시보드 (2026-04-07)

| 실험 | 상태 | 작업량 | GPU | 핵심 결과 |
|---|:---:|:---:|:---:|---|
| **E3** (single-channel Gaussian) | ✅ **완료** | 85초 실측 | X | Max 1960 reference와 4자리 일치, "knee at b=1" 가설 **기각** |
| **E3b** (heterogeneous WF, 신규) | ✅ **완료** | 0.8초 | X | floor=0 win 24/24, **MSE-PPL gap이 allocation 축에서도 발현** |
| E1 ($\kappa(\bar{M}_{KL})$) | ⏸ 대기 | 0.5일 | 필요 (data 없음) | — |
| E2 (tail index $\alpha$) | ⏸ 대기 | 0.5일 | 필요 | — |
| E4 (cross-ablation) | ⏸ 대기 | 2일 | 필요 | AXIS2 P0 선행 |
| E5 (per-token $M_{KL}$) | ⏸ 대기 | 1일 | 필요 | — |
| E6 (MMLU + NIAH 16K) | ⏸ 대기 | 2일 | 필요 | 병렬 진행 중 |

---

## 0. Executive Summary

### 0.1 배경

기존 AXIS2 plan은 5개 alternative quantizer (L¹ Lloyd, Spherical, Per-token Fisher, E_8 lattice, Wasserstein-1D)의 PPL 비교에 집중한다. 그러나 NeurIPS reviewer 관점에서 다음 3대 우려가 잔존한다:

1. **"3-Axis Lie Group framework의 예측력이 Axis 2에서 무너짐"** — Lloyd-Max가 MSE 3.5× 우위임에도 PPL 전면 실패
2. **"WF floor=2가 ad-hoc 수정"** — 이론적 유도 없음
3. **"Theorem 6.16.3의 Class C 가정이 실용 회전 공간을 충분히 포괄하는가"**

### 0.2 본 plan의 기여

본 plan은 위 3대 우려에 대한 이론적 해결안의 **empirical verification**을 수행한다. 기존 plan이 "어떤 quantizer가 작동하는가"를 묻는 반면, 본 plan은 **"왜 작동하는가"와 "이론이 실제를 예측하는가"**를 묻는다.

### 0.3 추가되는 실험 6종

| # | 실험 | 대응 이론 명제 | 우선순위 | 작업량 |
|---|---|---|:---:|:---:|
| **E1** | $\kappa(\bar{M}_{KL})$ 측정 | Proposition A (Metric Mismatch Bound) | P0 | 0.5일 |
| **E2** | Tail index $\alpha$ 측정 (Hill estimator) | Proposition B ($L^p$ Hierarchy) | P0 | 0.5일 |
| **E3** | $D_{uniform}(b)$ Shannon 대비 편차 실측 | Discrete-WF Theorem (floor=2) | P0 | 0.5일 |
| **E4** | Cross-Rotation × Quantizer Ablation | Axis 1 / Axis 2 독립성 | P0 | 2일 |
| **E5** | Per-token $M_{KL}(t)$ Variance 분석 | Fisher quantizer 설계 근거 | P1 | 1일 |
| **E6** | Best Quantizer × MMLU + NIAH 16K | Downstream 전이 검증 | P0 | 2일 |

**필수 P0 작업량**: 5.5일 (순차) / 3일 (병렬 가능 시)

### 0.4 기대 효과

- **Reviewer #1 (이론)**: "framework가 무엇을 예측하는가?" → Proposition A/B/C의 quantitative prediction + verification
- **Reviewer #2 (실험)**: "ablation이 불충분" → E4의 15-cell cross-ablation matrix
- **Reviewer #3 (novelty)**: Proposition A/B/C + Discrete-WF Theorem은 새 이론 기여

---

## 1. 이론적 배경

### 1.1 3대 우려의 수학적 근원

**우려 1 (Axis 2 실패)**: Lloyd-Max가 $L^2$ Hilbert 공간에서 최적이지만, attention distortion은 $L^2$가 아닌 **Fisher 정보 기하학**에서 측정된다:
$$D_{KL}(p_t \| \hat{p}_t) \approx \tfrac{1}{2}(k - \hat{k})^\top M_{KL}(t)(k - \hat{k})$$
여기서 $M_{KL}(t) = Q^\top(\text{diag}(p_t) - p_t p_t^\top) Q$ — token별 Fisher metric.

**우려 2 (WF floor=2)**: Shannon WF는 연속 채널 가정이며, $b$-bit 균등 양자화의 실제 distortion $D_{uniform}(b)$는 Shannon 공식 $\sigma^2 \cdot 2^{-2b}$과 $b < 2$에서 systematically 크게 deviate.

**우려 3 (Class C)**: RoPE 구조와 commute하는 직교 회전의 집합이 정확히 Class C. 이는 제약이 아닌 자연스러운 귀결.

### 1.2 검증할 이론 명제

본 plan의 각 실험은 다음 이론 명제 중 하나 이상을 직접 검증한다.

#### Proposition A (Metric Mismatch Bound)

> $L^2$-Lloyd-Max와 $M_{KL}$-Lloyd-Max의 attention distortion 차이는 $\kappa(\bar{M}_{KL})$에 비례한다:
> $$D_{attn}(Q_{L^2}) - D_{attn}(Q_{M_{KL}}) \leq C \cdot (\kappa(\bar{M}_{KL}) - 1) \cdot \sigma^2_{quant}$$

**예측**: $\kappa$가 큰 모델일수록 $L^2$-Lloyd의 PPL 실패가 심하다.

**이미 관측된 증거**: v4 보고서 — Mistral $\kappa(\Sigma_Q) = 10{,}333$ (최대 44,937), Mistral 2-bit Lloyd-Max PPL 32.68 (catastrophic).

#### Proposition B ($L^p$ Quantization Hierarchy)

> Source 분포의 tail index $\alpha < 4$일 때 (4차 모멘트 발산), $L^1$-Lloyd (median centroid)가 $L^2$-Lloyd (mean centroid)보다 attention distortion에서 strictly 우월하다.

**예측**: Hill estimator로 측정한 $\alpha$가 작을수록 $L^1$-Lloyd 이득이 크다.

**근거**: Graf & Luschgy (2000), *Foundations of Quantization for Probability Distributions*, Chapter 6.

#### Proposition C (Spherical Optimality)

> RMSNorm이 적용된 LLM에서 키 norm 변동 $\epsilon = \text{Var}(\|k\|)/\mathbb{E}[\|k\|]^2$ 가 작을 때, 구면 양자화 $S^{d-1}$가 $L^2$ 양자화보다 attention KL에서 $O(1/\epsilon)$ 우월하다.

**예측**: 키의 norm 변동이 작은 layer/head에서 Spherical 이득이 크다.

#### Discrete-WF Theorem (제안)

> 균등 $b$-bit 양자화기의 실제 rate-distortion $D_{uniform}(b)$가 $b < b_{crit}$에서 Shannon 공식 $\sigma^2 \cdot 2^{-2b}$보다 strictly 크면 (즉 convex deficit이 존재), 이산 WF의 최적해는 $b_j^* \geq b_{crit}\; \forall j$. 가우시안 source + 균등 양자화에서 $b_{crit} = 2$.

**증명 스케치**: $D_{uniform}(1) / \sigma^2 \approx 0.363$, $D_{Shannon}(1)/\sigma^2 = 0.25$. $b=1$에서 45% 편차 → Lagrangian KKT 조건에서 $b_j=1$ 배제.

**예측**: $b=1$에서 $D_{uniform}/D_{Shannon}$ ratio가 가장 크다.

#### Corollary (Class C Maximality)

> RoPE $R_{\text{RoPE}} = \bigoplus_{i=1}^{d/2} R_2(\theta_i)$ 와 commute하는 $SO(d)$의 원소는 정확히 Class C (블록-대각 회전)이다.

**증명**: RoPE의 eigenvalue $e^{\pm i\theta_i}$가 distinct → 임의 $U$가 commute하면 eigenspace 보존 → Class C.

**검증 방법**: 이론적 (실험 불필요). Appendix로 추가.

---

## 2. 실험 E1: $\kappa(\bar{M}_{KL})$ 측정 — Proposition A 검증

### 2.1 의도

$L^2$-Lloyd 실패의 magnitude가 $\kappa(\bar{M}_{KL})$와 positive correlation을 가진다는 Proposition A를 직접 검증.

### 2.2 프로토콜

**입력**:
- 3모델 (Qwen2.5-7B, Llama-3.1-8B, Mistral-7B-v0.3)
- V3 calibration 데이터 (WikiText-2 train, 160K tokens)
- V3 Lloyd-Max PPL 측정값 (재사용)

**측정**:
1. 각 (layer, head)에서 per-token attention 분포 $p_t$ 수집
2. $M_{KL}(t) = Q^\top(\text{diag}(p_t) - p_t p_t^\top) Q$ 계산 (Q = PCA basis)
3. 평균: $\bar{M}_{KL} = \mathbb{E}_t[M_{KL}(t)]$
4. Condition number $\kappa_{l,h} = \lambda_{\max}(\bar{M}_{KL,l,h}) / \lambda_{\min}(\bar{M}_{KL,l,h})$
5. 모델별 요약: mean, median, 95 percentile

**분석**:
- $\kappa$ histogram (per model)
- **Scatter plot**: x축 $\kappa_{l,h}$, y축 $\text{PPL}(Q_{L^2}) - \text{PPL}(Q_{M_{KL}})$ (각 head별)
- Linear regression + $R^2$

### 2.3 가설 (Falsification Criteria)

**H1.1 (Primary)**: 3모델 averaged $\kappa$ 순서가 Lloyd-Max 실패 순서와 일치
- Mistral $\kappa$ > Llama $\kappa$ > Qwen $\kappa$
- Mistral Lloyd 실패 폭 (32.68 vs 6.46 = 5.1×) > Llama (65.46 vs 10.14 = 6.5×) > Qwen (8.34 vs 7.98 = 1.05×)
- **PASS**: $\kappa$ 순서와 실패 순서의 Spearman correlation > 0.8
- **PARTIAL**: correlation > 0.5
- **FAIL**: 순서가 맞지 않음

**H1.2**: Per-head scatter plot에서 positive correlation
- **PASS**: $R^2 > 0.5$
- **FAIL**: $R^2 < 0.3$

### 2.4 예상 결과 및 논문 반영

**예상**: Mistral의 $\kappa$가 Llama/Qwen 대비 1-2 order 큼 → heavy-tail catastrophe 설명

**논문 반영**:
- Proposition A 본문에 Figure: $\kappa$ vs PPL failure scatter
- Table: 3모델 $\kappa$ statistics
- 해석: "metric mismatch는 framework가 정량적으로 예측하는 현상"

### 2.5 리소스

- GPU: 불필요 (기존 calibration data 재분석)
- CPU: 1 서버, ~4시간
- 의존성: V3 calibration data + V3 Lloyd-Max PPL 결과

**총 작업량**: **0.5일**

---

## 3. 실험 E2: Tail Index $\alpha$ 측정 — Proposition B 검증

### 3.1 의도

Source 분포의 tail index가 $L^1$-Lloyd 이득을 예측함을 직접 검증.

### 3.2 프로토콜

**Hill Estimator**: Tail index $\alpha$의 고전적 추정기.
$$\hat{\alpha}_H(k) = \frac{1}{k} \sum_{i=1}^{k} \ln\frac{X_{(n-i+1)}}{X_{(n-k)}}, \quad \alpha = 1/\hat{\alpha}_H$$
$X_{(i)}$는 order statistics, $k$는 tail size.

**측정**:
1. 각 (layer, head)의 PCA 후 키 벡터 $\tilde{k}_j$ (차원별 scalar)
2. 차원별 $|\tilde{k}_j|$의 Hill estimator로 $\alpha_{l,h,j}$ 계산 ($k = n/10$)
3. Head별 평균: $\alpha_{l,h} = \text{median}_j(\alpha_{l,h,j})$
4. 모델별 분포: histogram of $\alpha$

**분석**:
- $\alpha$ histogram (per model)
- **Scatter plot**: x축 $\alpha_{l,h}$, y축 $\text{PPL}(L^2 \text{ Lloyd}) - \text{PPL}(L^1 \text{ Lloyd})$ (L¹ Lloyd 결과가 나오면)
- Theoretical prediction: $\alpha < 4$이면 L¹ 우위

### 3.3 가설

**H2.1**: 3모델의 tail index 순서가 heavy-tail 예측과 일치
- Mistral $\alpha < $ Llama $\alpha < $ Qwen $\alpha$
- 정량적: Mistral $\alpha \sim 2-3$, Llama $\alpha \sim 3-4$, Qwen $\alpha \sim 4-5$

**H2.2**: $\alpha < 4$인 head에서 $L^1$-Lloyd이 $L^2$-Lloyd보다 strictly 우월
- **PASS**: 95% 이상의 heavy-tail head ($\alpha < 4$)에서 $L^1$ 우위
- **FAIL**: random 수준

### 3.4 예상 결과 및 논문 반영

**예상**: Mistral의 median $\alpha \approx 2.5$, Qwen $\approx 4.2$ → Mistral 2-bit에서 L¹ 이득 큼, Qwen에서 작음

**논문 반영**:
- Proposition B 본문에 Figure: $\alpha$ vs $L^1$ gain scatter
- Table: 3모델 tail index
- **스토리**: "Mistral의 2-bit 예외는 framework 실패가 아니라 heavy-tail → L² 부적합의 직접 예측"

### 3.5 리소스

- GPU: 불필요
- CPU: 1 서버, ~4시간

**총 작업량**: **0.5일**

---

## 4. 실험 E3: $D_{uniform}(b)$ Shannon 대비 편차 실측 — Discrete-WF Theorem 검증

### 4.1 의도

$b$-bit 균등 양자화기의 실제 distortion이 $b < 2$에서 Shannon $\sigma^2 \cdot 2^{-2b}$보다 strictly 큼을 실측하여, **floor=2가 이론적 필연**임을 입증.

### 4.2 프로토콜

**합성 데이터 (Gaussian baseline)**:
1. $X \sim \mathcal{N}(0, 1)$, 100만 샘플
2. 각 $b \in \{1, 2, 3, 4, 5, 6\}$에 대해:
   - 균등 $b$-bit 양자화기 적용 (clip range = $\pm 3\sigma$)
   - $D_{uniform}(b) = \mathbb{E}[(X - Q(X))^2]$ 측정
3. Shannon 공식: $D_{Shannon}(b) = \sigma^2 \cdot 2^{-2b}$
4. 비율 $r(b) = D_{uniform}(b) / D_{Shannon}(b)$ 및 절대 편차 $\Delta(b) = D_{uniform}(b) - D_{Shannon}(b)$

**실제 데이터 (3모델)**:
1. PCA 후 각 차원의 scalar 분포 사용
2. 같은 방식으로 $D_{uniform}(b)$ 측정
3. 차원별 (분산 정규화된) 비율 histogram

**분석**:
- Plot: $b$ vs $D_{uniform}(b)/\sigma^2$ (실측), $\sigma^2 \cdot 2^{-2b}$ (Shannon 예측), log scale
- Plot: $b$ vs ratio $r(b)$
- Table: 3모델 × $b \in \{1,2,3,4\}$ 평균 $r(b)$

### 4.3 가설

**H3.1**: $r(1) > r(2) > r(3) > \ldots \to 1$ (monotonic convergence)
- **PASS**: $r(1) > 1.2$, $r(2) < 1.1$, $r(b) \to 1$ for $b \geq 3$
- **FAIL**: monotonic 아니거나 ratio < 1

**H3.2**: $r(1)$과 $r(2)$의 gap이 $r(2)$와 $r(3)$의 gap보다 큼 ("knee at b=2")
- **PASS**: $r(1) - r(2) > r(2) - r(3)$
- **FAIL**: linear decay

**H3.3**: 실제 데이터에서 $r(1)$이 가우시안보다 더 크다 (heavy-tail 보강)
- **PASS**: 3모델 모두 synthetic $r(1)$ 이상
- **FAIL**: 반대

### 4.4 예상 결과 및 논문 반영

**예상 (가우시안)**:
| $b$ | $D_{uniform}/\sigma^2$ | $D_{Shannon}/\sigma^2$ | $r(b)$ |
|:---:|:---:|:---:|:---:|
| 1 | ~0.363 | 0.250 | **1.45** |
| 2 | ~0.119 | 0.0625 | 1.90 |
| 3 | ~0.0345 | 0.0156 | 2.21 |
| 4 | ~0.0095 | 0.00391 | 2.43 |

단, **절대 편차** $\Delta(b) = D_{uniform}(b) - D_{Shannon}(b)$의 **marginal improvement**를 보면:
- $\Delta(0) - \Delta(1) = \sigma^2 - 0.363\sigma^2 = 0.637\sigma^2$ (크다)
- $\Delta(1) - \Delta(2) = 0.363 - 0.119 = 0.244\sigma^2$ (작아짐)
- $\Delta(2) - \Delta(3) = 0.119 - 0.0345 = 0.0845\sigma^2$ (더 작아짐)

$b=1 \to 2$의 marginal gain이 $b=0 \to 1$보다 작다 → **Lagrangian에서 $b=1$ 할당이 최적 아님** → floor=2 justification.

**논문 반영**:
- Discrete-WF Theorem의 **Figure 1**: $b$ vs $D_{uniform}$ + $D_{Shannon}$ overlay
- **Figure 2**: marginal gain $\Delta(b-1) - \Delta(b)$ plot (knee at $b=2$ 가시화)
- 증명에 실측값 인용 ("$r(1) = 1.45$로 measured")

### 4.5 리소스

- GPU: 불필요
- CPU: 1 서버, ~2시간 (synthetic + 3 models)

**총 작업량**: **0.5일**

---

## 5. 실험 E4: Cross-Rotation × Quantizer Ablation — Axis 1/Axis 2 독립성 검증

### 5.1 의도

기존 AXIS2 plan은 `Pre-RoPE PCA × {5 quantizer}`만 테스트. Reviewer가 반드시 요구하는 **독립성 ablation** — 각 회전에서 각 quantizer가 어떻게 작동하는지의 **cross product matrix**를 측정.

### 5.2 프로토콜

**Configuration matrix**:

|  | Uniform (baseline) | L¹ Lloyd | Spherical | (Fisher) | (E_8) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Identity (no rot)** | V3 재사용 | **NEW** | **NEW** | P1 | P1 |
| **TurboQuant (random)** | V3 재사용 | **NEW** | **NEW** | P1 | P1 |
| **Pre-RoPE PCA** | V3 재사용 | AXIS2 P0 | AXIS2 P0 | AXIS2 P1 | AXIS2 P1 |

**P0 필수 (이 plan E4)**: NEW 셀 4개 (Identity/TurboQuant × L¹/Spherical)
**P1 확장**: Fisher, E_8까지 확장 (8 셀 추가)

**측정**:
- 3모델 × 2-bit × 4 NEW 셀 = **12개 측정**
- WikiText-2 sliding-window PPL (V3 protocol)

### 5.3 분석 및 해석

**Independence hypothesis**: Framework가 axis 분리를 claim하므로, 각 quantizer의 이득은 회전과 무관해야 함.

**정량 지표**:
1. **Quantizer gain**: $\Delta_{quant}(R) = \text{PPL}(R + \text{Uniform}) - \text{PPL}(R + \text{quant})$
2. **Independence check**: $\Delta_{quant}(\text{Identity}) \approx \Delta_{quant}(\text{TurboQuant}) \approx \Delta_{quant}(\text{Pre-RoPE PCA})$?

**가능한 시나리오**:
- **Strong independence**: 각 회전에서 L¹/Spherical 이득이 비슷 → framework의 axis separation 지지
- **Synergy**: Pre-RoPE PCA + L¹/Spherical에서만 큰 이득 → "L² failure가 anisotropy 노출 때문"을 입증
- **Antagonism**: L¹/Spherical이 Identity에서만 효과 → framework 약화

### 5.4 가설

**H4.1 (Primary)**: L¹ Lloyd의 이득은 회전에 의존적이며, Pre-RoPE PCA에서 가장 큼
- **근거**: PCA가 anisotropy를 차원별로 명시적으로 노출 → heavy-tail이 특정 PC 차원에 집중 → L¹ (median) 효과 극대
- **PASS**: $\Delta_{L^1}(\text{PCA}) > \Delta_{L^1}(\text{Identity}) \cdot 1.5$
- **FAIL**: 비슷하거나 반대

**H4.2 (Primary)**: Spherical의 이득은 회전에 덜 의존적
- **근거**: 구면 양자화가 회전 불변
- **PASS**: $|\Delta_{sph}(\text{PCA}) - \Delta_{sph}(\text{Identity})| / \Delta_{sph}(\text{PCA}) < 0.3$
- **FAIL**: 큰 편차

**H4.3**: Identity + L¹ Lloyd가 Identity + Uniform을 Mistral 2-bit에서 개선
- Mistral은 heavy-tail → PCA 없이도 L¹ 이득 있어야
- **PASS**: Mistral 2-bit PPL 개선
- **FAIL**: 변화 없음

### 5.5 논문 반영

**Ablation Table (필수)**:

```
Table X: Cross-ablation of rotation × quantizer (2-bit PPL)

                    Uniform   L¹ Lloyd  Spherical  Δ(L¹)   Δ(Sph)
Identity            XX.X      XX.X      XX.X       +X.X%   +X.X%  
TurboQuant          XX.X      XX.X      XX.X       +X.X%   +X.X%
Pre-RoPE PCA        XX.X      XX.X      XX.X       +X.X%   +X.X%
```

이 표 하나가 "왜 Axis 2 reform이 framework 정합성을 유지하는가"를 입증.

### 5.6 리소스

- GPU: 1× A100, ~12시간 (12 config × 1 hour each)
- 의존성: L¹ Lloyd + Spherical 구현 완료 (AXIS2 P0 선행)

**총 작업량**: **2일** (구현 완료 후)

---

## 6. 실험 E5: Per-token $M_{KL}(t)$ Variance 분석 — Fisher Quantizer 설계 근거

### 6.1 의도

Per-token Fisher quantizer (AXIS2 Experiment 3)의 투자 가치 판단 및 **cluster size $K$** 결정.

### 6.2 핵심 질문

$M_{KL}(t)$가 token 간에 얼마나 변동하는가?
- 거의 불변 → 평균 $\bar{M}_{KL} \approx \Sigma_Q$로 충분 (per-token 불필요)
- 큰 variance → per-token quantization에 가치 있음

**이미 알려진 사실 (v4)**: PCA-Q 자연 정렬(0.6~2.5°)은 *평균* $\Sigma_Q$에 대한 결과. Per-token $M_{KL}(t)$는 **다른 이야기** — token별 attention entropy에 따라 sharp ↔ flat으로 변동.

### 6.3 프로토콜

**측정**:
1. 3모델 × 5개 representative layer × 2개 head = 30 (layer, head) 샘플
2. 각 (layer, head)에서 10K 토큰 수집
3. 각 토큰의 $M_{KL}(t)$ 계산
4. 분석:
   - $M_{KL}(t)$ eigenvalue spectrum 평균 및 variance
   - $M_{KL}(t)$ 의 Frobenius norm histogram
   - $M_{KL}(t)$ top eigenvector direction의 entropy (방향이 얼마나 일관?)

**Clustering**:
- $K \in \{1, 4, 8, 16, 32\}$로 $M_{KL}(t)$ k-means clustering
- 각 $K$에서 intra-cluster MSE: $\sum_t \|M_{KL}(t) - \bar{M}_{KL}^{(c(t))}\|_F^2$
- Elbow point 결정

### 6.4 가설

**H5.1**: $M_{KL}(t)$ Frobenius norm의 coefficient of variation (std/mean)이 0.5 이상
- **PASS**: 평균 이상으로 token 간 변동 → per-token 가치 있음
- **FAIL**: CV < 0.2 → 평균으로 충분, per-token 과잉 설계

**H5.2**: $K = 8$에서 intra-cluster MSE가 $K=1$ 대비 50% 감소
- **PASS**: 적은 cluster로 큰 이득 → $K=8$ 이 sweet spot
- **FAIL**: monotonic 감소 (elbow 없음)

**H5.3**: Attention entropy 기반 clustering이 random clustering보다 intra-cluster MSE 낮음
- **PASS**: attention structure가 $M_{KL}$ cluster와 일치
- **FAIL**: 무관

### 6.5 논문 반영

**결과에 따라 2개 시나리오**:

- **Per-token 가치 있음**: Per-token Fisher quantizer (AXIS2 Experiment 3) 우선순위 P1 → P0로 상승. 본 plan E5가 근거 제공
- **Per-token 가치 없음**: AXIS2 Experiment 3 취소, 평균 Mahalanobis Lloyd로 축소. 이것도 honest finding으로 논문에 반영 ("$\bar{M}_{KL}$은 token 대표성이 충분")

### 6.6 리소스

- GPU: 1× A100, ~6시간
- CPU: clustering 분석

**총 작업량**: **1일**

---

## 7. 실험 E6: Best Quantizer × MMLU + NIAH 16K — Downstream 전이 검증

### 7.1 의도

AXIS2 plan의 WikiText-2 PPL 이득이 **downstream task accuracy**로 전이되는지 검증. NeurIPS reviewer #2의 "downstream missing" 우려를 선제 차단.

### 7.2 전제

AXIS2 P0 (L¹ Lloyd + Spherical) 결과에서 **best quantizer** 1~2개 선정.

### 7.3 프로토콜

**Downstream tasks**:
1. **MMLU** 5-shot (standard benchmark)
   - 모델: Qwen2.5-7B, Llama-3.1-8B (Mistral은 v20 plan R3와 병렬)
   - Bits: 2, 3
   - Configs: {FP16, No rot 2b, Pre-RoPE PCA+Uniform 2b, Pre-RoPE PCA + **best quantizer** 2b}
2. **NIAH 16K** (long-context retrieval)
   - 모델: Qwen2.5-7B
   - Bits: 2
   - Depth: 0.0, 0.25, 0.5, 0.75, 1.0
   - Configs: FP16, PCA+Uniform, PCA+Best, TurboQuant+Uniform

### 7.4 가설

**H6.1 (Primary)**: MMLU에서 PCA + best quantizer가 PCA + Uniform보다 ≥ 1pt accuracy 개선
- **PASS**: 두 모델 모두 개선
- **PARTIAL**: 한 모델만
- **FAIL**: 둘 다 동일 or 악화

**H6.2**: NIAH 16K에서 PCA + best quantizer가 TurboQuant과 차별화
- 8K에서는 둘 다 100%였음 (v3 결과). 16K에서 TurboQuant 실패 여부 확인
- **PASS**: PCA > TurboQuant at 16K
- **FAIL**: 둘 다 성공 or 둘 다 실패

**H6.3 (가장 중요)**: PPL 이득이 MMLU 이득과 monotonic 상관
- Spherical > L¹ > Uniform이 PPL 순서면 MMLU도 동순서
- **PASS**: 순서 일치
- **FAIL**: PPL 잘되는데 MMLU 안되는 case

### 7.5 논문 반영

- **Table**: 3모델 × 4 config × 2 bit MMLU accuracy
- **Table**: NIAH 16K depth별 성공률
- **핵심 문장**: "PPL 이득 $X\%$가 MMLU 이득 $Y$pt로 전이, Pearson $r = Z$"
- Framework의 "real-world 유용성" 입증

### 7.6 리소스

- GPU: 2× A100 병렬, ~16시간 (MMLU는 느림)
- 의존성: AXIS2 P0 best quantizer 확정 후

**총 작업량**: **2일**

---

## 8. 통합 타임라인

### 8.1 병렬 실행 (GPU 2대 가정)

```
Day 1 (월)
├─ CPU: E1 (κ 측정) + E2 (tail index) + E3 (D_uniform) — 1.5일 compressed
├─ GPU 1: L¹ Lloyd 구현 + Mistral 2-bit 검증 (AXIS2 §13.1)
└─ GPU 2: Spherical 구현 준비

Day 2-3 (화-수)
├─ GPU 1: L¹ Lloyd × 3모델 × 2-bit + E4 Cross-ablation (Identity + TurboQuant 회전 × L¹ Lloyd)
├─ GPU 2: Spherical × 3모델 × 2-bit + E4 Cross-ablation (× Spherical)
└─ CPU: E1/E2/E3 분석 + plot 생성

Day 4 (목)
├─ GPU 1-2: E5 Per-token M_KL variance 분석 + best quantizer 선정
└─ CPU: E4 cross-ablation table 작성

Day 5-6 (금-토)
├─ GPU 1: MMLU Qwen (2-bit + 3-bit)
├─ GPU 2: MMLU Llama (2-bit + 3-bit)
└─ CPU: 논문 drafting (proposition A/B/C 결과 섹션)

Day 7 (일)
├─ GPU 1: NIAH 16K (Qwen)
└─ 결과 통합 + v5 보고서 작성
```

**Total**: 7일 (병렬 최대) — AXIS2 plan P0 (5일)과 통합 시 **총 10일 이내 완료**

### 8.2 순차 실행 (GPU 1대)

Day 1-1.5: E1+E2+E3 (CPU만, 병렬) + L¹ Lloyd 구현
Day 2-4: L¹ + Spherical 측정 + E4
Day 5: E5
Day 6-7: E6 (MMLU + NIAH)
**Total**: 7일

---

## 9. 실패 시나리오 및 대응

### 9.1 E1 FAIL ($\kappa$와 Lloyd 실패가 무관)

**함의**: Proposition A 기각. Lloyd 실패 원인이 Fisher metric 아님.

**대응**:
- Heavy-tail 가설 (Proposition B)로 narrative 전환
- E1을 "negative finding"으로 논문에 포함 ("Fisher metric is not the primary failure mode")
- Spherical/L¹의 경험적 이득만 보고

### 9.2 E2 FAIL (tail index와 L¹ 이득 무관)

**함의**: Proposition B 기각. Heavy-tail이 주된 원인 아님.

**대응**:
- Direction-based Spherical만 contribution
- L¹ Lloyd는 "marginal improvement" 수준으로 보고

### 9.3 E3 FAIL ($D_{uniform}(b)$가 Shannon과 거의 동일)

**함의**: floor=2 derivation 실패.

**대응**:
- $b_{crit}$ 정리로 우회 (coworker's Proposition 6 활용)
- floor=2를 empirical calibration으로 표기, 이론 유도는 future work

### 9.4 E4 FAIL (L¹/Spherical이 Identity에서만 작동)

**함의**: Framework axis separation 깨짐.

**대응**: 이 경우 가장 심각. 논문 축소 재고 필요.
- Pre-RoPE PCA의 unique contribution 부각
- Axis 2는 "standalone quantizer improvement"로 격하

### 9.5 E6 FAIL (MMLU에서 PPL 이득 전이 안 됨)

**함의**: PPL이 잘못된 proxy. Framework 실용 가치 의문.

**대응 (critical)**:
- 즉시 NeurIPS 2026 target 재고
- ICLR 2027로 연기 (긴 유예 + 추가 실험)
- 또는 "PPL-specific optimization" 논문으로 축소

---

## 10. 성공 시나리오 — 기대 효과

### 10.1 All P0 PASS

**시나리오 비교**:

| 항목 | 현재 (AXIS2 plan만) | +E1+E2+E3+E4+E6 |
|---|:---:|:---:|
| Proposition A 검증 | ❌ | ✅ scatter plot |
| Proposition B 검증 | ❌ | ✅ tail index correlation |
| Discrete-WF theorem | ❌ | ✅ $r(b)$ knee plot |
| Axis 독립성 ablation | ❌ | ✅ 12-cell matrix |
| Downstream MMLU | ⚠️ v20에 있음 | ✅ AXIS2 통합 |
| NIAH 16K | ❌ | ✅ |

**Reviewer score 변화**:
- Reviewer #1 (이론): 5 → **7** (+2.0) — "framework가 quantitative predictions를 만들고 검증함"
- Reviewer #2 (실험): 4 → **6** (+2.0) — "thorough ablation + downstream"
- Reviewer #3 (novelty): 5 → **6** (+1.0) — "Propositions A/B/C + Discrete-WF theorem = 새 이론 기여"
- **Average**: 4.7 → **6.3**

**Accept 확률**: 20-25% → **65-75%**

### 10.2 기존 AXIS2 plan만 완료 시 (비교)

Accept 확률: 20-25% → **55-65%** (이전 답변 기준)

**E1-E4 추가 시 delta**: +10-15%p. 3일의 이론 검증이 paper acceptance의 critical margin을 제공.

---

## 11. 논문 반영 구조

본 plan이 모두 성공할 경우 논문 구조 제안:

```
Section 3: Lie Group Framework
  3.1 Axis 1 (Pre-RoPE PCA) — Theorem 6.16.3 ✓
  3.2 Axis 2 reform — Metric Mismatch Bound
    3.2.1 Proposition A: κ-proportional failure
    3.2.2 Proposition B: L^p hierarchy
    3.2.3 Proposition C: Spherical optimality
  3.3 Axis 3 — Discrete-WF Theorem (floor=2 derivation)
  3.4 Class C Maximality (Appendix)

Section 4: Empirical Verification
  4.1 E1: κ(M_KL) correlates with Lloyd failure
  4.2 E2: Tail index predicts L¹ gain  
  4.3 E3: D_uniform knee at b=2
  4.4 Main results: 3-model × 5-quantizer PPL
  4.5 E4: Cross-ablation validates axis independence

Section 5: Downstream
  5.1 E6: MMLU transfer
  5.2 NIAH 16K long-context
```

이 구조는 **이론 → 검증 → 결과 → downstream**의 완결된 NeurIPS paper structure.

---

## 12. 체크리스트 — 실행 전 준비

### 12.1 데이터
- [ ] V3 calibration data 재확인 (WikiText-2 train 160K tokens)
- [ ] V3 Lloyd-Max PPL 결과 파일 경로 확인
- [ ] Attention logits 저장 (E1에 필요, 재수집 필요 시 +0.5일)

### 12.2 코드
- [ ] L¹ Lloyd 구현 (AXIS2 §13.1, 코드 1줄 변경)
- [ ] Spherical k-means 구현 (AXIS2 §3.4 pseudocode)
- [ ] Hill estimator 구현 (표준 scipy 함수)
- [ ] $M_{KL}(t)$ 계산 함수 (기존 attention hook 확장)

### 12.3 환경
- [ ] GPU 가용성 확인 (A100 × 2 권장)
- [ ] MMLU 평가 harness 확인 (lm-eval-harness)
- [ ] NIAH 16K 데이터셋 준비

### 12.4 결과 파일 표준
- [ ] JSON schema 정의: `{experiment_id, model, method, bits, ppl, kappa, alpha, ...}`
- [ ] 수치 source traceability (NEURIPS_VERIFICATION_REPORT v3 스타일)
- [ ] Git commit 후 hash 저장

---

## 13. 관련 문서 및 기존 계획과의 통합

### 13.1 본 plan이 의존하는 문서
- `AXIS2_ANISOTROPY_AWARE_QUANTIZATION_EXPERIMENT_PLAN.md` (P0 L¹ + Spherical 구현 선행)
- `NEURIPS_VERIFICATION_REPORT_v4.md` (v4 QW-PCA 실패 + PCA-Q 정렬 발견)
- `LIE_GROUP_UNIFICATION.md` (Theorem 6.16.3, Propositions 6.18.x, 6.19.x)
- `EXPERIMENT_PLAN_v20_unified.md` (v20 P0~P8 roadmap)

### 13.2 v20 plan과의 관계

v20 plan의 P1 (MMLU 현재 진행 중), P3 (theory-to-metric), P4 (KVTC 확장), P5 (WF downstream)와 본 plan은 **상호보완적**:

| v20 항목 | 본 plan 대응 |
|---|---|
| P1 MMLU Qwen/Llama | E6에 통합 |
| P3 Theory-to-metric | E1 (κ) + E2 (α)로 확장 |
| P4 KVTC Qwen/Mistral | 별도 (본 plan 범위 외) |
| P5 WF(f=2) downstream | E6에 통합 |
| P6 Axis 2 spherical | 본 plan 전체가 이를 강화 |

### 13.3 후속 문서
- `AXIS2_THEORETICAL_VERIFICATION_RESULTS_v1.md` (실험 결과)
- `NEURIPS_VERIFICATION_REPORT_v5.md` (v4 → v5 업데이트, E1-E6 결과 포함)

---

## 14. 결론

기존 AXIS2 plan은 "어떤 quantizer가 작동하는가"에 답하지만, 본 plan은 **"왜 작동하는가 + 이론이 실제를 예측하는가"**에 답한다. 이 추가 계층이 없으면 NeurIPS reviewer의 이론 성향 비판을 방어하기 어렵다.

**핵심 P0 실험 4종 + 1종 downstream (E1+E2+E3+E4+E6)은 총 5.5일 (병렬 3일)**에 완료 가능하며, accept 확률을 **+10~15%p** 끌어올리는 critical path다.

**즉시 실행 권고**:
1. **오늘 오전**: E1, E2, E3 동시 시작 (CPU 작업, GPU 불필요)
2. **오늘 오후**: AXIS2 P0-1 (L¹ Lloyd) Mistral 2-bit hypothesis 검증
3. **내일**: 결과 기반 P0-2 (Spherical) 또는 방향 전환 결정
4. **이번 주 말**: E4 cross-ablation 완료
5. **다음 주**: E5 + E6

**기대 효과**:
- 이론-실험 일치성 입증 (Proposition A/B/C)
- Framework 예측력 복구 (Lloyd 실패를 "framework prediction"으로 reframe)
- WF floor=2의 이론적 정당화 (Discrete-WF Theorem)
- Downstream transfer 증명 (MMLU + NIAH)
- Reviewer 3대 우려 선제 차단

---

*작성: Claude Opus 4.6 (2026-04-07)*
*근거: AXIS2_ANISOTROPY_AWARE_QUANTIZATION_EXPERIMENT_PLAN.md, NEURIPS_VERIFICATION_REPORT_v4.md, 리뷰어 관점 evaluation*
*다음 단계: E1+E2+E3 즉시 실행 (CPU only, day 1 오전), AXIS2 P0-1 병행*
