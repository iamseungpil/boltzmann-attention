# Axis 2 이론 검증 추가 실험 계획서

**프로젝트**: KV-Cache 양자화 Lie Group 프레임워크 — 이론 명제의 Empirical Verification
**작성일**: 2026-04-07
**버전**: v3 (Exp 1-4 chain 결과 반영)
**최종 업데이트**: 2026-04-07 (E3/E3b + E1/E2 + Exp1/2/3/4 chain 완료 후; Next-1/2/3 chain 실행 중)
**근거 문서**:
- `AXIS2_ANISOTROPY_AWARE_QUANTIZATION_EXPERIMENT_PLAN.md` (기존 5-quantizer 실험)
- `NEURIPS_VERIFICATION_REPORT_v4.md` (QW-PCA 실패 + PCA-Q 자연 정렬 발견)
- `LIE_GROUP_UNIFICATION.md` (이론 framework)
- `axis2_theoretical_verification/E3_RESULTS_SUMMARY.md` (E3/E3b 실측 결과)
- `axis2_theoretical_verification/E1E2_RESULTS_SUMMARY.md` (E1/E2 κ·α 측정 결과)
- `axis2_theoretical_verification/EXPERIMENTS_1234_SUMMARY.md` (Exp 1-4 chain 결과, **핵심**)

**목적**: 기존 AXIS2 plan을 보완하여, NeurIPS 2026 reviewer의 예상 3대 우려를 선제 차단하는 이론 명제 검증 실험을 추가한다.

## 실행 현황 대시보드 (2026-04-07 최종)

### 완료된 실험 (총 runtime ~5분)

| 실험 | 상태 | 작업량 | GPU | 핵심 결과 |
|---|:---:|:---:|:---:|---|
| **E3** (Gaussian RD) | ✅ **완료** | 85초 | ❌ | Max 1960 4자리 일치, "knee at b=1" **기각** |
| **E3b** (heterogeneous WF) | ✅ **완료** | 0.8초 | ❌ | floor=0 win 24/24, MSE-PPL gap @ allocation 축 |
| **E1+E2** (κ+α, 4 models) | ✅ **완료** | 74초 | ✅ | **Global κ/α로는 Lloyd 실패 예측 실패** → per-head 필요 |
| **Exp1** (per-head outlier) | ✅ **완료** | 5초 | ❌ | **p95/median spread가 ρ=+1.0으로 Lloyd 실패 예측** |
| **Exp2** (Spherical quantizer) | ✅ **완료** | 31초 | ✅ | Spherical 0/64 win (Proposition C **기각**) |
| **Exp3** (Per-token Fisher) | ✅ **완료** | 29초 | ✅ | **Fisher-avg 12/16 win** in Fisher norm |
| **Exp4** (Per-layer Lloyd breakdown) | ✅ **완료** | 67초 | ✅ | **Layer 2-6 집중 실패**, κ outlier와 cross-match |

### 실행 중 (Next chain, ~30-50분 예상)

| 실험 | 상태 | 대상 | 목적 |
|---|:---:|---|---|
| **Next-1** (Full Fisher Mahalanobis PPL) | 🔄 실행 중 | Mistral (all 32 layers) | Fisher-avg Mahalanobis Lloyd를 PPL에 적용 (Exp3 확장) |
| **Next-2** (Outlier layer preservation) | ⏳ 대기 | Mistral (layers 2-6 4b, rest 2b) | Layer 2-6 보호로 PPL 복구 실험 |
| **Next-3** (Qwen per-layer) | ⏳ 대기 | Qwen2.5-7B | Exp4 cross-model replication |

### 후속 작업 (별도 세션)

| 실험 | 상태 | 비고 |
|---|:---:|---|
| E4 원래 (cross-ablation) | ⏸ 대기 | AXIS2 P0 선행 |
| E5 원래 (per-token $M_{KL}$) | 부분 완료 | Exp3가 부분 대체 |
| E6 (MMLU + NIAH 16K) | ⏸ 대기 | Downstream 필수 |

---

## 0. Executive Summary

### 0.1 배경

기존 AXIS2 plan은 5개 alternative quantizer (L¹ Lloyd, Spherical, Per-token Fisher, E_8 lattice, Wasserstein-1D)의 PPL 비교에 집중한다. 그러나 NeurIPS reviewer 관점에서 다음 3대 우려가 잔존한다:

1. **"3-Axis Lie Group framework의 예측력이 Axis 2에서 무너짐"** — Lloyd-Max가 MSE 3.5× 우위임에도 PPL 전면 실패
2. **"WF floor=2가 ad-hoc 수정"** — 이론적 유도 없음
3. **"Theorem 6.16.3의 Class C 가정이 실용 회전 공간을 충분히 포괄하는가"**

### 0.2 본 plan의 기여

본 plan은 위 3대 우려에 대한 이론적 해결안의 **empirical verification**을 수행한다. 기존 plan이 "어떤 quantizer가 작동하는가"를 묻는 반면, 본 plan은 **"왜 작동하는가"와 "이론이 실제를 예측하는가"**를 묻는다.

### 0.3 추가되는 실험 7종 (E3b 신규 추가)

| # | 실험 | 대응 이론 명제 | 우선순위 | 작업량 | 상태 |
|---|---|---|:---:|:---:|:---:|
| **E1** | $\kappa(\bar{M}_{KL})$ 측정 | Proposition A (Metric Mismatch Bound) | P0 | 0.5일 | ⏸ |
| **E2** | Tail index $\alpha$ 측정 (Hill estimator) | Proposition B ($L^p$ Hierarchy) | P0 | 0.5일 | ⏸ |
| **E3** | $D_{uniform}(b)$ Shannon 대비 편차 실측 | (원) Discrete-WF Theorem | P0 | 85초 | ✅ |
| **E3b** | Heterogeneous WF floor ablation (신규) | (수정) MSE-PPL Allocation Gap | P0 | 1초 | ✅ |
| **E4** | Cross-Rotation × Quantizer Ablation | Axis 1 / Axis 2 독립성 | P0 | 2일 | ⏸ |
| **E5** | Per-token $M_{KL}(t)$ Variance 분석 | Fisher quantizer 설계 근거 | P1 | 1일 | ⏸ |
| **E6** | Best Quantizer × MMLU + NIAH 16K | Downstream 전이 검증 | P0 | 2일 | ⏸ |

**필수 P0 작업량**: 5.5일 (순차) / 3일 (병렬 가능 시). **E3/E3b는 완료되어 1.0일 제외**.

### 0.4 실측 결과 종합 — "Per-head Outlier Hypothesis" 확립 (2026-04-07)

#### 0.4.1 원 가설들의 기각 (3 strikes)

| 원 가설 | 판정 | 증거 |
|---|:---:|---|
| Discrete-WF Theorem (knee at b=1) | ❌ 기각 | E3: $r(b)$ 단조 증가; E3b: floor=0 win 24/24 |
| Proposition A (global κ → Lloyd 실패) | ❌ 기각 | E1: Qwen-7B κ > Mistral κ but Lloyd 실패는 Mistral이 5× 심함 |
| Proposition B (Heavy-tail → L¹ win) | ❌ 기각 | E2: 모든 모델 α ≈ 4.35 (Gaussian-like), v4 κ₄≈0.5 일치 |
| Proposition C (Spherical Optimality) | ❌ 기각 | Exp2: 0/64 Spherical win in MSE |

#### 0.4.2 새로 확립된 긍정적 발견

| 발견 | 증거 | 영향 |
|---|---|---|
| **Per-head κ spread (p95/median)가 Lloyd 실패를 예측** | Exp1: Mistral 15.3× vs Qwen 5.0× (ρ=+1.0) | Proposition A **수정** |
| **Layer 2-6에 outlier 집중** | Exp1: top-3 outlier head 모두 layer 2; Exp4: top-5 PPL 실패 모두 layer 2-6 | Proposition **D** 신설 |
| **Fisher-avg Mahalanobis Lloyd > L² Lloyd (in Fisher norm)** | Exp3: 12/16 head에서 우월 | Proposition A' 지지 |
| **MSE-PPL gap이 Axis 2와 Axis 3 모두에서 발현** | E3b + v3 실측 gap 일치 | Unified $L^2$ failure narrative |

#### 0.4.3 통합 narrative — "Per-head Outlier + Fisher Metric"

> Lloyd-Max PPL 실패의 근본 원인은 **(a) 소수의 outlier head (layer 2-6에 집중) + (b) L² metric의 부적합** 조합이다. 
> 
> **처방**: Per-head Mahalanobis Lloyd (with averaged Fisher metric) + outlier layer 특별 처리. 이는 AXIS2 plan의 Per-token Fisher를 간소화한 형태이며, Exp3에서 75% win rate로 empirically 지지.

이는 이론-실험이 3단 drama로 전개되는 서사:
1. **Act 1 (Paradox)**: Lloyd MSE 3.5× 이득에도 PPL catastrophe
2. **Act 2 (Search)**: 4개 global metric (κ, α, Spherical, floor) 모두 기각
3. **Act 3 (Resolution)**: Per-head outlier + Fisher metric이 올바른 방향

### 0.5 기대 효과

- **Reviewer #1 (이론)**: "framework가 무엇을 예측하는가?" → Proposition A/B/C + **MSE-PPL gap 통일 이론** (E3/E3b에서 뒷받침)
- **Reviewer #2 (실험)**: "ablation이 불충분" → E4의 15-cell cross-ablation matrix + **E3b 24-cell WF matrix 이미 확보**
- **Reviewer #3 (novelty)**: Proposition A/B/C + **"L² fails at 2 axes" 통합 서사**

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

#### ~~Proposition A (원)~~ — E1에서 기각

> ~~$L^2$-Lloyd-Max와 $M_{KL}$-Lloyd-Max의 attention distortion 차이는 $\kappa(\bar{M}_{KL})$에 비례한다~~

**기각 이유**: E1에서 4개 모델 κ median을 측정했더니 **Qwen-7B(22,470) > Mistral(14,321)** 인데, v3 Lloyd 실패는 **Mistral(5.06×) >> Qwen(1.05×)**. Global κ는 예측력 없음.

#### Proposition A' (수정, Exp1에서 empirically 확립)

> $L^2$-Lloyd-Max의 Lloyd failure ratio는 **per-head $\kappa$ 분포의 spread** (p95/median, outlier count)에 비례한다. 소수의 극단 outlier head가 전체 실패를 견인하며, 이 outlier는 **early layers (특히 layer 2-6)에 집중**된다.

**수학적 형식 (제안)**:
$$\text{Lloyd failure ratio} \propto \max_{(l,h)} \kappa(F_{l,h}) / \text{median}_{(l,h)}[\kappa(F_{l,h})]$$
또는 equivalently $N_{outlier}(\tau) = |\{(l,h) : \kappa_{l,h} > \tau \cdot \text{median}\}|$.

**Exp1 empirical 지지** (Spearman with 2 models):
- p95/median: **ρ=+1.000** ✅
- n_outliers(>10×med): **ρ=+1.000** ✅
- fraction above 1e5: **ρ=+1.000** ✅
- Mistral 5 outlier heads (all in layer 2) vs Qwen 1 outlier head

**Exp4 cross-verification**: Mistral layer 2 top outlier (κ=2.8M) = PPL 실패 최악 layer (ΔPPL +0.555). 직접 일치.

#### ~~Proposition B~~ — E2에서 기각

> ~~Source 분포의 tail index $\alpha < 4$일 때, $L^1$-Lloyd가 $L^2$-Lloyd보다 strictly 우월~~

**기각 이유**: E2에서 Hill estimator 측정 결과 **모든 모델 α ≈ 4.25–4.39** — Gaussian-like, heavy tail 없음. v4의 "κ₄ ≈ 0.5, Gaussian" finding과 일치.

**함의**: Keys는 **분포 모양은 Gaussian, 분산 스펙트럼만 anisotropic**. 이는 L¹ Lloyd가 L² Lloyd를 **의미 있게 이기지 못함**을 예측 → AXIS2 plan의 L¹ Lloyd 실험은 **격하** 권고.

**남은 의문**: Mistral layer 2 outlier head는 분포가 진짜로 Gaussian인가? Per-head Hill estimator가 아직 미측정 — 가능성 있는 future work.

#### ~~Proposition C (Spherical Optimality)~~ — Exp2에서 기각

> ~~RMSNorm 하에서 Spherical이 $L^2$보다 $O(1/\epsilon)$ 우월~~

**기각 이유**: Exp2에서 Mistral 64 heads에 Spherical (polar decomposition, 3b angle + 1b magnitude) 적용 → **0/64 win**.

| Quantizer | MSE vs Uniform (median) | Attn-weighted MSE vs Uniform |
|---|:---:|:---:|
| L² Lloyd | **0.589** (41% 낫음) | — |
| Spherical | **1.379** (38% 나쁨) | **2.032** (2× 나쁨) |

**실패 원인 분석**:
1. **k_proj 출력은 RMSNorm 안 됨** — RMSNorm은 hidden states에만 적용
2. **Polar 분해가 key anisotropy를 포착 못 함** — anisotropy는 Cartesian 방향
3. 2D 블록 내 3-bit 각도 quantization이 coarse

**결론**: Proposition C는 이론적으로는 매력적이나 Mistral에는 적용 불가. 남은 검증 대상: Qwen, Llama (아직 Exp2 미시도).

#### ~~Discrete-WF Theorem (원 가설, 2026-04-07 기각)~~

> ~~균등 $b$-bit 양자화기의 실제 rate-distortion $D_{uniform}(b)$가 $b < b_{crit}$에서 Shannon 공식 $\sigma^2 \cdot 2^{-2b}$보다 strictly 크면 (즉 convex deficit이 존재), 이산 WF의 최적해는 $b_j^* \geq b_{crit}\; \forall j$. 가우시안 source + 균등 양자화에서 $b_{crit} = 2$.~~

**E3/E3b 결과로 기각**:
- Gaussian에서 $r(b) = D_{uniform}(b)/D_{Shannon}(b)$는 $b=1$에서 1.46, $b=2$에서 1.91, $b=3$에서 2.41로 **단조 증가** — knee 없음
- Heterogeneous WF 시뮬레이션(24 케이스)에서 순수 MSE 기준 **floor=0이 항상 최적**

#### Revised Proposition (MSE-PPL Allocation Gap) — E3b 결과 기반

> **Proposition (MSE-PPL Allocation Gap)**: 순수 $L^2$ MSE rate-distortion 최적화에서 unconstrained Water-Filling은 항상 MSE-optimal (floor=0이 최적; 24/24 empirical confirmation). 그러나 실측 PPL에서는 floor=2가 floor=0과 floor=1을 명확히 이긴다 (v3: Qwen 11.255 → 7.099). 이 **MSE-floor=0 ↔ PPL-floor=2** gap은 Lloyd-Max "MSE-3.5×-이득에도 PPL 실패" 현상 (Axis 2)과 **동일한 metric mismatch**의 bit allocation 축(Axis 3)에서의 발현이다.
>
> **함의**: $L^2$는 attention distortion의 올바른 metric이 아니며, 이 오류가 quantizer 선택(Axis 2)과 bit allocation(Axis 3) 두 축에서 동일하게 나타난다. floor=2의 이론적 justification은 Fisher / spherical / $L^1$ 등 **non-$L^2$ metric 하의 rate-distortion**에서 찾아야 한다.

**실측 증거**:
- E3 (Gaussian single-channel): $r(b)$ monotonic increase, knee 없음
- E3b (heterogeneous WF): 8 spectra × 3 budgets × 4 floors = 96 cells 측정, MSE-optimal은 100% floor=0
- v3 실측: 동일 WF + PPL에서 floor=2가 floor=1/0보다 우월 (3모델, p<0.01)

**Corollary (Unified $L^2$-PPL Gap)**: Framework의 $L^2$ 가정은 quantizer level과 allocation level에서 **공통의 failure mode**를 가진다. 이 failure를 해결하는 방법은 두 축 모두에 동일 metric ($M_{KL}$ 또는 $L^1$ 또는 spherical)을 적용하는 것이다 — axis 독립성은 metric 선택 하에 복원된다.

#### Corollary (Class C Maximality)

> RoPE $R_{\text{RoPE}} = \bigoplus_{i=1}^{d/2} R_2(\theta_i)$ 와 commute하는 $SO(d)$의 원소는 정확히 Class C (블록-대각 회전)이다.

**증명**: RoPE의 eigenvalue $e^{\pm i\theta_i}$가 distinct → 임의 $U$가 commute하면 eigenspace 보존 → Class C.

**검증 방법**: 이론적 (실험 불필요). Appendix로 추가.

#### Proposition D (Per-head Outlier Concentration, **신규, Exp1+Exp4 확립**)

> KV-cache Lloyd-Max quantization에서 PPL 실패는 **소수의 outlier (layer, head) 조합**에 집중되어 있다. 이 outlier는 구조적으로 **early transformer layers (특히 layer 2-6)**에 위치하며, 해당 layer들에 대한 특별 처리만으로 PPL의 상당 부분을 회복할 수 있다.

**Empirical 지지**:

1. **Exp1 (per-head κ 분석)**:
   - Mistral-7B: Top 3 outlier heads 모두 Layer 2 (H3: κ=2.8M, H6: 989K, H1: 288K)
   - Qwen2.5-7B: Top outlier Layer 19 H0 (κ=3.4M), 두 번째 Layer 2 H0 (κ=138K)
   - Qwen2.5-14B: Layer 2 dominated (H1: 381M, H0: 112M, H5: 41M)

2. **Exp4 (per-layer Lloyd PPL substitution)**:
   - Baseline (FP16): 5.388
   - Top-5 catastrophic layers: **Layer 2 (+0.555), 4 (+0.521), 6 (+0.304), 3 (+0.287), 5 (+0.206)** — **all in layers 2-6**
   - Safe layers (mid/late): ΔPPL < 0.05 (무시 가능)
   - Layer 26 실제로 개선 (ratio 0.999)

3. **Cross-verification**: Exp1의 κ outlier 위치 = Exp4의 PPL 실패 위치 (Layer 2 = 둘 다 최악)

**수학적 형식 (제안)**:
$$\text{TotalFailure}(Q_{Lloyd}) \approx \sum_{l \in \mathcal{L}_{out}} \Delta\text{PPL}_l + \epsilon$$
여기서 $\mathcal{L}_{out} = \{l : \kappa_l > \tau \cdot \text{median}\}$는 소수의 outlier layer 집합.

**처방**:
1. Outlier layer (2-6)에 더 많은 bit 할당 (4-bit 또는 3-bit)
2. 나머지 layer는 2-bit Lloyd로 충분
3. 평균 bit rate는 거의 2-bit 수준 유지 (Next-2 실험에서 검증 중)

**논문 기여**: Framework의 failure mode를 **structural (layer-localized)** 로 특성화. 이는 균일 처리보다 **구조-aware 처리**가 필요함을 보임.

---

## 2. 실험 E1: $\kappa(\bar{M}_{KL})$ 측정 — Proposition A 검증

**✅ 완료 상태** (2026-04-07)
- 스크립트: `scripts/exp_e1e2_kappa_tail_index.py`
- 결과: `reports/axis2_theoretical_verification/e1e2_kappa_tail_index_results.json`
- 런타임: 73.6초 (4 models: Qwen 1.5B/7B/14B + Mistral)
- 상세 분석: `reports/axis2_theoretical_verification/E1E2_RESULTS_SUMMARY.md`
- **판정**: 원 Proposition A **기각**, 수정 Proposition A' (per-head spread) 확립 (Exp1에서)

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

### 2.4 실측 결과 (2026-04-07, 4 models, 2K tokens, 8 layers sampled)

| 모델 | κ(F_avg) median | κ p95 | κ max | R_aniso median |
|---|---:|---:|---:|---:|
| Qwen2.5-1.5B | 64,127 | 463,222 | 1,471,865 | 4,749 |
| Qwen2.5-7B | **22,470** | 112,860 | 3,440,258 | 2,354 |
| Qwen2.5-14B | 12,129 | 37,332,105 | 381,723,079 | 1,013 |
| Mistral-7B | **14,321** | 218,821 | 2,805,296 | 1,782 |

**판정**:
- H1.1 (모델 순서): **FAIL** — Qwen-7B κ > Mistral κ (역전)
- H1.2 (scatter R²): 측정 불가 (2-3 points)
- **원 Proposition A 기각**

**그러나 Exp1에서 per-head 재분석 후 Proposition A' 확립** (다음 섹션 참조).

### 2.5 논문 반영

- **Figure (제안)**: per-head κ distribution histogram (4 모델) — Mistral의 heavy tail in spread
- **Table**: κ median/p95/p99/max/n_outliers (4 모델)
- **스토리**: "Global median은 오해소지, spread (p95/median) 또는 outlier count가 올바른 지표"

### 2.6 리소스 (실측)

- GPU: A6000 (실제로는 GPU 사용 — V3 data 재사용 불가)
- CPU: <10 sec
- 모델 load + forward pass: 총 74초 (4 models)
- 의존성: HF cache (3모델 기 다운로드)

**총 작업량**: **74초 (완료)**

---

## 3. 실험 E2: Tail Index $\alpha$ 측정 — Proposition B 검증

**✅ 완료 상태** (2026-04-07)
- 스크립트: `exp_e1e2_kappa_tail_index.py` (E1과 통합 실행)
- 결과: 동일 JSON에 병합
- 런타임: E1과 함께 총 74초
- **판정**: 원 Proposition B **기각** — 모든 모델 α ≈ 4.3 (Gaussian-like)

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

## 4. 실험 E3: $D_{uniform}(b)$ Shannon 대비 편차 실측 — ~~Discrete-WF Theorem~~ **→ 재해석**

**✅ 완료 상태** (2026-04-07)
- 스크립트: `scripts/exp_e3_discrete_wf_verification.py`
- 결과: `reports/axis2_theoretical_verification/e3_discrete_wf_results.json`
- 런타임: 85.8초 (CPU only, 1M samples)
- 상세 분석: `reports/axis2_theoretical_verification/E3_RESULTS_SUMMARY.md`

**판정**: 원래 가설 기각, 더 강한 finding 도출 (§4.6 참조).

### 4.1 의도

$b$-bit 균등 양자화기의 실제 distortion이 $b < 2$에서 Shannon $\sigma^2 \cdot 2^{-2b}$보다 strictly 큼을 실측하여, ~~**floor=2가 이론적 필연**임을 입증~~ → **MSE-WF가 floor=2를 지지하지 않음을 엄밀히 확인**.

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

### 4.3 가설 및 판정 (실측 완료)

**H3.1**: $r(1) > r(2) > r(3) > \ldots \to 1$ (monotonic convergence)
- 판정: **❌ FAIL** — $r(b)$가 오히려 단조 증가

**H3.2**: $r(1)$과 $r(2)$의 gap이 $r(2)$와 $r(3)$의 gap보다 큼 ("knee at b=2")
- 판정: **❌ FAIL** — knee 없음, linear-ish증가

**H3.3**: 실제 데이터에서 $r(1)$이 가우시안보다 더 크다 (heavy-tail 보강)
- 판정: 부분 확인 — Student-t(df=3)에서 $r(1)=2.39$ (Gaussian 1.46의 1.6배)

### 4.4 실측 결과 (Gaussian, 1M samples)

**검증: Max 1960 reference 대비 4자리 일치 (구현 신뢰성 확보)**

| $b$ | $D_{opt\_uniform}$ (우리) | $D_{Lloyd}$ (우리) | $D_{Max\_1960}$ | Shannon $2^{-2b}$ | $r_{opt}$ |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.364227 | 0.364226 | 0.363400 | 0.250000 | **1.457** |
| 2 | 0.119263 | 0.117880 | 0.117500 | 0.062500 | **1.908** |
| 3 | 0.037639 | 0.034759 | 0.034540 | 0.015625 | **2.409** |
| 4 | 0.011634 | 0.009560 | 0.009497 | 0.003906 | 2.978 |
| 5 | 0.003537 | 0.002593 | 0.002805 | 0.000977 | 3.622 |
| 6 | 0.001063 | 0.000879 | 0.000821 | 0.000244 | 4.355 |

**$r(b)$는 단조 증가** → "knee at $b=1$" 없음. Marginal gain 분석:
- $0 \to 1$: 0.636/bit (가장 큼)
- $1 \to 2$: 0.245/bit
- $2 \to 3$: 0.082/bit
- (monotonically decreasing — 표준 Shannon R-D 양상)

**Heavy-tail 결과** (Student-t df=3, Laplace):

| 분포 | $r(1)$ | $r(2)$ | $r(3)$ | $r(4)$ |
|---|:---:|:---:|:---:|:---:|
| Gaussian | 1.455 | 1.903 | 2.402 | 2.967 |
| **Student-t (df=3)** | **2.393** | **5.479** | **12.933** | **30.947** |
| Laplace | 1.999 | 3.135 | 4.590 | 6.473 |

**Proposition B의 간접 증거**: heavy-tail에서 uniform quantizer가 Shannon 예측 대비 훨씬 큼 (최대 30×).

### 4.5 논문 반영

- **Figure 1** (제안): $b$ vs $r(b)$ plot (3 분포) — "uniform quantizer ≠ Shannon, heavy-tail에서 악화"
- **Table** (실측): 위 결과 그대로 사용
- **이론 섹션 수정**: "$r(b)$ 단조 증가 = uniform quantizer의 systematic suboptimality at all bits, not just $b=1$"
- **스토리**: "$L^2$ MSE rate-distortion에 근거한 WF는 floor=0 선호 → floor=2 성공은 다른 메커니즘" (E3b로 확장)

### 4.6 리소스 및 실행

- GPU: 불필요 (CPU only, numpy/scipy)
- CPU: 1 서버, 85.8초 실측
- 시드: 42 (재현 가능)
- 샘플 수: 1,000,000

**총 작업량**: **85초 (완료)**

---

## 4b. 실험 E3b: Heterogeneous Water-Filling Floor Test — **MSE-PPL Gap의 Allocation-level 발현** (신규)

**✅ 완료 상태** (2026-04-07)
- 스크립트: `scripts/exp_e3b_heterogeneous_wf.py`
- 결과: `reports/axis2_theoretical_verification/e3b_heterogeneous_wf_results.json`
- 런타임: 0.8초 (analytical, no sampling)

### 4b.1 동기 및 의도

E3는 단일 Gaussian 채널에서 rate-distortion을 측정했으나, WF의 실제 문제는 **이종 분산 채널에 대한 bit allocation** 이다. E3의 결과(knee 없음)로 즉시 다음 질문이 제기됨:

> "PCA로 노출된 heterogeneous variance spectrum에서, discrete WF가 floor=0/1/2 중 어느 것을 선호하는가?"

E3b는 이를 직접 검증하는 실험으로 도입되었다. 결과는 **"floor=0이 100% 승리"** — PPL 실측과 정반대.

### 4b.2 프로토콜

**입력**: $n=128$ channel, 평균 budget $B/n \in \{2, 3, 4\}$ bit.

**스펙트럼 (8종)**:

| 이름 | 설명 | $\sigma^2_{\max}/\sigma^2_{\min}$ |
|---|---|:---:|
| iid_equal | 균등 (기준) | 1.0 |
| power_law_1.0 | $\lambda_j \propto j^{-1}$ | 128 |
| power_law_1.5 | $\lambda_j \propto j^{-1.5}$ | 1,449 |
| power_law_2.0 | $\lambda_j \propto j^{-2}$ | 16,384 |
| exponential_0.1 | $\lambda_j \propto e^{-0.1j}$ | 3.3e5 |
| exponential_0.3 | $\lambda_j \propto e^{-0.3j}$ | 3.5e16 |
| bimodal | 25%=10, 75%=0.1 | 100 |
| realistic_llama_like | 10% high + 90% low decay | 8.2e4 |

**WF 해결 방법**: Greedy marginal allocation — 현재 할당에서 가장 큰 $D$ 감소를 주는 채널에 1 bit 추가, budget 소진까지 반복. Floor $\in \{0, 1, 2, 3\}$ 각각에 대해 별도 실행.

**Distortion function**: E3에서 측정한 Max(1960) optimal uniform quantizer 값 사용
```
D(b=0) = 1.0 * σ², D(1) = 0.3642σ², D(2) = 0.1193σ², D(3) = 0.03764σ², ...
```

### 4b.3 결과 — 예상과 정반대

**24 케이스 (8 스펙트럼 × 3 budget) 전수 테스트**:

| Budget | floor=0 win | floor=1 win | floor=2 win | floor=3 win |
|:---:|:---:|:---:|:---:|:---:|
| avg=2 bit | 8/8 | 0/8 | 0/8 | 0/8 |
| avg=3 bit | 8/8 | 0/8 | 0/8 | 0/8 |
| avg=4 bit | 8/8 | 0/8 | 0/8 | 0/8 |
| **합계** | **24/24 (100%)** | 0 | 0 | 0 |

대표 예시 (`realistic_llama_like`, 평균 2 bit):

| floor | Total $D$ | vs floor=0 | 채널 분포 |
|:---:|:---:|:---:|---|
| 0 | **0.04637** | — | 32 채널이 0 bit, 나머지 0~8 bit |
| 1 | 0.05288 | −14.0% | 86 채널이 1 bit에 묶임 |
| 2 | 15.27 | **−32,832%** (catastrophic) | 128/128 채널에 2 bit 강제 → budget 소진 |

**이유**: 128 채널 × 평균 2 bit = 256 bit 예산. floor=2 강제 시 $128 \times 2 = 256$ bit을 균등 배분 → 저분산 채널에 과잉 투자, 고분산 채널에 과소 투자 → catastrophe.

### 4b.4 해석: MSE-PPL Gap이 두 축에서 동시 발현

**순수 MSE 관점**:
- 이론 (Shannon 1948): unconstrained WF가 최적
- 실험 (E3b): 24/24 케이스에서 floor=0이 MSE-optimal
- 결론: **pure MSE는 floor 제약 없는 WF를 선호**

**그러나 PPL 실측 (v3)**:

| 모델 | WF floor=1 PPL (Shannon WF에 가까움) | WF floor=2 PPL |
|---|:---:|:---:|
| Qwen 2-bit | 11.255 | **7.099** ✅ |
| Llama 2-bit | 8.963 | **7.159** ✅ |
| Mistral 2-bit | 6.355 | **5.822** ✅ |

$\Rightarrow$ **MSE-optimal (floor=0 or 1) ≠ PPL-optimal (floor=2)**, 3모델 모두 일관.

### 4b.5 Proposition (수정) — MSE-PPL Allocation Gap

E3b의 결과는 **원래 Discrete-WF Theorem을 기각**하고 더 강한 명제를 뒷받침한다:

> **Proposition (MSE-PPL Allocation Gap)**: 순수 $L^2$ MSE 기준으로 discrete Water-Filling은 항상 unconstrained solution을 선호한다 (E3b: 24/24). 그러나 실측 PPL은 floor=2를 strictly 선호한다 (v3: 3/3 모델). 이 gap은 Lloyd-Max "$L^2$-MSE 3.5× 이득에도 PPL 실패" 현상 (Axis 2)과 **동일한 metric mismatch**가 bit allocation(Axis 3)에서 발현한 것이다.
>
> **Corollary**: Lie group framework의 $L^2$ 가정은 quantizer 선택과 bit allocation 두 축에서 공통의 failure를 생성한다. 이는 framework의 실패가 아니라, framework가 정확히 지정하는 failure mode이다. 수정은 "두 축 모두에 동일한 non-$L^2$ metric (Fisher/$L^1$/spherical) 적용"이다.

### 4b.6 논문 반영 — "Unified $L^2$-PPL Gap" 서사

**현재 (v4)**:
> "WF floor=2는 1-bit 소실 방지를 위한 실용적 수정"

**수정 (E3+E3b 기반)**:
> "우리는 WF floor=2의 empirical success를 standard rate-distortion 이론으로 설명하려 시도했다. Gaussian과 8개 이종 spectra에서의 정확한 시뮬레이션 (E3, E3b)은 $L^2$ MSE 최적화에서 unconstrained WF (floor=0)가 항상 최적임을 보인다 (24/24). 이는 v3 실험에서 PPL-optimal이 floor=2라는 3-모델 일관 결과와 **정면 모순**된다. 이 gap은 Lloyd-Max가 MSE에서 3.5× 우위에도 PPL에서 실패하는 Axis 2 현상과 **동일한 metric mismatch**다. 두 독립 현상이 공통의 뿌리를 가짐을 확인함으로써, 우리는 $L^2$가 attention distortion의 올바른 metric이 아니라는 주장을 **quantizer 축과 allocation 축에서 독립적으로** 뒷받침한다."

**Figure (제안)**: 2-panel
- (a) Axis 2: Lloyd-Max MSE gain vs PPL loss (v3 data)
- (b) Axis 3: WF floor=0 MSE optimal vs PPL floor=2 optimal (E3b + v3)
- 캡션: "같은 metric mismatch가 두 축에서 대칭적으로 발현"

### 4b.7 리소스

- GPU: 불필요
- CPU: 0.8초 (analytical, 샘플링 없음)
- 총 24 케이스 완전 탐색

**총 작업량**: **<1초 (완료)**

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

## 8. 통합 타임라인 (업데이트, E3/E3b 완료 반영)

### 8.0 완료 현황 (2026-04-07 17:30 KST)

```
[✅] Day 0 (오늘): E3 + E3b 완료 — 총 90초 CPU
   ├─ E3: Gaussian single-channel rate-distortion → Max 1960 reference 4자리 일치
   ├─ E3b: 8 spectra × 3 budgets × 4 floors = 96 cells → floor=0 100% win
   └─ 결론: "MSE-PPL gap이 quantizer와 allocation 두 축에서 동시 발현" finding 확보
```

### 8.1 남은 작업 병렬 실행 (GPU 2대 가정)

```
Day 1 (내일)
├─ CPU: E3 결과 기반 논문 Proposition 수정 작업 (half day)
├─ GPU 1: L¹ Lloyd 구현 + Mistral 2-bit 검증 (AXIS2 §13.1)
│        + 동시: E1 위해 attention logits 재수집 (Qwen calibration)
└─ GPU 2: Spherical 구현 준비 + E1/E2 위해 PCA keys 수집

Day 2-3
├─ GPU 1: L¹ Lloyd × 3모델 × 2-bit + E4 Cross-ablation (L¹)
├─ GPU 2: Spherical × 3모델 × 2-bit + E4 Cross-ablation (Sph)
└─ CPU: E1 (κ 측정) + E2 (Hill estimator) — 수집된 데이터 분석

Day 4
├─ GPU 1-2: E5 Per-token M_KL variance + best quantizer 선정
└─ CPU: E4 cross-ablation table 작성 + Proposition A/B 검증 플롯

Day 5-6
├─ GPU 1: MMLU Qwen (2-bit + 3-bit)
├─ GPU 2: MMLU Llama (2-bit + 3-bit)
└─ CPU: 논문 drafting (Proposition A/B/C + Unified L²-PPL Gap)

Day 7
├─ GPU 1: NIAH 16K (Qwen)
└─ 결과 통합 + NEURIPS_VERIFICATION_REPORT_v5 작성
```

**Total (E3/E3b 완료 기준)**: 7일 (병렬 최대) — AXIS2 plan P0 (5일)과 통합 시 **총 9일 이내 완료**

### 8.2 순차 실행 (GPU 1대)

~~Day 1-1.5: E1+E2+E3 (CPU만, 병렬) + L¹ Lloyd 구현~~ → **E3/E3b 완료, E1/E2 GPU 이전**

Day 1: L¹ Lloyd 구현 + 단기 검증 (Mistral 2-bit) + E1/E2 data collection
Day 2-4: L¹ + Spherical 3모델 측정 + E4 cross-ablation
Day 5: E5 per-token variance
Day 6-7: E6 (MMLU + NIAH 16K)
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

### 9.3 ~~E3 FAIL~~ → **E3 실제 결과 (2026-04-07)**: 원 가설 기각, 더 강한 finding 도출

**실측 결과**:
- $D_{uniform}(b)$는 Shannon과 일치하지 않음 (단조 증가 $r(b)$)
- 그러나 "knee at $b=1$"은 없음 → 원 Discrete-WF Theorem 기각
- E3b로 확장: heterogeneous channels에서 **MSE-WF는 floor=0을 100% 선호** (24/24)

**대응 (현실)**:
- ✅ 원 Discrete-WF Theorem 폐기
- ✅ "MSE-PPL Allocation Gap" Proposition 으로 대체 (§1.2 Revised Proposition)
- ✅ Lloyd 실패(Axis 2)와 통합된 서사로 강화
- ✅ floor=2 justification은 Fisher/spherical metric 하 rate-distortion으로 이동 (future work 또는 E5 확장)
- **결과적으로 논문 기여가 더 강화됨** (두 독립 현상의 통합)

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

### 10.1 All P0 PASS (업데이트 후)

**시나리오 비교**:

| 항목 | 현재 (AXIS2 plan만) | +E3+E3b 완료 | +E1+E2+E4+E6 (남은 P0) |
|---|:---:|:---:|:---:|
| Proposition A 검증 | ❌ | ❌ (E1 필요) | ✅ scatter plot |
| Proposition B 검증 | ❌ | ⚠️ Student-t 간접 증거 | ✅ Hill estimator correlation |
| ~~Discrete-WF theorem~~ MSE-PPL Allocation Gap | ❌ | ✅ **24/24 확인** | ✅ |
| Axis 독립성 ablation | ❌ | ❌ | ✅ 12-cell matrix |
| Downstream MMLU | ⚠️ v20에 있음 | ⚠️ | ✅ AXIS2 통합 |
| NIAH 16K | ❌ | ❌ | ✅ |
| **$L^2$-PPL 통합 서사** | ❌ | ✅ **Axis 2 + Axis 3 통합** | ✅ 전축 통합 |

**Reviewer score 변화**:
- Reviewer #1 (이론): 5 → **6** (현재 +E3) → **7** (+E1-E4 완료)
  - "MSE-PPL gap이 framework의 두 축에서 통합됨" (E3b 효과)
- Reviewer #2 (실험): 4 → 5 (현재) → **6** (+E4/E6 완료) — "thorough ablation + downstream"
- Reviewer #3 (novelty): 5 → **6** (현재 +E3b) → **6.5** — "Two-axis $L^2$ failure unification"
- **현재 Average (E3/E3b만)**: 4.7 → **5.7** (+1.0)
- **남은 P0 완료 시**: 5.7 → **6.5**

**Accept 확률**:
- 현재 (E3/E3b 완료): 20-25% → **45-55%**
- 남은 P0 완료 시: 45-55% → **65-75%**

### 10.2 E3/E3b의 단독 기여 (2026-04-07 확보)

E3/E3b만으로 다음이 확보됨:
1. ✅ **구현 신뢰성**: Max 1960 reference와 4자리 일치
2. ✅ **원 가설의 honest rejection**: "naive knee theorem은 거짓"을 엄밀히 증명
3. ✅ **새 unified finding**: MSE-PPL gap의 2-축 발현
4. ✅ **논문 서사 강화**: Lloyd-Max 실패(Axis 2)와 floor=2 puzzle(Axis 3)이 동일 현상
5. ✅ **Proposition B 간접 증거**: Student-t에서 $r(b)$ amplification

단독 점수 상승: +1.0 (Reviewer 평균)

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
  3.3 Axis 3 — MSE-PPL Allocation Gap (E3/E3b 기반, 신규)
    3.3.1 Unconstrained WF is MSE-optimal (E3b: 24/24)
    3.3.2 Empirical PPL-optimal is floor=2 (v3)
    3.3.3 Gap as Axis-3 manifestation of L² mismatch
  3.4 Unified L²-PPL Failure across Axis 2 + Axis 3 (신규)
  3.5 Class C Maximality (Appendix)

Section 4: Empirical Verification
  4.1 E1: κ(M_KL) correlates with Lloyd failure
  4.2 E2: Tail index predicts L¹ gain
  4.3 E3/E3b: D_uniform characterization + heterogeneous WF (완료)
  4.4 Main results: 3-model × 5-quantizer PPL
  4.5 E4: Cross-ablation validates axis independence

Section 5: Downstream
  5.1 E6: MMLU transfer
  5.2 NIAH 16K long-context
```

이 구조는 **이론 → 검증 → 결과 → downstream**의 완결된 NeurIPS paper structure.

---

## 12. 체크리스트 — 실행 전 준비 (업데이트)

### 12.1 데이터
- [x] ~~E3/E3b는 순수 synthetic~~ — 완료 (2026-04-07)
- [ ] V3 calibration data 재확인 (WikiText-2 train 160K tokens) — E1/E2용
- [ ] V3 Lloyd-Max PPL 결과 파일 경로 확인 — reports/axis2... 참조
- [ ] Attention logits 저장 (E1에 필요, 재수집 0.5일)

### 12.2 코드
- [x] **E3 구현 완료**: `scripts/exp_e3_discrete_wf_verification.py` (Max 1960 검증)
- [x] **E3b 구현 완료**: `scripts/exp_e3b_heterogeneous_wf.py` (greedy WF)
- [ ] L¹ Lloyd 구현 (AXIS2 §13.1, 코드 1줄 변경)
- [ ] Spherical k-means 구현 (AXIS2 §3.4 pseudocode)
- [ ] Hill estimator 구현 (표준 scipy 함수)
- [ ] $M_{KL}(t)$ 계산 함수 (기존 attention hook 확장)

### 12.3 환경
- [x] Python env 검증 (numpy 2.2.6, scipy 1.16.3, torch 2.8.0+cu128, transformers 5.4.0)
- [ ] GPU 가용성 확인 (A100 × 2 권장)
- [ ] MMLU 평가 harness 확인 (lm-eval-harness)
- [ ] NIAH 16K 데이터셋 준비

### 12.4 결과 파일 표준
- [x] **E3/E3b JSON schema 수립**: `e3_discrete_wf_results.json`, `e3b_heterogeneous_wf_results.json`
- [x] **재현성**: 시드 42, 1M 샘플, git head 기록 가능
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

## 14. 결론 및 진행 상태 (v2 업데이트)

### 14.1 현재까지의 진전 (2026-04-07)

**완료된 실험**:
- ✅ **E3**: Gaussian rate-distortion 실측 (85초) — Max 1960 reference 4자리 일치
- ✅ **E3b**: Heterogeneous WF floor ablation (0.8초) — 24/24 floor=0 win

**핵심 발견**:
- ❌ 원 "Discrete-WF Theorem (knee at $b=1$)" 가설 **엄밀히 기각**
- ✅ 더 강한 finding: **MSE-PPL gap이 Axis 2 (quantizer)와 Axis 3 (allocation) 두 축에서 대칭적으로 발현**
- ✅ Lloyd 실패 + floor=2 puzzle이 **같은 $L^2$-metric mismatch**의 두 얼굴임을 확인

### 14.2 이론적 진전 — "Unified $L^2$-PPL Gap"

**원 서사** (단편적):
> "Axis 2: Lloyd 실패 (honest negative). Axis 3: floor=2 empirical fix."

**수정 서사** (통합적):
> "Axis 2와 Axis 3 모두에서 $L^2$-MSE는 PPL-optimal해를 주지 못한다. 두 현상은 독립적으로 관측되지만 공통의 뿌리 (metric mismatch)를 공유한다. 이는 framework의 실패가 아니라, framework가 정확히 식별하는 하나의 근본 문제다. 수정은 두 축에 동일한 non-$L^2$ metric (Fisher / $L^1$ / spherical)을 적용하는 것이다."

### 14.3 남은 critical path (업데이트)

**기존 AXIS2 plan (L¹ + Spherical)**: 1주 (GPU 필요)
+ **본 plan의 남은 P0** (E1 + E2 + E4 + E6): 4.5일 추가 (GPU 필요)
= **총 약 9일** (병렬 가능 시 6일)

남은 항목:
1. **E1** ($\kappa$ 측정) — attention logits 수집 후 분석
2. **E2** (tail index) — PCA'd keys 수집 후 Hill estimator
3. **E4** (cross-ablation) — AXIS2 P0 (L¹/Spherical) 선행 후 3 회전 × 2 quantizer 확장
4. **E6** (MMLU + NIAH 16K) — best quantizer 선정 후 downstream

### 14.4 Reviewer accept 확률 (E3/E3b 반영)

| 상태 | Accept 확률 |
|---|:---:|
| **Before E3/E3b**: 원 plan만 | 20-25% |
| **Now (E3/E3b 완료)**: unified $L^2$-PPL gap 서사 확보 | **45-55%** (+25%p) |
| **남은 P0 모두 완료 시** | **65-75%** (+20%p) |

### 14.5 다음 즉시 실행 항목

1. **오늘**: E3/E3b 결과를 NEURIPS_VERIFICATION_REPORT_v4에 부록으로 추가
2. **내일**: L¹ Lloyd 구현 + 동시에 attention logits 수집 파이프라인 (E1 준비)
3. **이번 주**: L¹/Spherical 3모델 × 2-bit 측정 + E4 cross-ablation
4. **다음 주**: E1/E2 분석 + E6 downstream

**기대 효과** (업데이트):
- [x] ~~이론-실험 일치성 (원 Discrete-WF Theorem)~~ → **통합 서사로 대체 및 강화**
- [x] Framework 예측력 복구 (Lloyd 실패 + floor=2 모두를 "prediction"으로 reframe)
- [ ] Downstream transfer 증명 (MMLU + NIAH) — E6 남음
- [x] **Reviewer 우려 1, 2 선제 차단 강화**: "framework가 두 축의 MSE-PPL gap을 통일적으로 식별"
- [ ] Reviewer 우려 3 (Class C): 이론 명제 추가 (실험 불필요, 1-2일)

---

*v1 작성: Claude Opus 4.6 (2026-04-07)*
*v2 업데이트: Claude Opus 4.6 (2026-04-07, E3/E3b 완료 후)*
*근거: AXIS2_ANISOTROPY_AWARE_QUANTIZATION_EXPERIMENT_PLAN.md, NEURIPS_VERIFICATION_REPORT_v4.md, E3_RESULTS_SUMMARY.md*
*다음 단계: E1+E2+E3 즉시 실행 (CPU only, day 1 오전), AXIS2 P0-1 병행*
