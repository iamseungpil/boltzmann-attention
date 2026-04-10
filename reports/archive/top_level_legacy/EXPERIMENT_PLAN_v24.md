# 실험 계획 v24: 병렬 검증 — D_attn, AAPQ, Anti-PCA, WF (2026-04-09)

## 검증 기준 (선행 연구 기반)

### PPL 기준 (WikiText-2, 49K test)

| 모델 | FP16 | 기존 SOTA (3-bit 근처) | 우리 PCA+Uniform 3-bit |
|------|:----:|:---------------------:|:---------------------:|
| Mistral-7B | 5.58 | TurboQuant 3.5b: ~5.6 (추정) | 5.67 |
| Qwen-7B | 6.56 | KVTC 적응형: ~6.6 (추정) | 6.76 |
| Llama-8B | 6.40 | KIVI 2b: ~7.0 | 6.67 |

**성공 기준**: 새 방법이 PCA+Uniform 대비 PPL 0.1+ 개선하면 유의미.
**SOTA 경쟁 기준**: TurboQuant 3.5b 수준 (FP16 대비 +0.1 이내)에 3-bit으로 도달하면 경쟁력.

### D_attn 기준 (새 지표, 양자화 없이 측정)

D_attn = Σ_j [R^T Σ_Q R]_jj × [R^T Σ_K R]_jj

**성공 기준**: Anti-PCA 또는 최적화 회전의 D_attn이 PCA 대비 10%+ 감소.

## 선행 연구 요약 (우리와 관련된 것)

| 논문 | 학회 | 핵심 기법 | 우리와의 관계 |
|------|:----:|----------|-------------|
| KVTC | ICLR'26 | Shared PCA + DP + entropy coding | 우리: per-head PCA (증명) + attention-weighted |
| TurboQuant | ICLR'26 | Random rot + Lloyd + QJL | 우리: structured rot (PCA/Anti-PCA) |
| SpinQuant | ICLR'25 | Learned rot (Cayley SGD on task loss) | 우리: D_attn proxy (cheaper) |
| KVQuant | NeurIPS'24 | Pre-RoPE + NUQ + Fisher + outlier | 우리: rotation 이론 추가 |
| MixKVQ | arXiv'25 | Channel-level query-aware bit alloc | 우리: PCA-dim level |
| AQUA-KV | ICML'25 | Cross-layer predict + residual quant | 직교적 (결합 가능) |
| Expected Attention | ICLR'26 | Closed-form future attention | 직교적 |
| A2ATS | ACL'25 | Query-aware VQ | 우리: scalar quant (Class C) |

### 우리의 novelty (정직하게)

| 주장 | 새로운가? | 근거 |
|------|:--------:|------|
| D_mse 회전 불변성 (per-dim adaptive에서) | **예** | 아무도 명시적으로 증명 안 함 |
| D_attn 회전 비불변성 | **예** | 아무도 분석 안 함 |
| Rearrangement inequality → Anti-PCA | **예** | 아무도 시도 안 함 |
| PCA가 PPL을 개선하는 진짜 이유 분해 | **예** | decorrelation/WF/Gaussianity 분리 |
| Per-PCA-dim attention-weighted bit allocation | **예** | MixKVQ는 channel-level |
| 624/624 L∞ 측정 | **예** | |

## 병렬 실험 (4 GPUs, E8)

### 실험 1: D_attn 측정 (GPU 0)

```
의도: 5가지 회전에서 D_attn을 측정하여 이론적 예측 검증
가설: D_attn(Anti-PCA) < D_attn(Random) < D_attn(PCA) < D_attn(NoRot)
       (rearrangement inequality 예측)
검증: Mistral-7B calibration data에서 Σ_K, Σ_Q 추출 → D_attn 계산
      양자화 없이, 순수 행렬 연산
      5 layers × 8 heads = 40 data points
metric: D_attn 값 + 회전별 순위
해석:
  가설 맞으면 → Anti-PCA/최적화 회전으로 PPL 개선 가능성
  가설 틀리면 → D_attn이 올바른 proxy가 아니거나 rearrangement 적용 불가
시간: 30분
```

### 실험 2: AAPQ (GPU 1)

```
의도: attention-weighted bit allocation이 uniform보다 PPL을 개선하는가?
가설: PCA + attention-WF(avg=3, floor=2) PPL < PCA + Uniform 3-bit PPL
검증: Mistral-7B, 3-bit avg, WikiText-2 49K test
      importance_j = λ_j × σ²_q,j (attention-weighted)
      vs eigenvalue-only WF: importance_j = λ_j
      vs uniform 3-bit
metric: PPL
해석:
  attention-WF < eigenvalue-WF < uniform → "attention 가중이 효과적"
  attention-WF ≈ eigenvalue-WF < uniform → "WF 자체가 효과, attention 미미"
  모두 비슷 → "3-bit에서 WF 효과 미미"
성공 기준: PPL 0.1+ 개선 (5.67 → 5.57 이하)
시간: 1시간
```

### 실험 3: Anti-PCA PPL (GPU 2)

```
의도: D_attn을 줄이는 Anti-PCA가 실제 PPL도 개선하는가?
가설: Anti-PCA 3-bit PPL < PCA 3-bit PPL (D_attn 감소 → PPL 감소)
검증: Mistral-7B, 3-bit, WikiText-2 49K test
      Anti-PCA = PCA 후 차원을 σ²_q 역순으로 재배열
metric: PPL + D_attn
해석:
  Anti-PCA PPL < PCA PPL → "rearrangement이 PPL에도 유효" → 강한 결과
  Anti-PCA PPL > PCA PPL → "D_attn이 PPL의 올바른 proxy가 아님"
  Anti-PCA PPL ≈ PCA PPL → "차원 순서 무관, 다른 요인이 지배"
성공 기준: PPL 0.1+ 개선
시간: 1시간
```

### 실험 4: PCA + WF avg=3 (GPU 3)

```
의도: WF bit allocation이 3-bit avg에서 효과가 있는가?
가설: PCA + eigenvalue-WF(avg=3, floor=2) PPL < PCA + Uniform 3-bit PPL
검증: Mistral-7B, WikiText-2 49K test
      WF: b_j = 3 + 0.5 × log₂(λ_j / GM(λ)), floor=2
metric: PPL + per-dim bit 분포
해석:
  WF < Uniform → "non-uniform allocation이 효과적" → AAPQ 방향 유효
  WF ≈ Uniform → "3-bit에서 WF 효과 미미" → 더 낮은 bit 필요
성공 기준: PPL 0.05+ 개선
시간: 1시간
```

## 결과 조합별 해석

| D_attn 순위 | AAPQ | Anti-PCA | WF | 논문 방향 |
|------------|:----:|:--------:|:--:|----------|
| Anti<Random<PCA | 효과적 | 효과적 | 효과적 | **최선: D_attn 이론 + 방법 + 결과** |
| Anti<Random<PCA | 효과적 | 무효 | 효과적 | AAPQ 중심, Anti-PCA는 분석만 |
| 순서 다름 | 무효 | 무효 | 효과적 | WF 중심, D_attn 이론 약화 |
| 모두 비슷 | 무효 | 무효 | 무효 | 이론 논문 (D_mse 불변성이 핵심) |
