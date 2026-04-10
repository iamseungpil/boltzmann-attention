# FOKVQ PPL 벤치마크 코드 리뷰: cowork 구현 vs 우리 구현 비교 분석

## 0. 개요

본 문서는 cowork(iamseungpil)이 Azure ML 환경용으로 작성한 `exp4_2_standard_ppl_benchmark.py`와
우리의 검증된 구현 `exp10_1_wikitext2_ppl.py`를 비교 분석하여,
PPL 결과 부진의 원인을 코드 수준에서 규명한다.

- **cowork 코드**: [iamseungpil/boltzmann-attention](https://github.com/iamseungpil/boltzmann-attention) 내 `exp4_2_standard_ppl_benchmark.py`
- **우리 코드**: `docs/architecture/patent/exp10_1_wikitext2_ppl.py` (검증 완료)
- **목적**: PPL 결과 부진의 근본 원인을 코드 비교를 통해 분석

---

## 1. [치명적 버그] PCA 투영 시 센터링 누락

cowork 코드에서 PCA basis 계산 시에는 센터링을 수행하지만,
실제 양자화 투영 단계에서는 센터링을 하지 않는다. 이것이 가장 심각한 버그이다.

### cowork 코드 (`pca_basis_np`)

```python
def pca_basis_np(matrix: np.ndarray) -> np.ndarray:
    centered = matrix - matrix.mean(axis=0, keepdims=True)  # 센터링 O
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return vt.T.astype(np.float32)
```

### cowork 코드 (`quantize_nonuniform_with_basis`)

```python
def quantize_nonuniform_with_basis(keys, basis, target_bits, topk_frac, ...):
    coeffs = torch.matmul(keys, basis)  # <-- 센터링 없이 직접 투영!
    # ... quantize coeffs ...
    return torch.matmul(coeffs_q, basis.transpose(-1, -2))  # mean 복원 없음
```

### 우리 코드 (`fokvq_quantize_head`)

```python
def fokvq_quantize_head(K_head, bits_avg, gamma=0.3):
    K_f = K_head.float()
    mean = K_f.mean(dim=0)
    centered = K_f - mean                    # <-- 센터링 O
    cov = (centered.T @ centered) / max(centered.shape[0] - 1, 1)
    evals, evecs = torch.linalg.eigh(cov)
    K_pca = centered @ evecs                 # <-- 센터링된 데이터 투영
    # ... quantize K_pca ...
    K_recon = K_q @ evecs.T + mean           # <-- mean 복원
```

### 영향

PCA 계수에 DC offset(평균 성분)이 섞여 들어간다.
이로 인해 고유값 기반 bit 할당이 왜곡되며, 수십 PPL 악화가 발생할 수 있다.

### 수학적 설명

- PCA basis U는 centered 데이터 (X - mu)의 공분산에서 유도됨
- 올바른 투영: `c = (x - mu)U` --> 역변환: `x_hat = cU^T + mu`
- cowork의 투영: `c' = xU = (x - mu)U + muU` --> muU 항이 오염됨
- 오염된 c'의 분산 = Var(c) + (muU)^2 --> bit 할당 기준이 틀어짐

핵심은 `muU` 항이 각 PCA 축의 계수에 상수 오프셋으로 더해지면서,
원래 분산이 작은(= 적은 bit로 충분한) 하위 축에도 큰 값이 실리게 된다는 점이다.
결과적으로 bit 할당 자체가 무의미해진다.

---

## 2. Bit 할당 전략: 이진 분할 vs 연속 할당

### cowork: top-k 이진 분할

```python
# bit_schedule 함수: 2bit -> (5,1), 3bit -> (6,2), 4bit -> (7,3)
# top 25% axes -> high bits, bottom 75% -> low bits
coeffs_hi = symmetric_quantize_last_dim(coeffs[..., :top_k], high_bits)
coeffs_lo = symmetric_quantize_last_dim(coeffs[..., top_k:], low_bits)
```

- d=64일 때 2-bit 타겟: 16차원 5bit, 48차원 1bit
- **1-bit 양자화 = sign 정보만 남음** --> 48/64 차원의 magnitude 정보가 완전히 소실됨
- 이는 전체 차원의 75%에 해당하며, 2-bit 타겟에서 치명적

### 우리 코드: eigenvalue^gamma 기반 연속 할당

```python
ev_pos = np.maximum(ev_np, 1e-10)
w = ev_pos ** gamma          # gamma = 0.3
w /= w.sum()
ib = np.clip(np.round(w * d * bits_avg).astype(int), 1, 8)
# 예시 (2-bit avg): [5, 4, 4, 3, 3, 3, 2, 2, 2, 2, 1, 1, ...] -- 부드러운 그라디언트
```

- 중간 차원들이 2~3bit를 받아 정보를 보존
- Rate-distortion theory의 reverse water-filling에 더 가까운 할당
- 급격한 절벽(5bit vs 1bit)이 아닌 부드러운 그라디언트

### 영향

특히 2-bit 타겟에서 이진 분할은 catastrophic하다.
전체 차원의 75%가 sign-only(1-bit)가 되면, 양자화 오차가 급격히 증가한다.
연속 할당은 모든 차원에 최소 1bit를 보장하면서도 중요도에 비례한 할당이 가능하다.

---

## 3. 양자화 함수: Symmetric vs Asymmetric

### cowork: `symmetric_quantize_last_dim` -- zero 중심 대칭

```python
abs_max = tensor.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
scale = abs_max / qmax
quant = (tensor / scale).round().clamp(qmin, qmax)
return quant * scale
```

- 데이터가 [3, 7] 범위일 때 --> scale은 7 기준 --> [-7, 7] 범위를 양자화
- [-7, 3] 구간은 실제 데이터가 없음 --> **dynamic range의 57%가 낭비**

### 우리 코드: Asymmetric min-max 양자화

```python
c_min = col.min()
c_max = col.max()
n_lev = 2 ** int(ib[i])
step = max((c_max - c_min).item(), 1e-8) / (n_lev - 1)
K_q[:, i] = torch.round((col - c_min) / step) * step + c_min
```

- 데이터가 [3, 7] 범위일 때 --> 정확히 [3, 7]만 양자화
- 전체 dynamic range를 100% 활용

### 영향

센터링 누락(섹션 1)과 결합 시 더욱 심각해진다.
PCA 계수에 큰 mean offset이 있으면, symmetric 양자화는 그 offset을 수용하기 위해
실제 신호가 차지하는 범위보다 훨씬 넓은 구간을 양자화 레벨로 나눈다.
이는 유효 bit 수를 실질적으로 감소시킨다.

---

## 4. Calibration PCA vs Runtime PCA

### cowork: 캘리브레이션 기반 (train set 16 samples, max 512 tokens)

```python
def build_fokvq_bases(model, tokenizer, calibration_texts, device, max_len):
    # train set에서 per-layer per-head PCA basis 미리 계산
```

### 우리 코드: 런타임 PCA (현재 K 캐시에서 직접 계산)

```python
def fokvq_quantize_head(K_head, bits_avg, gamma=0.3):
    # 현재 head의 K 텐서에서 실시간 PCA
    cov = (centered.T @ centered) / max(centered.shape[0] - 1, 1)
```

### 분석

- **캘리브레이션 방식**: 배포에 실용적 (추론 시 PCA 비용 없음), 하지만 train/test 분포 불일치 위험
- **런타임 방식**: 정확도 높음, 하지만 매 window마다 PCA 계산 비용
- 이 차이는 정당한 trade-off이며, 그 자체로는 PPL 악화의 주요 원인은 아님
- **단, 캘리브레이션이 16 samples로 매우 적음** -- 충분한 통계 확보가 불가능하다.
  PCA basis의 품질이 낮아질 수 있으며, 특히 하위 고유벡터의 방향이 불안정해진다.

---

## 5. Sliding Window NLL 계산 (cowork이 더 나은 부분)

### cowork: prefix 마지막 logit + eval logits 결합

```python
combined_logits = torch.cat([
    prefix_outputs.logits[:, -1:, :],  # prefix -> eval[0] 예측
    outputs.logits[:, :-1, :],         # eval[0:N-1] -> eval[1:N] 예측
], dim=1)
token_nll, num_tokens = negative_log_likelihood_from_logits(combined_logits, target_ids)
```

- stride 전체 토큰을 평가 대상에 포함

### 우리 코드: eval self-prediction만 (stride-1 토큰)

```python
shift_logits = logits[:, :-1, :]   # (1, stride-1, vocab)
shift_labels = eval_ids[:, 1:]     # (1, stride-1)
```

- window당 1 토큰 누락

### 분석

cowork 방식이 표준 sliding window PPL 프로토콜에 더 부합한다.
prefix의 마지막 hidden state가 eval 구간 첫 토큰을 예측하는 logit을 생성하므로,
이를 포함하는 것이 정확하다.

window당 1 토큰 누락이 전체 PPL에 미치는 영향은 크지 않지만(stride가 클수록 무시 가능),
수정된 코드에서는 cowork의 방식을 채용할 가치가 있다.

---

## 6. 요약: PPL 악화 원인 우선순위

| 순위 | 문제 | 예상 PPL 영향 | 수정 난이도 |
|------|------|--------------|------------|
| **1** | PCA 투영 시 센터링 누락 | 매우 큼 (수십 PPL) | 2줄 추가 |
| **2** | 이진 bit 분할 (75%가 1-bit) | 큼 (2-bit에서 치명적) | 연속 할당으로 교체 |
| **3** | Symmetric 양자화 | 중간 | Asymmetric으로 교체 |
| **4** | 캘리브레이션 basis 불일치 | 작음~중간 | 런타임 PCA 또는 샘플 수 증가 |

문제 1~3이 결합되면 영향이 곱셈적으로 증폭된다.
센터링 누락으로 PCA 계수에 offset이 생기고, 이진 분할로 75% 차원이 1-bit가 되고,
symmetric 양자화로 그 1-bit마저 dynamic range가 낭비되는 연쇄 구조이다.

---

## 7. 수정 코드

`exp4_2_standard_ppl_benchmark_v2.py`에 위 문제들을 모두 수정한 버전을 제공한다.

핵심 수정 사항:

1. **센터링 추가**: PCA 투영 전 mean 제거, 역변환 후 mean 복원
2. **연속 bit 할당**: eigenvalue^gamma (gamma=0.3) 가중치 기반 per-dimension 할당
3. **Asymmetric 양자화**: min-max 기반으로 전체 dynamic range 활용
4. **Sliding window**: cowork의 combined logit 방식 채용 (더 표준적)

---

**Date**: 2026-04-02
**Author**: CDP PoC Team
**References**:
- cowork repo: https://github.com/iamseungpil/boltzmann-attention
- Our exp10_1: docs/architecture/patent/exp10_1_wikitext2_ppl.py
