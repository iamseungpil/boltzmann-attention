# Axis 2 비등방성 인지 양자화 실험 계획

**프로젝트**: KV-Cache 양자화의 Axis 2 (양자화기) 재설계 — 비등방성 활용
**작성일**: 2026-04-06
**버전**: v1 — Spherical / Lattice / Fisher / L^p / Wasserstein 양자화 비교
**근거 문서**: NEURIPS_VERIFICATION_REPORT_v3.md (V3 negative result)

---

## 0. Executive Summary

V3 검증에서 확인된 가장 중요한 음성 결과: **Lloyd-Max는 MSE에서 3.5x 이득에도 불구하고 PPL에서 모든 비트(2/3/4-bit) × 모든 모델(Qwen/Llama/Mistral)에서 catastrophic failure**.

| 모델 | 2-bit Pre-RoPE PCA + Uniform | 2-bit Pre-RoPE PCA + Lloyd-Max | Lloyd 패배 |
|------|----------------------------|-------------------------------|-----------|
| Llama-3.1-8B | 10.14 | **65.46** | **6.5×** |
| Mistral-7B | 6.46 | **32.68** | **5.1×** |
| Qwen2.5-7B | 7.98 | 8.34 | 1.05× |

**근본 원인 진단**:
- Pre-RoPE PCA가 키 분포의 비등방성을 명시적으로 노출함 (R_aniso: Qwen 4.27, Llama 7.97, Mistral 131.62)
- Lloyd-Max는 L²(Banach Hilbert) metric의 등방성 cell 모양을 사용
- 결과: PCA가 노출한 비등방성을 quantizer가 cell shape에 활용하지 못함
- → **MSE-PPL gap의 직접 발현**

**본 실험의 목적**: Lloyd-Max를 대체할 비등방성 인지 양자화기 5종을 구현·비교하여, V3 음성 결과를 contribution으로 전환한다.

**5개 후보**:
1. **Spherical Quantization** (P0, 1주) — attention의 inner product 본질에 정렬
2. **L¹ Lloyd-Max (Median)** (P0, 3일) — heavy-tail robust
3. **Per-token Fisher Quantization** (P1, 1주) — softmax sensitivity 직접 활용
4. **Lattice Quantization (E_8 stretched)** (P1, 2주) — provable rate-distortion 우월
5. **Wasserstein-1D Quantization** (P2, 1주) — 분포 모양 보존

**예상 결과**: 적어도 하나의 방법이 Pre-RoPE PCA + Uniform baseline을 PPL에서 추가로 능가하면 axis 2가 negative result에서 contribution으로 전환됨.

---

## 1. 이론적 배경

### 1.1 Lloyd-Max의 본질적 한계 (수학적 진단)

Lloyd-Max 양자화기는 다음 변분 문제의 해:
$$Q^*_j = \arg\min_{Q_j} \mathbb{E}_k\bigl[(k_j - Q_j(k_j))^2\bigr]$$

**숨겨진 가정**:
1. **Metric**: Euclidean L² (모든 방향 동등)
2. **Generator**: Bregman divergence with $F(x) = \|x\|^2 / 2$
3. **Cell shape**: 등방성 ball
4. **Codebook**: per-dimension scalar (PCA 후)

PCA 후 차원들이 분산순으로 정렬되지만, **각 차원 안에서의 양자화는 여전히 등방성**입니다. Lloyd-Max는 차원별 분산만 활용하고, **차원 간 비등방성 구조 (R_aniso)를 cell shape에 통합하지 못합니다**.

### 1.2 비등방성을 활용하는 양자화의 수학적 요건

이상적 양자화기는 다음을 만족해야 합니다:

(A) **Cell shape의 비등방성 적응**:
$$\text{cell}(c) = \{k : d_W(k, c) \leq r\}, \quad d_W(k, c) = (k-c)^\top W (k-c)$$
여기서 $W$는 분포의 비등방성을 반영해야 함 ($W = \Sigma_K^{-1}$ 또는 $\Sigma_Q$).

(B) **Heavy-tail에 대한 robustness**:
Mistral의 R_aniso=131.62은 sub-exponential tail의 직접 증거. L² centroid는 outlier에 끌려가므로 dense bulk의 정밀도 손실. **Median (L¹) 또는 robust estimator** 필요.

(C) **PPL과의 직접 정렬**:
PPL 변화는 KL divergence로 정량화되며, 2차 전개 시 Fisher information $F$가 metric. **Fisher metric quantization**이 attention 보존에 직접 정렬.

### 1.3 V3 음성 결과의 해석

V3 보고서 9.3절: "MSE-optimal 스칼라 양자화기 설계가 attention distortion 최소화를 보장하지 않음을 강력히 시사"

이건 **L² Banach 공간의 한계**입니다. 다른 Banach 공간 또는 다른 metric으로 양자화를 재정의하면 이 한계를 우회할 수 있습니다.

---

## 2. 비교 대상 baseline (V3 데이터 기준)

모든 실험은 다음 baseline과 비교합니다:

### 2.1 V3에서 측정된 baseline (재현 불필요, 직접 인용)

| 방법 | Qwen2.5-7B | Llama-3.1-8B | Mistral-7B-v0.3 | 출처 |
|------|------------|--------------|----------------|------|
| **FP16** | 6.5559 | 6.3983 | 5.5717 | `ppl_*.json:L3` |
| **TurboQuant 2-bit** | 9.3315 | 11.2638 | 6.3708 | `ppl_*.json:L11` |
| **Pre-RoPE PCA + Uniform 2-bit** | 7.9804 | 10.1375 | 6.4614 | `ppl_*.json:L17` |
| **Pre-RoPE PCA + Lloyd-Max 2-bit** | 8.3433 | 65.4625 | 32.6844 | `ppl_*.json:L23` |
| **Pre-RoPE PCA + WF(f=2) 2-bit** | **7.0985** | **7.1588** | **5.8222** | `v15_*.json` |

### 2.2 본 실험의 비교 메트릭

각 새 quantizer를 다음 형태로 평가:
$$\text{Pre-RoPE PCA} + \underbrace{\textbf{[새 quantizer]}}_{\text{axis 2}} + \text{WF(f=2)}$$

목표: WF(f=2) baseline (Qwen 7.10, Llama 7.16, Mistral 5.82)을 PPL에서 능가.

**합격 기준**:
- **Strong PASS**: 3모델 모두 PPL 개선 (≥1%)
- **Partial PASS**: 1-2 모델 PPL 개선
- **FAIL**: 모든 모델에서 baseline 이하

---

## 3. 실험 1: Spherical Quantization (P0, 우선순위 1)

### 3.1 동기

Attention score는 $a_{ij} = q_i \cdot k_j / \sqrt{d}$로 **본질적으로 inner product**. RMSNorm이 적용된 LLM에서 키 벡터의 norm은 거의 일정 (median 변동 < 5%). 따라서:

$$\text{softmax}(q \cdot k / \sqrt{d}) \approx \text{softmax}\bigl(\|q\| \cdot \|k\| \cdot \cos\theta / \sqrt{d}\bigr)$$

여기서 $\theta$는 $q$와 $k$의 각도. **방향 정보가 attention의 dominant factor**입니다.

따라서 KV cache 양자화의 핵심은:
- $\|k\|$ (magnitude): 1차원, 1-bit으로도 충분
- $k / \|k\|$ (direction): $S^{d-1}$ 위의 점, 양자화 핵심

### 3.2 수학적 정의

**Spherical decomposition**:
$$k = \|k\| \cdot \hat{k}, \quad \hat{k} \in S^{d-1}$$

**Spherical quantization**:
1. Magnitude: $\|k\| \to Q_r(\|k\|)$ (1-bit uniform 또는 log-uniform)
2. Direction: $\hat{k} \to Q_S(\hat{k}) \in \mathcal{C}_S$ where $\mathcal{C}_S$ is a spherical codebook

**Spherical codebook 구성** (von Mises-Fisher mixture 기반):
1. Calibration set의 키들을 normalize: $\hat{k}_i = k_i / \|k_i\|$
2. Spherical k-means로 $N$개 centroid 학습 (Lloyd algorithm on sphere)
3. Centroid는 단위 벡터: $c_l \in S^{d-1}$
4. Assignment: $Q_S(\hat{k}) = \arg\max_l \langle \hat{k}, c_l \rangle$ (cosine similarity)

**복원**:
$$\hat{k}_{\text{recon}} = Q_r(\|k\|) \cdot Q_S(\hat{k})$$

### 3.3 비트 할당

총 b-bit per dimension. d=128에서 d × b = 256 bit (2-bit case). 분배:
- Magnitude: 8 bit (1-bit per scalar value, 충분)
- Direction codebook 크기: $N = 2^{(256 - 8)} = 2^{248}$ ... **불가능 (codebook explosion)**

**대안 1: Per-block spherical quantization**

키를 8차원 블록 16개로 분해. 각 블록에 대해:
- 블록 magnitude: 1 bit
- 블록 direction (7-sphere $S^7$): $2^{2 \cdot 8 - 1} = 2^{15}$ 코드 → 15 bit
- 블록당 16 bit, 총 16 × 16 = 256 bit ✓

**대안 2: Product spherical quantization**

차원을 64 × 2로 분해. 각 2차원 블록에서:
- 2D direction (각도 θ): 3 bit (= 8 angular bins)
- 2D magnitude: 1 bit
- 블록당 4 bit, 총 64 × 4 = 256 bit ✓

대안 2가 RoPE의 2D 회전 구조와 자연스럽게 정합 → **PolarQuant과의 연결성 확보**.

### 3.4 알고리즘 (Pseudocode)

```python
def spherical_quantize_calibrate(K_calib, n_blocks=64, block_dim=2, bits_per_block=4):
    """
    K_calib: (n_tokens, d) calibration keys
    Output: codebook (per-block spherical centroids)
    """
    n, d = K_calib.shape
    assert d == n_blocks * block_dim
    
    codebooks = []
    for b in range(n_blocks):
        # Extract block
        K_block = K_calib[:, b*block_dim:(b+1)*block_dim]  # (n, block_dim)
        
        # Decompose magnitude + direction
        mags = np.linalg.norm(K_block, axis=1, keepdims=True)  # (n, 1)
        dirs = K_block / (mags + 1e-10)                        # (n, block_dim) on sphere
        
        # Spherical k-means for direction
        n_dir_codes = 2 ** (bits_per_block - 1)  # half bits for direction
        dir_centroids = spherical_kmeans(dirs, k=n_dir_codes, max_iter=50)
        
        # Uniform quantization for magnitude (in log scale)
        n_mag_codes = 2 ** 1  # 1 bit
        mag_quantizer = LogUniformQuantizer(mags, levels=n_mag_codes)
        
        codebooks.append({
            'dir_centroids': dir_centroids,
            'mag_quantizer': mag_quantizer,
        })
    return codebooks


def spherical_quantize_encode(K, codebooks, n_blocks=64, block_dim=2):
    """Encode K using spherical codebooks per block."""
    K_recon = np.zeros_like(K)
    for b in range(n_blocks):
        K_block = K[:, b*block_dim:(b+1)*block_dim]
        mags = np.linalg.norm(K_block, axis=1, keepdims=True)
        dirs = K_block / (mags + 1e-10)
        
        # Direction: nearest centroid by cosine similarity
        sims = dirs @ codebooks[b]['dir_centroids'].T  # (n, n_dir_codes)
        dir_idx = np.argmax(sims, axis=1)
        dir_recon = codebooks[b]['dir_centroids'][dir_idx]
        
        # Magnitude: scalar quantization
        mag_recon = codebooks[b]['mag_quantizer'].encode_decode(mags)
        
        # Reconstruct block
        K_recon[:, b*block_dim:(b+1)*block_dim] = mag_recon * dir_recon
    return K_recon


def spherical_kmeans(X, k, max_iter=50):
    """Spherical k-means: centroids on unit sphere."""
    n, d = X.shape
    # Initialize: random k samples
    indices = np.random.choice(n, k, replace=False)
    centroids = X[indices]
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    
    for it in range(max_iter):
        # Assign to nearest centroid (cosine similarity = dot product on sphere)
        sims = X @ centroids.T  # (n, k)
        labels = np.argmax(sims, axis=1)
        
        # Update centroids: spherical mean = normalized sum
        new_centroids = np.zeros_like(centroids)
        for j in range(k):
            mask = labels == j
            if mask.sum() > 0:
                mean = X[mask].sum(axis=0)
                new_centroids[j] = mean / (np.linalg.norm(mean) + 1e-10)
            else:
                new_centroids[j] = centroids[j]
        
        # Check convergence
        if np.allclose(new_centroids, centroids, atol=1e-6):
            break
        centroids = new_centroids
    return centroids
```

### 3.5 실험 프로토콜

**모델**: Qwen2.5-7B, Llama-3.1-8B, Mistral-7B-v0.3 (V3 기준)

**비트 영역**: 2-bit, 3-bit, 4-bit (V3와 동일)

**구성**:
| Method | Axis 1 (rotation) | Axis 2 (quantizer) | Axis 3 (bit allocation) |
|--------|-------------------|-------------------|-----------------------|
| Baseline | Pre-RoPE PCA | **Uniform** | WF(f=2) |
| **New** | Pre-RoPE PCA | **Spherical (block=2)** | WF(f=2) |
| Reference (V3) | Pre-RoPE PCA | Lloyd-Max (failed) | WF(f=2) |

**캘리브레이션**:
- Dataset: WikiText-2 train, 160K tokens
- Per-layer, per-head spherical codebook 학습
- Codebook 학습 시간: ~30분/모델 (single GPU)

**평가**:
- WikiText-2 test PPL (V3와 동일 protocol)
- 추가: PG-19 PPL (out-of-distribution check)

### 3.6 가설 (Falsification Criteria)

**H1 (Primary)**: Pre-RoPE PCA + Spherical 2-bit < Pre-RoPE PCA + Uniform 2-bit (3모델)
- **PASS**: 3모델 모두 ≥ 1% 개선
- **PARTIAL**: 1-2 모델 개선
- **FAIL**: 모든 모델 악화 또는 변동 < 1%

**H2 (Mistral 회복)**: Spherical이 Mistral 2-bit 예외를 회복 (TurboQuant 6.371 능가)
- **PASS**: PPL < 6.37
- **FAIL**: PPL ≥ 6.37

**H3 (Llama 추가 이득)**: Llama 2-bit에서 WF(f=2) 7.16 → ≤ 7.0
- **PASS**: ≥ 2% 추가 개선
- **FAIL**: 추가 개선 < 2%

### 3.7 예상 시간 및 자원

| 단계 | 시간 | 자원 |
|------|------|------|
| Spherical k-means 구현 + 단위 테스트 | 1일 | CPU |
| Calibration (3모델) | 1일 | 1× A100 |
| PPL 측정 (3모델 × 3비트) | 2일 | 1× A100 |
| 결과 분석 + 보고 | 1일 | — |
| **총** | **5일** | — |

---

## 4. 실험 2: L¹ Lloyd-Max (Median Quantizer) (P0, 우선순위 2)

### 4.1 동기

Lloyd-Max는 L² metric에서 centroid = mean을 사용. **Heavy-tail 분포에서 mean은 outlier에 끌려갑니다**. Mistral의 R_aniso=131.62는 명백한 heavy-tail 증거.

L¹ Lloyd-Max는 centroid = **median**을 사용:
$$Q^*_j(L^1) = \arg\min_{Q_j} \mathbb{E}\bigl[|k_j - Q_j(k_j)|\bigr]$$

Median은 outlier에 robust → dense bulk에 더 정확한 codebook 배치.

### 4.2 수학적 정의

**Lloyd algorithm for L¹**:
1. Initialize: percentile-based codebook
2. Repeat:
   - Assignment: $k_{tj} \to$ nearest centroid in L¹
   - Update: centroid_l = median{$k_{tj} : k_{tj}$ assigned to $l$}
3. Until convergence

**가우시안에서의 등가성**: 가우시안은 mean = median이므로 L¹ Lloyd ≈ L² Lloyd. 차이는 **heavy-tail 분포에서만 발생**.

### 4.3 알고리즘 (Pseudocode)

```python
def l1_lloyd_quantize_1d(col, bits, n_iter=20):
    """L^1 Lloyd-Max (median-based) for 1D data."""
    n_levels = 2 ** bits
    
    # Initialize: percentile-based
    percentiles = np.linspace(0, 100, n_levels + 2)[1:-1]
    centroids = np.percentile(col, percentiles)
    
    for it in range(n_iter):
        # Assignment in L^1 (which is the same as L^2 for 1D scalar)
        dists = np.abs(col[:, None] - centroids[None, :])
        labels = np.argmin(dists, axis=1)
        
        # Update: centroid = MEDIAN (not mean)
        new_centroids = centroids.copy()
        for j in range(n_levels):
            mask = labels == j
            if mask.sum() > 0:
                new_centroids[j] = np.median(col[mask])  # L^1 centroid
        
        if np.allclose(new_centroids, centroids, atol=1e-6):
            break
        centroids = new_centroids
    
    # Encode and decode
    dists = np.abs(col[:, None] - centroids[None, :])
    labels = np.argmin(dists, axis=1)
    return centroids[labels]
```

### 4.4 실험 프로토콜

위 Spherical과 동일한 모델/비트/baseline. Quantizer만 L¹ Lloyd로 교체.

### 4.5 가설

**H1**: Mistral 2-bit Lloyd-Max (32.68) → L¹ Lloyd로 < 10
- **PASS**: 큰 폭 개선 (heavy-tail robustness 입증)
- **PARTIAL**: 일부 개선
- **FAIL**: 여전히 catastrophic

**H2**: Qwen 2-bit Lloyd-Max (8.34) → L¹ Lloyd로 ≤ 8.0
- 작은 개선 예상 (Qwen은 less heavy-tailed)

**H3**: Llama 2-bit Lloyd-Max (65.46) → L¹ Lloyd로 < 15
- 극단적 개선 예상 (Llama가 가장 heavy-tailed catastrophic)

### 4.6 예상 시간

| 단계 | 시간 |
|------|------|
| L¹ Lloyd 구현 (Lloyd-Max 기존 코드 수정) | 0.5일 |
| 측정 (3모델 × 3비트) | 1일 |
| 분석 + 보고 | 0.5일 |
| **총** | **2일** |

이게 가장 빠른 실험. **즉시 실행 가능**.

---

## 5. 실험 3: Per-Token Adaptive Fisher Quantization (P1)

### 5.1 동기

따름정리 6.19.13에서 우리는 $\bar{M}_{KL} \approx \Sigma_Q$로 근사했으나, V3 결과에 비추어 이 근사는 부정확합니다. 정확한 metric은 **token마다 다른** $M_{KL}(t)$:
$$M_{KL}(t) = Q^\top \bigl(\text{diag}(p_t) - p_t p_t^\top\bigr) Q$$

여기서 $p_t = \text{softmax}(q_t K^\top / \sqrt{d})$는 토큰 $t$의 attention 분포.

**핵심 통찰**: 평균($\Sigma_Q$)을 사용하면 토큰별 변동성을 잃습니다. Per-token quantizer가 더 정확합니다.

### 5.2 수학적 정의

**Per-token Mahalanobis quantization**:
$$Q^*(k_j; t) = \arg\min_c (k_j - c)^\top M_{KL}(t) (k_j - c)$$

토큰 $t$의 attention 분포에 따라 metric이 달라짐.

**구현 전략**:
1. Calibration 시 각 (layer, head, position) 별로 $p_t$ 분포 추정
2. Position을 K개 cluster로 묶음 (e.g., position-binned by attention entropy)
3. 각 cluster마다 다른 Mahalanobis quantizer 학습

### 5.3 알고리즘 (Pseudocode)

```python
def fisher_quantize_calibrate(K_calib, attention_logs, n_clusters=8, bits=2):
    """
    K_calib: (n_tokens, d) keys
    attention_logs: list of attention distributions p_t for each token
    """
    n_tokens, d = K_calib.shape
    
    # Compute per-token Fisher metric
    fishers = []
    for t in range(n_tokens):
        p = attention_logs[t]  # (n_keys,)
        # F = Q^T (diag(p) - pp^T) Q  (in our framework, Q = identity in PCA basis)
        # Simplified: F is roughly proportional to p variance
        F = np.diag(p) - np.outer(p, p)  # (n_keys, n_keys)
        fishers.append(F.diagonal())  # use diagonal for tractability
    
    # Cluster tokens by Fisher diagonal pattern
    fisher_features = np.stack(fishers)  # (n_tokens, n_keys)
    cluster_labels = kmeans(fisher_features, n_clusters)
    
    # Per-cluster Mahalanobis quantizer
    quantizers = []
    for c in range(n_clusters):
        mask = cluster_labels == c
        K_cluster = K_calib[mask]
        # Use cluster-mean Fisher as Mahalanobis weight
        W_c = np.diag(fisher_features[mask].mean(axis=0))
        # Standard Lloyd in W_c-weighted space
        K_white = K_cluster @ sqrtm(W_c)  # whitening
        codebook = lloyd_max_multidim(K_white, bits)
        quantizers.append({
            'codebook': codebook,
            'W_inv_sqrt': inv(sqrtm(W_c)),
        })
    
    return quantizers, cluster_labels
```

### 5.4 가설

**H1**: Per-token Fisher가 averaged $\Sigma_Q$ (MK Lloyd)를 PPL에서 능가
- **PASS**: V3 MK Lloyd 결과 대비 개선
- **FAIL**: 비슷하거나 악화

**H2**: 클러스터 수 K를 늘리면 PPL이 monotonically 개선
- K=1 (averaged): MK Lloyd
- K=8: 중간
- K=64: full per-token (computationally infeasible)

### 5.5 예상 시간

| 단계 | 시간 |
|------|------|
| Per-token attention log 수집 | 1일 |
| Fisher metric 계산 + 클러스터링 | 1일 |
| Per-cluster quantizer 학습 + 측정 | 2일 |
| 분석 + 보고 | 1일 |
| **총** | **5일** |

---

## 6. 실험 4: Lattice Quantization (E_8 Stretched) (P1)

### 6.1 동기

Conway-Sloane lattice 이론: $E_8$ lattice는 8차원 sphere packing의 최적해이며, 스칼라 quantization 대비 **0.65 dB rate-distortion 이득** (Eyuboglu-Forney 1992).

비등방성 처리: lattice를 $\Sigma^{1/2}$로 stretch하여 PCA basis의 비등방성에 적응.

### 6.2 수학적 정의

**Stretched lattice**:
$$\Lambda_{\Sigma} = \{\Sigma^{1/2} \cdot z : z \in \Lambda_{E_8}\}$$

**Quantization**: 
$$Q_\Lambda(k) = \arg\min_{\lambda \in \Lambda_\Sigma} \|k - \lambda\|^2$$

E_8 lattice의 closest point search는 $O(d)$ 복잡도 (Conway-Sloane Algorithm 1).

**Per-block 구조**:
- head_dim = 128 = 16 × 8
- 16개의 8차원 블록
- 각 블록에 stretched E_8 적용

### 6.3 알고리즘 개요

```python
def e8_stretched_quantize(K, Sigma_K, bits_per_block=16):
    """
    E_8 lattice quantization with anisotropic stretching.
    K: (n, d) keys, d = 128 = 16 * 8
    Sigma_K: (d, d) covariance
    """
    n, d = K.shape
    n_blocks = d // 8
    K_recon = np.zeros_like(K)
    
    for b in range(n_blocks):
        # Block extraction
        K_block = K[:, b*8:(b+1)*8]
        Sigma_block = Sigma_K[b*8:(b+1)*8, b*8:(b+1)*8]
        
        # Stretching transform (whitening)
        L = sqrtm(Sigma_block)
        L_inv = inv(L)
        K_white = K_block @ L_inv.T  # → roughly isotropic
        
        # E_8 quantization in whitened space
        # E_8 generator matrix (Conway-Sloane)
        # ... use Algorithm 2.2 from Conway-Sloane
        K_white_quant = e8_closest_point(K_white, scale=2**(-bits_per_block/16))
        
        # Inverse stretch
        K_recon[:, b*8:(b+1)*8] = K_white_quant @ L.T
    
    return K_recon


def e8_closest_point(x, scale=1.0):
    """
    Find closest E_8 lattice point to x (scaled).
    E_8 = D_8 ∪ (D_8 + (1/2, 1/2, ..., 1/2))
    where D_8 = {z ∈ Z^8 : Σ z_i is even}
    """
    x_scaled = x / scale
    
    # Two cosets of D_8
    # Coset 1: round to D_8
    z1 = np.round(x_scaled)
    # Force even sum
    parity = z1.sum(axis=-1) % 2
    if parity.any():
        # Flip the coordinate with largest residual
        residuals = np.abs(x_scaled - z1)
        worst = np.argmax(residuals, axis=-1)
        for i, w in enumerate(worst):
            if parity[i]:
                z1[i, w] += 1 if x_scaled[i, w] > z1[i, w] else -1
    
    # Coset 2: shifted by (1/2, ..., 1/2)
    z2 = np.round(x_scaled - 0.5) + 0.5
    parity2 = (2 * z2).sum(axis=-1) % 2
    # ... similar parity correction
    
    # Choose closer coset
    d1 = np.linalg.norm(x_scaled - z1, axis=-1)
    d2 = np.linalg.norm(x_scaled - z2, axis=-1)
    result = np.where(d1[:, None] < d2[:, None], z1, z2)
    return result * scale
```

### 6.4 가설

**H1**: E_8 stretched가 Pre-RoPE PCA + Uniform보다 PPL ≤ 5% 개선
- 이론 예측: 0.65 dB ≈ 7% MSE 개선 → 5% PPL 개선 가능
- **PASS**: ≥ 3% 개선
- **FAIL**: < 1% 개선

**H2**: Mistral 2-bit에서 비등방성 활용으로 PPL 최저
- **PASS**: WF(f=2) 5.82 → < 5.7

### 6.5 예상 시간

| 단계 | 시간 |
|------|------|
| E_8 closest point 알고리즘 구현 + 단위 테스트 | 4일 |
| Stretched lattice quantizer 통합 | 2일 |
| 측정 (3모델 × 3비트) | 3일 |
| 분석 + 보고 | 1일 |
| **총** | **10일 (2주)** |

---

## 7. 실험 5: Wasserstein-1D Quantization per PCA Component (P2)

### 7.1 동기

Wasserstein-2 quantization은 1D에서 closed-form. PCA 후 차원이 (대각화로) 분리되므로, **각 차원에 1D Wasserstein quantizer**를 적용 가능.

1D Wasserstein-2의 최적 quantizer는 Lloyd-Max와 일치 (centroidal Voronoi tessellation). 그러나 **empirical distribution에 기반**한 직접 계산:
$$c_i^* = \int_{F^{-1}((i-1/n))}^{F^{-1}(i/n)} x \cdot dF(x) / [F(F^{-1}(i/n)) - F(F^{-1}((i-1)/n))]$$

여기서 $F$는 empirical CDF.

### 7.2 알고리즘

```python
def wasserstein_1d_quantize(col, bits):
    """
    1D Wasserstein-2 quantizer based on empirical CDF.
    For uniform partition of CDF (equiprobable bins).
    """
    n_levels = 2 ** bits
    
    # Sort to get empirical CDF
    sorted_col = np.sort(col)
    n = len(sorted_col)
    
    # Equiprobable bins (uniform on CDF)
    bin_edges_idx = np.linspace(0, n, n_levels + 1).astype(int)
    
    # Centroid of each bin = conditional mean
    centroids = []
    for i in range(n_levels):
        start, end = bin_edges_idx[i], bin_edges_idx[i+1]
        if start < end:
            centroids.append(sorted_col[start:end].mean())
        else:
            centroids.append(sorted_col[start])
    centroids = np.array(centroids)
    
    # Encode
    dists = np.abs(col[:, None] - centroids[None, :])
    labels = np.argmin(dists, axis=1)
    return centroids[labels]
```

이건 **uniform-CDF Lloyd-Max**입니다 — Bennett companding의 이산 버전. Heavy-tail에 robust.

### 7.3 가설 및 예상 시간

**H1**: Lloyd-Max보다 안정적, Uniform과 비슷하거나 우월

| 단계 | 시간 |
|------|------|
| 구현 + 측정 | 3일 |
| **총** | **3일** |

---

## 8. 종합 실험 매트릭스

### 8.1 비교 표 (이상적 결과)

| Quantizer | 이론 metric | Banach 공간 | Heavy-tail robust | PPL 예상 (Llama 2-bit) |
|-----------|------------|------------|------------------|---------------------|
| Uniform | 없음 (baseline) | $L^2$ uniform | △ | 7.16 (V3 baseline) |
| Lloyd-Max | $L^2$ centroid | $L^2$ (Hilbert) | ✗ | 65.46 (V3 fail) |
| **L¹ Lloyd (median)** | $L^1$ median | $L^1$ (Banach) | ✓ | **< 10 (예상)** |
| **Spherical** | Cosine | $S^{d-1}$ | ✓ | **< 7.0 (예상)** |
| **Per-token Fisher** | $M_{KL}(t)$ | Riemannian | ✓ | ? (variable) |
| **E_8 lattice** | $L^2$ + lattice | $L^2$ + structure | ✓ | **< 7.0 (예상)** |
| **Wasserstein-1D** | $W_2$ on empirical | $\mathcal{P}(\mathbb{R})$ | ✓ | < 8.0 (예상) |

### 8.2 우선순위 및 timeline

| 우선순위 | 실험 | 시간 | Critical path |
|---------|------|------|--------------|
| **P0-1** | L¹ Lloyd (median) | **2일** | 가장 빠름, 가장 단순 |
| **P0-2** | Spherical (block=2) | **5일** | 이론적으로 가장 정렬 |
| **P1-1** | Per-token Fisher | 5일 | M_KL 직접 검증 |
| **P1-2** | E_8 lattice | 10일 | provable 우월성 |
| **P2-1** | Wasserstein-1D | 3일 | 가장 단순한 robust 대안 |

**Critical path**: P0-1 → P0-2 → 결과에 따라 P1 진행

총 P0 완료: **1주 (5일)**
총 P0+P1 완료: **3주**

---

## 9. 검증 환경

### 9.1 모델 (V3와 동일)

| 모델 | HF identifier | head_dim | layer 수 | KV head 수 |
|------|---------------|---------|---------|-----------|
| Qwen2.5-7B | `Qwen/Qwen2.5-7B` | 128 | 28 | 4 (GQA-7) |
| Llama-3.1-8B | `meta-llama/Llama-3.1-8B` | 128 | 32 | 8 (GQA-4) |
| Mistral-7B-v0.3 | `mistralai/Mistral-7B-v0.3` | 128 | 32 | 8 (GQA-4) |

### 9.2 데이터셋

**Calibration**: WikiText-2 train, 160K tokens (V3와 동일)
**Evaluation**: 
- WikiText-2 test (V3 비교용, primary metric)
- PG-19 (out-of-distribution check)
- (Optional) NIAH 8K, 16K (long-context check)

### 9.3 비트 영역

**Primary**: 2-bit (V3에서 가장 흥미로운 영역, Lloyd-Max 실패 영역)
**Secondary**: 3-bit, 4-bit (V3와 비교)

### 9.4 GPU 자원

| 작업 | GPU | 시간 |
|------|-----|------|
| Calibration (per model) | 1× A100/H100 | ~30분 |
| PPL evaluation (per config) | 1× A100/H100 | ~10분 |
| 총 (모든 P0+P1, 3모델 × 3비트 × 5방법) | 1× A100 | ~24시간 |

---

## 10. 결과 보고 양식

각 실험은 다음 표 형태로 보고:

```markdown
## 실험 X 결과: [Quantizer 이름]

### 모델별 PPL (2-bit)
| 모델 | FP16 | Pre-RoPE PCA + Uniform (V3) | Pre-RoPE PCA + [New] | 개선율 |
|------|------|----------------------------|---------------------|--------|
| Qwen2.5-7B | 6.56 | 7.10 (V3 WF f=2) | ? | ?% |
| Llama-3.1-8B | 6.40 | 7.16 (V3 WF f=2) | ? | ?% |
| Mistral-7B | 5.57 | 5.82 (V3 WF f=2) | ? | ?% |

### Hypothesis 검증
- H1: [statement] → PASS / PARTIAL / FAIL (근거)
- H2: [statement] → PASS / PARTIAL / FAIL (근거)
- H3: [statement] → PASS / PARTIAL / FAIL (근거)

### 해석
- ...

### 데이터 출처
- Source files: ...
- Hash: ...
```

---

## 11. 본 실험이 NeurIPS 논문에 미치는 영향

### 11.1 시나리오 분석

**시나리오 A: P0-1 (L¹ Lloyd) PASS, P0-2 (Spherical) PASS**

```
새 contribution:
  Theorem (Anisotropy-Aware Quantization Hierarchy):
    L^p Lloyd-Max with p = 1 (median) strictly dominates p = 2 (mean)
    on heavy-tailed sources, validated on Mistral, Llama 2-bit.
    
    Spherical quantization on S^(d-1) via block decomposition
    achieves additional X% PPL improvement by aligning with 
    attention's inner product structure.

영향:
  Axis 2: negative result → positive contribution
  Soundness: 6.5 → 7.5 (음성 → 양성 전환)
  Novelty: 7.0 → 7.5 (새 quantizer family)
  Overall: 6.5 → 7.5+ (clear accept)
```

**시나리오 B: P0 PASS, P1 PASS**

```
새 contribution:
  + Per-token Fisher quantizer (M_KL의 정확한 사용)
  + E_8 lattice with anisotropic shaping

영향:
  Soundness: 7.5 → 8.0 (full theoretical chain)
  Novelty: 7.5 → 8.0 (multiple new methods)
  Overall: 7.5 → 8.0 (strong accept)
```

**시나리오 C: P0 FAIL (가장 보수적)**

```
음성 결과 확정:
  Lloyd-Max뿐 아니라 L^1, Spherical도 fail
  → MSE/spherical/median 모두 PPL과 disconnect
  → "MSE-PPL fundamental gap"이 더 강력한 음성 결과
  
영향:
  Axis 2: 음성 결과로 확정, but 더 강한 evidence
  Soundness: 6.5 → 7.0 (체계적 검증)
  Overall: 6.5 → 7.0 (weak accept, but honest)
```

### 11.2 손실 시나리오 (실험을 안 했을 때)

**Reviewer Q (예상)**: "Why doesn't your axis 2 work? Have you tried L^1 or spherical alternatives?"

**현재 (실험 X)**: "We acknowledge Lloyd-Max fails empirically, leave alternatives to future work"
→ Reviewer가 "incomplete framework"로 score -0.5

**실험 후 (성공/실패 무관)**: "We tested 5 alternative metrics; results show [X]"
→ Reviewer가 "thorough investigation"으로 score +0.3

**ROI**: 실험 1주 → score +0.5~1.0

---

## 12. Risk Analysis

### 12.1 기술적 risk

| Risk | 확률 | 영향 | 완화 |
|------|------|------|------|
| Spherical k-means가 high-dim에서 수렴 안 함 | 중 | 중 | block size를 2-4로 작게 |
| L¹ Lloyd가 가우시안에서는 L²와 동일 (개선 없음) | 낮 | 낮 | Mistral에서 차별 예상 |
| E_8 implementation 버그 | 중 | 중 | 표준 라이브러리 사용 (cocoa) |
| Per-token Fisher computational cost | 높 | 중 | cluster로 근사 |

### 12.2 결과 risk

| Risk | 확률 | 대응 |
|------|------|------|
| 모든 alternative가 Uniform보다 못함 | 30% | 시나리오 C (음성 결과 강화) |
| 1개만 PASS, 나머지 FAIL | 50% | 시나리오 A (single new contribution) |
| 2개 이상 PASS | 20% | 시나리오 B (multiple contributions) |

### 12.3 timeline risk

| Risk | 완화 |
|------|------|
| P0 5일 안에 못 끝남 | L¹ Lloyd만 (2일)이라도 완료 |
| GPU 가용성 | V3 결과 재현 불필요, 비교만 (적은 GPU 사용) |
| 코드 버그 | 단위 테스트 + 작은 모델로 검증 후 본 실험 |

---

## 13. 즉시 실행 가능한 첫 단계 (Day 1)

### 13.1 L¹ Lloyd 구현 (오전 4시간)

기존 Lloyd-Max 코드에서 단 한 줄만 변경:
```python
# Before:
new_centroids[k] = col[mask].mean()

# After:
new_centroids[k] = np.median(col[mask])
```

### 13.2 Mistral 2-bit 측정 (오후 4시간)

V3와 동일한 protocol로 Mistral만 먼저 측정. Mistral이 가장 heavy-tailed이고 Lloyd-Max가 가장 catastrophic하게 실패한 모델.

**Day 1 결과**: Mistral 2-bit Pre-RoPE PCA + L¹ Lloyd PPL 측정값
- 만약 < 10이면 hypothesis 강력 지지 → P0-1 전체 진행
- 만약 ≈ 32 (변화 없음)이면 hypothesis 기각 → 다른 방향 (Spherical) 우선

이 single-day 실험으로 전체 P0의 viability를 빠르게 평가 가능.

---

## 14. 본 문서의 git 위치 및 후속 작업

**저장 위치**: `docs/architecture/paper/FOKVQ/paper/lie_group/AXIS2_ANISOTROPY_AWARE_QUANTIZATION_EXPERIMENT_PLAN.md`

**관련 기존 문서**:
- `NEURIPS_VERIFICATION_REPORT_v3.md` (V3 음성 결과의 출처)
- `LIE_GROUP_UNIFICATION.md` (이론 framework)
- `AXIS2_AXIS3_FAILURE_DIAGNOSIS_EXPERIMENT_DESIGN.md` (이전 진단 실험)

**후속 문서 (실험 후 작성)**:
- `AXIS2_ANISOTROPY_RESULTS_v1.md` (실험 결과)
- `THEOREM_6_19_X_ANISOTROPY_HIERARCHY.md` (새 정리 작성)

---

## 15. 참고문헌

### 비등방성 양자화 이론

- **Conway, J. H., & Sloane, N. J. A. (1999)**. *Sphere Packings, Lattices and Groups*. Springer. (Lattice quantization 표준)
- **Eyuboglu, M. V., & Forney, G. D. (1992)**. "Lattice and trellis quantization with lattice- and trellis-bounded codebooks." *IEEE Trans. Inf. Theory*, 38(2). (E_8 RD bound)
- **Hamkins, J., & Zeger, K. (1997)**. "Asymptotically dense spherical codes." *IEEE Trans. Inf. Theory*, 43(6). (Spherical quantization)
- **Bourne, D. P., & Roper, S. M. (2021)**. "Centroidal power diagrams, Lloyd's algorithm and applications to optimal location problems." *SIAM J. Numer. Anal.* (W_2 quantization)
- **Amari, S. (2016)**. *Information Geometry and Its Applications*. Springer. (Fisher metric)
- **Graf, S., & Luschgy, H. (2000)**. *Foundations of Quantization for Probability Distributions*. Springer. (L^p quantization 이론)

### KV cache 양자화 (선행 연구)

- **Hooper et al. (2024)**. "KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization." *NeurIPS 2024*.
- **Liu et al. (2024)**. "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache." *ICML 2024*.
- **Ashkboos et al. (2024)**. "QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs." *NeurIPS 2024*.
- **Liu et al. (2024)**. "SpinQuant: LLM Quantization with Learned Rotations." *ICLR 2025*.
- **Staniszewski & Łańcucki (2026)**. "KV Cache Transform Coding for Compact Storage in LLM Inference." *ICLR 2026*. (KVTC, 가장 가까운 prior)

---

## 16. 결론 및 권고

본 실험 계획은 V3 검증의 가장 중요한 음성 결과 (Lloyd-Max PPL 재앙)를 정리들의 한계 진단으로 받아들이고, 비등방성 인지 양자화기 5종을 체계적으로 비교하여 axis 2를 contribution으로 전환합니다.

**즉시 시작 권고**:
1. **Day 1 (오늘)**: L¹ Lloyd 구현 + Mistral 2-bit 측정 (8시간)
2. **Day 2-5 (이번 주)**: Spherical quantization 구현 + 3모델 × 2-bit 측정
3. **Day 6 (다음 주 시작)**: 결과 분석 → P1 방향 결정

**기대 효과**:
- Best case: 새 contribution 1-2개 추가, NeurIPS overall 6.5 → 7.5+
- Worst case: MSE-PPL gap의 더 강력한 음성 evidence, overall 6.5 → 7.0
- Either way: reviewer Q 차단, framework completeness 입증

---

*작성: Claude Opus 4.6 (2026-04-06)*
*근거: NEURIPS_VERIFICATION_REPORT_v3.md (V3 negative result), LIE_GROUP_UNIFICATION.md (이론 framework)*
*다음 단계: P0-1 (L¹ Lloyd) 즉시 실행, Mistral 2-bit 측정으로 hypothesis 빠른 검증*
