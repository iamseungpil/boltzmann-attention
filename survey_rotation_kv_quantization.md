# Rotation-based KV-Cache Quantization Methods: Comprehensive Survey
## NeurIPS 2026 Unified Lie Group Framework를 위한 선행연구 분석

---

## 1. 비교 총괄표

| # | Method | Venue (검증) | 핵심 변환 | 수학적 구조 | Lie Group 명시 | 최적성 증명 | 대표 성능 |
|---|--------|-------------|----------|------------|---------------|------------|----------|
| 1 | **TurboQuant** | ICLR 2026 (Google) | Random SO(d) via QR분해 | Beta분포 + Lloyd-Max scalar | **No** | Shannon bound ~2.7x 이내 | 3.5bit quality-neutral |
| 2 | **KVTC** | ICLR 2026 (NVIDIA) | Pre-RoPE PCA + DP bit alloc | 선형 decorrelation + entropy coding | **No** | No (heuristic DP) | 20x compression |
| 3 | **PolarQuant** | NeurIPS 2025 | 2D polar coord (r, theta) | 극좌표 변환 + RoPE 구조 활용 | **No** | No | 4.2x compression |
| 4 | **CommVQ** | ICML 2025 (Apple/UMass) | RoPE-commutative codebook | [[x,y],[-y,x]] 구조의 VQ | **No** | No | 2-bit: 87.5% 절감 |
| 5 | **SpinQuant** | ICLR 2025 (Meta) | Learned SO(d) via Cayley SGD | Stiefel manifold 최적화 | **암묵적** (Stiefel) | No (empirical) | W4A4KV4: 2.9pt gap |
| 6 | **RotorQuant** | Preprint 2026 (Scrya) | Cl(3,0) Clifford rotors | Block-diag 3D rotors, RxR~ | **No** (Clifford만) | No | PPL 6.91 vs TQ 7.07 |
| 7 | **IsoQuant** | Preprint 2026 | SO(4) isoclinic via quaternion | so(4) ~ su(2)_L + su(2)_R | **Yes** (so(4) 분해) | No (MSE만 비교) | 4.5-4.7x speedup over RQ |
| 8 | **QuaRot** | NeurIPS 2024 | Random Hadamard | Computational invariance | **No** | No | LLaMA2-70B: +0.47 PPL |
| 9 | **KIVI** | ICML 2024 | 없음 (비대칭 양자화) | Per-channel K / Per-token V | **No** | No | 2-bit, 2.6x memory |
| 10 | **KVQuant** | NeurIPS 2024 | Pre-RoPE per-channel | Dense-and-Sparse outlier 분리 | **No** | No | 3-bit: <0.1 PPL loss |

### 추가 발견 논문

| Method | Venue | 핵심 기여 |
|--------|-------|----------|
| **ParoQuant** | ICLR 2026 (Z Lab) | Sparse Givens rotation + channel scaling |
| **RotateKV** | IJCAI 2025 | Outlier-aware FWHT + Pre-RoPE grouped-head rotation |
| **KVLinC** | Preprint 2025 | Hadamard rotation + linear correction adapter |
| **SQuat** | Preprint 2025 | Query subspace-orthogonal 양자화 오차 |
| **PALU** | ICLR 2025 | Low-rank projection + Hadamard outlier 제거 |
| **QJL** | AAAI 2025 | 1-bit JL transform, unbiased inner product estimator |
| **GPTQ-Babai** | ICLR 2026 | GPTQ = Babai's nearest plane (lattice geometry) |

---

## 2. 개별 논문 상세 분석

### 2.1 TurboQuant (Google Research, ICLR 2026)

**저자**: Amir Zandieh, Majid Daliri, Majid Hadian, Vahab Mirrokni

**핵심 방법**:
- 입력 벡터 x에 random orthogonal matrix Pi (QR분해로 생성) 곱셈: y = Pi * x
- 회전 후 각 좌표가 Beta 분포 따름: f_X(x) = Gamma(d/2)/(sqrt(pi)*Gamma((d-1)/2)) * (1-x^2)^((d-3)/2)
- 고차원에서 N(0, 1/d)로 수렴 (concentration of measure)
- 좌표별 독립성을 이용해 Lloyd-Max scalar quantizer 적용
- **2단계**: MSE quantizer (b-1 bit) + 잔차에 QJL 1-bit sign quantization -> unbiased inner product estimator

**정보이론적 증명**:
- Shannon distortion-rate lower bound: D(B) >= 2^(-2B/d) for unit sphere
- TurboQuant MSE: D_mse <= (sqrt(3)*pi/2) * (1/4^b), Shannon bound 대비 ~2.7x factor
- **이것은 vector quantizer에 대한 information-theoretic optimality 증명이며, 현재까지 KV cache 양자화에서 유일한 formal bound**

**벤치마크**:
- Gemma, Mistral 모델 (LongBench, NIAH, ZeroSCROLLS, RULER, L-Eval)
- 3.5 bit: quality-neutral / 2.5 bit: marginal degradation
- Llama-3.1-8B LongBench-E: 2.5-bit 49.44 vs FP16 50.06
- Gemma 4B: 1-bit K + Q4 V -> PPL 36.00 vs 35.99 baseline

**이론적 갭**:
- Rotation을 SO(d) 원소로 명시하지만 **Lie group 구조를 활용하지 않음** (QR 분해는 Haar measure sampling에 해당하나 이를 언급하지 않음)
- Random rotation의 optimality를 증명하지 않음 (임의 rotation이면 충분하다는 것만 보임)
- **PCA rotation vs random rotation의 비교가 없음**

---

### 2.2 KVTC (NVIDIA, ICLR 2026)

**저자**: Konrad Staniszewski, Adrian Lancucki

**핵심 방법**:
- **PCA 기반 decorrelation**: calibration data에서 PCA basis V를 한 번 계산
- Pre-RoPE key에 적용 (RoPE 적용 전 양자화하여 outlier 구조 보존)
- **DP 기반 bit allocation**: PCA 순서 (분산 내림차순)에 따라 고분산 성분에 더 많은 bit 할당
  - 후미 주성분에 0 bit 할당하여 조기 차원 축소 효과
- **Entropy coding**: DEFLATE 알고리즘으로 양자화 심볼 압축
- Attention sink (4 oldest tokens) + sliding window (128 recent tokens) 보호

**벤치마크**:
- Llama 3, Mistral NeMo, R1-Qwen 2.5
- AIME25, GSM8K, LiveCodeBench, LongBench, MATH-500, MMLU, Qasper, RULER
- 20x compression에서 1점 이내 성능 유지, 특정 설정에서 40x 가능

**이론적 갭**:
- PCA가 왜 양자화 오차 최소화에 최적인지 증명 없음 (KLT가 decorrelation에 최적이라는 것은 알려져 있지만, 양자화 MSE 최소화와의 관계는 비자명)
- DP bit allocation이 global optimum인지, 또는 convex relaxation인지 불명확
- **Lie group 관점에서 PCA rotation은 data-dependent SO(d) 원소 선택에 해당하나 이를 formalize하지 않음**
- Pre-RoPE 선택의 이론적 근거가 부족 (경험적 관찰만)

---

### 2.3 PolarQuant (NeurIPS 2025)

**저자**: Songhao Wu, Ang Lv, Xiao Feng 등 (두 버전 존재: arxiv 2502.02617 / 2502.00527)

**핵심 방법**:
- Post-RoPE key vector를 2D 쌍으로 분할: (K[2j], K[2j+1])
- 극좌표 변환: r[j] = sqrt(K[2j]^2 + K[2j+1]^2), theta[j] = atan2(K[2j+1], K[2j]) + pi
- **핵심 관찰**: RoPE가 2D 쌍에 회전을 적용하므로, outlier가 한 차원에만 있어도 극좌표에서는 radius에 반영되고 angle은 smooth
- Radius: n-bit (non-negative, no zero-point), Angle: m-bit (range [0, 2pi))
- 주 실험: m=4, n=4 bits (실효 4.16 bits)

**벤치마크**:
- Llama-3.1-8B-Instruct
- LongBench (6 task categories), NIAH (4K-104K context), MMLU, GSM8K
- PPL 미보고, task score 위주
- 4.2x compression, 1.27x decode speedup

**이론적 갭**:
- 극좌표 변환이 SO(2) 회전과 동치라는 사실을 **인식하지 못함**
  - RoPE 자체가 block-diagonal SO(2) 회전이고, 극좌표는 이 SO(2) action의 orbit parametrization
- Random preconditioning 버전 (2502.02617)은 random SO(d) + recursive polar = TurboQuant의 변형에 해당
- **두 PolarQuant 버전 사이의 관계가 혼란**: 하나는 RoPE 구조 활용, 다른 하나는 random preconditioning
- 내적 계산의 table-lookup 주장은 두 번째 버전에서만

---

### 2.4 CommVQ (Apple/UMass, ICML 2025)

**저자**: Junyan Li, Yang Zhang 등 (UMass, MIT, Princeton, Apple 소속)

**핵심 방법**:
- **RoPE-commutative codebook**: codebook C가 [[x,y],[-y,x]] 형태일 때 R*C = C*R (모든 2D rotation R에 대해)
  - 이는 **SO(2)의 center** (혹은 commutant)에 해당
  - 수학적으로 이 구조는 complex number multiplication과 동치
- **Additive quantization**: encoder (linear -> activation -> linear + Gumbel-softmax) -> discrete codes -> codebook decode via matrix multiplication
- **EM training**: E-step (nearest clustering), M-step (codebook update)

**벤치마크**:
- LLaMA-3.1-8B, LLaMA-2-7B, Mistral-7B
- LongBench: CommVQ-2 47.98 vs FP16 48.05 (거의 무손실)
- GSM8K: CommVQ-2 76.04% vs FP16 76.27%
- CommVQ-1 (1-bit): 44.94 on LongBench (vs VQLLM-1: 27.42)
- 128K context on single RTX 4090

**이론적 갭**:
- [[x,y],[-y,x]] 구조가 SO(2)의 centralizer라는 것을 **명시적으로 Lie group으로 formalize하지 않음**
- Commutativity가 RoPE의 SO(2) 구조에서 자연스럽게 나온다는 사실을 algebraic하게만 처리
- **RoPE equivariance의 일반화 (높은 차원의 Lie group으로)에 대한 논의 없음**
- Codebook optimality에 대한 이론적 보장 없음

---

### 2.5 SpinQuant (Meta, ICLR 2025)

**저자**: Zechun Liu, Changsheng Zhao, Igor Fedorov 등 (Meta)

**핵심 방법**:
- **4개의 rotation matrix**: R1 (residual), R2 (head), R3 (online, value), R4 (online, FFN)
  - R1, R2: weight에 흡수 가능 (offline)
  - R3, R4: online Hadamard transform으로 고정 (효율성)
- **Cayley SGD on Stiefel manifold**: R1, R2를 학습
  - Stiefel manifold St(d,d) = {R in R^(dxd) : R^T R = I} 위에서 최적화
  - Cayley transform: R(t+1) = (I - tau*A)^(-1) * (I + tau*A) * R(t), where A is skew-symmetric
  - 이것은 **so(d) Lie algebra의 exponential map의 근사**에 해당
- 핵심 관찰: random rotation 간 **최대 13점 차이** (downstream zero-shot)

**벤치마크**:
- LLaMA-2 7B/13B/70B, LLaMA-3 8B
- W4A4KV4: LLaMA-2-7B에서 FP16 대비 2.9점 gap
- LLM-QAT 대비 19.1점, SmoothQuant 대비 25.0점 개선
- LLaMA-3 8B: QuaRot 대비 45.1% relative gap reduction

**이론적 갭**:
- Cayley SGD가 so(d) Lie algebra에서의 gradient flow라는 것을 **인식하지 못함**
  - Cayley transform = exponential map의 [1,1] Pade approximant
  - Stiefel manifold = SO(d) (when n=d)
- **왜 rotation이 양자화를 돕는지에 대한 이론적 분석 없음** (경험적 관찰만)
- Random rotation의 분산이 크다는 것을 보이지만, **어떤 rotation이 optimal인지 characterize하지 않음**
- PCA rotation과의 비교 부재

---

### 2.6 RotorQuant (Scrya, Preprint 2026)

**저자**: John D. Pope

**핵심 방법**:
- TurboQuant의 dense d x d rotation을 **block-diagonal Cl(3,0) rotors**로 대체
- 벡터를 3D 블록으로 분할, 각 블록에 Clifford rotor: R = exp(B/2), B는 bivector
- Rotor에는 4개의 non-zero component (scalar + 3 bivector)
- Sandwich product RxR~ 로 회전 적용, ~100 multiply-adds per vector
- **Parameters**: 372 (vs TurboQuant 16,384 for d=128)

**벤치마크**:
- Llama 3.1 8B at 10.3x compression
- PPL: 6.91 (RotorQuant) vs 7.07 (TurboQuant) vs 6.63 (FP16)
- Decode: 28% faster (119 vs 93 tok/s)
- Prefill: 5.3x faster
- O(d) complexity vs O(d log d) for TurboQuant

**이론적 갭**:
- Clifford algebra를 사용하지만 **Lie group Spin(3)과의 관계를 formalize하지 않음**
  - Cl(3,0)의 even subalgebra = quaternion algebra, Spin(3) ~ SU(2)의 double cover
- Block-diagonal 구조가 왜 충분한지 이론적 근거 없음
- **3D block이 왜 optimal인지** (vs 2D or 4D) 분석 없음
- Grade-aware quantization의 이론적 기반 부재

---

### 2.7 IsoQuant (Preprint 2026)

**저자**: Zhongping Ji

**핵심 방법**:
- **SO(4) isoclinic decomposition**: so(4) ~ su(2)_L + su(2)_R (Lie algebra 동형사상)
- 4D 블록을 quaternion으로 표현: T(v) = q_L v q_R_bar
- **IsoQuant-Full**: 양측 quaternion (q_L, q_R), SO(4)의 full 6 DOF
- **IsoQuant-Fast**: 좌측만 (q_L v), SO(3) subgroup, 4 vs 8 params/block
- d=128: 32개의 4D 블록, SIMD float4와 자연스럽게 정렬

**성능** (stage-1 quantize-dequantize만):
- FMAs at d=128: Full 1,024 / Fast 512 (vs RotorQuant 2,408 / TurboQuant 16,384)
- Mean kernel speedup: 4.5-4.7x over RotorQuant, peak 6x+
- MSE: RotorQuant와 동등 또는 약간 낮음

**이론적 갭**:
- **유일하게 so(4) Lie algebra를 명시적으로 사용하는 논문**
- 그러나 **end-to-end LLM 실험 없음** (synthetic vectors만)
- Block-diagonal SO(4)^g 구조가 full SO(d)에 비해 어떤 optimality를 가지는지 증명 없음
- **왜 4D가 최적 블록 크기인지** (2D, 3D, 8D 대비) 이론적 분석 없음
- RoPE와의 상호작용 미분석

---

### 2.8 QuaRot (ETH Zurich, NeurIPS 2024)

**저자**: Saleh Ashkboos, Amirkeivan Mohtashami, Maximilian L. Croci 등

**핵심 방법**:
- **Computational invariance**: weight와 activation 사이에 Hadamard rotation 삽입
  - QW * x = Q * (WR^T) * (Rx) -- rotation이 상쇄됨
- Weight, activation, KV cache 모두 4-bit 양자화
- Online Hadamard transform을 attention module에 적용하여 key/value outlier 제거
- Hadamard matrix: H_n = [[H_{n/2}, H_{n/2}], [H_{n/2}, -H_{n/2}]] / sqrt(2)

**벤치마크**:
- LLaMA-2 7B/13B/70B
- LLaMA2-70B 4-bit: WikiText-2 PPL +0.47 (3.79 vs 3.32 baseline)
- 99% zero-shot performance 유지
- Prefill 3.33x speedup, memory 3.89x saving
- 6-bit, 8-bit: lossless (round-to-nearest, no calibration)

**이론적 갭**:
- Hadamard matrix는 **SO(d)의 특수 원소** (orthogonal + det=+1일 때)이나 이를 Lie group으로 논의하지 않음
- Computational invariance는 **group action의 equivariance**와 동치이나 이를 formalize하지 않음
- **왜 Hadamard가 random rotation보다 좋거나 나쁜지** 이론적 비교 없음
- KV cache에 대한 독립적 bit-width 분석 부재 (W/A/KV 함께 4-bit만)

---

### 2.9 KIVI (ICML 2024)

**저자**: Zirui Liu, Jiayi Yuan, Hongye Jin 등

**핵심 방법**:
- **Rotation 없음**: 순수 양자화 전략
- Key: per-channel 양자화 (outlier가 특정 channel에 집중)
- Value: per-token 양자화 (outlier 분포가 다름)
- Tuning-free, plug-and-play

**벤치마크**:
- LLaMA, Falcon, Mistral
- 2-bit에서 "거의 동일한 품질"
- 2.6x peak memory 절감, 4x batch size, 2.35-3.47x throughput

**이론적 갭**:
- Key의 per-channel outlier 구조가 **왜** 존재하는지 분석 없음
- Per-channel이 최적인지 증명 없음
- Rotation 기반 방법과의 비교/결합 없음 (이후 논문들이 이를 개선)

---

### 2.10 KVQuant (UC Berkeley, NeurIPS 2024)

**저자**: Coleman Hooper, Sehoon Kim 등

**핵심 방법**:
- **Pre-RoPE per-channel Key quantization**: RoPE 적용 전 양자화하여 channel outlier 구조 보존
- **Dense-and-Sparse**: 1% outlier를 별도 저장, 나머지 dense quantization
- Per-layer sensitivity-weighted non-uniform datatypes
- 10M context on 8-GPU A100 시스템

**벤치마크**:
- LLaMA, Llama-2, Llama-3, Mistral
- 3-bit: <0.1 PPL degradation (WikiText-2, C4)
- 7B model: 1M context on single A100-80GB
- 1.7x speedup vs FP16

**이론적 갭**:
- Pre-RoPE가 왜 더 좋은지에 대한 **이론적 분석 부재** (경험적 관찰)
- Non-uniform datatype 설계의 optimality 미증명
- Rotation 기반 전처리와의 결합 미탐구

---

## 3. 핵심 이론적 분석

### 3.1 "Lie group"을 명시적으로 사용하는 논문

| 논문 | Lie Group 사용 | 수준 |
|------|--------------|------|
| IsoQuant | so(4) ~ su(2)_L + su(2)_R | **유일하게 Lie algebra 분해를 명시적 사용** |
| SpinQuant | Stiefel manifold (=SO(d)) | 암묵적: Cayley SGD는 so(d)에서의 gradient flow |
| ParoQuant | Givens rotation | SO(2)의 embedding이지만 Lie group으로 논의 안 함 |
| RotorQuant | Spin(3) via Cl(3,0) | Clifford algebra만 언급, Lie group 미연결 |
| 나머지 전부 | **없음** | rotation을 단순 linear algebra로 취급 |

**결론**: 양자화 문헌에서 Lie group/Lie algebra를 체계적으로 사용하는 논문은 **사실상 존재하지 않음**. IsoQuant이 가장 가까우나 end-to-end 실험이 없고, Lie group 이론의 깊이 있는 활용 (representation theory, optimal transport on G, etc.)은 전무.

### 3.2 PCA Rotation의 Optimality에 대한 증명

**현황**: PCA rotation이 양자화에 optimal이라는 formal proof를 제시하는 논문은 **발견되지 않음**.

관련 사실들:
- **KLT (Karhunen-Loeve Transform)** = PCA는 decorrelation에 최적 (이는 잘 알려짐)
- **Transform coding 이론** (Shannon): Gaussian source에 대해 KLT + independent scalar quantizer가 optimal
- TurboQuant: random rotation이 **임의의 분포**에 대해 near-optimal (data-independent)
- KVTC: PCA를 사용하지만 optimality를 주장하지 않음
- **Gap**: data-dependent PCA가 data-independent random rotation보다 **언제, 왜** 더 좋은지에 대한 formal analysis가 없음

### 3.3 Pre-RoPE vs Post-RoPE PCA 비교

**현황**: 체계적 비교 논문은 **발견되지 않음**.

관찰 사항:
- **KVQuant** (NeurIPS 2024): Pre-RoPE per-channel quantization이 경험적으로 더 좋음 (RoPE가 channel 간 magnitude를 섞어서 post-RoPE에서는 outlier가 분산)
- **KVTC** (ICLR 2026): Pre-RoPE PCA 사용
- **PolarQuant** (NeurIPS 2025): Post-RoPE 극좌표 변환 (RoPE의 SO(2) 구조를 직접 활용)
- **RotateKV** (IJCAI 2025): Pre-RoPE grouped-head rotation

**핵심 이론적 질문** (미해결):
1. RoPE = block-diagonal SO(2)^(d/2) action이므로, Pre-RoPE 분포와 Post-RoPE 분포는 SO(2)^(d/2) orbit 관계
2. PCA의 eigenvectors가 이 group action에 대해 어떤 equivariance 성질을 가지는가?
3. 최적 양자화 변환이 RoPE의 Lie group 구조와 어떻게 상호작용하는가?

---

## 4. Unified Lie Group Framework를 위한 Novelty Space 분석

### 4.1 기존 방법들의 Lie Group 재해석

| 방법 | Lie Group 재해석 |
|------|-----------------|
| TurboQuant random rotation | Haar measure on SO(d)에서 sampling |
| KVTC PCA rotation | Data-dependent SO(d) 원소 (KLT basis를 rotation으로 해석) |
| SpinQuant learned rotation | SO(d) 위의 gradient descent (Cayley = Lie algebra exponential의 Pade approx) |
| PolarQuant 2D polar | SO(2) orbit parametrization (angle = group coordinate) |
| CommVQ commutative codebook | SO(2)의 centralizer (= C*의 구조) |
| QuaRot Hadamard | Hadamard subgroup of O(d) |
| RotorQuant Clifford rotors | Spin(3) ~ SU(2) block-diagonal action |
| IsoQuant quaternion | SO(4) ~ (SU(2) x SU(2)) / Z_2 isoclinic decomposition |
| ParoQuant Givens | Product of SO(2) embeddings in SO(d) |
| RotateKV FWHT | Walsh-Hadamard subgroup action |

### 4.2 열린 이론적 질문 (논문 novelty 후보)

1. **Optimal rotation characterization**: 주어진 분포 P에 대해, 어떤 R in SO(d)가 양자화 MSE를 최소화하는가? PCA가 optimal인 조건은?

2. **Block-diagonal optimality**: SO(d) full rotation vs SO(k)^(d/k) block-diagonal -- 어떤 조건에서 block-diagonal로 충분한가? Block size k의 optimal 선택은?

3. **RoPE equivariance**: RoPE = block-diag SO(2)^(d/2)일 때, 양자화 변환 T가 RoPE와 commute해야 하는가? CommVQ의 commutativity 조건을 일반화할 수 있는가?

4. **Distortion-rate theory on Lie groups**: TurboQuant의 Shannon bound를 SO(d) 위의 분포로 일반화할 수 있는가?

5. **Gradient flow on SO(d)**: SpinQuant의 Cayley SGD를 so(d) 위의 Riemannian gradient flow로 formalize하고, 수렴 보장을 줄 수 있는가?

6. **Hierarchy of subgroups**: SO(2) < SO(3) < SO(4) < ... < SO(d) 의 subgroup chain에서 각 level의 양자화 trade-off를 characterize할 수 있는가?

### 4.3 가장 강력한 Novelty 방향

**방향 1**: "PCA는 Gaussian source에 대한 optimal SO(d) rotation이며, random Haar rotation은 minimax optimal이다" -- 이 두 extreme을 하나의 framework으로 통합

**방향 2**: "RoPE의 SO(2)^(d/2) 구조와 양자화 rotation의 SO(d) 구조 사이의 상호작용을 Lie group representation theory로 분석" -- Pre-RoPE vs Post-RoPE의 이론적 해결

**방향 3**: "Block-diagonal SO(k)^(d/k) 구조의 optimality를 Lie algebra 분해로 증명" -- RotorQuant(k=3), IsoQuant(k=4), PolarQuant(k=2), ParoQuant(k=2)를 통합

---

## 5. 비트별 대표 성능 수치 종합

### Key Cache Quantization PPL (WikiText-2 또는 동등, Llama 계열)

| Method | 2-bit | 3-bit | 4-bit | Model |
|--------|-------|-------|-------|-------|
| TurboQuant | - | ~+0.5% PPL | quality-neutral | Gemma/Mistral |
| KVTC | - | ~20x compress | - | Llama 3 |
| KVQuant | - | <0.1 PPL loss | - | LLaMA/Llama-2/3 |
| KIVI | ~same quality | - | - | LLaMA/Mistral/Falcon |
| QuaRot | - | - | +0.47 PPL (70B) | LLaMA-2 |
| CommVQ | 87.5% reduction | - | - | LLaMA-3.1-8B |
| RotorQuant | - | 6.91 (10.3x) | - | Llama 3.1 8B |
| PolarQuant | - | - | 4.2x compress | Llama 3.1 8B |
| SpinQuant | - | - | W4A4KV4: 2.9pt gap | LLaMA-2-7B |

*참고: 대부분의 논문이 서로 다른 메트릭, 모델, 설정을 사용하여 직접 비교가 어려움*

---

## 6. 결론 및 권고

### 현재 landscape 요약

1. **TurboQuant** (ICLR 2026)이 유일하게 information-theoretic optimality bound를 제공하며, 현재 이론적 gold standard
2. **Lie group을 체계적으로 사용하는 논문은 사실상 전무** -- IsoQuant이 so(4) 분해를 쓰지만 end-to-end 검증 없음
3. **PCA vs random rotation의 optimality 비교**가 공백 -- 가장 큰 이론적 기회
4. **RoPE와 양자화 rotation의 상호작용**에 대한 formal theory가 없음 -- CommVQ가 commutativity를 exploit하지만 일반화 부재
5. **Block size 선택의 이론적 근거** 없음 -- 2D (PolarQuant) vs 3D (RotorQuant) vs 4D (IsoQuant) vs d (TurboQuant) 사이의 trade-off가 경험적으로만 알려짐

### NeurIPS 2026 논문을 위한 구체적 권고

Unified Lie group framework 논문은 다음을 포함해야 함:

1. **SO(d) 위의 양자화 distortion 함수 F(R)의 정의 및 분석** -- R in SO(d)가 주어졌을 때 양자화 MSE가 어떻게 결정되는지
2. **F(R)의 minimizer characterization** -- Gaussian case: PCA = optimal, general case: random Haar = minimax
3. **Block-diagonal subgroup SO(k)^(d/k)로의 restriction** -- approximation error bound
4. **RoPE equivariance 조건의 formalization** -- 어떤 조건에서 양자화가 RoPE와 commute해야/하지 않아야 하는지
5. **실험적 검증**: TurboQuant, KVTC, SpinQuant, PolarQuant, CommVQ를 unified framework의 special case로 재현

---

*Survey 작성일: 2026-04-04*
*NeurIPS 2026 submission deadline 기준 novelty 분석*
