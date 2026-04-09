# QCQA: Query-aware Cascaded Quantized Attention
## Theory — Hypothesis — Verification Plan (v1, 2026-04-09)

---

## 1. 의도 (Intent)

긴 문맥에서 query에 맞게 KV cache를 동적으로 활용하여, 같은 저장량에서 TurboQuant보다 나은 retrieval accuracy를 달성한다.

**핵심 제약**: 저장량(storage)은 prefill 시점에 고정. Query-dynamic은 decode-time computation에서만 가능. 따라서 "저장을 바꾸는 것"이 아니라 "같은 저장을 query에 맞게 다르게 읽는 것".

---

## 2. 이론 (Theory)

### 2.1 문제 설정

Key cache K가 b-bit으로 양자화되어 K̂로 저장. Decode time에 query q가 주어졌을 때:

```
True attention: α_t = softmax(q^T k_t / √d)
Quantized attention: α̂_t = softmax(q^T k̂_t / √d)  
Score error: η_t = q^T ε_t / √d,  ε_t = k_t - k̂_t
```

### 2.2 Score Error의 Query-Dependent Structure

Per-dim uniform quantization에서 ε_{t,j} ~ Uniform(-Δ_j/2, Δ_j/2) 독립:

```
Var(η_t | q) = σ²_t(q) = (1/d) Σ_j q_j² Δ_j² / 12
```

**핵심**: σ_t(q)는 query-dependent. 같은 양자화된 token이라도, query가 어떤 방향을 보느냐에 따라 score noise가 다르다.

### 2.3 Pairwise Flip Risk

Token t와 현재 quantized winner b의 ranking flip probability:

```
P_flip(t,b | q,K̂) = P(s_t > s_b | ŝ_b > ŝ_t)
                   ≈ Φ(−(ŝ_b − ŝ_t) / ν_{tb}(q))
```

여기서 ν_{tb}(q) = √(σ²_t(q) + σ²_b(q)) = √(2σ²(q)) (동일 quantization 가정)

**이 metric은 세 가지 정보를 결합**:
1. Margin (ŝ_b − ŝ_t): score gap — 크면 안전
2. Query direction noise (σ(q)): q가 noisy dim을 가리키면 위험 
3. Quantization structure (Δ_j): per-dim step size

### 2.4 Theorem: Optimal Refinement Selection

**Theorem 1 (Greedy Optimality)**: Budget r개 토큰을 FP16으로 재계산할 때, argmax-mismatch probability의 union bound:

```
P(mismatch) ≤ Σ_{t∉R} P_flip(t,b|q,K̂)
```

이 bound를 최소화하는 R*는 flip risk가 가장 높은 r개:

```
R* = top-r tokens by P_flip(t,b|q,K̂)
```

**Proof**: Rearrangement inequality. Refinement은 선택된 토큰의 P_flip을 0으로 만듦. Greedy가 bound 감소를 최대화.

### 2.5 TurboQuant과의 결합

TurboQuant = Random rotation + Lloyd-Max 2-bit + QJL 1-bit residual (total 3 bits)

**QCQA + TurboQuant 결합**:
- Storage: TurboQuant과 동일 (3 bits = 2-bit base + 1-bit QJL)
- TurboQuant default: QJL residual을 모든 토큰에 uniform 적용
- **QCQA 변형**: QJL residual을 **flip risk 기반으로 selective 적용**

하지만 이건 SRA와 같은 zero-sum 문제 (non-selected tokens 악화).

**더 나은 결합: Cascaded Computation**
1. TurboQuant 3-bit으로 전체 attention 계산 (1st pass) — TurboQuant과 동일
2. Flip risk가 높은 top-r 토큰만 FP16으로 재계산 (2nd pass) — 추가 computation
3. 2nd pass 결과로 attention weights 업데이트

이건 zero-sum이 아님: 모든 토큰이 3-bit attention을 받고, 추가로 critical 토큰만 refined.
추가 비용: r/N × attention 계산 (작은 overhead)

### 2.6 Gap Theorem

**Theorem 2 (Sparsity-Dependent Gap)**:

s-sparse attention (상위 s개 토큰이 attention mass의 (1-δ) 차지)에서:
- Cascaded attention의 argmax-mismatch rate가 uniform보다 낮을 조건:

```
Gain ∝ P_flip(challenger) × (1 - s/N)
```

- NIAH (s=1, N=4096): Gain 최대
- Dense PPL (s≈N): Gain → 0 (overhead만 추가)

---

## 3. 가설 (Hypotheses)

### H1: Flip Risk Calibration
P_flip 예측이 actual flip과 잘 calibrate됨 (Brier score < 0.1).

**검증**: FP16 vs 2-bit에서 실제 flip 빈도 수집, predicted P_flip과 비교.

### H2: Flip Risk > Raw Score for Selection
같은 budget r에서, flip risk 기반 선택이 raw score 기반보다 true winner recall이 높음.

**검증**: Recall@r (r = 16, 32, 64) by flip_risk vs by raw_score.

### H3: Cascaded TurboQuant + QCQA NIAH Recovery
TurboQuant 3-bit base + flip-risk-guided top-r FP16 refinement로 NIAH > TurboQuant alone.

**검증**: NIAH accuracy at r = {16, 32, 64, 128}. 비교: TurboQuant alone vs QCQA-cascaded.

### H4: Sparsity-Gap Correlation
Attention sparsity가 높은 layer/task에서 QCQA gain이 더 큼.

**검증**: Per-layer attention entropy vs QCQA gain scatter plot.

### H5: PPL Parity
Dense attention에서 QCQA의 2nd pass overhead가 negligible하고 PPL ≈ TurboQuant.

**검증**: WikiText-2 PPL 비교.

---

## 4. Kill Criteria

- H1 실패 (Brier > 0.2): flip risk metric 자체가 불량 → 다른 metric 탐색
- H2 실패 (flip_risk ≤ raw_score recall): metric이 raw score보다 못함 → method 불필요
- H3 실패 (NIAH 미개선): cascaded approach가 NIAH에 도움 안됨 → 방향 폐기

---

## 5. 실험 순서

### Phase 1: Theory Validation (GPU 0-3, 지금 실행 중)
| GPU | 실험 | 검증 |
|:---:|------|------|
| 0 | Mistral 2-bit flip calibration | H1, H2 |
| 1 | TurboQuant NIAH baselines | H3 baseline |
| 2 | Mistral 3-bit flip calibration | H1 control |
| 3 | Qwen 2-bit flip calibration | H1 generalization |

### Phase 2: Method Validation (Phase 1 통과 시)
| GPU | 실험 | 검증 |
|:---:|------|------|
| 0 | QCQA oracle: true-flip-risk top-r + FP16 recompute, NIAH | H3 |
| 1 | QCQA proxy: 2-bit-estimated flip-risk top-r + FP16 recompute, NIAH | H3 |
| 2 | QCQA + TurboQuant cascade: 3-bit 1st pass + top-r FP16 2nd pass, NIAH | H3 + TurboQuant 결합 |
| 3 | PPL evaluation of QCQA vs TurboQuant | H5 |

### Phase 3: Full Benchmark (Phase 2 통과 시)
- Multi-model (Mistral, Qwen, LLaMA)
- Multi-task (NIAH, LongBench, PPL)
- Sparsity-gap analysis (H4)
- SOTA comparison table

---

## 6. 기존 방법과의 포지셔닝

| Method | Token selection | Quant-noise-aware | Margin-aware | Theory |
|--------|:-:|:-:|:-:|:-:|
| H2O | Attention score eviction | ✗ | ✗ | ✗ |
| FIER | 1-bit key proxy | ✗ | ✗ | ✗ |
| Quest | Page-level max score | ✗ | partial | ✗ |
| RocketKV | Two-stage eviction | ✗ | ✗ | ✗ |
| TurboQuant | None (uniform 3-bit) | ✗ | ✗ | Rate-distortion |
| **QCQA** | **Flip risk metric** | **✓** | **✓** | **Argmax-mismatch bound** |

---

## 7. TurboQuant 결합: Refinement Hierarchy

### Architecture (Codex review 반영)
Cascaded refinement with escalating precision:

```
Level 0: Base quantization (2-bit or 3-bit TurboQuant) → all tokens
Level 1: QJL residual correction → tokens with flip_risk > θ₁  
Level 2: Medium-bit refinement (4-bit stored) → tokens with flip_risk > θ₂ after Level 1
Level 3: FP16 shadow (CPU offload) → tokens with flip_risk > θ₃ after Level 2
```

각 level에서 flip risk 재평가 → 필요한 토큰만 다음 level로 escalate.

### Memory Budget (Codex 지적 반영)
| Storage | Per-token bits | Where |
|---------|:-:|---|
| Base (TurboQuant) | 3 | GPU HBM |
| QJL residual | 이미 포함 (TurboQuant의 일부) | GPU HBM |
| 4-bit refinement key | +4 | GPU HBM (선택적) 또는 CPU |
| FP16 shadow | +16 | CPU DRAM (offload) |

Equal-memory 비교: 3-bit TurboQuant + QCQA vs 4-bit uniform
- QCQA HBM: 3 bits/token (TurboQuant과 동일)
- QCQA 추가: r개 토큰의 FP16 fetch from CPU = bandwidth cost만 추가
- Uniform 4-bit: 4 bits/token (33% 더 많은 HBM)

### NeurIPS Framing (Codex 추천)
논문의 claim은 **TurboQuant combo가 아닌 일반 원리**:
"Margin-and-uncertainty-aware refinement selection is a better use of cascaded attention budget than raw-score selection."
TurboQuant은 하나의 instantiation.

## 8. Reviewer Preemption

### Objection 1: Memory claim misleading
→ HBM과 total storage 분리 보고. Equal-HBM과 equal-total 두 설정에서 비교.

### Objection 2: Gaussian tail approximation brittle
→ Sub-Gaussian concentration bound로 약화. q^T Σ_ε q 일반 form 사용. Calibration plot으로 경험적 검증.

### Objection 3: 2nd pass may miss important tokens  
→ Interval-overlap criterion (uncertainty interval이 top-k threshold와 겹치면 선택) 추가. Failure analysis 포함.
