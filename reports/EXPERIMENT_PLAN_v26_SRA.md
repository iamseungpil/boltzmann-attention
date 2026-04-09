# 실험 계획 v26: Selective Residual Activation (SRA) — 2026-04-09

## 의도
Query에 맞게 long context를 동적으로 압축하여, 같은 bit budget에서 TurboQuant보다 나은 retrieval 성능 달성.

## 이론 Framework

### Core Insight
TurboQuant의 QJL residual은 모든 토큰에 uniform하게 적용. 하지만 sparse attention task (NIAH)에서는 소수 토큰만 중요. Residual을 query-time에 선택적으로 적용하면 이론적으로 최적.

### Theorem A: Attention-Weighted Error Decomposition
E[||Av - Âv||²] = Σ_{t,t'} C_{tt'}(q) · E[ε_t^T ε_{t'}]
where C_{tt'} = α_t(δ_{tt'} - α_{t'}) · ||v_t - v̄||²
→ Distortion weights depend on query through α

### Theorem B: Optimal Bit Allocation (Water-Filling)
b_t* = max(0, λ + ½log₂(w_t))
where w_t = α_t · ||v_t - v̄||²
→ 중요 토큰에 더 많은 bits

### Theorem C: Gap Between Uniform and Query-Optimal
Gap = AM(w)/GM(w) ≥ Θ(N/s)
→ s-sparse attention에서 gap ∝ N/s
→ NIAH (s=1): gap ~ N = 4096x
→ PPL (s≈N): gap → 1 (no benefit)

## Method: SRA

### Storage (same as TurboQuant)
- Random rotation + Lloyd-Max 2-bit base
- 1-bit QJL residual for all tokens
- Total: 3 bits/dim

### Inference (different from TurboQuant)
1. Compute proxy attention with 2-bit keys: proxy_α = softmax(q · K̂₂/√d)
2. Select top-k tokens by proxy_α
3. Apply 1-bit residual to top-k tokens: K̂₃[top-k] = K̂₂[top-k] + correction
4. Recompute attention with mixed-precision K (3-bit for top-k, 2-bit for rest)

### TurboQuant comparison
| | TurboQuant | SRA |
|---|---|---|
| Storage | 3 bits (2+1 QJL) | 3 bits (2+1 QJL) |
| Residual application | All tokens (uniform) | Top-k only (selective) |
| Query-dependent | No | Yes |
| NIAH (predicted) | ≤ uniform 3-bit (0.8) | > uniform 3-bit (~1.0) |
| PPL (predicted) | Same as TurboQuant | Same (gap → 0) |

## 가설 (반증 가능)

H1: SRA oracle (true attention weights로 selective) > TurboQuant uniform on NIAH
H2: SRA with proxy (2-bit score 기반 선택) ≈ SRA oracle (proxy recall 충분)
H3: SRA PPL ≈ TurboQuant PPL (dense attention에서 차이 없음)
H4: Gap은 attention sparsity에 비례 (Theorem C 검증)

## Kill Criterion
SRA oracle이 TurboQuant을 이기지 못하면 즉시 중단.

## 실험 순서

### Week 1: Oracle & Proxy
1. TurboQuant baseline 재현 (v3 인프라 사용)
2. SRA oracle: true attention → top-k residual → NIAH 비교
3. SRA proxy: 2-bit attention → top-k residual → NIAH 비교
4. Sparsity-gap correlation across layers

### Week 2: Full Benchmark
5. PPL (WikiText-2, C4)
6. LongBench
7. Multi-model (Qwen, LLaMA)

### Week 3-4: Paper
8. Theorems formal proof
9. Paper writing

## 핵심 리스크

1. **Proxy recall**: 2-bit proxy가 true top-k를 놓칠 수 있음
   → 이미 검증: needle median rank = 5-6 at 2-bit (OK)

2. **Two-pass overhead**: proxy + recompute = 2x attention 비용
   → 두 번째 pass는 top-k만이므로 overhead = k/N

3. **CliffKV와의 관계**: CliffKV가 실패한 건 patcher 버그 때문
   → v3 patcher 사용 필수. SRA는 v3 기반으로 구현.

4. **TurboQuant이 이미 QJL로 residual 적용**: 우리가 추가로 하는 건 "selective activation"
   → 저장량 동일, 적용 방식만 다름 → 공정한 비교
