# Experiment 5: Lie-Group Structured Generator Variants

**Date**: 2026-04-04
**Track**: H5 (Lie-group extensions beyond plain PCA)
**Phase**: 3 (Search for constructive Lie variants)
**Prerequisite evidence**: H1 (anisotropy confirmed), H2 (PCA > random/identity on surrogates), Exp 3-1 (full SO(d) Givens optimization FAILS to beat PCA: MSE 938.81 vs 220.45)

---

## Context

Exp 3-1 showed that unconstrained optimization on SO(d) via Givens rotations does not beat PCA on K-reconstruction surrogate metrics. This establishes that the bottleneck is not "insufficient optimization" — rather, the full SO(d) search space is too large for the calibration data available, leading to overfitting or poor convergence.

H5 tests the conjecture that **structured** subgroups of SO(d), motivated by Lie-algebraic reasoning, can outperform agnostic baselines by reducing the effective parameter count while preserving the covariance-aligned properties that make PCA strong.

---

## Exp 5-1: Block-Diagonal Rotation in PCA Space

```
Experiment: Exp 5-1 (block_diagonal)
Intent: Test whether local refinement within blocks of PCA-ordered dimensions
  improves quantization beyond PCA alone. Block-diagonal rotations form a
  sub-Lie-algebra so(b)^{d/b} ⊂ so(d), reducing parameters while preserving
  hierarchical variance structure.
Hypothesis: For at least one block size b ∈ {2, 4, 8, 16}, the composition
  PCA ∘ block_diag_SO(b) achieves lower quantization MSE than PCA alone on
  held-out tokens, without degrading beyond PCA on KL divergence.
Protocol:
  1. Extract per-layer per-head K vectors from GPT-2 Medium (calibration: 40 samples, max_seq_len=128)
  2. Split K vectors: 75% calibration, 25% held-out
  3. Compute PCA basis from calibration split
  4. For each block size b ∈ {2, 4, 8, 16}:
     a. Partition d=64 dimensions into d/b blocks (in PCA order)
     b. Optimize each block rotation in SO(b) to minimize 3-bit quantization MSE on calibration split
     c. Compose: R_final = PCA_basis @ block_diag(R_1, ..., R_{d/b})
  5. Compute surrogate metrics on held-out split:
     - K reconstruction MSE at 2, 3, 4 bits
     - Attention KL divergence (requires Q vectors) at 2, 3, 4 bits
Verification:
  - Smoke: composed R_final is orthogonal (max |R^T R - I| < 1e-4)
  - Smoke: all quantized outputs are finite
  - Success: MSE(block_b) < MSE(PCA) for at least one b on held-out, p < 0.05 paired t-test across layers
  - Stronger: KL(block_b) < KL(PCA) on held-out
Failure interpretation:
  - If no block size beats PCA on held-out MSE, the local coupling between
    adjacent PCA components is not quantization-relevant, and block-diagonal
    Lie structure adds nothing over the covariance eigenbasis.
Next action:
  - If SUCCESS on surrogates → integrate into exp4_2 harness for PPL evaluation
  - If FAIL → proceed to Exp 5-2
```

## Exp 5-2: Complex Unitary Residual

```
Experiment: Exp 5-2 (complex_unitary)
Intent: Test whether treating pairs of dimensions as complex numbers and
  applying U(d/2) ⊂ SO(d) gives a more natural parameterization for
  rotation optimization. U(d/2) has d²/4 parameters vs d(d-1)/2 for SO(d),
  and the complex structure respects the natural pairing of key dimensions.
Hypothesis: Complex unitary rotation (PCA pre-rotation + U(d/2) refinement)
  achieves lower quantization MSE than PCA alone on held-out tokens at 2-3 bits.
Protocol:
  1. Same K extraction and train/held-out split as Exp 5-1
  2. After PCA pre-rotation, pair consecutive dimensions: (dim_0, dim_1), (dim_2, dim_3), ...
  3. Parametrize refinement as U = exp(iH) where H is d/2 × d/2 Hermitian
  4. Map U back to a real d × d orthogonal matrix via the standard embedding
  5. Optimize H to minimize 3-bit quantization MSE on calibration split
  6. Evaluate surrogate metrics on held-out split
Verification:
  - Smoke: real embedding of U is orthogonal (max |R^T R - I| < 1e-4)
  - Smoke: U is unitary (max |U^H U - I| < 1e-4)
  - Success: MSE(complex) < MSE(PCA) on held-out, p < 0.05
Failure interpretation:
  - If complex unitary fails, the natural complex pairing of key dimensions
    does not provide quantization-relevant structure. The Lie framing cannot
    claim U(d/2) as a constructive improvement.
Next action:
  - If SUCCESS → integrate into exp4_2 harness for PPL
  - If FAIL → proceed to Exp 5-3
```

## Exp 5-3: Commutator-Regularized Optimization

```
Experiment: Exp 5-3 (commutator_reg)
Intent: Test whether constraining the Lie-algebra search to near-commute with
  the key covariance matrix Σ provides a regularization that improves
  generalization from calibration to held-out. PCA is exactly the solution
  that commutes with Σ (when eigenvalues are distinct), so commutator
  regularization interpolates between PCA and free optimization.
Hypothesis: Optimizing R = exp(A) where A ∈ so(d) with regularization
  loss λ||[exp(A), Σ]||_F achieves lower held-out MSE than both PCA (too rigid)
  and unregularized Givens (too flexible, overfits calibration).
Protocol:
  1. Same K extraction and train/held-out split as Exp 5-1
  2. Compute covariance Σ from calibration K per layer/head
  3. Parametrize rotation as R = exp(A) where A is skew-symmetric
  4. Loss = quantization_MSE(R @ K_cal) + λ × ||R Σ R^T - Σ||_F
  5. Sweep λ ∈ {0.01, 0.1, 1.0, 10.0} with gradient-free optimization (Powell)
  6. Evaluate best-λ variant on held-out split
Verification:
  - Smoke: R is orthogonal, outputs are finite
  - Success: MSE(commutator_reg, best λ) < MSE(PCA) on held-out AND
             MSE(commutator_reg, best λ) < MSE(Givens_unreg) on held-out
  - This would demonstrate that Lie-algebraic structure provides a principled
    regularization that neither PCA nor free optimization achieves
Failure interpretation:
  - If commutator-regularized rotation does not beat PCA: the covariance
    eigenbasis is already optimal within the space of near-commuting rotations,
    and the Lie framing adds no constructive value.
  - If it beats PCA but not Givens: regularization helps generalization but
    the commutator structure is not the right inductive bias.
Next action:
  - If SUCCESS → integrate into exp4_2 harness for PPL
  - If ALL 5-1, 5-2, 5-3 FAIL → Conclude Phase 3 as negative; lock paper to
    "theory + empirical analysis" (Phase 4 decision rule #2)
```

---

## Shared Evaluation Protocol

### Models
- **Primary**: GPT-2 Medium (24L/16H/d=64) — matches all existing baselines
- **Layers sampled**: Every 3rd layer (L0, L3, L6, ..., L21) — matches Exp 3-1

### Calibration
- 40 samples from WikiText-2 train set
- max_seq_len = 128
- 75/25 calibration/held-out split within extracted K vectors

### Surrogate Metrics (evaluated on held-out split)
1. **K reconstruction MSE**: ||K - dequant(quant(R @ K)) @ R^T||² at 2, 3, 4 bits
2. **Attention KL divergence**: KL(softmax(QK^T/√d) || softmax(Q K_quant^T/√d))

### Statistical Tests
- Paired t-test across layers for MSE comparison
- Wilcoxon signed-rank for KL comparison (non-parametric backup)
- Pre-registered significance level: p < 0.05

### Self-Test Requirements
- All rotation matrices are orthogonal (max |R^T R - I| < 1e-4)
- All quantized outputs are finite (no NaN/Inf)
- Reconstruction MSE is non-negative and decreases with higher bit-width
- Identity rotation reproduces the same MSE as non-rotated quantization

---

## Decision Flow

```
Exp 5-1 (block_diag) → SUCCESS on surrogate? ─ Yes → integrate into PPL harness
                                               └ No  → Exp 5-2

Exp 5-2 (complex_unitary) → SUCCESS? ─ Yes → integrate into PPL harness
                                       └ No  → Exp 5-3

Exp 5-3 (commutator_reg) → SUCCESS? ─ Yes → integrate into PPL harness
                                      └ No  → Phase 3 NEGATIVE
                                              Paper = "theory + analysis" (rule #2)
```

If ANY variant succeeds on surrogates AND subsequent PPL:
→ Phase 4 decision rule #1 applies (promote to "Lie-guided method" paper)

If surrogate success but PPL failure:
→ Same diagnosis as Exp 4-2: the bottleneck is the post-rotation quantizer
→ Paper framing shifts to "Lie structure identifies better rotations, but
   the scalar quantizer does not yet exploit this advantage"
