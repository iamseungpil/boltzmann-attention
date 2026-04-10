# Corollary 6.7 Drop Memo — Phase-Closure Claim Retired

**Date**: 2026-04-10
**Author**: auto (Claude Code)
**Status**: FINAL — Cor 6.7 removed from paper main body, demoted to Appendix negative result

---

## Decision

**Drop Corollary 6.7 (Exact Phase-Closure in the Facet Subspace Complement) from the main paper.**

The formal claim remains mathematically correct, but its empirical precondition
`q ⊥ Range(B)` is structurally violated on Qwen2.5-7B. Three independent lines
of evidence converge:

## Evidence Summary

### 1. Ontology-based ε_q diagnostic (3 centering variants)

| Variant | Centering | r_ont | MMLU ε_q | MetaTool ε_q | Separation |
|---------|-----------|------:|----------|-------------|-----------|
| v2      | none      | 20    | 0.1865   | 0.1973      | +0.0109   |
| v2c     | tool-grand| 31    | 0.1718   | 0.1888      | **+0.0170** |
| v2w     | wiki-mean | 20    | 0.1170   | 0.1320      | +0.0150   |

All three are **Row 3** (separation < +0.05). Best separation is +0.017 (v2c).
The isotropic floor for r=20 is 0.1563; MMLU ε_q hovers near or below floor,
meaning queries project onto Range(B) no more than onto a random 20-dim subspace.

### 2. W_Q^T·W_K SVD architectural baseline (Option (b))

| Metric | MMLU | MetaTool | Separation |
|--------|------|----------|-----------|
| ε_q    | 0.0901 | 0.0955 | **+0.0054** |

The **architectural upper bound** from weight SVD gives only +0.0054 separation.
No linear K-subspace derived from weights alone can separate task types at the
query level. This is definitive: the Qwen2.5-7B Q-K coupling is approximately
isotropic in head-space.

Per-layer profile: no layer exceeds +0.012 separation (L23 max). Singular value
spectrum is flat (S[19]/S[0] = 0.62 at L14).

### 3. Root cause: language-mean dominance

At Qwen2.5-7B's K-space, the top singular direction of category-mean K vectors
has |cos| > 0.99 with the K-mean of arbitrary English text. Two contributors:
1. `b_K` (qkv_bias=True, architectural constant)
2. `W_K · E[x_t]` (massive activation channels, layer-norm accumulation)

Centering removes the sum but does not create task-discriminative directions
because the residual content is shared between tool and non-tool queries.

## What Cor 6.7 Was and Why It Fails

**Cor 6.7 statement**: If `q ⊥ Range(B)`, then `qaMSE(q; E) = 0` (no
first-order attention perturbation for non-domain queries).

**Why it fails empirically**: Cor 6.7 requires that non-tool queries (MMLU)
have negligible projection onto Range(B). Measured ε_q := ||B^T q||² / ||q||²
shows MMLU queries project at ε_q ≈ 0.17 (v2c), far from the required ≈ 0.
The precondition `q_MMLU ⊥ Range(B)` is violated because:
- Tool and non-tool queries share the same verb/domain vocabulary at the
  K-projection level
- The Q-K coupling architecture (W_Q^T W_K) does not privilege any subspace
  for task-type discrimination

## What Survives

| Claim | Status | Mechanism |
|-------|--------|-----------|
| +11.15pp MetaTool Subtask1 lift | **CONFIRMED** | Thm 6.1 + Mode A/B/C |
| Theorem 6.1 (qaMSE → output bound) | **Proven** | Appendix B |
| Theorem 6.2 (rotation invariance) | **Proven** | Appendix B |
| Cor 6.3 (MSE → qaMSE bridge) | **Proven** | Appendix B |
| Cor 6.4–6.6 (Mode A/B/C) | **Proven** | §B.6 |
| Cor 6.7 (phase-closure) | **DROPPED** | Precondition violated |
| Cor 6.8–6.12 | **Appendix only** | Depend on Cor 6.7 context |

## Paper Restructuring

### Old frame (retire)
> "Cor 6.7 phase-closure: MMLU stays at 0 because q ⊥ Range(B), MetaTool
> gets selective bias"

### New frame
> "Thm 6.1 + Mode A/B sharpening: facet-gated K-bias produces task-specific
> attention re-concentration via K-norm amplification (low-temperature effect)
> and Var_s[V] asymmetry, yielding +11.15pp on MetaTool Subtask 1 with bounded
> −4pp MMLU cost explained by Mode C bulk-tail (Cor 6.6)"

### Appendix treatment
- Move Cor 6.7 statement + proof to Appendix C as "Negative result: why
  phase-closure does not explain the observed lift"
- Include the 3-variant centering table + W_Q·W_K SVD as empirical evidence
- Cite as: "While the formal claim is correct, its precondition is not
  satisfied in practice on GQA models with qkv_bias"

## Files Produced During Investigation

| File | Content |
|------|---------|
| `diag_epsilon_q_qwen25_7b.json` | v1 ontology ε_q diagnostic |
| `diag_epsilon_q_qwen25_7b_v2.json` | v2 clean ontology |
| `diag_epsilon_q_qwen25_7b_v2c.json` | v2c tool-grand centering |
| `diag_epsilon_q_qwen25_7b_v2w.json` | v2w wiki-mean centering |
| `diag_gate_distribution_qwen25_7b.json` | g_f gate distribution |
| `diag_gate_distribution_qwen25_7b_bossplit.json` | BOS-split gate |
| `diag_wqwk_svd_baseline_qwen25_7b.json` | W_Q^T·W_K SVD baseline |
| `build_qwen_metatool_b_ont_v2{,c,w}.json` | v2 build reports |

## Framing Traps to Avoid

1. "Cor 6.7 실패는 MMLU와 관련 없다" — **WRONG**. Failure IS about MMLU
   (precondition violation). The lift mechanism is not about MMLU (task-agnostic).
2. "Ontology rotation cannot achieve tool selection" — **WRONG**. +11.15pp is
   confirmed via a different mechanism (Thm 6.1 + Mode A/B/C).
3. "We need Cor 6.7 for the paper" — **WRONG**. Paper core is Thm 6.1/6.2 +
   Mode classification + empirical lift. Cor 6.7 was an aspirational bonus.
