# Steering Paper Evidence Summary

Date: 2026-04-16

## 1. Purpose

This document is the canonical evidence summary for the steering paper in [paper/neurips2026_steering_v2](/home/v-seungplee/boltzmann-attention/paper/neurips2026_steering_v2) and [paper/neurips2026_steering_ko](/home/v-seungplee/boltzmann-attention/paper/neurips2026_steering_ko). Its job is to separate three things that had been mixed across older reports:

1. what is actually verified,
2. what is only a planning hypothesis,
3. what should stay out of the paper.

The paper should be driven by the first category only.

## 2. Central Intent

The paper is not a generic activation-steering paper and it is not a cache-compression paper. The paper-safe center is now two-tiered:

> Tier 1: for multi-tool sequential selection, ontology-guided query-side contraction has the correct sign and basis specificity, while stationary key-side amplification on the same basis is structurally mismatched.
>
> Tier 2: within Qwen Subtask4, two stronger operator variants emerge, namely full-subspace single-pass contraction and iterative emitted-state contraction with early-layer key assistance.

Tier 1 is cross-family and main-text safe. Tier 2 is real but currently Qwen-only, so it belongs in the operator-comparison layer of the paper rather than in the headline claim.

## 3. Verified Main Findings

### 3.1 Main benchmark: MetaTool Subtask4

Subtask4 is the main benchmark because each query requires exactly two tools. That makes it the cleanest local test of sequential coverage rather than single-step ranking.

| Model | Method | Macro F1 | Exact | Jaccard | Interpretation |
|---|---|---:|---:|---:|---|
| Qwen2.5-7B-Instruct | `no_steer` | 0.7307 | 0.5252 | 0.6673 | baseline |
| Qwen2.5-7B-Instruct | `ocq_qbias_b-0.1` | 0.7471 | 0.5272 | 0.6791 | positive sign |
| Qwen2.5-7B-Instruct | `ocq_bias_a0.3` | 0.6850 | 0.4728 | 0.6194 | negative sign |
| Llama-3.1-8B-Instruct | `no_steer` | 0.6227 | 0.5030 | 0.5872 | baseline |
| Llama-3.1-8B-Instruct | `ocq_qbias_b-0.1` | 0.6271 | 0.5070 | 0.5919 | positive sign |
| Llama-3.1-8B-Instruct | `ocq_bias_a0.3` | 0.3105 | 0.2616 | 0.2951 | catastrophic negative sign |

Interpretation:

- The absolute Q-bias gain is modest.
- The important pattern is sign consistency across two model families.
- The K-bias contrast is much stronger than the Q-bias lift: the same ontology basis is useful on the query side and destructive on the stationary key side.

### 3.2 Basis specificity is stronger than the raw gain

Qwen null-control results:

| Method | Real basis | Feature-shuffled | Random |
|---|---:|---:|---:|
| Q-bias `beta=-0.1` F1 | 0.7471 | 0.7254 | 0.7068 |
| K-bias `alpha=0.3` F1 | 0.6850 | 0.0000 | 0.0000 |

Interpretation:

- The effect is not explained by “any rank-matched perturbation works.”
- The ontology basis is the only basis that keeps the sign positive on the query side.
- On the key side, the real basis is the only basis that does not collapse completely.

This is the strongest empirical evidence in the current package.

### 3.3 The useful regime is narrow

Qwen Subtask4 Q-bias sweep:

| Method | F1 |
|---|---:|
| `no_steer` | 0.7307 |
| `ocq_qbias_b-0.1` | 0.7471 |
| `ocq_qbias_b-0.3` | 0.6224 |
| `ocq_qbias_b-0.5` | 0.6135 |

Interpretation:

- The method is not robust to coefficient magnitude.
- The paper should not imply that stronger suppression is better.
- The right interpretation is a localized coverage bias, not a generic confidence amplifier.

### 3.4 Small K-plus-Q interaction exists but is not yet headline material

Qwen Subtask4 micro-sweep:

| Method | F1 |
|---|---:|
| `no_steer` | 0.7307 |
| `ocq_qbias_b-0.1` | 0.7471 |
| `ocq_qkv_a0.025_v0_q-0.1` | 0.7529 |
| `ocq_qkv_a0.05_v0_q-0.1` | 0.7502 |
| `ocq_qkv_a0.075_v0_q-0.1` | 0.7270 |
| `ocq_qkv_a0.1_v0_q-0.1` | 0.7317 |
| `ocq_qkv_a0.15_v0_q-0.1` | 0.7266 |

Interpretation:

- There may be a narrow interaction band for very small `alpha_K`.
- The effect is too localized to anchor the paper without cross-model confirmation.
- This belongs in the appendix unless it reproduces cleanly on Llama as well.

### 3.5 Qwen operator comparison: two stronger variants are real

The 2026-04-17 result bundle adds a Qwen-only operator comparison on the same Subtask4 benchmark:

| Method | Macro F1 | Exact | Delta F1 vs baseline | Interpretation |
|---|---:|---:|---:|---|
| `no_steer` | 0.7307 | 0.5252 | 0.0000 | baseline |
| `ocq_qbias_b-0.1` | 0.7471 | 0.5272 | +0.0165 | cross-family-safe Q-only point |
| `ocq_qbias_b-0.03` | 0.7535 | 0.5332 | +0.0229 | full-subspace single-pass contraction |
| `iterative_kq` | 0.7524 | 0.5473 | +0.0218 | emitted-state contraction + early-layer K assist |
| `multipass_kq` | 0.7410 | 0.5091 | +0.0104 | recall-leaning but less F1-efficient |

Interpretation:

- The stronger `+2.2pp` story is real on Qwen.
- The two best Qwen operators are close in F1 but differ in behavior: `ocq_qbias_b-0.03` is precision-first, while `iterative_kq` gives the strongest Exact improvement.
- This does not yet replace the cross-family sign story because the stronger variants have not been locked on Llama under the same protocol.

### 3.6 Stability does not come from “perturbing less”

Effective perturbation magnitude:

| Basis | Mean `||ΔK q||` |
|---|---:|
| Real ontology | 621.30 |
| Random | 399.56 |
| Feature-shuffled | 291.98 |

Interpretation:

- The ontology basis does not win by being a weaker edit.
- The real basis perturbs more and still remains the only stable useful direction.
- This supports a “specific stable subspace” interpretation.

### 3.7 Perturbation theorem is diagnostic, not decorative

Empirical bound checks:

| Model | Layer | Median `LHS/RHS` |
|---|---:|---:|
| Qwen2.5-7B-Instruct | 13 | `2.357e-08` |
| Llama-3.1-8B-Instruct | 15 | `6.372e-08` |

Interpretation:

- The bound is extremely conservative on the tested interventions.
- The theorem supports the use of qaMSE-style perturbation language as a diagnostic.
- It does not prove benchmark accuracy and must not be framed that way.

## 4. Structural Interpretation

The strongest internally coherent story, after reading the v1-to-v3 reports together, is the following.

### 4.1 What failed

- Stationary key-side amplification can help single-target emphasis, but it has no explicit coverage state.
- Once the prompt requires two distinct tools, repeated amplification of the same ontology-aligned mass becomes a mismatch.
- Earlier broad framings that tried to make K-side boosting the main positive story for tool use are not supported by the current Subtask4 evidence.

### 4.2 What survived

- Query-side suppression on an ontology basis has the correct sign for discouraging repeated dominant facet reuse.
- The gain is small but sign-consistent.
- The basis-specificity evidence is much stronger than the raw average gain.
- On Qwen, stronger operator variants exist, but they should currently be read as operator-selection evidence rather than as cross-family mechanism proof.

### 4.3 What the paper should say

- The paper should argue for a task-structure mismatch, not universal superiority.
- The closest contrast is SEKA/AdaSEKA because they are the nearest key-side spectral steering family.
- The safest main claim is about sign, intervention site, and basis specificity on sequential tool selection.

## 5. Closest Prior Comparison Status

### 5.1 What is safe to say

- SEKA and AdaSEKA are the closest prior methods in form and spirit.
- Their natural operating point is key-side highlighting or amplification.
- A fair comparison requires the same prompt template, scorer, decode policy, and source basis handling.

### 5.2 What is not safe to say yet

- A clean headline claim that we beat SEKA or AdaSEKA on MetaTool Subtask4.
- Any claim based on broken-tokenizer runs, hardware-sensitive wrappers, or paper-proxy reimplementations.
- Any cross-paper score comparison with unmatched prompt formatting or scoring strictness.

### 5.3 Why this matters

The paper becomes vulnerable if it sounds like “we beat SEKA” without a matched table. The safer framing is:

> the current evidence already isolates a mechanism mismatch for stationary key-side steering, and the decisive external check is a matched SEKA/AdaSEKA comparison under one fixed evaluation protocol.

## 6. Main-Paper vs Appendix Allocation

### 6.1 Main paper

- Subtask4 main table on Qwen and Llama.
- Ontology null controls.
- One compact mechanism figure.
- Narrow structural theory.

### 6.2 Appendix

- Small-alpha K-plus-Q interaction.
- Scorer-robustness notes.
- Perturbation-bound details.
- Baseline integration notes and matched-comparison protocol.
- Qwen operator comparison beyond the main cross-family table.
- External robustness benchmarks only when their evaluators are official enough to be auditable.

## 7. Claims to Remove or Weaken

These claims appeared in older reports or broader drafts and should not drive the current paper.

- “K-side steering is the main positive accuracy contribution for tool use.”
- “The theory predicts benchmark accuracy directly.”
- “The paper is also a cache-compression contribution.”
- “A large SEKA gap is already established.”
- “Dynamic layer-adaptive Q+K is already a verified main method.”
- “tau2-bench already validates the main paper claim.”

## 8. External Benchmark Status

### 8.1 tau2-bench

- Current repo state: ontology builder exists, but the available evaluator is `eval_tau2bench_epsilon_q.py`, which is a per-step next-tool and $\epsilon_q$ diagnostic rather than the official full multi-turn success harness.
- Decision: tau2 is placeholder-only for the current paper package.
- Reason: it is too far from the locked Subtask4 claim, and the available code does not yet support a fair headline result.

### 8.2 BFCL

- Current repo state: `eval_bfcl.py` exists, but it uses function-name set matching rather than official BFCL AST-based scoring.
- Decision: BFCL is supporting-only, and only as a screening benchmark until the official scorer is integrated.
- Reason: it is useful for stress-testing single-tool versus multi-tool behavior, but the present evaluator is still a proxy.

## 9. Immediate Paper-Completion Priorities

### Priority 1

Lock the two-method operator comparison across model families:

- `ocq_qbias_b-0.03`,
- `iterative_kq`,
- plus `no_steer` and stationary `K-bias` under the same protocol.

This is the highest-value next step because it decides whether the Qwen `+2.2pp` story is paper-safe or just a single-model curiosity.

### Priority 2

Add the cleanest possible stepwise coverage analysis on Subtask4:

- first-tool accuracy,
- second-tool accuracy,
- repeated-first-facet failure rate.

This is the single most valuable missing experiment because it tests the claimed mechanism directly.

### Priority 3

Stabilize one matched external comparison:

- SEKA,
- AdaSEKA,
- or one clearly documented external baseline family under one locked protocol.

External breadth comes after this point. If an additional benchmark is needed before submission, BFCL is the only reasonable candidate, and even then only after its scorer is official enough to audit.

## 9. Out of Scope for This Paper

- PCA or KV-cache compression as a main contribution.
- Broad multi-benchmark sprawl without mechanism alignment.
- Dynamic routing or recursive residual cache ideas as headline contributions.
- Claims about exact emitted-facet tracking without an explicit stateful controller.

## 10. Bottom Line

The paper is viable if it stays narrow.

The evidence already supports a coherent contribution:

1. sequential multi-tool selection changes the preferred intervention sign,
2. query-side ontology suppression is the only stable positive direction currently verified across both Qwen and Llama,
3. the ontology basis matters in a way that random low-rank controls do not reproduce.

If the paper tries to say more than that, it becomes much easier to break.
