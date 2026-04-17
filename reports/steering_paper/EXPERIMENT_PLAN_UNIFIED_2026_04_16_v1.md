# Unified Steering-Paper Experiment Plan

Date: 2026-04-16
Version: v1 (to be iterated with Codex critic)
Paper target: `paper/neurips2026_steering_v2`

## 0. Central claim

For multi-tool sequential selection, the correct intervention pattern is **Layer-Adaptive K+Q**: apply ontology-guided K-bias on the first quarter of transformer layers to encode tool-specific direction, and apply Q-coverage on the remaining layers to steer away from already-emitted facets. **Stationary K-side amplification** (SEKA family) is structurally mismatched for multi-tool tasks because it has no emitted-state memory and therefore cannot redirect the second decision. Query-only coverage is safe but insufficient alone, because it lacks the precision boost that K-bias provides during information encoding.

We prove this claim along three axes:

1. **Performance**: Layer-Adaptive K+Q beats stationary K, Q-only, SEKA, and a PCA-basis ablation on MetaTool Subtask4 and τ²-bench across Qwen and Llama families.
2. **Mechanism**: Stepwise decomposition shows the gain concentrates in the second-tool decision, and the repeated-first-tool rate falls for Layer-Adaptive but not for stationary K.
3. **Specificity**: The catalog-derived ontology basis `B_ont` remains the only basis that keeps the sign positive on the query side; feature-shuffled, random, and data-driven PCA bases all lose the effect.

Every experiment in this plan is tied to one of those three axes.

## 1. Methods compared

Every table uses the same five method families. Each method is named with the exact tag accepted by the evaluator CLI so that the result JSON, the paper table, and the code are auditable from a single string.

| Family | Tag | Description | Where implemented |
|---|---|---|---|
| Baseline | `no_steer` | Pure `model.generate` | `eval_metatool_subtask1.run_method` |
| Stationary K amplification | `ocq_bias_a0.3` | `K' = K + α·B_ont·B_ontᵀ·K`, all layers, all tokens | `install_kbias_hooks` |
| Query-only coverage | `ocq_qbias_b-0.03` | `Q' = Q + β·B_ont·B_ontᵀ·Q`, all layers, all tokens | `install_q_bias_hooks` |
| **Layer-Adaptive K+Q (ours)** | `ocq_ladapt_k0.05_q-0.03` | K on first L/4 layers, Q on remaining | `install_layer_adaptive_hooks` |
| SEKA (matched protocol) | `real_seka_amp1.0` | Canonical SEKA: `k_proj` hook on last-10 layers, marker-selected user tokens, `P_pos = B B^T` | `external/SEKA/src/model/seka_llm.py` via `eval_subtask4_with_real_seka.py` |
| PCA basis ablation | `ocq_qbias_b-0.03` with `--pca-baseline-path` | Same Q-coverage operator, but `B_ont` replaced by top-r PCA directions of calibration-set K activations | `build_pca_baseline_basis.py` |

We also run three controls as ablation rows:

- `ocq_bias_a0.05` — weak stationary K at the same rank to rule out an `α` mismatch.
- `ocq_ladapt_k0.05_q0` — K-early only, no Q, to isolate the K contribution.
- `ocq_ladapt_k0_q-0.03` — Q-only, no K, to isolate the Q contribution and compare against `ocq_qbias_b-0.03` under the same dispatch path.

## 2. Datasets and benchmarks

### 2.1 MetaTool Subtask4 (N=497)

Each query requires exactly two distinct tools from a catalog of 15 (retail, news, finance, etc.). This is the purest local test of sequential coverage.

### 2.2 τ²-bench (retail + airline, N=114 + 60)

Multi-turn tool-use benchmark with ground-truth action sequences (0–13 per task). We evaluate the **first-turn tool-selection set** against `evaluation_criteria.actions`. We do not evaluate full multi-turn success; the paper states this limitation explicitly.

Domain pool for the paper:

- retail (15 tools, N=114),
- airline (separate domain ontology, N=60),
- telecom and banking are listed as appendix extension rows.

### 2.3 MetaTool Subtask1 (parity regime, N=995)

Single-tool selection. Used only as a parity check that our K-bias matches SEKA-class performance in the single-concept regime. Not the main benchmark.

### 2.4 Null controls (basis specificity, Subtask4 and τ²-retail)

Three bases at the same rank `r=24`:

- real `B_ont` (catalog-derived),
- feature-shuffled `B_ont` (same entries, facet columns permuted),
- random orthonormal (Gaussian → QR).

Plus the PCA basis from §2.5 as a data-driven fourth basis.

### 2.5 PCA basis (for §4 ablation only)

Computed once per model: stack K activations of `n_kv` heads over 256 Subtask4 prompts, pad/crop to 512 tokens, do per-(layer, kv_head) SVD on the float32 K matrix, keep top-24 right-singular vectors. Saved as a `(L, H_kv, d, r)` tensor matching `B_ont` shape.

## 3. Models

Qwen family size sweep (confirming that Layer-Adaptive is not a single-size artifact):

- `Qwen/Qwen2.5-1.5B-Instruct`
- `Qwen/Qwen2.5-3B-Instruct`
- `Qwen/Qwen2.5-7B-Instruct` (primary)
- `Qwen/Qwen2.5-14B-Instruct`

Cross-family:

- `meta-llama/Llama-3.1-8B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.3` (Subtask1 and Subtask4 only; no τ² row because its chat template does not cleanly carry a tools list)

## 4. Experiment matrix

Every cell is one result JSON. Tables below show the expected direction where a result has not yet been locked. Placeholder predictions are written with the sign expected by the theory and with a confidence label (H = high, M = medium, L = low) based on whether a related observation exists.

### M1. τ²-bench retail, Qwen size sweep (primary performance table)

| Model size | `no_steer` | `ocq_bias_a0.3` | `ocq_qbias_b-0.03` | `ocq_ladapt_k0.05_q-0.03` | `real_seka_amp1.0` | `pca_qbias_b-0.03` |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-1.5B | placeholder | expected ↓ (H) | expected ≈ (M) | **expected ↑ (H)** | expected ↓ (H) | expected ≈ `no_steer` (M) |
| Qwen2.5-3B | placeholder | expected ↓ (H) | expected + (H) | **expected ↑↑ (H)** | expected ↓ (H) | expected ≈ `no_steer` (M) |
| Qwen2.5-7B | placeholder | expected ↓↓ (H) | expected + (H) | **expected ↑↑ (H)** | expected ↓↓ (H) | expected ≈ `no_steer` (M) |
| Qwen2.5-14B | placeholder | expected ↓ (M) | expected + (M) | **expected ↑ (M)** | expected ↓ (M) | expected ≈ `no_steer` (M) |

Success criterion: Layer-Adaptive positive on at least three of four sizes, and strictly better than Q-only on the primary size (7B).

Failure rule: If Layer-Adaptive does not beat Q-only at 7B, the paper downgrades Layer-Adaptive from main contribution to a diagnostic operator and promotes Q-only to the headline.

### M2. τ²-bench airline, Qwen 7B + Llama 8B

| Model | `no_steer` | `ocq_bias_a0.3` | `ocq_qbias_b-0.03` | `ocq_ladapt_k0.05_q-0.03` | `real_seka_amp1.0` | `pca_qbias_b-0.03` |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-7B | placeholder | expected ↓ (H) | expected + (M) | **expected ↑ (H)** | expected ↓ (H) | expected ≈ `no_steer` (M) |
| Llama-3.1-8B | placeholder | expected ↓↓ (H) | expected + (M) | **expected ↑ (M)** | expected ↓↓ (H) | expected ≈ `no_steer` (M) |

Success criterion: Sign consistency with M1 on retail.

### M3. MetaTool Subtask4 locked results (already auditable)

| Model | `no_steer` | `ocq_bias_a0.3` | `ocq_qbias_b-0.1` | `ocq_qbias_b-0.03` | `ocq_ladapt_k0.05_q-0.03` | `real_seka_amp1.0` | `pca_qbias_b-0.03` |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-7B F1 | 0.7307 | 0.6850 | 0.7471 | 0.7535 | **0.7514** (v5) | placeholder | placeholder |
| Qwen2.5-7B Exact | 0.5252 | 0.4728 | 0.5272 | 0.5332 | **0.5473** | placeholder | placeholder |
| Llama-3.1-8B F1 | 0.6227 | 0.3105 | 0.6271 | placeholder | placeholder | placeholder | placeholder |
| Llama-3.1-8B Exact | 0.5030 | 0.2616 | 0.5070 | placeholder | placeholder | placeholder | placeholder |

Placeholder cells are completed in M4 (cross-family lock). Layer-Adaptive Qwen cell is taken verbatim from `reports/layer_adaptive_2026_04_17/` on the develop branch (PAPER_DRAFT_v4 §4.2, `k_early_only` iterative).

### M4. Cross-family lock (Subtask4)

Rerun `ocq_qbias_b-0.03`, `ocq_ladapt_k0.05_q-0.03`, `real_seka_amp1.0`, `pca_qbias_b-0.03` on Llama-3.1-8B-Instruct with identical prompt template, decode policy, and scorer as the Qwen row. The result JSON schema must match the Qwen JSON.

Failure rule: If Layer-Adaptive sign flips on Llama, the paper reframes as "Qwen-family result with Llama direction confirmed for Q-only only".

### M5. Stepwise mechanism decomposition (Subtask4)

Using the `stepwise` block already emitted by `eval_metatool_subtask4.py`, compare six methods:

| Method | first_tool_hit | second_tool_hit | second_distinct_hit | repeated_first_tool |
|---|---:|---:|---:|---:|
| `no_steer` | placeholder | placeholder | placeholder | placeholder |
| `ocq_bias_a0.3` | ≈ `no_steer` (H) | ↓↓ (H) | ↓↓ (H) | **↑↑ (H)** |
| `ocq_qbias_b-0.03` | ≈ `no_steer` (H) | ↑ (M) | ↑ (M) | ↓ (M) |
| `ocq_ladapt_k0.05_q-0.03` | **↑ (H)** | **↑↑ (H)** | **↑↑ (H)** | ↓ (M) |
| `real_seka_amp1.0` | ↑ (M) | ↓ (H) | ↓ (H) | **↑↑ (H)** |
| `pca_qbias_b-0.03` | ≈ `no_steer` (H) | ≈ `no_steer` (M) | ≈ `no_steer` (M) | ≈ `no_steer` (M) |

**Intent**: this is the mechanism argument. If Layer-Adaptive is real, the gain must show up in the `second_distinct_hit` column and the `repeated_first_tool` column must drop. If the gain shows up only in `first_tool_hit`, the paper cannot call the effect a sequential-coverage mechanism.

Success criterion: Layer-Adaptive beats all other methods in both `second_distinct_hit` and `repeated_first_tool` reduction.

Failure rule: If Layer-Adaptive wins only on `first_tool_hit`, the discussion section replaces sequential-coverage language with a precision-shift interpretation.

### M6. Basis-specificity controls (Subtask4)

| Basis | `ocq_qbias_b-0.03` F1 (Qwen) | `ocq_bias_a0.3` F1 (Qwen) |
|---|---:|---:|
| Real `B_ont` | 0.7471 (locked) | 0.6850 (locked) |
| Feature-shuffled | 0.7254 (locked) | 0.0000 (locked) |
| Random orthonormal | 0.7068 (locked) | 0.0000 (locked) |
| PCA-calibration | placeholder | placeholder |

Expected direction: PCA row collapses or is neutral for both operators. This separates "catalog-derived ontology" from any rank-matched subspace.

### M7. Subtask1 parity row (appendix)

| Model | Baseline | `ocq_bias_a0.3` | Δ |
|---|---:|---:|---:|
| Llama-3.1-8B | 62.31% | 77.39% | **+15.08pp** (locked) |
| Qwen2.5-7B | 75.58% | 86.73% | **+11.16pp** (locked) |

Same operator is destructive on Subtask4 (M3). This single comparison is the cleanest evidence of the stationary-K mismatch.

### M8. ε_q stopping criterion (appendix)

AUROC of `ε_q` predicting failure on held-out Subtask4 smoke (N=100). Locked at 0.976 on develop. This is a deployment-time claim, not a mechanism claim, and is appendix-only.

## 5. Layer-Adaptive schedule sweep (ablation, appendix)

Qwen2.5-7B, Subtask4 N=497, `α=0.05, β=-0.03`:

| Schedule | K layers | Q layers | Expected F1 |
|---|---|---|---:|
| `uniform` | 0..L | 0..L | ≈ 0.685 (H) — K everywhere breaks late layers |
| `k_early_only` | 0..L/4 | 0..L | **≈ 0.751 (locked)** |
| `layer_adaptive` | 0..L/5 full, L/5..3L/4 weak | L/5..L | ≈ 0.751 (locked) |
| `q_late_only` | 0..L | 3L/4..L | ≈ 0.73 (M) — K everywhere still dominates |
| `k_mid_only` | L/4..3L/4 | 0..L | ≈ 0.71 (L) — new control |

Intent: show the U-shape MSE story predicts which schedules succeed. Schedules that apply K to late layers (including `uniform` and `q_late_only`) must not beat `k_early_only`.

## 6. Execution plan and GPU budget

| Experiment | GPUs | Estimated runtime |
|---|---|---|
| M1 Qwen size sweep × 5 methods × N=114 | 1× A100 80GB, sequential | ~12 h |
| M2 retail + airline × 2 models × 5 methods | 1× A100 80GB | ~8 h |
| M3 Subtask4 lock (Qwen + Llama) × 5 methods × N=497 | 1× A100 80GB | ~10 h |
| M4 cross-family lock | ~4 h | |
| M5 stepwise decomposition | already computed from M3 `stepwise` block | |
| M6 basis controls (PCA row) | 1× A100 80GB | ~2 h |
| M7 Subtask1 parity | already locked on both models | |
| §5 schedule sweep | 1× A100 80GB | ~5 h |

Total net time: ≈ 41 h on one A100. Parallelize across two nodes to finish in a day.

## 7. Protocol lock

All paper-facing runs use:

- `attn_implementation="eager"` (deterministic) for Subtask4 and τ²; `sdpa` only when `real_seka` path requires it; the JSON records the chosen implementation.
- `do_sample=False, max_new_tokens=300` for Subtask4, `max_new_tokens=512` for τ²-retail.
- Same scorer (`generation_tool_call_set_v1` for Subtask4; `tau2_action_set_v1` for τ²).
- Same prompt template identifier (`fc_chat_template_v1` for Subtask4; `tau2_fc_chat_template_v1` for τ²).
- Every result JSON records `prompt_template_id`, `scorer`, `decode_policy`, `runtime_config`, and `basis_source`.

## 8. Gate: what must be true before the paper can submit

- G1: M1 run with all five methods on at least three Qwen sizes, with Layer-Adaptive positive on at least two of the three.
- G2: M3 Layer-Adaptive row on Qwen AND Llama.
- G3: M5 stepwise table where Layer-Adaptive `second_distinct_hit` is greater than every other method.
- G4: M6 PCA row producing a collapse or neutral result relative to real `B_ont`.
- G5: M4 SEKA row filled with canonical SEKA via `eval_subtask4_with_real_seka.py` using an A100-reproduced ES=0.952 baseline as a protocol anchor (§8 below).
- G6: Every paper table caption cross-references the exact result JSON path that produced its numbers.

## 9. SEKA reproduction protocol lock

The canonical SEKA CounterFact number (ES=0.952 on Qwen3-4B-Base) is the baseline that signals our SEKA harness is correctly wired. We reproduce it on A100 before running any SEKA row in M1–M4:

1. Run `external/SEKA/benchmarks/eval_fact_gen.py` with the exact command in `reports/COWORKER_SEKA_REPRO_GUIDE_2026_04_16.md` §1.4.
2. Expected result: `efficacy_metrics.json → score.mean ≈ 0.95` within ±0.02.
3. If the A100 run reproduces, `eval_subtask4_with_real_seka.py` inherits the same canonical implementation and any subsequent SEKA row is comparable.
4. If the A100 run does not reproduce, the paper cites SEKA as a reference method rather than a head-to-head comparison, and M1 SEKA rows are labeled "canonical SEKA (A100 reproduction pending)" with the CounterFact divergence reported in the appendix.

## 10. Reviewer objections we pre-empt

- "SEKA vs ours uses different layer range." — M1 runs both `ocq_bias_a0.3` (all layers) and `real_seka_amp1.0` (last-10 layers); every paper row reports the layer policy. The paper argues about intervention *type* (stationary K vs layer-adaptive K+Q), not about scheduling superiority.
- "Q-only might already be enough." — M3 and M5 include `ocq_qbias_b-0.03` and `ocq_ladapt_k0_q-0.03` as separate rows so the K-early contribution is isolated.
- "The catalog ontology might be specific to MetaTool." — τ²-bench retail and airline use τ²-native tool catalogs and domain-specific `B_ont`s built via `build_tau2_ontology.py`.
- "PCA basis should also work if this is about rank." — M6 falsifies this.
- "τ² is multi-turn, set metrics are too loose." — Paper states the scope explicitly: τ² in this paper is the first-turn action-set task; full multi-turn success is a follow-up.

## 11. Out of scope for this paper

- PCA or KV-cache compression as a paper claim.
- Full τ²-bench multi-turn success; we do first-turn action-set selection only.
- Recursive cache or architecture modifications.
- Claims that the perturbation bound predicts accuracy.
- Any AdaSEKA proxy path in paper-facing Subtask4 tables (blocked in `eval_metatool_subtask4.py`).

## 12. Deliverables

- Six result JSON directories under `reports/`:
  `tau2_retail_size_sweep_2026_04_17/`, `tau2_airline_2026_04_17/`, `subtask4_cross_family_2026_04_17/`, `stepwise_2026_04_17/`, `basis_controls_2026_04_17/`, `schedule_sweep_2026_04_17/`.
- One reproduction bundle: `reports/canonical_seka_A100_2026_04_16/` containing the ES=0.952 A100 replica.
- One frozen paper draft: `paper/neurips2026_steering_v2/` with every table cell either filled with a result-JSON citation or explicitly marked as a placeholder with expected direction.

## 13. Final acceptance test for this plan

The plan passes when, reading only §4 tables and §7 protocol lock, a reviewer could reconstruct exactly which command produces every number in the paper. If any table cell cannot be traced to a command and a JSON path, the plan fails.
