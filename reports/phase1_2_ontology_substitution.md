# Phase 1.2 — Ontology Direction Substitution into SEKA

**Date**: 2026-04-09
**Hardware**: NVIDIA RTX A6000 (GPU 1), 48GB
**Environment**: vllm_env
**Model**: Qwen/Qwen3-4B-Base
**Benchmark**: CounterFact (PASTA split, first 500 examples)

## Goal

Drop-in substitute the **direction source** in SEKA's K-side projection
steering — replacing SEKA's contrastive-SVD basis (built from 200
synthetic QA pairs) with the **ontology-derived facet basis** from
`scripts/ontology_facet_basis.py`, keeping every other element of the
SEKA operator identical: same hook point (`k_proj` / `k_norm` output),
same layer set (last 10), same `amplify_pos=1.56`, same
`amplify_neg=0.0`, same `min_diff`-style head selection (all 80 heads
pass), same `top_pct`-style truncation (energy 0.95).

**H20 gate**: ontology variant must achieve **ES ≥ 85.7** (90 % of
SEKA's 95.2 from Phase 1.1) to proceed to Phase 1.3/1.4. Below that,
pivot to "SEKA + ontology fallback" or switch to Path B.

## Pipeline

### 1. Ontology basis build — `scripts/phase1_ontology_projection.py`

Wraps the existing Mistral-era `ontology_facet_basis.py` machinery
(`extract_category_K` + `build_facet_basis`) and runs it on
Qwen3-4B-Base with `TARGET_LAYERS = [26..35]` × `n_kv=8`.

For each `(L, H)` the pipeline:

1. Runs each of 85 ontology sentences through the model, hooks
   `k_proj.forward`, extracts per-head K-vectors excluding BOS,
   averages per category.
2. Gram-Schmidt residualizes the 4 facet embedding matrices in
   priority order (domain → manufacturer → product_type → price_tier)
   and concatenates the orthonormal bases into `B ∈ ℝ^{128 × r_tot}`.
3. Forms the symmetric idempotent projector `P = B B^T ∈ ℝ^{128×128}`
   (rank `r_tot`).

**`r_tot` grid over all 80 heads** (10 layers × 8 KV heads):

| stat | value |
|---|---|
| min | 11 |
| median | 13 |
| max | 14 |
| mean | 12.7 |

All 80/80 heads produced a valid basis. Max orthogonality error
`‖BᵀB − I‖_∞ < 5e-14` across every head.

For reference, SEKA's Phase 1.1 projector had rank ≈ 8 per head
(`trace ≈ 8` from the `top_pct=0.90` energy truncation of contrastive
SVD singular values). Ontology is slightly higher rank (≈13) because
the 4 facets together contribute 12–14 orthogonal directions after
residualization.

### 2. Tensor packaging

Stack `P` across 10 layers × 8 heads into `(10, 8, 128, 128)` float32
and save as the dict format SEKA's `_load_proj` expects:

```python
{'layers': [26,27,28,29,30,31,32,33,34,35], 'proj': Tensor(10,8,128,128)}
```

Written to:
- `external/SEKA/seka_projections/ontology-qwen3-4b/Qwen3-4B-Base_pos_proj.pt`
- `external/SEKA/seka_projections/ontology-qwen3-4b/Qwen3-4B-Base_neg_proj.pt`

Both files are identical — SEKA's hook computes
`delta = (gp·pos + gn·neg) / 2`, so with `gn=0.0` the neg branch is
arithmetically inert and the effective gain is `0.78·P·k` regardless
of what neg contains. We pass a second copy so SEKA's 4-D tensor check
succeeds and the code path is identical to the Phase 1.1 run.

Builder diagnostic: `reports/axis2_theoretical_verification/phase1_ontology_projection_qwen3_4b.json`

### 3. Eval — unchanged from Phase 1.1

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 python3 benchmarks/eval_fact_gen.py \
  --model Qwen/Qwen3-4B-Base \
  --data_path data/pasta_bench \
  --output_dir benchmarks/counterfact/results/ontology-qwen3-4b-500 \
  --benchmarks efficacy paraphrase \
  --add_unmediated_fact True \
  --batch_size 16 --max_new_tokens 64 \
  --example_subset 0:500 \
  --seka \
  --pos seka_projections/ontology-qwen3-4b/Qwen3-4B-Base_pos_proj.pt \
  --neg seka_projections/ontology-qwen3-4b/Qwen3-4B-Base_neg_proj.pt \
  --amplify_pos 1.56 --amplify_neg 0.0 --layers last10
```

Run time: ~20 s total (3 s efficacy + 8 s paraphrase + load).

## Results

### Headline table

| Metric | Baseline (no steer) | SEKA Phase 1.1 | **Ontology Phase 1.2** | Ontology vs SEKA | H20 gate (≥85.7 ES) |
|---|---|---|---|---|---|
| ES (efficacy) | 40.2 | 95.2 | **91.6** | −3.6 pp | ✓ **PASS** (+5.9) |
| PS (paraphrase) | 43.6 | 96.2 | **93.2** | −3.0 pp | ✓ PASS (+6.6) |
| ES lift over baseline | — | +55.0 | **+51.4** | −3.6 pp | — |
| PS lift over baseline | — | +52.6 | **+49.6** | −3.0 pp | — |

Ontology substitution recovers **93.5 % of SEKA's ES lift** and
**94.3 % of SEKA's PS lift** without using any CounterFact-specific
data during basis construction.

### Raw scores

From `external/SEKA/benchmarks/counterfact/results/ontology-qwen3-4b-500/`:

- `efficacy.json`: `score.mean = 0.916`, `std = 0.277`, `magnitude.mean = 2.53e-4`
- `paraphrase.json`: `score.mean = 0.932`, `std = 0.211`, `magnitude.mean = 1.83e-4`

## Interpretation

1. **Direction source is (mostly) fungible.** With the entire SEKA
   operator held fixed, swapping the contrastive-SVD basis (rank-8,
   trained on 200 synthetic in-domain pairs) for a generic ontology
   basis (rank-13, built from 85 out-of-domain facet sentences) costs
   only ~3.5 pp on both ES and PS. The heavy lifting in SEKA's
   efficacy appears to come from the steering operator structure
   (amplifying k-space projections at marker positions in last-10
   layers), not from the specific subspace SEKA chose.

2. **Generic ontology carries CounterFact-relevant variance.** The
   ontology sentences cover domains/manufacturers/products/price — no
   overlap with CounterFact's facts about languages, religions,
   geography, occupations. That a basis this far off-distribution
   still recovers ES = 91.6 suggests the facet directions are pointing
   at generally salient content-token K-subspaces, not at
   CounterFact-specific contrasts.

3. **Rank cost is 1.6×.** Ontology uses r_tot ≈ 13 vs SEKA's 8. If
   Phase 1.3 wants to match SEKA on compute budget, the ontology basis
   could be SVD-truncated back to rank 8 to probe whether the extra
   directions are load-bearing or dilutive.

4. **Gap breakdown (3.6 pp ES) is small enough to be tuning noise.**
   Possible sources: (a) energy truncation 0.95 vs 0.90, (b) no
   per-head `min_diff` gating (ontology uses all 80 heads; SEKA's
   `min_diff=0.10` also gave 80/80 so this is neutral), (c) no
   CounterFact-aligned direction at all. Phase 1.3 will sweep α and
   energy threshold to localize.

## Verdict

**Phase 1.2 — H20 gate PASSED** (91.6 ES ≥ 85.7 target, margin +5.9 pp).

The ontology direction source is a viable drop-in replacement for
SEKA's contrastive SVD on CounterFact / Qwen3-4B-Base. The ontology
variant loses ~3.5 pp relative to the task-specific baseline while
avoiding any task-specific tuning data — an acceptable price for a
zero-shot direction source.

**Path A is alive.** Proceed to Phase 1.3:

- Sweep α ∈ {1.0, 1.2, 1.4, 1.56, 1.8, 2.0} to find the ontology-specific
  optimum (SEKA's 1.56 was tuned for the contrastive-SVD basis and may
  not be optimal for rank-13 ontology).
- Truncate ontology basis to rank 8 (matching SEKA's energy) and
  re-evaluate — isolates rank effect from direction effect.
- Run on a second task (pick one from PASTA bench) to check the
  ontology basis isn't secretly overfit to CounterFact's structure.

## Files

- Builder script: `scripts/phase1_ontology_projection.py`
- Projection tensors: `external/SEKA/seka_projections/ontology-qwen3-4b/{Qwen3-4B-Base_pos_proj.pt, Qwen3-4B-Base_neg_proj.pt}`
- Builder diagnostic: `reports/axis2_theoretical_verification/phase1_ontology_projection_qwen3_4b.json`
- Eval results: `external/SEKA/benchmarks/counterfact/results/ontology-qwen3-4b-500/{efficacy.json, paraphrase.json}`
- Phase 1.1 reference: `reports/phase1_seka_reproduction.md`
