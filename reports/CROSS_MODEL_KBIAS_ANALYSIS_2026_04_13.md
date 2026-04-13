# Cross-Model K-Bias Steering Analysis

**Date**: 2026-04-13
**Author**: mais (with Claude Code)
**Status**: Experiment complete, analysis ongoing

---

## 1. Executive Summary

K-bias ontology steering (`K' = K + α·B·B^T·K`) was tested on 3 models using MetaTool Subtask1 (995 queries, tool selection with 10 candidates). Results reveal a dramatic model-dependent response: Qwen and Llama achieve +10pp lift, while Mistral catastrophically fails at -32pp. Root cause analysis identifies **min-truncation in B_ont construction** as the primary aggravating factor and **weak baseline prompt-following** as the secondary cause.

## 2. Baseline Cross-Model Results (original B_ont, α=0.3)

| Model | Mode | n_kv | no_steer | ocq_bias α=0.3 | Δ | no_match rate |
|-------|------|------|----------|----------------|---|---------------|
| Qwen2.5-7B | C | 4 | 75.58% | 86.73% | **+11.16pp** | 4.6% → 4.2% |
| Llama-3.1-8B | A | 8 | 80.60% | 90.85% | **+10.25pp** | 17.1% → 7.0% |
| Mistral-7B-v0.3 | A | 8 | 61.01% | 29.15% | **-31.86pp** | 36.6% → 66.6% |

**Key observation**: Both Llama and Mistral are Mode A (position-sink attention), yet Llama succeeds (+10.25pp) while Mistral fails catastrophically (-31.86pp). **Mode A is NOT the root cause.**

### 2.1 Sink-Skip Experiment (skip position 0 from K-bias)

| Model | ocq_bias α=0.3 | sinkskip α=0.3 | Δ(sink fix) |
|-------|----------------|-----------------|-------------|
| Llama-3.1-8B | 90.85% (+10.25pp) | 86.73% (+6.13pp) | -4.12pp (worse) |
| Mistral-7B | 29.15% (-31.86pp) | 19.40% (-41.61pp) | -9.75pp (worse) |

**Conclusion**: Skipping position-0 (BOS/sink) **hurts both models**. Sink position is NOT the problem — it carries useful K-bias signal.

## 3. Root Cause: min-Truncation in B_ont Construction

### 3.1 The Truncation Mechanism

B_ont is built per (layer, head) via Gram-Schmidt orthogonalization. Each head produces a variable number of ontology directions (rank). The final tensor is truncated to `r_ont = min(all heads)` for uniform shape.

### 3.2 Per-Model Rank Distribution

| Statistic | Qwen2.5-7B | Llama-3.1-8B | Mistral-7B |
|-----------|-----------|-------------|-----------|
| n_kv (KV heads) | 4 | 8 | 8 |
| Total (L,H) pairs | 112 | 256 | 256 |
| **Median rank** | **28** | **29** | **28** |
| **min rank (= r_ont)** | **24** | **19** | **13** |
| Bottleneck head | L1_H3 | L0_H7 | **L0_H2, L0_H4** |
| Truncation loss | 4 cols | 10 cols | **15 cols** |

**All three models have median rank ≈ 28.** The difference is entirely in the **worst-case bottleneck head**. Mistral L0_H2 has domain facet rank = 3 (vs Llama L0_H7 domain rank = 8), forcing the entire 256-head tensor down to r_ont = 13.

### 3.3 Why Mistral L0 is Low-Rank

Layer 0 K-space is the direct output of `W_K[0]` applied to token embeddings. Mistral-7B-v0.3's `W_K[0]` compresses ontology category K-vectors into a lower-dimensional subspace than Llama's. With 8 KV heads (vs Qwen's 4), the probability of having one pathologically low-rank head increases.

## 4. Ablation Experiments

### 4.1 Experiment 1: Mistral B_ont Rebuilt Without L0 (skipL0)

Excluded Layer 0 from B_ont construction → min rank rises from 13 to **21**.

| Model | B_ont | r_ont | ocq_bias α=0.3 | Δ vs no_steer |
|-------|-------|-------|----------------|---------------|
| Mistral | original | 13 | 29.15% | -31.86pp |
| Mistral | **skipL0** | **21** | 52.06% | **-8.94pp** |

**23pp improvement** from simply removing L0 bottleneck. L0 removal accounts for ~72% of the original damage.

### 4.2 Experiment 2: Llama B_ont Force-Truncated to r=13

Applied Mistral's truncation level (r=13) to Llama's B_ont to test if truncation alone causes failure.

| Model | B_ont | r_ont | ocq_bias α=0.3 | Δ vs no_steer |
|-------|-------|-------|----------------|---------------|
| Llama | original | 19 | 90.85% | +10.25pp |
| Llama | **trunc r=13** | **13** | 86.83% | **+6.23pp** |

**Llama remains positive at r=13** (+6.23pp vs Mistral's -31.86pp at the same r=13). **Truncation is not the sole cause — Mistral has an additional model-specific vulnerability.**

### 4.3 Experiment 3: Mistral Adaptive B_ont (pad-to-max, r=33)

Each head retains its full Gram-Schmidt rank; low-rank heads (L0) have zero-padded columns that contribute nothing in the hook.

| Model | B_ont | r_ont | ocq_bias α=0.3 | Δ vs no_steer |
|-------|-------|-------|----------------|---------------|
| Mistral | original | 13 | 29.15% | -31.86pp |
| Mistral | skipL0 | 21 | 52.06% | -8.94pp |
| Mistral | **adaptive** | **33** | 45.13% | **-15.88pp** |

**Adaptive is WORSE than skipL0** (-15.88pp vs -8.94pp). L0 heads with their natural low-rank (13) basis still inject harmful bias — they should be excluded entirely, not padded.

## 5. Diagnosis Summary

### 5.1 Factor Decomposition

| Factor | Contribution | Evidence |
|--------|-------------|----------|
| **L0 bottleneck min-truncation** | ~72% of damage | skipL0 recovers 23pp (from -32 to -9) |
| **Mistral base model weakness** | ~28% of damage | Llama at r=13 gives +6.23pp; Mistral at r=21 gives -8.94pp |
| Attention mode (A vs C) | 0% | Llama (Mode A) succeeds at +10.25pp |
| Sink position | 0% (negative) | Sinkskip worsens both models |

### 5.2 Mistral Base Model Weakness

| Metric | Llama-3.1-8B | Mistral-7B-v0.3 |
|--------|-------------|-----------------|
| no_steer accuracy | 80.60% | 61.01% |
| no_steer no_match rate | 17.1% | **36.6%** |

Mistral's 36.6% no_match (model fails to output any tool name) indicates weak prompt-following on the MetaTool format. K-bias amplifies attention in the ontology subspace, but if the model is already uncertain about what to generate, the amplification pushes it further off-distribution.

## 6. Recommendations

### 6.1 For Build Pipeline (Immediate Fix)

1. **Use `--pad-to-max` + `--target-layers` excluding L0** (or any layer with effective rank < threshold)
2. Do NOT use min-truncation for cross-model B_ont — it creates a "weakest link" bottleneck
3. Per-head adaptive rank is the correct approach, but pathological heads (rank << median) should be zeroed out

### 6.2 For Paper

- **Primary evidence**: Qwen2.5-7B (+11.16pp) and Llama-3.1-8B (+10.25pp) — both significant lift at α=0.3
- **Mistral**: Report as negative result with diagnosed root cause (min-truncation + weak baseline)
- **Cross-model claim**: "K-bias steering generalizes across Qwen (Mode C) and Llama (Mode A) architectures when B_ont rank is sufficient and the base model has adequate prompt-following capability"

### 6.3 Next Steps

1. **H2 validation**: Test Mistral-7B-Instruct-v0.3 to confirm baseline fragility hypothesis
2. **Qwen + Llama α sweep**: Find optimal α per model (current α=0.3 may not be optimal for Llama)
3. **τ²-bench multi-turn**: Extend to multi-turn tool use benchmarks with the validated Qwen/Llama pair

## 7. Files Produced

| File | Description |
|------|------------|
| `reports/axis2_theoretical_verification/mistral_sinkskip_full995.json` | Mistral × {no_steer, bias, sinkskip} |
| `reports/axis2_theoretical_verification/llama31_sinkskip_full995.json` | Llama × {no_steer, bias, sinkskip} |
| `reports/axis2_theoretical_verification/mistral_skipL0_full995.json` | Mistral skipL0 B_ont eval |
| `reports/axis2_theoretical_verification/llama31_r13_truncation_full995.json` | Llama forced r=13 eval |
| `reports/axis2_theoretical_verification/mistral_adaptive_full995.json` | Mistral adaptive r=33 eval |
| `reports/axis2_theoretical_verification/build_mistral_b_ont_skipL0.json` | skipL0 build diagnostic |
| `reports/axis2_theoretical_verification/build_mistral_b_ont_adaptive.json` | adaptive build diagnostic |
| `external/SEKA/seka_projections/ontology-mistral-7b-v03-metatool-skipL0/B_ont.pt` | skipL0 basis |
| `external/SEKA/seka_projections/ontology-mistral-7b-v03-metatool-adaptive/B_ont.pt` | adaptive basis |
| `external/SEKA/seka_projections/ontology-llama31-8b-metatool-r13/B_ont.pt` | Llama truncated basis |

## 8. Reproduction Commands

```bash
# Sink-skip experiments (original B_ont)
python scripts/ocq/eval_metatool_subtask1.py \
  --model mistralai/Mistral-7B-v0.3 --device cuda:0 \
  --methods no_steer ocq_bias_a0.3 ocq_bias_a0.3_sinkskip \
  --b-ont external/SEKA/seka_projections/ontology-mistral-7b-v03-metatool/B_ont.pt \
  --skip-sink-tokens 1 --max-samples 0

# Mistral skipL0 rebuild
python scripts/ocq/build_qwen_metatool_b_ont.py \
  --model mistralai/Mistral-7B-v0.3 --device cuda:0 \
  --target-layers "1,2,...,31" \
  --out external/SEKA/seka_projections/ontology-mistral-7b-v03-metatool-skipL0/B_ont.pt

# Mistral adaptive rebuild
python scripts/ocq/build_qwen_metatool_b_ont.py \
  --model mistralai/Mistral-7B-v0.3 --device cuda:0 \
  --target-layers all --pad-to-max \
  --out external/SEKA/seka_projections/ontology-mistral-7b-v03-metatool-adaptive/B_ont.pt
```
