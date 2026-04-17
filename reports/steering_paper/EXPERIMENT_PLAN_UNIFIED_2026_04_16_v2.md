# Unified Steering-Paper Experiment Plan (v2)

Date: 2026-04-16
Supersedes: `EXPERIMENT_PLAN_UNIFIED_2026_04_16_v1.md` (deprecated)
Central claim now anchors on **beating SEKA on multi-tool tool-use**.
Paper target: `paper/neurips2026_steering_ko` (canonical) mirrored to `paper/neurips2026_steering_v2`.

## 0. Central claim (one-liner)

> SEKA and AdaSEKA are state-of-the-art for single-concept editing but **structurally mismatched** for multi-tool tool use because stationary key-side amplification cannot encode emitted-tool state (Theorem 4.1). We present two history-free operator families — **Q-coverage** and **Layer-Adaptive K+Q** — that beat SEKA and AdaSEKA on MetaTool Subtask4 and τ²-bench retail across Qwen and Llama, with the gap quantifiable via a stepwise coverage decomposition that directly confirms Theorem 4.1's repeat-bias prediction.

Three-tier instantiation:

- **T1 (performance).** On MetaTool Subtask4, both ours operators strictly beat SEKA/AdaSEKA at every matched amp, across Qwen and Llama.
- **T2 (mechanism).** The gap concentrates in `second_distinct_hit_rate` and inversely in `repeated_first_tool_rate`, exactly as Theorem 4.1 / Corollary 4.2 predict.
- **T3 (robustness).** The gap persists across Qwen sizes {1.5B, 3B, 7B, 14B}, and the catalog ontology basis is decisive (feature-shuffled, random, and PCA-of-K alternatives fail).

## 1. Operator family and comparison axis

The paper compares six methods under one protocol:

| Family | Tag | Description |
|---|---|---|
| Baseline | `no_steer` | pure `model.generate` |
| SEKA (matched) | `real_seka_amp{0.5,1.0,2.0}` | canonical SEKA via `external/SEKA/src/model/seka_llm.py`, last-10 layers, marker-masked user tokens |
| AdaSEKA (matched) | `canonical_adaseka_amp{1.0,3.0}` | per-facet expert selection, `scripts/diagnostics_2026_04_16/eval_subtask4_with_adaseka.py` |
| Stationary K (ours) | `ocq_bias_a0.3` | `K' = K + α·B_ont·B_ontᵀ·K`, all layers, all tokens — used as null-K baseline |
| **Q-coverage (ours)** | `ocq_qbias_b-0.03` | `Q' = Q + β·B_ont·B_ontᵀ·Q`, all layers |
| **Layer-Adaptive K+Q (ours)** | `ocq_ladapt_k0.05_q-0.03` | K on first L/4 layers + Q everywhere |

## 2. Main-body experiments (E1–E6)

Every experiment ties to a theorem/corollary and has an explicit falsification rule.

### E1 — MetaTool Subtask4 head-to-head
- **Intent**: T1 on multi-tool with k=2.
- **Hypothesis** (Thm 4.1, Cor 4.2): SEKA/AdaSEKA < `no_steer` by a large margin; ours > `no_steer`.
- **Verification**: N=497 full, Qwen2.5-7B + Llama-3.1-8B, all six methods.
- **Success**: both ours strictly > every SEKA/AdaSEKA row on both models.
- **Kill rule**: if Q-coverage ≤ SEKA on either model at matched protocol, central claim fails.
- **Budget**: 12 methods × 2 models × 497 samples × ~0.5s/sample ≈ 1.5 GPU-h.

### E2 — τ²-bench retail cross-benchmark
- **Intent**: T1 on long-sequence multi-turn.
- **Hypothesis** (Thm 4.3, Cor 4.6): Q-coverage > SEKA; Layer-Adaptive may under-perform Q-only in k≥4 regime but never worse than SEKA.
- **Verification**: N=114 full, Qwen 7B primary, Llama 8B secondary.
- **Success**: Q-coverage > every SEKA row; Layer-Adaptive > SEKA even if ≤ Q-only.
- **Kill rule**: if both ours ≤ SEKA on τ²-retail, central claim fails.
- **Budget**: ~1 GPU-h.

### E3 — Stepwise mechanism (more important than aggregate F1)
- **Intent**: T2 mechanism — quantify the "why" of winning.
- **Hypothesis** (Cor 4.2): SEKA raises `repeated_first_tool_rate`; ours lowers it and raises `second_distinct_hit_rate`.
- **Verification**: Reuse Subtask4 per-sample JSONs; summarize stepwise block across methods.
- **Success**: Layer-Adaptive K+Q strictly dominates on `second_distinct_hit_rate`.
- **Kill rule**: if gain is only in `first_tool_hit_rate`, discussion loses sequential-coverage framing.
- **Budget**: post-processing only, 0 GPU-h.

### E4 — Basis ablation (decisive)
- **Intent**: T3 — show catalog ontology is necessary, not merely "any low-rank basis".
- **Hypothesis**: PCA-of-K basis fails (or at best is neutral) for Q-coverage and catastrophic for K-only, matching random/shuffled controls.
- **Verification**: Four bases × two operators on Qwen Subtask4 N=497.
- **Success**: PCA row collapses for K-only; is ≤ shuffled for Q-coverage.
- **Kill rule**: if PCA matches real for Q-coverage, E4 reframes to "any data-driven low-rank subspace suffices; catalog ontology is optimal".
- **Budget**: `build_pca_baseline_basis.py` 20 min + 2 eval runs ≈ 45 min.

### E5 — Qwen model-size sweep
- **Intent**: T3 — gap is not a single-size artifact.
- **Hypothesis** (Thm 4.1 is size-independent): gap preserved at 1.5B/3B/14B.
- **Verification**: Four Qwen sizes × three methods (`no_steer`, Q-coverage, Layer-Adaptive, best SEKA) × Subtask4 N=497 + τ²-retail N=60.
- **Success**: Layer-Adaptive or Q-coverage > best SEKA at all four sizes.
- **Kill rule**: if gap shrinks to zero at 14B, flag as "architecture-sensitive" in limitations.
- **Budget**: 4 sizes × 4 methods × 497 samples ≈ 6 GPU-h (7B already done).

### E6 — Regime split ΔF1 vs k
- **Intent**: T3 — turn weakness into design principle.
- **Hypothesis** (Cor 4.6): Layer-Adaptive crosses Q-only near k=3. SEKA stays <0 everywhere.
- **Verification**: bin per-sample results by ground-truth k on Subtask1 (k=1), Subtask4 (k=2), τ²-retail (k∈[0,13]).
- **Success**: monotone crossing between Layer-Adaptive and Q-only near k∈[3,4]; SEKA negative at all k.
- **Kill rule**: if Layer-Adaptive loses to Q-only even at k=2, Proposition 4.5's empirical premise E1 is falsified.
- **Budget**: post-processing + one Subtask1 re-run (~30 min).

## 3. Appendix / supporting experiments

- **A1 Canonical SEKA CounterFact A100 reproduction** (gate for all `[REPRO]` rows in the manifest).
- **A2 α/β/layer-boundary sweep** — ablation of `(α, β, τ)` on Qwen Subtask4.
- **A3 ε_q stopping criterion** — locked AUROC=0.976, deployment diagnostic.
- **A4 Null-control full table** — real/shuffled/random 4×2 matrix beyond the paper main.
- **A5 Subtask1 parity** — locked K-bias Llama +15.08pp, Qwen +11.16pp as evidence that ours is multi-tool-specific design.
- **A6 LoRA cross-reference** — compare to LoRA finetuning on same model (`reports/lora_v4_2026_04_15/`).

## 4. Explicitly excluded

- Spider, ChartQA (user directive).
- Full τ² multi-turn success metric (scope).
- KV-cache compression (separate track).
- Recursive / architecture-modifying operators.
- AdaSEKA proxy path in paper-facing Subtask4 (blocked at code level in `eval_metatool_subtask4.run_method`).

## 5. Timeline (post-approval)

| Day | Work |
|---|---|
| 1 | E5 launch on 1.5B/3B (2 GPU-h); E4 PCA basis build + run; Llama Layer-Adaptive fill for E1. |
| 2 | E1 AdaSEKA canonical rows; E2 τ²-retail N=114 full; E6 regime-split aggregation. |
| 3 | E5 14B run; E3 stepwise aggregation; figure regeneration from real JSONs. |
| 4 | SEKA canonical A100 reproduction attempt; paper table fill; final PDF compile. |

Total: ≈25 GPU-h on one A100 node, sequentialized in <4 calendar days.

## 6. Falsification summary

A single-page checklist for the paper is in `paper/neurips2026_steering_ko/sections/09_appendices.tex` Appendix B. If any of the following holds, the central claim is downgraded:

1. E1 Q-coverage ≤ SEKA on either model.
2. E2 both ours ≤ SEKA on τ²-retail.
3. E3 gain on `first_tool_hit` only (no second-tool improvement).
4. E4 PCA basis reproduces real basis for both Q and K.
5. E5 gap vanishes at any Qwen size.

None of the currently locked evidence falsifies the claim. E4 PCA row and E1 Llama Layer-Adaptive are the two most informative missing measurements.
