# Unified Experiment Plan v3 — Rotational Steering for Multi-Tool Selection

**Date**: 2026-04-17
**Supersedes**: `EXPERIMENT_PLAN_UNIFIED_2026_04_16_v2.md`
**Paper target**: `paper/neurips2026_steering_ko` (canonical), `paper/neurips2026_steering_v2` (mirror)

## 0. Purpose of this document

This document enumerates \emph{only the experiments that must be run} before submission, each stated in the intent / hypothesis / verification form. Every experiment here ties directly to a table cell or figure in the current paper draft; experiments that are already locked on develop do not appear in this list.

## 1. Central claims the plan must support

- **C1. Tool-selection performance.** Rotational steering (Q-coverage and layer-adaptive K+Q) produces consistent gains over stationary activation steering (SEKA, AdaSEKA) on multi-tool benchmarks on Qwen2.5-7B-Instruct.
- **C2. Efficiency.** The method requires no weight training and no prompt expansion, works purely as an inference-time hook, and its runtime/memory overhead is small and measurable.

**Model scope.** All claims are made on Qwen2.5-7B-Instruct as the primary model. Size-sweep (appendix) covers Qwen2.5-Instruct 1.5B/3B/7B/14B. Llama experiments are explicitly excluded from this plan cycle.

Every experiment below is justified by which of C1 or C2 it supports, and which paper table or figure cell it fills.

## 2. Blocking gate: canonical SEKA reproduction

### G1. Canonical SEKA CounterFact on A100 (blocking)

- **Intent.** Before the paper can make any head-to-head statement vs SEKA, an external implementation must reproduce the canonical CounterFact efficacy score in \emph{our} environment. Otherwise every SEKA row is ``reference citation only,'' and the paper's comparison narrative collapses to structural reasoning without empirical head-to-head.
- **Hypothesis.** On A100 with `torch==2.3.0+cu121`, `transformers==4.51.3`, Qwen3-4B-Base, the pre-built `P_{\mathrm{pos}}`, `amplify_pos=1.0, amplify_neg=0.8, layers=last10, N=500`, the efficacy score lands in `0.95 ± 0.02`.
- **Verification.** Run `external/SEKA/benchmarks/eval_fact_gen.py` with the exact arguments of `reports/COWORKER_SEKA_REPRO_GUIDE_2026_04_16.md` §1.4. Success criterion is `efficacy_metrics.json::score.mean ∈ [0.93, 0.97]`.
- **Decision rule.** On pass, all `[REPRO]`-gated rows below become eligible to fill. On fail, the paper's SEKA/AdaSEKA rows are rendered as ``canonical SEKA (reference)'' and the comparison narrative is downgraded to ``diagnosis of structural limit from SEKA's own form.''
- **Cost estimate.** About 10 minutes once A100 is available.

## 3. Claim C1 (tool-selection performance): must-run experiments

### E1. Canonical SEKA and AdaSEKA rows on MetaTool Subtask4 (Qwen)

- **Intent.** Fill the canonical SEKA and AdaSEKA cells of Table 1 for Qwen2.5-7B-Instruct. These cells currently hold ``placeholder'' markers with expected directions.
- **Hypothesis.** Corollary~\ref{cor:seka-repeat} predicts `repeated_first_tool_rate` rises and therefore F1 falls under SEKA on multi-tool. Expected direction: `Δ F1 ≪ 0`.
- **Verification.** N=497, greedy decoding. Run `scripts/ocq/eval_subtask4_with_real_seka.py` with `amplify={0.5, 1.0, 2.0}` for canonical SEKA and `scripts/diagnostics_2026_04_16/eval_subtask4_with_adaseka.py` with `amplify={1.0, 3.0}` for canonical AdaSEKA. Report macro F1 and Exact.
- **Decision rule.** If either canonical SEKA or AdaSEKA produces `Δ F1 ≥ 0`, the stationary-K limit argument needs revision. Predicted `Δ F1 < 0` directly confirms the argument.
- **Blocker.** G1.
- **Cost estimate.** ~2 GPU-hours.

### E2. τ²-bench retail N=114 full on Qwen

- **Intent.** Fill the full-N row of Table 2. Currently N=30 smoke is locked with Q-coverage `+3.74pp` and layer-adaptive `-3.73pp` — the regime-split signature predicted by Corollary~\ref{cor:regime}.
- **Hypothesis.** At N=114, Q-coverage `ΔF1 ≥ +2pp` (signal $>$ smoke noise), layer-adaptive `ΔF1 ∈ [-5, 0]pp` as the long-sequence regime asserts, and SEKA/AdaSEKA `ΔF1 ≤ -2pp`.
- **Verification.** `scripts/ocq/eval_tau2_bench.py --model Qwen/Qwen2.5-7B-Instruct --domain retail --max-samples 114 --methods no_steer ocq_qbias_b-0.03 ocq_ladapt_k0.05_q-0.03 real_seka_amp1.0 canonical_adaseka_amp1.0`.
- **Decision rule.** Confirmation of the smoke pattern at N=114 establishes the regime-split empirically. Inversion would refute Corollary~\ref{cor:regime}.
- **Blocker.** G1 for SEKA/AdaSEKA rows only.
- **Cost estimate.** ~1.5 GPU-hours.

### E3. PCA-of-K basis row on Subtask4

- **Intent.** Fill the PCA-of-K cell of Table 3 (basis $\times$ operator). This is the \emph{decisive} ablation that separates ``gain from the basis'' from ``gain from the operator.''
- **Hypothesis.** For stationary K, PCA basis collapses toward 0 as random/shuffled bases do, confirming that stationary K depends entirely on semantic subspace alignment. For Q-coverage, PCA basis produces an intermediate or neutral result — between real (`0.7471`) and random (`0.7068`).
- **Verification.**
  1. Run `scripts/ocq/build_pca_baseline_basis.py --model Qwen/Qwen2.5-7B-Instruct --dataset .../Task2-Subtask4.json --n-calib 256 --rank 24 --out reports/pca_baseline_bases/qwen25_7b_st4_r24.pt`.
  2. Run `scripts/ocq/eval_metatool_subtask4.py --b-ont <pca>.pt --methods ocq_qbias_b-0.1 ocq_bias_a0.3 --max-samples 497`.
- **Decision rule.** Both PCA cells `≤` the real cells and the K cell near zero confirms ``catalog ontology is itself part of the operator.'' PCA indistinguishable from real for Q-coverage narrows the claim to ``low-rank structure suffices but the catalog is optimal.''
- **Blocker.** None.
- **Cost estimate.** PCA build ~20 minutes + evaluation ~1 hour.

### E4. Position-level stepwise aggregation

- **Intent.** Fill Figure 3 and Table 4 with real numbers. These are currently arrow-direction placeholders.
- **Hypothesis.** The `repeated_first_tool_rate` is higher under stationary K than under \texttt{no\_steer}, and lower under Q-coverage and layer-adaptive. Simultaneously `second_distinct_hit_rate` is higher for our operators.
- **Verification.** No new GPU runs. The stepwise block is already emitted into the per-sample JSONs of existing Qwen Subtask4 runs and E1 outputs. A post-processing script `scripts/ocq/summarize_stepwise.py` (to be added) aggregates these into the paper's Figure 3 / Table 4 numbers.
- **Decision rule.** Joint directionality confirms the core mechanism claim. Any method with the predicted sign reversed is reported truthfully; absence of the predicted joint directionality downgrades the mechanism claim.
- **Blocker.** E1 must produce result JSONs.
- **Cost estimate.** 0 GPU; ~30 minutes of post-processing work.

## 4. Claim C2 (efficiency): must-run experiments

### E5. Inference-latency measurement

- **Intent.** Convert the ``to measure'' cell of Table 6 into a number. The paper currently predicts overhead under 1% of total inference time.
- **Hypothesis.** Hook installation adds $O(Tdr)$ matmul per attention call, so per-query wall-clock overhead is small relative to `no_steer`.
- **Verification.** On Qwen2.5-7B, run the Subtask4 N=497 evaluation four times: `no_steer`, Q-coverage `b-0.03`, layer-adaptive `k0.05_q-0.03`, canonical SEKA `amp=1.0` (conditional on G1). Log `per_query_ms` and `total_runtime_s` in the result JSON. Report mean, median, and 95th percentile of per-query latency.
- **Decision rule.** Overhead $\le$ 5% of `no_steer` runtime confirms the efficiency claim. Overhead $\ge$ 20% would require softening the ``cheap'' framing.
- **Blocker.** None for Q-coverage and layer-adaptive; G1 for the SEKA row.
- **Cost estimate.** ~2 GPU-hours.

### E6. Memory-footprint verification

- **Intent.** Confirm that the hook adds only the theoretical basis storage (~2.6 MB on Qwen 7B) to GPU memory, not extra activations or gradients.
- **Hypothesis.** Peak GPU memory with the hook installed minus peak without hook equals the size of `B_ont` within measurement noise.
- **Verification.** Instrument `scripts/ocq/eval_metatool_subtask4.py` to log `torch.cuda.max_memory_allocated()` before hook install, after 10 queries, and at shutdown. Compare the delta with the theoretical size of `B_ont`.
- **Decision rule.** Match within $\pm 20\%$ confirms the theoretical footprint. Large excess would indicate unintended intermediate caching that the paper needs to disclose.
- **Blocker.** None.
- **Cost estimate.** Piggybacks on E6; marginal cost 0.

## 5. Supporting ablations (appendix, optional)

### E7. Qwen model-size sweep on Subtask4 (appendix Table 5)

- **Intent.** Check that the sign persists across Qwen 1.5B, 3B, 14B sizes. Theorems 1--3 are size-independent structural/analytical results, so failure here would indicate an implementation or environmental factor, not a theoretical one.
- **Hypothesis.** Sign preserved at all sizes.
- **Verification.** Adapt `scripts/ocq/run_tau2_size_sweep.sh` to Subtask4. Run `no_steer`, SEKA `amp=1.0`, Q-coverage, layer-adaptive across the four sizes.
- **Blocker.** None for ours; G1 for SEKA.
- **Cost estimate.** ~4 GPU-hours.

### E8. Phase 2.5 layer boundary sweep (coworker track, appendix Table 8)

- **Intent.** Check whether the $\tau=1/4$ choice in layer-adaptive is on the Pareto frontier or can be improved. Strengthens or falsifies the empirical premise E1 of Proposition~\ref{prop:ladapt-intuition}.
- **Hypothesis.** LS-2 ($\tau=1/4$) is on the frontier; LS-5 (K+Q overlap) performs no better than LS-2; LS-6 (Q late-only) underperforms LS-2.
- **Verification.** 6 configs × 2 $\beta$ × 2 benchmarks = 24 runs using `scripts/ocq/eval_subtask4_dynamic_qk_v2.py` with varying `--layer-mode` and `--beta`. Expected GPU time 6 hours.
- **Blocker.** None.
- **Cost estimate.** 6 GPU-hours; owned by coworker.

## 6. Scope exclusions and why

- **Full multi-turn success on $\tau^2$-bench (with user simulator).** Out of scope: we evaluate only first-turn action-set selection. Reason: the simulator introduces orthogonal variance that obscures the steering effect.
- **Spider and ChartQA benchmarks.** Out of scope per prior user direction. These are not multi-tool; they would not exercise the mechanism Theorem~\ref{thm:no-memory} targets.
- **Joint steering + KV-cache compression (e.g., ours + KIVI).** Noted in the paper as ``composable'' but not empirically validated here. A natural follow-up but not needed to support C1 or C2.

### E9. Rank ablation on the ontology basis

- **Intent.** The paper fixes $r=24$ per head. A small ablation showing that $r=24$ is reasonable on the Pareto frontier of performance vs storage strengthens the design choice.
- **Hypothesis.** $r=24$ is at or near the knee: $r=8$ underfits the 4-facet ontology (performance drops), $r=48$ overfits (performance saturates), storage scales linearly with $r$.
- **Verification.** Four runs on Qwen Subtask4 N=497 with Q-coverage $\beta=-0.03$: $r\in\{8, 16, 24, 48\}$ by truncating or extending the Gram-Schmidt output. Report F1 and per-head basis memory.
- **Decision rule.** $r=24$ within 0.5pp of the best among tested $r$ confirms the choice. A better value triggers a paper footnote and a run to re-lock the main tables at that $r$.
- **Blocker.** None.
- **Cost estimate.** ~1 GPU-hour.

## 7. Execution order, aggregate cost, and wall-clock schedule

Assuming one A100 node with `~15 GPU-hours/day` realistic throughput. All experiments are Qwen-only (Qwen2.5-7B-Instruct for main body; 1.5B/3B/14B only in E7 size sweep).

| Day | Items | GPU-hours | Claim | Critical path? |
|---|---|---:|---|---|
| 1 AM | G1 SEKA reproduction | 0.2 | both | yes |
| 1 AM | E3 PCA basis build + run | 1.3 | C1 | yes |
| 1 PM | E2 τ²-retail N=114 | 1.5 | C1 | yes |
| 1 PM | E1 canonical SEKA + AdaSEKA on Subtask4 | 2.0 | C1 | yes |
| 2 AM | E5 + E6 latency + memory profile | 2.0 | C2 | yes |
| 2 AM | E4 stepwise aggregation (CPU) | 0.5 | C1 | yes |
| 2 PM | E7 size sweep (appendix, optional) | 4.0 | appendix | no |
| 3+ | E8 layer boundary sweep (coworker) | 6.0 | appendix | no |
| 3+ | E9 rank ablation | 1.0 | appendix | no |

**Core path** (Day 1--2 AM): about 7 GPU-hours over one and a half calendar days, sufficient to lock every main-body cell. **Appendices** (Day 2 PM+): about 11 additional hours; not on the critical path for submission.

## 8. Falsification triggers

- If G1 fails, SEKA/AdaSEKA cells in Tables 1 and 2 become ``canonical reference only.''
- If E1 shows `Δ F1 ≥ 0` for canonical SEKA on Qwen, the stationary-K limit argument needs revision.
- If E3 PCA cell is indistinguishable from real on Q-coverage, the ``catalog ontology is part of the operator'' strong claim is downgraded.
- If E4 shows `repeated_first_tool_rate` does not rise under stationary K above `no_steer`, Corollary~\ref{cor:seka-repeat} is empirically refuted, and the mechanism story is downgraded.
- If E5 shows per-query latency overhead $\ge 20\%$ of `no_steer`, the efficiency claim is softened.

Any single falsification triggers a targeted edit to the corresponding paper section; two or more falsifications trigger a scope retreat.
