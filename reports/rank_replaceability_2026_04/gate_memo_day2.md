# Day-2 Gate Memo — E1 Rank Measurement Results (2026-04-29)

> Companion to: `reports/EXPERIMENT_PLAN_v27_rank_bounded_replaceability_2026_04_28.md`
> Detailed tables: `reports/rank_replaceability_2026_04/analysis_summary.md`

## Headline

**Effective rank $r^*(\tau{=}0.95)$ of the prefix-attention output function $\Phi_P$ is *very small* across all 8 (model, task) combinations measured.** Mean $r^* \in [1.00, 2.25]$, median = 1 universally. Per Theorem 1, this places us deep inside **Corollary 1.1 (static replaceability sufficient condition)**.

## Headline table (τ = 0.95)

| Model | Task | N | r*_mean | r*_med | r*_max | high-rank heads (≥8) |
|---|---|---:|---:|---:|---:|---:|
| Qwen2.5-7B | metatool_st4 | 256 | **2.25** | 1 | **40** | 43 / 784 (5.5%) |
| Qwen2.5-7B | τ²-retail | 114 | 1.59 | 1 | 28 | 19 / 784 (2.4%) |
| Qwen2.5-7B | τ²-telecom | 256 | 1.01 | 1 | 2 | 0 / 784 |
| Qwen2.5-7B | τ²-airline | 50 | 1.54 | 1 | 23 | 17 / 784 (2.2%) |
| Llama-3.1-8B | metatool_st4 | 256 | **1.38** | 1 | **13** | 6 / 1024 (0.6%) |
| Llama-3.1-8B | τ²-retail | 114 | 1.06 | 1 | 4 | 0 / 1024 |
| Llama-3.1-8B | τ²-telecom | 256 | 1.00 | 1 | 1 | 0 / 1024 |
| Llama-3.1-8B | τ²-airline | 50 | 1.10 | 1 | 4 | 0 / 1024 |

## Theorem 1 verdict

Plan v27 §3 decision tree:

```
mean r*(0.95) < 16 (모든 layer/head)
 └─ E3 static recovery 직진
     └─ Paper Story A: "Static rank-k intervention suffices"
```

We are well below the $k_0 = 32$ threshold for the strong "static replaceability sufficient" branch. **Corollary 1.1 fires across all measured (model, task) cells.** The Theorem 1 prediction "rank-$r^*$ static intervention recovers $\tau$-fraction of prefix-attention energy" is therefore the active claim to test next (E3).

## Bimodality (Theorem 1 Corollary 1.2 signal)

| Model | Task | Mean | Max | High-rank heads |
|---|---|---:|---:|---:|
| Qwen2.5-7B | metatool_st4 | 2.25 | 40 | 5.5% |
| Qwen2.5-7B | τ²-retail | 1.59 | 28 | 2.4% |
| Qwen2.5-7B | τ²-airline | 1.54 | 23 | 2.2% |
| Llama-3.1-8B | metatool_st4 | 1.38 | 13 | 0.6% |

A small but *non-zero* fraction of heads have $r^* \ge 8$ (the "mixed" or "high-rank" heads). These are concentrated more in Qwen than in Llama. **This is exactly the predicted bimodal pattern** (mostly low-rank heads with a small high-rank tail). Implication for paper: rank-$k$ intervention on these specific heads is the binding constraint; on the bulk of heads, even rank-1 suffices.

The high-rank heads cluster in a few specific layers (see `analysis_summary.md` layer profiles). For Qwen MetaTool ST4, layer 0 and layers 24–27 (final block) carry most of the mass — consistent with the companion paper's "layer 28 amplification" mechanism observation.

## Caveats (preregistered + new)

1. **τ² prefix is *generic system prompt only*, not full tool catalog.** My loader for `tau2_*` tasks uses `DEFAULT_TOOL_SYSTEM_PROMPT` plus the user instruction; the actual `RETAIL_TOOLS` / `TELECOM_TOOLS` / `AIRLINE_TOOLS` JSON catalogs (~1–2k tokens of tool schemas) are not injected. **τ² numbers above are therefore lower bounds** on $r^*$ for the realistic prefix. Re-running with full catalogs is the obvious extension; included in next sprint.
2. **Prefix length is short (93–148 tokens).** This favors low rank artificially. Real production agentic prompts (Anthropic Opus 4.7 harness, Graphify) are 2–8k tokens. We do not yet know how $r^*$ scales with prefix length; this is a critical open question for the practical claim.
3. **N = 256 is small relative to $d_h = 128$.** For high-rank heads measured at $r^* \approx 30+$, the SVD is on a $(N, d_h) = (256, 128)$ matrix where rank can saturate at $\min(256, 128) = 128$. The reported maxes (40, 71) are well below this saturation, but for higher-confidence high-tail estimates we should run $N \ge 512$.
4. **Last-position-only query state.** I measured $\Phi_P$ at the last token position only (where generation begins). Multi-step agentic flow has additional decision points; rank may grow with sequence depth. This becomes relevant for E5 (Q-bias 1st-order check) and E6 (sanity).

## Risk: too-good-to-be-true check

Rank means as low as 1.00 (Qwen telecom) raise an eyebrow. Two checks recommended before committing the headline:

- **Sanity A**: Verify that random-prefix control gives even *lower* rank (it should — a generic random prefix has near-zero query-conditional content). If random ≥ real, the measurement pipeline has a bug.
- **Sanity B**: Verify that random-query control (task-irrelevant queries) gives *higher* rank (random queries should have less consistent attention patterns over the prefix, hence higher rank). If random-queries ≤ real, the prefix is being treated as boilerplate regardless of query content.

These are E1 controls planned in the original plan but not yet executed. Recommend running both before publishing any headline number externally.

## Decision (next sprint)

**Branch chosen**: Plan v27 decision tree branch *strong static gate*. Proceed directly to:

- **E3 (static recovery)**: Construct $V_k$ for $k \in \{1, 2, 4, 8, 16\}$ from the measured top-eigenvectors per (layer, head), inject into frozen Qwen / Llama, evaluate task accuracy on MetaTool ST4 (we have eval_metatool_subtask4.py harness already). Pass criterion: rank-$r^*(0.95)$ static intervention recovers $\ge$ full-prompt-acc $-5$pp.
- **E1 controls (sanity)**: random-prefix and random-query variants on Qwen MetaTool only (cheap). Prerequisite for any external claim.
- **E1 extension (full prefix)**: re-run τ²-bench with actual tool catalogs from `eval_tau2_bench.py`. This addresses Caveat 1.

Skipping E2 (layer/head specialization) as a separate experiment — already covered in `analysis_summary.md` layer profiles.

E5 (Q-bias Taylor sign check) deferred until after E3 confirms the static story holds. If E3 passes cleanly, E5's purpose (justifying *why* Q-bias works) becomes secondary.

E6 (random basis ablation) still needed but lower priority — the 8 cells already give cross-family + cross-domain robustness.

## Wallclock

Total compute wallclock: ~3 minutes (8 runs × ~25–40s each on A6000). E1 of the plan is empirically cheap.

## Files

- Raw JSONs: `reports/rank_replaceability_2026_04/{qwen,llama}_{metatool_st4,tau2_{retail,telecom,airline}}_n256.json`
- Aggregated tables: `reports/rank_replaceability_2026_04/analysis_summary.md`
- Logs: `reports/rank_replaceability_2026_04/logs/*.log`
- Scripts: `scripts/rank_replaceability/{measure_phi_rank.py,analyze_rank_results.py}`

## Honest framing for paper draft

The result is *almost suspiciously clean* in the static-replaceability direction. Two things to do before claiming Theorem 1's empirical sufficiency in §5 of `PAPER_DRAFT_v0.md`:

1. Run the random-prefix / random-query controls (Sanity A/B). If both pass, the result is solid.
2. Re-run with actual τ² tool catalogs (full prefix). If $r^*$ stays small even at 2k+ token prefixes, this is a *strong* paper claim. If $r^*$ explodes with prefix length, the right framing is "rank scales sub-linearly with prefix length" rather than "rank is small".

Until both are done, §5 of the paper should remain placeholder. The current run is *evidence consistent with* Theorem 1, not yet a definitive empirical anchor.
