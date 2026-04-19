# Paper Result Manifest

This manifest is the audit trail for every number, table, and figure in the Korean paper `neurips2026_steering_ko` (and its English mirror `neurips2026_steering_v2`). Every cell that contains a concrete number points to a result JSON produced by a named script with locked hyperparameters. Placeholder cells point to the script that *will* produce them.

## Convention

- `[LOCKED]` — the JSON exists on develop or main and the number is copied verbatim.
- `[PENDING]` — the experiment is specified; the script exists; results have not yet been run.
- `[REPRO]` — the row is gated on the SEKA canonical reproduction (Appendix B).

Result JSONs follow the schema in `FACT_BASE.md`: they always include
`prompt_template_id`, `scorer`, `decode_policy`, `runtime_config`, `basis_source`.

## Tables

### Table 1 (E1 — MetaTool Subtask4 head-to-head)

| Row | Status | JSON path | Script |
|---|---|---|---|
| Qwen `no_steer` F1=0.7307 | [LOCKED] | `reports/qkv_alpha_microsweep_2026_04_15/full497_alpha_microsweep.json::results[no_steer]` | `scripts/ocq/eval_metatool_subtask4.py --methods no_steer --max-samples 497` |
| Qwen stationary K `a0.3` F1=0.6850 | [LOCKED] | same JSON, method `ocq_bias_a0.3` | same script, `--methods ocq_bias_a0.3` |
| Qwen Q-coverage `b-0.03` F1=0.7535 | [LOCKED] | develop `reports/beta_sweep_2026_04_16/qwen_st4_qbias_b-0.03.json` | `scripts/ocq/eval_metatool_subtask4.py --methods ocq_qbias_b-0.03 --max-samples 497` |
| Qwen Layer-Adaptive K+Q F1=0.7514 | [LOCKED] | develop `reports/layer_adaptive_2026_04_17/qwen_k_early_only_a0.05_b-0.05_N497.json` | `scripts/ocq/eval_subtask4_dynamic_qk_v2.py --alpha 0.05 --beta -0.05 --layer-mode k_early_only --n_queries 497` |
| Qwen SEKA amp=1.0 | [REPRO] | → `reports/matched_seka_2026_04_17/qwen_st4_seka_amp1.0.json` | `scripts/ocq/eval_subtask4_with_real_seka.py --model Qwen/Qwen2.5-7B-Instruct --amplify 1.0 --max-samples 497` |
| Qwen AdaSEKA amp=1.0 | [REPRO] | → `reports/matched_adaseka_2026_04_17/qwen_st4_adaseka_amp1.0.json` | `scripts/diagnostics_2026_04_16/eval_subtask4_with_adaseka.py --amplify 1.0 --max-samples 497` |
| Llama `no_steer` F1=0.6227 | [LOCKED] | develop `reports/wave_2026_04_15_pm/gpu0/llama_inst_st4_full497.json::no_steer` | `scripts/ocq/eval_metatool_subtask4.py --model meta-llama/Llama-3.1-8B-Instruct --methods no_steer` |
| Llama stationary K `a0.3` F1=0.3105 | [LOCKED] | same JSON, `ocq_bias_a0.3` | same script, `--methods ocq_bias_a0.3` |
| Llama Q-coverage `b-0.1` F1=0.6271 | [LOCKED] | same JSON, `ocq_qbias_b-0.1` | same script, `--methods ocq_qbias_b-0.1` |
| Llama Layer-Adaptive K+Q | [PENDING] | → `reports/layer_adaptive_2026_04_17/llama_k_early_only.json` | `scripts/ocq/eval_subtask4_dynamic_qk_v2.py --model meta-llama/Llama-3.1-8B-Instruct --alpha 0.05 --beta -0.05 --layer-mode k_early_only --n_queries 497` |
| Llama SEKA/AdaSEKA | [REPRO] | → `reports/matched_seka_2026_04_17/llama_*` | same SEKA scripts with Llama model flag |

### Table 2 (E2 — τ²-bench retail)

| Row | Status | JSON path | Script |
|---|---|---|---|
| Qwen N=30 smoke (all methods) | [LOCKED] | develop `reports/tau2_2026_04_17/retail_smoke_N30.json` | `scripts/ocq/eval_tau2_bench.py --model Qwen/Qwen2.5-7B-Instruct --domain retail --max-samples 30` |
| Qwen N=114 full (all methods) | [PENDING] | → `reports/tau2_2026_04_17/qwen25_7b_retail_full.json` | same script with `--max-samples 114` |
| SEKA/AdaSEKA on τ²-retail | [REPRO] | → `reports/matched_seka_tau2_2026_04_17/qwen_retail_seka.json` | τ²-adapted SEKA driver (requires extending `eval_subtask4_with_real_seka.py` to tau2-bench dataset path) |

### Table 3 (E3 — Stepwise coverage decomposition)

| Row | Status | JSON path | Script |
|---|---|---|---|
| All six methods, stepwise block | [PENDING] | → derived from the Table 1 run JSONs | `eval_metatool_subtask4.py` already emits the `stepwise` block; post-processing script `scripts/ocq/summarize_stepwise.py` (to create) aggregates across JSONs |

### Table 4 (E4 — Basis ablation)

| Row | Status | JSON path | Script |
|---|---|---|---|
| Real `B_ont`, Q-coverage 0.7471 | [LOCKED] | develop `reports/null_controls_2026_04_15/qwen_qbias_real.json` | `scripts/ocq/eval_metatool_subtask4.py --b-ont .../qwen_B_ont.pt --methods ocq_qbias_b-0.1` |
| Feature-shuffled, Q 0.7254 | [LOCKED] | develop `reports/null_controls_2026_04_15/qwen_qbias_shuffled.json` | same script with shuffled `--b-ont` file |
| Random orthonormal, Q 0.7068 | [LOCKED] | develop `reports/null_controls_2026_04_15/qwen_qbias_random.json` | same with random `--b-ont` |
| Real, stationary K 0.6850 | [LOCKED] | same file, `ocq_bias_a0.3` method | same |
| Shuffled/random, stationary K 0.0 | [LOCKED] | same files, `ocq_bias_a0.3` | same |
| PCA-of-K, Q and K | [PENDING] | → `reports/null_controls_2026_04_17/qwen_pca_baseline.json` | `scripts/ocq/build_pca_baseline_basis.py` → `eval_metatool_subtask4.py --b-ont <pca>.pt` |

### Table 5 (E5 — Qwen model-size sweep)

| Row | Status | JSON path | Script |
|---|---|---|---|
| Qwen 7B F1=0.7307 / 0.7535 / 0.7514 | [LOCKED] | see Table 1 | see Table 1 |
| Qwen 1.5B/3B/14B | [PENDING] | → `reports/size_sweep_2026_04_17/qwen25_{1.5,3,14}b_st4.json` | `scripts/ocq/run_tau2_size_sweep.sh` (adapt for Subtask4) |
| τ²-retail 30/114 size sweep | [PENDING] | → `reports/tau2_size_sweep_2026_04_17/` | `scripts/ocq/run_tau2_size_sweep.sh` |

### Table 6 (implicit — E6 Regime-split)

| Row | Status | JSON path | Script |
|---|---|---|---|
| Subtask1 `k=1` K-bias locked numbers | [LOCKED] | develop `reports/subtask1_main_2026_04_15/` | `scripts/ocq/eval_metatool_subtask1.py` |
| Subtask4 `k=2` | [LOCKED] | see Table 1 | see Table 1 |
| τ²-retail `k∈[4,13]` per-k bins | [PENDING] | derived from Table 2 per-sample JSON | `scripts/ocq/summarize_regime_split.py` (to create) aggregates per-`k` bins |

## Figures

| Fig | Status | Source script | Input JSONs |
|---|---|---|---|
| Fig 1 (conceptual) | [PLACEHOLDER] | `scripts/build_placeholder_figures.py::fig1_concept` | none (schematic) |
| Fig 2 (Δ F1 vs k) | [PLACEHOLDER] | `scripts/build_placeholder_figures.py::fig2_delta_vs_k` | → replace by `scripts/build_fig2_from_jsons.py` once Table 2/6 run |
| Fig 3 (stepwise) | [PLACEHOLDER] | `scripts/build_placeholder_figures.py::fig3_stepwise` | → replace by `scripts/build_fig3_from_jsons.py` once Table 1 + stepwise aggregation done |
| Fig 4 (basis heatmap) | [PLACEHOLDER] | `scripts/build_placeholder_figures.py::fig4_basis` | → replace after Table 4 PCA row lands |
| Fig 5 (size sweep) | [PLACEHOLDER] | `scripts/build_placeholder_figures.py::fig5_size_sweep` | → replace after Table 5 pending cells land |
| Fig legacy (main/ablations/stability) | [LOCKED] | `scripts/build_figures_from_results.py` in v2 repo | legacy null-control and bound JSONs |

## Directory layout

```
paper/neurips2026_steering_ko/
├── main.tex, content.tex, refs.bib, neurips_2026.sty
├── MANIFEST.md                 ← this file
├── FACT_BASE.md                ← locked numbers and JSON schema
├── sections/                   ← 01–09
├── figures/                    ← fig1..fig5 (new) + fig_main/ablations/stability (legacy)
└── scripts/
    ├── build_placeholder_figures.py
    ├── (future) build_fig2_from_jsons.py
    ├── (future) build_fig3_from_jsons.py
    ├── (future) build_fig4_from_jsons.py
    └── (future) build_fig5_from_jsons.py

scripts/ocq/                    ← code repo, shared with develop
├── eval_metatool_subtask1.py           (hook installers + Subtask1 driver)
├── eval_metatool_subtask4.py           (paper-facing Subtask4 driver, stepwise block)
├── eval_subtask4_dynamic_qk_v2.py      (iterative K+Q, ladapt schedule)
├── eval_tau2_bench.py                  (τ² retail/airline/telecom/banking)
├── eval_subtask4_with_real_seka.py     (canonical SEKA on Subtask4)
├── build_pca_baseline_basis.py         (E4 PCA basis builder)
├── build_tau2_ontology.py              (τ² per-domain B_ont)
└── run_tau2_size_sweep.sh              (E5 driver)

scripts/diagnostics_2026_04_16/
├── eval_subtask4_with_adaseka.py       (canonical AdaSEKA per-facet experts)
└── build_adaseka_experts_from_bont.py  (per-facet SVD expert builder)
```

## SEKA reproduction gate

Every row marked `[REPRO]` is blocked on the SEKA canonical CounterFact reproduction of ES$=0.952\pm 0.02$ on A100. The protocol is in the paper Appendix B; the log of the A6000 reproduction failure is `reports/seka_repro_2026_04_16/`. Until this gate passes, all `[REPRO]` rows render as "canonical SEKA (reference)" rather than head-to-head comparison.
