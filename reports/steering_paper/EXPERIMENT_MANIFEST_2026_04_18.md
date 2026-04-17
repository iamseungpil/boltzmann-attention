# Experiment Manifest — Steering Paper (2026-04-18)

Navigable index of every experiment artifact referenced by
`paper/neurips2026_steering_ko`. Paths prefixed `develop:` live only on
`origin/develop`; bare paths live on local `main`. All other paths are
repo-relative.

Scope: Table 1 (`sections/06_experiments.tex:15-47`), active cross-model
runs, pending experiments, script catalog, bug log, and develop-only
resources.

---

## 1. Locked Qwen results — Table 1 (main body)

F1 column values copied verbatim from `sections/06_experiments.tex` rows
20-45. Sources cross-checked against `paper/neurips2026_steering_ko/MANIFEST.md`.

### MetaTool Subtask4 (N=497)

| F1     | Method (paper label)          | Source JSON                                                                 | Script                                 |
|--------|-------------------------------|-----------------------------------------------------------------------------|----------------------------------------|
| 0.7307 | `no_steer`                    | develop:`reports/subtask4_overnight/st4_real_N0.json::no_steer`             | `scripts/ocq/eval_metatool_subtask4.py` |
| 0.6850 | stationary K `a0.3` (collapse)| develop:`reports/subtask4_overnight/st4_real_N0.json::ocq_bias_a0.3`        | same, `--methods ocq_bias_a0.3`        |
| 0.7535 | Q-rotation `b-0.03` (best)    | develop:`reports/beta_sweep_2026_04_16/qwen_st4_qbias_b-0.03.json` *        | same, `--methods ocq_qbias_b-0.03`     |
| 0.7514 | Layer-adaptive K+Q            | develop:`reports/layer_adaptive_2026_04_17/qwen_k_early_only_a0.05_b-0.05_N497.json` | `scripts/ocq/eval_subtask4_dynamic_qk_v2.py --alpha 0.05 --beta -0.05 --layer-mode k_early_only --n_queries 497` |
| *placeholder* | canonical SEKA amp=1.0  | pending develop:`reports/matched_seka_2026_04_17/qwen_st4_seka_amp1.0.json` | `scripts/ocq/eval_subtask4_with_real_seka.py --model Qwen/Qwen2.5-7B-Instruct --amplify 1.0 --max-samples 497` |

*The `b-0.03` row is claimed [LOCKED] in MANIFEST.md but the referenced
develop path is not in `git ls-tree origin/develop`; see Gap N1.
The `qwen_st4_ladapt_full_N497.json` on develop reports ladapt F1=0.7446
with method `ocq_ladapt_k0.05_q-0.03`. The paper's 0.7514 comes from a
different layer schedule file (`qwen_k_early_only_a0.05_b-0.05_N497.json`);
verify at copy-in time.

### τ²-retail (N=114)

| F1     | Method                     | Source                                              |
|--------|----------------------------|-----------------------------------------------------|
| 0.4679 | `no_steer`                 | `reports/tau2_2026_04_17/retail_full_v2.json`       |
| 0.4737 | stationary K `a0.05`       | same, method `ocq_bias_a0.05`                       |
| 0.5190 | Q-rotation `b-0.03` (best) | same, method `ocq_qbias_b-0.03`                     |
| 0.4829 | Layer-adaptive K+Q         | same, method `ocq_ladapt_k0.05_q-0.03`              |
| *placeholder* | SEKA amp=1.0        | pending (SEKA reproduction gate)                    |

CLI: `python scripts/ocq/eval_tau2_bench.py --model Qwen/Qwen2.5-7B-Instruct --device cuda:0 --b-ont external/SEKA/seka_projections/ontology-qwen25-7b-tau2-retail/B_ont.pt --methods no_steer ocq_bias_a0.05 ocq_bias_a0.1 ocq_qbias_b-0.03 ocq_ladapt_k0.05_q-0.03 --domain retail --max-samples 114 --out reports/tau2_2026_04_17/retail_full_v2.json`

### τ²-telecom (N=200)

| F1     | Method                         | Source                                          |
|--------|--------------------------------|-------------------------------------------------|
| 0.2512 | `no_steer`                     | `reports/tau2_2026_04_17/telecom_N200.json`     |
| 0.3691 | stationary K `a0.05`           | same, `ocq_bias_a0.05`                          |
| 0.4349 | Q-rotation `b-0.03`            | same, `ocq_qbias_b-0.03`                        |
| 0.3604 | Layer-adaptive K+Q             | same, `ocq_ladapt_k0.05_q-0.03`                 |
| 0.4990 | **Q-rotation `b+0.05` (best)** | **MISSING — see Gap N2**                        |

CLI (what was actually run): `python scripts/ocq/eval_tau2_bench.py --model Qwen/Qwen2.5-7B-Instruct --b-ont external/SEKA/seka_projections/ontology-qwen25-7b-tau2-telecom/B_ont.pt --methods no_steer ocq_bias_a0.05 ocq_qbias_b-0.03 ocq_ladapt_k0.05_q-0.03 --domain telecom --max-samples 200 --out reports/tau2_2026_04_17/telecom_N200.json`. The `ocq_qbias_b0.05` method that produces the paper's headline +24.78pp is NOT in any JSON in `reports/tau2_2026_04_17/` or on develop. Needs a dedicated re-run to lock.

### τ²-airline (N=50)

| F1     | Method                     | Source                                               |
|--------|----------------------------|------------------------------------------------------|
| 0.3285 | `no_steer`                 | develop:`reports/tau2_2026_04_17/airline_full.json`  |
| 0.3242 | Q-rotation `b-0.03`        | same, `ocq_qbias_b-0.03`                             |
| 0.3669 | Layer-adaptive K+Q (best)  | same, `ocq_ladapt_k0.05_q-0.03`                      |

Airline JSON is develop-only; not yet cherry-picked to local main.

---

## 2. Active Llama cross-model runs (in progress 2026-04-17→18)

Master log: `logs/llama_sweep_master.log`. Per-phase logs:
`logs/l1a_llama_retail_bont.log`, `logs/l1b_llama_telecom_bont.log`,
`logs/l2_llama_retail_sweep.log`.

| Phase | Status | Artifact |
|-------|--------|----------|
| L1a Llama retail B_ont build | done | `external/SEKA/seka_projections/ontology-llama31-8b-tau2-retail/B_ont.pt` (shape `(32,8,128,10)`) |
| L1b Llama telecom B_ont build | done | `external/SEKA/seka_projections/ontology-llama31-8b-tau2-telecom/B_ont.pt` (shape `(32,8,128,10)`) |
| L2 retail Q-sweep N=114 | running (β=-0.10 locked F1=0.5021 vs baseline 0.5059) | target `reports/tau2_2026_04_18/llama31_retail_N114.json` |
| L3 telecom Q-sweep N=200 | pending | target `reports/tau2_2026_04_18/llama31_telecom_N200.json` |
| L4 MetaTool ST4 layer-adaptive | pending (drop first if deadline tight) | target `reports/llama_cross_model_2026_04_18/st4_ladapt_llama.json` |

Launcher: `scripts/ocq/run_llama_cross_model_sweep.sh`. Smoke artifact
already present: `reports/tau2_2026_04_18/llama31_retail_smoke3.json`.

---

## 3. Pending experiments (gaps in paper)

| Tag | Fills | Blocker / cost |
|-----|-------|----------------|
| G1 canonical SEKA A100 reproduction | ST4 row "canonical SEKA amp=1.0" + every τ² `[REPRO]` cell | Blocker: A6000 reproduction of CounterFact ES=0.952±0.02 failed (logs `reports/seka_repro_2026_04_16/`). Needs A100. Cost ~4 GPU-h once env is clean. |
| E3 PCA-of-K basis ablation | Table 4 "PCA-of-K, Q and K" | Script to build: `scripts/ocq/build_pca_baseline_basis.py`; eval reuses `eval_metatool_subtask4.py`. Cost ~2 GPU-h. |
| E7 Qwen 1.5B/3B/14B size sweep | Table 5 pending rows | Driver: `scripts/ocq/run_tau2_size_sweep.sh` (ST4 adaptation needed). Cost ~12 GPU-h for three sizes × ST4+retail. |
| E8 Phase 2.5 layer boundary sweep | Supporting fig for layer-adaptive schedule | Coworker task. Cost ~6 GPU-h. |
| B1 β* logit-lens discriminative-G | `sections/05_theory.tex` β* claim gating (C3) on retail+telecom | Script drafted: `scripts/ocq/measure_beta_star_logit.py`. Needs B2 per-sample best-β ground truth. Cost ~0.5 GPU-h. |
| N1 Re-lock ST4 `b-0.03` F1=0.7535 | Table 1 row | Develop path cited in MANIFEST.md not found by `git ls-tree`; re-run `eval_metatool_subtask4.py --methods ocq_qbias_b-0.03 --max-samples 497`. |
| N2 Re-lock telecom `b+0.05` F1=0.4990 | Table 1 headline +24.78pp cell | Not in any JSON. Re-run `eval_tau2_bench.py --domain telecom --max-samples 200 --methods ocq_qbias_b0.05 ocq_qbias_b0.03 ocq_qbias_b0.10`. Cost ~3 GPU-h. |

---

## 4. Scripts catalog

| Script | One-liner | Paper claim it supports |
|--------|-----------|-------------------------|
| `scripts/ocq/build_tau2_ontology.py` | τ² per-domain ontology JSON builder (model-agnostic). | τ² B_ont pipeline for retail/telecom/airline. |
| `scripts/ocq/build_qwen_metatool_b_ont.py` | Builds `B_ont.pt` of shape `(L, H_kv, d, r)` from a facet ontology; name is historical, works for any HF model via `--model`. | Ontology basis for all Qwen/Llama rows. |
| `scripts/ocq/eval_tau2_bench.py` | τ² eval driver with K/Q hooks (`no_steer`, `ocq_bias_a*`, `ocq_qbias_b*`, `ocq_ladapt_*`). | Rows 27-44 of Table 1; L2/L3 Llama strand. |
| `scripts/ocq/eval_metatool_subtask4.py` | MetaTool ST4 driver, stepwise block emitter. | ST4 `no_steer`, `ocq_bias_a0.3`, `ocq_qbias_b-0.03` rows. |
| `scripts/ocq/eval_subtask4_dynamic_qk_v2.py` | ST4 layer-adaptive K+Q (iterative, `k_early_only` schedule). | ST4 layer-adaptive 0.7514 row + L4. |
| `scripts/ocq/eval_subtask4_with_real_seka.py` | Canonical SEKA on ST4 (amp sweep). | ST4 SEKA row (gated by G1). |
| `scripts/diagnostics_2026_04_16/eval_subtask4_with_adaseka.py` | Canonical AdaSEKA per-facet experts. | AdaSEKA comparison row. |
| `scripts/ocq/measure_beta_star.py` | β* schema-G predictor (fails on telecom, 31.2% agreement). | Appendix honest-failure reference. |
| `scripts/ocq/measure_beta_star_logit.py` | β* logit-lens discriminative-G variant (drafted 2026-04-17). | B1: if passes, retains main-body β* framing. |
| `scripts/ocq/run_llama_cross_model_sweep.sh` | L1-L4 master launcher. | Active Llama runs (§2). |
| `scripts/ontology_facet_basis.py` | Shared facet → basis primitives imported by B_ont builders. | Infrastructure. |

Typical CLIs are inline in MANIFEST.md entries and `EXPERIMENT_PLAN_UNIFIED_2026_04_18_v4.md`.

---

## 5. Bugs fixed (2026-04-17 session)

| File:line | Bug | Fix |
|-----------|-----|-----|
| `scripts/ontology_facet_basis.py:56-59` | Hardcoded `os.environ['CUDA_VISIBLE_DEVICES']='1'` (coworker's 2-GPU env), hijacked single-GPU runs on import. | Changed to `os.environ.setdefault('CUDA_VISIBLE_DEVICES', os.environ.get('CUDA_VISIBLE_DEVICES', '0'))`; caller's device pin survives import. |
| `scripts/ocq/eval_tau2_bench.py:613` | `domain_tools` referenced inside `run_method` without being passed in; NameError during first τ² run. | Added `domain_tools` as explicit keyword argument to `run_method`; propagated from outer scope. |
| `scripts/ocq/build_tau2_ontology.py:65` | Hardcoded `/home/woori/workspace_common/...` path (coworker workstation). | Replaced with `REPO = Path(__file__).resolve().parents[2]` + `REPO/"external/tau2-bench/src/tau2/domains"`. |
| `scripts/ocq/build_qwen_metatool_b_ont.py:50` | Same hardcoded `/home/woori/...` path. | Same `Path(__file__).resolve().parents[2]` portability fix. |

All fixes are in the working tree (see `git status`); none committed yet.

---

## 6. Develop-branch-only resources

| Resource | Notes |
|----------|-------|
| develop:`math/paper/benchmark_design/PAPER_DRAFT_v4.md` | 1934-line English markdown paper draft. NOT the canonical NeurIPS submission — canonical is `paper/neurips2026_steering_ko` on main. |
| develop:`math/paper/lie_group/THEOREM_SUPPLEMENTS_2026_04_16.md` | Formal β* proof + improvement roadmap. Backported to `paper/neurips2026_steering_ko/sections/09_appendices.tex` by today's agent. |
| develop:`reports/tau2_2026_04_17/{retail_full_v2,telecom_N200,airline_full,banking_full_v2,tau2_*_ontology}.json` | `retail_full_v2` and `telecom_N200` cherry-picked to local `reports/tau2_2026_04_17/`; `airline_full` and banking variants still develop-only. |
| develop:`reports/beta_star_2026_04_17/{retail_smoke30_allpos,telecom_smoke*}.json` | Two files cherry-picked to local (`retail_smoke30_allpos.json`, `telecom_smoke20_schema.json`); the `telecom_smoke10.json`, `telecom_smoke20_allpos_v2.json`, `telecom_smoke20_allschemas_last.json` still develop-only. |
| develop:`reports/layer_adaptive_2026_04_17/qwen_st4_ladapt_full_N497.json` | ST4 layer-adaptive eval (F1=0.7446 for `ocq_ladapt_k0.05_q-0.03`). Needs reconciliation with the paper's 0.7514 figure (different schedule file `qwen_k_early_only_a0.05_b-0.05_N497.json`). |
| develop:`reports/qkv_alpha_microsweep_2026_04_15/full497_alpha_microsweep.json` | Source of ST4 `no_steer` 0.7307 per MANIFEST.md. |
| develop:`reports/subtask4_overnight/st4_real_N0.json` | Source of ST4 `ocq_bias_a0.3` 0.6850. |
| develop:`reports/coworker_reproduction_2026_04_16/B_ont/{qwen25-7b-metatool,llama31-8b-metatool,mistral-7b-v03-metatool-skipL0-padmax}_B_ont.pt` | Cross-model B_ont checkpoints for MetaTool. |

---

## Cross-links

- `paper/neurips2026_steering_ko/MANIFEST.md` — per-cell audit trail (locked/pending/repro).
- `reports/steering_paper/EXPERIMENT_PLAN_UNIFIED_2026_04_18_v4.md` — full L/B/E plan with CLIs, decision rules, cost.
- `reports/steering_paper/PAPER_ALIGNMENT_AUDIT_2026_04_16.md` — last alignment audit.
- `reports/steering_paper/STEERING_EVIDENCE_SUMMARY_2026_04_16.md` — claim-evidence ledger.
