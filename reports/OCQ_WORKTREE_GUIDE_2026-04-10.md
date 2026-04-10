# OCQ Worktree Guide (2026-04-10)

## Goal

This guide explains the compact layout of the current `ba-ocq-develop`
worktree after cleanup.

## Top-level rule

Only the current OCQ experiment line should be visible at the top level.
Historical material is archived, not deleted.

## Canonical layout

### Reports

- `reports/COWORKER_REQUEST_cross_model_2026_04_10.md`
  - original develop-branch request
- `reports/OCQ_CROSS_MODEL_STATUS_2026-04-10.{md,tex,pdf}`
  - current evidence snapshot
- `reports/OCQ_E8_EXECUTION_POLICY_2026-04-10.md`
  - where and how to run
- `reports/OCQ_NEXT_PLAN_2026-04-10.md`
  - canonical next-step plan
- `reports/OCQ_STAGE2_NARROWED_PLAN_2026-04-10.md`
  - older narrowing doc kept for reference
- `reports/OCQ_WORKTREE_GUIDE_2026-04-10.md`
  - this file

### Archived reports

- `reports/archive/top_level_legacy/`
  - old top-level reports moved out of the main surface

### Scripts

- `scripts/ocq/`
  - core builders and evaluators
- `scripts/ocq_e8_qwen_control.sh`
  - current Qwen rerun runner
  - not yet the full control-bundle runner
- `scripts/ocq_e8_qwen_mmlu_safety.sh`
  - regression/safety runner
- `scripts/ocq_e8_mistral_low_alpha.sh`
  - completed follow-up runner retained for reproducibility only

### Archived runners

- `scripts/archive/ocq_legacy_runners/`
  - historical one-off runners no longer considered canonical

### Results

- `results/ocq/smoke/`
  - tiny smoke artifacts
- `results/ocq/first1/`
  - bounded first-layer exploratory runs
- `results/ocq/cross_model/`
  - canonical E8-complete cross-model and safety results

## Current primary result files

- `results/ocq/cross_model/qwen25_7b_metatool_alpha_sweep_995.json`
- `results/ocq/cross_model/mistral_7b_v03_metatool_alpha_sweep_995.json`
- `results/ocq/cross_model/mistral_7b_v03_metatool_low_alpha_995.json`
- `results/ocq/cross_model/qwen25_7b_mmlu_safety_1000.json`

## Cleanup policy

1. Do not delete evidence-bearing artifacts unless they are exact duplicates.
2. Move historical top-level files into archive instead of leaving them mixed
   with active OCQ docs.
3. Keep only active E8 runners at script top level.
4. Copy remote-complete E8 artifacts back into local `results/ocq/cross_model`.
