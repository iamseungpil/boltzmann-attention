# OCQ Publish Manifest (2026-04-10)

This manifest marks the canonical OCQ/E8 archive for the April 10, 2026 push.

## Scope

- parser-safe evaluator hardening for MetaTool
- Qwen/Mistral cross-model OCQ status
- E8 execution policy and next-step plan
- control-bundle smoke outputs and cross-model result files

## Canonical Docs

- `reports/OCQ_CROSS_MODEL_STATUS_2026-04-10.md`
- `reports/OCQ_E8_EXECUTION_POLICY_2026-04-10.md`
- `reports/OCQ_NEXT_PLAN_2026-04-10.md`
- `reports/OCQ_WORKTREE_GUIDE_2026-04-10.md`
- `reports/COWORKER_REQUEST_cross_model_2026_04_10.md`

## Canonical Code

- `scripts/ocq/eval_metatool_subtask1.py`
- `scripts/ocq/make_control_b_ont.py`
- `scripts/ocq/build_metatool_ontology_v2.py`
- `scripts/ocq/build_qwen_metatool_b_ont.py`
- `scripts/ocq_e8_qwen_control_bundle.sh`
- `scripts/ocq_e8_qwen_control.sh`
- `scripts/ocq_e8_qwen_mmlu_safety.sh`
- `scripts/ocq_e8_mistral_low_alpha.sh`

## Canonical Results

- `results/ocq/cross_model/`
- `reports/axis2_theoretical_verification/`
- `reports/figures/ocq_cross_model_alpha_sweep_2026_04_10.png`

## Notes

- Legacy top-level reports were moved under `reports/archive/top_level_legacy/`.
- Legacy OCQ runner scripts were moved under `scripts/archive/ocq_legacy_runners/`.
- This archive is intended to be mirrored both to GitHub and to a Hugging Face dataset repo.
