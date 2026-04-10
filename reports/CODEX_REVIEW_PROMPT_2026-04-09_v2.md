# Problem Description

Validate whether the revised documentation and plan in `/home/v-seungplee/boltzmann-attention` now correctly narrow the current research state, or whether they still overclaim.

## Files to Review

- `reports/RESEARCH_STATUS_2026-04-09.md`
- `reports/EXPERIMENT_PLAN_v30_proxy_to_storage.md`
- `reports/selective_refinement_status_2026-04-09.tex`
- `paper/neurips2026_ko/FACT_BASE.md`
- `scripts/exp_query_exploit.py`
- `scripts/exp_cliffkv_niah.py`

## Ground Truth Constraints

- Current positive `two_pass` / `cliffkv` results are attention-path proxy diagnostics, not storage-valid compressed-cache measurements.
- `QDRP` only helps in a toy low-budget synthetic regime and currently loses its main real-trace story.
- `TRIC` loses to a shared linear predictor in current diagnostics.

## Questions

1. Do the revised docs now correctly match the code and evidence?
2. What overclaims, if any, still remain?
3. Is the v30 plan the right next-step sequence?
4. What is the smallest remaining fix before launching new bounded experiments?

Answer briefly but harshly. If the docs are finally aligned, say that directly. If not, identify the exact remaining sentence or plan item that is still wrong.
