# F13 FunnelRot — Status

**Status**: spec-ready (2026-04-19)
**Primary spec**: `math/paper/lie_group/NEW_THEOREM_TEST.md §5` + memory `phase_f13_funnelrot_spec_2026_04_19.md`
**Script**: `scripts/new_theorem_test/train_f12_facetrot_qk.py` with `--schedule ladapt --rot-pairs 2 --skip-layer-28`
**Gate**: None (parallel/superset of F12b; can run before or after F12)
**Cells**: F13a (=F12b repro) / F13b primary ladapt+R4+skip-L28 / F13c projection-only / F13d L28-intervene neg ctrl / F13e uniform ablation / F13f R=16 ablation
**Decision**: F13b > F13a by ≥+2pp → ladapt-schedule contributory; else non-load-bearing
**Cost**: 12–18 GPU-hr (6 cells)
**Log**: (empty; update here when run starts)
