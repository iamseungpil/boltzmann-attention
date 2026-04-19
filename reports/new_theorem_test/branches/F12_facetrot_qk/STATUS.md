# F12 FacetRot-QK — Status

**Status**: **done-falsified** (2026-04-19)
**Primary spec**: `math/paper/lie_group/NEW_THEOREM_TEST.md §5` + memory `phase_f12_facetrot_qk_spec_2026_04_19.md`
**Script**: `scripts/new_theorem_test/train_f12_facetrot_qk.py` (complete) + `build_f12_facet_subspace.py` + `eval_subtask4_facetrot_qk.py`
**Gate**: None (primary track) — unconditionally executed.

## Result

F12b primary cell (R=32, uniform L18-27, 5 epochs LoRA, MetaTool Subtask4 N=147 held-out):

| | Baseline | F12b | Δ (pp) |
|---|---:|---:|---:|
| F1 | 0.764 | **0.728** | **−3.63** |
| recall | 0.759 | 0.704 | −5.44 |
| precision | 0.776 | 0.776 | 0.00 |
| Exact | 0.571 | 0.497 | −7.48 |
| emitted_two_rate | 0.952 | 0.796 | **−15.6** |

Train loss: 1.333 → 0.624 (monotonic, 5 epochs). θ_max 29.5°, ‖θ‖_F=3.67. No convergence failure — the rotation LoRA trained successfully but regresses downstream generation.

Per pre-reg decision tree: **null branch** (0.71-0.74 F1) / **harmful vs actual baseline**. No +3pp gate clearance → no main-text promotion of Thm 6.14 Hybrid / Lemma 6.14.A.

## Artifacts

- `external/SEKA/seka_projections/f12-qwen25-7b-metatool-facet-subspace/facet_subspace.pt` (F=75, L=28, H=4, d=128, R=32)
- `external/SEKA/seka_projections/f12b-qwen25-7b-r32-uniform/f12_checkpoint.pt`
- `reports/f12_metatool/f12b_train_log.txt`
- `reports/f12_metatool/f12b_eval_log.txt`
- `reports/f12_metatool/f12b_eval_n147.json`
- Memory: `phase_f12_facetrot_qk_executed_falsified_2026_04_19.md`

## Diagnosis

Train/eval mismatch pathology: teacher-forced CE converges, but free-running greedy decode loses 15.6pp in emitted_two_rate. Precision unchanged; recall −5.44pp. Rotation at L18-27 with L28 intact biases early-EOS.

## Next

- F13 FunnelRot (ladapt + R=4 + skip L28) — strict superset. Pathology may be mitigated by skipping mid-upper K-rotation via ladapt staging.
- If F13 also null: H-Order canary + §6.3 scope-limit strengthening become primary.
