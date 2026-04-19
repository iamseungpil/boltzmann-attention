# F13 FunnelRot — Status

**Status**: **done-positive (primary cell)** (2026-04-19)
**Primary spec**: `math/paper/lie_group/NEW_THEOREM_TEST.md §5` + memory `phase_f13_funnelrot_spec_2026_04_19.md`
**Script**: `scripts/new_theorem_test/train_f12_facetrot_qk.py` with `--schedule ladapt --rot-pairs 2 --skip-layer-28 True --steered-layers all`

## Result (F13b primary cell)

| | Baseline | F13b | Δ (pp) |
|---|---:|---:|---:|
| F1 | 0.764 | **0.803** | **+3.85** |
| F_0.5 | 0.770 | 0.810 | +3.97 |
| EU | 0.571 | 0.612 | +4.08 |
| Jaccard | 0.702 | 0.741 | +3.97 |
| Exact | 0.571 | 0.612 | +4.08 |
| precision | 0.776 | 0.816 | +4.08 |
| recall | 0.759 | 0.796 | +3.74 |
| emitted_two_rate | 0.952 | 0.952 | 0.00 |

Per pre-reg decision tree: **strong positive** (0.80-0.84 bracket) → ICLR ceiling 6.5-7.0.

Train dynamics: CE loss 0.955 → 0.896 over 5 epochs (non-monotonic, final θ_max 24.9°, 191 trainable scalars). Compared to F12b (1220 scalars, CE 1.333 → 0.624), F13 has cleaner eval generalization despite higher train-CE loss.

## Ablation matrix

| Cell | Status | Result |
|---|---|---|
| F13a (= F12b repro, R=32 uniform) | done-null | F1 0.728 (−3.63pp) |
| **F13b (primary, R=4 ladapt skip-L28)** | **done-positive** | **F1 0.803 (+3.85pp)** |
| F13c (projection-only, no SO(2)) | not run | — |
| F13d (ladapt, no L28 skip) | not run | — |
| F13e (uniform + R=4) | not run | — |
| F13f (R=16 ablation) | not run | — |

F13b already clears headline gate; ablations optional but would isolate what's load-bearing (ladapt vs low-rank vs L28-skip).

## Artifacts

- `external/SEKA/seka_projections/f13-qwen25-7b-metatool-facet-subspace/facet_subspace.pt` (R=4, F=75)
- `external/SEKA/seka_projections/f13b-qwen25-7b-r4-ladapt-skipl28/f12_checkpoint.pt`
- `reports/f12_metatool/f13b_train_log.txt`
- `reports/f12_metatool/f13b_eval_log.txt`
- `reports/f12_metatool/f13b_eval_n147.json`
- Memory: `phase_f13_funnelrot_executed_positive_2026_04_19.md`

## Next

- ICLR §1-§7 integration: Thm 6.14 promote to main text; FunnelRot recipe as §4 method; §5 headline table; §6.3 scope-limit reframe.
- Optional: F13c/d/e/f ablations (each ~2 GPU-hr) to strengthen mechanism claim.
- Gate: SAAF Stage 2 + F14 MetaFocus NOT unlocked (F13 < 3pp evaluates FALSE since F13 is +3.85pp). This is the positive-result outcome — no fallback needed.
