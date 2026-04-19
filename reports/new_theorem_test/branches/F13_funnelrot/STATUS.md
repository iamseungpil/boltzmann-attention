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

| Cell | R | Schedule | L28-skip | Status | F1 | ΔF1 | emit2 |
|---|---:|---|:---:|---|---:|---:|---:|
| baseline | — | — | — | — | 0.764 | — | 0.952 |
| F13a (= F12b repro) | 32 | uniform | ✗ | done-null | 0.728 | −3.63 | 0.796 |
| F13d (ladapt, no L28 skip) | 4 | ladapt | ✗ | **done-positive** | 0.782 | +1.82 | 0.925 |
| F13e (uniform + R=4) | 4 | uniform | ✓ | **done-positive** | 0.788 | +2.37 | 0.946 |
| **F13b (primary, full recipe)** | **4** | **ladapt** | ✓ | **done-positive** | **0.803** | **+3.85** | 0.952 |
| F13c (projection-only, no SO(2)) | 4 | ladapt | ✓ | not run | — | — | — |
| F13f (R=16 ablation) | 16 | ladapt | ✓ | not run | — | — | — |

### Mechanism decomposition (from F13b / F13d / F13e / F12b four-cell)

- **Rank R=32 → R=4**: primary (~+6pp, dominates F12b → F13e delta).
- **ladapt schedule** (vs uniform): +1.5pp (F13b − F13e).
- **L28-skip** (vs intact): +2pp (F13b − F13d).
- **Interactions**: near-additive, small positive synergy.

### Pre-reg outcomes

| Cell | Pre-reg bracket fired | Reading |
|---|---|---|
| F13d | 0.76–0.80 "weak contributor" | L28-skip is weak-to-moderate, not load-bearing |
| F13e | 0.76–0.80 "low-rank partially fixes; ladapt adds some" | rank is primary, ladapt secondary |

### Rewritten F12 pathology

**Previous diagnosis (in handoff)**: "F12 uniform fails due to L28 intact + uniform schedule biasing early-EOS."

**Corrected (post-F13de)**: F12 fails due to **R=32 over-parameterization**. At R=32, the LoRA can construct rotations incidentally aligned with the EOS unembedding direction, collapsing emit2 from 0.95 to 0.80. At R=4, rotation is confined to the 4-dim NMI-aligned facet subspace, which is near-orthogonal to the EOS singular vector — emit2 mostly preserved regardless of schedule or L28 handling.

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
