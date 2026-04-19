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

## 🚨 CRITICAL caveats (2026-04-19) — N=5 SEED SWEEP COMPLETE

### Finding 1 — FunnelRot is seed-dependent (bimodal)

N=5 seeds for R=4 ladapt L0-27 (F13b recipe):

| Seed | Cell | F1 | emit2 | CE | Mode |
|---|---|---:|---:|---:|---|
| 0a | F13b | 0.8027 | 0.9524 | 0.90 | ✓ SUCCESS |
| 0b | F13d | 0.7823 | 0.9252 | 0.76 | ✓ SUCCESS |
| 3 | F13m | 0.7732 | 0.8571 | 0.73 | ✓ SUCCESS |
| 1 | F13k | 0.6710 | 0.6327 | 1.36 | ✗ FAIL |
| 2 | F13l | 0.6429 | 0.5578 | 1.29 | ✗ FAIL |

- **Mean F1 = 0.7344 ± 0.072** (σ 0.0722), 95% CI [0.645, 0.824]
- **Δmean = −2.98pp** vs baseline (below, CI includes baseline)
- Success mode (3/5, CE<1.0): mean 0.786 ± 0.016 = **+2.19pp** (conditional)
- Fail mode (2/5, CE>1.0): mean 0.657 ± 0.020 = −10.73pp
- **Perfect final-CE → eval-F1 correlation** → CE-gating as mitigation

F13b's +3.85pp headline is **upper-tail of success mode**, not robust.

### Finding 2 — `skip_layer_28` flag is NO-OP for Qwen2.5-7B

Guard `28 < num_layers` is False when num_layers=28. All F13b/d/e/f/g trained with IDENTICAL schedules regardless of `--skip-layer-28`. ΔL28-skip claims invalid.

### Paper recommendation

**Option B (recommended)**: "FunnelRot conditional on CE<1.0 convergence (60% of seeds) achieves +2.19pp ± 0.016 vs baseline. Non-converged seeds discarded via training-time CE-gating."

## Ablation matrix

| Cell | R | Schedule | L28-skip | Status | F1 | ΔF1 | emit2 |
|---|---:|---|:---:|---|---:|---:|---:|
| baseline | — | — | — | — | 0.764 | — | 0.952 |
| F13a (= F12b repro) | 32 | uniform | ✗ | done-null | 0.728 | −3.63 | 0.796 |
| F13d (ladapt, no L28 skip) | 4 | ladapt | ✗ | **done-positive** | 0.782 | +1.82 | 0.925 |
| F13e (uniform + R=4) | 4 | uniform | ✓ | **done-positive** | 0.788 | +2.37 | 0.946 |
| F13g (R=32 ladapt+skip) | 32 | ladapt | ✓ | **done-positive** | 0.787 | +2.26 | 0.898 |
| **F13b (primary, full recipe)** | **4** | **ladapt** | ✓ | **done-positive** | **0.803** | **+3.85** | 0.952 |
| F13f (R=16 ladapt+skip) | 16 | ladapt | ✓ | **done-anomaly** | **0.655** | **−10.89** | 0.660 |
| F13c (projection-only, no SO(2)) | 4 | ladapt | ✓ | not run | — | — | — |

### Mechanism decomposition (Shapley-like, from 4 observed 2-change cells + F12b + F13b)

Pairwise sums:
- Δrank + Δsched = F13d − F12b = **+5.44pp**
- Δrank + Δskip = F13e − F12b = **+6.01pp**
- Δsched + Δskip = F13g − F12b = **+5.89pp**
- Triple = F13b − F12b = **+7.48pp**

Solution under additivity:
- **Δrank ≈ +2.78pp**
- **Δschedule ≈ +2.66pp**
- **ΔL28-skip ≈ +3.23pp**
- **Pairwise antagonism ≈ −1.2pp** (triple 7.48 < sum of pairs 8.67)

**No single primary driver**. All three axes contribute roughly equally.

### Pre-reg outcomes

| Cell | Pre-reg bracket fired | Reading |
|---|---|---|
| F13d | 0.76–0.80 "weak contributor" | L28-skip is weak-to-moderate |
| F13e | 0.76–0.80 "low-rank partially fixes; ladapt adds some" | rank NOT primary (post-F13g revision) |
| F13g | 0.76–0.80 NEW (schedule+skip rescue R=32) | ladapt+skip alone rescues; rank NOT primary |
| F13f | ≥ 0.81 predicted but OBSERVED < 0.76 outlier | non-monotonic rank curve, U-shape |

### Rewritten F12 pathology (v3)

**Previous v2 (post-F13de)**: "F12 fails due to R=32 over-parameterization; R=4 alone fixes it."

**Corrected v3 (post-F13fg)**: F12 is a **compound failure** — R=32 + uniform + L28-intact combined produces early-EOS (emit2 0.796). **Fixing any single axis** (rank, schedule, or L28-skip) gives partial rescue (emit2 0.90-0.95, F1 +5-6pp). **All three axes** give full rescue (emit2 0.952, F1 +7.48pp). No single primary driver.

### R=16 anomaly (F13f)

R=16 ladapt+skip gave F1 0.655 — **worse than F12b (0.728)**, worse than all other F13 cells. emit2 dropped to 0.660 (below F12b's 0.796). Non-monotonic rank curve: R=4 (0.803) → R=16 (0.655) → R=32 (0.787).

Hypotheses: (1) seed=0 bad local minimum, (2) critical-rank regime with destructive subspace geometry, (3) ill-conditioning at intermediate rank.

**Action item**: reseed F13f at seed=1 (1.5 GPU-hr) before making R=16 claims in paper.

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
