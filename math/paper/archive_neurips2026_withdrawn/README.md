# NeurIPS 2026 Submission — WITHDRAWN (2026-04-19)

## Status
**Withdrawn from NeurIPS 2026 submission cycle.** Project effort consolidated
onto the ICLR 2027 single track at `math/paper/iclr2027/PAPER_DRAFT_ICLR_v1.md`.

## Withdrawal date
2026-04-19

## Reasons for withdrawal

1. **Mechanism overreach in §1.0 falsified by own data.**
   The "SEKA-class K-side spectral steering is structurally incapable of
   multi-selection" claim is directly refuted by Telecom canonical-AdaSEKA
   form +28.89pp (a stationary K-side operator producing measurable
   multi-tool lift). Pivoting away from the mechanism claim leaves the
   thesis hollow.

2. **§4 Facet-Gated K-Bias Operator structure mathematically void.**
   F8d 7-dataset NMI probe established that τ²-bench's 4-facet ontology
   has effective independent-axis count ≈ 1.3 (not 4). Phase C catalog
   permutation (4 variants: real / facet_values_perm / tool_names_perm /
   full_random) all produced attn_fro ratio ∈ [1.98, 2.69]. Catalog
   *content* is not load-bearing; the load-bearing object is the
   K-subspace at tool-related token positions, which is a Gram-Schmidt
   span-invariant pipeline output. The "facet-gated" framing of §4 is
   thus mathematically void within the linear $\delta K = \alpha BB^\top K$
   regime.

3. **Phase D static-weight $d^*$ family closed (Failure verdict).**
   Per-(L, h, q) angular alignment of $d_{\text{emp}}$ against H₁
   (lm_head pull-through) and H₃ ($W_K$ top-SV) on Qwen × τ² {Telecom,
   Retail} × N=50 each (~4000 measurements per hypothesis): H₁ at
   0.86–0.91× random baseline, H₃ at 1.10–1.23×. 0% of heads pass 30°
   threshold for either. Static-weight-geometric mechanism explanation
   for Q-sign asymmetry is ruled out.

4. **Multiple in-text retractions undermine reviewer trust.**
   §5.4.1.1 retracts the original Thm 6.1 cross-model decomposition.
   §5.5.2 retracts the contrastive K-bias smoke +5.8pp signal as
   small-N variance artifact. §3.6.4 admits Thm 6.21 was "formalized
   after observing" §5.4.1.1 measurements. 1500+ line draft length
   amplifies these credibility risks.

5. **Deadline pressure with no path to recover trust capital.**
   4-claim pivot (per `handoff_paper_edit_2026_04_19.md`) requires 1+
   week rewrite under D-29 NeurIPS deadline. Marginal benefit of pivot
   (P=0.45 × score=5.5 = 0.45 ceiling) is dominated by reallocating
   the same week to ICLR strengthening (6.25 → 7.0).

## What was retained (for cherry-pick into ICLR or future venues)

The following experiment artifacts and theoretical results from this
withdrawn track are still active assets:

- **All `reports/` data** — preserved at repository root, not archived.
  Cited from ICLR draft Discussion / Appendix as needed.
- **`external/SEKA/seka_projections/` B_ont files** — reused in ICLR
  Phase B1 + B2 experiments.
- **Theorem 6.13 (OCQ Categorical-Channel Optimality)** — orthogonal
  contribution; candidate for separate ICML 2027 / workshop submission
  on KV-cache compression (not in scope of either ICLR or NeurIPS
  steering thesis).
- **`reports/steering_paper/` markdown** — left in place as it contains
  shared experiment plans and audit notes used by both tracks.

## What was archived here

- `benchmark_design/` — all NeurIPS PAPER_DRAFT_v1/v2/v3 markdown +
  IMPACT_ORIENTED_BENCH_2026_04_14.md.
- `paper/archive_neurips2026_withdrawn/neurips2026_*` — three LaTeX
  build directories (English, Korean, steering Korean variants).

## Active track going forward

**ICLR 2027** — `math/paper/iclr2027/PAPER_DRAFT_ICLR_v1.md`
Two-Level Argmax-Subspace Selectivity in Pretrained Transformers.
Phase A (universality) + Phase B1 (cross-bench) + Phase B2 (mechanism
location) + Phase C (falsifier-as-refinement) + Phase D (Failure
verdict, transparent reporting) + Appendix A.1 (Banking OOD) + B.1
(Phase D negative). Pending: §5.7 H-J breadth scatter, §5.X NMI
diagnostic merge, F9 MetaTool VD/flat completion, Phase B2
cross-model verification (Llama, Mistral), F10 per-facet α.
