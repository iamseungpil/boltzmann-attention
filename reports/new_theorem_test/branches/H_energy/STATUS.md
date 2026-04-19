# H-Energy — Hopfield energy well Status

**Status**: spec-draft (2026-04-19). Postprocess only; zero GPU.
**Primary spec**: memory `cognitive_geometric_reframe_group6_2026_04_19.md` §4.H-Energy
**Data source**: saved tensors from F9 (MetaTool Subtask1 N=200), F10 (4 cells), F11 (4 cells at α*=0.02). All in `reports/new_theorem_test/`.
**Formula**: `E(q; K) = -lse(qK^T / √d)`. Measure `ΔE_gt - ΔE_distractor` intervention before/after.
**Decision**:
  - `ΔE gap > 0 consistently across cells` → 6A/6B (Ramsauer Hopfield / Hoover Energy Transformer) mechanistic anchor; include in paper §5.X Discussion even if F12/F13 null
  - `ΔE gap noisy / zero` → energy framing is metaphor only, not load-bearing
**Gate**: None (postprocess; run anytime existing tensors accessible).
**Cost**: 0 GPU-hr (postprocess). ~30 min CPU for all cells.
**Log**: (empty)
