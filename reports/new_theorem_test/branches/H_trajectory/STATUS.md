# H-Trajectory — Residual trajectory Status

**Status**: preliminary (2026-04-19). Spec rationale only; protocol detail pending.
**Primary spec**: memory `cognitive_geometric_reframe_group6_2026_04_19.md` §4.H-Trajectory
**Task**: MetaTool / BFCL N=100 per-step residual stream recording. Compare ontology-guided CoT vs ad-hoc CoT trajectory divergence.
**Metric candidates** (TBD):
  - Per-layer `‖r_t - r_{t-1}‖` stream (step-wise movement)
  - Path length = ∑ ‖r_t - r_{t-1}‖
  - Lyapunov-style divergence: ‖r_t^{ont} - r_t^{ad-hoc}‖
**Gate**: **H-Order ≥ +2pp** (no point measuring trajectory if ordering has no effect).
**Decision**:
  - `divergence_ont < divergence_ad-hoc consistently` → ontology-guided residual stream more stable; separate follow-on paper candidate
  - `no systematic difference` → trajectory framing dormant
**Cost**: 4–6 GPU-hr (per-step residual stream recording needs larger memory footprint).
**Log**: (empty)
