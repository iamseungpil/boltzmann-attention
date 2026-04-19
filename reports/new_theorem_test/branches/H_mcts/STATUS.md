# H-MCTS — Attention-subspace exploration Status

**Status**: **dormant** (2026-04-19). Depended transitively on F14 pilot positive OR H-Meta ≥ +2pp — both closed by F13b +3.85pp. **Do not execute.** Revive only if post-F13 ablation work reveals need for subspace exploration (unlikely given F13b's uniform +3.7-4.1pp across all metrics without exploration noise).
**Primary spec**: memory `user_5question_synthesis_f14_metafocus_2026_04_19.md` §5 H-MCTS
**Concept**: `α_explore * noise in B_ont⊥` (subspace-restricted exploration to control search-space explosion).
**Sweep**: α_explore ∈ {0, 0.05, 0.1, 0.2} — 4-cell grid on MetaTool Subtask4 N=200.
**Null control**: noise in B_ont-aligned subspace (no exploration; pure perturbation to match ‖Δ‖).
**Gate**: F14 pilot positive OR H-Meta ≥ +2pp (need some signal that meta-layer exists before adding exploration).
**Decision**: `α_explore > 0 improves multi-tool coverage (Subtask4)` → F14 MCTS component validated; AlphaGo-attention bridge confirmed.
**Risks**:
  - **Reward Goodhart** if reward is "ontology coverage" — token spam. Use H-Energy-inspired intrinsic signal instead.
  - **Search-space** even restricted to B_ont⊥ is L×H×T — may still be too large. Profile first.
**Cost**: 4–6 GPU-hr.
**Log**: (empty)
