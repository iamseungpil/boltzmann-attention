# F14 MetaFocus — Synthesis pilot Status

**Status**: **dormant** (2026-04-19). Gate closed by F13b +3.85pp (triple-gate requires F12<+3 ∧ F13<+3 ∧ H-Ord≥+2; F13 ≥ +3pp breaks gate). **Do not execute.** Revive only if ablations (F13d/e) reveal F13b result was cocktail-artifact and real lift < +3pp, or if cross-bench replication fails broadly.
**Primary spec**: `NEW_THEOREM_TEST.md §2.5.2 v5.3 (6M)` + memory `user_5question_synthesis_f14_metafocus_2026_04_19.md` §4
**Architecture sketch**:
```
Frozen Qwen2.5-7B
├── Base attention (L0-L27) unchanged
└── Meta layer (F12/F13 extension):
    ├── Ontology-axis Q/K rotation R_ont(axis, step)
    ├── Exploration bonus α_explore · N(0,σ²) in B_ont⊥  (H-MCTS component)
    ├── Intrinsic reward: ontology coverage / Hopfield E decrease / trajectory divergence
    └── Hierarchical meta-observation indexed by axes  (H-HOT component)
```
**Hard gate** (ALL three required):
  - F12 < +3pp
  - F13 < +3pp
  - H-Order ≥ +2pp
**3-cell pilot** (post-gate):
  1. Meta-only (H-Meta component)
  2. Meta + MCTS exploration (H-Meta + H-MCTS)
  3. Full synthesis (all components)
**Decision**:
  - `+5pp vs baseline` → strong paper §5.X primary claim
  - `+2–5pp` → ablation required; identify which component contributes
  - `0–2pp noise` → component gains cancel; scope-limit
  - `negative` → F14 dormant; Group 6 axis archived
**MOFCISS precedent warning**: F14 has 5 tuning axes (meta rotation / MCTS α / intrinsic reward weight / hierarchical depth / ontology order). MUST run α-sensitivity calibration BEFORE full 3-cell pilot. F11 α=0.3 catastrophic (F1=0); α*=0.02 after calibration. Skip calibration → null result uninterpretable.
**Cost**: 8–12 GPU-hr (pilot only; full ablations are H-Meta/H-MCTS/H-HOT separately).
**Log**: (empty)
