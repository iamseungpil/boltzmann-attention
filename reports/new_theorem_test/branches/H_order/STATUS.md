# H-Order — Ontology-ordering canary Status

**Status**: spec-draft (2026-04-19). Rationale in memory; script NOT written.
**Primary spec**: memory `cognitive_geometric_reframe_group6_2026_04_19.md` §4.H-Order
**Task**: MetaTool Subtask4 N=200. 10 random-ordered exemplar permutations vs 1 ontology-ordered.
**Null control**: shuffled ontology labels (keep permutation structure, randomize axis labels).
**Bonus metric**: head commutator `‖[P_verb, P_domain]‖_F` vs ordering-sensitivity correlation. `r > 0.3` ⇒ 비가환성 실증 (6C/6G anchor).
**Gate**: None (cheap canary, 1–2 GPU-hr, runs anytime after F12/F13).
**Decision**:
  - `ontology ≥ 90th %ile of random` → gate opens for H-Trajectory + F14 pilot
  - `ontology in 50–90th %ile` → weak; further design needed
  - `ontology < 50th %ile` → Group 6 axis dormant; F14 scope abandoned
**Cost**: 1–2 GPU-hr (prompt-level only; no weight hook)
**Log**: (empty)
