# Experiment Branch Tracking — Brainstorm-driven (2026-04-19 —)

**Purpose**: Track all experimental branches spawned from 2026-04-19 brainstorm sessions.
Each branch has an independent execution path, gate, status, and artifact dir under `branches/`.

**Mode**: Brainstorm / pre-reg only. No GPU execution until gates clear.
**Paper target**: ICLR 2027 (single track; NeurIPS 2026 withdrawn).
**Authoritative spec source**: `math/paper/lie_group/NEW_THEOREM_TEST.md`.

---

## §1. Branch registry

| ID   | Name                       | Origin                     | Status        | GPU-hr    | Gate (precondition)                         | Artifact dir                      |
|------|----------------------------|----------------------------|---------------|-----------|---------------------------------------------|-----------------------------------|
| F12  | FacetRot-QK                | F11 null (Phase F)         | spec-ready    | 8–10      | None (primary alt, direct)                  | `branches/F12_facetrot_qk/`       |
| F13  | FunnelRot (ladapt + L28)   | F12 + B2 L28 + Thm 6.14    | spec-ready    | 12–18     | None (parallel/superset of F12b)            | `branches/F13_funnelrot/`         |
| H-Ord| Ontology-Ordering canary   | Group 6 / 5Q Q1            | spec-draft    | 1–2       | None (cheap canary)                         | `branches/H_order/`               |
| H-En | Hopfield Energy well       | Group 6 (6A/6B)            | spec-draft    | 0 (pp)    | F9/F10/F11 saved tensors exist              | `branches/H_energy/`              |
| H-Tr | Residual Trajectory        | Group 6 (6D/6H/6J)         | spec-draft    | 4–6       | H-Order ≥ +2pp (signal exists)              | `branches/H_trajectory/`          |
| H-Mt | Meta-attention layer       | 5Q Q2+Q5                   | preliminary   | 10–15     | F12<+3pp AND F13<+3pp                       | `branches/H_meta/`                |
| H-MC | Attention-subspace MCTS    | 5Q Q3                      | preliminary   | 4–6       | F14 pilot positive OR H-Mt ≥ +2pp           | `branches/H_mcts/`                |
| H-HT | Higher-order HOT ablation  | 5Q Q5                      | preliminary   | 12–18     | F14 pilot positive                          | `branches/H_hot/`                 |
| F14  | MetaFocus synthesis pilot  | 5Q integration             | preliminary   | 8–12      | F12<+3pp ∧ F13<+3pp ∧ H-Ord≥+2pp            | `branches/F14_metafocus/`         |
| E1   | Engineering-Interface BM   | 2026-04-19 reframe session | spec-draft    | 2–4       | Runs alongside F12/F13 (piggyback)          | `branches/E1_engineering_reframe/`|

**Status legend**:
- `spec-ready` — protocol written, scripts committed (or mostly committed), runnable once approved
- `spec-draft` — rationale + 4-outcome pre-reg exists in memory/paper, script NOT written
- `preliminary` — concept only; §2.5 of NEW_THEOREM_TEST.md; spec pending
- `in-progress` — GPU execution underway
- `done-positive` / `done-null` / `done-falsified` — terminal
- `dormant` — gate failed; archive branch

**Current state snapshot (2026-04-19)**: 2 `spec-ready` (F12, F13), 4 `spec-draft` (H-Order, H-Energy, H-Trajectory, E1), 4 `preliminary` (H-Meta, H-MCTS, H-HOT, F14).

---

## §2. Dependency graph

```
        ┌─────────────────────────────────────────────────────┐
        │  PRIMARY TRACK (F-series, K-bias attention)         │
        │                                                     │
        │  F12 ── (null) ──► F13 ── (null) ──┐                │
        │   │                 │              ▼                │
        │   │                 │         ┌─── H-Order ─┐       │
        │   │                 │         │             │       │
        │   ▼                 ▼         ▼             ▼       │
        │  done-pos         done-pos   H-Energy    (null)     │
        │  (paper §5)       (paper §5) (postproc)  dormant    │
        │                                 │                   │
        │                                 ▼                   │
        │                              H-Trajectory           │
        │                                 │                   │
        │                                 ▼                   │
        │                              F14 pilot              │
        │                                 │                   │
        │                          ┌──────┼──────┐            │
        │                          ▼      ▼      ▼            │
        │                        H-Meta H-MCTS H-HOT          │
        └─────────────────────────────────────────────────────┘

        ┌─────────────────────────────────────────────────────┐
        │  CROSS-CUT TRACK (E1, engineering-interface)        │
        │                                                     │
        │  E1 runs piggyback on F12/F13 inputs, emits         │
        │  cost/variance/HP-dim metrics. Independent of       │
        │  gate; outputs feed §1/§6 reframe.                  │
        └─────────────────────────────────────────────────────┘
```

---

## §3. Execution priority queue (post-F11 falsification, 2026-04-19)

| Order | Branch   | Rationale                                                     |
|-------|----------|---------------------------------------------------------------|
| 1     | F12      | Primary alt to F11. Complete HF hook + train + eval.          |
| 2     | F13      | Strict superset of F12b. Run parallel or sequential.          |
| 3     | H-Energy | Postprocess-only. Zero GPU cost. Run anytime.                 |
| 4     | H-Order  | Cheap canary (1–2 hr). Run after F12/F13 unless either +3pp.  |
| 5     | E1       | Piggyback on F12/F13 generation. Add CoT/ToT baselines.       |
| 6     | H-Traj   | Only if H-Order positive.                                     |
| 7     | F14      | Only if full gate (F12<+3 ∧ F13<+3 ∧ H-Ord≥+2).               |
| 8+    | H-Meta / H-MCTS / H-HOT | F14 ablations. After F14 pilot.                |

---

## §4. Per-branch status detail

Each branch has a SPEC.md under its `branches/<id>/` directory.
Status updates, artifacts (JSON, tensors), and run logs go in the same dir.

### §4.F12 — FacetRot-QK
- Spec: `NEW_THEOREM_TEST.md §5` + memory `phase_f12_facetrot_qk_spec_2026_04_19.md`
- Script: `scripts/new_theorem_test/train_f12_facetrot_qk.py` (skeleton, 8 TODOs)
- Unblocker: build `build_f12_facet_subspace.py` → HF hook registration → GQA reshape → RoPE commute → MetaTool loader → CE loss → Lipschitz penalty
- Cell plan: F12a uniform α=0.05 / F12b uniform α=0.10 / F12c ladapt
- Decision: `+3pp` → paper §5.X primary; `0–3pp` → partial, see F13; `null` → gate F13/H-Ord

### §4.F13 — FunnelRot
- Spec: `NEW_THEOREM_TEST.md §5` + memory `phase_f13_funnelrot_spec_2026_04_19.md`
- Superset of F12b via `--schedule ladapt --rot-pairs 2 --skip-layer-28`
- Cells: F13a (=F12b repro) / F13b primary ladapt+R4+skip-L28 / F13c projection-only / F13d L28-intervene negative / F13e uniform ablation / F13f R=16 ablation
- Decision: F13b > F13a by ≥ +2pp → ladapt-schedule is contributory; else ladapt non-load-bearing

### §4.H-Order — Ontology-ordering canary
- Spec: memory `cognitive_geometric_reframe_group6_2026_04_19.md` §4.H-Order
- Task: MetaTool Subtask4 N=200, 10 random perms vs 1 ontology-ordered, prompt-level only
- Bonus: head commutator `‖[P_verb, P_domain]‖_F` vs ordering sensitivity correlation (r > 0.3 → 비가환성 실증)
- Decision: `ontology ≥ 90th %ile random` → proceed to H-Trajectory + F14 pilot; else Group 6 dormant

### §4.H-Energy — Hopfield energy well
- Spec: memory `cognitive_geometric_reframe_group6_2026_04_19.md` §4.H-Energy
- Postprocess saved tensors from F9/F10/F11 (no new GPU). `E = -lse(qK^T / √d)`
- Measure: `ΔE_gt - ΔE_distractor` intervention before/after
- Decision: `ΔE gap > 0 consistently` → 6A/6B mechanistic anchor for paper §5.X Discussion

### §4.H-Trajectory — Residual trajectory
- Spec: memory `cognitive_geometric_reframe_group6_2026_04_19.md` §4.H-Trajectory
- Per-step residual stream recording; ontology-guided CoT vs ad-hoc CoT divergence
- Gate: H-Order ≥ +2pp first (otherwise no signal to measure)
- Decision: `divergence_ont < divergence_ad-hoc` → follow-on paper candidate

### §4.H-Meta — Meta-attention layer
- Spec: memory `user_5question_synthesis_f14_metafocus_2026_04_19.md` §5 H-Meta
- F12/F13 infra + meta-layer (LoRA rank comparable). Null = identity rotation + identity gate
- Gate: F12 AND F13 both <+3pp (otherwise main-track dominates)
- Decision: `meta > pure-LoRA baseline ≥ +2pp` → paper §5.X novel meta-layer

### §4.H-MCTS — Subspace exploration
- Spec: memory `user_5question_synthesis_f14_metafocus_2026_04_19.md` §5 H-MCTS
- 4-cell sweep α_explore ∈ {0, 0.05, 0.1, 0.2}. Null = noise in B_ont-aligned subspace
- Decision: `α_explore > 0 improves multi-tool coverage` → F14 MCTS component validated

### §4.H-HOT — Higher-order ablation
- Spec: memory `user_5question_synthesis_f14_metafocus_2026_04_19.md` §5 H-HOT
- 3-variant: (i) stacked depth control / (ii) ontology-typed routing / (iii) step-state aware
- Decision: `(ii),(iii) ≥ (i) + 2pp` → novel meta-layer qualitatively distinct; else "depth suffices"

### §4.F14 — MetaFocus synthesis pilot
- Spec: NEW_THEOREM_TEST.md §2.5.2 v5.3 + memory `user_5question_synthesis_f14_metafocus_2026_04_19.md`
- 3-cell pilot combining meta-layer + MCTS + intrinsic reward
- Hard gate: F12<+3pp ∧ F13<+3pp ∧ H-Order≥+2pp
- MOFCISS-precedent warning: α sensitivity pre-calibration MANDATORY before full run
- Decision: `+5pp` → strong paper; `+2–5pp` → ablate; `0–2pp` → scope-limit; `negative` → dormant

### §4.E1 — Engineering-Interface Benchmark
- Spec: NEW (this session). Draft in `branches/E1_engineering_reframe/SPEC.md`
- Claim: attention-level F12/F13 ≈ CoT/ToT outcome-space, but cheaper/simpler/more standardized
- 4 measurable axes: inference-token-cost, HP-dim, paraphrase-variance, peak-accuracy
- Baselines to add: CoT (Wei 2022), ToT (Yao 2023), Self-Consistency (Wang 2022)
- Decision: if F12/F13 marginal, E1 becomes fallback paper framing (§1 Introduction reframe)
- Piggyback: runs on F12/F13 same prompts; adds ≤ 3 GPU-hr

---

## §5. Cross-branch decision matrix (consolidated)

| F12 outcome | F13 outcome | H-Order | Next primary action                      | Paper framing                              |
|-------------|-------------|---------|------------------------------------------|--------------------------------------------|
| ≥ +3pp      | any         | skip    | Write §5.X around F12; E1 for interface  | Capability + interface win                 |
| 0–3pp       | ≥ +3pp      | skip    | Write §5.X around F13                    | Capability + interface win                 |
| 0–3pp       | 0–3pp       | ≥ +2pp  | H-Energy postprocess + F14 pilot         | Narrow lift + meta-layer §5.X              |
| 0–3pp       | 0–3pp       | < +2pp  | E1 standalone; Group 6 dormant           | Engineering-interface fallback (E1 main)   |
| null        | null        | < +2pp  | Scope-limit; publish failure template    | Failure template paper                     |

---

## §6. Gate conditions (consolidated)

**Hard (must clear before execution):**
- **F14**: `F12 < +3pp` AND `F13 < +3pp` AND `H-Order ≥ +2pp`
- **H-Trajectory**: `H-Order ≥ +2pp`
- **H-Meta**: `F12 < +3pp` AND `F13 < +3pp`
- **H-MCTS, H-HOT**: `F14 pilot ≥ +2pp over baseline`

**Soft (recommended order):**
- H-Order before H-Trajectory
- H-Energy postprocess before any new GPU run (costs nothing, informs 6A/6B anchor)
- E1 piggyback on F12/F13 — no extra GPU if done during the same run

**Precedent warning (MOFCISS 2026-04-19)**: any branch with per-layer coefficient spec (H-Meta, F14) MUST include α sensitivity calibration BEFORE full-cell run. F11 α=0.3 produced F1=0; α*=0.02 after calibration. Skip calibration → null result not interpretable.

---

## §7. Change log

| Date       | Change                                                                                      |
|------------|---------------------------------------------------------------------------------------------|
| 2026-04-19 | Document created. 10 branches registered. F12/F13 spec-ready, E1 added this session.        |

Add new rows here when status transitions (`spec-draft → spec-ready → in-progress → done-*` or `dormant`).

---

## §8. Cross-references

- **Primary spec source**: `math/paper/lie_group/NEW_THEOREM_TEST.md` §5 (F12/F13), §2.5.2 (Group 6 / F14)
- **Paper draft**: `math/paper/iclr2027/PAPER_DRAFT_ICLR_v1.md`
- **Brainstorm handoffs**:
  - `handoff_2026_04_19_f12_execution.md` (F12/F13 engineering)
  - `brainstorm_handoff_2026_04_19_late_night.md` (Group 6 / F14 resumption)
  - `cognitive_geometric_reframe_group6_2026_04_19.md` (H-Order/H-Energy/H-Trajectory)
  - `user_5question_synthesis_f14_metafocus_2026_04_19.md` (H-Meta/H-MCTS/H-HOT/F14)
- **Failure precedents** (gate rationale):
  - `phase_f10_query_conditional_gating_2026_04_19.md` (F10 null)
  - `phase_f11_mofciss_executed_falsified_2026_04_19.md` (F11 null + α spec warning)
