# Impact-Oriented Benchmark Redesign (2026-04-14)

Scope: Re-evaluate Phase B benchmark purely from **paper-impact** lens (NeurIPS/ICLR/ACL main track), not deployment alignment. Triggered by review of Netsru Q&A 2026-04-14 where deployment-side framing was isolated out.

## 1. Diagnosis of current plan

- Thesis (post Cor 6.7 drop, see `cor67_drop_confirmed_2026_04_10`): K-side **F-simultaneous** facet-catalog ontology bias is a geometrically/causally distinct steering mechanism from Q-side CAA/ITI/PASTA/ASA/Focus-Directions.
- Current plan (`phase_b_tool_selection_plan`): MetaTool Subtask1 + τ²-bench top-1 + cross-model. First signal +11.15pp (`metatool_subtask1_first_signal_2026_04_09`).
- Gap: benchmarks operate in **F=1 regime**; the theory's F-simultaneous claim is invisible in the numbers. Reviewer-1 objection surface.
- Current ceiling: EMNLP Findings. Not main-track.

## 2. Required axes for main-track impact

### A. Compositional / multi-facet benchmark (highest priority)
- Build or adopt a benchmark that exercises F≥2 simultaneous facets (e.g., tool × domain × schema-slot).
- Candidates: BFCL-v3 multi-turn + parallel; NexusRaven; Gorilla-v3; or self-built ~2K compositional set.
- Purpose: create a regime where Q-side 1-of-M methods structurally cannot match, making mechanism difference visible in accuracy.

### B. Scaling curve
- Qwen2.5 family 0.5B / 1.5B / 3B / 7B / 14B / 32B (accept 4-point compromise: 0.5 / 3 / 7 / 32).
- Tests emergent vs architectural vs scale-invariant. 32B is justified as a curve point, not a deployment target.

### C. Mechanistic interpretability
- Activation patching: swap B_ont columns between paired examples, measure behavioral delta.
- Per-head contribution decomposition; leverage existing L0 / L27h1 rank-1 head findings (`cor67_gate_distribution_diagnostic_2026_04_10`).
- Reframe Cor 6.7 hard-gate failure as honest negative: soft energy-ratio gate works, hard gate closes rank-1 heads to zero. This becomes a mechanism story, not a dead result.

### D. Zero-shot ontology transfer (OOD generalization)
- B_ont built on MetaTool → applied zero-shot to ToolAlpaca / τ²-bench with no rebuild.
- Reframes "continual tool addition" from deployment talking point to an **OOD generalization** experiment.
- This is the only clean scientific statement of FT-superiority.

### E. Reproduced baselines (not cited)
- CAA, ASA (2026), Focus Directions (Zhu 2025), PASTA, LoRA r=8 matched-data, RAG prompt baseline.
- Recipes already in `baseline_recipes_attention_steering`. Run on same data/compute as ours.

### F. Safety retention (mandatory appendix)
- MMLU / HH-RLHF refusal / ToxiGen before vs after steering.
- Must explicitly disambiguate: Cor 6.7 MMLU −10.50pp was **hard-gate variant only**; soft variant retains.
- Without this: desk-reject risk.

## 3. Drop (deployment bias)

- "32B because client uses it" — zero impact argument; only keep as curve point.
- Llama-Nemotron specific inclusion — unnecessary unless scaling-family coverage.
- Two-stage tool+parse verification — already in BFCL, not novel.
- Planning–Execution separation — deployment architecture, not claim-connected.
- MCP scenarios — N/A.

## 4. Reframed thesis sentence

> "Prior Q-side steering literature (CAA, ITI, PASTA, ASA, Focus-Directions) addresses 1-of-M discriminative routing. We show that in the **F-simultaneous** compositional regime — which arises naturally in tool/plan selection with multi-facet schemas — a K-side ontology basis yields a geometrically distinct, causally identifiable, and scaling-invariant steering mechanism, validated by activation patching and zero-shot ontology transfer."

## 5. Final benchmark matrix

| Axis | Content | Impact role |
|---|---|---|
| Core accuracy | MetaTool Subtask1 + **BFCL-v3 multi-turn parallel** + τ²-bench | F-simultaneous demonstration |
| Scaling | Qwen2.5 {0.5, 3, 7, 32}B curve | emergent vs arch |
| Transfer | MetaTool→ToolAlpaca zero-shot B_ont | OOD generalization |
| Mechanistic | Activation patching + per-head; ε_q / var_frac / gain_over_uniform | reviewer credibility |
| Safety | MMLU + HH-RLHF refusal + ToxiGen; soft vs hard gate disambiguation | reject defense |
| Baselines | CAA / ASA / FocusDir / PASTA / LoRA r=8 / RAG — **all reproduced** | matched comparison |

## 6. Cheapest, highest-yield pair

If only two axes can be added before submission: **(A) compositional benchmark** + **(C) activation patching**. These alone move the paper from Findings to main-track bubble.

## 7. Status

- Doc created 2026-04-14.
- Supersedes the tool-selection-only framing in `phase_b_tool_selection_plan` for benchmark section (keeps week-1 kill-switch gating; does not change pivot decision).
