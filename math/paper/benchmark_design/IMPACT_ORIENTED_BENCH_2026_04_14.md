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

## 7. Verification gate — MUST clear before executing Section 5 matrix

Added 2026-04-14 after coworker codex N=20 closed-set smoke flipped the sign of the a0.3 effect on MetaTool Subtask1 (original: no_steer 0.75 / a0.3 0.65; opaque: 0.85 / 0.75). The +11.15pp first_line headline (see memory `metatool_subtask1_first_signal_2026_04_09`) is under suspicion of being a parser/answerability artifact.

**Gate conditions (all four must resolve before expanding to compositional / scaling / transfer axes):**

| # | Experiment | Pass condition | Fail consequence |
|---|---|---|---|
| G1 | Full N=995 closed-set logprob × {no_steer, a0.3 real, a0.3 random, a0.3 featshuffle} × {original, opaque}; McNemar paired | a0.3 real − no_steer ≥ +2pp AND real > random > featshuffle holds | Retract accuracy headline; shift paper thesis to mechanism-only |
| G2 | Teacher-forced vs greedy vs beam per-sample decomposition on same 995 | Identify which of (parser artifact) / (decoding dynamics) / (real selection) owns the effect | Prevents "parser artifact" mis-label of a real decoding-bias effect |
| G3 | Uncertain-subset (no_steer top-1 margin bottom 30%) re-measurement | Effect size ≥ full-set effect in uncertain subset | Confirms no_steer saturation is hiding signal, if present |
| G4 | ΔLP per-tool-category distribution | Effect concentrated in semantically multi-facet categories, not uniform | Directly seeds F-simultaneous compositional story for Section 5 axis A |

**If G1 fails, paper thesis pivots from:**
> "K-side ontology bias yields accuracy gains on tool selection distinct from Q-side steering"

**to:**
> "K-side ontology direction is geometrically specific (vs random/featshuffle) and causally identifiable via activation patching even when closed-set accuracy gain is modest or absent"

The Section 5 matrix still executes under the pivoted thesis — compositional bench / scaling curve / activation patching / zero-shot transfer remain valuable. Only the headline accuracy number is retracted.

**Probabilistic bet (2026-04-14 snapshot):**
- Headline-accuracy main-track route: 30–40% survival
- Mechanism-route (geometric specificity + causal): 70%+ survival
- Shift effort toward mechanism-route evidence (axes C + D in Section 5).

**Scripts status:** codex's label_logprob scorer lives on a separate worktree (`ba-ocq-develop/`). Our checkout's `scripts/ocq/eval_metatool_subtask1.py` is generation-based only. Either merge codex branch or reimplement scorer locally as a `--scorer label_logprob` flag. Reimplementation is ~100 LoC.

## 8. Status

- Doc created 2026-04-14.
- Verification gate (§7) added 2026-04-14 after codex smoke result.
- Supersedes the tool-selection-only framing in `phase_b_tool_selection_plan` for benchmark section (keeps week-1 kill-switch gating; does not change pivot decision).
- Section 5 matrix execution is **BLOCKED on §7 gate**.
