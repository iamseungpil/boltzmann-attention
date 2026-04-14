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
| G1 | Full N=995 closed-set logprob × {no_steer, a0.3 real, a0.3 random, a0.3 featshuffle} × {Qwen, Llama}; McNemar paired | a0.3 real − no_steer ≥ +2pp on BOTH models AND real > random > featshuffle holds | Retract accuracy headline; shift paper thesis to mechanism-only |
| G2 | Teacher-forced vs greedy vs beam per-sample decomposition on same 995 | Identify which of (parser artifact) / (decoding dynamics) / (real selection) owns the effect | Prevents "parser artifact" mis-label of a real decoding-bias effect |
| G3 | Uncertain-subset (no_steer top-1 margin bottom 30%) re-measurement | Effect size ≥ full-set effect in uncertain subset | Confirms no_steer saturation is hiding signal, if present |
| G4 | ΔLP per-tool-category distribution | Effect concentrated in semantically multi-facet categories, not uniform | Directly seeds F-simultaneous compositional story for Section 5 axis A |
| G5 | Mistral skipL0+padmax label_logprob survival | a0.3 ≥ −8pp (i.e. ≥90% of generation-scorer recovery retained) | B_ont construction fix does not transfer to stricter scorer; re-examine min-truncation hypothesis |
| G6 | Mistral-Instruct H2 validation | Instruct variant closes ≥5pp of the remaining −4.32pp vs base | Confirms 14%-base-weakness decomposition; completes Mistral diagnosis |

**If G1 fails, paper thesis pivots from:**
> "K-side ontology bias yields accuracy gains on tool selection distinct from Q-side steering"

**to:**
> "K-side ontology direction is geometrically specific (vs random/featshuffle) and causally identifiable via activation patching even when closed-set accuracy gain is modest or absent"

The Section 5 matrix still executes under the pivoted thesis — compositional bench / scaling curve / activation patching / zero-shot transfer remain valuable. Only the headline accuracy number is retracted.

**Probabilistic bet (2026-04-14 snapshot):**
- Headline-accuracy main-track route: 30–40% survival
- Mechanism-route (geometric specificity + causal): 70%+ survival
- Shift effort toward mechanism-route evidence (axes C + D in Section 5).

**Scripts status (updated 2026-04-14 after github archive inspection):**
- Codex's `label_logprob` scorer is NOT on any pushed branch. The `ba-ocq-develop` worktree is codex's local dir and unpushed. We cannot audit codex's implementation.
- `origin/archive/ocq-e8-2026-04-10` does contain codex's `first_line` parser-safe scorer + `make_control_b_ont.py` (random_orthonormal, feature_shuffle). This is cherry-pick-ready.
- We have reimplemented `label_logprob` locally (`--scorer label_logprob --lp-normalize {sum,mean}`) in our develop. Our N=20 smoke gives +10pp (sum) / +5pp (mean) — **opposite sign from codex's −10pp**. Sign discrepancy is most likely an implementation-detail difference (tokenization, prompt boundary, length normalization, BOS handling), not a genuine artifact question.

**Scorer sensitivity — headline is not scorer-invariant (new G1 requirement):**

| Scorer | N | no_steer | a0.3 | Δ |
|---|---|---|---|---|
| substring_any (legacy) | 995 | 75.58% | 86.73% | +11.15pp |
| first_line (parser-safe, codex archive) | 995 | 73.57% | 83.12% | +9.55pp |
| label_logprob sum (our reimpl) | 20 | 30% | 40% | +10pp |
| label_logprob mean (our reimpl) | 20 | 15% | 20% | +5pp |
| label_logprob (codex, impl unseen) | 20 | 75% | 65% | **−10pp** |

G1 is revised: the paper must report **at least two scorers** (first_line full 995 + label_logprob full 995 in both sum and mean normalization) with explicit sensitivity discussion. A single "closed-set accuracy" number hides a 20pp swing across scorer variants.

**Cross-model status (CORRECTED — see `reports/CROSS_MODEL_KBIAS_ANALYSIS_2026_04_13.md`):**

Prior "cross-model dead" framing was obsolete. Actual state:

| Model | Mode | n_kv | no_steer | a0.3 | Δ | verdict |
|---|---|---|---|---|---|---|
| Qwen2.5-7B | C | 4 | 75.58% | 86.73% | **+11.16pp** | alive |
| **Llama-3.1-8B** | **A** | **8** | **80.60%** | **90.85%** | **+10.25pp** | **alive** |
| Mistral-7B-v0.3 (original) | A | 8 | 61.01% | 29.15% | −31.86pp | diagnosed |
| Mistral-7B-v0.3 (skipL0+padmax) | A | 8 | 61.01% | 56.68% | **−4.32pp** | 86% recovered |

**2-family positive confirmed across Mode C (Qwen GQA n_kv=4) and Mode A (Llama GQA n_kv=8).** Mistral failure fully decomposed:
- 86% = B_ont construction defect (min-truncation: L0_H2 rank=3 forces r_ont=13 globally). Fix: skipL0+padmax — validated.
- 14% = Mistral base fragility (no_match rate 36.6% vs Llama 17.1% / Qwen 4.6%).
- Mode A/C, sink position, truncation alone all ruled out as causes.

Implications for §5 matrix:
- Scaling curve must be run on **both Qwen and Llama families** (not Qwen-only).
- Mistral-Instruct H2 validation is a required experiment (confirms base-weakness = 14% component).
- "Honest counterexample with mechanistic diagnosis" is now a paper strength, not a weakness.

**Control-basis fairness critique (codex's `make_control_b_ont.py`):**
- `feature_shuffle` is row-permutation of real B_ont — preserves Frobenius norm but destroys axis semantics. Collapses K onto arbitrary directions; more destructive than random by construction. "real > random > featshuffle" ordering is therefore partially tautological and should not be the main evidence for geometric specificity.
- `random_orthonormal` uses fresh random + QR per (L,H) but does **not** match intervention norm. Real B_ont has rank-1 heads where ‖α·B·Bᵀ·k‖ is large; random can be energetically quieter or louder, making the comparison unfair.
- **Fix:** add a **norm-matched random control** — rescale each (L,H) random basis so ‖α·B_rand·B_randᵀ·k_layer_mean‖_F matches ‖α·B_real·B_realᵀ·k_layer_mean‖_F on a held-out dev slice.

## 8. Currently-executing experimental pipeline (2026-04-14)

Status snapshot: both GPUs at 99% utilization; three waves auto-chained.

- **Wave 1** (GPU0+GPU1, in progress): Qwen2.5-7B × real B_ont × {sum, mean} scorer × full 995. ETA ~50 min from 18:05.
- **Wave 2** (GPU0+GPU1, queued, chain PID 343600): Qwen × {random, featshuffle} controls × {sum, mean} × full 995. ~1.7 h.
- **Wave 3a** (GPU0+GPU1, queued, chain PID 356966):
  - GPU0: Llama-3.1-8B × real B_ont × {sum, mean} scorer.
  - GPU1: Mistral-7B-v0.3 × skipL0+padmax B_ont × {sum, mean}.
- **Wave 3b** (GPU0+GPU1, queued after Wave 3a):
  - GPU0: Llama × {random, featshuffle} controls × {sum, mean}.
  - GPU1: Mistral-7B-**Instruct**-v0.3 × skipL0+padmax × {sum, mean} (H2 base-weakness validation).

Total wall-clock for §7 gate resolution: ~8–10 h. Resolves G1, G5, G6 on two models plus Mistral counterexample. G2 (scorer decomposition), G3 (uncertain subset), G4 (per-category) are derivable from the persample JSONL dumps after these runs.

## 9. Status

- Doc created 2026-04-14.
- Verification gate (§7) added 2026-04-14 after codex smoke result.
- Cross-model section corrected 2026-04-14 after finding `CROSS_MODEL_KBIAS_ANALYSIS_2026_04_13.md`: 2-family positive confirmed, Mistral fully diagnosed.
- Pipeline extended 2026-04-14 to include Llama label_logprob + controls + Mistral skipL0+padmax + Mistral-Instruct H2.
- Supersedes the tool-selection-only framing in `phase_b_tool_selection_plan` for benchmark section (keeps week-1 kill-switch gating; does not change pivot decision).
- Section 5 matrix execution is **BLOCKED on §7 gate** (G1+G5+G6 minimum).
