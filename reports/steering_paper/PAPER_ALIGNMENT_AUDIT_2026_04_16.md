# Steering Paper Alignment Audit

Date: 2026-04-16

## Findings First

### 1. The main paper was one experiment short of its own mechanism claim

Severity: high

The paper repeatedly interprets the Q-side result as a coverage effect on the second decision, but the shipped `Subtask4` evaluator had only set-level metrics. That meant the paper could show a sign-consistent aggregate gain, but not the claimed bridge from intervention sign to second-tool recovery.

Action taken:

- `scripts/ocq/eval_metatool_subtask4.py` now emits stepwise metrics:
  - `first_tool_hit_rate`
  - `second_tool_hit_rate`
  - `second_distinct_hit_rate`
  - `second_recovery_given_first_hit_rate`
  - `repeated_first_tool_rate`
  - emitted-sequence rates
- The per-sample dump now preserves the raw predicted tool sequence, not only the deduplicated set.

Consequence:

- The code is now capable of testing the paper's mechanism claim directly.
- The paper should still avoid saying that the mechanism is already proved until those stepwise numbers are actually run on the locked protocol.

### 2. The standard Subtask4 driver still exposed an AdaSEKA proxy path

Severity: high

This was the most dangerous paper-integrity issue in the evaluation stack. The repo already contained a real SEKA wrapper, but the standard Subtask4 path could still route `adaseka_*` tags to a Q-side proxy that is not the official algorithm.

Action taken:

- `scripts/ocq/eval_metatool_subtask4.py` now rejects the AdaSEKA proxy path for paper-facing Subtask4 evaluation and instructs the user to use the real external wrapper instead.

Consequence:

- This removes the easiest route to accidental proxy contamination in the main benchmark.
- The Subtask1 file still contains proxy helpers for exploratory work; they must remain out of any paper-facing baseline table.

### 3. Paper-critical runs were not auditable enough

Severity: medium

Before this audit, the result JSONs did not consistently record prompt-template identity, scorer identity, decoding policy, and runtime configuration. That is enough for an internal experiment, but not enough for a reviewer-facing matched-baseline claim.

Action taken:

- `scripts/ocq/eval_metatool_subtask4.py` now writes:
  - `prompt_template_id`
  - `scorer`
  - `decode_policy`
  - `runtime_config`
- `scripts/ocq/eval_subtask4_with_real_seka.py` now writes the same protocol metadata and the same stepwise metrics structure.

Consequence:

- Future matched SEKA tables can be audited from the result bundle itself instead of from memory or shell history.

### 4. The main paper is viable only if it stays narrow

Severity: strategic

The current evidence is enough for a clean narrow paper:

1. query-side ontology suppression is modestly positive on Subtask4 across Qwen and Llama,
2. stationary key-side amplification is negative on both and catastrophic on Llama,
3. ontology-specific null controls are far stronger than the raw gain,
4. the perturbation theorem is a diagnostic theorem, not an accuracy theorem.

It is not enough for the broader develop-branch story about step-adaptive control, large SEKA superiority, or a joint steering-compression paper.

## Verified Evidence That Remains Paper-Safe

| Axis | Safe statement | Current evidence |
|---|---|---|
| Main effect | Q-bias is positive and K-bias is negative on Subtask4 across Qwen/Llama | `STEERING_EVIDENCE_SUMMARY_2026_04_16.md` |
| Specificity | Real ontology basis beats shuffled/random controls on Qwen | same |
| Operating regime | Useful Q-bias region is narrow around `beta=-0.1` | same |
| Theory scope | Perturbation bound is conservative and diagnostic | same |
| Closest-prior status | Real matched SEKA/AdaSEKA table is still required before any superiority claim | same + `COWORKER_REQUEST_2026_04_15.md` |

## Paper Claims That Must Stay Out

- Any claim that exact step-adaptive emitted-facet tracking is already implemented.
- Any claim that AdaSEKA has been fairly beaten without an official matched wrapper.
- Any claim that compression is a main validated contribution of the current steering paper.
- Any claim that the perturbation theorem predicts benchmark accuracy directly.

## Code Changes Completed In This Audit

| File | Change | Why it matters |
|---|---|---|
| `scripts/ocq/eval_metatool_subtask4.py` | Added stepwise coverage metrics and protocol metadata | closes the largest code-paper gap |
| `scripts/ocq/eval_metatool_subtask4.py` | Blocks AdaSEKA proxy for paper-facing Subtask4 runs | prevents the most likely baseline-labeling mistake |
| `scripts/ocq/eval_subtask4_with_real_seka.py` | Added matching stepwise metrics and protocol metadata | enables auditable matched-baseline runs |

## Build Warning

Do not treat `paper/neurips2026_steering_v2/build_latex.py` as a safe rebuild path for the curated paper. In its current state it regenerates sections from `math/paper/benchmark_design/PAPER_DRAFT_v3.md`, which can overwrite the narrowed main-paper text with the broader develop draft. The safe rebuild path for the curated paper is direct LaTeX compilation from `paper/neurips2026_steering_v2/main.tex`.

## Locked Next Steps

1. Run `Subtask4` stepwise coverage decomposition first.
2. Run one matched real-baseline comparison second.
3. Extend null controls to Llama if practical.
4. Only then decide whether the paper can keep the stronger coverage framing in the main text.

## Bottom Line

The project is not in the state implied by the broad develop draft, but it is not broken either.

The right interpretation is harsher and simpler: the main branch now has a paper that is defensible because it says less. The correct path is to strengthen that narrow paper with stepwise coverage evidence and one real matched baseline table, not to import broader claims from `develop`.
