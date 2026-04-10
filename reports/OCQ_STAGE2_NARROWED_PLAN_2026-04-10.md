# OCQ Stage 2 Narrowed Plan (2026-04-10)

## Why this document exists

The current Phase B document set overreaches relative to the code and evidence.
Codex review and local bounded experiments both indicate that the defensible
near-term scope is narrower:

- supported today: single-turn MetaTool + MMLU safety checks
- not supported today: tau2 / BFCL multi-turn main claim, Q-gated methods D/E,
  AdaSEKA-on-new-benchmark claims

This plan freezes the immediate intent, hypothesis, and verification logic so
the next loop stays evidence-first.

## Current factual state

### Runnable today

- `scripts/ocq/build_metatool_ontology_v2.py`
- `scripts/ocq/build_qwen_metatool_b_ont.py`
- `scripts/ocq/eval_metatool_subtask1.py`
- `scripts/ocq/eval_mmlu_subset.py`

### Not runnable today

- `eval_tau2.py` does not exist.
- `eval_multi_turn.py` does not exist.
- BFCL/Tau2 drivers are documented but not implemented.
- Q-gated / QK-gated methods D/E are documented but not implemented.

## Codex-reviewed risks

1. The paper claim and the evidence do not currently match.
   - Evidence: flat `ocq_bias_a0.3` on single-turn MetaTool.
   - Unsupported claim: context-integrated, Q-gated, multi-turn tool selection.
2. MetaTool can reward lexical copying / parser artifacts.
3. The ontology effect may be a random-subspace or language-mean effect.
4. Alpha sensitivity is large enough to threaten robustness.
5. Current facet-gated implementation is K-local, not query-aware closure.

## Intent / Hypothesis / Verification

### Intent N1 — Verify the runnable OCQ surface before making stronger claims

**Hypothesis.**
The minimal single-turn OCQ stack can be made portable and executable in a
clean worktree without hand edits during each run.

**Verification.**
- ontology builder v2 completes
- B_ont builder completes on bounded first-layer smoke
- MetaTool eval completes for `no_steer`, `ocq_bias_a0.3`, `ocq_facet_gated_a1.0`
- MMLU eval completes for the same methods

**Status.** Passed.

### Intent N2 — Test whether the current evidence really supports facet-gated OCQ

**Hypothesis.**
If the current paper direction is right, even a bounded first-layer setting
should show that `ocq_facet_gated` is at least competitive with flat bias on
tool selection while not harming MMLU.

**Verification.**
- MetaTool 50-sample bounded run
- MMLU 100-sample bounded run
- compare `no_steer` vs `ocq_bias_a0.3` vs `ocq_facet_gated_a1.0`

**Observed result.**
- MetaTool 50:
  - `no_steer`: 70.0%
  - `ocq_bias_a0.3`: 80.0% (`+10.0pp`)
  - `ocq_facet_gated_a1.0`: 62.0% (`-8.0pp`)
- MMLU 100:
  - `no_steer`: 70.0%
  - `ocq_bias_a0.3`: 66.0% (`-4.0pp`)
  - `ocq_facet_gated_a1.0`: 67.0% (`-3.0pp`)

**Status.** Failed.

Interpretation: current bounded evidence supports a narrow “flat K-bias can
help MetaTool” statement more than a “facet-gated OCQ is better and safe”
statement.

### Intent N3 — Falsify trivial explanations before scaling up

**Hypothesis.**
If the MetaTool gain is real, it should survive at least some of the obvious
artifact controls.

**Required next verifications.**
1. Candidate-name opacity control
   - rename tool names to opaque IDs while preserving descriptions
2. Parser robustness control
   - substring match vs exact normalized match vs constrained decode
3. Random-projector control
   - same-rank random orthonormal projector vs ontology projector
4. Label-shuffle control
   - shuffled ontology categories vs real ontology
5. Alpha sweep for facet-gated
   - current `a1.0` is not enough evidence either way

**Status.** Not yet run.

## Decision gate

### Proceed only if all of the following become true

1. `ocq_facet_gated` recovers and matches or beats flat bias on at least one
   bounded single-turn setting.
2. MMLU degradation is within noise or clearly smaller than flat bias.
3. At least one artifact control rejects the lexical/random-subspace story.

### Otherwise narrow the story

If the above gates fail, the viable paper becomes one of:

- static catalog-derived K-bias as a single-turn tool-selection baseline, or
- a negative/diagnostic paper about why facet-gating currently underperforms
  despite the narrative appeal.

## Immediate next tasks

1. Add artifact-control modes to MetaTool eval.
2. Add random-projector baseline.
3. Sweep `ocq_facet_gated` alpha on the bounded first-layer setting.
4. Log intervention norms / gate mass on MMLU instead of accuracy only.
