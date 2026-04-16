# Develop Integration Decision

Date: 2026-04-16

## Purpose

This memo records what to take from `origin/develop` and what to leave out after the latest refresh to commit `37877a7`.

The constraint is strict: the paper remains a narrow steering paper on multi-tool sequential selection. Any integration that widens the story without first tightening the mechanism evidence is a regression.

## Branch Reality

- `main` contains the cleaned steering paper, report bundle, and external robustness drivers.
- `origin/develop` does not contain that cleaned paper/report tree.
- Therefore broad merging from `develop` is unsafe and unjustified.

## Integration Decision

### Integrate now

- `scripts/ocq/eval_subtask4_with_real_seka.py`
  - Reason: the added `--p-pos` path is low-risk and directly improves matched-baseline reproducibility.
  - Result: integrated locally, with explicit path existence checks.

### Keep only as local exploratory references

- `scripts/ocq/eval_subtask4_dynamic_qk.py`
- `scripts/ocq/eval_subtask4_dynamic_qk_v2.py`
- `scripts/ocq/eval_subtask4_facet_adaptive_v3.py`
- `scripts/ocq/eval_subtask4_facet_adaptive_v4.py`

Reason:

- they are exploratory stateful steering drivers,
- they introduce their own mechanism assumptions,
- they are not yet tied to the paper's validated claim structure,
- and they would pull the paper away from the current narrow main story.

These files are worth revisiting only after the main paper has a completed stepwise coverage analysis and a matched SEKA/AdaSEKA table.

### Reject for now

- Broad replacement of `eval_metatool_subtask1.py` from `develop`
- Broad replacement of `eval_metatool_subtask4.py` from `develop`
- Any branch movement that deletes or bypasses:
  - `paper/neurips2026_steering_v2`
  - `paper/neurips2026_steering_ko`
  - `reports/steering_paper`

Reason:

- The local main branch already contains validated paper-facing changes and local Q/K scheduling support.
- The `develop` refactor does not justify dropping those additions.

## Code-Quality Assessment

### Verified good

- Current main evaluator set compiles cleanly:
  - `eval_metatool_subtask1.py`
  - `eval_metatool_subtask4.py`
  - `eval_subtask4_with_real_seka.py`
  - `eval_bfcl.py`
  - `eval_tau2bench_epsilon_q.py`
- CLI smoke checks pass.
- Method-tag parsing smoke checks pass, including:
  - `ocq_qbias_b-0.1`
  - `ocq_qkv_a0.025_v0_q-0.1`
  - `ocq_qk_layered_a0.3_q-0.1`

### Main risk in develop additions

The exploratory drivers are not obviously syntactically broken, but they are large single-file controllers added through repeated `auto:` commits. The risk is not syntax. The risk is hidden protocol drift:

- changed prompt sequencing,
- changed stopping logic,
- changed turn aggregation,
- changed internal state assumptions,
- and no paper-locked evaluation contract.

That is exactly how scope creep becomes benchmark noise.

## Next Coding Cycle

The next cycle should use the following intent-hypothesis-validation contract.

### C1. Matched SEKA/AdaSEKA baseline stabilization

- Intent: make the closest-prior comparison auditable.
- Hypothesis: under a locked protocol, stationary SEKA-style key-side steering does not overturn the current sign story on Subtask4.
- Validation:
  - same prompt template,
  - same scorer,
  - same decode policy,
  - same ontology source handling,
  - logged environment metadata.

### C2. Stepwise coverage instrumentation

- Intent: test the mechanism directly.
- Hypothesis: if Q-bias acts as a coverage surrogate, the main gain should appear in second-tool recovery or repetition reduction, not only first-tool ranking.
- Validation:
  - first-tool accuracy,
  - second-tool accuracy,
  - repeated-first-facet failure rate,
  - per-sample predicted sequence trace.

### C3. One external robustness benchmark

- Intent: check whether the single-tool vs multi-tool asymmetry persists outside MetaTool.
- Hypothesis: BFCL `parallel` or `multiple` should be more aligned with the current claim than a single-function subset.
- Validation:
  - compare `simple` against `parallel/multiple`,
  - keep exactly the same hook and decoding setup as much as possible.

## Layer-Adaptive Q+K Decision

Preserve the local layer-adaptive Q+K support in:

- `scripts/ocq/eval_metatool_subtask1.py`
- `scripts/ocq/eval_metatool_subtask4.py`

but keep it explicitly non-headline.

Reason:

- it is already integrated and smoke-checked,
- it is a plausible bounded extension of the sign-story,
- but it is not yet validated strongly enough to drive the paper.

That means:

- do not remove it,
- do not broaden around it,
- do not claim it in the paper until it survives the same intent-hypothesis-validation discipline as the main method.
