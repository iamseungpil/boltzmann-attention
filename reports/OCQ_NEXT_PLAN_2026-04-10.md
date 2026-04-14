# OCQ Next-Step Plan (2026-04-10, codex-reviewed refresh)

## Purpose

This is the canonical next-step document for the current OCQ line in
`ba-ocq-develop`.

The project is still in **mechanism isolation mode**, not benchmark expansion
mode. The immediate question is no longer "do controls exist?" That part is
mostly done. The immediate question is:

> if free-generation parsing is removed, does the Qwen ontology effect still
> survive as a closed-set tool-selection effect?

## Current factual state

### Completed Qwen evidence

Parser-safe `first_line` runs on MetaTool `995` are complete for:

- original names
- opaque local IDs
- rank-matched random control
- feature-shuffle control

Core numbers:

| Setting | `no_steer` | `a0.2` | `a0.3` |
|---|---:|---:|---:|
| original | 33.57 | 13.57 | 36.38 |
| original replication | 33.57 | 12.16 | 36.78 |
| opaque | 29.25 | 5.23 | 42.61 |
| random | 33.57 | 6.03 | 11.96 |
| random + opaque | 29.25 | 3.22 | 8.24 |
| featshuffle | 33.57 | 12.46 | 1.41 |
| featshuffle + opaque | 29.25 | 2.31 | 1.81 |

### What these numbers actually mean

1. `a0.2` is dead.
   - It collapses under the stricter scorer in both original and opaque modes.
   - Stop calling it a balanced operating point.

2. Real Qwen `a0.3` survives matched null controls.
   - It is far above random and far above feature shuffle.
   - That kills the lazy "any projector works" story.

3. The remaining ambiguity is not ontology-vs-random.
   - The remaining ambiguity is answerability-vs-discrimination.
   - `a0.3` improves overall top-1, but much of the gain is still mediated by
     emitting a parseable candidate label more often.

4. Opaque mode does **not** prove "deep semantics" by itself.
   - It preserves descriptions and examples.
   - It removes label names, not the semantic content of the prompt.

### Key decomposition

Original parser-safe:

- `no_steer`
  - matched rate: `0.4322`
  - conditional accuracy: `0.7767`
  - none rate: `0.2905`
- `a0.3`
  - matched rate: `0.4603`
  - conditional accuracy: `0.7904`
  - none rate: `0.1869`

Opaque parser-safe:

- `no_steer`
  - matched rate: `0.3729`
  - conditional accuracy: `0.7844`
- `a0.3`
  - matched rate: `0.5759`
  - conditional accuracy: `0.7400`

Interpretation:

- In original mode, `a0.3` helps both matched rate and conditional accuracy.
- In opaque mode, the large gain is dominated by matched-rate rescue, not by
  better conditional discrimination.

## What changed today

The evaluator now has a new closed-set debug mode:

- `--scoring-mode label_logprob`

This mode:

- keeps the original prompt body
- scores terminated label continuations `"{choice}\\n"`
- removes `no_match` by construction
- logs top candidate scores for bounded smoke runs

Related files:

- [eval_metatool_subtask1.py](/home/v-seungplee/ba-ocq-develop/scripts/ocq/eval_metatool_subtask1.py)
- [ocq_e8_qwen_control_bundle_label_logprob.sh](/home/v-seungplee/ba-ocq-develop/scripts/ocq_e8_qwen_control_bundle_label_logprob.sh)
- [ocq_e8_qwen_replication_label_logprob.sh](/home/v-seungplee/ba-ocq-develop/scripts/ocq_e8_qwen_replication_label_logprob.sh)

Smoke status:

- local 1-sample Qwen smoke passed
- `label_logprob` produced closed-set predictions with `no_match = 0`

Important caveat:

- this is a debugging scorer, not yet publication-grade evidence
- it is a different evaluation interface from free-form generation

## Reviewer-visible blockers

### B1. Free-generation ambiguity is not dead until closed-set runs complete

The `first_line` control suite is strong enough to reject trivial random-subspace
stories, but still mixes two phenomena:

- did the model emit a parseable label?
- was the emitted label correct?

`label_logprob` is the direct next diagnostic because it removes the first one.

### B2. Prompt-template / in-context-learning confound remains live

MetaTool prompts contain examples before the final query. OCQ could be helping
tool routing, example retrieval, or prompt-template continuation. Those are not
the same mechanism.

### B3. Control energy is not yet matched

Random and feature-shuffle controls match shape and rank, but not necessarily
the induced `||ΔK||` or projected-energy distribution on actual activations.

### B4. Multi-turn expansion is still premature

Launching tau2/BFCL now would multiply ambiguity before the single-turn Qwen
mechanism is isolated.

## Intent / Hypothesis / Verification

### Intent P1 — Remove parser dependence from the main Qwen claim

**Hypothesis.**
If the real ontology effect is not just a parser artifact, `a0.3` should still
beat `no_steer` under `label_logprob`.

**Verification.**

1. run Qwen `no_steer` vs `a0.3` under `label_logprob`
2. run both original and opaque modes
3. report:
   - top-1
   - conditional accuracy
   - margin to second-best candidate from `top_scores`

**Stop rule.**
If real `a0.3` loses its gain under `label_logprob`, the current positive story
was mostly a generation/parser effect.

### Intent P2 — Test ontology specificity under closed-set scoring

**Hypothesis.**
Under `label_logprob`, real ontology `a0.3` should remain clearly above
rank-matched random and feature-shuffle controls.

**Verification.**

1. closed-set control bundle:
   - original
   - opaque
   - random
   - random + opaque
   - featshuffle
   - featshuffle + opaque
2. methods:
   - `no_steer`
   - `ocq_bias_a0.3`

**Stop rule.**
If the ontology gap collapses once parsing is removed, the current
ontology-specific interpretation is not stable.

### Intent P3 — Separate answerability gain from discrimination gain

**Hypothesis.**
The original-mode Qwen gain contains a real discrimination component, but the
opaque-mode gain is mostly answerability / commitment rescue.

**Verification.**

1. compare `first_line` and `label_logprob` side by side
2. compute delta decomposition:
   - matched-rate delta
   - conditional-accuracy delta
   - top-1 delta
3. inspect score margins on a bounded shard

**Stop rule.**
If `label_logprob` removes nearly all of the gain in both original and opaque
modes, do not keep claiming semantic routing improvement.

### Intent P4 — Test the prompt-template confound before new benchmarks

**Hypothesis.**
Part of the current effect may come from the few-shot prompt structure, not
just query-to-tool semantics.

**Verification.**

1. build three prompt variants on a bounded shard:
   - full prompt
   - no in-context examples
   - mismatched or shuffled examples
2. run `no_steer` vs real `a0.3` with `label_logprob`

**Stop rule.**
If the effect vanishes when examples are removed or mismatched, narrow the
claim to "helps this prompt template" rather than "tool routing" broadly.

### Intent P5 — Delay multi-turn and cross-model expansion

**Hypothesis.**
Expanding to tau2/BFCL or restarting cross-model claims before P1-P4 complete
would dilute the scientific story.

**Verification.**

- do not launch tau2, BFCL, or new cross-model sweeps until at least P1 and P2
  are complete

**Stop rule.**
If P1 or P2 fail, expansion is blocked.

## Ordered execution plan

### Immediate

1. run Qwen `label_logprob` replication on E8
   - original
   - opaque
   - methods: `no_steer`, `a0.3`
2. run Qwen `label_logprob` control bundle on E8
   - original / opaque / random / featshuffle
   - methods: `no_steer`, `a0.3`

### After that

3. add projected-energy diagnostics on a bounded shard
4. run prompt-variant ablation on a bounded shard

### Explicitly deferred

5. Llama / Mistral expansion
6. tau2 / BFCL
7. MMLU follow-up beyond regression checking

## Brutal summary

- `a0.2` is finished.
- `a0.3` is the only live setting.
- random/featshuffle controls are already strong enough to justify keeping the
  direction alive.
- the main remaining risk is that the Qwen gain is still partially an
  answer-format effect.
- the correct next move is **closed-set Qwen debugging on E8**, not benchmark
  expansion.
