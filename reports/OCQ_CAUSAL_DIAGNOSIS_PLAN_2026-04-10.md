# OCQ Causal Diagnosis Plan (2026-04-10)

## One-line status

Real ontology `a0.3` on Qwen is still alive, but the burden of proof shifted:
the next experiment must show that the gain survives **without** free-generation
parsing.

## Current claims by strength

### Defensible now

1. On Qwen / MetaTool, real ontology `a0.3` beats matched random and
   feature-shuffle controls under `first_line` parser-safe scoring.
2. `a0.2` is not a viable operating point under the stricter scorer.
3. Opaque-name control rejects the simplistic "tool-name lexical copying only"
   explanation.

### Not yet defensible

1. "This is a clean ontology-routing effect."
2. "This generalizes cross-model."
3. "This justifies tau2/BFCL expansion."
4. "This is safe by default."

## Competing causes

| Cause | Supported? | Current reading |
|---|---|---|
| ontology-specific semantic alignment | yes | alive |
| parser / answer-format rescue | yes | major remaining confound |
| lexical tool-name copying | mostly no | weakened by opaque control |
| any random projector works | no | killed by random/featshuffle |
| few-shot prompt-template continuation | unresolved | still live |
| generic norm / energy amplification | unresolved | still live |

## Hard next experiments

### D1. Closed-set replication

- Intent: remove `no_match` and parser dependence
- Hypothesis: real `a0.3` still beats `no_steer`
- Verification: `label_logprob`, original + opaque

### D2. Closed-set control bundle

- Intent: re-test ontology specificity after removing parsing
- Hypothesis: ontology stays above random + featshuffle
- Verification: dedicated E8 control launcher

### D3. Prompt-factorization shard

- Intent: test whether OCQ is helping tool routing or merely the MetaTool prompt
- Hypothesis: some gain remains even when examples are removed or mismatched
- Verification: full/no-example/shuffled-example ablation on a bounded shard

### D4. Energy diagnostics

- Intent: rule out "real projector simply injects a gentler perturbation"
- Hypothesis: ontology still wins after matching projected-energy statistics
- Verification: log `||alpha * P_B K|| / ||K||` per layer/head on a held-out
  shard, then calibrate controls

## E8 execution order

1. `scripts/ocq_e8_qwen_replication_label_logprob.sh`
2. `scripts/ocq_e8_qwen_control_bundle_label_logprob.sh`
3. projected-energy debug shard
4. prompt-variant debug shard

## Kill criteria

- If `label_logprob` removes the real `a0.3` gain, stop the positive story.
- If ontology no longer beats random/featshuffle under `label_logprob`, stop
  the ontology-specific story.
- If the gain survives only with the exact MetaTool few-shot template, narrow
  the claim aggressively.
