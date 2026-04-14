# OCQ Cross-Model Status Report (2026-04-10, refreshed)

**Project**: OCQ / catalog-derived K-bias  
**Scope**: single-turn MetaTool mechanism diagnosis  
**State**: Qwen narrowly positive, Mistral negative, multi-turn deferred

## Executive summary

The broad transfer story is still false.

- Qwen has one live setting: `a0.3`
- Qwen `a0.2` is no longer credible under stricter scoring
- Mistral remains negative
- the main open question is no longer cross-model transfer
- the main open question is whether the Qwen gain survives after removing
  free-generation parsing

## Current evidence

### Qwen / parser-safe `first_line`

| Setting | `no_steer` | `a0.2` | `a0.3` |
|---|---:|---:|---:|
| original | 33.57 | 13.57 | 36.38 |
| replication | 33.57 | 12.16 | 36.78 |
| opaque | 29.25 | 5.23 | 42.61 |
| random | 33.57 | 6.03 | 11.96 |
| featshuffle | 33.57 | 12.46 | 1.41 |

### What is actually established

1. Real ontology `a0.3` is above matched null controls on Qwen.
2. `a0.2` is unstable and should be dropped.
3. Opaque naming weakens the lexical-copying objection.
4. The remaining ambiguity is answer-format rescue vs true conditional
   discrimination.

### What is not established

1. Cross-model generalization
2. Multi-turn benefit
3. Publication-grade ontology-specific mechanism

## Code status

The evaluator now includes a closed-set debugging scorer:

- `--scoring-mode label_logprob`

New launchers:

- [ocq_e8_qwen_replication_label_logprob.sh](/home/v-seungplee/ba-ocq-develop/scripts/ocq_e8_qwen_replication_label_logprob.sh)
- [ocq_e8_qwen_control_bundle_label_logprob.sh](/home/v-seungplee/ba-ocq-develop/scripts/ocq_e8_qwen_control_bundle_label_logprob.sh)

Smoke status:

- 1-sample Qwen smoke passed locally
- the closed-set path eliminates `no_match` by construction

## Immediate next move

Run the Qwen closed-set debug wave on E8:

1. replication: original + opaque, `no_steer` vs `a0.3`
2. control bundle: real / random / featshuffle under the same scorer

Do **not** reopen tau2, BFCL, or new cross-model sweeps before that.

See [OCQ_NEXT_PLAN_2026-04-10.md](/home/v-seungplee/ba-ocq-develop/reports/OCQ_NEXT_PLAN_2026-04-10.md) and [OCQ_CAUSAL_DIAGNOSIS_PLAN_2026-04-10.md](/home/v-seungplee/ba-ocq-develop/reports/OCQ_CAUSAL_DIAGNOSIS_PLAN_2026-04-10.md).
