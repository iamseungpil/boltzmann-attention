# OCQ Cross-Model Status Report (2026-04-10, post-follow-up)

**Project**: OCQ / catalog-derived flat K-bias  
**Scope**: single-turn MetaTool and safety regression checks  
**State**: Qwen positive, Mistral negative, low-alpha rescue failed, Llama blocked

## Executive summary

The current OCQ evidence does not support a broad transfer story.

- `Qwen/Qwen2.5-7B` shows a strong positive result on MetaTool.
- `mistralai/Mistral-7B-v0.3` shows a strong negative result on the same setup.
- lowering alpha on Mistral does not recover the method
- `Qwen/Qwen2.5-7B` at `a0.2` is not obviously destructive on MMLU, but that is
  only a regression check, not a mechanism validation

The correct reading is narrow:

> flat catalog-derived K-bias currently works on one model / one benchmark and
> fails on another model family.

## Core results

### Qwen / MetaTool `995`

| Method | Top-1 | Delta vs `no_steer` |
|---|---:|---:|
| `no_steer` | 73.57 | baseline |
| `ocq_bias_a0.2` | 81.11 | +7.54 |
| `ocq_bias_a0.3` | 83.12 | +9.55 |

### Mistral / MetaTool `995`

| Method | Top-1 | Delta vs `no_steer` |
|---|---:|---:|
| `no_steer` | 55.98 | baseline |
| `ocq_bias_a0.2` | 44.12 | -11.86 |
| `ocq_bias_a0.3` | 37.89 | -18.09 |

### Mistral / low-alpha MetaTool `995`

| Method | Top-1 | Delta vs `no_steer` |
|---|---:|---:|
| `ocq_bias_a0.05` | 53.17 | -2.81 |
| `ocq_bias_a0.10` | 49.55 | -6.43 |
| `ocq_bias_a0.15` | 47.34 | -8.64 |
| `ocq_bias_a0.20` | 44.12 | -11.86 |

This closes the cheap "alpha was just too large" escape hatch.

### Qwen / MMLU subset `1000`

| Method | Accuracy | Delta vs `no_steer` |
|---|---:|---:|
| `no_steer` | 72.0 | baseline |
| `ocq_bias_a0.2` | 72.9 | +0.9 |
| `ocq_bias_a0.3` | 70.5 | -1.5 |

This suggests `a0.2` is the only plausible operating point for a balanced
story. `a0.3` is the MetaTool winner, but not the clean safety winner.

## Basis diagnostics

| Model | Target layers | `r_min` | `r_median` | `r_max` |
|---|---:|---:|---:|---:|
| Qwen2.5-7B | `first1` | 25 | 27 | 30 |
| Mistral-7B-v0.3 | `first1` | 14 | 15 | 19 |

The smaller retained rank on Mistral is a real confound and one reason the next
Mistral step must be factored, not another blind sweep.

## What is ruled out

1. Broad cross-model transfer
2. "Just lower alpha and Mistral will recover"
3. Any immediate multi-turn claim

## What is still unresolved

1. Whether the Qwen gain is ontology-specific or a parser / null-control
   artifact
2. Whether Mistral fails because of placement, basis quality, or norm mismatch
3. Whether a parser-safe scorer preserves the Qwen lift

## Evaluation caveats

### Parser

The MetaTool evaluation had a real parser bug in candidate extraction. That has
now been fixed in the local evaluator by using delimiter-aware candidate
parsing and explicit invalid-row dropping. The next Qwen runs must use the
corrected evaluator.

### Scoring

MetaTool scoring is still based on unconstrained free generation and candidate
string matching. Exact or parser-safe scoring remains a required next step.

### MMLU

The current MMLU result is a regression check on one sampled subset, not a hard
"safe everywhere" statement.

## Immediate next move

The next experiments are not tau2, BFCL, or Llama. They are:

1. Qwen parser-safe and null-control experiments
2. Qwen operating-point replication (`a0.2` vs `a0.3`)
3. Mistral factored diagnosis

See [OCQ_NEXT_PLAN_2026-04-10.md](/home/v-seungplee/ba-ocq-develop/reports/OCQ_NEXT_PLAN_2026-04-10.md).
