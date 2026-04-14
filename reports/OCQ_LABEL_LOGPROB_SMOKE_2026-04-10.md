# OCQ Label-LogProb Smoke Note (2026-04-10)

## What changed

We added a closed-set debug scorer:

- `--scoring-mode label_logprob`

This scorer removes `no_match` by construction and chooses from the closed set
`{10 candidates, None}` by terminated label log-prob.

## Smoke protocol

- node: E8
- model: `Qwen/Qwen2.5-7B`
- target layers: `first1`
- sample count: `20`
- scoring mode: `label_logprob`
- normalization: `mean`

## Smoke results

### Replication / original names

| Method | Top-1 | matched |
|---|---:|---:|
| `no_steer` | 0.75 | 1.00 |
| `ocq_bias_a0.2` | 0.70 | 1.00 |
| `ocq_bias_a0.3` | 0.65 | 1.00 |

### Opaque names

| Method | Top-1 | matched |
|---|---:|---:|
| `no_steer` | 0.85 | 1.00 |
| `ocq_bias_a0.2` | 0.85 | 1.00 |
| `ocq_bias_a0.3` | 0.75 | 1.00 |

### Random control

| Method | Top-1 | matched |
|---|---:|---:|
| `no_steer` | 0.75 | 1.00 |
| `ocq_bias_a0.2` | 0.80 | 1.00 |
| `ocq_bias_a0.3` | 0.45 | 1.00 |

### Random + opaque

| Method | Top-1 | matched |
|---|---:|---:|
| `no_steer` | 0.85 | 1.00 |
| `ocq_bias_a0.2` | 0.75 | 0.95 |
| `ocq_bias_a0.3` | 0.70 | 1.00 |

### Feature shuffle

| Method | Top-1 | matched |
|---|---:|---:|
| `no_steer` | 0.75 | 1.00 |
| `ocq_bias_a0.2` | 0.60 | 0.85 |
| `ocq_bias_a0.3` | 0.20 | 0.85 |

## Immediate interpretation

This is a bad sign for the current positive story.

1. Once parser dependence is removed, the baseline becomes strong.
2. Real `a0.3` is still less destructive than random or feature shuffle.
3. But on this 20-sample smoke, real `a0.3` does **not** beat `no_steer`.
4. Therefore the earlier `first_line` gain now looks likely to be mediated
   substantially by parser / answerability / abstention effects.

## What this does and does not prove

### It does suggest

- the next full run should be a **closed-set replication**, not another
  benchmark expansion
- the main risk identified in the codex review was real

### It does not prove

- that OCQ is dead overall
- that real ontology offers no benefit under closed-set scoring

The smoke size is only `20`, so a full `995` replication is still needed before
declaring the direction dead.

## Decision

Do not treat `first_line` results as the main evidence anymore.

Priority order:

1. full `995` Qwen `label_logprob` replication
2. if negative, stop benchmark expansion and rewrite the claim narrowly
3. only if positive, rerun the full closed-set control bundle
