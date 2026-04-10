# OCQ E8 Execution Policy (2026-04-10, refreshed)

## Default rule

All non-trivial OCQ experiments should run on **E8**, not on the local machine.
Local execution is reserved for:

- parser smoke tests
- import / syntax checks
- tiny bounded checks that do not need a full model run

## Why

1. The local GPU is currently occupied by an unrelated long-running Llama job.
2. E8 has the right capacity for repeated OCQ evaluation waves.
3. The current work no longer needs local exploratory runs; it needs clean,
   reproducible, queueable experiments.

## Observed GPU state at refresh time

### Local

- GPU 0: occupied by unrelated `llm-addiction` run

### E8

- GPU 0: occupied by unrelated `ptca` / Gemma workload
- GPU 1: idle
- GPU 2: idle
- GPU 3: idle

## Allocation rule

- default OCQ usage for the next wave: **GPU 2 + GPU 3**
- keep **GPU 1** available as spare for retries or overflow
- avoid **GPU 0** unless the unrelated workload ends

## Active runner surface

### Keep active

- [ocq_e8_qwen_control.sh](/home/v-seungplee/ba-ocq-develop/scripts/ocq_e8_qwen_control.sh)
  - current Qwen rerun surface
  - not yet the full parser-safe/null-control bundle
- [ocq_e8_qwen_control_bundle.sh](/home/v-seungplee/ba-ocq-develop/scripts/ocq_e8_qwen_control_bundle.sh)
  - bounded and full Qwen control-bundle surface
- [ocq_e8_qwen_mmlu_safety.sh](/home/v-seungplee/ba-ocq-develop/scripts/ocq_e8_qwen_mmlu_safety.sh)
  - secondary regression runner

### Completed reproducibility runners

- [ocq_e8_mistral_low_alpha.sh](/home/v-seungplee/ba-ocq-develop/scripts/ocq_e8_mistral_low_alpha.sh)
  - kept only because the low-alpha falsification is part of the evidence base
  - not part of the forward active surface

### Archived legacy runners

Historical one-off runners were moved to:

- [scripts/archive/ocq_legacy_runners](/home/v-seungplee/ba-ocq-develop/scripts/archive/ocq_legacy_runners)

They should not be used as the default launch surface.

## Launch order

1. Implement parser-safe scoring and null-control code paths
   - local smoke only
2. Qwen control bundle
   - parser-safe MetaTool
   - random / shuffled / opaque controls
   - preferred GPU: 2
3. Qwen operating-point replication
   - `no_steer`, `a0.2`, `a0.3`
   - preferred GPU: 3
4. Mistral factored diagnosis
   - after Qwen harness is stable
   - preferred GPU: 3 or 1

## What is explicitly deferred

- tau2 main wave
- BFCL main wave
- Llama rerun without verified model access
- any "multi-turn main claim" experiment

## Run hygiene

Every E8 run should:

1. write to a dedicated JSON output path
2. write to a dedicated `.log`
3. preserve `seed`, `start_idx`, `max_new_tokens`, and parser diagnostics
4. be copied back into local `results/ocq/cross_model` after completion

Hard gate:

- do not launch the Qwen control bundle, Qwen operating-point rerun, or the
  next Mistral diagnosis until parser-safe scoring and all three null controls
  exist in code and pass a dry run on a bounded shard
