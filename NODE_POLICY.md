# Node Policy (2026-04-09)

This repository currently owns one reserved AMLT holder for reproducible boltzmann-attention runs.

## Reserved Node

1. `metacognition_e8`
   - AMLT experiment: `metacognition-e8-recovery-0402`
   - owner: `boltzmann-attention`
   - intended use:
     1. boltzmann-attention experiments
     2. boltzmann evaluation follow-ups
     3. result generation for paper/report updates

## Cross-Project Boundary

The shared node partition is:

1. `metacognition_eval`, `metacognition_train_b`
   - main `metacognition`
2. `metacognition_run_c`
   - `metacognition-behavior-uncertainty`
3. `metacognition_e8`
   - `boltzmann-attention`
4. `rsp_grpo_exp` / `advanced-bream`
   - separate softprompt-GRPO line

This repository should not consume the other projects' reserved holders unless the user explicitly
changes policy.

## Safety Rule

1. Do not kill or repurpose another project's AMLT holder.
2. If `metacognition_e8` is temporarily idle, keep the holder intact for this repository.
