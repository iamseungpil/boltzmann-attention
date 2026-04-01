# DHS Autoresearch Log

**Date**: 2026-03-31  
**Node**: `eval-e8`  
**Protocol**: autoresearch

## Goal

Validate the DHS/HEAT track on top of the stabilized Boltzmann baseline, starting from the theory-critical Phase 1 experiments before moving into mechanism and practical-value experiments.

## Scope

- `Exp 1.1`: dual-exponential superiority
- `Exp 5.1`: sequence-length valley-depth scaling
- `Exp 4.1`: GQA multimodality

## Iteration 1

- Review:
  - Existing repository contains Boltzmann/FOKVQ precursor scripts only.
  - DHS/HEAT documents define experiments, but dedicated scripts were missing.
- Change:
  - Added `scripts/dhs_phase1.py` as a unified Phase 1 experiment runner.
  - Script covers:
    - dual-exponential vs baseline fitting
    - sequence-length valley-depth scaling
    - GQA multimodality analysis
- Verification:
  - `python3 -m py_compile scripts/dhs_phase1.py` passed locally.
  - Local runtime smoke test was limited because the local shell environment does not expose `torch`.
- Decision:
  - Keep.
- Learning:
  - Phase 1 can be started immediately by reusing standard attention outputs rather than waiting for bespoke DHS infrastructure.

## Iteration 2

- Review:
  - `eval-e8` GPUs were idle and `/scratch` had sufficient free space.
  - Remote `grpo` environment already had `torch`, `transformers`, and `scipy`.
- Change:
  - Synced `scripts/dhs_phase1.py` to:
    - `/mnt/input/boltzmann/dhs_scripts/dhs_phase1.py`
  - Launched three background experiments:
    - `dual_exp_gpt2`
    - `length_scaling_gpt2`
    - `gqa_multimodality_qwen3b`
- Verification:
  - Remote launch commands returned PIDs:
    - `176413`
    - `176405`
    - `176433`
  - AML SSH proxy became intermittently unstable during monitoring, so result collection is still in progress.
- Decision:
  - Keep.
- Learning:
  - Background launch is reliable enough; status polling should use short commands with retries because long AML SSH sessions time out.

## Iteration 3

- Review:
  - The first background launch pattern did not actually keep the DHS jobs alive.
  - A direct foreground smoke test for `dual_exp_gpt2` succeeded, so the issue was launch wrapping rather than experiment code.
- Change:
  - Patched `scripts/dhs_phase1.py`:
    - fixed config attribute lookup for Qwen
    - constrained GPT-2 length-scaling to valid position lengths
  - Relaunched Phase 1 jobs with simpler background shell wrapping.
- Verification:
  - `dual_exp_gpt2` completed successfully.
    - dual-exp beats Poly-4 in `95.31%` of heads by `R^2`
    - dual-exp beats Poly-4 in `95.31%` of heads by `AIC`
    - mean `R^2`: dual-exp `0.924`, Poly-4 `0.113`
    - paired t-test: `t=56.29`, `p=2.51e-187`
  - `length_scaling_gpt2` completed successfully.
    - used lengths: `256, 512, 768, 1024`
    - median valley depth increased monotonically: `194.17 -> 341.40 -> 430.23 -> 662.47`
    - Spearman(`log L`, median valley depth) = `1.0`
  - `gqa_multimodality_qwen3b` completed successfully after the Qwen config fix.
    - mean per-Q-head dual-exp `R^2 = 0.871`
    - fraction of KV groups flagged multimodal = `0.236`
- Decision:
  - Keep.
- Learning:
  - Gate A is currently strongly positive for GPT-2.
  - Sequence-length scaling is also strongly positive on GPT-2.
  - Qwen shows strong per-head dual-exp structure, but KV-group multimodality is weaker and more mixed than the most aggressive DHS wording suggests; this should likely be framed as partial support rather than a clean headline win.

## Current Remote Artifacts

- Scripts:
  - `/mnt/input/boltzmann/dhs_scripts/dhs_phase1.py`
- Logs:
  - `/mnt/input/boltzmann/dhs_autoresearch/dual_exp_gpt2.log`
  - `/mnt/input/boltzmann/dhs_autoresearch/length_scaling_gpt2.log`
  - `/mnt/input/boltzmann/dhs_autoresearch/gqa_multimodality_qwen3b.log`
- Expected outputs:
  - `/mnt/input/boltzmann/dhs_results/dual_exp_gpt2.json`
  - `/mnt/input/boltzmann/dhs_results/length_scaling_gpt2.json`
- `/mnt/input/boltzmann/dhs_results/gqa_multimodality_qwen3b.json`

## Local Copies

- `results/dhs_dual_exp_gpt2.json`
- `results/dhs_length_scaling_gpt2.json`
- `results/dhs_gqa_multimodality_qwen3b.json`

## Next

1. Interpret `gqa_multimodality_qwen3b` more carefully and decide whether it is PASS or PARTIAL under the integrated plan.
2. If Phase 1 remains positive enough, implement Phase 2:
   - RoPE-kappa coupling
   - residual-reheating coupling

## Iteration 4

- Review:
  - Phase 1 produced two strong positives and one partial-support result.
  - This is sufficient to continue into Phase 2 mechanism checks.
- Change:
  - Added `scripts/dhs_phase2.py`.
  - Implemented:
    - `rope_kappa` on Qwen by sweeping `rope_theta`
    - `residual_reheating` on GPT-2 by patching residual scaling inside GPT-2 blocks
- Verification:
  - `rope_kappa` smoke test passed.
    - `theta=1e4 -> median kappa2=0.713`
    - `theta=1e6 -> median kappa2=2.000`
    - directional effect is positive, not negative
  - `residual_reheating` smoke test initially failed due to a GPT-2 block forward signature mismatch.
  - Patched the residual-scaling wrapper to match the current Transformers `past_key_value` / `cache_position` API.
  - `residual_reheating` smoke test then passed.
- Decision:
  - Keep.
- Learning:
  - The RoPE sweep is executable and already suggests that the current causal prediction about the sign of `theta -> kappa2` may need revision.
  - Residual intervention is technically viable for the full run.

## Iteration 5

- Change:
  - Launched full Phase 2 runs on `eval-e8`:
    - `rope_kappa_qwen3b`
    - `residual_reheating_gpt2`
- Verification:
  - Remote logs show model loading has started.
  - GPU memory is allocated on the corresponding devices.
- Decision:
  - Keep running.
