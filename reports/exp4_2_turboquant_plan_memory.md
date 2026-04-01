# Exp 4-2 TurboQuant-Style Plan Memory

**Date**: 2026-04-01  
**Scope**: `scripts/fokvq/exp4_2_standard_ppl_benchmark.py`

## Objective

Add a stronger external-style baseline to `Exp 4-2` without overclaiming reproduction quality.

## Approved Interpretation

- baseline name in the harness: `turboquant`
- implementation claim: `turboquant-style`
- included:
  - random orthogonal rotation per layer/head
  - scalar Lloyd-Max codebook fit on standard normal samples
  - rotated-domain scalar quantization with per-tensor scale restoration
- excluded until separately implemented and verified:
  - official-repo reproduction claim
  - OJP/QJL or any other paper-specific acceleration/distillation components

## Verification Rules

- deterministic self-tests must pass before any benchmark run
- self-tests must verify:
  - orthogonality of the random rotation
  - monotonic and size-correct Lloyd-Max codebook construction
  - Lloyd-Max scalar quantization is not materially worse than a simple scalar uniform baseline on held-out Gaussian samples
  - output shape/dtype/finite-value preservation
  - higher bitwidth reduces reconstruction error on a synthetic tensor
- benchmark acceptance rule:
  - runner must finish without crash
  - token counts must align across compared methods
  - result JSON must be written successfully

## Keep / Discard Policy

- keep a change only if self-tests pass and smoke evaluation finishes with finite PPL
- discard or revise if:
  - any method returns non-finite values
  - token accounting diverges
  - the turboquant path breaks the shared harness

## Known Risks

- this is not yet an official TurboQuant reproduction
- random rotation can make short-slice results noisy, so claims must remain conservative
- if `turboquant-style` is weak, it still serves as a paper-aligned ablation baseline rather than evidence of superiority
