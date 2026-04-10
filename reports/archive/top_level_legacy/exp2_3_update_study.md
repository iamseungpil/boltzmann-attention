# Exp 2-3 Update Study

**Date**: 2026-03-31  
**Project**: boltzmann-attention  
**Experiment**: Query-dependent vs static bit allocation rerun

## Executive Summary

| Item | Result |
|------|--------|
| Goal | Resolve whether Exp 2-3 is genuinely inconclusive or a data-integrity artifact |
| Authoritative rerun | Completed on `eval-e8` |
| Key finding | Query-dependent allocation differs from static allocation |
| Static vs Q-dep rank correlation | 0.6976 +/- 0.1737 |
| Static vs Q-dep allocation L1 | 1.1701 +/- 0.1261 |
| Query-vs-query rank correlation | 0.7460 +/- 0.0817 |
| Query-vs-query allocation L1 | 0.4935 +/- 0.2185 |
| Scientific verdict | Exp 2-3 is resolved; dynamicity exists, but end-to-end superiority is still untested |

The rerun matches the current local `results/exp2_3_results.json` on all scientific metrics. The previous inconclusive state was caused by conflicting historical artifacts, not by instability in the current implementation.

## 1. Autoresearch Loop

### Iteration 1

- Change: ran `scripts/fokvq/exp2_3_query_dependent_bitalloc.py` on `eval-e8` using `HF_HOME=/mnt/input/hf_cache`
- Verification: run failed before model execution
- Decision: discard
- Reason: `/mnt/input` had no free space for Hugging Face cache

### Iteration 2

- Change: reran on `eval-e8` with `HF_HOME=/scratch/hf_cache`
- Verification: run completed successfully and wrote `/mnt/input/fokvq/results/exp2_3/results.json`
- Decision: keep
- Reason: authoritative rerun reproduced the current local JSON metrics

## 2. Implementation Notes

- Remote node: `eval-e8`
- AML job: `serene_beard_f30tbngxfz`
- GPU SKU: `80G4-A100-NvLink`
- Python env: `/opt/conda/envs/grpo`
- Added dependency: `scipy==1.13.1`
- Experiment script: `scripts/fokvq/exp2_3_query_dependent_bitalloc.py`

## 3. Result Interpretation

If static allocation were already equivalent to query-dependent allocation, rank correlation would be near 1 and L1 distance near 0. The rerun shows neither:

- Static vs query-dependent rank correlation is only `0.6976`
- Static vs query-dependent allocation distance is `1.1701`
- Query-vs-query allocation distance remains `0.4935`

This means the per-dimension importance profile changes with the query. The effect is strongest in early layers, where static-vs-query-dependent rank correlation drops to `0.208-0.445` and allocation L1 rises to `1.325-1.516`.

The result does **not** prove that a dynamic scheduler beats static PCA in perplexity or latency. It proves a narrower statement: query-conditioned allocation structure exists and should be treated as a separate optimization axis from static rotation/axis choice.

## 4. Artifacts

- Local result JSON: `results/exp2_3_results.json`
- Remote result JSON: `/mnt/input/fokvq/results/exp2_3/results.json`
- Figure: `reports/figures/exp2_3_query_dynamicity.png`
- Updated report source: `reports/boltzmann_rq_report_ko.tex`

## 5. Next Experiments

1. Keep PCA as the static rotation baseline and add a lightweight query-conditioned bit scheduler on top.
2. Measure end-to-end `PPL / latency / memory` trade-off rather than only rank/L1 differences.
3. Repeat the same scheduler experiment on a GQA model to test whether query-conditioned scheduling becomes more valuable when KV sharing is stronger.
