# Problem Description

Review and harden the current research plan in `/home/v-seungplee/boltzmann-attention` for three candidate directions around KV-cache compression and retrieval:

1. Selective refinement / two-pass reranking
2. Query-dynamic risk paging (QDRP)
3. Tiny recursive innovation cache (TRIC)

The immediate goal is not to market a method. It is to decide, harshly, what is still scientifically defensible, what must be reframed as a diagnostic only, and what should be dropped before spending more GPU.

## Codebase Context

Locations:
- `reports/EXPERIMENT_PLAN_v27_dynamic_recursive.md`: prior plan for QDRP and TRIC
- `reports/EXPERIMENT_PLAN_v28_selective_refine.md`: prior selective-refinement plan
- `reports/EXPERIMENT_PLAN_v29_selective_refine_diagnostics.md`: current narrowed plan
- `reports/SELECTIVE_REFINEMENT_REPORT_2026-04-09.md`: current report draft
- `scripts/exp_query_exploit.py`: attention-path intervention experiments (`two_pass`, `query_dequant`, `sharp_temp`, controls)
- `scripts/exp_cliffkv_niah.py`: selective promotion proxy experiment
- `scripts/exp_query_risk_paging.py`: synthetic QDRP diagnostic
- `scripts/exp_tric_recursive_predictor.py`: synthetic TRIC diagnostic
- `tmp_remote_results/mistral_all_methods_4k.json`
- `tmp_remote_results/mistral_two_pass_4k.json`
- `tmp_remote_results/qwen_two_pass_4k.json`
- `results/e8_2026-04-09/Mistral-7B-v0_3_fixcheck_cliffkv_mistral_4k.json`
- `reports/autoresearch_dynamic_recursive_log.tsv`
- `reports/autoresearch_query_exploit_log.tsv`
- `reports/autoresearch_selective_refine_log.tsv`

## Current Situation

Empirical state:
- `two_pass` looks strong on bounded 4K NIAH.
  - Mistral same-harness: `baseline_2bit=0.333`, `two_pass_k16=1.000`
  - Mistral sweep: `k=8,16,32,64` all `1.000`
  - Qwen sweep: `k=8,16,32,64` all `1.000`
- `query_dequant` fails on the same grid.
  - Best Mistral 4K result: `0.333`, equal to baseline
- `sharp_temp` fails on the same grid.
  - All tested temperatures: `0.000`
- QDRP synthetic:
  - budget=1: `risk_vs_score_recover=+0.3770`
  - budget=2: pure risk fails, hybrid is basically neutral (`+0.0005`)
- TRIC synthetic:
  - recursive predictor beats copy-last but still loses badly to shared linear
  - `recursive_gain_vs_linear` remains negative

Critical validity concern:
- In `scripts/exp_query_exploit.py`, the patcher explicitly says the underlying KV cache stays FP16 and low-bit logic is only on the attention path.
  - See `current_limitations()` and the patched forward using `key_fp16 = key_states.clone()`.
- In `scripts/exp_cliffkv_niah.py`, the patched forward also clones full-precision keys after cache update and builds mixed precision on the fly.
  - This appears to be an attention-path proxy, not true stored compressed-cache evaluation.

Therefore the main risk is that some recent report language may still overstate the storage validity or novelty of the current positive results.

## Proposed Approaches

### Approach 1: Narrow selective refinement to a diagnostic claim
Concept:
- Treat `two_pass` and `cliffkv` results as decode-time attention-path intervention diagnostics only.
- Run selector diagnostics (`recent_k`, `sink_k`, `random_k`) and harder 8K/16K budget frontiers.
- Do not make compressed-cache storage claims yet.

Pros:
- Scientifically honest with current implementation.
- Fastest route to identifying whether the signal is real or just a harness artifact.

Cons:
- Novelty becomes weak.
- Could collapse to “top-score promotion baseline” rather than a publishable method.

### Approach 2: Pivot selective refinement to true storage-valid implementation
Concept:
- Rewrite cache write/update path so base cache is actually quantized before storage.
- Maintain a real promoted side-buffer or stored residual path.
- Re-run fair-budget experiments.

Pros:
- Restores validity of compression/storage claims.
- Enables proper equal-storage/equal-HBM comparisons.

Cons:
- More engineering work.
- Current positive signal might disappear once the FP16 fallback path is removed.

### Approach 3: Re-center on QDRP or TRIC
Concept:
- Use selective-refinement results only as a side diagnostic and instead push a more novel method:
  - QDRP: query-conditional risk-based page activation
  - TRIC: shared recursive predictor plus innovation residual coding

Pros:
- More novel than simple two-pass promotion.
- Better paper differentiation if they work.

Cons:
- Current diagnostics are weak:
  - QDRP loses to raw score on real trace
  - TRIC loses to shared linear even on synthetic

## Questions for Codex

Please provide a harsh review with file references where useful:

1. Fatal flaws:
   - What is scientifically invalid or overclaimed right now?
2. Addressable issues:
   - What can be fixed with reframing or bounded extra experiments?
3. Failure modes:
   - What alternative explanations still threaten the current positive `two_pass` result?
4. Best next step:
   - Which of the three approaches above would you pursue next and why?
5. Concrete plan:
   - Give a prioritized next-experiment sequence with explicit kill criteria.

Do not be polite. Assume NeurIPS/ICML review standards. Distinguish clearly between:
- diagnostic-only results
- real method-valid results
- directions that should be dropped now
