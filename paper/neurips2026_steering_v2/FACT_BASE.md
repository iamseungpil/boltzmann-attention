# Fact Base

## Verified paper scope

This paper is scoped as a steering paper on multi-tool selection. The following claims are allowed in the English and Korean drafts because they are directly backed by local result files.

## Verified facts

| ID | Claim | Source |
|---|---|---|
| F1 | On Qwen2.5-7B-Instruct MetaTool Subtask4 (N=497), `no_steer` has macro F1 0.7307 and `ocq_qbias_b-0.1` has macro F1 0.7471. | `reports/qkv_joint_2026_04_15/full497_qbias_sweep.json` |
| F2 | On the same Qwen Subtask4 benchmark, `ocq_qbias_b-0.1` improves Exact from 0.5252 to 0.5272 and Jaccard from 0.6673 to 0.6791. | `reports/qkv_joint_2026_04_15/full497_qbias_sweep.json` |
| F3 | On Qwen Subtask4, `ocq_qbias_b-0.1` with random `B_ont` yields F1 0.7068; feature-shuffled `B_ont` yields F1 0.7254. | `reports/wave_2026_04_15_pm/gpu1/qwen_st4_qbias_b-0.1_random_bont.json`, `reports/wave_2026_04_15_pm/gpu1/qwen_st4_qbias_b-0.1_featshuffle_bont.json` |
| F4 | On Qwen Subtask4, real K-bias `ocq_bias_a0.3` yields F1 0.6850, while random and feature-shuffled K-bias both yield F1 0.0000. | `reports/subtask4_overnight/st4_real_N0.json`, `reports/subtask4_overnight/st4_random_N0.json`, `reports/subtask4_overnight/st4_featshuffle_N0.json` |
| F5 | On Llama-3.1-8B-Instruct MetaTool Subtask4 (N=497), `no_steer` has macro F1 0.6227, `ocq_qbias_b-0.1` has 0.6271, and `ocq_bias_a0.3` has 0.3105. | `reports/wave_pm2_2026_04_15/gpu1/llama_st4_qbias_full497.json`, `reports/wave_2026_04_15_pm/gpu0/llama_inst_st4_full497.json` |
| F6 | On Qwen Subtask4, the Q-bias sweep peaks at `beta=-0.1` among the evaluated values `{0,-0.1,-0.3,-0.5}`; larger magnitude suppression degrades sharply. | `reports/qkv_joint_2026_04_15/full497_qbias_sweep.json` |
| F7 | On Qwen Subtask4, adding a very small K-bias on top of Q-bias reaches F1 0.7529 at `alpha_K=0.025` and 0.7502 at `alpha_K=0.05`, above the 0.7471 Q-only result. | `reports/qkv_alpha_microsweep_2026_04_15/full497_alpha_microsweep.json` |
| F8 | In the effective-magnitude diagnostic, the real ontology perturbation has larger mean `||ΔK q||` (621.30) than the random (399.56) and feature-shuffled (291.98) controls. | `reports/stability_effective_magnitude_2026_04_15/result.json` |
| F9 | For Theorem 6.1 verification on Qwen layer 13, the median empirical ratio `LHS/RHS` is `2.357e-08` over 2800 head-query measurements. | `reports/theory_verify_2026_04_14/thm61_qwen_L13_a0.3_N100.json` |
| F10 | For Theorem 6.1 verification on Llama layer 15, the median empirical ratio `LHS/RHS` is `6.372e-08` over 3200 head-query measurements. | `reports/thm61_llama_2026_04_15/llama_L15_a0.3_N100.json` |
| F11 | The Q-space stability proxy on Qwen Subtask4 smoke (N=100) yields AUROC 0.9756 for predicting `F1 >= 0.5` using `min_eps_q`. | `reports/thm620_smoke/eps_q_predictor_N100.json` |
| F12 | On Qwen Subtask1 strict label-logprob scoring, the real ontology basis is consistently better than random and feature-shuffled controls under both sum and mean scorers. | `reports/codex_verify_2026_04_14/full995_sum_original.json`, `reports/codex_verify_2026_04_14/full995_sum_random.json`, `reports/codex_verify_2026_04_14/full995_sum_featshuffle.json`, `reports/codex_verify_2026_04_14/full995_mean_original.json`, `reports/codex_verify_2026_04_14/full995_mean_random.json`, `reports/codex_verify_2026_04_14/full995_mean_featshuffle.json` |
| F13 | On Llama-3.1-8B base Subtask1 strict label-logprob scoring, the real ontology basis is better than both random and feature-shuffled controls under both sum and mean scorers. | `reports/codex_verify_2026_04_14/full995_sum_llama31_real.json`, `reports/codex_verify_2026_04_14/full995_sum_llama31_random.json`, `reports/codex_verify_2026_04_14/full995_sum_llama31_featshuffle.json`, `reports/codex_verify_2026_04_14/full995_mean_llama31_real.json`, `reports/codex_verify_2026_04_14/full995_mean_llama31_random.json`, `reports/codex_verify_2026_04_14/full995_mean_llama31_featshuffle.json` |

## Claims to avoid

| Claim | Reason |
|---|---|
| Exact step-adaptive facet-history tracking is already implemented and evaluated. | The current `ocq_qbias_b-*` hook is a history-free ontology projection on `q_proj`, not a literal emitted-facet tracker. |
| SEKA is beaten directly on a clean full-scale matched implementation. | The available SEKA artifact is marked as a broken-tokenizer run and should not anchor the main story. |
| Compression is a main validated contribution of the current steering paper. | Compression evidence lives in older drafts and side experiments; the current paper should mention it, if at all, only as secondary context. |
| The AUROC predictor is a mature deployment result. | The available evidence is a smoke result on N=100, not a completed multi-turn evaluation. |
