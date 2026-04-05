# Fact Base

## Verified Facts

| ID | Claim | Source | Verification |
|----|-------|--------|--------------|
| F1 | Exp 2-1 overall within-corpus cosine similarity is 0.8193 and cross-corpus similarity is 0.4111, giving a separability gap of 0.4082. | `results/exp2_1_results.json` | direct file inspection |
| F2 | Exp 2-1 two-sample t-test statistic is 30.50 with p-value 1.64e-64. | `results/exp2_1_results.json` | direct file inspection |
| F3 | Exp 2-2 shows `lda_outperforms_pca = false`. | `results/exp2_2_results.json` | direct file inspection |
| F4 | In Exp 2-2 at 8/2 allocation, PCA mean KL is 1.8265 while LDA mean KL is 11.8657. | `results/exp2_2_results.json` | direct file inspection |
| F5 | In Exp 2-2 at 8/2 allocation, PCA mean K-MSE is 59.79 while LDA mean K-MSE is 1790.11. | `results/exp2_2_results.json` | direct file inspection |
| F6 | Exp 2-3 reports `query_dependent_allocation_differs = true` and `allocation_varies_across_queries = true`. | `results/exp2_3_results.json` | direct file inspection |
| F7 | Exp 2-3 overall static-vs-query-dependent rank correlation mean is 0.6976 and allocation L1 mean is 1.1701. | `results/exp2_3_results.json` | direct file inspection |
| F8 | Exp 2-3 overall query-vs-query rank correlation mean is 0.7460 and query-vs-query L1 mean is 0.4935. | `results/exp2_3_results.json` | direct file inspection |
| F9 | Exp 3-1 mean MSE values are identity 961.01, random 1047.17, PCA 220.45, optimized 938.81. | `results/exp3_1_results.json` | direct file inspection |
| F10 | Exp 3-1 reports `optimized_is_best = false` and `optimized_significantly_beats_pca = false`. | `results/exp3_1_results.json` | direct file inspection |
| F11 | Boltzmann Exp 1-1 gives GPT-2 rho -0.6574 (p = 4.82e-04) and Qwen rho +0.1081 (p = 0.530). | `results/experiment_log_v2.md`, `reports/boltzmann_rq_report_ko.tex` | local report cross-check |
| F12 | Boltzmann Exp 1-2 gives overall disagreement 39.8% and mean rho(C, ||k||^2) = 0.275. | `results/experiment_log_v2.md`, `reports/boltzmann_rq_report_ko.tex` | local report cross-check |
| F13 | Boltzmann Exp 1-2 PPL results are baseline 139.85, uniform-3bit 130.38, variance-3bit 127.62, and specific-heat-3bit 264.48. | `results/experiment_log_v2.md`, `results/critic_corrections.md` | local report cross-check |
| F14 | The authoritative Exp 2-3 rerun was completed on `eval-e8` and matched the current local JSON on all scientific metrics. | `reports/exp2_3_update_study.md` | direct file inspection |
| F15 | Phase 4-1 preliminary custom-text PPL results report GPT-2 fp16 17.59, uniform-3bit 17.71, fokvq-3bit 17.54, and Qwen fp16 8.21, uniform-3bit 210.60, fokvq-3bit 8.44. | `phase4_1_ppl_results.docx` | direct document inspection |
| F16 | The new plan document explicitly says standard WikiText-2 test-set PPL under sliding-window protocol is still required for NeurIPS. | `FOKVQ_ADDITIONAL_EXPERIMENT_PLAN_v9.docx` | direct document inspection |
| F17 | KIVI is an asymmetric KV-cache quantization method that quantizes keys per-channel and values per-token. | arXiv + official GitHub | web verification |
| F18 | KVQuant and ZipCache are NeurIPS 2024 KV-cache quantization baselines. | NeurIPS proceedings | web verification |
| F19 | QuIP# is a weight-only post-training quantization method, not a KV-cache quantization method. | arXiv + official GitHub | web verification |
| F20 | TurboQuant is an ICLR 2026 online vector quantization method positioned for KV-cache compression. | arXiv / Google Research blog | web verification |
| F21 | NeurIPS 2026 main-track submissions are limited to 9 content pages, excluding references, checklist, and appendices. | official NeurIPS 2026 template | web verification |

| F22 | 3-model axis 1 verification: Pre-RoPE PCA+WF is MSE-best at all bits for Qwen2.5-7B (112 heads), Llama-3.1-8B (256 heads), Mistral-7B (256 heads). | `verify_3axis_unified.py` logs | direct experiment 2026-04-05 |
| F23 | 3-model Cor 6.16.4(d): Post-RoPE PCA+WF loses to TurboQuant at 2-bit in all 3 models. | `verify_3axis_unified.py` logs | direct experiment 2026-04-05 |
| F24 | Lloyd-Max centering fix: After centering, Lloyd-Max gains over uniform are 3.5x(2-bit), 2.1x(3-bit), 1.6x(4-bit) consistently across 3 models. | `verify_3axis_unified.py` logs | direct experiment 2026-04-05 |
| F25 | R_aniso values: Qwen2.5-7B=4.27, Llama-3.1-8B=7.97, Mistral-7B=131.62. | `verify_3axis_unified.py` logs | direct experiment 2026-04-05 |
| F26 | TurboQuant-to-PreRoPE actual gain ratio at 4-bit: Qwen 3.39x, Llama 3.56x, Mistral 3.80x (increases with R_aniso). | `verify_3axis_unified.py` logs | direct experiment 2026-04-05 |

| F27 | Qwen2.5-7B PPL Table: Pre-RoPE PCA+Uni is PPL-best at all bits (2bit: 7.98, 3bit: 6.76, 4bit: 6.60). TurboQuant: 9.33/6.82/6.61. | `ppl_table_Qwen_Qwen2.5-7B_*.json` | direct experiment 2026-04-05 |
| F28 | Lloyd-Max HURTS PPL despite helping MSE (Qwen 3bit: Lloyd 7.30 vs Uniform 6.76). Non-Gaussian tails cause codebook mismatch. | `ppl_table_Qwen_Qwen2.5-7B_*.json` | direct experiment 2026-04-05 |
| F29 | Water-filling HURTS PPL especially at 2-bit (Qwen: WF 11.37 vs Uni 7.98). Low-variance channels get catastrophically few bits. | `ppl_table_Qwen_Qwen2.5-7B_*.json` | direct experiment 2026-04-05 |

| F30 | 3-model PPL Table 1: PreRoPE PCA+Uni is PPL-best at 2-3bit for Qwen (7.98/6.76) and Llama (10.14/6.67), but TurboQuant wins at Mistral 2-bit (6.37 vs 6.46). | `ppl_table_*.json` | direct experiment 2026-04-05 |
| F31 | Lloyd-Max CATASTROPHIC at PPL for all 3 models: Qwen 8.34, Llama 65.46, Mistral 32.68 at 2-bit vs PCA+Uni 7.98/10.14/6.46. | `ppl_table_*.json` | direct experiment 2026-04-05 |
| F32 | Water-filling inconsistent: helps Llama 2-bit (8.79 vs 10.14) but hurts Qwen 2-bit (11.37 vs 7.98) and Mistral 3-4bit. | `ppl_table_*.json` | direct experiment 2026-04-05 |
| F33 | Mistral exception: TurboQuant beats PCA at 2-bit (6.37 vs 6.46) despite R_aniso=131.62. Extreme anisotropy concentrated in few heads destabilizes PCA calibration. | `ppl_table_*.json` | direct experiment 2026-04-05 |

| F34 | V15-1 KVTC comparison: Per-head PCA beats shared PCA (KVTC style) by 46.3% at 2-bit PPL on Llama-3.1-8B (10.14 vs 18.87). 3-bit: 2.1%, 4-bit: 0.4%. | `v15_meta-llama_Llama-3.1-8B_*.json` | direct experiment 2026-04-05 |

| F35 | V15-3: Adaptive Lloyd-Max still fails at PPL. Uniform > Adaptive Lloyd > Gaussian Lloyd at 2-bit (7.98 > 8.12 > 8.34). Scalar quantizer structural limit confirmed. | `v15_Qwen_Qwen2.5-7B_*.json` | direct experiment 2026-04-05 |
| F36 | V15-4: WF floor=2 SUCCEEDS. Beats Uniform at 2-bit by 11.0% (7.10 vs 7.98) and 3-bit by 1.1% (6.68 vs 6.76). WF failure was caused by min_bits=1 allowing catastrophic 1-bit channels. | `v15_Qwen_Qwen2.5-7B_*.json` | direct experiment 2026-04-05 |

| F37 | WF floor=2 generalizes: beats TurboQuant on ALL 3 models at 2-bit. Qwen +23.9% (7.10 vs 9.33), Llama +36.5% (7.16 vs 11.26), Mistral +8.6% (5.82 vs 6.37). | `v15_*.json` | direct experiment 2026-04-05 |
| F38 | Mistral exception RESOLVED: PCA+WF(floor=2) at 5.822 beats TurboQuant at 6.371, reversing the PCA+Uni(6.461) > TurboQuant(6.371) failure. | `v15_mistralai_*.json` | direct experiment 2026-04-05 |

| F39 | V16-2 Ablation: Axis 1 gain 1.32x, Axis 3 gain 1.12x, combined 1.48x at 2-bit Qwen. WF benefits TurboQuant too (9.33→8.52). | `v16_ablation_*.json` | direct experiment 2026-04-05 |

| F40 | Gaussianity check: PCA components have mean excess kurtosis 0.49-0.58 (mild non-Gaussian), ~12% with kurtosis>1, Shapiro-Wilk pass rate ~22%. Consistent across 3 models. | `gaussianity_check_*.json` | direct experiment 2026-04-05 |
| F41 | 4-bit WF floor=2 ≈ Uniform for all 3 models (difference <0.1%). WF only helps at 2-3 bit. | `4bit_wf_*.json` | direct experiment 2026-04-05 |

## Unverified or Not Yet Ready

| Claim | Reason |
|-------|--------|
| FOKVQ is state of the art on standard WikiText-2 PPL | standard full-protocol table is not finished |
| FOKVQ beats TurboQuant, KIVI, KVQuant, and ZipCache on a fair common benchmark | current comparison axis is not yet fully harmonized |
| FOKVQ generalizes to Llama-3-8B with standard PPL and latency evidence | planned, not yet complete |

