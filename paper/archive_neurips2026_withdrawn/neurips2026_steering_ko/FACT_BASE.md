# Fact Base

## Verified Results

| ID | Claim | Value | Source |
|---|---|---:|---|
| F1 | Qwen Subtask4 `no_steer` macro F1 | 0.7307 | `reports/qkv_joint_2026_04_15/full497_qbias_sweep.json` |
| F2 | Qwen Subtask4 `ocq_qbias_b-0.1` macro F1 | 0.7471 | `reports/qkv_joint_2026_04_15/full497_qbias_sweep.json` |
| F3 | Qwen Subtask4 `ocq_bias_a0.3` macro F1 | 0.6850 | `reports/subtask4_overnight/st4_real_N0.json` |
| F4 | Llama Subtask4 `no_steer` macro F1 | 0.6227 | `reports/wave_pm2_2026_04_15/gpu1/llama_st4_qbias_full497.json` |
| F5 | Llama Subtask4 `ocq_qbias_b-0.1` macro F1 | 0.6271 | `reports/wave_pm2_2026_04_15/gpu1/llama_st4_qbias_full497.json` |
| F6 | Llama Subtask4 `ocq_bias_a0.3` macro F1 | 0.3105 | `reports/wave_2026_04_15_pm/gpu0/llama_inst_st4_full497.json` |
| F7 | Qwen Q-bias feature-shuffled control | 0.7254 | `reports/wave_2026_04_15_pm/gpu1/qwen_st4_qbias_b-0.1_featshuffle_bont.json` |
| F8 | Qwen Q-bias random control | 0.7068 | `reports/wave_2026_04_15_pm/gpu1/qwen_st4_qbias_b-0.1_random_bont.json` |
| F9 | Qwen K-bias feature-shuffled control | 0.0000 | `reports/subtask4_overnight/st4_featshuffle_N0.json` |
| F10 | Qwen K-bias random control | 0.0000 | `reports/subtask4_overnight/st4_random_N0.json` |
| F11 | Best small-alpha augmentation on Qwen | 0.7529 at `alpha_K=0.025` | `reports/qkv_alpha_microsweep_2026_04_15/full497_alpha_microsweep.json` |
| F12 | Effective perturbation magnitude, real basis | 621.30 | `reports/stability_effective_magnitude_2026_04_15/result.json` |
| F13 | Effective perturbation magnitude, random basis | 399.56 | `reports/stability_effective_magnitude_2026_04_15/result.json` |
| F14 | Effective perturbation magnitude, feature-shuffled basis | 291.98 | `reports/stability_effective_magnitude_2026_04_15/result.json` |
| F15 | Theorem 6.1 empirical median ratio on Qwen | `2.357e-08` | `reports/theory_verify_2026_04_14/thm61_qwen_L13_a0.3_N100.json` |
| F16 | Theorem 6.1 empirical median ratio on Llama | `6.372e-08` | `reports/thm61_llama_2026_04_15/llama_L15_a0.3_N100.json` |

