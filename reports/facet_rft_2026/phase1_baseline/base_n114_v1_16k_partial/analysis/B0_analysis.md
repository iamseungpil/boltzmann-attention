# B0 Vanilla Baseline — Phase 1 N=114 Trials=4 Analysis

- results file: `reports/facet_rft_2026/phase1_baseline/base_n114/B0_telecom_base.json/results.json`
- git_commit: `unknown`
- agent: `openai/Qwen2.5-7B-Instruct` · max_steps=200 · seed=42

## 1. Headline metrics

| Metric | Value |
|---|---|
| Total simulations | 456 |
| Evaluated (exclude infra errors) | 421 |
| Infrastructure errors | 35 |
| **Avg reward (evaluated)** | **0.0475** |
| Avg reward (all, infra=0) | 0.0439 |
| Full-credit rate (reward≥1.0) | 20/421 = 0.0475 |
| Wilson 95% CI (full-credit) | [0.0310, 0.0722] |

## 2. pass^k (task-level)

"Pass" = at least 1 of k trials achieves reward ≥ threshold and is not an infrastructure error.

| k | passed tasks | pass^k |
|---|---|---|
| 1 | 2/114 | 0.0175 |
| 2 | 8/114 | 0.0702 |
| 3 | 11/114 | 0.0965 |
| 4 | 15/114 | 0.1316 |

## 3. Termination breakdown

| Termination | Count | Share |
|---|---|---|
| user_stop | 254 | 0.557 |
| max_steps | 153 | 0.336 |
| infrastructure_error | 35 | 0.077 |
| too_many_errors | 14 | 0.031 |

## 4. Termination × Reward cross-tab

| Termination | n | avg_reward | pass_rate (reward≥1) |
|---|---|---|---|
| user_stop | 254 | 0.0787 | 0.0787 |
| max_steps | 153 | 0.0000 | 0.0000 |
| infrastructure_error | 35 | 0.0000 | 0.0000 |
| too_many_errors | 14 | 0.0000 | 0.0000 |

## 5. Persona breakdown

| Persona | n_total | n_evaluated | avg_reward (evaluated) |
|---|---|---|---|
| Easy | 152 | 140 | 0.0429 |
| Hard | 144 | 136 | 0.0735 |
| None | 160 | 145 | 0.0276 |

## 6. Per-category breakdown (task_id prefix)

| Category | n_sims | avg_reward | top termination |
|---|---|---|---|
| service_issue | 116 | 0.1379 | user_stop |
| mobile_data_issue | 144 | 0.0278 | user_stop |
| mms_issue | 196 | 0.0000 | user_stop |

## 7. Trial variance (within-task across 4 trials)

- Mean per-task reward std: **0.0692**
- Tasks with ≥2 evaluated trials: 114/114

## 8. Duration (seconds per simulation)

- mean=128.3, median=112.2, p95=269.9
- min=0.0, max=322.8
- evaluated only mean=138.9

## 9. Message count per simulation

- mean=96.1, median=56.0, p95=201.0, max=206

## 10. Reward basis distribution

- ENV_ASSERTION: 254 (0.809)
- ACTION: 60 (0.191)

- DB match (evaluated): 37/254 = 0.1457

## 11. Worst 10 tasks by avg reward (across 4 trials)

- `[mobile_data_issue]data_mode_off|data_usage_exceeded[PERSONA:None]` → avg_reward=0.0000
- `[mobile_data_issue]data_usage_exceeded|user_abroad_roaming_enabled_off[PERSONA:Easy]` → avg_reward=0.0000
- `[mobile_data_issue]data_saver_mode_on|data_usage_exceeded[PERSONA:Easy]` → avg_reward=0.0000
- `[mobile_data_issue]airplane_mode_on|data_saver_mode_on|user_abroad_roaming_disabled_on[PERSONA:None]` → avg_reward=0.0000
- `[mobile_data_issue]airplane_mode_on|bad_network_preference|user_abroad_roaming_enabled_off[PERSONA:Easy]` → avg_reward=0.0000
- `[mobile_data_issue]bad_vpn|data_saver_mode_on|user_abroad_roaming_disabled_on[PERSONA:None]` → avg_reward=0.0000
- `[mobile_data_issue]data_mode_off|data_usage_exceeded|user_abroad_roaming_disabled_off[PERSONA:Hard]` → avg_reward=0.0000
- `[mobile_data_issue]bad_network_preference|bad_vpn|user_abroad_roaming_disabled_off[PERSONA:Hard]` → avg_reward=0.0000
- `[mobile_data_issue]bad_network_preference|data_saver_mode_on|data_usage_exceeded[PERSONA:Hard]` → avg_reward=0.0000
- `[mobile_data_issue]bad_network_preference|user_abroad_roaming_enabled_off[PERSONA:Hard]` → avg_reward=0.0000

## 12. Best 10 tasks (avg reward > 0)

- `[service_issue]airplane_mode_on|break_apn_settings|contract_end_suspension|lock_sim_card_pin[PERSONA:None]` → avg_reward=0.5000
- `[service_issue]break_apn_settings|lock_sim_card_pin|overdue_bill_suspension|unseat_sim_card[PERSONA:Easy]` → avg_reward=0.5000
- `[service_issue]airplane_mode_on|contract_end_suspension|lock_sim_card_pin|unseat_sim_card[PERSONA:Hard]` → avg_reward=0.5000
- `[service_issue]airplane_mode_on|lock_sim_card_pin[PERSONA:Easy]` → avg_reward=0.5000
- `[mobile_data_issue]airplane_mode_on|bad_network_preference[PERSONA:Hard]` → avg_reward=0.5000
- `[service_issue]airplane_mode_on|break_apn_settings|lock_sim_card_pin|overdue_bill_suspension[PERSONA:Hard]` → avg_reward=0.2500
- `[service_issue]airplane_mode_on|break_apn_settings|contract_end_suspension|lock_sim_card_pin|unseat_sim_card[PERSONA:Eas...` → avg_reward=0.2500
- `[service_issue]airplane_mode_on|break_apn_settings|lock_sim_card_pin[PERSONA:None]` → avg_reward=0.2500
- `[service_issue]airplane_mode_on|lock_sim_card_pin|overdue_bill_suspension|unseat_sim_card[PERSONA:Easy]` → avg_reward=0.2500
- `[service_issue]contract_end_suspension|lock_sim_card_pin[PERSONA:Hard]` → avg_reward=0.2500

## 13. Hallucination retries

- Total retries used: 0
- Sims with ≥1 retry: 0/456
