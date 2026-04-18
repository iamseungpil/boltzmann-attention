# Paired Bootstrap Significance — 2026-04-18

10000 iterations per comparison, paired by task_id. Two-sided p-value.

| Comparison | N | ΔF1 | 95% CI | p | Significant |
|---|---:|---:|---|---:|:---:|
| **Qwen retail** ladapt (fixed) − no_steer | 114 | +6.05pp | [+1.57, +10.70] | 0.007 | ✅ |
| **Qwen retail** ladapt − Q-only(−0.03) | 114 | +0.94pp | [−3.93, +5.76] | 0.714 | ❌ tied |
| **Qwen telecom** ladapt (fixed) − no_steer | 200 | +26.76pp | [+23.31, +30.23] | <0.001 | ✅ |
| **Qwen telecom** ladapt − Q+0.10 | 200 | +4.03pp | [+0.12, +7.77] | 0.044 | ✅ |
| **Qwen ST4** Q-only(−0.03) − ladapt | 497 | +0.89pp | [−0.76, +2.57] | 0.298 | ❌ tied |
| **Qwen airline** ladapt − no_steer | 50 | +3.83pp | [−1.58, +9.14] | 0.165 | ❌ underpowered |
| **Llama retail** ladapt − no_steer | 114 | −1.68pp | [−4.37, +0.93] | 0.207 | ❌ tied |
| **Llama telecom** ladapt − no_steer | 200 | +11.62pp | [+9.34, +13.93] | <0.001 | ✅ |

## Interpretation for paper

**What is statistically defensible:**
1. Layer-adaptive beats `no_steer` significantly on Qwen retail, Qwen telecom, Llama telecom.
2. Layer-adaptive beats Q-only significantly ONLY on Qwen telecom (p=0.044). On ST4 and retail the two are statistically indistinguishable.
3. Llama retail ladapt is NOT significantly worse than no_steer (p=0.207, CI contains 0) — paper must NOT claim "ladapt hurts on Llama retail"; correct wording is "no significant effect."
4. Qwen airline N=50 is underpowered (p=0.165) — the +3.84pp headline cannot be claimed as "significant."

**What the paper should say instead of "ladapt is winner":**
- "Layer-adaptive K+Q matches or exceeds Q-only across Qwen domains, with a statistically significant advantage on Qwen telecom."
- "No statistically significant difference between layer-adaptive and Q-only on Qwen retail or MetaTool ST4; both are significantly better than no_steer on retail."
- "Llama retail shows no significant effect of either operator (baseline ceiling). Llama telecom reproduces the Qwen layer-adaptive gain significantly (+11.62pp, p<0.001)."

## Llama telecom β+0.05 format collapse

Separate audit of per-sample `pred_tools` shows:

| Method | empty / 200 | F1=0 / 200 |
|---|---:|---:|
| β=−0.10 | 0 | 61 |
| β=−0.05 (best) | 0 | 4 |
| β=−0.03 | 0 | 10 |
| β=+0.01 | 0 | 30 |
| **β=+0.03** | **145** | 151 |
| **β=+0.05** | **200** | 200 |
| **β=+0.10** | **55** | 79 |
| ladapt | 0 | 10 |

Positive Q-rotation destabilises Llama's tool-call format. At β=+0.05 every prompt produces a non-parseable output (empty `pred_tools`), not "wrong tool". This is format-stability collapse, not semantic misalignment — same operator family still transfers, but its magnitude/polarity must be calibrated per model.
