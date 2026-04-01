# Critic Corrections Log

## Round 1 (Boltzmann 1-1, 1-2)
- HeadVar middle: 0.036 → **0.037** (FIXED in TeX + log)
- "reheat final 0.76": specify L34-L35 (FIXED in TeX)

## Round 2 (1-3, 2-1, 2-2)
- Exp 2-2 KL at 8/2: claimed PCA=4.75, LDA=13.28 → **actual PCA=1.83, LDA=11.87**
  - 4.75 is kl_per_layer[4], not mean
  - 13.28 does not exist in data
  - Source: exp2_2_results.json comparisons[1].pca_mean_kl=1.8265, lda_mean_kl=11.8657
- Exp 2-2 logic: V_MSE is nearly identical PCA vs LDA (within 1%).
  K_MSE drives the conclusion. V_MSE should be acknowledged.

## Verified (all correct)
- All 1-3 numbers: 3.376, 3.255, -0.122, t=-11.79, p=3.15e-11, W=0.0, 0/24
- All 2-1 numbers: 0.819, 0.411, 0.408, t=30.50, p=1.6e-64, U=5177, 0.343
- 2-2 K_MSE ratios: 26x (6/2), 30x (8/2) — verified

---

## Round 3 (experiment_log_v2.md full verification — 2026-03-29)

### FAIL items

#### F1: GQA KV0 Mean — Logical Contradiction in Text
- **Claim**: "KV Group 0의 평균 T_eff = 0.548 (H5 제외 시)", gap = 0.506
  (source: experiment_log_v2.md §2.5, cites exp1_1_results.json layer_head_teff[5])
- **Actual**:
  - KV0 mean WITH H5 = 0.548 ✓ (matches JSON)
  - KV0 mean WITHOUT H5 (excl H5) = 0.615
  - If excl H5: gap = 0.615 - 0.042 = 0.573
  - If incl H5: gap = 0.548 - 0.042 = 0.506 ✓
- **Issue**: The text says "(H5 제외 시)" = "excluding H5", but 0.548 is the value INCLUDING H5. The stated exclusion is false.
- **Fix**: Change "KV Group 0의 평균 T_eff = 0.548 (H5 제외 시)" to "KV Group 0의 평균 T_eff = 0.548 (H5 포함)" and update gap logic accordingly. The gap 0.506 is numerically correct when H5 is included; the "(H5 제외 시)" parenthetical is wrong.

#### F2: Exp 1-2 Mid-Late Layer Group rho Mean
- **Claim**: Mid-Late (L12-L17) mean rho(C, ||k||^2) = 0.183
  (source: experiment_log_v2.md §3.3 table, cites exp1_2_results.json layer_stats)
- **Actual**: L12=0.207, L13=0.288, L14=0.221, L15=0.272, L16=0.183, L17=0.104 → mean = 0.212
- **Discrepancy**: claimed 0.183, actual 0.212 (diff = 0.029, ~15.8% error)
- **Fix**: Update Mid-Late group rho to 0.212 in the table.

#### F3: Exp 1-2 Middle Layer Group rho Mean (minor)
- **Claim**: Middle (L6-L11) mean rho = 0.219
- **Actual**: L6=0.269, L7=0.072, L8=0.403, L9=0.130, L10=0.239, L11=0.241 → mean = 0.226
- **Discrepancy**: claimed 0.219, actual 0.226 (diff = 0.007, ~3.2% error)
- **Fix**: Update Middle group rho to 0.226.

#### F4: FOKVQ Cross-Analysis Numbers — Source File Not Found
- **Claims in §4.2 and §5.1**: rho(T_eff, r_eff) = -0.084 (p=0.695), rho(C(q), Facet Gain) = -0.63 (p=0.001), T_eff vs Facet Gain rho = +0.30 (p=0.16), r_eff range 7.17-9.50, STRONG 100% (72/72), 2-bit KL 61-69%, 3-bit 60-65%, 4-bit 31-73%, 8-bit reversal 5.0-7.9x, r_eff vs Facet Gain rho = -0.6811 (p=1.13e-53)
- **Source cited**: FOKVQ_FINAL_REPORT.md
- **Actual**: FOKVQ_FINAL_REPORT.md does NOT EXIST as a .md file. Files found are .docx variants at /home/v-seungplee/boltzmann-attention/:
  - FOKVQ_FINAL_REPORT.docx
  - FOKVQ_ADDITIONAL_EXPERIMENT_PLAN.docx
  - FOKVQ_PAPER_v3.docx
  - FOKVQ_EXPERIMENT_RESULTS_REPORT.docx
- **Status**: UNCITABLE — all FOKVQ cross-analysis numbers lack verifiable .md source. Cannot confirm or deny correctness.
- **Fix**: Cite the correct source (.docx filename + section), or export relevant sections to a .md file that can be machine-verified.

### WARNING items (not FAIL, but noteworthy)

#### W1: Critic Log Round 2 "0.343" for Exp 2-1
- The critic_corrections.md Round 2 lists "0.343" as a verified value for Exp 2-1.
- The JSON field `separability_gap` = 0.408. No field with value 0.343 exists at the top-level result.
- Value 0.343 appears in `pair_similarities.code_vs_conversational[22]` (L22 pair similarity).
- The "0.343" in the Round 2 verified list is ambiguous/misattributed. It should be clarified what metric it refers to (it does not appear to be a main result metric).

#### W2: Exp 2-3 — Degenerate Result (NaN-heavy)
- exp2_3_results.json contains NaN for all rank_corr fields except L0.
- `query_dependent_allocation_differs = false`, `allocation_varies_across_queries = false`.
- alloc_l1_distance for L1-L23 ≈ exactly 1.0 (within floating-point noise), suggesting the allocation reduces to a constant (no variance in key norms → single bucket assignment).
- This is a potential implementation issue (degenerate allocation), not a valid negative result. The experiment result should be flagged as inconclusive rather than interpreted as "static allocation suffices."

### Verified correct in v2 log

- GPT-2 rho=-0.6574, p=4.82e-04 (exp1_1_gpt2_results.json) ✓
- All 24 GPT-2 per-layer T_eff values (exp1_1_gpt2_results.json layer_teff_mean) ✓
- GPT-2 global min at L19=0.221, final reheat L23=0.520 ✓
- GPT-2 head variance: L0=0.0080, L11=0.032, L4=0.029, L19=0.009 ✓
- Qwen rho=+0.1081, p=0.530 ✓
- Qwen Layer 5 all 16 head T_eff values ✓
- Qwen KV1 mean = 0.042, gap = 0.506 (when KV0 incl H5; see F1 for text error) ✓
- Qwen head variance top-5 layers: 0.0857, 0.0683, 0.0661, 0.0577, 0.0526 ✓
- 15.9x ratio L5/L0 head variance ✓
- Exp 1-2 avg_disagreement=39.8%, avg_correlation=0.275 ✓
- All PPL values: baseline 139.85, 2-bit 419.41, variance 4.85e+08, specific_heat 4.85e+08, 3-bit 130.38/127.62/264.48, 4-bit 110.19/122.85/125.56 ✓
- All per-layer disagreement rates and individual rho_C_k values ✓
- Layer group Early mean: disagr=30.1%, rho=0.526 ✓; Late: disagr=45.2%, rho=0.136 ✓
- Exp 1-3: mean_diff=-0.122, t=-11.79, p=3.15e-11, W=0.0, 0/24 ✓
- Exp 2-1: within=0.819, cross=0.411, gap=0.408, t=30.50, p=1.6e-64, U=5177 ✓
- Exp 2-2: 8/2 PCA KL=1.826, LDA KL=11.866, K_MSE ratios 26x/30x ✓
- Exp 2-2: V_MSE nearly identical PCA vs LDA (within 1%) ✓

### Overall Verdict for v2 log: FAIL
Reasons: F1 (text/logic error), F2 (numerical mismatch 15.8%), F3 (minor 3.2%), F4 (uncitable FOKVQ numbers).

---

## Round 4 (Exp 2-4 and Exp 3-1 verification — 2026-03-29)

### Exp 2-4: AFOD vs PCA vs Random vs Identity

**Overall verdict: PASS**

Verified correct:
- All 12 mean MSE values (4 methods × 3 bit widths) — each confirmed as mean of 24 per-layer values ✓
- All 12 mean KL values — each confirmed as mean of 24 per-layer KL values ✓
- afod_wins_mse = 2 — correct: AFOD < PCA for 3-bit and 4-bit MSE, not 2-bit ✓
- pca_beats_random_mse = 3 — correct: PCA < Random for all three bit widths ✓
- All afod_improvement_mse_pct and afod_improvement_kl_pct values — correctly computed as mean of per-layer (pca-afod)/pca×100, NOT from global means ✓
- All statistical test t-statistics and p-values (afod_vs_pca, pca_vs_random, pca_vs_identity) reported as-is from the JSON ✓

One noteworthy structural point (not a FAIL):
- improvement_pct fields are computed as the mean of per-layer relative improvements, not from the ratio of global means. This can give qualitatively different signals at the global level (e.g., 2-bit AFOD has global_mean_improvement = -0.74% by global means, but +0.23% by per-layer mean). The JSON is internally self-consistent; any external report should clarify which formula was used.

### Exp 3-1: Continuous Rotation Optimization (Lie Group)

**Overall verdict: PASS with one minor anomaly**

Verified correct:
- All 8 per-layer mse_identity, mse_pca, mse_random, mse_optimized values ✓
- All per-layer improvement_vs_identity_pct, improvement_vs_pca_pct, improvement_vs_random_pct — formula: (baseline-opt)/baseline×100 ✓
- Overall means: identity=961.01, random=1047.17, pca=220.45, optimized=938.81 ✓
- mean_improvement_vs_pca_pct = -391.82 — mean of per-layer improvement values ✓
- Ranking: pca (1) > optimized (2) > identity (3) > random (4) ✓
- optimized_is_best=False: correct (optimized MSE 938.8 >> pca MSE 220.5) ✓
- optimized_significantly_beats_pca=False: correct — t=-3.45 (p=0.011) significant but in wrong direction ✓
- opt_history_start == mse_identity for all 8 layers ✓
- All optimization_time_s and num_function_evals present ✓
- Total runtime = 3718.5s = 1.03h; sum of per-layer times = 3673.4s (overhead 45s plausible) ✓

Minor anomaly (not a FAIL):
- L21 opt_history_end (1751.73) differs from mse_optimized (1748.49) by 3.25 (0.19%). Likely because opt_history_end records the optimizer callback value while mse_optimized is computed by a full separate evaluation. All improvement percentages use mse_optimized (the correct value).

Key scientific finding confirmed by data:
- Lie group optimization (64 Givens planes, 200 iters) does NOT outperform PCA at any layer: mean_improvement_vs_pca_pct = -391.8% (optimized is far worse than PCA). Optimized beats identity by only 0.5-4.9% per layer. PCA dominates over random (t=3.68, p=0.008).
- optimized_vs_identity: t=2.07, p=0.078 — marginally non-significant improvement over identity alone.

---

## Round 5 (boltzmann_full_report_ko.tex full verification — 2026-03-29)

### Data Integrity Alert: exp2_3_results.json replaced

The exp2_3_results.json file that existed at the time of Round 3 verification (and which the reporter used for boltzmann_full_report_ko.tex §2.3) was overwritten at 17:31 on 2026-03-29. The restored file (17:34) contains a **completely different dataset**:

| Field | Old data (TeX basis) | New data (current JSON) |
|-------|---------------------|------------------------|
| query_dependent_allocation_differs | false | true |
| allocation_varies_across_queries | false | true |
| overall alloc_l1_distance_mean | ~1.024 | 1.1701 |
| overall static_vs_qdep_rank_corr_mean | NaN (23/24 layers) | 0.6976 (0/24 NaN) |
| l1_gt_0 t_stat | 472.7 | 575.1 |
| Layer 0 rank_corr_mean | -0.348 | 0.239 |
| Layer 0 Q-Q rank_corr_mean | 0.600 | 0.577 |

The TeX report §2.3 and related synthesis claims (§4 verdict table, §5 conclusion) are based on the OLD data and are factually inconsistent with the current exp2_3_results.json.

### FAIL items

#### F5: Exp 2-3 Section — Complete Data Mismatch (CRITICAL)

All numerical claims in §2.3 (Table tab:exp23_main) are wrong relative to current exp2_3_results.json:

| Claimed in TeX (old data) | Actual in current JSON | Status |
|---------------------------|------------------------|--------|
| Mean L1 distance = 1.024 | 1.1701 | MISMATCH |
| L1 std = 0.126 | 0.1261 | OK (matches) |
| Layers with valid rank rho = 1/24 | 24/24 (no NaN) | MISMATCH |
| Layer 0 rank rho = −0.348 ± 0.348 | 0.239 ± 0.047 | MISMATCH |
| Layer 0 Q-Q rho = 0.600 ± 0.488 | 0.577 ± 0.144 | MISMATCH |
| Layers 1–23 L1 ≈ 1.000 (std < 1e-7) | L1 range 1.037–1.515, std 0.019–0.044 | MISMATCH |
| query_dependent_allocation_differs = false | true | MISMATCH |
| allocation_varies_across_queries = false | true | MISMATCH |
| t-stat (L1 > 0) = 472.7 | 575.1 | MISMATCH |

Downstream impacts of F5:
- §4 verdict table row 2-3: "L1 ≈ 1.0 / Static Sufficient" — wrong conclusion given current data
- §5 conclusion bullet: "정적 PCA 할당이 query-dependent 동적 할당과 동일한 결과를 산출한다 (23/24 layers에서 L1 ≈ 1.0)" — wrong given current data
- Abstract: "query-dependent 동적 할당과 AFOD 초기화 모두 정적 PCA 대비 유의한 개선을 보이지 못하였다" — the query-dependent part is no longer supported (current data shows differs=true, varies=true)

**Fix**: The team must determine which exp2_3 dataset is authoritative — the old (degenerate) result or the new (non-degenerate) result. The TeX §2.3 section, verdict table row, conclusion bullet, and abstract must all be rewritten once the authoritative data is confirmed. The old interpretation (STATIC SUFFICIENT) is NOT supported by the current JSON.

**Note on L1 std**: L1 std = 0.126 (old) ≈ 0.1261 (new). This one value coincidentally matches across both datasets and is not evidence that the data is the same.

### Verified correct in boltzmann_full_report_ko.tex

#### Exp 2-4 Section (§2.4) — PASS
All TeX claims verified against restored exp2_4_results.json (confirmed byte-identical to previously verified data):
- All 12 mean MSE values (Table tab:exp24_mse) — all within rounding ✓
- All t-statistics and p-values (Table tab:exp24_stat) — all within 2 significant figures ✓
- All afod_improvement_mse_pct values (0.2%, 0.6%, −0.5%) — all within 1 decimal place rounding ✓
- PCA vs Random ratios (5.6x at 2-bit, 2.3x at 4-bit) — confirmed ✓
- afod_wins_mse = 2, pca_beats_random_mse = 3 — confirmed ✓

#### Exp 3-1 Section (§3.1) — PASS
All TeX claims verified against restored exp3_1_results.json (confirmed byte-identical to previously verified data):
- All 4 overall mean MSE values (Table tab:exp31_rank) — all within rounding ✓
- All 8 × 4 per-layer MSE values (Table tab:exp31_layer) — all within rounding ✓
- All per-layer improvement_vs_pca_pct values — all within rounding ✓
- All 3 statistical test t/p values (t=−3.45/p=0.011, t=3.68/p=0.008, t=2.07/p=0.078) ✓
- mean_improvement_vs_pca_pct = −391.8% ✓
- 4.3x ratio (938.8/220.5 = 4.26, reported as 4.3x — acceptable 1 sig fig) ✓
- optimized_is_best=False, optimized_significantly_beats_pca=False ✓

### Overall Verdict for boltzmann_full_report_ko.tex: FAIL
Reason: F5 — Exp 2-3 section and downstream synthesis text are based on superseded/replaced data that contradicts current exp2_3_results.json. The scientific conclusion for Exp 2-3 may need to be reversed depending on which dataset is authoritative.
