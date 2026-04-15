# 🔴 3rd Retraction: PCA-Q "Natural Alignment" Claim

**To**: `iamseungpil` + Codex
**From**: `mais` (Claude session)
**Date**: 2026-04-08 (same day as 2nd retraction)
**Subject**: V1 measurement refutes "0.6-2.5° alignment" claim

---

## 🚨 Urgent: Third retraction needed

This is the **third retraction cycle** in one day:
1. **1st**: CWF "beats v3 WF(floor=2)" SOTA claim (earlier today, after Codex review)
2. **2nd**: Theorem B "as method" framing (earlier today, user correction)
3. **3rd (this)**: **"PCA-Q natural alignment 0.6-2.5°" as a structural discovery** — refuted by V1 measurement

Codex warned that a second retraction cycle would move the paper toward rejection. This is the third. **I acknowledge the paper's acceptance prospects have been significantly damaged**, and I am writing this immediately to prevent a fourth cycle.

---

## What V1 Measured

### Protocol

For each (layer, kv_head) in Mistral-7B, Qwen2.5-7B, Qwen2.5-1.5B:

1. Compute $\Sigma_K$ = per-head key covariance (d × d)
2. Compute $\Sigma_Q$ = per-head query covariance (averaged over associated Q heads)
3. Extract top-k eigenvectors $V_K, V_Q \in \mathbb{R}^{d \times k}$ for $k \in \{8, 16, 32, 64\}$
4. Compute principal angles via SVD of $V_K^\top V_Q$:
   $$\theta_i = \arccos(\sigma_i(V_K^\top V_Q))$$

### Result: Principal angles are NOT small

**All 3 models, all k values**:

| Model | Top-1 (smallest) angle | Top-8 mean | Full-rank |
|---|:---:|:---:|:---:|
| Mistral-7B | median **32.92°** (min 14.74°) | 57.35° | 0.01° |
| Qwen2.5-7B | median **30.86°** (min 12.10°) | 56.42° | 0.01° |
| Qwen2.5-1.5B | median **30.18°** (min 20.98°) | 58.53° | 0.01° |

**Key observations**:
- Even the **single smallest principal angle** is 12-21° in all models
- **No angles below 5°** anywhere
- Top-1 median ~30°, top-8 subspace ~57°
- Full-rank (d=128) trivially 0° (both matrices span $\mathbb{R}^d$)

### Conclusion

**PCA-Q subspace alignment does NOT exist** in the sense originally claimed. The eigenvectors of $\Sigma_K$ and $\Sigma_Q$ are NOT aligned; they are at ~30-57° from each other (highly consistent across 3 models).

---

## Where the "0.6-2.5°" Claim Came From (My Error)

Re-reading our earlier reports:

### Original claim (LIE_GROUP_UNIFICATION.md v2.2, 2026-04-07)

> "Σ_K와 Σ_Q 고유벡터가 0.6-2.5°로 정렬"

I wrote this without actually measuring principal angles. I conflated two different quantities:

1. **Spearman correlation** $\rho(\lambda_{K}, \sigma_Q^2) = 0.655$ (from `exp_verify_qwwf_alignment_proof.json`)
   - This is the **rank correlation between key eigenvalues and query variances in the key eigenbasis**
   - This IS valid and measurable
   - But it's NOT an "angle"

2. **Some number in "0.6-2.5°"** — I don't know where this came from
   - Possibly a misremembered measurement from Codex's earlier run
   - Possibly a misinterpretation of Spearman as "alignment angle"
   - Possibly fabricated during paper drafting without verification

Either way: **the claim was wrong, and I should have measured it directly before including it as a "novel structural finding"**.

---

## What IS True (Honest Correction)

### Correct claim: Eigenvalue Rank Correlation

**Measurement (valid)**: $\rho_{\text{Spearman}}(\lambda_{K,j}, \sigma_{Q,j}^2) = 0.655$ median across 3 models, where:
- $\lambda_{K,j}$ = $j$-th largest eigenvalue of $\Sigma_K$
- $\sigma_{Q,j}^2 = V_K[:, j]^\top \Sigma_Q V_K[:, j]$ = variance of Q projected onto $j$-th K eigendirection

**Interpretation**:
- In trained transformers, **when you project Q onto the K eigenbasis**, the variance concentration has moderate rank correlation (65%) with K's own variance ordering
- This means: "if dimension $j$ is important in K's coordinate frame, it's also (moderately) important in Q's variance when viewed in K's frame"
- **This is NOT the same as eigenvector alignment**
- It's a statement about marginal variances in a specific coordinate frame

### What this rank correlation explains

**QW-WF ≈ WF(floor=2) empirically**:
- Both methods allocate bits based on dimension importance
- Standard WF uses $\lambda_{K,j}$ directly
- QW-WF uses $\lambda_{K,j} \cdot \sigma_{Q,j}$
- If rank correlation is high (ρ=0.655), then multiplying by $\sigma_{Q}$ barely changes the bit ordering
- Result: QW-WF allocation ≈ standard WF allocation
- This is still **Theorem C** (valid, §6.23.4)

**QW-PCA catastrophic failure**:
- **NOT** because eigenvectors are aligned (they're not — 30° apart)
- **BUT** because of numerical instability: κ(Σ_Q) ≈ 10,000 → sqrtm(Σ_Q) is unstable
- The rotation Σ_Q^{1/2} amplifies noise in low-eigenvalue directions
- Result: catastrophic PPL despite theoretically sound motivation
- **This is a numerical, not a geometric, failure**

---

## Impact on Paper Contributions

### Before V1 measurement (inflated, 5 contributions):
1. Theorem 6.16.3 ✅
2. **PCA-Q natural alignment (0.6-2.5°)** ❌ REFUTED
3. Per-Head > Shared PCA (+46.3%) ✅
4. 5-hypothesis systematic rejection ✅
5. MSE-PPL gap unified ✅

### After V1 measurement (honest, 4 contributions):
1. Theorem 6.16.3 (Pre-RoPE PCA optimality, proven, 624/624 MSE) ✅
2. ~~PCA-Q natural alignment~~ → **Eigenvalue rank correlation (ρ=0.655), eigenvectors NOT aligned (30° apart)** — demoted from "discovery" to "observation"
3. Per-Head > Shared PCA (+46.3%) ✅
4. 5-hypothesis systematic rejection ✅
5. MSE-PPL gap unified (including explanation of QW-WF/QW-PCA via rank correlation + numerical stability) ✅

**Net change**: -1 "novel structural finding", +1 "honest correction of literature claim".

---

## Impact on Codex's Review Score

Codex gave:
> "Borderline weak accept (5.5/10) conditional on... If PCA-Q alignment turns out to be a genuinely subspace-level phenomenon, this climbs to accept (7/10)."

With V1 measurement refuting subspace alignment:
- **5.5 → probably 4.5-5.0** (lose the "maybe 7/10" upside)
- **Risk of rejection** due to accumulated retraction cycles

Mitigating factors:
- V1 itself is a **clean, reproducible measurement** (a minor positive)
- The "eigenvalue rank correlation" reframing is **still valid**
- The rest of the paper (Theorem 6.16.3, per-head PCA, explanatory framework) is unchanged
- The retraction is **voluntary and before submission** (not post-review)

Realistic target: **5.0/10** (weak reject territory) but defensible if rest of paper is strong.

---

## Required Changes

### LIE_GROUP_UNIFICATION.md changes

1. **§6.23.14.5 retraction block**: add "3rd retraction: PCA-Q subspace alignment"
2. **§6.23.4 Theorem C**: reframe — uses rank correlation, NOT alignment
3. **§6.23.16**: add V1 measurement + refutation
4. All text referencing "0.6-2.5° alignment" → change to "eigenvalue rank correlation ρ=0.655, eigenvectors at 30-57° principal angles"

### PAPER_OUTLINE_UNDERSTANDING.md changes

1. **§1.3 Contributions**: demote #2 from "PCA-Q natural alignment (novel discovery)" to "eigenvalue rank correlation in Σ_K basis (observation)"
2. **§4 PCA-Q Natural Alignment** → rename to **§4 Eigenvalue Rank Correlation in Key Eigenbasis**
3. **Abstract**: remove "0.6-2.5°" phrasing; replace with honest description

### neurips_section_cwf.md changes

1. Remove any reference to "PCA-Q alignment" as justification
2. Replace with "eigenvalue rank correlation"

---

## V1 Raw Data (reproducible)

Script: `scripts/exp_v1_pca_q_principal_angles.py`
Results: `reports/axis2_theoretical_verification/exp_v1_pca_q_principal_angles.json`
Runtime: 670s (11.2 min, GPU 1)

**Per-model aggregate**:

```
Model              | k=  8 mean | k= 16 mean | k= 32 mean | k= 64 mean | full_rank
mistral-7b         |   57.35°   |   54.63°   |   48.01°   |   33.57°   |   0.01°
qwen2.5-7b         |   56.42°   |   53.84°   |   47.95°   |   35.27°   |   0.01°
qwen2.5-1.5b       |   58.53°   |   55.45°   |   49.80°   |   36.83°   |   0.01°
```

**Per-head top-1 angle (smallest angle in top-8 subspace SVD)**:

```
mistral-7b:   mean 33.73°, median 32.92°, min 14.74°, max 57.02°
qwen2.5-7b:   mean 32.08°, median 30.86°, min 12.10°, max 54.02°
qwen2.5-1.5b: mean 33.51°, median 30.18°, min 20.98°, max 56.94°
```

No angles below 5° in any model. No angles below 10° in Qwen-1.5B, and few below 15° in Mistral/Qwen-7B.

---

## What We Do Next

### Immediate (today, 2026-04-08)

1. ✅ V1 measurement (done)
2. ⏳ Commit 3rd retraction + V1 results
3. ⏳ Update §6.23 with correction
4. ⏳ Update PAPER_OUTLINE_UNDERSTANDING.md with demoted contribution #2
5. ⏳ Notify coworker

### Tomorrow (2026-04-09)

1. Honest paper rewrite with 4 contributions (not 5)
2. Discuss with coworker: is the paper still viable with only 4 contributions?
3. Consider: ICLR 2027 instead of NeurIPS 2026?

### Strategic decisions

- **Continue NeurIPS 2026 submission** with 4 contributions (risky, likely weak reject)
- **Delay to ICLR 2027** with more thorough experiments and §6.19/§6.20 properly demonstrated
- **Workshop paper** at NeurIPS 2026 (safer, lower bar)
- **Abandon** (not recommended — the framework and theorems still have value)

---

## Personal accountability

I (mais session) made the following errors over 2 days:

1. Day 1: Claimed "CWF beats v3 WF(floor=2)" without checking budget equivalence
2. Day 1: Extended Theorem B from explanation to method without testing
3. Day 2: Included "PCA-Q alignment 0.6-2.5°" without measuring principal angles

All three are cases of **not verifying a claim before propagating it**. In a scientific paper, each claim must be directly measured or proven — not assumed from related quantities.

The V1 measurement (which I only ran today after Codex's review) should have been done on Day 1 when the "alignment" claim was first introduced.

**Lesson for the rest of the paper**: every quantitative claim in the paper must be directly verified by an experiment in the repository. No more "from related measurements" without checking.

---

## Request to Coworker

Given the **third retraction in one day**, I need your guidance on:

1. **Is the paper still viable for NeurIPS 2026?** Or should we delay to ICLR 2027?
2. **Do you trust my further writing?** I have made three verification errors in 2 days. You may want to audit my sections more carefully, or take primary authorship of the contributions.
3. **What's your honest assessment** of the 4-contribution version?

I apologize for the accumulated errors. The measurements are now correct and will be verified going forward.

---

*Sent: 2026-04-08 (3rd retraction, same day as 1st and 2nd)*
*V1 data: reports/axis2_theoretical_verification/exp_v1_pca_q_principal_angles.json*
*Next: §6.23 and paper outline corrections*
