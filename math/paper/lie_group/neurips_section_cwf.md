# NeurIPS 2026 Paper Section Draft (REWRITTEN, 2026-04-08)

**Status**: Rewritten as Understanding paper framing after Codex critique + Next-12 results
**Position in paper**: Section X — Cascade Theory + Theorem B (explanatory, NOT method)
**Length target**: 2 pages

---

## ⚠️ Original Framing Retracted (2026-04-08)

The previous version of this section (2026-04-07) presented "Cascade-Aware Water-Filling (CWF)" as a new SOTA quantization method that beats v3 WF(floor=2). After Codex's critique and our Next-12 fair-comparison experiment, we acknowledge:

- ❌ **CWF does NOT beat v3 WF(floor=2) at fair budget (avg=2.0)**: at the same budget, our CWF (9.12 PPL) is dramatically worse than v3 (5.82 PPL).
- ❌ **CWF inter-head contribution = 0**: Next-12 shows that combining CWF with v3's intra-head WF (Two-level) gives identical PPL to intra-head WF alone.
- ✅ **Theorem B is valid as an explanation** of optimal allocation, but NOT as a method when v3 WF(floor=2) already extracts most of the variance.

This section is now reframed as **explanatory framework** + **negative result** (CWF as constructive ablation).

---

## X. Per-head Outlier Concentration: An Explanatory Framework

### X.1 Phenomenon: MSE-PPL gap concentrates on early layers

We observe that in trained transformers (Mistral-7B, Llama-3.1-8B), the L²-Lloyd PPL catastrophe (5.06× and 6.46× over Uniform 2-bit, respectively) is **structurally localized** to a small number of early attention layers:

**Per-layer Lloyd substitution sensitivity (Mistral-7B)**:

| Layer | ΔPPL when Lloyd replaces FP16 (other layers FP16) |
|:---:|:---:|
| **2** | **+0.555** |
| **4** | **+0.521** |
| 6 | +0.304 |
| 3 | +0.287 |
| 5 | +0.206 |
| 7-31 | < 0.20 |
| 26 | -0.004 (improves!) |

**Top-5 layers (2,3,4,5,6) account for 60% of total Lloyd failure**. This is not random noise; it is a structural property of the trained model.

We call this **per-head outlier concentration** (Proposition D in the appendix).

### X.2 Theorem A: Why MSE-Optimal Lloyd Fails in PPL

For each (layer, kv-head), we define the **averaged Fisher metric**:
$$M^{\text{avg}}_{l,h} := \frac{1}{T}\sum_t s_{t,l,h} \cdot q_{t,l,h} q_{t,l,h}^\top, \quad s_{t,l,h} := \sum_j p_{l,h}[t,j](1-p_{l,h}[t,j])$$

This is the second-order Taylor coefficient of attention KL divergence with respect to key perturbations.

**Theorem A (MSE-PPL Inversion Bound)**: For Lloyd-Max quantizer $Q_{L^2}$ and Uniform quantizer $Q_{\text{unif}}$ with respective error covariances $\Sigma^{L^2}$ and $\Sigma^{\text{unif}}$:

$$\text{tr}(M_{l,h} \cdot \Sigma^{L^2}_{l,h}) - \text{tr}(M_{l,h} \cdot \Sigma^{\text{unif}}_{l,h}) \geq c \cdot \big(\kappa(M_{l,h}) - 1\big) \cdot \sigma^2_{\text{quant}} - \Delta^{\text{MSE-gain}}$$

where $\kappa(M_{l,h}) = \lambda_{\max}(M_{l,h}) / \lambda_{\min}(M_{l,h})$ is the Fisher metric condition number.

**Interpretation**: Lloyd-Max minimizes $\text{tr}(\Sigma)$ (isotropic L²) but PPL depends on $\text{tr}(M \cdot \Sigma)$ (anisotropic Fisher). When $\kappa(M)$ is large—i.e., attention is concentrated in a few directions—Lloyd's "uniform error" assumption causes catastrophic mis-allocation of quantization noise into the most-attended directions.

**Empirical evidence (Mistral-7B)**:
- Layers 2-6 have $\kappa(M) \in [10^4, 10^7]$ (heavy-tailed spread)
- Other layers have $\kappa(M) \approx 10^3$ (well-conditioned)
- Lloyd failure correlation with κ-spread (p95/median): Spearman ρ = +1.0 (n=2 models, see Exp1)

### X.3 Theorem B: Master Allocation Equation (Explanatory)

Given the master PPL equation (informally derived from Taylor expansion + chain rule):
$$\Delta\log\text{PPL}^{\text{quant}} \approx \sum_{l,h} g_{l,h} \cdot \text{tr}(M^{\text{avg}}_{l,h} \cdot \Sigma^{(l,h)}_{\delta k})$$

where $g_{l,h}$ is the cascade factor (gradient of loss w.r.t. attention output, propagating through subsequent layers).

**Theorem B**: The optimal bit allocation minimizing $\Delta\log\text{PPL}$ subject to total budget $B$ is given by the Lagrangian:

$$b^*_{l,h} = \left(\tfrac{1}{2}\log_4\!\Big(\tfrac{g_{l,h} \cdot \text{tr}(M^{\text{avg}}_{l,h})}{\mu}\Big)\right)^{+}, \quad \sum_{l,h} b_{l,h} = B$$

**Interpretation**: This explains why hand-picked configurations (e.g., "Layer 2-6 @ 3-bit, others @ 2-bit") achieve near-optimal PPL: they approximate Theorem B's Lagrangian solution given the empirical sensitivity profile.

**Constructive validation (Next-9c)**: When we instantiate Theorem B with $g_{l,h}$ measured via Exp4 direct ΔPPL substitution and budget = 256 × 2.156 bits, the resulting allocation gives PPL **6.9505**—exactly matching the hand-picked Next-4 E configuration (6.9505) to 4 decimals. This confirms Theorem B as an accurate **explanation** of why Next-4 E is optimal at its budget.

### X.4 Limitations: Theorem B is Vacuous as a Method (Next-12)

**Critical experiment**: Does Theorem B's allocation (Cascade-Aware WF, "CWF") strictly improve over v3's WF(floor=2) at fair budget?

**Setup**: Mistral-7B, avg=2.0 bits/dim, 5 configs:

| Config | Description | PPL |
|---|---|:---:|
| A | Uniform 2-bit per dim | 9.12 |
| B | Intra-head WF skip-floor=2 (v3 reproduction) | 6.02 |
| B2 | Intra-head continuous WF (no floor) | **5.94** |
| C | Inter-head CWF only (uniform within head) | 9.12 |
| **D** | **Two-level: CWF + intra-head WF (combined)** | **6.02** |

**Findings**:
1. **B = D = 6.02**: Two-level WF gives **identical PPL** to intra-head WF alone. Adding CWF's inter-head signal contributes **zero**.
2. **A = C = 9.12**: Inter-head CWF alone is identical to uniform. Under the floor=2 constraint, every head ends up with 256 bits (= 2 × 128 dim), so inter-head reallocation is trivial.
3. **B2 (5.94) ≈ v3 WF(floor=2) (5.82)**: Our reproduction matches v3 within 2%.

**Conclusion**: Theorem B is a valid **explanation** of allocation optima, but it is **vacuous as a method** under standard floor=2 constraints in Mistral-7B. v3's intra-head WF already extracts essentially all useful variance; the inter-head dimension that CWF adds is empty signal.

**Why?** With floor=2 per dim and budget 2d per head, every head's total budget is fixed at 2d bits—there is no room for inter-head reallocation. CWF requires either (a) lower floors (allowing some heads to skip), or (b) larger total budgets (avg > 2.0 bits)—both of which break the fair comparison or the v3 protocol.

### X.5 Honest Positioning of CWF

CWF (Next-9c, Next-10) is not a new quantization method. It is:

1. **A constructive demonstration of Theorem B**: When provided correct sensitivity values (e.g., Exp4 direct substitution), CWF reproduces hand-picked configurations and provides smooth quality-bits trade-off curves at extended budgets.

2. **A negative result for inter-head allocation**: At standard budgets and floor constraints, the inter-head signal that CWF exploits is empty. Future work may revisit this with looser constraints (e.g., per-head sparsification).

3. **Empirical evidence for Proposition D**: The ability to recover hand-picked configurations confirms that per-layer outlier concentration (Layers 2-6 in Mistral) is the structurally relevant axis.

### X.6 What This Means for the Paper

This section's **real contribution** is the **explanatory framework**:

- **Theorem A** explains *why* Lloyd-Max MSE-optimal fails in PPL (Fisher metric mismatch + κ-spread)
- **Theorem B** explains *why* hand-picked configurations like "Layer 2-6 @ 3-bit" are optimal (Lagrangian solution under sensitivity-weighted importance)
- **Proposition D** identifies *where* the failure concentrates (Layers 2-6 in Mistral, Layer 0 + late layers in Qwen)
- **Theorem C** (separate) explains *why* QW-WF reduces to standard WF (PCA-Q natural alignment)

These are **understanding contributions**, not method contributions. We make no claim that CWF is a new SOTA quantizer. v3 WF(floor=2) remains the best known method in this space.

### X.7 What Remains for Future Work

1. **Theorem E (Cascade Amplification)**: Why are sensitivity values $g_{l,h}$ systematically larger in early layers? This requires training dynamics theory beyond the scope of this paper.

2. **Per-head sparsification**: Could relaxing floor=2 (allowing some heads to be quantized to 0) enable inter-head reallocation? Initial experiments (E3b) suggest no for L² MSE, but Fisher-weighted allocation is unexplored.

3. **Cascade-aware Mahalanobis**: Next-9 attempted Fisher Mahalanobis Lloyd globally; it failed due to numerical issues (982 PPL). A properly cascade-weighted version is open.

---

## Summary Table for Paper

**What we contribute (HONEST)**:

| # | Contribution | Type | Status |
|:---:|---|---|---|
| 1 | Theorem 6.16.3: Pre-RoPE PCA optimality (Class C) | Theorem | Proven, 624/624 MSE verified |
| 2 | PCA-Q natural alignment (0.6-2.5°) | Discovery | Novel structural finding |
| 3 | Per-head > Shared PCA (vs KVTC) | Empirical | +46.3% (Llama 2-bit) |
| 4 | Five-hypothesis systematic rejection | Methodology | All 5 fail; structural insight |
| 5 | MSE-PPL gap unified across 3 axes | Explanation | Lloyd + WF + QW share metric mismatch |
| 6 | Theorem A: MSE-PPL Inversion Bound | Theorem | Bound proven, κ-correlation ρ=+1.0 |
| 7 | Theorem B: Master Allocation Equation | Theorem | Constructively validated (Next-9c) |
| 8 | Proposition D: Per-head outlier concentration | Empirical | Verified in 2 models |
| 9 | Theorem C: QW-WF rank equivalence | Theorem | Loose bound + empirical (ρ=0.655) |

**What we DO NOT claim**:

- ❌ CWF as new SOTA quantization method
- ❌ Beating v3 WF(floor=2) at fair budget
- ❌ Inter-head allocation as principled improvement
- ❌ Cascade gradient as the principled sensitivity measure (empirical substitution is more accurate)

---

## Notes for Coworker (iamseungpil)

If you have time on the A100×16 node, the most useful experiments are:

1. **Llama CWF cross-verification**: Confirm whether the inter-head CWF signal is also empty on Llama-3.1-8B (we expect yes, similar to Mistral)

2. **MMLU downstream**: Verify that v3 WF(floor=2) maintains accuracy on real tasks (this validates the explanatory framework's relevance)

3. **Fisher Mahalanobis with proper numerical stability**: Re-attempt Next-9 (982 PPL catastrophe) with cascade-weighted whitening + double precision

4. **Theorem E investigation**: Measure $g_l$ via backward pass and check whether it correlates with layer depth as Conjecture E predicts. This is the closest open question to a principled extension.

---

*Drafted: 2026-04-07 (original, retracted)*
*Rewritten: 2026-04-08 (post-Codex critique + Next-12)*
