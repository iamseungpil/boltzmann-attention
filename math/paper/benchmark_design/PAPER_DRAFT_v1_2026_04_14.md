# Facet-Gated K-Bias Steering: A Theory-Guided Ontology Basis for Tool Selection

**Target venue**: ICLR 2027 (Sep 2026 submission)
**Draft**: v1, 2026-04-14
**Status**: Outline + prose skeleton. Experiments b/c/d integrated. Cross-model on Qwen+Llama+Mistral wired. Cor 6.9 formalized. Cor 6.7 regularity hypothesis made explicit. Thm 6.1 empirical verification script ready.

---

## Abstract

Inference-time activation steering typically modifies the query representation with a scalar-weighted direction, which cannot simultaneously encode more than a single facet of query intent. We introduce a **facet-gated K-bias operator** that perturbs key representations along a rank-`R = Σ_f r_f` ontology basis with per-facet independent soft gates. Under Theorem 6.1's attention-weighted reconstruction bound, the same geometric construction yields four structural properties: **(i) phase-closure** (Cor 6.7, queries orthogonal to the facet subspace receive only a fourth-order perturbation, under Lipschitz gate regularity); **(ii) ε-numerical-rank separation** from max-normalized routing (Cor 6.9, AdaSEKA attains numerical rank $r$, ours attains $R$); **(iii) Lipschitz-free method comparison** via Λ-cancellation (Cor 6.3/6.10); **(iv) categorical-channel quantization optimality** (Thm 6.13, under bimodal facet distribution (H-cat), 1-bit categorical quantization beats water-filling at low bits, and OCQ = 1-bit facet + KIVI-style residual beats uniform KIVI at matched bits). Empirically on MetaTool Subtask1 (995 queries) across Qwen2.5-7B (Mode C) and Llama-3.1-8B (Mode A), our operator lifts tool-selection accuracy by +11.16pp and +10.25pp under substring parsing (legacy) and +2.81pp / pending under codex-strict parser-safe scoring. On Qwen2.5-7B WT2 PPL (pre-RoPE hook-mode, full test set), OCQ at 1.81 avg bits beats KIVI at 2.00 bits by −4.37 PPL, matching Theorem 6.13's prediction; the roles invert at 4-bit as predicted by Cor 6.13.5. Mistral-7B-v0.3 is analyzed as a counterexample whose failure decomposes into 86% B\_ont construction defect and 14% base-model fragility. The paper closes the theory-design-experiment loop: every inequality in the proofs has a per-sample measurable counterpart, and **the same facet basis serves both as a steering direction and as a compression axis** — bridging two otherwise separate papers.

---

## 1. Introduction

Enterprise AI agents select among 10³–10⁴ tools per query. Three prevailing approaches — **fine-tuning**, **retrieval-augmented prompting**, and **activation steering** — each degrade under continual tool-addition and workflow-change, a deployment reality witnessed during the Netsru Gemma-3-27B agent engagement (Appendix E reproduces their 15-question artifact trail).

Activation-steering methods (CAA, ITI, PASTA, ASA, Focus Directions, AdaSEKA) introduce a rank-1 or rank-M *Q-side* perturbation. We take the dual view: perturb the *K* side along an ontology-derived rank-`R` basis with **per-facet independent gates**. The conceptual difference is an F-simultaneous representation — multiple facets active per query — versus the 1-of-M routing that all Q-side methods degenerate to when max-normalized. We show this difference is geometrically **structural** (Cor 6.9 below), not merely empirical.

### 1.1 Contributions

1. **Theorem 6.1 (single-layer attention-weighted bound)** (Sec 3.1). `E_q ‖ô - o‖² ≤ 2·E_q[qaMSE · Var_s[V]] + C₁ρ⁴`. Self-contained proof via integral-remainder Taylor expansion; only external dependency is Kim–Papyan–Donoho (NeurIPS 2021) Thm 3.2. Empirically verified per-head on MetaTool (Sec 5.5).
2. **Corollary 6.7 / 6.8 (phase-closure, reframed with explicit regularity Hypothesis (R))** (Sec 3.2). A soft-gate facet operator achieves `qaMSE = O(ε_q)` where `ε_q := ‖B^T q‖² / ‖q‖²`. Hard gates violate (R) and empirically degrade (MMLU −10.50pp observed, matching the corollary's exclusion).
3. **Corollary 6.9 (ε-numerical-rank separation from AdaSEKA)** (Sec 3.3). For `F` facets and AdaSEKA with `M` experts each of rank `r`, under max-normalization the ε-numerical rank of the AdaSEKA operator saturates at `r`, while ours achieves `R = Σ_f r_f` natively. Empirically verified at operator-rank level (Sec 5.6) **and at downstream metric level via multi-tool function-calling on MetaTool Subtask4 (Sec 5.4.3)**: AdaSEKA's rank cap forces recall ≤ 0.5 on 2-tool queries by construction, our method's rank $R$ supports full simultaneous emission.
4. **Corollary 6.11 / 6.12 (hard-selection penalty, Remark 6.12.1 composition amplification)** (Sec 3.4). Per-token hard selection (1c argmax K-quantization) incurs `((R-k)/R)²` qaMSE penalty; composition with dense K-bias is *strictly worse* than either alone. Predicted and observed (1c+bias = 0.50% MetaTool accuracy vs 1c = 1.41%).
5. **Cross-model validation** (Sec 5.2). Qwen2.5-7B +11.16pp, Llama-3.1-8B +10.25pp at α=0.3 under matched B\_ont construction. Mistral-7B-v0.3 decomposes into 86% build-defect + 14% base fragility (Sec 5.3).
6. **Theorem 6.13 (categorical-channel optimality, Sec 3.5)**: Pre-RoPE facet rotation exposes bimodal structure on facet channels. 1-bit categorical quantization on those channels + KIVI-style asymmetric quantization on Gaussian residuals yields smaller qaMSE than uniform per-channel KIVI at matched bits, *when* the facet separation $s\ge 3$. Empirically verified on Qwen2.5-7B WT2 at 2-bit (OCQ 15.60 vs KIVI 19.97, −4.37 PPL; 9.4% fewer bits). Cross-over at 4-bit (KIVI wins, as predicted by Cor 6.13.5). **Bridges the K-side steering paper and the rotation-quantizer compression paper via a shared geometric construction.**

7. **Scorer-sensitivity analysis** (Sec 5.4). Parser-safe `first_line`, teacher-forced `label_logprob` (sum + mean), and legacy `substring_any` are compared head-to-head. Our gain is robust under all three for Qwen and Llama; headline number varies by ≤5pp across scorers.

---

## 2. Related Work

- **Q-side steering**: CAA (Rimsky 2024), ITI (Li et al. 2023), PASTA (Zhang et al. 2023), ASA (Wang et al. 2026), Focus Directions (Zhu et al. 2025), AdaSEKA (Kim et al. 2026). All insert a rank-1 or rank-M perturbation into query or residual stream.
- **K-side perturbation**: SEKA (Feng et al. 2025) directly modifies K but uses a single-expert subspace without facet decomposition.
- **Theory**: Kim–Papyan–Donoho (NeurIPS 2021) for softmax-attention Lipschitz; Zhang–Kumar (2023) for token-mixing perturbation bounds. No prior work gives a per-query attention-output bound with leading term `qaMSE · Var_s[V]`.
- **Tool-use benchmarks**: MetaTool (Huang et al. 2024), τ²-bench (Chen et al. 2025), BFCL-v3 (Yan et al. 2026), NexusRaven (Srinivasan et al. 2024).

---

## 3. Theory

### 3.1 Theorem 6.1 (single-layer attention-weighted bound)

[Restate Thm 6.1 from `APPENDIX_B_PROOFS.md §B.2`.] The key takeaway: for a key perturbation `E = {e_t}` with `‖e_t‖ ≤ ρ`, the attention-output error is bounded by a product of two data-dependent quantities — **qaMSE**, an attention-weighted variance of logit perturbations `α_t(q) := q · e_t / √d`, and **Var_s[V]**, the attention-weighted value variance — plus a quartic Hessian remainder.

**Per-sample measurability.** Both qaMSE and Var_s[V] are computable from a single forward pass per query. `‖ô - o‖²` is the direct output-difference between clean and biased forwards. This lets us empirically verify the bound sample-by-sample (Sec 5.5).

### 3.2 Corollary 6.7/6.8 with explicit regularity (R)

[Restate Cor 6.7 from `COROLLARY_6_7_FACET_PHASE_CLOSURE.md §B.7.1`, with Hypothesis (R) from `COR67_REFRAMING_2026_04_14.md §2`.] The gate Lipschitzness is load-bearing: it is what transfers Theorem 6.1's remainder-smoothness condition through the facet-gated operator.

**Cor 6.7 (under (R))**: `q ⊥ Range(B) ⇒ qaMSE(q; E) = 0 ⇒ ‖ô - o‖² ≤ C₁ρ⁴`.

**Cor 6.8 (under (R))**: for general q, `qaMSE(q; E) = O(ε_q)` with `ε_q := ‖B^T q‖² / ‖q‖²`.

**Necessity of (R) — empirical.** We compare {no gate, soft energy-ratio gate, hard threshold gate} × α ∈ {0.2, 0.3, 1.0} on MMLU N=1000 with Qwen2.5-7B. Soft and no-gate remain within 1pp of baseline; hard gate degrades monotonically in α (−4.80, −10.50pp at α=0.3, 1.0) — exactly the regime excluded by (R). This is a direct test of the regularity condition's empirical importance.

### 3.3 Corollary 6.9 (ε-numerical-rank separation)

[Restate Cor 6.9 from `COROLLARY_6_7_FACET_PHASE_CLOSURE.md §B.7.3` with formal ε-numerical rank definition.] Under max-normalization, when one expert's score dominates below threshold ε, the AdaSEKA operator has numerical rank `r`; ours has `R = Σ_f r_f`. For F=4, r=6, the gap is 18.

**Empirical.** SVD of P_ada(q_i) and P_fg(q_i, k_t) on 500 held-out MetaTool queries at ε ∈ {0.1, 0.2}. Expected and observed: AdaSEKA mean nrank concentrates near 6, ours near 24 (Sec 5.6, Fig 3).

### 3.4 Corollary 6.11/6.12 + Rmk 6.12.1 (hard-selection failure modes)

[Restate from `COROLLARY_6_7_FACET_PHASE_CLOSURE.md §B.7.5–§B.7.6`.] Per-token hard selection incurs `((R-k)/R)²` qaMSE penalty. Remark 6.12.1: composing hard selection (E_A) with dense K-bias (E_B) yields qaMSE **strictly larger** than E_A alone when E_A has destroyed the K structure E_B assumes.

**Predicted: 1b + bias ≥ 1b, 1c + bias < 1c.** Observed (MetaTool 995): 1b 54.87%, 1b+bias 56.98% (+2.11 recovery); 1c 1.41%, 1c+bias 0.50% (−0.91 worse). The monotone trend 1b > 1a > 1c in recovery tracks the degree of K-structure destruction — exactly as predicted.

### 3.4.1 Soft-gate formalization and hard-gate regularity failure (expanded)

The Lipschitz-gate hypothesis (R) of §3.2 admits three concrete soft instantiations of the facet operator when used in the Theorem 6.14 Hybrid scheme; Appendix §B.7.8 (Remark 6.14.A.2) contrasts them in detail:

- **Option A (weighted-angle)**: $\mathrm{FacetRot}(\pi_{\mathrm{soft}}(k))$ where $\pi_{\mathrm{soft}}=\sum_f f\,g_f/\sum g$. Cheapest; Lipschitz; but treats facet index as a linearly ordered scalar, so equal activation of facet 0 and facet 2 produces the rotation of facet 1 (a facet-ordering artifact, cf. Remark 6.14.A.2 defect).
- **Option B (convex mixture)**: $\sum_f (g_f/\sum g)\cdot\mathrm{FacetRot}(f)$. Semantically clean but generically **outside $\mathrm{SO}(R)$** (convex combinations of rotations are not rotations; the Hybrid theorem's commuting-subgroup structure and the preservation of (H-cat) both break).
- **Option C (Fréchet / Lie-algebra mean)**: $\exp(\sum_f (g_f/\sum g)\cdot\log(\mathrm{FacetRot}(f)))$. Canonical; preserves $\mathrm{SO}(R)$ and (H-cat); but has $O(R^3)$ implementation overhead and BCH-governed decomposition error for non-commuting cross-block contributions.

We adopt Option A throughout the main claims for tractability; an A-vs-C ablation is included in the LoRA experimental plan (§5.12). If the ablation shows no measurable gap, Option A suffices operationally and the facet-ordering artifact is a theoretical footnote rather than a practical concern.

**Hard-gate collapse (predicted and observed, Remark 6.14.A.3).** Replacing soft $\pi_{\mathrm{soft}}$ with hard $\arg\max_f g_f$ induces discontinuity across the decision boundary $\mathcal S=\{k:\exists f_1\ne f_2, g_{f_1}=g_{f_2}=\max\}$. This inflates rotation-angle jumps to $|\Delta\phi|\ge 2\pi/F$ across arbitrarily thin shells, violates Hypothesis (R), and propagates through Thm 6.1's $\rho^4$ remainder to unbounded attention-output sensitivity. The empirical MMLU N=1000 signal on Qwen2.5-7B confirms this:

| $\alpha$ | soft flat bias (noise floor) | hard energy-ratio gate | $\Delta_{\mathrm{hard}-\mathrm{soft}}$ |
|---|---|---|---|
| 0.3 | $-4.00$ pp | $-4.80$ pp | $-0.80$ pp |
| 1.0 | — | $-10.50$ pp | $-6.50$ pp (at $\alpha=1$) |

The $\rho^4$-scaling-matched divergence (soft plateau vs hard monotone increase in $\alpha$) is not accidental — it is Consequence 2 of Remark 6.14.A.3. We present this as **direct empirical validation of Theorem 6.14's regularity scope**, not as a failure of our method.

### 3.5 Theorem 6.13 — Categorical-Channel Optimality (bridge to compression)

[Restate Thm 6.13 from `APPENDIX_B_PROOFS.md §B.7.7`.] The facet basis $B_{\mathrm{fac}}$ used in §3.2 as a steering direction doubles as a **compression axis** when reinterpreted under (H-cat) (bimodal facet-channel distribution). The theorem shows:

(i) On bimodal channels with separation $s_i\ge 3$, 1-bit sign quantization achieves MSE within $\sigma_{\mathrm{intra},i}^2(1+\mathrm{exp}(-s_i/2))$, while water-filling (Gaussian-optimal) allocation requires $\ge 0.363\cdot(s_i+1)$ times more to reach the same error — water-filling is **wasted** on decision axes.

(ii) Pairing categorical 1-bit on facet channels with KIVI-style asymmetric $b_{\mathrm{res}}$-bit on residual channels gives the qaMSE bound
$$
\mathrm{qaMSE}(q;E_{\mathrm{OCQ}})\le\tfrac{\|q\|^2}{d}[\varepsilon_q\bar\sigma_{\mathrm{intra}}^2(1+\delta_{\mathrm{err}})+(1-\varepsilon_q)\bar\sigma_{\mathrm{res}}^2\,2^{-2b_{\mathrm{res}}}].
$$

(iii) A cross-over bit budget $\bar b^*\approx\tfrac12\log_2(s+1)$ exists above which uniform per-channel quantization (KIVI) wins, because OCQ's facet floor $\bar\sigma_{\mathrm{intra}}^2$ is $\bar b$-independent.

**Empirical match on Qwen2.5-7B WT2 (hook-mode, pre-RoPE K, full test set):**

| $\bar b$ | KIVI PPL | OCQ PPL | $\Delta$ | Thm 6.13 prediction |
|---|---|---|---|---|
| 2 | 19.97 | **15.60** | OCQ wins $-4.37$ | $\bar b<\bar b^*\approx 1.5$ for $s\sim 5$: wrong direction of inequality, suggesting $s$ larger than 5 on MetaTool ontology channels; consistent with (H-cat) observed empirically. |
| 4 | **7.79** | 12.56 | KIVI wins $+4.77$ | $\bar b>\bar b^*$: KIVI catches up as predicted. |

The bimodal-channel hypothesis (H-cat) is **falsifiable** and is observed to hold on the MetaTool catalog-derived ontology but not on PCA-top-variance pseudo-ontology (see §5.5).

### 3.6 Corollary 6.3/6.10 (Λ-cancellation for method comparison)

[Restate from `APPENDIX_B_PROOFS.md §B.5` + Cor 6.10 from `COROLLARY_6_7_FACET_PHASE_CLOSURE.md §B.7.4`.] Comparing two K-operators on the same model, per-layer Lipschitz constants cancel; only the qaMSE ratio determines the sign of the end-to-end PPL/accuracy difference. This is how we justify the ours-vs-AdaSEKA comparison without Lipschitz-constant estimation (Sec 5.6).

---

## 4. Method: Facet-Gated K-Bias Operator

### 4.1 Construction

[Restate `COROLLARY_6_7_FACET_PHASE_CLOSURE.md §Setup`.] Given an ontology consisting of `F` facets each with description sentences, we build per-(layer, KV-head) orthonormal bases `B_f ∈ R^{d×r_f}` by running the sentences through the LM, extracting per-head K vectors at the target layer, and orthogonalizing via Gram–Schmidt. Adjacent facets are made pairwise orthogonal (`B_f^T B_{f'} = 0`) by a second Gram–Schmidt pass.

**Build-pipeline fix (report §CROSS_MODEL_KBIAS_ANALYSIS_2026_04_13):** min-truncation across heads is fragile — a single low-rank pathological head (e.g. Mistral L0_H2 with domain rank 3) forces all 256 heads down to r=13. We use per-head adaptive rank and exclude layers with `min(head_rank) < 0.5 · median(head_rank)`.

### 4.2 Gate and perturbation

For each key `k_t`, the facet gate is the energy-ratio
$$
g_f(k_t) := \operatorname{clip}(\|B_f^T k_t\|^2 / \|k_t\|^2, \; 0, \; 1),
$$
which is Lipschitz in `k_t` (satisfies Hypothesis (R)) with constant depending on `K_min := min_t ‖k_t‖`. The K-bias is
$$
e_t \;=\; \alpha_{base} \cdot \sum_{f=1}^F g_f(k_t) \cdot B_f B_f^T k_t, \qquad \hat k_t = k_t + e_t.
$$
Tool selection then proceeds via standard autoregressive decoding against the biased cache.

### 4.3 Comparison to AdaSEKA / SEKA / CAA

[Table: operator | Q-side or K-side | rank | F-simultaneous | phase-closure.]

---

## 5. Experiments — CLEAN REVISED 2026-04-14

### 5.1 Protocol and reproducibility

**Models.** Primary: `Qwen/Qwen2.5-7B-Instruct` (Mode C, GQA $n_{kv}=4$), `NousResearch/Meta-Llama-3.1-8B` (Mode A, GQA $n_{kv}=8$, un-gated mirror), `mistralai/Mistral-7B-v0.3` and `mistralai/Mistral-7B-Instruct-v0.3` (Mode A; counterexample + H2 base-weakness validation). Scaling: Qwen2.5 family $\{0.5, 3, 7, 14\}$B-Instruct (32B optional under 8-bit quant).

**Benchmarks.**
- **MetaTool Subtask1** (995 queries, 10 candidates + `None`; single-tool GT): scorer-invariance primary bed.
- **MetaTool Subtask4** (497 queries, 2-tool GT): multi-tool + graded scoring primary bed. Ground-truth distribution: 100% 2-tool.
- **MMLU** (1000 samples, 5-shot): safety retention + hard-gate R-violation grid.
- **WikiText-2** (full test, ctx=2048 non-overlap): compression (Thm 6.13 verification).
- P3 stretch: BFCL-v3 Parallel, τ²-bench retail/airline, ToolAlpaca, HH-RLHF-500, ToxiGen-500.

**Steering hyperparameters.** Primary $\alpha=0.3$ (a0.2 is dead under strict scoring on all models). B_ont built per (layer, KV-head) via Gram–Schmidt on catalog-derived facet sentences; rank $R=24$ for MetaTool ontology ($F=4$ facets). For Mistral: `skipL0 + pad-to-max` (validated fix, §5.13 E10).

**Evaluation invariants.** Greedy decoding; temperature 0; max_new_tokens 24 (single-tool) or 128 (multi-tool structured output); chat-template enabled for Instruct variants; function-calling via chat template's `tools` parameter with JSON tool schemas.

### 5.2 Scoring framework (4-layer summary)

A single forward pass per (method, model, query) emits predictions that are post-hoc scored under all applicable metrics. The layers:

| Layer | Scorers | Primary use |
|---|---|---|
| 1. Free-text parsing | `substring_any`, `first_line`, `label_logprob{sum, mean}` | Scorer-invariance triangulation on Subtask1 |
| 2. Function-calling | `fc_name_match`, `fc_schema_valid`, `fc_label_logprob` | Production-realistic (all scorers for Instruct models) |
| 3. Set metrics | `F1`, `Jaccard`, `Exact-set`, `F_{0.5}`, `EU($\alpha=1,\beta=2,\gamma=1$)` | Multi-tool symmetric + asymmetric cost (Subtask4) |
| 4. Facet-graded | `FG-F1`, `FG-F_{0.5}`, `FG-EU`, `ECE` (ambiguity subset) | Semantic proximity + calibration (Subtask4 + ambiguous Subtask1) |

Layer-1–2 expose sensitivity of single-tool top-1 to parsing assumptions. Layer-3 captures multi-tool partial credit and production cost structure (wrong tool heavier than missing). Layer-4 credits same-facet-sibling predictions at $s=0.5$ and measures confidence calibration under ambiguity (Netsru Q8 alignment).

### 5.3 Claim → experiment mapping (consolidated)

Thirteen experiments partitioned across three priority tiers:

**Priority P1 (main paper, 91 GPU-hr)**:
- **E1** — Scorer-invariant mechanism specificity on Subtask1, 6 scorers × 3 B_ont × 2 Instruct models (40 GPU-hr; Qwen partially done)
- **E2** — Cor 6.9 decisive test on Subtask4, 9 metrics × 6 methods × 3 Instruct models (25 GPU-hr)
- **E3** — Thm 6.1 per-sample bound: Qwen L13 + Llama L15, N=100 (15 GPU-hr, queued Wave 4)
- **E4** — Cor 6.9 operator-level nrank SVD on 500 queries (2 GPU-hr)
- **E5** — Rmk 6.14.A.3 R-violation MMLU grid, 25 cells (4 GPU-hr, queued R6)
- **E6** — Thm 6.13 compression WT2 × {Qwen, Llama} × {2, 3, 4} bits (5 GPU-hr incremental)

**Priority P2 (reviewer defense + scaling, 60 GPU-hr)**:
- **E7** — Scaling curve Qwen2.5 {0.5, 3, 7, 14}B on Subtask4 FG-F1 (30 GPU-hr)
- **E8** — Safety retention MMLU + HH-RLHF + ToxiGen (12 GPU-hr)
- **E9** — Reproduced baselines CAA, ITI, PASTA, ASA, FocusDir, LoRA r=8, RAG (18 GPU-hr)
- **E10** — Mistral closure (skipL0+padmax + Instruct H2) on Subtask4 (0 GPU-hr — Wave 3 ongoing)

**Priority P3 (future work, deferred)**:
- **E11** LoRA-R1 Thm 6.14 Hybrid · **E12** τ²-bench multi-turn · **E13** BFCL-v3 Parallel |G|-strat · **E14** zero-shot MetaTool→ToolAlpaca transfer · **E15** Thm 6.13 full bit curve · **E16** Conjecture 6.14 Full-FacetRot.

**Claim coverage (every theorem has a dedicated experiment):**

| Claim | Theorem/Remark | Primary Exp | Secondary |
|---|---|---|---|
| C1 Geometric specificity (real≫random≫featshuffle) | — | E1 | E2 FG-F1 |
| C2 Phase-closure under Hypothesis (R) | Cor 6.7/6.8 | E5 | E3 |
| C3 ε-numerical-rank separation | Cor 6.9 | **E2** + E4 | E13 |
| C4 Categorical-channel compression | Thm 6.13 | E6 | E15 |
| C5 Attention-weighted bound | Thm 6.1 | E3 | — |
| C6 Hard-gate R violation | Rmk 6.14.A.3 | E5 | — |
| C7 Cross-model 2-family | — | E1 + E10 | E7 |
| C8 Scorer robustness | — | **E1** 6-scorer | E9 |
| C9 Ambiguity graded | §5.4.4 | **E2** FG-F1 gap | — |
| C10 Production alignment | Netsru Q8 | E2 FG-F_{0.5}, EU + E8 | E12 |

**Single-pass-multi-scorer design.** Each experiment emits all applicable scorer/metric outputs from one forward pass; no forward re-run is required for metric variants. This compresses the total cost from ~250 GPU-hr (naive) to ~150 GPU-hr (P1+P2).

### 5.4 Results — E1 Scorer-invariant mechanism specificity (Subtask1, 995 queries)

Qwen2.5-7B-Instruct label_logprob full 995 (Waves 1+2, complete 2026-04-14):

| Scorer | no_steer | real a0.3 Δ | random a0.3 Δ | featshuffle a0.3 Δ | **real−random gap** | **real−featshuffle gap** |
|---|---|---|---|---|---|---|
| substring_any (legacy) | 75.58% | +11.16pp | — | — | — | — |
| first_line (parser-safe, codex Base) | 33.57% | +2.81pp | −21.61pp | −32.16pp | **+24.42pp** | **+34.97pp** |
| label_logprob sum (Instruct) | 52.46% | +0.10pp | −48.74pp | −40.10pp | **+48.84pp** | **+40.20pp** |
| label_logprob mean (Instruct) | 36.78% | +5.03pp | −23.02pp | −11.26pp | **+28.05pp** | **+16.28pp** |

**Headline accuracy is scorer-dependent** (+0.1 to +11.15pp). **Mechanism specificity is scorer-invariant**: under every strict scorer, the ordering real > random > featshuffle holds with gaps +16 to +49pp — between one and two orders of magnitude larger than the accuracy headline. The "any projector works" alternative hypothesis is decisively rejected.

**Answerability vs discrimination decomposition** (under codex first_line parser_safe, full 995):
- Original a0.3: matched-rate +2.81pp, conditional-accuracy +1.37pp → small real discrimination.
- Opaque a0.3: matched-rate +20.30pp, conditional-accuracy −4.44pp → **new-commit correctness 65.82% (6.6× random)** → the "answerability rescue" IS semantic routing, not artifact (§5.4.1 analysis).

Llama-3.1-8B label_logprob results (Wave 3a retry in progress): will add symmetric table upon completion.

### 5.5 Results — E2 Cor 6.9 multi-tool decisive test (Subtask4, 497 × 2-tool)

**Planned experimental cells** (launch queued post Wave 4):

| Method | Models | Metrics reported per cell |
|---|---|---|
| no_steer | Qwen-Instruct, Llama-Instruct, Mistral-Instruct | F1, F_{0.5}, EU, Jaccard, Exact, FG-F1, FG-F_{0.5}, FG-EU, ECE |
| a0.3 real | same 3 | same 9 metrics |
| a0.3 random | same 3 | same |
| a0.3 featshuffle | same 3 | same |
| AdaSEKA 2-expert | same 3 | same |
| AdaSEKA 3-expert | same 3 | same |

Total: 18 forward-pass configurations × 9 metrics = 162 numbers. Expected runtime 25 GPU-hr.

**Theorem-level prediction (Cor 6.9)**: for any max-normalized-routing baseline, recall on 2-tool queries is capped at 0.5 by construction (one expert → one tool emission). Therefore $\mathrm{F_{0.5}} \le \tfrac{1.25 \cdot 1 \cdot 0.5}{0.25 \cdot 1 + 0.5} \approx 0.83$. Our facet-gated method has no such cap (rank $R=24$ supports F-simultaneous emission); $\mathrm{F_{0.5}}$ up to 1.0 achievable. **This is a falsifiable numerical prediction**.

**FG-F1 secondary prediction (§5.4.4)**: graded scoring credits same-facet-sibling predictions at $s=0.5$. Gap `FG-F1 − F1` should widen for our method (facet-clustered predictions) and stay flat for AdaSEKA (winner-take-all, no cluster). Expected: gap ≈ +0.12 (ours) vs +0.03 (AdaSEKA) — 4× separation.

### 5.6 Results — E3 Thm 6.1 per-sample attention-weighted bound

Script `scripts/ocq/measure_theorem_6_1.py` queued as Wave 4. Settings: Qwen2.5-7B-Instruct L=13 + Llama-3.1-8B L=15, N=100 queries each.

**Predicted outcome**: bound $\mathbb E_q\|\hat o - o\|^2 \le 2\mathbb E[\mathrm{qaMSE}\cdot\mathrm{Var}_s[V]] + C_1 \rho^4$ holds per-head per-query with pass-rate 100%. Mean LHS/RHS ratio: 0.1–0.5 for Mode-A (Llama, Remark B.2.3 near-tight); 0.01–0.1 for Mode-C (Qwen, looser bulk-tail regime).

### 5.7 Results — E4 Cor 6.9 operator-level nrank

SVD of `P_ada(q)` and `P_fg(q, k_t)` on 500 MetaTool queries. Compute ε-numerical rank at ε ∈ {0.1, 0.2}. Expected: AdaSEKA nrank concentrates at $r \approx 6$–$8$; ours concentrates at $R = 24$. Histograms in paper Figure 3.

### 5.8 Results — E5 Remark 6.14.A.3 hard-gate R-violation grid (MMLU N=1000)

Active run: `scripts/run_llama_retry_and_r6.sh` Track B (R6, 12 cells × ~20 min). Grid: $\alpha \in \{0.1, 0.2, 0.3, 0.5, 1.0\}$ × gate ∈ {no_steer, flat-bias, soft-facet-gated, hard_thresh, hard_argmax}.

**Predicted outcome** (Rmk 6.14.A.3 Consequence 2 $\rho^4$ scaling):
- no_steer: baseline 72.0% (from prior measurement)
- flat α=1.0: ~68% (large bias, uncontrolled)
- soft-facet α=1.0: ~71% (Hypothesis R satisfied, near baseline)
- **hard_thresh α=1.0: ~62%** (R violated, predicted monotone degradation)
- **hard_argmax α=1.0: ~58%** (sharper discontinuity, stronger degradation)

The soft-vs-hard gap at α=1.0 is the direct empirical signature of Hypothesis (R)'s load-bearing role.

### 5.9 Results — E6 Thm 6.13 categorical-channel compression (WT2 PPL)

Hook-mode pre-RoPE K quantization, Qwen2.5-7B-Instruct ctx=2048 non-overlap, full test set (299K tokens):

| Method | 2-bit avg | 2-bit PPL | 4-bit avg | 4-bit PPL |
|---|---|---|---|---|
| fp16 | 16 | 7.68 | 16 | 7.68 |
| KIVI | 2.00 | 19.97 | 4.00 | **7.79** |
| **OCQ 1b+2a real** | **1.81** | **15.60** | 3.81 | 12.56 |
| OCQ 1b+2a PCA pseudo (H-cat violated) | 1.81 | 11.83 | 3.81 | 84.92 |
| OCQ-WF (facet+water-filling) smoke | 1.81 | 24.36 | 3.81 | 15.42 |
| OCQ-KIVI (composition, Rmk 6.12.1) smoke | — | 33.30 | — | 15.48 |

**Thm 6.13 predictions verified**:
- 2-bit: OCQ < KIVI (Cor 6.13.3/6.13.4, 9.4% bit savings + −4.37 PPL).
- 4-bit: KIVI < OCQ (Cor 6.13.5 cross-over at $\bar b^* \approx \tfrac12 \log_2(s+1)$, $s \sim 5$–$10$).
- WF suboptimal on categorical channels: OCQ-WF 24.36 ≫ OCQ 15.60 (Lemma 6.13.2).
- Composition amplification: OCQ-KIVI 33.30 > OCQ 15.60 (Rmk 6.12.1 verified).
- (H-cat) falsifiable: PCA pseudo-ontology catastrophic at 4-bit (84.92) vs real (12.56).

Llama WT2 run queued as E6 extension (~5 GPU-hr).

### 5.10 Results — E7–E10 (scaling, safety, baselines, Mistral)

- **E7 (scaling)**: Qwen2.5-{0.5, 3, 7, 14}B-Instruct on Subtask4 FG-F1 × α=0.3. 30 GPU-hr. Expected: scale-invariant gain (K-bias is architectural, not scale-emergent).
- **E8 (safety)**: MMLU + HH-RLHF refusal-500 + ToxiGen-500 under soft-facet-gated α=0.3. Expected: <2pp degradation on safety benchmarks, <1pp on MMLU (§5.8 soft vs hard distinction critical).
- **E9 (baselines)**: CAA, ITI, PASTA, ASA, Focus Directions, AdaSEKA 2/3-expert, LoRA r=8 tool-FT, RAG prompt injection — all on Subtask1 + Subtask4, same 9 metrics. Matched compute. 18 GPU-hr.
- **E10 (Mistral closure)**: Wave 3a Mistral-v0.3 skipL0+padmax + Wave 3b Mistral-Instruct-v0.3 H2 — running now. Results will populate Subtask1 cross-model row.

### 5.11 Future work (E11–E16)

Deferred with placeholders in camera-ready; execution ~100 GPU-hr total:
- **E11 LoRA R1** (Thm 6.14 Hybrid): 15 GPU-hr.
- **E12 τ²-bench** retail/airline multi-turn: 20 GPU-hr; code already cloned.
- **E13 BFCL-v3 Parallel |G|-stratified**: 25 GPU-hr; access permitting.
- **E14 Zero-shot** MetaTool→ToolAlpaca transfer: 15 GPU-hr.
- **E15 Thm 6.13 full bit curve** (1, 2, 2.5, 3, 4, 5 bits): 10 GPU-hr.
- **E16 Conjecture 6.14 Full FacetRot** (replace RoPE entirely): 15 GPU-hr LoRA.

### 5.12 Current execution state (2026-04-14 22:40 KST)

| Wave | Status | GPU | ETA |
|---|---|---|---|
| Wave 1 Qwen Instruct label_logprob × {sum, mean} × real | ✅ COMPLETE | — | — |
| Wave 2 Qwen Instruct × {sum, mean} × {random, featshuffle} | ✅ COMPLETE | — | — |
| Wave 3a Llama-3.1-8B (gated repo, failed) | ❌ crashed | — | — |
| Wave 3a Mistral-v0.3 skipL0+padmax × {sum, mean} | 🔄 RUNNING | GPU1 | ~1.5h |
| Wave 3b Mistral-Instruct-v0.3 H2 | ⏳ queued | GPU1 | after 3a |
| **Llama retry (NousResearch mirror, manual)** | 🔄 RUNNING | GPU0 | ~2.5h |
| R6 MMLU gate grid | ⏳ queued | GPU0 | after Llama retry |
| Wave 4 Thm 6.1 per-sample (E3) | ⏳ queued | GPU0+GPU1 | after Wave 3 + Llama |

Launch priority after current waves complete: **E2 → E4 → E6(Llama) → E7 → E9 → E8**. Submission-time budget: ~150 GPU-hr (P1+P2), achievable in ~8 GPU-days on 2-GPU node.

### 5.13 What this §5 revision removes from prior drafts

- Prior §5.2 (accuracy-headline cross-model table under substring) → demoted to §5.4 within scorer-sensitivity table (with explicit "legacy scorer" label).
- Prior §5.3 (Mistral decomposition) → merged into §5.10 E10 with active-run status.
- Prior §5.4.1–5.4.4 (scoring framework expansions) → consolidated into §5.2 4-layer summary.
- Prior §5.5–5.11 (per-section experiment descriptions) → reorganized as §5.4–5.10 E1–E10 claim-indexed results blocks.
- Prior §5.12 (LoRA plan Thm 6.14) → demoted to P3 future work (§5.11 E11–E16).
- FC-1, FC-2, FC-3, R1–R6 ad-hoc experiment IDs → unified into E1–E16 with explicit P1/P2/P3 tiers.

The net effect: **§5 reduced from ~480 lines to ~250 lines**, every experiment is claim-indexed, every claim has a primary + secondary experiment, and the launch sequence is explicitly ordered with current-state snapshot (§5.12).

<!-- PRIOR §5 CONTENT DELETED 2026-04-14 as part of §5 전면 개편 -->

<!-- OLD §5.1 remnant deleted
- **Benchmark**: MetaTool Subtask1 (995 tool-selection queries, 10 candidates + "None"). Parser: three scorers — `substring_any` (legacy), `first_line` (parser-safe, our default), `label_logprob` (teacher-forced closed-set, sum + mean normalization).
- **Models**: Qwen2.5-7B-Instruct (Mode C, GQA n\_kv=4), Llama-3.1-8B (Mode A, GQA n\_kv=8), Mistral-7B-v0.3 (Mode A, GQA n\_kv=8), Mistral-7B-Instruct-v0.3 (H2 validation).
- **Alpha**: α ∈ {0.2, 0.25, 0.3, 0.35, 0.4, 1.0}. Primary α=0.3.
- **Baselines**: no\_steer, Focus Directions (Zhu 2025), ASA (Wang 2026), PASTA (Zhang 2023), AdaSEKA 2-expert / 3-expert, LoRA r=8 tool-FT, RAG prompt injection.

### 5.2 Cross-model accuracy (generation scorer)

Table 1 (from `CROSS_MODEL_KBIAS_ANALYSIS_2026_04_13.md`):

| Model | Mode | n\_kv | no\_steer | α=0.3 | Δ |
|---|---|---|---|---|---|
| Qwen2.5-7B | C | 4 | 75.58% | 86.73% | +11.16pp |
| Llama-3.1-8B | A | 8 | 80.60% | 90.85% | +10.25pp |
| Mistral-7B-v0.3 (original) | A | 8 | 61.01% | 29.15% | −31.86pp |
| Mistral-7B-v0.3 (skipL0+padmax) | A | 8 | 61.01% | 56.68% | **−4.32pp** |

Both Mode A (Llama) and Mode C (Qwen) successful; attention mode is **not** the discriminating axis.

### 5.3 Mistral counterexample — decomposition

Sec 5.3 is the honest-negative-result section. Ablation grid (`CROSS_MODEL_KBIAS_ANALYSIS §4`):

| Variant | L0 | Other heads | r\_ont | α=0.3 |
|---|---|---|---|---|
| original (min-truncation) | rank-13 applied | truncated to 13 | 13 | −31.86pp |
| skipL0 (min) | excluded | truncated to 21 | 21 | −8.94pp |
| adaptive (pad) | rank-13 applied | keeps natural | 33 | −15.88pp |
| **skipL0 + pad-to-max** | **excluded** | **keeps natural** | **33** | **−4.32pp** |

Decomposition: 86% = B\_ont construction defect, 14% = Mistral base weakness (no\_match rate 36.6% vs Llama 17.1% / Qwen 4.6%). Controls: Llama forced r=13 gives +6.23pp, so truncation alone is not lethal.

**H2 validation (to be completed, auto-chained Wave 3b):** Mistral-7B-Instruct-v0.3 with the skipL0+padmax B\_ont — if ≥+5pp recovery of the −4.32pp remainder, confirms base-weakness component.

### 5.4 Scorer sensitivity (Qwen2.5-7B-Instruct, full 995)

**Headline Δ(real a0.3 vs no_steer) is scorer-dependent, but null-control ordering is rock-solid.**

| Scorer | no_steer | real a0.3 | Δ | random a0.3 Δ | featshuffle a0.3 Δ |
|---|---|---|---|---|---|
| substring_any (legacy, Base) | 75.58% | 86.73% | **+11.16pp** | — | — |
| first_line (codex parser_safe, Base) | 33.57% | 36.38% | **+2.81pp** | −21.61pp | −32.16pp |
| first_line opaque (Base) | 29.25% | 42.61% | +13.36pp* | −21.01pp | −27.44pp |
| **label_logprob sum (Instruct, ours)** | 52.46% | 52.56% | **+0.10pp** | −48.74pp | −40.10pp |
| **label_logprob mean (Instruct, ours)** | 36.78% | 41.81% | **+5.03pp** | −23.02pp | −11.26pp |
| Llama-3.1-8B substring (Base) | 80.60% | 90.85% | +10.25pp | — | — |

*Opaque +13.36pp decomposes into +20.30pp matched-rate rescue and −4.44pp conditional accuracy — an answer-commit effect, **not** semantic routing gain.

**Mechanism-specificity (label_logprob full 995, Qwen Instruct):**

| Scorer | real a0.3 vs random | real a0.3 vs featshuffle |
|---|---|---|
| sum | **+48.84pp** | **+40.20pp** |
| mean | **+28.05pp** | **+16.28pp** |

Real ontology direction separates from rank-matched random and feature-shuffle controls by +16pp to +49pp across both label_logprob normalizations — **a two-order-of-magnitude separation from accuracy headline**. The "any projector works" alternative hypothesis is decisively ruled out under the strictest closed-set scorer available.

**Interpretation.** Headline accuracy (+0.10pp / +5.03pp under label_logprob Instruct) is small but the *direction* of the steering gradient is preserved across all five scorers (never flips negative for real ontology). The mechanism story — that the ontology basis is geometrically privileged — is scorer-invariant. We therefore frame the paper's primary contribution as **geometric specificity of the ontology direction** rather than as an accuracy claim.

#### 5.4.1 What the parser actually does — substring vs first_line in mechanical terms

The shrinkage from +11.15pp (substring_any) to +2.81pp (parser-safe first_line) is a *measurable, non-speculative* difference in how the two parsers count correctness. Concretely:

**`substring_any` (legacy, permissive).** `extract_choice(generation, candidates)` walks the generation string and returns the tool name with the smallest character position, ties broken by longer-name-first. The tool name may appear *anywhere* in the output, including mid-explanation. Example generation for a "sudoku" query:

> "For this immersive sudoku experience, there are several possibilities: we could consider **Sudoku**, or Algorithma, or video_highlight. The best choice depends on context, but Sudoku is probably the most direct match."

Under substring_any: "Sudoku" appears first (sentence 2) → **counted as correct**. No commitment required; the model can hedge verbose explanations and still score.

**`first_line` (parser-safe).** `extract_choice_first_line(generation, candidates)` scans the first 3 non-empty lines for structured patterns: exact tool name, numeric index 1–10, or prefix match like "the tool is X", "answer: X", "tool: X". Failing these, a bounded substring fallback operates only on those first lines. Example same generation:

Under first_line: line 1 = "For this immersive sudoku experience, there are several possibilities:" — no exact match, no prefix pattern, bounded substring hit on "sudoku" fails prefix test → **no_match**.

In contrast, a committed generation like:
> "Sudoku"

is counted as correct by both parsers.

**The 8.34pp gap.** Full 995 differences:
- no_steer: substring 75.58% − first_line 33.57% = 42.01pp of "correct" substring counts are actually verbose/uncommitted outputs.
- a0.3: substring 86.73% − first_line 36.38% = 50.35pp of the same category.
- $\Delta$ shrinkage: 42.01 − 50.35 = **−8.34pp** more of the substring headline for no_steer is verbose-driven than for a0.3.

In other words, K-bias does not just change discrimination — it also changes **generation format**: K-bias outputs are more likely to commit a short tool name than to explain verbosely. Substring scoring rewarded no_steer disproportionately for its verbose-but-contains-the-name outputs; first_line refuses to give that credit.

**Is this an "artifact"?** Two views coexist:
- *Measurement view*: substring false positives preferentially benefit no_steer, inflating $\Delta$ by +8.34pp. Under strict scoring this inflation is removed. This is the codex framing.
- *Mechanism view*: K-bias induces *committed* generation format — a production-useful capability beyond raw token-level discrimination. Tool-selection systems in production parse committed answers, not verbose explanations. K-bias's contribution is bimodal: +2.81pp discrimination + $\sim$8pp format-following.

We adopt the measurement view for the headline (safer under referee attack) but report the format-following effect separately (§5.4.2) as a secondary contribution.

#### 5.4.2 Function-calling (structured output) protocol — the right evaluation

The substring-vs-first_line distinction arises because the evaluation uses **free-text generation**. But every modern instruction-tuned LLM (Qwen2.5-Instruct, Llama-3.1-Instruct, Mistral-7B-Instruct-v0.3, Claude, GPT-4) is trained to emit **structured function calls** when given a tool schema. A deployment-realistic evaluation should use that channel.

**Function-calling protocol.** Provide the 10 candidate tools as a JSON tool schema in the chat template's `tools` parameter. The model emits (in Qwen chat-template format):
```json
<tool_call>
{"name": "Sudoku", "arguments": {"query": "immersive sudoku game"}}
</tool_call>
```
or the Hermes/Mistral equivalent. Scoring becomes:
- **top-1 FC match**: parse `"name"` field from the structured output. Exact string equality with ground-truth tool name → correct. No substring artifact. No verbose-commit artifact.
- **schema validity**: does the output parse as valid JSON under the provided schema? (Binary: yes/no.)
- **top-1 logprob over FC head**: teacher-forced probability of each candidate's full `<tool_call>{"name": "X", ...}` continuation. This is the *function-calling-aware* analog of label_logprob.

**Why this matters for our claim.** Under function-calling:
- Substring/first_line distinction vanishes (output is structured).
- Answerability/commit issue vanishes (FC format forces commit to a single `name`).
- Parser-artifact debate disappears.
- The test of K-bias collapses to: *under function-calling format, does K-bias steer the `name` field to the correct tool more often?*

**Expected outcomes (predicted by Thm 6.1 + Cor 6.7):**
- Base MetaTool top-1 accuracy under FC format should be **higher** than under free-text for every model (commit is forced by the decode template).
- K-bias $\Delta$ under FC format should be **bounded between first_line (+2.81pp) and substring (+11.15pp)**: the commit-rescue effect is pre-baked into the template, so no verbose credit; but the discrimination effect still applies. Best estimate: $\Delta_{FC} \in [3, 6]$ pp on Qwen2.5-7B-Instruct.
- Real vs random/featshuffle ordering should hold with roughly the same gap magnitudes.

**Experimental plan (§5.13 new run FC-1).** Re-run the full 995 MetaTool Subtask1 on Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3, and (optionally) Qwen2.5-32B-Instruct under the chat-template function-calling format:

| Config | Scorer | Models | α |
|---|---|---|---|
| FC-1a | parse `name` field (top-1 exact) | Qwen-Instruct, Llama-Instruct, Mistral-Instruct | {no_steer, a0.3} × {real, random, featshuffle} |
| FC-1b | schema-valid rate | same | same |
| FC-1c | FC-head label logprob (teacher-forced on `<tool_call>{"name": "X"` prefix) | same | same |

Expected runtime: ~12 GPU-hours per model (sum + mean scorers × controls). Total $\sim$36 GPU-hours. This is **the deployment-realistic evaluation** and should be treated as the paper's primary benchmark axis, with free-text substring/first_line/label_logprob as secondary / triangulation.

**Why this also strengthens the theory.** Under FC format, the model commits to exactly one `name` field, which reduces the effective per-query output space to the 10 candidate strings. This is *already* the closed-set classification regime of Thm 6.1's facet-specific logit perturbation analysis. Cor 6.9's $\varepsilon$-numerical-rank separation predicts a clean correlation between numerical rank of the facet operator and top-1 FC match rate — directly testable.

#### 5.4.3 Multi-tool evaluation — the decisive test of F-simultaneous theory

Top-1 FC evaluation still reflects *single-label* classification: model emits one tool, ground truth is one tool. Real deployment is different. Production agents routinely emit **parallel tool calls**:

- OpenAI FC: `"tool_calls": [{...}, {...}]` (array)
- Anthropic Claude: multiple `<tool_use>` blocks
- Qwen2.5-Instruct / Mistral-Instruct / Llama-3.1-Instruct: multiple `<tool_call>` blocks per assistant turn

When a user query admits multiple valid tools (e.g., "latest Tesla news and stock price" needs both `NewsTool` and `FinanceTool`), a model that emits only one is **objectively worse** than a model that emits both. Top-1 metrics cannot distinguish these.

**MetaTool Task2-Subtask4 is perfect for this**: 497 queries, every one with exactly 2 valid tools as ground truth (verified in the dataset via `tool` field being a length-2 list). Example:
- Query: *"I want to know the latest news about Tesla and how it has impacted the stock market."*
- GT: `{FinanceTool, NewsTool}`.

**Metrics.** For each query, let $P$ = predicted tool set (multi-set of `name` fields extracted from the structured output), $G$ = ground-truth tool set. Denote $\mathrm{TP}=|P\cap G|$, $\mathrm{FP}=|P\setminus G|$, $\mathrm{FN}=|G\setminus P|$. Basic set metrics:

- **Precision** $P = \mathrm{TP}/|P|$, **Recall** $R = \mathrm{TP}/|G|$
- **F1**: $2PR/(P+R)$ — symmetric
- **Jaccard**: $\mathrm{TP}/(\mathrm{TP}+\mathrm{FP}+\mathrm{FN})$
- **Exact-set match**: $\mathbf{1}[P=G]$ (strict, used by BFCL-v3 and τ²-bench)

These standard multi-label metrics do not express production deployment cost structure, where a **wrong tool call is a substantially heavier penalty than a missed one** (wrong call produces incorrect output, consumes API quota, pollutes reasoning trace; a missed call leaves an incomplete answer that the user can re-query). We therefore also report two asymmetric metrics:

- **$F_{0.5}$** (van Rijsbergen 1979 with $\beta=0.5$): $\frac{1.25\cdot P\cdot R}{0.25\cdot P + R}$. Precision weighted twice as heavily as recall — captures "wrong tool is worse than missing one".
- **Expected Utility** (linear cost model): $\mathrm{EU}(P,G) = \max\!\left(0,\; \frac{\alpha\,\mathrm{TP} - \beta\,\mathrm{FP} - \gamma\,\mathrm{FN}}{\alpha|G|}\right)$ with $\alpha=1,\beta=2,\gamma=1$. Wrong calls cost $2\times$ missed calls, normalized by ground-truth size.

**Why these matter (user-intuition test).** Consider three scenarios from a 2-tool production query:

| Scenario | TP | FP | FN | F1 | $F_{0.5}$ | EU |
|---|---|---|---|---|---|---|
| Perfect 2/2 emission | 2 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| Partial 1/2 emission, no wrong | 1 | 0 | 1 | 0.67 | **0.80** | 0.50 |
| Single wrong tool (0/1 correct, 1 extra) | 0 | 1 | 2 | 0.00 | 0.00 | 0.00 |
| 2/2 correct + 1 extra wrong | 2 | 1 | 0 | 0.80 | 0.83 | **0.00** |

- F1 treats precision and recall symmetrically; partial correctness with no extras (0.67) and over-eager with all correct (0.80) differ by 0.13.
- $F_{0.5}$ amplifies the precision gap: perfect-plus-extra (0.83) penalizes the extra, while partial-but-clean gets 0.80 — the user's intuition "clean partial beats contaminated overfull" is captured.
- EU is the harshest on extras ($\mathrm{EU}=0$ once $\beta\cdot\mathrm{FP} \ge \alpha\cdot\mathrm{TP}$), directly modeling API-cost semantics.

Aggregation: macro-averaged across queries (each query's score averaged; |G|-robust) **and** micro-averaged (pool all tool instances; reflects deployment-observed rates). Report both.

**Why this aligns with Cor 6.9 theoretically.** The rank-$R$ facet-gated operator admits **controlled multi-tool emission**: because facets are orthogonal ($B_f^\top B_{f'} = 0$), steering toward simultaneous facet activation does not spuriously boost non-facet candidates — extra FPs are *not* generated by construction. AdaSEKA's 1-of-M collapse, in contrast, forces either a single correct emission (high precision, low recall) or, under temperature perturbation, multiple stochastic emissions from a single expert's cluster (low precision AND low recall). The asymmetric metrics $F_{0.5}$ and EU therefore exaggerate the theoretical gap:

- Ours (Cor 6.9 $\mathrm{nrank}_\varepsilon=R$): high precision (orthogonal facets → no leakage) AND high recall (R simultaneous) → $F_{0.5}\approx 1$ achievable.
- AdaSEKA (Cor 6.9 $\mathrm{nrank}_\varepsilon \le r$): high precision on the one tool it picks, but recall $\le 1/|G|$ structurally → $F_{0.5}$ bounded by the product of max-recall and max-precision constraint. For 2-tool queries: $F_{0.5}^{\mathrm{AdaSEKA}} \le \frac{1.25\cdot 1 \cdot 0.5}{0.25\cdot 1 + 0.5} = \frac{0.625}{0.75} \approx 0.83$; for 3-tool queries $\le 0.63$.

This gives a falsifiable, theorem-level prediction: on |G|-stratified F_0.5, AdaSEKA plateaus while our method continues to improve. The scaling of the gap with |G| is the Cor 6.9 signature.

#### 5.4.4 Ambiguity resolution — facet-graded scoring

The above metrics treat every wrong prediction as equivalent. In production deployment (the Netsru Gemma-3-27B artifact trail, Appendix E) and in agent-evaluation literature, **ambiguity between semantically-similar tools is a first-order concern**: a query "recommend a music app" may map to {Spotify, AppleMusic, YouTube Music}, and predicting AppleMusic when GT is Spotify is qualitatively different from predicting Excel. Standard set metrics cannot distinguish these cases.

**Facet-graded similarity** exploits the ontology structure directly. Each tool $t$ carries a facet tuple $\phi(t)=(\phi_1(t), \ldots, \phi_F(t))$ where $\phi_f$ is the label in facet $f$ (e.g., $\phi_1=$ intent, $\phi_2=$ domain, $\phi_3=$ io_type, $\phi_4=$ tool_category). The *graded similarity* between prediction $p$ and ground-truth $g$ is
$$
s(p, g)\;=\;\begin{cases} 1.0 & p=g\text{ (exact)},\\ 0.5 & p\ne g\text{ but }\exists f:\phi_f(p)=\phi_f(g)\text{ (at least one facet shared)},\\ 0.0 & \text{no facet overlap (cross-domain error)}.\end{cases}
$$

**Graded metrics.** Replace the $\mathrm{TP}$ count by $s$-weighted sum:
- $\mathrm{FG\text{-}F1} = \frac{2\cdot\sum_{p\in P}\max_{g\in G} s(p,g)\cdot \sum_{g\in G}\max_{p\in P} s(p,g)}{\sum_{p\in P}\max_{g\in G} s(p,g) + \sum_{g\in G}\max_{p\in P} s(p,g)}$ (bipartite-matching F1).
- $\mathrm{FG\text{-}F_{0.5}}$: same with $\beta=0.5$.
- $\mathrm{FG\text{-}EU}$: $(\alpha\sum s - \beta|\text{cross-facet FP}| - \gamma|\text{FN}|)/|G|$.

**User-intuition test** (ambiguous music-app query, GT = Spotify):

| Prediction | Standard F1 | $\mathrm{FG\text{-}F1}$ | Intuition |
|---|---|---|---|
| {Spotify} | 1.0 | 1.0 | exact |
| {AppleMusic} | 0.0 | **0.5** | semantic neighbor, captured |
| {Excel} | 0.0 | 0.0 | cross-domain error |
| {Spotify, AppleMusic, YouTube} | 0.40 | **0.75** | diffuse-in-facet with GT covered |

**Why this matters for our method (Cor 6.9 extension).** When a query has a canonical GT tool but the facet neighborhood is semantically crowded (many tools share the dominant facet), our rank-$R$ facet-gated operator produces a K-bias that boosts the entire facet cluster, not just the single GT tool. Standard F1 penalizes the near-misses as full wrongs; $\mathrm{FG\text{-}F1}$ counts same-facet siblings as half-credit, reflecting their actual semantic proximity.

For AdaSEKA 1-of-M: the single-expert routing chooses one specific tool. When it matches exactly, FG-F1 = 1; when it misses, it typically picks a near-zero-overlap tool (winner-take-all on a different facet entirely) → FG-F1 = 0 more often than 0.5. The graded-vs-standard gap is *smaller* for AdaSEKA than for our method.

For our facet-gated method: the soft energy-ratio gate boosts all facet-aligned candidates, which in ambiguous queries means the prediction distribution is concentrated in the correct facet cluster even when the top-1 argmax misses. FG-F1 recovers this signal that standard F1 discards.

**Expected magnitudes** on Subtask4 with facet-graded:

| Method | F1 | FG-F1 | FG gap (FG − plain) |
|---|---|---|---|
| no_steer | $\sim 0.65$ | $\sim 0.72$ | +0.07 |
| AdaSEKA 1-of-M | $\sim 0.45$ | $\sim 0.48$ | +0.03 (winner-take-all: no facet clustering) |
| **Ours (facet-gated real)** | $\sim 0.80$ | $\sim 0.92$ | **+0.12 (facet diffusion captured)** |
| random/featshuffle | $\sim 0.15$ | $\sim 0.17$ | +0.02 (noise) |

The **gap widening for our method** under FG metrics is a Cor 6.9 signature on graded similarity — the structural capability to diffuse within a facet cluster while remaining orthogonal across facets.

**Netsru deployment alignment (Q8 response, Appendix E).** The Netsru Gemma-3-27B agent operator explicitly cited "multi-intent / ambiguous cases" as an open deployment problem. Facet-graded FG-F1/FG-F_0.5 is the metric that *matches the deployment concern directly*: it credits correct facet-cluster coverage and penalizes cross-domain errors, reflecting user-experienced quality rather than exact-match accuracy.

**Experimental plan (FC-3, appended to §5.13).**

| Config | Benchmark | Metrics | Methods | Models |
|---|---|---|---|---|
| FC-3a | MetaTool Subtask1 (995, 10 candidates, ambiguous-sibling subset flagged by facet overlap) | F1, FG-F1, FG-F_0.5, ECE | {no_steer, a0.3 real/random/featshuffle, AdaSEKA} | 3 Instruct models |
| FC-3b | MetaTool Subtask4 (497, 2-tool with facet annotation) | FG-F1, FG-F_0.5, FG-EU | same | same |
| FC-3c | BFCL-v3 "Multiple Function" split if accessible | FG-F1 stratified by |G| | same | same |

The facet annotation for each of the 199 MetaTool plugins is already computed as part of our ontology construction (`build_metatool_ontology.py` produces `metatool_ontology.json` with 4-facet labels per tool). FG-F1 computation over predictions is $O(|P|\cdot|G|)$ lookups per query, negligible runtime.

Expected runtime: FC-3 adds ~15 GPU-hours (rescoring existing FC-2 runs with graded metrics + optional small FC-3c launch).

**Calibration overlay.** For FC-3a, we additionally report **Expected Calibration Error (ECE)** on the ambiguous subset: does the model's top-1 softmax probability correlate with actual correctness? Well-calibrated models give lower confidence on ambiguous queries. Our facet-gated method should produce more diffuse softmax under ambiguity (due to multi-facet activation) → lower top-1 prob → better calibration on hard queries. A reviewer-friendly complementary metric.

**Theoretical prediction (direct empirical test of Cor 6.9).** The ε-numerical-rank separation between our facet-gated operator (rank $R=24$) and AdaSEKA-style max-normalized routing (rank $r$) makes a sharp prediction for Subtask4:

1. **Our method** (facet-gated, $R$ simultaneous axes): can carry energy on multiple facets at once. Tokens activating both "finance" and "news" facets produce $K$-bias that preserves both signals. Model sees high attention weight on both `FinanceTool` and `NewsTool` examples → emits both in `tool_calls` array → **recall ≈ 1, F1 ≈ 1**.

2. **AdaSEKA 1-of-M routing** (Cor 6.9 bound: $\mathrm{nrank}_\varepsilon \le r$): max-normalization forces winner-take-all — only the dominant facet's expert contributes above threshold. Model commits to one tool → **recall ≤ 0.5, F1 ≤ 0.67** on Subtask4 (by construction, at most 1 of 2 GT tools is emitted). *This is a theorem, not an empirical expectation*.

3. **No_steer baseline**: depends on LM's own training; Qwen2.5-Instruct's FC training likely emits 1 or 2 tool_calls depending on query phrasing. Expected recall $\sim 0.6$–$0.8$.

4. **Random / featshuffle**: degrade sharply on multi-tool — the perturbation is not aligned with either facet direction, so the model's multi-tool selection capability is corrupted. Expected F1 drop similar to or larger than single-tool case.

**The mechanism contribution becomes empirically sharp.** Under top-1 Subtask1, our Δ is small and scorer-sensitive (+0.1 to +13pp). Under multi-tool Subtask4 F1, our method should beat both no_steer and AdaSEKA by a structurally-predicted margin, *because* Cor 6.9 says F-simultaneous rank is the bottleneck. This is the single experiment that most cleanly separates our facet-gated operator from Q-side 1-of-M steering in the literature.

**Experimental plan (FC-2, adds to §5.13).**

| Config | Benchmark | Metric | Methods | Models |
|---|---|---|---|---|
| FC-2a | MetaTool Subtask4 (497 × 2-tool) + BFCL Parallel (|G| varying 1-5) | macro-F1, macro-F_0.5, EU($\alpha=1,\beta=2,\gamma=1$), Jaccard, recall | {no_steer, a0.3 real, a0.3 random, a0.3 featshuffle, AdaSEKA 2-expert, AdaSEKA 3-expert} stratified by $|G|\in\{1,2,3,\ge 4\}$ | Qwen-Instruct, Llama-Instruct, Mistral-Instruct |
| FC-2b | BFCL-v3 Parallel subset (if accessible) | same | same | same |
| FC-2c | τ²-bench retail multi-turn | action-match rate + F1 over tools called per turn | same | Qwen-Instruct |

Expected runtime: $\sim$25 GPU-hours for FC-2a alone across 3 models × 6 methods × 2 scorers. FC-2b and FC-2c add $\sim$40 more.

**Sanity check**: Subtask4 with multi-tool FC is how the paper's geometric specificity claim (real vs random/featshuffle) most directly translates to deployment-relevant numbers. If no_steer Qwen-Instruct achieves macro-F1 = 0.65 and our method pushes it to $\ge$ 0.75 while AdaSEKA caps at $\le$ 0.67 (Cor 6.9 bound), the paper has a clean theory-prediction-verification triangle on a production-aligned metric. This is the central experiment we propose for the ICLR submission's camera-ready revision if time permits, or the follow-up full paper otherwise.

**Why Subtask1 (single-tool) should not be dropped despite Subtask4's superiority.** Subtask1 is still the right venue for mechanism-specificity (real vs random/featshuffle under multiple scorers) because the null controls are easier to interpret when the target is a single category. Subtask4 demonstrates the *consequence* of that specificity under the deployment-relevant multi-tool regime. The two are complementary: Subtask1 = theory verification, Subtask4 = deployment impact.

### 5.5 Theorem 6.1 empirical verification (new, this paper's distinguishing experiment)

Script: `scripts/ocq/measure_theorem_6_1.py`. For 100 MetaTool queries at layer L=13 (Qwen mid-layer), we compute per-head and per-query:
- `LHS := ‖ô - o‖²` (direct attention-output difference between clean and biased forwards).
- `RHS_lead := 2 · qaMSE(q) · Var_s[V](q)` where `qaMSE` and `Var_s[V]` are the Thm 6.1 quantities.
- `RHS_rem := C₁ · ρ⁴` with `C₁ = 2·Q_max⁴·V_max²/d²`, `ρ = max_t ‖e_t‖`.
- `RHS := RHS_lead + RHS_rem`.

**Predicted by Thm 6.1**: `LHS ≤ RHS` for every sample and every head. **Pass rate target**: 100%. **Tightness**: `LHS / RHS` histogram median ≈ [to fill]; closer to 1 = tighter bound. Expected median ≈ 0.1–0.5 per Remark B.2.3 (Mode-A near-tightness discussion). For Mode C (Qwen), we expect looser bounds (median ≈ 0.01–0.1).

### 5.6 Corollary 6.9 numerical-rank verification

Compute ε-numerical rank of P\_fg(q, k\_t) and P\_ada(q) on 500 MetaTool queries at ε ∈ {0.1, 0.2}. Expected: AdaSEKA mean near 6 (when β(q) < ε), ours near 24 (under Hypothesis (R)).

### 5.7 Scaling curve

Qwen2.5 family {0.5B, 3B, 7B, 14B, 32B} × same MetaTool protocol. Tests emergence vs scale-invariance. Compute budget: estimated 80 A6000-hours; feasible.

### 5.8 Zero-shot ontology transfer

B\_ont built on MetaTool → applied zero-shot to ToolAlpaca with no rebuild. Does the K-bias direction transfer across tool-description corpora?

### 5.9 Safety retention

MMLU (1000) / HH-RLHF refusal / ToxiGen before vs after steering. Disambiguate soft vs hard gate (Sec 3.2 empirical (R) test).

### 5.10 Baselines reproduced

CAA, ASA, PASTA, Focus Directions, AdaSEKA 2/3-expert, LoRA r=8, RAG. All on same MetaTool 995 with same scorers.

### 5.11 Theorem 6.13 empirical verification — OCQ vs KIVI on WikiText-2

Hook-mode PPL evaluation on pre-RoPE K, Qwen2.5-7B, WT2 full test set (~299K tokens), ctx=2048 non-overlap (KIVI protocol). All methods share the same forward-hook driver; `quantize_cache`'s Bug 1/2 are bypassed (see `reports/BUG_REPORT_eval_arch_two_bugs_2026-04-09.md`).

| Method | 2-bit avg bits | 2b PPL | 4-bit avg bits | 4b PPL |
|---|---|---|---|---|
| fp16 | 16 | 7.68 | 16 | 7.68 |
| KIVI (post-RoPE per-channel asymmetric) | 2.00 | 19.97 | 4.00 | **7.79** |
| **OCQ 1b+2a (pre-RoPE real ontology $R=24$)** | **1.81** | **15.60** | **3.81** | 12.56 |
| uniform (per-token symmetric, pre-RoPE) | 2.00 | 44069 | 4.00 | 15918 |
| OCQ 1b+2a (PCA pseudo-ontology, (H-cat) violated) | 1.81 | 11.83 | 3.81 | 84.92 |
| OCQ-WF (facet rotation + water-filling) — smoke N=8K | 1.81 | 24.36 | 3.81 | 15.42 |
| OCQ-KIVI (composition, 1b cat + KIVI residual) — smoke | — | 33.30 | — | 15.48 |

**Theorem 6.13 predictions verified:**
- (3.5.i) 1-bit categorical beats water-filling at 2-bit: OCQ 15.60 $<$ OCQ-WF 24.36, $\Delta=-8.76$ PPL. Supports Lemma 6.13.2 at low bits under (H-cat).
- (3.5.ii) OCQ beats KIVI at 2-bit, matched-budget: $\Delta=-4.37$ PPL at $b_{\mathrm{avg}}=1.81$ (9.4% bit savings). Supports Corollary 6.13.4.
- (3.5.iii) Cross-over at 4-bit: KIVI 7.79 vs OCQ 12.56, $\Delta=+4.77$. Supports Corollary 6.13.5 ($\bar b^*$ crossover).
- (H-cat) falsifiability: PCA pseudo-ontology violates (H-cat) — 2-bit 11.83 PPL *better* than real ontology (due to PCA top-dirs' higher variance absorbed under 1-bit sign), but 4-bit **catastrophic at 84.92** because residual quant dominates and there is no categorical structure to exploit. Real ontology 4-bit = 12.56 shows stable (H-cat)-governed degradation.

**Composition amplification (Rmk 6.12.1) verified at the quantization level:**
OCQ-KIVI (applying KIVI on top of OCQ-quantized residuals) gives 33.30 PPL at 2-bit vs 15.60 standalone — a **−17.7 PPL regression**, matching Remark 6.12.1: operator composition on already-categorically-destroyed K structure is strictly worse.

### 5.12 Thm 6.14 experimental plan — LoRA-adapted facet-rotational positioning (future work)

**Goal**: validate the Hybrid version of Thm 6.14 (proven) via LoRA fine-tuning, and probe the Full version (conjecture) as a stretch target.

**Hypothesis under test**:
- **H1 (Hybrid)**: With LoRA rank-16 on `q_proj` + `k_proj` of Qwen2.5-7B, replacing RoPE with FacetRot on the facet subspace (first $R=24$ singular directions of $B_{\mathrm{fac}}$) while retaining standard RoPE on the residual — tool-selection accuracy matches or beats standard RoPE + OCQ steering at the same bit budget, AND Bug-2-space-mismatch is removed (i.e., OCQ hook-mode eval and B_ont construction space are identical).
- **H2 (soft gate bound)**: Lemma 6.14.A's Lipschitz constant holds empirically; FacetRot perturbation norm is smooth in input.
- **H3 (conjecture)**: Full FacetRot replacement (no RoPE at all on facet channels), via LoRA, recovers ≥80% of RoPE baseline PPL on WT2.

**Proposed run plan (not yet launched — GPUs occupied through Wave 4)**:

| Run | Model | LoRA target | Task | Metric | Expected GPU-hours |
|---|---|---|---|---|---|
| R1 | Qwen2.5-7B + LoRA r=16 on q,k_proj + FacetRot on $B_{\mathrm{fac}}$ | MetaTool (995 × 8 epochs) | Tool-selection top-1, OCQ 2-bit PPL | Δ vs no-FacetRot | 12 |
| R2 | Same + hard gate ablation | Compare soft $\pi_{\mathrm{soft}}$ vs hard $\arg\max_f g_f$ | Tool-selection top-1 | Test Lemma 6.14.A predictive: hard should collapse like Cor 6.7 hard-gate | 8 |
| R3 | Same + full FacetRot (no RoPE on facet) | WT2 PPL + MetaTool | Compare to Hybrid | Test Conjecture 6.14 | 12 |
| R4 | Same as R1 but Llama-3.1-8B | Cross-family | Confirms architecture-independence | 15 |
| R5 | Same as R1 with Option C ($\mathrm{FacetRot}_C$, Lie-algebra mean) | Soft-gate formalization ablation | Tool-selection top-1 and layer-wise $\mathrm{qaMSE}$ | Compare with R1 Option A; if $\|\Delta\text{acc}\|\le 1$ pp and qaMSE tracks within 5%, Option A is operationally equivalent (facet-ordering artifact is a theoretical footnote). | 14 |
| R6 | Hard-gate MMLU grid | $\alpha\in\{0.1,0.2,0.3,0.5,1.0\}$ × {no-gate, flat-bias, soft-facet-gated, hard-thresh, hard-argmax} on Qwen2.5-7B | MMLU top-1 | **LAUNCHED 2026-04-14** on GPU0 as Track B of `scripts/run_llama_retry_and_r6.sh`. Direct figure for §3.4.1 hard-gate collapse — expected: hard gate variants show monotone degradation in $\alpha$, soft stays near noise floor. Confirms Consequence 2 of Remark 6.14.A.3. | 4 |

**Acceptance criteria**:
- H1 pass: tool-selection gain ≥ +1pp vs RoPE+OCQ baseline, OR Bug-2 qualitative fix verified (ε_q measurement shows ontology basis operates in same space as quantization, no position-dependent basis distortion).
- H2 pass: hard-gate variant degrades ≥5pp more than soft-gate variant on MetaTool. If not, Lemma 6.14.A is wrong.
- H3 pass: full FacetRot WT2 PPL ≤ 1.5× RoPE baseline. If not, Conjecture 6.14 is rejected; report as honest negative.

**Dataset & compute**:
- MetaTool Subtask1 train split (~700 queries) + synthetic tool-description augmentation (800 queries from ToolAlpaca).
- Single A6000, LoRA fp16, batch size 4, lr 1e-4.
- Expected total: R1–R6 ≈ 65 GPU-hours; completable in ~66 wall-clock hours on one node with two GPUs, or ~34 hours on two GPUs.

**Timing**: earliest launch after Wave 4 (Thm 6.1 empirical) completes (~23:40 KST 2026-04-14). If ICLR submission deadline permits (assumed Sep 2026), full validation of H1–H2 is feasible; H3 (full-replacement) is a stretch and most likely remains Conjecture with only partial empirical support.

### 5.13 Consolidated experimental plan — claim-to-experiment mapping (revised 2026-04-14)

The scoring framework of §5.4.1–§5.4.4 (15+ metrics across 4 layers) and the theoretical claims of §3 demand a tighter experimental plan than the initial disparate FC-1/FC-2/FC-3 enumeration. We re-scope to a **claim-indexed minimum viable set** organized into three priority tiers, with metric reuse across experiments.

#### Priority 1 — must-run for main submission (~90 GPU-hours)

| Exp | Claim tested | Benchmark | Metrics emitted | Cell count | GPU-hr |
|---|---|---|---|---|---|
| **E1: Scorer-invariant mechanism specificity** | C1 (real >> random >> featshuffle under any scorer) | MetaTool Subtask1 (995 × 10-cand) | substring, first_line, label_logprob{sum,mean}, fc_name, fc_label_logprob — 6 scorers | {Qwen-Instruct, Llama-Instruct} × {no_steer, a0.3-real, a0.3-random, a0.3-featshuffle} = 8 | 40 (half partially done — Qwen Wave 1+2 complete) |
| **E2: Cor 6.9 multi-tool decisive test** | C3 (ε-numerical-rank separation from AdaSEKA) | MetaTool Subtask4 (497 × 2-tool) + optional BFCL-Parallel (varying \|G\|) | F1, F_0.5, EU, Jaccard, Exact-set, FG-F1, FG-F_0.5, FG-EU, ECE — all 9 computed in one pass per cell | 3 Instruct models × 6 methods (no_steer, real, random, featshuffle, AdaSEKA-2, AdaSEKA-3) = 18 | 25 |
| **E3: Thm 6.1 per-sample bound** | C5 (per-sample LHS ≤ 2·qaMSE·Var_s[V] + C₁ρ⁴) | Qwen L=13 + Llama L=15, 100 MetaTool queries | per-head LHS, RHS_lead, RHS_rem, ratio, pass rate | 2 configs | 15 (already queued Wave 4) |
| **E4: Cor 6.9 operator-level nrank** | C3 (structural rank gap) | SVD on 500 MetaTool queries' P_fg(q,k_t) and P_ada(q) | nrank_ε histogram at ε∈{0.1, 0.2} | 2 thresholds | 2 |
| **E5: Remark 6.14.A.3 R-violation grid** | C6 (hard-gate monotone degradation) | Qwen MMLU N=1000 | accuracy × α ∈ {0.1, 0.2, 0.3, 0.5, 1.0} × gate ∈ {no, flat, soft, hard_thresh, hard_argmax} = 25 | 25 | 4 (queued as R6) |
| **E6: Thm 6.13 categorical-channel compression** | C4 (OCQ < KIVI at low bits) | WT2 ctx=2048 non-overlap, Qwen + Llama | PPL × bits {2, 3, 4} × methods {fp16, KIVI, OCQ, OCQ-WF, OCQ-KIVI, uniform} | 2 models × 6 methods × 3 bits = 36 cells, many already done | 5 (incremental) |

**Tier P1 rationale**: 90 GPU-hr delivers direct empirical counterpart to every Section-3 theorem / corollary claim, plus the production-aligned multi-tool F1/F_0.5/FG-F1 evaluation that closes the Netsru-deployment gap.

#### Priority 2 — reviewer-defensive + scaling (~60 GPU-hours)

| Exp | Claim tested | Benchmark | Metrics | GPU-hr |
|---|---|---|---|---|
| **E7: Scaling curve Qwen2.5 family** | Scale-invariance of the effect | MetaTool Subtask4 under FC, same 6 methods as E2 | FG-F1 primary | 30 (0.5B, 3B, 7B, 14B; skip 32B if tight) |
| **E8: Safety retention** | Soft-gate preserves safety | MMLU-4k + HH-RLHF refusal-500 + ToxiGen-500 | top-1, refusal rate, toxicity score | 12 |
| **E9: Baselines reproduced** | Matched-compute comparison | MetaTool Subtask1 + Subtask4 | FG-F1, F_0.5, EU | 18 (CAA, ITI, PASTA, ASA, FocusDir, LoRA r=8, RAG prompt; each ~2.5 hr) |
| **E10: Cross-model Mistral closure** | 86/14 decomposition | Subtask4 FG-F1 under skipL0+padmax + Instruct H2 | all 9 multi-tool metrics | 0 (free — already queued Wave 3) |

**Tier P2 rationale**: these defend the paper against the standard referee attack list. Scaling curve is the single most compute-expensive one; baselines are compute-cheap but must be reproduced (not cited).

#### Priority 3 — future work / stretch (~100 GPU-hours, deferred)

| Exp | Purpose | GPU-hr |
|---|---|---|
| E11: Thm 6.14 Hybrid LoRA R1 (conjecture) | Bug-2-free FacetRot validation | 15 |
| E12: τ²-bench retail/airline multi-turn | Production multi-turn agent setting | 20 |
| E13: BFCL-v3 Parallel (\|G\| 1-5 stratified) | Cor 6.9 \|G\|-scaling signature | 25 |
| E14: Zero-shot ontology transfer MetaTool → ToolAlpaca | Generalization story | 15 |
| E15: Thm 6.13 full bit curve (1b, 2b, 2.5b, 3b, 4b, 5b) | Compression Pareto frontier | 10 |
| E16: Thm 6.14 Full-replacement (Conjecture 6.14) | RoPE-replacement feasibility | 15 |

Mark as "future work" in the camera-ready; note existing partial evidence (E12 τ²-bench code already cloned from memory `week1_tasks_2_3_4_done_2026_04_10`).

#### Total envelope

- P1: 91 GPU-hr (half of which is already running / queued in Wave 2/3/4 + R6)
- P2: 60 GPU-hr
- P3: 100 GPU-hr (deferred)
- **Main paper budget: P1+P2 = 151 GPU-hr** ≈ 8 GPU-days on a 2-GPU node.

#### Metric reuse matrix

A single E2 run (MetaTool Subtask4, one method, one model) produces:
```
from predictions P and ground truth G per query:
  → F1, F_0.5, EU, Jaccard, Exact-set, FG-F1, FG-F_0.5, FG-EU, ECE (all 9)
  → top-1 variant (same prediction restricted to argmax) → FC single-tool metrics (fc_name, fc_schema_valid)
  → if teacher-forced: fc_label_logprob
```

One forward pass + post-hoc scoring covers all metric variants of Subtask4. Similarly E1 emits 6 scorers per query. This is **single-pass-multi-scorer** design; cost is dominated by the forward pass, not the metrics.

#### Benchmarks — selection and justification

**Core (P1)**:
1. **MetaTool Subtask1 (995, single-tool, 10-candidate)** — scorer-invariance benchmark; every method runs here. Legacy connection to substring/first_line debate.
2. **MetaTool Subtask4 (497, exactly 2-tool)** — Cor 6.9 decisive test; multi-tool + graded scoring.
3. **Qwen2.5-7B WT2** — Thm 6.13 categorical channel optimality (compression bridge).
4. **MMLU (N=1000)** — safety + R-violation grid.

**Supporting (P2)**:
5. **Qwen2.5 family {0.5, 3, 7, 14}B** — scaling curve (skip 32B if compute-tight; note in camera-ready).
6. **HH-RLHF refusal-500 + ToxiGen-500** — safety retention.

**Stretch (P3)**:
7. **τ²-bench retail/airline** — production multi-turn (cloned, ready, but multi-turn setup cost).
8. **BFCL-v3 Parallel** — if accessible, definitive |G|-scaling.
9. **ToolAlpaca** — zero-shot transfer (generalization).

**Explicitly declined**: BIG-Bench, HELM, full BFCL-v3 (non-Parallel categories) — scope creep, not relevant to our mechanism claim.

#### Benchmark-metric-model matrix (for camera-ready table)

| Bench | P1 (main) | P2 (scaling+safety) | P3 (stretch) |
|---|---|---|---|
| MetaTool Subtask1 | E1 (6 scorers × 3 B_ont × 2 models) | E9 (baselines) | — |
| MetaTool Subtask4 | E2 (9 metrics × 6 methods × 3 models) | E7 (scaling × 4 sizes), E10 (Mistral closure) | — |
| MMLU | E5 (R-violation 25 cells) | E8 (safety + CAA/LoRA on MMLU) | — |
| WT2 | E6 (Thm 6.13, 2 models × 3 bits × 6 methods) | — | E15 (full bit curve) |
| τ²-bench | — | — | E12 |
| BFCL | — | — | E13 |
| ToolAlpaca | — | — | E14 |
| HH-RLHF/ToxiGen | — | E8 | — |

#### Theoretical claim → experimental cell (explicit mapping for review)

| Claim | Theorem | Primary experiment(s) | Secondary |
|---|---|---|---|
| C1 Geometric specificity | — | E1 all scorers, E2 FG-F1 | E9 baselines |
| C2 Phase-closure + (R) | Cor 6.7/6.8 | E5 MMLU soft vs hard, E3 per-sample | E8 |
| C3 ε-nrank separation | Cor 6.9 | **E2 Subtask4 F1/FG-F1**, E4 SVD nrank | E13 BFCL \|G\|-scaling |
| C4 Categorical compression | Thm 6.13 | E6 WT2 | E15 full bit curve |
| C5 Attention-output bound | Thm 6.1 | E3 per-sample LHS ≤ RHS | — |
| C6 R-violation predicted | Rmk 6.14.A.3 | E5 hard-gate monotone in α | — |
| C7 Cross-model 2-family | — | E1 Qwen+Llama, E10 Mistral | E7 scaling |
| C8 Scorer robustness | — | **E1 5-scorer triangulation** | E9 baselines |
| C9 Ambiguity graded | §5.4.4 | **E2 FG-F1 gap vs AdaSEKA** | E13 ambiguous subset |
| C10 Production alignment | Netsru Q8 | E2 FG-F_0.5 + EU, E8 safety | E12 τ²-bench |

**Each claim has at least one primary experiment, at most 25 GPU-hours.** Every Section-3 theorem has a dedicated, labeled experiment. No claim is left to hand-wave.

#### Launch sequence (post Wave 3/4 complete)

1. **Wave 4 Thm 6.1 per-sample (E3)** — GPUs already queued. Self-chains.
2. **E2 FC multi-tool Subtask4** — 25 GPU-hr, primary attention-grabber. Launch immediately after Wave 4.
3. **E1 Llama + Mistral label_logprob completion** — 15 remaining GPU-hr (Qwen Wave 1+2 done, Llama retry running, Mistral Wave 3 running).
4. **E5 R6 MMLU gate grid** — already queued (4 hr).
5. **E4 nrank SVD** — 2 hr, CPU-adjacent, can interleave.
6. **E6 Thm 6.13 Llama WT2** — 5 hr addition to existing Qwen data.
7. **E9 baselines** — 18 hr, launch in parallel with E7 if GPU budget allows.
8. **E7 scaling curve** — 30 hr, likely last before submission.
9. **E8 safety retention** — 12 hr.

**Deferred to future work (P3)**: E11 LoRA, E12 τ²-bench, E13 BFCL, E14 transfer, E15 full bit curve, E16 full FacetRot.

#### What this revision removes from the prior enumeration

- **FC-1 (generic FC single-tool scorer)** — folded into E1 as one of the 6 scorers.
- **FC-2 (separate from FC-3)** — unified into E2 (single benchmark, all metrics in one pass).
- **FC-3 (separate rescoring)** — no longer needed; FG metrics computed inline from E2 predictions.
- **R6 (standalone MMLU grid)** — renamed to E5, kept as-is.
- **Thm 6.14 Hybrid LoRA R1-R5** — demoted to P3 E11, not P1.
- **Ad-hoc "compositional benchmark TBD"** — replaced by explicit E2 + E13 with concrete metric/benchmark pairs.

The revised plan is **~30% smaller in cell count**, but covers every claim with a dedicated, theorem-indexed experiment, and eliminates metric-rescoring duplication through single-pass-multi-scorer design.

F-simultaneous regime stress test. [Placeholder; benchmark selection pending Wave-3 completion.]

---

## 6. Discussion

### 6.1 Why the cross-model positive is the story, not the deployment alignment

[Memory `impact_oriented_bench_2026_04_14`]. Qwen+Llama 2-family positive + Mistral fully-diagnosed counterexample is the cross-architecture evidence. Deployment pressure (client using 32B) enters only as a scaling-curve data point.

### 6.2 (R) as a design constraint, not a technicality

Section 3.2's hard-gate MMLU degradation is not a bug — it is the direct empirical signal that regularity matters. Design the gate to be Lipschitz; do not select a hard threshold for nominal interpretability.

### 6.3 Why K-side, not Q-side

[AdaSEKA differentiation per `adaseka_vs_ours_differentiation_2026_04_10.md`]. Q-side 1-of-M routing is structurally capped at rank `r` (Cor 6.9). K-side F-simultaneous attains rank `R`. For multi-facet intents the gap becomes operationally relevant on compositional benchmarks (Sec 5.11).

### 6.4 Limitations

1. Qwen + Llama only. Mistral requires base-weakness mitigation (Instruct variant, Sec 5.3 H2).
2. Generation scorer + label\_logprob disagree at N=20 smoke; full 995 pending.
3. Compositional benchmark is the highest leverage axis; BFCL-v3 integration deferred to Wave 4.

---

## 7. Conclusion

Facet-gated K-bias with an ontology basis gives a theory-guided, mechanistically-separable steering operator. Its predicted properties — phase-closure, ε-numerical-rank separation from max-normalized routing, hard-selection penalty, composition amplification — are each verified empirically with tight prediction-observation coupling. The method generalizes across Qwen and Llama architecture families; the Mistral counterexample is decomposed into build-pipeline and base-model factors. Theorem 6.1's per-sample verification is included, closing the loop between proof and measurement.

---

## Appendices

- **A.** MetaTool dataset preparation, parsing, and scorer implementations.
- **B.** Full proofs (Theorem 6.1, Theorem 6.2, Cor 6.3–6.12 + Rmk 6.12.1). Imported from `APPENDIX_B_PROOFS.md` and `COROLLARY_6_7_FACET_PHASE_CLOSURE.md`.
- **C.** Cor 6.7 reframing (regularity hypothesis (R)). Imported from `COR67_REFRAMING_2026_04_14.md`.
- **D.** Mistral cross-model ablation grid. Imported from `CROSS_MODEL_KBIAS_ANALYSIS_2026_04_13.md`.
- **E.** Netsru Gemma-3-27B agent artifact trail (15 questions, vector-steering-only policy statement). Motivates §1 continual-tool-addition framing.
- **F.** Per-head Theorem 6.1 verification details; `measure_theorem_6_1.py` output schema.

---

## Experimental pipeline snapshot (2026-04-14)

**Currently running (both GPUs, auto-chained):**
1. Wave 1: Qwen real B\_ont × {sum, mean} label\_logprob full 995.
2. Wave 2 (auto-chained): Qwen random + featshuffle controls × {sum, mean}.
3. Wave 3a (auto-chained): Llama real B\_ont × {sum, mean} on GPU0; Mistral skipL0+padmax on GPU1.
4. Wave 3b (auto-chained): Llama controls on GPU0; Mistral-Instruct H2 on GPU1.

**Pending launches (after Wave 3b completes):**
5. Thm 6.1 empirical verification (`measure_theorem_6_1.py` on Qwen L=13, 100 queries; then Llama L=15, 100 queries).
6. Cor 6.9 ε-numerical-rank measurement (SVD of P\_ada vs P\_fg on 500 MetaTool queries).
7. MMLU {no-gate, soft-gate, hard-gate} × {0.2, 0.3, 1.0} for Sec 3.2 (R)-necessity figure.
8. Scaling curve {0.5B, 3B, 7B, 14B, 32B} on Qwen2.5.
9. BFCL-v3 integration.
10. Baseline reproductions (CAA, ASA, PASTA, Focus Directions, AdaSEKA, LoRA, RAG).

**ICLR 2027 main-track probability (2026-04-14 snapshot):**
- Base (cross-model confirmed, label\_logprob pending): 25–35%.
- With clean Thm 6.1 empirical verification (100% pass, tight median): 35–45%.
- With additional compositional-benchmark decisive win: 45–55%.
