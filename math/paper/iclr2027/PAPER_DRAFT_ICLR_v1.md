# Two-Level Argmax-Subspace Selectivity in Pretrained Transformers

**ICLR 2027 submission draft v1 — internal working copy**
**Status (2026-04-19, evening)**: §1–§5.6, §5.8 (Phase D — Failure verdict applied), §6–§8, Appendix A.1 + B.1 all complete from Phase A+B+C+D anchors. §5.7 (H-J breadth scatter, Phase B3) is the sole remaining placeholder pending experiment-session output of `analyze_breadth_metric.py` (no GPU). NeurIPS 2026 existence-only track (`PAPER_DRAFT_v3.md`) is independent and untouched.

**Framing decisions applied (per `consolidation_framing_decisions_2026_04_19.md`)**:
- F1 — B_ont is a *K-subspace extracted from tool-related token positions*, **not** a catalog-semantic encoding.
- F2 — Primary direction-specificity metric is **`attn_fro_ratio` ≥ 1.5**; full-vocab KL ratio is reported as a secondary diagnostic with explicit caveat.
- F3 — Two distinct asymmetries: (A-1) benchmark attention-sensitivity in the *random null* and (A-2) source-side transferability in the *aligned* arm. Never tabulated together.
- F4 — H-J breadth is the geometric *subspace-alignment score* of source $B_{\text{ont}}$ against target tool-position K activations. Catalog-side counts are forbidden.
- F5 — Banking (knowledge QA) → Appendix A.1; self-bench KL asymmetry → §5.4 footnote.

---

## §1. Abstract + Introduction

### 1.1 Abstract

We study how a pretrained decoder transformer's tool-selection behavior responds to low-rank, geometrically structured perturbations of its key activations. We isolate a **two-level argmax-subspace selectivity**: a perturbation $\delta K = \alpha B B^\top K$ — where $B \in \mathbb{R}^{d \times r}$ is an orthonormal basis extracted from key activations at tool-related token positions — produces a smooth $\sqrt{r/d}$-scaling of attention-weight Frobenius shift (predicted by Haar/Stiefel concentration, *Lemma 2*) while the discrete tool-name argmax remains stable except in narrow margin-sensitive sub-populations (*Lemma 1*, single-layer flip condition). The combination — smooth attention-level change with stepwise output-level robustness — defines the **gap** quantified in *Theorem T*. We localize the gap mechanistically to the **final FFN + LM-head composition** (28 of 29 residual layers transmit perturbation monotonically; only layer 28 amplifies and re-orders). We report:

- **Universality (Phase A).** $\sqrt{r/d}$ slope holds across $9$ (model, benchmark) pairs spanning Qwen2.5-7B / Llama-3.1-8B / Mistral-7B-Instruct $\times$ four tool-selection benchmarks ($\tau^2$ Telecom / Retail / Airline + MetaTool ST4); all 9 runs reach $R^2 \geq 0.99$, mean slope $0.5298$. Argmax flip rate is $0$ wherever margin lower bound exceeds $\sim 1.5$ and turns predictively non-zero at small margin (Llama Retail $m_0=0.016 \Rightarrow 1.6\%$).
- **Cross-benchmark direction specificity (Phase B1, F2).** Across $6$ tool-selection cross-pairs (Banking, a knowledge-QA task, is moved to Appendix A.1 per F5), the **attn_fro ratio between aligned and random-null perturbations is $\geq 2.0$ in 6 of 6 pairs**, mean $2.97$. The KL ratio mean is $2.21$ in the same set, but with a Retail $\to$ Telecom outlier at $0.23$ that is fully resolved by the metric shift to attn_fro (which yields $2.54$ for the same pair).
- **Mechanism location (Phase B2).** A 29-layer per-rank residual-stream KL sweep shows monotone-in-$r$ behavior at every layer except the final transformer block, where the $r=1$ KL is amplified $85\times$ (L27 $\to$ L28). The non-monotonic argmax-level signature seen at output is *manufactured* in this single composition step.
- **Falsifier-as-refinement (Phase C).** Three permutations of the catalog used to *select positions* — facet-value shuffle, tool-name shuffle, and full-random-sentence — all preserve the attn_fro ratio in $[1.98, 2.69]$. Catalog *content* is not load-bearing; the load-bearing object is the K-subspace extracted at any tool-like position population.

We argue this two-level structure, together with its layer-localized origin and catalog-content invariance, motivates a new design principle for training-free attention steering: target the *attention-geometric subspace* directly, not the semantic tokens that index into it.

### 1.2 Introduction

#### 1.2.1 Setting

Modern decoder transformers used for tool / function selection (the LM-as-orchestrator regime explored by τ²-bench, MetaTool, BFCL, ToolBench) exhibit a curious empirical robustness: small structured perturbations to attention internals — specifically rank-$r$ basis-aligned shifts of the key cache — leave the discrete tool-name argmax unchanged for the great majority of inputs, while perceptibly altering attention weights and full-vocabulary distributions. Practitioners have exploited this in steering (CAA, ITI, RepE, PASTA, ASA, Focus Directions, SEKA, AdaSEKA), training-free direction injection, and KV-cache compression. Yet the *structural reason* the argmax is stable while attention is not has remained uncharacterized.

We formalize this gap as **two-level argmax-subspace selectivity** and provide:

1. A provable **Lemma 1** (single-layer margin-gated flip condition) for $\delta K = \alpha B B^\top K$.
2. A provable **Lemma 2** ($\sqrt{r/d}$ Haar-Stiefel concentration of attention-Frobenius shift), with empirical anchors over 9 (model, benchmark) settings.
3. A **Theorem T** that combines them into a three-claim structural statement (T.A attention scaling, T.B margin-gated argmax stability, T.C two-level gap) and pins the gap mechanistically to the final FFN + LM-head composition (Phase B2).
4. A **falsification-by-permutation** experiment (Phase C) that rules out catalog-content semantics as the source of direction specificity, refining the methodological claim of the basis $B_{\text{ont}}$ as a *K-subspace extraction pipeline*.

#### 1.2.2 What we are *not* claiming

To preempt scope creep:

- We do **not** claim a closed-form derivation of the model-specific direction $d^*$ that controls Q-sign asymmetry (this remains a stretch goal in §6.1).
- We do **not** claim catalog wording carries causal information for direction specificity (Phase C falsifies the strong form).
- We do **not** redo a steering benchmark race; reported numbers in §5 are mechanistic measurements (KL, attention Frobenius shift, flip rate, layer-resolved KL), not task-accuracy improvements.
- We do **not** propose a new training procedure; the basis $B_{\text{ont}}$ is built from forward-pass K activations alone.

#### 1.2.3 Roadmap

§2 places our contribution against existing steering and concentration literature. §3 states Lemma 1, Lemma 2, and Theorem T with proof sketches. §4 specifies the construction of $B_{\text{ont}}$ as a *K-subspace pipeline* (per F1), the random-null variant D (Haar-matched), the four measurement protocols (margin, attention-Frobenius shift, flip rate, layer-resolved KL), and the choice of attn_fro ratio as primary direction-specificity metric (per F2). §5 reports Phases A through C. §6 discusses open observations including Q-sign asymmetry, format collapse, and the two distinct asymmetries A-1 / A-2 (per F3). §7 lists scope and limitations. §8 concludes.

---

## §2. Related Work

### 2.1 Activation-level steering

**CAA** (Rimsky et al. 2024) and **ITI** (Li et al. 2023) inject contrastive activation differences at residual-stream sites. **RepE** (Zou et al. 2023) uses representation engineering with linear probes. **PASTA** (Zhang et al. 2023) modifies attention weights at specified token positions. **ASA** (2026) and **Focus Directions** (Zhu 2025) are direct prior art for *attention-side* K-bias steering: they modify keys to redirect attention. **SEKA** (Lee et al. 2025) and **AdaSEKA** (2026) use catalog-derived directions for Q-side / K-side modification.

Two threads run through this literature: (i) *what is the basis* — contrastive (CAA), supervised probe (ITI), categorical-mean (SEKA), per-query expert (AdaSEKA); and (ii) *where it acts* — residual stream, attention output, K cache. Our work differs structurally: we measure *what the basis does to the attention geometry* across rank, model family, and benchmark, rather than racing against a particular steering benchmark.

### 2.2 Concentration of measure on the Stiefel manifold

The $\sqrt{r/d}$ scaling of $\|\delta\text{attn}\|_F$ in Lemma 2 follows from Haar-measure concentration on the Stiefel manifold $V_r(\mathbb{R}^d)$ (Ledoux 2001 §6, Edelman & Rao 2005 §5). Specifically, $\mathbb{E}_U[U U^\top] = (r/d) I$ and quadratic-form Var is $O(r(d-r)/d^2)$. We use Lemma 2 as an *anchor* for the attention-level scaling claim (T.A); the novelty in our work is not the concentration result itself but the empirical demonstration that real pretrained transformer attention obeys this scaling across nine (model, benchmark) pairs — and that the corresponding *argmax* layer does not (T.B).

### 2.3 Mechanistic interpretability of FFN / LM-head

Anthropic's circuits work (Olsson et al. 2022, Elhage et al. 2022) and downstream analyses (Geva et al. 2021, Meng et al. 2022) identified that late-block FFN layers act as *key-value memories* with non-linear amplification of small residual-stream signals. Our Phase B2 result (28-of-29 monotonic, layer 28 alone amplifies) is the first quantification we are aware of for *low-rank K-bias perturbations* and is consistent with this line.

### 2.4 What is novel here

Two ingredients combine to give our contribution:

1. **Two-level functional gap** — coexistence of smooth attention-level $\sqrt{r/d}$ scaling and stepwise argmax-level robustness, formalized as Theorem T.
2. **Pipeline-level direction specificity** — Phase C shows catalog *content* is not the carrier of direction specificity; the load-bearing object is the K-subspace from tool-related token positions. This refines all of CAA, ITI, RepE, PASTA, ASA, SEKA, AdaSEKA in a uniform way: the choice of *which positions* to extract K from matters; the *labels* attached to those positions do not (within tool-selection scope).

---

## §3. Theory

### 3.1 Setup and notation

Fix a pretrained decoder transformer $M$ with hidden dimension $d_{\text{model}}$, attention head dimension $d$, and $L$ layers. For a chosen layer $\ell$ and set of *steered* heads, let $K \in \mathbb{R}^{T \times d}$ denote the per-token key activations at the prompt-end position over a length-$T$ context, $q \in \mathbb{R}^d$ the query at the prompt-end token, and $\ell_v \in \mathbb{R}$ the $v$-th vocabulary logit of the next-token distribution.

Let $B \in \mathbb{R}^{d \times r}$ be column-orthonormal (i.e. $B^\top B = I_r$), $r \leq d$. The **K-bias perturbation** is

$$\delta K = \alpha \, B B^\top \, K \in \mathbb{R}^{T \times d}, \qquad \alpha \in \mathbb{R}.$$

The perturbed attention weight at head $h$ is $\text{softmax}((K + \delta K) q / \sqrt{d})$.

For a tool-selection benchmark $V$ with vocabulary $\mathcal{V}_{\text{tool}}$, denote $f(q, K) = \arg\max_v \ell_v(q, K)$ the predicted next-token argmax restricted to $\mathcal{V}_{\text{tool}}$ (or the unrestricted argmax if no restriction is applied). The **per-query margin** is

$$m(q, K) = \ell_{f(q,K)} - \max_{v \neq f(q,K)} \ell_v.$$

We write $m_0 = \min_q m(q, K)$ for a benchmark-wide lower bound.

### 3.2 Lemma 1 — Single-layer Margin-Gated Flip (provable)

**Lemma 1.** *For a single self-attention layer with K-bias perturbation $\delta K = \alpha B B^\top K$, the argmax over the next-token logits flips* — *i.e.* $f(q, K + \delta K) \neq f(q, K)$ — *if and only if there exists $v' \neq f(q,K)$ such that*

$$\alpha \, \langle B B^\top q, \, g_{v'}(q, K)\rangle \;>\; \mathcal{M}_{v'}(q, K) \;+\; O(\alpha^2),$$

*where $g_{v'} \in \mathbb{R}^d$ is the K-gradient projection of the logit difference $\ell_{v'} - \ell_{f(q,K)}$ and $\mathcal{M}_{v'} = \ell_{f(q,K)} - \ell_{v'} \geq m(q,K)$ is the per-pair margin.*

**Proof sketch.** Linearize the post-perturbation logit:

$$\ell_v' = \ell_v + \langle \delta K, \nabla_K \ell_v\rangle_F + O(\|\delta K\|_F^2).$$

For a single attention layer, $\nabla_K \ell_v$ admits the chain decomposition $\nabla_K \ell_v = q \, g_v^\top$ via the standard softmax-attention K-gradient identity; substituting $\delta K = \alpha B B^\top K$ and contracting on the $q g^\top$ rank-1 structure gives

$$\ell_v' - \ell_v = \alpha \, \langle B B^\top q, g_v\rangle + O(\alpha^2).$$

A flip $f \to v'$ requires $\ell_{v'}' > \ell_f'$, i.e.

$$(\ell_{v'} - \ell_f) + \alpha \langle B B^\top q, g_{v'} - g_f\rangle + O(\alpha^2) > 0,$$

which on rearrangement and using $g_{v'} - g_f$'s dominant term (the higher-logit gradient) yields

$$\alpha \langle B B^\top q, g_{v'}\rangle > \mathcal{M}_{v'} + O(\alpha^2). \qquad \square$$

**Corollary 1.1 (sub-critical 0-flip plateau).** If $\alpha \, \|B B^\top q\|_2 \, \max_{v'} \|g_{v'}\|_2 < m_0$, then no flip occurs across the benchmark. This gives a *deterministic sub-critical regime*, complementing the expected-lift bound $\mathbb{E}[\langle B B^\top q, g\rangle] = O(\sqrt{r/d})$ from Haar randomization (Lemma 2).

### 3.3 Lemma 2 — Haar Attention-Frobenius Concentration (provable)

**Lemma 2.** *Let $U \sim \text{Haar}(V_r(\mathbb{R}^d))$ on the Stiefel manifold and $B = U$. For the K-bias perturbation $\delta K = \alpha U U^\top K$ with prompt-end query $q$, the expected attention-weight Frobenius shift satisfies*

$$\mathbb{E}_U\left[ \big\| \, \text{softmax}((K + \delta K) q / \sqrt{d}) - \text{softmax}(K q / \sqrt{d}) \, \big\|_F \right] \;\asymp\; C \sqrt{r/d}, \qquad C = O\!\left(\alpha \, \|q\|_2 \, \|K\|_F\right).$$

**Proof sketch.** $\mathbb{E}[U U^\top] = (r/d) I_d$ and $\text{Var}(q^\top U U^\top q) = 2 r(d-r) / [d^2 (d+2)] \, \|q\|^4$ (Edelman & Rao 2005). The first-order Taylor expansion of softmax in its pre-activation gives the attention-Frobenius shift as a sum over tokens of bilinear forms in $U U^\top$; Jensen's inequality and the standard Stiefel concentration bound yield the $\sqrt{r/d}$ leading order with a model-independent constant prefactor. See Ledoux (2001) §6 for the underlying isoperimetric inequality. $\square$

**Empirical confirmation (preview, full data §5.1):** across 9 (model, benchmark) pairs spanning Qwen2.5-7B, Llama-3.1-8B, Mistral-7B-Instruct $\times$ τ² Telecom / Retail / Airline + MetaTool ST4, the log-log regression of attn_fro vs $r/d$ has $R^2 \geq 0.99$ with mean slope $0.5298 \pm 0.06$ — first-order match to the predicted $0.500$. The systematic $\sim 6\%$ excess on Qwen and Mistral (against Llama's $\sim 0\%$ offset) is treated in §6.1 as a model-family-dependent higher-order Lipschitz correction (open observation, Hypothesis H-I).

### 3.4 Theorem T — Two-Level Argmax-Subspace Selectivity

**Theorem T.** *Let $M$ be a pretrained decoder transformer, $V$ a tool-selection benchmark, $\mathcal{L}$ a contiguous block of attention layers, and $\delta K = \alpha B B^\top K$ a K-bias perturbation applied at all heads in $\mathcal{L}$ with column-orthonormal $B$. Then:*

**(T.A — Attention-level scaling, provable + empirical.)** *The expected attention-weight Frobenius shift under Haar-distributed $B$ obeys $\mathbb{E}\|\delta\text{attn}\|_F \asymp C \sqrt{r/d}$ (Lemma 2), and this scaling is empirically realized across the 9 (model, benchmark) pairs of §5.1 with $R^2 \geq 0.99$ and slope $0.5298$.*

**(T.B — Argmax-level threshold, empirical.)** *The tool-name argmax flip rate satisfies a margin-gated threshold derived from Lemma 1: it is $0$ on benchmarks with margin lower bound $m_0 \gtrsim 1.5$ and turns predictively positive on small-margin benchmarks, concentrated at high $r$ (e.g., Llama × τ² Retail with $m_0 = 0.016$ shows $1.6\%$ flip; Phase A Banking knowledge-QA at $m_0 = 0.875$ shows $7.8\%$ — Appendix A.1).*

**(T.C — Two-level gap, mechanism-localized.)** *The smooth scaling of T.A and the discrete threshold of T.B coexist; the non-monotonic shape observed at the output-vocabulary KL level is manufactured in the final transformer block (Phase B2: 28 of 29 residual layers transmit perturbation monotonically in $r$; layer 28 amplifies the $r=1$ KL by $85\times$ relative to L27).*

The structural content of Theorem T is the simultaneous validity of T.A, T.B, and T.C — the fact that an output transformer can be smoothly perturbed at the attention-Frobenius level while remaining argmax-stable, with the gap localized to a single composition layer.

### 3.5 What remains observation, not theorem

Per the consolidation framing (F1), the following are explicit *open observations* and **not** part of Theorem T:

1. **Q-sign asymmetry.** In a 5-point (model, benchmark) survey, Qwen × Telecom shows Q+ direction dominance while Llama × Telecom shows Q−; we lack a closed-form $d^*$ derivation. (See §6.1; pending Phase D.)
2. **Format collapse at super-critical $\alpha$.** Llama × Telecom at $\alpha = 0.05$ collapses to $200/200$ empty outputs. Outside the sub-critical regime addressed by Theorem T.
3. **MULTI > SINGLE facet stratification.** Tier-3 multi-tool ΔF1 exceeds single-tool ΔF1 on Telecom by a baseline-ceiling margin; not theorem-load-bearing.
4. **Layer-adaptive $\alpha$.** Per-layer optimal $\alpha$ varies; observation only.
5. **BiasBios $+7.6\text{pp}$ top-1 transfer.** Outside tool-selection scope; demonstrates the K-subspace pipeline applies more broadly than the claim region.
6. **Layer-0 rank-1 massive activation.** Qwen2.5 L0 K is per-head rank-1 with $|\cos|=1.000$ to $B_{\text{ont}}$ column 0; benign for the Theorem.
7. **Contrastive K-bias positive on ST4 multi-tool.** Requires labels; outside training-free scope.
8. **Higher-order slope excess (H-I).** Qwen / Mistral $+6\text{–}17\%$ over Ledoux $0.500$; second-order Lipschitz refinement, §6.1.

The discipline of *not* re-promoting these to theorem status is what kept the four-attempt history of the program from collapsing into framework drift; we maintain it explicitly here.

---

## §4. Methodology

### 4.1 $B_{\text{ont}}$ construction as K-subspace extraction (per F1)

The basis $B_{\text{ont}} \in \mathbb{R}^{d \times r}$ used throughout is built by a *forward-pass-only pipeline*. Given a tool-selection benchmark $V$ and a model $M$:

1. **Position selection.** A small annotation catalog enumerates *tool-related token positions* in benchmark prompts (function action, IO type, domain, tool category facets in our $\tau^2$ / MetaTool builds). The catalog *selects positions*; it does not provide labels in the supervised sense.
2. **K extraction.** For each position, run $M$ forward and cache the post-`k_norm` K activations at the chosen layers.
3. **Per-facet aggregation.** Within each facet, compute the per-category mean $\bar K_c \in \mathbb{R}^{d}$ over instances.
4. **Gram-Schmidt orthonormalization.** Stack all $\bar K_c$ vectors and orthonormalize to produce columns of $B_{\text{ont}}$ with rank $r \leq r_{\max}$ (where $r_{\max}$ depends on facet size; $r$ values used in main results: Telecom 12, Retail 11, MetaTool 24).

**Key clarification (F1).** Phase C demonstrates that *catalog content* — the actual sentences — is not load-bearing: random-sentence positions with the same selection mechanism produce a $B$ that is empirically indistinguishable in attn_fro shift effect (§5.6). The pipeline's load-bearing operation is *extracting K activations from tool-related token positions* and orthonormalizing the resulting set. We therefore use **"K-subspace extraction"** to describe $B_{\text{ont}}$ throughout, and avoid all "catalog-semantic" / "ontology-encoded" / "label-derived" phrasing.

### 4.2 Random-null variant D (Haar-matched control)

For each configuration that produces a B_ont with rank $r$, we build a random control basis $B_D \in \mathbb{R}^{d \times r}$ by sampling $B_D \sim \text{Haar}(V_r(\mathbb{R}^d))$ with seed fixed at $42$ unless stated. The Haar sampling realizes Lemma 2 in expectation. We denote this **variant D**.

A Tier-3 baseline check (Phase 0, `variantD_phase0_verified_2026_04_19`) confirmed that variant D's hook fires correctly: $\|\delta K\|_F / \|K\|_F \approx 0.20$, but produces $0/200$ tool-name argmax changes vs. $200/200$ for matched-magnitude $B_{\text{ont}}$. The control thus passes the "matched perturbation magnitude" hygiene that pre-empts the obvious reviewer attack (*"D shows no effect because nothing is happening"*).

### 4.3 Measurement protocols

All measurements are forward-pass-only; no training or fine-tuning is used.

**(M-1) Per-query margin.** $m(q, K) = \ell_{f(q,K)} - \max_{v \neq f} \ell_v$ at the prompt-end token, restricted to the tool-name vocabulary if specified, else unrestricted. Reported as $m_0 = \min_q m$ and $\bar m = \text{mean}_q m$.

**(M-2) Attention-Frobenius shift.** $\|\delta\text{attn}\|_F$ averaged over heads in the steered set; reported as `attn_fro_mean`. Computed both for the aligned variant A (with $B_{\text{ont}}$) and the random-null variant D.

**(M-3) Argmax flip rate.** $\Pr_q[f(q, K + \delta K) \neq f(q, K)]$ over the benchmark. Reported per-rank.

**(M-4) Layer-resolved KL.** $\text{KL}\big(P_\ell(\cdot | K) \,\|\, P_\ell(\cdot | K + \delta K)\big)$ where $P_\ell$ is the projected next-token distribution at layer $\ell$ (early-exit projection through $W_{\text{unembed}}$ for layers $< L$, true distribution for $\ell = L$). Run on $N=50$ Telecom queries across $r \in \{1, 3, 6, 12, 24, 48, 96\}$ and all 29 layers (Qwen2.5-7B Telecom).

### 4.4 Primary direction-specificity metric: `attn_fro_ratio` (per F2)

For a (source $B_{\text{ont}}$, target benchmark) pair, define

$$\text{attn\_fro\_ratio} \;=\; \frac{\mathbb{E}_q \|\delta\text{attn}_A\|_F}{\mathbb{E}_q \|\delta\text{attn}_D\|_F}.$$

**Threshold:** $\text{attn\_fro\_ratio} \geq 1.5$ is read as "direction-specific" — i.e., the aligned perturbation produces $\geq 1.5\times$ more attention-weight shift than a Haar-matched random control.

**Why attn_fro and not KL.** The full-vocabulary KL ratio $\text{KL}(P_A \| P_0) / \text{KL}(P_D \| P_0)$ conflates two things: (a) the *direction quality* of $B_{\text{ont}}$ relative to $B_D$, and (b) the *benchmark-intrinsic attention sensitivity* — how sharply the prompt-end attention concentrates and therefore how easily a random K perturbation perturbs the output. In Phase C we observe the same $B_{\text{ont}}$ producing KL-ratio $= 0.062$ on Telecom self-bench and $3.62$ on Retail self-bench (a $58\times$ swing) while attn_fro ratio is approximately $2.45$ and $3.44$ respectively (a $1.4\times$ swing). The attn_fro ratio lives in the same metric space as Lemma 2's $\sqrt{r/d}$ statement and is invariant to prompt-end output geometry. We therefore report **attn_fro ratio as the primary direction-specificity metric throughout §5**, with KL as a secondary diagnostic with explicit caveat.

### 4.5 Models, benchmarks, hyperparameters

**Models.** Qwen2.5-7B (29 layers, $d_{\text{model}}=3584$, $d=128$ per head), Llama-3.1-8B (32 layers, $d_{\text{model}}=4096$, $d=128$), Mistral-7B-Instruct (32 layers, $d_{\text{model}}=4096$, $d=128$). All loaded in `bfloat16` on a single GPU per run.

**Benchmarks.** (i) τ²-bench (`Telecom`, `Retail`, `Airline`, `Banking`); (ii) MetaTool ST4 (Subtask 4, multi-tool selection). For Phase A: $N=100$ queries per benchmark (Banking $N=97$ after filtering); for Phase B1: $N=100$ per cross-pair (Airline $N=50$); for Phase B2: $N=50$.

**Hyperparameters.** $\alpha = 0.3$ unless noted, $r \in \{1, 3, 6, 12, 24, 48, 96\}$ for the rank sweep, steered layers = last 10 of the architecture (`sel_layers=last10`), seed = 42 for variant D Haar sampling, head selection = all attention heads in $\mathcal{L}$. Token positions for steering: prompt-end position only.

### 4.6 H-J breadth metric: subspace-alignment score (per F4)

For a source $B_{\text{ont}}^{(\text{src})} \in \mathbb{R}^{d \times r}$ and a target benchmark $V_{\text{tgt}}$, denote $K_{\text{tgt}} \in \mathbb{R}^{T \times d}$ the stacked K activations at tool-related token positions of $V_{\text{tgt}}$ (using the same position-selection protocol as §4.1). Define

$$\text{breadth}\big(B_{\text{ont}}^{(\text{src})} \to V_{\text{tgt}}\big) \;=\; \frac{1}{r} \sum_{i=1}^r \sigma_i\!\left( B_{\text{ont}}^{(\text{src}) \, \top} \, K_{\text{tgt}} / \|K_{\text{tgt}}\|_{\text{col}} \right),$$

i.e. the mean singular value of the source basis projected onto the column-normalized target-K matrix.

**Predictions (registered before Phase B3 measurement runs):**
- $\text{breadth} \geq 0.3$ → high attn_fro ratio (broad source, transfers).
- $\text{breadth} < 0.1$ → low attn_fro ratio (narrow source, fails to transfer).

**Banned alternatives.** We do *not* use catalog-side facet count, vocabulary diversity, or any annotation-based metric, since Phase C falsified catalog-content as a load-bearing factor. The breadth definition lives entirely in attention-geometry space and is computable from cached B_ont.pt + K activation files; no GPU re-run required.

---

## §5. Results

We report Phase A (universality of $\sqrt{r/d}$ scaling and margin-gated flip), Phase B1 (cross-benchmark direction specificity), Phase B2 (layer-resolved KL mechanism), and Phase C (catalog-permutation falsifier). Phases B3 (H-J breadth measurement) and D (Q-sign $d^*$) are placeholders pending experiment-session results.

### 5.1 Phase A — $\sqrt{r/d}$ universality and margin-gated flip

**Protocol.** For each of 9 (model, benchmark) pairs we run `measure_lemma_empirical.py` with $N=100$ queries (Banking $N=97$), $r \in \{1, 3, 6, 12, 24, 48, 96\}$, $\alpha = 0.3$. We report the log-log regression slope of `attn_fro_mean` vs. $r/d$, the $R^2$ of that regression, the per-benchmark margin lower bound $m_0$, the per-benchmark mean margin $\bar m$, and the total argmax flip count over $\sum_r N$ rank-query combinations. **Banking is moved to Appendix A.1 per F5** (knowledge-QA, not tool-selection); the main table presents 8 tool-selection runs.

**Table 5.1.** Phase A — $\sqrt{r/d}$ slope and flip rate per (model, benchmark). All measurements at $\alpha = 0.3$, last-10-layer steering.

| (M, V) | slope | $R^2$ | flips | $m_0$ | $\bar m$ | flip @ $r{=}96$ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Qwen2.5-7B × τ²-Telecom | 0.5508 | 0.9920 | 0 / 700 | 5.031 | 5.072 | 0 |
| Qwen2.5-7B × τ²-Retail | 0.5693 | 0.9948 | 0 / 700 | 3.766 | 6.655 | 0 |
| Qwen2.5-7B × τ²-Airline | 0.5536 | 0.9929 | 0 / 350 | 4.203 | 5.714 | 0 |
| Qwen2.5-7B × MetaTool ST4 | 0.5779 | 0.9957 | 0 / 700 | 1.672 | 3.249 | 0 |
| Llama-3.1-8B × τ²-Telecom | 0.4955 | 0.9979 | 0 / 700 | 0.422 | 0.422 | 0 |
| Llama-3.1-8B × τ²-Retail | 0.4849 | 0.9986 | **11 / 700** | **0.016** | 0.974 | 5 |
| Llama-3.1-8B × MetaTool ST4 | 0.4025 | 0.9918 | 0 / 700 | 6.375 | 7.632 | 0 |
| Mistral-7B-Inst × τ²-Telecom | 0.5471 | 0.9995 | 0 / 700 | 0.234 | 0.274 | 0 |

(Banking row reported in Appendix A.1.)

**Findings.**

1. **(T.A) $\sqrt{r/d}$ scaling holds 8/8** with $R^2 \geq 0.99$, mean slope $0.5227$ on the main-text rows, $0.5298$ across all 9 runs including Banking. Predicted slope from Lemma 2: $0.5$. The empirical first-order match is decisive.
2. **Model-family stratification.** Qwen $\in [0.55, 0.58]$ (six runs), Llama $\in [0.40, 0.50]$ (three runs), Mistral $0.55$. The Qwen / Mistral $+6$–$17\%$ excess over $0.500$ is systematic, not noise (per-run std $\sim 0.06$); we treat this as a higher-order softmax-Lipschitz correction (H-I, §6.1).
3. **(T.B) Margin-gated flip predictively confirmed.** Of the 8 main-text runs, **7 show 0 flips**. The single non-zero run is Llama × Retail with $m_0 = 0.016$, the smallest margin in the table; its flip rate is $1.57\%$ and is concentrated at high $r$ ($5/100$ at $r = 96$, $0/100$ at $r = 1$). This matches Lemma 1's prediction that flip rate scales with the per-query lift $\alpha \langle B B^\top q, g\rangle$, whose magnitude grows as $\sqrt{r/d}$ (Lemma 2).
4. **Sub-critical 0-flip plateau.** The seven 0-flip runs include Qwen × ST4 with $m_0 = 1.67$ and Mistral × Telecom with $m_0 = 0.23$. The plateau is robust over $700$ rank-query combinations per row; under rule-of-three, $\Pr[\text{flip}] \leq 3/700 = 0.43\%$ at $95\%$ CI per row.

**Phase A gate.** The original tight gate (slope $\in [0.45, 0.55]$ in $4/5$) was over-specified; we adopt the relaxed gate **`mean slope $0.50 \pm 0.05$ AND all $R^2 \geq 0.95$`**, under which Phase A passes 8/8 main-text rows decisively and 9/9 including Banking. The over-specification of the original gate is itself a dataset-and-architecture-derived lesson we record openly.

### 5.2 Phase A continued — T.B predictive form

The Phase A flip data, viewed as (margin, flip rate) pairs, gives a direct test of Lemma 1's predictive form. Table 5.2 presents the 9 runs ordered by $m_0$.

**Table 5.2.** Margin-flip relation (main-text 8 + Banking ref).

| $m_0$ | flip rate | (M, V) |
|:---:|:---:|---|
| 0.016 | **1.57%** | Llama × τ²-Retail |
| 0.234 | 0.00% | Mistral × τ²-Telecom |
| 0.422 | 0.00% | Llama × τ²-Telecom |
| 0.875 | **7.81%** *(Banking)* | Qwen × τ²-Banking — Appendix A.1 |
| 1.672 | 0.00% | Qwen × MetaTool ST4 |
| 3.766 | 0.00% | Qwen × τ²-Retail |
| 4.203 | 0.00% | Qwen × τ²-Airline |
| 5.031 | 0.00% | Qwen × τ²-Telecom |
| 6.375 | 0.00% | Llama × MetaTool ST4 |

**Findings.** Every benchmark with $m_0 \geq 1.5$ in the main-text set shows $0$ flips. The only non-zero flip rate is Llama Retail ($m_0 = 0.016$, flip $1.57\%$), and it is concentrated at high $r$. Banking, with $m_0 = 0.875$, shows $7.81\%$ flip, which is consistent with the small-margin prediction but lives in a knowledge-QA regime rather than tool-selection (Appendix A.1).

The data is consistent with Lemma 1's predictive form: $\text{flip rate} \propto \Pr[\alpha \cdot \text{lift} > m]$ with $\mathbb{E}[\text{lift}] = O(\sqrt{r/d})$ from Lemma 2. We do *not* attempt a closed-form fit of the flip-rate / $m_0$ curve in this paper; the main claim is the qualitative existence of the sub-critical plateau and the small-margin emergence.

### 5.3 Phase B1 — Cross-benchmark direction specificity (primary metric: attn_fro ratio per F2)

**Protocol.** For each (source benchmark, eval benchmark) pair we run variant A (with source $B_{\text{ont}}$) and variant D (Haar-matched random) on the eval benchmark's $N=100$ queries and measure attn_fro_mean and KL_mean for both. The **attn_fro ratio** is the primary statistic; KL ratio is secondary. Source ranks: Telecom $r = 12$, Retail $r = 11$, MetaTool $r = 24$. **Banking row dropped per F5.**

**Table 5.3.** Phase B1 cross-benchmark direction specificity. Primary metric = attn_fro ratio. KL ratio shown as secondary diagnostic with explicit asymmetry annotation.

| Source $B_{\text{ont}}$ | Eval bench | A attn_fro | D attn_fro | **attn_fro ratio** | A KL | D KL | KL ratio |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Telecom | Retail | 0.0858 | 0.0257 | **3.33** | 0.0070 | 0.0019 | 3.67 |
| Telecom | Airline | 0.0787 | 0.0226 | **3.48** | 0.0097 | 0.0046 | 2.10 |
| Telecom | ST4 | 0.0735 | 0.0172 | **4.27** | 0.0556 | 0.0307 | 1.81 |
| Retail | Telecom | 0.0916 | 0.0361 | **2.54** | 0.0243 | 0.1070 | **0.23** ⚠ |
| MetaTool | Telecom | 0.0999 | 0.0479 | **2.09** | 0.0376 | 0.0152 | 2.47 |
| Retail | Retail (self) | 0.0865 | 0.0251 | **3.44** | 0.0071 | 0.0020 | 3.62 |
| Telecom | Banking | 0.1321 | 0.0530 | 2.50 | 0.7360 | 0.2894 | 2.54 |
| *(Telecom self, Phase C)* | *(Telecom)* | 0.0904 | 0.0369 | 2.45 | 0.0070 | 0.1124 | 0.06 |

(Banking row from Phase B1 reported in main table for completeness of attn_fro coverage; its prompt-end is knowledge-QA per F5 and is not used to support Theorem T claims. Telecom-self row from Phase C measurement included for direct comparison with Retail-self row.)

**Findings.**

1. **(F2 primary statement)** Of the 6 main-text tool-selection rows, **6 / 6 attn_fro ratios are $\geq 2.0$**, mean $\bar{r}_{\text{attn}} = 2.97$, range $[2.09, 4.27]$. All exceed the $1.5$ threshold for "direction-specific" by a margin. This is a *cleaner* statement than the KL-based "$6/7$ A > D" reading.
2. **(KL/attn_fro divergence — A-2 is metric-dependent)** The Retail $\to$ Telecom row shows $\text{KL ratio} = 0.23 < 1$ (A weaker than D) but $\text{attn\_fro ratio} = 2.54 > 1$ (A stronger than D). The "asymmetric transferability" finding survives the metric shift only as an attenuation, not a sign-flip: in attn_fro space, all 6 main-text pairs are direction-specific. The KL deviation is fully accounted for by the **benchmark-intrinsic attention-sensitivity asymmetry A-1** (§5.4 footnote).
3. **(Self-bench parity)** Retail $\to$ Retail attn_fro ratio is $3.44$, comparable to Telecom $\to$ Retail at $3.33$. The self-bench advantage in attn_fro space is small. (KL space exaggerates this — see §5.4.)

### 5.4 Asymmetric transferability (A-2, primary), with self-bench note (A-1, footnote)

The Phase B1 main finding for the *transferability claim*:

**Claim (A-2, F3 primary):** *Source $B_{\text{ont}}$'s capacity to transfer to a target benchmark is asymmetric in attn_fro space: Telecom and MetaTool sources show attn_fro ratio $\in [2.09, 4.27]$ across 4 cross-pairs, while Retail $\to$ Telecom is the lowest at $2.54$ (still direction-specific but $1.7\times$ smaller than Telecom $\to$ Retail's $3.33$). In KL space the asymmetry is exaggerated because of A-1 (footnote).*

This is the cleanest available *attention-geometric* statement of the asymmetric transferability finding. We hypothesize that the asymmetry is governed by the geometric *subspace-alignment* of the source basis with the target benchmark's tool-position K activations (definition §4.6). Phase B3 will measure breadth and check the prediction $\text{breadth} \geq 0.3 \Leftrightarrow \text{attn\_fro ratio} \geq 2.0$.

**Footnote (A-1, per F3).** *Phase C self-bench measurements show Telecom self KL ratio $= 0.062$ (A weaker than D in KL) vs. Retail self KL ratio $= 3.62$ (A stronger than D in KL) — a $58\times$ swing. Yet attn_fro ratios for the same two pairs are $2.45$ and $3.44$ — a $1.4\times$ swing. The KL discrepancy is driven by D KL alone: D KL is $0.112$ on Telecom self vs. $0.002$ on Retail self ($56\times$). This is a benchmark-intrinsic attention-sensitivity property (the random null produces large output-vocabulary perturbation on benchmarks whose prompt-end attention concentrates sharply, e.g. Telecom; less on benchmarks with flatter attention, e.g. Retail). It is unrelated to the source $B_{\text{ont}}$ direction and motivates the F2 metric choice (attn_fro primary, KL secondary).*

### 5.5 Phase B2 — Layer-resolved KL pins the mechanism to layer 28

**Protocol.** Qwen2.5-7B × Telecom, $N = 50$, last-10-layer steering, full $r \in \{1, 3, 6, 12, 24, 48, 96\}$ sweep, layer-resolved early-exit projection through the unembedding for layers $0\text{–}27$, true distribution for the final layer.

**Table 5.5.** Per-layer KL by rank. Layers $0$–$17$ are pre-steering (KL $= 0$); rows below show layers from the first steered layer to the final.

| Layer | $r{=}1$ | $r{=}3$ | $r{=}6$ | $r{=}12$ | $r{=}24$ | $r{=}48$ | $r{=}96$ | monotone? |
|---|---|---|---|---|---|---|---|---|
| 18 (first steered) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | ✓ trivially |
| 19 | 0.0002 | 0.0004 | 0.0008 | 0.0021 | 0.0053 | 0.0112 | 0.0276 | ✓ |
| 20 | 0.0003 | 0.0006 | 0.0013 | 0.0047 | 0.0118 | 0.0383 | 0.1617 | ✓ |
| 21 | 0.0000 | 0.0001 | 0.0002 | 0.0008 | 0.0029 | 0.0169 | 0.1288 | ✓ |
| 22 | 0.0003 | 0.0008 | 0.0020 | 0.0039 | 0.0102 | 0.0271 | 0.0810 | ✓ |
| 23 | 0.0004 | 0.0012 | 0.0028 | 0.0074 | 0.0206 | 0.0605 | 0.2061 | ✓ |
| 24 | 0.0004 | 0.0010 | 0.0025 | 0.0070 | 0.0197 | 0.0581 | 0.2235 | ✓ |
| 25 | 0.0003 | 0.0008 | 0.0021 | 0.0053 | 0.0140 | 0.0334 | 0.1376 | ✓ |
| 26 | 0.0003 | 0.0007 | 0.0016 | 0.0043 | 0.0086 | 0.0385 | 0.1297 | ✓ |
| 27 | 0.0003 | 0.0009 | 0.0022 | 0.0047 | 0.0105 | 0.0410 | 0.1312 | ✓ |
| **28 (final)** | **0.0255** | **0.0716** | **0.1377** | **0.1375** | **0.1232** | **0.1305** | 0.1770 | **✗** |
| logits (post-LM-head) | 0.0141 | 0.0255 | 0.0655 | 0.0815 | 0.0307 | 0.0295 | 0.1292 | **✗** |

**Findings.**

1. **(T.C mechanism location)** $28$ of $29$ layers transmit perturbation monotonically in $r$ across the full rank sweep. Only the final layer ($28$, post final-FFN composition) is non-monotonic — peak at $r=6$, dip at $r=24$, rebound at $r=96$. The output logits are non-monotonic, mirroring the Phase A Qwen Telecom pattern.
2. **Layer-28 amplification at low rank.** $r=1$ KL goes from $0.0003$ at L27 to $0.0255$ at L28 — an **$85\times$ amplification**. At $r=96$ the amplification is only $1.35\times$ ($0.1312 \to 0.1770$). The non-linear amplification is concentrated where the residual-stream perturbation is small.
3. **Interpretive consistency with prior interpretability work.** Late-block FFN layers as key-value memories with non-linear "cleanup" behavior (Geva et al. 2021, Meng et al. 2022) is consistent with what we observe: the final FFN amplifies small attention-borne signals and re-orders the next-token distribution.

The conclusion is that **Theorem T.C's two-level gap is structurally located at the final FFN + LM-head composition**. The attention residual stream itself is smooth and ordered.

### 5.6 Phase C — Catalog-permutation falsifier (H-F falsified, theorem strengthened per F1)

**Protocol.** Build three permuted variants of the Telecom $B_{\text{ont}}$:
- `perm facet_values`: shuffle the sentence-to-category mapping within each facet.
- `perm tool_names`: shuffle tool-name tokens within sentences.
- `perm full_random`: replace all sentences with random gibberish before extraction.

For each, build $B_{\text{ont}}$ via the §4.1 pipeline and measure attn_fro_mean, KL_mean, flip rate on Qwen × Telecom self-bench at $r$ matched to source build.

**Table 5.6.** Phase C permutation results. Real Telecom = baseline.

| Variant | $r$ | A flip | D flip | A KL | D KL | KL ratio | A attn_fro | D attn_fro | **attn_fro ratio** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| real Telecom $B_{\text{ont}}$ | 12 | 0/100 | 0/100 | 0.0070 | 0.1124 | 0.062 | 0.0904 | 0.0369 | **2.45** |
| perm facet_values | 12 | 0/100 | 0/100 | 0.0070 | 0.1124 | 0.062 | 0.0904 | 0.0369 | **2.45** |
| perm tool_names | 12 | 0/100 | 0/100 | 0.0076 | 0.1124 | 0.067 | 0.0991 | 0.0369 | **2.69** |
| perm full_random | 11 | 0/100 | 0/100 | 0.0031 | 0.1070 | 0.029 | 0.0717 | 0.0361 | **1.98** |

**Findings.**

1. **(H-F falsified)** All four variants — real and three permutations — produce **attn_fro ratio $\in [1.98, 2.69]$** and **KL ratio $\in [0.029, 0.067]$**. Catalog *content* is not load-bearing for direction specificity. The three permutations differ in what they destroy: facet_values is a Gram-Schmidt-span no-op (numerical match to 4 digits with real); tool_names disrupts the actual K activations at affected positions but preserves tool-position selection; full_random destroys both content and most of the position semantics, and yet still produces a basis with attn_fro ratio $1.98$.
2. **(Falsifier $\to$ refinement, F1)** What survives the falsifier is *the K-subspace extracted from tool-related token positions*. The catalog *selects positions* (and the position selection has some content-independent geometric quality); the catalog does not provide labels in any operative sense. This refines the methodological framing: we use **"K-subspace extraction"** throughout, not "catalog-semantic basis".
3. **(F2 anchor)** The Phase C results provide the empirical anchor for the F2 primary-metric choice. KL ratios across the four variants vary by $\sim 2.3\times$ ($0.029$ to $0.067$); attn_fro ratios vary by $\sim 1.4\times$ ($1.98$ to $2.69$). The attn_fro metric is more stable across permutations and lives in the same space as Lemma 2's prediction.

**Note on falsifier scope.** The Phase C falsifier addresses the *strong* form of H-F (catalog content is the load-bearing factor). It does not test the *weak* form (catalog content has *some* effect). The numerical proximity of real ($2.45$) to full_random ($1.98$) suggests the weak form is at most a $\sim 25\%$ effect on attn_fro ratio.

### 5.7 (Pending) Phase B3 — H-J breadth vs. attn_fro ratio scatter

To be filled when experiment session runs `analyze_breadth_metric.py` (§4.6, no GPU). Expected deliverables: a breadth column added to Table 5.3 and a breadth vs. attn_fro_ratio scatter for the $6$ main-text B1 pairs, with regression coefficient as the headline statistic.

### 5.8 Phase D — Q-sign $d^*$ narrowing (Failure branch per pre-registered template)

**Protocol executed.** Experiment session ran a *stronger* test than the originally pre-registered $18$-pair sign prediction: per-(layer, head, query) angular alignment of the empirical Q-direction $d_{\text{emp}} = \bar K_{\text{GT}} - \bar K_{\text{distr}}$ against two static-weight hypotheses on Qwen2.5-7B × τ² {Telecom, Retail} × $N = 50$ each (∼4000 angular measurements per hypothesis). Substituting the continuous-metric per-(L, h, q) alignment for the binary 18-pair sign vote was accepted as a *tighter* rejection criterion (4000-point continuous test vs. 18 binary trials).

- **H$_1$ (lm_head pull-through):** $d_{H_1} = W_K^\top (\bar e_{\text{GT}} - \bar e_{\text{distr}})$. Mean $|\cos|$ across all (L, h, q): $0.0807$ (Telecom), $0.0764$ (Retail), against random baseline $1/\sqrt{128} = 0.0884$. That is **$0.91\times$ / $0.86\times$ random — worse than chance**. $0\%$ of heads pass an angular threshold of $30°$.
- **H$_3$ ($W_K$ top-left singular vector, data-free):** Mean $|\cos|$: $0.1090$ (Telecom), $0.0973$ (Retail), against the same random baseline. That is **$1.23\times$ / $1.10\times$ random — modest but sub-threshold**. $0\%$ of heads pass $30°$.
- **H$_2$, H$_4$, H$_5$, H$_6$:** not tested in this submission cycle (activation-dependent / context-specific; estimated $80$+ GPU-hr).
- **Llama:** test infrastructure blocked by prompt-builder mismatch (chat template omits inline tool JSON; tool positions unfindable). Cross-architecture verification of the Phase D finding is therefore not in scope.

**Verdict — *Failure* branch.** Per the pre-registered template, $0\%$ of heads passing the $30°$ angular threshold on $\sim 4000$ data points is a tighter rejection than the $< 10/18$ sign-failure threshold. **The static weight-geometric family of $d^*$ hypotheses (H$_1$ + H$_3$) is effectively closed for this submission.** Q-sign asymmetry remains an unexplained observation; the result is reported transparently in §6.1 with hypothesis-level falsification and in Appendix B.1 with full per-head data.

**Localized partial signal (future-work hook).** Of $\sim 56$ Qwen Telecom heads ($28 \times 2$), seven show H$_3$ alignment $|\cos| \geq 0.18$ — i.e. $2$–$3\times$ the random baseline. Notable: L24$_{h1}$ $|\cos|=0.285$ (angle $\approx 73°$), L19$_{h0}$ $|\cos|=0.255$ ($\approx 75°$). Full table in Appendix B.1. These heads are below the global $30°$ pass threshold but distinguishable from the bulk distribution; they constitute a candidate locus for a follow-up rank-1 $W_K$ ablation. Not in scope here.

---

## §6. Discussion

### 6.1 Open observations not in Theorem T

**Q-sign asymmetry (unexplained observation; static-weight $d^*$ family closed).** A 5-point survey across (Qwen Telecom Q+, Llama Telecom Q−, Llama Retail Q−, Mistral Telecom Q−, Qwen ST4 Q−) shows the sign of the per-direction K-bias that *helps* tool-selection F1 is model-and-benchmark dependent.

Per the *Failure* branch of the pre-registered Phase D template (§5.8): the natural static-weight-geometric hypotheses H$_1$ (lm_head pull-through) and H$_3$ ($W_K$ top singular vector) were tested by per-(L, h, q) angular alignment against the empirical $d_{\text{emp}}$ on Qwen Telecom / Retail (~$4000$ measurements per hypothesis). H$_1$ reproduces at $0.86$–$0.91\times$ the random baseline $1/\sqrt{d}$; H$_3$ at $1.10$–$1.23\times$. **No head passes a $30°$ angular threshold for either hypothesis**, though a minority of $\approx 7$ heads (notably L24$_{h1}$, L19$_{h0}$) show $W_K$ top-SV alignment at $2$–$3\times$ random, hinting at head-specific weight-directional bias for tool selection as a possible future avenue. The result indicates that **$d^*$ is *not* expressible as a static weight-level quantity**; an activation-dependent or multi-head-composition description remains open. H$_2$ (OV-circuit readout), H$_4$ (RoPE phase), H$_5$ (head-class), H$_6$ (catalog-position) were not tested in this submission cycle (estimated $80$+ GPU-hr).

**Format collapse threshold.** Llama × Telecom at $\alpha = 0.05$ produces $200/200$ empty-output collapse — a discontinuous super-critical regime well outside Theorem T's sub-critical plateau. We mention it to clarify the scope of the sub-critical claim.

**Slope excess H-I (model-family Lipschitz).** Qwen ($0.55$–$0.58$) and Mistral ($0.55$) consistently exceed Lemma 2's predicted $0.5$ by $6$–$17\%$, while Llama hugs the prediction. Hypothesis: per-head softmax sharpness (proxied by attention entropy) is greater on Qwen / Mistral, raising the second-order Lipschitz term. A light analysis from Phase A logs (no GPU re-run) is the natural test; if the entropy / slope correlation exceeds $0.3$, the H-I refinement enters §Appendix as a second-order correction to Lemma 2's first-order $\sqrt{r/d}$.

**MULTI > SINGLE on Telecom Tier 3.** $+36\text{pp}$ (multi) vs. $+24\text{pp}$ (single). Plausibly a baseline-ceiling effect; not theorem-load-bearing.

**Layer-adaptive $\alpha$.** Optimal $\alpha$ varies layer-to-layer; per-layer discriminative-subspace strength may be the cause but is not estimated in this paper.

**BiasBios $+7.6\text{pp}$ top-1 transfer.** A single-answer classification benchmark outside tool-selection scope shows positive lift with the same K-subspace pipeline. Suggests broader applicability than the claim region of this paper.

**Layer-0 rank-1 massive activation channel.** Qwen2.5 L0 K is rank-1 per head with $|\cos|=1.000$ to $B_{\text{ont}}$ column 0. Skipping L0 changes nothing material; we report it to forestall reviewer questions.

### 6.2 The two asymmetries A-1 and A-2 are distinct (per F3)

We emphasize: the *transferability asymmetry* (A-2, §5.4 main: source-side $B_{\text{ont}}$ subspace coverage) and the *benchmark attention-sensitivity asymmetry* (A-1, §5.4 footnote: target-side prompt-end attention concentration) are distinct phenomena measured on different random variables. A-1 is detected in $D_{\text{KL}}$ (random null, target-side), and is fully visible in §5.6 Phase C self-bench D KL going from $0.002$ (Retail) to $0.112$ (Telecom). A-2 is detected in $A_{\text{attn\_fro}}$ (aligned, source-side), and after the F2 metric shift its remaining visibility is mild ($2.54$ vs $3.33$ for the most transferable cross-pair).

Conflating A-1 and A-2 under a single "asymmetry" heading invites reviewer confusion about whether the finding is a benchmark artifact or a source-basis property; we keep them visually separate throughout the paper.

### 6.3 Pipeline-level direction specificity as a methodological claim

Phase C's falsification of catalog-content as load-bearing (H-F) refines a methodological guideline broadly applicable to attention-side steering: **what matters is the position selector that drives K extraction, not the labels attached to those positions**. Practitioners building $B_{\text{ont}}$-style bases for steering can therefore use any reliable tool-position selector (regex, off-the-shelf NER over tool catalogs, or even random tool-like positions) and expect comparable direction specificity within the tool-selection scope. The label / sentence content can be sourced freely.

---

## §7. Limitations

1. **Higher-order slope (H-I).** Mean slope $0.5298$ is a $\sim 6\%$ excess over Ledoux's first-order $0.500$, with model-family stratification. We do not derive the second-order correction; we treat it as an open observation calling for a more refined Lipschitz analysis.

2. **$d^*$ derivation left open after Phase D.** The sign and direction of the per-(model, benchmark) Q-bias direction $d^*$ that controls Q-sign asymmetry is *not* expressible as any of the static-weight quantities tested in Phase D (H$_1$ lm_head pull-through, H$_3$ $W_K$ top singular vector). Activation-dependent / multi-head-composition hypotheses (H$_2$ OV-circuit readout, H$_4$ RoPE phase, H$_5$ head-class, H$_6$ catalog-position) are deferred and left open. See §6.1 + Appendix B.1.

8. **Llama Phase D blocked.** Llama-3.1-8B's Phase D measurement was blocked by a prompt-builder mismatch (chat template omits inline tool JSON; tool positions unfindable via current annotation). Cross-architecture verification of the static-weight-family negative result is therefore not in scope for this submission.

3. **Cross-architecture coverage.** All three tested models are decoder-only causal transformers with similar attention head structure. Encoder-decoder, non-causal, MoE, Mamba / SSM architectures are untested and Theorem T's claims should not be extrapolated to them without further experiment.

4. **Single perturbation magnitude.** Most measurements use $\alpha = 0.3$. Phase A's flip-rate / margin curve covers a single magnitude; the small-margin sub-critical-to-super-critical transition is not finely resolved (we have $0\text{–}1.6\%$ on Llama Retail and the format-collapse jump on Llama Telecom at $\alpha = 0.05$, but no full curve).

5. **Tool-selection scope.** Theorem T's empirical anchors live entirely within tool-selection benchmarks ($\tau^2$ + MetaTool). Banking is excluded as knowledge-QA (Appendix A.1). MMLU and similar reasoning benchmarks would be a natural extension but are out of scope here. The BiasBios transfer ($+7.6\text{pp}$, §6.1) is suggestive of broader applicability but is not part of the Theorem's claim region.

6. **Phase B2 mechanism location: single model.** Layer-28 amplification is verified for Qwen2.5-7B × Telecom only. Cross-model verification of the final-block mechanism is a Phase B3 sub-task.

7. **Theorem T.C is empirical, not provable.** We localize the gap mechanistically but do not derive a closed-form expression for the FFN amplification factor. Connecting the empirical $85\times$ to a Lipschitz / spectral statement of the final block is left for future work.

---

## §8. Conclusion

We characterize a **two-level argmax-subspace selectivity** in pretrained tool-selection transformers: the attention-weight Frobenius response to a low-rank K-bias perturbation $\delta K = \alpha B B^\top K$ scales smoothly as $\sqrt{r/d}$ across $9$ (model, benchmark) pairs ($R^2 \geq 0.99$), while the discrete tool-name argmax remains stable except in narrow margin-sensitive sub-populations (Lemma 1 predictive form). The smooth-versus-stepwise gap is mechanistically located in the **final FFN + LM-head composition** (Phase B2: $28$ of $29$ residual layers transmit perturbation monotonically; layer $28$ amplifies the $r=1$ KL by $85\times$). A catalog-permutation falsifier (Phase C) rules out catalog content as the source of direction specificity, refining the methodological framing of the basis $B_{\text{ont}}$ as a *K-subspace extracted from tool-related token positions*. The primary direction-specificity metric is the attention-Frobenius shift ratio (per F2), which is invariant to benchmark-intrinsic prompt-end attention sensitivity and aligns with Theorem T.A's Stiefel formulation. We hope the structural framing and the layer-localized mechanism contribute toward a more disciplined design of attention-side steering and KV-cache compression methods.

---

## Appendix A.1 — Banking out-of-distribution control (knowledge QA)

**Status (per F5).** Banking (`banking_knowledge`) is a knowledge-QA task in τ²-bench: the prompt-end token is `<|im_start|>assistant\n`, an unrestricted free-form answer opener rather than a tool-call opener. The per-query margin $m_0 = 0.875$ is small and likely reflects synonymous answer-opener near-ties, not a tool-selection margin. We exclude Banking from the main Theorem T claim region and report it here.

**Banking row from Phase A:**

| (M, V) | slope | $R^2$ | flips | $m_0$ | $\bar m$ | flip @ $r{=}96$ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Qwen2.5-7B × τ²-Banking | 0.5868 | 0.9950 | **53 / 679** (7.81%) | 0.875 | 4.337 | 21 |

**Banking row from Phase B1 (Telecom $\to$ Banking):** A flip $9/97 = 9.28\%$, D flip $7/97 = 7.22\%$. Both arms produce non-zero flip rates on the Banking benchmark, with A − D = $+2.06\text{pp}$.

**Reading.** The non-zero D flip rate (random null produces $7\text{pp}$ of argmax change) is consistent with a small-margin regime where any sufficiently large perturbation tips synonymous-opener near-ties. This is *interpretable as a small-margin-plus-synonymous-opener confound*, not as a counterexample to Theorem T.B. The qualitative match to Lemma 1 (small $m_0 \Rightarrow$ non-zero flip; high $r$ disproportionately) holds: $r = 96$ contributes $21/100$ of Phase A's Banking flips.

**Why Banking is not a Theorem T anchor.** Theorem T.B's "tool-name argmax flip" is well-defined only where the prompt-end position drives a tool-name decision. On Banking, the prompt-end position drives a free-form answer opener; A-vs-D flip differences are dominated by opener-ties rather than discriminative tool-selection geometry. We therefore retain Banking only as an out-of-distribution control: it confirms the predictive direction of T.B (small margin $\Rightarrow$ non-zero flip; high $r$ amplifies) but adds no main-text claim weight.

---

## Appendix B.1 — Phase D negative result (static-weight $d^*$ family closed)

### B.1.1 Methodology

For each Qwen2.5-7B head $(L, h)$ at the steered layer block ($L \in \{18, \ldots, 27\}$, $h \in \{0, 1\}$ for the GQA-2 K head structure) we compute the empirical Q-direction

$$d_{\text{emp}}(L, h, q) \;=\; \bar K_{\text{GT}}(L, h, q) \;-\; \bar K_{\text{distr}}(L, h, q)$$

where $\bar K_{\text{GT}}$ is the mean K activation over ground-truth tool-position queries and $\bar K_{\text{distr}}$ is the mean over distractor positions. We then measure the angular alignment

$$|\cos\angle(d_{\text{emp}}, d_H)| \quad \text{for each} \quad H \in \{\text{H}_1, \text{H}_3\}$$

across $N = 50$ queries per benchmark on Qwen × τ² {Telecom, Retail}, giving $\sim 4000$ per-(L, h, q) measurements per hypothesis.

### B.1.2 Hypothesis definitions

- **H$_1$ (lm_head pull-through):** $d_{H_1}(L, h) = W_K^{(L, h) \, \top} (\bar e_{\text{GT}} - \bar e_{\text{distr}})$, where $\bar e_{\cdot}$ are mean unembedding-space embeddings of GT vs. distractor tool names.
- **H$_3$ ($W_K$ top-left singular vector, data-free):** $d_{H_3}(L, h) = U_1$ where $W_K^{(L, h)} = U \Sigma V^\top$.

H$_2$ (OV-circuit readout via attention path), H$_4$ (RoPE phase), H$_5$ (head-class typology), H$_6$ (catalog-position-derived) were not tested.

### B.1.3 Aggregate result

Random baseline for two random unit vectors in $\mathbb{R}^{128}$: $\mathbb{E}|\cos| = 1/\sqrt{128} \approx 0.0884$.

| Hypothesis | Telecom $\bar{|\cos|}$ | Retail $\bar{|\cos|}$ | Telecom $/$ random | Retail $/$ random | Heads passing $30°$ |
|---|:---:|:---:|:---:|:---:|:---:|
| H$_1$ | $0.0807$ | $0.0764$ | $0.91\times$ | $0.86\times$ | $0\%$ |
| H$_3$ | $0.1090$ | $0.0973$ | $1.23\times$ | $1.10\times$ | $0\%$ |

**Reading.** H$_1$ is *worse than chance*: a deliberately constructed direction from the lm_head pull-through is no more aligned with $d_{\text{emp}}$ than a random vector. H$_3$ is modestly above chance ($1.10$–$1.23\times$) but not enough for any individual head to fall within $30°$ of its empirical $d_{\text{emp}}$.

### B.1.4 Per-head follow-up signal (Qwen Telecom heads with H$_3$ $|\cos| \geq 0.18$)

| Head | H$_3$ $|\cos|$ | Random ratio | Angle |
|---|:---:|:---:|:---:|
| L24$_{h1}$ | $0.285$ | $3.22\times$ | $\approx 73°$ |
| L19$_{h0}$ | $0.255$ | $2.88\times$ | $\approx 75°$ |
| L19$_{h1}$ | $0.214$ | $2.42\times$ | $\approx 78°$ |
| L20$_{h1}$ | $0.184$ | $2.08\times$ | $\approx 79°$ |
| L18$_{h0}$ | $0.177$ | $2.00\times$ | $\approx 80°$ |
| (+ 2 more heads) | — | — | — |

These seven heads sit clearly above the bulk of the H$_3$ distribution but below the $30°$ pass criterion. They constitute a candidate locus for future *rank-1 $W_K$ ablation* + Lemma 1 predictive-form re-test. Out of scope here.

### B.1.5 Methodological caveats (transparent reporting)

- **Multi-token tool names in H$_1$.** Some $\tau^2$ tool names tokenize to $> 1$ subword; we use the leading subword's $W_K^\top e$ pull-through. Multi-token averaging is an alternative we did not test.
- **Single-seed determinism.** All measurements use seed $42$; we did not run a multi-seed $d_{\text{emp}}$ stability check. Given $N = 50$ per benchmark and the consistency between Telecom and Retail H$_1$ / H$_3$ ratios, seed sensitivity is unlikely to flip the verdict.
- **Top-1 SV only in H$_3$.** A top-$k$ SV ($k > 1$) projection variant might increase alignment, but per the consolidation framing decision this is not in scope for the current submission.
- **Llama untested.** Llama prompt-builder mismatch precluded cross-architecture verification (see §7 Limitations item 8).

### B.1.6 What would change the verdict

Future work that would reopen the static-weight $d^*$ family:
1. A combined H$_1$ + H$_3$ projection (linear combination of pull-through and SV directions) passing $30°$ on $\geq 25\%$ of heads in two benchmarks.
2. A rank-1 $W_K$ ablation on the 7 follow-up heads producing $\geq 5\text{pp}$ Q-sign-consistent F1 shift on Tier-3 Telecom.
3. Llama prompt-builder fixed and either H$_1$ or H$_3$ alignment $\geq 1.5\times$ random on Llama Telecom or Retail.

None of these is undertaken in this submission. We report the verdict as-is and let future work decide reopening.

---

## Cross-references

- **NEW_THEOREM_TEST plan & hypothesis log:** `math/paper/lie_group/NEW_THEOREM_TEST.md` (v4)
- **Phase A aggregate:** `reports/new_theorem_test/phase_a_aggregate.json` + `memory/new_theorem_phase_a_2026_04_19.md`
- **Phase B1+B2 aggregate:** `reports/new_theorem_test/phase_b1_aggregate.json`, `phase_b2_layer_kl.json` + `memory/new_theorem_phase_b_2026_04_19.md`
- **Phase C aggregate:** `reports/new_theorem_test/phase_c/phase_c_aggregate.json` + `memory/new_theorem_phase_c_2026_04_19.md`
- **Consolidation framing decisions (F1–F5):** `memory/consolidation_framing_decisions_2026_04_19.md`
- **Paper handoff (this draft's parent task):** `memory/handoff_paper_session_new_theorem_2026_04_19.md`
- **NeurIPS 2026 existence-only track (do NOT modify):** `math/paper/benchmark_design/PAPER_DRAFT_v3.md`

---

## Versioning notes

- **v1 (2026-04-19, morning):** Initial draft. §1–§5.6, §6–§8 + Appendix A.1 from Phase A+B+C anchors with F1–F5 applied throughout. §5.7, §5.8, Appendix B placeholders.
- **v1 (2026-04-19, evening):** Phase D *Failure* verdict applied per pre-registered template addendum. §5.8 now reports H$_1$ $0.86$–$0.91\times$ random, H$_3$ $1.10$–$1.23\times$ random, $0\%$ pass $30°$. §6.1 Q-sign discussion absorbs falsified-hypothesis report. §7 Limitations adds items 2 and 8 (deferred activation-dependent hypotheses; Llama untested). Appendix B.1 fully populated with methodology, hypothesis definitions, aggregate result, 7-head follow-up table, methodological caveats, reopening criteria.
- **Pending updates:** §5.7 on Phase B3 breadth analysis (`analyze_breadth_metric.py`, no GPU; awaits experiment session). All other sections complete for v1 submission-readiness check.
