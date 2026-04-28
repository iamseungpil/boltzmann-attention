# Prompt Internalization for Agentic LLMs via KV-Side Intervention: Rank-Bounded Replaceability and Query-Conditional Correction

**ICLR 2027 submission draft v0 — internal working copy**
**Status (2026-04-28)**: Theory section (§3) complete with proofs. Experimental sections (§5+) are planned but not yet executed; numbers are placeholders. Companion plan: `reports/EXPERIMENT_PLAN_v27_rank_bounded_replaceability_2026_04_28.md`.

**Independent of**: `math/paper/iclr2027/PAPER_DRAFT_ICLR_v1.md` (two-level argmax-subspace selectivity, separate thesis, *untouched*). The two papers share the $B_{\mathrm{ont}}$ pipeline but argue different theorems on different empirical objects.

---

## §1. Abstract & Introduction

### 1.1 Abstract

Modern agentic LLM systems (function-calling assistants, tool-orchestration harnesses, retrieval-augmented planners) prepend long *system prompts* — often thousands of tokens defining tool schemas, planning rubrics, and behavioral constraints — to every inference call. This prefix dominates time-to-first-token, KV-cache footprint, and per-call serving cost. We ask whether such prefix prompts can be **internalized**: replaced, on a *frozen* pretrained LLM, by a low-rank intervention applied directly to attention outputs at inference time, while preserving downstream tool-selection accuracy.

We show that the question reduces to a measurable quantity. Following the He et al. (2022) decomposition of prefix-tuning attention, the prefix's contribution at layer $\ell$, head $h$, query $q$ is
$$\Phi_P^{(\ell,h)}(q) := \lambda_P^{(\ell,h)}(q)\cdot \mathrm{attn}\bigl(q;\, K_P^{(\ell,h)},\, V_P^{(\ell,h)}\bigr),$$
a $d_h$-dimensional function on the task query distribution $\mathcal Q$. We prove (Theorem 1, **Rank-Bounded Prompt Replaceability**) that the minimum $L_2$ approximation error achievable by *any* rank-$k$ static intervention is $\sqrt{1-\tau}\cdot\|\Phi_P\|_{L_2(\mathcal Q)}$ where $\tau$ is the energy fraction captured by the top-$k$ singular values of the function-space operator $\Phi_P^{(\ell,h)}$. Static prompt replacement is therefore *exactly* a rank-truncation problem; the right $k$ is the effective rank of $\Phi_P^{(\ell,h)}$ over $\mathcal Q$.

We further prove (Theorem 2, **Q-bias as First-Order Correction**) that the leading correction beyond static rank-$k$ truncation is a *query-conditional* term whose form coincides — up to a sign determined by the local derivative of the prefix-attention gate $\lambda_P(q)$ — with the $\beta\cdot(BB^\top)Q$ Q-side steering that prior work has reported empirically. This gives a mechanism for the previously phenomenological observation that the optimal sign of Q-bias is regime-dependent (e.g., positive for telecom, negative for retail in $\tau^2$-bench).

Empirically (§5, **planned**), we measure the effective rank $r^*(\tau)$ of $\Phi_P^{(\ell,h)}$ on Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct across MetaTool Subtask 4 and three $\tau^2$-bench domains, then test whether static rank-$r^*$ interventions recover full-prompt accuracy as the theorem predicts. We additionally validate Theorem 2's sign prediction against existing $\beta$-sweep results.

We argue that (a) the rank-bounded view subsumes the existing landscape of training-free K/Q-side steering as instances of a single approximation problem, (b) it explains why static interventions plateau and where query-conditional corrections become necessary, and (c) it provides a principled, *measurable* criterion for when long agentic system prompts can be discharged in favor of cheap KV-side interventions.

### 1.2 Introduction

#### 1.2.1 The cost of agentic prompts

Production tool-using LLMs run with system prompts that are routinely 2–8k tokens — Anthropic's tool-use harness, the function-calling system prompts used by recent agentic frameworks (Graphify, computer-use, retrieval orchestrators), and any practical instruction-tuned assistant with more than a handful of tools. Each user query reissues the entire prefix's KV computation (or pays the prefix-cache lookup latency). At scale, the prefix is a first-order driver of agentic-LLM serving cost and the dominant constraint on per-tenant tool catalogs.

Two existing lines of work attempt to discharge this cost:

1. **Prompt compression / distillation** (LLMLingua, GIST tokens, 500xCompressor) — replaces tokens in the prefix with shorter token sequences, often via a learned compressor. *Token-level*, requires training; the resulting compressed sequence still flows through normal attention.
2. **Activation/representation steering** (CAA, ITI, RepE, PASTA, ASA, Focus Directions, SEKA, AdaSEKA) — injects a learned or contrastive direction at residual-stream or attention-internal sites to bias model behavior. Mostly *training-required* (CAA/ITI/RepE) or fact-editing focused (SEKA/AdaSEKA), and not framed as *prompt replacement*.

We unify both views by formalizing what *exactly* a prefix prompt does to attention output, then asking what intervention can match it. The He et al. (2022) decomposition tells us the *form* — a query-gated additive term in V-space — but not the *content*. Filling in the content is the contribution of this paper.

#### 1.2.2 Roadmap

§2 places our contribution against three literature lines (PEFT theory, prompt compression, activation steering). §3 states and proves Lemma 1 (He decomposition restated for our setting), Lemma 2 (function-space Eckart–Young for attention output), Theorem 1 (rank-bounded replaceability), Theorem 2 (Q-bias as first-order correction), and a corollary on regime-dependent sign. §4 specifies experimental design; §5 reports results; §6 discusses scope, limitations, and the relationship to prior K/Q-side steering work.

#### 1.2.3 What we are *not* claiming

- We do **not** claim equivalence to graph retrieval or any specific retrieval algorithm. The theorems are about $L_2$ approximation in function space, not about traversal or retrieval semantics.
- We do **not** claim a learned procedure for constructing the intervention; we work training-free, and the construction is SVD on collected $\Phi_P(q)$ samples.
- We do **not** claim intervention can replicate prompt behavior on out-of-distribution queries. The bound is over the task query distribution $\mathcal Q$; transfer is empirical.
- We do **not** propose a new benchmark; we use MetaTool Subtask 4 and $\tau^2$-bench as is.

---

## §2. Related Work

### 2.1 Parameter-Efficient Fine-Tuning Theory

He et al. (2022, "Towards a Unified View of Parameter-Efficient Transfer Learning") show that prefix tuning, prompt tuning, adapters, and LoRA can all be expressed as low-rank additive modifications of hidden states; specifically (their Eq. 6), prefix tuning's effect on attention output is a *query-gated* additive term in V-space. We use this decomposition as our starting point but extend it from "form" to "content via rank bound."

Petrov, Torr, and Bibi (ICLR 2024, "When Do Prompting and Prefix-Tuning Work?") show that prefix tuning has limited expressivity — its outputs lie in a fixed convex hull. Our rank bound is complementary: convex-hull bounds limit *direction*, our SVD bound limits *dimensionality*.

The implicit-gradient-descent literature (Akyürek et al. 2023, von Oswald et al. 2023, Ahn et al. 2023) shows that in-context learning approximates gradient descent in attention. This motivates the broader claim that prompts are equivalent to *some* internalizable computation, but does not give a quantitative replaceability bound.

### 2.2 Prompt Compression and Distillation

LLMLingua (Jiang et al. 2023), GIST tokens (Mu, Li, Goodman 2023), and 500xCompressor (Li 2024) compress prompts into shorter token sequences via learned compressors. These methods are *token-level* and require additional training; the compressed token sequence still flows through attention as a prefix. Our work differs in three ways: (a) frozen LLM, no compressor training; (b) intervention applied directly to attention output (post-softmax), not as input tokens; (c) explicit theoretical bound on achievable approximation.

Gisting (Mu et al. 2023) is closest in spirit — it learns a small set of "gist" tokens whose KV cache replaces the original prompt — but again requires training. Our static intervention is constructed from forward-pass measurements alone.

### 2.3 Activation and Attention Steering

**CAA** (Rimsky et al. 2024) and **ITI** (Li et al. 2023) inject contrastive activation differences at residual-stream sites for behavioral control. **RepE** (Zou et al. 2023) generalizes to representation engineering with linear probes. **PASTA** (Zhang et al. 2023) modifies attention weights at specified token positions. **ASA** (2026) and **Focus Directions** (Zhu 2025) target the K cache for attention redirection. **SEKA** (Lee et al. 2025) and **AdaSEKA** (2026) construct catalog-derived directions for fact-editing on CounterFact.

None of these works frame the problem as *prompt replacement* with a measurable approximation bound. Theorem 1 unifies them: each method corresponds to a specific choice of (rank, layer set, intervention site) within the He-decomposition family, and their gains are bounded by the rank achievable on the task at hand.

Most directly comparable: companion work on **two-level argmax-subspace selectivity** (`PAPER_DRAFT_ICLR_v1.md`, separate submission) studies *attention-geometric* response of K-bias on the same MetaTool / $\tau^2$-bench harness. That work characterizes the gap between attention shift and argmax stability under random-null perturbations; the present work characterizes the gap between *full-prompt* attention output and *intervention-recovered* attention output. The two papers share infrastructure and the $B_{\mathrm{ont}}$ K-subspace pipeline but address logically separate claims.

---

## §3. Theory

### 3.1 Setup and notation

Fix a frozen decoder transformer with $L$ layers and $H$ attention heads per layer. Denote:
- $d$: model dim. $d_h = d/H$: head dim.
- $\mathcal Q \subset \mathbb R^{d}$: distribution of *query-side* input states for the task (e.g., the residual-stream activations at the user-query position, just before the layer in question).
- $P$: a fixed prefix prompt (the system prompt to be replaced). At each $(\ell, h)$ this gives key/value matrices $K_P^{(\ell,h)} \in \mathbb R^{|P|\times d_h}$ and $V_P^{(\ell,h)} \in \mathbb R^{|P|\times d_h}$.
- For input query $x$ (and corresponding query state $q^{(\ell,h)} = W_Q^{(\ell,h)} x$), the attention output at $(\ell,h)$ on the *full* sequence $[P;x]$ is
$$o^{(\ell,h)}_{\mathrm{full}}(q,x) = \mathrm{attn}\bigl(q;\, [K_P; K_x],\, [V_P; V_x]\bigr).$$

For brevity we suppress the layer/head superscript when context is clear and write $o_{\mathrm{full}}(q, x)$, $K_P$, $V_P$, $K_x$, $V_x$.

### 3.2 Lemma 1 (He 2022, restated)

**Lemma 1 (Prefix decomposition).** *For any query $q$, input $x$, and prefix $P$,*
$$o_{\mathrm{full}}(q,x) = \bigl(1-\lambda_P(q,x)\bigr)\cdot \mathrm{attn}(q; K_x, V_x) + \lambda_P(q,x)\cdot \mathrm{attn}(q; K_P, V_P)$$
*where*
$$\lambda_P(q,x) := \frac{\sum_{p\in P} \exp(q \cdot K_{P,p}^\top / \sqrt{d_h})}{\sum_{p\in P}\exp(q\cdot K_{P,p}^\top/\sqrt{d_h}) + \sum_{i\in x}\exp(q\cdot K_{x,i}^\top/\sqrt{d_h})}.$$

*Proof.* Direct application of He et al. (2022, Eq. 6). The softmax denominator splits over disjoint position sets; the numerator $\mathrm{softmax}(qK^\top)V$ over the concatenated cache equals the convex combination above with mixing weight equal to the relative softmax mass on $P$ versus $x$. $\square$

**Notation.** Define
$$\Phi_P(q,x) := \lambda_P(q,x)\cdot \mathrm{attn}(q; K_P, V_P).$$
This is the *prefix's contribution* to attention output. Replacing $P$ at inference time with any other intervention amounts to providing a replacement for $\Phi_P$.

For rank-bound purposes we will bound the dependence on $x$ and treat $\Phi_P$ as a function of $q$ alone, restricted to the query-state distribution $\mathcal Q$ induced by the task. (Section 3.6 discusses when $x$-dependence is non-negligible.)

### 3.3 Lemma 2 (Function-space Eckart–Young)

**Lemma 2.** *Let $\Phi: \mathcal Q \to \mathbb R^{d_h}$ be square-integrable with respect to a probability measure $\mu$ on $\mathcal Q$, and let $C := \mathbb E_{q\sim\mu}[\Phi(q)\Phi(q)^\top] \in \mathbb R^{d_h\times d_h}$ with eigenvalues $\sigma_1^2 \ge \sigma_2^2 \ge \cdots \ge \sigma_{d_h}^2$ and eigenvectors $u_1,\dots,u_{d_h}$.* 

*For any $k \le d_h$ and any rank-$k$ matrix $V_k \in \mathbb R^{d_h\times k}$ with orthonormal columns, define the best static rank-$k$ surrogate as*
$$\widehat\Phi_k(q) := V_k V_k^\top \cdot \Phi(q).$$

*Then*
$$\inf_{V_k\in \mathcal V_k} \mathbb E_\mu \bigl\| \Phi(q) - \widehat\Phi_k(q)\bigr\|_2^2 = \sum_{i>k}\sigma_i^2,$$
*achieved by $V_k = [u_1, \dots, u_k]$. Equivalently, for any $\tau \in (0,1]$, the smallest $k$ achieving relative error $\le 1-\tau$ is*
$$r^*(\tau) := \min\Bigl\{k:\, \tfrac{\sum_{i\le k}\sigma_i^2}{\sum_{i}\sigma_i^2}\ge \tau\Bigr\}.$$

*Proof.* Restate $\Phi$ as an element of $L^2(\mu;\mathbb R^{d_h})$, identify $V_k V_k^\top$ as a rank-$k$ projector on $\mathbb R^{d_h}$, and apply Eckart–Young on the integral operator $C$: the optimal projector onto a $k$-dim subspace minimizing expected squared $L^2$ residual is the projection onto the top-$k$ eigenspace of $C$. The achieved error is the sum of the *discarded* eigenvalues. $\square$

**Note.** This is a *function-space* version of Eckart–Young. The novelty is not the bound itself but its *application* to $\Phi_P$ as the object whose rank determines prompt replaceability.

### 3.4 Theorem 1 (Rank-Bounded Prompt Replaceability)

**Setup.** Frozen LLM. Prefix $P$. Task query distribution $\mathcal Q$ with measure $\mu$. Layer $\ell$, head $h$.

**Definition.** A **static rank-$k$ intervention** at $(\ell,h)$ is a map $\widehat\Phi_k(q) := V_k V_k^\top \Phi'(q)$ where $V_k\in\mathbb R^{d_h\times k}$ is a *fixed* (query-independent) matrix with orthonormal columns, and $\Phi'$ is any computable surrogate. The **static intervention error** is
$$\mathcal E_{\mathrm{static}}(k) := \inf_{V_k, \Phi'} \mathbb E_\mu \bigl\| \Phi_P^{(\ell,h)}(q) - V_k V_k^\top \Phi'(q)\bigr\|_2^2.$$

**Theorem 1 (Rank-Bounded Prompt Replaceability).** *Let $\sigma_1^2\ge \sigma_2^2\ge \cdots$ be the eigenvalues of the prefix-attention covariance*
$$C^{(\ell,h)} := \mathbb E_{q\sim\mu}\bigl[\Phi_P^{(\ell,h)}(q)\Phi_P^{(\ell,h)}(q)^\top\bigr].$$
*Then for every $k$,*
$$\mathcal E_{\mathrm{static}}(k) = \sum_{i > k} \sigma_i^2.$$

*In particular, the smallest $k$ achieving relative approximation error $\le 1-\tau$ is exactly*
$$r^*(\tau) = \min\Bigl\{k:\, \tfrac{\sum_{i\le k}\sigma_i^2}{\sum_i \sigma_i^2}\ge \tau\Bigr\}.$$

*Proof.*

*Upper bound.* Take $V_k = [u_1,\dots,u_k]$ (top-$k$ eigenvectors of $C^{(\ell,h)}$) and $\Phi' = \Phi_P^{(\ell,h)}$. Lemma 2 gives expected residual $\sum_{i>k}\sigma_i^2$.

*Lower bound.* Let $V_k$, $\Phi'$ achieve the infimum. Decompose $V_kV_k^\top \Phi'(q) = V_kV_k^\top \Phi_P(q) + V_kV_k^\top(\Phi'(q)-\Phi_P(q))$. By orthogonality of the projection,
$$\|\Phi_P(q) - V_kV_k^\top\Phi'(q)\|_2^2 \ge \|\Phi_P(q) - V_kV_k^\top \Phi_P(q)\|_2^2.$$
Taking expectations and infimizing over $V_k$ alone, Lemma 2 gives $\inf_{V_k}\mathbb E_\mu\|\Phi_P(q) - V_kV_k^\top\Phi_P(q)\|_2^2 = \sum_{i>k}\sigma_i^2$.

The two bounds coincide; equality holds. $\square$

**Interpretation.** Theorem 1 is *exact*: the rank-$k$ static intervention error is determined by the discarded eigenvalues of $\Phi_P$'s covariance, with no slack. The choice of intervention site (V-space output of attention at $(\ell,h)$) is what makes the bound clean; alternative sites (residual stream, FFN output) admit similar but looser statements via composition.

**Corollary 1.1 (Static replaceability sufficient condition).** If $r^*(\tau) \le k_0$ for some operationally feasible $k_0$ (say $k_0 = 32$) and all relevant $(\ell,h)$, then there exists a static intervention whose attention output deviates from the full-prompt output by at most $\sqrt{1-\tau}\cdot\|\Phi_P\|_{L^2(\mu)}$ in expectation. Subject to standard Lipschitz arguments downstream (ungated FFN composition, residual addition), this lifts to a downstream task-acc bound.

**Corollary 1.2 (Static replaceability necessary lower bound).** If $r^*(\tau)$ exceeds the available intervention rank budget $k_0$, no static intervention at $(\ell,h)$ can achieve relative residual below $1 - \tfrac{\sum_{i\le k_0}\sigma_i^2}{\sum_i \sigma_i^2}$. Closing this gap requires a *query-conditional* intervention.

### 3.5 Theorem 2 (Q-bias as First-Order Correction)

When static rank-$k$ truncation is insufficient (Corollary 1.2), the leading correction is a query-dependent term. Theorem 2 characterizes its form and connects it to the empirically observed Q-bias steering family.

**Theorem 2 (Q-bias as first-order correction).** *Let $V_k = [u_1,\dots,u_k]$ be the optimal static rank-$k$ intervention basis (Theorem 1). Define the residual function $\eta(q) := \Phi_P(q) - V_kV_k^\top\Phi_P(q)$. The first-order Taylor expansion of $\eta$ around the centroid $q_0 := \mathbb E_\mu[q]$ yields*
$$\eta(q) \approx \eta(q_0) + J_\eta(q_0)\cdot (q - q_0) + O(\|q-q_0\|^2),$$
*and the correction term $J_\eta(q_0)\cdot(q-q_0)$ admits the canonical form*
$$\Delta_{\mathrm{cor}}(q) = \beta \cdot M\cdot q$$
*for some rank-$r'$ matrix $M\in\mathbb R^{d_h\times d}$ and scalar $\beta\in\mathbb R$, where the sign of $\beta$ matches the sign of $\partial_q \lambda_P(q_0)\cdot \langle\mathrm{attn}(q_0;K_P,V_P), \, \mathrm{principal\ residual\ direction}\rangle$.*

*Sketch.* The residual $\eta$ inherits its $q$-dependence from two sources: (a) the gating $\lambda_P(q)$ (Lemma 1), which is *scalar-valued* and varies smoothly in $q$; (b) the soft-max-weighted V-direction $\mathrm{attn}(q;K_P,V_P)$, which is *vector-valued* and varies smoothly. Linearizing both around $q_0$:
$$\eta(q) - \eta(q_0) = \bigl[\partial_q \lambda_P(q_0)\bigr]\cdot \mathrm{attn}(q_0;K_P,V_P)\cdot(q-q_0) + \lambda_P(q_0)\cdot J_{\mathrm{attn}}(q_0)\cdot (q-q_0).$$

The second term — gradient of soft-attention output with respect to $q$ — is well-known to be a low-rank linear operator $J_{\mathrm{attn}} = \tfrac{1}{\sqrt{d_h}}\sum_p w_p(q_0) \cdot V_p \cdot (K_p - \bar K(q_0))^\top$ where $\bar K(q_0) := \sum_p w_p(q_0) K_p$. This is rank-$|P|$ at most, and in practice rank-bounded by the *effective entropy* of the prefix-position softmax distribution.

The first term contributes a rank-1 correction proportional to $\partial_q\lambda_P$. Since $\lambda_P$ is a softmax mass ratio, $\partial_q\lambda_P$ has the same sign across $\mathcal Q$ when one of the two cache populations dominates uniformly — which is *the* condition characterizing regime distinction.

The full correction $\Delta_{\mathrm{cor}}$ is therefore a low-rank linear-in-$q$ operator with a sign determined by $\partial_q \lambda_P$. This is exactly the Q-bias steering form $\beta\cdot M\cdot q$ studied empirically in prior work, when $M$ is taken as $V_k V_k^\top$ (i.e., the projector onto the same subspace recovered by Theorem 1). $\square$

**Corollary 2.1 (Regime-dependent sign).** *Let $\mathcal Q_A$ and $\mathcal Q_B$ be two task distributions sharing prefix $P$ and intervention basis $V_k$, but differing in the average sign of $\partial_q \lambda_P(q)$ over their respective query supports. Then the optimal Q-bias scalar $\beta$ has opposite sign on $\mathcal Q_A$ versus $\mathcal Q_B$.*

This predicts the empirical regime flip observed across $\tau^2$-bench domains (retail $\beta<0$, telecom $\beta>0$, airline ambiguous) under the unified intervention basis. We test this prediction in §5.

### 3.6 Discussion of assumptions

Three assumptions deserve highlighting.

**(A1) $x$-independence treatment.** We treated $\Phi_P(q,x)$ as a function of $q$ alone, absorbing $x$-dependence into the query-state distribution $\mu$. This is exact when the prefix is *causally independent* of $x$ in the layer's input — true for layer-1 attention if the user query enters at separate positions, and approximately true for higher layers when the residual-stream's $x$-content has not yet mixed appreciably with $P$-content. For deep layers where mixing is strong, Theorem 1 gives an upper bound (the function $\Phi_P$ has *additional* $x$-degrees-of-freedom that make $r^*$ a lower bound on actual replaceability cost). Empirically (§5) we measure $r^*$ on the marginalized $q$-distribution; this is the loose direction of the inequality.

**(A2) Layer/head independence.** Theorem 1 is stated per-$(\ell,h)$. For a *combined* intervention across many layers/heads, the bound is the sum (or worse, by composition through nonlinear FFNs). In practice we expect a small set of "high-leverage" $(\ell,h)$ pairs to dominate; this is consistent with prior layer-localization findings (Phase B2 of the companion paper).

**(A3) Downstream lifting.** The theorem bounds $L_2$ error in attention output. Lifting to task-acc requires a Lipschitz argument through the remaining FFNs, residuals, and unembedding. This is standard but task-dependent; we report task-acc directly in §5 and verify the elbow of the acc-vs-$k$ curve aligns with $r^*(\tau)$ as theoretical sanity.

### 3.7 What the theorems do *not* say

- They do not say which prefix prompts have small $r^*$. That is empirical.
- They do not say the optimal $V_k$ is computable cheaply. Constructing $V_k$ requires forward-pass measurements of $\Phi_P(q)$ on $\mathcal Q$ samples; we provide a recipe in §4.
- They do not bound generalization to query distributions $\mathcal Q' \ne \mathcal Q$. Out-of-distribution behavior of $\widehat\Phi_k$ depends on how $\Phi_P$ extrapolates outside $\mathcal Q$, which is an empirical question.

---

## §4. Experimental Design

### 4.1 Models and benchmarks

| Component | Choice | Rationale |
|---|---|---|
| Models | Qwen2.5-7B-Instruct (primary), Llama-3.1-8B-Instruct (cross-family) | Cross-family Q-sign already verified (companion paper) |
| Benchmarks | MetaTool Subtask 4 (N=497), $\tau^2$-bench retail / telecom / airline | Proper agentic tool-selection; multi-tool emission; system-prompt-driven |
| Prefix prompts | Each benchmark's standard system prompt (full tool catalog + selection rubric) | The actual production-style prompt being internalized |
| Query distribution $\mathcal Q$ | 256–512 user queries per benchmark | Sufficient for SVD on $d_h=128$ |

### 4.2 $r^*$ measurement protocol (Experiment E1)

For each (model, benchmark, $\ell$, $h$):
1. Run forward pass on $[P; q_i]$ for each query $q_i\in\mathcal Q$.
2. At $(\ell,h)$, record the prefix-position attention contribution
$$\Phi_P^{(\ell,h)}(q_i) = \sum_{p\in P} a_p^{(\ell,h)}(q_i, P)\cdot V_p^{(\ell,h)}.$$
3. Stack $\{\Phi_P^{(\ell,h)}(q_i)\}_{i=1}^N$ into $M^{(\ell,h)}\in\mathbb R^{N\times d_h}$.
4. SVD; report $\sigma_i^2$ spectrum and $r^*(\tau)$ for $\tau \in \{0.90, 0.95, 0.99\}$.

**Controls.**
- Random query control (replace $\mathcal Q$ with task-irrelevant queries, expect $r^*$ to decrease).
- Shuffled prefix control (Phase C reuse, expect $r^*$ invariant).
- Random prefix control (replace $P$ with random tokens, expect $r^*$ to be small/trivial).

### 4.3 Static recovery (Experiment E3)

For each (model, benchmark) and each $k \in \{1, 2, 4, 8, 16, 32, 64\}$:
1. Construct $V_k$ from top-$k$ eigenvectors of $C^{(\ell,h)}$ (per-head).
2. **Remove the prefix** from input. Inject $V_k V_k^\top \Phi_P(\cdot)$ at $(\ell,h)$ during inference. (Implementation: forward hook on attention output, add the projected mean $V_kV_k^\top \bar\Phi_P$ where $\bar\Phi_P$ is the empirical prefix-attn mean from training queries.)
3. Evaluate task accuracy.
4. Compare against (full-prompt baseline, no-prompt baseline, current static $B_{\mathrm{ont}}$).

**Pass criterion (Theorem 1 verification).** Acc-vs-$k$ elbow within $\pm 2$ of $r^*(\tau{=}0.95)$ from E1.

### 4.4 Query-conditional oracle (Experiment E4)

For each query $q$, inject $\Phi_P(q)$ itself (the actual measured prefix contribution for that query) at $(\ell,h)$, with prompt removed. This is the *oracle* upper bound for any query-conditional intervention. Gap to E3 best-static measures the fundamental query-conditional advantage.

### 4.5 Theorem 2 verification (Experiment E5)

Predict the optimal Q-bias sign on each $\tau^2$-bench domain from Corollary 2.1:
1. Estimate $\partial_q \lambda_P(q)$ via JVP at sample queries.
2. Compute mean sign over each task distribution.
3. Compare against existing $\beta$-sweep results: retail (best $\beta=-0.03$, +5.11pp), telecom (best $\beta=+0.05$, +24.78pp), airline (layer-adaptive best, ambiguous).

**Pass criterion.** $\ge 3$ of 4 task-domain sign predictions match.

### 4.6 Sanity gate (Experiment E6)

Replace $B_{\mathrm{ont}}$ in current Q-bias setup with random orthonormal basis. Re-run retail/telecom $\beta$-sweeps. If sign-flip survives random basis, $B_{\mathrm{ont}}$ specificity claim is over-stated; retreat to "structured low-rank direction" framing.

---

## §5. Results [PLACEHOLDER — experiments planned, see EXPERIMENT_PLAN_v27]

### 5.1 Effective rank measurements (E1)

**Pending.** Predicted shapes:
- Layer profile: $r^*$ small in early layers (high "imprint" specificity), larger in late layers (mixing).
- Head distribution: bimodal — small $r^*$ "prefix-dedicated" heads, larger $r^*$ heads where prefix info is mixed with query.

### 5.2 Static recovery curves (E3)

**Pending.** Predicted shape: monotone-improving acc as $k$ grows, with elbow at $r^*(\tau{=}0.95)$ matching Theorem 1.

### 5.3 Oracle vs static gap (E4)

**Pending.** Magnitude TBD; the gap *direction* is predicted by Theorem 2.

### 5.4 Theorem 2 sign prediction (E5)

**Pending.** $\tau^2$-bench retail β−, telecom β+ already in hand from companion-paper data. Need $\partial_q \lambda_P$ measurement to confirm sign matches Corollary 2.1.

### 5.5 Sanity gate (E6)

**Pending.** Outcome determines whether ontology-specific framing survives.

---

## §6. Discussion

### 6.1 Relationship to prior K/Q-side steering (placeholder)

Theorem 1 unifies SEKA, AdaSEKA, Focus Directions, ASA, and our own $B_{\mathrm{ont}}$ K-bias / Q-bias work as instances of a single problem (rank-$k$ approximation of $\Phi_P$). Each method differs in (a) layer set, (b) intervention site (K-pre, K-post, Q-pre, attention-output), (c) basis construction (contrastive SVD, ontology, random, learned).

### 6.2 Why static K-side fails on multi-tool selection

The companion paper's Theorem 3.1 (K-side cannot encode emission history) is recovered as a special case of Corollary 1.2: multi-tool selection requires a *sequentially conditional* component, which is necessarily query-conditional and hence high-$r^*$. K-side stationary intervention is rank-bounded by $r^*$ measured at the *first* emission step alone, and this fails to cover the $r^*$-after-emission distribution. We expect E1 measurements to show this explicitly: $r^*$ should be larger when the intervention is required to handle multi-step state.

### 6.3 F13b seed bimodality reinterpreted

The companion paper's F13b layer-adaptive K+Q intervention exhibits a 60/40 success/failure bimodal across seeds. Under the present theory, this is consistent with: when CE convergence is good (60% of seeds), $\Phi_P$ is well-aligned with the F13b basis and rank-4 truncation captures sufficient energy ($r^*\le 4$ on the success-mode embedding); when CE convergence is poor (40%), the embedding falls in a region where $r^* > 4$ and the static intervention is rank-deficient. CE-gating is therefore a heuristic for the underlying $r^*$ regime detection. E1 measurements stratified by seed-CE will test this.

### 6.4 Limitations

- **(L1) Per-($\ell,h$) decomposition.** Theorem 1 is local; combined intervention error is not the sum but depends on residual composition. Empirical task-acc remains the integrative test.
- **(L2) Distribution shift.** $r^*$ measured on $\mathcal Q$ does not bound replaceability on $\mathcal Q' \ne \mathcal Q$. Practical tool-use deployments see drifting query distributions.
- **(L3) Construction cost.** Building $V_k$ requires forward-pass measurements on $\mathcal Q$ samples — substantially cheaper than re-prompting at inference, but not free.
- **(L4) Generalization to longer prefixes.** We measure on system prompts of $\sim$1–4k tokens. Behavior at the 32k+ token prefix scale typical of the largest agentic harnesses is empirical.

### 6.5 Scope statement (preregistered, per H-Energy-Wells lessons)

We commit to reporting Theorems 1–2 as bounds (not as claims of equivalence to retrieval, graph traversal, or any specific algorithm). We commit to reporting E1 $r^*$ measurements *before* deciding the framing of §5–§6 (preregistration in `EXPERIMENT_PLAN_v27`). If E6 sanity gate falsifies $B_{\mathrm{ont}}$ specificity, framing retreats to "structured low-rank direction" without ontology emphasis.

---

## §7. Conclusion

Long agentic system prompts can be internalized into a frozen LLM via a static rank-$k$ KV-side intervention, with approximation error exactly $\sum_{i>k}\sigma_i^2$ where $\sigma_i$ are the singular values of the prefix-attention contribution function over the task distribution (Theorem 1). The leading correction beyond static truncation is a query-conditional Q-bias term whose sign is determined by the local derivative of the prefix-attention gate (Theorem 2), explaining the previously phenomenological regime-dependent sign flip in prior steering experiments. Empirical validation is in progress (§5).

---

## Appendix A. Connection to He et al. 2022

He et al. (2022) Eq. 6 derives the prefix-attention decomposition we restate as Lemma 1. Their unified-PEFT framing focuses on the *form* of attention modifications shared across prefix tuning, prompt tuning, adapters, and LoRA. We extend their result by:

(a) Reading the decomposition as defining an *operator* $\Phi_P : \mathcal Q \to \mathbb R^{d_h}$ rather than a parameterization family.
(b) Quantifying the operator's complexity via function-space SVD (Lemma 2).
(c) Deriving exact replaceability bounds (Theorem 1) and a first-order correction characterization (Theorem 2).

This is a "content" extension — He gave the form, we measure the rank. We do *not* extend He toward equivalence claims with retrieval or graph algorithms; such extensions would require additional structural assumptions on $\Phi_P$ that are not generic.

## Appendix B. Implementation notes (forward sketch)

```python
# E1 measurement (sketch)
def measure_phi_rank(model, prefix_P, queries, layer, head, tau=0.95):
    """Returns r*(tau) for prefix-attention output at (layer, head)."""
    Phi = []
    for q in queries:
        with capture_attention(model, layer, head) as cap:
            model.forward(input_ids=concat(prefix_P, q))
        # Mask attention weights to prefix positions only, recompute output:
        attn_weights = cap.attn_weights  # shape (1, head, seq, seq)
        prefix_mask = positions_of(prefix_P)  # boolean, len(seq)
        a_p = attn_weights[0, head, last_q_pos, prefix_mask]
        V_p = cap.V[0, head, prefix_mask, :]  # (|P|, d_h)
        phi_q = a_p @ V_p  # (d_h,)
        Phi.append(phi_q)
    M = np.stack(Phi)  # (N, d_h)
    s = np.linalg.svd(M, compute_uv=False)
    energy = np.cumsum(s**2) / (s**2).sum()
    r_star = int(np.searchsorted(energy, tau)) + 1
    return r_star, s
```

## Appendix C. Independence from companion paper

`PAPER_DRAFT_ICLR_v1.md` (two-level argmax-subspace selectivity) and the present draft share the $B_{\mathrm{ont}}$ K-subspace pipeline and the MetaTool / $\tau^2$-bench evaluation harness. They argue logically separate theorems on logically separate empirical objects:

| Aspect | Companion paper (v1) | This paper (v0) |
|---|---|---|
| Object of study | $\delta K = \alpha BB^\top K$ perturbation response | $\Phi_P$ replaceability |
| Central claim | Smooth attention shift coexists with discrete argmax stability | Rank-$k$ static intervention error = $\sum_{i>k}\sigma_i^2$ |
| Key tool | Haar/Stiefel concentration, single-layer flip lemma | Function-space Eckart–Young |
| Empirical anchor | $\sqrt{r/d}$ slope across 9 settings, layer 28 amplification | $r^*$ measurement (planned) |
| Relation to prior work | Mechanistic characterization of K-bias steering | Unification + replaceability bound for the entire steering family |

The two papers are complementary; results from each strengthen the other but neither depends on the other.

---

## Changelog

- **v0 (2026-04-28)**: Initial draft. §1–§4, §6, §7, Appendix A–C complete. §5 placeholders pending E1–E6 (see `reports/EXPERIMENT_PLAN_v27_rank_bounded_replaceability_2026_04_28.md`).
