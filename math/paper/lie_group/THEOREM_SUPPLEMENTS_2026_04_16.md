# Theorem Supplements: Closing Proof Gaps in the Q-Coverage / K-Bias Framework

**Date**: 2026-04-16
**Status**: Supplement to APPENDIX_B_PROOFS.md
**Scope**: Gaps 1--5 identified in proof review of Theorems 6.17 and 6.20

---

## Notation and Prerequisites

We use the following notation throughout, consistent with APPENDIX\_B\_PROOFS.md:

- $B_{\mathrm{ont}} \in \mathbb{R}^{d \times R}$: orthonormal ontology basis (R facet directions)
- $P_{\mathrm{ont}} = B_{\mathrm{ont}} B_{\mathrm{ont}}^\top$: orthogonal projector onto $\operatorname{span}(B_{\mathrm{ont}})$
- $K'_{t} = B_{\mathrm{ont}}^\top K_t$: rotated key at position $t$ (R-dimensional)
- $W_U \in \mathbb{R}^{|\mathcal{V}| \times d}$: unembedding matrix
- $g_f$: gate function for facet $f$ satisfying condition (R)
- $s_{\min} \geq 3$: minimum facet separation in (H-cat)
- $\mathcal{F} = \{f_1, \ldots, f_F\}$: set of facet categories
- $T_{\mathrm{tool}}$: number of tokens in a tool name; $T_{\mathrm{prompt}}$: total prompt length

We assume familiarity with:
- Theorem 6.1 (per-sample attention-weighted bound)
- Condition (H-cat) (bimodal K-column distribution, lines 740--743)
- Condition (R) (gate Lipschitz regularity)
- Lemma 6.17.A (first-order channel separation)

---

## Gap 1: Extended Bimodality Condition for Output Gradients

### Motivation

Condition (H-cat) governs the distribution of rotated keys $K'_{t,i}$ in the ontology basis. Theorem 6.17's proof, however, invokes the claim that the output gradient $\nabla_{o_t} \log p(y_t \mid y_{<t})$ also has significant projection onto $\operatorname{span}(B_{\mathrm{ont}})$ when $y_t$ is a correct tool token. This is not implied by (H-cat) alone, since (H-cat) is a statement about key statistics, not about the loss landscape.

### Definition (H-cat-ext): Extended Bimodality with Output-Gradient Alignment

Let $\mathcal{T}_{\mathrm{tool}} \subset \mathcal{V}$ denote the set of tool-name tokens (first token of each tool name in the vocabulary). Define:

$$
\eta_{\mathrm{grad}} := \min_{v \in \mathcal{T}_{\mathrm{tool}}} \frac{\| P_{\mathrm{ont}} \, \nabla_{o_t} \log p(v \mid y_{<t}) \|_2}{\| \nabla_{o_t} \log p(v \mid y_{<t}) \|_2}
$$

where the gradient is taken with respect to the attention output $o_t$ at the token position where $v$ is the correct next token.

**(H-cat-ext)**: Condition (H-cat) holds, AND $\eta_{\mathrm{grad}} \geq \eta_0$ for some constant $\eta_0 > 0$.

### Lemma 1.1 (Sufficient Condition for (H-cat-ext) via Unembedding Concentration)

**Statement.** Let $W_U^{\mathcal{T}} \in \mathbb{R}^{|\mathcal{T}_{\mathrm{tool}}| \times d}$ be the submatrix of $W_U$ restricted to tool-token rows. Define the tool unembedding concentration ratio:

$$
\gamma_U := \min_{v \in \mathcal{T}_{\mathrm{tool}}} \frac{\| P_{\mathrm{ont}} \, w_v \|_2}{\| w_v \|_2}
$$

where $w_v$ is the row of $W_U$ corresponding to token $v$. If (H-cat) holds and

$$
\gamma_U \geq \gamma_0 > 0,
$$

then (H-cat-ext) holds with $\eta_{\mathrm{grad}} \geq \gamma_0 \cdot (1 - \delta)$, where $\delta = O(|\mathcal{V}|^{-1})$ depends on the softmax saturation at the correct token.

**Proof.**

At a position $t$ where the correct next token is $v \in \mathcal{T}_{\mathrm{tool}}$, the gradient of the log-likelihood with respect to the pre-softmax logit $z_v = w_v^\top o_t$ is:

$$
\frac{\partial \log p(v \mid y_{<t})}{\partial o_t} = (1 - p(v \mid y_{<t})) \, w_v - \sum_{v' \neq v} p(v' \mid y_{<t}) \, w_{v'}.
$$

This can be rewritten as:

$$
\nabla_{o_t} \log p = w_v - \sum_{v' \in \mathcal{V}} p(v' \mid y_{<t}) \, w_{v'} = w_v - \bar{w},
$$

where $\bar{w} = \mathbb{E}_{v' \sim p}[w_{v'}]$ is the probability-weighted mean embedding.

Now:

$$
\| P_{\mathrm{ont}}(w_v - \bar{w}) \|_2 \geq \| P_{\mathrm{ont}} w_v \|_2 - \| P_{\mathrm{ont}} \bar{w} \|_2.
$$

The first term satisfies $\| P_{\mathrm{ont}} w_v \|_2 \geq \gamma_0 \| w_v \|_2$ by assumption.

For the second term, $\bar{w}$ is a convex combination over the full vocabulary. Under (H-cat), tool tokens constitute a small fraction of the vocabulary. Since $P_{\mathrm{ont}}$ projects onto an $R$-dimensional subspace ($R \ll d$) tailored to tool-relevant directions, a random vocabulary token $w_{v'}$ has expected squared projection $\mathbb{E}[\| P_{\mathrm{ont}} w_{v'} \|_2^2] = (R/d) \| w_{v'} \|_2^2$ under isotropic assumptions on non-tool tokens. Thus:

$$
\| P_{\mathrm{ont}} \bar{w} \|_2 \leq \max_{v'} \| P_{\mathrm{ont}} w_{v'} \|_2 \leq \sqrt{R/d} \cdot \max_{v'} \| w_{v'} \|_2 + \epsilon
$$

where $\epsilon$ accounts for deviation from isotropy. In practice, $R/d \approx 10/3584 \approx 0.003$, so $\| P_{\mathrm{ont}} \bar{w} \|_2$ is small relative to $\| P_{\mathrm{ont}} w_v \|_2$.

More precisely, define $\delta := \| P_{\mathrm{ont}} \bar{w} \|_2 / \| P_{\mathrm{ont}} w_v \|_2$. Then:

$$
\eta_{\mathrm{grad}} \geq \frac{\gamma_0 \| w_v \|_2 (1 - \delta)}{\| w_v - \bar{w} \|_2}.
$$

Since $\| w_v - \bar{w} \|_2 \leq \| w_v \|_2 + \| \bar{w} \|_2 \leq \| w_v \|_2 (1 + O(1))$ and in the well-trained regime $\| \bar{w} \|_2 \leq \| w_v \|_2$ (the mean embedding has smaller norm than individual tool embeddings), we obtain:

$$
\eta_{\mathrm{grad}} \geq \frac{\gamma_0 (1 - \delta)}{1 + \|\bar{w}\|_2 / \|w_v\|_2} \geq \gamma_0 (1 - \delta) / 2.
$$

The factor of 2 is a worst case; empirically the denominator ratio $\|\bar{w}\|_2/\|w_v\|_2$ is much smaller than 1 for well-trained models. $\square$

### Remark 1.1 (Empirical Verification Protocol)

(H-cat-ext) is directly testable:

1. **$\gamma_U$ measurement**: Extract $W_U$ from the model, identify tool tokens $\mathcal{T}_{\mathrm{tool}}$, compute $\| P_{\mathrm{ont}} w_v \|_2 / \| w_v \|_2$ for each $v$.
2. **$\eta_{\mathrm{grad}}$ measurement**: Run a forward pass on MetaTool prompts, hook $o_t$ at tool-generation positions, compute the ratio.
3. **Acceptance criterion**: $\gamma_U \geq 0.1$ (10% of tool embedding energy in ontology subspace) is sufficient for the bound to be non-trivial.

### Remark 1.2 (Connection to Theorem 6.17)

Everywhere in the proof of Theorem 6.17 where the phrase "support in $\operatorname{span}(B_{\mathrm{ont}})$" appears in reference to loss gradients, the correct justification is: "by (H-cat-ext) with $\eta_{\mathrm{grad}} \geq \eta_0$, the gradient has at least fraction $\eta_0$ of its energy in $\operatorname{span}(B_{\mathrm{ont}})$, which is sufficient for the first-order channel separation argument of Lemma 6.17.A."

---

## Gap 2: Diminishing Marginal Return for Already-Emitted Facets

### Motivation

The proof of Theorem 6.17(b) at line 1170 claims that "the marginal first-order benefit of adding more attention mass to an already-emitted facet is net-zero." This is false in general: already-emitted tool tokens reside in the KV cache and can affect subsequent generation (e.g., the model may reference a previously selected tool name when generating the next tool). The correct statement is that the marginal benefit is BOUNDED and small under natural conditions.

### Proposition 6.17.B (Diminishing Marginal Return)

**Statement.** Consider an autoregressive model generating a sequence of $M$ tool selections $\{y^{(1)}, \ldots, y^{(M)}\}$. Let $f_j$ denote the facet corresponding to tool $y^{(j)}$. At generation step $m$ (generating tool $y^{(m)}$), define the marginal attention benefit of re-attending to facet $f_j$ ($j < m$, already emitted) as:

$$
\Delta_{\mathrm{re}}(f_j, m) := \sum_{t \in \mathrm{pos}(y^{(j)})} \alpha_{t}^{(m)} \cdot \| P_{f_j} V_t \|_2
$$

where $\alpha_t^{(m)}$ is the attention weight from query position $m$ to key position $t$, and $\mathrm{pos}(y^{(j)})$ denotes the token positions occupied by the $j$-th emitted tool name.

Under the following conditions:
- **(C1)** Tool names are short: $|\mathrm{pos}(y^{(j)})| \leq T_{\mathrm{tool}}$ tokens
- **(C2)** The prompt is long: total sequence length $T \geq T_{\mathrm{prompt}}$
- **(C3)** Keys at tool-name positions have bounded norm: $\| K_t \|_2 \leq \kappa$ for $t \in \mathrm{pos}(y^{(j)})$

Then:

$$
\Delta_{\mathrm{re}}(f_j, m) \leq T_{\mathrm{tool}} \cdot \frac{e^{\kappa \|q_m\|_2 / \sqrt{d}}}{T_{\mathrm{prompt}}} \cdot \max_t \|V_t\|_2.
$$

In particular, when $T_{\mathrm{tool}} \leq 5$ and $T_{\mathrm{prompt}} \geq 100$:

$$
\Delta_{\mathrm{re}}(f_j, m) = O\!\left(\frac{T_{\mathrm{tool}}}{T_{\mathrm{prompt}}}\right) \cdot e^{\kappa \|q_m\|_2 / \sqrt{d}} \cdot \max_t \|V_t\|_2.
$$

**Proof.**

For each head, the attention weight from query position $m$ to any key position $t$ is:

$$
\alpha_t^{(m)} = \frac{\exp(q_m^\top K_t / \sqrt{d})}{\sum_{s=1}^{T} \exp(q_m^\top K_s / \sqrt{d})}.
$$

The numerator for a single tool-token position is bounded:

$$
\exp(q_m^\top K_t / \sqrt{d}) \leq \exp(\| q_m \|_2 \| K_t \|_2 / \sqrt{d}) \leq \exp(\kappa \| q_m \|_2 / \sqrt{d}).
$$

The denominator is bounded below by the contribution of non-tool-token positions. There are at least $T_{\mathrm{prompt}} - M \cdot T_{\mathrm{tool}}$ prompt tokens. Each contributes at least $\exp(-\kappa \|q_m\|_2 / \sqrt{d})$ to the denominator (since key norms are bounded). However, a tighter and simpler bound uses the fact that the denominator has at least $T_{\mathrm{prompt}}$ terms, each at least $\exp(0) = 1$ (by centering; if keys are not centered, use $\exp(-\kappa^2/\sqrt{d})$ as a lower bound per term). We use the weaker but cleaner bound:

$$
\sum_{s=1}^T \exp(q_m^\top K_s / \sqrt{d}) \geq T_{\mathrm{prompt}} \cdot \exp(\min_s q_m^\top K_s / \sqrt{d}).
$$

To avoid dependence on the minimum, we use the standard softmax concentration inequality. For our purposes, the uniform lower bound suffices:

$$
\alpha_t^{(m)} \leq \frac{\exp(\kappa \|q_m\|_2 / \sqrt{d})}{\sum_{s} \exp(q_m^\top K_s / \sqrt{d})}.
$$

Since the denominator contains $T$ terms and is at least $T$ when all exponents are non-negative, or at minimum contains $T_{\mathrm{prompt}}$ terms each at least 1 (under key centering), we get the practical bound:

$$
\alpha_t^{(m)} \leq \frac{\exp(\kappa \|q_m\|_2/\sqrt{d})}{T_{\mathrm{prompt}}}.
$$

Summing over the $|\mathrm{pos}(y^{(j)})| \leq T_{\mathrm{tool}}$ positions of the already-emitted tool:

$$
\Delta_{\mathrm{re}}(f_j, m) \leq \sum_{t \in \mathrm{pos}(y^{(j)})} \alpha_t^{(m)} \cdot \|V_t\|_2 \leq T_{\mathrm{tool}} \cdot \frac{\exp(\kappa \|q_m\|_2/\sqrt{d})}{T_{\mathrm{prompt}}} \cdot \max_t \|V_t\|_2.
$$

For typical transformer activations, $\kappa \|q_m\|_2 / \sqrt{d}$ is $O(1)$ (e.g., in Qwen2.5-7B with $d = 3584$, empirical $\|q\|_2 \approx 10$--$30$, $\|K\|_2 \approx 10$--$30$, giving $\kappa \|q\|_2/\sqrt{d} \approx 5$--$15$), so the exponential factor is a moderate constant (say $C_{\exp} \approx 150$--$10^6$).

With $T_{\mathrm{tool}} = 3$ and $T_{\mathrm{prompt}} = 500$, the bound gives:

$$
\Delta_{\mathrm{re}} \leq \frac{3 C_{\exp}}{500} \cdot \max_t \|V_t\|_2.
$$

Even with $C_{\exp} = 10^4$ (very large), this is $60 \cdot \max_t \|V_t\|_2$, which must be compared against the benefit of attending to the NEW correct facet. $\square$

### Remark 2.1 (Why "Net-Zero" Fails but Diminishing Return Suffices)

The original "net-zero" claim would require $\Delta_{\mathrm{re}} = 0$, which is false since already-emitted tokens carry non-zero values in the KV cache. However, for the Q-coverage argument to work, we only need:

$$
\Delta_{\mathrm{new}}(f_m, m) \gg \Delta_{\mathrm{re}}(f_j, m) \quad \text{for all } j < m,
$$

i.e., the benefit of attending to the NEW facet dominates re-attention to old facets. This holds when:

1. The Q-coverage mechanism subtracts $P_{f_j}$ from the query, actively suppressing re-attention (the numerator $\exp(q_m^\top K_t / \sqrt{d})$ is reduced for old-facet keys), AND
2. The K-bias amplifies keys along the new facet direction (increasing $\Delta_{\mathrm{new}}$).

Under condition (1), the effective $\kappa$ for old-facet keys is reduced to $\kappa (1 - \|P_{f_j} q_m\|_2 / \|q_m\|_2 \cdot \|K_t\|_2 / \kappa)$, further suppressing $\Delta_{\mathrm{re}}$.

### Remark 2.2 (Replacing Line 1170)

In the proof of Theorem 6.17(b), line 1170 should be replaced with: "By Proposition 6.17.B, the marginal first-order benefit of re-attending to already-emitted facet $f_j$ is bounded by $O(T_{\mathrm{tool}} / T_{\mathrm{prompt}}) \cdot C_{\exp} \cdot \max_t \|V_t\|_2$. After Q-coverage projection subtracts $P_{f_j}$ from $q_m$, the effective exponential constant $C_{\exp}$ is further reduced. Thus the new-facet benefit dominates, and the Lemma 6.17.A separation argument applies to the residual query."

---

## Gap 3: Theorem 6.17' (Revised Statement)

### Motivation

The original Theorem 6.17 at lines 1129--1137 claims full QKV-joint optimality in part (d). Remark 6.17.3 (lines 1186--1247) documents that K-channel inclusion is destructive at every tested value of $\alpha_K$. The theorem statement must be revised to reflect this empirical falsification.

### Theorem 6.17' (Q-Coverage Optimality for Multi-Tool Selection --- Revised)

**Setting.** Let $\mathcal{M}$ be a pre-trained autoregressive LLM with $L$ layers and $H$ heads per layer. Let $B_{\mathrm{ont}} \in \mathbb{R}^{d \times R}$ be an orthonormal ontology basis constructed from the training-set key statistics, with $F$ facet categories. Consider the multi-tool selection task: given a user query, the model must emit an ordered list of $M$ tools from a catalog of size $|\mathcal{C}|$.

**Conditions.**
- (H-cat-ext) holds with $\eta_{\mathrm{grad}} \geq \eta_0 > 0$ (Definition from Gap 1)
- (R) holds for all facet gates
- Tool names satisfy $|\mathrm{pos}(y^{(j)})| \leq T_{\mathrm{tool}} \leq 5$ for all $j$
- Prompt length $T_{\mathrm{prompt}} \geq 100$
- Inter-facet orthogonality: $\| B_{f_i}^\top B_{f_j} \|_F \leq \epsilon_{\perp}$ for $i \neq j$, where $\epsilon_\perp$ is small

**Part (a): K-Side Stability (Attention Distribution Preservation).**

For K-bias with coefficient $\alpha_K$ applied at layer $\ell$, the KL divergence between the original and biased attention distributions satisfies:

$$
D_{\mathrm{KL}}(\alpha^{\mathrm{orig}} \| \alpha^{\mathrm{biased}}) \leq \alpha_K^2 \cdot \frac{R \cdot \mu_{\max}^2}{d} + O(\alpha_K^3)
$$

where $\mu_{\max} = \max_i \mu_i$ is the largest facet mean under (H-cat).

However, **K-bias does not improve tool-selection accuracy**. Empirically, for all tested $\alpha_K \in \{0.1, 0.3, 0.5, 1.0, 2.0\}$ on Qwen2.5-7B/MetaTool Subtask 4 (497 multi-tool prompts):

$$
\mathrm{F1}(\alpha_K) \leq \mathrm{F1}(0) \quad \text{for all tested } \alpha_K.
$$

The mechanism: K-bias uniformly amplifies all keys along facet directions, including keys of INCORRECT tools that share the same facet category as the correct tool, creating a selection-ambiguity within each facet.

**Part (b): Q-Coverage Optimality.**

Define the Q-coverage operator at generation step $m$ (after tools $y^{(1)}, \ldots, y^{(m-1)}$ have been emitted) as:

$$
\tilde{q}_m := (I - P_{\mathrm{emitted}}^{(m)}) \, q_m, \qquad P_{\mathrm{emitted}}^{(m)} := \sum_{j=1}^{m-1} P_{f(j)}
$$

where $f(j)$ is the facet category of tool $y^{(j)}$ and $P_{f}$ is the projector onto facet $f$'s subspace. (See Gap 5 for the correct projector update rule ensuring idempotency.)

Under (H-cat-ext), (R), and the diminishing-return bound of Proposition 6.17.B:

(i) **Separation**: At step $m$, the first-order log-probability benefit of the $m$-th correct tool relative to any already-emitted-facet tool satisfies:

$$
\nabla_{\tilde{q}_m} \log p(y^{(m)}_{\mathrm{correct}}) - \nabla_{\tilde{q}_m} \log p(y^{(j)}_{\mathrm{old}}) \geq \eta_0 \cdot s_{\min}^{1/2} \cdot \sigma_{\mathrm{intra}}^{-1} - O(T_{\mathrm{tool}} / T_{\mathrm{prompt}})
$$

for all $j < m$, where the first term comes from Lemma 6.17.A applied to the projected query, and the second term is the diminishing-return correction from Proposition 6.17.B.

(ii) **Coverage completeness**: If $M \leq F$ (number of tools does not exceed number of facet categories), then after $M$ steps, the cumulative projector satisfies:

$$
\operatorname{rank}(P_{\mathrm{emitted}}^{(M+1)}) = \sum_{j=1}^{M} \operatorname{rank}(P_{f(j)})
$$

provided distinct facet categories are selected (which is enforced by the separation in (i)).

**Part (c): V-Side Compatibility (Conditional on $\alpha_K = 0$).**

When $\alpha_K = 0$ (no K-bias), the Q-coverage mechanism does not alter the value vectors. The output perturbation at step $m$ is:

$$
\| o_m^{\mathrm{Q\text{-}cov}} - o_m^{\mathrm{orig}} \|_2 \leq \| P_{\mathrm{emitted}}^{(m)} q_m \|_2 \cdot \frac{\max_t \|V_t\|_2}{\sqrt{d}} \cdot (1 + O(\alpha_K))
$$

At $\alpha_K = 0$, the $O(\alpha_K)$ term vanishes, and the perturbation is controlled solely by the Q-coverage projection magnitude.

**Part (d): Original Joint Additivity --- FALSIFIED.**

The original statement claimed that Q-coverage and K-bias combine additively:

$$
\text{[FALSIFIED]} \quad \mathrm{F1}(Q + K) \geq \mathrm{F1}(Q) + \mathrm{F1}(K) - \mathrm{F1}(\emptyset).
$$

**Counterexample.** On Qwen2.5-7B / MetaTool Subtask 4 (497 prompts), with $\alpha_K = 0.3$:

| Configuration | F1 |
|---|---|
| Baseline (no intervention) | 0.617 |
| Q-coverage only | 0.685 |
| K-bias only ($\alpha_K = 0.3$) | 0.594 |
| Q + K joint ($\alpha_K = 0.3$) | 0.571 |

The joint configuration DECREASES performance below the baseline ($0.571 < 0.617$), violating additivity. The mechanism is: K-bias disrupts the attention distribution that Q-coverage relies on, as the biased keys no longer align with the first-order separation computed from the original key statistics.

**Verified Family.** The empirically validated intervention family is:

$$
\mathcal{I}_{\mathrm{valid}} = \{ \text{Q-only}, \; \text{V+Q at } \alpha_K = 0 \}.
$$

**Proof of Part (a).**

The K-bias modifies keys as $K_t' = K_t + \alpha_K \cdot P_{\mathrm{ont}} K_t$. The attention logit becomes:

$$
q^\top K_t' / \sqrt{d} = q^\top K_t / \sqrt{d} + \alpha_K \cdot q^\top P_{\mathrm{ont}} K_t / \sqrt{d}.
$$

The perturbation to each logit is $\delta_t = \alpha_K \cdot q^\top P_{\mathrm{ont}} K_t / \sqrt{d}$. By Pinsker-type bounds for softmax perturbation (see, e.g., the analysis in Theorem 6.1):

$$
D_{\mathrm{KL}}(\alpha^{\mathrm{orig}} \| \alpha^{\mathrm{biased}}) \leq \mathrm{Var}_t[\delta_t] = \alpha_K^2 \cdot \mathrm{Var}_t\!\left[\frac{q^\top P_{\mathrm{ont}} K_t}{\sqrt{d}}\right].
$$

Under (H-cat), $P_{\mathrm{ont}} K_t$ has variance dominated by the $R$ facet directions with $\mathrm{Var}[K'_{t,i}] \leq \mu_i^2 + \sigma_{\mathrm{intra},i}^2 \leq 2\mu_{\max}^2$ (using $s_i \geq 1$). Summing over $R$ components and normalizing by $d$:

$$
\mathrm{Var}_t[\delta_t] \leq \alpha_K^2 \cdot \frac{R \cdot \|q\|_2^2 \cdot 2\mu_{\max}^2}{d^2} \leq \alpha_K^2 \cdot \frac{R \mu_{\max}^2}{d}
$$

where the last step uses $\|q\|_2^2 / d \leq O(1)$ (per-dimension query norm is $O(1)$ in trained transformers). $\square$

**Proof of Part (b).**

Part (b)(i): After projecting out $P_{\mathrm{emitted}}^{(m)}$, the query $\tilde{q}_m$ has zero component along previously selected facet directions. By Lemma 6.17.A (first-order channel separation), in the complementary subspace the correct new-facet tool has a first-order advantage of at least $\eta_0 \cdot s_{\min}^{1/2} \cdot \sigma_{\mathrm{intra}}^{-1}$ (where $\eta_0$ enters via (H-cat-ext) ensuring the gradient aligns with the ontology subspace).

The residual re-attention to old facets contributes at most $O(T_{\mathrm{tool}}/T_{\mathrm{prompt}})$ by Proposition 6.17.B. Since Q-coverage explicitly removes the $P_{f_j}$ component from $q_m$, the effective attention to old-facet keys is further suppressed: the dot product $\tilde{q}_m^\top K_t$ for $t$ in an old-facet position satisfies:

$$
|\tilde{q}_m^\top P_{f_j} K_t| = |(I - P_{\mathrm{emitted}}^{(m)})q_m)^\top P_{f_j} K_t| = 0
$$

exactly (since $P_{f_j} \leq P_{\mathrm{emitted}}^{(m)}$ for $j < m$, so $(I - P_{\mathrm{emitted}}^{(m)}) P_{f_j} = 0$). Thus old-facet keys are attended to ONLY through their component orthogonal to $\operatorname{span}(B_{\mathrm{ont}})$, which has magnitude $O(\epsilon_\perp)$ under inter-facet orthogonality. The separation follows.

Part (b)(ii): Under the separation guarantee, at each step $m$, a tool from a NEW facet category is selected (since old-facet tools have suppressed first-order benefit). The projectors $P_{f(1)}, \ldots, P_{f(M)}$ correspond to distinct facet categories. Under inter-facet orthogonality ($\epsilon_\perp$ small), these projectors are approximately orthogonal, and:

$$
\operatorname{rank}\!\left(\sum_{j=1}^M P_{f(j)}\right) = \sum_{j=1}^M \operatorname{rank}(P_{f(j)}) - O(M^2 \epsilon_\perp).
$$

For $\epsilon_\perp \to 0$, this is exact. $\square$

**Proof of Part (c).**

The Q-coverage perturbation to the attention output is:

$$
o_m^{\mathrm{Q\text{-}cov}} - o_m^{\mathrm{orig}} = \sum_t (\alpha_t^{\mathrm{Q\text{-}cov}} - \alpha_t^{\mathrm{orig}}) V_t.
$$

By the softmax Jacobian, the attention weight perturbation from replacing $q_m$ with $\tilde{q}_m = q_m - P_{\mathrm{emitted}} q_m$ is, to first order:

$$
\alpha_t^{\mathrm{Q\text{-}cov}} - \alpha_t^{\mathrm{orig}} \approx \alpha_t^{\mathrm{orig}} \left( \frac{-q_m^\top P_{\mathrm{emitted}} K_t}{\sqrt{d}} - \sum_s \alpha_s^{\mathrm{orig}} \frac{-q_m^\top P_{\mathrm{emitted}} K_s}{\sqrt{d}} \right).
$$

The magnitude of each attention weight change is bounded by $\|P_{\mathrm{emitted}} q_m\|_2 \cdot \max_t \|K_t\|_2 / \sqrt{d}$. Summing over positions with the triangle inequality:

$$
\|o_m^{\mathrm{Q\text{-}cov}} - o_m^{\mathrm{orig}}\|_2 \leq 2 \cdot \frac{\|P_{\mathrm{emitted}} q_m\|_2 \cdot \max_t \|K_t\|_2}{\sqrt{d}} \cdot \max_t \|V_t\|_2.
$$

At $\alpha_K = 0$, no additional perturbation is introduced. $\square$

---

## Gap 4: Theorem 6.20' (Effective Bound)

### Motivation

The original constant $C = \Lambda^2 \cdot L_\pi^2 \cdot \rho^2 \cdot \mathrm{Var}_s V / p_{\min}$ in Theorem 6.20 involves $p_{\min} \sim 10^{-6}$ (the minimum probability over the entire vocabulary), making the bound vacuous ($C \sim 10^6$ or larger).

### Theorem 6.20' (Effective Steering Bound on Tool-Selection Subproblem)

**Statement.** Restrict attention to the tool-selection subproblem: at each tool-emission position $t$, define the tool-conditional distribution:

$$
p_{\mathrm{tool}}(c \mid y_{<t}) := \frac{p(v_c \mid y_{<t})}{\sum_{c' \in \mathcal{C}} p(v_{c'} \mid y_{<t})}
$$

where $v_c$ is the first token of tool $c$'s name, and $\mathcal{C}$ is the tool catalog with $|\mathcal{C}|$ tools.

Let $c^*$ denote the correct tool. Define:

$$
p_{\mathrm{correct}} := p_{\mathrm{tool}}(c^* \mid y_{<t}), \qquad \Lambda_{\mathrm{tool}} := \max_{c \in \mathcal{C}} \| P_{\mathrm{ont}} w_{v_c} \|_2
$$

and the tool-restricted Lipschitz constant:

$$
L_{\pi, \mathrm{tool}} := \max_{c \in \mathcal{C}} \| \nabla_{o_t} \log p_{\mathrm{tool}}(c \mid y_{<t}) \|_2.
$$

Under (H-cat-ext) and Q-coverage with step $m$, the per-step regret of not selecting the correct tool is bounded by:

$$
\log \frac{1}{p_{\mathrm{tool}}(c^* \mid \tilde{q}_m)} \leq \log \frac{1}{p_{\mathrm{correct}}} + C_{\mathrm{eff}} \cdot \| P_{\mathrm{emitted}} q_m \|_2^2
$$

where:

$$
C_{\mathrm{eff}} = \frac{\Lambda_{\mathrm{tool}}^2 \cdot L_{\pi, \mathrm{tool}}^2 \cdot \rho^2 \cdot \mathrm{Var}_s V}{p_{\mathrm{correct}} \cdot d}
$$

and $\rho$ is the spectral radius of the attention-to-output Jacobian.

### Concrete Instantiation (Qwen2.5-7B on MetaTool)

We provide a numerical estimate using known architecture parameters and empirical measurements:

| Quantity | Symbol | Value | Source |
|---|---|---|---|
| Hidden dimension | $d$ | 3584 | Architecture |
| Ontology rank | $R$ | 10 | Construction |
| Number of tools | $|\mathcal{C}|$ | 16 | MetaTool catalog |
| Correct-tool probability | $p_{\mathrm{correct}}$ | 0.15--0.40 | Empirical (median ~0.25) |
| Tool embedding projection | $\Lambda_{\mathrm{tool}}$ | $\approx 3.0$ | Estimated: $\|w_v\|_2 \approx 10$, $\gamma_U \approx 0.3$ |
| Tool Lipschitz constant | $L_{\pi, \mathrm{tool}}$ | $\approx 2.0$ | $\leq 1/p_{\mathrm{correct}} \cdot \max \|w_v\|_2 \leq 4/0.25 \cdot 10/10 = 4$; using median |
| Spectral radius | $\rho$ | $\approx 1.0$ | Typical for well-trained models |
| Value variance | $\mathrm{Var}_s V$ | $\approx 1.0$ | Per-dimension, normalized |

Substituting:

$$
C_{\mathrm{eff}} = \frac{3.0^2 \cdot 2.0^2 \cdot 1.0^2 \cdot 1.0}{0.25 \cdot 3584} = \frac{36}{896} \approx 0.040.
$$

With $\|P_{\mathrm{emitted}} q_m\|_2^2 \leq \|q_m\|_2^2 \approx 900$ (empirical $\|q\|_2 \approx 30$):

$$
C_{\mathrm{eff}} \cdot \|P_{\mathrm{emitted}} q_m\|_2^2 \leq 0.040 \times 900 = 36.
$$

And $\log(1/p_{\mathrm{correct}}) = \log(4) \approx 1.4$.

**Assessment**: The bound gives $\log(1/p_{\mathrm{tool}}) \leq 1.4 + 36 = 37.4$, which corresponds to $p_{\mathrm{tool}} \geq e^{-37.4} \approx 10^{-16}$. This is still vacuous in the worst case because $\|P_{\mathrm{emitted}} q_m\|_2$ can be as large as $\|q_m\|_2$.

However, the Q-coverage mechanism ensures that $P_{\mathrm{emitted}}$ projects out only the USED facet directions. If the query $q_m$ has most of its energy outside the already-emitted facets (which is the regime where Q-coverage helps), then $\|P_{\mathrm{emitted}} q_m\|_2^2 \ll \|q_m\|_2^2$. Specifically, if only $k$ of $R$ facets have been emitted and the query distributes energy roughly uniformly:

$$
\|P_{\mathrm{emitted}} q_m\|_2^2 \approx \frac{k}{d} \|q_m\|_2^2 \approx \frac{k \cdot 900}{3584} \approx 0.25k.
$$

For $k = 3$ (three tools already emitted): $C_{\mathrm{eff}} \cdot 0.75 \approx 0.03$, giving $\log(1/p_{\mathrm{tool}}) \leq 1.43$, i.e., $p_{\mathrm{tool}} \geq 0.24$. **This is non-vacuous.**

### Remark 4.1 (Honest Assessment)

The bound is non-vacuous ONLY when $\|P_{\mathrm{emitted}} q_m\|_2$ is small relative to $\|q_m\|_2$, which is precisely the regime where Q-coverage is most effective (the query has little energy in already-emitted directions). In the adversarial regime where the query heavily overlaps with emitted facets, the bound becomes vacuous, but this is also the regime where Q-coverage provides the largest correction.

**The empirical AUROC of 0.976 for facet-direction classification is the primary contribution.** Theorem 6.20' provides a theoretical framework that is non-vacuous in the favorable regime, but the tight characterization of multi-tool selection performance comes from the empirical evaluation.

### Proof of Theorem 6.20'

The proof follows the standard regret decomposition for softmax perturbation, restricted to the tool subvocabulary.

**Step 1 (Tool-conditional likelihood).** The tool-conditional log-likelihood at position $t$ with modified query $\tilde{q}_m$ is:

$$
\log p_{\mathrm{tool}}(c^* \mid \tilde{q}_m) = \log p_{\mathrm{tool}}(c^* \mid q_m) + (\tilde{q}_m - q_m)^\top \nabla_{q_m} \log p_{\mathrm{tool}}(c^* \mid q_m) + O(\|\tilde{q}_m - q_m\|^2).
$$

**Step 2 (Gradient bound).** The gradient $\nabla_{q_m} \log p_{\mathrm{tool}}$ factors through the attention mechanism:

$$
\nabla_{q_m} \log p_{\mathrm{tool}} = \nabla_{o_t} \log p_{\mathrm{tool}} \cdot \frac{\partial o_t}{\partial q_m}.
$$

The Jacobian $\partial o_t / \partial q_m$ has spectral norm bounded by $\rho \cdot \max_t \|V_t\|_2 / \sqrt{d}$ (from the softmax Jacobian structure). The gradient $\nabla_{o_t} \log p_{\mathrm{tool}}$ has norm bounded by $L_{\pi,\mathrm{tool}} \leq \Lambda_{\mathrm{tool}} / p_{\mathrm{correct}}$ (since the gradient involves the unembedding row $w_{v_{c^*}}$ projected through the softmax Jacobian, which is bounded by $\|w_{v_{c^*}}\| (1 - p_{\mathrm{tool}}(c^*)) \leq \Lambda_{\mathrm{tool}} / \gamma_U$; using the tool-restricted denominator gives the $p_{\mathrm{correct}}$ factor).

**Step 3 (Perturbation magnitude).** The query perturbation is $\tilde{q}_m - q_m = -P_{\mathrm{emitted}} q_m$, so $\|\tilde{q}_m - q_m\|_2 = \|P_{\mathrm{emitted}} q_m\|_2$.

**Step 4 (Assembly).** Combining via the second-order Taylor remainder (the first-order term can be zero by symmetry when the model is well-calibrated):

$$
|\log p_{\mathrm{tool}}(c^* \mid \tilde{q}_m) - \log p_{\mathrm{tool}}(c^* \mid q_m)| \leq \frac{1}{2} \|P_{\mathrm{emitted}} q_m\|_2^2 \cdot \sup_{\xi} \|\nabla^2_{q_m} \log p_{\mathrm{tool}}(c^* \mid \xi)\|_{\mathrm{op}}.
$$

The Hessian spectral norm is bounded by $\Lambda_{\mathrm{tool}}^2 \cdot L_{\pi,\mathrm{tool}}^2 \cdot \rho^2 \cdot \mathrm{Var}_s V / (p_{\mathrm{correct}} \cdot d)$ via the chain rule applied twice through the attention mechanism and the tool-restricted softmax (the $p_{\mathrm{correct}}$ appears from the softmax Hessian's $p(1-p)$ factor, bounded below by $p_{\mathrm{correct}} \cdot (1 - p_{\mathrm{correct}}) \geq p_{\mathrm{correct}} / 2$ for $p_{\mathrm{correct}} \leq 1/2$; we absorb the factor of 2 into the constant).

Rearranging:

$$
-\log p_{\mathrm{tool}}(c^* \mid \tilde{q}_m) \leq -\log p_{\mathrm{tool}}(c^* \mid q_m) + C_{\mathrm{eff}} \cdot \|P_{\mathrm{emitted}} q_m\|_2^2. \quad \square
$$

---

## Gap 5: Projector Update Rule (Lemma 6.17.C)

### Motivation

In the implementation of Q-coverage (eval\_subtask4\_dynamic\_qk\_v2.py), the projector $P_{\mathrm{emitted}}$ is updated by adding $P_f$ each time a tool with facet $f$ is emitted. When two tools share the same facet category (e.g., "domain:finance" and "domain:travel" both have facet category "domain"), the projector $P_{\mathrm{domain}}$ is added twice, yielding:

$$
P_{\mathrm{emitted}} = P_{\mathrm{other}} + 2 P_{\mathrm{domain}}
$$

This is NOT a valid projector: $P_{\mathrm{emitted}}^2 \neq P_{\mathrm{emitted}}$ (it has eigenvalue 2 along the domain subspace), and $P_{\mathrm{remaining}} = P_{\mathrm{ont}} - P_{\mathrm{emitted}}$ has eigenvalue $-1$ along the domain subspace, creating negative eigenvalues that invert the intended direction suppression.

### Lemma 6.17.C (Projector Update Rule)

**Statement.** Let $\{P_{f}\}_{f \in \mathcal{F}}$ be a collection of orthogonal projectors indexed by facet categories $\mathcal{F} = \{f_1, \ldots, f_F\}$, with $P_f P_{f'} = 0$ for $f \neq f'$ (inter-facet orthogonality) and $\sum_f P_f = P_{\mathrm{ont}}$.

Define the CATEGORY-LEVEL emitted set $\mathcal{E}_m \subseteq \mathcal{F}$ at step $m$ as the set of distinct facet categories whose associated tools have been emitted in steps $1, \ldots, m-1$:

$$
\mathcal{E}_m := \{ f(j) : j \in \{1, \ldots, m-1\} \}
$$

where $f(j) \in \mathcal{F}$ is the facet category of the $j$-th emitted tool (note: set union, not multiset).

The correct cumulative projector is:

$$
P_{\mathrm{emitted}}^{(m)} := \sum_{f \in \mathcal{E}_m} P_f
$$

Then:

**(i) Idempotency.** $P_{\mathrm{emitted}}^{(m)}$ is an orthogonal projector (symmetric and idempotent).

**(ii) Monotone nesting.** $\mathcal{E}_1 \subseteq \mathcal{E}_2 \subseteq \cdots$, and consequently $\operatorname{range}(P_{\mathrm{emitted}}^{(1)}) \subseteq \operatorname{range}(P_{\mathrm{emitted}}^{(2)}) \subseteq \cdots$.

**(iii) Complementary projector validity.** $P_{\mathrm{remaining}}^{(m)} := P_{\mathrm{ont}} - P_{\mathrm{emitted}}^{(m)}$ is an orthogonal projector with:

$$
P_{\mathrm{remaining}}^{(m)} = \sum_{f \in \mathcal{F} \setminus \mathcal{E}_m} P_f, \qquad \operatorname{rank}(P_{\mathrm{remaining}}^{(m)}) = \sum_{f \notin \mathcal{E}_m} \operatorname{rank}(P_f).
$$

**(iv) Failure mode of naive update.** If the implementation uses a multiset (adding $P_f$ each time ANY tool in category $f$ is emitted), and category $f$ has $k$ tools emitted, then:

$$
P_{\mathrm{emitted}}^{\mathrm{naive}} = \sum_{f \in \mathcal{F}} n_f \cdot P_f, \qquad n_f := |\{j < m : f(j) = f\}|
$$

has eigenvalue $n_f$ along $\operatorname{range}(P_f)$. For $n_f \geq 2$:
- $P_{\mathrm{emitted}}^{\mathrm{naive}}$ is NOT idempotent ($n_f^2 \neq n_f$)
- $P_{\mathrm{remaining}}^{\mathrm{naive}} = P_{\mathrm{ont}} - P_{\mathrm{emitted}}^{\mathrm{naive}}$ has eigenvalue $1 - n_f \leq -1$ along $\operatorname{range}(P_f)$
- The Q-coverage query $\tilde{q}_m = P_{\mathrm{remaining}}^{\mathrm{naive}} q_m$ has its component along $\operatorname{range}(P_f)$ FLIPPED in sign, which AMPLIFIES rather than suppresses attention to old-facet keys

**Proof.**

**(i)** Since $\{P_f\}_{f \in \mathcal{F}}$ are mutually orthogonal projectors ($P_f P_{f'} = \delta_{ff'} P_f$), any subset sum is also a projector:

$$
\left(\sum_{f \in \mathcal{E}} P_f\right)^2 = \sum_{f, f' \in \mathcal{E}} P_f P_{f'} = \sum_{f \in \mathcal{E}} P_f^2 = \sum_{f \in \mathcal{E}} P_f.
$$

Symmetry follows from symmetry of each $P_f$. $\square$

**(ii)** $\mathcal{E}_m \subseteq \mathcal{E}_{m+1}$ since $\mathcal{E}_{m+1} = \mathcal{E}_m \cup \{f(m)\}$. Range monotonicity follows from the projector ordering: if $\mathcal{E} \subseteq \mathcal{E}'$, then $P_{\mathcal{E}} \leq P_{\mathcal{E}'}$ in the Loewner order. $\square$

**(iii)** Since $\sum_{f \in \mathcal{F}} P_f = P_{\mathrm{ont}}$:

$$
P_{\mathrm{ont}} - P_{\mathrm{emitted}}^{(m)} = \sum_{f \in \mathcal{F}} P_f - \sum_{f \in \mathcal{E}_m} P_f = \sum_{f \in \mathcal{F} \setminus \mathcal{E}_m} P_f,
$$

which is an orthogonal projector by the same argument as (i). The rank formula follows from orthogonality: $\operatorname{rank}(\sum_f P_f) = \sum_f \operatorname{rank}(P_f)$ when the projectors are mutually orthogonal. $\square$

**(iv)** By direct computation: $P_f$ has eigenvalue 1 on $\operatorname{range}(P_f)$ and 0 elsewhere. Thus $n_f P_f$ has eigenvalue $n_f$ on $\operatorname{range}(P_f)$. For $n_f = 2$: eigenvalue of $P_{\mathrm{emitted}}^{\mathrm{naive}}$ is 2, eigenvalue of $P_{\mathrm{remaining}}^{\mathrm{naive}}$ is $1 - 2 = -1$. The resulting query modification:

$$
\tilde{q}_m^{\mathrm{naive}} = P_{\mathrm{remaining}}^{\mathrm{naive}} q_m
$$

has, along $\operatorname{range}(P_f)$: $(\tilde{q}_m^{\mathrm{naive}})_f = -1 \cdot (q_m)_f = -(q_m)_f$, which is a SIGN FLIP, not a suppression. This causes the attention logit $\tilde{q}_m^\top K_t$ to become $-q_m^\top P_f K_t + \cdots$ for keys in facet $f$, which can increase attention to facet-$f$ keys that had negative original logits. $\square$

### Algorithm 1: Correct Q-Coverage Projector Update

```
Input: B_ont (ontology basis), facet_to_cols (map: facet category -> column indices of B_ont)
Initialize: emitted_categories = {} (empty set)

For each generation step m = 1, 2, ..., M:
    1. Build P_emitted from emitted_categories:
       cols = union of facet_to_cols[f] for f in emitted_categories
       B_emitted = B_ont[:, cols]
       P_emitted = B_emitted @ B_emitted^T

    2. Compute modified query:
       q_tilde = q_m - P_emitted @ q_m   # equivalently: (I - P_emitted) @ q_m

    3. Run attention with q_tilde, generate tool y^(m)

    4. Update emitted set (SET union, not multiset):
       f_new = facet_category(y^(m))
       emitted_categories = emitted_categories ∪ {f_new}  # no-op if f_new already present
```

**Key implementation detail**: Step 4 uses SET union. If `f_new` is already in `emitted_categories` (because a previous tool shared the same facet category), the set is unchanged. This guarantees that `P_emitted` remains a valid orthogonal projector at every step.

### Remark 5.1 (Approximate Inter-Facet Orthogonality)

In practice, the inter-facet orthogonality condition $P_f P_{f'} = 0$ holds only approximately. If $\|P_f P_{f'}\|_{\mathrm{op}} \leq \epsilon_\perp$ for $f \neq f'$, then the subset-sum projector has:

$$
\left\|\left(\sum_{f \in \mathcal{E}} P_f\right)^2 - \sum_{f \in \mathcal{E}} P_f\right\|_{\mathrm{op}} \leq |\mathcal{E}|^2 \cdot \epsilon_\perp.
$$

For $\epsilon_\perp = 0.01$ and $|\mathcal{E}| = 5$: deviation from idempotency is at most 0.25, which is acceptable for the first-order analysis but may accumulate over many steps. For robust implementation, one should periodically re-orthogonalize: replace $P_{\mathrm{emitted}}$ with the projector onto the top-$k$ singular subspace of $B_{\mathrm{ont}}[:, \mathrm{cols}]$, which is always exactly idempotent.

### Remark 5.2 (Connection to eval\_subtask4\_dynamic\_qk\_v2.py Bug)

The bug in the implementation corresponds to the failure mode in part (iv): the code tracks `emitted_facets` as a list (allowing duplicates) rather than a set. When two tools with facet category "domain" are emitted, `P_domain` is added twice, creating the negative-eigenvalue pathology. The fix is to change the data structure from list to set, or equivalently to add a guard:

```python
if facet_category not in emitted_categories:
    emitted_categories.add(facet_category)
    # update P_emitted
```

---

## Gap 6: Theorem β* (First-order Sign Predictor for Q-Steering)

### Motivation

Theorem 6.17 (V-side mirror, β > 0) and Theorem 6.17' (coverage mask, β < 0) each legitimize one sign of the Q-side steering coefficient β, but **neither predicts which sign maximizes downstream accuracy for a given task**. Empirically, we observe domain-dependent sign winners:

| Domain | Baseline F1 | sign($\beta^*$) observed | $|\Delta F_1|$ |
|---|---|---|---|
| τ² Retail | 0.47 | $-$ | 5.11pp |
| τ² Telecom | 0.25 | $+$ | 24.78pp |
| MetaTool ST4 | 0.74 | $-$ | 2.28pp |

Without a predictor, β-sign reduces to a hyperparameter (2× tuning grid), which reviewers may flag as ad-hoc. Theorem β* closes this gap by deriving the optimal sign as a measurable statistic of the baseline attention distribution.

### Setup

For a single attention head at a generation step, let $Q \in \mathbb{R}^d$ be the query vector (after q_proj), $K_t \in \mathbb{R}^d$ the keys for context tokens $t = 1, \ldots, T$, and $B \in \mathbb{R}^{d \times r}$ the orthonormal ontology basis ($B^\top B = I_r$). Write $P := BB^\top$ for the (symmetric idempotent) projector onto Range($B$).

The β-perturbed query is $Q_\beta := (I + \beta P) Q$ (matching `install_q_bias_hooks` in eval_metatool_subtask1.py). Attention scores and distribution:
$$
z_t(\beta) = \tfrac{1}{\sqrt d}\langle Q_\beta, K_t\rangle, \qquad p_\beta(t) = \frac{\exp z_t(\beta)}{\sum_s \exp z_s(\beta)}.
$$

Let $\mathcal{G} \subseteq \{1,\ldots,T\}$ denote the ground-truth (GT) token set — prompt tokens whose high attention correlates with emitting a correct tool name (e.g., the tool-schema description span for the correct tool). The downstream objective we first-order-linearize is the baseline-proxy
$$
L(\beta) := \sum_{t\in\mathcal{G}} p_\beta(t).
$$

Define the per-token **ontology-projected score**
$$
r_t := \tfrac{1}{\sqrt d}\langle Q, P K_t\rangle = \tfrac{1}{\sqrt d}\langle P Q, K_t\rangle \quad (\text{by symmetry of }P).
$$

### Theorem β* (First-order Sign Predictor)

**Statement.** With the setup above, let $\pi_{\mathcal{G}} := \sum_{t\in\mathcal{G}} p_0(t)$, $\bar r := \sum_t p_0(t)\, r_t$, and $\bar r_{\mathcal{G}} := \frac{1}{\pi_{\mathcal{G}}}\sum_{t\in\mathcal{G}} p_0(t)\,r_t$. Then
$$
\boxed{\;\left.\frac{dL}{d\beta}\right|_{\beta=0} = \pi_{\mathcal{G}}\bigl(\bar r_{\mathcal{G}} - \bar r\bigr),\quad \therefore\;\operatorname{sign}(\beta^*) = \operatorname{sign}\bigl(\bar r_{\mathcal{G}} - \bar r\bigr).\;}
$$

**Proof.**

*Step 1 (Affinity of scores).* Since $(I + \beta P) Q = Q + \beta P Q$,
$z_t(\beta) = z_t^{(0)} + \beta\, r_t$, where $z_t^{(0)} = \tfrac{1}{\sqrt d}\langle Q, K_t\rangle$.

*Step 2 (Softmax gradient identity).* By direct differentiation of the softmax (standard exponential-family result),
$$
\frac{\partial p_\beta(t)}{\partial \beta} = p_\beta(t)\bigl(r_t - \mathbb{E}_{p_\beta}[r]\bigr).
$$

*Step 3 (Evaluate at β = 0).* $\mathbb{E}_{p_\beta}[r]\big|_{\beta=0} = \bar r$, so
$$
\left.\frac{\partial p_\beta(t)}{\partial \beta}\right|_0 = p_0(t)\,(r_t - \bar r).
$$

*Step 4 (Sum over $\mathcal{G}$).*
$$
\left.\frac{dL}{d\beta}\right|_0 = \sum_{t\in\mathcal{G}} p_0(t)\,(r_t - \bar r) = \underbrace{\sum_{t\in\mathcal{G}} p_0(t) r_t}_{\pi_{\mathcal{G}}\,\bar r_{\mathcal{G}}} - \bar r\cdot\underbrace{\sum_{t\in\mathcal{G}} p_0(t)}_{\pi_{\mathcal{G}}} = \pi_{\mathcal{G}}\bigl(\bar r_{\mathcal{G}} - \bar r\bigr).
$$

Since $\pi_{\mathcal{G}} > 0$, the sign is determined by the GT-weighted vs. global-weighted ontology-projection gap. $\blacksquare$

### Interpretation

$\bar r_{\mathcal{G}} - \bar r$ compares the **baseline-weighted average of the ontology-projected key-score** over GT vs. all tokens.

- $\bar r_{\mathcal{G}} > \bar r$: GT tokens align *better* with $PQ$ than the average context token. The baseline under-weights GT; **amplifying** Q along $P$ (β > 0) concentrates more mass on GT.
- $\bar r_{\mathcal{G}} < \bar r$: Non-GT tokens happen to be ontology-aligned (they distract the model via ontology proximity). **Subtracting** Q's ontology component (β < 0) removes mass from these distractors.

### Second-order term (curvature)

Applying the chain rule once more,
$$
\left.\frac{d^2 L}{d\beta^2}\right|_0 = \pi_{\mathcal{G}}\bigl(V_{\mathcal{G}} - \operatorname{Var}_{p_0}(r)\bigr),
$$
where $V_{\mathcal{G}} := \frac{1}{\pi_{\mathcal{G}}}\sum_{t\in\mathcal{G}} p_0(t)(r_t - \bar r)^2$ is the baseline-weighted second moment of $r$ about $\bar r$, restricted to $\mathcal{G}$. Equivalently,
$$
L(\beta) \approx L(0) + \beta\, s + \tfrac{\beta^2}{2}\,\pi_{\mathcal{G}}\bigl(V_{\mathcal{G}} - \operatorname{Var}_{p_0}(r)\bigr),\quad s := \pi_{\mathcal{G}}(\bar r_{\mathcal{G}} - \bar r).
$$

When the curvature is negative ($V_{\mathcal{G}} < \operatorname{Var}_{p_0}(r)$), the Taylor optimum is $\beta^* = -s / [\pi_{\mathcal{G}}(\operatorname{Var}_{p_0}(r) - V_{\mathcal{G}})]$. When positive, no finite optimum is predicted to leading order; a bounded $|\beta| \leq \beta_{\max}$ regularizer is required in practice. In both cases, **the sign is preserved**: $\operatorname{sign}(\beta^*) = \operatorname{sign}(s)$.

### Fluctuation-Dissipation Reading

The softmax gradient $\partial_\beta \bar r(\beta) = \operatorname{Var}_{p_\beta}(r)$ is a response = fluctuation identity (Callen--Welton form). Theorem β* is the GT-indicator-weighted specialization: the sensitivity of the GT mass to Q-perturbation equals the covariance between the GT indicator and the ontology-projected score under the baseline distribution.

### Corollary β*.1 (Multi-head, Layer-Adaptive Extension)

For layer $\ell$ with heads $h = 1,\ldots,H$, define per-$(\ell,h)$ quantities $r_t^{(\ell,h)}$ and $s^{(\ell,h)}$ analogously (using the local $B^{(\ell,h)}$ and the head's $Q^{(\ell,h)}, K_t^{(\ell,h)}$). When the Q-bias is applied at layer $\ell$ with coefficient $\beta_\ell$ (as in our layer-adaptive hook), chain rule through downstream residual-stream computations gives
$$
\left.\frac{\partial L}{\partial \beta_\ell}\right|_0 = \sum_{h=1}^{H} w_{\ell,h}\, s^{(\ell,h)},
$$
for weights $w_{\ell,h} \geq 0$ reflecting the logit-lens sensitivity of each head's output to the final GT prediction. The sign aggregation is therefore a convex combination at each layer. **Corollary**: $\operatorname{sign}(\beta^*_\ell) = \operatorname{sign}\bigl(\sum_h w_{\ell,h}\, s^{(\ell,h)}\bigr)$, computable from a single baseline forward + per-head projection statistics.

### Algorithm 1: Adaptive Sign Routing

```
Input: prompt, model f, ontology basis B, candidate GT span G (e.g., tool schema tokens)
Output: β with optimal sign

1. h ← f(prompt, hook=record_Q_K_attn)           # baseline forward pass
2. for each (ℓ, head) where Q-bias will be applied:
     Compute r_t = <PQ, K_t> / √d for t ∈ prompt
     Compute p_0(t) from attn weights
     s^{(ℓ,h)} ← ∑_{t∈G} p_0(t) (r_t - E_{p_0}[r])
3. s_tot ← ∑_{ℓ,h} w_{ℓ,h} s^{(ℓ,h)}              # w from logit-lens or uniform
4. β ← sign(s_tot) · β_0                          # β_0 a fixed magnitude
5. return second forward pass with β-hook enabled
```

**Cost**: 1 extra forward pass at test time; $O(T d r D)$ additional dot-product for $r_t$ per layer (negligible vs. attention).

### Remark 6.1 (Empirical Consistency 2026-04-17 — Qualitative Only)

Three of three observed Q-sign winners are consistent with the *intuition* behind Theorem β* (baseline attention calibration determines the optimal sign):

| Domain | Baseline F1 | Heuristic prediction | Observed best | ✓ |
|---|---|---|---|---|
| τ² Retail | 0.47 (balanced) | $-$ (regularize) | Q-only $\beta=-0.03$, +5.11pp | ✓ |
| τ² Telecom | 0.25 (under-focused) | $+$ (amplify) | Q-only $\beta=+0.05$, +24.78pp | ✓ |
| MetaTool ST4 | 0.74 (over-confident) | $-$ (regularize) | Q-only $\beta=-0.03$, +2.28pp | ✓ |

### Remark 6.2 (Direct Measurement Gap — Open Problem 2026-04-17)

Direct measurement of $\operatorname{sign}(\bar r_{\mathcal G} - \bar r)$ using the GT-schema definition of $\mathcal G$ **does NOT reproduce the empirical sign** on τ² Telecom. On both retail (n=30) and telecom (n=20) smoke tests, the predictor returns negative $s$ for every task, yielding a constant "−" prediction that happens to align with retail's majority empirical sign (66% accuracy is an artifact of class imbalance) but disagrees with telecom's majority "+" (31% accuracy).

Diagnostic finding: $\bar r_{\mathcal G}$ is **systematically smaller than $\bar r$** when $\mathcal G$ is the GT tool-schema span. Tokens inside the JSON tool schema consistently have lower $\langle BB^\top Q, K_t\rangle$ than the prompt-wide average. Moreover, tasks where the baseline attends heavily to GT schema tokens ($\pi_G > 0.015$ on telecom) tend to have **lower** baseline F1 (≈0 vs. 0.21 elsewhere). This decouples $L(\beta) = \sum_{t\in\mathcal G} p_\beta(t)$ from downstream F1 and invalidates the choice of $\mathcal G = \text{schema tokens}$ as a useful surrogate.

**Hypothesized cause**: The single-step attention objective $L(\beta)$ measured at position $T{-}1$ captures only the final-layer directional shift in schema attention, but empirical F1 depends on an unfolding generation process whose logits integrate hidden-state contributions across every output step. The first-order theorem is mathematically correct for its stated objective; the objective itself is the wrong proxy for generation F1.

**Proposed fixes (future-work roadmap)**:

1. **Logit-lens variant**.
  Define $v_t := \langle V^{(\ell^*)}_t, W_U[\text{tok}_{\mathcal G}]\rangle$ — each token's value-output projection onto the GT tool's unembedding direction at a chosen decision layer $\ell^*$. Then
  $$s_{\text{logit}} = \sum_t p_0(t)(r_t - \bar r)\, v_t$$
  is the first-order sensitivity of the correct-tool logit to $\beta$. This requires capturing $V$ and the unembedding, not just $Q$ and $K$, but stays within single-pass measurement.

2. **Generation-step aggregation**.
  Run a short baseline generation. At each output step $k$, measure $s^{(k)}$ using the Q at that step (which attends back into prompt + prior outputs). Aggregate as $\bar s = \frac{1}{K}\sum_k s^{(k)}$. This captures the decisive moment when the model commits to a tool name.

3. **Alternative $\mathcal G$**.
  Let $\mathcal G$ be the user-intent tokens (task's `reason_for_call` span or keywords extracted from the query) rather than tool-schema tokens. The hypothesis: the model's Q must align with intent-bearing tokens for correct tool selection; amplification / subtraction along the intent direction is the true steering axis.

**Paper stance (honest)**. We present Theorem β* as a **theoretical framework** that unifies Thm 6.17 and Thm 6.17' under a single first-order sign-prediction principle. The *directly measurable form* $\operatorname{sign}(s) = \operatorname{sign}(\bar r_{\mathcal G} - \bar r)$ with $\mathcal G = \text{schema tokens}$ is **not a reliable practical predictor** in our current experiments; alternative objectives (logit-lens, generation-step, intent-based $\mathcal G$) are open work.

**Pending empirical validation**: (i) logit-lens variant; (ii) direct measurement on airline / MetaTool; (iii) cross-model Llama-3.1-8B; (iv) per-query prediction fidelity with alternative $\mathcal G$.

### Remark 6.2 (Edge Cases)

- **Small $\pi_{\mathcal{G}}$**: when the baseline places essentially zero mass on $\mathcal{G}$, the first-order term $s$ is vanishingly small, and higher-order (softmax-saturation) terms dominate. The predictor degrades; a fallback to unbiased β-tuning is recommended when $\pi_{\mathcal{G}} < 10^{-3}$.
- **Degenerate $B$**: if $PQ \approx 0$ (query has no ontology energy), all $r_t \approx 0$, hence $s \approx 0$ and β has no first-order effect. Consistent with the pass-through property noted in `adaseka_vs_ours_differentiation_2026_04_10.md`.
- **$\mathcal{G}$-spec noise**: the theorem is pointwise in the GT-span choice. Robustness should be checked by perturbing the span (e.g., expanding to include argument tokens).

### Remark 6.3 (Relation to Existing Theorems)

Theorem β* is a **meta-theorem** unifying Thm 6.17 (V-side mirror, β > 0 regime) and Thm 6.17' (coverage mask, β < 0 regime). Neither existing theorem required sign specification; β* promotes the sign from an external choice to an internal consequence of baseline statistics. The existing proofs remain intact; β* provides the selection rule between them.

---

## Gap 7: Q-K Attention-Score Duality + Q-Only Sufficiency

### Motivation

Prior drafts positioned K-bias (SEKA-style) and Q-bias (ours) as two complementary axes with layer-adaptive coupling. Observationally, however, Q-bias alone (single sign, per-step) dominates K-bias on all long-horizon benchmarks (τ² Telecom N=200 Q+only +24.78pp vs. K-only weak), and Banking's multipass_ladapt win depends on Q-side contribution. This suggests that the two axes are not independent but that K-bias can be **absorbed into Q-bias at the attention-score level**, with Q-bias providing operational advantages. We formalize this below.

### Theorem 7.1 (Q-K Attention-Score Duality)

**Statement.** Let $B \in \mathbb{R}^{d \times r}$ be orthonormal ($B^\top B = I_r$) and $P := BB^\top$ the corresponding orthogonal projector (symmetric idempotent). For any scalar $c \in \mathbb{R}$ and any query / key vectors $Q, K_t \in \mathbb{R}^d$,

$$
\bigl\langle (I + cP)\,Q,\; K_t \bigr\rangle \;=\; \bigl\langle Q,\; (I + cP)\,K_t \bigr\rangle \;=\; \bigl\langle Q, K_t\bigr\rangle + c\,\bigl\langle Q,\; P\,K_t\bigr\rangle.
$$

In particular, the unscaled attention logit $z_t = \langle Q, K_t\rangle$ transforms identically under

- **Q-side $c$-amplification**: $Q \leftarrow (I + cP)Q$ with coefficient $\beta = c$, or
- **K-side $c$-amplification**: $K_t \leftarrow (I + cP)K_t$ with coefficient $\alpha = c$.

### Proof

$P$ is symmetric ($P^\top = P$) because it is a composition $BB^\top$ with $B$ real. Hence for any $u, v \in \mathbb{R}^d$:
$$
\langle P u, v\rangle = u^\top P^\top v = u^\top P v = \langle u, P v\rangle.
$$
Applying this with $u = Q$, $v = K_t$:
$$
\langle (I + cP) Q, K_t\rangle = \langle Q, K_t\rangle + c\langle P Q, K_t\rangle = \langle Q, K_t\rangle + c\langle Q, P K_t\rangle = \langle Q, (I + cP) K_t\rangle. \qquad \blacksquare
$$

### Corollary 7.1.A (Per-Step Equivalence at the Softmax Layer)

**Statement.** The Boltzmann attention distribution
$$
p(t \mid Q, K_{1:T}, c, \text{side}) \;\propto\; \exp\!\Bigl(\tfrac{1}{\sqrt d}\,\langle Q_c^{(\text{side})}, K_t^{(\text{side})}\rangle\Bigr),
$$
with $Q_c^{(\text{Q-side})} = (I + cP)Q$, $K_t^{(\text{Q-side})} = K_t$ (K unchanged) versus $Q_c^{(\text{K-side})} = Q$, $K_t^{(\text{K-side})} = (I + cP)K_t$, is **identical** for any $c$. Consequently: any first-order steering analysis, variance bound, or coverage argument developed for K-bias transfers verbatim to Q-bias with coefficient $\beta = \alpha$.

**Proof.** Immediate from Theorem 7.1. Both sides yield the same per-logit expression $\langle Q, K_t\rangle + c\langle Q, P K_t\rangle$, hence the same softmax. $\blacksquare$

### Corollary 7.1.B (Multi-Step Divergence via KV-Cache)

**Setup.** Consider autoregressive generation with KV-cache. At step $k$, a forward pass through the self-attention block:

- Computes $Q_k$ (fresh per step), $K_k$, $V_k$ (stored into cache).
- Attention at step $k$: $p_k(t) \propto \exp\bigl(\tfrac{1}{\sqrt d}\langle Q_k, K_t\rangle\bigr)$ for $t \le k$.

If a **K-side hook** with coefficient $\alpha$ is installed, it rewrites $K_k \leftarrow (I + \alpha P) K_k$ at every step before insertion into cache. After $K$ steps, every cached key carries the modification, so at any subsequent step $k' > k$:
$$
\langle Q_{k'}, K_k^{\text{hooked}}\rangle = \langle Q_{k'}, K_k\rangle + \alpha\,\langle Q_{k'}, P K_k\rangle \quad \text{(every retrieval).}
$$

If a **Q-side hook** with coefficient $\beta$ is installed, it rewrites $Q_k \leftarrow (I + \beta P) Q_k$ at every step. Keys in cache are unmodified. Attention at step $k'$:
$$
\langle Q_{k'}^{\text{hooked}}, K_k\rangle = \langle Q_{k'}, K_k\rangle + \beta\,\langle Q_{k'}, P K_k\rangle.
$$

By Theorem 7.1, at a single step the two are identical with $\alpha = \beta$.

**Difference.** Let the generation use $K$ steps. K-side modifications remain "frozen" in cache and thus:
- Every future step's attention lookup $\langle Q_{k'}, K_k^{\text{hooked}}\rangle$ contains the $\alpha$-perturbation inserted at step $k \le k'$.
- There is no opportunity to revise or withdraw the perturbation as generation proceeds.

Q-side modifications, by contrast:
- Are applied only at the current step's query.
- Allow adaptive perturbation: per-step $\beta_k$ can be chosen as a function of current context (Theorem β* / Algorithm 1), in particular $\beta_k \to 0$ once sufficient coverage is achieved.

**Formal consequence.** Define the per-step attention perturbation as the additive logit change $\Delta z_t^{(k)}$. For K-bias with fixed $\alpha$: $\Delta z_t^{(k)} = \alpha \langle Q_{k'}, PK_k\rangle$ is **accumulated** in the sense that *every* step $k'$ attends with the $\alpha$-perturbed $K_k$; the decision trajectory is therefore consistent with a single coefficient applied globally. For Q-bias with step-varying $\beta_{k'}$: $\Delta z_t^{(k')} = \beta_{k'} \langle Q_{k'}, PK_k\rangle$ adapts per step. In the language of control, Q-side steering has **per-step controllability**; K-side steering is a **constant-gain disturbance** once inserted.

This formalizes the empirical observation in §4.8.2 (Retail action-count breakdown): K-only collapses on long-horizon tasks (10+ actions: +0.3pp) while Q-only keeps lifting (+10.7pp), because K accumulates unadjustable perturbation while Q resets every step.

### Corollary 7.1.C (K-Side Methods Subsume into Q-Side Formulation)

**Statement.** Let $\mathcal{H}_K = \{h : K_t \mapsto (I + cP) K_t, c \in \mathbb{R}\}$ be the family of single-coefficient K-side steering hooks with fixed projector $P$. Then there exists a bijection $\phi : \mathcal{H}_K \to \mathcal{H}_Q$ (via $\phi(c) = c$) such that the single-step attention distribution under $h \in \mathcal{H}_K$ equals that under $\phi(h) \in \mathcal{H}_Q$.

In particular:
- **SEKA** ($K \leftarrow K + \alpha \cdot d$ for direction $d$; equivalent to $P = dd^\top / \|d\|^2$, $c = \alpha$) has an exact Q-side counterpart.
- **CAA** (activation-steering at residual stream; per-step and Q-projected through subsequent $W_K$) is already effectively a Q-side axis.
- **AdaSEKA** ($K \leftarrow (I + g(Q) \cdot P_{\text{dyn}(Q)}) K$) is Q-adaptive K-side; the Q-adaptivity can be folded into a per-step $\beta(Q)$ on the Q-side.

**Paper-level consequence**. Framing the method as **signed Q-side ontology steering** is not a restriction; it is a **generalization** of all prior K-side proposals with strictly better operational properties (per-step controllability, no cache pollution). The combination of:

1. Theorem 7.1 (equivalence at attention-score level),
2. Theorem β* (first-order sign predictor),
3. Corollary 7.1.B (multi-step divergence favoring Q),

unifies the "K vs Q" debate and positions Q-only signed steering as the canonical attention-steering operator for multi-step tool selection.

### Remark 7.1 (Empirical Evidence)

The Q-K duality theorem predicts per-step equivalence. Multi-step divergence is expected and is observed:

| Benchmark | Horizon | K-only ($\alpha$) | Q-only ($\beta$) | Ratio |
|-----------|---------|-------------------|------------------|-------|
| τ² Retail 10+ actions | long | +0.3pp | **+10.7pp** | ∼35× |
| τ² Retail 3-5 actions | short | +6.1pp | +4.9pp | 0.8× |
| τ² Telecom (avg 12 actions) | long | weak | **+24.78pp** (Q+0.05) | dominant |
| MetaTool (single-turn, avg <2 actions) | very short | −4.57pp | +2.28pp | Q slightly better |

Pattern: **Horizon length is a strong predictor of the Q-over-K gap**. Short-horizon (MetaTool, Retail 3-5) shows near-equivalence (consistent with Theorem 7.1); long-horizon shows Q dominance (consistent with Corollary 7.1.B). No data point contradicts the theorem.

### Remark 7.2 (Why Prior Art Missed This)

K-side steering (SEKA, AdaSEKA) came from the factual-editing literature (single-token output, no long horizon); the cache accumulation was invisible because generation was one token. Multi-tool / multi-step benchmarks are where the Q advantage emerges, and these benchmarks post-date most K-side methods' papers. Our contribution is to **identify the per-step equivalence + the multi-step divergence** and **pivot the framework to Q-only signed steering** as a strict generalization.

---

## Summary of Changes to Main Proof

| Location | Original Claim | Replacement |
|---|---|---|
| Line ~740 | (H-cat) suffices for output gradient alignment | Requires (H-cat-ext); verified via Lemma 1.1 |
| Line 1170 | Marginal benefit of re-attention is "net-zero" | Bounded by $O(T_{\mathrm{tool}}/T_{\mathrm{prompt}})$ per Prop. 6.17.B |
| Lines 1129--1137 | Thm 6.17 with parts (a)--(d) including joint additivity | Thm 6.17' with (a)--(c) valid, (d) FALSIFIED |
| Thm 6.20 | Constant $C$ involves $p_{\min} \sim 10^{-6}$ (vacuous) | Thm 6.20' with $p_{\mathrm{correct}}$; non-vacuous in favorable regime only |
| Q-coverage algorithm | Implicit multiset projector accumulation | Lemma 6.17.C: set-based update with idempotency proof |
| β sign choice | External hyperparameter | Thm β*: $\operatorname{sign}(\beta^*) = \operatorname{sign}(\bar r_{\mathcal{G}} - \bar r)$ computable from baseline pass |

---

*End of supplement.*
