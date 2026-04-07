# Attention-Weighted Reconstruction Bound — Derivation v1

**Date**: 2026-04-08
**Status**: Working draft for Paper 1 Section "MSE → PPL bridge"
**Supersedes**: The informal sketch in V2_THEORY_AND_3MODES Section 3.2

---

## 1. Setup

Single-head self-attention with causal mask. Let
- $q \in \mathbb{R}^d$ : query at a fixed target position
- $K \in \mathbb{R}^{T \times d}$ : keys at positions $1, \ldots, T$, rows $k_t$
- $V \in \mathbb{R}^{T \times d_v}$ : values, rows $v_t$
- $\tilde K = K + E$ : reconstructed keys after quantization, $E \in \mathbb{R}^{T \times d}$
- $s_t(q) := \operatorname{softmax}(q K^\top / \sqrt{d})_t$ : FP16 attention weights
- $s'_t(q) := \operatorname{softmax}(q \tilde K^\top / \sqrt{d})_t$ : quantized attention
- $o(q) := \sum_t s_t(q) v_t$ : FP16 attention output
- $o'(q) := \sum_t s'_t(q) v_t$ : quantized attention output (V unchanged here)
- $\Delta o(q) := o'(q) - o(q)$ : output perturbation

We assume values $V$ are not quantized; the K-side analysis below is the
object of interest. The V-side is analogous by symmetry and treated
separately (Appendix C of the eventual paper).

---

## 2. First-order expansion of softmax

Define the raw score perturbation
$$\alpha_t(q) := \frac{q \cdot e_t}{\sqrt{d}}, \qquad e_t := \tilde k_t - k_t,$$
so that
$$\frac{q \tilde k_t}{\sqrt{d}} = \frac{q k_t}{\sqrt{d}} + \alpha_t(q).$$

The softmax Jacobian at $s_t$ with respect to its logits is
$$\frac{\partial s_t}{\partial \ell_{t'}} = s_t (\delta_{tt'} - s_{t'}).$$

Hence the first-order perturbation of the attention weights is
$$s'_t - s_t = \sum_{t'} \frac{\partial s_t}{\partial \ell_{t'}} \alpha_{t'}
  + O(\|\alpha\|^2)
  = s_t \left( \alpha_t - \sum_{t'} s_{t'} \alpha_{t'} \right) + O(\|\alpha\|^2).$$

Introducing the $s$-weighted expectation operator
$\langle f \rangle_s := \sum_{t'} s_{t'} f_{t'}$, this reads
$$s'_t - s_t = s_t \left( \alpha_t - \langle \alpha \rangle_s \right)
  + O(\|\alpha\|^2). \tag{2.1}$$

Note the zero-mean structure: $\sum_t (s'_t - s_t) = 0$ to first order, as
required by the simplex constraint.

---

## 3. Attention output perturbation

Substituting $(2.1)$ into $\Delta o(q) = \sum_t (s'_t - s_t) v_t$ gives
$$\Delta o(q) = \sum_t s_t \left( \alpha_t - \langle \alpha \rangle_s \right) v_t
  + O(\|\alpha\|^2). \tag{3.1}$$

Since $\sum_t s_t (\cdot - \langle \alpha \rangle_s)$ acts as a centered
operator, we can rewrite $(3.1)$ as
$$\boxed{\Delta o(q) = \sum_t s_t(q) \cdot \alpha_t(q) \cdot
  \bigl( v_t - o(q) \bigr) + O(\|\alpha\|^2)} \tag{3.2}$$

where we used $\sum_t s_t (v_t - o) = 0$ to add the $-\langle \alpha \rangle_s \cdot o$
term without changing anything. Equation $(3.2)$ is the central object of
our theory.

### 3.1 Interpretation of each factor

$(3.2)$ says the output perturbation at query $q$ is a sum over keys $t$ of
three factors:

1. $s_t(q)$ : **attention weight** (how much this key currently matters)
2. $\alpha_t(q) = \tfrac{q \cdot e_t}{\sqrt{d}}$ : **query-projected key error**
   (the component of $e_t$ in the direction $q$ — orthogonal-to-$q$ error
   components do not affect the logit of key $t$)
3. $v_t - o(q)$ : **value deviation** (how far $v_t$ is from the current
   attention output; if $v_t = o(q)$ this token's contribution to the
   perturbation is zero, because up-weighting or down-weighting it has no
   effect on the mean $o(q)$)

A key error $e_t$ only harms the output if **all three** are large:

- **If** $s_t \approx 0$ (low attention): perturbation is multiplied by zero,
  so Lloyd's large tail-token errors at low-attention positions do not hurt.
  This is why raw MSE is a conservative over-estimate.
- **If** $q \perp e_t$: the logit of key $t$ is unchanged, so $s_t$ is
  unchanged. This is why a rotation axis choice that aligns the error into
  directions perpendicular to typical queries is effective — and why per-head
  Pre-RoPE PCA is the right rotation: it concentrates error into the dominant
  eigendirections of $\Sigma_K$, and queries tend to align with *those* same
  eigendirections only in sink heads. In non-sink heads, the query direction
  is roughly independent of the eigendirections, and $q \cdot e_t$ is small.
- **If** $v_t \approx o(q)$: up-weighting key $t$ doesn't change the output
  much, because its value is already "average".

---

## 4. Expected output perturbation magnitude

Taking expectation over queries $q$ drawn from the calibration/eval
distribution and summing the Frobenius square of $(3.2)$:
$$\mathbb{E}_q \|\Delta o(q)\|^2
  \approx \mathbb{E}_q \left\| \sum_t s_t(q) \alpha_t(q) (v_t - o(q)) \right\|^2.
  \tag{4.1}$$

We develop $(4.1)$ into a sum of per-token squared contributions plus
cross-terms:
$$\mathbb{E}_q \|\Delta o(q)\|^2
  = \mathbb{E}_q\!\!\sum_{t_1, t_2}\!\!s_{t_1} s_{t_2} \alpha_{t_1} \alpha_{t_2}
    (v_{t_1} - o(q)) \cdot (v_{t_2} - o(q)).$$

### 4.1 Diagonal-dominant bound (when queries are independent of errors)

If we assume that for distinct $t_1 \neq t_2$ the cross-terms average to
zero (reasonable if query directions at different target positions are
roughly uncorrelated, and $(v_{t_1}-o)\cdot(v_{t_2}-o)$ has no systematic
sign), we get the diagonal-dominant approximation
$$\mathbb{E}_q \|\Delta o(q)\|^2
  \gtrsim \mathbb{E}_q \sum_t s_t(q)^2 \alpha_t(q)^2 \|v_t - o(q)\|^2
  \tag{4.2}$$

This bound identifies **qaMSE** (query-aligned MSE) as the primary metric:
$$\text{qaMSE} := \mathbb{E}_q \sum_t s_t(q)^2 \cdot \frac{(q \cdot e_t)^2}{d}
  \cdot \|v_t - o(q)\|^2.
  \tag{4.3}$$

Three approximations of qaMSE with decreasing precision (and decreasing
measurement cost):

**(A) Full qaMSE** (exact per equation (4.3)): requires capturing
$q, v_t, o(q)$ for each query. Measured in v2af.

**(B) qaMSE without v-deviation weighting**:
$$\text{qaMSE}^{(1)} := \mathbb{E}_q \sum_t s_t(q) \cdot (q \cdot e_t)^2 / d.
  \tag{4.4}$$
Approximation: $s_t^2 \to s_t$ (drop one factor of $s$), $\|v_t - o\|^2 \to $ constant.
Measured in v2af.

**(C) awMSE** (attention-weighted MSE, v2ae candidate):
$$\text{awMSE} := \mathbb{E}_q \sum_t s_t(q) \cdot \|e_t\|^2.
  \tag{4.5}$$
Approximation: replace query projection $(q\cdot e_t)^2$ by $\|q\|^2 \|e_t\|^2$
and absorb the $\|q\|^2$ factor. This is the coarsest approximation; it
over-estimates error in directions orthogonal to $q$.

**(D) Raw MSE**:
$$\text{MSE} := \frac{1}{T}\sum_t \|e_t\|^2. \tag{4.6}$$
Approximation: $s_t \to 1/T$, $(q\cdot e_t)^2 \to \|q\|^2\|e_t\|^2$. This is
what Lloyd-Max minimizes, but it does not know about attention.

### 4.2 Ordering of metric tightness

In general,
$$\text{MSE} \geq_{?} \text{awMSE} \geq_{?} \text{qaMSE}^{(1)} \geq_{?} \text{qaMSE} \approx \mathbb{E}\|\Delta o\|^2$$
is NOT a strict inequality in every case; each pair differs by an
approximation that can be positive or negative. But the **rank correlation**
with $\mathbb{E}\|\Delta o\|^2$ should be monotone: finer metrics should
correlate more tightly with the ground-truth output error.

---

## 5. Mode-specific corollaries

### 5.1 Mode A — Localized positional sink (Mistral-7B)

In Mode A, a specific head has query directions aligned with the top
eigenvector of $\Sigma_K$. Concretely, for such a head, the attention mass
$s_t(q)$ is heavily concentrated on a single token $t^* = 0$ (BOS), and
$q \cdot k_{t^*}$ is large positive (in the direction of the top
eigenvector).

Let $u_1$ be the top PCA eigenvector of $\Sigma_K$ for this head. Then
$k_{t^*} \approx \lambda_1^{1/2} u_1$ (to leading order), and $q$ is aligned
with $u_1$ (otherwise the attention would not concentrate on $t^*$).

The Lloyd error at $t^*$ has a large component in the $u_1$ direction because
that's the direction where the tail token lives — the bulk centroids cannot
reach it. So $e_{t^*} \approx c \cdot u_1$ for some large $c$, and
$q \cdot e_{t^*}$ is large.

Combining:
- $s_{t^*}(q) \approx 0.6 \text{ to } 0.8$ (v2w)
- $(q \cdot e_{t^*}/\sqrt{d})^2 \sim c^2 / d$, large
- $\|v_{t^*} - o(q)\|^2 \sim \|v_{t^*}\|^2$ (v_{t^*} is not the current output mean)

All three factors are large simultaneously → the $t^*$ term dominates the
sum in (4.3) → the covariance term in the awMSE decomposition
(Cov(a_t, \|e_t\|^2)) that we measured directly in v2ae is positive and
large.

**Pos-sink fix** sets $e_{t^*} = 0$, killing the $t^*$ term entirely. No
other term in (4.3) is comparable in magnitude (other tokens have small
$s_t$), so $\mathbb{E}\|\Delta o\|^2$ drops to near-zero. This matches v2h:
`Lloyd + pos-sink_k=1 ≈ FP16` on Mistral.

**Grid** caps each $(q \cdot e_t)^2$ by bounding $\|e_t\|$, but the $t^*$
term still has the full $s_{t^*}$ factor ≈ 0.8, so Grid only mildly helps.
This matches v2p: `Grid = 6.43, Lloyd + pos-sink = 5.99`.

### 5.2 Mode B — Distributed structural tail (Nemo-12B)

In Mode B, there is no single dominant $t^*$. Instead there is a set $S$ of
"delimiter" positions (paragraph breaks, BOS, common words) spread
throughout the sequence, each with moderate attention $s_t \sim 0.05$ to
$0.15$ but with a large query alignment $q \cdot e_t$ (because the queries
of these heads genuinely want to read delimiter tokens).

Now the diagonal sum in (4.3) has many medium-sized terms rather than one
dominant term. The total $\sum_{t \in S} s_t (q \cdot e_t)^2 \|v_t - o\|^2$
is the sum of many summands, each moderate.

**Pos-sink** only zeros $t = 0$, leaving all the other delimiters
un-protected → limited help (v2u: Lloyd + pos-sink barely improves Nemo at
L=32K).

**Token-sink** zeros all $t \in S$, but only if the calibration-selected
$S$ matches the eval content (v2ab/v2ad: works at L=2K, partially works at
L=8K, fails at L=32K because eval delimiters ∉ cal-fit S).

**Grid** uniformly caps each $(q \cdot e_t)^2$ via $\|e_t\| \leq \Delta/2$,
covering all delimiters (known and unknown) with a single deterministic
bound. This explains why Grid is the robust winner on Nemo at long context
(v2u: Grid = 7.68 < Lloyd + sink = 14.84).

### 5.3 Mode C — Bulk-tail (Qwen)

In Mode C, the K distribution is close to Gaussian with modest anisotropy
(κ_max < 10^5). Queries and keys are not strongly aligned with any specific
axis. The attention weights $s_t(q)$ are diffuse: no single position
dominates.

With diffuse $s_t$, the diagonal sum in (4.3) is approximately
$\frac{1}{T} \sum_t (q \cdot e_t)^2 \|v_t - o\|^2$, which is close to raw
MSE weighted by an isotropic $v$ factor. In this limit, Lloyd (L²-optimal
on raw MSE) is also close to optimal on qaMSE, and there is no strong
incentive to switch to Grid.

**Pos-sink** is a small safety margin: it helps because pos 0 has slightly
elevated attention ($s_0 \sim 0.3$ per v2aa), but the effect is modest.

**Token-sink** is *harmful* on small models because the cal-selected
$S$-set is heavily content-specific (Qwen-1.5B cal is dominated by
"Senjō no Valkyria" article tokens per v2ad). At eval, those tokens do not
appear at the same frequency, so protecting them adds an error term
*without* canceling any real covariance contribution. The protected tokens
are now "islands" of FP16 in a distribution that Lloyd was already fitting,
and the discontinuity introduces a bias. This produces *negative*
improvement: Qwen-1.5B PPL rises by +2 to +6 with self-cal token sink.

The "catastrophic token sink" result is therefore not a bug — it is
predicted by the theory once we recognize that the cal-eval distributional
mismatch causes the *wrong* $S$-set to be protected.

---

## 6. Empirical validation program (v2ae, v2af)

We measure the following per (layer, head) per model:

| Symbol | Definition | Cost | Measured in |
|---|---|---|---|
| `raw_MSE` | $\tfrac{1}{T}\sum_t \|e_t\|^2$ | $O(Td)$ | v2ae, v2af |
| `awMSE`   | $\sum_t a_t \|e_t\|^2$ (a_t = mean attention) | $O(Td)$ | v2ae, v2af |
| `qaMSE`   | $\sum_q \sum_t s_t(q) (q\cdot e_t)^2$ | $O(Q T d)$ | **v2af** |
| `exact`   | $\sum_q \|\Delta o(q)\|^2$ via running softmax | $O(Q T (d + d_v))$ | **v2af** |

We then compute the Lloyd/Grid ratio for each metric and compare against the
Lloyd/Grid PPL ratio.

**Prediction**: `r_qa` and `r_exact` should agree with `r_ppl` in sign on
all 4 models (Mistral, Nemo, Qwen-7B, Qwen-1.5B), whereas `r_raw` fails on
3/4 and `r_aw` fails on 2/4 (v2ae data).

If this prediction holds, we have direct empirical support for equation
(4.3) as the correct first-order metric and for the Section 5 mode
corollaries.

---

## 7. Open points to tighten before paper submission

1. **Cross-term neglect in (4.2)**: Prove that under suitable independence
   of query directions, the off-diagonal $t_1 \neq t_2$ contributions are
   $o(1)$ of the diagonal. This is where the bound becomes an approximation
   rather than an inequality.

2. **Value-side symmetry**: Derive the analogous bound for V-quantization
   (set $E_V = \tilde V - V$). The structure is similar but simpler (no
   softmax nonlinearity on the V side).

3. **Second-order softmax term**: The $O(\|\alpha\|^2)$ term in (2.1)
   becomes important when $\alpha$ is large (which happens for Lloyd on
   tail tokens in Mode A). Write out the second-order correction explicitly
   and argue that it is bounded by qaMSE + something-smaller.

4. **Mode boundary smoothness**: Formulate the modes as a continuous
   classifier in $(pos0\_attn, \kappa_{max})$ space, with the optimal method
   being a smooth function rather than a discrete choice. Bound the
   sub-optimality of choosing the wrong method near a boundary.

5. **Bit-width scaling**: At $b$ bits, Lloyd has $\|e_t\|^2 \sim 2^{-2b}$
   in expectation. Derive how the Lloyd vs Grid crossover moves as $b$
   changes. Predict: at higher $b$ the covariance term shrinks and Lloyd
   recovers, consistent with the observation that 3-bit rarely shows
   catastrophes.

6. **Connection to Theorem 6.16.3**: Show that Pre-RoPE PCA rotation
   minimizes `qaMSE` in the limit of isotropic queries, recovering
   Theorem 6.16.3 as a special case. This unifies the rotation theorem and
   the attention-weighted bound under a single framework.

---

## 8. Why this differs from prior work

- **KVLINC (2510.05373)**: motivates attention error empirically, does not
  provide a formal bound. Their solution (linear correction adapters) is
  orthogonal to our analysis.
- **Expected Attention (2510.00636)**: defines $\|\Delta h_{ti}\| = a_{ti}\|W_o v_i\|$
  as a V-side contribution metric (for eviction), not a K-side error bound.
  Does not provide the attention-weighted reconstruction form.
- **KVQuant, KIVI, KVTC**: propose outlier-handling or rotation strategies
  without a first-principles quantization error bound.
- **Classical rate-distortion** (Gallager, Cover-Thomas): provides Lloyd-Max
  L² optimality under the Gaussian assumption, no attention structure.
- **TurboQuant (2504.19874)**: near-optimal distortion for online VQ, uses
  raw MSE as the loss and does not address attention-weighted effects.
- **Radio (2505.03031)**: rate-distortion optimization for LLM compression,
  focuses on weight compression rather than KV-cache activations.

As far as we can tell from the April 2026 literature, the **attention-weighted
reconstruction bound with the Mode A/B/C corollaries is not present in
any existing paper**. The closest statements are empirical observations
(Expected Attention's contribution metric) or raw-MSE bounds (TurboQuant),
neither of which connects to the mode-dependent choice of quantizer.

---

*Draft v1, 2026-04-08, mais. Subject to revision after v2af results land.*
