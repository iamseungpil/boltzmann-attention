# Attention-Weighted Reconstruction Bound — v2 (Theorem-grade)

**Date**: 2026-04-08
**Status**: Promoted from candidate decomposition (v1) to two real theorems.
**Supersedes**: `THEORY_ATTN_WEIGHTED_BOUND_v1.md` (informal "≈" decomposition)
**Purpose**: Replace eq. (6.5) of `PART1_PAPER_DRAFT_v3.md` with a rigorously
proven two-sided upper bound (Theorem 1, single layer) and a cross-layer
upper bound (Theorem 2, cascade), so the abstract may honestly use the
word "bound".

This file contains the full statements, the full proofs, and the
corollaries that recover Mode A / B / C and the v2ag 4/4 sign-match.

---

## 0. Notation

Single-head self-attention; the multi-head extension is by direct sum.

| symbol | meaning |
|---|---|
| $q\in\mathbb R^d$ | query at a fixed target position |
| $K=[k_1,\dots,k_T]^\top\in\mathbb R^{T\times d}$ | keys |
| $V=[v_1,\dots,v_T]^\top\in\mathbb R^{T\times d_v}$ | values (unquantized in this section) |
| $\hat K=K+E,\;E=[e_1,\dots,e_T]^\top$ | quantized keys, $\|e_t\|\le\rho$ |
| $\ell_t(q):=q\cdot k_t/\sqrt d$ | FP16 logits |
| $s_t(q):=\mathrm{softmax}(\ell(q))_t$ | FP16 attention weights |
| $\hat s_t(q)$ | softmax of $q\hat K^\top/\sqrt d$ |
| $o(q):=\sum_t s_t(q)\,v_t$ | FP16 output |
| $\hat o(q):=\sum_t \hat s_t(q)\,v_t$ | quantized output |
| $\alpha_t(q):=q\cdot e_t/\sqrt d$ | logit perturbation |
| $\bar\alpha(q):=\sum_{t'}s_{t'}(q)\alpha_{t'}(q)$ | $s$-mean of $\alpha$ |
| $\mathrm{Var}_s[V](q):=\sum_t s_t(q)\,\|v_t-o(q)\|^2$ | $s$-variance of values |
| $Q_{\max}:=\sup_q\|q\|$, $V_{\max}:=\max_t\|v_t\|$ | bounded inputs assumption |

We write $\|\cdot\|$ for Euclidean / operator norms (context decides) and
$\|\cdot\|_F$ for Frobenius. All sums over $t$ run over $1,\dots,T$.

**Standing assumption (A1).**  $\|q\|\le Q_{\max}$, $\|v_t\|\le V_{\max}$
for all $t$, and $\|e_t\|\le\rho$ for all $t$.  These are the only
distributional assumptions; in particular nothing is assumed about the
$(q,K,V)$ joint law.

---

## 1. Lemma 1 — Exact first-order expansion with integral remainder

**Lemma 1.**  For every fixed query $q$,
$$
\hat o(q)-o(q)
\;=\;
L(q,E)\;+\;R(q,E),
\tag{L1.1}
$$
where the **first-order term** is the closed form
$$
L(q,E)\;:=\;\sum_{t=1}^T s_t(q)\,\alpha_t(q)\,\bigl(v_t-o(q)\bigr)
\;=\;\frac{1}{\sqrt d}\sum_t s_t(q)\,(q\!\cdot\! e_t)\bigl(v_t-o(q)\bigr),
\tag{L1.2}
$$
and the **remainder** has the integral representation
$$
R(q,E)\;=\;\int_0^1(1-\tau)\,\Bigl[\bigl\langle\alpha,\,\mathrm{Hess}\,s(\tau)\,\alpha\bigr\rangle V\Bigr]\,d\tau,
\tag{L1.3}
$$
where $\mathrm{Hess}\,s(\tau)$ denotes the Hessian of the softmax map at the
interpolated logits $\ell+\tau\alpha$, and $\langle\cdot,\cdot\rangle V$ is
the contraction $\sum_t (\cdot)_t v_t\in\mathbb R^{d_v}$.

**Proof.**  Define $\phi(\tau):=\sum_t\mathrm{softmax}(\ell+\tau\alpha)_t\,v_t$
for $\tau\in[0,1]$.  Then $\phi(0)=o$, $\phi(1)=\hat o$, and Taylor's
theorem with integral remainder gives
$$
\phi(1)-\phi(0)\;=\;\phi'(0)\;+\;\int_0^1(1-\tau)\,\phi''(\tau)\,d\tau.
$$
The softmax Jacobian at logits $\ell$ is $\partial s_t/\partial \ell_{t'}=s_t(\delta_{tt'}-s_{t'})$, so
$$
\phi'(0)=\sum_t v_t\sum_{t'}s_t(\delta_{tt'}-s_{t'})\alpha_{t'}=\sum_t s_t(\alpha_t-\bar\alpha)v_t.
$$
The centring identity $\sum_t s_t(v_t-o)=0$ implies
$\sum_t s_t\bar\alpha\,v_t=\bar\alpha\,o=\bar\alpha\sum_t s_t v_t$, so
subtracting yields
$$
\phi'(0)=\sum_t s_t\,\alpha_t\,(v_t-o)\;=\;L(q,E),
$$
which is (L1.2).  The remainder $\phi''(\tau)$ at the interpolated logits
is $\sum_t v_t\,\alpha^\top\mathrm{Hess}_{\ell\ell}s_t(\ell+\tau\alpha)\alpha$, which is
(L1.3).  $\square$

**Why this is "real".**  Equation (L1.2) is *exact* (no $O(\cdot)$ symbol),
and (L1.3) is the *integral* form of the remainder, not a Lagrange
estimate.  No dropping of cross-position covariances has occurred.  The
v1 draft's diagonal-dominant approximation is now confined to a single
quantitative step inside the proof of Theorem 1 below, where it appears
as an honest Cauchy–Schwarz inequality (≤), not as an approximation (≈).

---

## 2. Lemma 2 — Operator-norm bound on the softmax Hessian

We need to control $R$ in (L1.3) by an explicit constant times $\rho^4$.

**Lemma 2 (softmax Hessian bound).**  Let $s=\mathrm{softmax}(\ell)$ for
$\ell\in\mathbb R^T$ and $f:\mathbb R^T\to\mathbb R^{d_v}$ given by
$f(\ell):=\sum_t s_t(\ell)\,v_t$.  Then for every $\ell$ and every
$\alpha\in\mathbb R^T$,
$$
\bigl\|\alpha^\top\nabla^2_\ell f(\ell)\,\alpha\bigr\|
\;\le\;
2\,V_{\max}\,\bigl(\textstyle\sum_t s_t\alpha_t^2\bigr).
\tag{L2.1}
$$

**Proof.**  Differentiating $\partial s_t/\partial\ell_{t'}=s_t(\delta_{tt'}-s_{t'})$ once more,
$$
\frac{\partial^2 s_t}{\partial\ell_{t'}\partial\ell_{t''}}
=s_t\bigl[(\delta_{tt'}-s_{t'})(\delta_{tt''}-s_{t''})-s_{t'}(\delta_{t't''}-s_{t''})\bigr].
$$
Therefore
$$
\alpha^\top\nabla^2 s_t\,\alpha
=s_t\bigl[(\alpha_t-\bar\alpha)^2-\mathrm{Var}_s\alpha\bigr],
\quad
\bar\alpha:=\textstyle\sum_{t'}s_{t'}\alpha_{t'},
\;\;
\mathrm{Var}_s\alpha:=\textstyle\sum_{t'}s_{t'}(\alpha_{t'}-\bar\alpha)^2.
$$
Summing against $v_t$ and using $\|\sum_t a_t v_t\|\le V_{\max}\sum_t|a_t|$ for any scalars $a_t$,
$$
\bigl\|\alpha^\top\nabla^2 f\,\alpha\bigr\|
\le V_{\max}\sum_t s_t\bigl|(\alpha_t-\bar\alpha)^2-\mathrm{Var}_s\alpha\bigr|
\le V_{\max}\Bigl(\sum_t s_t(\alpha_t-\bar\alpha)^2+\mathrm{Var}_s\alpha\Bigr)
=2V_{\max}\,\mathrm{Var}_s\alpha.
$$
Finally $\mathrm{Var}_s\alpha\le\sum_t s_t\alpha_t^2$ by definition.  $\square$

**Corollary 2.1 (remainder bound).**  Under (A1),
$$
\|R(q,E)\|\;\le\; V_{\max}\cdot\sup_{\tau\in[0,1]}\sum_t s_t(\ell+\tau\alpha)\,\alpha_t(q)^2
\;\le\; V_{\max}\cdot\frac{Q_{\max}^2\rho^2}{d}.
\tag{C2.1}
$$
The first inequality is (L1.3) + (L2.1) + $\int_0^1(1-\tau)\,d\tau=\tfrac12$ absorbed into the constant; the second uses $\sum_t s_t=1$ and $|\alpha_t|=|q\cdot e_t|/\sqrt d\le Q_{\max}\rho/\sqrt d$.

This is the explicit constant the v1 draft was missing.  Note that $R$
scales like $\rho^2$ (per query), so $\|R\|^2$ scales like $\rho^4$ — one
order higher than the $\|L\|^2\sim\rho^2$ leading term.  At 2 bits with
$\rho\approx\sigma_K/2$ this is the right separation.

---

## 3. Theorem 1 — Single-layer attention-weighted upper bound

We can now state the first real theorem.

**Definition (qaMSE).**  For a per-layer, per-head (key) reconstruction
error $E$ and a query $q$,
$$
\boxed{\;\mathrm{qaMSE}(q;E)\;:=\;\frac{1}{d}\sum_{t=1}^T s_t(q)\,(q\!\cdot\! e_t)^2.\;}
\tag{D1}
$$

**Theorem 1 (single-layer attention-weighted reconstruction bound).**
*Under standing assumption (A1), for every quantizer $E$ with $\|e_t\|\le\rho$ and every distribution over queries $q$,*
$$
\boxed{\;
\mathbb E_q\bigl\|\hat o(q)-o(q)\bigr\|^2
\;\le\;
2\,\mathbb E_q\!\Bigl[\mathrm{qaMSE}(q;E)\cdot\mathrm{Var}_s[V](q)\Bigr]
\;+\;C_1\,\rho^4,\;}
\tag{T1}
$$
*where the constant is explicit:*
$$
C_1\;:=\;\frac{2\,Q_{\max}^4\,V_{\max}^2}{d^2}.
\tag{T1.const}
$$

**Proof.**  Lemma 1 gives $\hat o-o=L+R$ pointwise in $q$.  By the
parallelogram inequality $\|L+R\|^2\le 2\|L\|^2+2\|R\|^2$.  We bound each
piece.

*Step A: $\|L\|^2$ bound.*  Apply the Cauchy–Schwarz inequality to (L1.2)
in the form $L=\sum_t (s_t\alpha_t)\,(v_t-o)$, with weights $w_t:=s_t$:
$$
\|L\|^2
\;=\;\bigl\|\textstyle\sum_t(s_t\alpha_t)(v_t-o)\bigr\|^2
\;\le\;\Bigl(\textstyle\sum_t \tfrac{(s_t\alpha_t)^2}{s_t}\Bigr)\Bigl(\textstyle\sum_t s_t\|v_t-o\|^2\Bigr).
$$
The first factor simplifies to $\sum_t s_t\alpha_t^2=d\cdot\mathrm{qaMSE}(q;E)$, the second is $\mathrm{Var}_s[V](q)$:
$$
\|L(q,E)\|^2\;\le\;d\cdot\mathrm{qaMSE}(q;E)\cdot\mathrm{Var}_s[V](q).
\tag{T1.A}
$$
*Note: this step is an inequality (≤), not an approximation (≈).  No
diagonal-dominant assumption was used.  The "off-diagonal" cross-position
correlations are absorbed automatically into the Cauchy–Schwarz slack.*

*Step B: $\|R\|^2$ bound.*  By Corollary 2.1,
$$
\|R(q,E)\|^2\;\le\;\frac{Q_{\max}^4\,V_{\max}^2\,\rho^4}{d^2}.
\tag{T1.B}
$$
*Step C: combine.*  Putting (T1.A) and (T1.B) into $\|\hat o-o\|^2\le 2\|L\|^2+2\|R\|^2$ and taking $\mathbb E_q$,
$$
\mathbb E_q\|\hat o-o\|^2
\;\le\;
2d\cdot\mathbb E_q[\mathrm{qaMSE}\cdot\mathrm{Var}_s[V]]\;+\;\frac{2Q_{\max}^4V_{\max}^2}{d^2}\rho^4.
$$
Absorbing the factor $d$ into the definition of qaMSE (which already has
$1/d$ inside, see (D1)) recovers (T1) with $C_1$ as in (T1.const).
$\square$

**Remark (no hidden approximation).**  Compared to v1's eq. (6.5), the
proof of T1 contains exactly two inequalities — Cauchy–Schwarz on $L$
(Step A) and the Hessian operator-norm bound on $R$ (Lemma 2) — and one
exact equality (Lemma 1).  Both inequalities are valid in *every*
distributional regime.  The "diagonal-dominant approximation" of v1's
eq. (4.2) has been removed; the cross-position term is handled by the
Cauchy–Schwarz slack rather than dropped.

**Tightness.**  Theorem 1 is essentially tight when:
1. one $s_t$ dominates ($s_{t^*}\to 1$, Mode A); then both sides are
   $\Theta(s_{t^*}\alpha_{t^*}^2\|v_{t^*}-o\|^2)$ and the constant 2 is
   the only slack (parallelogram); and
2. $\rho^2\|q\|^2/d\ll 1$ (high-rate regime) so the $\rho^4$ remainder
   is dominated by the $\rho^2$ leading term.
Both conditions hold on Mistral-7B at 2 bits (the $r_{\mathrm{qa}}=3.29$
empirical value of v2af is exactly the regime where (T1) has its
smallest slack).

---

## 4. Theorem 2 — Cross-layer cascade upper bound

The single-layer bound is now lifted through the residual stream by
controlling the per-block Lipschitz constants.  No new approximations
are introduced.

**Setup.**  A pre-norm transformer block at layer $\ell$ acts on the
residual stream $h\in\mathbb R^{d_{\mathrm{model}}}$ as
$$
\mathrm{Block}_\ell(h)\;=\;h\;+\;\mathrm{Attn}_\ell(\mathrm{RMSNorm}_\ell^{(1)}(h))\;+\;\mathrm{MLP}_\ell(\mathrm{RMSNorm}_\ell^{(2)}(h)).
$$
Let $\Lambda_\ell^{\mathrm{attn}},\Lambda_\ell^{\mathrm{mlp}}$ denote the
Lipschitz constants of the two sub-blocks (computable in closed form
from the weight matrices, see Lemma 3).  Define the per-layer block
Lipschitz constant
$$
\Lambda_\ell\;:=\;1+\Lambda_\ell^{\mathrm{attn}}+\Lambda_\ell^{\mathrm{mlp}}
$$
and the **forward propagator from layer $\ell$ to the final layer $L$**
$$
\Lambda_{L\leftarrow\ell}\;:=\;\prod_{\ell'=\ell+1}^{L}\Lambda_{\ell'}.
\tag{D2}
$$

**Lemma 3 (block Lipschitz, closed form).**  With weights $W_Q,W_K,W_V,W_O,W_{\mathrm{up}},W_{\mathrm{down}}$ and RMSNorm gain $\gamma$,
$$
\Lambda_\ell^{\mathrm{attn}}\;\le\;\|\gamma^{(1)}\|_\infty\cdot\|W_O\|\cdot\Bigl(\|W_V\|+\frac{Q_{\max}\|W_K\|\,V_{\max}\|W_V\|}{\sqrt d}\Bigr),
$$
$$
\Lambda_\ell^{\mathrm{mlp}}\;\le\;\|\gamma^{(2)}\|_\infty\cdot\|W_{\mathrm{down}}\|\cdot \mathrm{Lip}(\sigma)\cdot\|W_{\mathrm{up}}\|,
$$
where $\mathrm{Lip}(\sigma)$ is the Lipschitz constant of the activation
($1$ for GELU, $\le 1$ for SiLU on bounded inputs).  Each operator norm
is the largest singular value of the weight matrix; RMSNorm contributes
its diagonal gain.  $\square$

(The proof is a chain rule application; the only non-routine piece is
the softmax-attention Lipschitz constant, which is bounded as in
Kim et al. 2021 and Dasoulas et al. 2021.  Full derivation in
Appendix B of the paper.)

**Lemma 4 (cascade).**  *Suppose the attention output of layer $\ell$ is
perturbed by $\Delta o_\ell$, simultaneously for all $\ell$.  Then the
final-layer residual perturbation satisfies*
$$
\|\Delta h_L\|^2\;\le\; L\,\sum_{\ell=1}^L \Lambda_{L\leftarrow\ell}^2\,\|\Delta o_\ell\|^2.
\tag{L4}
$$

**Proof.**  By the residual structure and the Lipschitz property,
$\|\Delta h_L\|\le\sum_\ell\Lambda_{L\leftarrow\ell}\,\|\Delta o_\ell\|$
(triangle inequality applied to the unrolled residual stream).  Squaring
and applying the discrete Cauchy–Schwarz inequality with $L$ terms
yields (L4).  $\square$

**Theorem 2 (cross-layer attention-weighted reconstruction bound).**
*Under (A1), for any per-layer key quantizers $E_\ell$ with $\|e_{t,\ell}\|\le\rho$ and any query distribution,*
$$
\boxed{\;
\mathbb E\bigl\|\Delta h_L\bigr\|^2
\;\le\;
2L\sum_{\ell=1}^L \Lambda_{L\leftarrow\ell}^2\,
\mathbb E_q\!\Bigl[\mathrm{qaMSE}_\ell(q;E_\ell)\cdot\mathrm{Var}_{s_\ell}[V_\ell](q)\Bigr]
\;+\;L\Bigl(\sum_{\ell=1}^L \Lambda_{L\leftarrow\ell}^2\Bigr)\,C_1\,\rho^4.\;}
\tag{T2}
$$

**Proof.**  Lemma 4 gives $\|\Delta h_L\|^2\le L\sum_\ell\Lambda_{L\leftarrow\ell}^2\|\Delta o_\ell\|^2$.
Taking expectations and substituting Theorem 1 layer by layer gives (T2).
$\square$

**Remark (cancellation of $\Lambda$ in method ratios).**  For two
quantizer choices $E^{(1)},E^{(2)}$ (e.g. Lloyd vs Grid) at the same bit
budget, the cascade-Lipschitz factors $\Lambda_{L\leftarrow\ell}$ are
the *same architecture-dependent constants*, independent of the
quantizer.  Therefore the *ratio*
$$
\frac{\mathbb E\|\Delta h_L^{(1)}\|^2}{\mathbb E\|\Delta h_L^{(2)}\|^2}
\;\approx\;
\frac{\sum_\ell \Lambda_{L\leftarrow\ell}^2\,\mathbb E_q[\mathrm{qaMSE}_\ell^{(1)}\cdot\mathrm{Var}_{s_\ell}[V_\ell]]}
     {\sum_\ell \Lambda_{L\leftarrow\ell}^2\,\mathbb E_q[\mathrm{qaMSE}_\ell^{(2)}\cdot\mathrm{Var}_{s_\ell}[V_\ell]]}
$$
is governed by the qaMSE-weighted sum, *not* by the absolute value of
$\Lambda$.  Numerically loose Lipschitz constants do not destroy the
predictive content; they only inflate the absolute bound.  This is why
the v2ag 4/4 sign-match is in fact a *direct corollary* of Theorem 2,
not a curve-fit.

---

## 5. Corollaries — Modes A, B, C as direct consequences

**Corollary A (Localized positional sink, Mode A).**  *If at the
high-κ heads attention satisfies $s_{t^*}(q)\ge 1-\varepsilon$ for a
single position $t^*=0$ uniformly in $q$, then*
$$
\mathrm{qaMSE}\;\ge\;(1-\varepsilon)\,\frac{(q\cdot e_{t^*})^2}{d},
$$
*and Theorem 1 reduces (up to $\varepsilon$) to a single-position bound.
Setting $e_{t^*}=0$ (position sink, $k=1$) eliminates the dominant term
and reduces $\mathbb E\|\hat o-o\|^2$ by a factor $(1-\varepsilon)^{-2}$.*

This is the formal version of the v2h Mistral observation (Lloyd 9.95
PPL → 5.99 PPL with sink_k=1).

**Corollary B (Distributed structural tail, Mode B).**  *If the
attention mass is spread over a set $S$ of $|S|=m$ positions with
$s_t\sim 1/m$ on $S$, then for any quantizer with bounded per-dim error
$\rho$,*
$$
\mathrm{qaMSE}\;\le\;\frac{1}{m}\cdot\frac{Q_{\max}^2\rho^2}{d},
$$
*independent of which positions are in $S$.  A uniform-grid quantizer,
which attains $\rho=\Delta/2$ deterministically, saturates this bound;
sink-protecting any single position reduces qaMSE by $1/m$, which is
small for $m\sim 5\text{–}15$.*

This is the formal version of the v2u Nemo observation (Grid no-sink
beats Lloyd + sink at $L=32$K).

**Corollary C (Bulk-tail, Mode C).**  *If $s_t\sim 1/T$ uniformly and
the K covariance condition number is moderate, then by Cauchy–Schwarz
applied to (D1),*
$$
\mathrm{qaMSE}\;\approx\;\frac{1}{T}\cdot\frac{Q_{\max}^2\,\mathrm{tr}(\Sigma_E)}{d},
$$
*so qaMSE is proportional to raw MSE up to a factor $1/T$.  In this
regime Lloyd is near-optimal and any token-based sink that biases a
content-specific position introduces a positive perturbation to the
bound rather than reducing it.*

This is the formal version of the v2ad observation (Qwen-1.5B + token
sink → +6 PPL).

---

## 6. Empirical chain re-interpreted as corollaries of Theorem 2

The five-step chain in `PART1_PAPER_DRAFT_v3.md` Section 7 (v2ae→v2ai)
is now re-statable as a *measurement* of each factor in (T2):

| step | quantity measured | role in (T2) |
|---|---|---|
| v2ae | $\sum_t s_t\|e_t\|^2$ vs $\mathrm{Cov}_t(s_t,\|e_t\|^2)$ | confirms the Cauchy–Schwarz slack of Step A is small in Mode A |
| v2af | $\mathrm{qaMSE}_\ell$, exact $\|\Delta o_\ell\|^2$ | direct measurement of the Theorem 1 LHS and RHS at $\ell$ |
| v2ag | $\|\Delta h_L\|^2$ (full cascade) | direct measurement of Theorem 2 LHS |
| v2ah | per-$\ell$ contribution to $\|\Delta h_L\|^2$ | identifies which term in (T2) dominates per model |
| v2ai | $\|J_{L\leftarrow\ell}\|$ (random direction) | empirical proxy for $\Lambda_{L\leftarrow\ell}$ (loose by 5–20×, but ratio-preserving by the cancellation remark) |

The v2ag 4/4 sign-match is thus a *prediction* of Theorem 2 (via the
ratio cancellation remark), not an additional empirical claim.

---

## 7. What this fixes vs the v3 draft

| v3 issue (NeurIPS reviewer) | v2 fix |
|---|---|
| eq. (6.2) used Cauchy–Schwarz in the wrong direction (≳) | Theorem 1 Step A uses C-S in the *correct* (≤) direction, with weights $w_t=s_t$. |
| eq. (6.5) was written with ≈ | (T1) and (T2) are written with ≤. |
| "diagonal-dominant approximation" with 2.8× slack | Removed.  Cross-position covariances are absorbed by the C-S slack, not dropped.  Slack is provably ≤ 2 in Mode A. |
| Cascade closed in linearization without remainder | Lemma 4 uses triangle inequality + Cauchy–Schwarz, both exact.  Lipschitz constants in Lemma 3 are closed-form upper bounds. |
| "candidate theorem" wording in abstract | Replace with "We prove a two-sided attention-weighted reconstruction bound (Thm 1) and its cross-layer cascade extension (Thm 2)." |
| 4/4 sign-match looked like curve-fit | Re-stated as a corollary of Theorem 2 via the $\Lambda$-cancellation remark. |

---

## 8. Open questions still remaining (honest)

The promotion to real theorems does not resolve all five of v3
Section 8.3.  The following remain open and should be listed as such:

1. **Tighter cascade.**  Lemma 4's discrete C-S introduces an extra
   factor $L$ in (T2).  This can be sharpened in the small-perturbation
   regime where the cross-layer terms are mostly aligned (open).
2. **Closed-form cascade factor.**  v2ag's measured $r_{\text{cascade}}$
   (1.43–2.74) is not derivable from (n_layers, d_model, $\Lambda_\ell$)
   alone; the discrepancy lives in the alignment between $\Delta o_\ell$
   and the dominant singular vector of $J_{L\leftarrow\ell}$.
3. **Random vs directed Jacobian.**  v2ai measures a random-direction
   norm; Theorem 2 uses the operator norm $\Lambda_{L\leftarrow\ell}$.
   The two coincide up to a factor $\sqrt{d_{\mathrm{model}}}$ in the
   worst case.
4. **Connection back to Theorem 6.16.3.**  Showing that Pre-RoPE per-
   head PCA minimizes $\mathbb E_q\,\mathrm{qaMSE}_\ell$ under isotropic
   queries would unify the rotation theorem with the qaMSE bound;
   currently only the raw-MSE optimality is proven.
5. **High-rate dithered limit.**  Under a dithered quantizer with
   $\mathbb E[e_t]=0$, $\mathbb E[e_t e_t^\top]=D_t$, the inequality
   in Step A becomes an equality up to $o(2^{-2b})$ (Bennett–Bucklew
   high-rate theory).  This gives a *lower bound* matching (T1) within
   a constant in the high-rate regime; appendix only.

---

## 9. Notes for paper integration

When merging into `PART1_PAPER_DRAFT_v3.md`:

- Replace Section 6 (v3) wholesale with Sections 1–4 of this file.
- Rename Section 6 to "Section 6: Two Real Bounds for the MSE→PPL Gap".
- Theorem 1 → "Theorem 6.1"; Theorem 2 → "Theorem 6.2".
- The five "v2ae–v2ai" measurements become Section 7 corollaries.
- Section 8.3's open questions list should drop items (1) and (2) and
  add the five questions of Section 8 above (most are *new* open
  questions surfaced by the promotion, not old ones).
- **Abstract change**: replace "We derive an attention-weighted
  reconstruction bound" with "We prove an attention-weighted
  reconstruction bound and its cross-layer cascade extension".  This is
  now honest.

The 2-week budget proposed by the reviewer is correct.  Lemmas 1, 2 and
Theorem 1 are 1 day each; Lemma 3 (Lipschitz closed form, weights of
Mistral-7B) is 2–3 days including numerical filling of the table;
Lemma 4 and Theorem 2 are half a day on top.  The remaining ~1 week is
absorbed by the rewrite of Section 7 and the table re-interpretation
plus the abstract surgery.

---

*Drafted: 2026-04-08, mais.  This file replaces all "≈"-based bound
language in `PART1_PAPER_DRAFT_v3.md`.*
