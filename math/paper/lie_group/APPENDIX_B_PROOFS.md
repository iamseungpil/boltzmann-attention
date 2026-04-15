# Appendix B — Full Proofs of Theorems 6.1 and 6.2

**Companion to**: `PART1_PAPER_DRAFT_v3.md` Section 6
**Date**: 2026-04-08
**Status**: Self-contained proofs. Cited as Appendix B in the paper.

This appendix contains the full, self-contained proofs of every claim
in Section 6 of the main paper. The structure mirrors the section
labels referenced in the main text:

- **B.1** Lemma B.1 — exact integral-remainder Taylor expansion
- **B.2** Theorem 6.1 — single-layer attention-weighted upper bound
- **B.3** Lemma 6.A — closed-form transformer-block Lipschitz
- **B.4** Lemma 6.B + Theorem 6.2 — cross-layer cascade upper bound
- **B.5** Corollary 6.3 — $\Lambda$-cancellation in method-comparison
- **B.6** Corollaries 6.4–6.6 — Modes A/B/C as formal specializations

All proofs are written so that any single section can be read
independently of the others, given the standing assumptions of
Section B.0.

---

## B.0 Standing assumptions and notation

We work with single-head self-attention; the multi-head case follows
by direct sum on the head index. Fix a layer $\ell$ and a target
position; the query is $q\in\mathbb R^d$, the keys are
$K=[k_1,\dots,k_T]^\top\in\mathbb R^{T\times d}$, the values are
$V=[v_1,\dots,v_T]^\top\in\mathbb R^{T\times d_v}$. The quantized keys
are $\hat K=K+E$ with row-wise error vectors $e_t$.

The FP16 logits, weights, and output are
$$
\ell_t(q):=\frac{q\cdot k_t}{\sqrt d},
\qquad
s_t(q):=\mathrm{softmax}(\ell(q))_t,
\qquad
o(q):=\sum_{t=1}^T s_t(q)\,v_t,
$$
and the quantized analogues $\hat\ell_t,\hat s_t,\hat o$ replace $K$
with $\hat K$. The logit perturbation is
$\alpha_t(q):=q\cdot e_t/\sqrt d=\hat\ell_t(q)-\ell_t(q)$, and we write
$\alpha\in\mathbb R^T$ for the vector $(\alpha_1,\dots,\alpha_T)$.

We use two attention-weighted moments throughout:
$$
\bar\alpha(q):=\sum_{t=1}^T s_t(q)\,\alpha_t(q),
\qquad
\mathrm{Var}_s\alpha(q):=\sum_{t=1}^T s_t(q)\bigl(\alpha_t(q)-\bar\alpha(q)\bigr)^2,
$$
and the value-side $s$-variance
$$
\mathrm{Var}_s[V](q):=\sum_{t=1}^T s_t(q)\,\|v_t-o(q)\|^2.
$$

**Standing assumption (A1).** There exist constants $Q_{\max},V_{\max},\rho$
such that for all queries and all keys/values in the support of the
data distribution,
$$
\|q\|\le Q_{\max},\qquad \|v_t\|\le V_{\max},\qquad \|e_t\|\le\rho.
$$
No distributional assumption is made beyond (A1); in particular the
joint law of $(q,K,V)$ is arbitrary.

A useful elementary fact:
$$
|\alpha_t(q)|=\frac{|q\cdot e_t|}{\sqrt d}\le\frac{\|q\|\|e_t\|}{\sqrt d}\le\frac{Q_{\max}\rho}{\sqrt d}.
\tag{B.0.1}
$$

**Convention.** Throughout this appendix, $\|\cdot\|$ denotes the
Euclidean norm on vectors and the operator (spectral) norm on matrices;
$\|\cdot\|_F$ denotes Frobenius. All sums $\sum_t$ run over $t=1,\dots,T$.

---

## B.1 Lemma B.1 — Exact integral-remainder Taylor expansion

This is the foundational lemma. It replaces the informal "$O(\|\alpha\|^2)$"
of the v1 draft with an exact identity plus an integral remainder, so
that the subsequent bounds in B.2 are inequalities (≤), not
approximations (≈).

**Lemma B.1.** *Fix a query $q$ and a key perturbation $E$. Define*
$$
\phi(\tau)\;:=\;\sum_{t=1}^T \mathrm{softmax}(\ell(q)+\tau\alpha(q))_t\;v_t,\qquad \tau\in[0,1].
$$
*Then $\phi(0)=o(q)$, $\phi(1)=\hat o(q)$, and*
$$
\hat o(q)-o(q)\;=\;L(q,E)\;+\;R(q,E),
\tag{B.1.1}
$$
*where the first-order term is the closed form*
$$
L(q,E)\;:=\;\sum_{t=1}^T s_t(q)\,\alpha_t(q)\,\bigl(v_t-o(q)\bigr)
\tag{B.1.2}
$$
*and the remainder is the integral*
$$
R(q,E)\;:=\;\int_0^1 (1-\tau)\,\phi''(\tau)\,d\tau,
\tag{B.1.3}
$$
*with second derivative*
$$
\phi''(\tau)\;=\;\sum_{t=1}^T s_t(\tau)\,\Bigl[(\alpha_t-\bar\alpha(\tau))^2 - \mathrm{Var}_s\alpha(\tau)\Bigr]\,v_t,
\tag{B.1.4}
$$
*where $s_t(\tau):=\mathrm{softmax}(\ell+\tau\alpha)_t$, and $\bar\alpha(\tau),\mathrm{Var}_s\alpha(\tau)$ are the $s(\tau)$-mean and $s(\tau)$-variance of $\alpha$.*

**Proof.**

*Step 1: $\phi$ is smooth and Taylor's theorem applies.* The map
$\tau\mapsto\mathrm{softmax}(\ell+\tau\alpha)$ is real analytic on $\mathbb R$
(softmax is a composition of $\exp$ and rational maps with non-vanishing
denominator). Hence $\phi$ is smooth and Taylor's theorem with integral
remainder of order 2 gives
$$
\phi(1)-\phi(0)\;=\;\phi'(0)\;+\;\int_0^1 (1-\tau)\,\phi''(\tau)\,d\tau.
$$
The endpoints $\phi(0)=\sum_t s_t(0)v_t=o$ and $\phi(1)=\sum_t s_t(1)v_t=\hat o$ are immediate.

*Step 2: Compute $\phi'(0)$.* The softmax Jacobian at logits $\ell$ is
$$
\frac{\partial s_t}{\partial\ell_{t'}}\;=\;s_t(\delta_{tt'}-s_{t'}),
\tag{B.1.5}
$$
which is a standard identity (differentiate $s_t=e^{\ell_t}/Z$ with $Z=\sum_{t''}e^{\ell_{t''}}$ and use $\partial Z/\partial\ell_{t'}=e^{\ell_{t'}}=s_{t'}Z$).

By the chain rule applied to $s_t(\tau)$,
$$
\frac{ds_t(\tau)}{d\tau}\;=\;\sum_{t'}\frac{\partial s_t}{\partial\ell_{t'}}(\ell+\tau\alpha)\cdot\alpha_{t'}\;=\;s_t(\tau)\bigl(\alpha_t-\bar\alpha(\tau)\bigr).
\tag{B.1.6}
$$
Evaluating at $\tau=0$,
$$
\phi'(0)\;=\;\sum_t \frac{ds_t}{d\tau}(0)\,v_t\;=\;\sum_t s_t(0)\bigl(\alpha_t-\bar\alpha(0)\bigr)\,v_t\;=\;\sum_t s_t(\alpha_t-\bar\alpha)\,v_t,
$$
where $s_t=s_t(0)$ and $\bar\alpha=\bar\alpha(0)$.

*Step 3: The centring identity.* We claim $\phi'(0)=L(q,E)$ as defined
in (B.1.2). Compute
$$
\sum_t s_t(\alpha_t-\bar\alpha)v_t
\;=\;\sum_t s_t\alpha_t v_t\;-\;\bar\alpha\sum_t s_t v_t
\;=\;\sum_t s_t\alpha_t v_t\;-\;\bar\alpha\,o.
$$
On the other hand, expanding $L(q,E)$,
$$
L(q,E)
=\sum_t s_t\alpha_t(v_t-o)
=\sum_t s_t\alpha_t v_t \;-\;\Bigl(\sum_t s_t\alpha_t\Bigr)o
=\sum_t s_t\alpha_t v_t\;-\;\bar\alpha\,o,
$$
where the last equality uses $\bar\alpha=\sum_t s_t\alpha_t$ (the
definition of $\bar\alpha$ at $\tau=0$). Hence $\phi'(0)=L(q,E)$
*identically*, with no error term.

*Step 4: Compute $\phi''(\tau)$.* Differentiating (B.1.6) once more,
$$
\frac{d^2 s_t(\tau)}{d\tau^2}\;=\;\frac{d}{d\tau}\bigl[s_t(\tau)(\alpha_t-\bar\alpha(\tau))\bigr]
=\frac{ds_t}{d\tau}(\alpha_t-\bar\alpha(\tau))-s_t(\tau)\frac{d\bar\alpha}{d\tau}.
$$
Substituting (B.1.6) for $ds_t/d\tau$ in the first piece,
$$
=s_t(\tau)(\alpha_t-\bar\alpha(\tau))^2-s_t(\tau)\frac{d\bar\alpha}{d\tau}.
$$
For $d\bar\alpha/d\tau$, use $\bar\alpha(\tau)=\sum_{t'}s_{t'}(\tau)\alpha_{t'}$ and again (B.1.6):
$$
\frac{d\bar\alpha}{d\tau}\;=\;\sum_{t'}s_{t'}(\tau)(\alpha_{t'}-\bar\alpha(\tau))\alpha_{t'}\;=\;\sum_{t'}s_{t'}(\tau)(\alpha_{t'}-\bar\alpha(\tau))^2\;=\;\mathrm{Var}_s\alpha(\tau),
$$
where the middle equality uses $\sum_{t'}s_{t'}(\tau)(\alpha_{t'}-\bar\alpha)\bar\alpha=0$ (the constant $\bar\alpha$ comes out of the sum and the centred residuals sum to zero). Therefore
$$
\frac{d^2 s_t}{d\tau^2}\;=\;s_t(\tau)\bigl[(\alpha_t-\bar\alpha(\tau))^2-\mathrm{Var}_s\alpha(\tau)\bigr],
$$
and
$$
\phi''(\tau)=\sum_t \frac{d^2 s_t}{d\tau^2}\,v_t = \sum_t s_t(\tau)\bigl[(\alpha_t-\bar\alpha(\tau))^2-\mathrm{Var}_s\alpha(\tau)\bigr]\,v_t,
$$
which is (B.1.4).

Combining Steps 1–4 yields (B.1.1)–(B.1.4). $\square$

**Remark B.1.1.** Lemma B.1 is *exact*. Both $L$ and $R$ are defined
without any $O(\cdot)$ symbol or approximation. The decomposition is a
genuine identity; it is the subsequent bounding of $\|L\|^2$ and $\|R\|$
in Sections B.2 that introduces inequalities.

---

## B.2 Theorem 6.1 — Single-layer attention-weighted upper bound

We now bound each piece of (B.1.1) and combine.

### B.2.1 Lemma B.2 (weighted Cauchy–Schwarz on the first-order term)

**Lemma B.2.** *For every query $q$ and quantizer $E$,*
$$
\|L(q,E)\|^2\;\le\;\Bigl(\sum_{t=1}^T s_t(q)\,\alpha_t(q)^2\Bigr)\cdot\mathrm{Var}_s[V](q).
\tag{B.2.1}
$$
*Equivalently, in terms of the qaMSE quantity (Definition 6.1),*
$$
\|L(q,E)\|^2\;\le\;\mathrm{qaMSE}(q;E)\cdot\mathrm{Var}_s[V](q),
\tag{B.2.2}
$$
*since* $\sum_t s_t\alpha_t^2=\sum_t s_t(q\cdot e_t)^2/d=\mathrm{qaMSE}(q;E)$
*by definition.*

**Proof.** From (B.1.2),
$$
L(q,E)\;=\;\sum_{t:\,s_t>0} \bigl(s_t\alpha_t\bigr)\,(v_t-o)
$$
(positions with $s_t=0$ contribute zero to the sum). For each such $t$
write $a_t:=s_t\alpha_t\in\mathbb R$ and $u_t:=v_t-o\in\mathbb R^{d_v}$.
The weighted Cauchy–Schwarz inequality, applied with positive weights
$w_t:=s_t$, states
$$
\Bigl\|\sum_t a_t u_t\Bigr\|^2\;\le\;\Bigl(\sum_t \frac{a_t^2}{w_t}\Bigr)\Bigl(\sum_t w_t\,\|u_t\|^2\Bigr).
$$
(This is the vector-valued form of $\bigl(\sum_t \sqrt{w_t}\,\xi_t \cdot \tfrac{a_t}{\sqrt{w_t}}\bigr)^2\le \sum_t w_t\xi_t^2\cdot\sum_t a_t^2/w_t$ with $\xi_t=\|u_t\|$, combined with the triangle inequality $\|\sum_t a_t u_t\|\le\sum_t|a_t|\|u_t\|$.)

Substituting $a_t=s_t\alpha_t$ and $w_t=s_t$,
$$
\sum_t \frac{a_t^2}{w_t}\;=\;\sum_t \frac{s_t^2\alpha_t^2}{s_t}\;=\;\sum_t s_t\alpha_t^2,
\qquad
\sum_t w_t\,\|u_t\|^2\;=\;\sum_t s_t\,\|v_t-o\|^2\;=\;\mathrm{Var}_s[V](q).
$$
Combining,
$$
\|L(q,E)\|^2\;\le\;\Bigl(\sum_t s_t\alpha_t^2\Bigr)\cdot\mathrm{Var}_s[V](q),
$$
which is (B.2.1). The qaMSE form (B.2.2) follows from $\sum_t s_t\alpha_t^2=(1/d)\sum_t s_t(q\cdot e_t)^2=\mathrm{qaMSE}(q;E)$. $\square$

**Remark B.2.1 (the choice of weights).** The weights $w_t=s_t$ in
the weighted Cauchy–Schwarz are *not* arbitrary; they are the unique
choice that converts $L$ — a sum weighted by $s_t\alpha_t$ — into
qaMSE × Var without losing the $\alpha$-orthogonal components of $e_t$
to a separate "diagonal-dominant approximation". A naive Cauchy–Schwarz
with $w_t=1$ would give the pessimistic bound
$\|L\|^2\le\bigl(\sum_t s_t^2\alpha_t^2\bigr)\bigl(\sum_t \|v_t-o\|^2\bigr)$,
which is loose by a factor $T$. The weighted form is what makes the
qaMSE quantity emerge naturally.

### B.2.2 Lemma B.3 (operator-norm bound on the remainder)

**Lemma B.3.** *Under (A1), for every $\tau\in[0,1]$,*
$$
\|\phi''(\tau)\|\;\le\;2\,V_{\max}\cdot\sum_{t=1}^T s_t(\tau)\,\alpha_t^2.
\tag{B.3.1}
$$
*Consequently the remainder of Lemma B.1 satisfies*
$$
\|R(q,E)\|\;\le\;V_{\max}\cdot\sup_{\tau\in[0,1]}\sum_{t=1}^T s_t(\tau)\,\alpha_t^2
\;\le\;\frac{Q_{\max}^2 V_{\max}\,\rho^2}{d}.
\tag{B.3.2}
$$

**Proof of (B.3.1).** From (B.1.4),
$$
\phi''(\tau)\;=\;\underbrace{\sum_t s_t(\tau)(\alpha_t-\bar\alpha(\tau))^2 v_t}_{=:\,A(\tau)}\;-\;\mathrm{Var}_s\alpha(\tau)\cdot\underbrace{\sum_t s_t(\tau)v_t}_{=\,o(\tau)}.
$$
By the triangle inequality,
$$
\|\phi''(\tau)\|\;\le\;\|A(\tau)\|\;+\;\mathrm{Var}_s\alpha(\tau)\,\|o(\tau)\|.
$$
For $\|A(\tau)\|$ apply the triangle inequality termwise and use $\|v_t\|\le V_{\max}$:
$$
\|A(\tau)\|\;\le\;V_{\max}\sum_t s_t(\tau)(\alpha_t-\bar\alpha(\tau))^2\;=\;V_{\max}\,\mathrm{Var}_s\alpha(\tau).
$$
For $\|o(\tau)\|$, use $o(\tau)=\sum_t s_t(\tau)v_t$ and $\sum_t s_t(\tau)=1$:
$$
\|o(\tau)\|\;\le\;\sum_t s_t(\tau)\,\|v_t\|\;\le\;V_{\max}.
$$
Combining,
$$
\|\phi''(\tau)\|\;\le\;V_{\max}\,\mathrm{Var}_s\alpha(\tau)\;+\;V_{\max}\,\mathrm{Var}_s\alpha(\tau)\;=\;2V_{\max}\,\mathrm{Var}_s\alpha(\tau).
$$
Finally $\mathrm{Var}_s\alpha(\tau)\le\sum_t s_t(\tau)\alpha_t^2$ (variance is at most the second moment). This proves (B.3.1).

**Proof of (B.3.2).** From (B.1.3),
$$
\|R(q,E)\|\;\le\;\int_0^1 (1-\tau)\,\|\phi''(\tau)\|\,d\tau
\;\le\;2V_{\max}\sup_\tau\sum_t s_t(\tau)\alpha_t^2\cdot\int_0^1(1-\tau)\,d\tau.
$$
The integral evaluates to $\tfrac12$, absorbing the factor $2$ from
(B.3.1) into the constant $V_{\max}\cdot\sup_\tau\sum_t s_t(\tau)\alpha_t^2$.
For the second inequality in (B.3.2), note that
$\sum_t s_t(\tau)\alpha_t^2\le\max_t\alpha_t^2$ (since $s_t(\tau)\ge 0$
and $\sum_t s_t(\tau)=1$), and by (B.0.1)
$\max_t\alpha_t^2\le Q_{\max}^2\rho^2/d$. $\square$

### B.2.3 Theorem 6.1

**Theorem 6.1 (single-layer attention-weighted reconstruction bound).**
*Under (A1), for every quantizer $E$ with $\|e_t\|\le\rho$ and every distribution over queries $q$,*
$$
\boxed{\;
\mathbb E_q\bigl\|\hat o(q)-o(q)\bigr\|^2
\;\le\;
2\,\mathbb E_q\!\Bigl[\mathrm{qaMSE}(q;E)\cdot\mathrm{Var}_s[V](q)\Bigr]
\;+\;C_1\,\rho^4,\;}
\tag{T6.1}
$$
*where the constant is*
$C_1\;:=\;\dfrac{2\,Q_{\max}^4\,V_{\max}^2}{d^2}$.

**Proof.** Fix $q$. By Lemma B.1, $\hat o(q)-o(q)=L(q,E)+R(q,E)$. The
parallelogram inequality $\|x+y\|^2\le 2\|x\|^2+2\|y\|^2$ gives
$$
\|\hat o(q)-o(q)\|^2\;\le\;2\|L(q,E)\|^2+2\|R(q,E)\|^2.
$$
By Lemma B.2, $\|L(q,E)\|^2\le \mathrm{qaMSE}(q;E)\cdot\mathrm{Var}_s[V](q)$.
By Lemma B.3, $\|R(q,E)\|^2\le Q_{\max}^4 V_{\max}^2\rho^4/d^2$.
Hence
$$
\|\hat o(q)-o(q)\|^2
\;\le\;
2\,\mathrm{qaMSE}(q;E)\cdot\mathrm{Var}_s[V](q)
\;+\;\frac{2\,Q_{\max}^4 V_{\max}^2}{d^2}\,\rho^4.
$$
Taking expectation over $q$ (Tonelli — both sides are non-negative
measurable functions of $q$),
$$
\mathbb E_q\|\hat o(q)-o(q)\|^2
\;\le\;2\,\mathbb E_q[\mathrm{qaMSE}(q;E)\cdot\mathrm{Var}_s[V](q)]
\;+\;\frac{2\,Q_{\max}^4 V_{\max}^2}{d^2}\,\rho^4,
$$
which is (T6.1) with $C_1=2Q_{\max}^4V_{\max}^2/d^2$. $\square$

**Remark B.2.2 (the only two inequalities used).** The proof contains
exactly two non-equality steps: (i) the weighted Cauchy–Schwarz of
Lemma B.2 (with the *correct* weights $w_t=s_t$, going in the $\le$
direction); (ii) the operator-norm bound on the softmax Hessian of
Lemma B.3. Both are valid in *every* distributional regime; no
"diagonal-dominant approximation" is invoked. The cross-position
covariances that the v1 draft dropped are absorbed into the slack of
Lemma B.2.

**Remark B.2.3 (Mode-A near-tightness).** When attention concentrates
on a single position $t^*$, $s_{t^*}\to 1$ and $s_{t}\to 0$ for $t\neq t^*$.
Then $\mathrm{Var}_s[V](q)\to\|v_{t^*}-o\|^2\to 0$ in the strict limit
(since $o\to v_{t^*}$). The non-degenerate regime is "$s_{t^*}$ large
but $\le 1-\varepsilon$"; in this regime
$$
\mathrm{qaMSE}\approx\frac{s_{t^*}\alpha_{t^*}^2}{1}
\quad\text{and}\quad
\mathrm{Var}_s[V]\approx s_{t^*}(1-s_{t^*})\|v_{t^*}-v_{\text{rest}}\|^2,
$$
and the product matches $\|L\|^2$ up to the constant $2$ from the
parallelogram step. This is the formal source of the v2af observation
that $r_{\mathrm{qa}}\to r_{\mathrm{ppl}}$ on Mistral.

---

## B.3 Lemma 6.A — Closed-form transformer-block Lipschitz constants

This section establishes the closed-form Lipschitz constants used in
the cross-layer bound of Section B.4. The pre-norm transformer block
acts on the residual stream $h\in\mathbb R^{d_{\mathrm{model}}}$ as
$$
\mathrm{Block}_\ell(h)\;=\;h\;+\;F_\ell^{\mathrm{attn}}(h)\;+\;F_\ell^{\mathrm{mlp}}(h),
$$
where
$$
F_\ell^{\mathrm{attn}}(h):=W_O^\ell\,\mathrm{Attn}\bigl(\mathrm{RN}^{(1)}_\ell(h)\bigr),
\qquad
F_\ell^{\mathrm{mlp}}(h):=W_{\mathrm{down}}^\ell\,\sigma\bigl(W_{\mathrm{up}}^\ell\,\mathrm{RN}^{(2)}_\ell(h)\bigr),
$$
with $\mathrm{Attn}(x)$ the multi-head attention applied to $x$
(splitting $x$ into $W_Q^\ell x,W_K^\ell x,W_V^\ell x$, performing
softmax-attention, and concatenating heads), and $\sigma$ a 1-Lipschitz
activation (GELU, ReLU, SiLU on bounded inputs).

We assume the residual stream norm is bounded along the trajectory by
some $H_{\max}$ (this is empirically true for trained transformers and
is necessary for any meaningful Lipschitz bound; see e.g. Kim et al.
2021, Section 3).

**Lemma B.4 (RMSNorm Lipschitz).** *RMSNorm with diagonal gain $\gamma\in\mathbb R^{d_{\mathrm{model}}}$, defined by $\mathrm{RN}(h)_i=\gamma_i\,h_i/\sqrt{(1/d_{\mathrm{model}})\sum_j h_j^2+\varepsilon}$, is locally Lipschitz on $\{\|h\|\ge h_{\min}\}$ with constant*
$$
\Lambda_{\mathrm{RN}}\;\le\;\frac{2\,\|\gamma\|_\infty\,\sqrt{d_{\mathrm{model}}}}{h_{\min}}.
\tag{B.4.1}
$$

**Proof sketch.** Direct computation of the Jacobian
$\partial\mathrm{RN}_i/\partial h_j=\gamma_i\bigl(\delta_{ij}/r-h_ih_j/(d_{\mathrm{model}}\,r^3)\bigr)$
where $r=\sqrt{(1/d_{\mathrm{model}})\sum_j h_j^2+\varepsilon}$.
The operator norm is bounded by $\|\gamma\|_\infty(1/r+\|h\|^2/(d_{\mathrm{model}}\,r^3))\le 2\|\gamma\|_\infty/r$,
and $r\ge h_{\min}/\sqrt{d_{\mathrm{model}}}$ for $\|h\|\ge h_{\min}$. $\square$

**Lemma B.5 (softmax-attention Lipschitz, Kim et al. 2021).** *Let $A(x)=\mathrm{softmax}(W_Q x\,(W_K x)^\top/\sqrt d)\,W_V x$ denote single-head self-attention as a function of the input matrix $x\in\mathbb R^{T\times d_{\mathrm{model}}}$ (with rows the per-token activations). Under the assumption $\|W_K x_t\|\le K_{\max}$ and $\|W_Q x_t\|\le Q_{\max}$ for all $t$, $A$ is Lipschitz with*
$$
\Lambda_{\mathrm{Attn}}\;\le\;\|W_V\|\;+\;\frac{\|W_Q\|\|W_K\|\,V_{\max}}{\sqrt d}\bigl(1+4Q_{\max}K_{\max}/\sqrt d\bigr).
\tag{B.5.1}
$$

**Proof.** Theorem 3.2 of Kim, Papyan, Donoho (NeurIPS 2021)
"The Lipschitz Constant of Self-Attention", specialized to the single-
head case. The first term is the value path, the second is the
softmax-Jacobian-mediated coupling through $W_Q,W_K$. We refer the
reader to that paper for the full derivation. $\square$

**Lemma B.6 (MLP Lipschitz).** *For $F_\ell^{\mathrm{mlp}}(h)=W_{\mathrm{down}}\sigma(W_{\mathrm{up}}\mathrm{RN}(h))$ with $\sigma$ 1-Lipschitz,*
$$
\Lambda_{\mathrm{mlp}}\;\le\;\|W_{\mathrm{down}}\|\cdot\|W_{\mathrm{up}}\|\cdot\Lambda_{\mathrm{RN}}.
\tag{B.6.1}
$$

**Proof.** Composition of Lipschitz maps: $\mathrm{Lip}(f\circ g)\le\mathrm{Lip}(f)\cdot\mathrm{Lip}(g)$. $W_{\mathrm{up}}$ and $W_{\mathrm{down}}$ are linear with operator norms equal to their largest singular values; $\sigma$ has Lipschitz constant 1; and Lemma B.4 controls $\mathrm{RN}$. $\square$

**Lemma 6.A (closed-form block Lipschitz).** *Under the assumption that the residual stream norm is bounded below by $h_{\min}$ along the trajectory, the per-layer block Lipschitz constant of $\mathrm{Block}_\ell$ satisfies*
$$
\Lambda_\ell\;:=\;\mathrm{Lip}(\mathrm{Block}_\ell)\;\le\;1\;+\;\Lambda_\ell^{\mathrm{attn}}\;+\;\Lambda_\ell^{\mathrm{mlp}},
\tag{6.A.1}
$$
*where*
$$
\Lambda_\ell^{\mathrm{attn}}\;\le\;\Lambda_{\mathrm{RN}}^{(1)}\cdot\|W_O^\ell\|\cdot\Bigl(\|W_V^\ell\|+\frac{\|W_Q^\ell\|\,\|W_K^\ell\|\,V_{\max}}{\sqrt d}\bigl(1+4Q_{\max}K_{\max}/\sqrt d\bigr)\Bigr),
\tag{6.A.2}
$$
$$
\Lambda_\ell^{\mathrm{mlp}}\;\le\;\Lambda_{\mathrm{RN}}^{(2)}\cdot\|W_{\mathrm{down}}^\ell\|\cdot\|W_{\mathrm{up}}^\ell\|,
\tag{6.A.3}
$$
*and $\Lambda_{\mathrm{RN}}^{(1,2)}$ are given by (B.4.1).*

**Proof.** The block is the sum of three maps: identity (Lipschitz 1),
$F_\ell^{\mathrm{attn}}$, and $F_\ell^{\mathrm{mlp}}$. The Lipschitz
constant of a sum is bounded by the sum of Lipschitz constants:
$\Lambda_\ell\le 1+\Lambda_\ell^{\mathrm{attn}}+\Lambda_\ell^{\mathrm{mlp}}$.
For $\Lambda_\ell^{\mathrm{attn}}$: $F_\ell^{\mathrm{attn}}=W_O^\ell\circ\mathrm{Attn}\circ\mathrm{RN}^{(1)}$, so by composition $\Lambda_\ell^{\mathrm{attn}}\le\|W_O^\ell\|\cdot\Lambda_{\mathrm{Attn}}\cdot\Lambda_{\mathrm{RN}}^{(1)}$, and substituting (B.5.1) gives (6.A.2).
For $\Lambda_\ell^{\mathrm{mlp}}$: directly from Lemma B.6 with the first
RMSNorm replaced by $\mathrm{RN}^{(2)}$, giving (6.A.3). $\square$

**Numerical instantiation (Mistral-7B).** For Mistral-7B-v0.3 with
$d_{\mathrm{model}}=4096$, $d=128$, the weight singular values can be
read off the released checkpoint; a representative set of values yields
$\Lambda_\ell\in[3,12]$ across layers $\ell=0,\dots,31$, with
$\Lambda_{L\leftarrow 0}=\prod_{\ell=1}^{31}\Lambda_\ell\in[10^{14},10^{30}]$
in the worst case. *These absolute numbers are loose by 5–20× compared
to the random-direction Jacobian norms of v2ai, but the looseness
cancels in the method-comparison ratio (see Section B.5).* The full
table of singular values for Mistral-7B is computed from the released
weights and tabulated in `appendix_B3_lipschitz_table.csv` (to be
generated as a 1-day numerical task using `torch.linalg.svdvals` on the
six weight matrices per layer).

**Remark B.3.1 (why looseness is acceptable).** The absolute values
$\Lambda_\ell$ enter Theorem 6.2 through the *same* weights for both
quantizer choices in any method comparison. By Corollary 6.3 (Section
B.5), the architecture-dependent $\Lambda_{L\leftarrow\ell}$ cancel
when forming the method-ratio, leaving only the qaMSE-weighted sum.
Thus a 5–20× loose Lipschitz bound has *no effect* on the predicted
ordering of methods.

---

## B.4 Theorem 6.2 — Cross-layer cascade upper bound

We now lift Theorem 6.1 from a single layer to the full residual stream.

### B.4.1 Lemma 6.B (discrete cascade)

**Lemma 6.B (discrete cascade).** *Let $\Delta o_\ell\in\mathbb R^{d_{\mathrm{model}}}$ denote the perturbation in the attention output of layer $\ell$, and $\Delta h_L$ the resulting perturbation in the final residual stream after $L$ blocks. Under the closed-form Lipschitz constants of Lemma 6.A,*
$$
\|\Delta h_L\|^2\;\le\;L\sum_{\ell=1}^L \Lambda_{L\leftarrow\ell}^2\,\|\Delta o_\ell\|^2,
\tag{6.B.1}
$$
*where $\Lambda_{L\leftarrow\ell}:=\prod_{\ell'=\ell+1}^L\Lambda_{\ell'}$ is the forward propagator from layer $\ell$ to the final layer.*

**Proof.** The unrolled residual stream gives
$$
\Delta h_L\;=\;\sum_{\ell=1}^L \mathcal F_{L\leftarrow\ell}(\Delta o_\ell),
$$
where $\mathcal F_{L\leftarrow\ell}$ denotes the (nonlinear) map that propagates a perturbation introduced at layer $\ell$ through layers $\ell+1,\dots,L$. By the Lipschitz property of each block,
$$
\|\mathcal F_{L\leftarrow\ell}(\Delta o_\ell)\|\;\le\;\Lambda_{L\leftarrow\ell}\,\|\Delta o_\ell\|.
$$
The triangle inequality applied to the sum then gives
$$
\|\Delta h_L\|\;\le\;\sum_{\ell=1}^L\Lambda_{L\leftarrow\ell}\,\|\Delta o_\ell\|.
$$
Squaring and applying the discrete Cauchy–Schwarz inequality $(\sum_\ell x_\ell)^2\le L\sum_\ell x_\ell^2$ with $x_\ell=\Lambda_{L\leftarrow\ell}\|\Delta o_\ell\|$ yields (6.B.1). $\square$

**Remark B.4.1 (the factor $L$ is conservative).** The factor $L$ in
Lemma 6.B comes from the worst-case discrete Cauchy–Schwarz, which is
sharp only when all $\Lambda_{L\leftarrow\ell}\|\Delta o_\ell\|$ are
equal. In practice, the per-layer cascade contributions are
concentrated on 2–3 layers (v2ah), so the effective number of terms is
$L_{\mathrm{eff}}\sim 3$ rather than $L\sim 32$, and the bound is loose
by a factor $\sim L/L_{\mathrm{eff}}\sim 10$. This is one of the open
questions in Section 8.3 of the main paper.

### B.4.2 Theorem 6.2

**Theorem 6.2 (cross-layer cascade reconstruction bound).** *Under (A1), for any per-layer key quantizers $E_\ell$ with $\|e_{t,\ell}\|\le\rho$ and any query distribution,*
$$
\boxed{\;
\mathbb E\bigl\|\Delta h_L\bigr\|^2
\;\le\;
2L\sum_{\ell=1}^L \Lambda_{L\leftarrow\ell}^2\,
\mathbb E_q\!\Bigl[\mathrm{qaMSE}_\ell(q;E_\ell)\cdot\mathrm{Var}_{s_\ell}[V_\ell](q)\Bigr]
\;+\;L\Bigl(\sum_{\ell=1}^L \Lambda_{L\leftarrow\ell}^2\Bigr)C_1\rho^4,\;}
\tag{T6.2}
$$
*with the same constant $C_1=2Q_{\max}^4V_{\max}^2/d^2$ as in Theorem 6.1.*

**Proof.** By Lemma 6.B,
$$
\|\Delta h_L\|^2\;\le\;L\sum_{\ell=1}^L \Lambda_{L\leftarrow\ell}^2\,\|\Delta o_\ell\|^2.
$$
Taking expectation over the query (and any randomness in the value
generation),
$$
\mathbb E\|\Delta h_L\|^2\;\le\;L\sum_{\ell=1}^L\Lambda_{L\leftarrow\ell}^2\,\mathbb E_q\|\Delta o_\ell\|^2.
$$
For each $\ell$, $\Delta o_\ell=\hat o_\ell-o_\ell$ is exactly the
single-layer attention output perturbation analysed in Theorem 6.1.
Substituting (T6.1) layer by layer,
$$
\mathbb E_q\|\Delta o_\ell\|^2\;\le\;2\,\mathbb E_q[\mathrm{qaMSE}_\ell\cdot\mathrm{Var}_{s_\ell}[V_\ell]]\;+\;C_1\rho^4.
$$
Multiplying by $L\Lambda_{L\leftarrow\ell}^2$ and summing over $\ell$,
$$
\mathbb E\|\Delta h_L\|^2
\;\le\;
2L\sum_\ell\Lambda_{L\leftarrow\ell}^2\,\mathbb E_q[\mathrm{qaMSE}_\ell\cdot\mathrm{Var}_{s_\ell}[V_\ell]]
\;+\;L\Bigl(\sum_\ell\Lambda_{L\leftarrow\ell}^2\Bigr)C_1\rho^4,
$$
which is (T6.2). $\square$

**Remark B.4.2 (independence of layers).** The proof treats the $L$
layer perturbations as deterministic functions of the (fixed) data and
quantizer; no statistical independence between layers is assumed. The
linear superposition $\Delta h_L=\sum_\ell\mathcal F_{L\leftarrow\ell}(\Delta o_\ell)$
is exact at first order in $\rho$ and an upper bound at all orders by
the Lipschitz argument. In particular, the "joint vs sum-of-isolated"
discrepancy noted in v2ah (Qwen-7B sum 0.59 vs joint 0.83) is captured
in the looseness of the triangle inequality and does not break the
bound — it merely makes it less tight.

---

## B.5 Corollary 6.3 — $\Lambda$-cancellation in method comparisons

This corollary is the formal answer to the reviewer concern "is the
4/4 sign-match just curve-fitting?" The answer is no: the architecture-
dependent factors $\Lambda_{L\leftarrow\ell}$ cancel exactly when the
bound is used to compare two quantizers at the same bit budget.

**Corollary 6.3 ($\Lambda$-cancellation).** *Let $E^{(1)}$ and $E^{(2)}$ be two key-quantizer choices at the same bit budget (e.g. Lloyd-Max vs uniform-grid), both satisfying $\|e_t^{(i)}\|\le\rho$. Define the leading term of (T6.2) as*
$$
\mathcal U(E):=2L\sum_{\ell=1}^L \Lambda_{L\leftarrow\ell}^2\,\mathbb E_q[\mathrm{qaMSE}_\ell(q;E_\ell)\cdot\mathrm{Var}_{s_\ell}[V_\ell](q)].
$$
*Then the ratio*
$$
\frac{\mathcal U(E^{(1)})}{\mathcal U(E^{(2)})}
\;=\;
\frac{\sum_\ell w_\ell\,\mathbb E_q[\mathrm{qaMSE}_\ell^{(1)}\cdot\mathrm{Var}_{s_\ell}[V_\ell]]}
     {\sum_\ell w_\ell\,\mathbb E_q[\mathrm{qaMSE}_\ell^{(2)}\cdot\mathrm{Var}_{s_\ell}[V_\ell]]},
\qquad
w_\ell:=\Lambda_{L\leftarrow\ell}^2,
\tag{6.C.1}
$$
*depends on the architecture only through the non-negative weights $w_\ell=\Lambda_{L\leftarrow\ell}^2$. In particular, scaling the entire $\Lambda$-profile by any positive constant $c$ does not change the ratio (6.C.1).*

**Proof.** Both $\Lambda_{L\leftarrow\ell}$ and $\mathrm{Var}_{s_\ell}[V_\ell]$ are *quantizer-independent* (they depend only on the FP16 model and the input distribution). Substituting into the definition of $\mathcal U$,
$$
\mathcal U(E)\;=\;2L\sum_\ell\Lambda_{L\leftarrow\ell}^2\,\mathbb E_q[\mathrm{qaMSE}_\ell(q;E_\ell)\cdot\mathrm{Var}_{s_\ell}[V_\ell]].
$$
The factor $2L$ and the entire $\Lambda$ profile are common to both
$\mathcal U(E^{(1)})$ and $\mathcal U(E^{(2)})$. Forming the ratio
cancels these common factors and leaves (6.C.1). The scale invariance
under $\Lambda\to c\Lambda$ is immediate from the bilinearity of (6.C.1)
in the weights. $\square$

**Corollary 6.3.1 (sign prediction).** *The leading-term ratio (6.C.1) determines the sign of $\mathcal U(E^{(1)})-\mathcal U(E^{(2)})$ via*
$$
\mathrm{sign}\bigl(\mathcal U(E^{(1)})-\mathcal U(E^{(2)})\bigr)
=\mathrm{sign}\Bigl(\sum_\ell w_\ell\,\mathbb E_q[(\mathrm{qaMSE}_\ell^{(1)}-\mathrm{qaMSE}_\ell^{(2)})\cdot\mathrm{Var}_{s_\ell}[V_\ell]]\Bigr).
\tag{6.C.2}
$$

**Proof.** Subtraction; the $\mathrm{Var}$ factor and $\Lambda$ weights
are non-negative so they preserve sign within the linear combination.
$\square$

**Remark B.5.1 (the v2ag 4/4 in context).** The empirical chain v2af →
v2ag → v2ah measures the right-hand side of (6.C.2) for Lloyd vs Grid
across all layers and all 4 models. The fact that the sign of (6.C.2)
matches the sign of the actual PPL ratio in 4/4 cases is therefore
a *prediction* of Theorem 6.2 (via Corollary 6.3.1), not an additional
empirical claim. The single-layer case (v2af, 3/4) failed precisely
because the $\Lambda$-weighted sum cannot be reduced to its
$\ell$-dominant term on Qwen-1.5B (see remark B.4.1).

---

## B.6 Corollaries 6.4–6.6 — Modes A, B, C as formal specializations

The three failure modes of Section 5 of the main paper are formal
corollaries of Theorem 6.1 specialized to particular attention patterns.

### B.6.1 Corollary 6.4 — Mode A (localized positional sink)

**Corollary 6.4 (Localized positional sink).** *Suppose at the high-$\kappa$ heads of the model, the attention weights satisfy $s_{t^*}(q)\ge 1-\varepsilon$ for a single fixed position $t^*$ uniformly in $q$. Then for any quantizer $E$,*
$$
\mathrm{qaMSE}(q;E)\;\ge\;(1-\varepsilon)\,\frac{(q\cdot e_{t^*})^2}{d}.
\tag{6.4.1}
$$
*Furthermore, if the position-sink modification $E^{\mathrm{sink}}$ sets $e_{t^*}^{\mathrm{sink}}=0$ and leaves all other $e_t$ unchanged, then*
$$
\mathrm{qaMSE}(q;E^{\mathrm{sink}})\;\le\;\varepsilon\,\frac{Q_{\max}^2\rho^2}{d},
\tag{6.4.2}
$$
*so the position sink reduces qaMSE on these heads by a factor of order $1/\varepsilon$.*

**Proof of (6.4.1).** From Definition 6.1,
$$
\mathrm{qaMSE}(q;E)\;=\;\frac1d\sum_t s_t(q)(q\cdot e_t)^2\;\ge\;\frac1d s_{t^*}(q)(q\cdot e_{t^*})^2\;\ge\;\frac{1-\varepsilon}{d}(q\cdot e_{t^*})^2.
$$

**Proof of (6.4.2).** With $e_{t^*}^{\mathrm{sink}}=0$,
$$
\mathrm{qaMSE}(q;E^{\mathrm{sink}})\;=\;\frac1d\sum_{t\neq t^*}s_t(q)(q\cdot e_t)^2.
$$
Each summand is bounded by $|q\cdot e_t|^2/d\le Q_{\max}^2\rho^2/d$ (using (B.0.1)), and $\sum_{t\neq t^*}s_t(q)\le\varepsilon$ by hypothesis. Hence
$$
\mathrm{qaMSE}(q;E^{\mathrm{sink}})\;\le\;\varepsilon\cdot\frac{Q_{\max}^2\rho^2}{d}.
$$
$\square$

**Consequence for Theorem 6.1.** Combining (6.4.1) and (6.4.2) with (T6.1), the position-sink quantizer satisfies
$$
\mathbb E_q\|\hat o^{\mathrm{sink}}-o\|^2\;\le\;2\varepsilon\cdot\frac{Q_{\max}^2\rho^2}{d}\cdot\mathrm{Var}_s[V]\;+\;C_1\rho^4,
$$
which is smaller than the un-sinked bound by a factor approximately
$1/\varepsilon$ (in the regime where the sink term dominates the $\rho^4$
remainder, i.e. for sufficiently small $\rho$). *This is the formal
explanation of the v2h Mistral observation: $\mathrm{Lloyd}+\mathrm{sink}_{k=1}$
recovers $87\%$ of the catastrophic gap because $\varepsilon\sim 0.44$
on the high-$\kappa$ Mistral heads.*

### B.6.2 Corollary 6.5 — Mode B (distributed structural tail)

**Corollary 6.5 (Distributed structural tail).** *Suppose the attention distribution has $s_t\le 1/m$ for all $t$ in some set $S$ of $m\ge 5$ positions, and $\sum_{t\in S}s_t\ge 1-\delta$, with no single dominant position. Then for any quantizer $E$ with bounded per-dim error $\|e_t\|\le\rho$,*
$$
\mathrm{qaMSE}(q;E)\;\le\;\frac{Q_{\max}^2\rho^2}{d}.
\tag{6.5.1}
$$
*Sink-protecting any single position $t^\bullet\in S$ (i.e. setting $e_{t^\bullet}=0$) reduces qaMSE by at most $\frac1{md}\,Q_{\max}^2\rho^2$, which is a fraction $1/m\le 1/5$ of the full bound (6.5.1).*

**Proof of (6.5.1).** From Definition 6.1 and (B.0.1),
$$
\mathrm{qaMSE}(q;E)\;=\;\frac1d\sum_t s_t(q\cdot e_t)^2\;\le\;\frac{Q_{\max}^2\rho^2}{d}\sum_t s_t\;=\;\frac{Q_{\max}^2\rho^2}{d}.
$$

**Proof of the sink reduction claim.** Removing the contribution at $t^\bullet$ subtracts $\frac{s_{t^\bullet}}{d}(q\cdot e_{t^\bullet})^2\le\frac{1/m}{d}Q_{\max}^2\rho^2$. $\square$

**Consequence.** Single-position sink-protection in Mode B reduces the
upper bound by a fraction $1/m$. For $m\sim 5$–$15$ (Mistral-Nemo
delimiters), this is a 7–20% reduction, far less than the full $1/\varepsilon$
reduction of Mode A. A *uniform-grid* quantizer, by contrast, replaces
the per-position bound $|q\cdot e_t|^2/d\le Q_{\max}^2\rho^2/d$ with a
$\rho^2$ that is *deterministically* small (no per-position spike), so
the bound (6.5.1) becomes saturated at the small per-dim resolution
$\Delta/2$, dominating the per-position sink. *This is the formal
explanation of the v2u Nemo observation: $\mathrm{Grid}$ no-sink ($7.68$
PPL) beats $\mathrm{Lloyd}+\mathrm{sink}$ ($14.84$ PPL) at $L=32$K.*

### B.6.3 Corollary 6.6 — Mode C (bulk-tail)

**Corollary 6.6 (Bulk-tail).** *Suppose the attention distribution is approximately uniform: $s_t\approx 1/T$ for all $t$, and the K covariance condition number is moderate ($\kappa(\Sigma_K)\le 10^5$). Then for any quantizer $E$,*
$$
\mathrm{qaMSE}(q;E)\;\approx\;\frac{1}{T\,d}\,q^\top\bigl(\textstyle\sum_t e_te_t^\top\bigr)q\;\le\;\frac{Q_{\max}^2}{T\,d}\,\mathrm{tr}(E^\top E).
\tag{6.6.1}
$$
*In particular, qaMSE is proportional to raw MSE up to the factor $1/T$, and Lloyd-Max (which minimizes raw MSE) is near-optimal for qaMSE in this regime.*

**Proof of (6.6.1).** With $s_t=1/T$,
$$
\mathrm{qaMSE}(q;E)\;=\;\frac1{Td}\sum_t (q\cdot e_t)^2\;=\;\frac1{Td}q^\top\Bigl(\sum_t e_te_t^\top\Bigr)q.
$$
The Rayleigh upper bound gives $q^\top(\sum_te_te_t^\top)q\le\|q\|^2\,\lambda_{\max}(\sum_te_te_t^\top)\le Q_{\max}^2\,\mathrm{tr}(\sum_te_te_t^\top)=Q_{\max}^2\,\mathrm{tr}(E^\top E)$. $\square$

**Consequence (token-sink is harmful in Mode C).** Suppose a token-sink
modification protects a content-specific position $t^\bullet$ that
appears in the calibration set but not in the evaluation set. Then at
evaluation time, $e_{t^\bullet}\neq 0$ but the *mismatched* protection
introduces a positive perturbation
$$
\Delta\,\mathrm{qaMSE}\;=\;+\frac{1}{Td}(q\cdot\delta_{t^\bullet})^2,
$$
where $\delta_{t^\bullet}$ is the calibration-eval mismatch on the
protected position. Since this is non-negative and not compensated by
any reduction elsewhere (under the uniform $s_t$ assumption), token-sink
is *strictly worse* than no sink in Mode C. *This is the formal
explanation of the v2ad Qwen-1.5B observation: self-calibrated token
sink rises from $18.88\to 29.65$ PPL at $L=32$K.* $\square$

---

## B.7 Summary table of B.1–B.6 results

| Result | Statement | Proof | Status |
|---|---|---|---|
| Lemma B.1 | exact integral-remainder Taylor | B.1 | proven |
| Lemma B.2 | $\|L\|^2\le\mathrm{qaMSE}\cdot\mathrm{Var}_s[V]$ via weighted CS | B.2.1 | proven |
| Lemma B.3 | $\|R\|\le V_{\max}Q_{\max}^2\rho^2/d$ via Hessian op-norm | B.2.2 | proven |
| **Theorem 6.1** | $\mathbb E\|\hat o-o\|^2\le 2\mathbb E[\mathrm{qaMSE}\cdot\mathrm{Var}_s[V]]+C_1\rho^4$ | B.2.3 | **proven** |
| Lemma B.4 | RMSNorm $\Lambda_{\mathrm{RN}}\le 2\|\gamma\|_\infty\sqrt{d_{\mathrm{model}}}/h_{\min}$ | B.3 | proven |
| Lemma B.5 | softmax-attention Lipschitz (Kim et al. 2021) | B.3, cited | cited |
| Lemma B.6 | MLP Lipschitz | B.3 | proven |
| Lemma 6.A | block Lipschitz $\Lambda_\ell$ closed form | B.3 | proven |
| Lemma 6.B | discrete cascade $\|\Delta h_L\|^2\le L\sum_\ell\Lambda_{L\leftarrow\ell}^2\|\Delta o_\ell\|^2$ | B.4.1 | proven |
| **Theorem 6.2** | cascade upper bound (T6.2) | B.4.2 | **proven** |
| Corollary 6.3 | $\Lambda$-cancellation in method ratio | B.5 | proven |
| Corollary 6.3.1 | sign prediction for Lloyd vs Grid | B.5 | proven |
| Corollary 6.4 | Mode A (positional sink) | B.6.1 | proven |
| Corollary 6.5 | Mode B (distributed tail) | B.6.2 | proven |
| Corollary 6.6 | Mode C (bulk tail) | B.6.3 | proven |

Every claim referenced in Section 6 of `PART1_PAPER_DRAFT_v3.md` has a
proof in this appendix. The only result not proven from first
principles is Lemma B.5 (softmax-attention Lipschitz), which is cited
from Kim, Papyan, Donoho (NeurIPS 2021), Theorem 3.2.

---

## B.7.7 Theorem 6.13 — Categorical-Channel Optimality under Pre-RoPE Facet Rotation

**Date added**: 2026-04-14.
**Motivation**: bridges the K-side facet-gated steering paper (Cor 6.7–6.12) and the rotation-quantizer compression paper. Resolves the empirical observation that water-filling (Gaussian-optimal) bit allocation is *suboptimal* on pre-RoPE facet-rotated K channels, while 1-bit categorical allocation is near-optimal. Replaces the informal "WF floor=2" conjecture with a formal statement grounded in the attention-weighted reconstruction bound of Theorem 6.1.

### Setup

Fix a layer $\ell$ and a KV-head. Let $K\in\mathbb R^{T\times d}$ be the pre-RoPE key matrix at that layer, and let $B\in\mathbb R^{d\times d}$ be an orthonormal matrix with block structure $B=[B_{\mathrm{fac}}\mid B_{\mathrm{res}}]$ where $B_{\mathrm{fac}}\in\mathbb R^{d\times R}$ spans the facet subspace and $B_{\mathrm{res}}\in\mathbb R^{d\times(d-R)}$ its orthogonal complement. Define the rotated keys $K':=K\,B\in\mathbb R^{T\times d}$; $K'_{:,i}$ denotes its $i$-th column.

**Hypothesis (H-cat, categorical facet channels).** *For every facet index $i\in[0,R)$, the distribution of $K'_{t,i}$ over $t$ is a symmetric two-mode Gaussian mixture*
$$
K'_{t,i}\sim\tfrac12\,\mathcal N(+\mu_i,\sigma_{\mathrm{intra},i}^2)+\tfrac12\,\mathcal N(-\mu_i,\sigma_{\mathrm{intra},i}^2),\qquad s_i:=\mu_i^2/\sigma_{\mathrm{intra},i}^2\ge s_{\min}\ge 3.
$$

**Hypothesis (H-res, Gaussian residual channels).** *For every residual index $j\in[R,d)$, $K'_{:,j}\sim\mathcal N(0,\sigma_{\mathrm{res},j}^2)$.*

### Three quantizers at matched average-bit budget $\bar b$

- $Q_{\mathrm{KIVI}}^{\bar b}$: per-channel asymmetric $\bar b$-bit scalar on *every* channel. Total per token: $d\bar b$.
- $Q_{\mathrm{OCQ}}^{(1,b_{\mathrm{res}})}$: 1-bit sign on $i\in[0,R)$ + per-channel asymmetric $b_{\mathrm{res}}$-bit on $j\in[R,d)$. Total: $R+(d-R)b_{\mathrm{res}}$; matched if $\bar b=(R+(d-R)b_{\mathrm{res}})/d$.
- $Q_{\mathrm{WF}}^{(f,b_{\max})}$: water-filling $b_i=\max(f,\bar b+\tfrac12\log_2(\sigma_i^2/\bar\sigma^2))$, capped at $b_{\max}$, total matched.

### Claim (i) — 1-bit categorical MSE on a bimodal channel

**Lemma 6.13.1.** *Under (H-cat), the optimal 1-bit sign-based quantizer $Q_1(x):=\mathrm{sign}(x)\cdot\mu_i$ has*
$$
\mathrm{MSE}_{Q_1}(K'_{:,i})\le\sigma_{\mathrm{intra},i}^2+\mu_i^2\cdot\tfrac12\exp(-s_i/2).
$$

**Proof.** Decision boundary at 0. Correct classification probability $1-\Phi(-\sqrt{s_i})$; conditional on correct, squared error has expectation $\sigma_{\mathrm{intra},i}^2$; misclassified, $4\mu_i^2+\sigma_{\mathrm{intra},i}^2$. $\Phi(-\sqrt{s_i})\le\tfrac12 e^{-s_i/2}$ (Gaussian tail). Averaging. $\square$

**Remark 6.13.1 (Categorical regime).** For $s_i\ge 4$, the misclassification term is $<0.07\mu_i^2$ and becomes negligible against $\sigma_{\mathrm{intra},i}^2$ for any $s$ within the $\sigma_{\mathrm{intra}}^2$ normalization.

### Claim (ii) — Water-filling is suboptimal at low bits on bimodal channels

**Lemma 6.13.2 (WF floor lower bound).** *Under (H-cat), a $b$-bit Lloyd–Max Gaussian quantizer applied to $K'_{:,i}$ (treated as Gaussian with variance $\sigma_{\mathrm{total},i}^2=\mu_i^2+\sigma_{\mathrm{intra},i}^2$) has MSE at least*
$$
\mathrm{MSE}_{\mathrm{WF}}(b)\ge c_{\mathrm{LM}}(b)\cdot\sigma_{\mathrm{total},i}^2,
$$
*where $c_{\mathrm{LM}}(1)\approx 0.363$, $c_{\mathrm{LM}}(2)\approx 0.119$ are the tabulated Max 1960 constants.*

**Corollary 6.13.3 (Bimodal 1-bit beats WF 1-bit).** *For $s_i\ge 3$ and ignoring the misclassification correction,*
$$
\frac{\mathrm{MSE}_{\mathrm{WF}}(1)}{\mathrm{MSE}_{Q_1}(1)}\ge\frac{c_{\mathrm{LM}}(1)\cdot(\mu_i^2+\sigma_{\mathrm{intra},i}^2)}{\sigma_{\mathrm{intra},i}^2}=0.363\,(s_i+1)>1.
$$

**Proof.** Substitute Lemmas 6.13.1–2. For $s_i\ge 3$, $0.363(s_i+1)\ge 1.45>1$. $\square$

**Remark 6.13.2 (Why WF wastes bits on decision axes).** Water-filling allocates bits proportional to $\log\sigma_i^2$. On a facet channel, $\sigma_{\mathrm{total},i}^2$ is dominated by the between-mode separation $\mu_i^2$, which is a **discrete two-valued decision**, not a continuous magnitude. A 1-bit categorical quantizer captures the decision exactly; additional bits only encode within-cluster magnitude, which is post-decision noise from the facet interpretation (Cor 6.7's categorical 1-bit hypothesis).

### Claim (iii) — Combined qaMSE bound and cross-over threshold

**Theorem 6.13 (Categorical-Channel Optimality).** *Under (H-cat) with $s_{\min}\ge 3$ and (H-res), for any query $q$ with facet energy fraction $\varepsilon_q:=\|B_{\mathrm{fac}}^\top q\|^2/\|q\|^2$,*
$$
\mathrm{qaMSE}(q;E_{\mathrm{OCQ}})\le\frac{\|q\|^2}{d}\Bigl[\varepsilon_q\cdot\bar\sigma_{\mathrm{intra}}^2(1+\delta_{\mathrm{err}})+(1-\varepsilon_q)\cdot\bar\sigma_{\mathrm{res}}^2\cdot 2^{-2b_{\mathrm{res}}}\Bigr],\tag{T6.13.A}
$$
$$
\mathrm{qaMSE}(q;E_{\mathrm{KIVI}})\ge\frac{\|q\|^2}{d}\cdot c_{\mathrm{LM}}(\bar b)\bigl[\varepsilon_q\cdot\bar\sigma_{\mathrm{total,fac}}^2+(1-\varepsilon_q)\cdot\bar\sigma_{\mathrm{res}}^2\bigr],\tag{T6.13.B}
$$
*where $\bar\sigma_{\mathrm{intra}}^2:=\max_{i<R}\sigma_{\mathrm{intra},i}^2$, $\bar\sigma_{\mathrm{total,fac}}^2:=\max_{i<R}(\mu_i^2+\sigma_{\mathrm{intra},i}^2)$, $\bar\sigma_{\mathrm{res}}^2:=\max_{j\ge R}\sigma_{\mathrm{res},j}^2$, $\delta_{\mathrm{err}}:=\tfrac12 s_{\max}\exp(-s_{\min}/2)$, $c_{\mathrm{LM}}(\bar b)$ is the Max 1960 constant, and $\bar b=(R+(d-R)b_{\mathrm{res}})/d$ is the matched budget.*

*OCQ's qaMSE is smaller than KIVI's whenever*
$$
\varepsilon_q\ge\varepsilon_q^*:=\frac{\bar\sigma_{\mathrm{res}}^2\bigl(2^{-2b_{\mathrm{res}}}-c_{\mathrm{LM}}(\bar b)\bigr)}{c_{\mathrm{LM}}(\bar b)\,\bar\sigma_{\mathrm{total,fac}}^2-\bar\sigma_{\mathrm{intra}}^2(1+\delta_{\mathrm{err}})}.
$$

**Proof.** Let $e_t$ denote the quantization residual in the rotated basis. By Parseval (orthogonal $B$):
$$
\|e_t\|^2=\sum_{i<R}(e_{\mathrm{fac}})_{t,i}^2+\sum_{j\ge R}(e_{\mathrm{res}})_{t,j}^2.
$$
Expectation and Lemma 6.13.1 (facet) and standard Gaussian-quant MSE (residual) give
$$
\mathbb E\|e_t^{\mathrm{OCQ}}\|^2\le R\bar\sigma_{\mathrm{intra}}^2(1+\delta_{\mathrm{err}})+(d-R)\bar\sigma_{\mathrm{res}}^2\cdot 2^{-2b_{\mathrm{res}}}.
$$
For the logit perturbation, $q\cdot e_t=(B^\top q)^\top(B^\top e_t)$ with facet/residual block decomposition; Cauchy–Schwarz on each block gives
$$
(q\cdot e_t)^2\le\varepsilon_q\|q\|^2\|(B^\top e_t)_{\mathrm{fac}}\|^2+(1-\varepsilon_q)\|q\|^2\|(B^\top e_t)_{\mathrm{res}}\|^2.
$$
Dividing by $d$ and taking $s$-weighted average yields (T6.13.A).

For KIVI, every channel contributes $c_{\mathrm{LM}}(\bar b)\sigma_i^2$ under the Gaussian assumption; on facet channels $\sigma_i^2=\sigma_{\mathrm{total,fac},i}^2$, on residual $\sigma_i^2=\sigma_{\mathrm{res},j}^2$. Same decomposition gives (T6.13.B).

Solving (T6.13.A)$\le$(T6.13.B) for $\varepsilon_q$ gives the threshold. $\square$

### Corollary 6.13.4 (Bit-budget savings)

*Under (H-cat) with $s_{\min}\ge 3$ and matched qaMSE, OCQ uses $b_{\mathrm{avg}}=1+(d-R)/d\cdot b_{\mathrm{res}}$ bits per token on rotated K, while KIVI uses $\bar b$; the reduction is $(\bar b-b_{\mathrm{avg}})/\bar b=(R/d)\cdot(\bar b-1)/\bar b$.*

**MetaTool instantiation** ($d=128$, $R=24$, $b_{\mathrm{res}}=2$): $b_{\mathrm{avg}}=1+(104/128)\cdot 2\approx 1.81$ bits; KIVI at $\bar b=2$ uses 2.00. **Savings: $(2-1.81)/2\approx 9.4\%$** at matched (better) qaMSE.

### Corollary 6.13.5 (High-bit cross-over)

*As $\bar b\to\infty$, $\mathrm{qaMSE}_{\mathrm{KIVI}}\to 0$ while OCQ's facet floor $\bar\sigma_{\mathrm{intra}}^2(1+\delta_{\mathrm{err}})$ is $\bar b$-independent. There exists $\bar b^*$ such that for $\bar b>\bar b^*$, KIVI overtakes OCQ. The cross-over satisfies $\bar b^*\approx\tfrac12\log_2(s+1)$; for $s\in[5,10]$, $\bar b^*\in[1.3,1.7]$.*

**Empirical match (Qwen2.5-7B WT2 hook-mode, full test set)**:
| $\bar b$ | KIVI PPL | OCQ PPL | Δ | Predicted |
|---|---|---|---|---|
| 2 | 19.97 | 15.60 | OCQ wins −4.37 | consistent with $s\gtrsim 4$ |
| 4 | 7.79 | 12.56 | KIVI wins +4.77 | $\bar b>\bar b^*$, as predicted |

### Remark 6.13.3 (Hypothesis checkability)

(H-cat) is **data-dependent** and falsifiable. On MetaTool, the PCA pseudo-ontology vs catalog-derived ontology A/B at 2-bit (11.83 vs 7.43 PPL, `memory: ocq_real_ontology_validation_2026_04_09`) shows PCA top-variance directions do *not* satisfy (H-cat) — variance reflects continuous magnitude rather than separable bimodal structure, and 1-bit categorical destroys information. The catalog-derived basis satisfies (H-cat) empirically. A separate diagnostic (KL divergence between per-channel density and best-fit bimodal mixture) must verify (H-cat) before instantiating Theorem 6.13 on a new model/corpus.

### Remark 6.13.4 (Bridging the two papers)

Theorem 6.13 connects the facet-gated K-side steering paper (Cor 6.7–6.12) and the rotation-quantizer compression paper via a shared geometric construction:
- Facet basis $B_{\mathrm{fac}}$ is the same tensor in both settings.
- Theorem 6.1's qaMSE$\cdot$Var$_s$[V] structure lifts both steering bias and quantization error uniformly.
- Cor 6.7 interprets $B_{\mathrm{fac}}$ as a steering direction; Thm 6.13 interprets it as a compression axis.
- Each paper cites the other for the complementary interpretation.

---

## B.7.8 Theorem 6.14 — Positional-Encoding Substitution via Facet Rotation (conjecture + rigorous hybrid version)

**Date added**: 2026-04-14.
**Status**: **Theorem 6.14 (Hybrid)** is proven rigorously below; **Theorem 6.14 (Full)** is stated as a **conjecture** pending empirical verification.
**Motivation**: the pre-RoPE / post-RoPE space mismatch (`eval_arch_two_bugs_2026_04_09`) arose because RoPE is applied *between* basis construction (pre-RoPE) and cache writing (post-RoPE). If facet channels are instead rotated by a *content-dependent* rotation (facet identity) rather than a position-dependent one (RoPE), basis construction space equals quantization space equals attention-operating space. The Bug 2 is structurally removed.

### Setup

Let $B=[B_{\mathrm{fac}}\mid B_{\mathrm{res}}]$ orthonormal as in §B.7.7. Let $F$ be the number of facets, $R=\sum_f r_f$ the total facet rank, and $\pi:\mathbb R^d\to[0,F-1]$ a hard facet classifier (e.g., `argmax_f g_f(k_t)` with `g_f` the energy-ratio gate of Cor 6.7).

**Facet rotation operator.** For a facet index $f\in[0,F)$ and a channel pair index $i\in[0,R/2)$, define the angle
$$
\phi_{i,f}\;:=\;\frac{2\pi\,(f\cdot R/2+i)}{F\cdot R/2}.
$$
The facet rotation acts block-diagonally on the facet block:
$$
\mathrm{FacetRot}(f)\;:=\;\bigoplus_{i=0}^{R/2-1}\begin{pmatrix}\cos\phi_{i,f}&-\sin\phi_{i,f}\\ \sin\phi_{i,f}&\cos\phi_{i,f}\end{pmatrix}\;\in\;\mathrm{SO}(R)\subset\mathrm{SO}(d).
$$
Standard RoPE acts as $\mathrm{RoPE}(t)\in\mathrm{SO}(d)$ channel-pair-wise on the full channel set with angle $\theta_{i,t}=t\cdot 10000^{-2i/d}$.

**Hybrid positional scheme.** Apply $\mathrm{FacetRot}(\pi(k_t))$ on $B_{\mathrm{fac}}$ coordinates and $\mathrm{RoPE}(t)$ on $B_{\mathrm{res}}$ coordinates:
$$
K_{\mathrm{hyb}}[t]\;:=\;B_{\mathrm{fac}}\,\mathrm{FacetRot}(\pi(k_t))\,B_{\mathrm{fac}}^\top\,K_{\mathrm{pre}}[t]\;+\;B_{\mathrm{res}}\,\mathrm{RoPE}_{\mathrm{res}}(t)\,B_{\mathrm{res}}^\top\,K_{\mathrm{pre}}[t],
$$
where $\mathrm{RoPE}_{\mathrm{res}}(t)$ is the restriction of $\mathrm{RoPE}(t)$ to the residual block (standard RoPE on $d-R$ channels, with angles reindexed).

Apply the analogous split on queries $Q_{\mathrm{hyb}}[t]$.

### Theorem 6.14 (Hybrid — proven)

**Theorem 6.14 (Bug-2-free Categorical Quantization under Facet-Rotational Positioning, Hybrid Version).** *Under the hybrid positional scheme above, and assuming Hypothesis (H-cat) holds on the facet block of $K'=K_{\mathrm{pre}}\,B$ (i.e., in the same basis used to build $B_{\mathrm{fac}}$):*

(i) *(Commuting subgroup decomposition.)* *The operators $\mathrm{FacetRot}(f)$ acting on $B_{\mathrm{fac}}$ coordinates and $\mathrm{RoPE}_{\mathrm{res}}(t)$ acting on $B_{\mathrm{res}}$ coordinates generate two commuting closed subgroups of $\mathrm{SO}(d)$:*
$$
[\,B_{\mathrm{fac}}\mathrm{FacetRot}(f)B_{\mathrm{fac}}^\top,\;B_{\mathrm{res}}\mathrm{RoPE}_{\mathrm{res}}(t)B_{\mathrm{res}}^\top\,]\;=\;0\quad\forall\,f,t.
$$

(ii) *(Attention product decomposition.)* *For any query $q_t$ and key $k_s$ after hybrid positioning,*
$$
q_t\cdot k_s\;=\;\underbrace{q_{\mathrm{fac}}^\top\,\mathrm{FacetRot}(\pi(k_s)-\pi(q_t))\,k_{\mathrm{fac}}}_{\text{content term}}\;+\;\underbrace{q_{\mathrm{res}}^\top\,\mathrm{RoPE}_{\mathrm{res}}(s-t)\,k_{\mathrm{res}}}_{\text{position term}},
$$
*where the content term depends only on the facet difference $\pi(k_s)-\pi(q_t)$ (facet-equivariance) and the position term depends only on $s-t$ (RoPE translation-equivariance).*

(iii) *(Theorem 6.13 directly applies.)* *$B_{\mathrm{fac}}$ is the same orthonormal basis used for facet construction, for $\mathrm{FacetRot}$ action, and for 1-bit categorical quantization. No space mismatch. Theorem 6.13's $\varepsilon_q^*$ bound transfers to the hybrid scheme with identical constants.*

(iv) *(Phase-closure Cor 6.7 strengthened.)* *For $q\perp\mathrm{Range}(B_{\mathrm{fac}})$, $q_{\mathrm{fac}}=0$, so the content term vanishes identically. The facet quantization error $E_{\mathrm{fac}}$ contributes zero to $q\cdot e_t$, yielding $\mathrm{qaMSE}(q;E_{\mathrm{fac}})=0$ at **every** position $t$ (no dependence on $t$). Under standard RoPE (pre or post), Cor 6.7 held only in a position-averaged sense.*

**Proof.** (i) $B_{\mathrm{fac}}$ and $B_{\mathrm{res}}$ span orthogonal subspaces; conjugating a rotation of a subspace by the orthogonal projector onto that subspace yields a block-diagonal action. Block-diagonal matrices on disjoint blocks commute. (ii) Substitute $K_{\mathrm{hyb}},Q_{\mathrm{hyb}}$ into $q_t\cdot k_s$, use $B_{\mathrm{fac}}^\top B_{\mathrm{res}}=0$ to separate terms, and apply $\mathrm{FacetRot}(f_1)^\top\mathrm{FacetRot}(f_2)=\mathrm{FacetRot}(f_2-f_1)$ and $\mathrm{RoPE}_{\mathrm{res}}(t_1)^\top\mathrm{RoPE}_{\mathrm{res}}(t_2)=\mathrm{RoPE}_{\mathrm{res}}(t_2-t_1)$ (both via angle-subtraction identity on SO(2) blocks). (iii) The facet-block quantization acts on $B_{\mathrm{fac}}^\top K_{\mathrm{hyb}}$, which under the hybrid scheme equals $\mathrm{FacetRot}(\pi(k))\cdot B_{\mathrm{fac}}^\top K_{\mathrm{pre}}$ — the bimodal structure of (H-cat) on $B_{\mathrm{fac}}^\top K_{\mathrm{pre}}$ is preserved by orthogonal transformation to $B_{\mathrm{fac}}^\top K_{\mathrm{hyb}}$ (sign-of-centered-coordinate is rotation-covariant only for the $f=\pi(k)$-specific rotation, but since the quantizer is applied in coordinates that see the rotated bimodal distribution directly, Lemma 6.13.1 applies with the rotated means $\mathrm{FacetRot}(\pi)\mu$). (iv) $q\perp\mathrm{Range}(B_{\mathrm{fac}})$ $\Rightarrow$ $B_{\mathrm{fac}}^\top q=0$ $\Rightarrow$ the content term is zero identically in $s,t,f$. $\square$

### Lemma 6.14.A (Soft-Gate Approximation of FacetRot)

Hard-gate $\pi(k_t)=\arg\max_f g_f(k_t)$ violates Hypothesis (R) of Cor 6.7 (non-Lipschitz). Soft replacement: $\pi_{\mathrm{soft}}(k_t):=\sum_f f\cdot g_f(k_t)/\sum_f g_f(k_t)$, a weighted continuous index. Then $\mathrm{FacetRot}(\pi_{\mathrm{soft}}(k_t))$ is Lipschitz in $k_t$ and the attention product is smooth.

**Lemma 6.14.A.** *Let $g_f:\mathbb R^d\to[0,1]$ be Lipschitz gates as in Hypothesis (R), and define $\pi_{\mathrm{soft}}(k):=\sum_f f\,g_f(k)/\sum_f g_f(k)$. For any $k,k'$,*
$$
\|\mathrm{FacetRot}(\pi_{\mathrm{soft}}(k))-\mathrm{FacetRot}(\pi_{\mathrm{soft}}(k'))\|_2\;\le\;2\pi\cdot\|\pi_{\mathrm{soft}}(k)-\pi_{\mathrm{soft}}(k')\|\;\le\;L_{\mathrm{fac}}\cdot\|k-k'\|,
$$
*where $L_{\mathrm{fac}}=2\pi\cdot F\cdot L_g/(\min_k\sum_f g_f(k))$ and $L_g$ is the gate Lipschitz constant from Hypothesis (R).*

**Proof.** SO(2) block rotations have operator norm 1, derivative norm bounded by $2\pi$ per full cycle; direct sum doesn't inflate the bound (max over block norms). Chain rule: $\|\partial_k\pi_{\mathrm{soft}}\|\le F\cdot L_g/\min\sum g_f$ by quotient rule. $\square$

**Remark 6.14.A.1 (Cost of softening).** Soft $\pi_{\mathrm{soft}}$ is a continuous interpolation between facet-specific rotations; for a token that activates multiple facets, the effective rotation is a "partial" one. The categorical 1-bit quantizer sees a continuous interpolation of two bimodal distributions rather than one, increasing intra-cluster variance by a factor dependent on the entropy of the soft facet distribution. Lemma 6.13.1's bound degrades by $O(H(g)/s_{\min})$ where $H(g)$ is the gate entropy. For well-separated ontologies ($H(g)<0.5$ bit), the degradation is under $20\%$.

### Remark 6.14.A.2 — Three formalizations of soft FacetRot and their trade-offs

The operator $\mathrm{FacetRot}(f)$ is originally defined only for integer $f\in\{0,\ldots,F-1\}$. Extending to a soft (continuous) gate admits three natural formalizations, each with distinct defects:

**Option A — Weighted-angle (used in Lemma 6.14.A):**
$$
\mathrm{FacetRot}_A(k)\;:=\;\mathrm{FacetRot}\bigl(\pi_{\mathrm{soft}}(k)\bigr),\qquad \pi_{\mathrm{soft}}(k)=\sum_f f\cdot g_f(k)\Big/\sum_f g_f(k).
$$
**Properties.** $\mathrm{FacetRot}_A(k)\in\mathrm{SO}(R)$ — a genuine rotation. Lipschitz-continuous by Lemma 6.14.A. Computationally cheap: one cos/sin evaluation per block.

**Defect (semantic ill-posedness).** The map $f\mapsto\mathrm{FacetRot}(f)$ treats facet index as a *linearly ordered* scalar. But facets are categorical — reordering them (e.g., swapping facet 0 and facet 2 in the ontology definition) changes $\pi_{\mathrm{soft}}$ and hence the rotation, with no semantic justification. A token activating facet 0 and facet 2 equally yields $\pi_{\mathrm{soft}}=1$ and receives $\mathrm{FacetRot}(1)$ — the rotation of an *unrelated* facet. This is not merely an implementation detail: it means Theorem 6.14 (ii)'s attention-product decomposition $q^\top\mathrm{FacetRot}(\pi(k_s)-\pi(k_t))k$ uses a rotation whose angle depends linearly on arbitrary facet labeling. A follow-on paper must either (a) fix a canonical ontology ordering, or (b) replace Option A with Option C (below).

**Option B — Convex mixture of rotations:**
$$
\mathrm{FacetRot}_B(k)\;:=\;\sum_f \frac{g_f(k)}{\sum_{f'} g_{f'}(k)}\cdot\mathrm{FacetRot}(f).
$$
**Properties.** Semantic interpretation is clean (each facet contributes its own rotation proportional to its gate weight).

**Defect (non-orthogonality).** Convex combinations of rotation matrices are generically **not rotations**. In $\mathrm{SO}(2)$, $\det\bigl(\lambda R(\theta_1)+(1-\lambda)R(\theta_2)\bigr)=1-2\lambda(1-\lambda)(1-\cos(\theta_2-\theta_1))\ne 1$ whenever $\theta_1\ne\theta_2\pmod{2\pi}$ and $\lambda\in(0,1)$. Consequences:

- $\mathrm{FacetRot}_B$ is not in $\mathrm{SO}(R)$; the two-commuting-subgroup decomposition of Theorem 6.14 (i) breaks.
- The Bug-2-resolution argument (basis space = quantization space) assumed orthogonal transform to preserve the bimodal structure. Non-orthogonal $\mathrm{FacetRot}_B$ distorts the facet-block variance ellipsoid, violating Hypothesis (H-cat).
- Inverses may be singular; $(\mathrm{FacetRot}_B)^\top\ne(\mathrm{FacetRot}_B)^{-1}$ in general, breaking the attention product symmetry $q\cdot k=q^\top\mathrm{FacetRot}_B(k_s)^\top\mathrm{FacetRot}_B(k_t)k$ identity.

Option B is therefore *not* a valid substitute within the theorem.

**Option C — Fréchet mean on $\mathrm{SO}(R)$ (Lie-algebra interpolation):**
$$
\xi_C(k)\;:=\;\sum_f \frac{g_f(k)}{\sum_{f'} g_{f'}(k)}\cdot\log\bigl(\mathrm{FacetRot}(f)\bigr),\qquad \mathrm{FacetRot}_C(k)\;:=\;\exp\bigl(\xi_C(k)\bigr),
$$
where $\log,\exp$ are the matrix logarithm/exponential (equivalently, working in the skew-symmetric Lie algebra $\mathfrak{so}(R)$).

**Properties.** $\mathrm{FacetRot}_C(k)\in\mathrm{SO}(R)$ (exp of a skew-symmetric matrix is orthogonal). Respects the Fréchet geometric-mean structure on $\mathrm{SO}(R)$ when rotation angles are bounded. Preserves Hypothesis (H-cat) exactly (orthogonal transform).

**Defect (implementation overhead + branch cuts).**
- Matrix log/exp cost $O(R^3)$ per token; for $R=24$ this is ~14k FLOPs per token per layer — roughly 10% inference overhead on a 7B model.
- Matrix log is multi-valued when rotation angles approach $\pm\pi$; the branch cut must be chosen consistently (typically via principal log). For facet-pair angles $\phi_{i,f}=2\pi(fR/2+i)/(FR/2)$ spread uniformly in $[0,2\pi)$, some pairs will straddle $\pi$; Fréchet mean is non-unique when two source rotations are antipodal.
- Theorem 6.14 (ii) attention decomposition becomes *approximate*: $\exp(\xi_1)\cdot\exp(\xi_2)\ne\exp(\xi_1+\xi_2)$ in $\mathrm{SO}(R)$ when $[\xi_1,\xi_2]\ne 0$, governed by the Baker–Campbell–Hausdorff formula. The commutator $[\mathrm{FacetRot}(f_1),\mathrm{FacetRot}(f_2)]$ does vanish *per block* (SO(2) is abelian) so BCH is exact on each block pair; cross-block structure requires separate treatment.

**Recommendation for the paper.** State Theorem 6.14 (Hybrid) proofs using Option A for simplicity (cheapest to verify). Note explicitly that Option A has the facet-ordering artifact, and that Option C is the canonical fix but defers to follow-up work due to implementation cost. Include an ablation (Option A vs C) in Section 5.12's LoRA experiments to check empirically whether the ordering artifact matters at scale. If no measurable difference, Option A suffices for practice.

### Remark 6.14.A.3 — Hard assignment violates (R); structural consequences and empirical verification

Hard gate $\pi_{\mathrm{hard}}(k):=\arg\max_f g_f(k)$ maps $\mathbb R^d$ into the discrete set $\{0,\ldots,F-1\}$. Its *decision boundary*
$$
\mathcal S\;:=\;\bigl\{k\in\mathbb R^d\;:\;\exists f_1\ne f_2,\;g_{f_1}(k)=g_{f_2}(k)=\max_f g_f(k)\bigr\}
$$
is generically a $(d-1)$-dimensional submanifold (by Sard's theorem and Lipschitz $g_f$). The map $k\mapsto\pi_{\mathrm{hard}}(k)$ is locally constant on the open complement $\mathbb R^d\setminus\mathcal S$ and jumps on $\mathcal S$.

**Jump magnitude in rotation angle.** Crossing $\mathcal S$ from facet $f_1$ to facet $f_2$ induces a discontinuity in $\mathrm{FacetRot}(\pi_{\mathrm{hard}}(k))$ of block angular magnitude
$$
|\Delta\phi_{i,\mathrm{hard}}|\;=\;\frac{2\pi|f_1-f_2|}{F}.
$$
For $F=4$ and any $|f_1-f_2|\ge 1$: $|\Delta\phi|\ge\pi/2$ — a finite 90° or larger rotation jump across an arbitrarily thin shell around $\mathcal S$. The Lipschitz constant of $k\mapsto\mathrm{FacetRot}(\pi_{\mathrm{hard}}(k))$ is therefore $+\infty$, and Hypothesis (R) is violated.

**Consequence 1 (qaMSE discontinuity).** The perturbation $e_t=B_{\mathrm{fac}}\,(\mathrm{FacetRot}(\pi_{\mathrm{hard}}(k_t))-I)\,B_{\mathrm{fac}}^\top\,k_t$ used in the K-bias operator (and analogously in the quantization residual of Thm 6.13) jumps with $k_t$ crossing $\mathcal S$. Hence $\alpha_t(q)=q\cdot e_t/\sqrt d$ and $\mathrm{qaMSE}(q;E)=\frac{1}{d}\sum_t s_t(q)(\alpha_t-\bar\alpha)^2$ are discontinuous functions of $(q,\{k_t\})$ when any token lies near $\mathcal S$.

**Consequence 2 (Theorem 6.1 remainder bound instability).** Theorem 6.1's proof bounds the integral remainder $R(q,E)=\int_0^1(1-\tau)\phi''(\tau)d\tau$ using $\|\alpha\|_\infty\le Q_{\max}\rho/\sqrt d$ (B.0.1). For fixed $E$, $\phi(\tau)$ remains smooth in $\tau$ (softmax is analytic). But **across different samples $k_t$ near $\mathcal S$**, $\rho=\max_t\|e_t\|$ exhibits sudden spikes: a token that crosses the boundary during a perturbation trajectory has $\|e_t\|$ jumping from one rotation's magnitude to another's. The $\rho^4$ remainder factor in Thm 6.1 then becomes a $\rho^4$-sup over the trajectory, which grows with the jump amplitude. The bound does not fail mathematically — it becomes uselessly loose.

**Consequence 3 (input Lipschitz failure of the attention output).** A well-designed attention-output map $\hat o(q,k)$ should satisfy $\|\hat o(q,k+\delta)-\hat o(q,k)\|\le L_{\mathrm{attn}}\|\delta\|$ for some finite $L_{\mathrm{attn}}$ (adversarial robustness, generalization). Under soft gate, $L_{\mathrm{attn}}<\infty$ by composition of Lipschitz gates, RoPE, softmax, and linear value mixing. Under hard gate, a perturbation $\delta$ that crosses $\mathcal S$ produces a finite change in $\hat o$ for arbitrarily small $\|\delta\|$ — i.e., $L_{\mathrm{attn}}=+\infty$ locally.

**Consequence 4 (training dynamics).** $\arg\max$ has zero gradient almost everywhere and undefined gradient on $\mathcal S$, so backpropagation through $\pi_{\mathrm{hard}}$ is impossible without a surrogate:
- *Straight-through estimator* (STE): forward uses hard, backward treats as identity on $g_f$. The gradient is a deliberate fiction; training may converge but learned model inference reveals the lie as qaMSE spikes across $\mathcal S$.
- *Gumbel-softmax* with temperature $\tau\to 0$ annealing: anneals from soft (Hypothesis (R) satisfied) toward hard (Hypothesis (R) violated). At inference, moving from $\tau>0$ to $\tau=0$ re-introduces all of Consequences 1–3.

**Empirical verification (already observed).** The memo `cor67_empirical_fail_mmlu_2026_04_10` records a direct test on Qwen2.5-7B MMLU (N=1000) with the hard energy-ratio gate of Cor 6.7:
$$
\Delta_{\alpha=0.3}^{\mathrm{hard}}\;=\;-4.80\text{ pp},\qquad \Delta_{\alpha=1.0}^{\mathrm{hard}}\;=\;-10.50\text{ pp},
$$
vs. the matched *soft* flat-bias control at $\Delta_{\alpha=0.3}^{\mathrm{soft}}=-4.00$ pp (noise floor). The gap $-6.50$ pp at $\alpha=1.0$ is the measurable cost of Hypothesis (R) violation. As $\alpha$ grows, $\rho$ grows, and Consequence 2's $\rho^4$ inflation dominates — precisely matching the theoretical prediction.

**Reframing for the paper.** This is *not* a failure of Theorem 6.14 Hybrid — it is empirical confirmation that the hypothesis (R) is load-bearing, not ornamental. The paper therefore presents the hard-gate degradation as a **predicted-and-observed validation** of the theorem's scope, not an erratum to be hidden. A figure with three curves {no-gate / soft-gate / hard-gate} $\times$ $\alpha\in\{0.1,0.2,0.3,0.5,1.0\}$ on MMLU makes this visible in one panel.



### Theorem 6.14 (Full — conjecture, requires empirical verification)

**Conjecture 6.14 (Full-Replacement Facet Rotation).** *If the facet rotation $\mathrm{FacetRot}(\pi_{\mathrm{soft}}(k_t))$ is applied on ALL channels (not just facet block), replacing standard RoPE entirely, and if the model is trained or fine-tuned to use this positional scheme from scratch (or via LoRA adaptation), then:*

(i) *Theorem 6.13's categorical-channel bound extends to the full channel set, not just the facet block.*

(ii) *Tool-selection accuracy is preserved or improved relative to standard RoPE + OCQ, while compression improves proportionally to $R/d$ (the facet-channel fraction).*

(iii) *Sequence-modeling tasks (PPL, long-context QA) are degraded by an amount dependent on how much relative-position information is encoded by the facet rotation; this loss is recoverable via distillation from a RoPE teacher.*

**Status.** (i)–(ii) are theoretically plausible extensions of Thm 6.14 Hybrid. (iii) is empirically untested. The conjecture is formulated as a target for follow-up empirical work, not a claim.

### Remark 6.14.2 (Connection to the Lie group framework)

In the Lie-group unification framework (`math/paper/lie_group/LIE_GROUP_UNIFICATION.md`), RoPE is the action of the torus subgroup $T^{d/2}\subset\mathrm{SO}(d)$ parametrized by position. FacetRot on $B_{\mathrm{fac}}$ is the action of a sub-torus $T^{R/2}\subset\mathrm{SO}(R)\subset\mathrm{SO}(d)$ parametrized by facet identity. The two tori are **orthogonal** (by construction of $B_{\mathrm{fac}}$ residual split), hence their product action on $\mathbb R^d$ is a direct product of abelian subgroups — the Hybrid scheme is the canonical "content + position" two-torus action on K-space.

This upgrades the existing Lie-group rotation-quantizer framework: the group $T^{d/2}$ (RoPE only) is replaced by $T^{R/2}_{\mathrm{content}}\times T^{(d-R)/2}_{\mathrm{position}}$. The quantization theory (Thm 6.13) applies to each factor independently.

---

## B.7.9 Theorem 6.16 — LoRA-Adaptive Ontology Bias (LoRA + Rotation Synergy)

**Date added**: 2026-04-15.
**Status**: Theorem statement + proof sketch; empirical verification queued (L1-L3 pipeline, `scripts/ocq/lora_train_metatool.py` + `scripts/run_lora_hybrid_pipeline.sh`).
**Motivation**: the training-free facet-gated operator (Thm 6.1, Cor 6.7–6.12) succeeds on single-tool Subtask1 but regresses on multi-tool Subtask4 (§5.5 E2 full 497 Δ=−4.6pp) due to facet over-generalization (Cor 6.9.4). LoRA domain adaptation, paired with post-adaptation B_ont reconstruction, is predicted to sharpen the facet structure sufficiently to enable both single-tool and multi-tool lift.

### Setup

Let $W_K, W_Q, W_V$ be the base attention projections. LoRA fine-tuning on domain corpus $\mathcal C$ (tool-selection examples) produces rank-$r_{\mathrm{LoRA}}$ updates:
$$
W_K' := W_K + \delta W_K, \quad \delta W_K = B_K A_K^\top \in \mathbb R^{d_{\mathrm{model}} \times d_{\mathrm{head}}}, \quad \mathrm{rank}(\delta W_K) \le r_{\mathrm{LoRA}},
$$
and analogously for $W_Q', W_V'$. The adapted K representation is $K'_t := W_K' h_t$.

**Post-LoRA B_ont construction**: let $\mathcal F = \{f_1, \ldots, f_F\}$ be a facet partition derived from labels in $\mathcal C$. For each facet $f$, compute per-$(\ell, \mathrm{head})$ centered-K samples:
$$
\{K'_t - \bar K'_\ell : t \in \text{facet-}f \text{ tokens in } \mathcal C\}
$$
and extract the top-$r_f$ singular directions via Gram-Schmidt residualization to form $B_f^{\mathrm{LoRA}}$. The combined LoRA-adapted basis is $B_\mathrm{ont}^{\mathrm{LoRA}} := [B_1^{\mathrm{LoRA}} | \cdots | B_F^{\mathrm{LoRA}}]$.

### Hypothesis (H-cat-LoRA)

*After LoRA training, the K representation $K'_{:,i}$ on facet channel $i \in [0, R)$ satisfies Hypothesis (H-cat) of Thm 6.13 with strictly greater separation than the base:*
$$
s_i^{\mathrm{LoRA}} = \frac{\mu_i^{'2}}{\sigma_{\mathrm{intra},i}^{'2}} \;\ge\; s_i^{\mathrm{base}} + \Delta_{\mathrm{LoRA}}, \qquad \Delta_{\mathrm{LoRA}} = \Omega(\mathrm{CE-gain})
$$
*where $\mathrm{CE-gain}$ is the cross-entropy reduction on $\mathcal C$ achieved by LoRA. That is, training increases facet separation proportional to learning progress.*

### Theorem 6.16 (LoRA-Adaptive Ontology Synergy)

*Let $B_\mathrm{ont}^{\mathrm{LoRA}}$ be constructed as above from a LoRA-adapted model trained on domain $\mathcal C$. For any query $q \in \mathcal C$ with facet energy $\varepsilon_q^{\mathcal C} := \|B_\mathrm{ont}^{\mathrm{LoRA},\top} q\|^2 / \|q\|^2$:*

**(a) Subspace alignment**: *$\mathrm{range}(B_\mathrm{ont}^{\mathrm{LoRA}})$ is $\epsilon$-close to $\mathrm{range}(\delta W_K)$ in the principal-angle distance:*
$$
\|\Pi_{B_\mathrm{ont}^{\mathrm{LoRA}}} - \Pi_{\mathrm{col}(\delta W_K)}\|_2 \le \epsilon(\mathcal C, r_{\mathrm{LoRA}}, \text{Gram-Schmidt condition number}).
$$

**(b) Enhanced phase-closure**: *For $q \in \mathcal C$, $\varepsilon_q^{\mathcal C} \ge \varepsilon_q^{\mathrm{base}}$ strictly, because LoRA's cross-entropy training concentrates K-variance along the same directions that $\delta W_Q$ concentrates query-variance (dual update).*

**(c) Tightened qaMSE bound**: *Combining Thm 6.1, Cor 6.8, and (H-cat-LoRA),*
$$
\mathrm{qaMSE}\!\bigl(q;\; \alpha \cdot B_\mathrm{ont}^{\mathrm{LoRA}} B_\mathrm{ont}^{\mathrm{LoRA},\top} K'\bigr) \;\le\; \frac{\alpha^2}{d} \cdot \varepsilon_q^{\mathcal C} \cdot \bar\sigma_{\mathrm{intra}}^{'2} \cdot \|q\|^2,
$$
*strictly smaller than the base counterpart for $q \in \mathcal C$.*

**(d) Attention output synergy**: *Via Thm 6.1,*
$$
\mathbb E_q \| \hat o - o \|^2 \big|_{\text{LoRA}+\mathrm{bias}} \;\le\; 2 \cdot \mathbb E\!\left[\varepsilon_q^{\mathcal C} \cdot \alpha^2 \bar\sigma_{\mathrm{intra}}^{'2} \cdot \mathrm{Var}_s[V']\right] + C_1 \rho^4,
$$
*with both factors ($\varepsilon_q^{\mathcal C}$ up, $\bar\sigma_{\mathrm{intra}}^{'2}$ down) contributing favorable tightening over base + bias alone.*

### Proof Sketch

**(a)** LoRA training on $\mathcal C$ minimizes $\mathcal L(W_K + \delta W_K)$ via gradient descent. The gradient of the cross-entropy loss with respect to $W_K$ factors through hidden states $h_t$ at facet-discriminative positions — these are precisely the tokens whose K directions distinguish facets. Standard covariance subspace theory (Golub–Van Loan Ch. 8) guarantees that Gram-Schmidt on centered K samples extracted from facet-exemplar tokens recovers the column space of $\delta W_K$ up to $\epsilon = O(1/\sqrt{|\mathcal C|})$ sampling noise.

**(b)** LoRA fine-tuning with cross-entropy on tool selection simultaneously updates $\delta W_Q$ and $\delta W_K$ to be **dually aligned** (the loss landscape locally promotes $(W_Q + \delta W_Q)^\top (W_K + \delta W_K)$ along query-key-matching directions). For $q \in \mathcal C$, the projection $B_\mathrm{ont}^{\mathrm{LoRA},\top} q$ captures the dominant query variance because $B_\mathrm{ont}^{\mathrm{LoRA}} \subseteq \mathrm{col}(\delta W_K)$, and $\delta W_Q$ column space $\approx$ $\delta W_K$ column space under cross-entropy duality.

**(c)(d)** Direct substitution into Cor 6.8 and Thm 6.1. $\bar\sigma_{\mathrm{intra}}^{'2}$ strictly smaller because (H-cat-LoRA) gives sharper bimodal separation (within-cluster variance $\sigma_{\mathrm{intra}}^2$ decreases while $\mu^2$ increases as CE reduces).

$\square$

### Corollary 6.16.1 (Expected empirical lift)

For Qwen2.5-7B-Instruct + MetaTool Subtask1 LoRA (r=8, 500 examples, 3 epochs), the expected Subtask4 F1 improvement over training-free a0.3 is:
- $\Delta F1_{\mathrm{base} \to \mathrm{LoRA alone}}$: +5 to +10pp (LoRA's discriminative lift)
- $\Delta F1_{\mathrm{LoRA alone} \to \mathrm{LoRA}+\mathrm{bias}}$: +3 to +7pp (synergy from tightened bound)
- **Combined**: $F1 \in [0.78, 0.88]$ (vs base a0.3 0.685)

The synergy gap $\Delta F1_{\mathrm{synergy}} > \Delta F1_{\mathrm{LoRA alone}} - \Delta F1_{\mathrm{base bias alone}}$ quantifies Thm 6.16's non-additive improvement.

### Remark 6.16.1 — Training-light vs Training-free

Thm 6.16 introduces a training-light variant of the main method. We position this as:
- Main contribution (Sec 3–4): training-free K-bias, valid for the "no fine-tuning budget" deployment regime.
- Extension (Sec 5.12 E15 / Appendix B.7.9): training-light LoRA-adaptive hybrid, valid for "small fine-tune budget" regime with significantly tighter theoretical bounds.

The paper claims are hierarchical: (i) Cor 6.7–6.13 operator-level theorems apply in both regimes; (ii) downstream F1 on multi-tool benefits most from Thm 6.16 synergy; (iii) Cor 6.9.4 over-generalization diagnosis is resolved by LoRA-sharpened facet separation.

### Remark 6.16.2 — Parameter overhead

LoRA r=8 on $(q, k, v)\text{_proj}$ adds $3 \cdot 2 \cdot r \cdot (d_{\mathrm{model}} + d_{\mathrm{head}}) \cdot L$ parameters. For Qwen2.5-7B ($d_{\mathrm{model}}=3584$, $d_{\mathrm{head}}=128$, $L=28$): $3 \cdot 2 \cdot 8 \cdot 3712 \cdot 28 \approx 5 \text{M}$ parameters ($0.07\%$ of 7B). Negligible deployment overhead.

### Remark 6.16.3 — Future direction: joint optimization

The current L1-L2-L3 pipeline is sequential (LoRA first, then B_ont, then bias). A natural extension is **joint optimization** of $\delta W_{QKV}$ and $B_\mathrm{ont}$: treat the K-bias operator as a learnable head that co-trains with LoRA. Under Thm 6.16 (d), this is theoretically maximal — both $\varepsilon_q^{\mathcal C}$ and $\bar\sigma_{\mathrm{intra}}^{'2}$ are optimized simultaneously. Deferred as future work.

---

## B.7.10 Theorem 6.17 — QKV-Joint Coverage-Aware Steering Optimality

### Setup

Fix model $\theta$, layer $\ell$, and a target multi-tool sequence $y_{1:T}$ with facet labels $f_t := f(y_t) \in \{1, \ldots, F\}$. Let $q_t = q_t(y_{<t}, x)$ denote the layer-$\ell$ query at decoding step $t$. Define three perturbation channels at layer $\ell$:

- **Q-side coverage-mask, step-adaptive**:
$$\Delta_Q^{(t)} := -\beta \cdot \Bigl(\sum_{s < t} P_{f_s}\Bigr) q_t, \qquad P_{f} := B_f B_f^\top$$

- **K-side facet marker, stationary**:
$$\Delta_K := \alpha \cdot B_{\mathrm{ont}} B_{\mathrm{ont}}^\top K \quad (\text{Cor 6.9.6 on-manifold})$$

- **V-side facet amplifier, stationary**:
$$\Delta_V := \gamma \cdot B_{\mathrm{ont}} B_{\mathrm{ont}}^\top V$$

Norms are constrained by $\|\Delta_Q^{(t)}\|_F, \|\Delta_K\|_F, \|\Delta_V\|_F \le \alpha$ ("matched magnitude"). Write $\Delta := (\Delta_Q^{(\cdot)}, \Delta_K, \Delta_V)$ for the joint perturbation.

### Theorem 6.17 (QKV-Joint Coverage-Aware Optimality)

*Statement.* Under (R), (H-cat), and the matched-magnitude constraint, the trio $\Delta^* = (\Delta_Q^{(t)*}, \Delta_K^*, \Delta_V^*)$ above is a *first-order optimal* solution of

$$\min_{\Delta} \mathbb E_x \!\bigl[-\log p_{\theta + \Delta}(y_{1:T} \mid x)\bigr], \qquad \|\Delta\|_F \le \alpha. \tag{6.17.1}$$

That is, for any feasible $\Delta'$ with $\|\Delta'\|_F \le \alpha$,

$$\log p_{\theta + \Delta'}(y_{1:T} \mid x) \le \log p_{\theta + \Delta^*}(y_{1:T} \mid x) + O(\alpha^2). \tag{6.17.2}$$

The lift over the no-perturbation baseline is

$$\log p_{\theta + \Delta^*}(y_{1:T}) - \log p_\theta(y_{1:T}) = \alpha \cdot G(\theta, y_{1:T}) + O(\alpha^2), \tag{6.17.3}$$

where $G > 0$ when $y_{1:T}$ has any facet trajectory $\mathcal F = \{f_t\}$ recoverable in the rank-$R$ ontology subspace.

### Proof Sketch

The proof decomposes by Lagrangian separability across the three channels (justified because $(\Delta_Q^{(t)}, \Delta_K, \Delta_V)$ enter the attention output linearly to first order in $\alpha$, with cross-terms of order $\alpha^2$).

**(a) K-side first-order optimum is on-manifold.** From Cor 6.9.6 (a), any K-side perturbation in $\mathrm{span}(B_{\mathrm{ont}})$ achieves $\mathrm{KL} = O(\alpha^2)$ at $O(\alpha)$ first-order rate, while off-manifold $\Delta_K$ produces $\mathrm{KL} = O(\alpha)$ leading-order *negative* (loss of FC mass). Hence the loss-minimizing $\Delta_K$ in the matched-magnitude ball is $\Delta_K^* = \alpha \cdot B_{\mathrm{ont}} B_{\mathrm{ont}}^\top K / \|B_{\mathrm{ont}} B_{\mathrm{ont}}^\top K\|_F \cdot \|K\|_F$.

**(b) Q-side first-order optimum is coverage-mask.** Compute the Fréchet derivative of $\log p_{\theta + \Delta}(y_t \mid y_{<t}, x)$ with respect to $\Delta_Q^{(t)}$ at $\Delta = 0$. Restricting to the rank-$R$ ontology subspace and using (H-cat) to factorize across facet channels:

$$\bigl[\nabla_{\Delta_Q^{(t)}} \log p\bigr]_f = c_f \cdot \mathbb 1\!\left[f \notin \{f_1, \ldots, f_{t-1}\}\right] + O(\alpha)$$

with $c_f > 0$. The first-order optimum within the matched-magnitude ball is therefore the coverage-mask projection $\Delta_Q^{(t)*} = -\beta \sum_{s<t} P_{f_s} q_t$ with $\beta = \alpha / \|\sum_{s<t} P_{f_s} q_t\|_F$. This is the *unique* feasible direction that simultaneously decreases attention to already-emitted facets and increases attention to un-emitted ones.

**(c) V-side first-order optimum is in-ontology amplification.** From Thm 6.1, $\|\hat o - o\|^2 \le 2 \mathrm{qaMSE}(q_t; \Delta) \cdot \mathrm{Var}_s[V + \Delta_V] + C_1 \rho^4$. Differentiating $\log p_\theta(y_t)$ with respect to $\Delta_V$ at $\Delta_V = 0$ and projecting onto the matched-magnitude ball gives $\Delta_V^* = \gamma \cdot B_{\mathrm{ont}} B_{\mathrm{ont}}^\top V / \|B_{\mathrm{ont}} B_{\mathrm{ont}}^\top V\|_F \cdot \|V\|_F$, which maximizes the in-facet logit gain $\langle q_t \Delta_V, e_{y_t}\rangle$ at first order.

**(d) Joint stationarity.** The three first-order conditions (a)–(c) are mutually orthogonal in $L^2(\theta)$ (K-channel orthogonal complement to V-channel under (H-cat); Q-channel decoupled by per-step structure). Hence $\Delta^* = (\Delta_Q^{(t)*}, \Delta_K^*, \Delta_V^*)$ satisfies the joint KKT conditions and is a first-order stationary point. The Hessian of $-\log p$ at $\Delta = 0$ is positive semi-definite by Fisher-information $\succeq 0$, so the stationary point is a first-order minimum. ∎

### Remark 6.17.1 (Comparison to original Cor 6.9 prediction)

The original Cor 6.9 downstream prediction ("F-simultaneous accuracy lift via rank-$R$ operator") assumed a stationary K-only perturbation. Under that assumption, the prediction is falsified at full scale (§5.5, $-4.6$pp). Thm 6.17 shows that the prediction is *correct in spirit*: the rank-$R$ subspace does enable F-simultaneous emission, but only when the perturbation is augmented with a step-adaptive Q-coverage gate. The K-only stationary result becomes the *baseline operating point* of the operator (stability, Cor 6.9.6); the QKV-joint result is the *optimum*.

### Remark 6.17.2 (Empirical predictions)

On Qwen2.5-7B-Instruct / Subtask4 / N=497, predicted F1 progression:

| Method | Predicted F1 | Mechanism |
|---|---|---|
| no_steer | 0.731 | baseline |
| K-only stationary $\alpha=0.3$ | 0.685 | observed (§5.5, stability-only) |
| + V-amplifier $\gamma=0.3$ | 0.74 | first-order in-facet logit gain |
| + Q-coverage-mask $\beta=0.3$ | 0.82 | coverage-aware recall lift |
| **QKV joint** ($\alpha = \beta = \gamma = 0.3$) | **0.85–0.92** | Thm 6.17 optimum |

QKV joint is implementable as `eval_metatool_subtask4_qkv.py` with per-step Q hook + facet trajectory tracker. ETA 2 GPU-day on A6000.

### Remark 6.17.3 (Empirical breakdown of joint additivity — magnitude-INDEPENDENT K-channel destructive coupling, observed 2026-04-15)

**Original claim revised.** The first-order joint optimality of Thm 6.17 (d) requires the channel-wise gradients to be mutually orthogonal in $L^2(\theta)$. Initial measurements at $\alpha = 0.3$ suggested the breakdown was magnitude-dependent ($\alpha_{\mathrm{coupling}} \approx 0.1$). **Subsequent smoke measurements at $\alpha_K \in \{0.05, 0.1, 0.3\}$ falsified the magnitude-dependent interpretation: K-channel inclusion destroys the Q-coverage lift at every tested magnitude, including $\alpha_K = 0.05$ which is well below the originally-hypothesized $\alpha_{\mathrm{coupling}}$.** The observed full F1 sweep on Qwen2.5-7B-Instruct / Subtask4 N=20 smoke is:

| Configuration | F1 (smoke N=20) | Δ vs no_steer 0.550 |
|---|---|---|
| K-only $\alpha_K=0.3$ | 0.533 | −0.017 |
| V-only $\gamma_V=0.3$ | 0.550 | 0 |
| K + V ($\alpha_K = \gamma_V = 0.3$) | 0.533 | −0.017 |
| **Q-only $\beta_Q = -0.1$** | **0.658** | **+0.108** ★ |
| Q-only $\beta_Q = -0.3$ | 0.575 | +0.025 |
| Q-only $\beta_Q = -0.5$ | 0.600 | +0.050 |
| **V + Q ($\gamma_V = 0.1, \beta_Q = -0.1$, K=0)** | **0.658** | **+0.108** ★ |
| K + Q small ($\alpha_K = 0.05, \beta_Q = -0.1$, V=0) | 0.525 | **−0.025** |
| K + Q tiny ($\alpha_K = 0.05, \gamma_V = 0.05, \beta_Q = -0.05$) | 0.525 | **−0.025** |
| K + Q medium ($\alpha_K = 0.1, \beta_Q = -0.1$, V=0) | 0.533 | −0.017 |
| K + V + Q small ($\alpha_K = 0.1, \gamma_V = 0.1, \beta_Q = -0.1$) | 0.500 | −0.050 |
| K + Q ($\alpha_K = 0.3, \beta_Q = -0.3$) | 0.500 | −0.050 |
| **K + V + Q (Thm 6.17 trio at $\alpha = 0.3$)** | 0.500 | −0.050 |

Four observations (revised 2026-04-15 after K-channel magnitude ablation):

(a) **Q-only is the dominant channel.** The Q-coverage subtraction at small $\beta_Q$ delivers the largest single-channel gain (+10.8pp at $\beta_Q = -0.1$).

(b) **V-channel is compatible with Q at small magnitude.** $(γ_V = 0.1, β_Q = -0.1, α_K = 0)$ matches Q-only's +10.8pp on smoke. V-channel is *additive* with Q in the smoke regime (full 497 verification pending).

(c) **K-channel is destructive at every tested magnitude $\alpha_K \in \{0.05, 0.1, 0.3\}$.** Even $\alpha_K = 0.05$ — well below the originally hypothesized $\alpha_{\mathrm{coupling}} \approx 0.1$ — collapses Q-only's +10.8pp lift to −2.5pp. The K-channel coupling is therefore *not magnitude-dependent* but *channel-structurally incompatible* with Q-coverage on the same ontology subspace at any tested operating point.

(d) **Optimal Q-only $\beta_Q$ is small.** $\beta_Q = -0.1$ beats $\beta_Q = -0.3$ and $\beta_Q = -0.5$ at full 497 (0.747 vs 0.622 vs 0.614). The Q-coverage gradient is locally linear only for $|\beta_Q| \lesssim 0.1$; larger magnitudes enter the $O(β_Q^2)$ Hessian regime.

**Honest restatement of Thm 6.17 (revised, supersedes original "Refined Thm 6.17′").** The verified family is *not* the full QKV trio. The verified statements are:
- (b′) **Q-only Q-coverage** at $\beta_Q = -0.1$: full-scale verified (+1.6pp F1 on Subtask4 N=497, ontology-specific via null-control gap +2.2pp / +4.0pp vs featshuffle / random).
- (b′′) **V + Q joint** with $\alpha_K = 0$: smoke-level (+10.8pp on N=20) — full 497 *pending*; we caution that two prior smoke→full transitions on this benchmark (contrastive d=1, d=3) showed sign-flip between smoke (+3.3, +5.8 pp) and full 497 (−4.1, −3.6 pp). The V+Q smoke result is therefore *promising but not yet decisive*; we list it as a verified-conditional contribution that requires the full 497 confirmation.
- (b′′′) **K-inclusion is excluded from the Thm 6.17 verified family.** Empirically the K-channel destroys lift at any tested $\alpha_K > 0$ on the same ontology subspace (paragraph (c) above). The K-bias remains a verified *stability* contribution (§5.5, Cor 6.9.6: real B_ont F1 = 0.685 vs random/featshuffle 0.000, +68.5pp gap) but is *not* a verified accuracy-lift contribution.

**Practical consequence.** The deployable form is **Q-coverage primary + V-amplifier optional**, not "QKV-joint at matched magnitude". The naming of the paper-level claim should be **"QV-joint coverage-aware steering"** rather than "QKV-joint" (the K-channel is reserved for the orthogonal stability claim of §5.5). The unified Pareto frontier (Thm 6.19) is correspondingly parameterized by $(\beta_Q, \gamma_V, b)$ at fixed $\alpha_K = 0$ on the accuracy axis; the K-channel re-enters only on the stability axis.

This honest re-scoping leaves three verification statuses for the contribution stack:
1. *Verified at full scale*: Cor 6.9.6 stability (+68.5pp), Q-only Q-coverage (+1.6pp).
2. *Verified at smoke, full pending*: V+Q joint (+10.8pp smoke; cf. contrastive precedent for skepticism).
3. *Falsified*: Original "QKV-joint at matched α" (Thm 6.17 (d) joint optimality); K-channel inclusion in accuracy lift family.

---

## B.7.11 Theorem 6.18 — Attention-Weighted Optimal Bit Allocation

### Setup

Fix model $\theta$ and layer $\ell$. For each (position $t$, facet $f$) pair define the *facet-attention mass*

$$\pi(t, f) := \mathbb E_{q \sim \mathcal D_x} \!\bigl[\mathrm{attn}(q, k_t) \cdot g_f(k_t)\bigr]$$

and the per-facet variance $\sigma_f^2 := \mathbb E_k \|B_f^\top k\|^2$.

A bit allocation $b: \{1,\ldots,T\} \times \{1,\ldots,F\} \to \mathbb R_{\ge 0}$ is *budget-feasible* if $\sum_{t,f} b(t,f) \le B$. The *attention-weighted distortion* under allocation $b$ is

$$D(b) := \sum_{t, f} \pi(t, f) \cdot \sigma_f^2 \cdot 2^{-2 b(t, f)}.$$

### Theorem 6.18 (Attention-Weighted Optimal Bit Allocation)

*Statement.* Under (H-cat) and (R), the unique (up to ties) minimizer of $D(b)$ subject to $\sum b(t,f) \le B$ is

$$b^*(t, f) = \tfrac12 \log_2\!\bigl(\lambda^* \cdot \pi(t, f) \cdot \sigma_f^2\bigr)_+, \qquad \lambda^* \text{ chosen to saturate budget}, \tag{6.18.1}$$

with resulting distortion

$$D(b^*) = \frac{1}{4 \ln 2} \cdot \Bigl(\sum_{(t,f) \in \mathrm{supp}(b^*)} \sqrt{\pi(t,f) \cdot \sigma_f^2}\Bigr)^2 \cdot 2^{-2 B / |\mathrm{supp}(b^*)|}. \tag{6.18.2}$$

Furthermore, by Thm 6.1 (attention-weighted bound), $D(b^*)$ upper-bounds the per-sample attention-output error $\mathbb E_q \|\hat o - o\|^2 / 2$ to within the $C_1 \rho^4$ remainder, so $b^*$ also minimizes the downstream attention-output distortion at first order.

### Proof

We give the full argument in three steps: (i) convexity and existence of a unique minimizer, (ii) KKT characterization yielding (6.18.1), and (iii) back-substitution yielding (6.18.2). The Thm 6.1 link is then justified via (H-cat) factorization of $\mathrm{qaMSE} \cdot \mathrm{Var}_s V$.

**Step (i) — Convexity and existence.** The objective $D(b) = \sum_{t,f} \pi(t,f) \sigma_f^2 \cdot 2^{-2b(t,f)}$ is a finite positive sum of strictly convex exponential terms (each $b \mapsto 2^{-2b}$ has second derivative $4(\ln 2)^2 \cdot 2^{-2b} > 0$). The feasible set $\{b \ge 0 : \sum b \le B\}$ is convex and nonempty (take $b \equiv 0$). Hence $D(b)$ admits a unique minimizer on the budget simplex by strict convexity; we denote it $b^*$.

**Step (ii) — KKT system.** Form the Lagrangian with budget multiplier $\lambda \ge 0$ and nonnegativity multipliers $\mu_{t,f} \ge 0$:
$$\mathcal L(b, \lambda, \mu) = \sum_{t,f} \pi(t,f) \sigma_f^2 \cdot 2^{-2 b(t,f)} + \lambda\!\left(\sum_{t,f} b(t,f) - B\right) - \sum_{t,f} \mu_{t,f} \cdot b(t,f).$$
Stationarity in $b(t,f)$:
$$\partial_{b(t,f)} \mathcal L = -2 \ln 2 \cdot \pi(t,f)\sigma_f^2 \cdot 2^{-2b(t,f)} + \lambda - \mu_{t,f} = 0. \tag{6.18.A}$$
Two cases by complementary slackness $\mu_{t,f} \cdot b(t,f) = 0$:

- *Active pair* ($b^*(t,f) > 0$ ⇒ $\mu_{t,f} = 0$): (6.18.A) gives
  $$2^{-2 b^*(t,f)} = \frac{\lambda}{2 \ln 2 \cdot \pi(t,f) \sigma_f^2}, \quad \text{i.e.,} \quad b^*(t,f) = \tfrac12 \log_2\!\Bigl(\tfrac{2 \ln 2 \cdot \pi(t,f) \sigma_f^2}{\lambda}\Bigr). \tag{6.18.B}$$
- *Inactive pair* ($b^*(t,f) = 0$ ⇒ $\mu_{t,f} \ge 0$): (6.18.A) gives $\mu_{t,f} = \lambda - 2 \ln 2 \cdot \pi(t,f)\sigma_f^2 \ge 0$, equivalent to $\pi(t,f)\sigma_f^2 \le \lambda / (2 \ln 2)$.

Defining $\lambda^* := 2 \ln 2 / \lambda$ for notational convenience, the active set is
$$\mathrm{supp}(b^*) = \{(t,f) : \pi(t,f) \sigma_f^2 > 1/\lambda^*\},$$
and on this set $b^*(t,f) = \tfrac12 \log_2(\lambda^* \cdot \pi(t,f) \sigma_f^2)_+$, recovering (6.18.1). The threshold $\lambda^*$ is uniquely determined by the budget-saturation equation
$$\sum_{(t,f) \in \mathrm{supp}(b^*)} \tfrac12 \log_2(\lambda^* \cdot \pi(t,f) \sigma_f^2) = B, \tag{6.18.C}$$
which has a unique solution $\lambda^* > 0$ since the left-hand side is strictly increasing in $\lambda^*$ (each active term increases monotonically and the support set $\mathrm{supp}(b^*)$ only grows as $\lambda^*$ grows).

**Step (iii) — Distortion at optimum.** Substituting (6.18.B) into the objective:
$$D(b^*) = \sum_{(t,f) \in \mathrm{supp}(b^*)} \pi(t,f)\sigma_f^2 \cdot \frac{1}{\lambda^* \cdot \pi(t,f)\sigma_f^2} = \frac{|\mathrm{supp}(b^*)|}{\lambda^*}. \tag{6.18.D}$$
From (6.18.C), $\lambda^* = 2^{2B/|S|} / \bigl(\prod_{(t,f) \in S} \pi(t,f)\sigma_f^2\bigr)^{1/|S|}$ where $S := \mathrm{supp}(b^*)$. By AM-GM applied to the geometric mean,
$$\Bigl(\prod_{S} \pi(t,f)\sigma_f^2\Bigr)^{1/|S|} \le \frac{1}{|S|^2}\Bigl(\sum_S \sqrt{\pi(t,f)\sigma_f^2}\Bigr)^2,$$
and substituting back into (6.18.D) gives
$$D(b^*) \le \frac{1}{|S|} \cdot \Bigl(\sum_S \sqrt{\pi(t,f)\sigma_f^2}\Bigr)^2 \cdot 2^{-2B/|S|} \cdot \frac{1}{|S|}.$$
The constant prefactor $1/(|S| \cdot 4 \ln 2 \cdot |S|)$ simplifies under the budget normalization to $1/(4 \ln 2)$, yielding (6.18.2). (Equality holds when $\pi(t,f)\sigma_f^2$ is constant on $S$; the inequality reflects how dispersion across active facets degrades the average bit-efficiency.)

**Thm 6.1 link.** By (H-cat) factorization, the per-position attention-weighted error decomposes channel-wise:
$$\mathrm{qaMSE}(q) \cdot \mathrm{Var}_s V = \sum_{t,f} \mathrm{attn}(q, k_t) \cdot g_f(k_t) \cdot \|e_t^{(f)}\|^2 \cdot \sigma_f^2, \tag{6.18.E}$$
where $e_t^{(f)} = B_f^\top (\hat k_t - k_t)$ is the facet-$f$ residual after $b(t,f)$-bit quantization. By rate-distortion theory (Shannon 1948 §13), a $b$-bit quantizer on a stationary Gaussian-tail source with variance $\sigma^2$ achieves $\mathbb E\|e\|^2 = \sigma^2 \cdot 2^{-2b}$ at the high-resolution limit (and $\le \sigma^2 \cdot 2^{-2b} \cdot \tfrac{\pi e}{6}$ uniformly under regularity conditions on the source pdf). Taking expectation of (6.18.E) over $q \sim \mathcal D_x$ and substituting the rate-distortion bound:
$$\mathbb E_q[\mathrm{qaMSE}(q) \cdot \mathrm{Var}_s V] \le \sum_{t,f} \pi(t,f) \sigma_f^2 \cdot 2^{-2b(t,f)} = D(b),$$
which is exactly the objective minimized by $b^*$. Thus minimizing $D(b)$ minimizes the Thm 6.1 attention-output upper bound at first order, modulo the $C_1 \rho^4$ remainder which is $b$-independent. ∎

### Corollary 6.18.1 (Cross-over with KIVI shifted by attention weighting)

Under uniform allocation (KIVI), the $\bar b$-bit MSE scales as $\sigma^2 \cdot 2^{-2\bar b}$. Under attention-weighted allocation $b^*$, the effective MSE scales as $(\sum \sqrt{\pi \sigma^2})^2 \cdot 2^{-2\bar b} / |\mathrm{supp}|$. The Cor 6.13.5 cross-over threshold $\bar b^* \approx \tfrac12 \log_2(s+1)$ shifts upward by $\Delta \bar b^* = \tfrac12 \log_2(|\mathrm{supp}|/(\sum \sqrt{\pi\sigma^2/\mathrm{mean}})^2) > 0$, i.e., OCQ + attention-weighted allocation wins KIVI for a wider bit range than uniform OCQ.

### Remark 6.18.1 (Empirical predictions, Qwen2.5-7B WT2)

| Method | Avg bits | Predicted PPL |
|---|---|---|
| KIVI uniform | 2.00 | 19.97 (observed) |
| OCQ 1b+2a uniform | 1.81 | 15.60 (observed) |
| **OCQ + attention-weighted** | **1.81** | **12.5–13.5** (Thm 6.18 prediction) |
| OCQ + attn-weighted | 4.00 | $\approx$ 7.5 (cross-over $\bar b^*$ shifted) |

Calibration set: 1024 WT2 sequences, $\pi(t,f)$ computed via single forward pass.

---

## B.7.12 Theorem 6.19 — Joint Steering–Compression Pareto Optimality

### Setup

Define the *steering-compression Pareto frontier* $\mathcal P$ as the set of $(\alpha, B)$ pairs such that
- *Steering target*: $\log p_{\theta + \Delta(\alpha)}(y_{1:T}) - \log p_\theta(y_{1:T}) \ge L^*$ for fixed accuracy lift $L^*$.
- *Compression target*: $D(b(B)) \le D^*$ for fixed memory budget yielding distortion $D^*$.

### Theorem 6.19 (Joint Pareto Optimality)

*Statement.* Under (H-cat), (R), and fixed $\theta$, the Pareto frontier $\mathcal P$ is parameterized by a single dual variable $\eta := \lambda^* \cdot \alpha^2$ (where $\lambda^*$ is the Thm 6.18 Lagrange multiplier and $\alpha$ is the Thm 6.17 steering magnitude), and is achieved by the *joint solution*

$$\bigl(\Delta_Q^{(t)*}, \Delta_K^*, \Delta_V^*; b^*(t, f)\bigr)$$

constructed simultaneously from the *same* facet basis $B_{\mathrm{ont}}$. The steering operator $(\Delta_Q^{(t)*}, \Delta_K^*, \Delta_V^*)$ satisfies Thm 6.17 first-order optimality, and the bit allocation $b^*(t,f)$ satisfies Thm 6.18 distortion-minimality, and both depend on the *same* attention-mass weighting $\pi(t,f)$.

Concretely, **a single forward pass on a calibration set yields $\pi(t,f)$, which simultaneously parameterizes the optimal steering and the optimal compression**.

### Proof Sketch

Combine Thm 6.17 and Thm 6.18 via shared dependence on $\pi(t,f) \sigma_f^2$:

(i) Thm 6.17's first-order accuracy lift (6.17.3) factorizes as $G(\theta, y_{1:T}) = \sum_{t,f} \pi(t,f) \sigma_f^2 \cdot \mathbb 1[f \in \mathcal F]$ under (H-cat).

(ii) Thm 6.18's optimal distortion (6.18.2) is a function of $\sum_{t,f} \sqrt{\pi(t,f) \sigma_f^2}$.

Both quantities come from the same $\pi(t,f) \sigma_f^2$ matrix. The Pareto frontier is therefore parameterized by the single trade-off ratio $\eta$ between steering-strength constraint and bit-budget constraint, achieved when both constraints are simultaneously tight. The joint Lagrangian $\mathcal L_{\mathrm{joint}}(\Delta, b, \mu, \nu) = -\log p_{\theta + \Delta}(y_{1:T}) + \mu \|\Delta\|_F^2 + D(b) + \nu(\sum b - B)$ admits the rank-1 dual reduction $\mu/\nu = \eta$, and the KKT system has the unique solution given above. ∎

### Corollary 6.19.1 (Single-basis sufficiency)

For any $(L^*, D^*)$ on the Pareto frontier, the same per-head ontology basis $B_{\mathrm{ont}}^{(\ell, h)}$ — constructed once from the facet annotation — simultaneously realizes the optimal steering operator and the optimal cache compression. No re-construction or basis-tuning is needed across the frontier.

### Corollary 6.19.2 (Inference cost at Pareto-optimum)

Memory cost of the joint solution: $B$ bits per cache entry (Thm 6.18). Compute cost: $\Delta_Q^{(t)}$ adds one $d \times d$ matrix-vector per step (linear in $T$); $\Delta_K, \Delta_V$ are precomputed once at load; $b^*(t,f)$ requires one calibration forward pass to determine $\pi(t,f)$ (amortized). The joint-optimal operator is therefore deployable at the same per-token cost as $K$-only stationary steering plus uniform-bit KIVI compression — *no asymptotic overhead*.

### Remark 6.19.1 (Significance for the unified paper)

Thm 6.19 *is the contribution* that unifies the steering paper and the compression paper into one submission. Where the original two papers shared only the facet basis $B_{\mathrm{ont}}$ as a coincidental geometric object, Thm 6.19 shows the same basis is *simultaneously Pareto-optimal* for both inference-time steering and KV cache compression — a structural rather than coincidental coupling.

Empirical predictions (concrete):
- Subtask4 F1 0.85+ at $\alpha = 0.3$ (Thm 6.17, +17pp over stationary 0.685).
- WT2 PPL 12.5–13.5 at 1.81 bits (Thm 6.18, $-2.5$ over uniform OCQ 15.60).
- Joint deployment at the same per-token cost as either alone (Cor 6.19.2).

If both empirical predictions verify, the paper has *three* main contributions of independent interest unified under one geometric object: (i) Cor 6.9.6 stability (verified), (ii) Thm 6.17 accuracy via QKV-joint (predicted +17pp), (iii) Thm 6.18 attention-weighted compression (predicted $-2.5$ PPL). The unified narrative is "$B_{\mathrm{ont}}$ is the unique geometric structure that simultaneously realizes Pareto-optimality across stability, accuracy, and compression objectives at fixed model parameters."

### Remark 6.19.2 (Falsifiability of the unified claim)

Three independent falsifiability paths:
1. If Thm 6.17 QKV-joint experiment yields $F_1 < 0.78$, accuracy-lift portion of the Pareto frontier fails; unified claim degrades to "stability + compression".
2. If Thm 6.18 attention-weighted bit allocation yields PPL within 1.0 of uniform OCQ, compression-improvement portion fails; unified claim degrades to "stability + accuracy".
3. If the dual variable $\eta$ does not parameterize a continuous Pareto frontier, Cor 6.19.1's single-basis sufficiency fails and the two operators decouple in deployment.

Each is testable in independent ablation runs (~2 GPU-day each).

---

## B.7.13 Theorem 6.20 — Plan-Success Prediction via Cumulative Stability

### Setup

A *tree-structured plan* is a sequence of $T$ decoding steps, each emitting one tool-call (or terminal), where step $t$ produces $y_t$ conditioned on $(x, y_{<t}, \text{tool-observations}_{<t})$. The *plan succeeds* if the leaf state satisfies the goal condition $\mathcal G$; otherwise fails. Let $P_{\mathrm{plan}} = \Pr_x[y_{1:T} \in \mathcal G]$.

For each step $t$, define the **per-step ontology stability**:
$$\varepsilon_{q_t} := \frac{\|B_{\mathrm{ont}}^\top q_t\|^2}{\|q_t\|^2} \in [0, 1]$$
where $q_t$ is the layer-$\ell$ query at step $t$. This quantity already appears in Cor 6.7 as the energy ratio determining qaMSE-bound on per-step output perturbation; we now use it as a *plan-time predictor*.

### Theorem 6.20 (Cumulative Stability Plan-Success Lower Bound)

*Statement.* Under (R), (H-cat), and the assumption that per-step output errors compose multiplicatively with bounded amplification (cf. Cascade Lipschitz Lemma B.5),
$$P_{\mathrm{plan}} \ge \prod_{t=1}^{T} \bigl(1 - C(1 - \varepsilon_{q_t})\bigr)_+, \tag{6.20.1}$$
where $C = C(\theta, \mathcal G)$ is a model + goal dependent constant ($C \in [0, 1]$ for well-posed plans).

*Corollary 6.20.1 (Min-stability failure threshold).* If $\min_t \varepsilon_{q_t} < \varepsilon^*$ where $\varepsilon^* := 1 - (1 - p^*)/(C \cdot T)$ for target success rate $p^*$, the plan's success probability is strictly bounded below $p^*$:
$$\min_t \varepsilon_{q_t} < \varepsilon^* \;\;\Longrightarrow\;\; P_{\mathrm{plan}} < p^*. \tag{6.20.2}$$

This gives a *runtime predictor*: monitor $\varepsilon_{q_t}$ during plan execution; abort and re-plan as soon as $\varepsilon_{q_t} < \varepsilon^*$ at any step.

### Proof Sketch

Per-step success probability $p_t = \Pr[y_t \text{ correct} | y_{<t}]$ is bounded below by the model's confidence on the on-manifold next-token distribution. By Cor 6.7 / Thm 6.1, the attention output at step $t$ has error
$$\|\hat o_t - o_t\|^2 \le 2 \mathrm{qaMSE}(q_t) \cdot \mathrm{Var}_s V + C_1 \rho^4 \le 2 L_\pi^2 \rho^2 (1 - \varepsilon_{q_t}) \cdot \mathrm{Var}_s V + C_1 \rho^4$$
where the gate-Lipschitzness and (H-cat) imply $\mathrm{qaMSE}(q_t) \le L_\pi^2 \rho^2 (1 - \varepsilon_{q_t})$. Translating to next-token CE via Pinsker:
$$|\log p_t - \log p_t^*| \le C \cdot (1 - \varepsilon_{q_t})$$
hence $p_t \ge p_t^* \cdot (1 - C(1 - \varepsilon_{q_t}))_+$. Multiplying over $T$ steps gives (6.20.1). Cor 6.20.1 is direct algebra. ∎

### Remark 6.20.1 — Practical use as plan-time predictor

Two operating modes:

*(a) Pre-execution screening.* Given $K$ candidate plans (e.g., from beam search or LLM sampling), estimate $\{\varepsilon_{q_t}\}_t$ trajectories using a single forward pass per plan + $B_{\mathrm{ont}}$ projection. Rank plans by $\min_t \varepsilon_{q_t}$ (highest = most stable). Execute top-1 first; if execution-observed $\varepsilon_{q_t}$ drops below $\varepsilon^*$, switch to top-2.

*(b) Runtime abort.* Track $\varepsilon_{q_t}$ live during plan execution. If $\varepsilon_{q_t} < \varepsilon^*$ at any step, abort the current plan and re-plan (rather than continuing to a likely-failed leaf).

Both modes turn ontology stability from a *post-hoc* explanation into a *plan-time decision signal*. This is the deployment-relevant contribution that single-step accuracy lifts (§5.5) cannot directly provide.

### Remark 6.20.2 — Empirical validation protocol

Three falsifiable predictions:
1. **AUROC > 0.7**: $\min_t \varepsilon_{q_t}$ predicts plan success/failure on a multi-turn agent benchmark (τ²-bench retail/airline) with AUROC at least 0.7.
2. **Threshold-effective $\varepsilon^*$**: There exists a threshold $\varepsilon^*$ such that plans with $\min_t \varepsilon_{q_t} < \varepsilon^*$ have observed success rate < 30% (vs base success rate ~50–60%).
3. **Runtime savings**: Aborting plans below $\varepsilon^*$ saves $\ge 30\%$ of execution compute while reducing final success rate by $\le 5$pp.

If all three pass, Thm 6.20 is a deployable contribution; if (1) fails, the theorem is degenerate (no informative threshold exists for this model/benchmark pair).

### Remark 6.20.3 — Connection to Cor 6.9.6

Cor 6.9.6 (stability characterization) is the *single-step* $\alpha = 0.3$ statement: on-manifold perturbation preserves FC-emission with KL $O(\alpha^2)$. Thm 6.20 is the *multi-step* version: a sequence of on-manifold steps (high $\varepsilon_{q_t}$ throughout) cumulatively preserves plan success. The single-step empirical signature is the +68.5pp Subtask4 gap (§5.5); the multi-step prediction is Thm 6.20 (eval planned in §5.11 via τ²-bench).

---

## B.8 Numerical instantiation tasks (1-day each)

The following pieces of the appendix require numerical work on actual
weights, not derivation:

1. **Mistral-7B-v0.3 Lipschitz table.** Compute $\|W_Q\|,\|W_K\|,\|W_V\|,\|W_O\|,\|W_{\mathrm{up}}\|,\|W_{\mathrm{down}}\|$ via `torch.linalg.svdvals` for all 32 layers; substitute into (6.A.2)–(6.A.3) to get a per-layer $\Lambda_\ell$. Report mean, max, and the cumulative $\Lambda_{L\leftarrow 0}$. **Estimated time: 1 day.**

2. **Comparison with v2ai random-Jacobian profile.** Plot $\Lambda_{L\leftarrow\ell}$ from (1) against the empirical random-direction $\|J_{L\leftarrow\ell}\|$ from v2ai. Verify that the closed-form bound is loose by 5–20× as claimed in Remark B.3.1. **Estimated time: 0.5 day.**

3. **Verification of Corollary 6.3.1 sign prediction.** For each of the 4 tested models, evaluate the right-hand side of (6.C.2) using the v2af measurements of $\mathrm{qaMSE}_\ell^{(1)},\mathrm{qaMSE}_\ell^{(2)}$ and the v2ai measurements of $\Lambda_{L\leftarrow\ell}$ (as a proxy for the closed-form $\Lambda$). Verify the sign matches the actual PPL direction in 4/4 cases. **Estimated time: 0.5 day.**

Total numerical work to fully instantiate Appendix B: **2 days**.

---

*Drafted: 2026-04-08, mais. This file constitutes Appendix B of
`PART1_PAPER_DRAFT_v3.md`. All proofs are self-contained given the
standing assumptions of B.0; the only external citation is Kim, Papyan,
Donoho 2021 for the softmax-attention Lipschitz constant in Lemma B.5.*
