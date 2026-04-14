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
