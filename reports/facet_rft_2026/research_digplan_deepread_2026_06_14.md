# DiG-Plan (arXiv:2606.05728) — Forensic Deep-Read + Mathematical Analysis

**Date:** 2026-06-13 · **Analyst task:** single-paper deep-read for FRFT/diffusion-planning line · **Citation discipline:** all quotes fetched from arXiv abs + `arxiv.org/html/2606.05728`; math-lit citations verified against their own abstracts. Anything not directly verified is flagged **[UNVERIFIED]**.

---

## 0. 핵심 판정 (executive verdict)

**(a) Is DiG-Plan's headline claim credible after a fairness audit?**
The paper is **real and exists** (verified, see §A1). But the famous "0.32 vs 0.94" number is **NOT a TaskBench result — it is a synthetic 23-bit toy experiment** on a 2-layer/128-dim toy Transformer. Our memory note recorded it as if it were a "TaskBench-23 / 501-task" headline; that conflation is **wrong and must be corrected**. The 0.32→0.943 gap is **real but only demonstrates a known, almost-tautological fact**: a fixed-length bit-vector with 30% random masking gives a denoiser 10 *different* parallel reveal orders, while *greedy* (T=0) AR gives literally one deterministic sample repeated 10×. The comparison is **tilted by construction** (greedy AR has zero diversity by definition, so Pass@10 ≈ Pass@1). The honest, real-data result is the **TaskBench 0.661→0.729 ToolF1 (~+10% relative)**, and even there diffusion alone is *catastrophically worse* (ToolF1 0.128) — it only works bolted to an AR refiner.

**Credibility: the mechanism (early-commitment hurts AR coverage) is real and the paper is honest enough to show diffusion-only fails; but the headline 0.32/0.94 is a rhetorical toy result, not evidence of a 3× planning advantage.** Credible as *motivation*, NOT as a magnitude.

**(b) Is there a mathematical basis for diffusion > AR in planning?**
**Partially provable, mostly conditional.** What is genuinely provable: (i) fixed-order AR pays a *representational/diversity* cost on order-invariant (set-valued) targets relative to any-order/marginalized models [ARDM, XLNet — provable]; (ii) AR error compounds up to **quadratically** in sequence length under exposure bias (T·ε ≤ R ≤ T²·ε) [Arora et al. 2204.01171 — provable bound]. What is NOT provable: that diffusion *always* wins (both are universal joint-distribution approximators — §B4). The coverage advantage is dominantly about the **decoding regime (Pass@k with parallel stochastic reveals)**, not an intrinsic model-class supremacy.

**(c) Decision for us:** This *weakens* the case for expecting a large D-oracle from Dream-7B P-D on a real benchmark, and *re-frames* what to measure. See §D.

---

## A. Forensic read of DiG-Plan

### A1. Identity / existence — VERIFIED
- **Title (verbatim):** "DiG-Plan: Mitigating Early Commitment for Tool-Graph Planning via Diffusion Guidance"
- **Authors:** Yansi Li, Zhuosheng Zhang
- **arXiv:** 2606.05728, **v1**, submitted **2026-06-04**. The future-looking ID is **real** — it resolves on both `arxiv.org/abs/2606.05728` and `arxiv.org/html/2606.05728`. Venue: preprint (no venue stated).
- Affiliations not listed on the abs page. (Zhuosheng Zhang is associated with SJTU in prior work — **[UNVERIFIED for this paper]**.)

**Abstract (verbatim, load-bearing sentence):**
> "A controlled study shows that masked denoising raises Pass@10 solution coverage from 0.320 to 0.943 over AR sampling under matched compute. ... DiG-Plan employs a diffusion-based proposer to generate diverse tool sets via iterative refinement, followed by an AR refiner for dependency prediction. On TaskBench, DiG-Plan improves over AR baselines by a 10% relative margin..."

Note the abstract itself separates the **0.32/0.94 "controlled study"** from the **TaskBench "10% relative margin."** They are two different experiments. Our memory note merged them.

### A2. The task
"Tool-graph planning" = given an NL instruction, select the correct **subset** of tools from a library, then predict the **dependency edges** (a DAG). Output is a directed tool graph G̃=(Ṽ,Ẽ). The pipeline is **propose → refine → select**: a diffusion proposer emits diverse candidate tool *sets*; an AR refiner predicts dependency edges.
- **"AR refiner" (verbatim):** "refine each candidate into a directed tool graph G̃ₖ=(Ṽₖ,Ẽₖ) by predicting dependency edges with autoregressive decoding." → **The final system is a diffusion+AR hybrid, not diffusion-only.**

### A3. The "501 tasks" / TaskBench-23 — CUSTOM, not standard
- **Verbatim:** "We use N=501 instances, with 167 each for single-tool, chain, and DAG tasks." Compositional subset = "chain and DAG instances, yielding N=334 instances."
- "TaskBench-23" is the paper's **name for its setup** (23-tool universe in the toy study). The selection rule (how the 501/167-per-bucket were drawn from TaskBench's HuggingFace/Multimedia/DailyLife splits) is **NOT specified** in the fetched HTML. **[Selection procedure UNVERIFIED — flag: balanced-by-construction 167/167/167 is suspiciously clean and may oversample easy single-tool cases.]**

### A4. The headline numbers VERBATIM + what they actually are
**Table 1 (the famous numbers) is a SYNTHETIC TOY, verbatim context:**
> "To validate the early commitment hypothesis, we construct a controlled tool-set prediction task that isolates the combinatorial search component. We define a tool universe 𝒯={t₁,…,t₂₃} and represent tool subsets as fixed-length 23-bit vectors."
> Backbones: "identical small Transformer backbones with 2 layers, 128 hidden dimensions, and 4 attention heads for both AR and denoising models" on a "shared synthetic corpus from the same distribution," output "fixed ... to 23 bits," masking "30% of bits."
> **Table 1:** "Greedy AR: Pass@10=0.320±0.00; Masked denoising: Pass@10=0.943±0.02."
> **Same Table 1, k=1 quality:** "Greedy AR: 0.355±0.00; Masked denoising: 0.349±0.04" (comparable).

**The real TaskBench numbers:**
- Main: "DiG-Plan with Dream proposer achieves ToolF1 of 0.729 ... outperforming the AR two-stage baseline with ... 0.661" → +0.068 abs, ~+10.3% rel.
- **Table 3 (held-out compositional, N=334, K=10):** "AR proposer (T=1.3): Oracle@10=0.735, UnionPrec@10=0.575"; "Dream proposer: Oracle@10=0.787, UnionPrec@10=0.692."
- **Diffusion-only failure (verbatim):** "Dream DLM-only baseline achieves high EdgeRec of 0.335 but low ToolF1 of 0.128 ... end-to-end diffusion generation struggles with precise tool-name grounding and complex dependency prediction on compositional tasks, motivating our two-stage design."

**Models (main experiments):** diffusion proposer = **Dream 7B** (Ye et al. 2025), or LLaDA-8B-Instruct / LLaDA2.0-mini-preview; AR components = **Qwen2.5-7B-Instruct**. Both sides ~7–8B → roughly size-matched at the *system* level, but **different base families** (Dream vs Qwen) — a confound (see flag table).

### A5. Fairness audit — flag table

| # | Potential tilt | Evidence | Severity |
|---|---|---|---|
| F1 | **Greedy AR vs stochastic diffusion in Table 1.** Greedy (T=0) AR has *zero* sample diversity → Pass@10≡Pass@1 by construction. The 0.32 is not "AR is bad at coverage," it's "we disabled AR's sampling." | "Greedy AR" vs "stochastic denoising," both ±0.00 vs ±0.02 | **HIGH** — invalidates the 0.32/0.94 as a coverage comparison |
| F2 | **Toy ≠ benchmark.** Table 1 is a 2-layer/128-dim model on synthetic 23-bit vectors, presented in the abstract adjacent to TaskBench claims. | "controlled tool-set prediction task ... 23-bit vectors" | **HIGH** — magnitude does not transfer; real gap is ~10% |
| F3 | **Different base families.** Diffusion = Dream/LLaDA; AR = Qwen2.5. Not the same pretraining corpus/recipe. Cannot attribute gain purely to the diffusion *objective*. | "Dream 7B ... LLaDA-8B"; "Qwen2.5-7B-Instruct" | **MEDIUM** |
| F4 | **Post-hoc metric choice.** Headline metric is *union recall over K* (Pass@K), which structurally rewards diversity — exactly what diffusion's stochastic reveal maximizes. Single-sample ToolF1 is comparable (0.355 vs 0.349) / favors neither. | Eq. 3 UnionRecall@K; Table 1 k=1 tie | **MEDIUM** |
| F5 | **AR given high T=1.3 (mitigating, in AR's favor).** In Table 3 the AR proposer was sampled hot to *help* its diversity, yet still lost. This is a point of fairness *for* the paper. | "AR proposer (T=1.3)" | mitigating |
| F6 | **No latency/throughput/NFE accounting.** "matched compute" defined only as K=10 candidates; diffusion's multi-step iterative refinement NFE cost vs AR forward passes is not reported. | No limitations/latency section found | **MEDIUM** |
| F7 | **Selection of 501 tasks undocumented + perfectly balanced 167/167/167.** | "167 each" | **LOW-MED** |

### A6. Ablation honesty
- **Where diffusion loses — they DO show it:** diffusion-only ToolF1 0.128 (§A4). This is to the authors' credit and is the single most important honest disclosure in the paper.
- **Pass@1:** reported only as the toy k=1 tie (0.355 vs 0.349). No real-data Pass@1 table found.
- **Latency/throughput:** **absent.** No dedicated limitations section. Error analysis §5.5: "missing edges ... 82.9% of samples."
- **Metric definition (verbatim, Eq. 3):** "Pass@K=UnionRecall@K=|(∪ₖ Ṽₖ)∩V⋆|/max(|V⋆|,1)" → **oracle-style union recall**: does the union of K candidate tool-sets *cover* the ground-truth set. This is **coverage/recall, NOT "any-one-of-10 is fully correct."** It maps to **"the pool collectively contains the right tools"**, which is *adjacent to but not identical to* our D-oracle ("adds a NEW fully-correct plan"). See §D.

---

## B. Mathematical analysis — can diffusion be provably better at planning?

### B1. Factorization / order-invariance argument — **[provable (representational), conditional (benefit)]**
AR models p(y)=∏ₜ p(yₜ|y_<t) under **one fixed left-to-right order**. For a target that is a **set** (tool-set) or a DAG with **multiple valid topological orders**, the fixed order forces the model to commit to an arbitrary serialization. 
- **Provable fact (XLNet, arXiv:1906.08237):** the permutation objective maximizes E_z[Σ log p(y_{z_t}|y_{z_<t})] over all T! orders; "the same parameters are shared across all factorization orders ... in expectation each token has seen every other token." → an any-order model *marginalizes* over orderings; a single-order AR does not. This is a genuine representational difference.
- **Provable fact (ARDM, arXiv:2110.02037, Hoogeboom et al.):** ARDMs "unify and generalize order-agnostic autoregressive models and absorbing discrete diffusion as special cases" and "do not require causal masking." → **absorbing-state (masked) diffusion = order-agnostic AR in expectation.** This is the formal bridge: masked-diffusion training optimizes an ELBO that averages over reveal orders [MDLM, arXiv:2406.07524 — "Rao-Blackwellized objective," masked-diffusion ELBO; the absorbing-state/ELBO identity is standard though the fetched MDLM abstract excerpt did not quote it verbatim — **[partially UNVERIFIED quote, claim is standard]**].
- **What this gives:** if the target distribution is genuinely order-invariant, the *single-order* AR pays a strictly non-negative KL penalty for modeling a spurious order; an order-marginalized model does not. **Provable that the penalty is ≥0; conditional whether it is large** — for tool-SETS (small, unordered) it can matter; for the DAG *edges* (which DO have causal/dependency order) AR is the *right* inductive bias, which is exactly why DiG-Plan keeps AR for edges (§A4 / diffusion-only ToolF1 0.128).

### B2. Exposure bias / error cascade — **[provable bound, conditional that diffusion fixes it]**
- **Provable (Arora et al., "Why Exposure Bias Matters," arXiv:2204.01171):** regret of AR generation satisfies **T·ε ≤ R(p_θ,F) ≤ T²·ε**, ε = per-step error, T = sequence length. Lower bound = no accumulation; **upper bound = quadratic compounding** in length. This formalizes "an early wrong token corrupts everything downstream."
- **Does diffusion provably escape it?** **No clean theorem.** Iterative denoising *can revisit* earlier positions (non-causal), so it is not bound to monotone left-to-right error accumulation — but it introduces its own *unmasking-order* and *step-count* error sources. The reduction is **empirically plausible, not proven.** Tag: **conditional / empirical-only.** DiG-Plan's own "early commitment" framing is exactly this argument used as *motivation*, not theorem.

### B3. Coverage / mode-recall — **[conditional, links to verified VB/VF theorem]**
Pass@K union-recall grows with the **spread** of the generator. 
- A peaked (low-temp / greedy) AR distribution collapses K samples toward one mode → low union recall (this is precisely Table 1's 0.32). A higher-entropy sampler (hot AR *or* stochastic diffusion) raises union recall. So **the coverage advantage is primarily a property of the SAMPLING ENTROPY, not the model class** (F4).
- **Link to our verified result — Setlur et al., arXiv:2502.12118 ("Scaling Test-Time Compute Without Verification or RL is Suboptimal"):** verification-based selection beats verifier-free **"when the base ... LLM presents a heterogeneous distribution over correct solution traces (e.g., different lengths, styles)"**, formalized via **anti-concentration** (non-sharp reward distribution). 
- **The connection (conditional):** diffusion's order-agnostic stochastic reveals *plausibly* increase trace heterogeneity (more distinct correct tool-sets in the pool) → larger anti-concentration → larger VB headroom. **But 2502.12118 does NOT prove diffusion increases heterogeneity; it proves heterogeneity is what makes verification pay off.** So: *if* diffusion raises heterogeneity (empirical question), *then* the verified separation theorem says our VB selector gains. Tag: **conditional — the antecedent is exactly what we must measure.**

### B4. The honest limit — **[provable]**
Both AR transformers and (masked) diffusion transformers are **universal approximators of the joint p(y)** (AR by chain rule with sufficient capacity; diffusion via its ELBO converging to the data dist). Therefore **"diffusion is always better" is provably FALSE** — at the distribution-modeling limit they are equivalent. 
- **The advantage is therefore NOT in the model class but in (a) the inductive bias under finite capacity/data for order-invariant targets (§B1) and (b) the DECODING regime — parallel/stochastic Pass@k refinement (§B3).** DiG-Plan's own data confirm this: single-sample ToolF1 is a *tie* (0.355 vs 0.349); the gap appears *only* at K>1 (coverage) and *only* when AR is choked to greedy.
- **Where diffusion is NOT favorable:** strongly-ordered/causal sub-structure (the dependency EDGES) — DiG-Plan concedes this by keeping an AR refiner and reporting diffusion-only ToolF1 0.128.

### B5. Map to OUR D-oracle — **[empirical-only, with a provable upper bound]**
Our D-oracle asks: does the diffusion pool add **NEW fully-correct plans** that raise the pool's oracle ceiling, vs merely producing *different* (not more-correct) outputs?
- DiG-Plan's Pass@K = **union recall of tools**, which is *necessary but not sufficient* for a new fully-correct plan (covering the right tool-set ≠ producing a correct DAG — recall their edge errors hit 82.9% of samples). So **DiG-Plan's coverage win is an UPPER BOUND on, not a proof of, a D-oracle gain.**
- **Provable:** if diffusion's pool has strictly higher union recall AND the verifier can pick correctly, oracle@K is non-decreasing — the *ceiling* can only rise or stay. **Empirical:** whether that ceiling rise translates to *new fully-correct plans* (edges included) on OUR benchmark is unproven and, given the 82.9% edge-error rate, **likely modest.**

---

## C. Verified bibliography vs unverified leads

**Verified (fetched + quoted from primary source):**
- DiG-Plan — arXiv:2606.05728 v1 (2026-06-04), Li & Zhang. Real.
- ARDM "Autoregressive Diffusion Models" — arXiv:2110.02037, Hoogeboom, Gritsenko, Bastings, Poole, van den Berg, Salimans. Unifies order-agnostic AR + absorbing diffusion.
- MDLM "Simple and Effective Masked Diffusion Language Models" — arXiv:2406.07524, Sahoo, Arriola, Schiff, Gokaslan, et al. Masked-diffusion ELBO / Rao-Blackwellized objective.
- XLNet — arXiv:1906.08237, Yang et al. Permutation LM marginalizes over T! orders.
- "Why Exposure Bias Matters" — arXiv:2204.01171, Arora et al. Regret bound T·ε ≤ R ≤ T²·ε.
- Setlur, Rajaraman, Levine, Kumar — arXiv:2502.12118 "Scaling Test-Time Compute Without Verification or RL is Suboptimal." VB>VF under heterogeneous-trace (anti-concentration) condition. (This is the paper our note labels the "VB/VF separation theorem.")

**Unverified leads / flags:**
- DiG-Plan affiliations (likely SJTU via Z. Zhang) — **[UNVERIFIED]**.
- Selection procedure for the 501 TaskBench instances (167/167/167) — **[NOT in fetched HTML]**.
- Dream 7B (Ye et al. 2025) and LLaDA-8B (Nie et al. 2025) primary specs not re-verified here — cited as DiG-Plan reports them.
- MDLM's exact absorbing-state/ELBO sentence not quoted verbatim (claim is standard; **[partial]**).
- No DiG-Plan latency/NFE/limitations data exists to verify (**absent in paper**).

---

## D. Decision for us — does this strengthen or weaken Dream-7B P-D?

**Net: WEAKENS the magnitude expectation, SHARPENS the experiment design.**

1. **Do not quote 0.32→0.94 as a planning result.** It is a greedy-AR-vs-stochastic-diffusion toy on 23-bit vectors. Citing it in our writeups would be a fairness violation and is contradicted by the paper's own real number (~+10% rel). **Correct the memory note** that recorded "AR Pass@10 0.32 vs diffusion 0.94 on TaskBench-23 501-task."

2. **Realistic D-oracle expectation for Dream-7B P-D:** small-to-moderate. The closest real-data signal is DiG-Plan's **Oracle@10 0.735→0.787 (+0.052) and UnionPrec 0.575→0.692** — i.e. the pool gets *somewhat* richer and *cleaner* at K=10, but single-sample quality is a tie. Expect Dream-7B to **raise pool union/coverage, not single-shot accuracy.** Whether that yields *new fully-correct plans* (our strict D-oracle) is gated by edge/dependency correctness, which diffusion is bad at (ToolF1 0.128 standalone). **Budget for a modest D-oracle, dominated by tool-SET recall, not DAG-edge gains.**

3. **This is fundamentally a heterogeneity / VB-selector story (2502.12118), and that is OUR thesis.** DiG-Plan empirically supplies the missing antecedent for our verified separation theorem: a diffusion proposer plausibly raises pool heterogeneity → more anti-concentration → more headroom for a verifier/selector. **So P-D is worth running specifically to MEASURE Δheterogeneity of the Dream pool** — that is the load-bearing, provable-once-measured quantity, not raw Pass@k.

4. **Design guardrails for our P-D run (to avoid DiG-Plan's tilts):** (i) compare Dream-7B pool against AR sampled at *matched entropy* (hot T, e.g. ≥1.0), never greedy — else we reproduce F1 and overstate the gain; (ii) report Pass@1 and per-NFE/latency-matched budgets, not just Pass@10; (iii) separate **tool-set recall** from **full-plan (edge-inclusive) correctness** — our D-oracle must be the strict edge-inclusive one; (iv) report Δheterogeneity explicitly to connect to 2502.12118.

**Bottom line:** DiG-Plan is a *credible motivator and a useful negative result* (diffusion-only fails; hybrid needed), not evidence of a 3× planning win. Run Dream-7B P-D, but expect a coverage/heterogeneity gain (the provable lever for our VB selector), not a single-shot accuracy miracle — and measure Δheterogeneity as the primary outcome.
