# Architecture-for-Planning Related Work — Full-Text Deep-Read & Fit/Error Audit (diffusion excluded)

> 2026-06-14. Author = research agent (deep-read pass). Companion to `research_arch_planning_survey_2026_06_14.md`.
> **Strict citation discipline applied**: every paper below was opened at its arXiv abs and (where available) HTML full text in *this* session. Title/authors/version confirmed; planning/tool metrics quoted verbatim from the tables, not from memory or from the survey. Fairness flags are mine after reading method + setup. Numbers that I could only confirm from the survey (not re-derivable from the abs/HTML I could fetch) are explicitly marked.
> **Scope**: the ARCHITECTURE/DECODING-for-planning lines. Diffusion LMs excluded by design.
> **Our setting (the lens for every "FIT" verdict)**: Qwen2.5-7B + LoRA; K-sample generation + **gold-free** selection; **JSON-DAG** (set/graph of tool calls) output; deterministic gates; vLLM multi-LoRA deployment. Adoption filter #1 = "retraining-free (decoding-time) AND multi-LoRA-deployable"; filter #2 = "supplies a *heterogeneous* candidate to the gold-free selection pool (D-oracle>0 diversity)".

---

## 0. Verification ledger (what resolved, what did not)

| arXiv ID | Title (verified) | Version | Resolves? |
|---|---|---|---|
| 2601.13228 | Autoregressive Models Rival Diffusion Models at ANY-ORDER Generation | v1, 2026-01-19 | ✅ real |
| 2412.06769 | Training Large Language Models to Reason in a Continuous Latent Space (Coconut) | v3, rev 2025-11-03 | ✅ real |
| 2411.15100 | XGrammar: Flexible and Efficient Structured Generation Engine | v3, rev 2025-05-12 (MLSys'25) | ✅ real |
| 2404.03683 | Stream of Search (SoS): Learning to Search in Language | (orig 2024-04) | ✅ real |
| 2412.04703 | Transformers Struggle to Learn to Search | v2, 2024-12-06 / rev 2025-03-16 | ✅ real |
| 2310.05707 | Guiding Language Model Reasoning with Planning Tokens | v4, 2024-08-06 (COLM'24) | ✅ real |
| 2310.02226 | Think before you speak: Training LMs with Pause Tokens | v3, rev 2024-04-21 (ICLR'24) | ✅ real |
| 2402.14083 | Beyond A*: Better Planning with Transformers (Searchformer) | v2, rev 2024-04-26 | ✅ real |
| 2502.05171 | Scaling Test-Time Compute with Latent Reasoning (Huginn) | v2, rev 2025-02-17 | ✅ real |
| 2505.16782 | Reasoning Beyond Language: Survey on Latent CoT | v2, rev 2025-11-01 | ✅ real |
| 2305.11554 | ToolkenGPT (frozen LLM + tool embeddings) | v4, rev 2024-01-15 (NeurIPS'23 oral) | ✅ real |
| 2601.21358 | Latent CoT as Planning: Decoupling Reasoning from Verbalization (PLaT) | 2026-01-29 / rev 2026-02-04 | ✅ real (was an "Unverified lead") |
| 2603.13426 | Outcome-Aware Tool Selection for Semantic Routers (OATS) | 2026-03-13 ("WIP") | ✅ real (was an "Unverified lead") |
| **2603.27905** | "ATLAS-RTC: ... Token-Level Runtime Control" | claimed 2026-03 | **✗ DOES NOT RESOLVE** — see §6(d) |

All 10 priority IDs resolve. Two of the survey's "Unverified leads" (2601.21358, 2603.13426) now confirmed real. One survey lead (**2603.27905 ATLAS-RTC**) does **not** resolve to a verifiable paper and must not be cited.

---

## 1. The headline papers — claim, verbatim metric, fairness, FIT, ERROR

### A3 — "Autoregressive Models Rival Diffusion Models at ANY-ORDER Generation" (2601.13228v1)
Du, Fang, Yang, Zhang, Wei, Wang, Wang. Submitted 2026-01-19.

**Claim.** A general "any-order any-subset" AR objective: the joint is factorized at *group* level — verbatim `P(x_{1:n}) = ∏_{k=1}^{K} P(x_{G_k} | x_{G_{<k}})` with "randomly sampled group partitions and permutations." Headline: "A3 outperforms diffusion-based models while maintaining flexible decoding."

**Metrics (verbatim, Table 1; A3-8B / 3B / 1B vs Plaid-1B, Dream-7B, DiffuLlama-7B, LLaMA-3.1-8B):**
- TriviaQA: **A3-8B 19.4 vs DiffuLlama-7B 18.5 vs Dream-7B 18.3 — and LLaMA-3.1-8B 52.1.**
- PIQA: **A3-8B 78.1 vs DiffuLlama 63.3 vs Dream 55.8 — and LLaMA-3.1-8B 80.3.**
- Story infilling ROUGE (A3-8B): **19.2 / 4.6 / 18.6** (R-1/2/L); DiffuLlama is *higher* on infilling ROUGE.
- Training tokens: **A3 = 2B; DiffuLlama = 65B** (paper, verbatim: "A3 is trained on only 2B tokens, whereas DiffuLlama is trained on 65B").

**FAIRNESS — ⚠️ MUCH MORE TILTED THAN THE SURVEY STATED.** The survey marked A3 "✅ direct head-to-head with same-scale diffusion." That is half the picture. The decisive fact: **A3 loses badly to the AR topline (TriviaQA 19.4 vs 52.1; PIQA 78.1 vs 80.3).** The paper itself concedes: "still underperforms the AR baseline, this gap is likely attributable to limited training data." LLaMA-3.1-8B is a *standard L→R pretrained AR model used as a topline reference*, NOT an any-order competitor — so the title's "AR rivals diffusion" is **A3-the-any-order-AR rivaling diffusion, not standard AR**. And it "rivals" only because A3 is data-starved (2B) yet still edges data-rich diffusion (65B) — an *efficiency* argument, not a parity-budget win. There is **no same-budget any-order AR vs diffusion row**.

**FIT.** Conceptually a **candidate-generator arm** (an any-order LoRA could emit unordered tool-SETs). But it is a **retraining-from-objective** path, not decoding-time, and is unverified on 7B-instruct/LoRA. High risk.

**ERROR / divergence.** This is the DiG-Plan failure pattern in mirror image: it does *not* beat AR; it beats a deliberately weak diffusion baseline on an efficiency-normalized comparison. For our D-oracle>0 goal it *could* in principle supply an order-diverse candidate, but the evidence that any-order AR is even competitive with standard AR at our scale is absent.

**rw-sentence.** *A3 demonstrates that a single AR backbone can be trained for any-order/any-subset generation and edge same-scale diffusion under a small data budget, but it underperforms standard left-to-right AR, so it stands as a conceptual anchor that diffusion is unnecessary for unordered generation rather than evidence that any-order generation should replace our AR proposer.*

### Coconut — "Reason in a Continuous Latent Space" (2412.06769v3)
Hao, Sukhbaatar, Su, Li, Hu, Weston, Tian. Base = **GPT-2** (all experiments).

**Claim.** Feed the last hidden state back as the next input embedding (latent mode), so "continuous thoughts can encode multiple alternative next steps, allowing the model to perform a breadth-first search (BFS) rather than committing prematurely to a single deterministic path as in CoT."

**Metrics (verbatim, Table 1):** GSM8k — **CoT 42.9±0.2 vs Coconut 34.1±1.5**. ProsQA — **CoT 77.5±1.9 vs Coconut 97.0±0.3**.

**FAIRNESS — ✅ same-size, ⚠️ split result + instability.** Same GPT-2 weights, inference/training-procedure differences only. But Coconut **loses to AR-CoT on GSM8k** and only wins on the synthetic planning task ProsQA. Paper concedes training instability: at c=3 "a sharp spike in training loss, causing instability." Synthetic bench (ProsQA); GPT-2 scale.

**FIT.** Latent path → **internalization** layer. But retrofit on a pretrained 7B-instruct is unsupported (needs custom curriculum/architecture training); not LoRA-friendly; output is unverifiable latents (antithetical to our *verifiable* JSON-DAG).

**ERROR.** "BFS-in-latent" is exactly what our explicit K-sample pool already does *visibly*. Coconut hides the diversity where our gold-free verifier cannot gate it. Math-loses + instability + GPT-2 = narrow, fragile.

**rw-sentence.** *Coconut shows latent "continuous thoughts" can beat CoT on synthetic planning (ProsQA 97.0 vs 77.5) but loses on math (GSM8k 34.1 vs 42.9) and trains unstably on GPT-2, so its BFS-in-latent benefit is something our explicit, verifier-gated K-sample pool already obtains without sacrificing output verifiability.*

### Stream of Search (2404.03683)
Gandhi, Lee, Grand, Liu, Cheng, Sharma, Goodman. Model = **GPT-Neo 250M from scratch** (survey-sourced; abs confirms task/claims, not size). Task = **Countdown**.

**Claim/metrics (verbatim from abs).** "SoS pretraining increases search accuracy by 25% over models trained to predict only the optimal search trajectory." After STaR/APA finetuning: "the finetuned SoS models solve 36% of previously unsolved problems, including problems that cannot be solved by any of the heuristic solvers." Core idea: represent the *search process* (including fruitful mistakes/backtracking) as a flattened string.

**FAIRNESS — ✅ clean same-architecture baseline (optimal-trajectory-only), and surpasses its teacher solvers.** ⚠️ from-scratch 250M, single synthetic domain; 7B transfer unverified.

**FIT.** A **candidate-generator arm via trace-distillation (SFT)**: use our deterministic verifier to synthesize cost-aware search traces (with failures/backtracks), distill into a LoRA. Aligns with EXPERIMENT_DESIGN §3.10 build path. Retraining (medium cost), but vLLM multi-LoRA compatible.

**ERROR.** Genuinely beats its same-arch baseline and the teacher — the cleanest positive result in this set. Risk = scale-transfer (see Searchformer enc-dec caveat + the 2412.04703 ceiling warning).

**rw-sentence.** *SoS shows that distilling the full search trace — failures included — yields +25% search accuracy over optimal-only training and lets a small model surpass its teacher solvers, making "deterministic-verifier-generated trace distillation" our most credible internalization arm despite the 250M-from-scratch caveat.*

### Searchformer / "Beyond A*" (2402.14083v2)
Lehnert, Sukhbaatar, Su, Zheng, Mcvay, Rabbat, Tian. **Encoder-decoder Transformer from scratch.**

**Claim/metrics (verbatim).** Searchformer "optimally solves previously unseen Sokoban puzzles 93.7% of the time, while using up to 26.8% fewer search steps than the A* implementation," and "significantly outperforms baselines that predict the optimal plan directly with a 5-10× smaller model size and a 10× smaller training dataset."

**FAIRNESS — ✅ surpasses A* and the solution-only baseline; ⚠️ from-scratch enc-dec, maze/Sokoban.** Same family as SoS (search-dynamics bootstrapping). Decoder-only LLM / LoRA transfer unverified.

**FIT.** Same internalization-via-trace path as SoS; the "predict search dynamics, then bootstrap to fewer steps" recipe is the cost-aware angle. Retraining; architecture is enc-dec so it is a *recipe* to borrow, not a model to adopt.

**ERROR.** Strong keystone (beats the teacher search algorithm) but enc-dec-from-scratch — adoption = re-architecting, so only the *trace-bootstrapping recipe* transfers.

**rw-sentence.** *Searchformer surpasses A* itself (93.7% optimal Sokoban, −26.8% search steps) by bootstrapping search-dynamics traces, validating SoS-style trace internalization as a teacher-exceeding recipe — but as a from-scratch enc-dec it contributes the recipe, not the model, to our 7B-LoRA setting.*

### Planning Tokens (2310.05707v4)
Wang, Caccia, Ostapenko, Yuan, Wang, Sordoni. COLM'24. Bases = **Phi-1.5 (1.3B), Llama2-7B, Llama2-13B** (LoRA finetune).

**Metrics (verbatim, SQ-VAE variant):** Llama2-13B GSM8K **44.6→50.6 (+6.0)**, AQUA **41.3→43.9 (+2.6)**, MATH **7.2→8.5 (+1.3)**; Llama2-7B GSM8K **38.2→40.0 (+1.8)**, AQUA **36.6→41.3 (+4.7)**. Overall: "improve upon the baseline without planning tokens by **3.3% accuracy points on average** over three pre-trained language models." Parameter overhead: "negligible increase in trainable parameters (**0.001%**)."

**FAIRNESS — ✅ same-base finetune baseline, standard math benches, real 7B/13B, LoRA.** ⚠️ math-word-problem only (not DAG/tool); gains small but robust.

**FIT.** **Lowest-risk candidate-generator arm we have**: +0.001% params, LoRA-native, real-7B-validated. Attach a high-level plan-token before each A2 dirgraph step → a heterogeneous arm for the gold-free pool. Retraining but cheap.

**ERROR.** Robust same-base win, not tilted. Limits: small gains, demonstrated only on math word problems — DAG/tool transfer is our open question.

**rw-sentence.** *Planning Tokens give a real, robust +3.3pt average over a same-base LoRA finetune at +0.001% params on actual 7B/13B models, making per-step plan-token LoRA our cheapest, lowest-risk way to inject a heterogeneous candidate into the selection pool.*

### Pause / Filler Tokens (2310.02226 ICLR'24)
Goyal, Ji, Rawat, Menon, Kumar, Nagarajan. Decoder-only **130M & 1B**, C4 causal pretrain.

**Metrics (verbatim).** 1B model: "**18% EM** gain on … SQuAD," "**8%** on CommonSenseQA," and "**1%** accuracy on … GSM8k." Crucial caveat: gains require the model to be "**both pre-trained and finetuned with delays**" — finetune-only pause-injection is weak.

**FAIRNESS — ⚠️ task-dependent, reasoning gain negligible (+1pt GSM8k), and needs pause in pretraining (we can't redo Qwen pretraining).** Filler-token sibling (2404.15758): filler works on two algorithmic tasks but "Learning to use filler tokens is difficult and requires specific, dense supervision to converge."

**FIT.** Decoding/internalization adjacent, but the pretrain-coupling requirement kills the retraining-free appeal and the reasoning gain is ~+1pt.

**ERROR.** Honest but small; not tilted, just weak for our purpose.

**rw-sentence.** *Pause tokens give large gains only on extractive QA (+18% SQuAD) while reasoning barely moves (+1pt GSM8k) and the effect needs pause tokens in pretraining we cannot redo, so this is a distraction for our setting.*

### Transformers Struggle to Learn to Search (2412.04703v2) — the ceiling warning
Saparov … Najoung Kim, He He (9 authors).

**Claim (verbatim).** On graph-connectivity search: "This difficulty is not resolved even as the number of parameters is increased," and "performing search in-context (i.e., chain-of-thought) does not resolve this inability."

**FAIRNESS — ✅ controlled negative result.** Scale and CoT both fail to fix large-graph search.

**FIT.** A **guard**, not a method. Pin it before investing in any search/value-head internalization (SoS, Searchformer, value-heads): confirm our trees are *not* in the hard regime (our planning trees are shallow, ~2–7 conditions) before betting compute.

**rw-sentence.** *This controlled negative result — search difficulty unfixed by more parameters or CoT — is the ceiling warning to gate any search-internalization spend behind "are our trees shallow enough to escape this regime?".*

### Huginn / Latent Recurrent Depth (2502.05171v2)
Geiping, McLeish, Jain, Kirchenbauer, Singh, Bartoldson, Kailkhura, Bhatele, Goldstein. **3.5B params, 800B tokens, from scratch.**

**Claim (verbatim).** "iterating a recurrent block, thereby unrolling to arbitrary depth at test-time"; "does not require any specialized training data"; reaches up to ~50B-param-equivalent compute on reasoning.

**FAIRNESS / FIT.** ✅ legit architecture, but **from-scratch 800B-token pretrain, no LoRA/retrofit path** (confirms `SEARCH_INTERNALIZATION §9`). The "arbitrary test-time depth" is theoretically ideal for deterministic DAG evaluation, but there is no way to retrofit it onto Qwen-7B today.

**rw-sentence.** *Huginn confirms latent recurrent depth is only available from-scratch (3.5B/800B tokens, no LoRA path), so the theoretically-ideal "loop until done" remains unreachable for our pretrained-7B+LoRA setting.*

### Latent-CoT Survey (2505.16782v2) — the honest status line
Chen et al. **Verbatim from §7 Challenges:** "Current methods still underperform explicit CoT approaches, largely due to the instability of training." "Models trained with latent CoT techniques often struggle with novel problem structures or reasoning patterns not encountered during training." Plus: the "unobservable nature of the reasoning process … creates a significant training and alignment problem, making it difficult to apply direct supervision."

**FIT/verdict.** Reinforces the Coconut verdict: the *entire* latent-CoT family currently underperforms explicit CoT and generalizes poorly off-distribution. **Distraction** for us — and "difficult to apply direct supervision" is fatal for a setting whose whole leverage is a deterministic verifier supervising explicit structure.

### ToolkenGPT (2305.11554v4, NeurIPS'23 oral)
Hao, Liu, Wang, Hu. **LLM frozen; only toolken embeddings trained.** Verbatim: "represents each tool as a token (toolken) and learns an embedding for it, enabling tool calls in the same way as generating a regular word token"; "Once a toolken is triggered, the LLM is prompted to complete arguments." Per-domain accuracy numbers are PDF-only (HTML not parseable here) — **not quoted** (citation discipline).

**FAIRNESS/FIT.** ✅ frozen-LLM, zero-retrain tool addition, massive-tool scaling. ⚠️ doesn't use tool docs; mis-decides whether to call (Toolken+ 2410.12004 adds rerank/reject). For us: **patent-track (tool SELECTION internalization), not the thesis line** — our tool count is small so marginal utility is low. ◐

**rw-sentence.** *ToolkenGPT internalizes tool selection as frozen-LLM token embeddings with zero retraining, which fits the patent/tool-selection track but offers low marginal value to the small-tool-count thesis line.*

### PLaT (2601.21358) & OATS (2603.13426) — confirmed-real former leads
- **PLaT** (Wang, Peng, Liu; 2026-01): "Latent CoT as Planning: Decoupling Reasoning from Verbalization" — separates reasoning from verbalization. Same latent-CoT family → inherits the survey's underperformance/instability caveat; **cite as a latent-planning lead, not adopt.**
- **OATS** (Chen, Liu, Jiang, He, Liu; 2026-03, "WIP"): "Outcome-Aware Tool Selection for Semantic Routers" — embedding-interpolation tool selection *without LLM inference*, "zero-cost refinement outperforms more complex learned methods when data is sparse" on MetaTool/ToolBench. Sparse-data/zero-cost framing **echoes our P-D(-1) zero-cost-census philosophy**; relevant to the patent/tool-router track. Numbers are WIP — cite cautiously.

---

## 2. Cross-cut: candidate-generator arm vs decoding layer vs internalization; retrain vs decoding-time

| Line | diffusion alt/complement? | joins our gold-free pool as an arm? | retrain vs decoding-time |
|---|---|---|---|
| **XGrammar grammar-constrained** | complement (validity layer for *every* proposer) | no — it's a filter, not a generator | **decoding-time, zero-retrain** ✅ vLLM |
| **Planning Tokens** | orthogonal | **yes** (cheap heterogeneous LoRA arm) | retrain (LoRA, +0.001% params) |
| **SoS / Searchformer trace-distill** | complement (search-diversity as traces) | **yes** (different-distribution arm) | retrain (SFT trace distill) |
| **A3 any-order AR** | ★ conceptual *alternative* | maybe (order-diverse arm) | retrain (objective change, 7B unverified, high risk) |
| **Coconut / latent CoT / PLaT** | claimed alt, retrofit impossible | ❌ | retrain (unstable; not LoRA) |
| **Huginn recurrence** | alt | ❌ | from-scratch only |
| **Pause/filler** | orthogonal | weak | retrain incl. pretraining |
| **ToolkenGPT / OATS** | orthogonal (tool-selection) | yes (toolken arm) — patent track | retrain (embeddings, frozen LLM) |

**Insight.** Our "throw heterogeneous candidates into a gold-free pool" strategy is *isomorphic to combining several decoding/training interventions as orthogonal arms*. The diffusion proposer's job — "inject a different-distribution candidate" — can be served by (a) Planning-Token LoRA, (b) SoS-trace LoRA, (c) (high-risk) any-order LoRA, each retraining-but-vLLM-multi-LoRA-compatible. The only **zero-retrain** card is grammar-constrained decoding (a validity *floor*, not a diversity source).

---

## 3. Required closing verdicts

### (a) Does A3 any-order AR really rival diffusion? — FAIRNESS AUDIT → the "diffusion-not-required" claim is *strengthened conceptually but on weaker empirical footing than the survey implied*
**Yes, narrowly, and the comparison is tilted toward diffusion being weak rather than A3 being strong.** A3-8B edges DiffuLlama-7B/Dream-7B on TriviaQA (19.4 vs 18.5/18.3) and PIQA (78.1 vs 63.3/55.8) — a real same-family-scale win — **while trained on 2B tokens vs DiffuLlama's 65B** (an efficiency, not parity, argument). But three caveats the survey under-weighted: (1) **A3 loses heavily to the AR topline** (TriviaQA 19.4 vs 52.1; the paper admits "still underperforms the AR baseline"), so "AR rivals diffusion" means *any-order-AR rivals (weak) diffusion*, not that AR-style generation is strong here; (2) **diffusion wins story infilling on ROUGE**; (3) **no same-budget any-order-AR-vs-diffusion row exists.** 

Net effect on our "diffusion-not-required" claim: **strengthened as a framing anchor** — a single AR backbone *can* do any-order/any-subset generation and need not be inferior to diffusion — but **do not over-cite it as "AR beats diffusion."** Cite it precisely as: "any-order generation is achievable within the AR paradigm and is at least competitive with same-scale diffusion under a far smaller data budget." That is enough to justify our P-D(-1) decision to retire the diffusion proposer line in favor of a selector; it is **not** enough to claim any-order AR is a *strong* generator at our scale.

### (b) Top-2 architecture lines worth our compute
1. **Grammar-constrained decoding (XGrammar) — formalize it as an explicit architecture layer on the A2 JSON-DAG schema.** Zero-retrain, vLLM-native, already a dependency; guarantees the "right (parseable/schema-valid)" floor of every K-sample candidate → stabilizes the D-oracle denominator. Highest evidence, lowest cost.
2. **SoS/Searchformer-style trace distillation into a LoRA arm**, using our deterministic verifier to mint cost-aware search traces (failures+backtracks). Cleanest *teacher-exceeding* positive results in the set; aligns with EXPERIMENT_DESIGN §3.10. **Gate behind the 2412.04703 ceiling check** (confirm our shallow 2–7-condition trees are out of the hard-search regime first). *Runner-up worth a cheap pilot:* **Planning-Token LoRA** (+0.001% params) as a low-risk heterogeneous pool arm.

### (c) Distractions (do NOT spend compute)
- **Coconut / latent-CoT / PLaT (2412.06769, 2505.16782, 2601.21358)** — no 7B-LoRA retrofit, loses to AR on math, training-unstable, generalizes poorly off-distribution, outputs unverifiable latents (anti-thetical to our verifiable JSON-DAG). Our explicit K-sample pool already realizes "BFS-in-latent" visibly.
- **Huginn / recurrence retrofit (2502.05171)** — from-scratch only, no LoRA path.
- **Pause/filler tokens (2310.02226, 2404.15758)** — reasoning gain ~+1pt; needs pretraining-time injection; filler needs dense supervision.
- **Insertion/Levenshtein/SUNDAE re-implementation, pointer-net/graph-decoder head swap** — MT-era / non-standard heads, high re-architecting cost (concept-borrow only).
- **A3 any-order objective LoRA on 7B as an experiment** — keep as a *framing anchor only*; the 7B any-order objective swap is unverified and high-risk.

### (d) Future-dated IDs that do NOT resolve
- **2603.27905 "ATLAS-RTC … Token-Level Runtime Control"** — **does NOT resolve.** Direct abs fetch returned a template/hypothetical entry, and a targeted web search surfaced no matching paper (only unrelated 2603.xxxxx agent papers). **Remove from the bibliography / do not cite.** (By contrast, the other two formerly-unverified future-dated leads 2601.21358 and 2603.13426 DO resolve and are now confirmed real.) The survey's own §5 note that "PDF WebFetch cannot parse binaries → ToolkenGPT per-domain numbers omitted" remains correct and was respected here.

---

## 4. Net delta vs the survey
The survey's verbatim numbers all checked out. Two corrections/strengthenings: **(1) A3's fairness flag should be downgraded** from "✅ direct same-scale head-to-head" to "⚠️ tilted: loses to AR topline, efficiency-normalized 2B-vs-65B win over weak diffusion" — the "diffusion-not-required" anchor survives but must be cited precisely. **(2) Bibliography hygiene**: promote 2601.21358 (PLaT) and 2603.13426 (OATS) from "Unverified leads" to verified; **drop 2603.27905 (ATLAS-RTC) as non-resolving.**
