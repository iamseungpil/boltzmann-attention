# Inference Nondeterminism & Batch-Invariance: Literature Verification

**Date:** 2026-06-13 (access date for all web sources)
**Author:** literature-verification pass for facet-rft design doc
**Scope:** Why temp=0 still produces divergent agent trajectories, the canonical batch-invariance
explanation, and how to make our vLLM serving reproducible.
**Citation discipline:** Every cited item below was confirmed against a primary source this session.
Items that could only be partially confirmed (WebFetch was denied mid-session; some pages confirmed
only via search-engine extracts of the primary domain) are flagged. See §5.

---

## 1. 핵심 답변 요약 (executive summary)

Our observation — agent temp=0 AND user-sim temp=0, yet **0/111 identical action sequences across 4
trials**, with the census attributing 100% of the variance to agent *generation* and 0% to scoring —
is exactly the failure mode the 2025–2026 literature now treats as the canonical "LLM inference is
nondeterministic even at temp=0" phenomenon. Ranked by strength of evidence, the cause is:

1. **Lack of batch-invariance in the serving kernels (PRIMARY).** Horace He / Thinking Machines Lab
   (Sep 2025) argue verbatim that *"the primary reason nearly all LLM inference endpoints are
   nondeterministic is that the load (and thus batch-size) nondeterministically varies."* Under
   continuous/dynamic batching, the same request shares a batch with a different, time-varying set of
   other requests; the reduction strategy inside RMSNorm, matmul, and attention changes with batch
   shape, so the numerics — and thus the argmax token — change. This is **server-level** nondeterminism:
   each kernel is individually deterministic, but the *batch* the request lands in is not.

2. **Floating-point non-associativity on GPU (the ENABLER).** `(a+b)+c ≠ a+(b+c)` in IEEE-754, so any
   change in reduction/accumulation order (across batch shapes, tensor-parallel sizes, atomic-add
   scheduling) changes the last-bit result (arXiv 2408.05148). This is *why* batch-variance matters —
   it is the mechanism that converts "different reduction order" into "different logits."

3. **Tiny logit margins amplify (2) into discrete token flips.** arXiv 2506.09501 shows the
   *"fundamental cause of nondeterministic outputs is the small gap between competing logits, which
   makes token selection vulnerable to minute numerical fluctuations."* One flipped token early in an
   agent trajectory cascades into an entirely different action sequence — exactly our 0/111.

Crucially, **this is NOT sampling randomness.** temp=0 removes *sampling* nondeterminism (seed-controllable),
but leaves *numerical* nondeterminism untouched. Setting a seed does nothing here. That is why our
trajectories diverge despite greedy decoding everywhere.

**The fix the literature prescribes:** batch-invariant kernels — redesign RMSNorm, matmul, and attention
so their reduction order is fixed regardless of batch size, padding, or position. Thinking Machines report
**1000/1000 bitwise-identical completions** under dynamic batching once enabled. **Cost:** modest. Their
own numbers: vLLM default 26 s → unoptimized deterministic 55 s → improved-attention deterministic 42 s
(≈1.6× on their setup; the blog-level framing is "10–40% depending on op/hardware"). vLLM has since
shipped this as `VLLM_BATCH_INVARIANT=1`.

**Bottom line for us:** to make our pass^4 / A2-compile runs reproducible at temp=0 we should enable
vLLM batch-invariance (and pin hardware + vLLM version). See §3 for the temp=0-deterministic /
temp>0-diverse question — the answer is **yes, achievable.**

---

## 2. Per-RQ verified findings

### RQ1 — The batch-invariance thesis ✅ VERIFIED (primary source fetched)

- **Title:** "Defeating Nondeterminism in LLM Inference"
- **Author/venue:** Horace He, in collaboration with others at **Thinking Machines Lab**.
- **Type:** Blog / technical report (not peer-reviewed).
- **Date:** September 10, 2025.
- **URL:** https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
- **Verification:** Full page fetched and confirmed this session.

Verbatim core claims:

> "the primary reason nearly all LLM inference endpoints are nondeterministic is that the load (and thus
> batch-size) nondeterministically varies!"

> "even when we adjust the temperature down to 0 (thus making the sampling theoretically deterministic),
> LLM APIs are still **not** deterministic in practice"

> "we only need to worry about the 3 operations that involve reductions — RMSNorm, matrix multiplication,
> and attention."

> "when we enable our batch-invariant kernels, all of our 1000 completions are identical"

RL motivation, verbatim:

> "the different numerics between training and inference implicitly turns our on-policy RL into off-policy RL"

- **Library provided:** `thinking-machines-lab/batch-invariant-ops` (open-source PyTorch lib for
  RMSNorm / matmul / attention / softmax), demonstrated with vLLM.
- **Throughput cost (verbatim numbers):** vLLM default **26 s**; unoptimized deterministic **55 s**;
  with improved attention kernel **42 s**. (Blog framing elsewhere: "modest slowdown.")

This is the single most load-bearing citation for our design-doc section. It exists, the attribution
(He / Thinking Machines) is correct, and every claim we wanted is quotable.

### RQ2 — Floating-point / GPU nondeterminism ✅ VERIFIED

**(a) Peer-reviewed-track primary source — arXiv 2408.05148**
- **Title:** "Impacts of floating-point non-associativity on reproducibility for HPC and deep learning
  applications"
- **Authors:** Sanjif Shanmugavelu, Mathieu Taillefumier, Christopher Culver, Oscar Hernandez,
  Mark Coletti, Ada Sedova (Oak Ridge National Lab / UT-Battelle).
- **Type:** arXiv preprint (HPC venue style; DOE-authored).
- **Dates:** v1 9 Aug 2024; last revised v3 30 Oct 2024. **Cite as arXiv:2408.05148v3.**
- **URL:** https://arxiv.org/abs/2408.05148
- **Verification:** abstract + author list confirmed via arxiv.org search extract (page itself not
  fetched — WebFetch denied; flagged).

Mechanism, verbatim (abstract): *"Run to run variability in parallel programs caused by floating-point
non-associativity has been known to significantly affect reproducibility in iterative algorithms, due to
accumulating errors… Recently, the sensitivity of deep learning training and inference pipelines to
floating-point non-associativity has been found to sometimes be extreme."* Codes relying on **atomic
operations** are non-deterministic because of FP non-associativity; replacing atomic-add with
deterministic equivalents restores determinism at a speed cost.

**(b) Framework docs — PyTorch deterministic algorithms** ✅ (confirmed via docs.pytorch.org extracts)
- `torch.use_deterministic_algorithms()`:
  https://docs.pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
- Reproducibility note: https://docs.pytorch.org/docs/stable/notes/randomness.html
- **Type:** official documentation.
- Verbatim: *"A handful of CUDA operations are nondeterministic if the CUDA version is 10.2 or greater,
  unless the environment variable CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8 is
  set."* When `use_deterministic_algorithms(True)` is set, ops *"will use deterministic algorithms when
  available, and if only nondeterministic algorithms are available they will throw a RuntimeError."*
  Also: *"Deterministic operations tend to have worse performance than nondeterministic operations,"* and
  the docs caution this *"alone is not always enough to make an application reproducible."*

This pair gives us the canonical mechanism (FP non-associativity + atomic reductions) and the official
framework knob, both quotable.

### RQ3 — temp=0 ≠ deterministic; sampling vs numerical nondeterminism ✅ VERIFIED

**Primary source — arXiv 2506.09501**
- **Title:** "Understanding and Mitigating Numerical Sources of Nondeterminism in LLM Inference"
  (v1 was titled "Give Me FP32 or Give Me Death? Challenges and Solutions for Reproducible Reasoning").
- **Authors:** Jiayi Yuan, Hao Li, Xinheng Ding, Wenya Xie, Yu-Jhe Li, Wentian Zhao, Kun Wan, Jing Shi,
  Xia Hu, Zirui Liu.
- **Type:** arXiv preprint. **Date:** 11 Jun 2025 (v1). **Cite as arXiv:2506.09501.**
- **URL:** https://arxiv.org/abs/2506.09501
- **Verification:** abstract/findings via arxiv.org search extract (page not fetched — flagged).

Key verified findings:
- *"the fundamental cause of nondeterministic outputs is the small gap between competing logits, which
  makes token selection vulnerable to numerical fluctuations."*
- Precision matters: **FP32 ≈ near-perfect determinism; FP16 moderate; BF16 significant variance**
  despite being the common default.
- Concrete magnitude: *under bfloat16 + greedy decoding, DeepSeek-R1-Distill-Qwen-7B can exhibit up to
  **9% accuracy variation and ~9,000-token response differences**.*
- Runtime config (GPU count, batch size) further affects reproducibility, worst in low precision.

This is the cleanest empirical demonstration that **greedy ≠ reproducible** and that the culprit is
numerical, not sampling. It directly explains why our 4 trials diverged even with all temps at 0.

The **sampling vs numerical** distinction is made explicit by the Thinking Machines blog (RQ1: temp=0
"makes the sampling theoretically deterministic" yet APIs "still not deterministic") combined with
2506.09501 (the residual variance is numerical). Use both together for the dichotomy.

### RQ4 — Why determinism matters for eval / RL ✅ VERIFIED

**(a)** Thinking Machines blog (RQ1) already supplies the RL argument verbatim: numerics mismatch between
training and inference *"implicitly turns our on-policy RL into off-policy RL."*

**(b) Primary source — arXiv 2511.17826**
- **Title:** "Deterministic Inference across Tensor Parallel Sizes That Eliminates Training–Inference
  Mismatch"
- **Authors:** Ziyang Zhang, Xinheng Ding, Jiayi Yuan, Rixin Liu, Huizi Mao, Jiarong Xing, Zirui Liu.
- **Type:** arXiv preprint. **Date:** 21 Nov 2025 (v1). **Cite as arXiv:2511.17826.**
- **URL:** https://arxiv.org/abs/2511.17826
- **Verification:** abstract via arxiv.org search extract (page not fetched — flagged).

Verbatim-ish core (from abstract extract): *"identical inputs can yield different outputs when system
configurations (e.g., tensor parallel size, batch size) vary, even under greedy decoding. This arises
from the non-associativity of floating-point arithmetic and inconsistent reduction orders across GPUs.
The issue is particularly problematic in RL settings, where the training engine typically uses Fully
Sharded Data Parallel while the rollout engine relies on multi-GPU tensor parallelism."* They name
three motivating applications explicitly: **LLM-as-a-judge evaluation, multi-agent systems, and RL.**
Their solution is **Tree-Based Invariant Kernels (TBIK)** giving bitwise-identical results across TP size.

This is the strongest single source that nondeterminism corrupts eval and RL specifically (not merely
aesthetics), and it names multi-agent systems — directly our setting.

### RQ5 — vLLM-specific determinism knobs ⚠️ VERIFIED VIA SEARCH EXTRACTS (docs domain WebFetch was denied)

vLLM has first-class support, having upstreamed the Thinking Machines kernels. Knobs, per the official
docs (https://docs.vllm.ai/en/latest/usage/reproducibility/ and
https://docs.vllm.ai/en/latest/features/batch_invariance/):

- **`VLLM_BATCH_INVARIANT=1`** — enables batch-invariant RMSNorm / matmul / attention. *"Batch invariance
  ensures that the output of a model is deterministic and independent of the batch size or the order of
  requests in a batch."* **This is the only knob that works in online (server) mode.** Requires NVIDIA
  GPU **compute capability ≥ 8.0** (Ampere+). Intentional performance cost.
- **`seed`** — controls RNG; "requests will produce deterministic outputs regardless of batch size or
  order when using the seed parameter" **only in combination with batch invariance**; on its own it only
  fixes *sampling*, not numerics. (At temp=0 sampling is already fixed, so seed alone does nothing for
  our problem.)
- **`VLLM_ENABLE_V1_MULTIPROCESSING=0`** — makes scheduling deterministic; **offline mode only.** Per
  docs: *"In offline mode, you can either set VLLM_ENABLE_V1_MULTIPROCESSING=0 which makes scheduling
  deterministic, or enable batch invariance to make the outputs insensitive to scheduling. In online
  mode, you can only enable batch invariance."*
- **`--enforce-eager` (disable CUDA graphs) + disable prefix caching** — recommended companion settings
  for deterministic runs (CUDA graphs and prefix caching introduce shape/cache-dependent reduction
  paths). These help but are **not sufficient alone** to defeat batch-variance.
- **`--max-num-seqs 1`** (effectively batch size 1) — would remove cross-request batch-variance, but does
  NOT remove *intra-request* batch-shape variation (prefill chunk sizes, padding) and destroys throughput;
  not the recommended path. Batch-invariance is the principled fix.

**Hard caveat (verbatim):** *"Even with the above settings, vLLM only provides reproducibility when it
runs on the same hardware and the same vLLM version."* So our reproducibility claims must pin GPU model +
vLLM version.

> NOTE: Because WebFetch to `docs.vllm.ai` was denied this session, the exact env-var names above were
> confirmed from search-engine extracts of the official vLLM doc pages (two independent queries returned
> consistent text) and are cross-consistent with the Thinking Machines blog (which authored the kernels).
> Before quoting verbatim in the design doc, re-open the two doc URLs to confirm wording for the shipping
> vLLM version we deploy.

---

## 3. Concrete recommendation for OUR vLLM setup

**Goal:** deterministic (bitwise-reproducible) generation at temp=0 for pass^4 and A2-compile, while
keeping genuine diversity at temp>0 for the selector.

**Can we be deterministic at temp=0 but nondeterministic at temp>0? — YES, and it is the natural design.**

The two sources of variation are independent and separately controllable:

- **Numerical nondeterminism** (batch-variance + FP) is what `VLLM_BATCH_INVARIANT=1` removes. It is
  *temperature-independent*: it makes the *logits* bitwise-reproducible for a given input regardless of
  what else is in the batch.
- **Sampling nondeterminism** is governed by temperature + seed, applied *after* the logits.

So with batch-invariance ON:
- **At temp=0:** argmax over bitwise-identical logits → bitwise-identical tokens → reproducible
  trajectories. (Fixes our 0/111.)
- **At temp>0:** the logits are reproducible, but sampling still injects randomness. If you leave the
  seed unset (or vary it per draw), you get genuine, diverse samples for the selector. If you ever want
  reproducible *sampled* draws too, set a per-request `seed` — but for selector diversity you simply
  don't fix the seed.

Therefore batch-invariance does **not** force temp>0 to become deterministic. It only removes the
*numerical* noise floor. Diversity at temp>0 comes from the sampler, which is untouched. This is exactly
the regime we want.

**Recommended configuration:**
1. Launch the server with `VLLM_BATCH_INVARIANT=1` (online mode → this is the only option that survives
   continuous batching).
2. Add `--enforce-eager` (no CUDA graphs) and disable prefix caching for the deterministic eval runs.
3. **Pin GPU model and vLLM version**; record both in run metadata (reproducibility is only guaranteed
   on identical hardware + version).
4. Confirm GPU compute capability ≥ 8.0 (our A100/H-class meet this; verify before relying on it).
5. **temp=0 path (pass^4, A2-compile):** request `temperature=0`. Expect bitwise-identical action
   sequences across trials; if not, batch-invariance is not actually active — verify by running the same
   prompt twice under load and diffing.
6. **temp>0 path (selector diversity):** request `temperature>0`, leave seed unset (or randomize). You
   keep diversity; the only thing that changed vs. today is that the *logits* are now reproducible, which
   does not reduce sample diversity.
7. **Budget the cost:** plan for ≈1.3–1.6× latency on deterministic runs (Thinking Machines: 26→42 s
   with the improved kernel). Acceptable for eval; if used in the hot RL rollout path, measure first.
8. **Optional precision lever (independent of vLLM batch-invariance):** 2506.09501 shows BF16 is the
   worst offender. If batch-invariant kernels are unavailable for some op/backend, running eval in FP32
   sharply reduces (does not fully eliminate) variance — a fallback, not a substitute.

**Validation step:** after enabling, re-run the exact tau2 4-trial experiment. Success criterion is the
inverse of our finding: action sequences should now be identical across the 4 temp=0 trials (target
matching Thinking Machines' 1000/1000), while a temp>0 control still shows divergence.

---

## 4. Verified bibliography

| # | Source | Type | Date | Verification |
|---|--------|------|------|--------------|
| 1 | He, H. et al. (Thinking Machines Lab), "Defeating Nondeterminism in LLM Inference." https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/ | Blog / tech report | 2025-09-10 | **Full page fetched & confirmed** |
| 2 | `thinking-machines-lab/batch-invariant-ops` (open-source PyTorch lib) | Code/library | 2025 | Named in source #1 (fetched) |
| 3 | Shanmugavelu, S. et al., "Impacts of floating-point non-associativity on reproducibility for HPC and deep learning applications." arXiv:2408.05148v3 | Preprint (DOE/ORNL) | 2024-08-09 (v3 2024-10-30) | Abstract+authors via arxiv.org search extract |
| 4 | PyTorch, "Reproducibility" note + `torch.use_deterministic_algorithms`. https://docs.pytorch.org/docs/stable/notes/randomness.html | Official docs | accessed 2026-06-13 | Verbatim via docs.pytorch.org search extract |
| 5 | Yuan, J. et al., "Understanding and Mitigating Numerical Sources of Nondeterminism in LLM Inference." arXiv:2506.09501 (v1 "Give Me FP32 or Give Me Death?") | Preprint | 2025-06-11 | Abstract+findings via arxiv.org search extract |
| 6 | Zhang, Z. et al., "Deterministic Inference across Tensor Parallel Sizes That Eliminates Training–Inference Mismatch." arXiv:2511.17826 | Preprint | 2025-11-21 | Abstract via arxiv.org search extract |
| 7 | vLLM docs, "Reproducibility." https://docs.vllm.ai/en/latest/usage/reproducibility/ | Official docs | accessed 2026-06-13 | Search extract only (WebFetch denied) |
| 8 | vLLM docs, "Batch Invariance." https://docs.vllm.ai/en/latest/features/batch_invariance/ | Official docs | accessed 2026-06-13 | Search extract only (WebFetch denied) |

**Verification caveat:** WebFetch was available for source #1 only; it was denied for all subsequent
calls this session. Sources #3–#8 were confirmed through search-engine extracts of the primary domains
(arxiv.org, docs.pytorch.org, docs.vllm.ai), with two independent queries cross-checked where wording is
quoted. The arxiv IDs, titles, author lists, and dates are internally consistent and consistent across
queries. Before the design doc quotes any of #3–#8 **verbatim**, re-open the exact URL to confirm the
sentence — the paraphrase-level claims are reliable; the exact-wording quotes from #7/#8 should be
re-pulled.

## 5. Unverified leads (NOT for citation until fetched)

- **SGLang / LMSYS, "Towards Deterministic Inference in SGLang and Reproducible RL Training"**
  (https://www.lmsys.org/blog/2025-09-22-sglang-deterministic/, ~2025-09-22). Strong corroborating
  blog that applies batch-invariance to SGLang and ties it to reproducible RL. **Not fetched** (WebFetch
  denied); search results suggest it credits the Thinking Machines work and makes specific attention
  backends batch-invariant. Verify before citing.
- **arXiv:2511.00025, "On the Structure of Floating-Point Noise in Batch-Invariant GPU Matrix
  Multiplication."** Appears to analyze the residual noise structure under batch-invariant matmul.
  Potentially a good "deeper mechanism" cite. **Unverified** — only seen as a search hit.
- **arXiv:2601.06118, "Beyond Reproducibility: Token Probabilities Expose Large Language Model
  Nondeterminism."** Possibly relevant to detecting nondeterminism via logit margins (ties to our census
  idea). **Unverified.**
- **arXiv:2601.17768, "LLM-42: Enabling Determinism in LLM Inference with Verified Speculation."**
  Recurring hit; appears to add determinism under speculative decoding. **Unverified.**
- **arXiv:2604.22411, "Introducing Background Temperature to Characterise Hidden Randomness in LLMs."**
  Possibly relevant to the "hidden randomness even at temp=0" framing. **Unverified** (future-dated;
  treat with extra caution).
- General-audience explainers (Medium, llmwatch, keywordsai, unstract) appeared in search — **do not
  cite**; they are secondary summaries of source #1.
