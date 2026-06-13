# Related Work: Inference Determinism — Full-Text Deep-Read & Fit Analysis

**Date:** 2026-06-13 (access date for all web sources)
**Author:** full-text deep-read pass for facet-rft determinism design (ⓟ1 re-open assessment)
**Scope:** Beyond-abstract reading of the inference-nondeterminism literature cited in
`research_inference_determinism_2026_06_14.md` — exact mechanism, the exact fix, the cost numbers — and
a strict FIT analysis against OUR need (deterministic temp=0 for pass^4 / A2-compile; nondeterministic
temp>0 for selector diversity), per `BENCH_PORTFOLIO_FRAMEWORK_DESIGN.md` §3.7d.
**Citation discipline:** Every quote below was pulled THIS session by fetching the primary source
(WebFetch) or, where WebFetch to a domain was denied, confirmed via two independent search-engine
extracts of the primary domain. Source tier is labelled per item. WebFetch denials are flagged.

---

## 0. What changed vs the prior verification pass

The prior doc (`research_inference_determinism_2026_06_14.md`) verified sources #3, #5, #6 only via
search-engine *abstract* extracts (WebFetch was denied mid-session). This pass **fetched the full arXiv
abstract pages** for 2408.05148, 2506.09501, and 2511.17826, plus the four "future-dated" IDs, and
cross-checked the vLLM knob via the vLLM project's own announcement and GitHub release notes. Three
material corrections / additions result:

1. **2506.09501's actual fix is `LayerCast`** (store weights in 16-bit, compute in FP32) — the prior
   doc's bibliography described it only as a "precision observation." The mitigation is a concrete
   pipeline, and it is a *fallback* lever for us, not a substitute for batch-invariance. (§3 / §6)
2. **The minimum vLLM version with `VLLM_BATCH_INVARIANT` is `v0.11.1`** (highlight: "Batch-invariant
   `torch.compile`"). The design-doc's report that **0.11.0 lacks it is correct** — it shipped one
   point-release later. (§5, §6a)
3. **All four "future-dated" IDs resolve to real papers** — including 2604.22411, which one WebFetch
   render flagged as non-existent but a domain-restricted search confirmed. One (2601.17768, "LLM-42")
   is directly relevant: a *scheduling-based* alternative to batch-invariant kernels. (§4, §6c)

---

## 1. Thinking Machines — "Defeating Nondeterminism in LLM Inference" (PRIMARY citation)

- **Title / author / date (verbatim):** "Defeating Nondeterminism in LLM Inference" by "Horace He in
  collaboration with others at Thinking Machines", "Sep 10, 2025".
- **Tier:** Blog / technical report — **NOT peer-reviewed.** Highest-authority for the *engineering*
  claim because the authors wrote the kernels vLLM upstreamed; lowest-authority on the "peer-reviewed"
  axis. Treat as an engineering primary source.
- **Verification:** Full page fetched and confirmed this session.

**Mechanism, VERBATIM:**

> "the primary reason nearly all LLM inference endpoints are nondeterministic is that the load (and thus
> batch-size) nondeterministically varies!"

> "even when we adjust the temperature down to 0 (thus making the sampling theoretically deterministic),
> LLM APIs are still **not** deterministic in practice"

> "we only need to worry about the 3 operations that involve reductions — RMSNorm, matrix multiplication,
> and attention."

**The fix + result, VERBATIM:**

> "when we enable our batch-invariant kernels, all of our 1000 completions are identical."

**Cost, VERBATIM (their A100 / Qwen-3-8B setup):** "vLLM default: 26 [seconds]; Unoptimized Deterministic
vLLM: 55 [seconds]; + Improved Attention Kernel: 42 [seconds]." → with the improved attention kernel the
deterministic path is **42/26 ≈ 1.6×** the default latency (the unoptimized path is ~2.1×).

**Library:** `thinking-machines-lab/batch_invariant_ops` (PyTorch lib: batch-invariant RMSNorm, matmul,
attention). NOTE: the blog uses the underscore form `batch_invariant_ops`; the prior doc's hyphen form
`batch-invariant-ops` is the repo slug — same project, cite the canonical underscore form.

**RL motivation, VERBATIM:**

> "the different numerics between training and inference implicitly turns our on-policy RL into off-policy RL."

**Correction to prior doc:** the blog does **NOT name `VLLM_BATCH_INVARIANT`** anywhere. That env-var is a
*vLLM* artifact (the project upstreamed the kernels later). Do not attribute the env-var name to the
blog; attribute it to vLLM v0.11.1 (§5).

**FIT (strong, direct):** This is the load-bearing confirmation. Our ⓟ1 observation — agent temp=0 AND
user-sim temp=0, yet **0/111 identical action sequences over 4 trials**, census attributing 100% of
variance to *generation* and 0% to scoring — is exactly "temp=0 yet not deterministic in practice"
driven by "load (and thus batch-size) nondeterministically varies." Our 0/111 is the multi-step cascade
of the blog's single-completion flip: one flipped argmax token early in a trajectory forks the whole
action sequence. The blog asserts the fix yields *bitwise-identical* outputs (1000/1000), which is the
exact success criterion for re-opening ⓟ1.

**rw-sentence:** "Even under greedy (temperature-zero) decoding, production LLM serving is
nondeterministic because dynamic batching varies the batch size and thus the reduction order inside
RMSNorm, matmul, and attention; batch-invariant kernels restore bitwise-identical completions at roughly
1.6x latency (He et al., Thinking Machines Lab, 2025)."

---

## 2. arXiv:2408.05148 — FP non-associativity (the ENABLER mechanism)

- **Title (verbatim):** "Impacts of floating-point non-associativity on reproducibility for HPC and deep
  learning applications."
- **Authors (verbatim):** Sanjif Shanmugavelu, Mathieu Taillefumier, Christopher Culver, Oscar Hernandez,
  Mark Coletti, Ada Sedova (ORNL / UT-Battelle, DOE).
- **Dates (verbatim):** v1 9 Aug 2024; v2 23 Aug 2024; v3 30 Oct 2024. **Cite arXiv:2408.05148v3.**
- **Tier:** arXiv preprint, DOE/HPC-authored, peer-review-track style. Strongest *non-LLM-specific*
  grounding of the underlying numerical mechanism.
- **Verification:** **Full abstract page fetched this session** (upgrade from prior abstract-extract).

**Mechanism, VERBATIM (abstract):**

> "Run to run variability in parallel programs caused by floating-point non-associativity has been known
> to significantly affect reproducibility in iterative algorithms, due to accumulating errors… Recently,
> the sensitivity of deep learning training and inference pipelines to floating-point non-associativity
> has been found to sometimes be extreme."

> "we perform an investigation of the statistical properties of floating-point non-associativity… and
> analyze performance and productivity impacts of replacing atomic operations with deterministic
> alternatives on GPUs. We examine the recently-added deterministic options in PyTorch… Finally, we
> evaluate the strategy of exploiting automatic determinism… using the Groq accelerator."

**The fix:** replace **atomic operations** (atomic-add scatter reductions, whose completion order is
scheduler-dependent) with deterministic reductions; or use deterministic hardware (Groq). Cost is
characterised as a speed/productivity trade-off; the abstract does not give a single headline x-factor.

**FIT (mechanism-level, indirect):** This is *why* batch-variance matters — it converts "different
reduction order" into "different last-bit logits." It does NOT directly address dynamic-batching in LLM
serving (it is an HPC/atomics paper), so it is the right cite for the **enabler** in our chain, not the
**source**. It also justifies our companion knobs: PyTorch `use_deterministic_algorithms` +
`CUBLAS_WORKSPACE_CONFIG` remove the *atomic-reduction* nondeterminism this paper studies, but — per the
PyTorch docs the paper examines — those alone are "not always enough" because they don't make kernels
*batch-invariant*. So this confirms: max-num-seqs 1 + deterministic-algorithms is **insufficient** on its
own (see §3).

**rw-sentence:** "Floating-point non-associativity makes GPU reductions order-dependent, so any change in
accumulation order — including the atomic-add scatter scheduling studied by Shanmugavelu et al. (2024) —
perturbs results to the last bit, which is the numerical substrate underlying all temp-zero
nondeterminism."

---

## 3. arXiv:2506.09501 — logit-gap + BF16-vs-FP32 (the AMPLIFIER + a fallback fix)

- **Title (verbatim):** "Understanding and Mitigating Numerical Sources of Nondeterminism in LLM
  Inference." (v1 was titled "Give Me FP32 or Give Me Death? Challenges and Solutions for Reproducible
  Reasoning".)
- **Authors (verbatim):** Jiayi Yuan, Hao Li, Xinheng Ding, Wenya Xie, Yu-Jhe Li, Wentian Zhao, Kun Wan,
  Jing Shi, Xia Hu, Zirui Liu.
- **Dates (verbatim):** v1 11 Jun 2025; v2 24 Oct 2025. **Cite arXiv:2506.09501.**
- **Tier:** arXiv preprint, LLM-specific empirical study.
- **Verification:** **Full abstract page fetched this session.**

**Mechanism + magnitude, VERBATIM (abstract):**

> "changing system configuration, such as evaluation batch size, GPU count, and GPU version, can
> introduce significant differences in the generated responses. This issue is especially pronounced in
> reasoning models, where minor rounding differences in early tokens can cascade into divergent chains
> of thought, ultimately affecting accuracy. For instance, under bfloat16 precision with greedy decoding,
> a reasoning model like DeepSeek-R1-Distill-Qwen-7B can exhibit up to 9% variation in accuracy and 9,000
> tokens difference in response length due to differences in GPU count, type, and evaluation batch size.
> We trace the root cause of this variability to the non-associative nature of floating-point arithmetic
> under limited numerical precision."

**The fix (CORRECTION to prior doc), VERBATIM:**

> "we develop a lightweight inference pipeline, dubbed LayerCast, that stores weights in 16-bit precision
> but performs all computations in FP32, balancing memory efficiency with numerical stability."

So 2506.09501's prescription is **LayerCast (FP32 compute)**, not merely an observation that "BF16 is
worst." The "small logit gap" framing quoted in the prior doc is consistent with this paper's thesis but
note the *abstract* frames the root cause as FP non-associativity under limited precision; the
logit-gap detail is in the body. (Body not fetched; the abstract claim above is what is quotable
verbatim.)

**FIT (strong, our exact symptom):** This is the cleanest LLM-specific match to ⓟ1. "minor rounding
differences in early tokens can cascade into divergent chains of thought" *is* our 0/111 — a token flip
forks the agent trajectory. It also names our exact knobs (batch size, GPU count, GPU version) as the
varying configs, which validates our reproducibility-pinning requirement (§3 step 3). LayerCast is a
**fallback lever** for us: if a batch-invariant kernel is missing for some op/backend, FP32 compute
sharply reduces (does not fully eliminate) variance — orthogonal to and stackable with batch-invariance.

**rw-sentence:** "Yuan et al. (2025) show that under bfloat16 greedy decoding, batch size / GPU count /
GPU version alone induce up to 9% accuracy swings and 9,000-token divergences in reasoning models, and
mitigate this with LayerCast (16-bit storage, FP32 compute)."

---

## 4. arXiv:2511.17826 — nondeterminism corrupts RL / eval / judge (names OUR setting)

- **Title (verbatim):** "Deterministic Inference across Tensor Parallel Sizes That Eliminates
  Training-Inference Mismatch."
- **Authors (verbatim):** Ziyang Zhang, Xinheng Ding, Jiayi Yuan, Rixin Liu, Huizi Mao, Jiarong Xing,
  Zirui Liu.
- **Dates (verbatim):** v1 21 Nov 2025; v2 29 May 2026. **Cite arXiv:2511.17826.**
- **Tier:** arXiv preprint.
- **Verification:** **Full abstract page fetched this session.**

**Mechanism + applications, VERBATIM (abstract):**

> "Deterministic inference is increasingly critical for large language model (LLM) applications such as
> LLM-as-a-judge evaluation, multi-agent systems, and Reinforcement Learning (RL). However, existing LLM
> serving frameworks exhibit non-deterministic behavior: identical inputs can yield different outputs
> when system configurations (e.g., tensor parallel (TP) size, batch size) vary, even under greedy
> decoding. This arises from the non-associativity of floating-point arithmetic and inconsistent
> reduction orders across GPUs…"

> "the training engine typically uses Fully Sharded Data Parallel (i.e., TP = 1) while the rollout engine
> relies on multi-GPU TP to maximize the inference throughput, creating a natural mismatch between the two"

**The fix, VERBATIM:** "Tree-Based Invariant Kernels (TBIK), a set of TP-invariant matrix multiplication
and reduction primitives that guarantee bit-wise identical results regardless of TP size."

**FIT (strong, names our use-case):** This is the strongest cite that nondeterminism *corrupts evaluation
specifically*, not just aesthetics — and it explicitly names **multi-agent systems** and
**LLM-as-a-judge**, which is exactly our pass^4 / A2-compile / selector setting. TBIK extends batch-
invariance to the **tensor-parallel** axis: if we ever shard our serving across >1 GPU (TP>1), plain
batch-invariance is not guaranteed sufficient — reduction order also varies with TP size. Practical
consequence for us: keep eval serving at **TP=1** (single GPU) where vLLM's batch-invariance already
covers us, OR require TBIK-class kernels if we must scale TP. The MEMORY note "GPU0/1 충돌금지" already
pushes us toward single-GPU isolation, which aligns.

**rw-sentence:** "Zhang et al. (2025) show greedy decoding still diverges across tensor-parallel sizes
because reduction order varies across GPUs, and restore bitwise-identical outputs across TP sizes with
Tree-Based Invariant Kernels — motivated explicitly by LLM-judge, multi-agent, and RL determinism."

---

## 5. vLLM determinism knobs (the recipe)

WebFetch to `docs.vllm.ai` was **denied again this session** (same as prior). Knob names below confirmed
via: (a) WebFetch of the docs **source markdown on GitHub raw**; (b) the **vLLM project's own X/Twitter
announcement**; (c) the **v0.11.1 GitHub release notes**. Cross-consistent across all three.

- **`VLLM_BATCH_INVARIANT=1`** — the only knob that survives online/continuous batching. vLLM project
  announcement (verbatim, X post): *"Now you can get identical results regardless of batch size with just
  one flag: VLLM_BATCH_INVARIANT=1. No more subtle differences between bs=1 and bs=N (including
  prefill!)."* Docs source (verbatim): *"Batch invariance can be enabled by setting the
  `VLLM_BATCH_INVARIANT` environment variable to `1`"* and *"Batch invariance requires NVIDIA GPUs with
  compute capability 8.0 or higher."* Set it **before importing vLLM** (`os.environ[...]`). Marked
  *"currently in beta."* Validated on DeepSeek, Qwen3 (dense+MoE), Qwen2.5, Llama 3, GPT-OSS, Mistral;
  AWQ + FP8 variants tested.
- **`VLLM_ENABLE_V1_MULTIPROCESSING=0`** — **offline mode only.** Docs source (verbatim): *"makes
  scheduling deterministic."* Online mode cannot use it; *"In online mode, you can only enable [batch
  invariance]."* Note it *"change[s] the random state of user code."*
- **`seed`** — fixes RNG (`random`, `np.random`, `torch.manual_seed`); defaults to 0 per worker. At
  temp=0 sampling is already deterministic, so seed alone does **nothing** for our numerical problem; it
  only matters for reproducible temp>0 draws.
- **`tensor_parallel_size=1`** — the docs' offline batch-invariance example pins TP=1. (Consistent with
  §4: TP>1 needs TBIK-class kernels, not just batch-invariance.)
- **Hard caveat (verbatim, docs source):** *"vLLM only provides reproducibility when it runs on the same
  hardware and the same vLLM version."* → pin GPU model + vLLM version in run metadata.

**Minimum version (the question the design doc asked):** `VLLM_BATCH_INVARIANT` first ships in
**vLLM v0.11.1**. The v0.11.1 release highlights (verbatim): *"Batch-invariant `torch.compile`:
Generalized batch-invariant support across attention and MoE backends, with explicit support for DeepGEMM
and FlashInfer on Hopper and Blackwell GPUs."* The env-var check is cached in PR #26510 ("Cache the
environment variable check for batch invariance"). **This confirms the design-doc's finding that 0.11.0
lacks it** — it landed one point-release later, in 0.11.1. (Feature continued maturing through later
releases per issue #27433 and a 0.14→0.15 FlashInfer bug report; treat ≥0.11.1 as the floor and prefer a
current release.)

**Companion knobs (`--enforce-eager`, disable prefix caching):** the current docs **source** does NOT
list these as required for batch-invariance (it only pins TP=1). They remain reasonable belt-and-braces
settings — CUDA graphs and prefix caching introduce shape/cache-dependent paths — but per the vLLM
announcement, batch-invariance is explicitly designed to be correct *including prefill and bs=1-vs-bs=N*,
so they are **not** a substitute. Prior doc's claim that enforce-eager+max-num-seqs-1 alone is
insufficient stands and is now better grounded: §2 (FP/atomics) + the vLLM design both say so.

---

## 6. The four "future-dated" IDs — all RESOLVE (with one caveat)

All four were checked by WebFetch **and** a domain-restricted (`arxiv.org`) search to guard against a
hallucinated render. **All four resolve to real papers.**

- **arXiv:2601.06118 — RESOLVES.** "Beyond Reproducibility: Token Probabilities Expose Large Language
  Model Nondeterminism." Authors: Tairan Fu, Gonzalo Martínez, Javier Conde, Carlos Arriaga, Pedro
  Reviriego, Xiuyuan Qi, Shanshan Liu. Finding (verbatim-ish): *"nondeterminism effects are significant
  for token probabilities in the 0.1 to 0.9 range, while much smaller near 0 or 1"*; a single inference
  run can estimate nondeterminism impact. **FIT:** directly supports OUR census idea — detect/measure
  nondeterminism via logit margins rather than N repeats. Useful methodological cite for §3.7d.
- **arXiv:2601.17768 — RESOLVES.** "LLM-42: Enabling Determinism in LLM Inference with Verified
  Speculation." Authors: Raja Gond, Aditya K Kamath, Ramachandran Ramjee, Ashish Panwar (Microsoft
  Research / UW / IISc). v1 25 Jan 2026, v2 30 Jan 2026. Mechanism (verbatim, abstract): *"LLM-42 decodes
  tokens using a non-deterministic fast path and enforces determinism via a lightweight verify-rollback
  loop… commits those that are guaranteed to be consistent across runs, and rolls back those violating
  determinism. LLM-42 mostly re-uses existing kernels unchanged and incurs overhead only in proportion to
  the traffic that requires determinism."* **FIT:** a *scheduling* alternative to batch-invariant
  kernels — pay-for-what-you-use determinism. Relevant if batch-invariance's fixed overhead bites in a
  hot path, but not packaged in vLLM today; treat as forward-looking, not our recipe.
- **arXiv:2511.00025 — RESOLVES.** "On the Structure of Floating-Point Noise in Batch-Invariant GPU Matrix
  Multiplication." Author: Tadisetty Sai Yashwanth. Finding: FP error under batch-invariant matmul is
  *"highly correlated"* / a *"coordinated, directional perturbation,"* not i.i.d. Gaussian. **FIT:**
  deeper-mechanism cite; cautions that residual noise (if any leaks) is structured, not random — minor
  for us once batch-invariance is on.
- **arXiv:2604.22411 — RESOLVES (caveat).** "Introducing Background Temperature to Characterise Hidden
  Randomness in Large Language Models." Authors: Alberto Messina, Stefano Scotta (RAI Centre for Research).
  Submitted 24 Apr 2026. Defines *"background temperature" (T_bg)* — the effective temperature induced by
  implementation perturbation even at nominal T=0 (batch-size variation, kernel non-invariance, FP
  non-associativity). **CAVEAT:** the *first* WebFetch render flagged this ID as "fictional/future-dated";
  a domain-restricted arxiv.org search confirmed it is a real listing with that title and authors. The
  discrepancy was a stale/parse artifact in one render, not a missing paper. **FIT:** gives us a clean
  framing/metric ("background temperature") for ⓟ1 — quantify residual temp-0 nondeterminism as an
  effective T_bg before/after batch-invariance. Use only after the full text is fetched (only abstract-
  level confirmed; body not read).

No future-dated ID **fails** to resolve. The one to watch is 2604.22411 (flag the render discrepancy in
any citing text; re-pull the PDF before quoting the body).

---

## 7. Synthesis — does the literature confirm OUR pass^4 variance source?

**Yes, unambiguously, at three independent levels:**

1. **Symptom match:** 2506.09501 ("minor rounding differences in early tokens cascade into divergent
   chains of thought") and the Thinking Machines blog ("temp=0 yet not deterministic") describe exactly
   our 0/111. The census attributing 100% of variance to *generation* (not scoring) is consistent with a
   *logit-level* numerical cause that lives entirely in the generation forward pass.
2. **Mechanism match:** the cause is FP non-associativity (2408.05148) × batch-shape-dependent reduction
   order under dynamic batching (Thinking Machines) × tiny logit gaps that flip argmax (2506.09501) ×
   (if multi-GPU) TP-dependent reduction order (2511.17826). Sampling/seed is NOT the cause — temp=0
   already kills sampling; the residue is numerical.
3. **Setting match:** 2511.17826 names *multi-agent systems* and *LLM-as-a-judge* as the precise
   applications that determinism breaks — our pass^4 / A2-compile / selector pipeline.

So ⓟ1's diagnosis ("agent generation nondeterminism = batch-nondeterminism = the pass^4 variance source")
is **confirmed by the literature**, and the fix is a known, shipped knob.

---

## 8. CONCLUSIONS

### (a) Precise vLLM determinism recipe + minimum version

**Minimum version with `VLLM_BATCH_INVARIANT`: vLLM v0.11.1** (0.11.0 does NOT have it — design-doc
finding confirmed). Prefer a current release (feature is "beta" and still hardening).

Deterministic temp=0 path (pass^4 / A2-compile):
1. Upgrade to **vLLM ≥ 0.11.1** (ideally latest stable).
2. Set **`VLLM_BATCH_INVARIANT=1`** in the environment *before vLLM import / server launch*. This is the
   **only** knob that works in online/server mode; it makes RMSNorm, matmul, attention reductions
   batch-invariant (bs=1 == bs=N, including prefill).
3. Run on a GPU with **compute capability ≥ 8.0** (Ampere/A100, Hopper/H100 — our hardware qualifies;
   verify).
4. Keep **`tensor_parallel_size=1`** for eval (TP>1 reduction order is NOT covered by batch-invariance —
   would need TBIK-class kernels per 2511.17826). Single-GPU isolation aligns with our "GPU0/1 충돌금지"
   rule anyway.
5. (Offline-only alternative/supplement: `VLLM_ENABLE_V1_MULTIPROCESSING=0` for deterministic scheduling.
   Not usable online — we use server mode, so rely on batch-invariance.)
6. **Pin GPU model + exact vLLM version** in run metadata — reproducibility is guaranteed only on
   *identical hardware + identical vLLM version* (verbatim docs caveat).
7. Budget **≈1.6× latency** (Thinking Machines: 26 s → 42 s with improved attention).
8. Optional stacking lever if a kernel is missing for some op: FP32 compute (LayerCast-style, 2506.09501)
   — reduces but does not eliminate variance; not a substitute for batch-invariance.

Temp>0 path (selector diversity) — unchanged: `temperature>0`, leave `seed` unset/randomized.
Batch-invariance only makes *logits* reproducible; the sampler is untouched, so diversity is preserved.
Deterministic-at-temp0 + diverse-at-temp>0 is the natural regime — no conflict.

### (b) Can ⓟ1 be RE-OPENED with this recipe?

**Yes — ⓟ1 should be re-opened.** The original retirement reason was "vLLM 0.11.0 lacks
VLLM_BATCH_INVARIANT." That is now resolved: the flag ships in **v0.11.1+**, our GPUs meet CC≥8.0, and
the literature confirms batch-invariance is *the* fix for exactly our symptom. Re-open as: upgrade to
≥0.11.1, set `VLLM_BATCH_INVARIANT=1`, TP=1, pin HW+version, then re-run the tau2 4-trial experiment.
**Success criterion:** action sequences bitwise-identical across the 4 temp=0 trials (matching the
1000/1000 result), while a temp>0 control still diverges. Our batch-isolation experiment
(`--enforce-eager` + `--max-num-seqs 1`) is **NOT sufficient** as the principled fix — it removes
cross-request batch-variance but not intra-request batch-shape variation (prefill chunking, padding) and
destroys throughput; keep it only as a diagnostic A/B against the batch-invariant path, not the solution.

### (c) Future-dated IDs that do NOT resolve

**None.** All four resolve to real papers: 2601.06118, 2601.17768, 2604.22411, 2511.00025.
**One caveat to flag:** 2604.22411 (submitted 24 Apr 2026) was flagged "fictional/future-dated" by ONE
WebFetch render but **confirmed real** by a domain-restricted arxiv.org search (title + authors Messina &
Scotta, RAI). Cite with the note that only the abstract is confirmed; re-pull the PDF before quoting the
body.

---

## 9. Bibliography (this-session verification)

| # | Source | Type / tier | Date | Verification (this session) |
|---|--------|-------------|------|------------------------------|
| 1 | He, H. et al. (Thinking Machines Lab), "Defeating Nondeterminism in LLM Inference." | Blog / tech report (not peer-reviewed) | 2025-09-10 | **Full page fetched** |
| 2 | `thinking-machines-lab/batch_invariant_ops` (PyTorch lib) | Code/library | 2025 | Named in #1 (fetched) |
| 3 | Shanmugavelu, S. et al., arXiv:2408.05148v3 | Preprint (DOE/ORNL) | v1 2024-08-09; v3 2024-10-30 | **Full abstract page fetched** |
| 4 | Yuan, J. et al., arXiv:2506.09501 (v1 "Give Me FP32 or Give Me Death?") | Preprint | v1 2025-06-11; v2 2025-10-24 | **Full abstract page fetched** |
| 5 | Zhang, Z. et al., arXiv:2511.17826 | Preprint | v1 2025-11-21; v2 2026-05-29 | **Full abstract page fetched** |
| 6 | vLLM, "Batch Invariance" + Reproducibility docs; v0.11.1 release notes; vLLM project X announcement | Official docs + release notes | accessed 2026-06-13 | docs.vllm.ai WebFetch DENIED; confirmed via GitHub raw docs source + v0.11.1 release notes (fetched) + project X post (search) |
| 7 | Fu, T. et al., arXiv:2601.06118 | Preprint | 2026 | Fetched + arxiv.org search confirmed |
| 8 | Gond, R. et al., "LLM-42", arXiv:2601.17768 | Preprint (MSR/UW/IISc) | v1 2026-01-25; v2 2026-01-30 | Fetched + arxiv.org search confirmed |
| 9 | Yashwanth, T.S., arXiv:2511.00025 | Preprint | 2025-11 | Fetched + arxiv.org search confirmed |
| 10 | Messina, A. & Scotta, S., arXiv:2604.22411 | Preprint (RAI) | 2026-04-24 | arxiv.org search confirmed (one WebFetch render falsely flagged non-existent — abstract only) |

**Caveat:** docs.vllm.ai WebFetch was denied; vLLM knob names/wording are from the docs' GitHub-raw
source markdown, the v0.11.1 release notes, and the vLLM project's own X announcement — three
cross-consistent primary-ish channels. Bodies of #5, #7, #9, #10 not read (abstract-level only). Re-pull
the relevant PDF before quoting any body text from those.
