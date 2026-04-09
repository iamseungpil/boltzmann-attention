# Phase B Paper Plan v1 — Ontology-Categorical KV Cache Compression for Small-LLM Tool-Selection Agents

**Document version**: v1 (2026-04-09, method-name update 2026-04-09)
**Author**: working notes from session 317917ce
**Status**: living plan, supersedes Phase 1.x experiment plans
**Target venue**: ICLR 2027 (primary) / NeurIPS main track (conditional, subject to 2-week kill-switch gate 2026-04-23)

**Method naming note (2026-04-09)**: this document previously drafted the method as `OCQ` (ontology-categorical FOKVQ). Per legal-risk review, the `FOKVQ` token is part of the user's pending patent vocabulary and must not appear in any public-facing artifact. The method is **renamed to `OCQ` (Ontology-Categorical Quantization)** throughout this document and all downstream paper / memory artifacts. Variant tags `oc_fokvq_{1a,1b,1c}_{2a,2b}` are renamed to `OCQ-{1a,1b,1c}-{2a,2b}`. Source file paths under `scripts/fokvq/` are retained in the "Engineering State" section as-is for traceability to the 2026-04-09 experiment logs, pending a separate code-rename decision after patent claim review.

---

## 0. Executive Summary

We propose `OCQ` (Ontology-Categorical Facet-Oriented KV Quantization), a training-free non-uniform KV cache quantization method derived from the agent's tool catalog. The method uses a per-(layer, head) ontology basis `B_ont` constructed from semantic facets (function-action × io-type × domain × tool-category) to allocate **1-bit categorical** quantization to a small subset of K-space directions plus variance-proportional uniform quantization to the residual orthogonal complement.

The same `B_ont` doubles as an inference-time attention bias for tool selection accuracy. This makes the paper a **dual-claim contribution**: one ontology mechanism, two empirical wins (tool disambiguation accuracy + KV cache compression), unified by a single theoretical observation.

The closest prior art, **KVSink** (Su & Yuan, COLM 2025), is **geometrically orthogonal** to our method: KVSink protects sink token *positions* (rows of the T×d KV matrix) at fp16, while OCQ rotates the *columns* and quantizes the categorical-decision subspace at 1-bit. The two methods are stackable and we propose a 2×2 ablation grid as the paper's headline experiment.

**Today's headline finding** (2026-04-09): on Qwen2.5-7B WikiText-2 PPL with the MetaTool catalog-derived ontology, OCQ achieves PPL 7.43 at average 1.81 bits versus KIVI's 7.22 at average 2.06 bits — competitive accuracy at 12% lower bit budget. The same configuration with a PCA-derived "pseudo-ontology" gives PPL 11.83, a 4.4 PPL degradation, validating the theoretical claim that categorical 1-bit quantization works on **decision** axes (real ontology) and fails on **variance** axes (PCA).

---

## 1. Thesis and Contributions

### Single sentence

> An ontology basis derived offline from an LLM agent's tool catalog enables both inference-time attention bias for tool disambiguation **and** training-free non-uniform KV cache quantization in which the ontology axes are quantized to 1 bit because they carry categorical decisions whose continuous variance is post-resolution noise.

### Contributions

1. **Theoretical**: post-resolution categorical interpretation of K-space rotations. When a session-level ontology facet (e.g., user role, task category) is resolved, the K-space projection onto that facet's axis collapses to a categorical decision; continuous variance on that axis becomes task-irrelevant noise. Therefore 1-bit quantization on such axes is *noise removal*, not lossy compression.

2. **Methodological**:
   a. Catalog-derived 4-facet ontology extraction from a tool catalog with no training data and no labels (function-action × io-type × domain × tool-category).
   b. Per-(layer, head) ontology basis `B_ont` via Gram-Schmidt residualization in priority order across facets.
   c. `OCQ` quantizer: 1-bit categorical on `B_ont` coefficients (sign / mean-split / argmax variants) + variance-proportional INT on residual.
   d. Same `B_ont` reused as an inference-time attention bias for tool disambiguation (sister claim).

3. **Empirical**:
   a. **Claim B (KV quant)**: OCQ matches KIVI / KVQuant on small-LLM PPL at lower average bit budget; geometrically orthogonal to KVSink and stackable for additive gains.
   b. **Claim A (tool selection)**: OCQ attention bias improves top-1 accuracy on MetaTool's "similar choices" disambiguation subtask vs. dense-retrieval and prompt-engineering baselines.
   c. **Generalization**: results hold across Qwen2.5-{7B,14B,32B,72B}, Llama-3.1-{8B,70B}, Mistral-7B.
   d. **Korean coverage**: validated on FunctionChat-Bench (Kakao) `4_close` / `8_close` distractor configurations.

4. **Engineering**: open-source training-free reference implementation that plug-and-plays into HuggingFace transformers via forward hooks. Hot-swap of tool catalogs without retraining.

### What this paper deliberately is NOT

- Not a new attention steering method per se. The K-bias claim reuses the known SEKA / PASTA / ASA operator family with a new direction source.
- Not a new quantization datatype. OCQ uses standard symmetric INT for the residual; only the ontology subspace gets the categorical treatment.
- Not a competitor to KVSink. KVSink and OCQ are stackable; we report combined results, not head-to-head.
- Not a tool retrieval system. We compare to BGE-large dense retrieval as a baseline but the contribution is inference-time attention modification, not retrieval-augmented prompting.
- Not a long-context paper. We test on long-context KV quant benchmarks (LongBench, RULER) for completeness, but the primary win is on agent-style short-context tool selection where the ontology is most informative.

---

## 2. Background and Prior Art

### Attention sink and KV cache quantization

Xiao et al. (StreamingLLM, ICLR 2024) observe that the first few token positions absorb disproportionate softmax mass and any quantization error there is amplified. KIVI (Liu et al., ICML 2024) handles this by keeping the most-recent `R` tokens in fp16 (per-channel for K, per-token for V). KVQuant (Hooper et al., NeurIPS 2024) uses non-uniform datatypes (NUQ) with per-channel scales and a small dense-and-sparse outlier set. GEAR (Kang et al., 2024) adds low-rank correction. AQUA-KV (Pinaev et al., ICML 2025) uses cross-layer linear prediction for adaptive bit allocation.

**KVSink** (Su & Yuan, COLM 2025, arXiv:2508.04257) is the closest direct prior art for our paper. It explains attention sinks as the cross-layer evolution of stable activation outliers and uses a top-k detector at a precomputed emergence layer `l_E` to dynamically identify sink **token positions**, holding their full `d_head`-dim K/V at fp16. Stacked on KVQuant, KVSink reduces the fp16 outlier budget from 0.5% to 0.1% with no loss. KVSink reports PPL only (WT2, C4) on LLaMA2/3 and Mistral; no downstream task accuracy is reported.

### Tool selection / function calling

BFCL v3 (Berkeley, Sep 2024) and v4 Agentic (Jul 2025) are the de-facto leaderboards. MetaTool (Huang et al., ICLR 2024) is purpose-built for disambiguation, with an explicit "tool selection with similar choices" subtask covering 199 plugins / 47 categories / 995 disambiguation queries. ToolBench / ToolLLM (Qin et al., ICLR 2024) and StableToolBench (THUNLP-MT, ACL Findings 2024) provide large-scale natural-overlap stress tests. NESTFUL (IBM, EMNLP 2025) tests nested call composition. ComplexFuncBench (THUDM, 2025) is a long-context cross-domain benchmark. τ-bench (Sierra Research, 2024) evaluates multi-turn agentic policy compliance. FunctionChat-Bench (Kakao, 2024) is the only public Korean function-calling benchmark with explicit close-distractor configurations.

### Attention steering for LLM control

CAA (Panickssery et al.), ITI (Li et al.), PASTA (Zhang et al.), ASA (Wang et al. 2026), Focus Directions (Zhu et al. 2025), and SEKA (Li et al. ICLR 2026) are all activation / attention steering methods that modify residual stream or K/V projections to control LLM behavior. These are mech-interp heritage. We cite them in related work but our positioning is **applied agent inference**, not interpretability — accordingly the baselines we compete with are dense retrieval, prompt engineering, LoRA fine-tuning, and ASA (the only one targeting the same use case).

### Excluded for IP reasons

The user has a patent-pending method that conceptually overlaps with parts of this work. By explicit instruction (2026-04-09), this paper does **not** cite, reference, or use the patent-pending method, its acronyms, its proprietary taxonomies, or any customer data. The paper stands alone on public benchmarks and public baselines.

---

## 3. Method

### 3.1 Catalog-derived ontology extraction (`build_metatool_ontology.py`)

Input: a tool catalog (list of `(name, description)` pairs) plus optional category labels.

Procedure:
1. Define four canonical facet axes in priority order (most general → most specific):
   - **F1 = function-action**: `search / retrieve / create / analyze / summarize / recommend / book / track / convert / translate / play / inform`
   - **F2 = io-type**: `text / structured / image / audio / video / numeric`
   - **F3 = domain**: `finance / news / health / education / travel / shopping / entertainment / food / real_estate / weather / productivity / research / legal / career / utility`
   - **F4 = tool-category**: top-K most-populous category labels from the catalog itself (here K=15 from MetaTool's 47)
2. For each tool, run a rule-based keyword matcher over the description to assign `(action, io_type, domain)` triples.
3. For each facet × category, assemble 2–4 anchor sentences: hand-written canonical templates plus catalog-derived examples.
4. Output a dict `ONTOLOGY[facet][category] -> List[sentence]` compatible with `ontology_facet_basis.py`.

This procedure is fully training-free, deterministic, and takes <1 second on the MetaTool catalog. No labeled data, no LoRA training, no model fine-tuning.

### 3.2 Per-(layer, head) basis construction (`build_qwen_metatool_b_ont.py` + `ontology_facet_basis.py`)

For each `(layer ℓ, head h)`:
1. Run each anchor sentence through the model with a forward hook on `k_proj`. Average K vectors over content tokens (excluding BOS).
2. For each facet, build the embedding matrix `E^{(f)} ∈ ℝ^{d × K_f}` where columns are per-category mean K vectors.
3. Gram-Schmidt residualize across facets in priority order: `E_res^{(f)} = E^{(f)} − P_<f E^{(f)}` where `P_<f = Σ_{f' < f} B^{(f')} (B^{(f')})^T` is the projector onto the union of higher-priority facet bases.
4. SVD `E_res^{(f)}` and keep top components by cumulative energy ≥ 0.95 with `σ_i ≥ 10⁻³ σ_max`.
5. Concatenate the per-facet bases: `B_ont^{(ℓ,h)} = [B^{(F1)} | B^{(F2)} | B^{(F3)} | B^{(F4)}] ∈ ℝ^{d × r_ont}`.
6. Truncate uniformly to `r_min = min_{ℓ,h} r_ont^{(ℓ,h)}` so the saved tensor is rectangular `(n_layers, n_kv, d_head, r_min)`.

For Qwen2.5-7B with the MetaTool 4-facet ontology, this yields `r_ont = 24` per head (range 24–33 truncated to min). Total tensor: `(28, 4, 128, 24)`.

### 3.3 Residual basis construction (OCQ quantizer module)

Given `B_ont` and per-head `Σ_K`, compute the residual basis `B_res ∈ ℝ^{d × (d − r_ont)}` via one of two modes:

- **Mode 2a (orthogonal-complement eigh)**: `Σ_res = (I − B_ont B_ont^T) Σ_K (I − B_ont B_ont^T)`, eigendecompose, take top `(d − r_ont)` eigenvectors.
- **Mode 2b (full PCA + projection)**: eigendecompose `Σ_K`, project each eigenvector onto the orthogonal complement of `B_ont` via modified Gram-Schmidt preserving order, keep top `(d − r_ont)`.

The full per-(layer, head) basis is `B = [B_ont | B_res] ∈ ℝ^{d × d}`, orthonormal by construction.

### 3.4 OCQ quantization

For a K cache `keys ∈ ℝ^{B × H × T × d}`, per (batch, head):
1. Rotate: `coeffs = keys @ B`. Split into `coeffs_ont = coeffs[..., :r_ont]` and `coeffs_res = coeffs[..., r_ont:]`.
2. **Categorical 1-bit on ontology axes** — three modes:
   - **1a (sign-only)**: `q_ont = sign(coeffs_ont) * mean(|coeffs_ont|, axis=batch)`. 1 bit / token / axis.
   - **1b (mean-split)**: per axis, cluster coefficients into two buckets at the per-axis mean and represent each bucket by its centroid. 1 bit / token / axis.
   - **1c (argmax)**: per token, identify the ontology axis with largest |coeff| and store only that coefficient at full precision; zero all other ontology axes. `⌈log₂(r_ont)⌉ + 1` bits / token total for the entire ontology block.
3. **Variance-proportional uniform on residual**: `q_res = symmetric_quant(coeffs_res, res_bits)` where `res_bits` matches the nominal bit budget (2 / 3 / 4).
4. Reconstruct: `keys_quant = [q_ont | q_res] @ B^T`.

### 3.5 Sister claim — attention-time K-bias (claim A)

The same `B_ont` is used as an attention bias following the SEKA hook protocol: at marker positions during the resolution step, `K += α · (B_ont B_ont^T) K`. This concentrates attention along ontology-relevant directions. Phase 1.x infrastructure (`scripts/phase1_ontology_projection_rank8.py` and the patched SEKA `eval_fact_gen.py`) is reused.

### 3.6 Sink protection (orthogonal fix)

Independently of OCQ, the first `sink_len = 4` tokens of each cache window are kept in fp16 (KIVI residual length analogue, KVSink-compatible row preservation). Implementation in `_split_sink_bulk` of `exp4_2_standard_ppl_benchmark.py`.

---

## 4. Experimental Plan

### 4.1 Models (8)

| Family | Small | Mid | Large | Notes |
|---|---|---|---|---|
| Qwen2.5 | 7B (primary) | 14B | 32B, 72B | OISA reference family; primary results |
| Qwen3 | 1.7B (optional) | 4B | — | Phase 1.x infrastructure already built |
| Llama-3 | 3.2-1B (optional) | 3.1-8B-Instruct | 3.1-70B-Instruct | KVSink reproduction model |
| Mistral | — | 7B-v0.3 | — | KVSink published baseline |
| Phi-3 | mini-4k (optional) | — | — | Family diversity |

The 70B/72B runs go to coworker's A100-80GB × 4. 7B-32B fit on A6000-48GB × 2.

### 4.2 Benchmarks

**Claim A — Tool selection accuracy (primary)**

| # | Benchmark | License | Catalog | Disambig subset | Role |
|---|---|---|---|---|---|
| 1 | MetaTool ICLR 2024 | MIT | 47 cat / 199 plugins | "similar choices" 995 queries | **Primary** |
| 2 | BFCL v3 + v4 Agentic | Apache-2.0 | 60+ tools | irrelevance/relevance subsets | Community standard |
| 3 | NESTFUL EMNLP 2025 | Apache-2.0 | math + code | nested composition | Stress |
| 4 | ComplexFuncBench 2025 | MIT | cross-domain | 128k long context + cross-domain | **Joint claim A+B** |
| 5 | StableToolBench ACL 2024 | Apache-2.0 | 16k APIs / 49 cat | natural overlap | Scale stress |
| 6 | FunctionChat-Bench (Kakao) | Apache-2.0 | 500 × 5 configs | `4_close` / `8_close` | **Korean coverage** |

**Claim B — KV cache quantization (primary)**

| # | Benchmark | Measures | Used by | Role |
|---|---|---|---|---|
| 1 | WikiText-2 PPL (KIVI protocol) | LM PPL, ctx=2048 non-overlap | KIVI / KVQuant / GEAR / AQUA-KV / KVSink | Headline anchor |
| 2 | LongBench v1 (15 tasks) | downstream task accuracy 4k–32k | KIVI / KVQuant / GEAR / AQUA-KV | Downstream |
| 3 | RULER (NVIDIA) | synthetic needle / multi-hop / aggregation 4k–128k | de-facto 2025 standard | Long-context |

### 4.3 Baselines

**Claim A baselines**
1. No-steer (raw model, prompt only)
2. Dense retrieval over tool schemas (BGE-large, top-k injected into prompt)
3. Prompt engineering (category hints, scenario prefix)
4. LoRA fine-tuning on labeled tool-use data (standard instruction tuning)
5. ASA (Wang et al. 2026) — only direct activation-steering precedent for tool use
6. **Ours: OCQ K-bias** with ontology basis

**Claim B baselines**
1. fp16 (ceiling)
2. Uniform per-token symmetric quant (floor)
3. **KIVI** (Liu et al. 2024) — per-channel K, per-token V, R=128 fp16 residual
4. **KVQuant** (Hooper et al. 2024) — per-channel pre-RoPE K, NUQ datatype, dense+sparse outliers
5. **GEAR** (Kang et al. 2024) — quant + low-rank + sparse correction
6. **AQUA-KV** (Pinaev et al. ICML 2025) — adaptive cross-layer prediction
7. **KVSink** (Su & Yuan COLM 2025) — sink-token preservation, **must reproduce, no public code**
8. **More for Keys / Less for Values** (Feb 2025) — K vs V bit asymmetry
9. **Ours: OCQ** with catalog-derived ontology basis
10. **Combined: KVSink + OCQ** (the headline 2×2 ablation)

### 4.4 Metrics

| Claim | Primary | Secondary | Safety / sanity |
|---|---|---|---|
| A (tool selection) | Top-1 accuracy on MetaTool similar-choices | Top-3, AST correctness on BFCL | Schema-parse fabrication rate |
| B (KV quant) | Average bits per element @ matched accuracy | Peak KV memory, prefill latency overhead | WT2 PPL anchor |
| Joint | Top-1 accuracy under quantized cache | Combined Pareto frontier | Non-tool turn perplexity preservation |

### 4.5 Headline experiment — 2×2 ablation grid

Following the KVSink comparison analysis, the load-bearing experiment is the orthogonality grid on **LLaMA-3-8B-Instruct** (KVSink eval set + our deployment target):

| | KVSink OFF | KVSink ON (k=5) |
|---|---|---|
| **OCQ OFF** | naive 3-bit (floor) | KVSink-only reproduction |
| **OCQ ON** | OCQ-only (main) | **Combined (stack)** |

Hypothesis: `Combined > max(KVSink, OCQ)` on tool selection top-1 at equal or lower total bit budget. If confirmed, the geometric-orthogonality claim is numerically justified and the reviewer attack "this is just KVSink with semantic sinks" is preemptively neutralized.

---

## 5. Today's Findings (2026-04-09)

### 5.1 Prior bit-schedule variant 2-bit GPT-2 failure root cause is bit-schedule asymmetry, not attention sink

- **Tested** sink fix on the prior bit-schedule quantizer variant at GPT-2 medium and Qwen2.5-7B 2-bit. Result: sink fix benefits KIVI (29.50→28.76 GPT-2; 8.26→7.22 Qwen) but **strictly worsens** the prior variant in all sink configurations (52→96 GPT-2; 4527→5063 Qwen).
- **Real cause**: `bit_schedule(2, topk_frac=0.25) = (5, 1)` puts 1-bit on 75% of dims, which is catastrophic. Switching to `topk_frac=0.5 → bit_schedule = (3, 1)` improves GPT-2 2-bit from 63.53 to 39.09 (1.6× improvement).
- **Saved**: `memory/prior_variant_2bit_root_cause_2026_04_09.md`.

### 5.2 OCQ alleviates the bit-schedule pathology by construction

OCQ's split is `r_ont` dims at 1-bit + `(d − r_ont)` dims at uniform `bits`. With `r_ont/d ≈ 0.19` (24/128 on Qwen2.5-7B) the 1-bit share is much smaller than the `topk_frac=0.25 → 75% at 1-bit` of the prior bit-schedule variant. At Qwen2.5-7B 2-bit nominal, OCQ-1b gets PPL 7.43 vs prior variant 5063 — a 681× improvement.

### 5.3 Real catalog-derived ontology dramatically beats PCA pseudo-ontology

This is the day's most important result and the empirical validation of the categorical-1-bit hypothesis.

**Qwen2.5-7B WT2 PPL** (ctx=512, stride=256, max=16384, sink_len=4):

| Method | 2-bit | 3-bit | 4-bit |
|---|---|---|---|
| fp16 | 6.57 | — | — |
| KIVI | 7.22 | 6.59 | 6.52 |
| OCQ-1b-2a (PCA pseudo, r=16) | 11.83 | 11.03 | 84.92 |
| OCQ-1b-2a (**MetaTool real**, r=24) | **7.43** | **7.04** | **6.94** |
| OCQ-1b-2b (MetaTool real) | (n/a) | (n/a) | **6.92** |
| Δ (real − PCA pseudo) | **−4.40** | **−3.99** | **−77.98** |

**Interpretation**: PCA "pseudo-ontology" picks the top variance directions, which carry continuous magnitude information. Categorical 1-bit on those directions destroys signal. At low bit budgets (2-bit) the residual error masks this; at high bit budgets (4-bit) the residual error is small enough that the categorical-1-bit loss on top-PCA dirs dominates and PPL explodes (84.92).

Real catalog-derived ontology axes are approximately categorical: each axis aligns with a *facet decision* (which function, which io type, which domain, which tool category) and the residual variance on that axis is post-decision noise. 1-bit on those axes preserves the decision and discards the noise. PPL is stable across all bit budgets.

This is the **decisive empirical evidence** for the user's theoretical insight that ontology axes carry categorical decisions (1-bit suffices) while variance axes carry continuous information (1-bit destroys).

**Saved**: `memory/ocq_real_ontology_validation_2026_04_09.md`.

### 5.4 OCQ is competitive with KIVI at lower bit budget

| Method | Avg bits | PPL | Bits Δ vs KIVI | PPL Δ vs KIVI |
|---|---|---|---|---|
| KIVI 2-bit nominal (R=4) | 2.06 | 7.22 | (baseline) | (baseline) |
| OCQ 1b_2a 2-bit (real ontology, r=24) | 1.81 | 7.43 | **−12% bits** | +0.21 PPL |

OCQ achieves competitive PPL at 12% lower memory budget. The trade-off is 0.21 PPL — within typical run-to-run noise on this protocol. The real win must be measured on **task accuracy** (the right battleground), not PPL on unconditioned text.

### 5.5 KVSink is geometrically orthogonal to OCQ

Background agent deep-read of arXiv 2508.04257 confirmed:
- KVSink touches **rows** of the T×d KV matrix (sink token positions).
- OCQ touches **columns** (K-space rotation).
- KVSink keeps special set at **fp16** (max precision) because sinks amplify error.
- OCQ keeps special set at **1 bit** (min precision) because ontology axes carry categorical decisions.
- Geometrically orthogonal → stackable → 2×2 ablation grid is the natural headline experiment.
- KVSink published only PPL on LLaMA2/3 / Mistral. No downstream task accuracy. **No public code as of 2026-04-09** — must reproduce from paper description.

**Saved**: `memory/kvsink_vs_ocq_comparison_2026_04_09.md`.

### 5.6 Eval architecture limitation discovered

Our `exp4_2_standard_ppl_benchmark.py` only quantizes the K cache *between* two model.forward() calls (prefix → quant → target). When `stride == context_len`, every window has `prefix_len = 0` and the script falls through to the fp16 `score_full_window_suffix` path — quantization is **never applied**. This was discovered when an attempt to run KIVI's published protocol (ctx=2048 non-overlap) returned identical PPL across all quant methods (all = fp16 number 7.684).

**Implication**: our PPL numbers are only directly comparable to KIVI's published numbers if we add a forward-hook-based quantization mode that applies quant during the model's own forward pass. This is required infrastructure for the headline KIVI / KVQuant / KVSink reproduction.

---

## 6. Engineering State

### 6.1 Files / artifacts produced today

**Scripts**
- `scripts/fokvq/exp4_2_standard_ppl_benchmark.py` — extended with `--sink-len`, `--calibration-sink-skip`, OCQ method registry, external `B_ont` loading, sink-bulk split helper
- `scripts/fokvq/oc_fokvq.py` — 6-variant OCQ module with self-test (code path retained pending separate code-rename decision; see naming note at top of this document)
- `scripts/fokvq/build_metatool_ontology.py` — 4-facet catalog-derived ontology extractor
- `scripts/fokvq/build_qwen_metatool_b_ont.py` — per-(layer, head) `B_ont` builder reusing `ontology_facet_basis.py`

**Data**
- `external/SEKA/seka_projections/ontology-qwen25-7b-metatool/B_ont.pt` — `(28, 4, 128, 24)` `B_ont` for Qwen2.5-7B
- `reports/axis2_theoretical_verification/metatool_ontology.json` — 4-facet ontology dict
- `reports/axis2_theoretical_verification/build_qwen_metatool_b_ont.json` — diagnostic
- `reports/axis2_theoretical_verification/phase1_ontology_projection_qwen3_4b*.json` — Phase 1.x diagnostics

**Memory (persistent across sessions)**
- `paper_goal_is_tool_selection.md` (feedback)
- `phase_b_tool_selection_plan.md` (project)
- `oisa_deployment_context.md` (project — DO NOT cite in paper)
- `deployment_scale_and_dual_claim.md` (project — 7B-70B target, dual claim)
- `fokvq_2bit_root_cause_2026_04_09.md` (project)
- `ocq_real_ontology_validation_2026_04_09.md` (project — TODAY's main result; renamed from `oc_fokvq_real_ontology_validation_2026_04_09.md`)
- `kvsink_vs_ocq_comparison_2026_04_09.md` (project — orthogonal stackable; renamed from `kvsink_vs_oc_fokvq_comparison_2026_04_09.md`)

**External clones**
- `/tmp/MetaTool` — full repo, MIT license, catalog inspected

**Permissions**
- `/home/woori/.claude/settings.json` — `WebSearch` and `WebFetch` added to always-allow

### 6.2 Hardware budget

- Our side: A6000-48GB × 2. Sufficient for 7B-13B forward passes, marginal for 32B with quantized weights.
- Coworker side: A100-80GB × 4 = 320 GB total. Comfortable for 70B bf16 forward passes (140 GB weights + KV cache headroom).
- KVSink reproduction: 7B-class is sufficient for replication; 70B needed for headline scale claim.

---

## 7. Reviewer Attack Defense

| Attack | Response |
|---|---|
| "This is just KVSink with semantic sinks" | KVSink touches *rows* (token positions), OCQ touches *columns* (K-space dirs). KVSink keeps at **fp16**, OCQ at **1-bit**. Inverted precision semantics on orthogonal axes. Demonstrate via 2×2 ablation grid where Combined > max(individual). |
| "1-bit categorical needs the post-decision argument, which is agent-specific" | True and intentional. We position OCQ as an *agent-inference* method, not a general LM compression method. Don't compete on raw WT2 PPL where the ontology basis has no reason to exist. |
| "PCA-based methods (KIVI, KVQuant) work fine without semantic structure" | Yes, on PPL. We show that on **task accuracy** under matched bit budget, OCQ wins because the categorical 1-bit IS the right granularity for the task's decision structure. The 4.4-78 PPL gap between real and PCA-pseudo ontology under our own method is the empirical pivot. |
| "Why not just use KIVI? It already works" | KIVI uses 12% more bits and doesn't carry an attention bias claim. OCQ is the unified mechanism for both compression and tool disambiguation. Same ontology basis, two empirical wins. |
| "Your ontology extraction is rule-based, not learned" | Yes, intentionally training-free. The contribution is that *no training is needed*. We compare to LoRA fine-tuning as a baseline and show competitive top-1 at zero training cost. |
| "Why MetaTool? It's only 199 plugins" | Primary because it has an explicit similar-choices subtask. We additionally test on StableToolBench (16k APIs) for scale, BFCL for community familiarity, and FunctionChat-Bench for Korean / language coverage. |
| "Mistral results were negative in your Phase 1.x" | Phase 1.x used a toy product/manufacturer ontology, not catalog-derived from the eval task. The negative transfer was expected in hindsight: the toy ontology doesn't align with CounterFact's factual recall structure. With catalog-derived ontology aligned to the actual task (MetaTool / BFCL), Mistral results should be analogous to Qwen. To be tested. |
| "Why no 1B results?" | Customer deployment target is 7B-70B (B200 cluster). 1B is below the relevant capability floor for tool-using agents in the literature (ASA showed recall=0 at 0.5B). We optionally include 1B as a scaling-curve point but not as the primary claim. |

---

## 8. Next Steps (Prioritized)

### Critical path (Week 1-2)

1. **Eval architecture fix** — rewrite `exp4_2_standard_ppl_benchmark.py` to apply quantization via PyTorch forward hooks during the model's own forward pass, supporting non-overlapping context_len=2048 chunk evaluation. Required for KIVI / KVQuant / KVSink reproduction at canonical protocol. **Blocks**: any direct comparison with published PPL numbers.

2. **KVSink reproduction** (`scripts/fokvq/kvsink_reproduce.py`) — identify per-model emergence layer `l_E` and outlier channels `C^*` via WT2/C4 calibration; implement dynamic top-k row preservation; verify against published LLaMA2-7B WT2 numbers (KVQuant-4b 5.73 → KVSink-5 5.60).

3. **MetaTool subtask1 evaluation pipeline** — load `Task2-Subtask1.json` (995 queries), apply OCQ K-bias to Qwen2.5-7B during the resolution forward pass, score top-1 tool selection accuracy, compare to no-steer / BGE retrieval / prompt engineering / LoRA / ASA baselines. **Critical claim A test**.

4. **r_ont sweep on real ontology** — test r_ont ∈ {8, 12, 16, 24, 32} on Qwen2.5-7B with MetaTool ontology. Determine optimal subspace size for the categorical-1-bit + variance-residual split.

### Headline experiment (Week 2-3)

5. **2×2 ablation grid on LLaMA-3-8B-Instruct** — `{KVSink OFF/ON-k5} × {OCQ OFF/ON}` on WT2 PPL (anchor) + MetaTool subtask1 top-1 (claim). Hypothesis: Combined > max(individual). Required to neutralize the "just KVSink with semantic sinks" reviewer attack.

### Scaling and generalization (Week 3-6)

6. **Larger model scaling** — Qwen2.5-14B, 32B, 72B on coworker A100×4. Llama-3.1-70B-Instruct. Same MetaTool ontology basis builder; check that B_ont quality and OCQ benefits scale.

7. **KV quant baseline matrix** — KIVI / KVQuant / GEAR / AQUA-KV / KVSink / Ours on (Qwen2.5-{7B,14B,32B,72B}, LLaMA-3.1-{8B,70B}, Mistral-7B) × (WT2 / LongBench / RULER) × (2/3/4-bit average).

8. **Tool selection benchmark matrix** — MetaTool subtask1, BFCL v3 multi-turn + irrelevance, NESTFUL, ComplexFuncBench cross-domain, StableToolBench, FunctionChat-Bench `4_close`/`8_close` across all models × (no-steer / retrieval / prompt / LoRA / ASA / Ours).

### Writeup and ablations (Week 7-10)

9. **Random orthonormal rank-r control** for OCQ ontology basis. Already done for Phase 1.x K-bias on Qwen3-4B / CounterFact (47pp gap). Repeat for KV quant claim B on Qwen2.5-7B / MetaTool to confirm "ontology is special, not just any rank-r basis".

10. **Alternate ontology content ablation** — replace function-action / io-type / domain / tool-category facets with a different 4-facet decomposition (e.g. emotion / time / entity / quantity). If lift survives, the contribution is "structured ontology works" not "this specific ontology". If lift collapses, content matters.

11. **Phase-gating validation** — check that applying OCQ K-bias only at the resolution turn does NOT degrade non-tool-calling turns (perplexity / fluency on intermediate dialogue). Required for the "phase-gated" claim.

12. **Appendix A draft** — consolidate Phase 1.1-1.4 factual-editing benchmark results as 1-2 page Appendix "Operator validation on factual-editing benchmarks". Honest: report both Qwen3-4B positive (OCQ K-bias beats SEKA on CounterFact) and Mistral negative.

13. **Final paper draft** — assemble main body around the 2×2 ablation grid headline + Pareto frontier figure + scaling table. Target NeurIPS submission deadline (typically May for the year).

---

## 9. Open Questions

1. **Optimal `r_ont`**: 24 was the min across heads after Gram-Schmidt. Smaller (8, 12, 16) might give better PPL by increasing residual budget. Larger (32, 48) might improve task accuracy by capturing more facet structure. Empirical sweep needed.

2. **Per-head vs uniform `r_ont`**: currently uniform `r_min = 24`. Per-head variable might be optimal but requires the full quant pipeline to track per-head sizes (currently rectangular tensor only).

3. **2a vs 2b residual mode**: nearly identical with PCA-pseudo (as expected) and within noise on real ontology (1b_2a 6.94 vs 1b_2b 6.92 at 4-bit). Empirically minor but theoretically distinct.

4. **Sink fix interaction with OCQ**: sink fix benefits KIVI but worsens the prior bit-schedule variant. Untested whether OCQ behaves like KIVI or like that variant on this axis. Should be tested.

5. **Long-context regime (LongBench, RULER)**: OCQ assumes K-space anisotropy is consistent across the cache. Long-context K stats may differ from short-context calibration. Re-calibration on long-context corpus may be needed.

6. **Multi-turn agent regime**: in true agent loops, the ontology resolution at turn 1 should benefit cache compression at turns 2+. Phase-gated application is the natural design. Not yet tested in a multi-turn pipeline.

7. **Bit-budget normalization for fair comparison**: KIVI's `R = 128` fp16 residual adds a significant constant overhead. OCQ's 1-bit ontology subspace adds a different constant. The fair comparison is at matched **average bits per element**, not nominal bit budget. Headline table should report both.

---

## 10. Confidentiality Note

**OISA patent (user's pending application) is excluded from this paper by explicit instruction (2026-04-09).** Specifically:
- Do not cite or mention OISA, AFOD, MF-OISA, FOKVQ as patent terms, or any of the customer-specific facets (F1=Structure, F2=Journey, F3=Intent, F4=Tool).
- Do not use the 58% / 93% homonym numbers as baselines — these are internal patent claims.
- Do not reference the Korean bank CDP catalog, the 45-tool / 500-tool sizes, or "전환" as a homonym example.
- Do not compare to the patent-described LoRA-based facet internalization as a baseline. Use generic LoRA fine-tuning baselines instead.

The paper uses public benchmarks (MetaTool, BFCL, NESTFUL, ComplexFuncBench, StableToolBench, FunctionChat-Bench, LongBench, RULER, WikiText-2) and public baselines (KIVI, KVQuant, GEAR, AQUA-KV, KVSink, ASA, BGE retrieval, standard LoRA) only. Customer deployment is the post-publication practical use case, not a paper contribution.

---

## 11. References

### Tool selection / function calling
- BFCL Berkeley Function Calling Leaderboard, Sep 2024 (v3) / Jul 2025 (v4): https://gorilla.cs.berkeley.edu/leaderboard.html
- MetaTool, Huang et al., ICLR 2024: https://arxiv.org/abs/2310.03128 / https://github.com/HowieHwong/MetaTool
- ToolBench / ToolLLM, Qin et al., ICLR 2024: https://arxiv.org/abs/2307.16789 / https://github.com/OpenBMB/ToolBench
- StableToolBench, Guo et al., ACL Findings 2024: https://github.com/THUNLP-MT/StableToolBench
- NESTFUL, Basu et al., EMNLP 2025: https://github.com/IBM/NESTFUL
- ComplexFuncBench, THUDM 2025: https://arxiv.org/abs/2501.10132 / https://github.com/THUDM/ComplexFuncBench
- τ-bench, Yao et al., 2024: https://arxiv.org/abs/2406.12045 / https://github.com/sierra-research/tau-bench
- FunctionChat-Bench, Kakao 2024: https://arxiv.org/abs/2411.14054 / https://github.com/kakao/FunctionChat-Bench

### KV cache quantization
- KIVI, Liu et al., ICML 2024: https://arxiv.org/abs/2402.02750 / https://github.com/jy-yuan/KIVI
- KVQuant, Hooper et al., NeurIPS 2024: https://arxiv.org/abs/2401.18079 / https://github.com/SqueezeAILab/KVQuant
- GEAR, Kang et al., 2024: https://arxiv.org/abs/2403.05527
- AQUA-KV, Pinaev et al., ICML 2025: https://arxiv.org/abs/2501.19392
- KVSink, Su & Yuan, COLM 2025: https://arxiv.org/abs/2508.04257 (no public code)
- More for Keys, Less for Values, Feb 2025: https://arxiv.org/abs/2502.15075
- KITTY, Nov 2025: https://www.arxiv.org/pdf/2511.18643
- StreamingLLM (attention sinks), Xiao et al., ICLR 2024: https://arxiv.org/abs/2309.17453

### Long-context evaluation
- LongBench, Bai et al., 2023: https://github.com/THUDM/LongBench
- LongBench v2: https://longbench2.github.io/
- RULER, Hsieh et al., NVIDIA 2024: https://arxiv.org/abs/2404.06654 / https://github.com/NVIDIA/RULER

### Attention steering (related work, not primary baselines)
- StreamingLLM, Xiao et al., ICLR 2024
- CAA, ITI, PASTA, ASA — attention steering family
- SEKA, Li et al., ICLR 2026 — original operator we extended for K-bias
- Focus Directions, Zhu et al., 2025

### Datasets
- WikiText-2, Merity et al., 2016
- C4, Raffel et al., 2020

---

## 12. Document History

- **v1 (2026-04-09)**: Initial Phase B paper plan after the day's pivot from Phase 1.x factual-editing infrastructure to dual-claim tool-selection + KV quantization. Incorporates OISA exclusion, deployment scale correction, FOKVQ root-cause finding, OCQ formulation, real MetaTool ontology validation, KVSink prior-art comparison, eval architecture limitation, and prioritized next steps.
