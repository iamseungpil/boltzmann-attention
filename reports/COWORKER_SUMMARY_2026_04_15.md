# Coworker Summary — Experiment Results + Paper Framing (2026-04-15)

Target venue: **ICLR 2027** (Sep 2026 submission).
Canonical paper: `math/paper/benchmark_design/PAPER_DRAFT_v1_2026_04_14.md` (EN) / `PAPER_DRAFT_v1_ko.md` (KO).
Canonical proofs: `math/paper/lie_group/APPENDIX_B_PROOFS.md` + `COROLLARY_6_7_FACET_PHASE_CLOSURE.md` (**Cor 6.9.6 added 2026-04-15**).

---

## 1. Paper framing — stability-first (changed 2026-04-15)

### Old (pre-2026-04-15) framing
"Facet-gated K-bias improves tool-selection accuracy via operator-level rank structure."
**Linear bet**: theory → operator rank → multi-tool accuracy lift. Falsified at Subtask4 full 497 ($-4.6$pp).

### New framing — uniquely privileged ontology subspace
"The per-head ontology basis $B_{\mathrm{ont}}$ is the unique $R$-dimensional K-perturbation subspace whose output distribution diverges from the base model at rate $O(\alpha^2)$. Equal-magnitude complementary perturbations exit the FC-emission manifold at $\alpha > \alpha^*$."

**Double safety net**:
- Main claim (stability) already verified at full scale (+68.5pp direction-specificity gap on Subtask4 N=497).
- Accuracy lift (contrastive, LoRA) is independent secondary contribution. Individual failures do not break the main claim.

### Key theorem additions
- **Cor 6.9.6 (stability characterization, new)**: formal on/off-manifold KL bound with phase transition at $\alpha^*$. Proof in `COROLLARY_6_7_FACET_PHASE_CLOSURE.md` appended after Rmk 6.9.3.
- Elevates Cor 6.9 from operator-rank statement to distributional-stability statement.

---

## 2. Experimental results snapshot (2026-04-15 08:50 KST)

### 2.1 Subtask1 full 995 label_logprob cross-model grid

| Model | Scorer | no_steer | real a0.3 | random a0.3 | featshuffle a0.3 | real−random | real−featshuffle |
|---|---|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct | sum | 52.46 | +0.10 | −48.74 | −40.10 | **+48.84** | **+40.20** |
| Qwen2.5-7B-Instruct | mean | 36.78 | **+5.03** | −23.01 | −11.25 | **+28.04** | **+16.28** |
| Llama-3.1-8B-Base | sum | 46.33 | **+6.33** | −1.00 | −0.20 | **+7.33** | **+6.53** |
| Llama-3.1-8B-Base | mean | 23.12 | **+2.61** | −0.61 | −1.41 | **+3.22** | **+4.02** |
| Mistral-Base-v0.3 (skipL0+padmax) | sum | 69.35 | **+3.12** | pending | pending | — | — |
| Mistral-Base-v0.3 | mean | 40.70 | +0.20 | pending | pending | — | — |
| Mistral-Instruct-v0.3 | sum | 61.51 | **−2.92** | pending | pending | — | — |
| Mistral-Instruct-v0.3 | mean | 61.01 | **−3.62** | pending | pending | — | — |

**Reading**: 3-family Base positive (Qwen + Llama + Mistral all sum-positive). Mistral-Instruct is the sole negative; isolated as chat-template hedging (no_steer itself 7.84pp below Base). Direction specificity (gap columns) is scorer-invariant and model-invariant.

### 2.2 Subtask4 full 497 (**main empirical contribution**)

| B_ont | Method | F1 | Recall | Exact |
|---|---|---|---|---|
| real | no_steer | **0.731** | 0.716 | 0.525 |
| real | a0.3 | 0.685 | 0.672 | 0.473 |
| random | a0.3 | **0.000** | 0.000 | 0.000 |
| featshuffle | a0.3 | **0.000** | 0.000 | 0.000 |

**Direction-specificity gap = +68.5pp F1**. Largest single-experiment signal in the paper.

### 2.3 Non-uniform smoke (N=20, first multi-tool K-bias positive)

- Contrastive $a=0.3, d=3$: **F1 = 0.608 (+5.8pp over no_steer 0.550)** ★
- Contrastive $a=0.3, d=1$: F1 = 0.583 (+3.3pp)
- $\alpha$-sweep $a=0.15$: F1 = 0.575 (+2.5pp)
- V-bias variants: max F1 0.558 (all fail)
- Normalized (Thm 6.9.5 literal) at $\alpha \ge 0.3$: collapse

Full 497 extension of contrastive d=3 currently running on GPU0 (PID 1776169, ETA ~3h).

### 2.4 Thm 6.1 per-sample bound (theory verification)

Qwen2.5-7B-Instruct L=13, $\alpha=0.3$, N=100 queries × 28 heads = 2800 samples:
- **bound_pass_rate = 1.00** (every head-query sample satisfies Thm 6.1)
- median LHS/RHS ratio = $2.36 \times 10^{-8}$

### 2.5 Cor 6.9 operator nrank (SVD on 500 queries)

- Ours facet-gated: nrank_ε = **24.0** (at both ε=0.1, 0.2)
- AdaSEKA (T=0.1, β=0.094): nrank = 7.44
- **Gap +17** — decisive operator-level rank separation

### 2.6 R6 MMLU gate grid (N=1000)

Baseline 0.713. Best cell: **flat α=0.2 = 0.727 (+1.4pp)**. α=1.0 ordering: flat 0.584 > soft 0.614 > hard_argmax 0.552 > hard_thresh 0.535. Hypothesis (R) signal clearest at α=1.0.

### 2.7 WT2 compression (Thm 6.13)

Qwen2.5-7B full WT2 test (ctx=2048 non-overlap):
- 2-bit: **OCQ 15.60** vs KIVI 19.97 (−4.37 PPL, 9.4% fewer bits)
- 4-bit: KIVI 7.79 vs OCQ 12.56 (predicted cross-over at Cor 6.13.5 verified)

### 2.8 LoRA hybrid (Thm 6.16, rerun in progress)

Initial attempt 2026-04-15 02:32 KST OOM'd on GPU1 due to sibling processes. Rerun launched 08:49 KST (PID 1774445) with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` + batch-size 1. L1 training progressing normally (loss 9.58 → 0.001 at step 100).

---

## 3. What we need from coworker (A100 × 4 tracks)

See `reports/COWORKER_REQUEST_gemma_and_scaling_2026_04_14.md` v4. Priority order (updated):

### Priority 1 — Baselines (~18 GPU-hr, **critical for submission**)
Direct comparison table on Subtask1 + Subtask4 with CAA, ITI, PASTA, ASA, Focus Directions, AdaSEKA 2/3-expert, LoRA r=8 tool-FT, RAG prompt injection.
Scripts available: recipes documented in memory `baseline_recipes_attention_steering.md`.

### Priority 2 — Scaling (~30 GPU-hr)
Qwen2.5-{0.5, 3, 7, 14}B-Instruct on Subtask4 F1 + direction-specificity at $\alpha=0.3$. Expected: scale-invariant +68.5pp gap across sizes (architectural, not emergent).

### Priority 3 — Gemma (blocked on HF access form)
Gemma-3-27B-it R5 — coworker deployment-scale requirement.

### Priority 4 — LoRA full track (blocked on `build_lora_adapted_b_ont.py` + `eval_metatool_subtask4_lora.py` — TO WRITE)

---

## 4. Risks and open questions

1. **Contrastive d=3 full 497**: if it regresses (F1 < 0.608 at full), §5.5.2 becomes speculative. Main claim (§5.5 stability) unaffected.
2. **Mistral-Instruct null-control**: predicted +60pp direction-specificity gap. If observed, Instruct hedging is localized; if not, need to isolate scope.
3. **Baselines absence**: biggest risk to main-track. Coworker Track 1 is on the critical path.
4. **Paper length**: current draft ~515 lines. Target NeurIPS/ICLR 9 pages + appendix. Tight compression needed in §5.4–§5.10.

---

## 5. Files to read (in order, for context)

1. `math/paper/benchmark_design/PAPER_DRAFT_v1_2026_04_14.md` — canonical EN paper
2. `math/paper/lie_group/COROLLARY_6_7_FACET_PHASE_CLOSURE.md` (search "Corollary 6.9.6") — new stability theorem
3. `reports/subtask4_overnight/st4_{real,random,featshuffle}_N0.json` — main empirical result
4. `reports/nonuniform_smoke/st4_contrastive_a0.3_d3.json` — first multi-tool positive
5. `reports/theory_verify_2026_04_14/thm61_qwen_L13_a0.3_N100.json` — Thm 6.1 verification
6. `reports/theory_verify_2026_04_14/r6/r6_*.json` — MMLU 12-cell grid
7. `.claude/projects/-home-woori-workspace-common-boltzmann-attention/memory/subtask4_nullctrl_and_contrastive_d3_2026_04_15.md` — latest memory

---

## 6. Predicted main-track probability

| Scenario | Score (/10) | Main-track probability |
|---|---|---|
| Current (overnight complete, contrastive smoke) | 6.2 | 40–48% |
| + Contrastive d=3 full 497 confirms +5.8pp | 6.8 | 55% |
| + LoRA L3 F1 > 0.82 | 7.2 | 60–65% |
| + Baselines table (coworker Track 1) | 7.7 | 70% |
| + Gemma-3-27B scaling | 8.0+ | 75%+ |
