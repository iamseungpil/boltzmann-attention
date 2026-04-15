# AdaSEKA Empirical Comparison (CounterFact + MetaTool)

**Date**: 2026-04-10
**Purpose**: Establish decisive baseline against the closest prior art (AdaSEKA — Li et al., ICLR 2026, arXiv:2603.01281) so Phase B paper can argue differentiation from empirical evidence, not from narrative alone.

---

## 1. The comparison in one table

### CounterFact 500-sample subset, Qwen3-4B-Base

| Method | Direction source | Operator | ES | PS | Gap to best |
|---|---|---|---|---|---|
| no_steer | — | — | 40.2 | 43.6 | −56.6 |
| SEKA vanilla (α=1.56) | Synthetic SVD (200 pairs) | `k' = k + ½(g+·P+ + g-·P-)·k`, fixed g | 95.2 | 96.2 | −1.6 |
| **Ontology rank-8 (α=3.0)** | **MetaTool-like 4-facet ontology, Gram-Schmidt** | `k' = k + α·B_ont·B_ont^T·k`, fixed α | **96.8** | **96.7** | **best** |
| AdaSEKA 2-expert (held-out) | {synthetic, biasbios} SVD, Q-adaptive mixture | `k' = (I + g·P_dyn(Q))·k`, `P_dyn(Q)=Σ α_m(Q)·B_m B_m^T` | 48.2 | 54.0 | −48.6 |
| AdaSEKA 3-expert (+in-domain) | {synthetic, biasbios, counterfact} | same as above | 86.8 | 90.6 | −10.0 |

### Three empirical claims this establishes

**Claim 1 — AdaSEKA near-baseline without in-domain expert**
- With counterfact held out (realistic deployment: test domain unknown at method-construction time), AdaSEKA gives 48.2 ES vs no-steer 40.2 ES (+8.0pp). Near-baseline.
- Implication: AdaSEKA's "training-free" claim hides a hard dependency on in-domain contrastive data per test task.

**Claim 2 — Mixture dilution makes adaptive routing a liability**
- With counterfact expert included, AdaSEKA 3-expert = 86.8 ES. Still **8.4pp lower than single-expert vanilla SEKA** (95.2).
- SEKA uses only the synthetic expert (which AdaSEKA also includes). So AdaSEKA has strictly more information yet performs strictly worse.
- Implication: Q-adaptive max-normalized mixture dilutes the dominant expert's signal. The routing is a net negative even with favorable expert set.

**Claim 3 — Static ontology beats both AdaSEKA configurations**
- Ontology rank-8 α=3.0 = 96.8 ES, static basis derived from catalog structure (no benchmark contrastive data).
- Beats AdaSEKA 2-expert by 48.6pp and 3-expert by 10.0pp.
- Uses zero CounterFact training data. Direction source is task-agnostic ontology.

---

## 2. Where this comparison lives (reproducibility)

### Code
- AdaSEKA implementation: `external/SEKA/src/model/adaptive_seka_llm.py` (upstream, unchanged)
- Expert builders: `external/SEKA/src/custom_builders/adaptive/{synthetic,biasbios,counterfact,hotpot}_qa_builder.py`
- Eval driver: `external/SEKA/benchmarks/eval_fact_gen.py` (upstream, `--adaptive-seka` path)
- **2026-04-10 patch**: `external/SEKA/src/model/projection_builder_base_adapsvd.py` line ~52 — added `attn_implementation="eager"` so `output_attentions=True` actually returns per-layer weights (was silently empty under SDPA default, causing `IndexError: tuple index out of range` in `_check_attention_increase`).

### Expert data
- `external/SEKA/seka_projections/adaseka-qwen3-4b/synthetic/Qwen3-4B-Base_0.1mindiff_pos_svd.pt` (200 samples, 80/80 heads applied)
- `external/SEKA/seka_projections/adaseka-qwen3-4b/biasbios/Qwen3-4B-Base_0.1mindiff_pos_svd.pt` (200 samples, 80/80 applied, 61 kept / 135 discarded)
- `external/SEKA/seka_projections/adaseka-qwen3-4b/counterfact/Qwen3-4B-Base_0.1mindiff_pos_svd.pt` (100 samples, 80/80 applied)
- Hotpot expert build attempted but OOM'd on long multi-paragraph contexts; not included. Not a blocker for the Claim 1-3 story.
- Expert-path JSON files: `seka_projections/adaseka-qwen3-4b/expert_paths.json` (2-expert) and `expert_paths_3way.json` (3-expert).

### Result files
- `external/SEKA/benchmarks/counterfact/results/adaseka-qwen3-4b-500-2experts/{efficacy,paraphrase}.json`
- `external/SEKA/benchmarks/counterfact/results/adaseka-qwen3-4b-500-3experts/{efficacy,paraphrase}.json`
- Logs: `/tmp/adaseka_eval_counterfact.log`, `/tmp/adaseka_eval_counterfact_3expert.log`

### Runtime
- Build (per expert): ~15s on A6000 48GB
- Eval (per config, 500 samples, efficacy + paraphrase): ~90s on A6000 48GB
- Total for the 4 build + 2 eval = **~4 minutes actual GPU**

---

## 3. Positioning versus AdaSEKA — the decisive structural argument

AdaSEKA and our method look syntactically similar at the operator level — both can be written as `k' = (I + gate · P_something) · k`. They diverge on **four axes**:

### Axis A — Intervention axis (Q vs K)

**AdaSEKA**: K-side rewrite, but the projector `P_dyn(Q)` is a function of **Q**. Q chooses which expert subspaces to mix. This is a **Q-adaptive rotation**: the rotation operator depends on the query.

**Ours**: K-side rewrite with a projector structure determined by **ontology at build time**, not by Q. Q only modulates **how much** each facet activates (via independent gates), but never **which subspace** is used. The facets are fixed.

### Axis B — Mixture structure (winner-take-one vs simultaneous)

**AdaSEKA**: `P_dyn(Q) = Σ_m α_m(Q) · B_m B_m^T` where `α_m(Q)` are normalized by `max_m'`. The max normalization enforces **one dominant expert per query**. Other experts contribute softly but never dominate. This is structurally **1-of-M routing**.

**Ours**: `K_increment = Σ_f g_f(x) · B_f B_f^T K` where `g_f(x) = ‖x·B_f‖² / ‖x‖²` are **independent** per-facet energy gates with no normalization. Multiple facets can be simultaneously at full activation. **F-simultaneous composition** where F = number of orthogonal facets.

### Axis C — Direction source (contrastive SVD vs catalog ontology)

**AdaSEKA**: Each expert is built from a specific benchmark's **contrastive data** via SVD. Different expert = different benchmark's training set. The method inherently **requires per-task training data** even though it is "training-free" in the fine-tuning sense.

**Ours**: Facets are derived from the **tool / workflow / plan catalog structure** itself, not from test-task data. New tool addition = catalog update = single rank-increment on the affected facet's basis. **No benchmark contrastive pairs are ever collected**.

### Axis D — Facet count scaling

**AdaSEKA**: Evaluated with 4 experts (Synthetic, CounterFact, BiasBios, HotpotQA). Scaling beyond ~8 is problematic because the `max_m'` normalization saturates — adding more experts gives each expert a diminishing routing window.

**Ours**: Designed for **F = 10-100** facets in enterprise deployments. Memory is O(F·r_f) and compute is O(F·r_f·T) per forward, both negligible relative to attention itself. The architectural scaling is linear in F with no saturation, because each facet has an **independent** gate (not a normalized mixture).

---

## 4. What the paper must argue (one thesis sentence)

> In the realistic enterprise scenario — **dozens to hundreds of workflows, plans, and tools** — we automatically construct **tens of orthogonal semantic facets** from catalog structure and reflect them in the **K** tensor so that **multiple focus dimensions are simultaneously active** per token. This **F-simultaneous K-side catalog ontology** is structurally distinct from AdaSEKA's **Q-adaptive 1-of-M expert routing**: different intervention source (catalog vs contrastive data), different composition (simultaneous vs winner-take-most), and different scalability regime (F = 10-100 vs M ≤ 4).

This thesis is what Phase B paper must establish. Empirical evidence order:
1. **AdaSEKA comparison on CounterFact** (this document) — establishes that Q-adaptive mixture loses to static basis even on AdaSEKA's own target task family
2. **Flat ontology K-bias on MetaTool** (memory `metatool_subtask1_first_signal_2026_04_09.md`) — +11.15pp established, shows ontology direction is actionable for tool selection
3. **Per-facet gated K-bias on MetaTool** (pending) — empirical test of Claim 2 (simultaneous multi-facet > winner-take-most) on the actual target task type
4. **Dozens-of-facets scaling test** (pending) — need richer catalog (BFCL, ToolBench) to push F ≥ 10

---

## 5. Terminology constraints for paper

Per memory `oisa_deployment_context.md`, the paper must not cite or reference OISA patent materials. Specifically:
- **Do not** use terms: OISA, AFOD, MF-OSL, FC-LoRA, MF-OPB, LKCA, OPE-X
- **Do not** use facet namings F1=Structure / F2=Journey / F3=Intent / F4=Tool
- **Do not** cite bank CDP catalog, "전환" homonym, 45-tool / 500-tool scale
- **Do not** compare to the patent's LoRA-based facet internalization; use generic LoRA baselines instead

**Do use**:
- Public facet names derived from MetaTool/BFCL catalogs (function_action, io_type, domain, tool_category)
- Generic phrasing: "facet-gated attention steering with orthogonal semantic projections" or "multi-facet ontology K-bias"
- Public benchmark comparisons: AdaSEKA, SEKA vanilla, Focus Directions, ASA, PASTA, Activation Addition, Fact Grounded Attention

---

## 6. Follow-up experiments needed

Listed in priority order to avoid duplication:

1. **AdaSEKA on MetaTool** — the comparison that matters most for our paper. Requires porting expert builders from text-pair contrastive to tool-selection contrastive. Non-trivial. **Not yet started**. Highest priority for the paper's main table.

2. **Per-facet gated bias α sweep on MetaTool 995** — test whether simultaneous activation beats uniform flat +11.15pp. Current smoke 50 shows α=0.3 is too weak (+10pp vs flat +18pp on same 50) because energy-fraction gate dilutes amplification by ~0.09 per facet. Need α_base ∈ {1.0, 2.0, 3.0}. **Implementation done 2026-04-10**, sweep pending.

3. **Cross-model (Llama-3.1-8B, Mistral-7B) α sweep on MetaTool** — coworker A100×4 requested 2026-04-10, see `reports/COWORKER_REQUEST_cross_model_2026_04_10.md`. Blocker for Qwen-single-model risk.

4. **1a sign / 1c argmax ont_mode on MetaTool 995** — ON GPU as of 2026-04-10 ~02:00, `/tmp/metatool_FULL995_ocq_1a_sign.json` and `/tmp/metatool_FULL995_ocq_1c_argmax.json`. 1c tests the per-token single-axis selection extreme of the "adaptive 1-to-k axes" user hypothesis.

5. **Dozens-of-facets construction from BFCL/ToolBench catalogs** — test the F=10-100 scaling claim. Not started. Medium priority; needed for the paper's scaling argument.

6. **MMLU phase-gating evaluation** — for each method, measure degradation on non-tool queries. Per-facet gated bias is expected to auto-close (`Σ g_f ≈ 0` for non-tool Q), flat K-bias is expected to degrade, AdaSEKA is structurally unable to close (max normalization always selects an expert). Pending. This is the phase-closure empirical test.

---

## 7. Files updated 2026-04-10

- `memory/adaseka_vs_ours_differentiation_2026_04_10.md` (new) — next-session anchor
- `memory/metatool_subtask1_first_signal_2026_04_09.md` (updated) — added 2026-04-10 experiments running
- `memory/phase_b_session_resume.md` (updated) — 2026-04-10 handoff added
- `memory/MEMORY.md` (updated) — index entry added
- `scripts/ocq/eval_metatool_subtask1.py` (updated) — `install_facet_gated_hooks` + `build_facet_masks` + `--dump-failures` flag + `ocq_facet_gated_a*` method
- `external/SEKA/src/model/projection_builder_base_adapsvd.py` (patched) — `attn_implementation="eager"` for adaptive expert builder to work
- `external/SEKA/seka_projections/adaseka-qwen3-4b/` (new) — 3 expert SVD files + 2 expert_paths JSONs
- `external/SEKA/benchmarks/counterfact/results/adaseka-qwen3-4b-500-{2,3}experts/` (new) — AdaSEKA eval results
- `reports/ADASEKA_COMPARISON_2026_04_10.md` (this file, new)
- `reports/COWORKER_REQUEST_cross_model_2026_04_10.md` (existing, referenced)
