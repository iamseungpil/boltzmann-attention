# Unified Experiment Plan — Ontology-Based K-Bias Tool Selection

**Date**: 2026-04-13
**Contributors**: mais (develop branch), coworker (main branch)
**Goal**: Multi-facet ontology K-bias steering for enterprise tool selection

---

## 0. Current State Summary

### Our results (develop branch, 2026-04-13)

K-bias steering `K' = K + α·B_ont·B_ont^T·K` on MetaTool Subtask1 (995 queries):

| Model | no_steer | K-bias α=0.3 | Δ |
|-------|----------|-------------|---|
| Qwen2.5-7B | 75.58% | 86.73% | **+11.16pp** |
| Llama-3.1-8B | 80.60% | 90.85% | **+10.25pp** |
| Mistral-7B (fixed B_ont) | 61.01% | 56.68% | -4.32pp |

Key findings:
- K-bias works on Qwen (Mode C) and Llama (Mode A) — mode-agnostic
- Mistral failure root cause: min-truncation bottleneck in B_ont build (86%) + weak base model (14%)
- Fix: `--pad-to-max --target-layers 1-31` (skip pathological L0 heads)
- Cor 6.7 (phase-closure) dropped — architectural ceiling +0.0054

### Coworker results (main branch, 2026-04-08)

KV cache rotation-quantization interaction:

| Finding | Result |
|---------|--------|
| PCA ≈ Random for PPL | 3-bit: 5.691 vs 5.695 |
| Lloyd < Uniform always | 9/9 settings (3 models × 3 bits) |
| Per-head PCA >> Shared PCA | Llama 2-bit: 10.14 vs 18.87 (+46%) |
| D_attn ≠ PPL proxy | Inverse correlation |
| MMLU (Qwen 2-bit) | FP16 74.3% → NoRot 58.7% → PCA 67.9% |

---

## 1. Research Vision: Enterprise Multi-Facet Tool Selection

### 1.1 Problem

Enterprise environments have 45-1000+ tools with:
- **Name collision**: 3 different "conversion analysis" tools for marketing/analytics/executives
- **Multi-facet classification**: Same tool classified by purpose/function/structure/parameter independently
- **Denormalized metadata**: Inconsistent naming, path separators, ID-content mismatches

### 1.2 Our Approach: Ontology K-Bias Steering

Instead of stuffing 82K tokens of tool schemas into the prompt, **internalize the tool ontology into the model's K-space** via inference-time K-bias hooks. No fine-tuning needed.

Each tool is characterized by F facets (purpose, function, structure, parameter, domain, etc.). The K-bias projects attention onto facet-aligned subspaces, causing the model to "see" tool-relevant distinctions that it would otherwise miss.

---

## 2. Experiment Plan

### Phase 1: Foundation (DONE)

| # | Task | Status | Result |
|---|------|--------|--------|
| 1.1 | MetaTool Subtask1 single-turn eval | DONE | +11.16pp Qwen, +10.25pp Llama |
| 1.2 | Cross-model validation | DONE | 2/3 models positive |
| 1.3 | B_ont construction fix | DONE | pad-to-max + skipL0 |
| 1.4 | Cor 6.7 investigation | DONE | Dropped (architectural ceiling) |

### Phase 2: Ontology Redesign for Enterprise Facets (NEXT)

Current MetaTool ontology has 4 generic facets:
```
function_action (12 categories) — action verb: search/create/cancel...
io_type (6 categories)          — input/output schema type
domain (15 categories)          — high-level domain
tool_category (15 categories)   — CRUD-like grouping
```

**New enterprise-aligned facets** (aligned with OISA patent §5):

| Facet | Description | Example categories | Source |
|-------|-------------|-------------------|--------|
| **F1: Purpose** (도구의 목적) | What business goal does this tool serve? | analysis, monitoring, transaction, configuration, reporting | Tool docstring + name |
| **F2: Function** (도구의 기능/활용) | What computational action does it perform? | query, aggregate, filter, join, predict, generate | Tool implementation / action verbs |
| **F3: Structure** (도구의 구조) | What data structures does it operate on? | table, timeseries, graph, document, key-value | Input/output schema |
| **F4: Parameter** (도구의 파라미터) | What parameters does it accept? | date_range, entity_id, segment, metric_name, threshold | Tool signature |
| **F5: Domain** (도구의 도메인) | What business domain does it belong to? | marketing, finance, user_behavior, product, security | Namespace / team ownership |
| **F6: Stakeholder** (이해관계자) | Who typically uses this tool? | developer, marketer, analyst, executive, operator | Usage logs / team mapping |

**Experiments**:
- 2.1: Build 6-facet ontology from MetaTool + τ²-bench catalogs
- 2.2: Compare 4-facet vs 6-facet B_ont on MetaTool (does more facets help?)
- 2.3: Measure inter-facet orthogonality (NMI between facet assignments)
- 2.4: Per-facet contribution analysis (ablate one facet at a time)

### Phase 3: Multi-Facet Scoring for Tool Selection

**Core idea**: Each facet provides a "vote" for tool relevance. The final tool score is the sum of facet-aligned K-bias contributions.

```
score(tool_j | query) = Σ_f  α_f · sim(q, B_f · tool_j_embedding_f)
```

Where:
- `B_f` is the facet-f subspace basis
- `tool_j_embedding_f` is tool j's representation in facet f
- `α_f` is per-facet weight (learned or fixed)

**Experiments**:
- 3.1: Implement per-facet scoring in eval pipeline
- 3.2: Compare uniform α vs per-facet α on MetaTool
- 3.3: Test on τ²-bench retail (multi-turn, 20+ tools) with 6-facet ontology
- 3.4: Test on BFCL v3 (function calling benchmark)

### Phase 4: Integration with KV Cache Quantization (Coworker's Work)

**Connection**: Coworker proved per-head PCA >> shared PCA (+46% at 2-bit). Our B_ont IS a per-head ontology-aligned basis. Can B_ont serve dual purpose?

| Use case | Mechanism |
|----------|-----------|
| Tool selection steering | K' = K + α · B·B^T·K (amplify ontology directions) |
| KV cache compression | Quantize K in B_ont basis: ontology cols → 1-bit categorical, residual → KIVI |

**Experiments**:
- 4.1: OCQ (Ontology Cache Quantization) on Qwen/Llama with new 6-facet B_ont
- 4.2: Measure PPL with OCQ vs KIVI vs PCA-based quantization
- 4.3: Joint steering + quantization: can we steer AND compress simultaneously?
- 4.4: Compare B_ont-based quantization vs coworker's PCA-based quantization (MSE, L∞, PPL)

### Phase 5: Enterprise Deployment Validation

**Target**: Real CDP (Customer Data Platform) with 45 tools

| # | Task | Metric |
|---|------|--------|
| 5.1 | Build CDP ontology from tool catalog (4-8 facets) | NMI between facets |
| 5.2 | Homonym disambiguation ("전환" → marketing vs UX) | Disambiguation accuracy |
| 5.3 | Tool selection with K-bias on CDP queries | Top-1 accuracy, no_match rate |
| 5.4 | A/B test vs prompt-based tool selection | Latency, accuracy, token savings |

---

## 3. Priority and Timeline

### Week 1-2 (immediate)

| Priority | Task | Owner | GPU hours |
|----------|------|-------|-----------|
| P0 | Phase 2.1-2.2: 6-facet ontology build + eval | mais | 4h |
| P0 | Phase 3.1: Per-facet scoring implementation | mais | 2h |
| P1 | Phase 4.1-4.2: OCQ with new B_ont | coworker | 8h |
| P1 | Coworker: resolve PCA improvement cause (A-G) | coworker | 8h |

### Week 3-4

| Priority | Task | Owner |
|----------|------|-------|
| P0 | Phase 3.2-3.4: Multi-benchmark eval | mais |
| P1 | Phase 4.3: Joint steering + quantization | both |
| P2 | Phase 2.3-2.4: Facet orthogonality + ablation | mais |

### Week 5-8

| Priority | Task | Owner |
|----------|------|-------|
| P0 | Phase 5.1-5.4: CDP deployment validation | both |
| P1 | Paper writing: merge rotation-quantizer paper + K-bias paper | both |

---

## 4. Key Technical Decisions

### 4.1 B_ont Build Pipeline (Settled)

```bash
python build_qwen_metatool_b_ont.py \
  --model <MODEL> \
  --target-layers "1,2,...,31" \  # always skip L0
  --pad-to-max \                  # no min-truncation
  --ontology-json <6-FACET-ONTOLOGY>
```

### 4.2 Facet Design Principle (from OISA Patent)

Each facet must be:
1. **Independently classifiable**: A tool's purpose can change without changing its structure
2. **Orthogonality-verified**: NMI(F_i, F_j) < 0.3 for all i ≠ j
3. **Data-derivable**: Categories discovered from tool metadata, not hand-designed
4. **Gram-Schmidt separable**: After residualization, each facet captures unique K-space variance

### 4.3 Steering vs Quantization: Dual Use of B_ont

```
Inference-time K-bias hook:
  K_proj output → K + α·B·B^T·K → RoPE → attention

OCQ quantization hook:
  K_proj output → split into B^T·K (1-bit) + residual (KIVI n-bit) → store
  On retrieval: reconstruct K_approx → RoPE → attention
```

Both use the same B_ont. The difference:
- **Steering**: adds energy to ontology directions (α > 0)
- **Quantization**: preserves ontology directions at full precision, compresses residual

---

## 5. Success Criteria

| Milestone | Criterion | Deadline |
|-----------|-----------|----------|
| MetaTool 2-model | Qwen + Llama ≥ +8pp each | DONE |
| τ²-bench multi-turn | ≥ +5pp on retail domain | Week 3 |
| 6-facet vs 4-facet | 6-facet ≥ 4-facet on all benchmarks | Week 2 |
| OCQ + steering | PPL ≤ KIVI AND steering ≥ +5pp | Week 4 |
| CDP real deployment | Top-1 ≥ 85% on 45-tool selection | Week 6 |
| Paper submission | Unified rotation-quantizer + K-bias story | Week 8 |
