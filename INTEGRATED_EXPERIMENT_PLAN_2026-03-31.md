# Integrated Experiment Plan

Date: 2026-04-03  
Scope: FOKVQ / Lie-group interpretation / DHS-adjacent evidence  
Primary sources reviewed:
- `FOKVQ_PAPER_v3.docx`
- `FOKVQ_PAPER_v6.docx`
- `FOKVQ_FINAL_REPORT.docx`
- `FOKVQ_ADDITIONAL_EXPERIMENT_PLAN_v9.docx`
- `DHS_PAPER_v1.docx`
- `DHS_PAPER_v2.docx`
- `DHS_PATENT_EXPERIMENT_PLAN_v3.docx`
- `M-SSM_Universal_Framework.docx`
- current LaTeX draft under `paper/neurips2026_ko/`

## 1. Revised Paper Intent

The paper should no longer be organized around the strongest possible headline
`FOKVQ beats all KV-cache baselines`.

The paper should instead answer the following narrower but currently supported
question:

> Can KV-cache quantization be interpreted as a Lie-group coordinate-selection
> problem, and does the empirical evidence support PCA as a strong
> covariance-aligned static rotation even when the final low-bit quantizer is
> not yet state of the art?

This reframing matches the current evidence better than a pure systems-SOTA
story.

## 2. Claim Hierarchy

### 2.1 Claims we can already support

- K-space anisotropy is real and strong.
- Static rotation choice matters.
- PCA is a strong static baseline for that rotation choice.
- Query dependence exists, but it opens a scheduling problem more than a
  basis-relearning requirement.
- Current low-bit PPL gaps are more plausibly a post-rotation quantizer problem
  than a proof that covariance-aligned rotation is useless.

### 2.2 Claims we must not overstate

- `FOKVQ is SOTA on fair common PPL tables`
- `Lie-group framing alone guarantees better perplexity`
- `Qwen/GQA results already validate the full practical method`
- `MK-style universal theory is empirically established in this repo`

## 3. Working Hypotheses

All new experiments should be written in the same format:

- `Intent`: why the experiment exists
- `Hypothesis`: what measurable statement is being tested
- `Verification`: exact metrics and acceptance rule
- `Failure interpretation`: what conclusion to draw if the hypothesis fails

## 4. Active Experiment Tracks

### Track H1. Anisotropy is the prerequisite signal

**Intent**  
Establish that rotation-based reasoning is justified at all.

**Hypothesis**  
Head-wise key covariance is sufficiently anisotropic that a small number of
principal directions explain a disproportionate fraction of variance.

**Verification**
- effective-rank statistics across layers/heads
- within-corpus vs cross-corpus separability gap
- fraction of heads satisfying a pre-registered anisotropy threshold

**Failure interpretation**
- If anisotropy is weak, the Lie/PCA story collapses into a cosmetic rewrite of
  uniform quantization and should not be the paper center.

**Current status**
- Supported by existing GPT-2 assets.

### Track H2. PCA is a strong static generator choice

**Intent**  
Test the core mathematical reading of FOKVQ: not “PCA is magical,” but
“covariance-aligned generators outperform agnostic generators on static
surrogates.”

**Hypothesis**  
PCA basis outperforms identity, random rotation, LDA, and continuous
rotation-search baselines on reconstruction/quantization surrogates.

**Verification**
- KL divergence after quantization
- K reconstruction MSE
- axis ablation: `identity vs random vs PCA`
- optimization baseline: `continuous search vs PCA`

**Success criterion**
- PCA must beat identity/random clearly and beat or tie search-based baselines
  on the same surrogate.

**Failure interpretation**
- If PCA does not beat random or search-based baselines, the Lie-group framing
  cannot justify singling out covariance-aligned generators.

**Current status**
- Supported on GPT-2.

### Track H3. Query dependence is a scheduling signal, not yet a basis signal

**Intent**  
Prevent the paper from collapsing into the stronger unsupported claim that the
basis itself must be query-dependent.

**Hypothesis**  
Query-specific importance changes are real, but most of the gain can still be
described as dynamic allocation on top of a static basis.

**Verification**
- static-vs-query rank correlation
- allocation L1 distance
- query-vs-query variance
- optional: fixed-basis/dynamic-schedule vs dynamic-basis comparison

**Failure interpretation**
- If dynamic basis selection is required to recover performance, the current
  static-PCA paper must be narrowed further.

**Current status**
- Partially supported; basis-specific follow-up still useful.

### Track H4. The current bottleneck is the quantizer, not necessarily the basis

**Intent**  
Separate the empirical fate of the basis from the empirical fate of the
current FOKVQ implementation.

**Hypothesis**  
PCA-aligned rotation remains beneficial in axis ablations even when the full
FOKVQ quantizer loses to stronger low-bit baselines such as KIVI-style methods.

**Verification**
- same-harness standard PPL
- same-harness axis ablation
- compare:
  - `fp16`
  - `uniform`
  - `kivi-style`
  - `turboquant-style`
  - `identity`
  - `random`
  - `fokvq`

**Success criterion**
- Main-table superiority is not required.
- What is required is evidence that `PCA > random/identity` survives under a
  reasonably fair protocol.

**Failure interpretation**
- If PCA does not help even in axis ablations, the paper cannot claim that the
  Lie/PCA interpretation is empirically meaningful.

**Current status**
- Supported more strongly on surrogate metrics than on end-to-end PPL.

### Track H5. Lie-group extensions beyond plain PCA

**Intent**  
Explore whether the Lie framing generates new experimentally testable
variations rather than serving as post-hoc language only.

**Hypothesis**  
Structured generators informed by complex/block/unitary geometry can outperform
agnostic random generators, even if they do not yet beat KIVI.

**Verification**
- smoke: orthogonality/unitarity and finite reconstruction
- surrogate metrics before full PPL
- fair same-harness PPL only after smoke success

**Candidate methods**
- complex unitary residual
- banded complex unitary residual
- commutator-regularized basis
- RoPE-generator-aware structured rotations

**Failure interpretation**
- If all structured Lie variants fail to beat random or PCA on surrogate
  metrics, the paper should keep Lie framing descriptive rather than
  constructive.

## 5. Execution Order

### Phase 1. Freeze the current paper-safe evidence

**Intent**  
Lock down only the results that the current paper actually relies on.

**Hypothesis**  
The paper can stand on H1-H4 even if practical low-bit SOTA is not achieved.

**Verification**
- fact base matches actual result files
- paper text contains no unsupported positive claim

### Phase 2. Strengthen the basis-vs-quantizer separation

**Intent**  
Make the key interpretive claim hard to attack.

**Hypothesis**  
Negative main-table results can coexist with positive basis-level evidence.

**Verification**
- add or refresh axis-ablation figures/tables
- make sure interpretation is consistent across GPT-2 and Qwen runs

### Phase 3. Search for constructive Lie variants

**Intent**  
Attempt to turn the Lie perspective into an algorithmic gain.

**Hypothesis**  
At least one structured generator family beats random rotation and narrows the
gap to KIVI/turboquant-style baselines.

**Verification**
- bounded smoke -> critic -> improve loop
- only keep candidates that survive deterministic checks and common-harness PPL

### Phase 4. Decide final paper type

**Intent**  
Choose the right submission story from evidence, not preference.

**Decision rules**
- If a Lie variant wins convincingly on fair PPL:
  promote the paper toward `Lie-guided KV quantization method`.
- If only PCA/random separation survives:
  submit as `theory + empirical analysis of rotation choice`.
- If even that separation weakens:
  reduce to a descriptive/internal report, not a flagship methods paper.

## 6. Required Reporting Template

Every new experiment note should include the following block:

```text
Experiment:
Intent:
Hypothesis:
Protocol:
Verification:
Observed result:
Decision:
Failure interpretation:
Next action:
```

This is mandatory for all new plan documents and autoresearch logs.

## 7. Immediate Next Paper-Aligned Tasks

1. Keep the NeurIPS draft centered on Lie-group justification plus verified
   empirical support.
2. Use standard PPL tables as constraint evidence, not overclaim fuel.
3. Continue testing structured Lie variants only under fair same-harness
   evaluation.
4. Update every downstream plan file to use `Intent / Hypothesis / Verification /
   Failure interpretation`.
