# Integrated Experiment Plan

Date: 2026-03-31
Scope: Boltzmann/FOKVQ precursor results + DHS/HEAT paper validation
Primary sources:
- `results/experiment_log_v2.md`
- `reports/boltzmann_definitive_report_ko.tex`
- `DHS_PATENT_EXPERIMENT_PLAN_v3.docx`
- `HEAT_PAPER_v2.docx`

## 1. Goal

This plan has two goals.

1. Close the current Boltzmann/FOKVQ line with a reproducible, internally consistent result set.
2. Validate the DHS/HEAT hypothesis as a paper-ready theory with a small number of decisive experiments before committing to large end-to-end system work.

The key practical constraint is that the repository already contains substantial precursor evidence for temperature profiles, facet structure, and quantization behavior, but it does not yet contain the core DHS experiments such as dual-exponential fitting, RoPE-kappa coupling, anchor insertion, and position-aware quantization.

## 2. Current Status

### 2.1 Completed precursor experiments

The following results already exist and should be treated as baseline assets, not rerun by default.

| Area | Existing result |
|---|---|
| Boltzmann temperature | `1-1` GPT-2 weak success, Qwen GQA failure |
| Specific heat / quantization | `1-2` C(q) is different but not better than variance-based allocation |
| FFN spectral claim | `1-3` rejected |
| Facet separability | `2-1` supported |
| PCA vs LDA | `2-2` PCA wins clearly |
| AFOD init | `2-4` PCA sufficient |
| Lie-group rotation | `3-1` PCA near-optimal |
| SSM distillation | `3-2` not practical at useful state sizes |
| SSM initialization | `3-3` spectral is not best |

### 2.2 Unresolved baseline issue

`Exp 2-3` is not stable at the reporting level. The current JSON indicates that query-dependent allocation differs from static allocation, but older report text still reflects a degenerate negative result. This must be fixed before any DHS or HEAT paper uses the quantization story as supporting evidence.

### 2.3 Missing DHS/HEAT core experiments

The repository does not yet contain dedicated experiment code or frozen results for:

- Dual-exponential attention-profile fitting
- RoPE-kappa coupling
- Residual-reheating intervention
- GQA multimodality under HEAT fitting
- Sequence-length valley-depth scaling
- Sink-count auto-selection
- Position-aware KV quantization
- Anchor insertion for Lost in the Middle
- End-to-end Pareto and latency benchmarking

## 3. Decision Strategy

The work should be staged with explicit gates.

- Gate A: If dual-exponential fitting is not consistently better than simple baselines, do not push DHS/HEAT as a central theory paper.
- Gate B: If at least one mechanism-level prediction among RoPE-kappa coupling, residual-reheating coupling, or GQA multimodality fails cleanly, keep DHS/HEAT descriptive and avoid strong causal wording.
- Gate C: If position-aware quantization and anchor-based mitigation do not improve task or PPL metrics, do not position DHS/HEAT as a practical KV-cache management method.

This means the theory should be validated before expensive end-to-end systems work on A100.

## 4. Recommended Phases

## Phase 0. Baseline Freeze

Window: 2026-03-31 to 2026-04-01
Hardware: local GPU or existing environment
Goal: clean up the current story so later DHS work has a stable baseline.

### P0-1. Re-run and freeze `Exp 2-3`

Purpose:
- Produce one authoritative result for query-dependent vs static allocation.

Tasks:
- Re-run the experiment from a fixed configuration.
- Save raw outputs, summary JSON, and a short interpretation note.
- Update report text so it matches the new JSON.

Success criterion:
- One result file, one interpretation, no contradiction between JSON and report.

### P0-2. Build a baseline index

Purpose:
- Create a one-page mapping from completed experiments to reusable metrics, code, and figures.

Output:
- experiment id
- script path
- result path
- main conclusion
- whether the result is reused by DHS/HEAT

## Phase 1. DHS Theory Go/No-Go

Window: 2026-04-02 to 2026-04-07
Goal: decide whether DHS/HEAT is strong enough to deserve a paper-centered push.

### P1-1. Dual-exponential superiority

Source mapping:
- DHS `Exp 1.1`
- HEAT Prediction 1

Models:
- GPT-2 Medium
- Qwen2.5-7B
- Llama-3-8B

Comparison baselines:
- Poly-2
- Poly-4
- single-exponential
- optional: cosh, two-Gaussian if implementation cost is low

Primary metrics:
- per-head R^2
- AIC or BIC win rate
- fraction of heads where dual-exponential wins

Success criterion:
- GPT-2: dual-exponential beats Poly-4 for at least 80 percent of heads
- GQA models: dual-exponential beats Poly-4 for at least 70 percent of heads

Interpretation:
- This is the main Go/No-Go gate.

### P1-2. Sequence-length scaling

Source mapping:
- DHS `Exp 5.1`
- HEAT Prediction 5

Models:
- GPT-2 Medium first
- Llama-3-8B second if Phase 1 stays alive

Lengths:
- 256, 512, 1024, 2048, 4096
- 8192 only on the large-model track

Metric:
- Spearman correlation between log sequence length and valley depth

Success criterion:
- strong positive correlation, ideally above 0.8

Reason:
- This is cheaper than full end-to-end evaluation and directly checks a unique DHS prediction.

### P1-3. GQA multimodality

Source mapping:
- DHS `Exp 4.1`
- HEAT Prediction 4

Model:
- Llama-3-8B

Metric:
- per-Q-head fit quality
- per-KV-group aggregated profile shape
- multimodality test on aggregated distributions

Success criterion:
- per-Q-head remains roughly U-shaped
- averaged KV-group profile becomes multi-modal or clearly non-simple-U

Reason:
- This is the cleanest bridge from existing Qwen GQA findings to the DHS paper.

## Phase 2. Mechanism Validation

Window: 2026-04-08 to 2026-04-11
Goal: strengthen the paper from descriptive fit to mechanism-level evidence.

### P2-1. RoPE-kappa coupling

Source mapping:
- DHS `Exp 2.1`
- HEAT Prediction 2

Model family:
- Llama-3-8B variants preferred

Metric:
- correlation between RoPE base frequency and fitted recency decay constant kappa_2

Success criterion:
- negative correlation with practically meaningful effect size

### P2-2. Residual-reheating coupling

Source mapping:
- DHS `Exp 3.1`
- HEAT Prediction 3

Model:
- GPT-2 Medium

Intervention:
- scale residual strength at inference

Metric:
- correlation between residual scale and reheating magnitude

Success criterion:
- monotone positive relationship

Interpretation:
- If this fails, the reheating story should remain descriptive rather than causal.

## Phase 3. Practical Value Gate

Window: 2026-04-12 to 2026-04-17
Goal: test whether DHS/HEAT has practical value beyond explanation.

### P3-1. Position-aware quantization

Source mapping:
- DHS `Exp 7.1`
- HEAT practical implication on KV quantization

Models:
- GPT-2 Medium
- Qwen2.5-7B if GPT-2 result is promising

Baselines:
- uniform
- variance-based baseline from existing work
- KIVI-style position-agnostic setting
- optional combined FOKVQ + DHS if single-axis result is positive

Metrics:
- KL divergence
- WikiText-2 PPL
- stability under 2, 3, 4-bit settings

Success criterion:
- clear improvement over uniform at at least one useful bit budget without instability

Decision:
- If this fails, do not pitch DHS/HEAT as a quantization method.

### P3-2. Sink-count auto-selection

Source mapping:
- DHS `Exp 6.1`

Goal:
- test whether k derived from kappa_1 is better than fixed k=4

Metrics:
- retained quality under sink-preserving compression
- per-head or per-layer agreement with optimal retained length

Success criterion:
- at least modest but consistent gain over fixed-k heuristic

### P3-3. Layer-wise KV budget allocation

Source mapping:
- DHS `Exp 9.1`

Goal:
- compare U-shaped or T_eff-based layer budgeting against pyramid heuristics

Metrics:
- memory-budgeted PPL
- layerwise degradation pattern

Success criterion:
- equal or better quality under the same global cache budget

## Phase 4. Lost-in-the-Middle Intervention

Window: 2026-04-18 to 2026-04-22
Goal: test the strongest practical narrative for the paper.

### P4-1. Anchor insertion at the profile level

Source mapping:
- DHS `Exp 8.1`

Goal:
- verify that inserted anchors convert U-shaped profiles toward W-shaped profiles

Metrics:
- local peak formation at anchor positions
- valley-depth reduction

Success criterion:
- clear profile-shape change in the predicted direction

### P4-2. Anchor spacing from DHS parameters

Source mapping:
- DHS `Exp 8.2`

Goal:
- compare DHS-derived spacing against simple evenly spaced or oracle-lite variants

Success criterion:
- DHS spacing is competitive with or better than naive spacing

### P4-3. Lost in the Middle QA evaluation

Source mapping:
- DHS `Exp 8.3`

Models:
- Qwen2.5-7B or Llama-3-8B

Tasks:
- Lost in the Middle reproduction
- optional RULER follow-up if the base result is positive

Success criterion:
- middle-position accuracy improves meaningfully without major degradation at the ends

Decision:
- This is the highest-value practical result for a DHS/HEAT paper.

## Phase 5. End-to-End Systems Validation

Window: 2026-04-23 onward
Goal: only start if both theory and practical gates have passed.

### P5-1. Pareto evaluation

Source mapping:
- DHS `Exp 10.1`

Model:
- Qwen2.5-7B on A100

Metrics:
- memory
- latency
- PPL
- optional task metric such as MMLU or long-context QA

Compare against:
- full cache
- sink + recent window baselines
- quantization-only baseline
- anchor-only baseline
- integrated DHS configuration

### P5-2. O(1) latency claim

Source mapping:
- DHS `Exp 10.2`

Goal:
- verify that the proposed priority function is effectively constant-cost at runtime relative to dynamic scoring baselines

Decision:
- If the latency advantage is weak, keep the claim narrow and avoid systems-heavy positioning.

## 5. Hardware Plan

### Local / A6000 track

Use for:
- GPT-2 experiments
- baseline cleanup
- fitting code development
- early quantization prototypes

### A100 track

Use for:
- Llama-3-8B
- Qwen2.5-7B long-context runs
- Lost in the Middle evaluation
- Pareto and latency benchmarks

Practical rule:
- Do not spend A100 time until Phase 1 produces a positive signal.

## 6. Priority Order

If only six experiments can be done, run them in this order.

1. `Exp 2-3` freeze and cleanup
2. Dual-exponential superiority
3. GQA multimodality
4. RoPE-kappa coupling
5. Position-aware quantization
6. Lost in the Middle with anchors

This order maximizes information gain while keeping large-model cost under control.

## 7. Deliverables

Each experiment should produce the same minimal package:

- one script or notebook entry point
- one result JSON with raw metrics
- one short markdown interpretation
- one figure-ready CSV or image
- one line verdict: PASS, PARTIAL, FAIL, or INCONCLUSIVE

At the end of each phase, produce a phase summary page with:

- what passed
- what failed
- what changes in the paper claim
- whether to continue to the next phase

## 8. Final Recommendation

Do not start from the full 19-experiment DHS plan.

The right strategy is:

1. freeze the current baseline,
2. run the three theory-critical DHS experiments,
3. run one quantization experiment and one Lost-in-the-Middle intervention,
4. only then decide whether the paper is a theory paper, a descriptive analysis paper, or a practical KV-cache paper.

At the current state of the repository, the most likely efficient outcome is not a full patent-style program, but a narrower paper built around:

- dual-exponential fit quality,
- GQA as a boundary condition,
- one successful practical implication if available.
