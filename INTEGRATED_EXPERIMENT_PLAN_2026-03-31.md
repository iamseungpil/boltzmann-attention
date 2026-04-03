# Integrated Experiment Plan

Date: 2026-04-03  
Scope: FOKVQ / Lie-group rotation interpretation / benchmark redesign

## 0. Evidence Base And Source Policy

### 0.1 Authoritative sources

- `results/v3/gpt2-medium_full_quant_ppl_v3.json`
- `reports/exp4_2_separate_report_ko.md`
- `reports/exp4_2_autoresearch_log.md` Iteration 24-29
- benchmark-smoke JSON files produced by the current harness version
- bounded Qwen mechanistic smoke on `2026-04-03`

### 0.2 Deprecated or non-authoritative artifacts

- `results/v3/qwen2.5-7b_full_quant_ppl_v3.json`
  - this file is an older failed run with hook/runtime errors
  - it must not be used as the source of truth for current Qwen claims

### 0.3 Reporting rule

- Every report table must name its source artifact class:
  - raw JSON
  - experiment log
  - smoke-only wiring check
- No practical claim may rely on a stale failed JSON when a later corrected log exists.

## 1. Updated Objective

The repository now has enough evidence to reject a naive paper framing:

> “Current FOKVQ implementation is a finished low-bit winner over all strong baselines.”

The realistic paper framing is instead:

> “Rotation-based KV-cache quantization can be interpreted as a Lie-group
> coordinate-selection problem, PCA is a strong covariance-aligned static
> baseline, `fokvq_e2` is the strongest current structured candidate, and
> current experiments show where this interpretation is supported, where
> long-budget practical retention still fails, and where Hamiltonian-style
> diagnostics can only be used descriptively.”

This plan is built around that narrower but defensible claim.

## 2. Claim Hierarchy

### 2.1 Positive claims currently supported

- K-space anisotropy is real.
- PCA is a strong static generator choice on GPT-2 surrogate experiments.
- Query dependence exists, but it is more plausibly a scheduling issue than a
  proof that the basis itself must be dynamic.
- In Qwen/GQA, practical strength currently comes from recency-preserving
  residual-tail baselines.
- In bounded Qwen mechanistic smoke, `fokvq_e2` is the strongest verified
  structured candidate so far.

### 2.2 Claims currently unsupported

- Current FOKVQ is SOTA on fair same-harness PPL
- Current methods already beat `kivi_residual` in long-budget practical PPL
- Qwen results prove the full practical method
- Hamiltonian language is already experimentally validated as a causal story

## 3. Benchmark Ladder

Each benchmark must map to a hypothesis.

### B1. Quality retention

**Intent**  
Measure how much pure LM quality is preserved.

**Hypothesis**  
The practical quantizer quality is visible in same-harness full-K PPL.

**Verification**
- WikiText-2 chunked/full-K PPL
- `fp16`, `uniform`, `kivi_residual`, `turboquant_rand`, target methods

### B2. Rotation mechanism

**Intent**  
Test whether covariance-aligned or Lie-structured bases are better than agnostic
bases before any systems-level claim.

**Hypothesis**  
`identity < random < PCA/structured basis` should hold at least in the stress regime.

**Verification**
- same-harness axis ablation
- required methods: `identity`, `random`, `fokvq`, target structured methods

### B3. Retrieval depth

**Intent**  
Check whether a basis preserves useful cache geometry across positions, not just
average next-token likelihood.

**Hypothesis**  
If the geometry is meaningful, deep-needle retrieval should degrade more
gracefully than with weaker bases.

**Verification**
- NIAH depth sweep
- at least 3 depth points
- multiple context lengths

### B4. Optional external generalization

**Intent**  
Only after B1-B3 are stable, test whether the effect survives broader long-context tasks.

**Hypothesis**  
A method that survives B1-B3 deserves evaluation on broader task suites.

**Verification**
- optional external long-context benchmark suite

### B5. Hamiltonian descriptive diagnostics

**Intent**  
Use conservative geometry diagnostics to decide whether Hamiltonian-style
language belongs in the paper body or only in supporting discussion.

**Hypothesis**  
Methods that better preserve quadratic pair energy and the canonical
symplectic-form proxy tend to retain bounded mechanistic PPL better.

**Verification**
- sampled post-RoPE K tensors
- energy drift
- phase drift
- symplectic-form drift
- only descriptive correlation, never causal proof

## 4. Active Hypotheses

### H1. Anisotropy is the prerequisite signal

**Intent**  
Verify that rotation is worth discussing at all.

**Hypothesis**  
A small number of principal directions explain a disproportionate amount of K variance.

**Verification**
- effective rank
- corpus separability gap
- head-wise anisotropy summary

**Failure interpretation**
- If this fails, the Lie/PCA paper center collapses.

### H2. PCA is a strong static basis

**Intent**  
Test the core mechanistic claim on static rotation choice.

**Hypothesis**  
PCA outperforms identity, random, and weaker alternatives on reconstruction or
same-harness axis ablations.

**Verification**
- KL / MSE surrogates on GPT-2
- same-harness axis ablation on GPT-2 and Qwen where possible

**Failure interpretation**
- If PCA does not beat weaker bases, the theory section must be narrowed.

### H3. The current bottleneck is the post-rotation quantizer

**Intent**  
Separate basis evidence from final practical ranking.

**Hypothesis**  
A basis-level effect can coexist with negative main-table results.

**Verification**
- compare basis ablations and main-table PPL side by side

**Failure interpretation**
- If both are negative, the current story is only an internal report.

### H4. `fokvq_e2` is the main constructive candidate on Qwen

**Intent**  
Promote the only structured candidate with a new verified bounded win.

**Hypothesis**  
`fokvq_e2` remains stronger than plain `fokvq` and weak controls when moved
from bounded mechanistic smoke toward medium-budget practical PPL.

**Verification**
- bounded mechanistic smoke
- medium-budget `ppl_quality`

**Failure interpretation**
- if the bounded advantage disappears immediately, the contribution is
  mechanistic rather than practical

### H5. Complex/unitary residual variants remain exploratory

**Intent**  
Focus the constructive search on candidates compatible with RoPE geometry.

**Hypothesis**  
Complex/unitary residual transforms may explain RoPE geometry better, but they
must now justify themselves against `fokvq_e2`, not just weak controls.

**Verification**
- self-test
- preset smoke
- full run only after both pass
- Hamiltonian diagnostics may support them descriptively, but do not rescue
  negative practical or mechanistic results

**Failure interpretation**
- If these also fail, the Lie framing remains descriptive rather than algorithmic.

### H6. Hamiltonian framing is descriptive until diagnostics align

**Intent**  
Prevent the theory section from overclaiming beyond current evidence.

**Hypothesis**  
If Hamiltonian-style diagnostics align with bounded mechanistic ranking, the
paper may use Hamiltonian language as explanatory support.

**Verification**
- diagnostic ranking must be stable
- diagnostic ordering should not strongly contradict PPL ordering

**Failure interpretation**
- if alignment is weak, Hamiltonian framing moves to motivation or appendix only

## 5. Execution Protocol

Every new wave must follow:

1. define `Intent / Hypothesis / Verification / Failure interpretation`
2. run self-test
3. run one smoke in the matching benchmark preset
4. fix any harness or launcher issue first
5. only then launch a real experiment
6. keep only if the result improves the target metric without breaking the guard
7. write the outcome into the autoresearch log

## 6. Current Priority Order

1. keep the paper aligned with supported claims
2. preserve benchmark alignment in code and logs
3. finish mechanistic evidence before optional broad generalization
4. only then continue remote autoresearch waves
5. treat Hamiltonian results as descriptive unless the diagnostics remain stable
