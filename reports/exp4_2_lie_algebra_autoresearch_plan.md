# Exp 4-2 Lie-Rotation Autoresearch Plan

Date: 2026-04-03  
Target: `Qwen2.5-7B`, `post_rope`, full-K quantization  
Scope: `3/4-bit` practical regime, `2-bit` mechanistic stress tests, and a
secondary Hamiltonian-style descriptive diagnostic track

## 0. Evidence Base

### 0.1 Authoritative artifacts

- `reports/exp4_2_autoresearch_log.md` Iteration 24-29 for the corrected Qwen queue
- `reports/exp4_2_separate_report_ko.md` for GPT-2 same-harness control tables
- current-harness smoke JSON files with `benchmark_meta`
- bounded Qwen mechanistic smoke on `2026-04-03`
  - `FP16=11.8666`
  - `fokvq_e2(2b)=12.4561`, `kivi_residual(2b)=13.1795`,
    `turboquant_rand(2b)=14.5626`
  - `fokvq_e2(3b)=11.5092`, `turboquant_rand(3b)=11.7115`,
    `kivi_residual(3b)=11.8421`

### 0.2 Deprecated artifact

- `results/v3/qwen2.5-7b_full_quant_ppl_v3.json`
  - contains older hook/runtime failures
  - keep for history only, never as the source of a current claim

## 1. Revised Intent

The current paper-safe question is no longer:

> Can the current FOKVQ implementation beat every strong KV-cache baseline?

The current paper-safe question is:

> Can a Lie-group view of rotation selection explain why some cache-coordinate
> transforms help and others fail, can `fokvq_e2` serve as the strongest current
> structured candidate, and do Hamiltonian-style geometry diagnostics help
> describe the surviving effect without overclaiming causality?

This matters because the current evidence is asymmetric:

- `GPT-2 / MHA`: strong support for anisotropy and PCA as a static basis
- `Qwen / GQA / RoPE`: long-budget practical pressure still favors `kivi_residual`
- bounded `Qwen` mechanistic smoke now shows `fokvq_e2` beating both
  `kivi_residual` and `turboquant_rand`
- therefore the real open problem is not “does PCA exist,” but
  “which structured generator and quantizer survive once we separate
  mechanism, practical retention, and geometry diagnostics”

## 2. Benchmark Ladder

Experiments must be matched to the hypothesis they test.

### B1. PPL Quality Benchmark

**Intent**  
Measure pure LM quality retention under full-K quantization.

**Benchmark**  
WikiText-2 chunked/full-K PPL.

**Why this benchmark**
- comparable to existing KV-cache papers
- cheap enough for iterative same-harness loops
- good at exposing catastrophic quantizer failures

**Methods required**
- `fp16`
- `uniform`
- `kivi_residual`
- `turboquant_rand`
- target Lie/FOKVQ candidates

### B2. Rotation Mechanistic Benchmark

**Intent**  
Test whether covariance-aligned or Lie-structured generators beat agnostic
bases before making any practical claim.

**Benchmark**  
Same-harness axis-selection comparison:
- `identity`
- `random`
- `fokvq`
- structured Lie candidates

**Why this benchmark**
- directly measures the axis-selection hypothesis
- prevents negative main-table results from being misread as “rotation never helps”

### B3. Retrieval Depth Benchmark

**Intent**  
Check whether a structured transform preserves useful cache geometry across
different needle depths, not only average next-token likelihood.

**Benchmark**
- NIAH depth sweep with multiple context lengths
- minimum 3 depths; 5 preferred

**Why this benchmark**
- the paper’s current interpretation is about geometry, not only scalar MSE
- depth sensitivity is a cleaner retrieval proxy than one averaged score

### B3.5. Hamiltonian Descriptive Diagnostic

**Intent**
Test whether methods that preserve RoPE-pair geometry also preserve simple
Hamiltonian-style invariants on sampled cache tensors.

**Benchmark**
- sampled post-RoPE K tensors from the real model
- descriptive metrics:
  - quadratic energy drift
  - pair-phase drift
  - symplectic-form drift

**Why this benchmark**
- it does not claim Hamiltonian dynamics are proved
- it gives a falsifiable descriptive check for the proposed interpretation
- it can explain why some structured methods survive bounded smoke while others fail

### B4. Optional External Generalization

**Intent**  
Only after B1-B3 are stable, test broader long-context task generalization.

**Benchmark**
- LongBench or RULER-style tasks

**Why optional**
- useful for paper scope later
- but not necessary for the current smoke-critic-improve loop

### B5. Metric Robustness Benchmark

**Intent**
Check whether PPL alone is masking or distorting the practically relevant
effect.

**Benchmark**
- WikiText-2 full-K PPL
- NIAH depth sweep
- sampled attention-logit distortion
- sampled top-k attention rank overlap

**Why this benchmark**
- PPL is sensitive to average next-token likelihood
- cache quantization may preserve retrieval geometry while still hurting token
  likelihood on unrelated positions
- a method can be mechanistically meaningful without being a PPL winner

## 3. Correctness Gate

No scientific interpretation is allowed until this gate passes.

### G0. Harness correctness

**Intent**  
Ensure the harness actually measures the intended benchmark.

**Hypothesis**
- full-K quantization is active
- patched methods produce finite outputs
- benchmark preset and method set are aligned

**Verification**
- `--self-test` passes
- small preset smoke passes
- no unknown-method failures
- benchmark metadata is written to JSON

**Failure interpretation**
- any failure invalidates the current wave; fix harness before running a new experiment

## 4. Scientific Tracks

### H1. PCA/static basis remains meaningful on Qwen only if it survives axis ablation

**Intent**  
Test whether the paper’s mechanistic story survives in the harder GQA setting.

**Hypothesis**  
Even if the current full FOKVQ method loses to `kivi_residual`, a structured
basis should still beat `identity` or `random` in the same harness.

**Verification**
- run `rotation_mechanistic` preset
- require:
  - finite `fp16`, `identity`, `random`, `fokvq`
  - target method improves over at least one agnostic basis

**Failure interpretation**
- if the target method does not beat `identity/random`, the Lie/PCA story is
  not yet empirically anchored on Qwen

### H2. Recent-tail preservation is a necessary inductive bias on Qwen

**Intent**  
Preserve the strongest current positive empirical signal.

**Hypothesis**  
Methods that quantize only the older prefix and keep a recent FP16 tail are
more stable than transforms applied uniformly to the whole cache.

**Verification**
- compare `fokvq_e2` vs `fokvq_e2_residual`
- compare `complex_unitary_residual`-family methods against non-residual transforms
- use both PPL and average key MSE

**Failure interpretation**
- if residual-tail preservation gives no stability advantage, the practical
  story must be revised away from “recent tail + structured prefix”

### H3. `fokvq_e2` is the current main constructive candidate

**Intent**
Promote the only structured candidate that has a new verified win on bounded
Qwen mechanistic smoke.

**Hypothesis**
`fokvq_e2` is stronger than plain `fokvq`, and on bounded Qwen mechanistic
evaluation it can outperform `identity`, `random`, `kivi_residual`, and
`turboquant_rand`.

**Verification**
- keep the `rotation_mechanistic` preset as the first gate
- require the bounded ranking to remain:
  - better than `identity/random`
  - better than or tied with other structured candidates
- only then promote to `ppl_quality`

**Failure interpretation**
- if `fokvq_e2` loses its bounded mechanistic edge after a harness fix, it
  falls back to a secondary structured baseline rather than the lead candidate

### H4. Complex/unitary generators are exploratory Lie candidates, not the lead path

**Intent**  
Test whether RoPE-aware complex structure adds value beyond the currently
stronger `fokvq_e2` baseline.

**Hypothesis**  
Complex Hermitian/unitary generators may outperform:
- pair-local RoPE heuristics
- weaker real-rotation controls
- but they must now also justify themselves against `fokvq_e2`

**Verification**
- first on `rotation_mechanistic` and the Hamiltonian descriptive diagnostic
- then on `retrieval_depth` only if smoke passes
- keep only methods that:
  - are finite
  - are monotone in bit-width
  - beat or tie `fokvq_e2` on at least one meaningful axis

**Failure interpretation**
- if complex/unitary variants fail against `fokvq_e2` and weak controls, the
  Lie story stays descriptive, not algorithmic

### H5. Main-table superiority is optional; basis-level evidence is mandatory

**Intent**  
Prevent scope creep and overclaiming.

**Hypothesis**  
The paper can still make a real contribution if:
- structured basis choices are empirically meaningful
- but the current low-bit quantizer is not yet strong enough to beat `kivi_residual`

**Verification**
- report both:
  - practical ranking (`kivi_residual`, `turboquant_rand`, `fokvq`, variants)
  - mechanistic ranking (`identity`, `random`, structured basis variants)

**Failure interpretation**
- if both rankings are negative, the current paper must collapse to an internal
  report rather than a methods paper

### H6. Hamiltonian diagnostics are descriptive support, not a practical claim

**Intent**
Use conservative geometry diagnostics to support or reject Hamiltonian-style
language in the paper.

**Hypothesis**
Methods with lower quadratic energy drift and lower symplectic-form drift on
sampled post-RoPE K tensors will tend to be the methods that survive bounded
mechanistic PPL better.

**Verification**
- compute Hamiltonian-style metrics on the same sampled K tensors used for smoke
- compare metric ranking against bounded mechanistic PPL ranking
- only claim descriptive alignment, never causal proof

**Failure interpretation**
- if the diagnostic ranking and PPL ranking disagree strongly, Hamiltonian
  language must be demoted to motivation or appendix only

### H7. The practical gap is caused by objective mismatch, not by absence of structure

**Intent**
Explain why a method can win bounded mechanistic ranking and still lose the main
practical table.

**Hypothesis**
`fokvq_e2` improves basis alignment, but practical performance is still limited
by one or more of:
- tail/recency mismatch
- scalar quantizer mismatch after rotation
- head-group mismatch under GQA
- benchmark mismatch between retrieval geometry and average PPL

**Verification**
- compare PPL, NIAH, attention-logit distortion, and Hamiltonian-style drift
- compare plain vs residual-tail variants
- compare grouped-KV behavior on Qwen heads

**Failure interpretation**
- if no diagnostic separates practical losers from practical winners, the
  current theory needs a narrower descriptive claim

## 5. Active Candidate Queue

### E-M1. FOKVQ-E2 practical promotion

**Intent**
Test whether the new bounded mechanistic winner survives when the evaluation
budget is raised toward practical same-harness PPL.

**Hypothesis**
`fokvq_e2` remains better than plain `fokvq` and stays competitive with
`kivi_residual` and `turboquant_rand` in at least one low-bit regime.

**Verification**
- self-test
- bounded `rotation_mechanistic` smoke
- medium-budget `ppl_quality`
- keep only if it still improves over plain `fokvq` and does not collapse

**Failure interpretation**
- if bounded success disappears immediately under a larger budget, the result is
  mechanistic evidence only, not a practical promotion

### E-H1. Hamiltonian descriptive probe

**Intent**
Attach conservative Hamiltonian-style diagnostics to the same smoke runs already
used for mechanistic ranking.

**Hypothesis**
`fokvq_e2` and stable residual methods preserve simple pairwise energy and
symplectic structure better than weak controls.

**Verification**
- add descriptive metrics to the harness
- require finiteness and stable ordering on small smokes

**Failure interpretation**
- if the metrics are unstable or uninformative, drop them from the main paper

### E-M2. Why-not-practical ablation

**Intent**
Directly test why bounded wins do not yet become practical wins.

**Hypothesis**
One of the following bottlenecks dominates:
- `A1` recency bottleneck: recent FP16 tail matters more than global basis quality
- `A2` quantizer bottleneck: Lloyd-Max on rotated axes is still not the right
  distortion objective for downstream attention
- `A3` GQA bottleneck: one KV head shared by many Q heads weakens the value of a
  static K-only rotated basis
- `A4` metric bottleneck: PPL underweights the retrieval and geometry benefit

**Verification**
- `A1`: compare `fokvq_e2` vs `fokvq_e2_residual`
- `A2`: add attention-logit distortion and top-k overlap on sampled chunks
- `A3`: inspect grouped Q-vs-KV covariance mismatch and groupwise drift
- `A4`: compare ranking on PPL vs NIAH vs geometry metrics

**Failure interpretation**
- if all four fail, the method is likely descriptive-only under current design

### E-M3. PPL validity check

**Intent**
Test whether PPL is the wrong primary metric for the claim we care about.

**Hypothesis**
Methods that preserve retrieval geometry or attention structure can be
undervalued by average next-token PPL.

**Verification**
- keep PPL as the main practical metric
- add:
  - NIAH depth profiles
  - sampled attention-logit MSE
  - sampled top-k attention overlap
- look for cases where PPL is flat or negative but retrieval/attention metrics
  improve

**Failure interpretation**
- if all metrics agree that the method loses, the gap is algorithmic rather
  than metric-related

### E-C1. Complex Unitary Residual

**Intent**  
Use a Hermitian covariance and unitary basis on complex RoPE channels while
preserving the recent FP16 tail.

**Hypothesis**  
Global complex unitary structure is more faithful than pair-local RoPE
heuristics and generic real rotations.

**Verification**
- self-test
- `ppl_quality` smoke
- `rotation_mechanistic` smoke
- keep only if it beats at least one weaker structured baseline

**Failure interpretation**
- no better than plain `fokvq_e2_residual` means the complex basis is not yet
  adding useful inductive bias

### E-C2. Banded Complex Unitary Residual

**Intent**  
Allow low- and high-frequency RoPE bands to use different complex generators.

**Hypothesis**  
One global unitary basis is too rigid; frequency-banded bases should preserve
RoPE structure better.

**Verification**
- self-test
- `rotation_mechanistic` smoke against:
  - `complex_unitary_residual`
  - `banded_complex_unitary_residual`

**Failure interpretation**
- no improvement means frequency banding is not the missing ingredient

### E-C3. Query-Metric Complex Residual

**Intent**  
Inject query relevance without falling back to fragile full query-weighted PCA.

**Hypothesis**  
The useful complex basis is not purely `Sigma_K`-aligned but query-metric aware.

**Verification**
- self-test must show basis sensitivity under anisotropic `q_cov`
- `ppl_quality` smoke against `complex_unitary_residual`

**Failure interpretation**
- if it changes the basis but not the metric, query-metric weighting is likely
  not the practical bottleneck

### E-C4. Future commutator-regularized candidate

**Intent**  
Test the strongest pure Lie claim only after E-C1 to E-C3 are stable.

**Hypothesis**  
Generators that better commute with the RoPE operator should degrade less.

**Verification**
- synthetic commutator check first
- real benchmark only after smoke success

**Failure interpretation**
- lower commutator with no metric gain means the Lie claim is descriptive only

## 6. Execution Loop

Each candidate must follow the same strict loop.

1. implement one change
2. run `--self-test`
3. run one small preset smoke
4. fix any harness failure
5. only then run a real benchmark
6. keep only if the result improves the target ranking without breaking the guard
7. log the result in `exp4_2_autoresearch_log.md`

## 7. Promotion Rules

### Promote to full run

- self-test passes
- preset smoke passes
- at least one of:
  - practical gain over `fokvq` / `fokvq_e2`
  - mechanistic gain over `identity` / `random`
  - descriptive Hamiltonian metrics that align with the mechanistic ranking

### Keep as mechanistic evidence only

- does not beat `kivi_residual`
- but clearly beats weaker basis controls or clarifies a failure mode

### Discard

- unstable
- benchmark misaligned
- no gain over weaker controls
- only improves MSE while PPL and retrieval stay unchanged or regress

## 8. Current Go / No-Go Reading

- `kivi_residual` remains the long-budget practical leader.
- bounded `rotation_mechanistic` evidence is now positive for `fokvq_e2`.
- `fokvq_e2` is the main constructive candidate to promote next.
- complex/unitary residual methods are still worth testing, but as exploratory
  challengers rather than the default lead.
- Hamiltonian diagnostics are allowed only as descriptive support.
- the next expansion must explain the practical gap explicitly rather than only
  searching for another variant.
- no new remote run should start until:
  - self-test passes
  - preset smoke passes
  - the candidate is logged with `Intent / Hypothesis / Verification / Failure interpretation`
