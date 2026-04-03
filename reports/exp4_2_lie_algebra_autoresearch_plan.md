# Exp 4-2 Lie-Algebra Autoresearch Plan

Date: 2026-04-02
Target: `Qwen2.5-7B` `3/4-bit` full-quant PPL
Primary baseline to beat: `kivi_residual`
Secondary baselines: `fokvq`, `fokvq_e2`, `turboquant_rand`, `quip`

## 1. Goal

Find out whether a Lie-algebra-inspired orthogonal transform can beat
`kivi_residual` on the practically relevant `Qwen/GQA` `3/4-bit` regime.

This is explicitly narrower than the older “PCA beats everything” question.
The current evidence already shows:

- `GPT-2/MHA`: structured rotation is meaningful
- `Qwen/GQA`: `kivi_residual` is the strongest current same-harness control
- therefore the next useful question is whether a better `SO(d)` transform,
  not naive PCA alone, can close the remaining gap on `Qwen`

## 2. Working Hypotheses

### H-C0. Correctness Gate

Before any scientific comparison, the harness must satisfy:

- `16-bit patched == fp16`
- smoke output must be finite
- `4-bit` must not be worse than `3-bit` by a large margin for a stable method

If this gate fails, the wave is invalid and must be discarded.

### H-L1. Robust Lie Equalization

Naive variance equalization is too sensitive to outliers.
A robust covariance estimate before constructing the Givens rotation may
improve `3/4-bit` behavior.

Candidate method:
- `lie_eq_robust`

Acceptance signal:
- beats plain `lie_eq`
- does not regress badly against `fokvq_e2`
- stays within `+0.5 PPL` of `kivi_residual` at either `3-bit` or `4-bit`

### H-L2. Query-Safe Lie Weighting

Full Q-weighted PCA is GQA-fragile as a main method because query-head pooling
can collapse structure. But a weaker query-aware diagonal weighting may still
be safe.

Candidate method:
- `lie_qdiag`

Mechanism:
- pool query energy into a per-coordinate importance for each KV head
- construct a Lie transform using weighted K statistics rather than raw K

Acceptance signal:
- beats `fokvq` and `lie_eq` on `Qwen` `3-bit`
- remains finite and stable under GQA
- materially improves over `lie_eq_robust`

### H-L3. Robust + Query-Safe Combined Lie Transform

If the weak point is both outlier sensitivity and lack of query relevance,
combine the two.

Candidate method:
- `lie_qdiag_robust`

Acceptance signal:
- best among the Lie variants
- at least approaches `kivi_residual` closely at `3/4-bit`

### H-C1. Public-Gap Control Revision

The old same-harness `kivi` proxy was too weak. A more public-faithful
control with a residual FP16 tail changes the ranking.

Acceptance signal:
- `kivi_residual` beats old `kivi` on `Qwen` smoke
- and changes at least one prior ranking against `fokvq` or `fokvq_e2`

### H-C2. Random-Rotation Turbo Control

The Hadamard proxy may understate or distort the TurboQuant-style control.
A seeded random orthogonal rotation is a better same-harness proxy.

Acceptance signal:
- `turboquant_rand` is no worse than deterministic `turboquant`
- and is competitive with the best non-Lie control after `kivi_residual`

## 3. Iteration Protocol

Each iteration must follow the same loop.

1. implement one candidate
2. run self-test
3. run `Qwen 2048-token` smoke
4. compare against:
   - `kivi_residual`
   - `fokvq`
   - `fokvq_e2`
   - `lie_eq`
5. keep only if:
   - finite PPL
   - no harness breakage
   - empirical improvement over at least one relevant structured baseline
6. if promising:
   - schedule `16k` run
7. if not promising:
   - discard from the main queue

## 4. Ranking Rules

### Keep as main candidate

- improves `Qwen 3-bit` PPL over both `fokvq` and `fokvq_e2`
- and either beats `kivi_residual` or narrows the gap enough to justify a full run

### Keep as mechanistic ablation

- does not beat `kivi_residual`
- but teaches something clear, for example:
  - robust stats help
  - query-safe weighting helps
  - equalization hurts high-bit reconstruction

### Discard

- unstable
- clearly worse than `lie_eq` and `fokvq_e2`
- too slow relative to the information gained

## 5. Paper Update Triggers

### If a Lie method beats `kivi_residual` on `Qwen 3/4-bit`

Update paper claim to:
- PCA is not the final answer
- a structured `SO(d)` transform with the right inductive bias can outperform
  both naive PCA and stronger non-rotational baselines in the GQA setting

### If a Lie method helps but does not beat `kivi_residual`

Update paper claim to:
- `SO(d)` structure matters
- but the practical frontier is conditional and architecture-sensitive
- the contribution is mechanistic rather than SOTA-like

### If all Lie methods fail

Update paper claim to:
- structured rotation alone is insufficient at `Qwen 3/4-bit`
- the main scientific value is the boundary condition itself:
  MHA and GQA reward different quantization geometries

## 6. Immediate Queue

Run in this order:

1. integrate the latest remote safety fix review
2. close the outstanding `Qwen 16k` control/main runs
3. discard generic real-Lie candidates from the practical queue
4. test a first `RoPE-pair-preserving` unitary candidate
5. test residual-tail hybridization if the unitary candidate fails
6. test magnitude-phase quantization if the residual hybrid still trails
7. only if one candidate is promising:
   - promote to a wider `Qwen` compare run

## 6.1 Active 4-GPU Wave on `tops-caiman`

Current bounded queue after the Qwen eager-wrapper fix and the stronger
`kivi_residual` control revision:

- `GPU0`: `16384-token` full run still active
  - now completed
- `GPU0`: current smoke active
  - `fp16`, `kivi_residual`, `fokvq_e2_residual`, `rope_magphase`
- `GPU1`: previous smoke completed
  - `fp16`, `kivi_residual`, `fokvq_e2`, `fokvq_e2_residual`
- `GPU2`: currently free
  - reserved for the next promoted candidate
- `GPU3`: `16384-token` full run
  - now completed: `fp16`, `kivi_residual`, `quip`, `turboquant`, `turboquant_rand`

Decision rule for the next wave:

- if `rope_unitary` cannot materially improve over `fokvq` or close the gap to
  `kivi_residual`, discard it from the practical queue and reclaim `GPU1`
  for:
  - `residual-tail + structured transform`
  - or `magnitude-phase` quantization
- active next check:
  - `rope_magphase` failed catastrophically and is discarded
  - current next check:
    - test whether `complex_unitary_residual` can beat `fokvq_e2_residual`
    - and whether a Hermitian/unitary complex basis is better aligned with
      the intended Lie interpretation than direct phase quantization
- if `rope_unitary` is promising, promote only that candidate to a wider run
  before adding new complexity

## 6.2 Updated Control Interpretation

The baseline audit changed the queue.

- old `kivi` should now be treated as a weak same-harness proxy
- new `kivi_residual` is still not the official KIVI implementation, but it is a
  materially more faithful K-only proxy because it preserves a recent FP16 tail
- therefore the practical Qwen target is no longer just `kivi`, but primarily:
  - `kivi_residual`
  - then the best of `quip`, `kvquant`, `turboquant_rand`

Immediate implication:

- if a Lie method cannot challenge `kivi_residual` on `Qwen 3/4-bit`, it should
  not be carried as a practical winner
- it may still survive as a mechanistic result if it reveals a geometry effect
  that differs from PCA and from KIVI-style residual preservation

## 6.3 Validated Findings So Far

These are no longer speculative.

- `Qwen 2048-token smoke`
  - `fp16 = 6.0769`
  - old `kivi`: `7.3729 / 6.4674` at `3 / 4` bit
  - `kivi_residual`: `6.5734 / 6.1431`
  - `fokvq`: `9.7439 / 6.9743`
- `rope_unitary`: `14.3512 / 13.9959`
- `fokvq_e2`: `7.7454 / 7.3983`
- `fokvq_e2_residual`: `7.4612 / 7.4540`
- `rope_magphase`: `750.0469 / 424.5279`
- `Qwen 16k complete`
  - `fp16 = 6.7544`
  - `kivi_residual`: `7.2060 / 6.7738`
  - `fokvq`: `9.6555 / 7.4720`
  - `fokvq_e2`: `8.7232 / 8.4917`
  - `quip`: `9.1795 / 7.1394`
  - `turboquant`: `7.5671 / 7.0900`
  - `turboquant_rand`: `7.5587 / 7.0212`

Interpretation:

- the old `kivi` proxy understated the strength of the KIVI family
- `kivi_residual` is the correct practical control for the current same-harness queue
- plain `fokvq` is not competitive with that control on `Qwen 3-bit`
- current real-Lie candidates (`lie_qdiag`, `lie_qdiag_robust`) are too weak to
  remain in the practical frontier queue
- `rope_unitary` is also practically dead despite respecting pair locality
- `rope_magphase` shows that naive direct phase quantization is not the right
  way to operationalize the complex-rotation hypothesis
- `fokvq_e2_residual` helps at `3-bit`, but not enough to challenge the control
- the remaining open practical questions are:
  - whether a true complex/unitary basis can do better than pairwise rotation
  - whether any RoPE-aware candidate can close the remaining gap to `kivi_residual`

## 6.4 Literature-Driven Next Hypotheses

The next wave should not just retry the same PCA family. Recent KV-cache and
rotation literature suggests several distinct directions:

### H-N1. Residual-Tail + Rotation Hybrid

Current evidence suggests that preserving a recent FP16 tail is a very strong
control on `Qwen`. A natural next step is to combine:

- `kivi_residual` style recent-token preservation
- with a structured rotation on the older quantized prefix only

Candidate:
- `fokvq_residual`
- `lie_qdiag_residual`

Why this is plausible:
- KIVI-style residual preservation is empirically strong in the current queue
- rotation may help only on the older, harder-to-preserve region rather than on
  the entire cache

Acceptance signal:
- beats plain `kivi_residual` at either `3-bit` or `4-bit`

### H-N2. Learned / Calibrated Rotation Instead of Plain PCA

Rotation literature suggests that random rotations vary widely in quality and
that learned or calibrated rotations can outperform naive fixed choices.

Candidate:
- calibration-optimized orthogonal basis on a small held-out set
- low-rank or Cayley-style update starting from PCA or random rotation

Acceptance signal:
- beats plain `fokvq` and `turboquant_rand`
- narrows the gap to `kivi_residual`

### H-N3. Pivot / Salient Token Preservation

Some recent quantization work keeps a small critical token subset in higher
precision rather than treating every token uniformly.

Candidate:
- preserve a tiny pivot-token set plus quantize the rest
- combine token preservation with `fokvq_e2`

Acceptance signal:
- reduces the `3-bit` gap without large runtime cost

### H-N4. Head-Grouped Key Compression

Recent post-training KV compression work uses head grouping/reordering before
compression. That suggests our per-head independent basis may be too local.

Candidate:
- group KV heads by similarity
- fit a shared basis or shared codebook per group

Acceptance signal:
- improves `Qwen 3-bit` over per-head `fokvq_e2`

### H-N5. Hybrid Quantization + Low-Rank Residual

GEAR and ReCalKV indicate that pure scalar quantization may not be enough when
important residual structure is low-rank.

Candidate:
- `fokvq_e2 + rank-r residual`
- `kivi_residual + rank-r residual`

Acceptance signal:
- practical improvement at `3-bit` with modest runtime overhead

## 6.5 Complex-Rotation / Unitary Direction

Document review across the folder suggests that the strongest shared mathematical
object is not a generic real rotation, but an exponential map with phase
structure:

- `FOKVQ_PAPER_v3/v6`: quantization is framed as a Lie-group rotation
- `DHS_PAPER_v2` and `HEAT_PAPER_v2`: RoPE is interpreted as a rotation matrix
  whose frequency controls recency decay
- `M-SSM` and `M-SSM_Universal_Framework`: Transformer position encoding is
  written explicitly as `exp(i Θ · pos) ∈ U(d)` and tied to the same family as
  `exp(iH t)` and other exponential-map models

This suggests the current failed Lie variants may be using the wrong symmetry
class on `Qwen`.

### Core Diagnosis

Current `lie_eq` / `lie_qdiag` operate as generic real-valued `SO(d)` transforms.
That is plausible on `GPT-2` where there is no RoPE phase geometry, but on
`Qwen` the head dimensions come in paired coordinates that already represent
complex phases. A generic real rotation can mix unrelated pairs and destroy the
RoPE-compatible phase structure.

### H-U1. RoPE-Commuting Unitary Rotation

Restrict the transform to blockwise `SO(2)` / complex `U(d/2)` rotations that
respect the RoPE pairing.

Implementation sketch:
- reshape each head from `d` real dimensions into `d/2` complex channels
- parameterize the transform as block-2x2 rotations or a unitary matrix
- optionally constrain it to commute approximately with the RoPE generator `Θ`

Acceptance signal:
- beats current real-valued Lie variants on `Qwen`
- narrows the gap to `kivi_residual`

### H-U2. Magnitude-Phase Split Quantization

Treat each RoPE pair as a complex number `z = r e^{iφ}` and quantize:

- magnitude `r` with non-uniform / Lloyd-Max style allocation
- phase `φ` with circular quantization or protected high precision

Rationale:
- the documents repeatedly tie position information to phase
- current PCA-style methods only preserve Euclidean variance, not angular error

Acceptance signal:
- reduces long-range attention degradation at `3-bit`
- improves `Qwen` more than `GPT-2`, which would support the RoPE-phase claim

### H-U3. Commutator Test

Test whether respecting the RoPE generator matters by comparing:

- unconstrained real rotation
- complex/unitary rotation
- complex rotation with commutator penalty `||[H, Θ]||`

Interpretation:
- if lower commutator correlates with better PPL, that directly supports the
  complex-rotation Lie interpretation instead of a generic “any rotation helps”
  story

### H-U4. Residual-Tail + Unitary Prefix

Combine the strongest practical control with the strongest geometric prior:

- keep the recent residual tail in FP16 (`kivi_residual`)
- apply a RoPE-compatible unitary transform only to the older quantized prefix

This is the most plausible bridge between the current practical winner and the
folder’s Lie/HEAT theory.

## 7. Document Alignment Notes

- `phase4_1_ppl_results.docx` is no longer the authoritative Qwen evidence.
  It remains useful as historical motivation, but the active paper-facing readout
  must come from the `v3` full-quant harness because that is the first protocol
  that quantizes all K states rather than a diluted slice.
- `FOKVQ_ADDITIONAL_EXPERIMENT_PLAN_v9.docx` is still directionally correct:
  `Exp 4-2` remains the decisive benchmark. The difference is that the current
  queue is now narrowed to the credible Qwen/GQA battleground instead of broad
  GPT-2-first optimism.
- The practical paper story must be conditional on `Qwen 3/4-bit`:
  if a Lie candidate cannot at least challenge `fokvq_e2`, then the result is
  mechanistic only and should not replace PCA as the main practical method.
