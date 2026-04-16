# Steering Paper: Implementation Readiness and Proof Plan

Date: 2026-04-16

## 1. Git / Branch State

- Main working tree: `/home/v-seungplee/boltzmann-attention`
- Current branch: `main`
- Local uncommitted changes exist in paper files, figures, and report docs.
- Latest fetched `origin/develop`: `8b92ca8`
- Safe review worktree created at:
  - `/home/v-seungplee/boltzmann-attention-develop-review`

Because the main worktree is dirty, the correct way to inspect `develop` was:

1. `git fetch origin develop`
2. `git worktree add --detach /home/v-seungplee/boltzmann-attention-develop-review origin/develop`

This avoids corrupting the current paper edits.

## 2. Direct Readiness Assessment

This section answers one narrow question: can we run the high-priority steering experiments now without first doing infrastructure cleanup?

### 2.1 Scripts that are structurally ready

- `/home/v-seungplee/boltzmann-attention-develop-review/scripts/ocq/eval_metatool_subtask1.py`
- `/home/v-seungplee/boltzmann-attention-develop-review/scripts/ocq/eval_metatool_subtask4.py`
- `/home/v-seungplee/boltzmann-attention-develop-review/scripts/ocq/eval_bfcl.py`
- `/home/v-seungplee/boltzmann-attention-develop-review/scripts/ocq/eval_tau2bench_epsilon_q.py`

Observed status:

- `py_compile` passes.
- `--help` works.
- CLI is coherent enough for direct invocation.

Interpretation:

- The core OCQ/Q-bias evaluation path is runnable.
- BFCL and `tau2` support drivers exist and are compact enough for selective absorption.
- For our own method, the repo is near experiment-ready.

### 2.2 Scripts that exist but are not yet robust enough for unattended runs

- `/home/v-seungplee/boltzmann-attention-develop-review/scripts/ocq/eval_subtask4_with_real_seka.py`
- `/home/v-seungplee/boltzmann-attention-develop-review/scripts/diagnostics_2026_04_16/eval_subtask4_with_adaseka.py`
- `/home/v-seungplee/boltzmann-attention-develop-review/scripts/ocq/eval_subtask4_dynamic_qk_v2.py`

Observed status:

- `eval_subtask4_with_real_seka.py --help` fails if invoked outside the expected repo layout because it imports `src.model.seka_llm` from `external/SEKA`.
- The script inserts `external/SEKA` into `sys.path`, but still assumes the source tree is present and compatible.
- AdaSEKA script assumes cwd and relative path semantics that are fragile.

Interpretation:

- These are not missing implementations.
- They are environment-sensitive wrappers that require path hygiene before reliable batch use.
- The dynamic QK driver is useful for exploratory work, but it expands the method scope and should not be treated as a paper blocker.

### 2.3 Immediate implementation conclusion

The correct status is:

- `Q-bias / K-bias / null-control / micro-sweep` pipeline: ready enough to run.
- `matched SEKA / AdaSEKA` pipeline: conceptually implemented, operationally fragile.
- `BFCL` external robustness check: low-risk to integrate and smoke-test.
- `tau2` diagnostic proxy: useful, but should remain supporting evidence.
- Therefore the main blocker is not algorithm code, but reproducible baseline integration.

## 3. What Must Be Fixed Before Serious Runs

### F1. Make baseline wrappers self-contained

Target files:

- `/home/v-seungplee/boltzmann-attention-develop-review/scripts/ocq/eval_subtask4_with_real_seka.py`
- `/home/v-seungplee/boltzmann-attention-develop-review/scripts/diagnostics_2026_04_16/eval_subtask4_with_adaseka.py`

Required fixes:

1. Resolve `external/SEKA` path relative to `__file__` only.
2. Fail early with an explicit message if `external/SEKA` or projection files are missing.
3. Add a preflight mode:
   - imports
   - tokenizer load
   - one-sample generation
   - marker wrapping check
4. Log exact environment:
   - `CUDA_VISIBLE_DEVICES`
   - transformers version
   - attention implementation
   - model snapshot / commit hash if available

Success condition:

- `--help` works from repo root.
- `--preflight` succeeds without manual cwd tricks.

### F2. Standardize scorer and prompt format across all baselines

The paper is vulnerable if different methods are evaluated under different prompt templates or scorers.

Required lock:

1. One prompt template.
2. One decode policy.
3. One primary scorer.
4. Optional secondary scorer only in appendix.

Success condition:

- Every result JSON stores:
  - scorer name
  - prompt template ID
  - decoding params
  - dataset hash or path

### F3. Add stepwise analysis directly into the evaluation pipeline

Current weakness:

- Aggregate F1 exists.
- The central claim is about sequential coverage, but the pipeline does not yet make first-step vs second-step behavior the default output.

Required additions:

1. First-tool accuracy
2. Second-tool accuracy
3. Repeated-first-facet failure rate
4. Per-sample emitted sequence record

Success condition:

- Every main Subtask4 run emits both aggregate and stepwise metrics.

## 4. Recommended Experiment Plan

This is the experiment order that best improves the paper, not the order that maximizes raw experiment count.

### M1. Matched SEKA/AdaSEKA comparison

Intent:

- Establish the closest-prior comparison under a fair protocol.

Hypothesis:

- Query-side suppression remains competitive under matched evaluation.
- Stationary key-side amplification remains mismatched to Subtask4.

Validation:

1. Use official source in `external/SEKA`.
2. Use the same prompt format, scorer, and decoding across:
   - no_steer
   - Q-bias
   - K-bias
   - SEKA
   - AdaSEKA-2
   - AdaSEKA-3
3. Run Qwen first.
4. Run Llama second only after Qwen preflight is clean.

Decision rule:

- If matched SEKA/AdaSEKA cannot be stabilized, do not claim superiority.

### M2. Stepwise coverage breakdown

Intent:

- Test the actual mechanism claim rather than only the final F1.

Hypothesis:

- Q-bias should improve second-tool recovery more than first-tool accuracy.

Validation:

1. Add stepwise outputs to the Subtask4 evaluator.
2. Compare:
   - no_steer
   - Q-bias
   - K-bias
   - matched SEKA/AdaSEKA if M1 is clean

Decision rule:

- If second-tool improvement does not appear, the current coverage framing is too weak.

### M3. Llama null-control extension

Intent:

- Show ontology specificity is not Qwen-only.

Hypothesis:

- The real basis remains the only sign-consistent useful direction.

Validation:

1. Repeat feature-shuffled and random basis controls on Llama.
2. Report macro metrics and effective perturbation magnitude.

Decision rule:

- If Llama null controls do not preserve the same qualitative pattern, ontology specificity should be weakened in the main text.

### M4. Small-alpha interaction check

Intent:

- Determine whether the tiny K-side assist is real or sweep noise.

Hypothesis:

- If the positive band near `alpha_K = 0.025, 0.05` is real, it should reappear on a rerun and ideally on Llama.

Validation:

1. Rerun Qwen narrow sweep.
2. Add the same narrow sweep on Llama.

Decision rule:

- Robust on both models: mention in main or strong appendix.
- Only Qwen: appendix only.
- Unstable: remove from story.

### S1. Efficiency and trajectory bundle

Intent:

- Turn the paper from a small-score paper into a mechanism paper.

Validation:

1. Hook overhead per token
2. Ontology-energy over decoding steps
3. Two to three before/after tool-sequence examples

Decision rule:

- Include one compact figure in main paper.

## 5. What Not to Do Now

These directions are lower ROI for the current steering paper and should not block submission-quality progress:

- Large new benchmark expansion
- Compression-axis re-expansion
- New adaptive controller proposals
- Unmatched baseline tables
- More aggregate-score-only plots

## 6. Proof Assessment: Is the Current Proof Correct Enough?

Short answer:

- The current steering paper proof package is directionally reasonable but not yet at the clarity level of `/home/v-seungplee/energy_theory.pdf`.
- The biggest problem is not obvious formal invalidity.
- The biggest problem is proof presentation discipline.

### 6.1 What `energy_theory.pdf` does better

The energy paper gets four structural things right:

1. It distinguishes theorem scope from modeling assumptions up front.
2. Each theorem states exactly what is proved and under what assumptions.
3. Proofs are decomposed into named steps.
4. Discussion text is clearly separated from formal claims.

### 6.2 What the current steering proof package still lacks

1. Assumption hygiene
   - Structural claims and surrogate claims are still too close together.
   - The reader can still lose track of what is fully proved versus what is explanatory.

2. Proof architecture
   - The theorem/proposition sequence exists, but not yet in a fully explicit Setup -> Assumptions -> Statement -> Proof sketch -> Full proof pipeline.

3. Notational onboarding
   - The steering paper introduces variables inside statements faster than the energy paper does.

4. Formal scoping
   - The perturbation theorem is fine as a diagnostic result, but the paper must keep repeating that it is not an end-to-end accuracy theorem.

## 7. Required Proof Rewrite Style

The steering paper should imitate the *style* of `energy_theory.pdf`, not its topic.

### 7.1 Recommended proof structure for the steering paper

#### Section A. Setup and scope

- Define:
  - query vector
  - key matrix
  - ontology projector
  - history-free intervention class
- Add an explicit scope paragraph:
  - Proposition 1 and 2 are structural.
  - The perturbation theorem is diagnostic.
  - No theorem directly proves benchmark F1.

#### Section B. Structural propositions

- Proposition 1:
  - stationary key-side steering cannot encode coverage state
- Proposition 2:
  - negative query-side projection is the simplest history-free coverage surrogate

Each should follow:

1. Statement
2. One-sentence intuition
3. Short proof
4. One paragraph on what it does not prove

#### Section C. Diagnostic perturbation theorem

Recommended rewrite:

1. Assumptions
   - bounded query norm
   - bounded value norm
   - row-wise key perturbation norm bound
2. Define logits, softmax weights, qaMSE
3. State theorem
4. Proof in explicit named steps:
   - Step 1: exact interpolation identity
   - Step 2: first-order term bound
   - Step 3: remainder bound
   - Step 4: combine and take expectation

#### Section D. Interpretation layer

- Move all “why this matters for Q-bias vs K-bias” text outside the theorem environment.
- Keep interpretation paragraphs visually separated from proofs.

## 8. Concrete Paper Rewrite Tasks

### P1. Add an explicit assumptions block before the theory section

Target:

- `/home/v-seungplee/boltzmann-attention/paper/neurips2026_steering_v2/sections/04_3_theory.tex`
- `/home/v-seungplee/boltzmann-attention/paper/neurips2026_steering_ko/sections/05_theory.tex`

Required content:

1. Structural claims versus diagnostic theorem
2. What is proved
3. What is not proved

### P2. Rewrite theorem statements in “energy_theory” style

Required changes:

1. One statement = one claim.
2. No interpretation inside statement.
3. No hidden assumptions in the prose after the statement.

### P3. Rewrite appendix proofs into numbered steps

Target:

- appendix proof sections

Required change:

- Every proof should have labeled internal steps.

### P4. Add a proof-status table

Recommended small table:

| Claim | Type | Fully proved? | Depends on modeling assumptions? |
|---|---|---|---|
| Stationary key-side lacks coverage state | Structural | Yes | Minimal |
| Negative query-side projection has correct sign semantics | Structural | Yes | Minimal |
| Perturbation output bound | Diagnostic | Yes | Boundedness assumptions |
| Q-bias improves second-tool recovery | Empirical | No | Must be shown experimentally |

This single table will prevent reviewer confusion.

## 9. Final Verdict

### Implementation

- The core OCQ path is ready.
- The matched-baseline path is partially implemented but still fragile.
- Before large runs, baseline wrappers need one cleanup pass.

### Proofs

- The current proofs are not obviously wrong in the narrow sense.
- They are not yet presented with the discipline needed for a strong theory-facing paper.
- The right target is exactly the style of `energy_theory.pdf`:
  - explicit assumptions,
  - narrow statements,
  - stepwise proofs,
  - clean separation of proof and interpretation.

### Best next actions

1. Harden SEKA/AdaSEKA wrappers.
2. Add stepwise Subtask4 metrics.
3. Rewrite theory/appendix with explicit assumption and proof-status structure.
4. Only then run the matched baseline table that the paper actually needs.
