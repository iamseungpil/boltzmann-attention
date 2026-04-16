# Steering Paper Experiment Plan — 2026-04-16

## Purpose

This document is the paper-facing experiment plan for the steering paper in `paper/neurips2026_steering_v2`. It replaces the scattered planning notes for this paper with one execution order. The plan is split into `Main Experiments` and `Supporting Experiments` so that the paper story stays narrow:

1. The closest prior comparison is SEKA/AdaSEKA, but only under matched evaluation.
2. The central claim is about sequential coverage in multi-tool selection, not generic activation steering.
3. The paper should win on mechanism clarity before it tries to win on benchmark breadth.

## Planning Rule

Every experiment in this file must answer three questions explicitly:

1. What is the intent?
2. What hypothesis is actually being tested?
3. What result would count as validation or rejection?

If an experiment does not sharpen one of those three, it should not be run for the paper.

## Fact Base

- Qwen Subtask4: `no_steer` F1 `0.7307`, `Q-bias beta=-0.1` F1 `0.7471`, `K-bias alpha=0.3` F1 `0.6850`.
- Llama Subtask4: `no_steer` F1 `0.6227`, `Q-bias beta=-0.1` F1 `0.6271`, `K-bias alpha=0.3` F1 `0.3105`.
- Qwen Q-bias null controls: `featshuffle 0.7254`, `random 0.7068`.
- Qwen K-bias null controls: `featshuffle 0.0000`, `random 0.0000`.
- Qwen small `alpha_K` on top of Q-bias: `0.025 -> 0.7529`, `0.05 -> 0.7502`.
- Perturbation-bound medians: Qwen `2.357e-08`, Llama `6.372e-08`.

## Method Comparison to SEKA and AdaSEKA

These distinctions must remain explicit in both the paper and the evaluation plan.

| Axis | SEKA | AdaSEKA | Ours |
|---|---|---|---|
| Target task | Prompt highlighting / single-target emphasis | Input-adaptive prompt highlighting | Multi-tool sequential coverage |
| Direction source | Learned from synthetic QA contrasts | Learned expert bank from synthetic contrasts | Training-free ontology basis from tool catalog |
| Intervention site | Key-side amplification | Key-side amplification with input-conditioned expert choice | Query-side suppression, with key-side used only as a contrast condition |
| Core mechanism | Highlight chosen subspace | Choose or mix highlighted expert subspaces | Suppress already dominant ontology-aligned query mass |
| Statefulness | Stationary once chosen | Input-adaptive but still not explicit facet-coverage state | History-free approximation to coverage |
| Clean claim we can make | Closest conceptual predecessor | Closest adaptive predecessor | Different task and different sign semantics |

## Main Experiments

These are the experiments that decide whether the main paper is complete.

| ID | Experiment | Intent | Hypothesis | Validation |
|---|---|---|---|---|
| M1 | Matched SEKA/AdaSEKA table | Make the closest-prior comparison fair | Under one locked protocol, stationary key-side highlighting remains mismatched or at least does not overturn the sign story | Same prompt template, scorer, decode policy, and source basis handling for all methods |
| M2 | Stepwise coverage breakdown | Test the mechanism directly | Q-bias should help later coverage more than early ranking | First-tool accuracy, second-tool accuracy, repetition rate |
| M3 | Basis-specificity controls | Show the effect is not generic low-rank noise | Real ontology basis stays useful; shuffled or random controls do not | Real basis stays positive and stable while controls degrade |
| M4 | Small-alpha K-plus-Q interaction | Decide whether to keep or cut the pair story | If the micro-band is real, it should reproduce across families | Same narrow sweep on Qwen and Llama, compared to Q-only |

### M1. Matched SEKA and AdaSEKA head-to-head on MetaTool Subtask4

- Intent: establish whether the paper's mechanism claim survives under the closest prior methods evaluated with the same scorer, prompt template, ontology catalog, and decoding policy.
- Hypothesis: the current paper does not need to beat SEKA everywhere, but it should show that stationary key-side highlighting is mismatched to Subtask4 coverage and that any claim about a large gap is only credible under matched evaluation.
- Validation:
  - Use official `external/SEKA` implementations, not proxies.
  - Evaluate `no_steer`, SEKA, AdaSEKA-2, AdaSEKA-3, our `Q-bias`, and our `K-bias contrast`.
  - Fix one scorer and one prompt format for all methods.
  - Report F1, Exact, Jaccard, plus runtime overhead.
- Success criterion:
  - Minimum success: the matched table is internally coherent and does not reverse the central sign story.
  - Strong success: our `Q-bias` is competitive with or better than SEKA/AdaSEKA on Subtask4 while `K-bias` remains clearly worse.
- Paper placement: main paper.
- Failure interpretation:
  - If the baseline wrappers remain environment-fragile, the paper must avoid any superiority language and keep SEKA/AdaSEKA as a fairness protocol requirement rather than a completed claim.

### M2. Stepwise coverage breakdown on Subtask4

- Intent: test the actual mechanism claim instead of relying only on aggregate F1.
- Hypothesis: if query-side suppression acts as a coverage prior, it should help the second tool more than the first and reduce repeated-first-facet failures.
- Validation:
  - Decompose predictions into first-tool accuracy, second-tool accuracy, and repeated-first-facet failure rate.
  - Run for `no_steer`, `Q-bias`, `K-bias`, and matched SEKA/AdaSEKA if M1 succeeds.
  - Include Qwen and Llama.
- Success criterion:
  - The largest positive movement from `Q-bias` should appear on second-tool recovery or repetition reduction, not only on first-tool accuracy.
- Paper placement: main paper.
- Failure interpretation:
  - If the gain appears only on first-tool accuracy, the current coverage framing is overstated and should be weakened to a narrower query-side regularization story.

### M3. Basis-specificity and null-control verification

- Intent: show that the effect is not a generic low-rank perturbation.
- Hypothesis: replacing the ontology basis with feature-shuffled or random controls should erase the query-side gain and destroy key-side stability.
- Validation:
  - Keep the current Qwen null-control table.
  - Add the same null-control protocol on Llama if practical.
  - Report both end metrics and effective perturbation magnitude.
- Success criterion:
  - Real basis remains the only stable and positive direction while random or shuffled controls degrade.
- Paper placement: main paper.
- Failure interpretation:
  - If a shuffled or random control matches the real basis qualitatively, the ontology-specificity claim does not belong in the main paper.

### M4. Small-alpha K-plus-Q interaction

- Intent: determine whether a very small key-side term is genuinely complementary or just a sweep artifact.
- Hypothesis: the paper's primary method remains query-side suppression, but a narrow positive interaction band may exist for very small `alpha_K`.
- Validation:
  - Re-run the micro-sweep around `alpha_K in {0.0, 0.025, 0.05, 0.075, 0.1}` on Qwen and Llama under the same protocol.
  - Compare against Q-only.
- Success criterion:
  - Either confirm a narrow reproducible positive band or kill the interaction and keep the paper simpler.
- Paper placement: appendix unless the effect is robust across both model families.
- Failure interpretation:
  - If the band disappears on rerun or on Llama, cut the pair story from the main paper entirely.

## Supporting Experiments

These experiments strengthen interpretation but should not drive the main claim.

| ID | Experiment | Intent | Hypothesis | Validation |
|---|---|---|---|---|
| S1 | Subtask1 transfer | Check that the real basis is not destructive outside Subtask4 | Q-bias stays non-destructive but coverage-specific lift shrinks | Same scorer, same prompt template |
| S2 | Scorer robustness | Defuse scoring objections | Method ranking should stay qualitatively stable across strict and generation-based scorers | Locked prompts, two scorers only |
| S3 | Efficiency and latency | Show the hook is practical | Query-side steering adds bounded inference overhead | Tokens/sec or per-token latency |
| S4 | Ontology-energy traces and case studies | Give mechanism visibility | Q-bias should reduce repetitive ontology dominance after first emission | Traces and emitted sequence examples |
| S5 | Perturbation-bound diagnostics | Keep the theory attached to data | Bound ratios remain conservative in the real operating regime | Same ratio summary on main interventions |

### S1. Subtask1 transfer check

- Intent: verify that the coverage story is not unique to one benchmark slice.
- Hypothesis: the ontology basis should remain useful on single-selection disambiguation, but the coverage-specific gain should be smaller than on Subtask4.
- Validation:
  - Evaluate `no_steer`, `Q-bias`, and matched baselines on Subtask1 with the finalized scorer.
- Success criterion:
  - `Q-bias` remains non-destructive and basis-specific.
- Paper placement: appendix.

### S2. Scorer robustness

- Intent: preempt reviewer objections that the gains come from one scoring rule.
- Hypothesis: method ranking should be qualitatively stable across a strict label-logprob scorer and a generation-based scorer, even if absolute numbers differ.
- Validation:
  - Run the finalized main-method table under two scorers with identical prompt formatting.
  - Explicitly document any ranking changes.
- Success criterion:
  - The main sign pattern does not flip.
- Paper placement: appendix.

### S3. Efficiency and latency

- Intent: show the practical cost of the hooks and make the method look like a realistic inference-time intervention.
- Hypothesis: query-side steering adds modest overhead relative to matched SEKA/AdaSEKA baselines.
- Validation:
  - Measure tokens/sec or per-token latency on Qwen and Llama for `no_steer`, `Q-bias`, and matched SEKA/AdaSEKA.
- Success criterion:
  - Overhead is bounded and clearly reported.
- Paper placement: main paper only if the numbers are clean; otherwise appendix.

### S4. Ontology-energy traces and case studies

- Intent: give the mechanism one interpretable visual that is less cherry-pickable than a bare attention map.
- Hypothesis: under `Q-bias`, ontology-aligned query energy should drop after the first tool emission and the emitted tool sequence should become less repetitive.
- Validation:
  - Plot ontology-energy over decoding steps for 2-3 representative examples.
  - Show emitted tool sequences for `no_steer`, `Q-bias`, and one failed `K-bias` case.
- Success criterion:
  - The trace and example agree with the stepwise coverage breakdown from M2.
- Paper placement: one compact figure in main paper, extra examples in appendix.

### S5. Perturbation-bound diagnostics

- Intent: justify the theorem as a diagnostic tool rather than decorative math.
- Hypothesis: the empirical bound remains very conservative across the interventions used in the main paper.
- Validation:
  - Keep the current Qwen and Llama median ratio table.
  - If possible, add the same diagnostic for the small-`alpha_K` interaction point.
- Success criterion:
  - Ratios remain far below one and do not contradict the stability interpretation.
- Paper placement: appendix, with a short reference in the main body.

## Execution Order

1. M1 matched SEKA/AdaSEKA comparison.
2. M2 stepwise coverage breakdown.
3. M3 null-control extension if Llama is practical.
4. S3 efficiency and S4 trajectory visuals.
5. M4 small-alpha interaction decision.
6. S1, S2, S5 as cleanup and reviewer-defense material.

## Not In Scope

The following ideas may still be scientifically interesting, but they should not drive this paper:

- PCA or compression-driven rotation as a main story.
- Dynamic routing benchmarks that require a new paper framing.
- Layer-adaptive Q+K as a headline method before the stepwise coverage story is complete.
- Broad agent benchmark expansion without a tight mechanism bridge.

## Hard Decision Rules

- If M1 cannot be run with official SEKA/AdaSEKA code, the paper must avoid any headline superiority claim over SEKA.
- If M2 fails to show a second-tool or repetition reduction effect, the current coverage framing is too weak and the paper should revert to a narrower diagnostic-steering story.
- If M3 fails on a matched rerun, the ontology-specificity claim is not strong enough for the main paper.
- If M4 is unstable across model families, keep it out of the main story.

## Main-Paper vs Appendix Summary

### Main paper

- M1 matched baseline comparison
- M2 stepwise coverage breakdown
- M3 basis-specificity null controls
- One compact S4 mechanism figure

### Appendix

- M4 small-alpha interaction unless it becomes robust
- S1 Subtask1 transfer
- S2 scorer robustness
- S3 latency table if space is tight
- S4 extra cases
- S5 perturbation-bound diagnostics

## Codex Review Notes

The Codex review loop did not return a final structured report within the local timeout, but the partial trace was still useful. The extracted critique matched the current local assessment on three points:

1. The closest-prior comparison must be framed as a matched-evaluation problem, not a loose score contest.
2. The proof package must separate structural propositions from the perturbation theorem and present the theorem proof in explicit steps.
3. The highest-value new experiment is the matched SEKA/AdaSEKA Subtask4 table, followed immediately by the stepwise coverage breakdown.
