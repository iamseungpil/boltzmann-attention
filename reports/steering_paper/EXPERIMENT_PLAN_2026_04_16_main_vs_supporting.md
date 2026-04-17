# Steering Paper Experiment Plan — Canonical Alignment

Date: 2026-04-16

## Purpose

This document is the canonical paper-facing plan for the steering paper in `paper/neurips2026_steering_v2`. It replaces the earlier mixed planning state with one hard rule:

- the paper must be organized around claims that are already supported by auditable result files,
- stronger but narrower findings may appear only as secondary operator comparisons,
- external benchmark breadth comes after internal mechanism clarity.

## Locked Claim Hierarchy

The paper now has a two-tier claim structure.

### Tier 1: Main claim

For multi-tool sequential selection, ontology-guided query-side contraction has the correct sign and basis specificity, while stationary key-side amplification on the same basis is structurally mismatched.

This is the only cross-family claim currently safe for the main paper.

### Tier 2: Secondary claim

Within Qwen MetaTool Subtask4, two stronger operator variants improve on the earlier `beta=-0.1` point:

- `Q_full_Bont` single-pass contraction at `beta=-0.03`,
- `iterative_kq` with emitted-state query contraction and early-layer key assistance.

This claim is real but not yet cross-family. It must be framed as operator selection, not as the main headline, until Llama and matched-baseline runs exist.

## Fact Base

### Cross-family verified facts

- Qwen Subtask4 baseline: F1 `0.7307`, Exact `0.5252`, Jaccard `0.6673`.
- Qwen `ocq_qbias_b-0.1`: F1 `0.7471`, Exact `0.5272`.
- Qwen stationary `K-bias alpha=0.3`: F1 `0.6850`, Exact `0.4728`.
- Llama Subtask4 baseline: F1 `0.6227`, Exact `0.5030`, Jaccard `0.5872`.
- Llama `ocq_qbias_b-0.1`: F1 `0.6271`, Exact `0.5070`.
- Llama stationary `K-bias alpha=0.3`: F1 `0.3105`, Exact `0.2616`.
- Qwen null controls:
  - Q-bias real `0.7471`, feature-shuffled `0.7254`, random `0.7068`.
  - K-bias real `0.6850`, feature-shuffled `0.0000`, random `0.0000`.

### Qwen-only operator facts

- `Q_full_Bont` / `ocq_qbias_b-0.03`: F1 `0.7535`, Exact `0.5332`.
- `iterative_kq` with K-early: F1 `0.7524`, Exact `0.5473`.
- `multipass_kq`: F1 `0.7410`, Exact `0.5091`.
- `k_early_only` iterative KQ variant: F1 `0.7514`, Exact `0.5473`.

### External benchmark readiness facts

- `eval_bfcl.py` is a proxy evaluator based on predicted function-name sets, not official BFCL AST scoring.
- `eval_tau2bench_epsilon_q.py` is a per-step next-tool plus `epsilon_q` diagnostic, not full official tau2-bench conversation scoring.
- `build_tau2_ontology.py` exists, but the expected `eval_tau2.py` and `eval_multi_turn.py` are not present in the current repo state.

## Decision on External Benchmarks

### tau2-bench

Direct answer: not suitable as a main benchmark now.

Reasoning:

- The currently available driver is diagnostic rather than official.
- The task is farther from the locked Subtask4 mechanism claim than the current paper can afford.
- A weak or noisy tau2 result would blur the paper faster than it would strengthen it.

Decision:

- `tau2` is placeholder-only in this paper cycle.
- It may appear in a future extension once an official evaluation harness exists.

### BFCL

Direct answer: useful only as a supporting benchmark.

Reasoning:

- BFCL naturally separates simple from parallel or multiple function calls.
- That makes it a reasonable stress test for single-tool versus multi-tool behavior.
- But the current evaluator is still a proxy and must not be treated as official benchmark evidence.

Decision:

- BFCL is supporting-only.
- Any BFCL table must be marked as proxy until official AST evaluation is wired in.

## Main Experiments

These experiments decide whether the paper is complete.

| ID | Experiment | Intent | Hypothesis | Validation |
|---|---|---|---|---|
| M1 | Cross-family locked main table | Keep the paper grounded in what is actually safe | Q-side contraction stays positive and stationary K-side amplification stays negative on both Qwen and Llama | Same prompt, scorer, decode policy, metadata, and result schema across both models |
| M2 | Qwen operator comparison | Decide which stronger operator to present as the best current implementation | `Q_full_Bont` and `iterative_kq` both beat the earlier `beta=-0.1` point, with different precision/coverage behavior | Reproduce the same ranking under one locked evaluator and report F1, Exact, Jaccard, precision, recall |
| M3 | Stepwise coverage breakdown | Test whether the gain is actually about sequential recovery | Query-side methods should improve second-tool recovery or reduce repetition rather than only improve first-tool ranking | First-tool hit, second-tool hit, second distinct hit, second recovery given first hit, repeated-first-tool rate |
| M4 | Matched SEKA/AdaSEKA table | Keep the closest-prior comparison honest | Under a matched protocol, official SEKA-family baselines should not overturn the sign story | Official external wrappers only; same prompt template, same scoring contract, same decode policy |
| M5 | Basis-specificity extension | Show that the effect is not generic low-rank noise | Real ontology basis remains uniquely useful while shuffled or random controls fail qualitatively | Locked null-control protocol on Qwen, and on Llama if feasible |

### M1. Cross-family locked main table

- Intent: preserve one paper claim that remains defensible even if every stronger exploratory result later weakens.
- Hypothesis: `Q-bias` has the correct sign across model families and stationary `K-bias` does not.
- Validation:
  - Use the current Subtask4 evaluator only.
  - Keep prompt template, scorer, and decode policy fixed.
  - Report F1, Exact, Jaccard, precision, recall, and runtime metadata.
- Success criterion:
  - Q-side positive on both families.
  - K-side negative on both families.
- Paper role: main table.

### M2. Qwen operator comparison

- Intent: choose a best current implementation without pretending it is already a universal cross-family result.
- Hypothesis: `Q_full_Bont` and `iterative_kq` are both genuinely better than `beta=-0.1`, and `iterative_kq` should lead on Exact.
- Validation:
  - Compare `no_steer`, `ocq_qbias_b-0.1`, `ocq_qbias_b-0.03`, `iterative_kq`, and `multipass_kq`.
  - Use the same evaluator and result schema.
  - Include precision and recall so the precision-first versus recall-first tradeoff is explicit.
- Success criterion:
  - `Q_full_Bont` or `iterative_kq` reproduces near the current Qwen numbers.
  - `iterative_kq` remains best or tied-best on Exact.
- Paper role: main Qwen operator table or compact subsection.

### M3. Stepwise coverage breakdown

- Intent: force the mechanism story to cash out at the decision level.
- Hypothesis: query-side methods help by changing the second decision, not merely by slightly improving the first one.
- Validation:
  - Run the current stepwise metrics already emitted by `eval_metatool_subtask4.py`.
  - Compare `no_steer`, stationary `K-bias`, `ocq_qbias_b-0.1`, `ocq_qbias_b-0.03`, and `iterative_kq`.
  - Add SEKA-family rows only if M4 succeeds with official wrappers.
- Success criterion:
  - Improvement appears in second-tool hit, second distinct hit, or repetition reduction.
- Failure rule:
  - If the gain is only first-tool ranking, the discussion must weaken any explicit coverage language.
- Paper role: main mechanism table.

### M4. Matched SEKA and AdaSEKA comparison

- Intent: stop loose cross-paper comparisons and replace them with one fair protocol.
- Hypothesis: the closest prior methods may still be strong baselines, but they should not reverse the core sign story when evaluated fairly.
- Validation:
  - Use `eval_subtask4_with_real_seka.py` or official external wrappers only.
  - No proxy AdaSEKA path is allowed in paper-facing tables.
  - Emit the same protocol metadata as the main evaluator.
- Success criterion:
  - The resulting table is auditable from the JSON bundle alone.
  - The central sign story survives.
- Failure rule:
  - If wrappers remain unstable or broken, the paper must avoid superiority language.
- Paper role: main baseline table if matched; otherwise appendix protocol note only.

### M5. Basis-specificity extension

- Intent: keep the strongest empirical argument in the paper from being Qwen-only if possible.
- Hypothesis: the ontology basis remains qualitatively unique on Llama as well.
- Validation:
  - Extend the existing real / feature-shuffled / random protocol to Llama.
  - Report end metrics and perturbation magnitudes.
- Success criterion:
  - The real basis remains qualitatively distinct from controls.
- Paper role: main robustness row if practical, appendix if partial.

## Supporting Experiments

These experiments help interpretation but should not drive the paper's center.

| ID | Experiment | Intent | Hypothesis | Validation |
|---|---|---|---|---|
| S1 | BFCL proxy screening | Test whether single-tool versus multi-tool separation survives outside MetaTool | Query-side methods should look better on parallel or multiple settings than on simple settings | Must be labeled proxy, not official BFCL |
| S2 | Llama operator extension | Test whether Qwen-only stronger operators transfer | `Q_full_Bont` or `iterative_kq` should remain positive on Llama if they are real mechanism improvements | Re-run the same operator table on Llama |
| S3 | Scorer robustness | Defuse objections about one scoring rule | Qualitative ranking should survive across at least two scorer regimes | Locked prompts and documented scorer identity |
| S4 | Latency and overhead | Show practical deployability | Query-side edits add bounded inference overhead | Per-query runtime or per-token latency |
| S5 | Sequence traces and qualitative cases | Make the mechanism visible | Query-side edits should show less repetition in emitted sequences | Trace plots and side-by-side generations |
| S6 | tau2 placeholder slot | Keep space for later external expansion without contaminating the current paper | No paper claim should depend on tau2 until official scoring exists | Placeholder text only |

### S1. BFCL proxy screening

- Intent: stress-test the sign story on a community benchmark without overclaiming official benchmark validity.
- Hypothesis: the method should look more useful on parallel or multiple subsets than on simple subsets.
- Validation:
  - Use only the current proxy evaluator.
  - Mark every table and caption as proxy.
- Success criterion:
  - Qualitative separation supports the multi-tool argument.
- Paper role: appendix only, unless official BFCL evaluation is added.

### S2. Llama operator extension

- Intent: upgrade the Qwen-only stronger operator story if it survives.
- Hypothesis: at least one of `Q_full_Bont` or `iterative_kq` remains positive on Llama.
- Validation:
  - Same prompt template, scorer, decode policy, and output schema as Qwen.
- Success criterion:
  - Positive delta over `no_steer` on Llama.
- Paper role: if clean, this can promote Tier 2 from Qwen-only to cross-family.

### S3. Scorer robustness

- Intent: prevent evaluator-format objections from derailing the results.
- Hypothesis: absolute numbers may move, but the main ranking should not invert dramatically.
- Validation:
  - Compare the locked generation scorer with one stricter alternative only.
  - Do not add many scorers.
- Paper role: appendix.

### S4. Latency and overhead

- Intent: show that the method is a realistic inference-time intervention rather than an impractical analysis artifact.
- Hypothesis: Q-side edits add modest overhead relative to baseline and official SEKA-family comparisons.
- Validation:
  - Measure runtime from the same evaluation harness.
- Paper role: appendix or compact footnote table.

### S5. Sequence traces and qualitative cases

- Intent: make the operator behavior interpretable to a first-time reader.
- Hypothesis: `iterative_kq` and `Q_full_Bont` should visibly reduce repetition or improve second recovery on representative two-tool prompts.
- Validation:
  - Plot emitted sequence patterns or ontology-energy traces from the locked protocol.
- Paper role: one compact main figure plus appendix examples.

### S6. tau2 placeholder slot

- Intent: reserve a clean extension path without lying about readiness.
- Hypothesis: none for the current paper.
- Validation:
  - The manuscript may mention tau2 only as future external validation.
- Paper role: placeholder only.

## Execution Order

1. M1 cross-family locked main table sanity check.
2. M2 Qwen operator comparison rerun.
3. M3 stepwise coverage breakdown.
4. M4 matched SEKA/AdaSEKA comparison.
5. S2 Llama operator extension.
6. S4 and S5 practical/interpretability support.
7. S1 BFCL proxy screening if time remains.

## Readiness Gates

| Gate | Requirement | Status |
|---|---|---|
| G1 | Main claim backed by auditable cross-family result files | partial |
| G2 | Qwen stronger-operator story backed by current result files | complete |
| G3 | Stepwise mechanism claim backed by actual numbers, not only available code | incomplete |
| G4 | Closest-prior comparison uses official wrappers only | incomplete |
| G5 | External benchmark table uses official scorer, or is explicitly labeled proxy | incomplete |
| G6 | tau2 official full-task evaluator exists in repo | incomplete |

## Claims That Are Forbidden

- Any headline claim that tau2 validates the method.
- Any headline claim that BFCL results are official unless AST scoring is integrated.
- Any claim that `Q_full_Bont` or `iterative_kq` are cross-family best methods before Llama reruns exist.
- Any claim that SEKA or AdaSEKA are beaten without a matched official-wrapper table.
- Any claim that the theorem proves benchmark accuracy.

## Paper Layout Mapping

### Main body

- M1 cross-family sign table.
- M2 Qwen operator comparison.
- M3 stepwise coverage mechanism table.
- M5 basis-specificity table.

### Appendix

- M4 matched SEKA/AdaSEKA if not ready for main.
- S1 BFCL proxy screening.
- S2 Llama operator extension if late.
- S3 scorer robustness.
- S4 latency.
- S5 traces and qualitative examples.
- S6 tau2 placeholder note.

## Bottom Line

The correct paper is no longer the old narrow-only paper, but it is also not the broad develop-branch paper.

The aligned paper should say:

- one safe cross-family mechanism story,
- one stronger but currently Qwen-only operator-selection story,
- no external-benchmark bravado until the evaluators are official enough to trust.
