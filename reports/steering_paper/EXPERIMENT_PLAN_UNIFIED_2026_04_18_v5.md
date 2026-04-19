# Unified Experiment Plan v5 — NeurIPS Submission Track

**Date**: 2026-04-18  
**Paper target**: `paper/neurips2026_steering_ko`  
**Evidence base**: locked Qwen main-table JSONs, Llama retail/telecom reruns, `BOOTSTRAP_SIGNIFICANCE_2026_04_18.md`, and the current paper audit.

## 0. Why this plan exists

The paper is no longer blocked by lack of positive results. The remaining problem is different: the evidence is uneven. We already have a coherent story on Qwen and a useful cross-model signal on Llama telecom, but the proof stack, related-work positioning, and experiment portfolio are still misaligned with what NeurIPS reviewers will attack first. This plan therefore does **not** aim to add many more runs. It aims to close the few objections that can still sink the submission.

The paper's main claim is fixed as follows:

- **Main claim.** Multi-tool selection breaks stationary K-style steering, and layer-adaptive K+Q is the strongest currently locked counter-design within the same operator family.
- **Supporting claim.** Signed Q is still useful, but its best polarity and magnitude are regime- and model-dependent.
- **Non-claim.** The paper does not claim that layer-adaptive uniformly beats every signed-Q choice on every domain, nor that the first-order sign diagnostic is a deployment-ready router.
- **Scope control.** Until protocol-matched SEKA/AdaSEKA numbers are locked, the paper should describe the main empirical comparison as a result against the stationary K family, not as a definitive head-to-head win over the full spectral-steering literature.

In other words, the paper is allowed to be provocative about the failure mode, but not sloppy about the benchmark closure:

- **Allowed sharp framing.** ``Stationary key amplification is the wrong abstraction for multi-tool selection.''  
- **Not yet allowed.** ``We have already beaten the full SEKA/AdaSEKA line under a matched protocol.''

Every planned experiment below is evaluated against that fixed claim.

## 1. Locked evidence already sufficient for the paper

### E1. Qwen main table is already load-bearing

**Intent.** Establish the main empirical floor of the paper before spending compute elsewhere.

**Hypothesis.** On Qwen2.5-7B, layer-adaptive K+Q is the strongest locked K-using fallback: it matches or exceeds Q-only across the four core domains, with at least one significant win.

**Validation.** Use only locked JSONs and paired bootstrap:

- MetaTool ST4: `no_steer = 0.7307`, `Q-only = 0.7535`, `ladapt = 0.7514`
- `τ²` retail: `no_steer = 0.4679`, `Q-only = 0.5190`, `ladapt = 0.5284`
- `τ²` telecom: `no_steer = 0.2512`, `Q-only best = 0.4785`, `ladapt = 0.5188`
- `τ²` airline: `no_steer = 0.3285`, `ladapt = 0.3669`

Bootstrap summary is fixed in `BOOTSTRAP_SIGNIFICANCE_2026_04_18.md`.

**Interpretation.** This evidence is already strong enough to support an aggressive but accurate wording: layer-adaptive is the strongest \emph{locked} K-using design and a robust within-family fallback. It is not enough to support ``uniform winner.''

### E2. Retail action-count decomposition already explains the within-family split

**Intent.** Explain why retail shows only a small aggregate gap between ladapt and Q-only even though the two operators behave differently.

**Hypothesis.** Layer-adaptive wins short-horizon tasks, while Q-only wins medium/long horizon tasks, so the aggregate difference is a cancellation between distinct regimes.

**Validation.** Recomputed retail split:

- `≤2` actions: ladapt beats Q-only by `+9.2`pp
- `3–5` actions: Q-only beats ladapt by about `+3.8`pp
- `≥6` actions: Q-only beats ladapt by about `+4.6`pp

**Interpretation.** This is the best current empirical support for the cache-divergence story. It should be elevated to a main mechanism figure, not left as a secondary table.

### E3. Llama telecom already supports the narrow cross-model claim

**Intent.** Test whether the operator family survives a model-family change in a regime with enough headroom.

**Hypothesis.** On Llama-3.1-8B telecom, ladapt still improves over `no_steer`, even if the best signed-Q polarity changes.

**Validation.**

- `no_steer = 0.3845`
- `ladapt = 0.5007` (`+11.62`pp, bootstrap significant)
- best signed-Q is `β=-0.05 = 0.5474`
- positive `β` causes large-scale empty outputs

**Interpretation.** The right claim is not ``same beta transfers.'' The right claim is ``the operator family transfers, but polarity must be calibrated per model.''

These three blocks mean the paper already has a legitimate empirical core. The rest of the plan is about closing submission-time objections.

## 2. Submission blockers: experiments that still need to happen

### P0. Protocol-matched SEKA and AdaSEKA head-to-head

**Intent.** Close the most damaging reviewer objection: that the paper only beats an in-house stationary-K baseline, not the actual spectral steering literature.

**Hypothesis.** When run under the same evaluator, parser, prompt family, and ontology source, stationary K-family methods remain weaker than layer-adaptive on multi-tool tasks, especially on MetaTool ST4 and `τ²` telecom.

**Validation.**

- **Models.** Qwen2.5-7B only. Do not widen scope until this is locked.
- **Tasks.** MetaTool ST4, `τ²` retail, `τ²` telecom.
- **Methods.** `no_steer`, stationary K (`ocq_bias_*`), SEKA, AdaSEKA, Q-only best, ladapt.
- **Metrics.** F1, Recall, GT⊆Pred, nDCG, repeated-first-tool-rate, second-distinct-rate.
- **Rule.** Same prompt construction, same extraction logic, same decode policy, same max tokens, same ontology file class.

**Interpretation.** If this experiment is missing, the paper remains vulnerable no matter how good the Qwen/Llama tables look. This is the single highest-priority missing experiment.

### P1. Llama telecom failure-type classification

**Intent.** Turn the observed ``positive beta collapse'' into a precise failure analysis rather than a vague anecdote.

**Hypothesis.** Large positive Q-rotation on Llama telecom primarily breaks tool-call formatting, not semantic ranking. The dominant failure mode is empty or non-parseable output, not consistently wrong tool choice.

**Validation.**

- **Model.** Llama-3.1-8B-Instruct.
- **Task.** `τ²` telecom.
- **Methods.** `β=+0.03`, `β=+0.05`, `β=+0.10`, plus `β=-0.05` and ladapt as controls.
- **Protocol.** Run a small verbose audit (`N=20` is enough), save raw generations, classify first failure mode into `early EOS`, `plain-language drift`, `partial JSON`, `repetition loop`, `other`.
- **Main count already known.** Empty outputs are 145/200, 200/200, 55/200 for the three positive-β settings.

**Interpretation.** This experiment directly strengthens the paper's most interesting cross-model insight: layer-adaptive is safer not just because it preserves semantics, but because it avoids a format-stability regime that signed-Q can enter on tool-call-tuned models.

### P2. PCA-of-K basis ablation

**Intent.** Test whether the gain comes from any low-rank subspace or specifically from the ontology-aligned basis.

**Hypothesis.** Random and shuffled bases already fail; PCA will sit in between. Q-only may retain part of the gain under PCA, but K-including methods should depend more strongly on ontology semantics.

**Validation.**

- **Model.** Qwen2.5-7B.
- **Tasks.** MetaTool ST4 and `τ²` retail.
- **Methods.** Q-only best and ladapt under ontology basis vs PCA-of-K basis.
- **Success pattern.** PCA < ontology on at least one K-including setting.

**Interpretation.** This is not as critical as P0, but it helps defend novelty. Without it, a reviewer can say the paper merely discovered ``low-rank steering + schedule'' rather than ``ontology-aligned steering + schedule.''

### P2b. Cardinality-controlled evaluation

**Intent.** Rule out the cheap explanation that gains come mainly from emitting more tools rather than ranking the right tools higher.

**Hypothesis.** Under matched output cardinality, the relative advantage of Q-only and ladapt should remain visible on at least one benchmark, especially in nDCG@k or recall@k.

**Validation.**

- **Models.** Qwen2.5-7B.
- **Tasks.** MetaTool ST4 and `τ²` retail.
- **Protocol.** Re-score existing predictions at matched `k`, or re-decode with fixed tool budget where possible.
- **Metrics.** Precision@k, Recall@k, nDCG@k, GT⊆Pred@k.

**Interpretation.** If the effect vanishes under matched cardinality, the current mechanism story is overstated. If it survives, the paper gains a clean rebuttal to the ``it just emits more tools'' objection.

### P2c. Anti-repeat baseline

**Intent.** Test whether the main gain comes from a deep mechanism or from a simpler decode-time anti-repetition heuristic.

**Hypothesis.** A simple first-tool mask or anti-repeat penalty should help stationary K somewhat, but should still underperform Q-only or ladapt on second-distinct recovery.

**Validation.**

- **Models.** Qwen2.5-7B.
- **Tasks.** MetaTool ST4.
- **Methods.** `no_steer`, stationary K, stationary K + anti-repeat penalty, Q-only, ladapt.
- **Metrics.** F1, Exact, second-distinct-hit, second-recovery-given-first-hit.

**Interpretation.** This is one of the sharpest falsification tests for the current causal story. If a trivial decode-time anti-repeat heuristic closes most of the gap, the paper should frame its contribution as practical control rather than a distinct attention mechanism.

## 3. High-value but non-blocking experiments

### P3. MetaTool multi-metric extension

**Intent.** Show that the ST4 gain is not an F1 artifact.

**Hypothesis.** Q-only and ladapt should improve not only F1, but also coverage-sensitive metrics such as recall and second-distinct selection.

**Validation.**

- **Model.** Qwen2.5-7B.
- **Task.** MetaTool ST4.
- **Metrics.** F1, Recall, GT⊆Pred if available, first-hit, second-hit, second-distinct, repeated-first.

**Interpretation.** This extends the ``not F1 gaming'' rebuttal from `τ²` to the benchmark that most directly tests multi-tool sequencing.

### P4. Qwen size sanity sweep

**Intent.** Reduce the risk that the result is a 7B-only quirk.

**Hypothesis.** The exact magnitudes will move, but the stationary-K failure and the existence of a positive ladapt regime should survive at least one smaller or larger Qwen size.

**Validation.**

- **Models.** Qwen2.5-3B and Qwen2.5-14B.
- **Tasks.** MetaTool ST4 and `τ²` telecom only.
- **Methods.** `no_steer`, stationary K, Q-only best, ladapt.

**Interpretation.** This is a reviewer comfort experiment, not a load-bearing one. Run it only after P0--P2.

## 4. Experiments that should stay in appendix or be deferred

### D1. beta-star empirical predictor

**Intent.** Explore whether the first-order sign diagnostic can become a practical predictor.

**Hypothesis.** At best, a better proxy set may improve correlation with empirical best sign.

**Validation.** Logit-lens or discriminative-$\mathcal{G}$ variants.

**Interpretation.** This is no longer main-track critical. The paper should keep beta-star as a local explanation tool regardless of whether this lands.

### D2. Full layer-boundary sweep

**Intent.** Fine-tune the `L/4` split.

**Hypothesis.** A local Pareto frontier may exist, but it is unlikely to change the central claim.

**Validation.** LS-1 to LS-6 sweep on Qwen ST4 and `τ²` retail.

**Interpretation.** Helpful for polish, not necessary for acceptance. If time is tight, keep the placeholder table in the appendix and defer.

## 5. Recommended execution order

The execution order should follow reviewer risk, not curiosity.

1. **P0: SEKA/AdaSEKA head-to-head**
   - Without this, the paper remains structurally exposed.
2. **P1: Llama telecom failure-type classification**
   - Cheap and strengthens the cross-model story immediately.
3. **P2: PCA-of-K basis ablation**
   - Clarifies what part of the method is actually novel.
4. **P2b: Cardinality-controlled evaluation**
   - Closes the strongest metric-design objection.
5. **P2c: Anti-repeat baseline**
   - Sharp falsification of the causal mechanism story.
6. **P3: MetaTool multi-metric extension**
   - Small effort, closes a metric-quality objection.
7. **P4: Qwen size sanity sweep**
   - Only if time remains.

## 5.5 Benchmark-metric matrix

The execution plan is easiest to audit if each run has a fixed `(model, task, methods, metrics)` contract.

| Exp | Models | Tasks | Methods | Primary metrics | Why this is load-bearing |
|---|---|---|---|---|---|
| P0 | Qwen2.5-7B | MetaTool ST4, `τ²` retail, `τ²` telecom | `no_steer`, stationary K, SEKA, AdaSEKA, Q-only best, ladapt | F1, Recall, GT⊆Pred, nDCG, repeated-first, second-distinct | Closes the most obvious prior-work objection |
| P1 | Llama-3.1-8B | `τ²` telecom | `β=+0.03,+0.05,+0.10,-0.05`, ladapt | F1, empty-output rate, failure-type counts | Converts the current anecdotal collapse into a concrete mechanism figure |
| P2 | Qwen2.5-7B | MetaTool ST4, `τ²` retail | ontology basis vs PCA-of-K under Q-only and ladapt | F1, Exact, Jaccard | Separates ontology alignment from generic low-rank structure |
| P2b | Qwen2.5-7B | MetaTool ST4, `τ²` retail | existing predictions under matched `k` or fixed-budget decode | Precision@k, Recall@k, nDCG@k | Rules out the ``more emitted tools'' explanation |
| P2c | Qwen2.5-7B | MetaTool ST4 | stationary K with and without anti-repeat control, Q-only, ladapt | F1, second-distinct, second-recovery | Falsifies the claim that the gain is only a trivial no-repeat heuristic |
| P3 | Qwen2.5-7B | MetaTool ST4 | `no_steer`, Q-only, ladapt | F1, Exact, Jaccard, repeated-first, second-distinct | Strengthens the no-memory mechanism with the benchmark most aligned to it |
| P4 | Qwen2.5-{3B,14B} | MetaTool ST4, `τ²` telecom | `no_steer`, stationary K, Q-only best, ladapt | F1, Exact | Reviewer comfort experiment for size robustness |

## 6. What the paper should say while these runs are pending

The paper should not pretend these experiments are already done. It should distinguish three kinds of evidence.

- **Locked main evidence.** Qwen main table, retail horizon split, Llama telecom transfer, Llama positive-β collapse counts.
- **Submission-time validation.** SEKA/AdaSEKA head-to-head, PCA ablation, cardinality control, anti-repeat baseline, size sweep, layer-boundary sweep.
- **Deferred analysis.** beta-star predictor engineering.

If P0 is still missing at submission time, the related-work claim must remain narrow: the paper can say that stationary K-style steering is structurally mismatched to multi-tool selection and that our direct evidence is strongest on that family. It should not present itself as having closed the entire SEKA/AdaSEKA comparison.

This distinction allows the draft to be complete without being dishonest.

## 7. Concrete paper deliverables tied to this plan

Each planned experiment has a corresponding paper artifact.

- **P0** fills a new comparison table: `SEKA / AdaSEKA / stationary K / Q-only / ladapt`.
- **P1** fills a new figure: `beta vs F1` overlaid with `beta vs empty-output rate` on Llama telecom.
- **P2** fills a new appendix table: `ontology / shuffled / random / PCA`.
- **P2b** fills a rebuttal table: `matched-k metrics`.
- **P2c** fills a falsification table: `stationary K vs stationary K + anti-repeat`.
- **P3** expands the ST4 mechanism table with additional metrics.
- **P4** fills the appendix size-sanity table.

## 7.5 Claim language to use in the paper

The paper should sound sharper than a workshop draft, but every sharp sentence must still be tied to locked evidence.

- **Use:** ``Stationary K-style steering collapses on multi-tool selection even when it is effective for single-tool emphasis.''
- **Use:** ``Early-K / global-Q is the first locked counterexample showing that the failure is not low-rank steering itself, but the stationary deployment form.''
- **Use:** ``The core design freedom is not whether to steer, but where in depth to steer and which side of attention to modify.''
- **Avoid until P0 lands:** ``Our method outperforms SEKA/AdaSEKA.''
- **Avoid until P2 lands:** ``Ontology alignment is uniquely necessary.''
- **Avoid until P1 lands:** ``The Llama collapse is definitively a semantic-safe formatting failure.'' 

## 8. Bottom line

The paper is already past the ``no result'' phase. The remaining work is now selective: one protocol-matched baseline comparison, one failure-analysis figure, and one basis ablation will improve the submission more than any number of extra sweeps. If those land, the paper moves from ``promising but arguable'' to ``defensible under hostile review.''
