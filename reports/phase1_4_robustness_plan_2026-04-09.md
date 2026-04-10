# Phase 1.4 Robustness Plan for SEKA / AdaSEKA / Ontology (2026-04-09)

## Objective

Turn the current Phase 1.1-1.3 findings from "promising but locally reported" into a robustness-checked result that can survive obvious reviewer attacks.

Current risk is not weak headline numbers. Current risk is weak claim hygiene:

- the SEKA/AdaSEKA/ontology code path is not yet E8-ready by default
- `external/SEKA` and its datasets are absent on E8
- the strongest ontology claim still lacks the ablations that would separate a real semantic-basis effect from leakage, random-subspace luck, or routing artifacts

This plan is intentionally narrow. It prioritizes reproducibility and falsification over collecting more pretty numbers.

## Related Work Anchors

- **SEKA / AdaSEKA**: The official repository describes SEKA as low-rank key-space steering from synthetic QA pairs and AdaSEKA as a small ensemble of SEKA experts plus an input-dependent selector. That means any ontology claim must beat not just single-expert steering, but also the adaptive-routing story on its own turf. citeturn1view0
- **PASTA**: PASTA already established post-hoc attention steering as a viable inference-time control mechanism. This means "we also steer attention after training" is not enough novelty; the differentiator has to be the basis source, transfer behavior, or robustness story. citeturn0search1
- **CounterFact / model editing precedent**: CounterFact was built precisely to test whether factual interventions preserve specificity and generalization, so a 500-example success story without paired robustness analysis is not enough. citeturn0search10turn0search14
- **Bias in Bios**: Bias in Bios is a legitimate transfer benchmark, but it probes a different failure mode from CounterFact. Positive transfer helps, but it does not replace causal checks on the original factual-steering task. citeturn0search3turn0search15

## Intent / Hypothesis / Verification

### Intent R0: Audit executable surface before promising robustness runs

**Hypothesis.**
The current workspace is sufficient for SEKA-vs-ontology smoke, but not necessarily for AdaSEKA routing ablations or paired per-example statistics. Those must be treated as explicit dependencies, not assumed.

**Verification.**
- confirm `external/SEKA` exists and contains actual runnable code, not only output folders
- confirm whether AdaSEKA codepath exists in the upstream repo or in coworker patches
- confirm there is a per-example result artifact format that supports paired bootstrap and win/loss analysis
- if paired analysis is not already emitted, add a small local parser/analyzer before large runs

Update after E8 smoke:

- the upstream `efficacy.json` / `paraphrase.json` files already contain per-example `samples` with stable `id` fields
- this removes the need for an invasive upstream evaluator patch in the first robustness round
- first-round implementation should therefore be a thin export shim that flattens those `samples` into `example_id, efficacy, paraphrase` rows for paired bootstrap

**Kill criterion.**
If AdaSEKA or per-example result surfaces are absent, do not advertise R3 as runnable in the current round. Restrict the round to SEKA-vs-ontology robustness plus environment hardening.

### Intent R1: Reproduce the exact local SEKA and ontology pipeline on E8

**Hypothesis.**
If the local Phase 1.1-1.3 results are real and not machine-specific, then after portable path/device fixes the same scripts should build projections and run a small benchmark subset on E8 without manual code edits.

**Verification.**
- `scripts/ontology_facet_basis.py --self-test`
- `scripts/phase1_ontology_projection.py --self-test --require-external-seka`
- `scripts/phase1_ontology_projection_rank8.py --self-test --require-external-seka`
- verify that these self-tests are only path/dependency gates, not proof of full pipeline correctness
- 10-example smoke for:
  - baseline CounterFact
  - SEKA CounterFact
  - ontology CounterFact
- success condition: all three commands finish on E8 and write outputs to expected locations

**Kill criterion.**
If E8 still needs hand edits after the current portability patch, do not launch full robustness runs. Fix environment orchestration first.

### Intent R2: Test whether ontology really beats SEKA, or only looks better on a 500-example slice

**Hypothesis.**
The reported ontology rank-8 result (`ES=96.8`, `PS=96.7`) is only interesting if it survives a paired comparison against SEKA (`ES=95.2`, `PS=96.2`) on either the full CounterFact split or repeated paired subsamples.

**Verification.**
- run SEKA and ontology rank-8 on the same examples
- verify output format includes per-example predictions; if not, patch evaluation export first
- compute paired win/loss tables per example
- run paired bootstrap confidence intervals for `ES_delta` and `PS_delta`
- run at least one of:
  - full CounterFact
  - 5 disjoint 500-example shards with fixed seeds

Refined execution note:

- do not jump from 10-example smoke directly to full/fuller runs
- first run a 100-example instrumented shard on E8 with export + bootstrap wiring
- only then promote the identical harness to `0:500`

**Kill criterion.**
If ontology's lead disappears or confidence intervals cover a meaningful negative effect, downgrade the claim from "beats SEKA" to "competitive with SEKA."

### Intent R3: Test whether AdaSEKA's weakness is real mixture dilution rather than a bad configuration accident

**Hypothesis.**
If the coworker claim is real, soft or multi-expert AdaSEKA should underperform either single-expert SEKA or top-1 routed AdaSEKA even when an in-domain expert is available. If top-1 routing closes the gap, then the real story is routing dilution, not "adaptive is bad."

**Verification.**
- reproduce:
  - baseline
  - single-expert SEKA
  - AdaSEKA soft mixture / current official config
  - AdaSEKA top-1 hard routing
  - uniform-expert average as a negative control
- use the same evaluation subset and prompts for all runs
- record router entropy and per-example selected experts

**Kill criterion.**
If top-1 routing removes the gap, paper story becomes "mixture dilution in soft Q-adaptive routing." If both adaptive variants are unstable, paper story becomes "AdaSEKA is brittle in our setup," which is weaker.

### Intent R4: Rule out trivial explanations for ontology success

**Hypothesis.**
If ontology matters semantically, it should beat or at least reliably differ from a random same-rank projector and from a label-shuffled ontology basis.

**Verification.**
- controls:
  - random orthonormal rank-8 projector
  - shuffled ontology categories
  - truncated PCA-on-K covariance if easy to build
- compare against ontology rank-8 on identical examples

**Kill criterion.**
If random same-rank subspaces match ontology, the current paper story collapses. Then the result is not "ontology works"; it is "almost any rank-8 projector helps."

### Intent R5: Check transfer without overclaiming

**Hypothesis.**
A real reusable basis should transfer across at least one second task or one second model, but transfer magnitude can differ because gain `alpha` is task-specific.

**Verification.**
- second task: Bias in Bios, only after CounterFact robustness passes
- second model: one additional model family only if E8 smoke and first robustness round finish cleanly
- keep `alpha` sweep narrow: `{1.56, 2.0, 3.0}`

**Kill criterion.**
If transfer is inconsistent or only works with aggressive retuning plus degraded fluency, treat transfer as exploratory rather than main evidence.

## Execution Order

1. Surface audit for SEKA / AdaSEKA / paired-output availability
2. E8 bootstrap
3. E8 smoke
4. CounterFact paired robustness
5. ontology random/shuffle controls
6. AdaSEKA routing ablation only if codepath exists
7. optional transfer

Do not reorder this. Running transfer before falsifying the main CounterFact claim is wasted GPU.

Within step 4 use a sub-order:

1. `0:100` pilot with export and bootstrap
2. `0:500` main shard with the same harness if the pilot finishes cleanly

## Minimal E8 Requirements

- repo cloned under `/scratch/boltzmann/boltzmann-attention`
- `external/SEKA` cloned with `benchmarks/` and `src/` present
- SEKA datasets unpacked under `external/SEKA/data/`
- Python env with at least:
  - `torch`
  - `transformers`
  - `datasets`
  - `spacy`
  - `nltk`
  - `dataclasses-json`
  - SEKA repo requirements
- Hugging Face access to `Qwen/Qwen3-4B-Base`

## Concrete Smoke Matrix

### Smoke S1: Environment only

- portable self-tests must pass
- self-tests are necessary but shallow: they only validate path and dependency shape
- expected result: no missing path assumptions, no hardcoded `/home/woori/...`, no forced `CUDA_VISIBLE_DEVICES='1'`

### Smoke S2: Builder only

- build ontology projector for 1 layer and 1 head-compatible slice if runtime matters, otherwise full last-10 build
- expected result: write valid `.pt` payload and diagnostic JSON

### Smoke S3: Eval only

- 10 CounterFact examples for:
  - baseline
  - SEKA
  - ontology
- expected result: pipeline completes, writes JSON, and no format/path mismatch occurs

## What Counts as Success

- **Engineering success**: E8 runs without hand edits
- **Scientific success**: ontology remains at least competitive with SEKA after paired robustness and random-basis controls
- **Paper success**: either
  - ontology survives random/shuffle controls and adaptive-routing ablations, or
  - AdaSEKA mixture dilution becomes the real main result with ontology as a strong alternative baseline

## What Does Not Count

- one more 500-example point estimate without paired statistics
- a second task result without falsifying the main CounterFact claim
- self-created `external/SEKA/` directories that are not the actual repo
- "E8-ready" claims before dependency bootstrap is complete
