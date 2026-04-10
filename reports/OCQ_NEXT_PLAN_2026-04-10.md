# OCQ Next-Step Plan (2026-04-10, post-E8 refresh)

## Purpose

This is the canonical next-step document for the current OCQ line in
`ba-ocq-develop`. It replaces the stale version that still treated the E8
follow-up runs as pending.

The project is **not** in benchmark-expansion mode. It is in
**mechanism-validation and artifact-control mode**.

## Current factual state

### Completed evidence

- Qwen / MetaTool `995`
  - `no_steer`: `73.57%`
  - `ocq_bias_a0.2`: `81.11%` (`+7.54pp`)
  - `ocq_bias_a0.3`: `83.12%` (`+9.55pp`)
- Mistral / MetaTool `995`
  - `no_steer`: `55.98%`
  - `ocq_bias_a0.2`: `44.12%` (`-11.86pp`)
  - `ocq_bias_a0.3`: `37.89%` (`-18.09pp`)
- Mistral / low-alpha MetaTool `995`
  - `ocq_bias_a0.05`: `53.17%` (`-2.81pp`)
  - `ocq_bias_a0.10`: `49.55%` (`-6.43pp`)
  - `ocq_bias_a0.15`: `47.34%` (`-8.64pp`)
  - `ocq_bias_a0.20`: `44.12%` (`-11.86pp`)
- Qwen / MMLU subset `1000`
  - `no_steer`: `72.0%`
  - `ocq_bias_a0.2`: `72.9%` (`+0.9pp`)
  - `ocq_bias_a0.3`: `70.5%` (`-1.5pp`)

### Immediate conclusions

1. Cross-model transfer is currently false, not merely unproven.
2. Mistral low-alpha did not rescue the intervention. Alpha alone is not the
   explanation.
3. `a0.2` is the only plausible Qwen operating point for a balanced story.
4. The Qwen positive result is still vulnerable to parser and null-control
   objections.

## What claim is currently defensible

Only this:

> On `Qwen/Qwen2.5-7B`, catalog-derived flat K-bias improves single-turn
> MetaTool tool selection. The same intervention fails on
> `mistralai/Mistral-7B-v0.3`.

Do **not** say:

- "cross-model"
- "Qwen-family"
- "safe by phase-closure"
- "multi-turn context integration"

None of those is earned by the current evidence.

## Reviewer-visible blockers

### B1. MetaTool parser robustness

The original scorer was vulnerable:

- candidate-name parsing missed valid tool names such as `PDF&URLTool`
- evaluation was substring-based on unconstrained generation

The candidate parser has now been fixed in
[eval_metatool_subtask1.py](/home/v-seungplee/ba-ocq-develop/scripts/ocq/eval_metatool_subtask1.py),
and invalid candidate rows are now dropped and logged explicitly. Structured
first-line parser-safe scoring now also exists in code. Constrained decoding is
future hardening, not the current blocker.

### B2. Null controls are still missing

The Qwen gain has not yet beaten:

- rank-matched random projector
- shuffled-projector control
- opaque tool-name control

Until these are run, ontology-specificity is not established.

### B3. MMLU is only a regression check

MMLU helps reject "catastrophically destructive" settings. It does **not**
validate ontology specificity or tool-selection mechanism.

### B4. Develop request outruns reality

The develop-branch coworker request asks for:

- Llama cross-model sweep
- tau2
- BFCL
- multi-turn claims

The current code/evidence surface does not justify launching those as the next
main wave. Llama is gated, tau2/BFCL harnesses are not ready, and the Qwen gain
still lacks basic controls.

## Integrating the develop request with the current plan

### Requested by develop

1. Cross-model expansion (`R1`)
2. Multi-turn expansion (`R2`, `R3`)
3. MMLU phase-gating (`R4`)

Important:

- the original coworker request mixes an older Qwen sweep and an `all`-layers
  intent, while the current validated evidence is built on `first1`
- do not merge those into one comparison table without a config-frozen rerun

### Current disposition

1. `R1 / Mistral`
   - already completed
   - result is negative
2. `R1 / Llama`
   - blocked by HF gated access
   - do not schedule until access is verified
3. `R2 / tau2`
   - deferred
   - harness is not execution-ready
4. `R3 / BFCL`
   - deferred
   - same reason as tau2
5. `R4 / MMLU`
   - partially completed
   - use as secondary regression evidence only

## Intent / Hypothesis / Verification

### Intent P1 — Make the Qwen result reviewer-resistant

**Hypothesis.**
If the Qwen gain is real and not just a parser or lexical artifact, it should
survive tighter scoring and beat simple null controls.

**Verification.**

1. parser-safe MetaTool scoring
   - structured first-line parsing
   - constrained output only if first-line parsing proves too brittle
2. random-projector control
   - rank-matched random orthonormal basis
3. shuffled-projector control
   - feature-shuffled projector with matched rank and support
4. opaque-name control
   - preserve descriptions, replace tool names with opaque IDs

**Stop rule.**
If ontology projector does not beat matched null controls by at least `+3pp` on
Qwen under parser-safe scoring, kill the ontology-specific story.

### Intent P2 — Choose one Qwen operating point, not a vague alpha range

**Hypothesis.**
`a0.2` is likely the only defensible operating point because `a0.3` buys a
small extra MetaTool gain but introduces off-target cost on MMLU.

**Verification.**

- rerun `no_steer`, `a0.2`, `a0.3` with the corrected MetaTool parser
- if needed, repeat MMLU with a second seed or larger sample
- report the Qwen accuracy frontier instead of calling both "good"

**Stop rule.**
If `a0.2` also shows clear off-target degradation after replication, kill the
flat-bias safety story.

MMLU remains secondary here. It can reject obviously bad operating points, but
it cannot rescue an ontology-specificity claim that fails on MetaTool controls.

### Intent P3 — Diagnose Mistral with a factored design

**Hypothesis.**
Mistral failure is not explained by alpha alone. The remaining candidates are
layer-placement mismatch, basis mismatch, or intervention-norm mismatch.

**Verification.**

- `first1` vs `all`
- ontology projector vs rank-matched random projector
- log intervention norm / projected-energy diagnostics

**Stop rule.**
If Mistral remains negative under matched placement and norm-aware controls,
freeze it as a counterexample rather than trying to rescue it repeatedly.

### Intent P4 — Delay multi-turn expansion until P1 passes

**Hypothesis.**
Running tau2/BFCL before Qwen survives the control suite would only multiply
ambiguity.

**Verification.**

- do not launch tau2 or BFCL before P1 completes
- do not reopen Llama until access exists

**Stop rule.**
If P1 fails, multi-turn work is not the next step.

## GPU-aware execution order

### Actual GPU state at refresh time

- local machine
  - one A100 is occupied by an unrelated `llm-addiction` Llama run
  - local GPU should be treated as unavailable except for tiny smoke tests
- E8
  - GPU 0 is occupied by an unrelated job
  - GPUs 1, 2, 3 are idle

### Default allocation

- E8 GPU 1: primary OCQ experiment
- E8 GPU 2: secondary OCQ experiment
- E8 GPU 3: spare / retry / overflow

### Ordered queue

0. Implement parser-safe scoring and null-control code paths
   - exact or structured output scorer
   - random-projector control
   - shuffled-projector control
   - opaque-name path
   - local smoke only
1. Qwen parser-safe / null-control implementation smoke
   - local CPU or tiny local smoke only
2. Qwen control bundle on E8 GPU 1
   - parser-safe MetaTool
   - random / shuffled / opacity controls
3. Qwen replication frontier on E8 GPU 2
   - corrected parser
   - `no_steer`, `a0.2`, `a0.3`
4. Mistral factored diagnosis on E8 GPU 2 or 3
   - only after Qwen control harness is stable
5. Llama cross-model rerun
   - only if access becomes available
6. tau2 / BFCL
   - only if Qwen survives P1

Hard gate:

- do not launch step 2 or step 3 until parser-safe scoring and all three null
  controls exist in code and pass a bounded dry run

## Required code surface before the next E8 wave

### Already fixed

- MetaTool candidate parser now handles delimiter-based names and drops invalid
  candidate rows explicitly.
- parser-safe first-line scoring exists in code
- random-projector control exists in code
- shuffled-projector control exists in code
- opaque-name path exists in code

### Still required

1. bounded dry run of the control bundle
2. canonical full control-bundle runner adoption
3. better per-run provenance in JSON outputs
4. optional constrained-output scorer if first-line parsing proves too brittle

## Canonical artifacts

### Primary docs

- [COWORKER_REQUEST_cross_model_2026_04_10.md](/home/v-seungplee/ba-ocq-develop/reports/COWORKER_REQUEST_cross_model_2026_04_10.md)
- [OCQ_CROSS_MODEL_STATUS_2026-04-10.md](/home/v-seungplee/ba-ocq-develop/reports/OCQ_CROSS_MODEL_STATUS_2026-04-10.md)
- [OCQ_E8_EXECUTION_POLICY_2026-04-10.md](/home/v-seungplee/ba-ocq-develop/reports/OCQ_E8_EXECUTION_POLICY_2026-04-10.md)
- [OCQ_NEXT_PLAN_2026-04-10.md](/home/v-seungplee/ba-ocq-develop/reports/OCQ_NEXT_PLAN_2026-04-10.md)

### Primary runners

- [ocq_e8_qwen_control_bundle.sh](/home/v-seungplee/ba-ocq-develop/scripts/ocq_e8_qwen_control_bundle.sh)
- [ocq_e8_qwen_mmlu_safety.sh](/home/v-seungplee/ba-ocq-develop/scripts/ocq_e8_qwen_mmlu_safety.sh)

### Primary results

- [qwen25_7b_metatool_alpha_sweep_995.json](/home/v-seungplee/ba-ocq-develop/results/ocq/cross_model/qwen25_7b_metatool_alpha_sweep_995.json)
- [mistral_7b_v03_metatool_alpha_sweep_995.json](/home/v-seungplee/ba-ocq-develop/results/ocq/cross_model/mistral_7b_v03_metatool_alpha_sweep_995.json)
- [mistral_7b_v03_metatool_low_alpha_995.json](/home/v-seungplee/ba-ocq-develop/results/ocq/cross_model/mistral_7b_v03_metatool_low_alpha_995.json)
- [qwen25_7b_mmlu_safety_1000.json](/home/v-seungplee/ba-ocq-develop/results/ocq/cross_model/qwen25_7b_mmlu_safety_1000.json)
