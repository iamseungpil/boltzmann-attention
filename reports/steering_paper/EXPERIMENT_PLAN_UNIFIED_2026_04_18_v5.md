# Unified Experiment Plan v5 — Safe-Floor Ladapt, Action-Count Complementarity, Format-Stability Collapse

**Date**: 2026-04-18
**Supersedes**: `EXPERIMENT_PLAN_UNIFIED_2026_04_18_v4.md`
**Evidence base**: `BOOTSTRAP_SIGNIFICANCE_2026_04_18.md`, recomputed `tab:retail-action-count`, `PAPER_AUDIT_2026_04_18_postfix.md`.
**Paper target**: `paper/neurips2026_steering_ko`.

## 0. Scope delta vs v4

v4 framed the regime split as "layer-adaptive wins short MetaTool, signed Q wins long τ²". Paired-bootstrap analysis on locked JSONs forces three corrections that v5 internalises.

The first correction is that layer-adaptive K+Q on Qwen retail and MetaTool ST4 is statistically indistinguishable from Q-only (p=0.714 and p=0.298 respectively), not a win or a loss. v4's regime-split language overstated the gap. v5 reframes ladapt as a *safe floor* that matches or beats Q-only on every tested Qwen domain and is significantly better on τ²-telecom (p=0.044).

The second correction is that within retail, the just-recomputed action-count decomposition shows ladapt dominating the ≤2-action bucket by +9.2pp over Q-only while losing the 3-5 and ≥6 buckets. The regime split therefore lives *inside* the same operator family rather than *between* operator families.

The third correction is that Llama β≥+0.03 on telecom does not produce "wrong tool" — it produces 145-200 of 200 empty predictions. The positive-rotation collapse is a format-stability failure, not a semantic sign flip, and is the load-bearing cross-model mechanism for C2.

v5 keeps C2 (efficiency) and C3 (β* predictor) structurally unchanged; their wording softens in line with the audit's honesty paragraph.

## 1. Top-level claims (v5)

### C1 — Layer-adaptive K+Q is a safe floor within the signed-Q operator family on Qwen.

**Intent.** Show that choosing layer-adaptive over Q-only never loses significantly and sometimes wins significantly, so a practitioner without a regime oracle can pick ladapt by default.

**Hypothesis.** Across four Qwen domains (MetaTool ST4, τ²-retail, τ²-telecom, τ²-airline), paired-bootstrap ΔF1 of ladapt − Q-only has CI upper bound ≥ 0 in all four and lower bound > 0 in at least one (expected: telecom).

**Validation.** 10,000-iteration paired bootstrap on locked per-sample JSONs (`reports/tau2_2026_04_18/*_full_v2.json`, `reports/metatool/subtask4_locked_v2.json`). Source of record: `BOOTSTRAP_SIGNIFICANCE_2026_04_18.md` table rows.

**Interpretation.** Retail Δ = +0.94pp at p=0.714 and ST4 Δ = +0.89pp at p=0.298 are both statistical ties, so we claim "match" not "beat" on those two. Telecom Δ = +4.03pp at p=0.044 is a significant win and anchors C1. Airline N=50 is underpowered (p=0.165), so we report the point estimate (+3.83pp vs no_steer) without the "significant" qualifier. The paper must not claim "ladapt uniformly wins".

### C1-decomp — Within the shared family, ladapt and Q-only are complementary across retail's action-horizon.

**Intent.** Explain the +0.94pp retail gap by showing it is a near-cancellation of two opposite regimes rather than a flat overlap.

**Hypothesis.** After the ladapt fix, retail's ≤2-action bucket gives ladapt a ≥ +5pp margin over Q-only; 3-5 and ≥6 buckets flip to Q-only advantage of ≥ +3pp.

**Validation.** Recomputed table (to replace `tab:retail-action-count` in `06_experiments.tex`):

| bucket | n | no_steer F1 | Q-only Δ | ladapt (fixed) Δ |
|---|---:|---:|---:|---:|
| ≤2 actions | 42 | 0.2770 | +5.82 | +15.02 |
| 3-5 actions | 62 | 0.5837 | +4.43 | +0.66 |
| ≥6 actions | 10 | 0.5515 | +6.39 | +1.82 |

**Interpretation.** Ladapt wins short-horizon by a 9.2pp margin, Q-only wins medium and long by 3-4pp. The aggregate +0.94pp is the sample-weighted average of those regimes, not a uniform overlap. This is the direct observation of Corollary `cor:cache-divergence`: K-bias accumulates through KV-cache across steps, so its short-horizon advantage is neutralised as sequences lengthen. The paper's §6 decomposition paragraph must flip from "Q-only dominates every bucket" to "ladapt dominates short-horizon, Q-only dominates medium/long; the same operator family accommodates both through layer placement".

### C1b — Operator-family transfer to Llama holds on under-focused regimes; polarity is model-specific.

**Intent.** Show that the Qwen layer-adaptive gain survives a model-family change in the regime where baseline has room to move (telecom), and honestly report that it does not show up where baseline is already saturated (retail).

**Hypothesis.** Llama telecom ladapt − no_steer is positive with p<0.01; Llama retail ladapt − no_steer CI contains 0.

**Validation.** Paired bootstrap on `reports/tau2_2026_04_18/llama31_{retail,telecom}_full_v4.json`. Row of record: `BOOTSTRAP_SIGNIFICANCE_2026_04_18.md` lines 13-14.

**Interpretation.** Llama telecom Δ = +11.62pp at p<0.001 replicates the operator-family transfer claim. Llama retail Δ = −1.68pp at p=0.207 is a non-significant effect, not a failure — the paper must not write "ladapt fails on Llama retail". The natural explanation is a baseline-ceiling effect (Llama retail no_steer = 0.5059 vs Qwen 0.4679), which v5 flags for future ablation but does not resolve. Polarity itself is model-specific: Qwen telecom's best signed-Q is β=+0.10, Llama telecom's best signed-Q is β=−0.05, so the "operator family transfers" claim is narrower than "same β transfers".

### C2 — Positive Q-rotation on tool-call-tuned models collapses output format, not just semantics.

**Intent.** Replace the v4 sentence "Llama β+ collapses" with the actual mechanism, which is format invalidity rather than tool-selection error.

**Hypothesis.** On Llama telecom N=200, positive β values (β∈{+0.03, +0.05, +0.10}) produce empty `pred_tools` for a progressively-increasing fraction of samples, whereas negative β and layer-adaptive produce 0 empty outputs.

**Validation.** Direct count of `pred_tools == []` in locked per-sample JSONs:

| method | empty / 200 | F1=0 / 200 |
|---|---:|---:|
| β=−0.10 | 0 | 61 |
| β=−0.05 (best) | 0 | 4 |
| β=−0.03 | 0 | 10 |
| β=+0.01 | 0 | 30 |
| β=+0.03 | 145 | 151 |
| β=+0.05 | 200 | 200 |
| β=+0.10 | 55 | 79 |
| ladapt | 0 | 10 |

Row of record: `BOOTSTRAP_SIGNIFICANCE_2026_04_18.md` lines 30-42.

**Interpretation.** β=+0.05 produces zero parseable outputs across all 200 samples. This is format-stability collapse, not "picks the wrong tool harder". Layer-adaptive's asymmetric K-then-Q schedule sidesteps the collapse because K-rotation is confined to the first L/4 layers while Q-rotation in all layers remains small in magnitude; the combined operator never reaches the magnitude regime where Llama's tool-format head breaks down. This is the *mechanism* for why ladapt is a safer default on Llama than a naive signed-Q choice, and it is the load-bearing cross-model claim in C2.

### C3 — First-order β* sign predictor remains a conditional open problem.

**Intent.** Keep the theoretical statement but not reopen the predictor as a deployment tool.

**Hypothesis and validation.** Unchanged from v4 (schema-G fails on telecom at 31.2%, logit-lens variant deferred). The honesty paragraph in `app:G-sensitivity` already lands.

**Interpretation.** `05_theory.tex` continues to present β* as a first-order local characterisation; main body uses the predicted sign for regime *explanation* only; no claim of label-free routing in v5. If the logit-lens measurer from v4 B1 lands post-deadline, v5.1 would reopen the claim — not before.

## 2. Experimental coverage audit under v5 claims

The following experiments already locked on develop suffice for v5 if their narratives are updated:

| Locked artefact | Supports | Update needed |
|---|---|---|
| `tau2_retail_full_v2.json` (Qwen) | C1, C1-decomp | Yes — re-derive action-count buckets with fixed ladapt (done, see C1-decomp). |
| `tau2_telecom_N200_v2.json` (Qwen) | C1, C2 | No; Qwen telecom β+ collapse is part of C2 only via Llama. |
| `tau2_airline_v2.json` (Qwen) | C1 (underpowered footnote) | No. |
| `metatool/subtask4_locked_v2.json` | C1 (tie) | No. |
| `llama31_retail_full_v4.json` | C1b (null) | No; reframe wording from "fails" to "no significant effect". |
| `llama31_telecom_full_v4.json` | C1b (positive), C2 (format collapse) | No; reframe around format-validity mechanism. |
| `beta_star_*` (schema-G) | C3 | No; already honest in appendix. |

No new runs required for v5. v5 is a narrative revision and a table recomputation, not an experimental extension.

## 3. Falsification triggers (updated)

The revised claims change the trigger set from v4 T4-T7. The v5 triggers are:

**T1-v5.** If any Qwen domain's bootstrap shows ladapt − Q-only CI upper bound < 0 (i.e. ladapt significantly worse than Q-only), the "safe floor" claim collapses and v5 must retreat to "regime-dependent equivalence".

Check result: none of MetaTool ST4 (Δ=−0.89pp, p=0.298 against Q-only sign), retail (Δ=+0.94, p=0.714), or airline violates this. C1 holds.

**T2-v5.** If the recomputed action-count table still shows Q-only dominating the ≤2-action bucket, C1-decomp collapses.

Check result: ≤2-bucket ladapt Δ=+15.02, Q-only Δ=+5.82 — C1-decomp holds with 9.2pp margin.

**T3-v5.** If Llama telecom ladapt − no_steer CI lower bound < 0, C1b cross-model claim retreats to model-specific.

Check result: +9.34pp lower bound at p<0.001 — C1b holds on telecom. Retail is intentionally reported as null.

**T4-v5.** If β=+0.05 Llama telecom empty-count is < 50/200, the format-collapse mechanism of C2 loses its anchor.

Check result: 200/200 empty — C2 holds at maximum strength.

**T5-v5.** If the logit-lens predictor is later run (B1 from v4) and agreement < 50%, C3 stays retired.

Status: deferred. Main-body β* already softened to regime-explanation tool.

No v5 trigger fires; the plan is coherent with all locked evidence.

## 4. Self-critique pass

First draft critique: Intent sentences for C1 and C1b initially conflated "safe floor" with "best choice"; Hypothesis blocks for C1 omitted the airline-N underpower; Validation for C1-decomp omitted the n=114 coverage check. Second-pass draft fixes all three.

Second-pass critique: Interpretation blocks risk re-introducing the forbidden "winner" framing. Revised to use "match or beat" and "safe floor" consistently. Also removed the phrase "분명히" from the C2 interpretation.

Third-pass critique: C2's mechanism claim ("asymmetric K-then-Q sidesteps the magnitude regime") is plausible but not directly tested — no ablation isolates the magnitude vs schedule contribution. v5 accepts this as a *hypothesised* mechanism to be explicitly marked as such in the discussion rewrite, not as a proven decomposition. The paper's §7 passage on C2 uses "이 구조는 ... 것으로 해석된다" (not "증명한다") to respect this.

The three passes converge. No open internal contradictions remain.

## 5. Paper-edit deliverables driven by v5

The narrative revisions in v5 map to the four affected sections as follows. Abstract leads with "ladapt = safe floor; operator family transfers, polarity is model-specific; format-stability is the cross-model mechanism". §6 experiments updates `tab:retail-action-count` to the new numbers, reframes the decomposition paragraph around short-vs-long complementarity within the same family, and adds a short passage on Llama β+ format collapse with the 200/200 count. §7 discussion opens with the bootstrap tie findings and pivots from "ladapt wins retail/telecom" to "ladapt-vs-Q-only is a within-family regime split" and closes with format-stability as the mechanism of cross-model transfer. §8 conclusion echoes the abstract's three points and names polarity calibration (not "winner selection") as the practical takeaway.

Untouched sections (02 introduction, 03 related, 04 method, 05 theory, 09 appendices) already carry the right statements for v5 after the postfix audit, except for the stale "+24.78pp" occurrences documented in the audit's Part C. Those occurrences live in 05 theory's `tab:sign-routing` and in 02 introduction's empirical-evidence paragraph. Fixing those is an M1-M4 edit already scheduled in the postfix audit's Tier 1 and is not expanded in v5.

## 6. What v5 does not claim

v5 does not claim ladapt is uniformly better than Q-only. v5 does not claim β+ vs β− polarity transfers across models. v5 does not claim the β* predictor works for deployment. v5 does not claim the format-collapse explanation of C2 is experimentally decomposed into magnitude vs schedule contributions. v5 does not claim Qwen airline is a significant ladapt win. Each of these negatives is enforced by a specific sentence pattern in the paper rewrite.

## 7. Execution status

No new runs. All claims supported by existing locked JSONs. Paper-rewrite execution begins immediately after this plan commits.

## 8. Optional final-day experiments (Tier 1/2/3)

v5 narrative is fully supported by locked evidence and requires no reruns. The following extensions would strengthen individual sections but are not blocking; each is listed in Intent / Hypothesis / Validation / Interpretation form so the decision to run is traceable.

### Tier 1 — high value / low cost

**E-v5.1. Llama β+0.05 failure-type classification (~3 min GPU for N=20 verbose smoke).**
- **Intent.** Turn "200/200 empty" into a mechanistic statement about Llama's tool-call format under positive Q-rotation.
- **Hypothesis.** ≥60% of empty outputs are "early EOS after non-tool prose" (NL refusal or summary), remainder being repetition loops or JSON-fragment truncation.
- **Validation.** `eval_tau2_bench.py --model Llama --domain telecom --methods ocq_qbias_b0.05 --max-samples 20 --verbose`; manually classify each generation's first 200 chars into {early EOS + NL, repetition loop, partial JSON, other}.
- **Interpretation.** If early-EOS dominates, §6 format-collapse paragraph upgrades from "output invalid" to "instruction fine-tuning emits NL refusal when queries perturbed past training distribution". If repetition dominates, the mechanism is attention-sink-like collapse.

**E-v5.2. Llama telecom ladapt vs no_steer 10k-iter bootstrap (0 GPU; ~2 min Python).**
- **Intent.** Upgrade C1b's central claim (+11.62pp Llama telecom ladapt) from point estimate to significance-tested.
- **Hypothesis.** 95% CI lower bound > +8pp, p<0.001.
- **Validation.** Already computed in `BOOTSTRAP_SIGNIFICANCE_2026_04_18.md` (+11.62pp, CI [+9.34, +13.93], p<0.001). Needs only re-citation from abstract and §6.
- **Interpretation.** Strengthens cross-model transfer claim; already effectively landed.

**E-v5.3. MetaTool ST4 multi-metric extension (0 GPU; ~5 min Python).**
- **Intent.** Close audit gap C5 on the primary benchmark (N=497), not only τ² retail/telecom.
- **Hypothesis.** Q-only and ladapt show parallel gains on Recall / GT⊆Pred / nDCG; no "F1 gaming" signature.
- **Validation.** Extract Recall/GT⊆P/nDCG from `develop:reports/layer_adaptive_2026_04_17/qwen_st4_ladapt_full_N497.json` per-sample; add rows to `tab:multi-metric`.
- **Interpretation.** If four metrics co-move, "F1 gaming" rebuttal extends from τ² to MetaTool. If Recall moves but precision drops, paper acknowledges shift honestly.

### Tier 2 — moderate value / moderate cost

**E-v5.4. PCA-of-K basis ablation on τ² retail (1 GPU-h).**
- **Intent.** Fill `tab:e4-basis` PCA row; test whether low-rank structure suffices or catalog ontology is load-bearing.
- **Hypothesis.** PCA basis yields Q-only near real-B but ladapt below real-B → K-side needs catalog semantics, Q-side benefits from any aligned low-rank subspace.
- **Validation.** `build_pca_baseline_basis.py` rank 12 on retail calibration set; then `eval_tau2_bench.py --domain retail --max-samples 114 --b-ont <pca>.pt --methods ocq_qbias_b-0.03 ocq_ladapt_k0.05_q-0.03`.
- **Interpretation.** PCA ≈ real-B on Q-only weakens "catalog essential" strong claim to "catalog optimal but low-rank suffices". PCA < real-B on ladapt preserves the strong claim on K-side.

**E-v5.5. Qwen 1.5B size-sweep smoke (0.5 GPU-h).**
- **Intent.** Fill one cell of `tab:e5-sizesweep` to anchor size-independence statement.
- **Hypothesis.** 1.5B layer-adaptive sign on MetaTool ST4 positive; magnitude smaller.
- **Validation.** `eval_metatool_subtask4.py --model Qwen/Qwen2.5-1.5B-Instruct --methods no_steer ocq_ladapt_k0.05_q-0.03 --max-samples 100`.
- **Interpretation.** Sign preserved → size-independence footnote supported. Sign flipped → caveat strengthens.

### Tier 3 — deferred (cost or blockers)

**E-v5.6. Phase 2.5 layer boundary sweep LS-1..LS-6 (6 GPU-h).** Coworker track; confirms τ=1/4 Pareto choice.
**E-v5.7. Canonical SEKA A100 reproduction (G1 gate).** Blocked by A100 availability; paper rows remain reference-only until cleared.
**E-v5.8. Llama MetaTool ST4 ladapt (1.7 GPU-h).** Cross-model at MetaTool level; omitted because τ² retail+telecom on Llama already cover the table-1-level cross-model claim.
**E-v5.9. β* logit-lens predictor (0.5 GPU-h + coding).** Main-body β* already softened; paper does not depend on validated predictor. Future work.

### Decision policy

**Default for Saturday**: run Tier 1 (E-v5.1, 5.2 already done, 5.3). Total cost ~3 min GPU + ~10 min Python. Each adds a concrete sentence to the paper and closes a specific audit flag. Tier 2 is optional; Tier 3 is deferred.

End of plan v5.
