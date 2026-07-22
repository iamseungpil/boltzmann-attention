# [DRAFT v0 · 2026-07-19] Same-Rule Interference: Why Batched Per-Item Judgments Fail in LLM Agent Pipelines, and a Structural Fix

> Working draft (English, markdown → LaTeX later). Placeholders marked ⟦TBD⟧.
> Evidence ledger: RATE_SUBAGENT_DESIGN_2026_07_18.md §2i–2m · sim_results/*20260719*. All numbers below trace to
> committed logs; grades follow the repo evidence-ledger convention ([S]=trajectory-verified).
> ⚠️Honesty gates carried into the text: pass^1 live results · bench quirks footnoted · frame-dependence of k* stated.

## Abstract

LLM agent pipelines routinely delegate *per-item judgments* to a model in batches: given a policy document and a
list of records, judge every record in one call. We report a failure mode of this pattern discovered in a live
tool-use pipeline (τ²-bench banking): specific items are judged incorrectly *only when batched with other items
that engage the same policy clause*, and the failure is a silent retreat to a default answer, not a random error.
Through a chain of controlled probes on the live prompt (temperature 0, free-form generation) we exclude token
load, absolute context position, generation order, and category similarity, and isolate the interference source:
**previous items that inference-bind the same rule clause without any lexical name match**. Items anchored by
explicit lexical evidence are immune; judgment-dependent items fail with a sharp threshold at k*=2 interfering
items, with output instability (unit slips such as 5 → 500) at the threshold. A double dissociation rejects
quota-consumption, judgment-budget, and category-cue accounts. The mechanism-derived mitigation — capping
judgment batches at 2 items so every item sits at a list edge — closes the failure in probes (38/38), causes no
regression across all probed cells (166/167), and turns previously failing live tasks into passes, while prompt
mitigations fail entirely, consistent with prior reports. A logit-readout size sweep (1.5B–32B) shows the
staircase is not an artifact of one model scale. We position the phenomenon against serial-position and
proactive-interference literature: what is new is the *unit* of interference (the rule clause), the dissociation
methodology, the default-retreat failure mode, and a structural fix derived from the mechanism rather than from
folklore.

## 1. Introduction

Agent frameworks increasingly interpose deterministic scaffolding between an LLM and its tools: the scaffold
fetches records, groups them, and asks the model to *formalize* a judgment per record — e.g., "for each of these
credit-card transactions, report the reward rate the policy documents assign to it." Batching these judgments is
the natural implementation: one call per group amortizes the (large) document prefix.

We describe how this natural implementation silently corrupts a live agent pipeline, and trace the corruption to
an interference effect that, to our knowledge, has not been isolated at this granularity. The observable is
mundane: in a τ²-bench banking task, one transaction (a Patagonia purchase on an eco-themed card) is assigned the
default rate 1 instead of the elevated rate 5, causing the downstream dispute-detection logic to miss a
discrepancy and the task to fail. The cause is not mundane:

- The same item is judged **correctly at list edges and incorrectly inside the list** (positions 4–5 of 6), while
  the absolute token position barely moves (77.8%→85.3% of the prompt) and the effect is non-monotonic in it (§4).
- Interference is caused **only by co-batched items that engage the same policy clause by inference** — synthetic
  no-name eco retailers that force the model to apply the "certified sustainable retailers" clause without a
  lexical match. Items that consume the same judgment budget but resolve lexically (explicit exclusion-list
  merchants; explicit named partners) cause **no** interference, a double dissociation (§5).
- The failure has a **sharp threshold**: k*=2 interfering items in the generation frame, with the model's output
  destabilizing exactly at threshold (a 5 → 500 unit slip) before settling on the default (§5.3).
- **Prompt-level mitigations fail**; a structural change — batch ≤ 2 — eliminates the failure with zero measured
  regression and closes the affected live tasks (§6).

Contributions. (1) Discovery and forensic isolation of a clause-level interference effect in a *real* tool-use
pipeline, connected end-to-end to task pass/fail — not a synthetic benchmark-only observation. (2) A
four-step exclusion chain (token load / absolute position / generation order / category similarity) with an
input–output dissociation design and a release-from-PI manipulation, both absent from the LLM literature.
(3) Characterization of the failure mode: default-retreat with threshold instability, distinct from
recall/copy errors studied in prior work. (4) A mechanism-derived structural mitigation (batch ≤ 2) validated for
non-regression, plus evidence that the standard prompt-engineering repertoire does not work. (5) A
construction-variation robustness analysis (100% failure at k≥2 across 45 varied constructions) and a 1.5B–32B
scale sweep showing no scale in the family is immune, and an associative-memory
interpretation that unifies our findings with the proactive-interference literature (developed fully in a
companion mechanism paper).

## 2. Related work

*Serial-position effects in LLMs.* Guo & Vosoughi (ACL 2025 Findings) name serial-position effects for
option-selection tasks (label choice, primacy-dominated, mitigation inconsistent). Wang et al. (EMNLP 2023)
report label-choice primacy in ChatGPT; listwise-reranking position bias is documented in IR (ECIR 2026). Our
task is per-item enumerated judgment, not option selection, and we show the driver is not the position itself.

*Batch prompting.* Cheng et al. (2023) report accuracy degradation as batch size grows (≈4-item threshold) and
per-item answers that depend on batch position; BPE mitigates by permutation + voting. These works frame the
effect as lost-in-the-middle over absolute positions and do not separate token position from list ordinality,
nor manipulate inter-item similarity — the two dissociations that localize the cause here.

*Instruction-following at scale.* IFScale (Jaroslawicz et al., 2025) shows primacy favoritism over 500
simultaneous instructions. Consistent with our "edges are protected" observation, but no interference design.

*Proactive interference.* Unable to Forget (arXiv 2506.08184) establishes log-linear PI decay in key–value
overwrite recall; Remember First Forget Last (arXiv 2603.00270) shows primacy protection / recency collapse.
Both use explicit overwrite memory probes. We show PI-like interference **without any overwriting**, in a
document-grounded judgment task, gated by *clause-level semantic similarity* — and we import the classic
release-from-PI manipulation (Watkins; fan effect, Anderson) from cognitive psychology to prove it.

*Practitioner folklore.* "Split calls beyond ~7 items" circulates as guidance. We upgrade folklore to a measured
threshold, an identified unit of interference, and a mitigation with quantified non-regression.

*Positioning.* We claim neither the discovery of position effects nor of PI in LLMs. The claims are the unit
(rule clause), the dissociation methodology, the failure mode (default retreat + threshold instability), the
end-to-end task connection, and the mechanism-derived fix. ⟦TBD: table T1 prior-work capability matrix⟧

## 3. Setting: a gated tool-use pipeline on τ²-bench banking

⟦Figure F1: pipeline diagram + failure case⟧

We study τ²-bench `banking_knowledge`: an agent (Qwen2.5-32B-Instruct, GPTQ-Int8, vLLM, temp 0) serves a
simulated customer (gpt-5.2 user-sim, the leaderboard-recommended standard) under a deterministic scaffold that
routes tool calls, verifies arguments against tool schemas, and isolates *formalization*: for each
card × category cell of the customer's transactions, a sub-call receives ALL policy documents for that card plus
the cell's raw transaction records, and must return, per transaction, the applicable reward `base_rate` (plus raw
promo parameters). A deterministic engine then computes expected rewards (amount × rate × promo), flags
discrepancies against recorded rewards, and drives the dispute workflow. The engine never generates domain
values; the model is the only source of judgments (audited mechanism-by-mechanism, §2k of the evidence ledger).

Judgment structure of the critical cell. The eco-card's policy grants an elevated rate (5×) to the "Green"
category — the operative clause covers "certified sustainable retailers and eco-labeled products" — with an
explicit merchant exclusion list, and a general base rate of 1× otherwise. Judging a Green transaction is
therefore *inference*: the merchant (e.g., Patagonia) is not named anywhere; the model must apply the clause via
the bank's category field. Judging an excluded merchant (e.g., Target) is *lexical*: the exclusion list names it.
This distinction — inference-bound vs lexically-anchored engagement of the same clause — turns out to be the
whole story.

## 4. A live failure and what it is not

**The failure.** In task_028 (and its Phase-1 variant 018), the 6-row EcoCard-Green cell of one customer is
formalized with Patagonia at rate 1 (correct: 5), silently breaking dispute detection. Trajectory forensics
([S]) show the sub-call's output is well-formed JSON; only the value is wrong — a retreat to the base rate.

**Not token load.** The prompt is dominated by the fixed document prefix; 1-row and 6-row prompts differ
negligibly in length, and isolating rows into smaller cells does not change per-row token counts materially.

**Not absolute context position.** Holding 5 companion rows fixed and sliding Patagonia through list positions
0–5 moves its character offset only from 77.8% to 85.3% of the prompt, yet the judgment flips from correct
(positions 0–2) to incorrect (3–4) and back to correct (5, last):

| Patagonia list position | 0 | 1 | 2 | 3 | 4 | 5 (last) |
|---|---|---|---|---|---|---|
| judged rate | 5 ✓ | 5 ✓ | 5 ✓ | **1 ✗** | **1 ✗** | 5 ✓ |

Non-monotonic in offset (failure at 82% but success at 85%), within a ~350-token window: absolute-position
accounts (lost-in-the-middle) are out. What survives is *list* structure: interior items fail, edge items are
protected.

**Not generation order.** An input–output dissociation (2 cells × 4 arms): forcing the failing item to be
*emitted first* while remaining interior in the input still fails; placing it first in input while emitting it
last still passes (order compliance verified in the outputs). The damage is done in the prefill, before a single
output token — generation-side accounts (output-order pressure, autoregressive drift) are out.

**Not merchant identity.** Across cells, the failing item follows the cell, not the merchant: Patagonia fails in
one customer's 6-row cell, REI in another's 13-row cell, and a 4-row cell fails nowhere.

## 5. Isolating the cause: clause-level interference

### 5.1 Release from proactive interference

Fixing the target (vulnerable item) in the last position and swapping only the *preceding* items: similar
predecessors (same-category Green rows) reproduce the failure at k=2–4; dissimilar predecessors (the same
customer's non-Green rows, matched in count and format) never do. Similarity of the co-batched material — not
their number or the target's position — is the active ingredient. This is the classic release-from-PI signature.

### 5.2 Double dissociation: the unit is the clause, engaged by inference

Three predecessor types, target always last (two targets: Patagonia, REI; all temp-0 [S]):

| predecessors (k=4) | judgment needed? | same 5× clause? | lexical anchor? | target |
|---|---|---|---|---|
| A: no-name eco retailers (synthetic) | yes | yes — by inference | none | **FAIL** |
| B: exclusion-list merchants (Target, Walmart, …) | yes | engaged, denied | exclusion list names them | pass |
| C: explicit partners (Tesla Supercharger, …) | minimal | yes — by name | partner list names them | pass |

Arm B consumes as much "judgment budget" as A (each item must be weighed against the clause and denied) — yet no
interference: the quota and judgment-budget accounts die. Arm C grants the same 5× — the category-cue account
dies. Only arm A — items that bind the same clause *by inference, without a name match* — interferes. The
synthetic no-name brands additionally kill a memorization confound: these merchants cannot be in training data.

### 5.3 Threshold, and instability at threshold

Growing k (synthetic same-clause predecessors, target last): Patagonia is judged 5 at k∈{0,1} and 1 for all
k≥2. REI: 5 at k∈{0,1}; at exactly k=2 the output destabilizes to **500** — a unit slip (5 × the percent scale)
— then settles to 1 for k≥3. The transition is a step, not a drift, and the step boundary is where malformed
outputs appear. (In the live pipeline this unit slip is caught by a declared-range re-query; the interference
failure itself is not catchable that way because 1 is in range — it is a *plausible wrong answer*.)

### 5.4 Robustness: the effect survives every construction we varied

Because temp-0 generation is deterministic, we estimate variance over *constructions*: 3 semantically-preserving
instruction paraphrases × 3 disjoint synthetic brand sets × all distinct interior orders (2 at k=2, 3 at k=4),
k∈{0,2,4}, 48 unique prompts. Failure rates: k=0: 0/3; **k=2: 18/18** (Wilson 95% CI [0.82, 1.00]);
**k=4: 27/27** (CI [0.88, 1.00]). The value distribution sharpens the threshold-instability claim: at k=2,
12/18 cells retreat to the default (rate 1) and **6/18 produce the unit slip (rate 100)** — spread over two of
three paraphrases, all three brand sets, and both orders — while at k=4 all 27 cells settle on the default with
no slips. Instability is a property of the threshold, not of a particular wording, brand surface, or order.

### 5.5 Does scale fix it? No — every size in the family shows the staircase

A logit-readout probe (full-fidelity prompt; target-first primed output so the readout position is the judgment
token) sweeps model size over the same construction:

| model | k=0 P(5) | k=1 P(5)/P(1) | k=2 | k=4 | k=8 |
|---|---|---|---|---|---|
| 32B (Int8, server) | 0.999 | 0.154 / 0.756 | 0.024 / 0.970 | 0.025 / 0.966 | 0.100 / 0.863 |
| 14B (bf16) | 0.999 | 0.034 / 0.091 | 0.118 / 0.680 | 0.110 / 0.383 | 0.123 / 0.707 |
| 3B (bf16) | 0.983 | 0.259 / 0.548 | 0.100 / 0.446 | 0.120 / 0.476 | 0.046 / 0.565 |
| 1.5B (bf16) | 0.869 | 0.541 / 0.372 | 0.627 / 0.262 | 0.166 / 0.450 | 0.199 / 0.199 |

Every scale shows the staircase: a confident correct judgment at k=0 that collapses with interfering
predecessors. 32B, 14B and 3B collapse at k=1 in this frame; at exactly k=1 the 14B mass scatters to
third-candidate tokens (P(5)+P(1)=0.13) — the logit-space counterpart of the output instability we observe at
threshold in generation. The smallest model (1.5B) degrades most gradually and is the only one holding a
majority at k=2 — against our pre-registered expectation (smaller ⇒ earlier failure); a candidate reading
(not a claim) is that its weaker baseline binding (k=0 only 0.869) shrinks both the margin and the interference
that attacks it. Two honesty notes: (i) the primed-readout frame lowers the threshold relative to free
generation (k*=1 vs k*=2) — output-ordering freedom is itself protective, so thresholds are frame-dependent and
we compare only within frame; (ii) the served 32B is Int8-quantized, so we re-ran the 32B row in bf16 on the
CPU stack: k=0 P(5)=0.999 → k=1 P(1)=0.799 → k=2 P(1)=0.973 — the same collapse, ruling out a
quantization/serving-stack artifact. The claim we retain: **no scale in this family is immune**.
⟦optional: 0.5B row⟧

### 5.6 A second domain: the structure, not the banking surface

τ²-bench's official domains have procedural policies without per-item clause judgment, so for domain generality
we mirrored the *structure* in a synthetic domain with a realistic policy and a fully disjoint surface
vocabulary: airline checked-baggage fees — a waiver clause ("sports and adventure gear from certified
adventure-sports outfitters", fee $0, category tag authoritative), an explicit retailer exclusion list, a $35
standard fee, and a target item from a recognizable brand named nowhere in the documents (Burton snowboard,
category Sports). Logit-readout on the same 3B model: k=0 P($0)=1.000; similar (no-name outfitter gear)
predecessors collapse the target to the standard fee at k=1 (P($0)=0.076, k=4: 0.003); dissimilar predecessors
protect (P($0)=0.905/0.818/0.963 at k=1/2/4). Notably the whole prompt is 694–1456 tokens — interference at
under 900 tokens is a further, cross-domain refutation of context-length accounts. Honest note: at k=8
dissimilar, P($0) drops to 0.373 — in this short-prompt regime a nonspecific load effect appears at high k,
which we report rather than smooth over; the similarity gating is clean through k=4.

## 6. A structural fix, and why prompts don't work

**Prompt mitigations fail.** Two targeted prompt strengthenings — (a) explicit anti-fallback instruction
("do NOT fall back to the standard rate merely because you are unsure the merchant qualifies"), (b) an
authority clause for the category field ("the bank's own classification is final") — were deployed in the live
sub-prompt and did not eliminate the interference failures (honest record: two failed attempts preceded the
diagnosis). This matches the mitigation-resistance reported for PI (Unable to Forget: prompting "entirely
fails") and for batch effects (BPE resorts to permutation+voting rather than instructions).

**Mechanism-derived fix.** If interference requires interior list positions and k≥2 same-clause predecessors,
then batches of ≤2 have no interior and never exceed k=1 at the judgment point: every item is at an edge. We cap
the formalization batch at 2 (a scaffold parameter declared in the domain adapter, not a prompt change; the
engine still generates no values).

Validation ([S]): (i) targeted probes: 38/38 correct on the previously failing constructions; (ii)
non-regression: 166/167 across *all* probed cells of the 8 task users (the pre-existing miss is unrelated
⟦TBD: verify and footnote the 1 miss identity⟧); (iii) live: the two tasks whose failures traced to this cell
(018, 021) pass end-to-end with the cap in place; (iv) cost: the shared document prefix is served from prefix
cache, so the extra calls do not multiply prompt cost proportionally (hit-rate logged). Series scoreboard across
the 8-task banking suite: **018/020/021/022/027/029 pass** ([S], single-trial each — we report pass^1, not
pass^k); 026 is blocked by a benchmark gold-authoring bug (the bench's own documented flooring policy contradicts
its gold; census 9/10 golds floor, the exception is the blocking one) — reported upstream ⟦TBD: issue link⟧;
028 currently fails only on a context-window infrastructure limit (the corrected dispute flow lengthens the
conversation past the serving window; a lever trade-off we report honestly, mitigation in progress ⟦TBD⟧).

**What the fix is not.** Batch≤2 is not a general recommendation to shrink batches for accuracy; it is the
*specific* consequence of an identified mechanism (edge protection + threshold k*=2). Under a different clause
structure the safe batch size would differ; the diagnosis procedure (release-from-PI + dissociation probes) is
the transferable artifact.

## 7. Interpretation (frame only)

The pattern — similarity-gated interference from inference-bound predecessors, edge protection, sharp threshold,
default retreat — is what an associative-memory reading of attention predicts: same-clause predecessors act as
correlated (quasi-degenerate) retrieval patterns that dilute the target's clause-binding; lexically-anchored
items retrieve via near-duplicate token matching with a large margin and are immune. We use this as an
organizing frame here and *do not* claim mechanism identification in this paper; the companion mechanism study
(attention-mass curves, attention-mask knockout, temperature scaling) tests it directly. ⟦cross-ref Track B⟧

## 8. Limitations and honest notes

- Live results are pass^1 (single trial per task; deterministic agent but stochastic user-sim); multi-trial
  robustness for the live suite is future work. Probe results are temp-0 deterministic with construction-variation
  CIs instead (§5.4).
- One live benchmark (banking); the cross-domain replication (§5.6) is a structural mirror in a synthetic
  domain with a realistic policy, not a second live pipeline.
- The 32B agent is Int8-quantized in the live pipeline; the bf16 spot-check (§5.5) deconfounds the probe
  results but the live runs themselves remain Int8.
- Benchmark quirks are footnoted where they touch numbers: evaluator dict-comparison keys (optional-argument
  filling kills action match without affecting reward), user-sim JSON whitespace luck (no reward effect), and the
  026 gold bug.
- k*=2 is a property of this clause structure and frame, not a universal constant (frame-dependence shown in §5.5).

## References ⟦TBD: full bibliography⟧

Guo & Vosoughi 2025 (2406.15981) · Wang+ 2023 · Cheng+ 2023 · BPE · IFScale 2025 · Unable to Forget (2506.08184)
· Remember First Forget Last (2603.00270) · Liu+ 2023 (lost in the middle) · Context Length Alone Hurts (EMNLP'25)
· Ramsauer+ 2020 · Anderson (fan effect) · Watkins (release from PI) · τ²-bench.

---
### Figure/Table plan (from Track A skeleton)
- F1 pipeline + failure case (§3–4) · F2 exclusion chain: position sweep table + io-dissociation (§4)
- F3 double-dissociation bars (§5.2) · F4 k-staircase, overlaid by size (§5.3/5.5) · F5 batch≤2 non-regression + live (§6)
- T1 prior-work capability matrix (§2).
### Data provenance map
§4 table = rot_serial_20260719 · §5.1 = rpi_20260719 · §5.2 = mech_20260719 · §5.3 = ksweep_20260719 ·
§5.4 = a1_robustness_ci_20260719 · §5.5 = size_k_sweep_20260719 · §6 = bank_batch2_probe /
bank_shared_docs_probe_v5 / bank_redesign4·6·7b · all in reports/facet_rft_2026/sim_results/.
